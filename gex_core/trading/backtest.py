"""Walk-forward backtest for the gamma auto-trader."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from gex_core.exports import EXPORT_DIR
from gex_core.features import safe_float
from gex_core.history import _build_history_impl
from gex_core.market_time import (
    bars_between_timestamps,
    export_ts_eod_flatten,
    export_ts_is_trading_day,
    filter_trading_history,
    minutes_between_timestamps,
)
from gex_core.trading.advisor import _rule_based_advice
from gex_core.trading.config import (
    equity_from_mark,
    fix_magnet_exit_scale,
    max_entries_per_cycle,
    max_open_positions,
    min_entry_confidence,
    momentum_bars,
    stop_cooldown_bars,
    stop_loss_pct,
    take_profit_pct,
    trader_bar_minutes,
    trader_session_only,
)
from gex_core.trading.exits import (
    ExitProfile,
    ExitState,
    build_exit_profile,
    evaluate_exit,
)
from gex_core.trading.filters import MarketContext, market_context_from_snapshot
from gex_core.trading.paper_broker import (
    estimate_entry_premium,
    estimate_option_pnl_pct,
    mark_to_market_premium,
    pnl_usd,
)
from gex_core.trading.execution import map_execution_strike, resolve_backtest_execution_spot, uses_execution_mapping
from gex_core.trading.signals import compute_entry_candidates
from gex_core.trading.sizing import affordable_qty, resolve_contract_qty


@dataclass
class AccountLedger:
    starting_capital: float
    cash: float
    peak_equity: float
    skipped_insufficient_capital: int = 0
    equity_curve: list[dict[str, float | str]] = field(default_factory=list)

    @classmethod
    def create(cls, starting_capital: float) -> AccountLedger:
        return cls(
            starting_capital=starting_capital,
            cash=starting_capital,
            peak_equity=starting_capital,
        )

    def affordable_qty(self, premium: float, desired: float) -> int:
        unit_cost = premium * 100.0
        if unit_cost <= 0:
            return 0
        return max(0, min(int(desired), int(self.cash // unit_cost)))

    def debit_entry(self, premium: float, qty: float) -> None:
        self.cash -= premium * 100.0 * qty

    def credit_exit(self, exit_premium: float, qty: float) -> None:
        self.cash += exit_premium * 100.0 * qty

    def record_equity(self, ts: str, open_positions: list["_OpenPosition"], mark_spot: float) -> None:
        mtm = 0.0
        for pos in open_positions:
            pnl_pct = _position_pnl(pos, mark_spot)
            mtm += mark_to_market_premium(pos.entry_premium, pnl_pct) * 100.0 * pos.qty
        equity = self.cash + mtm
        self.peak_equity = max(self.peak_equity, equity)
        self.equity_curve.append({"ts": ts, "equity": round(equity, 2), "cash": round(self.cash, 2)})

    def max_drawdown_pct(self) -> float:
        if not self.equity_curve:
            return 0.0
        peak = self.starting_capital
        max_dd = 0.0
        for point in self.equity_curve:
            equity = float(point["equity"])
            peak = max(peak, equity)
            if peak > 0:
                max_dd = max(max_dd, (peak - equity) / peak)
        return max_dd


@dataclass
class _OpenPosition:
    entry_idx: int
    entry_ts: str
    option_type: str
    strike: float
    entry_spot: float
    entry_premium: float
    signal_type: str
    signal_gamma: float
    gamma_delta: float
    ai_confidence: float
    signal_strike: float | None = None
    magnet_strike: float | None = None
    qty: float = 1.0
    exit_profile: ExitProfile = field(default_factory=ExitProfile)
    exit_state: ExitState = field(default_factory=ExitState)


@dataclass
class _ClosedTrade:
    entry_ts: str
    exit_ts: str
    option_type: str
    strike: float
    signal_type: str
    entry_spot: float
    exit_spot: float
    entry_premium: float
    exit_premium: float
    pnl_pct: float
    pnl_usd: float
    exit_reason: str
    bars_held: int
    signal_strike: float | None = None
    qty: float = 1.0
    equity_after: float | None = None


@dataclass
class BacktestState:
    open_positions: list[_OpenPosition] = field(default_factory=list)
    closed_trades: list[_ClosedTrade] = field(default_factory=list)
    skipped_entries: int = 0
    blocked_duplicate: int = 0
    skipped_gamma_decline: int = 0
    skipped_strike_distance: int = 0
    skipped_filters: int = 0
    blocked_cooldown: int = 0
    skipped_no_execution_spot: int = 0
    skipped_weekends: int = 0
    skipped_low_confidence: int = 0
    skipped_wall_weak_gamma: int = 0
    skipped_wall_regime: int = 0
    skipped_wall_drift: int = 0
    last_wall_strike: float | None = None
    strike_cooldown: dict[tuple[float, str], str] = field(default_factory=dict)
    account: AccountLedger | None = None


def _mark_spot(signal_spot: float) -> float | None:
    if signal_spot <= 0:
        return None
    if uses_execution_mapping():
        return resolve_backtest_execution_spot(signal_spot=signal_spot)
    return signal_spot


def _snapshot_bar_minutes(row: dict) -> float:
    interval = safe_float(row.get("interval_minutes"), 0.0)
    if interval > 0:
        return interval
    return trader_bar_minutes()


def _max_exit_gap_minutes(bar_minutes: float) -> float:
    return max(bar_minutes * 2.0, 20.0)


def _strike_profiles_equal(a, b) -> bool:
    if a is None or b is None:
        return False
    if not isinstance(a, pd.Series) or not isinstance(b, pd.Series):
        return a == b
    if a.empty or b.empty:
        return False
    try:
        return a.equals(b)
    except Exception:
        return False


def _signal_previous_row(history: list[dict], idx: int) -> dict:
    """Last snapshot with a distinct strike profile (skip duplicate GEX bars)."""
    if idx <= 0:
        return history[0]
    cur = history[idx].get("strike")
    prev_idx = idx - 1
    while prev_idx > 0 and _strike_profiles_equal(cur, history[prev_idx].get("strike")):
        prev_idx -= 1
    return history[prev_idx]


def _resolve_trade_context(
    *,
    signal_strike: float,
    signal_spot: float,
) -> tuple[float, float, float] | None:
    """Return (execution_strike, execution_spot, signal_strike) or None if spot unavailable."""
    exec_spot = _mark_spot(signal_spot)
    if exec_spot is None or exec_spot <= 0:
        return None
    if uses_execution_mapping():
        exec_strike = map_execution_strike(
            signal_strike,
            signal_spot=signal_spot,
            execution_spot=exec_spot,
        )
        return exec_strike, exec_spot, signal_strike
    return signal_strike, exec_spot, signal_strike


def _has_open_duplicate(positions: list[_OpenPosition], *, strike: float, option_type: str) -> bool:
    for pos in positions:
        if pos.strike == strike and pos.option_type.lower() == option_type.lower():
            return True
    return False


def _memory_from_closed(closed: list[_ClosedTrade]) -> dict[str, Any]:
    if not closed:
        return {"performance": {"lessons": [], "by_signal": {}, "total_trades": 0, "win_rate": 0.0}}

    by_signal: dict[str, dict[str, float | int]] = {}
    wins = 0
    for trade in closed:
        bucket = by_signal.setdefault(
            trade.signal_type,
            {"count": 0, "wins": 0, "sum_pnl_pct": 0.0},
        )
        bucket["count"] = int(bucket["count"]) + 1
        bucket["sum_pnl_pct"] = float(bucket["sum_pnl_pct"]) + trade.pnl_pct
        if trade.pnl_pct > 0:
            bucket["wins"] = int(bucket["wins"]) + 1
            wins += 1

    for sig, stats in by_signal.items():
        count = int(stats["count"])
        stats["win_rate"] = float(stats["wins"]) / count if count else 0.0
        stats["avg_pnl_pct"] = float(stats["sum_pnl_pct"]) / count if count else 0.0

    total = len(closed)
    return {
        "performance": {
            "total_trades": total,
            "win_rate": wins / total if total else 0.0,
            "by_signal": by_signal,
            "lessons": [],
        }
    }


def _position_pnl(pos: _OpenPosition, spot: float) -> float:
    return estimate_option_pnl_pct(
        pos.option_type,
        entry_spot=pos.entry_spot,
        current_spot=spot,
        strike=pos.strike,
    )


def _apply_backtest_exit(
    state: BacktestState,
    pos: _OpenPosition,
    *,
    idx: int,
    exit_ts: str,
    exit_spot: float,
    pnl_pct: float,
    exit_reason: str,
    bars_held: int,
    sell_qty: float | None = None,
) -> None:
    trade_qty = pos.qty if sell_qty is None else sell_qty
    if trade_qty <= 0:
        return
    exit_premium = mark_to_market_premium(pos.entry_premium, pnl_pct)

    if exit_reason in {"take_profit_partial", "magnet_partial"} and trade_qty < pos.qty:
        equity_after = None
        if state.account:
            state.account.credit_exit(exit_premium, trade_qty)
            equity_after = state.account.cash
        partial_usd = pnl_usd(pos.entry_premium, exit_premium, trade_qty)
        state.closed_trades.append(
            _ClosedTrade(
                entry_ts=pos.entry_ts,
                exit_ts=exit_ts,
                option_type=pos.option_type,
                strike=pos.strike,
                signal_type=pos.signal_type,
                entry_spot=pos.entry_spot,
                exit_spot=exit_spot,
                entry_premium=pos.entry_premium,
                exit_premium=exit_premium,
                pnl_pct=pnl_pct,
                pnl_usd=partial_usd,
                exit_reason=exit_reason,
                bars_held=bars_held,
                signal_strike=pos.signal_strike,
                qty=trade_qty,
                equity_after=equity_after,
            )
        )
        pos.qty -= trade_qty
        pos.exit_state.partial_taken = True
        return

    _close_position(
        pos,
        exit_idx=idx,
        exit_ts=exit_ts,
        exit_spot=exit_spot,
        pnl_pct=pnl_pct,
        exit_reason=exit_reason,
        closed=state.closed_trades,
        state=state,
        qty=trade_qty,
        bars_held=bars_held,
    )


def _flatten_backtest_eod(
    state: BacktestState,
    *,
    idx: int,
    ts: str,
    mark_spot: float,
    bar_minutes: float,
) -> None:
    for pos in list(state.open_positions):
        pnl_pct = _position_pnl(pos, mark_spot)
        bars_held = bars_between_timestamps(pos.entry_ts, ts, bar_minutes=bar_minutes)
        _apply_backtest_exit(
            state,
            pos,
            idx=idx,
            exit_ts=ts,
            exit_spot=mark_spot,
            pnl_pct=pnl_pct,
            exit_reason="eod_flatten",
            bars_held=bars_held,
        )
    state.open_positions.clear()


def _close_position(
    pos: _OpenPosition,
    *,
    exit_idx: int,
    exit_ts: str,
    exit_spot: float,
    pnl_pct: float,
    exit_reason: str,
    closed: list[_ClosedTrade],
    state: BacktestState | None = None,
    qty: float | None = None,
    bars_held: int | None = None,
) -> None:
    trade_qty = pos.qty if qty is None else qty
    exit_premium = mark_to_market_premium(pos.entry_premium, pnl_pct)
    equity_after = None
    if state and state.account:
        state.account.credit_exit(exit_premium, trade_qty)
        equity_after = state.account.cash
    held = bars_held if bars_held is not None else exit_idx - pos.entry_idx
    closed.append(
        _ClosedTrade(
            entry_ts=pos.entry_ts,
            exit_ts=exit_ts,
            option_type=pos.option_type,
            strike=pos.strike,
            signal_type=pos.signal_type,
            entry_spot=pos.entry_spot,
            exit_spot=exit_spot,
            entry_premium=pos.entry_premium,
            exit_premium=exit_premium,
            pnl_pct=pnl_pct,
            pnl_usd=pnl_usd(pos.entry_premium, exit_premium, trade_qty),
            exit_reason=exit_reason,
            bars_held=held,
            signal_strike=pos.signal_strike,
            qty=trade_qty,
            equity_after=equity_after,
        )
    )
    pos.qty = max(0.0, pos.qty - trade_qty)


def _check_exits(
    state: BacktestState,
    *,
    idx: int,
    ts: str,
    signal_spot: float,
    row: dict,
    prev_ts: str | None = None,
    prev_signal_spot: float | None = None,
    apply_stop_cooldown: bool = True,
) -> None:
    mark_spot = _mark_spot(signal_spot)
    if mark_spot is None or mark_spot <= 0:
        return
    bar_minutes = _snapshot_bar_minutes(row)
    if prev_ts and prev_signal_spot is not None:
        gap = minutes_between_timestamps(prev_ts, ts)
        if gap is not None and gap > _max_exit_gap_minutes(bar_minutes):
            prev_mark = _mark_spot(prev_signal_spot)
            if prev_mark and prev_mark > 0 and state.open_positions:
                for pos in list(state.open_positions):
                    pnl_pct = _position_pnl(pos, prev_mark)
                    bars_held = bars_between_timestamps(pos.entry_ts, prev_ts, bar_minutes=bar_minutes)
                    exit_reason = "session_gap"
                    exit_pnl = pnl_pct
                    if bars_held >= pos.exit_profile.time_stop_bars:
                        exit_reason = "time_stop"
                    _close_position(
                        pos,
                        exit_idx=idx,
                        exit_ts=prev_ts,
                        exit_spot=prev_mark,
                        pnl_pct=exit_pnl,
                        exit_reason=exit_reason,
                        closed=state.closed_trades,
                        state=state,
                        qty=pos.qty,
                        bars_held=bars_held,
                    )
                state.open_positions = []
            return

    if export_ts_eod_flatten(ts) and state.open_positions:
        _flatten_backtest_eod(state, idx=idx, ts=ts, mark_spot=mark_spot, bar_minutes=bar_minutes)
        if state.account:
            state.account.record_equity(ts, state.open_positions, mark_spot)
        return

    remaining: list[_OpenPosition] = []
    for pos in state.open_positions:
        pnl_pct = _position_pnl(pos, mark_spot)
        bars_held = bars_between_timestamps(pos.entry_ts, ts, bar_minutes=bar_minutes)
        exit_reason, exit_pnl = evaluate_exit(
            pnl_pct,
            state=pos.exit_state,
            bars_held=bars_held,
            entry_spot=pos.entry_spot,
            strike=pos.strike,
            current_spot=mark_spot,
            option_type=pos.option_type,
            profile=pos.exit_profile,
            magnet_strike=pos.magnet_strike,
        )
        if exit_reason:
            sell_qty = pos.qty
            reason = exit_reason
            if exit_reason in {"take_profit_partial", "magnet_partial"} and pos.qty > 1:
                sell_qty = max(1, pos.qty // 2)
            elif exit_reason == "take_profit_partial":
                reason = "take_profit"
            _apply_backtest_exit(
                state,
                pos,
                idx=idx,
                exit_ts=ts,
                exit_spot=mark_spot,
                pnl_pct=exit_pnl,
                exit_reason=reason if reason != "take_profit_partial" else exit_reason,
                bars_held=bars_held,
                sell_qty=sell_qty,
            )
            if reason == "stop_loss" and apply_stop_cooldown:
                key = (pos.signal_strike or pos.strike, pos.option_type.lower())
                cooldown_bars = stop_cooldown_bars()
                cooldown_minutes = cooldown_bars * bar_minutes
                from datetime import timedelta

                from gex_core.exports import parse_timestamp

                expire = parse_timestamp(ts) + timedelta(minutes=cooldown_minutes)
                state.strike_cooldown[key] = expire.strftime("%Y-%m-%d_%H%M%S")
            if pos.qty > 0:
                remaining.append(pos)
        else:
            remaining.append(pos)
    state.open_positions = remaining
    if state.account:
        state.account.record_equity(ts, state.open_positions, mark_spot)


def _maybe_enter(
    state: BacktestState,
    *,
    idx: int,
    ts: str,
    row: dict,
    prev_row: dict,
    spot_history: list[float],
    max_open: int,
) -> None:
    if not export_ts_is_trading_day(ts):
        state.skipped_weekends += 1
        return

    spot = safe_float(row.get("spot"), 0.0)
    prev_spot = safe_float(prev_row.get("spot"), 0.0) or None
    exposure = row.get("strike")
    previous = prev_row.get("strike")

    pack = compute_entry_candidates(exposure, previous, spot=spot)
    if not pack.get("available"):
        skip = pack.get("skip_reason")
        if skip == "gamma_declined":
            state.skipped_gamma_decline += 1
        elif skip == "strike_too_far":
            state.skipped_strike_distance += 1
        return

    row_with_ts = dict(row)
    row_with_ts.setdefault("ts", ts)
    market = market_context_from_snapshot(row_with_ts, prev_spot=prev_spot, spot_history=spot_history)
    memory = _memory_from_closed(state.closed_trades)
    opened_this_cycle = 0

    for rec in pack.get("candidates") or []:
        if len(state.open_positions) >= max_open:
            return
        if opened_this_cycle >= max_entries_per_cycle():
            return

        signals = {**pack, "recommended": rec}
        advice = _rule_based_advice(signals, memory, market=market)
        if not advice.get("approve"):
            state.skipped_entries += 1
            if advice.get("filter"):
                state.skipped_filters += 1
            continue

        option_type = str(advice.get("option_type") or rec["option_type"])
        signal_strike = float(rec["strike"])
        magnet_strike_raw = float(rec.get("magnet_strike") or signal_strike)
        trade_ctx = _resolve_trade_context(signal_strike=signal_strike, signal_spot=spot)
        if trade_ctx is None:
            state.skipped_no_execution_spot += 1
            continue
        exec_strike, exec_spot, _ = trade_ctx
        if fix_magnet_exit_scale() and uses_execution_mapping():
            magnet_strike = map_execution_strike(
                magnet_strike_raw,
                signal_spot=spot,
                execution_spot=exec_spot,
            )
        else:
            magnet_strike = magnet_strike_raw
        cooldown_key = (magnet_strike_raw, option_type.lower())
        cooldown_until = state.strike_cooldown.get(cooldown_key)
        if cooldown_until and ts < cooldown_until:
            state.blocked_cooldown += 1
            continue
        if _has_open_duplicate(state.open_positions, strike=exec_strike, option_type=option_type):
            state.blocked_duplicate += 1
            continue

        confidence = float(advice.get("confidence", 0.5))
        if confidence < min_entry_confidence():
            state.skipped_low_confidence += 1
            continue
        size_mult = float(advice.get("size_multiplier") or 1.0)
        premium = estimate_entry_premium(exec_spot, exec_strike)
        equity = state.account.cash if state.account else None
        qty = float(
            resolve_contract_qty(
                confidence=confidence,
                premium=premium,
                entry_spot=exec_spot,
                strike=exec_strike,
                account_equity=equity,
                size_multiplier=size_mult,
            )
        )
        if state.account:
            qty = float(affordable_qty(premium, state.account.cash, qty))
            if qty < 1:
                state.account.skipped_insufficient_capital += 1
                continue
            state.account.debit_entry(premium, qty)
        elif qty < 1:
            state.skipped_entries += 1
            continue

        regime = str(row.get("regime") or "")
        expected_move = safe_float(row.get("expected_move_pct"), 0.0) or None
        profile = build_exit_profile(
            ai_confidence=confidence,
            gamma_delta=float(rec["gamma_delta"]),
            regime=regime,
            entry_spot=exec_spot,
            strike=exec_strike,
            expected_move_pct=expected_move,
            magnet_strike=magnet_strike,
        )
        state.open_positions.append(
            _OpenPosition(
                entry_idx=idx,
                entry_ts=ts,
                option_type=str(option_type),
                strike=exec_strike,
                entry_spot=exec_spot,
                entry_premium=premium,
                signal_type=str(rec["signal_type"]),
                signal_gamma=float(rec["gamma_bn"]),
                gamma_delta=float(rec["gamma_delta"]),
                ai_confidence=confidence,
                signal_strike=signal_strike if uses_execution_mapping() else None,
                magnet_strike=magnet_strike,
                qty=qty,
                exit_profile=profile,
            )
        )
        if state.account:
            state.account.record_equity(ts, state.open_positions, exec_spot)
        opened_this_cycle += 1


def _realized_pnl_usd(state: BacktestState) -> float:
    return float(sum(t.pnl_usd for t in state.closed_trades))


def _account_return_pct(state: BacktestState) -> float:
    if not state.account:
        return 0.0
    starting = state.account.starting_capital
    if starting <= 0:
        return 0.0
    if equity_from_mark() and state.account.equity_curve:
        ending = float(state.account.equity_curve[-1]["equity"])
        return (ending - starting) / starting
    return _realized_pnl_usd(state) / starting


def _account_ending_capital(state: BacktestState) -> float:
    if not state.account:
        return 0.0
    if equity_from_mark() and state.account.equity_curve:
        return float(state.account.equity_curve[-1]["equity"])
    return state.account.starting_capital + _realized_pnl_usd(state)


def _summarize(
    ticker: str,
    *,
    history_len: int,
    history: list[dict],
    state: BacktestState,
    stop_loss: float,
    take_profit: float,
    weekend_snapshots_excluded: int = 0,
) -> dict[str, Any]:
    trades = state.closed_trades
    date_from = str(history[0]["ts"]) if history else None
    date_to = str(history[-1]["ts"]) if history else None
    if not trades:
        result = {
            "ticker": ticker,
            "snapshots": history_len,
            "date_from": date_from,
            "date_to": date_to,
            "total_trades": 0,
            "open_at_end": len(state.open_positions),
            "skipped_entries": state.skipped_entries,
            "blocked_duplicate": state.blocked_duplicate,
            "skipped_gamma_decline": state.skipped_gamma_decline,
            "skipped_strike_distance": state.skipped_strike_distance,
            "skipped_filters": state.skipped_filters,
            "blocked_cooldown": state.blocked_cooldown,
            "skipped_no_execution_spot": state.skipped_no_execution_spot,
            "skipped_weekends": state.skipped_weekends,
            "skipped_low_confidence": state.skipped_low_confidence,
            "weekend_snapshots_excluded": weekend_snapshots_excluded,
            "message": "No trades triggered in walk-forward window",
            "stop_loss_pct": stop_loss,
            "take_profit_pct": take_profit,
        }
        if state.account:
            result["account"] = {
                "starting_capital": state.account.starting_capital,
                "ending_capital": round(state.account.cash, 2),
                "return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "skipped_insufficient_capital": state.account.skipped_insufficient_capital,
            }
        return result

    wins = sum(1 for t in trades if t.pnl_pct > 0)
    by_signal: dict[str, dict[str, Any]] = {}
    for trade in trades:
        bucket = by_signal.setdefault(
            trade.signal_type,
            {"count": 0, "wins": 0, "sum_pnl_pct": 0.0, "sum_pnl_usd": 0.0},
        )
        bucket["count"] += 1
        bucket["sum_pnl_pct"] += trade.pnl_pct
        bucket["sum_pnl_usd"] += trade.pnl_usd
        if trade.pnl_pct > 0:
            bucket["wins"] += 1

    for stats in by_signal.values():
        count = stats["count"]
        stats["win_rate"] = stats["wins"] / count
        stats["avg_pnl_pct"] = stats["sum_pnl_pct"] / count
        stats["avg_pnl_usd"] = stats["sum_pnl_usd"] / count

    by_exit: dict[str, int] = {}
    for trade in trades:
        by_exit[trade.exit_reason] = by_exit.get(trade.exit_reason, 0) + 1

    return {
        "ticker": ticker,
        "snapshots": history_len,
        "date_from": date_from,
        "date_to": date_to,
        "total_trades": len(trades),
        "wins": wins,
        "losses": len(trades) - wins,
        "win_rate": wins / len(trades),
        "avg_pnl_pct": sum(t.pnl_pct for t in trades) / len(trades),
        "total_pnl_usd": sum(t.pnl_usd for t in trades),
        "avg_bars_held": sum(t.bars_held for t in trades) / len(trades),
        "by_signal": by_signal,
        "by_exit_reason": by_exit,
        "open_at_end": len(state.open_positions),
        "skipped_entries": state.skipped_entries,
        "blocked_duplicate": state.blocked_duplicate,
        "skipped_gamma_decline": state.skipped_gamma_decline,
        "skipped_strike_distance": state.skipped_strike_distance,
        "skipped_filters": state.skipped_filters,
        "blocked_cooldown": state.blocked_cooldown,
        "skipped_no_execution_spot": state.skipped_no_execution_spot,
        "skipped_weekends": state.skipped_weekends,
        "skipped_low_confidence": state.skipped_low_confidence,
        "weekend_snapshots_excluded": weekend_snapshots_excluded,
        "stop_loss_pct": stop_loss,
        "take_profit_pct": take_profit,
        "execution_ticker": "SPY" if uses_execution_mapping() else ticker,
        "trades": [
            {
                "entry_ts": t.entry_ts,
                "exit_ts": t.exit_ts,
                "signal_type": t.signal_type,
                "option_type": t.option_type,
                "strike": t.strike,
                "signal_strike": t.signal_strike,
                "qty": t.qty,
                "pnl_pct": round(t.pnl_pct, 4),
                "pnl_usd": round(t.pnl_usd, 2),
                "exit_reason": t.exit_reason,
                "bars_held": t.bars_held,
                "equity_after": round(t.equity_after, 2) if t.equity_after is not None else None,
            }
            for t in trades
        ],
        **(
            {
                "account": {
                    "starting_capital": state.account.starting_capital,
                    "ending_capital": round(_account_ending_capital(state), 2),
                    "return_pct": round(_account_return_pct(state), 4),
                    "realized_pnl_usd": round(_realized_pnl_usd(state), 2),
                    "max_drawdown_pct": round(state.account.max_drawdown_pct(), 4),
                    "skipped_insufficient_capital": state.account.skipped_insufficient_capital,
                    "equity_curve": state.account.equity_curve,
                }
            }
            if state.account
            else {}
        ),
    }


def backtest_auto_trader(
    ticker: str,
    *,
    export_dir=EXPORT_DIR,
    lookback_days: int | None = 7,
    max_snapshots: int | None = 500,
    dedupe_identical_strikes: bool = True,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    max_open: int | None = None,
    starting_capital: float | None = None,
    history: list[dict] | None = None,
) -> dict[str, Any]:
    """Simulate auto-trader entries/exits over export snapshot history."""
    ticker = ticker.upper()
    stop_loss = stop_loss if stop_loss is not None else stop_loss_pct()
    take_profit = take_profit if take_profit is not None else take_profit_pct()
    max_open = max_open if max_open is not None else max_open_positions()

    if history is None:
        history = _build_history_impl(
            ticker,
            export_dir,
            lookback_days=lookback_days,
            max_snapshots=max_snapshots,
            dedupe_identical_strikes=dedupe_identical_strikes,
        )

    raw_len = len(history)
    weekend_snapshots_excluded = 0
    if trader_session_only():
        filtered = filter_trading_history(history, session_only=True)
        weekend_snapshots_excluded = raw_len - len(filtered)
        history = filtered

    if len(history) < 2:
        return {
            "ticker": ticker,
            "snapshots": len(history),
            "weekend_snapshots_excluded": weekend_snapshots_excluded,
            "total_trades": 0,
            "message": "Not enough history (need at least 2 snapshots)",
        }

    state = BacktestState()
    if starting_capital is not None and starting_capital > 0:
        state.account = AccountLedger.create(float(starting_capital))

    for idx in range(1, len(history)):
        row = history[idx]
        prev = _signal_previous_row(history, idx)
        ts = str(row["ts"])
        if not export_ts_is_trading_day(ts):
            state.skipped_weekends += 1
            continue

        spot = safe_float(row.get("spot"), 0.0)
        if spot <= 0:
            continue

        _check_exits(
            state,
            idx=idx,
            ts=str(row["ts"]),
            signal_spot=spot,
            row=row,
            prev_ts=str(history[idx - 1]["ts"]),
            prev_signal_spot=safe_float(history[idx - 1].get("spot"), 0.0) or None,
        )
        lookback = momentum_bars() + 2
        spot_history = [
            safe_float(history[j].get("spot"), 0.0)
            for j in range(max(0, idx - lookback + 1), idx + 1)
            if safe_float(history[j].get("spot"), 0.0) > 0
        ]
        _maybe_enter(
            state,
            idx=idx,
            ts=str(row["ts"]),
            row=row,
            prev_row=prev,
            spot_history=spot_history,
            max_open=max_open,
        )

    if state.open_positions:
        last = history[-1]
        last_signal_spot = safe_float(last.get("spot"), 0.0)
        last_idx = len(history) - 1
        last_spot = _mark_spot(last_signal_spot)
        if last_spot and last_spot > 0:
            bar_minutes = _snapshot_bar_minutes(last)
            for pos in list(state.open_positions):
                pnl_pct = _position_pnl(pos, last_spot)
                bars_held = bars_between_timestamps(pos.entry_ts, str(last["ts"]), bar_minutes=bar_minutes)
                _close_position(
                    pos,
                    exit_idx=last_idx,
                    exit_ts=str(last["ts"]),
                    exit_spot=last_spot,
                    pnl_pct=pnl_pct,
                    exit_reason="backtest_end",
                    closed=state.closed_trades,
                    state=state,
                    qty=pos.qty,
                    bars_held=bars_held,
                )
            state.open_positions.clear()
            if state.account:
                state.account.record_equity(str(last["ts"]), state.open_positions, last_spot)

    return _summarize(
        ticker,
        history_len=len(history),
        history=history,
        state=state,
        stop_loss=stop_loss,
        take_profit=take_profit,
        weekend_snapshots_excluded=weekend_snapshots_excluded,
    )


def _clone_snapshot(row: dict, *, ts: str, spot: float) -> dict:
    strike = row.get("strike")
    cloned = {k: v for k, v in row.items() if k not in {"strike", "ts", "spot", "ts_label"}}
    cloned["ts"] = ts
    cloned["ts_label"] = ts
    cloned["spot"] = spot
    if isinstance(strike, pd.Series):
        cloned["strike"] = strike.copy()
    else:
        cloned["strike"] = strike
    return cloned


def _sample_history_block(base: list[dict], rng: random.Random, block_size: int) -> list[dict]:
    if len(base) < 2:
        return list(base)
    max_start = max(1, len(base) - 1)
    start = rng.randint(1, max_start)
    end = min(len(base), start + block_size)
    return [_clone_snapshot(row, ts=str(row["ts"]), spot=float(row.get("spot") or 0.0)) for row in base[start - 1 : end]]


def build_bootstrap_history(
    base: list[dict],
    *,
    target_snapshots: int,
    block_size: int = 24,
    seed: int = 42,
) -> list[dict]:
    """Extend real export snapshots via block resampling with spot/gamma perturbation."""
    if len(base) < 2:
        return list(base)

    rng = random.Random(seed)
    extended: list[dict] = []
    block_idx = 0
    spot = float(base[0].get("spot") or 5000.0)

    while len(extended) < target_snapshots:
        block = _sample_history_block(base, rng, block_size=block_size)
        if not block:
            break

        for step, row in enumerate(block):
            if extended and step == 0:
                continue

            spot *= 1.0 + rng.uniform(-0.004, 0.004)
            ts = f"bootstrap_{block_idx:05d}_{step:04d}"
            orig_spot = safe_float(row.get("spot"), spot)
            snap = _clone_snapshot(row, ts=ts, spot=spot)

            strike = snap.get("strike")
            if isinstance(strike, pd.Series) and not strike.empty:
                jitter = 1.0 + rng.uniform(-0.02, 0.02)
                scaled_index = pd.to_numeric(strike.index, errors="coerce")
                if orig_spot > 0:
                    scale = (spot / orig_spot) * jitter
                    scaled_index = scaled_index * scale
                snap["strike"] = pd.Series(strike.values * jitter, index=scaled_index)

            extended.append(snap)
            if len(extended) >= target_snapshots:
                break

        block_idx += 1

    return extended


def backtest_auto_trader_bootstrap(
    ticker: str,
    *,
    target_trades: int = 1000,
    seed: int = 42,
    export_dir=EXPORT_DIR,
    lookback_days: int | None = 7,
    max_snapshots: int | None = 500,
    dedupe_identical_strikes: bool = True,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    max_open: int | None = None,
    starting_capital: float | None = None,
    max_synthetic_snapshots: int = 400_000,
) -> dict[str, Any]:
    """Bootstrap export history until at least ``target_trades`` close."""
    base = _build_history_impl(
        ticker.upper(),
        export_dir,
        lookback_days=lookback_days,
        max_snapshots=max_snapshots,
        dedupe_identical_strikes=dedupe_identical_strikes,
    )
    if len(base) < 2:
        return {
            "ticker": ticker.upper(),
            "snapshots": len(base),
            "total_trades": 0,
            "message": "Not enough history for bootstrap backtest",
        }

    steps = max(len(base) * 40, 8_000)
    result: dict[str, Any] = {}
    while steps <= max_synthetic_snapshots:
        history = build_bootstrap_history(base, target_snapshots=steps, seed=seed)
        result = backtest_auto_trader(
            ticker,
            history=history,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_open=max_open,
            starting_capital=starting_capital,
        )
        result["bootstrap"] = True
        result["base_snapshots"] = len(base)
        result["target_trades"] = target_trades
        if int(result.get("total_trades", 0)) >= target_trades:
            return result
        steps = int(steps * 1.6)

    result["message"] = (
        f"Reached snapshot cap ({max_synthetic_snapshots:,}) with "
        f"{result.get('total_trades', 0)} trades (target {target_trades})"
    )
    return result
