"""Walk-forward backtest for the low-GEX strike trader."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from gex_core.exports import EXPORT_DIR
from gex_core.features import safe_float
from gex_core.history import _build_history_impl
from gex_core.market_time import export_ts_is_trading_day, filter_trading_history
from gex_core.trading.backtest import (
    AccountLedger,
    BacktestState,
    _OpenPosition,
    _apply_backtest_exit,
    _check_exits,
    _close_position,
    _has_open_duplicate,
    _mark_spot,
    _position_pnl,
    _resolve_trade_context,
    _snapshot_bar_minutes,
    _summarize,
    bars_between_timestamps,
)
from gex_core.trading.config import (
    low_gex_reenter_each_bar,
    max_entries_per_cycle,
    max_open_positions,
    stop_loss_pct,
    take_profit_pct,
    trader_session_only,
)
from gex_core.trading.exits import build_exit_profile
from gex_core.trading.low_gex_signals import WallTarget, compute_wall_gex_signal
from gex_core.trading.paper_broker import estimate_entry_premium
from gex_core.trading.sizing import affordable_qty, resolve_contract_qty


def _flatten_for_reentry(
    state: BacktestState,
    *,
    idx: int,
    ts: str,
    signal_spot: float,
    row: dict,
) -> None:
    """Close all open positions at the current mark before a new bar entry."""
    mark_spot = _mark_spot(signal_spot)
    if mark_spot is None or mark_spot <= 0 or not state.open_positions:
        return
    bar_minutes = _snapshot_bar_minutes(row)
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
            exit_reason="bar_rotation",
            bars_held=bars_held,
        )
    state.open_positions.clear()
    if state.account:
        state.account.record_equity(ts, state.open_positions, mark_spot)


def _maybe_enter_wall_gex(
    state: BacktestState,
    *,
    idx: int,
    ts: str,
    row: dict,
    max_open: int,
    target: WallTarget = "min",
    reenter_each_bar: bool = False,
) -> None:
    if not export_ts_is_trading_day(ts):
        state.skipped_weekends += 1
        return

    spot = safe_float(row.get("spot"), 0.0)
    exposure = row.get("strike")
    if not isinstance(exposure, pd.Series):
        exposure = None

    pack = compute_wall_gex_signal(exposure, spot=spot, target=target)
    if not pack.get("available"):
        skip = str(pack.get("reason", ""))
        if "from spot" in skip:
            state.skipped_strike_distance += 1
        else:
            state.skipped_entries += 1
        return

    rec = pack["recommended"]
    opened_this_cycle = 0
    if len(state.open_positions) >= max_open:
        return
    if opened_this_cycle >= max_entries_per_cycle():
        return

    option_type = str(rec["option_type"])
    signal_strike = float(rec["strike"])
    wall_strike = float(rec.get("wall_strike") or signal_strike)
    trade_ctx = _resolve_trade_context(signal_strike=signal_strike, signal_spot=spot)
    if trade_ctx is None:
        state.skipped_no_execution_spot += 1
        return
    exec_strike, exec_spot, _ = trade_ctx

    if not reenter_each_bar and _has_open_duplicate(
        state.open_positions, strike=exec_strike, option_type=option_type
    ):
        state.blocked_duplicate += 1
        return

    confidence = 0.65
    premium = estimate_entry_premium(exec_spot, exec_strike)
    equity = state.account.cash if state.account else None
    qty = float(
        resolve_contract_qty(
            confidence=confidence,
            premium=premium,
            entry_spot=exec_spot,
            strike=exec_strike,
            account_equity=equity,
            size_multiplier=1.0,
        )
    )
    if state.account:
        qty = float(affordable_qty(premium, state.account.cash, qty))
        if qty < 1:
            state.account.skipped_insufficient_capital += 1
            return
        state.account.debit_entry(premium, qty)
    elif qty < 1:
        state.skipped_entries += 1
        return

    regime = str(row.get("regime") or "")
    expected_move = safe_float(row.get("expected_move_pct"), 0.0) or None
    profile = build_exit_profile(
        ai_confidence=confidence,
        gamma_delta=0.0,
        regime=regime,
        entry_spot=exec_spot,
        strike=exec_strike,
        expected_move_pct=expected_move,
        magnet_strike=wall_strike,
    )
    state.open_positions.append(
        _OpenPosition(
            entry_idx=idx,
            entry_ts=ts,
            option_type=option_type,
            strike=exec_strike,
            entry_spot=exec_spot,
            entry_premium=premium,
            signal_type=str(rec.get("signal_type") or ("max_gamma_strike" if target == "max" else "min_gamma_strike")),
            signal_gamma=float(rec["gamma_bn"]),
            gamma_delta=0.0,
            ai_confidence=confidence,
            signal_strike=signal_strike,
            magnet_strike=wall_strike,
            qty=qty,
            exit_profile=profile,
        )
    )
    if state.account:
        state.account.record_equity(ts, state.open_positions, exec_spot)


def backtest_wall_gex_trader(
    ticker: str,
    *,
    target: WallTarget = "min",
    export_dir=EXPORT_DIR,
    lookback_days: int | None = 7,
    max_snapshots: int | None = 500,
    dedupe_identical_strikes: bool = True,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    max_open: int | None = None,
    starting_capital: float | None = None,
    history: list[dict] | None = None,
    reenter_each_bar: bool | None = None,
) -> dict[str, Any]:
    """Simulate min/max GEX wall trades over export snapshot history."""
    ticker = ticker.upper()
    stop_loss = stop_loss if stop_loss is not None else stop_loss_pct()
    take_profit = take_profit if take_profit is not None else take_profit_pct()
    max_open = max_open if max_open is not None else max_open_positions()
    rotate = low_gex_reenter_each_bar() if reenter_each_bar is None else reenter_each_bar
    if rotate:
        max_open = max(max_open, max_entries_per_cycle())

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
            "strategy": "low_gex",
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
        ts = str(row["ts"])
        if not export_ts_is_trading_day(ts):
            state.skipped_weekends += 1
            continue

        spot = safe_float(row.get("spot"), 0.0)
        if spot <= 0:
            continue

        if not rotate:
            _check_exits(
                state,
                idx=idx,
                ts=ts,
                signal_spot=spot,
                row=row,
                prev_ts=str(history[idx - 1]["ts"]),
                prev_signal_spot=safe_float(history[idx - 1].get("spot"), 0.0) or None,
            )
        else:
            _flatten_for_reentry(state, idx=idx, ts=ts, signal_spot=spot, row=row)
        _maybe_enter_wall_gex(
            state,
            idx=idx,
            ts=ts,
            row=row,
            max_open=max_open,
            target=target,
            reenter_each_bar=rotate,
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

    result = _summarize(
        ticker,
        history_len=len(history),
        history=history,
        state=state,
        stop_loss=stop_loss,
        take_profit=take_profit,
        weekend_snapshots_excluded=weekend_snapshots_excluded,
    )
    result["strategy"] = "low_gex" if target == "min" else "high_gex"
    result["wall_target"] = target
    result["reenter_each_bar"] = rotate
    return result


def backtest_low_gex_trader(
    ticker: str,
    **kwargs: Any,
) -> dict[str, Any]:
    return backtest_wall_gex_trader(ticker, target="min", **kwargs)


def backtest_high_gex_trader(
    ticker: str,
    **kwargs: Any,
) -> dict[str, Any]:
    return backtest_wall_gex_trader(ticker, target="max", **kwargs)


def compare_wall_gex_backtest(
    ticker: str,
    *,
    lookback_days: int = 7,
    starting_capital: float = 500.0,
    reenter_each_bar: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run min vs max wall GEX backtests on the same history window."""
    dedupe = kwargs.get("dedupe_identical_strikes")
    if dedupe is None:
        dedupe = not reenter_each_bar
    history = _build_history_impl(
        ticker.upper(),
        EXPORT_DIR,
        lookback_days=lookback_days,
        max_snapshots=kwargs.get("max_snapshots", 5000),
        dedupe_identical_strikes=dedupe,
    )
    shared = {
        "history": history,
        "lookback_days": lookback_days,
        "starting_capital": starting_capital,
        "reenter_each_bar": reenter_each_bar,
        **{k: v for k, v in kwargs.items() if k not in ("max_snapshots", "dedupe_identical_strikes")},
    }
    low = backtest_wall_gex_trader(ticker, target="min", **shared)
    high = backtest_wall_gex_trader(ticker, target="max", **shared)
    return {
        "ticker": ticker.upper(),
        "lookback_days": lookback_days,
        "snapshots": low.get("snapshots", 0),
        "date_from": low.get("date_from"),
        "date_to": low.get("date_to"),
        "reenter_each_bar": reenter_each_bar,
        "low_gex": low,
        "high_gex": high,
        "comparison": {
            "low_trades": low.get("total_trades", 0),
            "high_trades": high.get("total_trades", 0),
            "low_win_rate": low.get("win_rate"),
            "high_win_rate": high.get("win_rate"),
            "low_pnl_usd": low.get("total_pnl_usd"),
            "high_pnl_usd": high.get("total_pnl_usd"),
            "low_return_pct": (low.get("account") or {}).get("return_pct"),
            "high_return_pct": (high.get("account") or {}).get("return_pct"),
            "low_max_dd": (low.get("account") or {}).get("max_drawdown_pct"),
            "high_max_dd": (high.get("account") or {}).get("max_drawdown_pct"),
            "low_skipped_distance": low.get("skipped_strike_distance", 0),
            "high_skipped_distance": high.get("skipped_strike_distance", 0),
        },
    }
