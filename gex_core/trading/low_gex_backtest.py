"""Walk-forward backtest for the low-GEX strike trader."""

from __future__ import annotations

from typing import Any

import pandas as pd

from gex_core.exports import EXPORT_DIR
from gex_core.features import safe_float
from gex_core.history import _build_history_impl
from gex_core.market_time import export_ts_is_trading_day, filter_trading_history
from gex_core.trading.backtest import (
    AccountLedger,
    BacktestState,
    _OpenPosition,
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
    max_entries_per_cycle,
    max_open_positions,
    stop_loss_pct,
    take_profit_pct,
    trader_session_only,
)
from gex_core.trading.exits import build_exit_profile
from gex_core.trading.low_gex_signals import compute_low_gex_signal
from gex_core.trading.paper_broker import estimate_entry_premium
from gex_core.trading.sizing import affordable_qty, resolve_contract_qty


def _maybe_enter_low_gex(
    state: BacktestState,
    *,
    idx: int,
    ts: str,
    row: dict,
    max_open: int,
) -> None:
    if not export_ts_is_trading_day(ts):
        state.skipped_weekends += 1
        return

    spot = safe_float(row.get("spot"), 0.0)
    exposure = row.get("strike")
    if not isinstance(exposure, pd.Series):
        exposure = None

    pack = compute_low_gex_signal(exposure, spot=spot)
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

    if _has_open_duplicate(state.open_positions, strike=exec_strike, option_type=option_type):
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
            signal_type="min_gamma_strike",
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


def backtest_low_gex_trader(
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
    """Simulate low-GEX wall trades over export snapshot history."""
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

        _check_exits(
            state,
            idx=idx,
            ts=ts,
            signal_spot=spot,
            row=row,
            prev_ts=str(history[idx - 1]["ts"]),
            prev_signal_spot=safe_float(history[idx - 1].get("spot"), 0.0) or None,
        )
        _maybe_enter_low_gex(
            state,
            idx=idx,
            ts=ts,
            row=row,
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

    result = _summarize(
        ticker,
        history_len=len(history),
        history=history,
        state=state,
        stop_loss=stop_loss,
        take_profit=take_profit,
        weekend_snapshots_excluded=weekend_snapshots_excluded,
    )
    result["strategy"] = "low_gex"
    return result
