"""Walk-forward backtest for the low-GEX strike trader."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from gex_core.exports import EXPORT_DIR
from gex_core.features import safe_float
from gex_core.history import _build_history_impl
from gex_core.market_time import (
    export_ts_entry_window_ok,
    export_ts_is_trading_day,
    filter_trading_history,
)
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
    trader_session_only,
    wall_entry_time_filter,
    wall_intraday_session,
    wall_reenter_on_shift,
    wall_reentry_after_stop,
    wall_max_hold_bars,
    wall_signal_filters_enabled,
    wall_stop_loss_pct,
    wall_take_profit_pct,
)
from gex_core.trading.exits import build_simple_exit_profile
from gex_core.trading.low_gex_signals import (
    WallTarget,
    compute_wall_gex_signal,
    wall_entry_quality_ok,
)
from gex_core.trading.paper_broker import estimate_entry_premium
from gex_core.trading.sizing import affordable_qty, resolve_contract_qty


def _flatten_on_wall_shift(
    state: BacktestState,
    *,
    idx: int,
    ts: str,
    signal_spot: float,
    row: dict,
    wall_strike: float,
) -> None:
    """Close positions tied to a prior wall when the GEX wall strike moves."""
    mark_spot = _mark_spot(signal_spot)
    if mark_spot is None or mark_spot <= 0 or not state.open_positions:
        return
    bar_minutes = _snapshot_bar_minutes(row)
    for pos in list(state.open_positions):
        prior_wall = float(pos.magnet_strike or pos.signal_strike or 0.0)
        if prior_wall <= 0 or abs(prior_wall - wall_strike) < 0.5:
            continue
        pnl_pct = _position_pnl(pos, mark_spot)
        bars_held = bars_between_timestamps(pos.entry_ts, ts, bar_minutes=bar_minutes)
        _apply_backtest_exit(
            state,
            pos,
            idx=idx,
            exit_ts=ts,
            exit_spot=mark_spot,
            pnl_pct=pnl_pct,
            exit_reason="wall_shift",
            bars_held=bars_held,
        )
    state.open_positions = [p for p in state.open_positions if p.qty > 0]
    if state.account:
        state.account.record_equity(ts, state.open_positions, mark_spot)


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
    window_pct: float = 0.12,
    reenter_each_bar: bool = False,
    reenter_on_shift: bool | None = None,
    reentry_after_stop: bool | None = None,
    entry_time_filter: bool | None = None,
    signal_filters: bool | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
) -> None:
    if not export_ts_is_trading_day(ts):
        state.skipped_weekends += 1
        return

    spot = safe_float(row.get("spot"), 0.0)
    exposure = row.get("strike")
    if not isinstance(exposure, pd.Series):
        exposure = None

    time_filter = wall_entry_time_filter() if entry_time_filter is None else entry_time_filter
    if time_filter and not export_ts_entry_window_ok(ts):
        state.skipped_filters += 1
        return

    pack = compute_wall_gex_signal(exposure, spot=spot, target=target, window_pct=window_pct)
    if not pack.get("available"):
        state.skipped_entries += 1
        return

    rec = pack["recommended"]
    wall_strike = float(rec.get("wall_strike") or rec["strike"])
    wall_gamma = float(rec["gamma_bn"])
    regime = str(row.get("regime") or "")
    quality_ok, quality_reason = wall_entry_quality_ok(
        wall_strike=wall_strike,
        wall_gamma=wall_gamma,
        regime=regime,
        last_wall_strike=state.last_wall_strike,
        signal_filters=signal_filters,
    )
    state.last_wall_strike = wall_strike
    if not quality_ok:
        reason = quality_reason.lower()
        if "below min" in reason or "|γ|" in quality_reason:
            state.skipped_wall_weak_gamma += 1
        elif "short-gamma" in reason:
            state.skipped_wall_regime += 1
        elif "drift" in reason:
            state.skipped_wall_drift += 1
        else:
            state.skipped_filters += 1
        return

    opened_this_cycle = 0
    if len(state.open_positions) >= max_open:
        return
    if opened_this_cycle >= max_entries_per_cycle():
        return

    option_type = str(rec["option_type"])
    signal_strike = float(rec["strike"])
    trade_ctx = _resolve_trade_context(signal_strike=signal_strike, signal_spot=spot)
    if trade_ctx is None:
        state.skipped_no_execution_spot += 1
        return
    exec_strike, exec_spot, _ = trade_ctx

    shift = wall_reenter_on_shift() if reenter_on_shift is None else reenter_on_shift
    if shift and not reenter_each_bar:
        _flatten_on_wall_shift(
            state,
            idx=idx,
            ts=ts,
            signal_spot=spot,
            row=row,
            wall_strike=wall_strike,
        )

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

    sl = stop_loss if stop_loss is not None else wall_stop_loss_pct()
    tp = take_profit if take_profit is not None else wall_take_profit_pct()
    hold_bars = wall_max_hold_bars()
    profile = build_simple_exit_profile(
        stop_loss=sl,
        take_profit=tp,
        time_stop_bars=hold_bars,
        max_hold_bars=hold_bars,
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
    window_pct: float = 0.12,
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
    reenter_on_shift: bool | None = None,
    reentry_after_stop: bool | None = None,
    intraday_session: bool | None = None,
    entry_time_filter: bool | None = None,
    signal_filters: bool | None = None,
) -> dict[str, Any]:
    """Simulate min/max GEX wall trades over export snapshot history."""
    ticker = ticker.upper()
    stop_loss = stop_loss if stop_loss is not None else wall_stop_loss_pct()
    take_profit = take_profit if take_profit is not None else wall_take_profit_pct()
    max_open = max_open if max_open is not None else max_open_positions()
    rotate = low_gex_reenter_each_bar() if reenter_each_bar is None else reenter_each_bar
    shift = wall_reenter_on_shift() if reenter_on_shift is None else reenter_on_shift
    after_stop = wall_reentry_after_stop() if reentry_after_stop is None else reentry_after_stop
    intraday = wall_intraday_session() if intraday_session is None else intraday_session
    time_filter = wall_entry_time_filter() if entry_time_filter is None else entry_time_filter
    quality_filter = wall_signal_filters_enabled() if signal_filters is None else signal_filters
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
    weekday_len = len(filter_trading_history(history, session_only=True, intraday_only=False))
    history = filter_trading_history(
        history,
        session_only=trader_session_only(),
        intraday_only=intraday,
    )
    weekend_snapshots_excluded = (raw_len - weekday_len) if trader_session_only() else 0
    off_hours_snapshots_excluded = (weekday_len - len(history)) if intraday else 0

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
                apply_stop_cooldown=not after_stop,
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
            window_pct=window_pct,
            reenter_each_bar=rotate,
            reenter_on_shift=shift,
            reentry_after_stop=after_stop,
            entry_time_filter=time_filter,
            signal_filters=quality_filter,
            stop_loss=stop_loss,
            take_profit=take_profit,
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
    result["window_pct"] = window_pct
    result["reenter_each_bar"] = rotate
    result["reenter_on_shift"] = shift
    result["reentry_after_stop"] = after_stop
    result["intraday_session"] = intraday
    result["entry_time_filter"] = time_filter
    result["off_hours_snapshots_excluded"] = off_hours_snapshots_excluded
    result["max_hold_bars"] = wall_max_hold_bars()
    result["signal_filters"] = quality_filter
    result["skipped_wall_weak_gamma"] = state.skipped_wall_weak_gamma
    result["skipped_wall_regime"] = state.skipped_wall_regime
    result["skipped_wall_drift"] = state.skipped_wall_drift
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
    reenter_each_bar: bool = False,
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
