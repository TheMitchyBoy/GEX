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
    DEFAULT_WALL_WINDOW_PCT,
    WallGexProfile,
    low_gex_reenter_each_bar,
    max_entries_per_cycle,
    max_open_positions,
    trader_session_only,
    wall_entry_time_filter,
    wall_gex_profile,
    wall_intraday_session,
    wall_reenter_on_shift,
    wall_reentry_after_stop,
    wall_signal_filters_enabled,
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
    min_shift_pts: float = 0.5,
    shift_cooldown_bars: int = 0,
) -> None:
    """Close positions tied to a prior wall when the GEX wall strike moves."""
    mark_spot = _mark_spot(signal_spot)
    if mark_spot is None or mark_spot <= 0 or not state.open_positions:
        return
    bar_minutes = _snapshot_bar_minutes(row)
    shifted = False
    for pos in list(state.open_positions):
        prior_wall = float(pos.magnet_strike or pos.signal_strike or 0.0)
        if prior_wall <= 0 or abs(prior_wall - wall_strike) < max(0.5, float(min_shift_pts)):
            continue
        shifted = True
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
    if shifted and shift_cooldown_bars > 0:
        state.entry_blocked_until_idx = max(state.entry_blocked_until_idx, idx + int(shift_cooldown_bars))
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
    min_gamma_bn: float | None = None,
    min_entry_drift_pts: float | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    max_hold_bars: int | None = None,
    wall_shift_min_pts: float | None = None,
    wall_shift_cooldown_bars: int | None = None,
    profile: WallGexProfile | None = None,
) -> None:
    profile = profile or wall_gex_profile(window_pct)
    if idx < state.entry_blocked_until_idx:
        state.skipped_wall_shift_cooldown += 1
        return

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
        min_gamma_bn=min_gamma_bn,
        min_drift_pts=min_entry_drift_pts,
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

    shift = profile.reenter_on_shift if reenter_on_shift is None else reenter_on_shift
    if shift and not reenter_each_bar:
        _flatten_on_wall_shift(
            state,
            idx=idx,
            ts=ts,
            signal_spot=spot,
            row=row,
            wall_strike=wall_strike,
            min_shift_pts=profile.shift_min_pts if wall_shift_min_pts is None else float(wall_shift_min_pts),
            shift_cooldown_bars=0 if wall_shift_cooldown_bars is None else int(wall_shift_cooldown_bars),
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

    sl = profile.stop_loss_pct if stop_loss is None else stop_loss
    tp = profile.take_profit_pct if take_profit is None else take_profit
    hold_bars = profile.max_hold_bars if max_hold_bars is None else max(1, int(max_hold_bars))
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
    min_gamma_bn: float | None = None,
    min_entry_drift_pts: float | None = None,
    max_hold_bars: int | None = None,
    wall_shift_min_pts: float | None = None,
    wall_shift_cooldown_bars: int | None = None,
) -> dict[str, Any]:
    """Simulate min/max GEX wall trades over export snapshot history."""
    ticker = ticker.upper()
    profile = wall_gex_profile(window_pct)
    stop_loss = profile.stop_loss_pct if stop_loss is None else stop_loss
    take_profit = profile.take_profit_pct if take_profit is None else take_profit
    resolved_hold_bars = profile.max_hold_bars if max_hold_bars is None else max(1, int(max_hold_bars))
    resolved_shift_min_pts = profile.shift_min_pts if wall_shift_min_pts is None else float(wall_shift_min_pts)
    max_open = max_open if max_open is not None else max_open_positions()
    rotate = low_gex_reenter_each_bar() if reenter_each_bar is None else reenter_each_bar
    shift = profile.reenter_on_shift if reenter_on_shift is None else reenter_on_shift
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
            min_gamma_bn=min_gamma_bn,
            min_entry_drift_pts=min_entry_drift_pts,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_hold_bars=resolved_hold_bars,
            wall_shift_min_pts=resolved_shift_min_pts,
            wall_shift_cooldown_bars=wall_shift_cooldown_bars,
            profile=profile,
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
    result["max_hold_bars"] = resolved_hold_bars
    result["near_wall"] = profile.near
    result["signal_filters"] = quality_filter
    result["min_gamma_bn"] = min_gamma_bn
    result["min_entry_drift_pts"] = min_entry_drift_pts
    result["wall_shift_min_pts"] = resolved_shift_min_pts
    result["wall_shift_cooldown_bars"] = wall_shift_cooldown_bars
    result["skipped_wall_weak_gamma"] = state.skipped_wall_weak_gamma
    result["skipped_wall_regime"] = state.skipped_wall_regime
    result["skipped_wall_drift"] = state.skipped_wall_drift
    result["skipped_wall_shift_cooldown"] = state.skipped_wall_shift_cooldown
    result["blocked_duplicate"] = state.blocked_duplicate
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
    window_pct: float = DEFAULT_WALL_WINDOW_PCT,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    max_hold_bars: int | None = None,
    max_snapshots: int | None = 5000,
    dedupe_identical_strikes: bool | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run min vs max wall GEX backtests on the same history window."""
    profile = wall_gex_profile(window_pct)
    if dedupe_identical_strikes is None:
        # Near-window profiles change slowly; dedupe collapses too many bars.
        dedupe_identical_strikes = False if profile.near else not reenter_each_bar
    history = _build_history_impl(
        ticker.upper(),
        EXPORT_DIR,
        lookback_days=lookback_days,
        max_snapshots=max_snapshots,
        dedupe_identical_strikes=dedupe_identical_strikes,
    )
    shared = {
        "history": history,
        "lookback_days": lookback_days,
        "starting_capital": starting_capital,
        "reenter_each_bar": reenter_each_bar,
        "window_pct": window_pct,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "max_hold_bars": max_hold_bars,
        **{
            k: v
            for k, v in kwargs.items()
            if k
            not in (
                "max_snapshots",
                "dedupe_identical_strikes",
                "window_pct",
                "stop_loss",
                "take_profit",
                "max_hold_bars",
            )
        },
    }
    low = backtest_wall_gex_trader(ticker, target="min", **shared)
    high = backtest_wall_gex_trader(ticker, target="max", **shared)
    low_return = (low.get("account") or {}).get("return_pct")
    high_return = (high.get("account") or {}).get("return_pct")
    if low_return is not None and high_return is not None:
        if low_return > high_return + 0.005:
            recommended = "min"
        elif high_return > low_return + 0.005:
            recommended = "max"
        else:
            recommended = "tie"
    else:
        recommended = None
    return {
        "ticker": ticker.upper(),
        "lookback_days": lookback_days,
        "window_pct": window_pct,
        "near_wall": profile.near,
        "dedupe_identical_strikes": dedupe_identical_strikes,
        "snapshots": low.get("snapshots", 0),
        "date_from": low.get("date_from"),
        "date_to": low.get("date_to"),
        "reenter_each_bar": reenter_each_bar,
        "stop_loss_pct": low.get("stop_loss_pct"),
        "take_profit_pct": low.get("take_profit_pct"),
        "max_hold_bars": low.get("max_hold_bars"),
        "reenter_on_shift": low.get("reenter_on_shift"),
        "recommended_side": recommended,
        "low_gex": low,
        "high_gex": high,
        "comparison": {
            "low_trades": low.get("total_trades", 0),
            "high_trades": high.get("total_trades", 0),
            "low_win_rate": low.get("win_rate"),
            "high_win_rate": high.get("win_rate"),
            "low_pnl_usd": low.get("total_pnl_usd"),
            "high_pnl_usd": high.get("total_pnl_usd"),
            "low_return_pct": low_return,
            "high_return_pct": high_return,
            "low_max_dd": (low.get("account") or {}).get("max_drawdown_pct"),
            "high_max_dd": (high.get("account") or {}).get("max_drawdown_pct"),
            "low_skipped_distance": low.get("skipped_strike_distance", 0),
            "high_skipped_distance": high.get("skipped_strike_distance", 0),
        },
    }
