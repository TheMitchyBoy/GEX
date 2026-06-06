"""Walk-forward backtest for the gamma auto-trader."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gex_core.exports import EXPORT_DIR
from gex_core.features import safe_float
from gex_core.history import _build_history_impl
from gex_core.trading.advisor import _rule_based_advice
from gex_core.trading.config import (
    max_open_positions,
    min_ai_confidence,
    stop_loss_pct,
    take_profit_pct,
)
from gex_core.trading.paper_broker import (
    estimate_entry_premium,
    estimate_option_pnl_pct,
    mark_to_market_premium,
    pnl_usd,
)
from gex_core.trading.signals import compute_gamma_signals


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


@dataclass
class BacktestState:
    open_positions: list[_OpenPosition] = field(default_factory=list)
    closed_trades: list[_ClosedTrade] = field(default_factory=list)
    skipped_entries: int = 0


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


def _close_position(
    pos: _OpenPosition,
    *,
    exit_idx: int,
    exit_ts: str,
    exit_spot: float,
    pnl_pct: float,
    exit_reason: str,
    closed: list[_ClosedTrade],
) -> None:
    exit_premium = mark_to_market_premium(pos.entry_premium, pnl_pct)
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
            pnl_usd=pnl_usd(pos.entry_premium, exit_premium),
            exit_reason=exit_reason,
            bars_held=exit_idx - pos.entry_idx,
        )
    )


def _check_exits(
    state: BacktestState,
    *,
    idx: int,
    ts: str,
    spot: float,
    stop_loss: float,
    take_profit: float,
) -> None:
    remaining: list[_OpenPosition] = []
    for pos in state.open_positions:
        pnl_pct = _position_pnl(pos, spot)
        exit_reason = None
        if pnl_pct <= -stop_loss:
            exit_reason = "stop_loss"
        elif pnl_pct >= take_profit:
            exit_reason = "take_profit"
        if exit_reason:
            _close_position(
                pos,
                exit_idx=idx,
                exit_ts=ts,
                exit_spot=spot,
                pnl_pct=pnl_pct,
                exit_reason=exit_reason,
                closed=state.closed_trades,
            )
        else:
            remaining.append(pos)
    state.open_positions = remaining


def _maybe_enter(
    state: BacktestState,
    *,
    idx: int,
    ts: str,
    spot: float,
    exposure,
    previous,
    max_open: int,
    min_confidence: float,
) -> None:
    if len(state.open_positions) >= max_open:
        return

    signals = compute_gamma_signals(exposure, previous, spot=spot)
    if not signals.get("available"):
        return

    memory = _memory_from_closed(state.closed_trades)
    advice = _rule_based_advice(signals, memory)
    if not advice.get("approve") or float(advice.get("confidence", 0)) < min_confidence:
        state.skipped_entries += 1
        return

    rec = signals["recommended"]
    option_type = advice.get("option_type") or rec["option_type"]
    strike = float(rec["strike"])
    premium = estimate_entry_premium(spot, strike)
    state.open_positions.append(
        _OpenPosition(
            entry_idx=idx,
            entry_ts=ts,
            option_type=str(option_type),
            strike=strike,
            entry_spot=spot,
            entry_premium=premium,
            signal_type=str(rec["signal_type"]),
            signal_gamma=float(rec["gamma_bn"]),
            gamma_delta=float(rec["gamma_delta"]),
            ai_confidence=float(advice.get("confidence", 0.5)),
        )
    )


def _summarize(
    ticker: str,
    *,
    history_len: int,
    state: BacktestState,
    stop_loss: float,
    take_profit: float,
) -> dict[str, Any]:
    trades = state.closed_trades
    if not trades:
        return {
            "ticker": ticker,
            "snapshots": history_len,
            "total_trades": 0,
            "open_at_end": len(state.open_positions),
            "skipped_entries": state.skipped_entries,
            "message": "No trades triggered in walk-forward window",
            "stop_loss_pct": stop_loss,
            "take_profit_pct": take_profit,
        }

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
        "stop_loss_pct": stop_loss,
        "take_profit_pct": take_profit,
        "trades": [
            {
                "entry_ts": t.entry_ts,
                "exit_ts": t.exit_ts,
                "signal_type": t.signal_type,
                "option_type": t.option_type,
                "strike": t.strike,
                "pnl_pct": round(t.pnl_pct, 4),
                "pnl_usd": round(t.pnl_usd, 2),
                "exit_reason": t.exit_reason,
                "bars_held": t.bars_held,
            }
            for t in trades
        ],
    }


def backtest_auto_trader(
    ticker: str,
    *,
    export_dir=EXPORT_DIR,
    lookback_days: int | None = 7,
    max_snapshots: int | None = 500,
    dedupe_identical_strikes: bool = False,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    max_open: int | None = None,
    min_confidence: float | None = None,
    history: list[dict] | None = None,
) -> dict[str, Any]:
    """Simulate auto-trader entries/exits over export snapshot history."""
    ticker = ticker.upper()
    stop_loss = stop_loss if stop_loss is not None else stop_loss_pct()
    take_profit = take_profit if take_profit is not None else take_profit_pct()
    max_open = max_open if max_open is not None else max_open_positions()
    min_confidence = min_confidence if min_confidence is not None else min_ai_confidence()

    if history is None:
        history = _build_history_impl(
            ticker,
            export_dir,
            lookback_days=lookback_days,
            max_snapshots=max_snapshots,
            dedupe_identical_strikes=dedupe_identical_strikes,
        )

    if len(history) < 2:
        return {
            "ticker": ticker,
            "snapshots": len(history),
            "total_trades": 0,
            "message": "Not enough history (need at least 2 snapshots)",
        }

    state = BacktestState()
    for idx in range(1, len(history)):
        row = history[idx]
        prev = history[idx - 1]
        spot = safe_float(row.get("spot"), 0.0)
        if spot <= 0:
            continue

        _check_exits(
            state,
            idx=idx,
            ts=str(row["ts"]),
            spot=spot,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        _maybe_enter(
            state,
            idx=idx,
            ts=str(row["ts"]),
            spot=spot,
            exposure=row.get("strike"),
            previous=prev.get("strike"),
            max_open=max_open,
            min_confidence=min_confidence,
        )

    if state.open_positions:
        last = history[-1]
        last_spot = safe_float(last.get("spot"), 0.0)
        last_idx = len(history) - 1
        if last_spot > 0:
            for pos in list(state.open_positions):
                pnl_pct = _position_pnl(pos, last_spot)
                _close_position(
                    pos,
                    exit_idx=last_idx,
                    exit_ts=str(last["ts"]),
                    exit_spot=last_spot,
                    pnl_pct=pnl_pct,
                    exit_reason="backtest_end",
                    closed=state.closed_trades,
                )
            state.open_positions.clear()

    return _summarize(
        ticker,
        history_len=len(history),
        state=state,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )
