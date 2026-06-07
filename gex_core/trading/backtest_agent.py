"""Agent-facing auto-trader backtest helpers using live configuration."""

from __future__ import annotations

import os
from typing import Any

from gex_core.trading.backtest import backtest_auto_trader
from gex_core.trading.config import (
    account_equity_usd,
    auto_trader_enabled,
    block_event_days,
    entry_time_filter_enabled,
    entry_window_after_open_min,
    entry_window_before_close_min,
    eod_flatten_enabled,
    execution_ticker,
    max_entries_per_cycle,
    max_open_positions,
    max_strike_distance_pct,
    min_flow_buy_ratio,
    min_gamma_delta,
    multi_strike_count,
    paper_trading_only,
    require_flow_alignment,
    require_spot_momentum,
    risk_per_trade_pct,
    signal_ticker,
    stop_loss_pct,
    strict_entry_filters,
    take_profit_pct,
    trader_bar_minutes,
    trader_session_only,
    use_risk_based_sizing,
    webull_underlying,
)


def _default_lookback_days() -> int:
    try:
        return max(1, int(os.environ.get("GEX_BACKTEST_LOOKBACK_DAYS", "7")))
    except (TypeError, ValueError):
        return 7


def _default_max_snapshots() -> int:
    try:
        return max(10, int(os.environ.get("GEX_BACKTEST_MAX_SNAPSHOTS", "500")))
    except (TypeError, ValueError):
        return 500


def current_trader_parameters() -> dict[str, Any]:
    """Snapshot of the auto-trader knobs currently in effect (from env/config)."""
    return {
        "auto_trader_enabled": auto_trader_enabled(),
        "paper_mode": paper_trading_only(),
        "signal_ticker": signal_ticker(),
        "execution_ticker": execution_ticker(),
        "webull_underlying": webull_underlying(),
        "stop_loss_pct": stop_loss_pct(),
        "take_profit_pct": take_profit_pct(),
        "max_open_positions": max_open_positions(),
        "max_entries_per_cycle": max_entries_per_cycle(),
        "multi_strike_count": multi_strike_count(),
        "min_gamma_delta": min_gamma_delta(),
        "max_strike_distance_pct": max_strike_distance_pct(),
        "strict_entry_filters": strict_entry_filters(),
        "require_flow_alignment": require_flow_alignment(),
        "min_flow_buy_ratio": min_flow_buy_ratio(),
        "require_spot_momentum": require_spot_momentum(),
        "block_event_days": block_event_days(),
        "entry_time_filter": entry_time_filter_enabled(),
        "entry_after_open_min": entry_window_after_open_min(),
        "entry_before_close_min": entry_window_before_close_min(),
        "eod_flatten": eod_flatten_enabled(),
        "trader_session_only": trader_session_only(),
        "bar_minutes": trader_bar_minutes(),
        "risk_sizing": use_risk_based_sizing(),
        "risk_per_trade_pct": risk_per_trade_pct(),
        "account_equity_usd": account_equity_usd(),
    }


def summarize_backtest_for_ai(result: dict[str, Any], *, include_trades: int = 5) -> dict[str, Any]:
    """Compact walk-forward summary suitable for LLM prompts."""
    trades = result.get("trades") or []
    tail = trades[-include_trades:] if include_trades > 0 else []
    return {
        "parameters": result.get("parameters") or current_trader_parameters(),
        "window": {
            "from": result.get("date_from"),
            "to": result.get("date_to"),
            "snapshots": result.get("snapshots"),
            "weekend_snapshots_excluded": result.get("weekend_snapshots_excluded"),
        },
        "message": result.get("message"),
        "total_trades": result.get("total_trades", 0),
        "wins": result.get("wins"),
        "losses": result.get("losses"),
        "win_rate": result.get("win_rate"),
        "avg_pnl_pct": result.get("avg_pnl_pct"),
        "total_pnl_usd": result.get("total_pnl_usd"),
        "avg_bars_held": result.get("avg_bars_held"),
        "by_signal": result.get("by_signal"),
        "by_exit_reason": result.get("by_exit_reason"),
        "account": result.get("account"),
        "skipped": {
            "entries": result.get("skipped_entries"),
            "filters": result.get("skipped_filters"),
            "gamma_decline": result.get("skipped_gamma_decline"),
            "strike_distance": result.get("skipped_strike_distance"),
            "cooldown": result.get("blocked_cooldown"),
            "duplicate": result.get("blocked_duplicate"),
        },
        "recent_trades": tail,
        "execution_ticker": result.get("execution_ticker"),
        "stop_loss_pct": result.get("stop_loss_pct"),
        "take_profit_pct": result.get("take_profit_pct"),
    }


def run_agent_backtest(
    ticker: str = "SPX",
    *,
    lookback_days: int | None = None,
    max_snapshots: int | None = None,
    starting_capital: float | None = None,
    compact: bool = True,
) -> dict[str, Any]:
    """Walk-forward backtest using the current trader configuration."""
    ticker = ticker.upper()
    params = current_trader_parameters()
    lookback = lookback_days if lookback_days is not None else _default_lookback_days()
    max_snaps = max_snapshots if max_snapshots is not None else _default_max_snapshots()
    capital = starting_capital if starting_capital is not None else params["account_equity_usd"]

    result = backtest_auto_trader(
        ticker,
        lookback_days=lookback,
        max_snapshots=max_snaps,
        starting_capital=capital,
    )
    result["parameters"] = params
    if compact:
        return summarize_backtest_for_ai(result)
    return result


def user_wants_backtest(message: str) -> bool:
    msg = (message or "").lower()
    triggers = (
        "backtest",
        "walk forward",
        "walk-forward",
        "simulate strategy",
        "simulate the strategy",
        "how would this strategy",
        "how would the strategy",
        "win rate on history",
        "historical performance",
        "past performance",
        "strategy performance",
        "test current parameters",
        "test these parameters",
        "run simulation",
    )
    return any(token in msg for token in triggers)


def format_backtest_reply(summary: dict[str, Any]) -> str:
    """Human-readable backtest blurb for rule-based chat fallback."""
    if summary.get("message") and not summary.get("total_trades"):
        return (
            f"Walk-forward backtest ({summary['parameters']['signal_ticker']} signals): "
            f"{summary['message']} "
            f"({summary['window'].get('snapshots', 0)} snapshots)."
        )

    account = summary.get("account") or {}
    win_rate = summary.get("win_rate")
    win_txt = f"{win_rate:.0%}" if isinstance(win_rate, (int, float)) else "n/a"
    ret = account.get("return_pct")
    ret_txt = f"{ret:+.1%}" if isinstance(ret, (int, float)) else "n/a"
    pnl = summary.get("total_pnl_usd")
    pnl_txt = f"${pnl:,.2f}" if isinstance(pnl, (int, float)) else "n/a"

    window = summary.get("window") or {}
    return (
        f"Walk-forward backtest on {window.get('snapshots', '?')} snapshots "
        f"({window.get('from', '?')} → {window.get('to', '?')}) "
        f"with current settings (risk {summary['parameters']['risk_per_trade_pct']:.0%}, "
        f"stop {summary['parameters']['stop_loss_pct']:.0%}, "
        f"target {summary['parameters']['take_profit_pct']:.0%}): "
        f"{summary.get('total_trades', 0)} trades, win rate {win_txt}, "
        f"total PnL {pnl_txt}, account return {ret_txt}."
    )
