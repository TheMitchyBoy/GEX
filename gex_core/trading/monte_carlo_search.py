"""Shared Monte Carlo helpers for auto-trader configuration search."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from gex_core.trading.backtest import backtest_auto_trader


@dataclass(frozen=True)
class TraderConfig:
    name: str
    env: dict[str, str] = field(default_factory=dict)
    stop_loss: float | None = None
    take_profit: float | None = None
    max_open: int | None = None


_TRADER_ENV_KEYS = (
    "GEX_TRADER_STRICT_FILTERS",
    "GEX_TRADER_MIN_GAMMA_DELTA",
    "GEX_TRADER_MIN_FASTEST_GAMMA_DELTA",
    "GEX_TRADER_STOP_LOSS_PCT",
    "GEX_TRADER_TAKE_PROFIT_PCT",
    "GEX_TRADER_PARTIAL_TP_PCT",
    "GEX_TRADER_TIME_STOP_BARS",
    "GEX_TRADER_STOP_COOLDOWN_BARS",
    "GEX_TRADER_MAX_OPEN",
    "GEX_TRADER_REQUIRE_MOMENTUM",
    "GEX_TRADER_REQUIRE_FLOW_ALIGN",
    "GEX_TRADER_REQUIRE_FLIP_SIDE",
    "GEX_TRADER_ENTRY_TIME_FILTER",
    "GEX_TRADER_MIN_ZERO_DTE_RATIO",
    "GEX_TRADER_MAX_IV_RANK",
    "GEX_TRADER_MOMENTUM_BARS",
    "GEX_TRADER_MIN_MAGNET_PROGRESS",
    "GEX_TRADER_MIN_FLOW_BUY_RATIO",
    "GEX_TRADER_BLOCK_EVENTS",
    "GEX_TRADER_EVENT_SIZE_MULT",
    "GEX_TRADER_RISK_SIZING",
    "GEX_TRADER_RISK_PER_TRADE_PCT",
    "GEX_TRADER_MAGNET_TOUCH_EXIT",
    "GEX_TRADER_EOD_FLATTEN",
    "GEX_TRADER_BAR_MINUTES",
    "GEX_TRADER_MIN_ENTRY_CONFIDENCE",
    "GEX_TRADER_STRONG_CONFIDENCE",
)


@contextmanager
def trader_env(config: TraderConfig) -> Iterator[None]:
    saved = {key: os.environ.get(key) for key in _TRADER_ENV_KEYS}
    try:
        for key in _TRADER_ENV_KEYS:
            os.environ.pop(key, None)
        for key, value in config.env.items():
            os.environ[key] = value
        yield
    finally:
        for key in _TRADER_ENV_KEYS:
            if saved[key] is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = saved[key]


def score_result(result: dict[str, Any]) -> float:
    """Rank configs: ROI and PnL first; penalize zero-trade and session-gap-only runs."""
    trades = int(result.get("total_trades") or 0)
    if trades == 0:
        return -1e9

    account = result.get("account") or {}
    ret = float(account.get("return_pct") or result.get("return_pct") or 0.0)
    dd = float(account.get("max_drawdown_pct") or 0.0)
    win_rate = float(result.get("win_rate") or 0.0)
    pnl = float(result.get("total_pnl_usd") or 0.0)

    by_exit = result.get("by_exit_reason") or {}
    session_gap = int(by_exit.get("session_gap") or 0)
    meaningful = trades - session_gap

    if meaningful <= 0 and pnl == 0.0 and ret == 0.0:
        return -1e6 + trades

    activity = min(meaningful if meaningful > 0 else trades, 20) / 20.0
    return ret * 1000.0 + pnl * 0.5 + win_rate * 50.0 - dd * 200.0 + activity * 10.0


def run_config_trial(
    config: TraderConfig,
    *,
    ticker: str,
    history: list[dict],
    starting_capital: float,
) -> dict[str, Any]:
    with trader_env(config):
        result = backtest_auto_trader(
            ticker,
            history=history,
            starting_capital=starting_capital,
            stop_loss=config.stop_loss,
            take_profit=config.take_profit,
            max_open=config.max_open,
        )
    account = result.get("account") or {}
    by_exit = result.get("by_exit_reason") or {}
    session_gap = int(by_exit.get("session_gap") or 0)
    total_trades = int(result.get("total_trades") or 0)
    return {
        "name": config.name,
        "score": score_result(result),
        "total_trades": total_trades,
        "meaningful_trades": max(0, total_trades - session_gap),
        "session_gap_trades": session_gap,
        "win_rate": result.get("win_rate"),
        "total_pnl_usd": result.get("total_pnl_usd", 0.0),
        "avg_pnl_pct": result.get("avg_pnl_pct"),
        "return_pct": account.get("return_pct", 0.0),
        "ending_capital": account.get("ending_capital"),
        "max_drawdown_pct": account.get("max_drawdown_pct", 0.0),
        "skipped_low_confidence": result.get("skipped_low_confidence", 0),
        "config": {
            "env": dict(config.env),
            "stop_loss": config.stop_loss,
            "take_profit": config.take_profit,
            "max_open": config.max_open,
        },
        "by_exit_reason": result.get("by_exit_reason"),
        "date_from": result.get("date_from"),
        "date_to": result.get("date_to"),
    }
