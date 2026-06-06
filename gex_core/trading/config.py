"""Auto-trader configuration from environment."""

from __future__ import annotations

import os


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def auto_trader_enabled() -> bool:
    return _flag("GEX_AUTO_TRADER", "0")


def paper_trading_only() -> bool:
    return _flag("GEX_TRADER_PAPER", "1")


def stop_loss_pct() -> float:
    try:
        return max(0.01, float(os.environ.get("GEX_TRADER_STOP_LOSS_PCT", "0.05")))
    except (TypeError, ValueError):
        return 0.05


def take_profit_pct() -> float:
    try:
        return max(0.05, float(os.environ.get("GEX_TRADER_TAKE_PROFIT_PCT", "0.35")))
    except (TypeError, ValueError):
        return 0.35


def max_open_positions() -> int:
    try:
        return max(1, int(os.environ.get("GEX_TRADER_MAX_OPEN", "3")))
    except (TypeError, ValueError):
        return 3


def min_ai_confidence() -> float:
    try:
        return float(os.environ.get("GEX_TRADER_MIN_AI_CONFIDENCE", "0.55"))
    except (TypeError, ValueError):
        return 0.55


def option_leverage() -> float:
    """Rough ATM option leverage for paper PnL vs underlying move."""
    try:
        return float(os.environ.get("GEX_TRADER_OPTION_LEVERAGE", "12"))
    except (TypeError, ValueError):
        return 12.0
