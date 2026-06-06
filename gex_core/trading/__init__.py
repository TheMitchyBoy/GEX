"""Gamma-driven auto-trader (paper mode by default) with AI advisor and trade memory."""

from gex_core.trading.engine import run_trading_cycle, trader_status
from gex_core.trading.journal import get_performance_summary, get_trade_memory_for_ai

__all__ = [
    "run_trading_cycle",
    "trader_status",
    "get_performance_summary",
    "get_trade_memory_for_ai",
]
