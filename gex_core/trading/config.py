"""Auto-trader configuration from environment."""

from __future__ import annotations

import os

from gex_core.env_bootstrap import parse_env_minutes


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def auto_trader_enabled() -> bool:
    return _flag("GEX_AUTO_TRADER", "0")


def trader_cycle_seconds() -> int:
    """How often the auto-trader evaluates exits/entries (0 = only on gamma refresh)."""
    try:
        return max(0, int(os.environ.get("GEX_TRADER_CYCLE_SECONDS", "15")))
    except (TypeError, ValueError):
        return 15


def trader_bar_minutes() -> float:
    """Minutes per gamma bar for time stops and cooldown semantics."""
    explicit = os.environ.get("GEX_TRADER_BAR_MINUTES", "").strip()
    if explicit:
        try:
            return max(0.1, float(explicit))
        except (TypeError, ValueError):
            pass
    return max(0.1, parse_env_minutes("GEX_REFRESH_INTERVAL_MINUTES", 2.0))


def trader_session_only() -> bool:
    return _flag("GEX_TRADER_SESSION_ONLY", "1")


def paper_trading_only() -> bool:
    return _flag("GEX_TRADER_PAPER", "1")


def stop_loss_pct() -> float:
    try:
        return max(0.01, float(os.environ.get("GEX_TRADER_STOP_LOSS_PCT", "0.05")))
    except (TypeError, ValueError):
        return 0.05


def far_otm_stop_loss_pct() -> float:
    try:
        return max(0.01, float(os.environ.get("GEX_TRADER_FAR_OTM_STOP_PCT", "0.03")))
    except (TypeError, ValueError):
        return 0.03


def far_otm_distance_pct() -> float:
    try:
        return max(0.001, float(os.environ.get("GEX_TRADER_FAR_OTM_DISTANCE_PCT", "0.01")))
    except (TypeError, ValueError):
        return 0.01


def take_profit_pct() -> float:
    try:
        return max(0.05, float(os.environ.get("GEX_TRADER_TAKE_PROFIT_PCT", "0.35")))
    except (TypeError, ValueError):
        return 0.35


def partial_take_profit_pct() -> float:
    try:
        return max(0.05, float(os.environ.get("GEX_TRADER_PARTIAL_TP_PCT", "0.08")))
    except (TypeError, ValueError):
        return 0.08


def trailing_stop_trigger_pct() -> float:
    try:
        return max(0.05, float(os.environ.get("GEX_TRADER_TRAIL_TRIGGER_PCT", "0.10")))
    except (TypeError, ValueError):
        return 0.10


def trailing_stop_floor_pct() -> float:
    try:
        return max(0.01, float(os.environ.get("GEX_TRADER_TRAIL_FLOOR_PCT", "0.05")))
    except (TypeError, ValueError):
        return 0.05


def time_stop_bars() -> int:
    try:
        return max(1, int(os.environ.get("GEX_TRADER_TIME_STOP_BARS", "6")))
    except (TypeError, ValueError):
        return 6


def time_stop_min_pnl_pct() -> float:
    try:
        return float(os.environ.get("GEX_TRADER_TIME_STOP_MIN_PNL_PCT", "0.05"))
    except (TypeError, ValueError):
        return 0.05


def time_stop_min_magnet_progress() -> float:
    """Minimum fraction of distance-to-strike closed to avoid a time stop."""
    try:
        return float(os.environ.get("GEX_TRADER_TIME_STOP_MIN_PROGRESS", "0.20"))
    except (TypeError, ValueError):
        return 0.20


def stop_cooldown_bars() -> int:
    try:
        return max(0, int(os.environ.get("GEX_TRADER_STOP_COOLDOWN_BARS", "2")))
    except (TypeError, ValueError):
        return 2


def strong_entry_confidence() -> float:
    try:
        return float(os.environ.get("GEX_TRADER_STRONG_CONFIDENCE", "0.80"))
    except (TypeError, ValueError):
        return 0.80


def strong_gamma_delta() -> float:
    try:
        return float(os.environ.get("GEX_TRADER_STRONG_GAMMA_DELTA", "0.08"))
    except (TypeError, ValueError):
        return 0.08


def magnet_proximity_pct() -> float:
    try:
        return float(os.environ.get("GEX_TRADER_MAGNET_PROXIMITY_PCT", "0.003"))
    except (TypeError, ValueError):
        return 0.003


def high_confidence_contracts() -> int:
    try:
        return max(1, int(os.environ.get("GEX_TRADER_HIGH_CONF_CONTRACTS", "2")))
    except (TypeError, ValueError):
        return 2


def max_open_positions() -> int:
    try:
        return max(1, int(os.environ.get("GEX_TRADER_MAX_OPEN", "2")))
    except (TypeError, ValueError):
        return 2


def max_entries_per_cycle() -> int:
    """Max new positions opened per trader/backtest snapshot cycle."""
    try:
        return max(1, int(os.environ.get("GEX_TRADER_MAX_ENTRIES_PER_CYCLE", "1")))
    except (TypeError, ValueError):
        return 1


def min_gamma_delta() -> float:
    try:
        return max(0.0, float(os.environ.get("GEX_TRADER_MIN_GAMMA_DELTA", "0.03")))
    except (TypeError, ValueError):
        return 0.03


def min_fastest_gamma_delta() -> float:
    try:
        return max(0.0, float(os.environ.get("GEX_TRADER_MIN_FASTEST_GAMMA_DELTA", "0.03")))
    except (TypeError, ValueError):
        return 0.03


def max_strike_distance_pct() -> float:
    try:
        return max(0.001, float(os.environ.get("GEX_TRADER_MAX_STRIKE_DISTANCE_PCT", "0.02")))
    except (TypeError, ValueError):
        return 0.02


def min_confluence_score() -> float:
    try:
        return float(os.environ.get("GEX_TRADER_MIN_CONFLUENCE", "50"))
    except (TypeError, ValueError):
        return 50.0


def require_spot_momentum() -> bool:
    return _flag("GEX_TRADER_REQUIRE_MOMENTUM", "1")


def block_event_days() -> bool:
    return _flag("GEX_TRADER_BLOCK_EVENTS", "1")


def require_flow_alignment() -> bool:
    return _flag("GEX_TRADER_REQUIRE_FLOW_ALIGN", "1")


def strict_entry_filters() -> bool:
    return _flag("GEX_TRADER_STRICT_FILTERS", "1")


def option_leverage() -> float:
    """Rough ATM option leverage for paper PnL vs underlying move."""
    try:
        return float(os.environ.get("GEX_TRADER_OPTION_LEVERAGE", "12"))
    except (TypeError, ValueError):
        return 12.0


def webull_app_key() -> str:
    return os.environ.get("GEX_WEBULL_APP_KEY", os.environ.get("WEBULL_APP_KEY", "")).strip()


def webull_app_secret() -> str:
    return os.environ.get("GEX_WEBULL_APP_SECRET", os.environ.get("WEBULL_APP_SECRET", "")).strip()


def webull_account_id() -> str:
    return os.environ.get("GEX_WEBULL_ACCOUNT_ID", os.environ.get("WEBULL_ACCOUNT_ID", "")).strip()


def webull_region() -> str:
    return os.environ.get("GEX_WEBULL_REGION", "us").strip().lower() or "us"


def webull_endpoint() -> str:
    return os.environ.get("GEX_WEBULL_ENDPOINT", "us-openapi.webullbroker.com").strip()


def signal_ticker() -> str:
    """Ticker used for gamma signals and GEX data (default SPX)."""
    return os.environ.get("GEX_SIGNAL_TICKER", "SPX").strip().upper() or "SPX"


def execution_ticker() -> str:
    """Underlying symbol for live/paper option orders (default SPY)."""
    raw = os.environ.get("GEX_EXECUTION_TICKER", os.environ.get("GEX_WEBULL_UNDERLYING", "SPY"))
    return raw.strip().upper() or "SPY"


def webull_fill_timeout_sec() -> float:
    try:
        return max(5.0, float(os.environ.get("GEX_WEBULL_FILL_TIMEOUT_SEC", "45")))
    except (TypeError, ValueError):
        return 45.0


def webull_fill_poll_sec() -> float:
    try:
        return max(0.5, float(os.environ.get("GEX_WEBULL_FILL_POLL_SEC", "2")))
    except (TypeError, ValueError):
        return 2.0


def webull_option_category() -> str:
    return os.environ.get("GEX_WEBULL_OPTION_CATEGORY", "US_OPTION").strip() or "US_OPTION"


def webull_underlying() -> str:
    """Webull option root symbol — defaults to execution ticker (SPY)."""
    raw = os.environ.get("GEX_WEBULL_UNDERLYING", "")
    if raw.strip():
        return raw.strip().upper()
    return execution_ticker()


def webull_contracts() -> int:
    try:
        return max(1, int(os.environ.get("GEX_TRADER_CONTRACTS", "1")))
    except (TypeError, ValueError):
        return 1


def webull_limit_buffer_pct() -> float:
    try:
        return max(0.0, float(os.environ.get("GEX_WEBULL_LIMIT_BUFFER_PCT", "0.05")))
    except (TypeError, ValueError):
        return 0.05


def webull_configured() -> bool:
    return bool(webull_app_key() and webull_app_secret() and webull_account_id())


def live_trading_allowed() -> bool:
    return (not paper_trading_only()) and webull_configured()


def require_live_confirm() -> bool:
    return _flag("GEX_TRADER_LIVE_CONFIRM", "1")


def account_equity_usd() -> float:
    try:
        return max(1.0, float(os.environ.get("GEX_TRADER_ACCOUNT_EQUITY", "500")))
    except (TypeError, ValueError):
        return 500.0


def risk_per_trade_pct() -> float:
    try:
        return max(0.001, float(os.environ.get("GEX_TRADER_RISK_PER_TRADE_PCT", "0.01")))
    except (TypeError, ValueError):
        return 0.01


def use_risk_based_sizing() -> bool:
    return _flag("GEX_TRADER_RISK_SIZING", "1")


def use_webull_account_equity() -> bool:
    """When live on Webull, size risk from broker net liquidation value."""
    return _flag("GEX_TRADER_USE_WEBULL_EQUITY", "1")


def webull_equity_cache_seconds() -> float:
    try:
        return max(5.0, float(os.environ.get("GEX_WEBULL_EQUITY_CACHE_SEC", "60")))
    except (TypeError, ValueError):
        return 60.0


def min_zero_dte_ratio() -> float:
    try:
        return max(0.0, float(os.environ.get("GEX_TRADER_MIN_ZERO_DTE_RATIO", "0")))
    except (TypeError, ValueError):
        return 0.0


def require_gamma_flip_side() -> bool:
    return _flag("GEX_TRADER_REQUIRE_FLIP_SIDE", "1")


def max_iv_rank() -> float:
    try:
        return float(os.environ.get("GEX_TRADER_MAX_IV_RANK", "1.0"))
    except (TypeError, ValueError):
        return 1.0


def min_magnet_progress_pct() -> float:
    try:
        return max(0.0, float(os.environ.get("GEX_TRADER_MIN_MAGNET_PROGRESS", "0.0")))
    except (TypeError, ValueError):
        return 0.0


def momentum_bars() -> int:
    try:
        return max(1, int(os.environ.get("GEX_TRADER_MOMENTUM_BARS", "2")))
    except (TypeError, ValueError):
        return 2


def entry_window_after_open_min() -> int:
    try:
        return max(0, int(os.environ.get("GEX_TRADER_ENTRY_AFTER_OPEN_MIN", "15")))
    except (TypeError, ValueError):
        return 15


def entry_window_before_close_min() -> int:
    try:
        return max(0, int(os.environ.get("GEX_TRADER_ENTRY_BEFORE_CLOSE_MIN", "30")))
    except (TypeError, ValueError):
        return 30


def eod_flatten_enabled() -> bool:
    return _flag("GEX_TRADER_EOD_FLATTEN", "1")


def eod_flatten_hour() -> int:
    try:
        return int(os.environ.get("GEX_TRADER_EOD_FLATTEN_HOUR", "15"))
    except (TypeError, ValueError):
        return 15


def eod_flatten_minute() -> int:
    try:
        return int(os.environ.get("GEX_TRADER_EOD_FLATTEN_MIN", "45"))
    except (TypeError, ValueError):
        return 45


def magnet_touch_exit_enabled() -> bool:
    return _flag("GEX_TRADER_MAGNET_TOUCH_EXIT", "1")


def dynamic_take_profit_enabled() -> bool:
    return _flag("GEX_TRADER_DYNAMIC_TP", "1")


def min_flow_buy_ratio() -> float:
    try:
        return float(os.environ.get("GEX_TRADER_MIN_FLOW_BUY_RATIO", "0.55"))
    except (TypeError, ValueError):
        return 0.55


def min_flow_aggressiveness() -> float:
    try:
        return float(os.environ.get("GEX_TRADER_MIN_FLOW_AGGRESSIVENESS", "0"))
    except (TypeError, ValueError):
        return 0.0


def event_day_size_multiplier() -> float:
    """0 = hard block on event days; 0.5 = half size; 1 = ignore events."""
    try:
        return max(0.0, float(os.environ.get("GEX_TRADER_EVENT_SIZE_MULT", "0")))
    except (TypeError, ValueError):
        return 0.0


def prefer_signal_type() -> str:
    return os.environ.get("GEX_TRADER_PREFER_SIGNAL", "").strip().lower()


def entry_time_filter_enabled() -> bool:
    return _flag("GEX_TRADER_ENTRY_TIME_FILTER", "1")


def multi_strike_count() -> int:
    try:
        return max(1, int(os.environ.get("GEX_TRADER_MULTI_STRIKE", "2")))
    except (TypeError, ValueError):
        return 2
