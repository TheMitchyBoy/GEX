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
        return max(0.05, float(os.environ.get("GEX_TRADER_PARTIAL_TP_PCT", "0.15")))
    except (TypeError, ValueError):
        return 0.15


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
        return float(os.environ.get("GEX_TRADER_TIME_STOP_MIN_PNL_PCT", "0.02"))
    except (TypeError, ValueError):
        return 0.02


def time_stop_min_magnet_progress() -> float:
    """Minimum fraction of distance-to-strike closed to avoid a time stop."""
    try:
        return float(os.environ.get("GEX_TRADER_TIME_STOP_MIN_PROGRESS", "0.15"))
    except (TypeError, ValueError):
        return 0.15


def stop_cooldown_bars() -> int:
    try:
        return max(0, int(os.environ.get("GEX_TRADER_STOP_COOLDOWN_BARS", "12")))
    except (TypeError, ValueError):
        return 12


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
        return max(1, int(os.environ.get("GEX_TRADER_MAX_OPEN", "1")))
    except (TypeError, ValueError):
        return 1


def min_ai_confidence() -> float:
    try:
        return float(os.environ.get("GEX_TRADER_MIN_AI_CONFIDENCE", "0.65"))
    except (TypeError, ValueError):
        return 0.65


def min_gamma_delta() -> float:
    try:
        return float(os.environ.get("GEX_TRADER_MIN_GAMMA_DELTA", "0.05"))
    except (TypeError, ValueError):
        return 0.05


def min_fastest_gamma_delta() -> float:
    try:
        return float(os.environ.get("GEX_TRADER_MIN_FASTEST_GAMMA_DELTA", "0.10"))
    except (TypeError, ValueError):
        return 0.10


def max_strike_distance_pct() -> float:
    try:
        return max(0.001, float(os.environ.get("GEX_TRADER_MAX_STRIKE_DISTANCE_PCT", "0.01")))
    except (TypeError, ValueError):
        return 0.01


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


def webull_underlying() -> str:
    """Webull option root symbol (SPX index options often SPX / SPXW)."""
    return os.environ.get("GEX_WEBULL_UNDERLYING", "SPX").strip().upper() or "SPX"


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
