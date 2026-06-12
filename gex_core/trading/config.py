"""Auto-trader configuration from environment."""

from __future__ import annotations

import os
from dataclasses import dataclass

from gex_core.env_bootstrap import parse_env_minutes


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def clear_all_filters() -> bool:
    """Master switch: disable signal, advisor, and strict entry gates."""
    return _flag("GEX_TRADER_CLEAR_FILTERS", "0")


def auto_trader_enabled() -> bool:
    return _flag("GEX_AUTO_TRADER", "0")


def wall_gex_auto_enabled() -> bool:
    """Background wall GEX loop in the web dashboard scheduler."""
    return _flag("GEX_WALL_GEX_AUTO", "0")


def wall_gex_cycle_seconds() -> int:
    """How often the wall GEX trader evaluates exits/entries."""
    explicit = os.environ.get("GEX_WALL_GEX_CYCLE_SECONDS", "").strip()
    if explicit:
        try:
            return max(5, int(explicit))
        except (TypeError, ValueError):
            pass
    try:
        return max(30, int(os.environ.get("GEX_WALL_GEX_CYCLE_SECONDS", "120")))
    except (TypeError, ValueError):
        return 120


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
        return max(0.01, float(os.environ.get("GEX_TRADER_STOP_LOSS_PCT", "0.06")))
    except (TypeError, ValueError):
        return 0.06


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
        return max(0.05, float(os.environ.get("GEX_TRADER_TAKE_PROFIT_PCT", "0.28")))
    except (TypeError, ValueError):
        return 0.28


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


def max_hold_minutes() -> float:
    """Hard max hold before flat exit (converted to bars via trader_bar_minutes)."""
    try:
        return max(1.0, float(os.environ.get("GEX_TRADER_MAX_HOLD_MINUTES", "30")))
    except (TypeError, ValueError):
        return 30.0


def max_hold_bars() -> int:
    return max(1, int(round(max_hold_minutes() / trader_bar_minutes())))


def time_stop_bars() -> int:
    """Stale-trade stop bars; default 7 (~14 min on 2-min snapshots)."""
    try:
        raw = os.environ.get("GEX_TRADER_TIME_STOP_BARS", "").strip()
        if raw:
            return max(1, int(raw))
        return 7
    except (TypeError, ValueError):
        return 7


def time_stop_min_pnl_pct() -> float:
    """Time-stop when PnL is below this (default 0 = underwater / stale losers only)."""
    try:
        return float(os.environ.get("GEX_TRADER_TIME_STOP_MIN_PNL_PCT", "0"))
    except (TypeError, ValueError):
        return 0.0


def time_stop_min_magnet_progress() -> float:
    """Minimum fraction of distance-to-strike closed to avoid a time stop."""
    try:
        return float(os.environ.get("GEX_TRADER_TIME_STOP_MIN_PROGRESS", "0.35"))
    except (TypeError, ValueError):
        return 0.35


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


def min_entry_confidence() -> float:
    """Minimum AI advisor confidence required to open a new position (0 = no floor)."""
    if clear_all_filters():
        return 0.0
    try:
        return max(0.0, min(1.0, float(os.environ.get("GEX_TRADER_MIN_ENTRY_CONFIDENCE", "0"))))
    except (TypeError, ValueError):
        return 0.0


def advisor_context_max_chars() -> int:
    """Max characters of UW + signal context sent to the entry advisor LLM."""
    try:
        return max(4000, int(os.environ.get("GEX_ADVISOR_CONTEXT_MAX_CHARS", "24000")))
    except (TypeError, ValueError):
        return 24000


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


def wall_stop_loss_pct() -> float:
    try:
        return max(0.01, float(os.environ.get("GEX_WALL_STOP_LOSS_PCT", "0.03")))
    except (TypeError, ValueError):
        return 0.03


def wall_take_profit_pct() -> float:
    try:
        return max(0.05, float(os.environ.get("GEX_WALL_TAKE_PROFIT_PCT", "0.22")))
    except (TypeError, ValueError):
        return 0.22


def wall_intraday_session() -> bool:
    """Keep only regular-session export snapshots in wall GEX backtests."""
    return _flag("GEX_WALL_INTRADAY_SESSION", "1")


def wall_entry_time_filter() -> bool:
    """Skip wall entries outside the entry window (after open / before close)."""
    return _flag("GEX_WALL_ENTRY_TIME_FILTER", "1")


def wall_signal_filters_enabled() -> bool:
    """Quality gates: min |gamma|, regime, wall drift (off by default)."""
    return _flag("GEX_WALL_SIGNAL_FILTERS", "0")


def wall_min_gamma_bn() -> float:
    """Skip walls with |net GEX| below this (Bn)."""
    try:
        return max(0.0, float(os.environ.get("GEX_WALL_MIN_GAMMA_BN", "0.5")))
    except (TypeError, ValueError):
        return 0.5


def wall_block_short_gamma() -> bool:
    """Skip entries when snapshot regime is short gamma."""
    return _flag("GEX_WALL_BLOCK_SHORT_GAMMA", "0")


def wall_min_drift_pts() -> float:
    """Require wall strike to move at least this many points vs prior bar (0=off)."""
    try:
        return max(0.0, float(os.environ.get("GEX_WALL_MIN_DRIFT_PTS", "0")))
    except (TypeError, ValueError):
        return 0.0


def wall_max_hold_bars() -> int:
    """Hard max hold for wall GEX trades (bars); default 8 ≈ 40 min on 5-min snapshots."""
    minutes = os.environ.get("GEX_WALL_MAX_HOLD_MINUTES", "").strip()
    if minutes:
        try:
            bar = max(1.0, low_gex_bar_minutes())
            return max(1, int(round(float(minutes) / bar)))
        except (TypeError, ValueError):
            pass
    try:
        return max(1, int(os.environ.get("GEX_WALL_MAX_HOLD_BARS", "8")))
    except (TypeError, ValueError):
        return 8


def wall_reenter_on_shift() -> bool:
    """Close and reopen when the target GEX wall strike moves."""
    return _flag("GEX_WALL_REENTER_ON_SHIFT", "1")


def near_wall_window_pct() -> float:
    """Strike search window for /near wall GEX view and backtests."""
    try:
        return max(0.005, min(0.05, float(os.environ.get("GEX_NEAR_WALL_WINDOW_PCT", "0.01"))))
    except (TypeError, ValueError):
        return 0.01


def near_wall_stop_loss_pct() -> float:
    """MC-tuned default for ±1% wall GEX (14d backtest)."""
    try:
        return max(0.01, float(os.environ.get("GEX_NEAR_WALL_STOP_LOSS_PCT", "0.03")))
    except (TypeError, ValueError):
        return 0.03


def near_wall_take_profit_pct() -> float:
    try:
        return max(0.05, float(os.environ.get("GEX_NEAR_WALL_TAKE_PROFIT_PCT", "0.28")))
    except (TypeError, ValueError):
        return 0.28


def near_wall_max_hold_bars() -> int:
    try:
        return max(1, int(os.environ.get("GEX_NEAR_WALL_MAX_HOLD_BARS", "10")))
    except (TypeError, ValueError):
        return 10


def near_wall_shift_min_pts() -> float:
    """Min wall strike move (pts) before flatten-on-shift fires for /near."""
    try:
        return max(0.5, float(os.environ.get("GEX_NEAR_WALL_SHIFT_MIN_PTS", "0.5")))
    except (TypeError, ValueError):
        return 0.5


def near_wall_reenter_on_shift() -> bool:
    """Near-spot: MC suggests disabling shift flatten (same PnL, fewer churn exits)."""
    return _flag("GEX_NEAR_WALL_REENTER_ON_SHIFT", "0")


DEFAULT_WALL_WINDOW_PCT = 0.12


@dataclass(frozen=True)
class WallGexProfile:
    """Resolved SL/TP/hold/shift settings for a wall GEX strike window."""

    window_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    max_hold_bars: int
    reenter_on_shift: bool
    shift_min_pts: float
    near: bool


def is_near_wall_window(window_pct: float) -> bool:
    """True when the strike search band is the /near ±1% view."""
    return float(window_pct) <= near_wall_window_pct() + 1e-9


def wall_gex_profile(window_pct: float | None = None) -> WallGexProfile:
    """Return wall GEX params for full (12%) or near-spot (±1%) windows."""
    wp = DEFAULT_WALL_WINDOW_PCT if window_pct is None else float(window_pct)
    if is_near_wall_window(wp):
        return WallGexProfile(
            window_pct=wp,
            stop_loss_pct=near_wall_stop_loss_pct(),
            take_profit_pct=near_wall_take_profit_pct(),
            max_hold_bars=near_wall_max_hold_bars(),
            reenter_on_shift=near_wall_reenter_on_shift(),
            shift_min_pts=near_wall_shift_min_pts(),
            near=True,
        )
    return WallGexProfile(
        window_pct=wp,
        stop_loss_pct=wall_stop_loss_pct(),
        take_profit_pct=wall_take_profit_pct(),
        max_hold_bars=wall_max_hold_bars(),
        reenter_on_shift=wall_reenter_on_shift(),
        shift_min_pts=0.5,
        near=False,
    )


def wall_reentry_after_stop() -> bool:
    """Allow a fresh entry at the same strike after a stop-loss (no strike cooldown)."""
    return _flag("GEX_WALL_REENTRY_AFTER_STOP", "1")


def low_gex_reenter_each_bar() -> bool:
    """Close open low-GEX positions every bar and open fresh toward the current wall."""
    return _flag("GEX_LOW_GEX_REENTER_EACH_BAR", "0")


def low_gex_bar_minutes() -> float:
    """Target bar cadence for low-GEX rotation (default 5 minutes)."""
    explicit = os.environ.get("GEX_LOW_GEX_BAR_MINUTES", "").strip()
    if explicit:
        try:
            return max(1.0, float(explicit))
        except (TypeError, ValueError):
            pass
    return max(1.0, parse_env_minutes("GEX_BACKFILL_INTERVAL_MINUTES", 5.0))


def min_gamma_delta() -> float:
    if clear_all_filters():
        return 0.0
    try:
        return max(0.0, float(os.environ.get("GEX_TRADER_MIN_GAMMA_DELTA", "0.02")))
    except (TypeError, ValueError):
        return 0.02


def min_fastest_gamma_delta() -> float:
    if clear_all_filters():
        return 0.0
    try:
        return max(0.0, float(os.environ.get("GEX_TRADER_MIN_FASTEST_GAMMA_DELTA", "0.03")))
    except (TypeError, ValueError):
        return 0.03


def max_strike_distance_pct() -> float:
    if clear_all_filters():
        return 1.0
    try:
        return max(0.001, float(os.environ.get("GEX_TRADER_MAX_STRIKE_DISTANCE_PCT", "0.02")))
    except (TypeError, ValueError):
        return 0.02


def min_confluence_score() -> float:
    if clear_all_filters():
        return 0.0
    try:
        return float(os.environ.get("GEX_TRADER_MIN_CONFLUENCE", "50"))
    except (TypeError, ValueError):
        return 50.0


def require_spot_momentum() -> bool:
    if clear_all_filters():
        return False
    return _flag("GEX_TRADER_REQUIRE_MOMENTUM", "0")


def block_event_days() -> bool:
    if clear_all_filters():
        return False
    return _flag("GEX_TRADER_BLOCK_EVENTS", "1")


def require_flow_alignment() -> bool:
    if clear_all_filters():
        return False
    return _flag("GEX_TRADER_REQUIRE_FLOW_ALIGN", "0")


def strict_entry_filters() -> bool:
    if clear_all_filters():
        return False
    return _flag("GEX_TRADER_STRICT_FILTERS", "0")


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


def webull_use_uat() -> bool:
    """Use Webull UAT/sandbox hosts (for integration testing)."""
    return _flag("GEX_WEBULL_USE_UAT", "0")


# Retired hostnames from older docs / examples (NXDOMAIN or wrong service).
_DEPRECATED_WEBULL_TRADE_HOSTS = {
    "us-openapi.webullbroker.com": "api.webull.com",
    "openapi.webullbroker.com": "api.webull.com",
}
_DEPRECATED_WEBULL_DATA_HOSTS = {
    "us-openapi.webullbroker.com": "broker-api.webull.com",
    "openapi.webullbroker.com": "broker-api.webull.com",
}


def _normalize_webull_trade_host(host: str) -> str:
    key = host.strip().lower()
    if webull_use_uat():
        return host.strip()
    replacement = _DEPRECATED_WEBULL_TRADE_HOSTS.get(key)
    return replacement if replacement else host.strip()


def _normalize_webull_data_host(host: str) -> str:
    key = host.strip().lower()
    if webull_use_uat():
        return host.strip()
    replacement = _DEPRECATED_WEBULL_DATA_HOSTS.get(key)
    return replacement if replacement else host.strip()


def webull_trade_endpoint() -> str:
    """Trading API host (see https://developer.webull.com/apis/docs/sdk/)."""
    explicit = os.environ.get("GEX_WEBULL_ENDPOINT", "").strip()
    if explicit:
        return _normalize_webull_trade_host(explicit)
    if webull_use_uat():
        return "us-openapi-alb.uat.webullbroker.com"
    return "api.webull.com"


def webull_data_endpoint() -> str:
    """Market data API host (options quotes, snapshots)."""
    explicit = os.environ.get("GEX_WEBULL_DATA_ENDPOINT", "").strip()
    if explicit:
        return _normalize_webull_data_host(explicit)
    if webull_use_uat():
        return "us-broker-api.uat.webullbroker.com"
    return "broker-api.webull.com"


def webull_endpoint() -> str:
    """Alias for the trading API host."""
    return webull_trade_endpoint()


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
        return max(0.001, float(os.environ.get("GEX_TRADER_RISK_PER_TRADE_PCT", "0.50")))
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


def webull_quote_cache_seconds() -> float:
    """TTL for option NBBO snapshots — reduces duplicate calls per quote refresh."""
    try:
        return max(0.5, float(os.environ.get("GEX_WEBULL_QUOTE_CACHE_SEC", "3")))
    except (TypeError, ValueError):
        return 3.0


def webull_position_cache_seconds() -> float:
    """TTL for broker open-position snapshots on the trade desk."""
    try:
        return max(5.0, float(os.environ.get("GEX_WEBULL_POSITION_CACHE_SEC", "30")))
    except (TypeError, ValueError):
        return 30.0


def min_zero_dte_ratio() -> float:
    try:
        return max(0.0, float(os.environ.get("GEX_TRADER_MIN_ZERO_DTE_RATIO", "0")))
    except (TypeError, ValueError):
        return 0.0


def require_gamma_flip_side() -> bool:
    if clear_all_filters():
        return False
    return _flag("GEX_TRADER_REQUIRE_FLIP_SIDE", "0")


def max_iv_rank() -> float:
    try:
        return float(os.environ.get("GEX_TRADER_MAX_IV_RANK", "1.0"))
    except (TypeError, ValueError):
        return 1.0


def min_magnet_progress_pct() -> float:
    if clear_all_filters():
        return 0.0
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
    return _flag("GEX_TRADER_EOD_FLATTEN", "0")


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
    return _flag("GEX_TRADER_MAGNET_TOUCH_EXIT", "0")


def magnet_touch_min_pnl_pct() -> float:
    """Minimum option PnL %% before magnet-touch exit (avoids scratching at the magnet)."""
    try:
        return max(0.0, float(os.environ.get("GEX_TRADER_MAGNET_TOUCH_MIN_PNL_PCT", "0.04")))
    except (TypeError, ValueError):
        return 0.04


def dynamic_take_profit_enabled() -> bool:
    return _flag("GEX_TRADER_DYNAMIC_TP", "0")


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


def max_gamma_only() -> bool:
    """Trade dominant |gamma| magnet (highest +γ or lowest −γ); drop fastest-increase fallback."""
    return _flag("GEX_TRADER_MAX_GAMMA_ONLY", "1")


def trade_negative_gamma_magnets() -> bool:
    """Enter on lowest −γ walls when |γ| dominates (off = trade highest +γ only)."""
    return _flag("GEX_TRADER_TRADE_NEGATIVE_GAMMA", "0")


def magnet_anchored_strikes() -> bool:
    """Trade strikes at the gamma magnet instead of nearest ATM."""
    return _flag("GEX_TRADER_MAGNET_ANCHORED_STRIKES", "0")


def fix_magnet_exit_scale() -> bool:
    """Map SPX magnet strikes to execution spot scale for exit progress checks."""
    return _flag("GEX_TRADER_FIX_MAGNET_EXIT_SCALE", "1")


def min_magnet_distance_pct() -> float:
    if clear_all_filters():
        return 0.0
    try:
        return max(0.0, float(os.environ.get("GEX_TRADER_MIN_MAGNET_DISTANCE_PCT", "0")))
    except (TypeError, ValueError):
        return 0.0


def dynamic_time_stop() -> bool:
    return _flag("GEX_TRADER_DYNAMIC_TIME_STOP", "0")


def equity_from_mark() -> bool:
    """When off, account return_pct uses realized closed-trade PnL (recommended)."""
    return _flag("GEX_TRADER_EQUITY_FROM_MARK", "0")


def regime_strict() -> bool:
    """Block entries in short-gamma regime."""
    if clear_all_filters():
        return False
    return _flag("GEX_TRADER_REGIME_STRICT", "0")


def magnet_partial_exit_enabled() -> bool:
    return _flag("GEX_TRADER_MAGNET_PARTIAL_EXIT", "0")


def magnet_partial_progress_pct() -> float:
    try:
        return max(0.5, float(os.environ.get("GEX_TRADER_MAGNET_PARTIAL_PROGRESS", "0.80")))
    except (TypeError, ValueError):
        return 0.80


def entry_time_filter_enabled() -> bool:
    if clear_all_filters():
        return False
    return _flag("GEX_TRADER_ENTRY_TIME_FILTER", "0")


def multi_strike_count() -> int:
    try:
        return max(1, int(os.environ.get("GEX_TRADER_MULTI_STRIKE", "2")))
    except (TypeError, ValueError):
        return 2
