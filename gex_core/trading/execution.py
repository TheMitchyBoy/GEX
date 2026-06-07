"""Map SPX gamma signals to SPY (or other) option execution."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime

from gex_core.trading.config import execution_ticker, signal_ticker

logger = logging.getLogger(__name__)

# Last observed execution_spot / signal_spot (e.g. SPY/SPX ≈ 0.09989, not 0.1).
_LAST_SPOT_RATIO: float | None = None
_LAST_RATIO_TS: float = 0.0
_RATIO_TTL_SEC = 300.0


def record_spot_ratio(*, signal_spot: float, execution_spot: float) -> float:
    """Cache live SPX/SPY scale for brief quote gaps."""
    global _LAST_SPOT_RATIO, _LAST_RATIO_TS
    if signal_spot <= 0 or execution_spot <= 0:
        return execution_spot / signal_spot if signal_spot > 0 else 0.0
    ratio = float(execution_spot) / float(signal_spot)
    _LAST_SPOT_RATIO = ratio
    _LAST_RATIO_TS = time.time()
    return ratio


def spot_scale_ratio(*, signal_spot: float, execution_spot: float) -> float:
    """Execution spot per unit of signal spot (SPY price / SPX price)."""
    if signal_spot <= 0 or execution_spot <= 0:
        raise ValueError("signal_spot and execution_spot must be positive")
    return float(execution_spot) / float(signal_spot)


def _cached_ratio() -> float | None:
    if _LAST_SPOT_RATIO is None:
        return None
    if time.time() - _LAST_RATIO_TS > _RATIO_TTL_SEC:
        return None
    return _LAST_SPOT_RATIO


def _fetch_live_spot(symbol: str) -> float | None:
    symbol = symbol.upper()
    try:
        from gex_core.uw_price_stream import get_uw_price_stream

        live = get_uw_price_stream().get_latest_price(symbol)
        if live and live > 0:
            return float(live)
    except Exception:
        pass
    try:
        import yfinance as yf

        hist = yf.Ticker(symbol).history(period="1d", interval="1m")
        if hist is not None and not hist.empty:
            px = float(hist["Close"].iloc[-1])
            if px > 0:
                return px
    except Exception as exc:
        logger.debug("Spot fetch failed for %s: %s", symbol, exc)
    return None


def resolve_execution_spot(*, signal_spot: float | None = None) -> float | None:
    """Live execution symbol spot (SPY). Never assumes SPX / 10."""
    exec_sym = execution_ticker().upper()
    sig_sym = signal_ticker().upper()

    live = _fetch_live_spot(exec_sym)
    if live and live > 0:
        if signal_spot and signal_spot > 0 and exec_sym != sig_sym:
            record_spot_ratio(signal_spot=float(signal_spot), execution_spot=live)
        return live

    if signal_spot and signal_spot > 0:
        ratio = _cached_ratio()
        if ratio is not None:
            return round(float(signal_spot) * ratio, 4)

    if exec_sym == sig_sym and signal_spot and signal_spot > 0:
        return float(signal_spot)

    return None


def map_execution_strike(
    signal_strike: float,
    *,
    signal_spot: float,
    execution_spot: float,
) -> float:
    """Scale SPX strike to SPY strike using live spot ratio (not ÷10)."""
    if signal_spot <= 0 or execution_spot <= 0:
        raise ValueError("Cannot map strike without both signal_spot and execution_spot")
    ratio = spot_scale_ratio(signal_spot=signal_spot, execution_spot=execution_spot)
    record_spot_ratio(signal_spot=signal_spot, execution_spot=execution_spot)
    return round(float(signal_strike) * ratio)


def build_webull_option_symbol(
    *,
    underlying: str,
    expire_date: str,
    option_type: str,
    strike: float,
) -> str:
    """Webull OSI-style symbol, e.g. SPY260606C00737500."""
    dt = datetime.strptime(expire_date, "%Y-%m-%d")
    yymmdd = dt.strftime("%y%m%d")
    cp = "C" if str(option_type).lower().startswith("c") else "P"
    strike_key = int(round(float(strike) * 1000))
    return f"{underlying.upper()}{yymmdd}{cp}{strike_key:08d}"


def execution_summary(*, signal_strike: float, signal_spot: float, execution_spot: float) -> dict[str, float | str]:
    ratio = spot_scale_ratio(signal_spot=signal_spot, execution_spot=execution_spot)
    exec_strike = map_execution_strike(signal_strike, signal_spot=signal_spot, execution_spot=execution_spot)
    return {
        "signal_ticker": signal_ticker(),
        "execution_ticker": execution_ticker(),
        "signal_strike": float(signal_strike),
        "execution_strike": float(exec_strike),
        "signal_spot": float(signal_spot),
        "execution_spot": float(execution_spot),
        "spot_ratio": float(ratio),
    }


def backtest_spot_ratio() -> float:
    """Default execution/signal spot ratio when live quotes are unavailable (SPY/SPX)."""
    try:
        return float(os.environ.get("GEX_BACKTEST_SPOT_RATIO", "0.09989"))
    except (TypeError, ValueError):
        return 0.09989


def resolve_backtest_execution_spot(*, signal_spot: float) -> float | None:
    """Historical execution spot from signal spot for walk-forward backtests."""
    exec_sym = execution_ticker().upper()
    sig_sym = signal_ticker().upper()
    if signal_spot <= 0:
        return None
    if exec_sym == sig_sym:
        return float(signal_spot)
    return round(float(signal_spot) * backtest_spot_ratio(), 4)


def uses_execution_mapping() -> bool:
    return execution_ticker().upper() != signal_ticker().upper()


def sync_execution_context(*, signal_spot: float) -> dict[str, float | str | None]:
    """Resolve execution spot + ratio for a known SPX signal spot."""
    exec_spot = resolve_execution_spot(signal_spot=signal_spot)
    ratio = None
    if exec_spot and signal_spot > 0:
        ratio = record_spot_ratio(signal_spot=signal_spot, execution_spot=exec_spot)
    return {
        "signal_ticker": signal_ticker(),
        "execution_ticker": execution_ticker(),
        "signal_spot": float(signal_spot),
        "execution_spot": float(exec_spot) if exec_spot else None,
        "spot_ratio": float(ratio) if ratio else None,
    }
