"""Map SPX gamma signals to SPY (or other) option execution."""

from __future__ import annotations

import logging
from datetime import datetime

from gex_core.trading.config import execution_ticker, signal_ticker

logger = logging.getLogger(__name__)


def map_execution_strike(
    signal_strike: float,
    *,
    signal_spot: float,
    execution_spot: float,
) -> float:
    """Scale an index signal strike to the execution underlying (e.g. SPX → SPY)."""
    if signal_spot > 0 and execution_spot > 0:
        return round(float(signal_strike) * float(execution_spot) / float(signal_spot))
    if execution_ticker().upper() == "SPY":
        return round(float(signal_strike) / 10.0)
    return round(float(signal_strike))


def resolve_execution_spot(*, signal_spot: float | None = None) -> float | None:
    """Best available spot for the execution symbol (SPY), falling back from SPX."""
    exec_sym = execution_ticker().upper()
    try:
        from gex_core.uw_price_stream import get_uw_price_stream

        live = get_uw_price_stream().get_latest_price(exec_sym)
        if live and live > 0:
            return float(live)
    except Exception:
        pass

    if exec_sym == "SPY" and signal_spot and signal_spot > 0:
        return round(float(signal_spot) / 10.0, 2)

    if signal_spot and signal_spot > 0 and exec_sym == signal_ticker().upper():
        return float(signal_spot)

    try:
        import yfinance as yf

        hist = yf.Ticker(exec_sym).history(period="1d", interval="1m")
        if hist is not None and not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as exc:
        logger.debug("yfinance spot fallback failed for %s: %s", exec_sym, exc)
    return None


def build_webull_option_symbol(
    *,
    underlying: str,
    expire_date: str,
    option_type: str,
    strike: float,
) -> str:
    """Webull OSI-style symbol, e.g. SPY260606C00590000."""
    dt = datetime.strptime(expire_date, "%Y-%m-%d")
    yymmdd = dt.strftime("%y%m%d")
    cp = "C" if str(option_type).lower().startswith("c") else "P"
    strike_key = int(round(float(strike) * 1000))
    return f"{underlying.upper()}{yymmdd}{cp}{strike_key:08d}"


def execution_summary(*, signal_strike: float, signal_spot: float, execution_spot: float) -> dict[str, float | str]:
    exec_strike = map_execution_strike(signal_strike, signal_spot=signal_spot, execution_spot=execution_spot)
    return {
        "signal_ticker": signal_ticker(),
        "execution_ticker": execution_ticker(),
        "signal_strike": float(signal_strike),
        "execution_strike": float(exec_strike),
        "signal_spot": float(signal_spot),
        "execution_spot": float(execution_spot),
    }
