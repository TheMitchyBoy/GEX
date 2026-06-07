"""Market-context features (realized volatility, returns, optional VIX).

Realized volatility and spot returns are derived purely from the snapshot
``spot`` history, so they work fully offline. VIX is optional and fetched via
``yfinance`` when available; any failure (no network, missing dependency)
degrades silently to ``0.0`` so forecasting never hard-fails in a sandbox.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import os

import numpy as np
import pandas as pd

from gex_core.features import safe_float

logger = logging.getLogger(__name__)

# Annualization assuming ~252 trading days; snapshot cadence is treated as
# one step. Callers that need calendar scaling can post-multiply.
TRADING_PERIODS = 252
SPX_YF_SYMBOL = "^GSPC"


def spot_log_returns(history: list[dict[str, Any]]) -> list[float]:
    """Log returns of the snapshot spot series (skips missing/zero spots)."""
    returns: list[float] = []
    prev: float | None = None
    for row in history:
        spot = safe_float(row.get("spot"), 0.0)
        if spot <= 0:
            prev = None
            continue
        if prev is not None and prev > 0:
            returns.append(math.log(spot / prev))
        prev = spot
    return returns


def realized_volatility(history: list[dict[str, Any]], window: int = 10) -> float:
    """Rolling realized vol (std of recent log returns).

    Returns the *per-step* standard deviation; this is a stationary, scale-free
    feature that captures the current volatility regime without any network
    dependency.
    """
    returns = spot_log_returns(history)
    if len(returns) < 2:
        return 0.0
    recent = returns[-window:]
    if len(recent) < 2:
        return 0.0
    return float(np.std(recent, ddof=1))


def annualized_realized_vol(history: list[dict[str, Any]], window: int = 10) -> float:
    return realized_volatility(history, window=window) * math.sqrt(TRADING_PERIODS)


def latest_spot_return(history: list[dict[str, Any]]) -> float:
    """Most recent step's simple spot return."""
    if len(history) < 2:
        return 0.0
    s0 = safe_float(history[-2].get("spot"), 0.0)
    s1 = safe_float(history[-1].get("spot"), 0.0)
    if s0 <= 0 or s1 <= 0:
        return 0.0
    return (s1 - s0) / s0


def _yf_close(symbol: str, *, period: str = "1mo") -> pd.Series | None:
    try:
        import yfinance as yf

        data = yf.Ticker(symbol).history(period=period)
        if data is None or data.empty or "Close" not in data:
            return None
        return data["Close"].dropna()
    except Exception as exc:  # pragma: no cover - network/optional dep path
        logger.debug("%s fetch unavailable: %s", symbol, exc)
        return None


def fetch_vix_level() -> float:
    """Best-effort current VIX level; ``0.0`` when unavailable (offline)."""
    closes = _yf_close("^VIX", period="5d")
    return float(closes.iloc[-1]) if closes is not None and not closes.empty else 0.0


def fetch_vix9d_level() -> float:
    closes = _yf_close("^VIX9D", period="5d")
    return float(closes.iloc[-1]) if closes is not None and not closes.empty else 0.0


def fetch_iv_rank(symbol: str = SPX_YF_SYMBOL, lookback_days: int = 252) -> float:
    """IV rank proxy from realized vol percentile over ~1y of closes."""
    closes = _yf_close(symbol, period="1y")
    if closes is None or len(closes) < 30:
        return 0.0
    returns = closes.pct_change().dropna()
    if len(returns) < 30:
        return 0.0
    window = min(lookback_days, len(returns))
    rolling = returns.rolling(10).std().dropna()
    if rolling.empty:
        return 0.0
    current = float(rolling.iloc[-1])
    history = rolling.tail(window)
    rank = float((history <= current).mean())
    return max(0.0, min(1.0, rank))


def fetch_skew_proxy() -> float:
    """Put/call IV skew proxy via VIX minus short-dated VIX."""
    vix = fetch_vix_level()
    vix9d = fetch_vix9d_level()
    if vix <= 0 or vix9d <= 0:
        return 0.0
    return float((vix - vix9d) / max(vix, 1.0))


def fetch_expected_move_pct(symbol: str = SPX_YF_SYMBOL) -> float:
    """Approximate 1-day expected move from VIX."""
    vix = fetch_vix_level()
    if vix <= 0:
        return 0.0
    return float(vix / 100.0 / (TRADING_PERIODS ** 0.5))


def fetch_cross_asset_returns() -> dict[str, float]:
    out = {"spy_return": 0.0, "tlt_return": 0.0}
    for symbol, key in (("SPY", "spy_return"), ("TLT", "tlt_return")):
        closes = _yf_close(symbol, period="5d")
        if closes is None or len(closes) < 2:
            continue
        prev, last = float(closes.iloc[-2]), float(closes.iloc[-1])
        if prev > 0:
            out[key] = (last - prev) / prev
    return out


def fetch_vol_regime() -> dict[str, float]:
    return {
        "vix_level": fetch_vix_level(),
        "vix9d_level": fetch_vix9d_level(),
        "iv_rank": fetch_iv_rank(),
        "skew_proxy": fetch_skew_proxy(),
        "expected_move_pct": fetch_expected_move_pct(),
    }


def _uw_ohlc_candle_size() -> str:
    return os.environ.get("GEX_UW_OHLC_CANDLE_SIZE", os.environ.get("GEX_SPX_PRICE_INTERVAL", "15m"))


def _uw_ohlc_limit() -> int:
    try:
        return max(50, int(os.environ.get("GEX_UW_OHLC_LIMIT", "800")))
    except (TypeError, ValueError):
        return 800


def _uw_price_enabled() -> bool:
    from gex_core.env_bootstrap import uw_api_configured

    return uw_api_configured()


def _merge_price_points(*series: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    merged: dict[str, float] = {}
    for points in series:
        if not points:
            continue
        for row in points:
            ts = row.get("ts")
            close = row.get("close")
            if not ts or close is None:
                continue
            try:
                merged[str(ts)] = float(close)
            except (TypeError, ValueError):
                continue
    return [{"ts": ts, "close": close} for ts, close in sorted(merged.items())]


def fetch_uw_price_history(
    ticker: str = "SPX",
    *,
    candle_size: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]] | None:
    """Recent price history from UW OHLC REST."""
    if not _uw_price_enabled():
        return None
    try:
        from gex_core.uw_loader import fetch_uw_ohlc_points

        points = fetch_uw_ohlc_points(
            ticker,
            candle_size=candle_size or _uw_ohlc_candle_size(),
            limit=limit or _uw_ohlc_limit(),
        )
        return points or None
    except Exception as exc:  # pragma: no cover - network path
        logger.debug("UW OHLC price history unavailable for %s: %s", ticker, exc)
        return None


def fetch_spx_price_history(
    period: str | None = None,
    interval: str | None = None,
    *,
    ticker: str = "SPX",
) -> list[dict[str, Any]] | None:
    """Recent price history as ``[{"ts": iso, "close": float}, ...]`` via UW OHLC."""
    _ = period
    candle_size = interval or _uw_ohlc_candle_size()
    return fetch_uw_price_history(ticker, candle_size=candle_size)


def fetch_spx_price(ticker: str = "SPX") -> float:
    """Latest price from UW websocket cache, then REST fallbacks."""
    if _uw_price_enabled():
        try:
            from gex_core.uw_price_stream import get_uw_price_stream

            live = get_uw_price_stream().get_latest_price(ticker)
            if live > 0:
                return live
        except Exception as exc:
            logger.debug("UW websocket price unavailable for %s: %s", ticker, exc)
        try:
            from gex_core.uw_loader import fetch_uw_stock_state_price, fetch_uw_spot

            state_price = fetch_uw_stock_state_price(ticker)
            if state_price > 0:
                return state_price
            return fetch_uw_spot(ticker)
        except Exception as exc:
            logger.debug("UW REST price fallback unavailable for %s: %s", ticker, exc)
    points = fetch_spx_price_history(ticker=ticker)
    if points:
        return float(points[-1]["close"])
    return 0.0


def fetch_spx_price_series_for_dashboard(
    ticker: str = "SPX",
    *,
    index_days: int | None = None,
    index_interval_minutes: int | None = None,
) -> tuple[list[dict[str, Any]], float, str]:
    """Merge UW websocket ticks, OHLC candles, and indexed export spots."""
    from gex_core.storage import fetch_index_spot_series

    index_days = int(os.environ.get("GEX_DASHBOARD_TIMELINE_DAYS", "90")) if index_days is None else index_days
    index_interval = (
        int(os.environ.get("GEX_BACKFILL_INTERVAL_MINUTES", "2"))
        if index_interval_minutes is None
        else index_interval_minutes
    )

    ws_points: list[dict[str, Any]] = []
    ohlc_points: list[dict[str, Any]] = []
    if _uw_price_enabled():
        try:
            from gex_core.uw_price_stream import get_uw_price_stream

            ws_points = get_uw_price_stream().get_price_points(ticker)
        except Exception as exc:
            logger.debug("UW websocket series unavailable for %s: %s", ticker, exc)
        history = fetch_uw_price_history(ticker)
        if history:
            ohlc_points = history

    indexed = fetch_index_spot_series(
        ticker,
        days=index_days,
        interval_minutes=index_interval,
        max_points=800,
    )
    indexed_points = [
        {"ts": row["ts_label"], "close": float(row["spot"])}
        for row in indexed
        if row.get("spot")
    ]

    live_points = _merge_price_points(ohlc_points, ws_points)
    points: list[dict[str, Any]] = []
    source = "snapshots"
    if indexed_points and live_points:
        live_start = live_points[0]["ts"]
        points = [p for p in indexed_points if p["ts"] < live_start]
        points.extend(live_points)
        source = "uw-live+snapshots"
    elif live_points:
        points = live_points
        source = "uw-live"
    elif indexed_points:
        points = indexed_points
        source = "snapshots"

    current = fetch_spx_price(ticker)
    if current <= 0 and points:
        current = float(points[-1]["close"])
    elif current <= 0 and indexed:
        current = float(indexed[-1]["spot"])

    return points, current, source


def _include_vol_regime() -> bool:
    flag = os.environ.get("GEX_INCLUDE_VOL_REGIME", "1").lower()
    return flag not in {"0", "false", "no"}


def attach_market_features(
    history: list[dict[str, Any]],
    *,
    vol_window: int = 10,
    include_vix: bool | None = None,
    include_vol_regime: bool | None = None,
    include_cross_asset: bool | None = None,
) -> list[dict[str, Any]]:
    """Annotate each snapshot in-place with realized-vol / return features.

    ``realized_vol`` and ``spot_return`` are computed from the trailing window
    up to and including that snapshot, so the features are causal (no
    lookahead) and safe to use as model inputs.
    """
    use_vol = _include_vol_regime() if include_vol_regime is None else include_vol_regime
    use_vix = use_vol if include_vix is None else include_vix
    use_cross = use_vol if include_cross_asset is None else include_cross_asset
    vol_regime = fetch_vol_regime() if use_vol else {}
    cross_asset = fetch_cross_asset_returns() if use_cross else {}
    vix = vol_regime.get("vix_level", 0.0) if use_vix else 0.0
    for i, row in enumerate(history):
        trailing = history[: i + 1]
        row["realized_vol"] = realized_volatility(trailing, window=vol_window)
        row["spot_return"] = latest_spot_return(trailing)
        row["vix_level"] = vix
        if vol_regime:
            row.update(vol_regime)
        if cross_asset:
            row.update(cross_asset)
    return history
