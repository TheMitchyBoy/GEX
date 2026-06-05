"""
Unusual Whales API loader for GEX analytics.

Fetches Gamma Exposure (GEX) by strike directly from the Unusual Whales API,
which uses verified trade-level data rather than open-interest assumptions.

Endpoints used
--------------
  GET /api/stock/{ticker}/greek-exposure/strike
      774 strikes, all expirations combined.  Returns call_gex, put_gex,
      call_delta, put_delta, call_charm, put_charm, call_vanna, put_vanna.
      Values are in raw dollars (divide by 1e6 to get Bn$/%).

  GET /api/stock/{ticker}/spot-exposures/strike
      ~50 ATM strikes.  Used for the current spot price (``price`` field)
      and intraday granularity (OI vs volume vs bid/ask gamma).

Authentication
--------------
Set the environment variable before running:

    export UW_API_KEY=your-key-here

Or pass ``api_key`` explicitly to :func:`fetch_uw_gex`.

Usage
-----
CLI::

    python main.py --ticker SPX

Python::
    from gex_core.uw_loader import fetch_uw_gex
    spot, agg = fetch_uw_gex("SPX")
    # agg is a GexAggregates namedtuple ready for charts and export
"""
from __future__ import annotations

import logging
import os
import time

import pandas as pd
import requests

from gex_core.pipeline import GexAggregates

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.unusualwhales.com"
_CLIENT_ID = "100001"

# Request resilience. Transient UW failures (timeouts, rate limits, gateway
# errors) should not abort an entire snapshot refresh. Retry a few times with
# exponential backoff; configurable via env for ops tuning.
_REQUEST_TIMEOUT = float(os.environ.get("UW_HTTP_TIMEOUT", "15"))
_MAX_RETRIES = int(os.environ.get("UW_HTTP_MAX_RETRIES", "3"))
_BACKOFF_BASE_SECONDS = float(os.environ.get("UW_HTTP_BACKOFF_SECONDS", "1.0"))
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# UW greek-exposure values are in millions of dollars (M$).
# Divide by 1e3 to convert M$ → Bn$ (our pipeline unit).
_GEX_SCALE = 1e3


# ─────────────────────────────────────────────────────────────────────────────
# Exposure normalization helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_net_exposure(
    frame: pd.DataFrame,
    *,
    call_col: str,
    put_col: str,
    net_col: str | None = None,
) -> pd.Series:
    """
    Build signed net exposure from call/put columns with UW-convention fallback.

    Prefer the explicit net column from UW when available. Otherwise infer whether
    put exposure is already signed (sum) or unsigned magnitude (difference).
    """
    if net_col and net_col in frame.columns:
        net = pd.to_numeric(frame[net_col], errors="coerce")
        if net.notna().any():
            return net

    calls = pd.to_numeric(frame.get(call_col), errors="coerce").fillna(0.0)
    puts = pd.to_numeric(frame.get(put_col), errors="coerce").fillna(0.0)

    sum_candidate = calls + puts
    diff_candidate = calls - puts

    # Heuristic: if puts are mostly negative, they are likely already signed.
    put_negative_share = float((puts < 0).mean()) if len(puts) else 0.0
    if put_negative_share >= 0.55:
        return sum_candidate
    return diff_candidate


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _api_key(api_key: str | None = None) -> str:
    key = api_key or os.environ.get("UW_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "Unusual Whales API key not found.\n"
            "Set it with:  export UW_API_KEY=<your-key>\n"
            "Or add UW_API_KEY to Cursor Cloud Agents → Secrets."
        )
    return key


def _retry_after_seconds(resp: requests.Response, attempt: int) -> float:
    """Backoff delay for a retryable response, honoring ``Retry-After`` if sent."""
    header = resp.headers.get("Retry-After") if resp is not None else None
    if header:
        try:
            return min(float(header), 30.0)
        except ValueError:
            pass
    return _BACKOFF_BASE_SECONDS * (2 ** attempt)


def _log_rate_limit(resp: requests.Response, path: str) -> None:
    """Log UW rate-limit/quota headers so throttling is visible in service logs."""
    if resp is None:
        return
    daily = resp.headers.get("x-uw-daily-req-count")
    daily_limit = resp.headers.get("x-uw-token-req-limit")
    per_min_remaining = resp.headers.get("x-uw-req-per-minute-remaining")
    per_min_reset = resp.headers.get("x-uw-req-per-minute-reset")
    if any(v is not None for v in (daily, daily_limit, per_min_remaining, per_min_reset)):
        logger.warning(
            "UW rate-limit on %s: status=%s daily=%s/%s per_minute_remaining=%s reset_ms=%s",
            path, resp.status_code, daily, daily_limit, per_min_remaining, per_min_reset,
        )


def _get(path: str, api_key: str | None = None, **params) -> list[dict]:
    """GET a UW API endpoint and return the ``data`` list.

    Retries transient failures (connection errors, timeouts, HTTP 429/5xx) with
    exponential backoff so a single hiccup does not abort a snapshot refresh.
    Non-retryable errors (e.g. 401/403/404) are raised immediately.
    """
    key = _api_key(api_key)
    url = f"{_BASE_URL}{path}"
    headers = {
        "Authorization": f"Bearer {key}",
        "UW-CLIENT-API-ID": _CLIENT_ID,
    }
    clean_params = {key: value for key, value in params.items() if value is not None}

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = requests.get(
                url, headers=headers, params=clean_params or None, timeout=_REQUEST_TIMEOUT
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                delay = _BACKOFF_BASE_SECONDS * (2 ** attempt)
                logger.warning(
                    "UW request %s failed (%s); retry %d/%d in %.1fs",
                    path, type(exc).__name__, attempt + 1, _MAX_RETRIES, delay,
                )
                time.sleep(delay)
                continue
            raise

        if resp.status_code in _RETRYABLE_STATUS:
            _log_rate_limit(resp, path)
            if attempt < _MAX_RETRIES:
                delay = _retry_after_seconds(resp, attempt)
                logger.warning(
                    "UW request %s returned %d; retry %d/%d in %.1fs",
                    path, resp.status_code, attempt + 1, _MAX_RETRIES, delay,
                )
                time.sleep(delay)
                continue

        resp.raise_for_status()
        payload = resp.json()
        if "error" in payload:
            raise RuntimeError(f"UW API error for {path!r}: {payload['error']}")
        return payload.get("data", [])

    # Exhausted retries on a transient network error.
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"UW request {path!r} failed after {_MAX_RETRIES} retries")


# ─────────────────────────────────────────────────────────────────────────────
# Spot price
# ─────────────────────────────────────────────────────────────────────────────

def fetch_uw_spot(
    ticker: str,
    api_key: str | None = None,
    date: str | None = None,
) -> float:
    """
    Return the current spot price for *ticker* from UW spot-exposures.

    Falls back to the ``price`` field on the first row of the response.
    """
    rows = _get(f"/api/stock/{ticker}/spot-exposures/strike", api_key=api_key, date=date)
    if not rows:
        raise ValueError(f"No spot-exposure data returned for {ticker!r}.")
    price = rows[0].get("price")
    if price is None:
        raise ValueError(f"Spot price missing from UW spot-exposures response for {ticker!r}.")
    return float(price)


# ─────────────────────────────────────────────────────────────────────────────
# Greek exposure by strike
# ─────────────────────────────────────────────────────────────────────────────

def fetch_uw_greek_exposure(
    ticker: str,
    api_key: str | None = None,
    date: str | None = None,
) -> pd.DataFrame:
    """
    Fetch the full GEX-by-strike table from ``/greek-exposure/strike``.

    Returns a DataFrame with columns:
      strike (float), call_gex, put_gex, net_gex,
      call_delta, put_delta, call_charm, put_charm, call_vanna, put_vanna
    All GEX/greek values are in Bn$/% (already scaled from raw dollars).
    """
    rows = _get(f"/api/stock/{ticker}/greek-exposure/strike", api_key=api_key, date=date)
    if not rows:
        raise ValueError(f"No greek-exposure data returned for {ticker!r}.")

    df = pd.DataFrame(rows)
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")

    # Convert all numeric greek columns from raw $ → Bn$
    greek_cols = [c for c in df.columns if c not in ("date", "strike")]
    for col in greek_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce") / _GEX_SCALE

    df["net_gex"] = _normalize_net_exposure(
        df,
        call_col="call_gex",
        put_col="put_gex",
        net_col="net_gex",
    )
    df = df.dropna(subset=["strike"]).sort_values("strike").reset_index(drop=True)
    if "date" in df.columns and df["date"].notna().any():
        df.attrs["market_date"] = str(df["date"].dropna().iloc[0])
    logger.info("Fetched %d strike rows from UW greek-exposure for %s", len(df), ticker)
    return df


def fetch_uw_greek_exposure_by_expiration(
    ticker: str,
    api_key: str | None = None,
    date: str | None = None,
) -> pd.Series:
    """Best-effort expiration-level GEX; empty series when endpoint unavailable."""
    for path in (
        f"/api/stock/{ticker}/greek-exposure/expiry",
        f"/api/stock/{ticker}/greek-exposure/expiration",
    ):
        try:
            rows = _get(path, api_key=api_key, date=date)
        except Exception:
            continue
        if not rows:
            continue
        df = pd.DataFrame(rows)
        exp_col = next((c for c in ("expiry", "expiration", "date") if c in df.columns), None)
        gex_col = next(
            (c for c in ("net_gex", "gex", "total_gex", "call_gex") if c in df.columns),
            None,
        )
        if not exp_col or not gex_col:
            continue
        values = pd.to_numeric(df[gex_col], errors="coerce").fillna(0.0) / _GEX_SCALE
        idx = pd.to_datetime(df[exp_col], errors="coerce")
        series = pd.Series(values.values, index=idx, dtype=float)
        series = series[~series.index.isna()].sort_index()
        if not series.empty:
            logger.info("Fetched %d expiration rows from UW %s for %s", len(series), path, ticker)
            return series
    return pd.Series(dtype=float)


def fetch_uw_spot_exposures(
    ticker: str,
    api_key: str | None = None,
    date: str | None = None,
) -> pd.DataFrame:
    """
    Fetch the intraday ATM spot-exposures table (~50 strikes around spot).

    Columns include: strike, price (spot), call_gamma_oi, put_gamma_oi,
    call_gamma_vol, put_gamma_vol, call_gamma_bid, put_gamma_bid, etc.

    Note: The raw values from this endpoint are in raw dollars (not M$);
    they should be interpreted as a relative comparison across strikes,
    not directly converted to Bn$.
    """
    rows = _get(f"/api/stock/{ticker}/spot-exposures/strike", api_key=api_key, date=date)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")  # spot price

    num_cols = [c for c in df.columns if c not in ("date", "ticker", "time", "strike", "price")]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "call_gamma_oi" in df.columns and "put_gamma_oi" in df.columns:
        df["net_gamma_oi"] = _normalize_net_exposure(
            df,
            call_col="call_gamma_oi",
            put_col="put_gamma_oi",
        )
        # spot-exposure endpoint is raw-dollar scale per 1% move
        df["net_gamma_oi_bn"] = pd.to_numeric(df["net_gamma_oi"], errors="coerce") / 1e9
    df = df.dropna(subset=["strike"]).sort_values("strike").reset_index(drop=True)
    if "date" in df.columns and df["date"].notna().any():
        df.attrs["market_date"] = str(df["date"].dropna().iloc[0])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main entry-point: GexAggregates from UW
# ─────────────────────────────────────────────────────────────────────────────

def fetch_uw_gex(
    ticker: str = "SPX",
    api_key: str | None = None,
    date: str | None = None,
) -> tuple[float, GexAggregates]:
    """
    Fetch GEX aggregates from the Unusual Whales API.

    Parameters
    ----------
    ticker : str
        Underlying ticker (default ``"SPX"``).
    api_key : str | None
        UW API key.  Falls back to the ``UW_API_KEY`` environment variable.

    Returns
    -------
    spot : float
        Current underlying price.
    agg : GexAggregates
        Ready-to-use aggregates in Bn$/%.

        - ``gex_by_strike``     — net GEX (call + put) per strike
        - ``cumulative_gex``    — running sum (gamma flip visible at zero-cross)
        - ``gex_by_expiration`` — empty Series (not available at this granularity)
        - ``surface_data``      — empty DataFrame (not available at this granularity)
        - ``total_gex_bn``      — total net GEX in Bn$/%

    Notes
    -----
    UW computes GEX from verified transaction data (buy/sell flags) rather
    than open-interest alone, so values may differ from CBOE-based estimates.
    """
    df = fetch_uw_greek_exposure(ticker, api_key=api_key, date=date)
    spot_df = fetch_uw_spot_exposures(ticker, api_key=api_key, date=date)
    spot = float(spot_df["price"].dropna().iloc[0]) if not spot_df.empty and "price" in spot_df.columns else fetch_uw_spot(
        ticker, api_key=api_key, date=date
    )
    logger.info("UW spot price for %s: %.2f", ticker, spot)

    gex_by_expiration = fetch_uw_greek_exposure_by_expiration(ticker, api_key=api_key, date=date)

    gex_by_strike = pd.Series(
        df["net_gex"].values,
        index=df["strike"].values,
        name="GEX",
        dtype=float,
    )
    gex_by_strike.index.name = "strike"
    gex_by_strike = gex_by_strike.sort_index()

    cumulative_gex = gex_by_strike.cumsum()
    total_gex_bn = float(gex_by_strike.sum())
    market_date = df.attrs.get("market_date")
    if market_date:
        gex_by_strike.attrs["market_date"] = market_date
        cumulative_gex.attrs["market_date"] = market_date
        if not gex_by_expiration.empty:
            gex_by_expiration.attrs["market_date"] = market_date

    surface_data = pd.DataFrame()
    if not df.empty and {"strike", "net_gex"}.issubset(df.columns):
        surface_data = df[["strike", "net_gex"]].rename(columns={"net_gex": "GEX"}).copy()
        if "call_charm" in df.columns:
            surface_data["charm"] = pd.to_numeric(df["call_charm"], errors="coerce").fillna(0.0) + pd.to_numeric(
                df["put_charm"], errors="coerce"
            ).fillna(0.0)
        if "call_vanna" in df.columns:
            surface_data["vanna"] = pd.to_numeric(df["call_vanna"], errors="coerce").fillna(0.0) + pd.to_numeric(
                df["put_vanna"], errors="coerce"
            ).fillna(0.0)

    gex_by_strike.attrs["greek_exposure_df"] = df
    gex_by_strike.attrs["spot_exposures_df"] = spot_df

    logger.info(
        "UW GEX for %s: total=%.3f Bn$, strikes=%d, gamma_flip near zero-crossing",
        ticker, total_gex_bn, len(gex_by_strike),
    )

    return spot, GexAggregates(
        gex_by_strike=gex_by_strike,
        gex_by_expiration=gex_by_expiration,
        cumulative_gex=cumulative_gex,
        surface_data=surface_data,
        total_gex_bn=total_gex_bn,
    )


def fetch_uw_charm_vanna(
    ticker: str = "SPX",
    api_key: str | None = None,
    date: str | None = None,
) -> dict[str, pd.Series]:
    """
    Return charm and vanna exposure by strike.

    Values are in UW's native units for each greek (charm: ∂Δ/∂t,
    vanna: ∂Δ/∂σ), scaled by the same _GEX_SCALE as GEX columns.
    They are useful for relative comparison across strikes but should NOT
    be directly compared to GEX (Bn$) values — the formulae differ.

    Keys: ``call_charm``, ``put_charm``, ``net_charm``,
          ``call_vanna``, ``put_vanna``, ``net_vanna``.
    """
    df = fetch_uw_greek_exposure(ticker, api_key=api_key, date=date)
    idx = df["strike"].values
    return {
        "call_charm":  pd.Series(df["call_charm"].values, index=idx),
        "put_charm":   pd.Series(df["put_charm"].values,  index=idx),
        "net_charm":   pd.Series((df["call_charm"] + df["put_charm"]).values, index=idx),
        "call_vanna":  pd.Series(df["call_vanna"].values, index=idx),
        "put_vanna":   pd.Series(df["put_vanna"].values,  index=idx),
        "net_vanna":   pd.Series((df["call_vanna"] + df["put_vanna"]).values, index=idx),
    }
