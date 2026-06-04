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
CLI:
    python main.py --ticker SPX --uw

Python:
    from gex_core.uw_loader import fetch_uw_gex
    spot, agg = fetch_uw_gex("SPX")
    # agg is a GexAggregates namedtuple ready for charts and export
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

import pandas as pd
import requests

from gex_core.pipeline import GexAggregates

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.unusualwhales.com"
_CLIENT_ID = "100001"

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


def _get(path: str, api_key: str | None = None, **params) -> list[dict]:
    """GET a UW API endpoint and return the ``data`` list."""
    key = _api_key(api_key)
    url = f"{_BASE_URL}{path}"
    headers = {
        "Authorization": f"Bearer {key}",
        "UW-CLIENT-API-ID": _CLIENT_ID,
    }
    resp = requests.get(url, headers=headers, params=params or None, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    if "error" in payload:
        raise RuntimeError(f"UW API error for {path!r}: {payload['error']}")
    return payload.get("data", [])


# ─────────────────────────────────────────────────────────────────────────────
# Spot price
# ─────────────────────────────────────────────────────────────────────────────

def fetch_uw_spot(ticker: str, api_key: str | None = None) -> float:
    """
    Return the current spot price for *ticker* from UW spot-exposures.

    Falls back to the ``price`` field on the first row of the response.
    """
    rows = _get(f"/api/stock/{ticker}/spot-exposures/strike", api_key=api_key)
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
) -> pd.DataFrame:
    """
    Fetch the full GEX-by-strike table from ``/greek-exposure/strike``.

    Returns a DataFrame with columns:
      strike (float), call_gex, put_gex, net_gex,
      call_delta, put_delta, call_charm, put_charm, call_vanna, put_vanna
    All GEX/greek values are in Bn$/% (already scaled from raw dollars).
    """
    rows = _get(f"/api/stock/{ticker}/greek-exposure/strike", api_key=api_key)
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
    logger.info("Fetched %d strike rows from UW greek-exposure for %s", len(df), ticker)
    return df


def fetch_uw_spot_exposures(
    ticker: str,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetch the intraday ATM spot-exposures table (~50 strikes around spot).

    Columns include: strike, price (spot), call_gamma_oi, put_gamma_oi,
    call_gamma_vol, put_gamma_vol, call_gamma_bid, put_gamma_bid, etc.

    Note: The raw values from this endpoint are in raw dollars (not M$);
    they should be interpreted as a relative comparison across strikes,
    not directly converted to Bn$.
    """
    rows = _get(f"/api/stock/{ticker}/spot-exposures/strike", api_key=api_key)
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
    return df.dropna(subset=["strike"]).sort_values("strike").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry-point: GexAggregates from UW
# ─────────────────────────────────────────────────────────────────────────────

def fetch_uw_gex(
    ticker: str = "SPX",
    api_key: str | None = None,
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
    # Fetch both endpoints in parallel would be nice; keeping it simple here.
    spot = fetch_uw_spot(ticker, api_key=api_key)
    logger.info("UW spot price for %s: %.2f", ticker, spot)

    df = fetch_uw_greek_exposure(ticker, api_key=api_key)

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

    logger.info(
        "UW GEX for %s: total=%.3f Bn$, strikes=%d, gamma_flip near zero-crossing",
        ticker, total_gex_bn, len(gex_by_strike),
    )

    return spot, GexAggregates(
        gex_by_strike=gex_by_strike,
        gex_by_expiration=pd.Series(dtype=float),  # not available at strike level
        cumulative_gex=cumulative_gex,
        surface_data=pd.DataFrame(),               # not available at strike level
        total_gex_bn=total_gex_bn,
    )


def fetch_uw_charm_vanna(
    ticker: str = "SPX",
    api_key: str | None = None,
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
    df = fetch_uw_greek_exposure(ticker, api_key=api_key)
    idx = df["strike"].values
    return {
        "call_charm":  pd.Series(df["call_charm"].values, index=idx),
        "put_charm":   pd.Series(df["put_charm"].values,  index=idx),
        "net_charm":   pd.Series((df["call_charm"] + df["put_charm"]).values, index=idx),
        "call_vanna":  pd.Series(df["call_vanna"].values, index=idx),
        "put_vanna":   pd.Series(df["put_vanna"].values,  index=idx),
        "net_vanna":   pd.Series((df["call_vanna"] + df["put_vanna"]).values, index=idx),
    }
