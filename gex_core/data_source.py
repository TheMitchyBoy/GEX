"""Unified GEX data loading: Unusual Whales primary, CBOE backup."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from gex_core.pipeline import GexAggregates, aggregate_gex, attach_signed_gex, fetch_options_payload, parse_payload
from gex_core.data_quality import clean_option_data

logger = logging.getLogger(__name__)

SOURCE_UW = "unusual_whales"
SOURCE_CBOE = "cboe"
SOURCE_GCS = "gcs"


@dataclass
class GexFetchResult:
    ticker: str
    source: str
    spot: float
    aggregates: GexAggregates
    option_data: pd.DataFrame | None = None
    data_quality: Any | None = None
    note: str = ""


def _fetch_uw(ticker: str, api_key: str | None = None) -> GexFetchResult:
    from gex_core.uw_loader import fetch_uw_gex

    spot, agg = fetch_uw_gex(ticker, api_key=api_key)
    return GexFetchResult(
        ticker=ticker.upper(),
        source=SOURCE_UW,
        spot=spot,
        aggregates=agg,
        option_data=None,
        note="Unusual Whales greek-exposure by strike",
    )


def _fetch_cboe(
    ticker: str,
    *,
    refresh: bool = False,
    cache_ttl_minutes: int = 15,
    max_dte: int = 365,
    strike_window_pct: float = 0.01,
    gcs_source: str | None = None,
    spot_override: float | None = None,
) -> GexFetchResult:
    from gex_core.cboe_loader import scrape_data

    spot, option_data, quality = scrape_data(
        ticker=ticker,
        refresh=refresh,
        cache_ttl_minutes=cache_ttl_minutes,
        gcs_source=gcs_source,
        spot_override=spot_override,
    )
    option_data = attach_signed_gex(spot, option_data)
    agg = aggregate_gex(
        option_data,
        spot=spot,
        max_dte=max_dte,
        strike_window_pct=strike_window_pct,
    )
    return GexFetchResult(
        ticker=ticker.upper(),
        source=SOURCE_GCS if gcs_source else SOURCE_CBOE,
        spot=spot,
        aggregates=agg,
        option_data=option_data,
        data_quality=quality,
        note="CBOE delayed quotes" if not gcs_source else f"GCS source {gcs_source}",
    )


def fetch_gex_data(
    ticker: str,
    *,
    prefer_uw: bool = True,
    uw_api_key: str | None = None,
    refresh: bool = False,
    cache_ttl_minutes: int = 15,
    max_dte: int = 365,
    strike_window_pct: float = 0.01,
    gcs_source: str | None = None,
    spot_override: float | None = None,
    force_cboe: bool = False,
) -> GexFetchResult:
    """
    Load GEX aggregates. Tries Unusual Whales when a key is available, then CBOE.

    Set ``force_cboe=True`` to skip UW (scheduled CBOE snapshots with full surface).
  """
    ticker = ticker.upper()
    api_key = uw_api_key or os.environ.get("UW_API_KEY")

    if gcs_source:
        return _fetch_cboe(
            ticker,
            refresh=refresh,
            cache_ttl_minutes=cache_ttl_minutes,
            max_dte=max_dte,
            strike_window_pct=strike_window_pct,
            gcs_source=gcs_source,
            spot_override=spot_override,
        )

    if prefer_uw and not force_cboe and api_key:
        try:
            return _fetch_uw(ticker, api_key=api_key)
        except Exception as exc:
            logger.warning("UW fetch failed for %s (%s); falling back to CBOE", ticker, exc)

    return _fetch_cboe(
        ticker,
        refresh=refresh,
        cache_ttl_minutes=cache_ttl_minutes,
        max_dte=max_dte,
        strike_window_pct=strike_window_pct,
        spot_override=spot_override,
    )
