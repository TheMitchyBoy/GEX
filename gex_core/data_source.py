"""GEX data loading from Unusual Whales only.

``fetch_gex_data`` is the single front door for fresh GEX. It returns a
``GexFetchResult`` with spot price and a ``GexAggregates`` bundle ready for
charts, export, and dashboard refresh. Requires ``UW_API_KEY`` in the environment
or an explicit ``uw_api_key`` argument.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from gex_core.pipeline import GexAggregates

SOURCE_UW = "unusual_whales"


@dataclass
class GexFetchResult:
    ticker: str
    source: str
    spot: float
    aggregates: GexAggregates
    market_date: str | None = None
    option_data: pd.DataFrame | None = None
    greek_exposure_df: pd.DataFrame | None = None
    spot_exposures_df: pd.DataFrame | None = None
    data_quality: Any | None = None
    note: str = ""


def fetch_gex_data(
    ticker: str,
    *,
    uw_api_key: str | None = None,
    market_date: str | None = None,
    **_ignored,
) -> GexFetchResult:
    """
    Fetch GEX aggregates from Unusual Whales.

    Requires ``UW_API_KEY`` or an explicit ``uw_api_key``.
    """
    from gex_core.uw_loader import fetch_uw_gex

    api_key = uw_api_key or os.environ.get("UW_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Unusual Whales API key required.\n"
            "Set: export UW_API_KEY=<your-key>"
        )

    ticker = ticker.upper()
    spot, agg = fetch_uw_gex(ticker, api_key=api_key, date=market_date)
    greek_df = agg.gex_by_strike.attrs.get("greek_exposure_df")
    spot_df = agg.gex_by_strike.attrs.get("spot_exposures_df")
    return GexFetchResult(
        ticker=ticker,
        source=SOURCE_UW,
        spot=spot,
        aggregates=agg,
        market_date=agg.gex_by_strike.attrs.get("market_date") or market_date,
        option_data=None,
        greek_exposure_df=greek_df if isinstance(greek_df, pd.DataFrame) else None,
        spot_exposures_df=spot_df if isinstance(spot_df, pd.DataFrame) else None,
        note="Unusual Whales greek-exposure by strike",
    )
