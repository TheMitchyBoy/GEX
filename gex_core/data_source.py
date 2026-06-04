"""GEX data loading from Unusual Whales only."""

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
    option_data: pd.DataFrame | None = None
    data_quality: Any | None = None
    note: str = ""


def fetch_gex_data(
    ticker: str,
    *,
    uw_api_key: str | None = None,
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
    spot, agg = fetch_uw_gex(ticker, api_key=api_key)
    return GexFetchResult(
        ticker=ticker,
        source=SOURCE_UW,
        spot=spot,
        aggregates=agg,
        option_data=None,
        note="Unusual Whales greek-exposure by strike",
    )
