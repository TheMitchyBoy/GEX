"""
Contract-level GEX math container.

The live data path uses Unusual Whales strike aggregates (see ``uw_loader``).
This module keeps the shared :class:`GexAggregates` container used across loaders,
charts, and export pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class GexAggregates:
    """Standard bundle of GEX views produced by loaders and aggregators."""

    gex_by_strike: pd.Series
    gex_by_expiration: pd.Series
    cumulative_gex: pd.Series
    surface_data: pd.DataFrame
    total_gex_bn: float
