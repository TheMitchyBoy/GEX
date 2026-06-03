"""Fetch, GEX computation, and aggregation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests

from gex_core.data_quality import DataQualityReport

CONTRACT_SIZE = 100
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("GEX_REQUEST_TIMEOUT", "10"))

_HTTP = requests.Session()
_HTTP.headers.update({"User-Agent": "GEX-Tracker/1.0"})


@dataclass(frozen=True)
class GexAggregates:
    gex_by_strike: pd.Series
    gex_by_expiration: pd.Series
    cumulative_gex: pd.Series
    surface_data: pd.DataFrame
    total_gex_bn: float


def fetch_options_payload(ticker: str) -> dict[str, Any]:
    endpoints = [
        f"https://cdn.cboe.com/api/global/delayed_quotes/options/_{ticker}.json",
        f"https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker}.json",
    ]
    last_error: Exception | None = None
    for endpoint in endpoints:
        try:
            response = _HTTP.get(endpoint, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as err:
            last_error = err
    raise RuntimeError(f"Could not fetch options data for {ticker}: {last_error}")


def parse_payload(payload: dict[str, Any]) -> tuple[float, pd.DataFrame]:
    if "data" not in payload:
        raise ValueError("Unexpected response format: missing 'data' field.")
    block = payload["data"]
    if "current_price" not in block or "options" not in block:
        raise ValueError("Unexpected response format: missing current price or options.")
    spot_price = float(block["current_price"])
    raw_options = block.get("options") or []
    return spot_price, pd.DataFrame(raw_options)


def attach_signed_gex(spot: float, data: pd.DataFrame) -> pd.DataFrame:
    """Vectorized per-contract signed GEX (notional $ per 1% move)."""
    gross = spot * data["gamma"].to_numpy(dtype=float) * data["open_interest"].to_numpy(dtype=float)
    gross *= CONTRACT_SIZE * spot * 0.01
    sign = np.where(data["type"].to_numpy() == "P", -1.0, 1.0)
    out = data.copy()
    out["GEX"] = gross * sign
    return out


def filter_by_dte(data: pd.DataFrame, max_dte: int) -> pd.DataFrame:
    today = datetime.today().date()
    end = today + timedelta(days=max_dte)
    mask = (data["expiration"].dt.date >= today) & (data["expiration"].dt.date <= end)
    return data.loc[mask]


def filter_by_strike_window(data: pd.DataFrame, spot: float, strike_window_pct: float) -> pd.DataFrame:
    lower = spot * (1 - strike_window_pct)
    upper = spot * (1 + strike_window_pct)
    return data.loc[(data["strike"] > lower) & (data["strike"] < upper)]


def aggregate_gex(
    data: pd.DataFrame,
    spot: float,
    max_dte: int = 365,
    strike_window_pct: float = 0.15,
) -> GexAggregates:
    """Single-pass strike/expiration/surface aggregates."""
    term = filter_by_dte(data, max_dte)
    gex_by_expiration = term.groupby("expiration", sort=True)["GEX"].sum() / 1e9

    gex_by_strike = data.groupby("strike", sort=True)["GEX"].sum() / 1e9
    cumulative_gex = gex_by_strike.sort_index().cumsum()

    surface_src = filter_by_strike_window(term, spot, strike_window_pct)
    surface = (
        surface_src.groupby(["expiration", "strike"], sort=True)["GEX"]
        .sum()
        .div(1e6)
        .reset_index()
    )
    total_gex_bn = float(data["GEX"].sum() / 1e9)
    return GexAggregates(
        gex_by_strike=gex_by_strike,
        gex_by_expiration=gex_by_expiration,
        cumulative_gex=cumulative_gex,
        surface_data=surface,
        total_gex_bn=total_gex_bn,
    )


def data_quality_report(before: int, after: int) -> DataQualityReport:
    """Backward-compatible helper when only row counts are available."""
    report = DataQualityReport(rows_in=before, rows_out=after)
    if before > after:
        report.removed["unspecified"] = before - after
    return report
