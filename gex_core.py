"""
Fast GEX analytics primitives and option data quality filters.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests

CONTRACT_SIZE = 100
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("GEX_REQUEST_TIMEOUT", "10"))
MIN_OPEN_INTEREST = int(os.environ.get("GEX_MIN_OPEN_INTEREST", "1"))
MIN_GAMMA = float(os.environ.get("GEX_MIN_GAMMA", "0"))
MAX_IV = float(os.environ.get("GEX_MAX_IV", "6.0"))
MAX_BID_ASK_SPREAD_PCT = float(os.environ.get("GEX_MAX_BID_ASK_SPREAD_PCT", "1.0"))
MAX_STRIKE_DISTANCE_PCT = float(os.environ.get("GEX_MAX_STRIKE_DISTANCE_PCT", "0.35"))

_SYMBOL_PATTERN = re.compile(
    r"^(?P<root>[A-Z]+)(?P<expiration>\d{6})(?P<type>[CP])(?P<strike_raw>\d{8})$"
)

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


def clean_option_data(data: pd.DataFrame, spot: float | None = None) -> pd.DataFrame:
    """Parse OCC symbols and apply quality filters."""
    if "option" not in data.columns:
        raise ValueError("Option payload is missing 'option' symbols.")
    if "gamma" not in data.columns or "open_interest" not in data.columns:
        raise ValueError("Option payload is missing required columns: gamma/open_interest.")

    cleaned = data.copy()
    symbols = cleaned["option"].astype(str).str.strip()
    parsed = symbols.str.extract(_SYMBOL_PATTERN)
    cleaned = cleaned.join(parsed)

    cleaned["strike"] = pd.to_numeric(cleaned["strike_raw"], errors="coerce") / 1000.0
    cleaned["gamma"] = pd.to_numeric(cleaned["gamma"], errors="coerce")
    cleaned["open_interest"] = pd.to_numeric(cleaned["open_interest"], errors="coerce")
    cleaned["expiration"] = pd.to_datetime(cleaned["expiration"], format="%y%m%d", errors="coerce")

    if "iv" in cleaned.columns:
        cleaned["iv"] = pd.to_numeric(cleaned["iv"], errors="coerce")
    if "bid" in cleaned.columns and "ask" in cleaned.columns:
        cleaned["bid"] = pd.to_numeric(cleaned["bid"], errors="coerce")
        cleaned["ask"] = pd.to_numeric(cleaned["ask"], errors="coerce")

    cleaned = cleaned.dropna(subset=["type", "strike", "gamma", "open_interest", "expiration"])
    cleaned = cleaned.loc[cleaned["open_interest"] >= MIN_OPEN_INTEREST]
    cleaned = cleaned.loc[cleaned["gamma"] > MIN_GAMMA]

    today = pd.Timestamp(datetime.today().date())
    cleaned = cleaned.loc[cleaned["expiration"].dt.normalize() >= today]

    if spot is not None and MAX_STRIKE_DISTANCE_PCT > 0:
        lower = spot * (1 - MAX_STRIKE_DISTANCE_PCT)
        upper = spot * (1 + MAX_STRIKE_DISTANCE_PCT)
        cleaned = cleaned.loc[(cleaned["strike"] >= lower) & (cleaned["strike"] <= upper)]

    if "iv" in cleaned.columns:
        cleaned = cleaned.loc[(cleaned["iv"].isna()) | ((cleaned["iv"] > 0) & (cleaned["iv"] <= MAX_IV))]

    if {"bid", "ask"}.issubset(cleaned.columns):
        mid = (cleaned["bid"] + cleaned["ask"]) / 2.0
        spread = (cleaned["ask"] - cleaned["bid"]).abs()
        has_quote = (cleaned["bid"] > 0) & (cleaned["ask"] > 0) & (cleaned["ask"] >= cleaned["bid"])
        spread_ok = spread <= (mid.abs() * MAX_BID_ASK_SPREAD_PCT).clip(lower=0.01)
        cleaned = cleaned.loc[(~has_quote) | (has_quote & spread_ok)]

    if "option" in cleaned.columns:
        cleaned = cleaned.sort_values("open_interest", ascending=False).drop_duplicates(
            subset=["option"], keep="first"
        )

    if "charm" in cleaned.columns:
        cleaned["charm"] = pd.to_numeric(cleaned["charm"], errors="coerce").fillna(0.0)

    if cleaned.empty:
        raise RuntimeError("No valid option rows remained after data cleaning/parsing.")
    return cleaned.reset_index(drop=True)


def attach_signed_gex(spot: float, data: pd.DataFrame) -> pd.DataFrame:
    """Vectorized per-contract signed GEX (notional $ per 1% move)."""
    out = data
    gross = spot * out["gamma"].to_numpy(dtype=float) * out["open_interest"].to_numpy(dtype=float)
    gross *= CONTRACT_SIZE * spot * 0.01
    sign = np.where(out["type"].to_numpy() == "P", -1.0, 1.0)
    out = out.copy()
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


def data_quality_report(before: int, after: int) -> dict[str, int]:
    return {"rows_before": before, "rows_after": after, "rows_removed": max(0, before - after)}
