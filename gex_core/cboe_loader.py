"""CBOE and GCS option chain loading with local JSON cache."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from gex_core.data_quality import clean_option_data
from gex_core.pipeline import fetch_options_payload, parse_payload

DATA_DIR = Path("data")
DEFAULT_CACHE_TTL_MINUTES = 15


def is_cache_fresh(cache_file: Path, cache_ttl_minutes: int) -> bool:
    if not cache_file.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
    return age < timedelta(minutes=cache_ttl_minutes)


def scrape_data(
    ticker: str,
    refresh: bool = False,
    cache_ttl_minutes: int = DEFAULT_CACHE_TTL_MINUTES,
    gcs_source: str | None = None,
    spot_override: float | None = None,
):
    """Return (spot, option_data, quality_report)."""
    if gcs_source:
        from gex_core.gcs_loader import load_gcs_options

        spot_price, option_data = load_gcs_options(
            gcs_source,
            ticker=ticker,
            spot=spot_override,
        )
        option_data, quality = clean_option_data(option_data, spot=spot_price)
        if option_data.empty:
            raise RuntimeError("No valid option rows after cleaning GCS data.")
        return spot_price, option_data, quality

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = DATA_DIR / f"{ticker.upper()}.json"

    if not refresh and is_cache_fresh(cache_file, cache_ttl_minutes):
        with cache_file.open(encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = fetch_options_payload(ticker)
        with cache_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f)

    spot_price, option_data = parse_payload(payload)
    option_data, quality = clean_option_data(option_data, spot=spot_price)
    if option_data is None or not isinstance(option_data, pd.DataFrame):
        raise RuntimeError("Failed to parse option data from CBOE payload.")
    if option_data.empty:
        raise RuntimeError("Option data was loaded but contains no rows.")
    return spot_price, option_data, quality
