"""Build reproducible export metadata stamped into summary JSON."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

from gex_core.data_quality import DataQualityConfig


def filter_config_hash() -> str:
    """Stable short hash of active data-quality filter settings."""
    cfg = DataQualityConfig()
    payload = {
        "GEX_DATA_FILTERS": os.environ.get("GEX_DATA_FILTERS", "1"),
        "GEX_MIN_OPEN_INTEREST": cfg.min_open_interest,
        "GEX_MIN_GAMMA": cfg.min_gamma,
        "GEX_MAX_IV": cfg.max_iv,
        "GEX_MAX_BID_ASK_SPREAD_PCT": cfg.max_bid_ask_spread_pct,
        "GEX_MAX_STRIKE_DISTANCE_PCT": cfg.max_strike_distance_pct,
        "GEX_DEDUPE_SYMBOLS": os.environ.get("GEX_DEDUPE_SYMBOLS", "1"),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:12]


def build_export_metadata(
    ticker: str,
    *,
    market_date: str | None,
    spot: float,
    total_gex_bn: float,
    regime: str,
    data_quality: dict[str, Any] | None = None,
    uw_endpoint: str = "greek-exposure/strike",
) -> dict[str, Any]:
    """Metadata block merged into ``{TICKER}_summary_{ts}.json``."""
    return {
        "export_schema_version": 2,
        "indexed_at_utc": datetime.now(timezone.utc).isoformat(),
        "uw_endpoint": uw_endpoint,
        "spot_source": "unusual_whales",
        "filter_config_hash": filter_config_hash(),
        "data_quality": data_quality,
        "ticker": ticker.upper(),
        "market_date": market_date,
        "spot": float(spot),
        "total_gex_bn_per_pct": float(total_gex_bn),
        "net_gamma_regime": regime,
    }
