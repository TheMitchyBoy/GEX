"""Runtime mode flags for processor vs full web stack."""

from __future__ import annotations

import os


def is_processor_mode() -> bool:
    return os.environ.get("GEX_PROCESSOR_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def summary_market_features_enabled() -> bool:
    """Optional yfinance VIX/cross-asset fields in summary_json (off in processor by default)."""
    default = "0" if is_processor_mode() else "1"
    flag = os.environ.get("GEX_SUMMARY_MARKET_FEATURES", default).strip().lower()
    return flag in {"1", "true", "yes", "on"}
