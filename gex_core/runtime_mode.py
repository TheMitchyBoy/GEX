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


def lightweight_market_context_enabled() -> bool:
    """Cached VIX/cross-asset context on the processor hot path (default on in processor mode)."""
    explicit = os.environ.get("GEX_LIGHTWEIGHT_MARKET_CONTEXT", "").strip().lower()
    if explicit in {"0", "false", "no", "off"}:
        return False
    if explicit in {"1", "true", "yes", "on"}:
        return True
    return is_processor_mode()


def reconcile_predictions_enabled() -> bool:
    """Resolve open llm_predictions rows when a new snapshot lands."""
    default = "1" if is_processor_mode() else "0"
    flag = os.environ.get("GEX_RECONCILE_PREDICTIONS", default).strip().lower()
    return flag in {"1", "true", "yes", "on"}
