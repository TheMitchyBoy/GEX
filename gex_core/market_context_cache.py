"""Cached lightweight market context for the processor hot path."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_DEFAULT_TTL_SEC = float(os.environ.get("GEX_MARKET_CONTEXT_CACHE_SEC", "300"))


def _cache_ttl() -> float:
    try:
        return float(os.environ.get("GEX_MARKET_CONTEXT_CACHE_SEC", str(_DEFAULT_TTL_SEC)))
    except (TypeError, ValueError):
        return _DEFAULT_TTL_SEC


def cached_vol_regime(*, force: bool = False) -> dict[str, Any]:
    """Fetch VIX/IV context at most once per TTL window."""
    key = "vol_regime"
    now = time.monotonic()
    cached = _CACHE.get(key)
    if not force and cached and (now - cached[0]) < _cache_ttl():
        return dict(cached[1])
    try:
        from gex_core.market_features import fetch_vol_regime

        payload = fetch_vol_regime() or {}
    except Exception:
        logger.debug("cached_vol_regime fetch failed", exc_info=True)
        payload = {}
    _CACHE[key] = (now, dict(payload))
    return dict(payload)


def cached_cross_asset_returns(*, force: bool = False) -> dict[str, Any]:
    key = "cross_asset"
    now = time.monotonic()
    cached = _CACHE.get(key)
    if not force and cached and (now - cached[0]) < _cache_ttl():
        return dict(cached[1])
    try:
        from gex_core.market_features import fetch_cross_asset_returns

        payload = fetch_cross_asset_returns() or {}
    except Exception:
        logger.debug("cached_cross_asset_returns fetch failed", exc_info=True)
        payload = {}
    _CACHE[key] = (now, dict(payload))
    return dict(payload)


def clear_market_context_cache() -> None:
    _CACHE.clear()
