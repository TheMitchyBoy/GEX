"""UW fetch metadata (rate limits, timing)."""

from __future__ import annotations

from typing import Any

_LAST: dict[str, Any] = {}


def set_last_uw_fetch_metadata(metadata: dict[str, Any]) -> None:
    _LAST.clear()
    _LAST.update(metadata)


def last_uw_fetch_metadata() -> dict[str, Any]:
    return dict(_LAST)
