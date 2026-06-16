"""In-process metrics surfaced by the processor health endpoint."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ProcessorMetrics:
    last_refresh_at: datetime | None = None
    last_uw_fetch_ms: float | None = None
    last_postgres_write_ms: float | None = None
    last_strikes_written: int = 0
    last_skipped_duplicate: bool = False
    last_validation_status: str | None = None
    last_error: str | None = None
    refresh_count: int = 0
    skipped_duplicate_count: int = 0
    rejected_count: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_refresh_at": self.last_refresh_at.isoformat() if self.last_refresh_at else None,
            "last_uw_fetch_ms": self.last_uw_fetch_ms,
            "last_postgres_write_ms": self.last_postgres_write_ms,
            "last_strikes_written": self.last_strikes_written,
            "last_skipped_duplicate": self.last_skipped_duplicate,
            "last_validation_status": self.last_validation_status,
            "last_error": self.last_error,
            "refresh_count": self.refresh_count,
            "skipped_duplicate_count": self.skipped_duplicate_count,
            "rejected_count": self.rejected_count,
            **self.extra,
        }


_LOCK = threading.Lock()
_METRICS = ProcessorMetrics()


def get_processor_metrics() -> ProcessorMetrics:
    with _LOCK:
        return ProcessorMetrics(
            last_refresh_at=_METRICS.last_refresh_at,
            last_uw_fetch_ms=_METRICS.last_uw_fetch_ms,
            last_postgres_write_ms=_METRICS.last_postgres_write_ms,
            last_strikes_written=_METRICS.last_strikes_written,
            last_skipped_duplicate=_METRICS.last_skipped_duplicate,
            last_validation_status=_METRICS.last_validation_status,
            last_error=_METRICS.last_error,
            refresh_count=_METRICS.refresh_count,
            skipped_duplicate_count=_METRICS.skipped_duplicate_count,
            rejected_count=_METRICS.rejected_count,
            extra=dict(_METRICS.extra),
        )


def record_refresh_result(
    *,
    uw_fetch_ms: float | None = None,
    postgres_write_ms: float | None = None,
    strikes_written: int = 0,
    skipped_duplicate: bool = False,
    validation_status: str | None = None,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    with _LOCK:
        _METRICS.last_refresh_at = datetime.now(timezone.utc)
        if uw_fetch_ms is not None:
            _METRICS.last_uw_fetch_ms = uw_fetch_ms
        if postgres_write_ms is not None:
            _METRICS.last_postgres_write_ms = postgres_write_ms
        _METRICS.last_strikes_written = strikes_written
        _METRICS.last_skipped_duplicate = skipped_duplicate
        _METRICS.last_validation_status = validation_status
        _METRICS.last_error = error
        if skipped_duplicate:
            _METRICS.skipped_duplicate_count += 1
        elif validation_status == "rejected":
            _METRICS.rejected_count += 1
        elif error is None:
            _METRICS.refresh_count += 1
        if extra:
            _METRICS.extra.update(extra)
