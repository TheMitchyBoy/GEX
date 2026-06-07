"""Shared SQLite connection helpers."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def connect_sqlite(path: Path, *, schema_sql: str | None = None, timeout: float = 30.0) -> sqlite3.Connection:
    """Open SQLite, enabling WAL when the database directory is writable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=timeout)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError as exc:
        logger.warning("SQLite WAL unavailable for %s (%s); using default journal mode", path, exc)
    if schema_sql:
        conn.executescript(schema_sql)
    return conn
