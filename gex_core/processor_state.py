"""Durable processor cursor state in PostgreSQL."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_state(key: str, default: str | None = None) -> str | None:
    from gex_core.db import use_postgres

    if not use_postgres():
        return default
    try:
        import psycopg

        from gex_core.db import database_url, ensure_postgres_schema

        ensure_postgres_schema()
        with psycopg.connect(database_url()) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT value FROM processor_state WHERE key = %s", (key,))
                row = cur.fetchone()
        return str(row[0]) if row else default
    except Exception:
        logger.debug("get_state failed for %s", key, exc_info=True)
        return default


def set_state(key: str, value: str) -> None:
    from gex_core.db import use_postgres

    if not use_postgres():
        return
    try:
        import psycopg

        from gex_core.db import database_url, ensure_postgres_schema

        ensure_postgres_schema()
        with psycopg.connect(database_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO processor_state (key, value, updated_at)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (key) DO UPDATE SET
                        value = EXCLUDED.value,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (key, value, _now_iso()),
                )
            conn.commit()
    except Exception:
        logger.exception("set_state failed for %s", key)


def last_backfilled_date(ticker: str) -> str | None:
    return get_state(f"backfill_last_date:{ticker.upper()}")


def mark_backfilled_through(ticker: str, market_date: str) -> None:
    set_state(f"backfill_last_date:{ticker.upper()}", market_date[:10])
