"""SQLite-backed chat session persistence."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

_CHAT_SCHEMA = """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            ticker TEXT,
            messages_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated
            ON chat_sessions (updated_at DESC);
        """


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect():
    from gex_core.trading.journal import db_path
    from gex_core.sqlite_util import connect_sqlite

    return connect_sqlite(db_path(), schema_sql=_CHAT_SCHEMA)


def _session_ttl_days() -> int:
    try:
        return max(1, int(os.environ.get("GEX_CHAT_PERSIST_DAYS", "14")))
    except (TypeError, ValueError):
        return 14


def load_session(session_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT session_id, ticker, messages_json, updated_at FROM chat_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    if not row:
        return None
    try:
        messages = json.loads(row["messages_json"])
    except json.JSONDecodeError:
        messages = []
    return {
        "session_id": row["session_id"],
        "ticker": row["ticker"],
        "messages": messages,
        "updated_at": row["updated_at"],
    }


def save_session(
    *,
    session_id: str,
    messages: list[dict[str, str]],
    ticker: str | None = None,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO chat_sessions (session_id, ticker, messages_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                ticker = excluded.ticker,
                messages_json = excluded.messages_json,
                updated_at = excluded.updated_at
            """,
            (session_id, (ticker or "").upper() or None, json.dumps(messages), _now_iso()),
        )
        conn.commit()


def delete_session(session_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        conn.commit()


def prune_old_sessions() -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(days=_session_ttl_days())
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM chat_sessions WHERE updated_at < ?",
            (cutoff.isoformat(),),
        )
        conn.commit()
        return int(cur.rowcount or 0)
