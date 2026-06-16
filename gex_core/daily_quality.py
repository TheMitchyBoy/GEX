"""Daily quality and prediction accuracy rollups."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_daily_quality_stats(
    *,
    ticker: str,
    market_date: str,
    status: str,
    quality_score: float | None,
    strike_count: int | None,
    data_lag_sec: float | None,
    uw_fetch_ms: float | None,
    postgres_write_ms: float | None,
) -> None:
    from gex_core.db import database_url, ensure_postgres_schema, use_postgres

    if not use_postgres():
        return
    ensure_postgres_schema()
    import psycopg

    ticker = ticker.upper()
    market_date = market_date[:10]
    with psycopg.connect(database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT payload_json FROM daily_quality_stats
                WHERE ticker = %s AND market_date = %s
                """,
                (ticker, market_date),
            )
            row = cur.fetchone()
            payload: dict[str, Any] = {}
            if row and row[0]:
                payload = row[0] if isinstance(row[0], dict) else json.loads(row[0])

            counts = payload.get("counts", {})
            counts[status] = int(counts.get(status, 0)) + 1
            payload["counts"] = counts
            payload["snapshots_total"] = int(payload.get("snapshots_total", 0)) + 1

            scores = payload.get("quality_scores", [])
            if quality_score is not None:
                scores.append(float(quality_score))
                payload["quality_scores"] = scores[-500:]
                payload["quality_score_avg"] = round(sum(scores) / len(scores), 4)

            lags = payload.get("data_lag_sec", [])
            if data_lag_sec is not None:
                lags.append(float(data_lag_sec))
                payload["data_lag_sec"] = lags[-500:]
                payload["data_lag_sec_avg"] = round(sum(lags) / len(lags), 2)

            strikes = payload.get("strike_counts", [])
            if strike_count is not None:
                strikes.append(int(strike_count))
                payload["strike_counts"] = strikes[-500:]
                payload["strike_count_avg"] = round(sum(strikes) / len(strikes), 2)

            timings = payload.get("uw_fetch_ms", [])
            if uw_fetch_ms is not None:
                timings.append(float(uw_fetch_ms))
                payload["uw_fetch_ms"] = timings[-500:]
                payload["uw_fetch_ms_avg"] = round(sum(timings) / len(timings), 2)

            writes = payload.get("postgres_write_ms", [])
            if postgres_write_ms is not None:
                writes.append(float(postgres_write_ms))
                payload["postgres_write_ms"] = writes[-500:]
                payload["postgres_write_ms_avg"] = round(sum(writes) / len(writes), 2)

            payload["updated_at"] = _now_iso()
            cur.execute(
                """
                INSERT INTO daily_quality_stats (ticker, market_date, payload_json, updated_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (ticker, market_date) DO UPDATE SET
                    payload_json = EXCLUDED.payload_json,
                    updated_at = EXCLUDED.updated_at
                """,
                (ticker, market_date, json.dumps(payload), _now_iso()),
            )
        conn.commit()


def update_prediction_accuracy_daily(ticker: str, *, market_date: str, outcome: dict[str, Any]) -> None:
    from gex_core.db import database_url, ensure_postgres_schema, use_postgres

    if not use_postgres():
        return
    ensure_postgres_schema()
    import psycopg

    ticker = ticker.upper()
    market_date = market_date[:10]
    with psycopg.connect(database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT payload_json FROM prediction_accuracy_daily
                WHERE ticker = %s AND market_date = %s
                """,
                (ticker, market_date),
            )
            row = cur.fetchone()
            payload: dict[str, Any] = {"resolved": 0, "sign_hits": 0, "bias_hits": 0, "regime_hits": 0}
            if row and row[0]:
                payload = row[0] if isinstance(row[0], dict) else json.loads(row[0])

            payload["resolved"] = int(payload.get("resolved", 0)) + 1
            if outcome.get("sign_hit"):
                payload["sign_hits"] = int(payload.get("sign_hits", 0)) + 1
            if outcome.get("bias_hit"):
                payload["bias_hits"] = int(payload.get("bias_hits", 0)) + 1
            if outcome.get("regime_hit"):
                payload["regime_hits"] = int(payload.get("regime_hits", 0)) + 1

            resolved = max(payload["resolved"], 1)
            payload["sign_hit_rate"] = round(payload.get("sign_hits", 0) / resolved, 4)
            payload["bias_hit_rate"] = round(payload.get("bias_hits", 0) / resolved, 4)
            payload["regime_hit_rate"] = round(payload.get("regime_hits", 0) / resolved, 4)
            payload["updated_at"] = _now_iso()

            cur.execute(
                """
                INSERT INTO prediction_accuracy_daily (ticker, market_date, payload_json, updated_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (ticker, market_date) DO UPDATE SET
                    payload_json = EXCLUDED.payload_json,
                    updated_at = EXCLUDED.updated_at
                """,
                (ticker, market_date, json.dumps(payload), _now_iso()),
            )
        conn.commit()
