"""Persist full GEX snapshot payloads to PostgreSQL for external dashboards."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_BATCH_SIZE = 1000


def export_csv_enabled() -> bool:
    import os

    from gex_core.db import use_postgres

    if not use_postgres():
        return True
    return os.environ.get("GEX_EXPORT_CSV", "0").strip().lower() in {"1", "true", "yes", "on"}


def _series_to_mapping(series: pd.Series) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in series.items():
        if pd.isna(value):
            continue
        out[str(key)] = float(value)
    return out


def _df_to_records(df: pd.DataFrame | None) -> list[dict[str, Any]] | None:
    if df is None or df.empty:
        return None
    records = json.loads(df.to_json(orient="records"))
    return records if records else None


def write_snapshot_to_postgres(
    ticker: str,
    ts: str,
    *,
    gex_by_strike: pd.Series,
    cumulative_gex: pd.Series,
    gex_by_expiration: pd.Series | None = None,
    surface_data: pd.DataFrame | None = None,
    greek_exposure_df: pd.DataFrame | None = None,
    summary: dict | None = None,
    summary_path: str | None = None,
    strike_path: str | None = None,
) -> None:
    """Upsert snapshot metadata, JSON payloads, and per-strike rows into Postgres."""
    from gex_core.db import database_url, ensure_postgres_schema, use_postgres

    if not use_postgres():
        return

    ensure_postgres_schema()
    import psycopg
    from psycopg.types.json import Json

    ticker = ticker.upper()
    summary = summary or {}
    indexed_at = datetime.now(timezone.utc).isoformat()
    expiration_json = _series_to_mapping(gex_by_expiration) if gex_by_expiration is not None else None
    surface_json = _df_to_records(surface_data)
    greek_json = _df_to_records(greek_exposure_df)

    with psycopg.connect(database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO snapshots (
                    ticker, ts, market_date, spot, total_gex, regime,
                    summary_path, strike_path, indexed_at,
                    summary_json, expiration_json, surface_json, greek_exposure_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, ts) DO UPDATE SET
                    market_date = EXCLUDED.market_date,
                    spot = EXCLUDED.spot,
                    total_gex = EXCLUDED.total_gex,
                    regime = EXCLUDED.regime,
                    summary_path = EXCLUDED.summary_path,
                    strike_path = EXCLUDED.strike_path,
                    indexed_at = EXCLUDED.indexed_at,
                    summary_json = EXCLUDED.summary_json,
                    expiration_json = EXCLUDED.expiration_json,
                    surface_json = EXCLUDED.surface_json,
                    greek_exposure_json = EXCLUDED.greek_exposure_json
                """,
                (
                    ticker,
                    ts,
                    summary.get("market_date"),
                    float(summary.get("spot") or summary.get("spot_price"))
                    if summary.get("spot") is not None or summary.get("spot_price") is not None
                    else None,
                    float(summary.get("total_gex_bn_per_pct"))
                    if summary.get("total_gex_bn_per_pct") is not None
                    else None,
                    str(summary.get("net_gamma_regime")) if summary.get("net_gamma_regime") else None,
                    summary_path,
                    strike_path,
                    indexed_at,
                    Json(summary),
                    Json(expiration_json) if expiration_json else None,
                    Json(surface_json) if surface_json else None,
                    Json(greek_json) if greek_json else None,
                ),
            )
            cur.execute(
                "DELETE FROM snapshot_strikes WHERE ticker = %s AND ts = %s",
                (ticker, ts),
            )
            strike_rows: list[tuple[Any, ...]] = []
            for strike, gex in gex_by_strike.items():
                if pd.isna(gex):
                    continue
                cum = cumulative_gex.get(strike)
                strike_rows.append(
                    (
                        ticker,
                        ts,
                        float(strike),
                        float(gex),
                        float(cum) if cum is not None and not pd.isna(cum) else None,
                    )
                )
            for start in range(0, len(strike_rows), _BATCH_SIZE):
                batch = strike_rows[start : start + _BATCH_SIZE]
                if not batch:
                    continue
                cur.executemany(
                    """
                    INSERT INTO snapshot_strikes (
                        ticker, ts, strike, gex_bn_per_pct, cumulative_gex_bn_per_pct
                    ) VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, ts, strike) DO UPDATE SET
                        gex_bn_per_pct = EXCLUDED.gex_bn_per_pct,
                        cumulative_gex_bn_per_pct = EXCLUDED.cumulative_gex_bn_per_pct
                    """,
                    batch,
                )
        conn.commit()
    logger.info("Wrote snapshot %s %s to PostgreSQL (%d strikes)", ticker, ts, len(gex_by_strike))
