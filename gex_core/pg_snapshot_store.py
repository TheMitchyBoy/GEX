"""Persist full GEX snapshot payloads to PostgreSQL for external dashboards."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_BATCH_SIZE = 1000


@dataclass
class SnapshotWriteResult:
    ts: str
    written: bool
    skipped_duplicate: bool = False
    validation_status: str = "ok"
    strikes_written: int = 0
    postgres_write_ms: float | None = None


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


def _strike_rows(
    ticker: str,
    ts: str,
    gex_by_strike: pd.Series,
    cumulative_gex: pd.Series,
) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for strike, gex in gex_by_strike.items():
        if pd.isna(gex):
            continue
        cum = cumulative_gex.get(strike)
        rows.append(
            (
                ticker,
                ts,
                float(strike),
                float(gex),
                float(cum) if cum is not None and not pd.isna(cum) else None,
            )
        )
    return rows


def _copy_strike_rows(cur: Any, table: str, rows: list[tuple[Any, ...]]) -> None:
    if not rows:
        return
    columns = "(ticker, ts, strike, gex_bn_per_pct, cumulative_gex_bn_per_pct)"
    with cur.copy(f"COPY {table} {columns} FROM STDIN") as copy:
        for row in rows:
            copy.write_row(row)


def _write_diagnostics(
    cur: Any,
    *,
    ticker: str,
    ts: str,
    status: str,
    validation: dict[str, Any] | None,
    uw_fetch_ms: float | None,
    postgres_write_ms: float | None,
) -> None:
    from psycopg.types.json import Json

    cur.execute(
        """
        INSERT INTO snapshot_diagnostics (
            ticker, ts, status, validation_json, uw_fetch_ms, postgres_write_ms, indexed_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (ticker, ts) DO UPDATE SET
            status = EXCLUDED.status,
            validation_json = EXCLUDED.validation_json,
            uw_fetch_ms = EXCLUDED.uw_fetch_ms,
            postgres_write_ms = EXCLUDED.postgres_write_ms,
            indexed_at = EXCLUDED.indexed_at
        """,
        (
            ticker,
            ts,
            status,
            Json(validation) if validation else None,
            uw_fetch_ms,
            postgres_write_ms,
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def _write_features(cur: Any, features: dict[str, Any]) -> None:
    from psycopg.types.json import Json

    cur.execute(
        """
        INSERT INTO snapshot_features (
            ticker, ts, prior_ts, snapshot_at, gamma_flip, call_wall, put_wall,
            pos_gamma_peak_strike, flip_distance_pct, wall_spread, gex_concentration,
            near_term_ratio, zero_dte_ratio, term_curvature, expiration_count,
            front_term_ratio, back_term_ratio, delta_gex, delta_spot, spot_return,
            regime_changed, surface_vector, strike_profile_hash, strike_count
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (ticker, ts) DO UPDATE SET
            prior_ts = EXCLUDED.prior_ts,
            snapshot_at = EXCLUDED.snapshot_at,
            gamma_flip = EXCLUDED.gamma_flip,
            call_wall = EXCLUDED.call_wall,
            put_wall = EXCLUDED.put_wall,
            pos_gamma_peak_strike = EXCLUDED.pos_gamma_peak_strike,
            flip_distance_pct = EXCLUDED.flip_distance_pct,
            wall_spread = EXCLUDED.wall_spread,
            gex_concentration = EXCLUDED.gex_concentration,
            near_term_ratio = EXCLUDED.near_term_ratio,
            zero_dte_ratio = EXCLUDED.zero_dte_ratio,
            term_curvature = EXCLUDED.term_curvature,
            expiration_count = EXCLUDED.expiration_count,
            front_term_ratio = EXCLUDED.front_term_ratio,
            back_term_ratio = EXCLUDED.back_term_ratio,
            delta_gex = EXCLUDED.delta_gex,
            delta_spot = EXCLUDED.delta_spot,
            spot_return = EXCLUDED.spot_return,
            regime_changed = EXCLUDED.regime_changed,
            surface_vector = EXCLUDED.surface_vector,
            strike_profile_hash = EXCLUDED.strike_profile_hash,
            strike_count = EXCLUDED.strike_count
        """,
        (
            features["ticker"],
            features["ts"],
            features.get("prior_ts"),
            features.get("snapshot_at"),
            features.get("gamma_flip"),
            features.get("call_wall"),
            features.get("put_wall"),
            features.get("pos_gamma_peak_strike"),
            features.get("flip_distance_pct"),
            features.get("wall_spread"),
            features.get("gex_concentration"),
            features.get("near_term_ratio"),
            features.get("zero_dte_ratio"),
            features.get("term_curvature"),
            features.get("expiration_count"),
            features.get("front_term_ratio"),
            features.get("back_term_ratio"),
            features.get("delta_gex"),
            features.get("delta_spot"),
            features.get("spot_return"),
            features.get("regime_changed"),
            Json(features.get("surface_vector")),
            features.get("strike_profile_hash"),
            features.get("strike_count"),
        ),
    )


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
    prepared: Any | None = None,
    uw_fetch_ms: float | None = None,
    force: bool = False,
) -> SnapshotWriteResult:
    """Upsert snapshot metadata, features, strikes, and diagnostics into Postgres."""
    from gex_core.db import database_url, ensure_postgres_schema, refresh_latest_snapshot_view, use_postgres
    from gex_core.snapshot_processing import PreparedSnapshot, atm_strike_series, prepare_snapshot_for_storage

    if not use_postgres():
        return SnapshotWriteResult(ts=ts, written=False)

    ensure_postgres_schema()
    import psycopg
    from psycopg.types.json import Json

    ticker = ticker.upper()
    prep: PreparedSnapshot
    if prepared is not None:
        prep = prepared
    else:
        prep = prepare_snapshot_for_storage(
            ticker,
            gex_by_strike=gex_by_strike,
            cumulative_gex=cumulative_gex,
            gex_by_expiration=gex_by_expiration,
            summary=summary or {},
            timestamp=ts,
            force=force,
        )
    ts = prep.ts
    summary = prep.summary
    validation_payload = prep.validation.to_dict()

    if prep.skipped_duplicate:
        start = time.perf_counter()
        with psycopg.connect(database_url()) as conn:
            with conn.cursor() as cur:
                _write_diagnostics(
                    cur,
                    ticker=ticker,
                    ts=ts,
                    status="skipped_duplicate",
                    validation=validation_payload,
                    uw_fetch_ms=uw_fetch_ms,
                    postgres_write_ms=(time.perf_counter() - start) * 1000,
                )
                cur.execute("NOTIFY gex_snapshot, %s", (json.dumps({"ticker": ticker, "ts": ts, "skipped": True}),))
            conn.commit()
        logger.info("Skipped duplicate snapshot %s %s", ticker, ts)
        return SnapshotWriteResult(
            ts=ts,
            written=False,
            skipped_duplicate=True,
            validation_status="skipped_duplicate",
            postgres_write_ms=(time.perf_counter() - start) * 1000,
        )

    if not prep.validation.ok:
        start = time.perf_counter()
        with psycopg.connect(database_url()) as conn:
            with conn.cursor() as cur:
                _write_diagnostics(
                    cur,
                    ticker=ticker,
                    ts=ts,
                    status="rejected",
                    validation=validation_payload,
                    uw_fetch_ms=uw_fetch_ms,
                    postgres_write_ms=(time.perf_counter() - start) * 1000,
                )
            conn.commit()
        logger.warning("Rejected snapshot %s %s: %s", ticker, ts, prep.validation.issues)
        return SnapshotWriteResult(
            ts=ts,
            written=False,
            validation_status="rejected",
            postgres_write_ms=(time.perf_counter() - start) * 1000,
        )

    indexed_at = datetime.now(timezone.utc).isoformat()
    expiration_json = _series_to_mapping(gex_by_expiration) if gex_by_expiration is not None else None
    surface_json = _df_to_records(surface_data)
    greek_json = _df_to_records(greek_exposure_df)
    spot = float(summary.get("spot") or summary.get("spot_price") or 0.0)
    atm = atm_strike_series(gex_by_strike, spot, window_pct=prep.atm_window_pct)
    atm_cumulative = cumulative_gex.reindex(atm.index) if not atm.empty else pd.Series(dtype=float)
    all_rows = _strike_rows(ticker, ts, gex_by_strike, cumulative_gex)
    atm_rows = _strike_rows(ticker, ts, atm, atm_cumulative)

    start = time.perf_counter()
    with psycopg.connect(database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO snapshots (
                    ticker, ts, market_date, spot, total_gex, regime,
                    summary_path, strike_path, indexed_at, snapshot_at, prior_ts,
                    summary_json, expiration_json, surface_json, greek_exposure_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, ts) DO UPDATE SET
                    market_date = EXCLUDED.market_date,
                    spot = EXCLUDED.spot,
                    total_gex = EXCLUDED.total_gex,
                    regime = EXCLUDED.regime,
                    summary_path = EXCLUDED.summary_path,
                    strike_path = EXCLUDED.strike_path,
                    indexed_at = EXCLUDED.indexed_at,
                    snapshot_at = EXCLUDED.snapshot_at,
                    prior_ts = EXCLUDED.prior_ts,
                    summary_json = EXCLUDED.summary_json,
                    expiration_json = EXCLUDED.expiration_json,
                    surface_json = EXCLUDED.surface_json,
                    greek_exposure_json = EXCLUDED.greek_exposure_json
                """,
                (
                    ticker,
                    ts,
                    summary.get("market_date"),
                    spot if spot > 0 else None,
                    float(summary.get("total_gex_bn_per_pct"))
                    if summary.get("total_gex_bn_per_pct") is not None
                    else None,
                    str(summary.get("net_gamma_regime")) if summary.get("net_gamma_regime") else None,
                    summary_path,
                    strike_path,
                    indexed_at,
                    prep.snapshot_at,
                    prep.prior_ts,
                    Json(summary),
                    Json(expiration_json) if expiration_json else None,
                    Json(surface_json) if surface_json else None,
                    Json(greek_json) if greek_json else None,
                ),
            )
            _write_features(cur, prep.features)
            cur.execute("DELETE FROM snapshot_strikes WHERE ticker = %s AND ts = %s", (ticker, ts))
            cur.execute("DELETE FROM snapshot_strikes_atm WHERE ticker = %s AND ts = %s", (ticker, ts))
            _copy_strike_rows(cur, "snapshot_strikes", all_rows)
            _copy_strike_rows(cur, "snapshot_strikes_atm", atm_rows)
            write_ms = (time.perf_counter() - start) * 1000
            _write_diagnostics(
                cur,
                ticker=ticker,
                ts=ts,
                status=prep.validation.status,
                validation=validation_payload,
                uw_fetch_ms=uw_fetch_ms,
                postgres_write_ms=write_ms,
            )
            cur.execute(
                "NOTIFY gex_snapshot, %s",
                (json.dumps({"ticker": ticker, "ts": ts, "status": prep.validation.status}),),
            )
        conn.commit()
    try:
        refresh_latest_snapshot_view()
    except Exception:
        logger.debug("latest_snapshot refresh failed", exc_info=True)
    logger.info(
        "Wrote snapshot %s %s to PostgreSQL (%d strikes, %d atm)",
        ticker,
        ts,
        len(all_rows),
        len(atm_rows),
    )
    return SnapshotWriteResult(
        ts=ts,
        written=True,
        validation_status=prep.validation.status,
        strikes_written=len(all_rows),
        postgres_write_ms=(time.perf_counter() - start) * 1000,
    )
