"""Write a matched GEX export set (strike, cumulative, summary, ...)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from gex_core.export_metadata import build_export_metadata
from gex_core.history import clear_history_cache

logger = logging.getLogger(__name__)


def write_snapshot_export(
    ticker: str,
    *,
    gex_by_strike: pd.Series,
    cumulative_gex: pd.Series,
    gex_by_expiration: pd.Series | None = None,
    surface_data: pd.DataFrame | None = None,
    greek_exposure_df: pd.DataFrame | None = None,
    summary: dict | None = None,
    export_dir: str | Path = "data/exports",
    timestamp: str | None = None,
) -> str:
    """Persist snapshot data to PostgreSQL (primary) and optional CSV/JSON exports."""
    from gex_core.pg_snapshot_store import export_csv_enabled, write_snapshot_to_postgres

    export_dir = Path(export_dir)
    timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    gex_by_expiration = gex_by_expiration if gex_by_expiration is not None else pd.Series(dtype=float)
    surface_data = surface_data if surface_data is not None else pd.DataFrame()

    if summary is not None and "data_source" not in summary:
        summary["data_source"] = "unusual_whales"
    if summary is not None:
        meta = build_export_metadata(
            ticker,
            market_date=summary.get("market_date"),
            spot=float(summary.get("spot") or summary.get("spot_price") or 0.0),
            total_gex_bn=float(summary.get("total_gex_bn_per_pct", 0.0)),
            regime=str(summary.get("net_gamma_regime", "N/A")),
            data_quality=summary.get("data_quality"),
            uw_endpoint=str(summary.get("uw_endpoint", "spot-exposures/strike")),
        )
        summary = {**meta, **summary}

    strike_path: str | None = None
    summary_path: str | None = None

    if export_csv_enabled():
        export_dir.mkdir(parents=True, exist_ok=True)
        strike_path = str(export_dir / f"{ticker.upper()}_gex_by_strike_{timestamp}.csv")
        gex_by_strike.rename("gex_bn_per_pct").to_csv(strike_path)
        cumulative_gex.rename("cumulative_gex_bn_per_pct").to_csv(
            export_dir / f"{ticker.upper()}_cumulative_gex_{timestamp}.csv"
        )
        gex_by_expiration.rename("gex_bn_per_pct").to_csv(
            export_dir / f"{ticker.upper()}_gex_by_expiration_{timestamp}.csv"
        )
        if surface_data is not None and not surface_data.empty:
            surface_data.to_csv(export_dir / f"{ticker.upper()}_gex_surface_{timestamp}.csv", index=False)
        if greek_exposure_df is not None and not greek_exposure_df.empty:
            greek_exposure_df.to_csv(
                export_dir / f"{ticker.upper()}_greek_exposure_{timestamp}.csv",
                index=False,
            )
        summary_path = str(export_dir / f"{ticker.upper()}_summary_{timestamp}.json")
        with Path(summary_path).open("w", encoding="utf-8") as f:
            json.dump(summary or {}, f, indent=2)

    try:
        write_snapshot_to_postgres(
            ticker,
            timestamp,
            gex_by_strike=gex_by_strike,
            cumulative_gex=cumulative_gex,
            gex_by_expiration=gex_by_expiration,
            surface_data=surface_data,
            greek_exposure_df=greek_exposure_df,
            summary=summary,
            summary_path=summary_path,
            strike_path=strike_path,
        )
        from gex_core.db import use_postgres
        from gex_core.runtime_mode import is_processor_mode
        from gex_core.storage import upsert_snapshot

        if not use_postgres():
            upsert_snapshot(
                ticker.upper(),
                timestamp,
                market_date=summary.get("market_date") if summary else None,
                spot=float(summary.get("spot")) if summary and summary.get("spot") is not None else None,
                total_gex=float(summary.get("total_gex_bn_per_pct")) if summary else None,
                regime=str(summary.get("net_gamma_regime")) if summary else None,
                summary_path=summary_path,
                strike_path=strike_path,
            )
        if not is_processor_mode():
            from gex_core.history import clear_history_cache

            clear_history_cache()
            try:
                from gex_core.prediction_log import reconcile_llm_predictions

                reconcile_llm_predictions(ticker.upper(), latest_ts=timestamp)
            except Exception:
                pass
    except Exception:
        logger.exception("Failed to persist snapshot %s %s", ticker, timestamp)
        raise

    return timestamp
