"""Headless GEX processor: fetch Unusual Whales data and write to PostgreSQL."""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from apscheduler.schedulers.blocking import BlockingScheduler

from gex_core.data_root import configure_data_paths
from gex_core.env_bootstrap import bootstrap_env, parse_env_minutes
from gex_core.processor_metrics import get_processor_metrics, record_refresh_result
from gex_core.refresh import refresh_tickers
from gex_core.refresh_schedule import adaptive_refresh_minutes, processor_refresh_enabled, should_refresh_now
from gex_core.tickers import PRIMARY_TICKER

logger = logging.getLogger(__name__)

_REFRESH_LOCK = threading.Lock()
_LAST_REFRESH_AT: datetime | None = None


def processor_tickers() -> list[str]:
    raw = os.environ.get("GEX_DEFAULT_TICKERS") or os.environ.get("TICKERS") or PRIMARY_TICKER
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def _scheduled_refresh_work() -> None:
    global _LAST_REFRESH_AT
    if not processor_refresh_enabled():
        logger.debug("Skipping refresh — market session inactive")
        return
    if not should_refresh_now(_LAST_REFRESH_AT):
        return
    with _REFRESH_LOCK:
        if not should_refresh_now(_LAST_REFRESH_AT):
            return
        try:
            refresh_tickers(processor_tickers())
            _LAST_REFRESH_AT = datetime.now(timezone.utc)
        except Exception as exc:
            record_refresh_result(error=str(exc), validation_status="error")
            raise


def _health_payload() -> dict:
    from gex_core.db import use_postgres
    from gex_core.market_time import is_trader_session_active, is_trading_weekday
    from gex_core.refresh_schedule import adaptive_refresh_minutes, processor_refresh_enabled
    from gex_core.storage import count_snapshots, export_age_minutes, latest_timestamp

    ticker = processor_tickers()[0]
    metrics = get_processor_metrics()
    snapshot_count = count_snapshots(ticker) if use_postgres() else None
    payload: dict = {
        "mode": "processor",
        "postgres": use_postgres(),
        "ticker": ticker,
        "snapshot_count": snapshot_count,
        "backfill_min_snapshots": int(os.environ.get("GEX_BACKFILL_MIN_SNAPSHOTS", "30")),
        "market_open": is_trading_weekday() and is_trader_session_active(),
        "refresh_enabled": processor_refresh_enabled(),
        "adaptive_refresh_minutes": adaptive_refresh_minutes(),
        "metrics": metrics.to_dict(),
    }
    if not use_postgres():
        payload["status"] = "degraded"
        payload["message"] = "DATABASE_URL is not set"
        return payload

    latest = latest_timestamp(ticker)
    age = export_age_minutes(ticker)
    payload["latest_ts"] = latest
    payload["export_age_minutes"] = age
    if metrics.last_uw_fetch_ms is not None:
        payload["uw_fetch_ms"] = metrics.last_uw_fetch_ms
    if metrics.last_postgres_write_ms is not None:
        payload["postgres_write_ms"] = metrics.last_postgres_write_ms
    max_age = adaptive_refresh_minutes() * 2
    if latest is None:
        payload["status"] = "warming"
    elif age is not None and age > max_age:
        payload["status"] = "stale"
    else:
        payload["status"] = "ok"
    return payload


class _HealthHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path not in {"/", "/health", "/health/live", "/health/ready"}:
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps(_health_payload()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_health_server() -> ThreadingHTTPServer:
    port = int(os.environ.get("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Processor health server listening on :%d", port)
    return server


def _maybe_bootstrap_history() -> None:
    """Background bootstrap is started from scripts/start_processor.sh."""
    return


def run_processor(*, extra_startup: Callable[[], None] | None = None) -> None:
    """Block forever, refreshing GEX snapshots on an interval."""
    bootstrap_env()
    configure_data_paths()

    from gex_core.db import ensure_postgres_schema, use_postgres

    if not use_postgres():
        logger.warning("DATABASE_URL is not set — processor will write CSV exports only")

    ensure_postgres_schema()

    health_server = start_health_server()
    atexit.register(health_server.shutdown)

    if extra_startup:
        extra_startup()
    _maybe_bootstrap_history()

    refresh_seconds = max(60.0, min(120.0, adaptive_refresh_minutes() * 60.0))
    scheduler = BlockingScheduler()
    scheduler.add_job(
        _scheduled_refresh_work,
        trigger="interval",
        seconds=refresh_seconds,
        id="gex_processor_refresh",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=120,
    )
    scheduler.add_job(_scheduled_refresh_work, trigger="date", id="gex_processor_bootstrap")
    logger.info("GEX processor started — refresh every %.0fs for %s", refresh_seconds, processor_tickers())
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown(wait=False)


def main() -> int:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    run_processor()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
