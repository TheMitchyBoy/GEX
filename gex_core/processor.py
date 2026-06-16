"""Headless GEX processor: fetch Unusual Whales data and write to PostgreSQL."""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from apscheduler.schedulers.blocking import BlockingScheduler

from gex_core.data_root import configure_data_paths
from gex_core.env_bootstrap import bootstrap_env, parse_env_minutes
from gex_core.refresh import DEFAULT_REFRESH_MINUTES, refresh_tickers
from gex_core.tickers import PRIMARY_TICKER

logger = logging.getLogger(__name__)

_REFRESH_LOCK = threading.Lock()


def processor_tickers() -> list[str]:
    raw = os.environ.get("GEX_DEFAULT_TICKERS") or os.environ.get("TICKERS") or PRIMARY_TICKER
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def _scheduled_refresh_work() -> None:
    with _REFRESH_LOCK:
        refresh_tickers(processor_tickers())


def _health_payload() -> dict:
    from gex_core.db import use_postgres
    from gex_core.storage import export_age_minutes, latest_timestamp

    ticker = processor_tickers()[0]
    payload: dict = {
        "mode": "processor",
        "postgres": use_postgres(),
        "ticker": ticker,
    }
    if not use_postgres():
        payload["status"] = "degraded"
        payload["message"] = "DATABASE_URL is not set"
        return payload

    latest = latest_timestamp(ticker)
    age = export_age_minutes(ticker)
    payload["latest_ts"] = latest
    payload["export_age_minutes"] = age
    max_age = parse_env_minutes("GEX_REFRESH_INTERVAL_MINUTES", DEFAULT_REFRESH_MINUTES) * 2
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
    if os.environ.get("GEX_STARTUP_BACKFILL", "").strip() != "1":
        return
    from gex_core.storage import list_indexed_timestamps

    ticker = processor_tickers()[0]
    if len(list_indexed_timestamps(ticker)) >= int(os.environ.get("GEX_FORECAST_MIN_SNAPSHOTS", "4")):
        return
    logger.info("Startup backfill: index sparse for %s", ticker)
    import subprocess
    import sys

    subprocess.Popen(
        [
            sys.executable,
            "scripts/gex_backfill_intraday.py",
            "--tickers",
            ",".join(processor_tickers()),
            "--intraday-days",
            os.environ.get("GEX_INTRADAY_BACKFILL_DAYS", "90"),
            "--daily-days",
            os.environ.get("GEX_DAILY_BACKFILL_DAYS", "90"),
            "--interval-minutes",
            os.environ.get("GEX_BACKFILL_INTERVAL_MINUTES", "10"),
        ],
        cwd=os.path.dirname(os.path.dirname(__file__)),
    )


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

    refresh_seconds = max(60.0, parse_env_minutes("GEX_REFRESH_INTERVAL_MINUTES", DEFAULT_REFRESH_MINUTES) * 60.0)
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
