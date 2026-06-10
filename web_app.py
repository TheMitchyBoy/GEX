"""
Flask SPX gamma dashboard.

Serves historical GEX snapshots from ``data/exports/``, renders Plotly charts,
runs weighted-KNN forecasts, and optionally auto-refreshes via APScheduler when
``UW_API_KEY`` is set. Snapshots live under ``data/exports/`` with an optional
SQLite index for fast history lookup.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, Response, abort, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.exceptions import HTTPException

from gex_core.backtest_metrics import backtest_delta_sign_accuracy
from gex_core.features import parse_gamma_flip_value, safe_float
from gex_core.market_exposure_agent import analyze_market_exposure, predict_market_exposure
from gex_core.gex_chatbot import build_welcome_message, chat_reply, reset_session
from gex_core.periscope import (
    build_periscope_context,
    build_slice_options,
    build_timeline_navigation,
    resolve_selected_timestamp,
)
from gex_core.periscope_api import (
    clear_periscope_api_cache,
    list_periscope_timestamps,
    should_use_api_for_date,
)
from gex_core.exports import EXPORT_DIR
from gex_core.history import (
    build_history,
    get_latest_ts,
    list_timestamps,
    load_snapshot_at_ts,
)
from gex_core.market_features import fetch_spx_price_series_for_dashboard
from gex_core.startup import deferred_web_startup
from gex_core.uw_price_stream import start_uw_price_stream
from gex_core.intelligence import (
    build_gamma_analysis_panel,
    build_data_quality_panel,
    build_model_accountability_panel,
    build_outcome_panel,
    build_strategy_assistant,
    build_term_structure_panel,
    build_today_regime_snapshot,
    build_watchlist_rows,
    compute_confluence_overlay,
    compute_forecast_probabilities,
    generate_alerts,
)
from gex_core.predict import (
    apply_flow_to_prediction,
    forecast_blocker_message,
    load_flow_predictions,
    predict_next_snapshot,
)
from gex_core.alert_dispatch import maybe_dispatch_alerts
from gex_core.env_bootstrap import bootstrap_env, parse_env_minutes, uw_api_configured, uw_api_key
from gex_core.refresh import DEFAULT_REFRESH_MINUTES, refresh_ticker, refresh_tickers
from gex_core.export_diagnostics import prediction_lookback_days
from gex_core.system_status import build_system_status
from gex_core.tickers import PRIMARY_TICKER, find_available_tickers, is_supported_ticker, supported_tickers

APP = Flask(__name__)
app = APP
logger = logging.getLogger(__name__)

bootstrap_env()


@APP.errorhandler(HTTPException)
def _api_http_error(exc: HTTPException):
    """Return JSON (not HTML) for /api/* HTTP errors."""
    if request.path.startswith("/api/"):
        return jsonify({"error": exc.description or exc.name}), exc.code
    return exc


@APP.errorhandler(Exception)
def _api_uncaught_error(exc: Exception):
    """Return JSON (not HTML) for unhandled /api/* exceptions."""
    if request.path.startswith("/api/"):
        logger.exception("Unhandled API error on %s", request.path)
        return jsonify({"error": str(exc) or "Internal server error"}), 500
    raise exc

# ─────────────────────────────────────────────────────────────────────────────
# Unusual Whales live data layer
# ─────────────────────────────────────────────────────────────────────────────

_UW_CACHE: dict[str, dict] = {}          # ticker → {spot, agg, ts, analysis}
_UW_CACHE_TTL = int(
    os.environ.get(
        "GEX_UW_CACHE_TTL_SECONDS",
        str(max(30, int(parse_env_minutes("GEX_REFRESH_INTERVAL_MINUTES", 10.0) * 60))),
    )
)  # keep cache aligned to refresh cadence by default

if not uw_api_configured():
    logger.warning(
        "UW_API_KEY is not set in this environment — live data is DISABLED. The "
        "dashboard will only show previously saved snapshots and forced refreshes "
        "will report 'Live data isn't configured'. Set UW_API_KEY in the service "
        "environment (e.g. host env / .env for docker compose, or your platform's "
        "config) to enable live fetches."
    )

# Last classified UW failure reason per ticker, so the dashboard and logs can
# report *why* a refresh failed instead of guessing.
_LAST_UW_ERROR: dict[str, str] = {}

_REFRESH_REASON_MESSAGES = {
    "not_configured": "Live data isn't configured on this server (UW_API_KEY is missing).",
    "auth": "The data provider rejected the request — the API key is invalid or lacks permission.",
    "rate_limited": "The data provider is rate-limiting requests right now.",
    "network": "Couldn't reach the data provider (timeout or connection error).",
    "error": "Couldn't fetch fresh data right now.",
}


def _classify_uw_error(exc: Exception) -> str:
    """Map a UW fetch exception to a coarse, user-meaningful reason code.

    Order matters: ``requests.HTTPError`` is itself a subclass of ``OSError``
    (== ``EnvironmentError``), so HTTP status and network categories are checked
    before the missing-key (plain ``EnvironmentError``) case.
    """
    import requests

    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403):
        return "auth"
    if status == 429:
        return "rate_limited"
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return "network"
    if isinstance(exc, EnvironmentError) and not isinstance(exc, requests.RequestException):
        return "not_configured"
    return "error"


def _uw_failure_reason(ticker: str) -> str:
    if not uw_api_configured():
        return "not_configured"
    with _uw_lock:
        return _LAST_UW_ERROR.get(ticker.upper(), "error")

_uw_lock = threading.Lock()


def _uw_cache_fresh(ticker: str) -> bool:
    entry = _UW_CACHE.get(ticker.upper())
    return bool(entry and (time.monotonic() - entry["ts"]) < _UW_CACHE_TTL)


def refresh_uw_data(ticker: str, force: bool = False) -> dict | None:
    """Fetch live UW GEX and run AI analysis; cache the result."""
    if not uw_api_configured():
        return None
    ticker = ticker.upper()
    if not force and _uw_cache_fresh(ticker):
        return _UW_CACHE[ticker]
    try:
        from gex_core.uw_loader import fetch_spot_gamma_aggregate_bn, fetch_uw_gex
        from gex_core.ai_analyst import analyze_dealer_gamma

        spot, agg = fetch_uw_gex(ticker, api_key=uw_api_key())
        gamma_flip = None
        spot_gamma_bn = fetch_spot_gamma_aggregate_bn(ticker, api_key=uw_api_key())
        analysis = analyze_dealer_gamma(
            ticker=ticker, spot=spot,
            gex_by_strike=agg.gex_by_strike,
            cumulative_gex=agg.cumulative_gex,
            total_gex_bn=agg.total_gex_bn,
            gamma_flip=gamma_flip,
        )
        entry = {
            "spot": spot, "agg": agg,
            "gamma_flip": gamma_flip,
            "spot_gamma_bn": spot_gamma_bn,
            "analysis": analysis,
            "ts": time.monotonic(),
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        }
        with _uw_lock:
            _UW_CACHE[ticker] = entry
            _LAST_UW_ERROR.pop(ticker, None)
        logger.info("UW data refreshed for %s: spot=%.2f, GEX=%.3f Bn$", ticker, spot, agg.total_gex_bn)
        return entry
    except Exception as exc:
        reason = _classify_uw_error(exc)
        with _uw_lock:
            _LAST_UW_ERROR[ticker] = reason
        logger.exception("UW data refresh failed for %s (reason=%s)", ticker, reason)
        return None


def get_uw_data(ticker: str) -> dict | None:
    """Return cached UW data, refreshing if stale."""
    if not uw_api_configured():
        return None
    ticker = ticker.upper()
    with _uw_lock:
        if _uw_cache_fresh(ticker):
            return _UW_CACHE[ticker]
    return refresh_uw_data(ticker)


IMG_DIR = Path(__file__).resolve().parent / "img"
FLOW_FEED_PATH = Path(os.environ.get("GEX_FLOW_FEED", "data/flow_sample.jsonl"))
REFRESH_TICKERS = supported_tickers()
REFRESH_MINUTES = DEFAULT_REFRESH_MINUTES


def _uw_live_enabled() -> bool:
    return os.environ.get("GEX_SHOW_UW_LIVE", "1").lower() in {"1", "true", "yes"}


def _select_snapshot(history: list, requested_ts: str | None, ticker: str | None = None) -> dict:
    if not history:
        raise ValueError("empty history")
    if requested_ts:
        for row in history:
            if row["ts"] == requested_ts:
                return row
        if ticker:
            loaded = load_snapshot_at_ts(ticker, requested_ts)
            if loaded:
                return loaded
    return history[-1]


def _dashboard_history(ticker: str) -> list[dict]:
    return build_history(
        ticker,
        lookback_days=int(os.environ.get("GEX_DASHBOARD_HISTORY_DAYS", "90")),
        max_snapshots=int(os.environ.get("GEX_DASHBOARD_HISTORY_MAX", "240")),
        dedupe_identical_strikes=True,
    )


def _prediction_history(ticker: str) -> list[dict]:
    """History for KNN — keep consecutive duplicate strike profiles for sample depth."""
    dedupe = os.environ.get("GEX_PREDICTION_DEDUP", "0").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    return build_history(
        ticker,
        lookback_days=prediction_lookback_days(ticker),
        max_snapshots=int(os.environ.get("GEX_PREDICTION_HISTORY_MAX", "240")),
        dedupe_identical_strikes=dedupe,
    )


def _dashboard_spx_price_context(ticker: str) -> tuple[list[dict], float, str]:
    points, current, source = fetch_spx_price_series_for_dashboard(ticker)
    if current <= 0:
        latest = get_latest_ts(ticker)
        if latest:
            loaded = load_snapshot_at_ts(ticker, latest)
            if loaded and loaded.get("spot"):
                current = float(loaded["spot"])
                if not points:
                    points = [{"ts": loaded.get("ts_label", latest), "close": current}]
                    source = "snapshots"
    return points, current, source


def _prediction_public_view(prediction: dict | None) -> dict | None:
    if not prediction:
        return None
    return {
        k: v
        for k, v in prediction.items()
        if k not in {"predicted_strike", "knn_strike", "flow_strike"}
    }


def _strategy_exposure_from_context(ctx: dict) -> tuple[pd.Series | None, pd.Series | None]:
    """Spot-exposures/strike net gamma profile for charts."""
    exposure = ctx.get("exposure_series")
    previous = ctx.get("previous_exposure")
    return exposure, previous


def _spx_redirect(**extra):
    args = request.args.to_dict(flat=True)
    args.update(extra)
    return redirect(url_for("index", **args))


def _admin_token_from_request(req) -> str:
    provided = req.headers.get("X-Admin-Token") or req.args.get("admin_token") or ""
    form = getattr(req, "form", None)
    if not provided and form:
        provided = form.get("admin_token") or ""
    if not provided and getattr(req, "is_json", False):
        get_json = getattr(req, "get_json", None)
        body = (get_json(silent=True) if callable(get_json) else None) or {}
        provided = body.get("admin_token") or ""
    return provided or ""


def _admin_action_authorized(req) -> bool:
    """Authorize state-changing admin actions via ``GEX_ADMIN_TOKEN``.

  When the token is unset, HTTP-triggered refreshes and dispatches are disabled.
    """
    token = os.environ.get("GEX_ADMIN_TOKEN")
    if not token:
        return False
    provided = _admin_token_from_request(req)
    return bool(provided) and secrets.compare_digest(provided, token)


def _run_ticker_refresh(ticker: str) -> tuple[bool, str | None]:
    """Refresh CSV exports and live UW cache for a ticker."""
    ticker = ticker.upper()
    clear_periscope_api_cache()
    refreshed_csv = False
    refreshed_live = False
    if uw_api_configured():
        try:
            refreshed_csv = refresh_ticker(ticker, force=True)
        except Exception:
            logger.exception("Force CSV refresh failed for %s", ticker)
        try:
            refreshed_live = refresh_uw_data(ticker, force=True) is not None
        except Exception:
            logger.exception("Force UW refresh failed for %s", ticker)
    ok = refreshed_csv or refreshed_live
    reason = None if ok else _uw_failure_reason(ticker)
    return ok, reason


def _render_periscope_dashboard(ticker: str = PRIMARY_TICKER):
    """Periscope-style market maker exposure view (replaces the legacy command center)."""
    ticker = ticker.upper()
    exposure = request.args.get("exposure", "gamma").lower()
    requested_ts = request.args.get("ts")
    requested_date = request.args.get("date")
    bootstrap_status = request.args.get("bootstrap")
    refresh_message = None

    has_exports = bool(list_periscope_timestamps(ticker, api_key=uw_api_key()))

    uw_entry = get_uw_data(ticker) if _uw_live_enabled() else None
    ctx = build_periscope_context(
        ticker=ticker,
        selected_ts=requested_ts,
        selected_date=requested_date,
        exposure=exposure,
        uw_entry=uw_entry,
        api_key=uw_api_key(),
    )
    timeline = ctx.get("timeline") or {}

    if bootstrap_status == "failed" and has_exports:
        bootstrap_status = "stale"
    elif (
        bootstrap_status is None
        and _uw_live_enabled()
        and uw_api_configured()
        and uw_entry is None
        and has_exports
        and timeline.get("is_latest", False)
    ):
        reason = _uw_failure_reason(ticker)
        if reason != "not_configured":
            bootstrap_status = "stale"
            refresh_message = _REFRESH_REASON_MESSAGES.get(reason, _REFRESH_REASON_MESSAGES["error"])

    selected = ctx.get("selected") or {}
    gamma_flip = parse_gamma_flip_value(ctx.get("gamma_flip"))
    if gamma_flip is not None:
        selected = {**selected, "gamma_flip": gamma_flip}
    gex_series, prev_series = _strategy_exposure_from_context(ctx)
    spot = ctx.get("spot")

    prev_spot = None
    history = ctx.get("history") or []
    sel_ts = ctx.get("selected_ts")
    if history and sel_ts:
        for i, row in enumerate(history):
            if row.get("ts") == sel_ts and i > 0:
                prev_spot = safe_float(history[i - 1].get("spot"), 0.0) or None
                break

    from gex_core.trading.strategy_viz import build_strategy_dashboard

    uw_bundle = _uw_bundle_for_context(ticker=ticker, ctx=ctx, uw_entry=uw_entry)
    strategy = build_strategy_dashboard(
        ticker=ticker,
        spot=spot,
        exposure=gex_series,
        previous_exposure=prev_series,
        snapshot=selected,
        prev_spot=prev_spot,
        uw_bundle=uw_bundle,
    )
    strategy_chart_json = strategy["chart_json"]
    strategy_state = strategy["state"]

    chat_welcome = build_welcome_message(
        ticker=ticker,
        spot=safe_float(spot, 0.0) or None,
        regime=ctx.get("regime", "N/A"),
        total_gex=safe_float(ctx.get("total_gex"), 0.0),
        gamma_flip=gamma_flip,
        exposure=exposure,
    )

    csv_source = selected.get("data_source") or ctx.get("data_path") or "unusual_whales"
    data_source = f"Unusual Whales · {ctx.get('selected_label', 'latest')} ({csv_source})"
    if uw_entry:
        data_source += " · live API"

    from gex_core.trading.engine import trader_status

    auto_trader = trader_status(ticker)

    return render_template(
        "periscope.html",
        ticker=ticker,
        exposure=exposure,
        spot=spot,
        regime=ctx.get("regime", "N/A"),
        total_gex=ctx.get("total_gex", 0.0),
        gamma_flip=gamma_flip,
        selected_ts=ctx.get("selected_ts"),
        selected_date=ctx.get("selected_date"),
        selected_label=ctx.get("selected_label"),
        timestamps=ctx.get("timestamps", []),
        timeline=timeline,
        replay_index=ctx.get("replay_index", 0),
        data_source=data_source,
        strategy_chart_json=strategy_chart_json,
        strategy_state=strategy_state,
        is_live_slice=timeline.get("is_latest", False),
        chat_welcome=chat_welcome,
        bootstrap_status=bootstrap_status,
        refresh_message=refresh_message,
        uw_configured=uw_api_configured(),
        refresh_minutes=REFRESH_MINUTES,
        vanna_charm_available=ctx.get("vanna_charm_available", False),
        auto_trader=auto_trader,
    )


def _ticker_api_payload(ticker: str, selected_ts: str | None = None) -> dict:
    ticker = PRIMARY_TICKER
    history = _dashboard_history(ticker)
    if not history:
        selected = {
            "regime": "N/A",
            "total_gex": 0.0,
            "pos_gex": 0.0,
            "neg_gex": 0.0,
            "call_wall": None,
            "put_wall": None,
            "gamma_flip": None,
            "spot": None,
            "near_term_ratio": 0.0,
        }
        return {
            "ticker": ticker,
            "has_history": False,
            "summary": None,
            "gamma_analysis": build_gamma_analysis_panel(selected),
            "term_structure": build_term_structure_panel(selected),
            "alerts": [],
            "strategy_notes": [],
            "model_accountability": build_model_accountability_panel(ticker, None, {}),
            "watchlist": [],
        }
    selected = _select_snapshot(history, selected_ts, ticker=ticker)
    prediction_history = _prediction_history(ticker)
    prediction_lookback = prediction_lookback_days(ticker)
    prediction = predict_next_snapshot(prediction_history, lookback_days=prediction_lookback)
    flow_overlay = None
    spot_for_flow = safe_float(selected.get("spot"), 0.0) or 4800.0
    try:
        flow_overlay = load_flow_predictions(FLOW_FEED_PATH, spot=float(spot_for_flow))
        prediction = apply_flow_to_prediction(prediction, flow_overlay)
    except Exception:
        logger.exception("Flow overlay failed for %s", ticker)
    try:
        backtest = backtest_delta_sign_accuracy(ticker)
    except Exception:
        logger.exception("Backtest metrics failed for %s", ticker)
        backtest = {}
    confluence = compute_confluence_overlay(selected, prediction, flow_overlay)
    today = build_today_regime_snapshot(selected, prediction)
    alerts = generate_alerts(history, selected, prediction)
    probs = compute_forecast_probabilities(selected, prediction, history)
    strategy_notes = build_strategy_assistant(selected, prediction, confluence)
    return {
        "ticker": ticker,
        "has_history": True,
        "selected_ts": selected.get("ts"),
        "summary": today,
        "gamma_analysis": build_gamma_analysis_panel(selected, prediction),
        "term_structure": build_term_structure_panel(selected, prediction),
        "prediction": _prediction_public_view(prediction),
        "probabilities": probs,
        "model_accountability": build_model_accountability_panel(ticker, prediction, backtest),
        "confluence": confluence,
        "alerts": alerts,
        "strategy_notes": strategy_notes,
        "data_quality": build_data_quality_panel(selected, history),
        "outcomes": build_outcome_panel(history, selected.get("ts")),
        "watchlist": build_watchlist_rows([ticker]),
    }


@APP.get("/health")
def health():
    status = build_system_status(PRIMARY_TICKER)
    # Liveness: always 200 when the process responds. Use ``ready`` / ``healthy``
    # in the JSON for readiness probes (fresh deploys have no exports yet).
    return jsonify(status), 200


@APP.get("/health/ready")
def health_ready():
    """Strict readiness probe — 503 until export history exists."""
    status = build_system_status(PRIMARY_TICKER)
    code = 200 if status.get("ready") else 503
    return jsonify(status), code


def _wall_gex_api_urls() -> dict[str, str]:
    return {
        "status": url_for("api_wall_gex_status"),
        "arm": url_for("api_wall_gex_arm"),
        "run": url_for("api_wall_gex_run"),
    }


def _wall_gex_live_data(ticker: str) -> tuple[float | None, pd.Series | None, dict[str, Any]]:
    """Load spot, exposure series, and a dry-run wall GEX signal."""
    from gex_core.trading.low_gex_engine import run_low_gex_trade

    ticker = ticker.upper()
    uw_entry = get_uw_data(ticker) if _uw_live_enabled() else None
    ctx = build_periscope_context(
        ticker=ticker,
        exposure="gamma",
        uw_entry=uw_entry,
        api_key=uw_api_key(),
    )
    spot = _resolve_trader_spot(ticker, safe_float(ctx.get("spot"), 0.0) or None)
    exposure, _ = _strategy_exposure_from_context(ctx)
    if not spot or exposure is None:
        return spot, exposure, {"ran": False, "reason": "No spot or gamma exposure available"}
    preview = run_low_gex_trade(
        ticker=ticker,
        spot=float(spot),
        exposure=exposure,
        execute=False,
    )
    return float(spot), exposure, preview


def _render_wall_gex_dashboard(ticker: str = PRIMARY_TICKER):
    from gex_core.trading.low_gex_engine import wall_gex_status

    ticker = ticker.upper()
    status = wall_gex_status(ticker)
    spot: float | None = None
    last_cycle: dict[str, Any] = {}
    try:
        spot, _, last_cycle = _wall_gex_live_data(ticker)
    except Exception as exc:
        logger.exception("Wall GEX signal preview failed for %s", ticker)
        last_cycle = {"ran": False, "reason": str(exc)}
    return render_template(
        "wall_gex.html",
        ticker=ticker,
        spot=spot,
        status=status,
        last_cycle=last_cycle,
        api_urls=_wall_gex_api_urls(),
    )


@APP.route("/")
def index():
    return _render_wall_gex_dashboard(PRIMARY_TICKER)


@APP.route("/gamma")
@APP.route("/periscope")
def gamma_dashboard():
    return _render_periscope_dashboard(PRIMARY_TICKER)


@APP.route("/ticker/<ticker>")
@APP.route("/ticker/<ticker>/")
def ticker_page(ticker):
    ticker = ticker.upper()
    if not is_supported_ticker(ticker):
        return _spx_redirect()
    return _render_wall_gex_dashboard(ticker)


@APP.route("/ticker/<ticker>/gamma")
def ticker_gamma_page(ticker):
    ticker = ticker.upper()
    if not is_supported_ticker(ticker):
        return _spx_redirect()
    return _render_periscope_dashboard(ticker)


@APP.post("/ticker/<ticker>/refresh")
def refresh_ticker_data(ticker):
    ticker = ticker.upper()
    if not is_supported_ticker(ticker):
        return _spx_redirect()
    if not _admin_action_authorized(request):
        abort(403)
    ok, reason = _run_ticker_refresh(ticker)
    has_exports = bool(list_periscope_timestamps(ticker, api_key=uw_api_key()))
    if ok:
        status = "ok"
    elif has_exports:
        status = "stale"
    else:
        status = "failed"
    extra = {}
    if status == "failed" and reason:
        extra["reason"] = reason
    return redirect(url_for("ticker_page", ticker=ticker, bootstrap=status, **extra))


@APP.post("/ticker/<ticker>/bootstrap")
def bootstrap_ticker_history(ticker):
    ticker = ticker.upper()
    if not is_supported_ticker(ticker):
        return _spx_redirect()
    if not _admin_action_authorized(request):
        abort(403)
    try:
        ok = refresh_ticker(ticker, force=True)
    except Exception:
        logger.exception("Manual GEX refresh failed for %s", ticker)
        return redirect(url_for("ticker_page", ticker=ticker, bootstrap="error"))

    status = "ok" if ok else "failed"
    return redirect(url_for("ticker_page", ticker=ticker, bootstrap=status))


@APP.post("/ticker/<ticker>/dispatch-alerts")
def dispatch_ticker_alerts(ticker):
    ticker = ticker.upper()
    if not is_supported_ticker(ticker):
        return jsonify({"ok": False, "message": "Unsupported ticker"}), 404
    if not _admin_action_authorized(request):
        abort(403)
    history = _dashboard_history(ticker)
    if not history:
        return jsonify({"ok": False, "message": "No history", "dispatched": False}), 404
    selected = _select_snapshot(history, None)
    prediction_history = _prediction_history(ticker)
    prediction = predict_next_snapshot(
        prediction_history,
        lookback_days=prediction_lookback_days(ticker),
    )
    alerts = generate_alerts(history, selected, prediction)
    status = maybe_dispatch_alerts(ticker, alerts, manual=True)
    code = 200 if status.get("ok") else 502
    return jsonify(status), code


@APP.get("/api/spx-price")
def api_spx_price():
    points, current, source = _dashboard_spx_price_context(PRIMARY_TICKER)
    return jsonify(
        {
            "ticker": PRIMARY_TICKER,
            "current": current if current > 0 else None,
            "source": source,
            "points": len(points),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )


@APP.get("/api/spot-stream")
def api_spot_stream():
    """SSE stream of live spot ticks from the UW price websocket cache only.

    Does not poll UW REST on this loop — that burned API quota at ~2 req/s when
    the websocket was disconnected. REST spot fallbacks belong on page load only.
    """
    from gex_core.uw_price_stream import get_uw_price_stream

    ticker = (request.args.get("ticker") or PRIMARY_TICKER).upper()
    poll_seconds = max(0.25, float(os.environ.get("GEX_SPOT_STREAM_POLL_SECONDS", "1")))

    def generate():
        stream = get_uw_price_stream()
        last_price: float | None = None
        while True:
            price = stream.get_latest_price(ticker)
            if price > 0 and price != last_price:
                last_price = price
                payload = json.dumps(
                    {
                        "ticker": ticker,
                        "spot": price,
                        "source": "uw-live",
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                yield f"data: {payload}\n\n"
            time.sleep(poll_seconds)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@APP.get("/api/periscope")
def api_periscope():
    ticker = PRIMARY_TICKER
    exposure = request.args.get("exposure", "gamma")
    requested_ts = request.args.get("ts")
    requested_date = request.args.get("date")
    uw_entry = get_uw_data(ticker) if _uw_live_enabled() else None
    ctx = build_periscope_context(
        ticker=ticker,
        selected_ts=requested_ts,
        selected_date=requested_date,
        exposure=exposure,
        uw_entry=uw_entry,
        api_key=uw_api_key(),
    )
    timeline = ctx.get("timeline") or {}
    return jsonify(
        {
            "ticker": ticker,
            "exposure": exposure,
            "spot": ctx.get("spot"),
            "regime": ctx.get("regime"),
            "total_gex": ctx.get("total_gex"),
            "gamma_flip": ctx.get("gamma_flip"),
            "selected_ts": ctx.get("selected_ts"),
            "selected_date": ctx.get("selected_date"),
            "selected_label": ctx.get("selected_label"),
            "mm_positions": ctx.get("mm_positions"),
            "vanna_charm_available": ctx.get("vanna_charm_available"),
            "timeline": timeline,
            "data_path": ctx.get("data_path"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )


@APP.get("/api/trader/status")
def api_trader_status():
    from gex_core.trading.engine import trader_status

    ticker = (request.args.get("ticker") or PRIMARY_TICKER).upper()
    return jsonify(trader_status(ticker))


@APP.post("/api/trader/arm")
def api_trader_arm():
    from gex_core.trading.config import live_trading_allowed, require_live_confirm
    from gex_core.trading.engine import arm_trader, trader_status

    payload = request.get_json(silent=True) or {}
    armed = bool(payload.get("armed", True))
    if armed and live_trading_allowed() and require_live_confirm() and not payload.get("live_confirm"):
        return jsonify(
            {
                "error": "Live Webull trading requires live_confirm: true in the request body.",
                "live_mode": True,
            }
        ), 400
    arm_trader(armed)
    ticker = (payload.get("ticker") or PRIMARY_TICKER).upper()
    return jsonify({"armed": armed, "status": trader_status(ticker)})


@APP.post("/api/trader/run")
def api_trader_run():
    """Manual paper-trading evaluation cycle."""
    from gex_core.trading.engine import run_trading_cycle

    ticker = PRIMARY_TICKER
    uw_entry = get_uw_data(ticker) if _uw_live_enabled() else None
    ctx = build_periscope_context(
        ticker=ticker,
        exposure="gamma",
        uw_entry=uw_entry,
        api_key=uw_api_key(),
    )
    spot = ctx.get("spot")
    if not spot:
        return jsonify({"error": "No spot price available"}), 503
    uw_bundle = _uw_bundle_for_context(ticker=ticker, ctx=ctx, uw_entry=uw_entry)
    exposure, previous_exposure = _strategy_exposure_from_context(ctx)
    result = run_trading_cycle(
        ticker=ticker,
        spot=float(spot),
        exposure=exposure,
        previous_exposure=previous_exposure,
        uw_bundle=uw_bundle,
        snapshot=ctx.get("selected"),
        previous_spot=_previous_spot_from_context(ctx),
        force=True,
    )
    return jsonify(result)


@APP.get("/api/trader/strategy")
def api_trader_strategy():
    """Live strategy signals, filter state, and chart spec for dashboard refresh."""
    from gex_core.trading.strategy_viz import build_strategy_dashboard

    ticker = (request.args.get("ticker") or PRIMARY_TICKER).upper()
    uw_entry = get_uw_data(ticker) if _uw_live_enabled() else None
    ctx = build_periscope_context(
        ticker=ticker,
        selected_ts=request.args.get("ts"),
        exposure="gamma",
        uw_entry=uw_entry,
        api_key=uw_api_key(),
    )
    history = ctx.get("history") or []
    prev_spot = None
    sel_ts = ctx.get("selected_ts")
    if history and sel_ts:
        for i, row in enumerate(history):
            if row.get("ts") == sel_ts and i > 0:
                prev_spot = safe_float(history[i - 1].get("spot"), 0.0) or None
                break
    uw_bundle = _uw_bundle_for_context(ticker=ticker, ctx=ctx, uw_entry=uw_entry)
    exposure, previous_exposure = _strategy_exposure_from_context(ctx)
    payload = build_strategy_dashboard(
        ticker=ticker,
        spot=ctx.get("spot"),
        exposure=exposure,
        previous_exposure=previous_exposure,
        snapshot=ctx.get("selected"),
        prev_spot=prev_spot,
        uw_bundle=uw_bundle,
    )
    return jsonify(payload)


@APP.get("/api/trader/suggestions")
def api_trader_suggestions():
    from gex_core.trading.advisor import build_suggestions

    ticker = (request.args.get("ticker") or PRIMARY_TICKER).upper()
    return jsonify({"ticker": ticker, "suggestions": build_suggestions(ticker)})


@APP.get("/api/wall-gex/status")
def api_wall_gex_status():
    from gex_core.trading.low_gex_engine import wall_gex_status

    ticker = (request.args.get("ticker") or PRIMARY_TICKER).upper()
    status = wall_gex_status(ticker)
    spot: float | None = None
    last_cycle: dict[str, Any] = {"ran": False, "reason": "Loading signal"}
    try:
        spot, _, last_cycle = _wall_gex_live_data(ticker)
    except Exception as exc:
        logger.exception("Wall GEX status failed for %s", ticker)
        last_cycle = {"ran": False, "reason": str(exc)}
    return jsonify({"ticker": ticker, "spot": spot, "status": status, "last_cycle": last_cycle})


@APP.post("/api/wall-gex/arm")
def api_wall_gex_arm():
    from gex_core.trading.config import live_trading_allowed, require_live_confirm
    from gex_core.trading.engine import arm_trader
    from gex_core.trading.low_gex_engine import wall_gex_status

    payload = request.get_json(silent=True) or {}
    armed = bool(payload.get("armed", True))
    if armed and live_trading_allowed() and require_live_confirm() and not payload.get("live_confirm"):
        return jsonify(
            {
                "error": "Live Webull trading requires live_confirm: true in the request body.",
                "live_mode": True,
            }
        ), 400
    arm_trader(armed)
    ticker = (payload.get("ticker") or PRIMARY_TICKER).upper()
    return jsonify({"armed": armed, "status": wall_gex_status(ticker)})


@APP.post("/api/wall-gex/run")
def api_wall_gex_run():
    """Manual wall GEX cycle (signal, exits, optional entry)."""
    from gex_core.trading.low_gex_engine import run_wall_gex_cycle

    payload = request.get_json(silent=True) or {}
    ticker = (payload.get("ticker") or PRIMARY_TICKER).upper()
    try:
        spot, exposure, preview = _wall_gex_live_data(ticker)
        if not spot:
            return jsonify({"error": "No spot price available", "last_cycle": preview}), 503
        if exposure is None:
            return jsonify({"error": "No gamma exposure available", "last_cycle": preview}), 503
        result = run_wall_gex_cycle(
            ticker=ticker,
            spot=float(spot),
            exposure=exposure,
            execute=True,
            force=True,
        )
        return jsonify(result)
    except Exception as exc:
        logger.exception("Wall GEX run failed for %s", ticker)
        return jsonify({"error": str(exc)}), 500


@APP.get("/trade")
@APP.get("/webull")
def webull_trade_dashboard():
    """Webull quick options trade desk with entry/exit price guidance."""
    from gex_core.trading.config import execution_ticker, signal_ticker
    from gex_core.trading.webull_quick_trade import dashboard_state

    ticker = (request.args.get("ticker") or PRIMARY_TICKER).upper()
    state = dashboard_state(signal_ticker_arg=ticker)
    return render_template(
        "webull_trade.html",
        ticker=ticker,
        signal_ticker=signal_ticker(),
        execution_ticker=execution_ticker(),
        initial_state=state,
    )


@APP.get("/api/webull/status")
def api_webull_status():
    from gex_core.trading.webull_quick_trade import dashboard_state

    ticker = (request.args.get("ticker") or PRIMARY_TICKER).upper()
    return jsonify(dashboard_state(signal_ticker_arg=ticker))


def _webull_strategy_trade_payload(ticker: str) -> dict[str, Any]:
    """Gamma strategy state + execution-mapped contract for the trade desk."""
    from gex_core.trading.strategy_viz import build_strategy_state
    from gex_core.trading.webull_quick_trade import build_recommended_trade

    uw_entry = get_uw_data(ticker) if _uw_live_enabled() else None
    ctx = build_periscope_context(
        ticker=ticker,
        selected_ts=request.args.get("ts"),
        exposure="gamma",
        uw_entry=uw_entry,
        api_key=uw_api_key(),
    )
    history = ctx.get("history") or []
    prev_spot = None
    sel_ts = ctx.get("selected_ts")
    if history and sel_ts:
        for i, row in enumerate(history):
            if row.get("ts") == sel_ts and i > 0:
                prev_spot = safe_float(history[i - 1].get("spot"), 0.0) or None
                break
    uw_bundle = _uw_bundle_for_context(ticker=ticker, ctx=ctx, uw_entry=uw_entry)
    exposure, previous_exposure = _strategy_exposure_from_context(ctx)
    state = build_strategy_state(
        ticker=ticker,
        spot=ctx.get("spot"),
        exposure=exposure,
        previous_exposure=previous_exposure,
        snapshot=ctx.get("selected"),
        prev_spot=prev_spot,
        uw_bundle=uw_bundle,
    )
    trade = build_recommended_trade(strategy_state=state)
    return {"strategy_state": state, "recommended_trade": trade}


@APP.get("/api/webull/signal")
def api_webull_signal():
    """Recommended gamma signal mapped to execution contract (SPX → SPY)."""
    ticker = (request.args.get("ticker") or PRIMARY_TICKER).upper()
    return jsonify(_webull_strategy_trade_payload(ticker))


@APP.get("/api/webull/quote")
def api_webull_quote():
    from gex_core.trading.webull_quick_trade import quote_payload

    try:
        strike = float(request.args.get("strike", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "strike required"}), 400
    if strike <= 0:
        return jsonify({"error": "strike must be positive"}), 400

    option_type = (request.args.get("type") or request.args.get("option_type") or "call").lower()
    underlying = (request.args.get("underlying") or request.args.get("execution") or "SPY").upper()
    expire_date = request.args.get("expire") or request.args.get("expire_date")
    entry_premium = request.args.get("entry_premium")
    peak_premium = request.args.get("peak_premium")
    spot = request.args.get("spot")

    try:
        entry_f = float(entry_premium) if entry_premium else None
    except (TypeError, ValueError):
        entry_f = None
    try:
        peak_f = float(peak_premium) if peak_premium else None
    except (TypeError, ValueError):
        peak_f = None
    try:
        spot_f = float(spot) if spot else None
    except (TypeError, ValueError):
        spot_f = None

    strategy_trade = None
    if request.args.get("strategy", "1") not in {"0", "false", "no"}:
        ticker = (request.args.get("ticker") or PRIMARY_TICKER).upper()
        try:
            strategy_trade = _webull_strategy_trade_payload(ticker).get("recommended_trade")
        except Exception:
            strategy_trade = None

    return jsonify(
        quote_payload(
            underlying=underlying,
            option_type=option_type,
            strike=strike,
            expire_date=expire_date,
            entry_premium=entry_f,
            peak_premium=peak_f,
            spot=spot_f,
            strategy_trade=strategy_trade,
        )
    )


@APP.post("/api/webull/order/buy")
def api_webull_order_buy():
    from gex_core.trading.config import live_trading_allowed, require_live_confirm
    from gex_core.trading.webull_quick_trade import execute_buy

    if not _admin_action_authorized(request) and os.environ.get("GEX_ADMIN_TOKEN"):
        return jsonify({"error": "Admin token required for live orders"}), 403

    payload = request.get_json(silent=True) or {}
    if live_trading_allowed() and require_live_confirm() and not payload.get("live_confirm"):
        return jsonify(
            {"error": "Live orders require live_confirm: true in the request body.", "live_mode": True}
        ), 400

    try:
        strike = float(payload.get("strike", 0))
        qty = int(payload.get("quantity") or payload.get("qty") or 1)
    except (TypeError, ValueError):
        return jsonify({"error": "invalid strike or quantity"}), 400

    limit = payload.get("limit_price")
    try:
        limit_f = float(limit) if limit is not None else None
    except (TypeError, ValueError):
        limit_f = None

    spot = payload.get("spot")
    try:
        spot_f = float(spot) if spot else 0.0
    except (TypeError, ValueError):
        spot_f = 0.0

    result = execute_buy(
        underlying=str(payload.get("underlying") or "SPY").upper(),
        option_type=str(payload.get("option_type") or "call").lower(),
        strike=strike,
        quantity=max(1, qty),
        limit_price=limit_f,
        expire_date=payload.get("expire_date"),
        spot=spot_f,
        ticker=str(payload.get("ticker") or PRIMARY_TICKER).upper(),
        price_style=str(payload.get("price_style") or "smart"),
        journal=bool(payload.get("journal", True)),
    )
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@APP.post("/api/webull/order/sell")
def api_webull_order_sell():
    from gex_core.trading.config import live_trading_allowed, require_live_confirm
    from gex_core.trading.webull_quick_trade import execute_sell

    if not _admin_action_authorized(request) and os.environ.get("GEX_ADMIN_TOKEN"):
        return jsonify({"error": "Admin token required for live orders"}), 403

    payload = request.get_json(silent=True) or {}
    if live_trading_allowed() and require_live_confirm() and not payload.get("live_confirm"):
        return jsonify(
            {"error": "Live orders require live_confirm: true in the request body.", "live_mode": True}
        ), 400

    try:
        strike = float(payload.get("strike", 0))
        qty = int(payload.get("quantity") or payload.get("qty") or 1)
    except (TypeError, ValueError):
        return jsonify({"error": "invalid strike or quantity"}), 400

    limit = payload.get("limit_price")
    entry = payload.get("entry_premium")
    try:
        limit_f = float(limit) if limit is not None else None
        entry_f = float(entry) if entry is not None else None
    except (TypeError, ValueError):
        limit_f = None
        entry_f = None

    trade_id = payload.get("trade_id")
    try:
        trade_id_i = int(trade_id) if trade_id is not None else None
    except (TypeError, ValueError):
        trade_id_i = None

    spot = payload.get("spot")
    try:
        spot_f = float(spot) if spot else 0.0
    except (TypeError, ValueError):
        spot_f = 0.0

    result = execute_sell(
        underlying=str(payload.get("underlying") or "SPY").upper(),
        option_type=str(payload.get("option_type") or "call").lower(),
        strike=strike,
        quantity=max(1, qty),
        limit_price=limit_f,
        expire_date=payload.get("expire_date"),
        entry_premium=entry_f,
        spot=spot_f,
        ticker=str(payload.get("ticker") or PRIMARY_TICKER).upper(),
        trade_id=trade_id_i,
        price_style=str(payload.get("price_style") or "smart"),
        journal=bool(payload.get("journal", True)),
    )
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@APP.get("/api/periscope/timeline")
def api_periscope_timeline():
    """Available dates and intraday slices for the session picker."""
    ticker = PRIMARY_TICKER
    timestamps = list_periscope_timestamps(ticker, api_key=uw_api_key())
    requested_ts = request.args.get("ts")
    requested_date = request.args.get("date")
    resolved = resolve_selected_timestamp(timestamps, ts=requested_ts, date=requested_date)
    timeline = build_timeline_navigation(timestamps, resolved, ticker=ticker)
    active_date = timeline.get("selected_date")
    slices_by_date = {}
    if active_date:
        slices_by_date[active_date] = build_slice_options(timestamps, active_date, ticker=ticker)
    return jsonify(
        {
            "ticker": ticker,
            "dates": timeline.get("available_dates", []),
            "slices_by_date": slices_by_date,
            "timeline": timeline,
            "data_path": (
                "uw_api"
                if active_date and should_use_api_for_date(active_date, api_key=uw_api_key())
                else "exports"
            ),
        }
    )


@APP.get("/api/agent/analyze")
def api_agent_analyze():
    ticker = PRIMARY_TICKER
    exposure = request.args.get("exposure", "gamma")
    requested_ts = request.args.get("ts")
    uw_entry = get_uw_data(ticker) if _uw_live_enabled() else None
    ctx = build_periscope_context(
        ticker=ticker,
        selected_ts=requested_ts,
        exposure=exposure,
        uw_entry=uw_entry,
        api_key=uw_api_key(),
    )
    selected = ctx.get("selected") or {}
    gex_series = ctx.get("exposure_series")
    uw_agg = uw_entry.get("agg") if uw_entry else None
    pred_history = _prediction_history(ticker)
    result = analyze_market_exposure(
        ticker=ticker,
        spot=safe_float(ctx.get("spot"), 0.0) or 5000.0,
        gex_by_strike=gex_series if gex_series is not None else pd.Series(dtype=float),
        cumulative_gex=selected.get("cumulative"),
        total_gex_bn=safe_float(ctx.get("total_gex"), 0.0),
        gamma_flip=ctx.get("gamma_flip"),
        history=pred_history if uw_agg else None,
        exposure_type=exposure,
        agg=uw_agg,
        spot_gamma_bn=uw_entry.get("spot_gamma_bn") if uw_entry else None,
        api_key=uw_api_key() if uw_agg else None,
        knn_prediction=predict_next_snapshot(
            pred_history,
            lookback_days=prediction_lookback_days(ticker),
        ) if uw_agg else None,
    )
    return jsonify(result)


@APP.get("/api/agent/backtest")
def api_agent_backtest():
    """Walk-forward auto-trader backtest using current env parameters."""
    from gex_core.trading.backtest_agent import current_trader_parameters, run_agent_backtest

    ticker = (request.args.get("ticker") or PRIMARY_TICKER).upper()
    if not is_supported_ticker(ticker):
        return jsonify({"error": "Unsupported ticker"}), 404

    lookback_days = request.args.get("lookback_days", type=int)
    max_snapshots = request.args.get("max_snapshots", type=int)
    starting_capital = request.args.get("starting_capital", type=float)
    compact = request.args.get("compact", "1").lower() not in {"0", "false", "no"}

    try:
        result = run_agent_backtest(
            ticker,
            lookback_days=lookback_days,
            max_snapshots=max_snapshots,
            starting_capital=starting_capital,
            compact=compact,
        )
    except Exception as exc:
        logger.exception("Agent backtest failed for %s", ticker)
        return jsonify({"error": str(exc), "parameters": current_trader_parameters()}), 500

    return jsonify(result)


@APP.get("/api/agent/monte-carlo-confidence")
def api_agent_monte_carlo_confidence():
    """Sweep AI advisor confidence thresholds and rank by walk-forward ROI."""
    from gex_core.trading.backtest_agent import (
        current_trader_parameters,
        run_agent_confidence_monte_carlo,
    )

    ticker = (request.args.get("ticker") or PRIMARY_TICKER).upper()
    if not is_supported_ticker(ticker):
        return jsonify({"error": "Unsupported ticker"}), 404

    lookback_days = request.args.get("lookback_days", type=int)
    max_snapshots = request.args.get("max_snapshots", type=int)
    starting_capital = request.args.get("starting_capital", type=float)
    min_conf_start = request.args.get("min_conf_start", type=float)
    min_conf_stop = request.args.get("min_conf_stop", type=float)
    min_conf_step = request.args.get("min_conf_step", type=float)
    strong_raw = request.args.get("strong_levels", "")
    strong_levels = [float(x.strip()) for x in strong_raw.split(",") if x.strip()] or None
    compact = request.args.get("compact", "1").lower() not in {"0", "false", "no"}

    kwargs: dict = {
        "ticker": ticker,
        "lookback_days": lookback_days,
        "max_snapshots": max_snapshots,
        "starting_capital": starting_capital,
        "compact": compact,
    }
    if min_conf_start is not None:
        kwargs["min_conf_start"] = min_conf_start
    if min_conf_stop is not None:
        kwargs["min_conf_stop"] = min_conf_stop
    if min_conf_step is not None:
        kwargs["min_conf_step"] = min_conf_step
    if strong_levels is not None:
        kwargs["strong_levels"] = strong_levels

    try:
        result = run_agent_confidence_monte_carlo(**kwargs)
    except Exception as exc:
        logger.exception("Confidence Monte Carlo failed for %s", ticker)
        return jsonify({"error": str(exc), "parameters": current_trader_parameters()}), 500

    return jsonify(result)


@APP.get("/api/agent/predict")
def api_agent_predict():
    """Feed all Unusual Whales data to the AI and return structured predictions."""
    ticker = PRIMARY_TICKER
    uw_entry = get_uw_data(ticker) if _uw_live_enabled() else None
    if not uw_entry or not uw_entry.get("agg"):
        return jsonify(
            {
                "error": "Live Unusual Whales data unavailable",
                "detail": _REFRESH_REASON_MESSAGES.get(_uw_failure_reason(ticker), "Configure UW_API_KEY"),
            }
        ), 503

    agg = uw_entry["agg"]
    spot = safe_float(uw_entry.get("spot"), 0.0)
    pred_history = _prediction_history(ticker)
    knn = predict_next_snapshot(pred_history, lookback_days=prediction_lookback_days(ticker))

    result = predict_market_exposure(
        ticker=ticker,
        spot=spot,
        gex_by_strike=agg.gex_by_strike,
        cumulative_gex=agg.cumulative_gex,
        total_gex_bn=agg.total_gex_bn,
        agg=agg,
        gamma_flip=uw_entry.get("gamma_flip"),
        spot_gamma_bn=uw_entry.get("spot_gamma_bn"),
        history=pred_history,
        knn_prediction=knn,
        api_key=uw_api_key(),
    )
    return jsonify(result)


def _chat_context(ticker: str, exposure: str, requested_ts: str | None) -> dict:
    """Build periscope + UW context for chat endpoints."""
    uw_entry = get_uw_data(ticker) if _uw_live_enabled() else None
    ctx = build_periscope_context(
        ticker=ticker,
        selected_ts=requested_ts,
        exposure=exposure,
        uw_entry=uw_entry,
        api_key=uw_api_key(),
    )
    selected = ctx.get("selected") or {}
    gex_series = ctx.get("exposure_series")
    uw_agg = uw_entry.get("agg") if uw_entry else None
    pred_history = _prediction_history(ticker)
    knn = (
        predict_next_snapshot(pred_history, lookback_days=prediction_lookback_days(ticker))
        if uw_agg
        else None
    )
    return {
        "ctx": ctx,
        "selected": selected,
        "gex_series": gex_series,
        "uw_entry": uw_entry,
        "uw_agg": uw_agg,
        "pred_history": pred_history,
        "knn": knn,
    }


@APP.post("/api/chat")
def api_chat():
    """Conversational GEX assistant backed by full Unusual Whales data."""
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    session_id = payload.get("session_id")
    exposure = str(payload.get("exposure") or request.args.get("exposure", "gamma")).lower()
    requested_ts = payload.get("ts") or request.args.get("ts")

    if not message:
        return jsonify({"error": "message is required"}), 400

    ticker = PRIMARY_TICKER
    chat_ctx = _chat_context(ticker, exposure, requested_ts)
    ctx = chat_ctx["ctx"]
    selected = chat_ctx["selected"]
    gex_series = chat_ctx["gex_series"]
    uw_entry = chat_ctx["uw_entry"]
    uw_agg = chat_ctx["uw_agg"]

    result = chat_reply(
        session_id=session_id,
        user_message=message,
        ticker=ticker,
        spot=safe_float(ctx.get("spot"), 0.0) or 5000.0,
        gex_by_strike=gex_series if gex_series is not None else pd.Series(dtype=float),
        cumulative_gex=selected.get("cumulative"),
        total_gex_bn=safe_float(ctx.get("total_gex"), 0.0),
        gamma_flip=ctx.get("gamma_flip"),
        exposure_type=exposure,
        agg=uw_agg,
        spot_gamma_bn=uw_entry.get("spot_gamma_bn") if uw_entry else None,
        history=chat_ctx["pred_history"] if uw_agg else None,
        knn_prediction=chat_ctx["knn"],
        api_key=uw_api_key() if uw_agg else None,
    )
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@APP.post("/api/chat/reset")
def api_chat_reset():
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id")
    if session_id:
        reset_session(str(session_id))
    return jsonify({"ok": True})


@APP.get("/api/latest-summary")
def api_latest_summary():
    ticker = PRIMARY_TICKER
    payload = _ticker_api_payload(ticker, request.args.get("ts"))
    return jsonify(payload)


@APP.get("/api/signals")
def api_signals():
    ticker = PRIMARY_TICKER
    payload = _ticker_api_payload(ticker, request.args.get("ts"))
    return jsonify(
        {
            "ticker": ticker,
            "selected_ts": payload.get("selected_ts"),
            "alerts": payload.get("alerts", []),
            "strategy_notes": payload.get("strategy_notes", []),
            "confluence": payload.get("confluence"),
            "probabilities": payload.get("probabilities"),
        }
    )


@APP.get("/api/watchlist")
def api_watchlist():
    tickers = find_available_tickers(EXPORT_DIR)
    return jsonify({"rows": build_watchlist_rows(tickers), "count": len(tickers)})


@APP.get("/widget/<ticker>")
def ticker_widget(ticker: str):
    ticker = PRIMARY_TICKER
    payload = _ticker_api_payload(ticker, request.args.get("ts"))
    summary = payload.get("summary") or {}
    confluence = payload.get("confluence") or {"score": 0.0, "label": "low"}
    theme = request.args.get("theme", "dark")
    compact = request.args.get("compact", "0") == "1"
    return render_template(
        "widget.html",
        ticker=ticker,
        summary=summary,
        confluence=confluence,
        updated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        theme=theme,
        compact=compact,
    )


@APP.route("/exports/<path:filename>")
def export_file(filename):
    return send_from_directory(EXPORT_DIR, filename)


# Backward-compatible path for existing deep links.
@APP.route("/data/exports/<path:filename>")
def export_file_legacy(filename):
    return send_from_directory(EXPORT_DIR, filename)


@APP.route("/img/<path:filename>")
def img_file(filename):
    path = IMG_DIR
    if not path.exists():
        abort(404)
    return send_from_directory(path, filename)


_scheduler: BackgroundScheduler | None = None
_scheduler_lock_path = Path(os.environ.get("GEX_SCHEDULER_LOCK", "data/.gex_scheduler.lock"))


def _previous_spot_from_context(ctx: dict) -> float | None:
    history = ctx.get("history") or []
    if len(history) >= 2:
        prev = safe_float(history[-2].get("spot"), 0.0)
        return prev if prev > 0 else None
    selected = ctx.get("selected") or {}
    timeline = ctx.get("timeline") or {}
    prev_ts = timeline.get("prev_ts")
    if prev_ts and history:
        for row in history:
            if row.get("ts") == prev_ts:
                prev = safe_float(row.get("spot"), 0.0)
                return prev if prev > 0 else None
    return safe_float(selected.get("spot"), 0.0) or None


def _uw_bundle_for_context(
    *,
    ticker: str,
    ctx: dict[str, Any],
    uw_entry: dict[str, Any] | None,
    fetch_extras: bool = False,
) -> dict[str, Any] | None:
    from gex_core.uw_context_bundle import try_build_uw_bundle_from_entry

    spot = safe_float(ctx.get("spot"), 0.0)
    if spot <= 0:
        return None
    return try_build_uw_bundle_from_entry(
        ticker=ticker,
        spot=spot,
        uw_entry=uw_entry,
        gamma_flip=ctx.get("gamma_flip"),
        history=ctx.get("history"),
        api_key=uw_api_key(),
        fetch_extras=fetch_extras,
    )


def _resolve_trader_spot(ticker: str, fallback: float | None) -> float | None:
    """Prefer UW websocket spot for high-frequency exit checks."""
    from gex_core.uw_price_stream import get_uw_price_stream
    from gex_core.trading.config import execution_ticker, signal_ticker
    from gex_core.trading.execution import record_spot_ratio

    live = get_uw_price_stream().get_latest_price(ticker)
    if live and live > 0:
        sig = signal_ticker().upper()
        exe = execution_ticker().upper()
        if ticker.upper() == sig and exe != sig:
            exec_live = get_uw_price_stream().get_latest_price(exe)
            if exec_live and exec_live > 0:
                record_spot_ratio(signal_spot=float(live), execution_spot=float(exec_live))
        return float(live)
    if fallback and fallback > 0:
        return float(fallback)
    return None


def _run_wall_gex_trader(ticker: str) -> dict[str, Any] | None:
    """Evaluate wall GEX signal and manage paper/live option trades."""
    from gex_core.trading.config import wall_gex_auto_enabled
    from gex_core.trading.journal import is_trader_armed
    from gex_core.trading.low_gex_engine import run_wall_gex_cycle

    if not wall_gex_auto_enabled() or not is_trader_armed():
        return None
    uw_entry = get_uw_data(ticker) if _uw_live_enabled() else None
    ctx = build_periscope_context(
        ticker=ticker,
        exposure="gamma",
        uw_entry=uw_entry,
        api_key=uw_api_key(),
    )
    spot = _resolve_trader_spot(ticker, safe_float(ctx.get("spot"), 0.0) or None)
    if not spot:
        return None
    exposure, _ = _strategy_exposure_from_context(ctx)
    if exposure is None:
        return None
    return run_wall_gex_cycle(
        ticker=ticker,
        spot=float(spot),
        exposure=exposure,
        execute=True,
    )


def _run_auto_trader(ticker: str) -> dict[str, Any] | None:
    """Evaluate gamma signals and manage paper option trades."""
    from gex_core.trading.config import auto_trader_enabled
    from gex_core.trading.engine import run_trading_cycle
    from gex_core.uw_context_bundle import build_uw_context_bundle

    if not auto_trader_enabled():
        return None
    uw_entry = get_uw_data(ticker) if _uw_live_enabled() else None
    ctx = build_periscope_context(
        ticker=ticker,
        exposure="gamma",
        uw_entry=uw_entry,
        api_key=uw_api_key(),
    )
    spot = _resolve_trader_spot(ticker, safe_float(ctx.get("spot"), 0.0) or None)
    if not spot:
        return None
    uw_bundle = None
    if uw_entry and uw_entry.get("agg") is not None:
        uw_bundle = build_uw_context_bundle(
            ticker=ticker,
            spot=float(spot),
            agg=uw_entry["agg"],
            gamma_flip=ctx.get("gamma_flip"),
            spot_gamma_bn=uw_entry.get("spot_gamma_bn"),
            api_key=uw_api_key(),
            fetch_extras=False,
        )
    exposure, previous_exposure = _strategy_exposure_from_context(ctx)
    return run_trading_cycle(
        ticker=ticker,
        spot=float(spot),
        exposure=exposure,
        previous_exposure=previous_exposure,
        uw_bundle=uw_bundle,
        snapshot=ctx.get("selected"),
        previous_spot=_previous_spot_from_context(ctx),
    )


def _auto_dispatch_alerts(ticker: str) -> None:
    """Generate alerts and run auto-dispatch. Background scheduler only."""
    history = _dashboard_history(ticker)
    if not history:
        return
    selected = _select_snapshot(history, None)
    prediction_history = _prediction_history(ticker)
    prediction = predict_next_snapshot(
        prediction_history,
        lookback_days=prediction_lookback_days(ticker),
    )
    alerts = generate_alerts(history, selected, prediction)
    maybe_dispatch_alerts(ticker, alerts, manual=False)


def _scheduled_refresh():
    # Staleness-gated (not force=True): when a manual/page refresh already wrote
    # a fresh snapshot this interval, skip the redundant UW fetch. This avoids
    # burning the UW per-minute/daily request budget on duplicate pulls.
    from gex_core.trading.config import trader_cycle_seconds

    try:
        refresh_tickers(REFRESH_TICKERS)
    except Exception:
        logger.exception("Scheduled GEX refresh failed")
        return
    for ticker in REFRESH_TICKERS:
        try:
            _auto_dispatch_alerts(ticker)
        except Exception:
            logger.exception("Auto-dispatch failed for %s", ticker)
        if trader_cycle_seconds() <= 0:
            try:
                _run_auto_trader(ticker)
            except Exception:
                logger.exception("Auto-trader cycle failed for %s", ticker)


def _scheduled_trader_tick():
    """High-frequency trader loop — exits on live spot, entries on latest gamma."""
    from gex_core.trading.config import auto_trader_enabled, trader_cycle_seconds

    if not auto_trader_enabled() or trader_cycle_seconds() <= 0:
        return
    for ticker in REFRESH_TICKERS:
        try:
            _run_auto_trader(ticker)
        except Exception:
            logger.exception("Auto-trader tick failed for %s", ticker)


def _scheduled_wall_gex_tick():
    """Wall GEX trader loop — exits on live spot, entries on lowest-γ wall."""
    from gex_core.trading.config import wall_gex_auto_enabled, wall_gex_cycle_seconds

    if not wall_gex_auto_enabled() or wall_gex_cycle_seconds() <= 0:
        return
    for ticker in REFRESH_TICKERS:
        try:
            _run_wall_gex_trader(ticker)
        except Exception:
            logger.exception("Wall GEX tick failed for %s", ticker)


def _acquire_scheduler_lock() -> bool:
    _scheduler_lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_file = _scheduler_lock_path.open("w")
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_file.write(str(os.getpid()))
        lock_file.flush()
        return True
    except OSError:
        return False


def start_background_refresh():
    global _scheduler
    from gex_core.trading.config import auto_trader_enabled, trader_cycle_seconds, wall_gex_auto_enabled, wall_gex_cycle_seconds

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    if os.environ.get("GEX_DISABLE_SCHEDULER", "").lower() in {"1", "true", "yes"}:
        return

    if _scheduler is not None:
        return

    if not _acquire_scheduler_lock():
        logger.info("Another process owns the GEX refresh scheduler lock")
        return

    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(
        _scheduled_refresh,
        trigger="interval",
        minutes=REFRESH_MINUTES,
        id="gex_refresh",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    _scheduler.start()
    atexit.register(lambda: _scheduler.shutdown(wait=False) if _scheduler else None)

    cycle_sec = trader_cycle_seconds()
    if auto_trader_enabled() and cycle_sec > 0:
        _scheduler.add_job(
            _scheduled_trader_tick,
            trigger="interval",
            seconds=cycle_sec,
            id="gex_trader_tick",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        logger.info("Auto-trader high-frequency loop every %ss", cycle_sec)

    wall_cycle_sec = wall_gex_cycle_seconds()
    if wall_gex_auto_enabled() and wall_cycle_sec > 0:
        _scheduler.add_job(
            _scheduled_wall_gex_tick,
            trigger="interval",
            seconds=wall_cycle_sec,
            id="gex_wall_gex_tick",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        logger.info("Wall GEX trader loop every %ss", wall_cycle_sec)

    if any(get_latest_ts(ticker) is None for ticker in REFRESH_TICKERS):
        _scheduler.add_job(
            _scheduled_refresh,
            trigger="date",
            id="gex_refresh_bootstrap",
            replace_existing=True,
            max_instances=1,
        )


deferred_web_startup(
    refresh_fn=start_background_refresh,
    price_stream_fn=lambda: start_uw_price_stream(_price_stream_tickers()),
)


def _price_stream_tickers() -> list[str]:
    from gex_core.trading.config import execution_ticker, signal_ticker

    tickers = list(REFRESH_TICKERS)
    for symbol in (signal_ticker(), execution_ticker()):
        if symbol and symbol not in tickers:
            tickers.append(symbol)
    return tickers


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    port = int(os.environ.get("PORT", "8080"))
    APP.run(host="0.0.0.0", port=port, debug=debug_mode)
