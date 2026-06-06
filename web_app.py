"""
Flask SPX gamma dashboard.

Serves historical GEX snapshots from ``data/exports/``, renders Plotly charts,
runs weighted-KNN forecasts, and optionally auto-refreshes via APScheduler when
``UW_API_KEY`` is set. All state is file-based; there is no database.
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

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, Response, abort, jsonify, redirect, render_template, request, send_from_directory, url_for

from gex_core.backtest_metrics import backtest_delta_sign_accuracy
from gex_core.calibration import calibrate_confidence
from gex_core.charts import (
    make_ai_insights_chart,
    make_0dte_movement_chart,
    make_cumulative_gex_chart,
    make_gex_profile_chart,
    make_positive_strike_chart,
    make_prediction_gamma_chart,
    make_spx_price_chart,
    make_timeline_chart,
    safe_float,
)
from gex_core.periscope_charts import build_periscope_charts
from gex_core.market_exposure_agent import analyze_market_exposure, predict_market_exposure
from gex_core.gex_chatbot import build_welcome_message, chat_reply, reset_session
from gex_core.periscope import (
    build_periscope_context,
    build_slice_options,
    build_timeline_navigation,
    group_timestamps_by_date,
    resolve_selected_timestamp,
)
from gex_core.periscope_api import (
    clear_periscope_api_cache,
    list_periscope_timestamps,
    should_use_api_for_date,
)
from gex_core.exports import EXPORT_DIR
from gex_core.history import (
    build_gamma_levels_timeline,
    build_history,
    get_latest_ts,
    list_tickers,
    list_timestamps,
    load_snapshot_at_ts,
    ts_label,
)
from gex_core.market_features import (
    fetch_spx_price,
    fetch_spx_price_history,
    fetch_spx_price_series_for_dashboard,
)
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
    simulate_spot_scenario,
)
from gex_core.predict import (
    apply_flow_to_prediction,
    forecast_blocker_message,
    load_flow_predictions,
    predict_next_snapshot,
    similar_setups,
)
from gex_core.alert_dispatch import maybe_dispatch_alerts
from gex_core.env_bootstrap import bootstrap_env, uw_api_configured, uw_api_key
from gex_core.refresh import DEFAULT_REFRESH_MINUTES, refresh_ticker, refresh_tickers
from gex_core.export_diagnostics import prediction_lookback_days, summarize_export_state
from gex_core.system_status import build_system_status
from gex_core.tickers import PRIMARY_TICKER, is_supported_ticker, supported_tickers

APP = Flask(__name__)
app = APP
logger = logging.getLogger(__name__)

bootstrap_env()

# ─────────────────────────────────────────────────────────────────────────────
# Unusual Whales live data layer
# ─────────────────────────────────────────────────────────────────────────────

_UW_CACHE: dict[str, dict] = {}          # ticker → {spot, agg, ts, analysis}
_UW_CACHE_TTL = int(
    os.environ.get(
        "GEX_UW_CACHE_TTL_SECONDS",
        str(max(30, int(os.environ.get("GEX_REFRESH_INTERVAL_MINUTES", "10")) * 60)),
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
        from gex_core.features import estimate_gamma_flip
        from gex_core.spot_exposure import spot_exposure_gamma_flip, spot_exposure_net_series

        spot, agg = fetch_uw_gex(ticker, api_key=uw_api_key())
        spot_df = agg.gex_by_strike.attrs.get("spot_exposures_df")
        spot_gamma = spot_exposure_net_series(spot_df, "gamma") if isinstance(spot_df, pd.DataFrame) else pd.Series(dtype=float)
        gamma_flip = spot_exposure_gamma_flip(spot_gamma) if not spot_gamma.empty else estimate_gamma_flip(agg.cumulative_gex)
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


def find_available_tickers(export_dir: Path | None = None):
    tickers = list_tickers(export_dir)
    return tickers if PRIMARY_TICKER in tickers else supported_tickers()


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


def _dashboard_skip_backtest() -> bool:
    raw = os.environ.get("GEX_DASHBOARD_SKIP_BACKTEST", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


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


def _fallback_predicted_strike(selected: dict, prediction: dict):
    """Scale the current strike profile when KNN neighbors lack per-strike targets."""
    import pandas as pd

    strike = selected.get("strike")
    if strike is None or (isinstance(strike, pd.Series) and strike.empty):
        return None
    strike = pd.Series(strike, dtype=float)
    cur_total = safe_float(selected.get("total_gex"), 0.0)
    delta = safe_float(prediction.get("predicted_delta_gex"), 0.0)
    if abs(cur_total) < 1e-9:
        return None
    return strike * (delta / cur_total)


def _build_predicted_strike_chart(
    prediction: dict | None,
    *,
    selected: dict,
    ticker: str,
    csv_spot: float | None,
) -> str | None:
    if not prediction:
        return None
    knn_strike = prediction.get("knn_strike")
    if knn_strike is None:
        knn_strike = prediction.get("predicted_strike")
    flow_strike = prediction.get("flow_strike")
    combined_strike = prediction.get("predicted_strike")
    if knn_strike is None and combined_strike is None:
        knn_strike = _fallback_predicted_strike(selected, prediction)
        combined_strike = knn_strike
    if knn_strike is None and combined_strike is None:
        return None
    return make_prediction_gamma_chart(
        knn_strike=knn_strike,
        combined_strike=combined_strike,
        flow_strike=flow_strike,
        ticker=ticker,
        spot=csv_spot,
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


def _previous_same_day_snapshot(history: list, selected: dict) -> dict | None:
    selected_ts = selected.get("ts")
    if not selected_ts:
        return None
    selected_day = selected_ts[:10]
    idx = -1
    for i, row in enumerate(history):
        if row.get("ts") == selected_ts:
            idx = i
            break
    if idx <= 0:
        return None
    for row in reversed(history[:idx]):
        if str(row.get("ts", ""))[:10] == selected_day:
            return row
    return None


def _build_0dte_movement_panel(selected: dict, previous: dict | None) -> dict:
    if not previous:
        return {
            "available": False,
            "message": "Waiting for the next same-day snapshot to measure 0DTE movement.",
        }
    import pandas as pd

    current_strike = pd.Series(selected.get("strike"), dtype=float).sort_index()
    previous_strike = pd.Series(previous.get("strike"), dtype=float).sort_index()
    if current_strike.empty or previous_strike.empty:
        return {
            "available": False,
            "message": "Strike-level data is unavailable for 0DTE movement.",
        }

    delta = current_strike.subtract(previous_strike, fill_value=0.0)
    top_abs = delta.abs().sort_values(ascending=False)
    top_strike = float(top_abs.index[0]) if not top_abs.empty else None
    top_delta = float(delta.loc[top_strike]) if top_strike is not None else 0.0
    spot_delta = safe_float(selected.get("spot"), 0.0) - safe_float(previous.get("spot"), 0.0)
    elapsed_minutes = None
    try:
        elapsed = datetime.strptime(selected["ts"], "%Y-%m-%d_%H%M%S") - datetime.strptime(previous["ts"], "%Y-%m-%d_%H%M%S")
        elapsed_minutes = max(0, int(elapsed.total_seconds() // 60))
    except Exception:
        pass

    return {
        "available": True,
        "previous_label": previous.get("ts_label"),
        "current_label": selected.get("ts_label"),
        "elapsed_minutes": elapsed_minutes,
        "net_delta": float(delta.sum()),
        "gross_delta": float(delta.abs().sum()),
        "positive_delta": float(delta[delta > 0].sum()),
        "negative_delta": float(delta[delta < 0].sum()),
        "top_strike": top_strike,
        "top_delta": top_delta,
        "spot_delta": spot_delta,
    }


def _safe_similar_setups(history: list) -> list:
    try:
        return similar_setups(history, top_n=5)
    except Exception:
        logger.exception("Similar setups lookup failed")
        return []


def _prediction_public_view(prediction: dict | None) -> dict | None:
    if not prediction:
        return None
    return {
        k: v
        for k, v in prediction.items()
        if k not in {"predicted_strike", "knn_strike", "flow_strike"}
    }


def _spx_redirect(**extra):
    args = request.args.to_dict(flat=True)
    args.update(extra)
    return redirect(url_for("index", **args))


def _render_periscope_dashboard(ticker: str = PRIMARY_TICKER):
    """Periscope-style market maker exposure view (replaces the legacy command center)."""
    ticker = ticker.upper()
    exposure = request.args.get("exposure", "gamma").lower()
    requested_ts = request.args.get("ts")
    requested_date = request.args.get("date")
    force_refresh = request.args.get("force_refresh", "").lower() in {"1", "true", "yes"}
    bootstrap_status = request.args.get("bootstrap")
    refresh_message = None

    if force_refresh:
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
        if refreshed_csv or refreshed_live:
            bootstrap_status = "ok"
        else:
            bootstrap_status = "failed"
            reason = _uw_failure_reason(ticker)
            refresh_message = _REFRESH_REASON_MESSAGES.get(reason, _REFRESH_REASON_MESSAGES["error"])

    has_exports = bool(list_periscope_timestamps(ticker, api_key=uw_api_key()))
    if bootstrap_status == "failed" and has_exports:
        bootstrap_status = "stale"

    uw_entry = get_uw_data(ticker) if _uw_live_enabled() else None
    _, _, price_source = _dashboard_spx_price_context(ticker)
    ctx = build_periscope_context(
        ticker=ticker,
        selected_ts=requested_ts,
        selected_date=requested_date,
        exposure=exposure,
        uw_entry=uw_entry,
        api_key=uw_api_key(),
    )
    price_points = ctx.get("price_points") or []
    timeline = ctx.get("timeline") or {}

    selected = ctx.get("selected") or {}
    gex_series = ctx.get("exposure_series")
    prev_series = ctx.get("previous_exposure")
    spot = ctx.get("spot")

    charts = build_periscope_charts(
        ticker=ticker,
        exposure_type=exposure,
        spot=spot,
        exposure_profile=ctx.get("exposure_profile"),
        exposure_extended=ctx.get("exposure_extended"),
        previous_exposure=prev_series,
        price_points=price_points,
        highlight_label=selected.get("ts_label"),
        mm_positions=ctx.get("mm_positions"),
        gamma_flip=ctx.get("gamma_flip"),
        call_wall=ctx.get("call_wall"),
        put_wall=ctx.get("put_wall"),
    )
    price_chart_json = charts.price
    exposure_chart_json = charts.exposures
    change_chart_json = charts.change
    cumulative_chart_json = charts.cumulative
    positions_chart_json = charts.positions
    ladder_chart_json = charts.ladder

    cumulative = selected.get("cumulative")
    uw_agg = uw_entry.get("agg") if uw_entry else None
    chat_welcome = build_welcome_message(
        ticker=ticker,
        spot=safe_float(spot, 0.0) or None,
        regime=ctx.get("regime", "N/A"),
        total_gex=safe_float(ctx.get("total_gex"), 0.0),
        gamma_flip=ctx.get("gamma_flip"),
        exposure=exposure,
    )

    csv_source = selected.get("data_source") or ctx.get("data_path") or "unusual_whales"
    data_source = f"Unusual Whales · {ctx.get('selected_label', 'latest')} ({csv_source})"
    if uw_entry:
        data_source += " · live API"

    return render_template(
        "periscope.html",
        ticker=ticker,
        exposure=exposure,
        spot=spot,
        regime=ctx.get("regime", "N/A"),
        total_gex=ctx.get("total_gex", 0.0),
        gamma_flip=ctx.get("gamma_flip"),
        selected_ts=ctx.get("selected_ts"),
        selected_date=ctx.get("selected_date"),
        selected_label=ctx.get("selected_label"),
        timestamps=ctx.get("timestamps", []),
        timeline=timeline,
        replay_index=ctx.get("replay_index", 0),
        data_source=data_source,
        price_chart_json=price_chart_json,
        exposure_chart_json=exposure_chart_json,
        change_chart_json=change_chart_json,
        cumulative_chart_json=cumulative_chart_json,
        positions_chart_json=positions_chart_json,
        ladder_chart_json=ladder_chart_json,
        is_live_slice=timeline.get("is_latest", False),
        chat_welcome=chat_welcome,
        bootstrap_status=bootstrap_status,
        refresh_message=refresh_message,
        uw_configured=uw_api_configured(),
        refresh_minutes=REFRESH_MINUTES,
        vanna_charm_available=ctx.get("vanna_charm_available", False),
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
    # Liveness: return 200 when exports exist so stale data does not block routing.
    code = 200 if status.get("ready") else 503
    return jsonify(status), code


@APP.route("/")
def index():
    return _render_periscope_dashboard(PRIMARY_TICKER)


@APP.route("/ticker/<ticker>")
@APP.route("/ticker/<ticker>/")
def ticker_page(ticker):
    ticker = ticker.upper()
    if not is_supported_ticker(ticker):
        return _spx_redirect()
    return _render_periscope_dashboard(ticker)


@APP.route("/legacy/ticker/<ticker>")
@APP.route("/legacy/ticker/<ticker>/")
def legacy_ticker_page(ticker):
    """Legacy full gamma dashboard (preserved for deep analysis)."""
    ticker = ticker.upper()
    if not is_supported_ticker(ticker):
        return _spx_redirect()
    bootstrap_status = request.args.get("bootstrap")
    force_refresh = request.args.get("force_refresh", "").lower() in {"1", "true", "yes"}
    force_refresh_failed = False
    refresh_message = None
    if force_refresh:
        refreshed_csv = False
        refreshed_live = False
        if not uw_api_configured():
            logger.warning("Forced refresh skipped for %s — UW_API_KEY is not configured", ticker)
        else:
            try:
                refreshed_csv = refresh_ticker(ticker, force=True)
            except Exception:
                logger.exception("Force CSV refresh failed for %s", ticker)
            try:
                refreshed_live = refresh_uw_data(ticker, force=True) is not None
            except Exception:
                logger.exception("Force UW refresh failed for %s", ticker)
        if refreshed_csv or refreshed_live:
            bootstrap_status = "ok"
        else:
            # Distinguish "no data at all" (hard failure) from "couldn't fetch
            # fresh data but a cached snapshot exists" (soft, stale). The latter
            # is downgraded once we confirm history is available below.
            force_refresh_failed = True
            reason = _uw_failure_reason(ticker)
            refresh_message = _REFRESH_REASON_MESSAGES.get(reason, _REFRESH_REASON_MESSAGES["error"])
            logger.warning("Forced refresh failed for %s (reason=%s)", ticker, reason)
            bootstrap_status = "failed"

    history = _dashboard_history(ticker)

    if force_refresh_failed and history:
        bootstrap_status = "stale"

    if not history:
        selected = {
            "ts_label": "No snapshot history available yet",
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
        uw_entry = get_uw_data(ticker)
        profile_json = None
        current_strike_chart_json = None
        if uw_entry and uw_entry.get("agg") is not None:
            uw_agg = uw_entry["agg"]
            selected.update(
                {
                    "regime": "LONG gamma" if uw_agg.total_gex_bn >= 0 else "SHORT gamma",
                    "total_gex": float(uw_agg.total_gex_bn),
                    "pos_gex": float(uw_agg.gex_by_strike[uw_agg.gex_by_strike > 0].sum()),
                    "neg_gex": float(uw_agg.gex_by_strike[uw_agg.gex_by_strike < 0].sum()),
                    "call_wall": float(uw_agg.gex_by_strike.idxmax()) if len(uw_agg.gex_by_strike) else None,
                    "put_wall": float(uw_agg.gex_by_strike.idxmin()) if len(uw_agg.gex_by_strike) else None,
                    "gamma_flip": uw_entry.get("gamma_flip"),
                    "spot": uw_entry.get("spot"),
                    "near_term_ratio": 0.0,
                }
            )
            profile_json = make_gex_profile_chart(
                uw_agg.gex_by_strike,
                ticker,
                spot=uw_entry.get("spot"),
                title="Gamma Exposure Map (UW Live)",
                cumulative_series=uw_agg.cumulative_gex,
                gamma_flip=uw_entry.get("gamma_flip"),
                call_wall=selected.get("call_wall"),
                put_wall=selected.get("put_wall"),
            )
            current_strike_chart_json = make_positive_strike_chart(
                uw_agg.gex_by_strike,
                ticker,
                "Current Position Focus (Positive GEX)",
                spot=uw_entry.get("spot") if uw_entry else None,
            )

        spx_price_points, spx_current_price, spx_price_source = _dashboard_spx_price_context(ticker)
        if spx_current_price <= 0 and uw_entry and uw_entry.get("spot"):
            spx_current_price = float(uw_entry["spot"])
        return render_template(
            "ticker.html",
            ticker=ticker,
            profile_json=profile_json,
            selected=selected,
            prediction=None,
            has_history=False,
            bootstrap_status=bootstrap_status,
            refresh_message=refresh_message,
            latest_ts=None,
            refresh_minutes=REFRESH_MINUTES,
            current_strike_chart_json=current_strike_chart_json,
            zero_dte_movement_chart_json=None,
            zero_dte_movement=None,
            predicted_strike_chart_json=None,
            uw_fetched_at=uw_entry["fetched_at"] if uw_entry else None,
            uw_profile_json=None,
            spx_price_chart_json=make_spx_price_chart(
                spx_price_points,
                ticker=ticker,
                price_source=spx_price_source,
            ),
            spx_current_price=spx_current_price or None,
            timestamps=[],
            selected_ts=None,
            timeline_chart_json=None,
            cumulative_chart_json=None,
            similar_setups=[],
            flow_overlay=None,
            data_source="Unusual Whales (live)" if uw_entry else "No data",
            spot_distance_to_flip=None,
            ai_insights_json=make_ai_insights_chart(uw_entry.get("analysis")) if uw_entry else None,
            today_regime=build_today_regime_snapshot(selected, None),
            gamma_analysis=build_gamma_analysis_panel(selected),
            alert_feed=[],
            forecast_probs=None,
            confluence_overlay=compute_confluence_overlay(selected, None, None),
            strategy_notes=build_strategy_assistant(selected, None, None),
            data_quality=build_data_quality_panel(selected, history),
            outcome_panel=None,
            term_structure=build_term_structure_panel(selected),
            model_accountability=build_model_accountability_panel(ticker, None, {}),
            scenario=None,
            scenario_pct=0.0,
            replay_index=0,
            prev_ts=None,
            next_ts=None,
            alert_dispatch_status=None,
            system_status=build_system_status(ticker),
        )

    uw_entry = get_uw_data(ticker) if _uw_live_enabled() else None
    uw_spot = uw_entry["spot"] if uw_entry else None
    uw_agg = uw_entry["agg"] if uw_entry else None
    uw_fetched_at = uw_entry["fetched_at"] if uw_entry else None

    requested_ts = request.args.get("ts")
    selected = _select_snapshot(history, requested_ts, ticker=ticker)
    prediction_history = _prediction_history(ticker)
    export_state = summarize_export_state(ticker, forecast_history=prediction_history)
    timestamps = list_timestamps(ticker)
    gamma_timeline = build_gamma_levels_timeline(ticker, history=history)
    replay_index = max(0, timestamps.index(selected["ts"])) if selected.get("ts") in timestamps else max(0, len(timestamps) - 1)
    prev_ts = timestamps[replay_index - 1] if replay_index > 0 else None
    next_ts = timestamps[replay_index + 1] if replay_index + 1 < len(timestamps) else None

    selected_spot = safe_float(selected.get("spot"), 0.0)
    csv_spot = selected_spot if selected_spot > 0 else (uw_spot if uw_spot else None)

    spx_price_points, spx_current_price, spx_price_source = _dashboard_spx_price_context(ticker)
    if spx_current_price <= 0:
        spx_current_price = safe_float(uw_spot, 0.0) or csv_spot or 0.0
    spx_price_chart_json = make_spx_price_chart(
        spx_price_points,
        history=history if not spx_price_points else None,
        ticker=ticker,
        gamma_flip=selected.get("gamma_flip"),
        call_wall=selected.get("call_wall"),
        put_wall=selected.get("put_wall"),
        price_source=spx_price_source,
    )

    profile_spot = spx_current_price if spx_current_price and spx_current_price > 0 else csv_spot
    profile_json = make_gex_profile_chart(
        selected.get("strike"),
        ticker,
        spot=profile_spot,
        title="Gamma Exposure Map (UW CSV)",
        cumulative_series=selected.get("cumulative"),
        gamma_flip=selected.get("gamma_flip"),
        call_wall=selected.get("call_wall"),
        put_wall=selected.get("put_wall"),
    )
    current_profile_series = selected.get("strike")
    previous_same_day = _previous_same_day_snapshot(history, selected)
    zero_dte_movement = _build_0dte_movement_panel(selected, previous_same_day)
    zero_dte_movement_chart_json = make_0dte_movement_chart(
        selected,
        previous_same_day,
        ticker,
        spot=profile_spot,
    )

    uw_profile_json = None
    if uw_agg is not None:
        uw_profile_json = make_gex_profile_chart(
            uw_agg.gex_by_strike,
            ticker,
            spot=uw_spot,
            title="Live Gamma Exposure Map · Unusual Whales",
            cumulative_series=uw_agg.cumulative_gex,
            gamma_flip=uw_entry.get("gamma_flip"),
            call_wall=float(uw_agg.gex_by_strike.idxmax()) if len(uw_agg.gex_by_strike) else None,
            put_wall=float(uw_agg.gex_by_strike.idxmin()) if len(uw_agg.gex_by_strike) else None,
        )

    prediction = None
    flow_overlay = None
    prediction_lookback = prediction_lookback_days(ticker)
    forecast_blocker = None
    try:
        prediction = predict_next_snapshot(prediction_history, lookback_days=prediction_lookback)
        if prediction is None:
            forecast_blocker = forecast_blocker_message(
                prediction_history,
                lookback_days=prediction_lookback,
                export_state=export_state,
            )
    except Exception:
        logger.exception("Prediction failed for %s", ticker)
        forecast_blocker = "Forecast failed with an internal error — check service logs."

    spot_for_flow = csv_spot or selected_spot or 4800.0
    try:
        flow_overlay = load_flow_predictions(FLOW_FEED_PATH, spot=float(spot_for_flow))
        prediction = apply_flow_to_prediction(prediction, flow_overlay)
    except Exception:
        logger.exception("Flow overlay failed for %s", ticker)

    backtest: dict = {}
    try:
        if not _dashboard_skip_backtest():
            backtest = backtest_delta_sign_accuracy(ticker, history=history)
        if prediction and backtest.get("accuracy") is not None:
            prediction = dict(prediction)
            prediction["backtest_sign_accuracy"] = backtest["accuracy"]
            prediction["backtest_n"] = backtest["n"]
            prediction["backtest_mae_delta"] = backtest.get("mae_delta")
            prediction["backtest_baseline_momentum_accuracy"] = backtest.get("baseline_momentum_accuracy")
            prediction["backtest_baseline_accuracy"] = backtest.get("baseline_accuracy")
            prediction["calibrated_confidence"] = calibrate_confidence(
                safe_float(prediction.get("confidence"), 0.0),
                backtest.get("accuracy"),
                backtest.get("n", 0) or 0,
            )
    except Exception:
        logger.exception("Backtest metrics failed for %s", ticker)
        backtest = {}

    current_strike_chart_json = make_positive_strike_chart(
        current_profile_series,
        ticker,
        "Current Position Focus (Positive GEX)",
        spot=profile_spot,
    )
    predicted_strike_chart_json = _build_predicted_strike_chart(
        prediction,
        selected=selected,
        ticker=ticker,
        csv_spot=csv_spot,
    )

    prediction_raw = prediction
    prediction = _prediction_public_view(prediction_raw)

    today_regime = build_today_regime_snapshot(selected, prediction_raw)
    alert_feed = generate_alerts(history, selected, prediction_raw)
    forecast_probs = compute_forecast_probabilities(selected, prediction_raw, history)
    confluence_overlay = compute_confluence_overlay(selected, prediction_raw, flow_overlay)
    strategy_notes = build_strategy_assistant(selected, prediction_raw, confluence_overlay)
    data_quality = build_data_quality_panel(selected, history)
    outcome_panel = build_outcome_panel(history, selected.get("ts"))
    term_structure = build_term_structure_panel(selected, prediction_raw)
    model_accountability = build_model_accountability_panel(ticker, prediction_raw, backtest)

    scenario_pct = safe_float(request.args.get("scenario_pct"), 0.0)
    scenario = simulate_spot_scenario(selected, scenario_pct / 100.0) if scenario_pct else None

    # Auto-dispatch runs only from the background scheduler (see
    # ``_auto_dispatch_alerts``); page renders never trigger webhooks. Manual
    # dispatch over HTTP requires a valid admin token to prevent unauthenticated
    # callers (links, crawlers, CSRF) from firing the webhook.
    alert_dispatch_status = None
    if request.args.get("dispatch_alerts") == "1":
        if _manual_dispatch_authorized(request):
            alert_dispatch_status = maybe_dispatch_alerts(ticker, alert_feed, manual=True)
        else:
            alert_dispatch_status = {
                "ok": False,
                "message": "Manual dispatch requires a valid admin token (set GEX_ADMIN_TOKEN).",
                "dispatched": False,
            }

    system_status = build_system_status(ticker)
    latest_raw = get_latest_ts(ticker)
    csv_source = selected.get("data_source") or "unusual_whales"
    data_source = f"Unusual Whales CSV · {selected['ts_label']} ({csv_source})"
    if uw_agg is not None:
        data_source += " · live API"
    spot_dist = None
    if selected.get("spot") and selected.get("gamma_flip"):
        spot_dist = abs(float(selected["spot"]) - float(selected["gamma_flip"]))

    return render_template(
        "ticker.html",
        ticker=ticker,
        profile_json=profile_json,
        uw_profile_json=uw_profile_json,
        uw_fetched_at=uw_fetched_at,
        spx_price_chart_json=spx_price_chart_json,
        spx_current_price=spx_current_price,
        selected=selected,
        timestamps=timestamps,
        selected_ts=selected.get("ts"),
        prediction=prediction,
        latest_ts=ts_label(latest_raw) if latest_raw else None,
        refresh_minutes=REFRESH_MINUTES,
        has_history=True,
        bootstrap_status=bootstrap_status,
        refresh_message=refresh_message,
        current_strike_chart_json=current_strike_chart_json,
        zero_dte_movement_chart_json=zero_dte_movement_chart_json,
        zero_dte_movement=zero_dte_movement,
        predicted_strike_chart_json=predicted_strike_chart_json,
        timeline_chart_json=make_timeline_chart(gamma_timeline or history, ticker),
        cumulative_chart_json=make_cumulative_gex_chart(
            selected.get("cumulative"), ticker, gamma_flip=selected.get("gamma_flip"),
        ),
        similar_setups=_safe_similar_setups(history),
        flow_overlay=flow_overlay,
        data_source=data_source,
        spot_distance_to_flip=spot_dist,
        ai_insights_json=make_ai_insights_chart(uw_entry.get("analysis")) if uw_entry else None,
        today_regime=today_regime,
        gamma_analysis=build_gamma_analysis_panel(selected, prediction_raw),
        alert_feed=alert_feed,
        forecast_probs=forecast_probs,
        confluence_overlay=confluence_overlay,
        strategy_notes=strategy_notes,
        data_quality=data_quality,
        outcome_panel=outcome_panel,
        term_structure=term_structure,
        model_accountability=model_accountability,
        forecast_blocker=forecast_blocker,
        forecast_snapshot_count=len(prediction_history),
        export_timestamp_count=len(timestamps),
        export_state=export_state,
        scenario=scenario,
        scenario_pct=scenario_pct,
        replay_index=replay_index,
        prev_ts=prev_ts,
        next_ts=next_ts,
        alert_dispatch_status=alert_dispatch_status,
        system_status=system_status,
    )


@APP.post("/ticker/<ticker>/bootstrap")
def bootstrap_ticker_history(ticker):
    ticker = ticker.upper()
    if not is_supported_ticker(ticker):
        return _spx_redirect()
    try:
        ok = refresh_ticker(ticker, force=True)
    except Exception:
        logger.exception("Manual GEX refresh failed for %s", ticker)
        return redirect(url_for("ticker_page", ticker=ticker, bootstrap="error"))

    status = "ok" if ok else "failed"
    return redirect(url_for("ticker_page", ticker=ticker, bootstrap=status))


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
    """SSE stream of live spot ticks from the UW price websocket cache."""
    from gex_core.uw_price_stream import get_uw_price_stream

    ticker = (request.args.get("ticker") or PRIMARY_TICKER).upper()

    def generate():
        stream = get_uw_price_stream()
        last_price: float | None = None
        while True:
            price = stream.get_latest_price(ticker)
            if price <= 0:
                try:
                    from gex_core.market_features import fetch_spx_price

                    price = float(fetch_spx_price(ticker) or 0.0)
                except Exception:
                    price = 0.0
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
            time.sleep(0.5)

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


def _manual_dispatch_authorized(req) -> bool:
    """Authorize a manual webhook dispatch request via the admin token.

    Dispatch is disabled by default: without ``GEX_ADMIN_TOKEN`` set, no HTTP
    caller can trigger the webhook. When configured, the token must be supplied
    via the ``admin_token`` query arg or ``X-Admin-Token`` header.
    """
    token = os.environ.get("GEX_ADMIN_TOKEN")
    if not token:
        return False
    provided = req.args.get("admin_token") or req.headers.get("X-Admin-Token") or ""
    return bool(provided) and secrets.compare_digest(provided, token)


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
    price_stream_fn=lambda: start_uw_price_stream(REFRESH_TICKERS),
)


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    port = int(os.environ.get("PORT", "8080"))
    APP.run(host="0.0.0.0", port=port, debug=debug_mode)
