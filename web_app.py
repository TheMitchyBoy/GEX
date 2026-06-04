from __future__ import annotations

import atexit
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, url_for

from gex_core.backtest_metrics import backtest_delta_sign_accuracy
from gex_core.charts import (
    make_ai_insights_chart,
    make_cumulative_gex_chart,
    make_gex_profile_chart,
    make_positive_strike_chart,
    make_prediction_gamma_chart,
    make_timeline_chart,
    safe_float,
)
from gex_core.exports import EXPORT_DIR
from gex_core.history import build_history, get_latest_ts, list_tickers, list_timestamps, ts_label
from gex_core.intelligence import (
    build_data_quality_panel,
    build_outcome_panel,
    build_strategy_assistant,
    build_today_regime_snapshot,
    build_watchlist_rows,
    compute_confluence_overlay,
    compute_forecast_probabilities,
    dispatch_alerts_to_webhook,
    generate_alerts,
    simulate_spot_scenario,
)
from gex_core.predict import (
    apply_flow_to_prediction,
    load_flow_predictions,
    predict_next_snapshot,
    similar_setups,
)
from gex_core.refresh import DEFAULT_REFRESH_MINUTES, DEFAULT_TICKERS, refresh_ticker, refresh_tickers

APP = Flask(__name__)
app = APP
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Unusual Whales live data layer
# ─────────────────────────────────────────────────────────────────────────────

_UW_CACHE: dict[str, dict] = {}          # ticker → {spot, agg, ts, analysis}
_UW_CACHE_TTL = int(
    os.environ.get(
        "GEX_UW_CACHE_TTL_SECONDS",
        str(max(30, int(os.environ.get("GEX_REFRESH_INTERVAL_MINUTES", "1")) * 60)),
    )
)  # keep cache aligned to refresh cadence by default
_UW_API_KEY = os.environ.get("UW_API_KEY")
_UW_ENABLED = bool(_UW_API_KEY)

_uw_lock = threading.Lock()


def _uw_cache_fresh(ticker: str) -> bool:
    entry = _UW_CACHE.get(ticker.upper())
    return bool(entry and (time.monotonic() - entry["ts"]) < _UW_CACHE_TTL)


def refresh_uw_data(ticker: str, force: bool = False) -> dict | None:
    """Fetch live UW GEX and run AI analysis; cache the result."""
    if not _UW_ENABLED:
        return None
    ticker = ticker.upper()
    if not force and _uw_cache_fresh(ticker):
        return _UW_CACHE[ticker]
    try:
        from gex_core.uw_loader import fetch_uw_gex
        from gex_core.ai_analyst import analyze_dealer_gamma
        from gex_core.features import estimate_gamma_flip

        spot, agg = fetch_uw_gex(ticker, api_key=_UW_API_KEY)
        gamma_flip = estimate_gamma_flip(agg.cumulative_gex)
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
            "analysis": analysis,
            "ts": time.monotonic(),
            "fetched_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        }
        with _uw_lock:
            _UW_CACHE[ticker] = entry
        logger.info("UW data refreshed for %s: spot=%.2f, GEX=%.3f Bn$", ticker, spot, agg.total_gex_bn)
        return entry
    except Exception:
        logger.exception("UW data refresh failed for %s", ticker)
        return None


def get_uw_data(ticker: str) -> dict | None:
    """Return cached UW data, refreshing if stale."""
    if not _UW_ENABLED:
        return None
    ticker = ticker.upper()
    with _uw_lock:
        if _uw_cache_fresh(ticker):
            return _UW_CACHE[ticker]
    return refresh_uw_data(ticker)


IMG_DIR = Path("img")
FLOW_FEED_PATH = Path(os.environ.get("GEX_FLOW_FEED", "data/flow_sample.jsonl"))
REFRESH_TICKERS = DEFAULT_TICKERS
REFRESH_MINUTES = DEFAULT_REFRESH_MINUTES


def find_available_tickers(export_dir: Path | None = None):
    return list_tickers(export_dir)


def _uw_live_enabled() -> bool:
    return os.environ.get("GEX_SHOW_UW_LIVE", "1").lower() in {"1", "true", "yes"}


def _select_snapshot(history: list, requested_ts: str | None) -> dict:
    if not history:
        raise ValueError("empty history")
    if requested_ts:
        for row in history:
            if row["ts"] == requested_ts:
                return row
    return history[-1]


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


def _ticker_api_payload(ticker: str, selected_ts: str | None = None) -> dict:
    history = build_history(ticker)
    if not history:
        return {
            "ticker": ticker,
            "has_history": False,
            "summary": None,
            "alerts": [],
            "strategy_notes": [],
            "watchlist": [],
        }
    selected = _select_snapshot(history, selected_ts)
    prediction = predict_next_snapshot(history)
    flow_overlay = None
    spot_for_flow = safe_float(selected.get("spot"), 0.0) or 4800.0
    try:
        flow_overlay = load_flow_predictions(FLOW_FEED_PATH, spot=float(spot_for_flow))
        prediction = apply_flow_to_prediction(prediction, flow_overlay)
    except Exception:
        logger.exception("Flow overlay failed for %s", ticker)
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
        "prediction": _prediction_public_view(prediction),
        "probabilities": probs,
        "confluence": confluence,
        "alerts": alerts,
        "strategy_notes": strategy_notes,
        "data_quality": build_data_quality_panel(selected, history),
        "outcomes": build_outcome_panel(history, selected.get("ts")),
        "watchlist": build_watchlist_rows([ticker]),
    }


@APP.route("/")
def index():
    tickers = find_available_tickers(EXPORT_DIR)
    watchlist_rows = build_watchlist_rows(tickers)
    ticker_cards = []
    for row in watchlist_rows:
        ticker_cards.append(
            {
                **row,
                "total_gex_text": f"{row['total_gex']:.3f}",
                "flip_distance_text": (
                    f"{row['flip_distance_pct'] * 100:.2f}%"
                    if row.get("flip_distance_pct") is not None
                    else "N/A"
                ),
                "confluence_text": f"{row['confluence_score']:.1f}",
                "stability_text": f"{row['regime_stability'] * 100:.0f}%",
            }
        )
    return render_template("index.html", tickers=ticker_cards)


@APP.route("/ticker/<ticker>")
@APP.route("/ticker/<ticker>/")
def ticker_page(ticker):
    ticker = ticker.upper()
    bootstrap_status = request.args.get("bootstrap")
    force_refresh = request.args.get("force_refresh", "").lower() in {"1", "true", "yes"}
    if force_refresh:
        refreshed_csv = False
        refreshed_live = False
        try:
            refreshed_csv = refresh_ticker(ticker, force=True)
        except Exception:
            logger.exception("Force CSV refresh failed for %s", ticker)
        try:
            refreshed_live = refresh_uw_data(ticker, force=True) is not None
        except Exception:
            logger.exception("Force UW refresh failed for %s", ticker)
        bootstrap_status = "ok" if (refreshed_csv or refreshed_live) else "failed"

    history = build_history(ticker)

    if not history:
        selected = {
            "ts_label": "No snapshot history available yet",
            "regime": "N/A",
            "total_gex": 0.0,
            "call_wall": None,
            "put_wall": None,
            "gamma_flip": None,
            "spot": None,
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
                    "call_wall": float(uw_agg.gex_by_strike.idxmax()) if len(uw_agg.gex_by_strike) else None,
                    "put_wall": float(uw_agg.gex_by_strike.idxmin()) if len(uw_agg.gex_by_strike) else None,
                    "gamma_flip": uw_entry.get("gamma_flip"),
                    "spot": uw_entry.get("spot"),
                }
            )
            profile_json = make_gex_profile_chart(
                uw_agg.gex_by_strike,
                ticker,
                spot=uw_entry.get("spot"),
                title="Market Maker Position (UW Live)",
            )
            current_strike_chart_json = make_positive_strike_chart(
                uw_agg.gex_by_strike,
                ticker,
                "Current Position Focus (Positive GEX)",
            )

        return render_template(
            "ticker.html",
            ticker=ticker,
            profile_json=profile_json,
            selected=selected,
            prediction=None,
            has_history=False,
            bootstrap_status=bootstrap_status,
            latest_ts=None,
            refresh_minutes=REFRESH_MINUTES,
            current_strike_chart_json=current_strike_chart_json,
            predicted_strike_chart_json=None,
            uw_fetched_at=uw_entry["fetched_at"] if uw_entry else None,
            uw_profile_json=None,
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
            alert_feed=[],
            forecast_probs=None,
            confluence_overlay=compute_confluence_overlay(selected, None, None),
            strategy_notes=build_strategy_assistant(selected, None, None),
            data_quality=build_data_quality_panel(selected, history),
            outcome_panel=None,
            scenario=None,
            scenario_pct=0.0,
            replay_index=0,
            prev_ts=None,
            next_ts=None,
            alert_dispatch_status=None,
        )

    uw_entry = get_uw_data(ticker) if _uw_live_enabled() else None
    uw_spot = uw_entry["spot"] if uw_entry else None
    uw_agg = uw_entry["agg"] if uw_entry else None
    uw_fetched_at = uw_entry["fetched_at"] if uw_entry else None

    requested_ts = request.args.get("ts")
    selected = _select_snapshot(history, requested_ts)
    timestamps = list_timestamps(ticker)
    replay_index = max(0, timestamps.index(selected["ts"])) if selected.get("ts") in timestamps else max(0, len(timestamps) - 1)
    prev_ts = timestamps[replay_index - 1] if replay_index > 0 else None
    next_ts = timestamps[replay_index + 1] if replay_index + 1 < len(timestamps) else None

    selected_spot = safe_float(selected.get("spot"), 0.0)
    csv_spot = selected_spot if selected_spot > 0 else (uw_spot if uw_spot else None)
    profile_json = make_gex_profile_chart(
        selected.get("strike"),
        ticker,
        spot=csv_spot,
        title="Market Maker Position (UW CSV)",
    )
    current_profile_series = selected.get("strike")

    uw_profile_json = None
    if uw_agg is not None:
        uw_profile_json = make_gex_profile_chart(
            uw_agg.gex_by_strike,
            ticker,
            spot=uw_spot,
            title="Live · Unusual Whales (latest API)",
        )

    prediction = None
    flow_overlay = None
    try:
        prediction = predict_next_snapshot(history)
    except Exception:
        logger.exception("Prediction failed for %s", ticker)

    spot_for_flow = csv_spot or selected_spot or 4800.0
    try:
        flow_overlay = load_flow_predictions(FLOW_FEED_PATH, spot=float(spot_for_flow))
        prediction = apply_flow_to_prediction(prediction, flow_overlay)
    except Exception:
        logger.exception("Flow overlay failed for %s", ticker)

    try:
        backtest = backtest_delta_sign_accuracy(ticker)
        if prediction and backtest.get("accuracy") is not None:
            prediction = dict(prediction)
            prediction["backtest_sign_accuracy"] = backtest["accuracy"]
            prediction["backtest_n"] = backtest["n"]
    except Exception:
        logger.exception("Backtest metrics failed for %s", ticker)

    current_strike_chart_json = make_positive_strike_chart(
        current_profile_series,
        ticker,
        "Current Position Focus (Positive GEX)",
    )
    predicted_strike_chart_json = None
    if prediction:
        knn_strike = prediction.get("knn_strike") or prediction.get("predicted_strike")
        flow_strike = prediction.get("flow_strike")
        combined_strike = prediction.get("predicted_strike")
        if knn_strike is not None or combined_strike is not None:
            predicted_strike_chart_json = make_prediction_gamma_chart(
                knn_strike=knn_strike,
                combined_strike=combined_strike,
                flow_strike=flow_strike,
                ticker=ticker,
                spot=csv_spot,
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

    scenario_pct = safe_float(request.args.get("scenario_pct"), 0.0)
    scenario = simulate_spot_scenario(selected, scenario_pct / 100.0) if scenario_pct else None

    alert_dispatch_status = None
    if request.args.get("dispatch_alerts") == "1":
        dispatched, message = dispatch_alerts_to_webhook(ticker, alert_feed)
        alert_dispatch_status = {"ok": dispatched, "message": message}

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
        selected=selected,
        timestamps=timestamps,
        selected_ts=selected.get("ts"),
        prediction=prediction,
        latest_ts=ts_label(latest_raw) if latest_raw else None,
        refresh_minutes=REFRESH_MINUTES,
        has_history=True,
        bootstrap_status=bootstrap_status,
        current_strike_chart_json=current_strike_chart_json,
        predicted_strike_chart_json=predicted_strike_chart_json,
        timeline_chart_json=make_timeline_chart(history, ticker),
        cumulative_chart_json=make_cumulative_gex_chart(
            selected.get("cumulative"), ticker, gamma_flip=selected.get("gamma_flip"),
        ),
        similar_setups=_safe_similar_setups(history),
        flow_overlay=flow_overlay,
        data_source=data_source,
        spot_distance_to_flip=spot_dist,
        ai_insights_json=make_ai_insights_chart(uw_entry.get("analysis")) if uw_entry else None,
        today_regime=today_regime,
        alert_feed=alert_feed,
        forecast_probs=forecast_probs,
        confluence_overlay=confluence_overlay,
        strategy_notes=strategy_notes,
        data_quality=data_quality,
        outcome_panel=outcome_panel,
        scenario=scenario,
        scenario_pct=scenario_pct,
        replay_index=replay_index,
        prev_ts=prev_ts,
        next_ts=next_ts,
        alert_dispatch_status=alert_dispatch_status,
    )


@APP.post("/ticker/<ticker>/bootstrap")
def bootstrap_ticker_history(ticker):
    ticker = ticker.upper()
    try:
        ok = refresh_ticker(ticker, force=True)
    except Exception:
        logger.exception("Manual GEX refresh failed for %s", ticker)
        return redirect(url_for("ticker_page", ticker=ticker, bootstrap="error"))

    status = "ok" if ok else "failed"
    return redirect(url_for("ticker_page", ticker=ticker, bootstrap=status))


@APP.get("/api/latest-summary")
def api_latest_summary():
    ticker = request.args.get("ticker", "SPX").upper()
    payload = _ticker_api_payload(ticker, request.args.get("ts"))
    return jsonify(payload)


@APP.get("/api/signals")
def api_signals():
    ticker = request.args.get("ticker", "SPX").upper()
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
    ticker = ticker.upper()
    payload = _ticker_api_payload(ticker, request.args.get("ts"))
    summary = payload.get("summary") or {}
    confluence = payload.get("confluence") or {"score": 0.0, "label": "low"}
    return render_template(
        "widget.html",
        ticker=ticker,
        summary=summary,
        confluence=confluence,
        updated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
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


def _scheduled_refresh():
    try:
        refresh_tickers(REFRESH_TICKERS, force=True)
    except Exception:
        logger.exception("Scheduled GEX refresh failed")


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


start_background_refresh()


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=8501, debug=True)
