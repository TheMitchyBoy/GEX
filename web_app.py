from __future__ import annotations

import atexit
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, abort, redirect, render_template, request, send_from_directory, url_for

from gex_core.backtest_metrics import backtest_delta_sign_accuracy
from gex_core.charts import (
    make_ai_insights_chart,
    make_cumulative_gex_chart,
    make_gex_profile_chart,
    make_positive_strike_chart,
    make_timeline_chart,
    safe_float,
)
from gex_core.exports import EXPORT_DIR
from gex_core.history import build_history, get_latest_ts, list_tickers, ts_label
from gex_core.predict import predict_next_snapshot, similar_setups
from gex_core.refresh import DEFAULT_REFRESH_MINUTES, DEFAULT_TICKERS, refresh_ticker, refresh_tickers

APP = Flask(__name__)
app = APP
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Unusual Whales live data layer
# ─────────────────────────────────────────────────────────────────────────────

_UW_CACHE: dict[str, dict] = {}          # ticker → {spot, agg, ts, analysis}
_UW_CACHE_TTL = 600                       # seconds (10 minutes)
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
REFRESH_TICKERS = DEFAULT_TICKERS
REFRESH_MINUTES = DEFAULT_REFRESH_MINUTES


def find_available_tickers(export_dir: Path | None = None):
    return list_tickers(export_dir)


def _safe_similar_setups(history: list) -> list:
    try:
        return similar_setups(history, top_n=5)
    except Exception:
        logger.exception("Similar setups lookup failed")
        return []


@APP.route("/")
def index():
    tickers = find_available_tickers(EXPORT_DIR)
    ticker_cards = []
    for ticker in tickers:
        history = build_history(ticker)
        latest = history[-1] if history else None
        ticker_cards.append(
            {
                "ticker": ticker,
                "history_count": len(history),
                "latest_ts": latest["ts_label"] if latest else "N/A",
                "total_gex": f"{latest['total_gex']:.3f}" if latest else "N/A",
                "regime": latest["regime"] if latest else "N/A",
            }
        )
    return render_template("index.html", tickers=ticker_cards)


@APP.route("/ticker/<ticker>")
@APP.route("/ticker/<ticker>/")
def ticker_page(ticker):
    ticker = ticker.upper()
    history = build_history(ticker)
    bootstrap_status = request.args.get("bootstrap")

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
            timeline_chart_json=None,
            cumulative_chart_json=None,
            similar_setups=[],
            data_source="Unusual Whales (live)" if uw_entry else "No data",
            spot_distance_to_flip=None,
            ai_insights_json=make_ai_insights_chart(uw_entry.get("analysis")) if uw_entry else None,
        )

    uw_entry = get_uw_data(ticker)
    if uw_entry:
        uw_spot = uw_entry["spot"]
        uw_agg = uw_entry["agg"]
        uw_fetched_at = uw_entry["fetched_at"]
    else:
        uw_spot = None
        uw_agg = None
        uw_fetched_at = None

    selected = history[-1]

    if uw_agg is not None:
        profile_json = make_gex_profile_chart(
            uw_agg.gex_by_strike,
            ticker,
            spot=uw_spot,
            title="Market Maker Position (UW Live)",
        )
        current_profile_series = uw_agg.gex_by_strike
    else:
        profile_json = make_gex_profile_chart(
            selected.get("strike"),
            ticker,
            spot=safe_float(selected.get("spot"), None) or None,
            title="Market Maker Position",
        )
        current_profile_series = selected.get("strike")

    prediction = None
    try:
        prediction = predict_next_snapshot(history)
    except Exception:
        logger.exception("Prediction failed for %s", ticker)

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
    if prediction and prediction.get("predicted_strike") is not None:
        predicted_strike_chart_json = make_positive_strike_chart(
            prediction.get("predicted_strike"),
            ticker,
            "Predicted Positive GEX by Strike",
        )
        prediction = {k: v for k, v in prediction.items() if k != "predicted_strike"}

    latest_raw = get_latest_ts(ticker)
    data_source = "Unusual Whales (live)" if uw_agg is not None else selected.get("data_source") or "CSV history"
    spot_dist = None
    if selected.get("spot") and selected.get("gamma_flip"):
        spot_dist = abs(float(selected["spot"]) - float(selected["gamma_flip"]))

    return render_template(
        "ticker.html",
        ticker=ticker,
        profile_json=profile_json,
        uw_fetched_at=uw_fetched_at,
        selected=selected,
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
        data_source=data_source,
        spot_distance_to_flip=spot_dist,
        ai_insights_json=make_ai_insights_chart(uw_entry.get("analysis")) if uw_entry else None,
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
