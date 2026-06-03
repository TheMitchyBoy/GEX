from __future__ import annotations

import atexit
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, abort, redirect, render_template, request, send_from_directory, url_for
from plotly.utils import PlotlyJSONEncoder

from gex_db.refresh import DEFAULT_REFRESH_MINUTES, DEFAULT_TICKERS, refresh_tickers
from gex_db.store import (
    get_latest_ts,
    get_snapshot,
    import_csv_exports,
    init_db,
    list_snapshots,
    list_tickers,
    list_timestamps,
    parse_ts,
)

APP = Flask(__name__)
app = APP
logger = logging.getLogger(__name__)

EXPORT_DIR = Path("data/exports")
IMG_DIR = Path("img")
REFRESH_TICKERS = DEFAULT_TICKERS
REFRESH_MINUTES = DEFAULT_REFRESH_MINUTES

CSV_RE = re.compile(
    r"^(?P<ticker>[A-Z0-9]+)_(?P<kind>gex_by_strike|gex_by_expiration|gex_surface|cumulative_gex)_(?P<ts>\d{4}-\d{2}-\d{2}_\d{6})\.csv$"
)


def ts_label(ts: str) -> str:
    return parse_ts(ts).strftime("%Y-%m-%d %H:%M:%S")


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def find_available_tickers(export_dir: Path | None = None):
    export_dir = export_dir or EXPORT_DIR
    export_dir.mkdir(parents=True, exist_ok=True)
    tickers = set(list_tickers())
    for file in export_dir.glob("*.csv"):
        match = CSV_RE.match(file.name)
        if match:
            tickers.add(match.group("ticker"))
    return sorted(tickers)


def collect_snapshot_files(ticker: str):
    snapshots = {}
    for file in EXPORT_DIR.glob(f"{ticker}_*.csv"):
        match = CSV_RE.match(file.name)
        if not match:
            continue
        ts = match.group("ts")
        kind = match.group("kind")
        snapshots.setdefault(ts, {})[kind] = file

    # only keep snapshots with minimum data needed
    filtered = {
        ts: files
        for ts, files in snapshots.items()
        if "gex_by_strike" in files and "gex_by_expiration" in files and "cumulative_gex" in files
    }
    return dict(sorted(filtered.items(), key=lambda item: item[0]))


def estimate_gamma_flip(cumulative: pd.Series):
    if cumulative.empty:
        return None

    values = cumulative.astype(float).values
    idx = pd.to_numeric(cumulative.index, errors="coerce").to_numpy(dtype=float)
    valid = ~np.isnan(idx)
    if int(np.sum(valid)) < 2:
        return None

    x = idx[valid]
    y = values[valid]
    signs = np.sign(y)

    for i in range(len(signs) - 1):
        if signs[i] == signs[i + 1]:
            continue
        x0, x1 = float(x[i]), float(x[i + 1])
        y0, y1 = float(y[i]), float(y[i + 1])
        if y1 == y0:
            return x0
        return x0 - y0 * (x1 - x0) / (y1 - y0)
    return None


def load_snapshot_metrics_from_row(row: dict):
    strike = row["strike"]
    ts = row["ts"]
    expiration = row["expiration"]
    cumulative = row["cumulative"]
    surface_df = row["surface_df"]

    exp_vals = pd.to_numeric(expiration, errors="coerce").fillna(0.0)
    total_gex = float(strike.sum())
    pos_gex = float(strike[strike > 0].sum())
    neg_gex = float(strike[strike < 0].sum())
    gex_std = float(strike.std()) if len(strike) > 1 else 0.0

    call_wall = float(strike.idxmax()) if len(strike) else None
    put_wall = float(strike.idxmin()) if len(strike) else None
    gamma_flip = estimate_gamma_flip(cumulative)

    near_term = float(exp_vals.head(3).sum()) if len(exp_vals) else 0.0
    term_total = float(exp_vals.sum()) if len(exp_vals) else 0.0
    near_term_ratio = near_term / term_total if term_total else 0.0

    surface_peak = 0.0
    if not surface_df.empty and "GEX" in surface_df.columns:
        surface_peak = float(pd.to_numeric(surface_df["GEX"], errors="coerce").abs().max())

    return {
        "ts": ts,
        "ts_label": ts_label(ts),
        "strike": strike,
        "exp_df": expiration.reset_index(),
        "cumulative": cumulative,
        "surface_df": surface_df,
        "surface_path": None,
        "strike_path": None,
        "exp_path": None,
        "cum_path": None,
        "total_gex": total_gex,
        "pos_gex": pos_gex,
        "neg_gex": neg_gex,
        "gex_std": gex_std,
        "abs_mean": float(strike.abs().mean()) if len(strike) else 0.0,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "gamma_flip": gamma_flip,
        "near_term_ratio": near_term_ratio,
        "surface_peak": surface_peak,
        "regime": "LONG gamma" if total_gex >= 0 else "SHORT gamma",
    }


def load_snapshot_metrics(ts: str, files: dict):
    strike_df = pd.read_csv(files["gex_by_strike"])
    exp_df = pd.read_csv(files["gex_by_expiration"])
    cum_df = pd.read_csv(files["cumulative_gex"])

    strike = pd.Series(
        pd.to_numeric(strike_df.iloc[:, 1], errors="coerce").fillna(0.0).values,
        index=pd.to_numeric(strike_df.iloc[:, 0], errors="coerce").fillna(0.0).values,
    )
    exp_vals = pd.to_numeric(exp_df.iloc[:, 1], errors="coerce").fillna(0.0)
    cumulative = pd.Series(
        pd.to_numeric(cum_df.iloc[:, 1], errors="coerce").fillna(0.0).values,
        index=pd.to_numeric(cum_df.iloc[:, 0], errors="coerce").fillna(0.0).values,
    )

    total_gex = float(strike.sum())
    pos_gex = float(strike[strike > 0].sum())
    neg_gex = float(strike[strike < 0].sum())
    gex_std = float(strike.std()) if len(strike) > 1 else 0.0
    abs_mean = float(strike.abs().mean()) if len(strike) else 0.0

    call_wall = float(strike.idxmax()) if len(strike) else None
    put_wall = float(strike.idxmin()) if len(strike) else None
    gamma_flip = estimate_gamma_flip(cumulative)

    near_term = float(exp_vals.head(3).sum()) if len(exp_vals) else 0.0
    term_total = float(exp_vals.sum()) if len(exp_vals) else 0.0
    near_term_ratio = near_term / term_total if term_total else 0.0

    surface_peak = 0.0
    surface_df = pd.DataFrame()
    if "gex_surface" in files:
        surface_df = pd.read_csv(files["gex_surface"])
        if "GEX" in surface_df.columns and not surface_df.empty:
            surface_peak = float(pd.to_numeric(surface_df["GEX"], errors="coerce").abs().max())

    return {
        "ts": ts,
        "ts_label": ts_label(ts),
        "strike": strike,
        "exp_df": exp_df,
        "cumulative": cumulative,
        "surface_df": surface_df,
        "surface_path": files.get("gex_surface"),
        "strike_path": files["gex_by_strike"],
        "exp_path": files["gex_by_expiration"],
        "cum_path": files["cumulative_gex"],
        "total_gex": total_gex,
        "pos_gex": pos_gex,
        "neg_gex": neg_gex,
        "gex_std": gex_std,
        "abs_mean": abs_mean,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "gamma_flip": gamma_flip,
        "near_term_ratio": near_term_ratio,
        "surface_peak": surface_peak,
        "regime": "LONG gamma" if total_gex >= 0 else "SHORT gamma",
    }


def load_snapshot_metrics_from_db(ticker: str, ts: str):
    row = get_snapshot(ticker, ts)
    if row is None:
        return None
    return load_snapshot_metrics_from_row(row)


def build_history(ticker: str):
    ticker = ticker.upper()
    snapshots = []

    db_rows = list_snapshots(ticker)
    if db_rows:
        for row in db_rows:
            try:
                metrics = load_snapshot_metrics_from_row(row)
                if metrics:
                    snapshots.append(metrics)
            except Exception:
                continue
    else:
        snapshot_files = collect_snapshot_files(ticker)
        for ts, files in snapshot_files.items():
            try:
                snapshots.append(load_snapshot_metrics(ts, files))
            except Exception:
                continue

    snapshots.sort(key=lambda row: row["ts"])
    return snapshots


def make_timeline_chart(history, ticker):
    if not history:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[row["ts_label"] for row in history],
            y=[row["total_gex"] for row in history],
            mode="lines+markers",
            line=dict(color="#4dabf7", width=2),
            marker=dict(size=7),
            name="Total GEX",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#adb5bd")
    fig.update_layout(
        title=f"{ticker} Total GEX Timeline",
        template="plotly_dark",
        height=320,
        margin=dict(l=20, r=20, t=45, b=20),
        xaxis_title="Snapshot",
        yaxis_title="GEX (Bn$ / %)",
    )
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def _load_surface_df(surface_path: Path | None = None, surface_df: pd.DataFrame | None = None):
    if surface_df is not None and not surface_df.empty:
        return surface_df
    if surface_path is None:
        return None
    return pd.read_csv(surface_path)


def make_heatmap(surface_path: Path | None = None, ticker: str = "", surface_df: pd.DataFrame | None = None):
    df = _load_surface_df(surface_path, surface_df)
    if df is None:
        return None
    if df.empty or not {"expiration", "strike", "GEX"}.issubset(df.columns):
        return None
    df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce")
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["GEX"] = pd.to_numeric(df["GEX"], errors="coerce")
    df = df.dropna(subset=["expiration", "strike", "GEX"])
    if df.empty:
        return None

    pivot = df.pivot_table(index="expiration", columns="strike", values="GEX", aggfunc="sum").fillna(0).sort_index()
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[float(c) for c in pivot.columns],
            y=[d.strftime("%Y-%m-%d") for d in pd.to_datetime(pivot.index)],
            colorscale="RdYlBu_r",
        )
    )
    fig.update_layout(
        title=f"{ticker} GEX Surface (selected snapshot)",
        template="plotly_dark",
        height=500,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def make_surface_scatter(
    surface_path: Path | None = None,
    ticker: str = "",
    surface_df: pd.DataFrame | None = None,
):
    df = _load_surface_df(surface_path, surface_df)
    if df is None:
        return None
    if df.empty or not {"expiration", "strike", "GEX"}.issubset(df.columns):
        return None
    df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce")
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["GEX"] = pd.to_numeric(df["GEX"], errors="coerce")
    df = df.dropna(subset=["expiration", "strike", "GEX"])
    if df.empty:
        return None

    fig = go.Figure(
        data=go.Scatter3d(
            x=df["strike"],
            y=df["expiration"].dt.strftime("%Y-%m-%d"),
            z=df["GEX"],
            mode="markers",
            marker=dict(size=4, color=df["GEX"], colorscale="Viridis", showscale=True),
        )
    )
    fig.update_layout(
        title=f"{ticker} 3D Surface (selected snapshot)",
        template="plotly_dark",
        height=500,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def prepare_training_rows(history):
    rows = []
    for i in range(len(history) - 1):
        cur = history[i]
        nxt = history[i + 1]
        rows.append(
            {
                "ts": cur["ts"],
                "features": np.array(
                    [
                        cur["total_gex"],
                        cur["pos_gex"],
                        cur["neg_gex"],
                        cur["gex_std"],
                        cur["near_term_ratio"],
                        cur["surface_peak"],
                        safe_float(cur["call_wall"], 0.0),
                        safe_float(cur["put_wall"], 0.0),
                        safe_float(cur["gamma_flip"], 0.0),
                    ],
                    dtype=float,
                ),
                "target_total_gex": nxt["total_gex"],
                "target_flip": nxt["gamma_flip"],
                "target_near_term_ratio": nxt["near_term_ratio"],
                "next_ts": nxt["ts"],
            }
        )
    return rows


def predict_next_snapshot(history):
    if len(history) < 4:
        return None

    current = history[-1]
    train = prepare_training_rows(history)
    if len(train) < 3:
        return None

    x_train = np.vstack([row["features"] for row in train])
    x_now = np.array(
        [
            current["total_gex"],
            current["pos_gex"],
            current["neg_gex"],
            current["gex_std"],
            current["near_term_ratio"],
            current["surface_peak"],
            safe_float(current["call_wall"], 0.0),
            safe_float(current["put_wall"], 0.0),
            safe_float(current["gamma_flip"], 0.0),
        ],
        dtype=float,
    )

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)

    z_train = (x_train - mean) / std
    z_now = (x_now - mean) / std

    distances = np.linalg.norm(z_train - z_now, axis=1)
    k = min(4, len(distances))
    nn_idx = np.argsort(distances)[:k]
    nn_dist = distances[nn_idx]
    weights = 1.0 / (nn_dist + 1e-6)
    weights = weights / weights.sum()

    pred_total = float(np.sum([weights[j] * train[i]["target_total_gex"] for j, i in enumerate(nn_idx)]))

    flip_targets = np.array([safe_float(train[i]["target_flip"], safe_float(current["gamma_flip"], 0.0)) for i in nn_idx], dtype=float)
    pred_flip = float(np.sum(weights * flip_targets))

    pred_ratio = float(np.sum([weights[j] * train[i]["target_near_term_ratio"] for j, i in enumerate(nn_idx)]))

    avg_dist = float(nn_dist.mean())
    confidence = max(0.0, min(1.0, 1.0 / (1.0 + avg_dist)))

    neighbors = []
    for rank, (i, d) in enumerate(zip(nn_idx, nn_dist), start=1):
        src = next(row for row in history if row["ts"] == train[i]["ts"])
        neighbors.append(
            {
                "rank": rank,
                "snapshot": src["ts_label"],
                "distance": float(d),
                "next_snapshot": ts_label(train[i]["next_ts"]),
                "next_total_gex": float(train[i]["target_total_gex"]),
            }
        )

    return {
        "predicted_total_gex": pred_total,
        "predicted_regime": "LONG gamma" if pred_total >= 0 else "SHORT gamma",
        "predicted_flip": pred_flip,
        "predicted_near_term_ratio": pred_ratio,
        "confidence": confidence,
        "neighbors": neighbors,
    }


def similar_setups(history, top_n=5):
    if len(history) < 3:
        return []

    current = history[-1]
    rows = []
    for i in range(len(history) - 1):
        row = history[i]
        feat = np.array(
            [
                row["total_gex"],
                row["gex_std"],
                row["near_term_ratio"],
                safe_float(row["gamma_flip"], 0.0),
                safe_float(row["call_wall"], 0.0),
                safe_float(row["put_wall"], 0.0),
            ],
            dtype=float,
        )
        rows.append((i, row, feat))

    current_feat = np.array(
        [
            current["total_gex"],
            current["gex_std"],
            current["near_term_ratio"],
            safe_float(current["gamma_flip"], 0.0),
            safe_float(current["call_wall"], 0.0),
            safe_float(current["put_wall"], 0.0),
        ],
        dtype=float,
    )

    matrix = np.vstack([feat for _, _, feat in rows])
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std == 0, 1.0, std)

    z_matrix = (matrix - mean) / std
    z_current = (current_feat - mean) / std
    distances = np.linalg.norm(z_matrix - z_current, axis=1)

    idx_sorted = np.argsort(distances)[: min(top_n, len(distances))]
    results = []
    for idx in idx_sorted:
        hist_idx, snap, _ = rows[idx]
        next_snap = history[hist_idx + 1] if hist_idx + 1 < len(history) else None
        sim_score = 1.0 / (1.0 + float(distances[idx]))
        results.append(
            {
                "snapshot": snap["ts_label"],
                "distance": float(distances[idx]),
                "similarity": sim_score,
                "regime": snap["regime"],
                "total_gex": snap["total_gex"],
                "next_snapshot": next_snap["ts_label"] if next_snap else None,
                "next_total_gex": next_snap["total_gex"] if next_snap else None,
                "next_regime": next_snap["regime"] if next_snap else None,
                "ts": snap["ts"],
            }
        )
    return results


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

    imgs = []
    if IMG_DIR.exists():
        imgs = sorted(IMG_DIR.glob(f"{ticker}_*.*"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not history:
        selected = {
            "ts_label": "No snapshot history available yet",
            "regime": "N/A",
            "total_gex": 0.0,
            "call_wall": None,
            "put_wall": None,
            "gamma_flip": None,
            "near_term_ratio": 0.0,
        }
        return render_template(
            "ticker.html",
            ticker=ticker,
            imgs=imgs,
            heatmap_json=None,
            scatter3d_json=None,
            timeline_json=None,
            timeline_options=[],
            selected=selected,
            prediction=None,
            similar_setups=[],
            strike_csv=None,
            exp_csv=None,
            cum_csv=None,
            has_history=False,
            bootstrap_status=bootstrap_status,
            latest_ts=None,
            refresh_minutes=REFRESH_MINUTES,
        )

    requested_ts = request.args.get("ts")
    ts_index = {row["ts"]: row for row in history}
    selected = ts_index.get(requested_ts, history[-1])

    heatmap_json = make_heatmap(
        selected.get("surface_path"),
        ticker,
        surface_df=selected.get("surface_df"),
    )
    scatter3d_json = make_surface_scatter(
        selected.get("surface_path"),
        ticker,
        surface_df=selected.get("surface_df"),
    )
    timeline_json = make_timeline_chart(history, ticker)

    prediction = predict_next_snapshot(history)
    similar = similar_setups(history, top_n=6)

    timeline_options = [
        {
            "ts": row["ts"],
            "label": row["ts_label"],
            "is_selected": row["ts"] == selected["ts"],
        }
        for row in history
    ]

    return render_template(
        "ticker.html",
        ticker=ticker,
        imgs=imgs,
        heatmap_json=heatmap_json,
        scatter3d_json=scatter3d_json,
        timeline_json=timeline_json,
        timeline_options=timeline_options,
        selected=selected,
        prediction=prediction,
        similar_setups=similar,
        strike_csv=selected["strike_path"].name if selected.get("strike_path") else None,
        exp_csv=selected["exp_path"].name if selected.get("exp_path") else None,
        cum_csv=selected["cum_path"].name if selected.get("cum_path") else None,
        latest_ts=ts_label(get_latest_ts(ticker)) if get_latest_ts(ticker) else None,
        refresh_minutes=REFRESH_MINUTES,
        has_history=True,
        bootstrap_status=bootstrap_status,
    )


@APP.post("/ticker/<ticker>/bootstrap")
def bootstrap_ticker_history(ticker):
    ticker = ticker.upper()
    try:
        from gex_db.refresh import refresh_ticker

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
    init_db()
    imported = import_csv_exports(EXPORT_DIR)
    if imported:
        logger.info("Imported %s historical CSV snapshots into database", imported)

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

    # Seed empty databases without blocking the gunicorn worker boot sequence.
    if any(not list_timestamps(ticker) for ticker in REFRESH_TICKERS):
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
