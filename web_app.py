from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from flask import Flask, abort, render_template, request, send_from_directory
from plotly.utils import PlotlyJSONEncoder

APP = Flask(__name__)
app = APP

EXPORT_DIR = Path("data/exports")
IMG_DIR = Path("img")

CSV_RE = re.compile(
    r"^(?P<ticker>[A-Z0-9]+)_(?P<kind>gex_by_strike|gex_by_expiration|gex_surface|cumulative_gex)_(?P<ts>\d{4}-\d{2}-\d{2}_\d{6})\.csv$"
)


def parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d_%H%M%S")


def ts_label(ts: str) -> str:
    return parse_ts(ts).strftime("%Y-%m-%d %H:%M:%S")


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def find_available_tickers(export_dir: Path):
    export_dir.mkdir(parents=True, exist_ok=True)
    tickers = set()
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
    idx = pd.to_numeric(cumulative.index, errors="coerce")
    valid = ~np.isnan(idx)
    if valid.sum() < 2:
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


def build_history(ticker: str):
    snapshot_files = collect_snapshot_files(ticker)
    snapshots = []
    for ts, files in snapshot_files.items():
        try:
            snapshots.append(load_snapshot_metrics(ts, files))
        except Exception:
            # skip malformed snapshot
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


def make_heatmap(surface_path: Path | None, ticker: str):
    if surface_path is None:
        return None
    df = pd.read_csv(surface_path)
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


def make_surface_scatter(surface_path: Path | None, ticker: str):
    if surface_path is None:
        return None
    df = pd.read_csv(surface_path)
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
def ticker_page(ticker):
    ticker = ticker.upper()
    history = build_history(ticker)
    if not history:
        abort(404)

    requested_ts = request.args.get("ts")
    ts_index = {row["ts"]: row for row in history}
    selected = ts_index.get(requested_ts, history[-1])

    imgs = []
    if IMG_DIR.exists():
        imgs = sorted(IMG_DIR.glob(f"{ticker}_*.*"), key=lambda p: p.stat().st_mtime, reverse=True)

    heatmap_json = make_heatmap(selected.get("surface_path"), ticker)
    scatter3d_json = make_surface_scatter(selected.get("surface_path"), ticker)
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
        strike_csv=selected["strike_path"].name,
        exp_csv=selected["exp_path"].name,
        cum_csv=selected["cum_path"].name,
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


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=8501, debug=True)
