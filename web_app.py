from __future__ import annotations

import atexit
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, abort, redirect, render_template, request, send_from_directory, url_for
from plotly.utils import PlotlyJSONEncoder

from gex_core.features import enrich_snapshot_metrics, estimate_gamma_flip
from gex_core.predict import load_flow_predictions, predict_next_snapshot, similar_setups
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
FLOW_FEED_PATH = Path(os.environ.get("GEX_FLOW_FEED", "data/flow_sample.jsonl"))
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


def estimate_gamma_flip_local(cumulative: pd.Series):
    return estimate_gamma_flip(cumulative)


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
    gamma_flip = estimate_gamma_flip_local(cumulative)

    near_term = float(exp_vals.head(3).sum()) if len(exp_vals) else 0.0
    term_total = float(exp_vals.sum()) if len(exp_vals) else 0.0
    near_term_ratio = near_term / term_total if term_total else 0.0

    surface_peak = 0.0
    if not surface_df.empty and "GEX" in surface_df.columns:
        surface_peak = float(pd.to_numeric(surface_df["GEX"], errors="coerce").abs().max())

    metrics = {
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
    return enrich_snapshot_metrics(metrics)


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
    gamma_flip = estimate_gamma_flip_local(cumulative)

    near_term = float(exp_vals.head(3).sum()) if len(exp_vals) else 0.0
    term_total = float(exp_vals.sum()) if len(exp_vals) else 0.0
    near_term_ratio = near_term / term_total if term_total else 0.0

    surface_peak = 0.0
    surface_df = pd.DataFrame()
    if "gex_surface" in files:
        surface_df = pd.read_csv(files["gex_surface"])
        if "GEX" in surface_df.columns and not surface_df.empty:
            surface_peak = float(pd.to_numeric(surface_df["GEX"], errors="coerce").abs().max())

    metrics = {
        "ts": ts,
        "ts_label": ts_label(ts),
        "ticker": None,
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
    return enrich_snapshot_metrics(metrics)


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
                    metrics["ticker"] = ticker
                    snapshots.append(metrics)
            except Exception:
                continue
    else:
        snapshot_files = collect_snapshot_files(ticker)
        for ts, files in snapshot_files.items():
            try:
                metrics = load_snapshot_metrics(ts, files)
                metrics["ticker"] = ticker
                snapshots.append(metrics)
            except Exception:
                continue

    snapshots.sort(key=lambda row: row["ts"])
    return snapshots


def make_timeline_chart(history, ticker):
    if not history:
        return None
    labels = [row["ts_label"] for row in history]
    totals = [safe_float(row.get("total_gex"), 0.0) for row in history]
    pos = [safe_float(row.get("pos_gex"), 0.0) for row in history]
    neg = [safe_float(row.get("neg_gex"), 0.0) for row in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=pos,
        fill="tozeroy",
        fillcolor="rgba(0,217,126,0.10)",
        line=dict(color=_GREEN, width=1),
        name="Positive GEX",
        hovertemplate="%{y:.3f} Bn$<extra>+GEX</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=neg,
        fill="tozeroy",
        fillcolor="rgba(255,71,87,0.10)",
        line=dict(color=_RED, width=1),
        name="Negative GEX",
        hovertemplate="%{y:.3f} Bn$<extra>-GEX</extra>",
    ))
    marker_colors = [_GREEN if t >= 0 else _RED for t in totals]
    fig.add_trace(go.Scatter(
        x=labels,
        y=totals,
        mode="lines+markers",
        line=dict(color=_BLUE, width=2.5),
        marker=dict(size=7, color=marker_colors, line=dict(color=_CHART_BG, width=1)),
        name="Total GEX",
        hovertemplate="%{y:.3f} Bn$<extra>Net GEX</extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.25)", line_width=1)
    _apply_base(
        fig,
        title=f"{ticker} · GEX Timeline",
        height=340,
        margin=dict(l=20, r=20, t=45, b=20),
        xaxis_title="Snapshot",
        yaxis_title="GEX (Bn$ / %)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def make_cumulative_gex_chart(cumulative, ticker, gamma_flip=None):
    """Area chart of cumulative GEX by strike with optional flip annotation."""
    if cumulative is None or (hasattr(cumulative, "empty") and cumulative.empty):
        return None
    try:
        x = [float(v) for v in cumulative.index]
        y = [float(v) for v in cumulative]
    except Exception:
        return None
    if not x:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=[max(v, 0) for v in y],
        fill="tozeroy",
        fillcolor="rgba(0,217,126,0.12)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=[min(v, 0) for v in y],
        fill="tozeroy",
        fillcolor="rgba(255,71,87,0.12)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        line=dict(color=_BLUE, width=2.5),
        name="Cumulative GEX",
        hovertemplate="Strike %{x:.0f}<br>%{y:.3f} Bn$<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.25)", line_width=1)
    if gamma_flip is not None:
        try:
            fig.add_vline(
                x=float(gamma_flip),
                line_dash="dot",
                line_color=_AMBER,
                line_width=2,
                annotation_text=f"Flip ~{float(gamma_flip):.0f}",
                annotation_font_color=_AMBER,
                annotation_position="top right",
            )
        except Exception:
            pass
    _apply_base(
        fig,
        title=f"{ticker} · Cumulative GEX",
        height=300,
        margin=dict(l=20, r=20, t=45, b=20),
        xaxis_title="Strike",
        yaxis_title="Cumulative GEX (Bn$ / %)",
        hovermode="x unified",
        showlegend=False,
    )
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def make_gex_breakdown_chart(history, ticker):
    """Stacked-relative bar chart showing positive / negative GEX composition over snapshots."""
    if not history or len(history) < 2:
        return None
    labels = [row["ts_label"] for row in history]
    pos = [safe_float(row.get("pos_gex"), 0.0) for row in history]
    neg = [safe_float(row.get("neg_gex"), 0.0) for row in history]
    totals = [safe_float(row.get("total_gex"), 0.0) for row in history]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Positive GEX",
        x=labels, y=pos,
        marker_color="rgba(0,217,126,0.72)",
        marker_line_width=0,
        hovertemplate="%{x}<br>+GEX: %{y:.3f} Bn$<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Negative GEX",
        x=labels, y=neg,
        marker_color="rgba(255,71,87,0.72)",
        marker_line_width=0,
        hovertemplate="%{x}<br>-GEX: %{y:.3f} Bn$<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=totals,
        mode="lines+markers",
        line=dict(color=_BLUE, width=2),
        marker=dict(size=6, color=_BLUE),
        name="Net GEX",
        hovertemplate="%{x}<br>Net: %{y:.3f} Bn$<extra></extra>",
    ))
    _apply_base(
        fig,
        barmode="relative",
        title=f"{ticker} · GEX Composition",
        height=320,
        margin=dict(l=20, r=20, t=45, b=20),
        xaxis_title="Snapshot",
        yaxis_title="GEX (Bn$ / %)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def _positive_gamma_view(strike_series: pd.Series | None, top_n: int = 40) -> pd.Series:
    if strike_series is None:
        return pd.Series(dtype=float)
    strike = pd.Series(strike_series, dtype=float)
    strike = strike[strike > 0]
    if strike.empty:
        return strike
    # Keep the most relevant positive strikes, then order by strike for readability.
    return strike.sort_values(ascending=False).head(top_n).sort_index()


_CHART_BG = "#07090f"
_GREEN = "#00d97e"
_RED = "#ff4757"
_BLUE = "#4dabf7"
_AMBER = "#f59e0b"

_BASE_LAYOUT = dict(
    paper_bgcolor=_CHART_BG,
    plot_bgcolor=_CHART_BG,
    font=dict(color="#c9d1d9", family="ui-monospace, monospace, sans-serif", size=11),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.15)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.15)"),
)


def _apply_base(fig: go.Figure, **extra) -> go.Figure:
    layout = dict(_BASE_LAYOUT)
    layout.update(extra)
    fig.update_layout(**layout)
    return fig


def make_gex_profile_chart(
    strike_series: pd.Series | None,
    ticker: str,
    spot: float | None = None,
    title: str = "GEX Profile",
    window_pct: float = 0.10,
    max_bars: int = 60,
) -> str | None:
    """
    Periscope-style horizontal bar chart of GEX by strike.

    Strikes run on the Y-axis (highest at top); bars extend right (green)
    for positive gamma and left (red) for negative gamma.  An optional
    horizontal guide line marks the current spot price.
    """
    if strike_series is None:
        return None
    strike = pd.Series(strike_series, dtype=float)
    if strike.empty:
        return None

    if spot is not None and spot > 0:
        lo, hi = spot * (1 - window_pct), spot * (1 + window_pct)
        window = strike.loc[(strike.index >= lo) & (strike.index <= hi)]
        if len(window) < 5:
            window = strike
    else:
        window = strike

    window = window.sort_index(ascending=True)
    if len(window) > max_bars:
        keep = window.abs().sort_values(ascending=False).head(max_bars).index
        window = window.loc[keep].sort_index(ascending=True)

    y_labels = [f"{int(s)}" for s in window.index]
    x_values = window.values.tolist()
    colors = [_GREEN if v >= 0 else _RED for v in x_values]

    fig = go.Figure(go.Bar(
        x=x_values,
        y=y_labels,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="Strike %{y}<br>GEX %{x:.3f} Bn$<extra></extra>",
    ))

    # Zero line
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.25)", line_width=1)

    # Spot guide
    if spot is not None and spot > 0:
        spot_label = f"{int(spot)}"
        if spot_label in y_labels:
            fig.add_hline(
                y=spot_label,
                line_color=_AMBER,
                line_dash="dash",
                line_width=1.5,
                annotation_text=f"Spot {int(spot)}",
                annotation_font_color=_AMBER,
                annotation_position="right",
            )

    _apply_base(
        fig,
        title=f"{ticker} · {title}",
        height=max(420, len(window) * 14),
        margin=dict(l=65, r=20, t=45, b=20),
        xaxis_title="GEX (Bn$ / %)",
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            tickfont=dict(family="ui-monospace, monospace", size=10),
        ),
        bargap=0.18,
    )
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def make_positive_strike_chart(strike_series: pd.Series | None, ticker: str, title: str):
    strike = _positive_gamma_view(strike_series)
    if strike.empty:
        return None

    x_vals = [float(x) for x in strike.index]
    y_vals = strike.values.tolist()
    fig = go.Figure(go.Bar(
        x=x_vals,
        y=y_vals,
        marker_color=_GREEN,
        marker_line_width=0,
        hovertemplate="Strike %{x:.0f}<br>GEX %{y:.3f} Bn$<extra></extra>",
    ))
    _apply_base(
        fig,
        title=f"{ticker} · {title}",
        height=300,
        margin=dict(l=20, r=20, t=45, b=20),
        xaxis_title="Strike",
        yaxis_title="Positive GEX (Bn$ / %)",
        yaxis=dict(rangemode="tozero", gridcolor="rgba(255,255,255,0.06)"),
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
            colorscale=[[0, _RED], [0.5, "#07090f"], [1, _GREEN]],
            hovertemplate="Strike %{x:.0f}<br>Expiry %{y}<br>GEX %{z:.2f} M$<extra></extra>",
        )
    )
    _apply_base(
        fig,
        title=f"{ticker} · GEX Surface Heatmap",
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

    gex_vals = df["GEX"].astype(float)
    fig = go.Figure(
        data=go.Scatter3d(
            x=df["strike"],
            y=df["expiration"].dt.strftime("%Y-%m-%d"),
            z=gex_vals,
            mode="markers",
            marker=dict(
                size=4,
                color=gex_vals,
                colorscale=[[0, _RED], [0.5, "#1e2030"], [1, _GREEN]],
                showscale=True,
                colorbar=dict(tickfont=dict(color="#c9d1d9")),
            ),
        )
    )
    _apply_base(
        fig,
        title=f"{ticker} · GEX Surface 3D",
        height=500,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return json.dumps(fig, cls=PlotlyJSONEncoder)


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
            cumulative_gex_json=None,
            breakdown_json=None,
            profile_json=None,
            timeline_options=[],
            selected=selected,
            prediction=None,
            similar_setups=[],
            flow_overlay=None,
            strike_csv=None,
            exp_csv=None,
            cum_csv=None,
            has_history=False,
            bootstrap_status=bootstrap_status,
            latest_ts=None,
            refresh_minutes=REFRESH_MINUTES,
            current_strike_chart_json=None,
            predicted_strike_chart_json=None,
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
    cumulative_gex_json = make_cumulative_gex_chart(
        selected.get("cumulative"),
        ticker,
        gamma_flip=selected.get("gamma_flip"),
    )
    breakdown_json = make_gex_breakdown_chart(history, ticker)
    profile_json = make_gex_profile_chart(
        selected.get("strike"),
        ticker,
        spot=safe_float(selected.get("spot"), None) or None,
        title="Dealer Gamma Profile",
    )

    prediction = predict_next_snapshot(history)
    current_strike_chart_json = make_positive_strike_chart(
        selected.get("strike"),
        ticker,
        "Current Positive GEX by Strike",
    )
    predicted_strike_chart_json = None
    if prediction and prediction.get("predicted_strike") is not None:
        predicted_strike_chart_json = make_positive_strike_chart(
            prediction.get("predicted_strike"),
            ticker,
            "Predicted Positive GEX by Strike",
        )
        prediction = {k: v for k, v in prediction.items() if k != "predicted_strike"}
    similar = similar_setups(history, top_n=6)

    flow_overlay = None
    if history:
        spot = safe_float(history[-1].get("spot"), 4800.0)
        try:
            flow_overlay = load_flow_predictions(FLOW_FEED_PATH, spot=spot)
            if prediction and flow_overlay:
                flow_delta = flow_overlay.get("predicted_flow_delta_gex_bn", 0.0)
                prediction["predicted_flow_delta_gex"] = flow_delta
                prediction["predicted_total_gex_with_flow"] = (
                    prediction["predicted_total_gex"] + flow_delta
                )
        except Exception:
            logger.debug("Flow overlay unavailable", exc_info=True)

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
        cumulative_gex_json=cumulative_gex_json,
        breakdown_json=breakdown_json,
        profile_json=profile_json,
        timeline_options=timeline_options,
        selected=selected,
        prediction=prediction,
        similar_setups=similar,
        flow_overlay=flow_overlay,
        strike_csv=selected["strike_path"].name if selected.get("strike_path") else None,
        exp_csv=selected["exp_path"].name if selected.get("exp_path") else None,
        cum_csv=selected["cum_path"].name if selected.get("cum_path") else None,
        latest_ts=ts_label(get_latest_ts(ticker)) if get_latest_ts(ticker) else None,
        refresh_minutes=REFRESH_MINUTES,
        has_history=True,
        bootstrap_status=bootstrap_status,
        current_strike_chart_json=current_strike_chart_json,
        predicted_strike_chart_json=predicted_strike_chart_json,
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
