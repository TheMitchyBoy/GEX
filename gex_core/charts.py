"""Shared Plotly chart builders for Flask and Streamlit dashboards."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

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


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _apply_base(fig: go.Figure, **extra) -> go.Figure:
    layout = dict(_BASE_LAYOUT)
    layout.update(extra)
    fig.update_layout(**layout)
    return fig


def _positive_gamma_view(strike_series: pd.Series | None, top_n: int = 40) -> pd.Series:
    if strike_series is None:
        return pd.Series(dtype=float)
    strike = pd.Series(strike_series, dtype=float)
    strike = strike[strike > 0]
    if strike.empty:
        return strike
    return strike.sort_values(ascending=False).head(top_n).sort_index()


def make_timeline_chart(history, ticker: str) -> str | None:
    if not history:
        return None
    labels = [row["ts_label"] for row in history]
    totals = [safe_float(row.get("total_gex"), 0.0) for row in history]
    pos = [safe_float(row.get("pos_gex"), 0.0) for row in history]
    neg = [safe_float(row.get("neg_gex"), 0.0) for row in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=pos, fill="tozeroy", fillcolor="rgba(0,217,126,0.10)",
        line=dict(color=_GREEN, width=1), name="Positive GEX",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=neg, fill="tozeroy", fillcolor="rgba(255,71,87,0.10)",
        line=dict(color=_RED, width=1), name="Negative GEX",
    ))
    marker_colors = [_GREEN if t >= 0 else _RED for t in totals]
    fig.add_trace(go.Scatter(
        x=labels, y=totals, mode="lines+markers",
        line=dict(color=_BLUE, width=2.5),
        marker=dict(size=7, color=marker_colors, line=dict(color=_CHART_BG, width=1)),
        name="Total GEX",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.25)", line_width=1)
    _apply_base(
        fig, title=f"{ticker} · GEX Timeline", height=340,
        margin=dict(l=20, r=20, t=45, b=20),
        xaxis_title="Snapshot", yaxis_title="GEX (Bn$ / %)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def make_cumulative_gex_chart(cumulative, ticker: str, gamma_flip=None) -> str | None:
    if cumulative is None or (hasattr(cumulative, "empty") and cumulative.empty):
        return None
    x = [float(v) for v in cumulative.index]
    y = [float(v) for v in cumulative]
    if not x:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[max(v, 0) for v in y], fill="tozeroy",
                             fillcolor="rgba(0,217,126,0.12)", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=[min(v, 0) for v in y], fill="tozeroy",
                             fillcolor="rgba(255,71,87,0.12)", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=_BLUE, width=2.5), name="Cumulative GEX"))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.25)", line_width=1)
    if gamma_flip is not None:
        try:
            fig.add_vline(x=float(gamma_flip), line_dash="dot", line_color=_AMBER, line_width=2,
                          annotation_text=f"Flip ~{float(gamma_flip):.0f}", annotation_font_color=_AMBER)
        except Exception:
            pass
    _apply_base(fig, title=f"{ticker} · Cumulative GEX", height=300,
                xaxis_title="Strike", yaxis_title="Cumulative GEX (Bn$ / %)", showlegend=False)
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def make_gex_breakdown_chart(history, ticker: str) -> str | None:
    if not history or len(history) < 2:
        return None
    labels = [row["ts_label"] for row in history]
    pos = [safe_float(row.get("pos_gex"), 0.0) for row in history]
    neg = [safe_float(row.get("neg_gex"), 0.0) for row in history]
    totals = [safe_float(row.get("total_gex"), 0.0) for row in history]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Positive GEX", x=labels, y=pos, marker_color="rgba(0,217,126,0.72)"))
    fig.add_trace(go.Bar(name="Negative GEX", x=labels, y=neg, marker_color="rgba(255,71,87,0.72)"))
    fig.add_trace(go.Scatter(x=labels, y=totals, mode="lines+markers", line=dict(color=_BLUE, width=2), name="Net GEX"))
    _apply_base(fig, barmode="relative", title=f"{ticker} · GEX Composition", height=320,
                xaxis_title="Snapshot", yaxis_title="GEX (Bn$ / %)", hovermode="x unified")
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def make_gex_profile_chart(
    strike_series: pd.Series | None,
    ticker: str,
    spot: float | None = None,
    title: str = "GEX Profile",
    window_pct: float = 0.10,
    max_bars: int = 60,
) -> str | None:
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
        x=x_values, y=y_labels, orientation="h", marker_color=colors, marker_line_width=0,
        hovertemplate="Strike %{y}<br>GEX %{x:.3f} Bn$<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.25)", line_width=1)
    if spot is not None and spot > 0:
        spot_label = f"{int(spot)}"
        if spot_label in y_labels:
            # add_hline breaks on categorical y-axes (Plotly TypeError); use shape instead.
            fig.add_shape(
                type="line",
                xref="paper",
                x0=0,
                x1=1,
                yref="y",
                y0=spot_label,
                y1=spot_label,
                line=dict(color=_AMBER, dash="dash", width=1.5),
            )
            fig.add_annotation(
                x=max(x_values) if x_values else 0,
                y=spot_label,
                text=f"Spot {int(spot)}",
                showarrow=False,
                font=dict(color=_AMBER, size=10),
                xanchor="left",
            )
    _apply_base(fig, title=f"{ticker} · {title}", height=max(420, len(window) * 14),
                xaxis_title="GEX (Bn$ / %)", bargap=0.18)
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def make_positive_strike_chart(strike_series: pd.Series | None, ticker: str, title: str) -> str | None:
    strike = _positive_gamma_view(strike_series)
    if strike.empty:
        return None
    fig = go.Figure(go.Bar(
        x=[float(x) for x in strike.index], y=strike.values.tolist(),
        marker_color=_GREEN, marker_line_width=0,
    ))
    _apply_base(fig, title=f"{ticker} · {title}", height=300,
                xaxis_title="Strike", yaxis_title="Positive GEX (Bn$ / %)",
                yaxis=dict(rangemode="tozero"))
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def make_heatmap(surface_path: Path | None = None, ticker: str = "", surface_df: pd.DataFrame | None = None):
    df = surface_df if surface_df is not None and not surface_df.empty else None
    if df is None and surface_path is not None:
        df = pd.read_csv(surface_path)
    if df is None or df.empty or not {"expiration", "strike", "GEX"}.issubset(df.columns):
        return None
    df = df.copy()
    df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce")
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["GEX"] = pd.to_numeric(df["GEX"], errors="coerce")
    df = df.dropna(subset=["expiration", "strike", "GEX"])
    if df.empty:
        return None
    pivot = df.pivot_table(index="expiration", columns="strike", values="GEX", aggfunc="sum").fillna(0)
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[float(c) for c in pivot.columns],
        y=[d.strftime("%Y-%m-%d") for d in pd.to_datetime(pivot.index)],
        colorscale=[[0, _RED], [0.5, "#07090f"], [1, _GREEN]],
    ))
    _apply_base(fig, title=f"{ticker} · GEX Surface Heatmap", height=500)
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def make_ai_insights_chart(analysis) -> str | None:
    if analysis is None:
        return None
    try:
        signals = analysis.signals
        colors = {"bullish": _GREEN, "bearish": _RED, "caution": _AMBER, "neutral": "#94a3b8"}
        bar_colors = [colors.get(s.sentiment, "#94a3b8") for s in signals]
        fig = go.Figure(go.Table(
            columnwidth=[140, 120, 380],
            header=dict(values=["<b>Signal</b>", "<b>Value</b>", "<b>Interpretation</b>"],
                        fill_color=_CHART_BG, font=dict(color="#c9d1d9", size=11)),
            cells=dict(
                values=[[s.label for s in signals], [s.value for s in signals], [s.detail for s in signals]],
                fill_color=[[_CHART_BG] * len(signals), bar_colors, [_CHART_BG] * len(signals)],
                font=dict(color="#c9d1d9", size=10), height=30,
            ),
        ))
        _apply_base(fig, height=max(260, len(signals) * 34 + 60), margin=dict(l=0, r=0, t=10, b=0))
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    except Exception:
        return None
