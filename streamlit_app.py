"""Streamlit dashboard for GEX exports, predictions, and live flow overlay.

Usage:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from gex_core.exports import find_exports_for_ticker, load_strike_series, parse_timestamp
from gex_core.features import compute_features_from_exports, enrich_snapshot_metrics
from gex_core.predict import load_flow_predictions, predict_next_snapshot, similar_setups

EXPORT_DIR = Path("data/exports")
IMG_DIR = Path("img")
FLOW_FEED_PATH = Path(os.environ.get("GEX_FLOW_FEED", "data/flow_sample.jsonl"))


def find_available_tickers(export_dir: Path):
    export_dir.mkdir(parents=True, exist_ok=True)
    tickers = set()
    for f in export_dir.glob("*.csv"):
        parts = f.name.split("_")
        if parts:
            tickers.add(parts[0])
    return sorted(tickers)


def latest_file_for(pattern: str, directory: Path):
    files = list(directory.glob(pattern))
    if not files:
        return None
    return sorted(files, key=lambda p: p.stat().st_mtime)[-1]


def load_surface_csv(path: Path):
    df = pd.read_csv(path)
    if "expiration" in df.columns:
        df["expiration"] = pd.to_datetime(df["expiration"])
    return df


def build_history_from_exports(ticker: str) -> list[dict]:
    exports = find_exports_for_ticker(ticker)
    timestamps = sorted(ts for ts, k in exports.items() if "gex_by_strike" in k)
    history = []
    prev_feats = None
    for ts in timestamps:
        info = exports[ts]
        feats = compute_features_from_exports(info, prev_features=prev_feats)
        prev_feats = feats
        strike = load_strike_series(info["gex_by_strike"])
        cumulative = (
            load_strike_series(info["cumulative_gex"])
            if "cumulative_gex" in info
            else pd.Series(dtype=float)
        )
        row = {
            "ts": ts,
            "ts_label": parse_timestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": ticker,
            "strike": strike,
            "cumulative": cumulative,
            "total_gex": feats["total_gex_bn"],
            "pos_gex": feats["pos_gex_bn"],
            "neg_gex": feats["neg_gex_bn"],
            "gex_std": feats["gex_std_bn"],
            "near_term_ratio": feats["near_term_ratio"],
            "surface_peak": feats.get("surface_peak", 0.0),
            "call_wall": feats.get("call_wall"),
            "put_wall": feats.get("put_wall"),
            "gamma_flip": feats.get("gamma_flip"),
            "regime": "LONG gamma" if feats["total_gex_bn"] >= 0 else "SHORT gamma",
            "abs_mean": abs(strike).mean() if len(strike) else 0.0,
        }
        history.append(enrich_snapshot_metrics(row))
    return history


_ST_BG = "#07090f"
_ST_GREEN = "#00d97e"
_ST_RED = "#ff4757"
_ST_BLUE = "#4dabf7"
_ST_AMBER = "#f59e0b"

_ST_BASE = dict(
    paper_bgcolor=_ST_BG,
    plot_bgcolor=_ST_BG,
    font=dict(color="#c9d1d9", family="ui-monospace, monospace, sans-serif", size=11),
)


def render_surface_heatmap(df: pd.DataFrame):
    if df.empty:
        st.info("No surface data to render.")
        return
    pivot = df.pivot(index="expiration", columns="strike", values="GEX").sort_index()
    fig = px.imshow(
        pivot.fillna(0).values,
        x=pivot.columns.astype(float),
        y=pivot.index.strftime("%Y-%m-%d"),
        aspect="auto",
        labels={"x": "Strike", "y": "Expiration", "color": "GEX (M$ / %)"},
        color_continuous_scale=[[0, _ST_RED], [0.5, _ST_BG], [1, _ST_GREEN]],
    )
    fig.update_layout(height=520, **_ST_BASE)
    st.plotly_chart(fig, use_container_width=True)


def render_surface_3d(df: pd.DataFrame):
    if df.empty:
        return
    gex = df["GEX"].astype(float)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=df["strike"].astype(float),
            y=df["expiration"].astype(str),
            z=gex,
            mode="markers",
            marker=dict(
                size=4,
                color=gex,
                colorscale=[[0, _ST_RED], [0.5, "#1e2030"], [1, _ST_GREEN]],
                showscale=True,
            ),
        )
    )
    fig.update_layout(height=560, **_ST_BASE)
    st.plotly_chart(fig, use_container_width=True)


def render_gex_profile(strike: pd.Series, spot: float | None = None, max_bars: int = 55):
    """
    Periscope-style horizontal bar 'profile' chart of GEX by strike.

    Strikes on Y-axis (highest at top), bars extend right (green = long gamma)
    or left (red = short gamma).
    """
    if strike is None or strike.empty:
        st.info("No strike data available for profile view.")
        return

    if spot is not None and spot > 0:
        lo, hi = spot * 0.90, spot * 1.10
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
    colors = [_ST_GREEN if v >= 0 else _ST_RED for v in x_values]

    fig = go.Figure(go.Bar(
        x=x_values,
        y=y_labels,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="Strike %{y}<br>GEX %{x:.3f} Bn$<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.2)", line_width=1)

    if spot is not None and spot > 0:
        spot_label = f"{int(spot)}"
        if spot_label in y_labels:
            fig.add_hline(
                y=spot_label,
                line_color=_ST_AMBER,
                line_dash="dash",
                line_width=1.5,
            )

    fig.update_layout(
        height=max(420, len(window) * 15),
        margin=dict(l=65, r=20, t=20, b=20),
        xaxis_title="GEX (Bn$ / %)",
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            tickfont=dict(family="ui-monospace, monospace", size=10),
        ),
        bargap=0.18,
        **_ST_BASE,
    )
    st.plotly_chart(fig, use_container_width=True)


def list_img_snapshots(ticker: str, img_dir: Path):
    if not img_dir.exists():
        return []
    return sorted(img_dir.glob(f"{ticker}_*.*"), key=lambda p: p.stat().st_mtime, reverse=True)


def render_predictions(ticker: str, history: list[dict]):
    st.subheader("GEX Change Prediction")
    if len(history) < 4:
        st.info("Need at least 4 snapshots for prediction.")
        return

    pred = predict_next_snapshot(history)
    if not pred:
        st.warning("Could not generate prediction.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted ΔGEX", f"{pred['predicted_delta_gex']:+.3f} Bn$")
    c2.metric("Predicted Total GEX", f"{pred['predicted_total_gex']:.3f} Bn$")
    c3.metric("Regime Flip Prob", f"{pred['regime_flip_probability'] * 100:.1f}%")
    c4.metric("Confidence", f"{pred['confidence'] * 100:.1f}%")

    st.caption(f"Predicted regime: **{pred['predicted_regime']}** · flip: {pred['predicted_flip']:.2f}")

    spot = float(history[-1].get("spot", 4800))
    flow = load_flow_predictions(FLOW_FEED_PATH, spot=spot)
    if flow["event_count"] > 0:
        st.markdown("**Live flow overlay**")
        fc1, fc2 = st.columns(2)
        fc1.metric("Flow ΔGEX", f"{flow['predicted_flow_delta_gex_bn']:+.4f} Bn$")
        fc2.metric("Combined forecast", f"{pred['predicted_total_gex'] + flow['predicted_flow_delta_gex_bn']:.3f} Bn$")
        if flow["top_signals"]:
            sig_df = pd.DataFrame(flow["top_signals"])
            st.dataframe(sig_df[["strike", "direction", "score", "recent_gex"]], use_container_width=True)

    similar = similar_setups(history, top_n=5)
    if similar:
        st.markdown("**Similar historical setups**")
        sim_df = pd.DataFrame(similar)[
            ["snapshot", "similarity", "total_gex", "next_delta_gex", "next_regime"]
        ]
        st.dataframe(sim_df, use_container_width=True)


def render_gex_timeline(history: list[dict]):
    if len(history) < 2:
        return
    labels = [h["ts_label"] for h in history]
    totals = [float(h.get("total_gex", 0) or 0) for h in history]
    pos = [float(h.get("pos_gex", 0) or 0) for h in history]
    neg = [float(h.get("neg_gex", 0) or 0) for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=pos,
        fill="tozeroy",
        fillcolor="rgba(0,217,126,0.10)",
        line=dict(color=_ST_GREEN, width=1),
        name="Positive GEX",
        hovertemplate="%{y:.3f} Bn$<extra>+GEX</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=neg,
        fill="tozeroy",
        fillcolor="rgba(255,71,87,0.10)",
        line=dict(color=_ST_RED, width=1),
        name="Negative GEX",
        hovertemplate="%{y:.3f} Bn$<extra>-GEX</extra>",
    ))
    marker_colors = [_ST_GREEN if t >= 0 else _ST_RED for t in totals]
    fig.add_trace(go.Scatter(
        x=labels, y=totals,
        mode="lines+markers",
        line=dict(color=_ST_BLUE, width=2.5),
        marker=dict(size=7, color=marker_colors),
        name="Total GEX",
        hovertemplate="%{y:.3f} Bn$<extra>Net GEX</extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.25)", line_width=1)
    fig.update_layout(
        height=340,
        xaxis_title="Snapshot",
        yaxis_title="GEX (Bn$ / %)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **_ST_BASE,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_cumulative_gex(cumulative: pd.Series, gamma_flip=None):
    """Area chart of cumulative GEX with gamma-flip annotation."""
    if cumulative is None or cumulative.empty:
        st.info("No cumulative GEX data available.")
        return
    x = [float(v) for v in cumulative.index]
    y = cumulative.astype(float).tolist()

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
        line=dict(color=_ST_BLUE, width=2.5),
        name="Cumulative GEX",
        hovertemplate="Strike %{x:.0f}<br>%{y:.3f} Bn$<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.25)", line_width=1)
    if gamma_flip is not None:
        try:
            fig.add_vline(
                x=float(gamma_flip),
                line_dash="dot",
                line_color=_ST_AMBER,
                line_width=2,
                annotation_text=f"Flip ~{float(gamma_flip):.0f}",
                annotation_font_color=_ST_AMBER,
                annotation_position="top right",
            )
        except Exception:
            pass
    fig.update_layout(
        height=300,
        xaxis_title="Strike",
        yaxis_title="Cumulative GEX (Bn$ / %)",
        hovermode="x unified",
        showlegend=False,
        **_ST_BASE,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_gex_breakdown(history: list[dict]):
    """Stacked-relative bar chart showing positive/negative GEX composition over time."""
    if len(history) < 2:
        return
    labels = [h["ts_label"] for h in history]
    pos = [float(h.get("pos_gex", 0) or 0) for h in history]
    neg = [float(h.get("neg_gex", 0) or 0) for h in history]
    totals = [float(h.get("total_gex", 0) or 0) for h in history]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Positive GEX", x=labels, y=pos,
        marker_color="rgba(0,217,126,0.72)",
        marker_line_width=0,
        hovertemplate="%{x}<br>+GEX: %{y:.3f} Bn$<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Negative GEX", x=labels, y=neg,
        marker_color="rgba(255,71,87,0.72)",
        marker_line_width=0,
        hovertemplate="%{x}<br>-GEX: %{y:.3f} Bn$<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=totals,
        mode="lines+markers",
        line=dict(color=_ST_BLUE, width=2),
        marker=dict(size=6, color=_ST_BLUE),
        name="Net GEX",
        hovertemplate="%{x}<br>Net: %{y:.3f} Bn$<extra></extra>",
    ))
    fig.update_layout(
        barmode="relative",
        height=320,
        xaxis_title="Snapshot",
        yaxis_title="GEX (Bn$ / %)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **_ST_BASE,
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(page_title="GEX Dashboard", layout="wide")
    st.title("GEX Dashboard")

    tickers = find_available_tickers(EXPORT_DIR)
    ticker = st.sidebar.selectbox("Ticker", options=tickers if tickers else ["SPX"], index=0)
    custom = st.sidebar.text_input("Or enter ticker manually", value="")
    if custom.strip():
        ticker = custom.strip().upper()

    show_images = st.sidebar.checkbox("Show image snapshots", value=True)
    show_heatmap = st.sidebar.checkbox("Show heatmap", value=True)
    show_3d = st.sidebar.checkbox("Show 3D scatter", value=False)
    positive_gamma_focus = st.sidebar.checkbox("Strike charts: positive gamma focus", value=True)
    show_predictions = st.sidebar.checkbox("Show GEX predictions", value=True)

    history = build_history_from_exports(ticker)

    if show_predictions and history:
        render_predictions(ticker, history)

    if history:
        tab_profile, tab_timeline, tab_breakdown, tab_cumulative = st.tabs(
            ["Dealer Profile", "GEX Timeline", "GEX Composition", "Cumulative GEX"]
        )
        latest = history[-1]
        with tab_profile:
            st.caption("Horizontal exposure profile — Periscope style. Green bars = dealer long gamma, red = short.")
            render_gex_profile(
                latest.get("strike", pd.Series(dtype=float)),
                spot=float(latest.get("spot", 0) or 0) or None,
            )
        with tab_timeline:
            render_gex_timeline(history)
        with tab_breakdown:
            if len(history) >= 2:
                render_gex_breakdown(history)
            else:
                st.info("Need at least 2 snapshots for composition chart.")
        with tab_cumulative:
            render_cumulative_gex(
                latest.get("cumulative", pd.Series(dtype=float)),
                gamma_flip=latest.get("gamma_flip"),
            )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Snapshots & Info")
        imgs = list_img_snapshots(ticker, IMG_DIR)
        if imgs and show_images:
            cols = st.columns(2)
            for i, img_path in enumerate(imgs[:6]):
                with cols[i % 2]:
                    st.image(str(img_path), caption=img_path.name, use_container_width=True)
        elif not imgs:
            st.info("No PNG snapshots found in img/.")

        surface_path = latest_file_for(f"{ticker}_gex_surface_*.csv", EXPORT_DIR)
        if surface_path:
            st.markdown(f"**Latest surface CSV:** {surface_path.name}")

    with col2:
        st.header("Interactive exports")
        strike_path = latest_file_for(f"{ticker}_gex_by_strike_*.csv", EXPORT_DIR)
        exp_path = latest_file_for(f"{ticker}_gex_by_expiration_*.csv", EXPORT_DIR)

        if surface_path:
            df_surface = load_surface_csv(surface_path)
            if not df_surface.empty:
                min_date = df_surface["expiration"].min().date()
                max_date = df_surface["expiration"].max().date()
                date_range = st.slider("Expiration range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
                strikes = sorted(df_surface["strike"].unique())
                strike_range = st.slider(
                    "Strike range",
                    min_value=int(min(strikes)),
                    max_value=int(max(strikes)),
                    value=(int(min(strikes)), int(max(strikes))),
                )
                df_filtered = df_surface.loc[
                    (df_surface["expiration"].dt.date >= date_range[0])
                    & (df_surface["expiration"].dt.date <= date_range[1])
                    & (df_surface["strike"] >= strike_range[0])
                    & (df_surface["strike"] <= strike_range[1])
                ]
                st.metric("Surface total GEX (M$)", f"{df_filtered['GEX'].sum():,.1f}")
                if show_heatmap:
                    st.subheader("GEX surface (heatmap)")
                    render_surface_heatmap(df_filtered)
                if show_3d:
                    st.subheader("GEX surface (3D scatter)")
                    render_surface_3d(df_filtered)

        if strike_path:
            st.subheader("GEX by strike")
            df_strike = pd.read_csv(str(strike_path), index_col=0, parse_dates=False)
            try:
                df_strike.index = df_strike.index.astype(float)
            except Exception:
                pass
            strike_chart = pd.DataFrame({"strike": df_strike.index, "gex": df_strike.iloc[:, 0]})
            strike_chart["gex"] = pd.to_numeric(strike_chart["gex"], errors="coerce").fillna(0.0)
            if positive_gamma_focus:
                strike_chart = strike_chart.loc[strike_chart["gex"] > 0].sort_values("gex", ascending=False).head(40).sort_values("strike")
            if strike_chart.empty:
                st.info(
                    "No positive gamma strikes available for this snapshot."
                    if positive_gamma_focus
                    else "No strike-level gamma data is available for this snapshot."
                )
            else:
                y_label = "Positive GEX (Bn$ / %)" if positive_gamma_focus else "GEX (Bn$ / %)"
                bar_colors = (
                    [_ST_GREEN] * len(strike_chart)
                    if positive_gamma_focus
                    else [_ST_GREEN if v >= 0 else _ST_RED for v in strike_chart["gex"]]
                )
                fig = go.Figure(go.Bar(
                    x=strike_chart["strike"],
                    y=strike_chart["gex"],
                    marker_color=bar_colors,
                    marker_line_width=0,
                    hovertemplate="Strike %{x:.0f}<br>GEX %{y:.3f} Bn$<extra></extra>",
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)", line_width=1)
                fig.update_layout(
                    title="Positive gamma strike focus" if positive_gamma_focus else "GEX by strike",
                    xaxis_title="Strike",
                    yaxis_title=y_label,
                    height=340,
                    **_ST_BASE,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No gex_by_strike CSV found in data/exports for this ticker.")

        if exp_path:
            st.subheader("GEX by expiration")
            df_exp = pd.read_csv(str(exp_path), index_col=0)
            exp_vals = pd.to_numeric(df_exp.iloc[:, 0], errors="coerce").fillna(0.0)
            exp_colors = [_ST_GREEN if v >= 0 else _ST_RED for v in exp_vals]
            fig2 = go.Figure(go.Bar(
                x=df_exp.index,
                y=exp_vals,
                marker_color=exp_colors,
                marker_line_width=0,
                hovertemplate="Expiry %{x}<br>GEX %{y:.3f} Bn$<extra></extra>",
            ))
            fig2.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)", line_width=1)
            fig2.update_layout(
                xaxis_title="Expiration",
                yaxis_title="GEX (Bn$ / %)",
                height=320,
                **_ST_BASE,
            )
            st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()
