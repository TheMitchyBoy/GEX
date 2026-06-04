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
        color_continuous_scale="RdYlBu_r",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)


def render_surface_3d(df: pd.DataFrame):
    if df.empty:
        return
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=df["strike"].astype(float),
            y=df["expiration"].astype(str),
            z=df["GEX"].astype(float),
            mode="markers",
            marker=dict(size=4, color=df["GEX"], colorscale="Viridis", showscale=True),
        )
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)


def list_img_snapshots(ticker: str, img_dir: Path):
    if not img_dir.exists():
        return []
    return sorted(img_dir.glob(f"{ticker}_*.*"), key=lambda p: p.stat().st_mtime, reverse=True)


def render_gamma_hero(history: list[dict]):
    """Primary focus: current gamma and future gamma prediction side by side."""
    if not history:
        st.info("No GEX history for this ticker.")
        return

    current = history[-1]
    pred = predict_next_snapshot(history) if len(history) >= 4 else None

    st.markdown(f"### {current.get('ticker', '')} · Gamma Outlook")
    col_current, col_future = st.columns(2)

    with col_current:
        st.markdown("#### Current Gamma")
        regime = current["regime"]
        st.metric(
            label=f"{regime} · {current['ts_label']}",
            value=f"{current['total_gex']:+.3f} Bn$ / %",
            delta=None,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Gamma Flip", f"{current['gamma_flip']:.0f}" if current.get("gamma_flip") else "N/A")
        c2.metric("Call Wall", f"{current['call_wall']:.0f}" if current.get("call_wall") else "N/A")
        c3.metric("Put Wall", f"{current['put_wall']:.0f}" if current.get("put_wall") else "N/A")
        c4, c5, c6 = st.columns(3)
        c4.metric("Near-term", f"{current['near_term_ratio'] * 100:.0f}%")
        c5.metric("+GEX", f"{current['pos_gex']:.2f}")
        c6.metric("−GEX", f"{current['neg_gex']:.2f}")

    with col_future:
        st.markdown("#### Future Gamma · Next Snapshot")
        if pred:
            st.metric(
                label=pred["predicted_regime"],
                value=f"{pred['predicted_total_gex']:+.3f} Bn$ / %",
                delta=f"{pred['predicted_delta_gex']:+.3f} ΔGEX",
            )
            f1, f2, f3 = st.columns(3)
            f1.metric("Pred. Flip", f"{pred['predicted_flip']:.0f}")
            f2.metric("Confidence", f"{pred['confidence'] * 100:.0f}%")
            f3.metric("Regime Flip", f"{pred['regime_flip_probability'] * 100:.0f}%")
            spot = float(current.get("spot", 4800))
            flow = load_flow_predictions(FLOW_FEED_PATH, spot=spot)
            if flow["event_count"] > 0:
                st.caption(
                    f"Flow overlay: {flow['predicted_flow_delta_gex_bn']:+.4f} Bn$ · "
                    f"combined {pred['predicted_total_gex'] + flow['predicted_flow_delta_gex_bn']:.3f} Bn$"
                )
        else:
            st.info("Need at least 4 snapshots to forecast the next gamma update.")

    st.divider()


def render_predictions_detail(ticker: str, history: list[dict]):
    """Secondary prediction details below the hero."""
    if len(history) < 4:
        return

    pred = predict_next_snapshot(history)
    if not pred:
        return

    similar = similar_setups(history, top_n=5)
    if similar:
        with st.expander("Similar historical setups"):
            sim_df = pd.DataFrame(similar)[
                ["snapshot", "similarity", "total_gex", "next_delta_gex", "next_regime"]
            ]
            st.dataframe(sim_df, use_container_width=True)

    spot = float(history[-1].get("spot", 4800))
    flow = load_flow_predictions(FLOW_FEED_PATH, spot=spot)
    if flow["event_count"] > 0 and flow.get("top_signals"):
        with st.expander("Live flow strike signals"):
            sig_df = pd.DataFrame(flow["top_signals"])
            st.dataframe(sig_df[["strike", "direction", "score", "recent_gex"]], use_container_width=True)


def render_gex_timeline(history: list[dict]):
    if len(history) < 2:
        return
    st.subheader("GEX Timeline")
    labels = [h["ts_label"] for h in history]
    totals = [h["total_gex"] for h in history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels, y=totals, mode="lines+markers", name="Total GEX"))
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(height=320, xaxis_title="Snapshot", yaxis_title="GEX (Bn$ / %)")
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(page_title="Gamma Tracker", layout="wide", page_icon="γ")

    tickers = find_available_tickers(EXPORT_DIR)
    ticker = st.sidebar.selectbox("Ticker", options=tickers if tickers else ["SPX"], index=0)
    custom = st.sidebar.text_input("Or enter ticker manually", value="")
    if custom.strip():
        ticker = custom.strip().upper()

    show_images = st.sidebar.checkbox("Show image snapshots", value=False)
    show_heatmap = st.sidebar.checkbox("Show heatmap", value=True)
    show_3d = st.sidebar.checkbox("Show 3D scatter", value=False)
    show_timeline = st.sidebar.checkbox("Show GEX timeline", value=True)

    history = build_history_from_exports(ticker)
    for row in history:
        row["ticker"] = ticker

    # Primary focus at top
    render_gamma_hero(history)
    render_predictions_detail(ticker, history)

    if show_timeline and len(history) >= 2:
        render_gex_timeline(history)

    st.markdown("#### Analysis & Exports")
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
            df_strike = pd.read_csv(str(strike_path), index_col=0)
            fig = px.bar(df_strike, x=df_strike.index.astype(float), y=df_strike.iloc[:, 0])
            st.plotly_chart(fig, use_container_width=True)

        if exp_path:
            st.subheader("GEX by expiration")
            df_exp = pd.read_csv(str(exp_path), index_col=0)
            fig2 = px.bar(df_exp, x=df_exp.index, y=df_exp.iloc[:, 0])
            st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()
