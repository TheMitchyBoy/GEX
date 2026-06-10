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

from gex_core.exports import EXPORT_DIR, find_exports_for_ticker, load_strike_series, parse_timestamp
from gex_core.features import compute_features_from_exports, enrich_snapshot_metrics
from gex_core.history import build_history as build_history_from_exports
from gex_core.backtest_metrics import backtest_delta_sign_accuracy
from gex_core.predict import apply_flow_to_prediction, load_flow_predictions, predict_next_snapshot, similar_setups
from gex_core.tickers import PRIMARY_TICKER, find_available_tickers

IMG_DIR = Path(__file__).resolve().parent / "img"
FLOW_FEED_PATH = Path(os.environ.get("GEX_FLOW_FEED", "data/flow_sample.jsonl"))


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

    backtest = backtest_delta_sign_accuracy(ticker)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted ΔGEX", f"{pred['predicted_delta_gex']:+.3f} Bn$")
    c2.metric("Predicted Total GEX", f"{pred['predicted_total_gex']:.3f} Bn$")
    c3.metric("Regime Flip Prob", f"{pred['regime_flip_probability'] * 100:.1f}%")
    c4.metric("Confidence", f"{pred['confidence'] * 100:.1f}%")

    st.caption(
        f"Predicted regime: **{pred['predicted_regime']}** · "
        f"flip: {pred['predicted_flip']:.2f} · "
        f"training snapshots: {pred.get('training_snapshot_count', 0)}"
    )

    term = pred.get("term_structure", {})
    if term:
        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("0DTE Share Forecast", f"{term.get('predicted_zero_dte_ratio', 0.0) * 100:+.1f}%")
        tc2.metric("Near-Term Share Forecast", f"{term.get('predicted_near_term_ratio', 0.0) * 100:+.1f}%")
        tc3.metric("Term Curvature Forecast", f"{term.get('predicted_term_curvature', 0.0):+.3f}")

    if backtest.get("n"):
        bt1, bt2, bt3 = st.columns(3)
        bt1.metric("Walk-forward Sign", f"{backtest['accuracy'] * 100:.1f}%")
        bt2.metric(
            "Momentum Baseline",
            f"{backtest['baseline_momentum_accuracy'] * 100:.1f}%"
            if backtest.get("baseline_momentum_accuracy") is not None
            else "N/A",
        )
        bt3.metric("ΔGEX MAE", f"{backtest['mae_delta']:.3f}" if backtest.get("mae_delta") is not None else "N/A")

    spot = float(history[-1].get("spot", 4800))
    flow = load_flow_predictions(FLOW_FEED_PATH, spot=spot)
    if flow["event_count"] > 0:
        pred_with_flow = apply_flow_to_prediction(pred, flow)
        st.markdown("**Live flow overlay**")
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("Raw Flow ΔGEX", f"{flow['predicted_flow_delta_gex_bn']:+.4f} Bn$")
        fc2.metric("Flow Weight", f"{pred_with_flow['flow_blend_weight'] * 100:.0f}%")
        fc3.metric("Combined forecast", f"{pred_with_flow['predicted_total_gex']:.3f} Bn$")
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


def render_ai_insights(analysis) -> None:
    """Render the AI dealer gamma analysis panel in Streamlit."""
    if analysis is None:
        st.info("AI analysis unavailable — configure UW_API_KEY.")
        return

    bias_color = "#00d97e" if analysis.bias == "bullish" else "#ff4757" if analysis.bias == "bearish" else "#f59e0b"

    col_bias, col_regime = st.columns([1, 2])
    with col_bias:
        st.markdown(
            f"<div style='font-size:1.4rem;font-weight:700;color:{bias_color};font-family:ui-monospace,monospace;'>"
            f"{analysis.bias.upper()}</div>"
            f"<div style='font-size:.75rem;color:#6e7681;'>Confidence: {analysis.confidence*100:.0f}%</div>",
            unsafe_allow_html=True,
        )
    with col_regime:
        st.markdown(
            f"<div style='font-size:.85rem;color:#c9d1d9;padding-top:.25rem;'>{analysis.regime_detail}</div>",
            unsafe_allow_html=True,
        )

    # Key levels
    c1, c2, c3, c4, c5 = st.columns(5)
    def _kv(col, label, val, color="#c9d1d9"):
        col.markdown(
            f"<div style='font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;color:#6e7681;font-family:ui-monospace,monospace;'>{label}</div>"
            f"<div style='font-size:1rem;font-weight:600;color:{color};font-family:ui-monospace,monospace;'>{val}</div>",
            unsafe_allow_html=True,
        )
    _kv(c1, "Total GEX", f"{analysis.total_gex_bn_per_pct:+.1f} Bn$",
        "#00d97e" if analysis.total_gex_bn_per_pct >= 0 else "#ff4757")
    _kv(c2, "Gamma Flip", f"{analysis.gamma_flip:.0f}" if analysis.gamma_flip else "N/A", "#f59e0b")
    flip_d = f"{analysis.flip_distance_pct:+.1f}%" if analysis.flip_distance_pct is not None else "N/A"
    _kv(c3, "Flip Dist", flip_d)
    _kv(c4, "Call Wall ▲", f"{analysis.call_wall:.0f}" if analysis.call_wall else "N/A", "#00d97e")
    _kv(c5, "Put Wall ▼", f"{analysis.put_wall:.0f}" if analysis.put_wall else "N/A", "#ff4757")

    st.markdown("---")
    st.markdown(f"<p style='font-size:.82rem;color:#94a3b8;line-height:1.6;'>{analysis.narrative}</p>",
                unsafe_allow_html=True)

    # Predictions
    if analysis.predictions:
        st.markdown("**Predictions**")
        for i, p in enumerate(analysis.predictions, 1):
            st.markdown(f"<div style='font-size:.8rem;margin-bottom:.35rem;'><span style='color:#4dabf7;font-family:ui-monospace,monospace;'>{i}.</span> {p}</div>",
                        unsafe_allow_html=True)

    # Signal table
    if analysis.signals:
        st.markdown("**Signals**")
        import pandas as _pd
        sig_rows = [{"Signal": s.label, "Value": s.value, "Interpretation": s.detail, "Tone": s.sentiment}
                    for s in analysis.signals]
        st.dataframe(
            _pd.DataFrame(sig_rows),
            use_container_width=True,
            hide_index=True,
        )


def main():
    st.set_page_config(page_title="SPX · Gamma Intelligence", layout="wide")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    tickers_from_exports = find_available_tickers(EXPORT_DIR)
    ticker = PRIMARY_TICKER
    st.sidebar.markdown("### SPX Gamma Dashboard")
    if tickers_from_exports:
        st.sidebar.caption("Using SPX exports from data/exports.")
    else:
        st.sidebar.caption("No SPX exports found yet; live data will show when UW_API_KEY is configured.")

    uw_enabled = bool(os.environ.get("UW_API_KEY"))
    if uw_enabled:
        st.sidebar.success("Unusual Whales: connected")
    else:
        st.sidebar.warning("UW_API_KEY not set — showing historical exports only")

    show_images = st.sidebar.checkbox("Show image snapshots", value=False)
    show_heatmap = st.sidebar.checkbox("Show heatmap", value=True)
    show_3d = st.sidebar.checkbox("Show 3D scatter", value=False)
    positive_gamma_focus = st.sidebar.checkbox("Strike focus: positive gamma only", value=True)
    show_predictions = st.sidebar.checkbox("Show KNN predictions", value=True)

    # ── Live UW data ─────────────────────────────────────────────────────────
    uw_spot: float | None = None
    uw_agg = None
    uw_analysis = None

    if uw_enabled:
        with st.spinner("Fetching live data from Unusual Whales…"):
            try:
                from gex_core.uw_loader import fetch_uw_gex
                from gex_core.ai_analyst import analyze_dealer_gamma
                from gex_core.features import resolve_gamma_flip

                uw_spot, uw_agg = fetch_uw_gex(ticker)
                greek_df = uw_agg.gex_by_strike.attrs.get("greek_exposure_df")
                uw_gamma_flip = resolve_gamma_flip(
                    spot=uw_spot,
                    gex_by_strike=uw_agg.gex_by_strike,
                    cumulative_gex=uw_agg.cumulative_gex,
                    greek_exposure_df=greek_df if isinstance(greek_df, pd.DataFrame) else None,
                )
                uw_analysis = analyze_dealer_gamma(
                    ticker=ticker, spot=uw_spot,
                    gex_by_strike=uw_agg.gex_by_strike,
                    cumulative_gex=uw_agg.cumulative_gex,
                    total_gex_bn=uw_agg.total_gex_bn,
                    gamma_flip=uw_gamma_flip,
                )
            except Exception as e:
                st.sidebar.error(f"UW fetch failed: {e}")

    # ── Header ────────────────────────────────────────────────────────────────
    hdr_col, spot_col = st.columns([3, 1])
    with hdr_col:
        st.title(f"{ticker} · SPX Gamma Intelligence")
    with spot_col:
        if uw_spot:
            st.metric("Live Spot (UW)", f"${uw_spot:,.2f}")

    # ── Main tabs ─────────────────────────────────────────────────────────────
    if uw_enabled and uw_agg is not None:
        tab_ai, tab_profile, tab_timeline, tab_breakdown, tab_cumulative, tab_exports = st.tabs(
            ["AI Analysis", "Dealer Profile", "GEX Timeline", "Composition", "Cumulative GEX", "Historical Exports"]
        )
    else:
        tab_ai = None
        tab_profile, tab_timeline, tab_breakdown, tab_cumulative, tab_exports = st.tabs(
            ["Dealer Profile", "GEX Timeline", "Composition", "Cumulative GEX", "Historical Exports"]
        )

    # ── AI Analysis tab ───────────────────────────────────────────────────────
    if tab_ai is not None:
        with tab_ai:
            render_ai_insights(uw_analysis)

    # ── Dealer Profile tab ────────────────────────────────────────────────────
    with tab_profile:
        if uw_agg is not None:
            st.caption("Live Unusual Whales dealer gamma profile — green = long gamma, red = short gamma.")
            render_gex_profile(uw_agg.gex_by_strike, spot=uw_spot)
        else:
            st.info("Connect Unusual Whales API to see the live dealer profile.")

    # ── History from exports (for timeline / composition / cumulative) ─────────
    history = build_history_from_exports(ticker)

    with tab_timeline:
        if history:
            render_gex_timeline(history)
        elif uw_agg is not None:
            st.info("No snapshot history yet — showing live cumulative GEX.")
            render_cumulative_gex(uw_agg.cumulative_gex, gamma_flip=uw_gamma_flip if uw_enabled else None)
        else:
            st.info("No data available.")

    with tab_breakdown:
        if len(history) >= 2:
            render_gex_breakdown(history)
        else:
            st.info("Need at least 2 historical snapshots for composition chart.")

    with tab_cumulative:
        if uw_agg is not None:
            st.caption("Live cumulative GEX from Unusual Whales — zero-crossing = gamma flip.")
            render_cumulative_gex(uw_agg.cumulative_gex, gamma_flip=uw_gamma_flip if uw_enabled else None)
        elif history:
            latest = history[-1]
            render_cumulative_gex(
                latest.get("cumulative", pd.Series(dtype=float)),
                gamma_flip=latest.get("gamma_flip"),
            )
        else:
            st.info("No data available.")

    # ── Historical exports tab ────────────────────────────────────────────────
    with tab_exports:
        if show_predictions and history:
            render_predictions(ticker, history)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Snapshots")
            imgs = list_img_snapshots(ticker, IMG_DIR)
            if imgs and show_images:
                cols = st.columns(2)
                for i, img_path in enumerate(imgs[:6]):
                    with cols[i % 2]:
                        st.image(str(img_path), caption=img_path.name, use_container_width=True)
            elif not imgs:
                st.info("No PNG snapshots in img/.")
            surface_path = latest_file_for(f"{ticker}_gex_surface_*.csv", EXPORT_DIR)
            if surface_path:
                st.markdown(f"**Latest surface CSV:** {surface_path.name}")

        with col2:
            st.subheader("Interactive exports")
            strike_path = latest_file_for(f"{ticker}_gex_by_strike_*.csv", EXPORT_DIR)
            exp_path = latest_file_for(f"{ticker}_gex_by_expiration_*.csv", EXPORT_DIR)

            if surface_path:
                df_surface = load_surface_csv(surface_path)
                if not df_surface.empty:
                    min_date = df_surface["expiration"].min().date()
                    max_date = df_surface["expiration"].max().date()
                    date_range = st.slider("Expiration range", value=(min_date, max_date),
                                           min_value=min_date, max_value=max_date)
                    strikes = sorted(df_surface["strike"].unique())
                    strike_range = st.slider("Strike range",
                                             min_value=int(min(strikes)), max_value=int(max(strikes)),
                                             value=(int(min(strikes)), int(max(strikes))))
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
                        st.subheader("GEX surface (3D)")
                        render_surface_3d(df_filtered)

            if strike_path:
                st.subheader("GEX by strike (historical)")
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
                    st.info("No positive gamma strikes available.")
                else:
                    y_label = "Positive GEX (Bn$)" if positive_gamma_focus else "GEX (Bn$)"
                    bar_colors = [_ST_GREEN] * len(strike_chart) if positive_gamma_focus else \
                                 [_ST_GREEN if v >= 0 else _ST_RED for v in strike_chart["gex"]]
                    fig = go.Figure(go.Bar(
                        x=strike_chart["strike"], y=strike_chart["gex"],
                        marker_color=bar_colors, marker_line_width=0,
                        hovertemplate="Strike %{x:.0f}<br>GEX %{y:.3f} Bn$<extra></extra>",
                    ))
                    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)", line_width=1)
                    fig.update_layout(xaxis_title="Strike", yaxis_title=y_label, height=340, **_ST_BASE)
                    st.plotly_chart(fig, use_container_width=True)

            if exp_path:
                st.subheader("GEX by expiration (historical)")
                df_exp = pd.read_csv(str(exp_path), index_col=0)
                exp_vals = pd.to_numeric(df_exp.iloc[:, 0], errors="coerce").fillna(0.0)
                exp_colors = [_ST_GREEN if v >= 0 else _ST_RED for v in exp_vals]
                fig2 = go.Figure(go.Bar(
                    x=df_exp.index, y=exp_vals,
                    marker_color=exp_colors, marker_line_width=0,
                    hovertemplate="Expiry %{x}<br>GEX %{y:.3f} Bn$<extra></extra>",
                ))
                fig2.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)", line_width=1)
                fig2.update_layout(xaxis_title="Expiration", yaxis_title="GEX (Bn$)", height=320, **_ST_BASE)
                st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()
