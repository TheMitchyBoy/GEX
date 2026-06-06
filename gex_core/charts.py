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


def _bar_width(values: list[float], fill_ratio: float = 0.86) -> float | None:
    """Pick a visually thick bar width for numeric strike axes."""
    if len(values) < 2:
        return None
    ordered = sorted(float(v) for v in values)
    diffs = [b - a for a, b in zip(ordered, ordered[1:]) if b > a]
    if not diffs:
        return None
    return max(1.0, pd.Series(diffs).median() * fill_ratio)


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
    spot = [safe_float(row.get("spot"), 0.0) or None for row in history]
    pos_gamma_peak = [
        safe_float(row.get("pos_gamma_peak_strike"), 0.0) or None for row in history
    ]
    if not any(v is not None for v in pos_gamma_peak):
        pos_gamma_peak = [safe_float(row.get("call_wall"), 0.0) or None for row in history]
    gamma_flip = [safe_float(row.get("gamma_flip"), 0.0) or None for row in history]
    call_wall = [safe_float(row.get("call_wall"), 0.0) or None for row in history]
    put_wall = [safe_float(row.get("put_wall"), 0.0) or None for row in history]
    near_term_ratio = [safe_float(row.get("near_term_ratio"), 0.0) for row in history]
    regimes = [row.get("regime", "N/A") for row in history]

    fig = go.Figure()

    def _band_bounds(levels: list[float | None], pct: float) -> tuple[list[float | None], list[float | None]]:
        lower = [(level * (1 - pct)) if level is not None else None for level in levels]
        upper = [(level * (1 + pct)) if level is not None else None for level in levels]
        return lower, upper

    def _add_distance_band(
        levels: list[float | None],
        pct: float,
        name: str,
        color: str,
        fillcolor: str,
    ) -> None:
        if not any(level is not None for level in levels):
            return
        lower, upper = _band_bounds(levels, pct)
        fig.add_trace(go.Scatter(
            x=labels,
            y=lower,
            mode="lines",
            line=dict(width=0, color=color),
            hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=labels,
            y=upper,
            mode="lines",
            fill="tonexty",
            fillcolor=fillcolor,
            line=dict(width=0, color=color),
            name=name,
            hoverinfo="skip",
        ))

    _add_distance_band(
        pos_gamma_peak,
        0.0015,
        "Max +γ proximity band (+/-0.15%)",
        _GREEN,
        "rgba(0,217,126,0.10)",
    )
    _add_distance_band(gamma_flip, 0.0025, "Gamma flip band (+/-0.25%)", "#e2e8f0", "rgba(226,232,240,0.08)")
    _add_distance_band(put_wall, 0.0015, "Put wall band (+/-0.15%)", _RED, "rgba(255,71,87,0.08)")

    if any(v is not None for v in put_wall):
        fig.add_trace(go.Scatter(
            x=labels,
            y=put_wall,
            mode="lines",
            line=dict(color=_RED, width=1.4, dash="dashdot"),
            marker=dict(size=4, color=_RED),
            name="Put wall",
            hovertemplate="%{x}<br>Put wall %{y:.0f}<extra></extra>",
        ))

    if any(v is not None for v in gamma_flip):
        fig.add_trace(go.Scatter(
            x=labels,
            y=gamma_flip,
            mode="lines",
            line=dict(color="#e2e8f0", width=1.6, dash="dot"),
            marker=dict(size=5, color="#e2e8f0", line=dict(color=_CHART_BG, width=1)),
            name="Gamma flip",
            hovertemplate="%{x}<br>Gamma flip %{y:.0f}<extra></extra>",
        ))

    if any(v is not None for v in pos_gamma_peak):
        fig.add_trace(go.Scatter(
            x=labels,
            y=pos_gamma_peak,
            mode="lines+markers",
            line=dict(color=_GREEN, width=2.6),
            marker=dict(size=7, color=_GREEN, line=dict(color=_CHART_BG, width=1)),
            name="Max +γ strike",
            hovertemplate="%{x}<br>Max +γ strike %{y:.0f}<extra></extra>",
        ))

    if any(v is not None for v in spot):
        def _distance_text(distance: float | None) -> str:
            return f"{distance:+.0f} pts" if distance is not None else "N/A"

        spot_custom = [
            [
                totals[idx],
                near_term_ratio[idx],
                regimes[idx],
                f"{pos_gamma_peak[idx]:.0f}" if pos_gamma_peak[idx] is not None else "N/A",
                _distance_text(
                    (spot[idx] - pos_gamma_peak[idx])
                    if spot[idx] is not None and pos_gamma_peak[idx] is not None
                    else None
                ),
                _distance_text(
                    (spot[idx] - gamma_flip[idx])
                    if spot[idx] is not None and gamma_flip[idx] is not None
                    else None
                ),
                _distance_text(
                    (spot[idx] - call_wall[idx])
                    if spot[idx] is not None and call_wall[idx] is not None
                    else None
                ),
                _distance_text(
                    (spot[idx] - put_wall[idx])
                    if spot[idx] is not None and put_wall[idx] is not None
                    else None
                ),
            ]
            for idx in range(len(labels))
        ]
        fig.add_trace(go.Scatter(
            x=labels,
            y=spot,
            customdata=spot_custom,
            mode="lines+markers",
            line=dict(color=_AMBER, width=3),
            marker=dict(size=8, color=_AMBER, line=dict(color=_CHART_BG, width=1.5)),
            name="SPX spot",
            hovertemplate=(
                "%{x}<br>SPX spot %{y:.2f}"
                "<br>Regime %{customdata[2]}"
                "<br>Max +γ strike %{customdata[3]}"
                "<br>Spot - max +γ %{customdata[4]}"
                "<br>Spot - flip %{customdata[5]}"
                "<br>Spot - call wall %{customdata[6]}"
                "<br>Spot - put wall %{customdata[7]}"
                "<br>Net GEX %{customdata[0]:+.3f} Bn$ / %"
                "<br>Near-term gamma share %{customdata[1]:.1%}<extra></extra>"
            ),
        ))

    level_values = [
        value
        for series in (spot, pos_gamma_peak, gamma_flip, call_wall, put_wall)
        for value in series
        if value is not None
    ]
    y_range = None
    if level_values:
        low, high = min(level_values), max(level_values)
        pad = max(20.0, (high - low) * 0.12)
        y_range = [low - pad, high + pad]

    _apply_base(
        fig,
        title=f"{ticker} · Spot vs Max +γ Strike",
        height=430,
        margin=dict(l=48, r=58, t=58, b=36),
        xaxis=dict(title="Snapshot", rangeslider=dict(visible=len(labels) > 4, thickness=0.08)),
        yaxis=dict(title="SPX spot and gamma strikes", zeroline=False, range=y_range),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def make_spx_price_chart(
    price_points: list[dict] | None,
    history=None,
    ticker: str = "SPX",
    gamma_flip: float | None = None,
    call_wall: float | None = None,
    put_wall: float | None = None,
    price_source: str | None = None,
) -> str | None:
    """Current SPX price line.

    Prefers live intraday prices (``price_points`` from UW websocket/OHLC); falls back to
    the per-snapshot spot series when the market feed is unavailable so the chart
    always renders. The most recent price is annotated as the current level.
    """
    x: list = []
    y: list[float] = []
    source = None
    if price_points:
        x = [p["ts"] for p in price_points]
        y = [safe_float(p["close"], 0.0) for p in price_points]
        source = "live"
    elif history:
        for row in history:
            spot = safe_float(row.get("spot"), 0.0)
            if spot > 0:
                x.append(row.get("ts_label"))
                y.append(spot)
        source = "snapshots"
    x = [xi for xi, yi in zip(x, y) if yi > 0]
    y = [yi for yi in y if yi > 0]
    if not y:
        return None

    if price_source:
        source = price_source

    current = y[-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line=dict(color=_AMBER, width=2.4),
        fill="tozeroy",
        fillcolor="rgba(245,158,11,0.08)",
        name="SPX price",
        hovertemplate="%{x}<br>SPX %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[x[-1]],
        y=[current],
        mode="markers+text",
        marker=dict(size=9, color=_AMBER, line=dict(color=_CHART_BG, width=1.5)),
        text=[f"{current:,.2f}"],
        textposition="middle left",
        textfont=dict(color=_AMBER, size=13),
        name="Current",
        hovertemplate="Current SPX %{y:.2f}<extra></extra>",
        showlegend=False,
    ))

    level_values = [
        safe_float(gamma_flip, 0.0),
        safe_float(call_wall, 0.0),
        safe_float(put_wall, 0.0),
    ]
    level_values = [v for v in level_values if v > 0]
    y_min = min(y)
    y_max = max(y)
    if level_values:
        y_min = min(y_min, min(level_values))
        y_max = max(y_max, max(level_values))

    for level, label, color in (
        (gamma_flip, "Flip", "#e2e8f0"),
        (call_wall, "Call wall", _GREEN),
        (put_wall, "Put wall", _RED),
    ):
        value = safe_float(level, 0.0)
        if value:
            fig.add_hline(
                y=value,
                line=dict(color=color, dash="dot", width=1.2),
                annotation_text=f"{label} {value:.0f}",
                annotation_position="right",
                annotation_font_color=color,
            )

    subtitle_map = {
        "live": "live · Unusual Whales",
        "uw-live": "live · Unusual Whales websocket",
        "uw-live+snapshots": "UW live + 90d backfill",
        "live+snapshots": "live + 90d backfill",
        "snapshots": "from saved snapshots",
    }
    subtitle = subtitle_map.get(source or "", "from saved snapshots")
    pad = max(5.0, (y_max - y_min) * 0.12)
    _apply_base(
        fig,
        title=f"{ticker} · Current Price ({subtitle})",
        height=320,
        margin=dict(l=48, r=70, t=58, b=36),
        xaxis=dict(title="Time", rangeslider=dict(visible=len(x) > 4, thickness=0.08)),
        yaxis=dict(title="SPX price", zeroline=False, range=[y_min - pad, y_max + pad]),
        hovermode="x unified",
        showlegend=False,
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
    title: str = "Gamma Exposure Map",
    window_pct: float = 0.035,
    max_bars: int = 90,
    cumulative_series: pd.Series | None = None,
    gamma_flip: float | None = None,
    call_wall: float | None = None,
    put_wall: float | None = None,
) -> str | None:
    if strike_series is None:
        return None
    strike = pd.Series(strike_series, dtype=float)
    if strike.empty:
        return None
    strike = strike.sort_index()
    if spot is not None and spot > 0:
        lo, hi = spot * (1 - window_pct), spot * (1 + window_pct)
        window = strike.loc[(strike.index >= lo) & (strike.index <= hi)]
        if len(window) < 5:
            window = strike
    else:
        window = strike
    window = window.sort_index(ascending=True)
    if len(window) > max_bars:
        step = max(1, len(window) // max_bars)
        sampled = window.iloc[::step]
        peaks = window.abs().sort_values(ascending=False).head(max(24, max_bars // 4))
        keep = sampled.index.union(peaks.index)
        for level in (spot, gamma_flip, call_wall, put_wall):
            if level is None:
                continue
            try:
                level_value = float(level)
            except (TypeError, ValueError):
                continue
            nearest_idx = (window.index.to_series().astype(float) - level_value).abs().idxmin()
            keep = keep.union(pd.Index([nearest_idx]))
        window = window.loc[keep].sort_index(ascending=True)

    strikes = [float(s) for s in window.index]
    gex_values = [float(v) for v in window.values]
    colors = [_GREEN if v >= 0 else _RED for v in gex_values]
    bar_width = _bar_width(strikes)

    cumulative = (
        pd.Series(cumulative_series, dtype=float).sort_index()
        if cumulative_series is not None
        else strike.cumsum()
    )
    cumulative_window = cumulative.reindex(window.index)
    if cumulative_window.isna().any():
        cumulative_window = strike.reindex(window.index).cumsum()

    fig = go.Figure(go.Bar(
        name="Net GEX by strike",
        x=strikes,
        y=gex_values,
        width=bar_width,
        marker_color=colors,
        marker_line_width=0,
        opacity=0.86,
        hovertemplate="Strike %{x:.0f}<br>Net GEX %{y:.3f} Bn$ / %<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        name="Cumulative GEX",
        x=strikes,
        y=[float(v) for v in cumulative_window.values],
        yaxis="y2",
        mode="lines",
        line=dict(color=_BLUE, width=2.4),
        hovertemplate="Strike %{x:.0f}<br>Cumulative %{y:.3f} Bn$ / %<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.25)", line_width=1)

    x_min, x_max = min(strikes), max(strikes)

    def _add_level(level: float | None, label: str, color: str, dash: str = "dash") -> None:
        if level is None:
            return
        try:
            value = float(level)
        except (TypeError, ValueError):
            return
        if value < x_min or value > x_max:
            return
        fig.add_vline(
            x=value,
            line=dict(color=color, dash=dash, width=1.6),
            annotation_text=f"{label} {value:.0f}",
            annotation_position="top",
            annotation_font_color=color,
        )

    _add_level(spot, "Spot", _AMBER)
    _add_level(gamma_flip, "Flip", "#e2e8f0", "dot")
    _add_level(call_wall, "Call wall", _GREEN, "dashdot")
    _add_level(put_wall, "Put wall", _RED, "dashdot")

    _apply_base(
        fig,
        title=f"{ticker} · {title}",
        height=520,
        margin=dict(l=48, r=58, t=72, b=52),
        xaxis=dict(
            title="SPX strike",
            range=[x_min, x_max],
            rangeslider=dict(visible=True, thickness=0.08),
            showspikes=True,
            spikecolor="rgba(255,255,255,0.25)",
        ),
        yaxis=dict(title="Net GEX (Bn$ / %)", zerolinecolor="rgba(255,255,255,0.20)"),
        yaxis2=dict(
            title="Cumulative GEX",
            overlaying="y",
            side="right",
            gridcolor="rgba(255,255,255,0)",
            zeroline=False,
        ),
        bargap=0.0,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def make_positive_strike_chart(strike_series: pd.Series | None, ticker: str, title: str) -> str | None:
    strike = _positive_gamma_view(strike_series)
    if strike.empty:
        return None
    strikes = [float(x) for x in strike.index]
    fig = go.Figure(go.Bar(
        x=strikes, y=strike.values.tolist(),
        width=_bar_width(strikes),
        marker_color=_GREEN, marker_line_width=0,
    ))
    _apply_base(fig, title=f"{ticker} · {title}", height=300,
                xaxis_title="Strike", yaxis_title="Positive GEX (Bn$ / %)",
                yaxis=dict(rangemode="tozero"), bargap=0.0)
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def make_0dte_movement_chart(
    current: dict | None,
    previous: dict | None,
    ticker: str,
    spot: float | None = None,
    window_pct: float = 0.025,
    max_bars: int = 80,
) -> str | None:
    """Same-day strike-level GEX movement, prioritized for 0DTE/intraday monitoring."""
    if not current or not previous:
        return None
    cur = pd.Series(current.get("strike"), dtype=float).sort_index()
    prev = pd.Series(previous.get("strike"), dtype=float).sort_index()
    if cur.empty or prev.empty:
        return None

    delta = cur.subtract(prev, fill_value=0.0).sort_index()
    if spot is None or spot <= 0:
        spot = safe_float(current.get("spot"), 0.0)
    if spot > 0:
        lo, hi = spot * (1 - window_pct), spot * (1 + window_pct)
        window = delta.loc[(delta.index >= lo) & (delta.index <= hi)]
        if len(window) < 5:
            window = delta
    else:
        window = delta

    if len(window) > max_bars:
        top = window.abs().sort_values(ascending=False).head(max_bars)
        window = window.loc[top.index].sort_index()
    if window.empty:
        return None

    strikes = [float(s) for s in window.index]
    delta_values = [float(v) for v in window.values]
    cur_values = [float(cur.get(s, 0.0)) for s in strikes]
    prev_values = [float(prev.get(s, 0.0)) for s in strikes]
    colors = [_GREEN if v >= 0 else _RED for v in delta_values]

    fig = go.Figure(go.Bar(
        name="ΔGEX since prior same-day snapshot",
        x=strikes,
        y=delta_values,
        width=_bar_width(strikes, fill_ratio=0.90),
        marker_color=colors,
        marker_line_width=0,
        customdata=list(zip(prev_values, cur_values)),
        hovertemplate=(
            "Strike %{x:.0f}"
            "<br>ΔGEX %{y:+.3f} Bn$ / %"
            "<br>Previous %{customdata[0]:+.3f}"
            "<br>Current %{customdata[1]:+.3f}<extra></extra>"
        ),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.25)", line_width=1)
    if spot and spot > 0 and min(strikes) <= spot <= max(strikes):
        fig.add_vline(
            x=float(spot),
            line=dict(color=_AMBER, dash="dash", width=1.7),
            annotation_text=f"Spot {float(spot):.0f}",
            annotation_position="top",
            annotation_font_color=_AMBER,
        )

    _apply_base(
        fig,
        title=f"{ticker} · 0DTE Movement Priority",
        height=360,
        margin=dict(l=48, r=24, t=64, b=42),
        xaxis=dict(title="SPX strike", range=[min(strikes), max(strikes)]),
        yaxis=dict(title="ΔGEX vs prior same-day snapshot", zerolinecolor="rgba(255,255,255,0.20)"),
        bargap=0.0,
        showlegend=False,
    )
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def _top_strikes_by_magnitude(*series: pd.Series | None, top_n: int = 36) -> list[float]:
    combined = pd.Series(dtype=float)
    for s in series:
        if s is None or (isinstance(s, pd.Series) and s.empty):
            continue
        combined = combined.add(pd.Series(s, dtype=float), fill_value=0.0)
    if combined.empty:
        return []
    ranked = combined.reindex(combined.abs().sort_values(ascending=False).index).head(top_n)
    return [float(x) for x in sorted(ranked.index)]


def make_prediction_gamma_chart(
    knn_strike: pd.Series | None,
    combined_strike: pd.Series | None,
    flow_strike: pd.Series | None,
    ticker: str,
    spot: float | None = None,
) -> str | None:
    """Large chart: KNN gamma forecast by strike with option-flow ΔGEX overlay."""
    knn = pd.Series(knn_strike, dtype=float) if knn_strike is not None else pd.Series(dtype=float)
    flow = pd.Series(flow_strike, dtype=float) if flow_strike is not None else pd.Series(dtype=float)
    combined = pd.Series(combined_strike, dtype=float) if combined_strike is not None else knn.add(flow, fill_value=0.0)
    if combined.empty and knn.empty and flow.empty:
        return None

    strikes = _top_strikes_by_magnitude(knn, flow, combined)
    if not strikes:
        return None

    knn_vals = [float(knn.get(s, 0.0)) for s in strikes]
    flow_vals = [float(flow.get(s, 0.0)) for s in strikes]
    combined_vals = [float(combined.get(s, 0.0)) for s in strikes]
    has_flow = any(abs(v) > 1e-9 for v in flow_vals)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="KNN forecast",
        x=strikes,
        y=knn_vals,
        marker_color=[_GREEN if v >= 0 else _RED for v in knn_vals],
        marker_line_width=0,
    ))
    if has_flow:
        fig.add_trace(go.Bar(
            name="Flow ΔGEX",
            x=strikes,
            y=flow_vals,
            marker_color=[_BLUE if v >= 0 else _AMBER for v in flow_vals],
            marker_line_width=0,
            opacity=0.85,
        ))
    fig.add_trace(go.Scatter(
        name="Combined forecast",
        x=strikes,
        y=combined_vals,
        mode="lines+markers",
        line=dict(color="#e2e8f0", width=2),
        marker=dict(size=6, color="#e2e8f0"),
    ))
    if spot is not None and spot > 0:
        fig.add_vline(
            x=float(spot),
            line=dict(color=_AMBER, dash="dash", width=1.5),
            annotation_text=f"Spot {int(spot)}",
            annotation_position="top",
        )

    fig.update_layout(barmode="group")
    _apply_base(
        fig,
        title=f"{ticker} · Predicted Gamma Change (KNN + Option Flow)",
        height=560,
        xaxis_title="Strike",
        yaxis_title="GEX (Bn$ / %)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=48, r=24, t=72, b=48),
    )
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
