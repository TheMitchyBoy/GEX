"""
Unusual Whales Periscope market-exposure charts.

Fresh Plotly builders matching https://unusualwhales.com/periscope/market-exposure :
  - top-left: SPX intraday price + replay slice highlight
  - top-right: MM exposure profile (~50 spot-exposure strikes, white prior dots)
  - bottom-left: MM positions summary
  - bottom-right: extended exposure (wider greek-exposure chain)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

from gex_core.features import select_atm_strike_series

# ── UW Periscope theme ───────────────────────────────────────────────────────
_BG = "#0a0e14"
_GREEN = "#10b981"
_RED = "#ef4444"
_SPOT = "#eab308"
_MUTED = "#64748b"
_LABEL = "#94a3b8"
_GRID = "rgba(148, 163, 184, 0.07)"
_ZERO = "rgba(255, 255, 255, 0.38)"

_EXPOSURE_LABEL = {"gamma": "Gamma", "vanna": "Vanna", "charm": "Charm"}
_PROFILE_MAX = 55
_EXTENDED_MAX = 96


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _encode(fig: go.Figure) -> str:
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def _base_layout(*, height: int, show_legend: bool = False) -> dict[str, Any]:
    return dict(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family="Inter, system-ui, sans-serif", size=11, color="#cbd5e1"),
        margin=dict(l=48, r=12, t=4, b=32),
        height=height,
        showlegend=show_legend,
    )


def _exposure_figure(
    series: pd.Series,
    *,
    spot: float | None,
    previous: pd.Series | None,
    height: int,
) -> go.Figure | None:
    """Horizontal strike profile — green/red bars + white prior-slice dots."""
    if series is None or series.empty:
        return None

    strikes = [float(s) for s in series.index]
    labels = [f"{s:.0f}" for s in strikes]
    values = [float(v) for v in series.values]
    colors = [_GREEN if v >= 0 else _RED for v in values]

    prev: dict[float, float] = {}
    if previous is not None and not previous.empty:
        prev = {float(k): float(v) for k, v in pd.Series(previous, dtype=float).items()}

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            opacity=0.93,
            hovertemplate="Strike %{y}<br>Net %{x:.3f} Bn<extra></extra>",
        )
    )

    prev_x = [prev.get(s) for s in strikes]
    if any(v is not None for v in prev_x):
        fig.add_trace(
            go.Scatter(
                y=labels,
                x=prev_x,
                mode="markers",
                marker=dict(color="#ffffff", size=7, line=dict(color=_BG, width=1.5)),
                hovertemplate="Strike %{y}<br>Prior 10m %{x:.3f} Bn<extra></extra>",
            )
        )

    if spot and spot > 0:
        nearest = f"{min(strikes, key=lambda s: abs(s - float(spot))):.0f}"
        fig.add_hline(y=nearest, line=dict(color=_SPOT, width=1.5, dash="dot"))

    fig.update_layout(
        **_base_layout(height=height),
        bargap=0.06,
        xaxis=dict(
            zeroline=True,
            zerolinecolor=_ZERO,
            zerolinewidth=1.5,
            gridcolor=_GRID,
            tickfont=dict(size=10, color=_MUTED),
        ),
        yaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=labels,
            gridcolor=_GRID,
            tickfont=dict(size=10, color=_LABEL),
        ),
    )
    return fig


def price_chart(
    price_points: list[dict[str, Any]] | None,
    *,
    ticker: str = "SPX",
    spot: float | None = None,
    highlight_label: str | None = None,
) -> str | None:
    """Top-left SPX price panel."""
    if not price_points:
        return None

    x = [p.get("ts") or p.get("time") for p in price_points]
    y = [_f(p.get("close") or p.get("price")) for p in price_points]
    pairs = [(a, b) for a, b in zip(x, y) if b > 0]
    if not pairs:
        return None
    x, y = zip(*pairs)

    fig = go.Figure(
        go.Scatter(
            x=list(x),
            y=list(y),
            mode="lines",
            line=dict(color=_SPOT, width=2),
            fill="tozeroy",
            fillcolor="rgba(234, 179, 8, 0.07)",
            hovertemplate="%{x}<br>SPX %{y:.2f}<extra></extra>",
        )
    )

    if highlight_label:
        labels = list(x)
        idx = next(
            (i for i, label in enumerate(labels) if str(label).startswith(str(highlight_label)[:16])),
            None,
        )
        if idx is None and highlight_label in labels:
            idx = labels.index(highlight_label)
        if idx is not None:
            fig.add_vrect(
                x0=idx - 0.45,
                x1=idx + 0.45,
                fillcolor="rgba(59, 130, 246, 0.25)",
                line_width=0,
                layer="below",
            )

    if spot and spot > 0:
        fig.add_hline(y=spot, line=dict(color="rgba(148,163,184,0.45)", dash="dot", width=1))

    fig.update_layout(
        **_base_layout(height=420),
        xaxis=dict(showgrid=False, tickfont=dict(size=10, color=_MUTED)),
        yaxis=dict(gridcolor=_GRID, tickfont=dict(size=10, color=_MUTED), zeroline=False),
    )
    return _encode(fig)


def exposure_profile_chart(
    series: pd.Series | None,
    *,
    spot: float | None,
    previous: pd.Series | None = None,
) -> str | None:
    """Top-right ~50-strike MM exposure profile."""
    fig = _exposure_figure(series if series is not None else pd.Series(dtype=float), spot=spot, previous=previous, height=420)
    return _encode(fig) if fig else None


def exposure_extended_chart(
    series: pd.Series | None,
    *,
    spot: float | None,
    previous: pd.Series | None = None,
) -> str | None:
    """Bottom-right extended strike profile."""
    raw = pd.Series(series, dtype=float).sort_index() if series is not None else pd.Series(dtype=float)
    if len(raw) > _EXTENDED_MAX and spot and spot > 0:
        raw = select_atm_strike_series(raw, spot, window_pct=0.09, min_strikes=15, max_strikes=_EXTENDED_MAX)
    fig = _exposure_figure(raw, spot=spot, previous=previous, height=max(580, min(760, 11 * len(raw) + 64)))
    return _encode(fig) if fig else None


def positions_chart(positions: dict[str, float] | None, *, ticker: str = "SPX") -> str | None:
    """Bottom-left MM positions — horizontal bars like UW summary panel."""
    if not positions:
        return None

    rows = [
        ("Call GEX", _f(positions.get("net_call_gex_bn"))),
        ("Put GEX", _f(positions.get("net_put_gex_bn"))),
        ("Call Δ", _f(positions.get("net_call_delta_bn"))),
        ("Put Δ", _f(positions.get("net_put_delta_bn"))),
    ]
    if not any(abs(v) > 1e-9 for _, v in rows):
        return None

    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [_GREEN if v >= 0 else _RED for v in values]

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="%{y}<br>%{x:.3f} Bn<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color=_ZERO, width=1.5))
    fig.update_layout(
        **_base_layout(height=420),
        bargap=0.2,
        xaxis=dict(gridcolor=_GRID, zeroline=False, tickfont=dict(size=10, color=_MUTED)),
        yaxis=dict(tickfont=dict(size=10, color=_LABEL)),
    )
    return _encode(fig)


@dataclass(frozen=True)
class PeriscopeChartBundle:
    price: str | None
    exposures: str | None
    positions: str | None
    exposures_extended: str | None


def build_periscope_charts(
    *,
    ticker: str,
    exposure_type: str,
    spot: float | None,
    exposure_profile: pd.Series | None,
    exposure_extended: pd.Series | None,
    previous_exposure: pd.Series | None,
    price_points: list[dict[str, Any]] | None,
    highlight_label: str | None,
    mm_positions: dict[str, float] | None,
) -> PeriscopeChartBundle:
    """Build all four UW Periscope panels from assembled context data."""
    profile = pd.Series(exposure_profile, dtype=float).sort_index() if exposure_profile is not None else pd.Series(dtype=float)
    if len(profile) > _PROFILE_MAX and spot and spot > 0:
        profile = select_atm_strike_series(profile, spot, window_pct=0.045, min_strikes=10, max_strikes=_PROFILE_MAX)

    extended = pd.Series(exposure_extended, dtype=float).sort_index() if exposure_extended is not None else pd.Series(dtype=float)

    return PeriscopeChartBundle(
        price=price_chart(price_points, ticker=ticker, spot=spot, highlight_label=highlight_label),
        exposures=exposure_profile_chart(profile, spot=spot, previous=previous_exposure),
        positions=positions_chart(mm_positions, ticker=ticker),
        exposures_extended=exposure_extended_chart(extended, spot=spot, previous=previous_exposure),
    )
