"""
Periscope charts tuned for options / dealer-gamma trading.

Four focused panels:
  1. Session price
  2. Net exposure by strike (ATM vertical profile + key levels)
  3. 10-minute change vs prior slice
  4. Cumulative exposure (gamma flip visualization)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

from gex_core.features import estimate_gamma_flip, select_atm_strike_series

# Theme
_BG = "#080b10"
_PANEL = "#0d1117"
_GREEN = "#22c55e"
_RED = "#ef4444"
_AMBER = "#f59e0b"
_BLUE = "#38bdf8"
_MUTED = "#64748b"
_TEXT = "#cbd5e1"
_GRID = "rgba(148, 163, 184, 0.08)"
_ZERO = "rgba(255, 255, 255, 0.35)"

_EXPOSURE_TITLE = {"gamma": "Net Gamma", "vanna": "Net Vanna", "charm": "Net Charm"}
_PROFILE_BARS = 34
_EXTENDED_BARS = 52


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _encode(fig: go.Figure) -> str:
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def _layout(*, height: int, title: str = "") -> dict[str, Any]:
    return dict(
        paper_bgcolor=_BG,
        plot_bgcolor=_PANEL,
        font=dict(family="Inter, system-ui, sans-serif", size=11, color=_TEXT),
        margin=dict(l=52, r=20, t=28 if title else 12, b=44),
        height=height,
        title=dict(text=title, x=0.02, font=dict(size=12, color=_MUTED)) if title else None,
        showlegend=False,
    )


def _atm_series(series: pd.Series, spot: float | None, *, max_bars: int = _PROFILE_BARS) -> pd.Series:
    s = pd.Series(series, dtype=float).sort_index()
    if s.empty:
        return s
    if spot and spot > 0:
        return select_atm_strike_series(s, spot, window_pct=0.04, min_strikes=8, max_strikes=max_bars)
    return s.head(max_bars)


def _bar_width(strikes: list[float]) -> float | None:
    if len(strikes) < 2:
        return None
    diffs = sorted(set(strikes))
    gaps = [b - a for a, b in zip(diffs, diffs[1:]) if b > a]
    if not gaps:
        return None
    return max(1.0, float(np.median(gaps)) * 0.82)


def _strike_tick_step(strikes: list[float]) -> float | None:
    if len(strikes) < 4:
        return None
    step = _bar_width(strikes)
    if step is None:
        return None
    if step <= 5:
        return 10.0
    if step <= 10:
        return 25.0
    return 50.0


def _symmetric_y_range(values: list[float], pad: float = 1.15) -> list[float]:
    if not values:
        return [-1, 1]
    m = max(abs(min(values)), abs(max(values)), 0.05)
    m *= pad
    return [-m, m]


def _add_strike_levels(
    fig: go.Figure,
    *,
    spot: float | None,
    gamma_flip: float | None,
    call_wall: float | None,
    put_wall: float | None,
    x_min: float,
    x_max: float,
) -> None:
    def _vline(x: float | None, name: str, color: str, dash: str = "dash") -> None:
        if x is None:
            return
        xv = _f(x, 0.0)
        if xv < x_min or xv > x_max:
            return
        fig.add_vline(x=xv, line=dict(color=color, width=1.5, dash=dash))
        fig.add_annotation(
            x=xv,
            y=1.04,
            xref="x",
            yref="paper",
            text=name,
            showarrow=False,
            font=dict(size=9, color=color),
        )

    _vline(spot, "Spot", _AMBER, "solid")
    _vline(gamma_flip, "Flip", "#e2e8f0", "dot")
    _vline(call_wall, "+γ", _GREEN, "dashdot")
    _vline(put_wall, "−γ", _RED, "dashdot")


def session_price_chart(
    price_points: list[dict[str, Any]] | None,
    *,
    spot: float | None = None,
    highlight_label: str | None = None,
) -> str | None:
    if not price_points:
        return None
    x_raw = [p.get("ts") or p.get("time") for p in price_points]
    y = [_f(p.get("close") or p.get("price")) for p in price_points]
    pairs = [(a, b) for a, b in zip(x_raw, y) if b > 0]
    if not pairs:
        return None
    x_raw, y = zip(*pairs)
    x = list(range(len(x_raw)))
    labels = list(x_raw)

    fig = go.Figure(
        go.Scatter(
            x=x,
            y=list(y),
            mode="lines",
            line=dict(color=_AMBER, width=2.2),
            hovertemplate="%{customdata}<br>SPX %{y:.2f}<extra></extra>",
            customdata=labels,
        )
    )

    if highlight_label:
        idx = next(
            (i for i, lab in enumerate(labels) if str(lab).startswith(str(highlight_label)[:16])),
            None,
        )
        if idx is None and highlight_label in labels:
            idx = labels.index(highlight_label)
        if idx is not None:
            fig.add_vrect(
                x0=idx - 0.5,
                x1=idx + 0.5,
                fillcolor="rgba(56, 189, 248, 0.18)",
                line_width=0,
                layer="below",
            )

    y_min, y_max = min(y), max(y)
    pad = max((y_max - y_min) * 0.08, 2.0)
    fig.update_layout(
        **_layout(height=380, title="Session price"),
        xaxis=dict(
            tickmode="array",
            tickvals=x[:: max(1, len(x) // 6)],
            ticktext=[labels[i] for i in x[:: max(1, len(x) // 6)]],
            tickfont=dict(size=9, color=_MUTED),
            showgrid=False,
        ),
        yaxis=dict(range=[y_min - pad, y_max + pad], gridcolor=_GRID, tickfont=dict(size=10, color=_MUTED)),
    )
    if spot and spot > 0:
        fig.add_hline(y=spot, line=dict(color="rgba(148,163,184,0.4)", dash="dot", width=1))
    return _encode(fig)


def exposure_by_strike_chart(
    series: pd.Series | None,
    *,
    spot: float | None,
    previous: pd.Series | None = None,
    exposure_type: str = "gamma",
    gamma_flip: float | None = None,
    call_wall: float | None = None,
    put_wall: float | None = None,
    max_bars: int = _PROFILE_BARS,
) -> str | None:
    """Vertical dealer exposure profile — the primary trading view."""
    if series is None or series.empty:
        return None

    window = _atm_series(series, spot, max_bars=max_bars)
    if window.empty:
        return None

    strikes = [float(s) for s in window.index]
    values = [float(v) for v in window.values]
    colors = [_GREEN if v >= 0 else _RED for v in values]
    width = _bar_width(strikes)

    prev = pd.Series(previous, dtype=float) if previous is not None else pd.Series(dtype=float)
    prev_y = [float(prev.get(s, np.nan)) for s in strikes]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=strikes,
            y=values,
            width=width,
            marker=dict(color=colors, line=dict(width=0)),
            opacity=0.9,
            name="Current",
            hovertemplate="Strike %{x:.0f}<br>Net %{y:.3f} Bn<extra></extra>",
        )
    )
    if any(np.isfinite(v) for v in prev_y):
        fig.add_trace(
            go.Scatter(
                x=strikes,
                y=prev_y,
                mode="markers",
                marker=dict(color="#ffffff", size=7, line=dict(color=_PANEL, width=1.5)),
                name="Prior 10m",
                hovertemplate="Strike %{x:.0f}<br>Prior %{y:.3f} Bn<extra></extra>",
            )
        )

    dtick = _strike_tick_step(strikes)
    xaxis = dict(
        gridcolor=_GRID,
        tickfont=dict(size=10, color=_MUTED),
        tickformat=",.0f",
    )
    if dtick:
        xaxis["dtick"] = dtick

    fig.update_layout(
        **_layout(height=400, title=_EXPOSURE_TITLE.get(exposure_type, "Exposure")),
        bargap=0.04,
        xaxis=xaxis,
        yaxis=dict(
            title=dict(text="Bn$ / 1% move", font=dict(size=10, color=_MUTED)),
            zeroline=True,
            zerolinecolor=_ZERO,
            zerolinewidth=1.5,
            gridcolor=_GRID,
            range=_symmetric_y_range(values + [v for v in prev_y if np.isfinite(v)]),
        ),
    )
    _add_strike_levels(
        fig,
        spot=spot,
        gamma_flip=gamma_flip,
        call_wall=call_wall,
        put_wall=put_wall,
        x_min=min(strikes),
        x_max=max(strikes),
    )
    return _encode(fig)


def exposure_change_chart(
    current: pd.Series | None,
    previous: pd.Series | None,
    *,
    spot: float | None,
    exposure_type: str = "gamma",
) -> str | None:
    """10-minute Δ exposure — highlights where dealer positioning shifted."""
    if current is None or current.empty:
        return None
    cur = _atm_series(current, spot)
    prev = pd.Series(previous, dtype=float) if previous is not None else pd.Series(dtype=float)
    if cur.empty:
        return None

    delta = cur.subtract(prev.reindex(cur.index), fill_value=0.0)
    strikes = [float(s) for s in delta.index]
    values = [float(v) for v in delta.values]
    colors = [_GREEN if v >= 0 else _RED for v in values]
    width = _bar_width(strikes)

    fig = go.Figure(
        go.Bar(
            x=strikes,
            y=values,
            width=width,
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="Strike %{x:.0f}<br>Δ %{y:+.3f} Bn<extra></extra>",
        )
    )
    dtick = _strike_tick_step(strikes)
    xaxis = dict(gridcolor=_GRID, tickfont=dict(size=10, color=_MUTED), tickformat=",.0f")
    if dtick:
        xaxis["dtick"] = dtick

    fig.update_layout(
        **_layout(height=360, title=f"10m change · {_EXPOSURE_TITLE.get(exposure_type, 'Exposure')}"),
        bargap=0.04,
        xaxis=xaxis,
        yaxis=dict(
            zeroline=True,
            zerolinecolor=_ZERO,
            gridcolor=_GRID,
            range=_symmetric_y_range(values),
        ),
    )
    if spot and spot > 0 and min(strikes) <= spot <= max(strikes):
        fig.add_vline(x=float(spot), line=dict(color=_AMBER, width=1.5, dash="dot"))
    return _encode(fig)


def cumulative_exposure_chart(
    series: pd.Series | None,
    *,
    spot: float | None,
    gamma_flip: float | None = None,
    max_bars: int = _EXTENDED_BARS,
) -> str | None:
    """Cumulative exposure vs strike — makes the gamma flip level obvious."""
    if series is None or series.empty:
        return None

    window = _atm_series(series, spot, max_bars=max_bars)
    if window.empty:
        return None

    strikes = [float(s) for s in window.index]
    cumulative = window.cumsum()
    y = [float(v) for v in cumulative.values]
    flip = gamma_flip
    if flip is None:
        flip = estimate_gamma_flip(cumulative)

    fill = _GREEN if y[-1] >= 0 else _RED
    fig = go.Figure(
        go.Scatter(
            x=strikes,
            y=y,
            mode="lines",
            line=dict(color=_BLUE, width=2.5),
            fill="tozeroy",
            fillcolor=f"rgba(56, 189, 248, 0.12)",
            hovertemplate="Strike %{x:.0f}<br>Cumul %{y:.3f} Bn<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color=_ZERO, width=1.5))

    dtick = _strike_tick_step(strikes)
    xaxis = dict(gridcolor=_GRID, tickfont=dict(size=10, color=_MUTED), tickformat=",.0f")
    if dtick:
        xaxis["dtick"] = dtick

    fig.update_layout(
        **_layout(height=400, title="Cumulative exposure (flip = zero cross)"),
        xaxis=xaxis,
        yaxis=dict(gridcolor=_GRID, zeroline=False),
    )
    if spot and spot > 0:
        fig.add_vline(x=float(spot), line=dict(color=_AMBER, width=1.5, dash="dot"))
    if flip is not None:
        fig.add_vline(x=float(flip), line=dict(color="#e2e8f0", width=1.5, dash="dot"))
    return _encode(fig)


def dealer_positions_chart(positions: dict[str, float] | None) -> str | None:
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
            x=labels,
            y=values,
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="%{x}<br>%{y:.3f} Bn<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color=_ZERO, width=1.5))
    fig.update_layout(
        **_layout(height=320, title="Dealer positioning (net)"),
        bargap=0.3,
        xaxis=dict(tickfont=dict(size=10, color=_MUTED)),
        yaxis=dict(
            gridcolor=_GRID,
            title=dict(text="Bn$", font=dict(size=10, color=_MUTED)),
        ),
    )
    return _encode(fig)


@dataclass(frozen=True)
class PeriscopeChartBundle:
    price: str | None
    exposures: str | None
    change: str | None
    cumulative: str | None
    positions: str | None = None


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
    gamma_flip: float | None = None,
    call_wall: float | None = None,
    put_wall: float | None = None,
) -> PeriscopeChartBundle:
    """Build the four trading-focused Periscope panels."""
    profile = pd.Series(exposure_profile, dtype=float).sort_index() if exposure_profile is not None else pd.Series(dtype=float)
    extended = pd.Series(exposure_extended, dtype=float).sort_index() if exposure_extended is not None else profile
    cum_source = extended if len(extended) > len(profile) else profile

    return PeriscopeChartBundle(
        price=session_price_chart(price_points, spot=spot, highlight_label=highlight_label),
        exposures=exposure_by_strike_chart(
            profile,
            spot=spot,
            previous=previous_exposure,
            exposure_type=exposure_type,
            gamma_flip=gamma_flip,
            call_wall=call_wall,
            put_wall=put_wall,
        ),
        change=exposure_change_chart(profile, previous_exposure, spot=spot, exposure_type=exposure_type),
        cumulative=cumulative_exposure_chart(cum_source, spot=spot, gamma_flip=gamma_flip),
        positions=dealer_positions_chart(mm_positions),
    )
