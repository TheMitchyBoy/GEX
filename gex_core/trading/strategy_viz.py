"""Dashboard visualization for the gamma auto-trader strategy."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder

from gex_core.charts import _bar_width, _strike_axis_layout
from gex_core.features import parse_gamma_flip_value, safe_float, select_dense_atm_strike_series
from gex_core.trading.advisor import advise_entry
from gex_core.trading.config import (
    max_strike_distance_pct,
    partial_take_profit_pct,
    stop_loss_pct,
    take_profit_pct,
    trailing_stop_floor_pct,
    trailing_stop_trigger_pct,
)
from gex_core.trading.exits import build_exit_profile, effective_stop_loss
from gex_core.trading.filters import MarketContext, evaluate_entry_filters
from gex_core.trading.journal import get_performance_summary, list_open_trades, list_recent_trades
from gex_core.trading.paper_broker import estimate_option_pnl_pct
from gex_core.trading.low_gex_signals import (
    compute_high_gex_signal,
    compute_low_gex_signal,
    wall_entry_quality_ok,
)
from gex_core.trading.signals import compute_gamma_signals

_BG = "#080b10"
_PANEL = "#0d1117"
_GREEN = "#22c55e"
_RED = "#ef4444"
_AMBER = "#f59e0b"
_BLUE = "#38bdf8"
_PURPLE = "#a78bfa"
_MUTED = "#64748b"
_TEXT = "#cbd5e1"
_TRAIL_CYAN = "#67e8f9"
_TRAIL_VIOLET = "#c4b5fd"
_GAMMA_CHANGE_MIN = 0.015


def _encode(fig: go.Figure) -> str:
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def _symmetric_x_range(values: list[float], pad: float = 1.15) -> list[float]:
    if not values:
        return [-1.0, 1.0]
    peak = max((abs(float(v)) for v in values), default=0.05)
    peak = max(peak, 0.05) * pad
    return [-peak, peak]


def _chart_exposure_window(
    series: pd.Series | None,
    spot: float,
    *,
    window_pct: float = 0.04,
    max_strikes: int = 65,
) -> pd.Series:
    """Dense ATM strike ladder for the magnet map (no peak-skip gaps)."""
    if series is None or series.empty or spot <= 0:
        return pd.Series(dtype=float)
    return select_dense_atm_strike_series(
        pd.Series(series, dtype=float).sort_index(),
        spot,
        window_pct=window_pct,
        max_strikes=max_strikes,
    )


def _normalize_exposure_trail(
    exposure_trail: list[dict[str, Any]] | None,
    previous_exposure: pd.Series | None,
) -> list[dict[str, Any]]:
    if exposure_trail:
        return list(exposure_trail)
    if isinstance(previous_exposure, pd.Series) and not previous_exposure.empty:
        return [{"ts": "", "label": "prior", "spot": None, "series": previous_exposure, "age": 1}]
    return []


def _gamma_change_points(
    current: pd.Series,
    trail: list[dict[str, Any]],
    *,
    window_index: pd.Index,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Largest |Δγ| strikes between consecutive recent slices."""
    if current.empty or not len(trail):
        return []
    chain = [item.get("series") for item in trail if isinstance(item.get("series"), pd.Series)]
    chain.append(current)
    points: dict[tuple[float, int], dict[str, Any]] = {}
    for step_idx in range(1, len(chain)):
        prev = chain[step_idx - 1]
        cur = chain[step_idx]
        label = str(trail[step_idx - 1].get("label") or trail[step_idx - 1].get("ts") or f"t-{step_idx}")
        aligned_prev, aligned_cur = prev.align(cur, join="outer", fill_value=0.0)
        delta = aligned_cur - aligned_prev
        scoped = delta.reindex(window_index, fill_value=0.0)
        for strike, dv in scoped.abs().nlargest(top_n).items():
            change = float(scoped.get(strike, 0.0))
            if abs(change) < _GAMMA_CHANGE_MIN:
                continue
            key = (float(strike), step_idx)
            prior = points.get(key)
            if prior is None or abs(change) > abs(float(prior.get("delta", 0.0))):
                points[key] = {
                    "strike": float(strike),
                    "delta": change,
                    "gamma": float(aligned_cur.get(strike, 0.0)),
                    "label": label,
                    "step": step_idx,
                }
    return list(points.values())


def _prior_exposure_series(
    trail: list[dict[str, Any]],
    previous_exposure: pd.Series | None,
) -> pd.Series | None:
    if trail:
        last = trail[-1].get("series")
        if isinstance(last, pd.Series) and not last.empty:
            return last
    if isinstance(previous_exposure, pd.Series) and not previous_exposure.empty:
        return previous_exposure
    return None


def _gamma_delta_vs_prior(window: pd.Series, prior: pd.Series | None) -> pd.Series:
    if prior is None or prior.empty or window.empty:
        return pd.Series(0.0, index=window.index)
    aligned = window.align(prior, join="left", fill_value=0.0)
    return aligned[0] - aligned[1]


def _gamma_bar_marker_styles(window: pd.Series, prior: pd.Series | None) -> tuple[list[str], list[float]]:
    """Base sign colors with opacity heat from |Δγ| vs the prior slice."""
    deltas = _gamma_delta_vs_prior(window, prior)
    max_abs = float(deltas.abs().max()) if not deltas.empty else 0.0
    max_abs = max(max_abs, _GAMMA_CHANGE_MIN)
    colors: list[str] = []
    opacities: list[float] = []
    for val, delta in zip(window.values, deltas.values):
        base = _GREEN if float(val) >= 0 else _RED
        heat = min(1.0, abs(float(delta)) / max_abs)
        opacities.append(0.55 + 0.4 * heat)
        if heat >= 0.65:
            colors.append(_AMBER if float(delta) >= 0 else "#fb923c")
        else:
            colors.append(base)
    return colors, opacities


def _add_gamma_bars(
    fig: go.Figure,
    *,
    window: pd.Series,
    prior: pd.Series | None,
    row: int = 1,
    col: int = 1,
) -> None:
    strikes = [float(s) for s in window.index]
    colors, opacities = _gamma_bar_marker_styles(window, prior)
    deltas = _gamma_delta_vs_prior(window, prior)
    fig.add_trace(
        go.Bar(
            x=window.values,
            y=strikes,
            orientation="h",
            width=_bar_width(strikes),
            marker=dict(
                color=colors,
                opacity=opacities,
                line=dict(width=0.6, color="rgba(255,255,255,0.12)"),
            ),
            name="Gamma",
            meta={"layer": "gamma_bars"},
            customdata=[
                f"Strike {s:.0f}<br>γ {g:+.3f} Bn<br>Δγ {d:+.3f} Bn"
                for s, g, d in zip(strikes, window.values, deltas.values)
            ],
            hovertemplate="%{customdata}<extra></extra>",
        ),
        row=row,
        col=col,
    )


def _add_level_guide(
    fig: go.Figure,
    *,
    y: float,
    color: str,
    name: str,
    x_range: list[float],
    row: int = 1,
    col: int = 1,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=[y, y],
            mode="lines",
            line=dict(color=color, width=1, dash="dot" if "flip" in name.lower() else "dash"),
            name=name,
            hovertemplate=f"{name}<br>%{{y:.0f}}<extra></extra>",
            meta={"layer": "levels"},
        ),
        row=row,
        col=col,
    )


def _signal_strike_trail(
    trail: list[dict[str, Any]],
    *,
    wall_mode: bool,
    window_pct: float,
    ticker: str | None = None,
) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    for item in trail:
        series = item.get("series")
        if not isinstance(series, pd.Series) or series.empty:
            continue
        spot_val = safe_float(item.get("spot"), 0.0)
        if spot_val <= 0:
            spot_val = safe_float(series.attrs.get("spot"), 0.0)
        if wall_mode:
            pack = compute_low_gex_signal(series, spot=spot_val or None, window_pct=window_pct)
        else:
            pack = compute_gamma_signals(series, None, spot=spot_val or None, ticker=ticker)
        rec = pack.get("recommended") or {}
        strike = safe_float(rec.get("strike"), 0.0)
        if strike <= 0:
            continue
        markers.append(
            {
                "strike": strike,
                "label": str(item.get("label") or item.get("ts") or ""),
                "gamma": safe_float(rec.get("gamma_bn"), 0.0),
                "age": int(item.get("age") or 1),
            }
        )
    return markers


def _decorate_gex_strike_panel(
    fig: go.Figure,
    *,
    window: pd.Series,
    spot_val: float,
    x_range: list[float],
    levels: dict[str, Any] | None,
    exposure_trail: list[dict[str, Any]] | None,
    previous_exposure: pd.Series | None,
    wall_mode: bool,
    window_pct: float,
    ticker: str | None = None,
    row: int = 1,
    col: int = 1,
) -> None:
    """History dots, Δγ highlights, level guides, and signal trail."""
    if window.empty or spot_val <= 0:
        return

    trail = _normalize_exposure_trail(exposure_trail, previous_exposure)
    peak = max(abs(x_range[0]), abs(x_range[1]), 0.05)

    # Prior-slice γ positions (fading cyan dots).
    max_age = max((int(item.get("age") or 1) for item in trail), default=1)
    for item in trail:
        series = item.get("series")
        if not isinstance(series, pd.Series) or series.empty:
            continue
        age = int(item.get("age") or 1)
        opacity = 0.18 + 0.22 * (1.0 - (age - 1) / max(max_age, 1))
        label = str(item.get("label") or item.get("ts") or "")
        hist = series.reindex(window.index, fill_value=0.0)
        fig.add_trace(
            go.Scatter(
                x=hist.values,
                y=[float(s) for s in hist.index],
                mode="markers",
                marker=dict(
                    size=5.5,
                    color=_TRAIL_CYAN,
                    opacity=opacity,
                    line=dict(width=0.5, color="rgba(103,232,249,0.35)"),
                    symbol="circle",
                ),
                name=f"γ {label}",
                hovertemplate="%{customdata}<br>Strike %{y:.0f}<br>γ %{x:+.3f} Bn<extra></extra>",
                customdata=[label] * len(hist),
                meta={"layer": "gamma_history"},
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    # Δγ pulse dots on current profile.
    changes = _gamma_change_points(window, trail, window_index=window.index, top_n=10)
    if changes:
        fig.add_trace(
            go.Scatter(
                x=[p["gamma"] for p in changes],
                y=[p["strike"] for p in changes],
                mode="markers",
                marker=dict(
                    size=[6 + min(14.0, abs(p["delta"]) * 18.0) for p in changes],
                    color=[_GREEN if p["delta"] >= 0 else _RED for p in changes],
                    opacity=0.92,
                    line=dict(width=1.2, color="rgba(255,255,255,0.55)"),
                    symbol="circle",
                ),
                name="Δγ",
                customdata=[f"{p['label']} · Δγ {p['delta']:+.3f} Bn" for p in changes],
                hovertemplate="%{customdata}<br>Strike %{y:.0f}<br>γ %{x:+.3f} Bn<extra></extra>",
                meta={"layer": "delta_pulse"},
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    # Signal / wall migration path (violet breadcrumbs + connecting line).
    markers = _signal_strike_trail(trail, wall_mode=wall_mode, window_pct=window_pct, ticker=ticker)
    if len(markers) >= 2:
        ordered = sorted(markers, key=lambda m: -int(m.get("age") or 1))
        path_x = [-peak * 0.07 * int(m.get("age") or 1) for m in ordered]
        path_y = [float(m["strike"]) for m in ordered]
        fig.add_trace(
            go.Scatter(
                x=path_x,
                y=path_y,
                mode="lines+markers",
                line=dict(color="rgba(196,181,253,0.55)", width=2, dash="dot"),
                marker=dict(
                    size=8,
                    color=_TRAIL_VIOLET,
                    opacity=0.75,
                    line=dict(width=1, color="rgba(255,255,255,0.45)"),
                ),
                name="Signal path",
                customdata=[m.get("label", "") for m in ordered],
                hovertemplate="%{customdata}<br>Strike %{y:.0f}<extra></extra>",
                meta={"layer": "signal_trail"},
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    elif markers:
        marker = markers[0]
        age = int(marker.get("age") or 1)
        fig.add_trace(
            go.Scatter(
                x=[-peak * 0.07 * age],
                y=[marker["strike"]],
                mode="markers",
                marker=dict(size=8, color=_TRAIL_VIOLET, opacity=0.75),
                name="Signal trail",
                hovertemplate=f"{marker['label']}<br>%{{y:.0f}}<extra></extra>",
                meta={"layer": "signal_trail"},
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    gamma_flip = parse_gamma_flip_value((levels or {}).get("gamma_flip"))
    if gamma_flip is not None and gamma_flip > 0:
        _add_level_guide(
            fig,
            y=gamma_flip,
            color="rgba(167,139,250,0.45)",
            name="Gamma flip",
            x_range=x_range,
            row=row,
            col=col,
        )
    for level_key, color, name in (
        ("call_wall", "rgba(34,197,94,0.35)", "Call wall"),
        ("put_wall", "rgba(239,68,68,0.35)", "Put wall"),
    ):
        level = safe_float((levels or {}).get(level_key), 0.0)
        if level > 0:
            _add_level_guide(fig, y=level, color=color, name=name, x_range=x_range, row=row, col=col)

    fig.add_annotation(
        x=0.01,
        y=spot_val,
        xref="paper",
        yref="y",
        text=f"Spot {spot_val:,.0f}",
        showarrow=False,
        font=dict(size=9, color=_AMBER),
        bgcolor="rgba(8,11,16,0.75)",
        borderpad=2,
        xanchor="left",
    )


def build_strategy_state(
    *,
    ticker: str,
    spot: float | None,
    exposure: pd.Series | None,
    previous_exposure: pd.Series | None,
    snapshot: dict[str, Any] | None = None,
    prev_spot: float | None = None,
    uw_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Current signals, filter checklist, and open-position marks for the dashboard."""
    spot_val = safe_float(spot, 0.0)
    snap = snapshot or {}
    if spot_val <= 0:
        spot_val = safe_float(snap.get("spot"), 0.0)

    signals = compute_gamma_signals(exposure, previous_exposure, spot=spot_val, ticker=ticker)
    market = MarketContext(
        spot=spot_val,
        prev_spot=prev_spot,
        gamma_flip=parse_gamma_flip_value(snap.get("gamma_flip")),
        regime=str(snap.get("regime") or ""),
        is_cpi_day=bool(snap.get("is_cpi_day")),
        is_nfp_day=bool(snap.get("is_nfp_day")),
        is_fomc_week=bool(snap.get("is_fomc_week")),
        flow_net_delta_gex_bn=safe_float(snap.get("flow_net_delta_gex_bn"), 0.0) or None,
    )

    advice = {"approve": False, "reason": "No signal", "source": "none"}
    filters = {"approve": False, "reason": "No signal"}
    if signals.get("available"):
        filters = evaluate_entry_filters(signals, market=market, uw_bundle=uw_bundle)
        advice = advise_entry(ticker=ticker, signals=signals, uw_bundle=uw_bundle, market=market)

    open_positions = list_open_trades(ticker)
    marked_positions = []
    for pos in open_positions:
        pnl = None
        if spot_val > 0:
            pnl = estimate_option_pnl_pct(
                pos["option_type"],
                entry_spot=float(pos["entry_spot"]),
                current_spot=spot_val,
                strike=float(pos["strike"]),
            )
        stop = effective_stop_loss(entry_spot=float(pos["entry_spot"]), strike=float(pos["strike"]))
        marked_positions.append(
            {
                **pos,
                "pnl_pct": pnl,
                "stop_pct": -stop,
                "target_pct": take_profit_pct(),
            }
        )

    rec = signals.get("recommended") or {}
    profile = None
    if rec.get("strike") and spot_val > 0:
        profile = build_exit_profile(
            ai_confidence=float(advice.get("confidence", 0.5)),
            gamma_delta=float(rec.get("gamma_delta", 0)),
            regime=market.regime,
            entry_spot=spot_val,
            strike=float(rec["strike"]),
        )

    return {
        "ticker": ticker.upper(),
        "spot": spot_val,
        "signals": signals,
        "filters": filters,
        "advice": advice,
        "open_positions": marked_positions,
        "performance": get_performance_summary(ticker),
        "recent_trades": list_recent_trades(limit=15, ticker=ticker),
        "rules": {
            "stop_loss_pct": stop_loss_pct(),
            "take_profit_pct": take_profit_pct(),
            "partial_take_profit_pct": partial_take_profit_pct(),
            "trail_trigger_pct": trailing_stop_trigger_pct(),
            "trail_floor_pct": trailing_stop_floor_pct(),
            "max_strike_distance_pct": max_strike_distance_pct(),
        },
        "exit_profile": {
            "hold_for_target": profile.hold_for_target if profile else False,
            "partial_take_profit": profile.partial_take_profit if profile else partial_take_profit_pct(),
            "trail_trigger": profile.trail_trigger if profile else trailing_stop_trigger_pct(),
            "trail_floor": profile.trail_floor if profile else trailing_stop_floor_pct(),
        }
        if profile
        else None,
        "levels": {
            "gamma_flip": market.gamma_flip,
            "call_wall": snap.get("call_wall"),
            "put_wall": snap.get("put_wall"),
            "regime": market.regime,
        },
    }


def build_strategy_chart(
    *,
    spot: float | None,
    exposure: pd.Series | None,
    state: dict[str, Any],
    window_pct: float = 0.04,
    max_strikes: int = 65,
    exposure_trail: list[dict[str, Any]] | None = None,
    previous_exposure: pd.Series | None = None,
    wall_mode: bool = False,
) -> go.Figure:
    """Two-row dashboard: GEX by strike + cumulative PnL."""
    spot_val = safe_float(spot or state.get("spot"), 0.0)
    signals = state.get("signals") or {}
    performance = state.get("performance") or {}
    recent = state.get("recent_trades") or []
    levels = state.get("levels") or {}
    open_positions = state.get("open_positions") or []

    trail_count = len(_normalize_exposure_trail(exposure_trail, previous_exposure))
    strike_title = (
        f"GEX by strike · ±{window_pct * 100:.1f}% · last {trail_count} γ snapshots"
        if window_pct < 0.04
        else f"GEX by strike · last {trail_count} γ snapshots & signals"
    )
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=(strike_title, "Cumulative PnL (recent trades)"),
    )

    window = _chart_exposure_window(
        exposure if isinstance(exposure, pd.Series) else pd.Series(dtype=float),
        spot_val,
        window_pct=window_pct,
        max_strikes=max_strikes,
    )
    prior = _prior_exposure_series(
        _normalize_exposure_trail(exposure_trail, previous_exposure),
        previous_exposure,
    )
    x_range = _symmetric_x_range([float(v) for v in window.values]) if not window.empty else [-1.0, 1.0]
    if not window.empty and spot_val > 0:
        _add_gamma_bars(fig, window=window, prior=prior, row=1, col=1)
        _decorate_gex_strike_panel(
            fig,
            window=window,
            spot_val=spot_val,
            x_range=x_range,
            levels=levels,
            exposure_trail=exposure_trail,
            previous_exposure=previous_exposure,
            wall_mode=wall_mode,
            window_pct=window_pct,
            ticker=state.get("ticker"),
        )
        fig.add_vline(
            x=0,
            line=dict(color="rgba(226, 232, 240, 0.55)", width=1.5),
            row=1,
            col=1,
        )

    if spot_val > 0:
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=[spot_val, spot_val],
                mode="lines",
                line=dict(color=_AMBER, width=2.5, dash="dot"),
                name="Spot",
                hovertemplate="Spot %{y:.2f}<extra></extra>",
                meta={"layer": "spot"},
            ),
            row=1,
            col=1,
        )

    for label, key, color, symbol in (
        ("Max +γ", "max_positive_gamma", _GREEN, "diamond"),
        ("Min −γ", "min_negative_gamma", _RED, "diamond"),
        ("Fastest Δγ", "fastest_gamma_increase", _PURPLE, "triangle-up"),
        ("Entry", "recommended", _AMBER, "star"),
    ):
        sig = signals.get(key) or {}
        strike = safe_float(sig.get("strike"), 0.0)
        if strike > 0:
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[strike],
                    mode="markers+text",
                    marker=dict(size=12, color=color, symbol=symbol),
                    text=[label],
                    textposition="middle right",
                    name=label,
                    hovertemplate=f"{label}<br>%{{y:.0f}} {sig.get('option_type', '')}<extra></extra>",
                    meta={"layer": "walls"},
                ),
                row=1,
                col=1,
            )

    for pos in open_positions:
        strike = safe_float(pos.get("strike"), 0.0)
        if strike <= 0:
            continue
        pnl = pos.get("pnl_pct")
        pnl_txt = f"{pnl:+.1%}" if pnl is not None else "open"
        color = _GREEN if (pnl or 0) >= 0 else _RED
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[strike],
                mode="markers",
                marker=dict(size=14, color=color, symbol="x", line=dict(width=2, color=_TEXT)),
                name=f"Open {pos.get('option_type')}",
                hovertemplate=f"OPEN {str(pos.get('option_type')).upper()} %{y:.0f}<br>PnL {pnl_txt}<extra></extra>",
                meta={"layer": "positions"},
            ),
            row=1,
            col=1,
        )

    closed = [t for t in reversed(recent) if t.get("status") == "closed" or t.get("exit_reason")]
    cum = 0.0
    xs, ys, colors = [], [], []
    for i, trade in enumerate(closed):
        cum += safe_float(trade.get("pnl_usd"), 0.0)
        xs.append(i + 1)
        ys.append(cum)
        colors.append(_GREEN if safe_float(trade.get("pnl_usd"), 0) >= 0 else _RED)

    if xs:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                line=dict(color=_BLUE, width=2),
                marker=dict(color=colors, size=7),
                name="Equity",
                hovertemplate="Trade #%{x}<br>Cum PnL $%{y:,.0f}<extra></extra>",
                meta={"layer": "equity"},
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dot", line_color=_MUTED, row=2, col=1)
    else:
        fig.add_annotation(
            text="No closed trades yet",
            xref="x2",
            yref="y2",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=_MUTED, size=12),
        )

    total_pnl = safe_float(performance.get("total_pnl_usd"), 0.0)
    win_rate = safe_float(performance.get("win_rate"), 0.0)
    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_PANEL,
        font=dict(family="Inter, system-ui, sans-serif", size=11, color=_TEXT),
        margin=dict(l=68, r=24, t=42, b=44),
        height=700,
        showlegend=False,
        clickmode="event+select",
        title=dict(
            text=f"WR {win_rate:.0%} · ${total_pnl:,.0f} cumulative · click strike → trade",
            x=0.01,
            y=0.99,
            font=dict(size=11, color=_MUTED),
        ),
    )
    if not window.empty:
        strike_axis = _strike_axis_layout([float(s) for s in window.index], spot_val, axis="y")
        fig.update_yaxes(**strike_axis, row=1, col=1)
    else:
        fig.update_yaxes(title_text="Strike", row=1, col=1, gridcolor="rgba(148,163,184,0.06)")
    fig.update_xaxes(
        title_text="Net gamma (Bn)",
        row=1,
        col=1,
        range=x_range,
        gridcolor="rgba(148,163,184,0.06)",
        zeroline=True,
        zerolinecolor="rgba(255,255,255,0.25)",
    )
    fig.update_xaxes(title_text="Trade #", row=2, col=1, gridcolor="rgba(148,163,184,0.06)")
    fig.update_yaxes(title_text="USD", row=2, col=1, gridcolor="rgba(148,163,184,0.06)")
    return fig


def build_strategy_dashboard(
    *,
    ticker: str,
    spot: float | None,
    exposure: pd.Series | None,
    previous_exposure: pd.Series | None,
    snapshot: dict[str, Any] | None = None,
    prev_spot: float | None = None,
    uw_bundle: dict[str, Any] | None = None,
    window_pct: float = 0.04,
    max_strikes: int = 65,
    exposure_trail: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    state = build_strategy_state(
        ticker=ticker,
        spot=spot,
        exposure=exposure,
        previous_exposure=previous_exposure,
        snapshot=snapshot,
        prev_spot=prev_spot,
        uw_bundle=uw_bundle,
    )
    fig = build_strategy_chart(
        spot=spot,
        exposure=exposure,
        state=state,
        window_pct=window_pct,
        max_strikes=max_strikes,
        exposure_trail=exposure_trail,
        previous_exposure=previous_exposure,
        wall_mode=False,
    )
    return {
        "state": state,
        "chart_json": _encode(fig),
        "window_pct": window_pct,
    }


def build_wall_strategy_state(
    *,
    ticker: str,
    spot: float | None,
    exposure: pd.Series | None,
    snapshot: dict[str, Any] | None = None,
    window_pct: float = 0.01,
) -> dict[str, Any]:
    """Wall GEX signals, quality filters, and open positions for the near-spot dashboard."""
    from gex_core.trading.config import wall_max_hold_bars, wall_stop_loss_pct, wall_take_profit_pct
    from gex_core.trading.low_gex_engine import is_wall_gex_trade, wall_gex_open_trades

    spot_val = safe_float(spot, 0.0)
    snap = snapshot or {}
    if spot_val <= 0:
        spot_val = safe_float(snap.get("spot"), 0.0)

    low = compute_low_gex_signal(exposure, spot=spot_val, window_pct=window_pct)
    high = compute_high_gex_signal(exposure, spot=spot_val, window_pct=window_pct)
    rec = low.get("recommended") or {}
    signals: dict[str, Any] = {
        "available": bool(low.get("available")),
        "reason": low.get("reason"),
        "recommended": rec if low.get("available") else None,
        "min_gamma_strike": low.get("min_gamma_strike"),
        "max_gamma_strike": high.get("max_gamma_strike") if high.get("available") else None,
        "master_direction": low.get("master_direction"),
    }

    filters = {"approve": False, "reason": "No signal"}
    if signals.get("available"):
        ok, reason = wall_entry_quality_ok(
            wall_strike=float(rec.get("strike") or 0),
            wall_gamma=float(rec.get("gamma_bn") or 0),
            regime=str(snap.get("regime") or ""),
        )
        filters = {"approve": ok, "reason": reason or "Wall quality OK"}

    advice = {
        "approve": bool(filters.get("approve")),
        "reason": rec.get("rationale") if filters.get("approve") else filters.get("reason", "No signal"),
        "source": "wall_gex",
        "confidence": 0.85 if filters.get("approve") else 0.4,
    }

    open_positions = []
    for pos in wall_gex_open_trades(ticker):
        pnl = None
        if spot_val > 0:
            pnl = estimate_option_pnl_pct(
                pos["option_type"],
                entry_spot=float(pos["entry_spot"]),
                current_spot=spot_val,
                strike=float(pos["strike"]),
            )
        stop = wall_stop_loss_pct()
        open_positions.append(
            {
                **pos,
                "pnl_pct": pnl,
                "stop_pct": -stop,
                "target_pct": wall_take_profit_pct(),
            }
        )

    closed = [
        t
        for t in list_recent_trades(limit=50, ticker=ticker)
        if is_wall_gex_trade(t)
    ]
    wins = sum(1 for t in closed if t.get("status") == "closed" and safe_float(t.get("pnl_usd"), 0) > 0)
    closed_done = [t for t in closed if t.get("status") == "closed" or t.get("exit_reason")]
    total_pnl = sum(safe_float(t.get("pnl_usd"), 0.0) for t in closed_done)
    performance = {
        "total_trades": len(closed_done),
        "win_rate": (wins / len(closed_done)) if closed_done else 0.0,
        "total_pnl_usd": total_pnl,
    }

    return {
        "spot": spot_val,
        "signals": signals,
        "filters": filters,
        "advice": advice,
        "open_positions": open_positions,
        "performance": performance,
        "recent_trades": closed[:15],
        "rules": {
            "stop_loss_pct": wall_stop_loss_pct(),
            "take_profit_pct": wall_take_profit_pct(),
            "partial_take_profit_pct": 0.0,
            "trail_trigger_pct": 0.0,
            "trail_floor_pct": 0.0,
            "max_strike_distance_pct": window_pct,
            "max_hold_bars": wall_max_hold_bars(),
        },
        "exit_profile": None,
        "levels": {
            "gamma_flip": parse_gamma_flip_value(snap.get("gamma_flip")),
            "call_wall": snap.get("call_wall"),
            "put_wall": snap.get("put_wall"),
            "regime": str(snap.get("regime") or ""),
        },
        "strategy_mode": "wall",
    }


def build_wall_strategy_chart(
    *,
    spot: float | None,
    exposure: pd.Series | None,
    state: dict[str, Any],
    window_pct: float = 0.01,
    max_strikes: int = 40,
    exposure_trail: list[dict[str, Any]] | None = None,
    previous_exposure: pd.Series | None = None,
) -> go.Figure:
    """Near-spot wall chart: low/high γ walls instead of gamma magnets."""
    spot_val = safe_float(spot or state.get("spot"), 0.0)
    signals = state.get("signals") or {}
    performance = state.get("performance") or {}
    recent = state.get("recent_trades") or []
    levels = state.get("levels") or {}
    open_positions = state.get("open_positions") or []

    trail_count = len(_normalize_exposure_trail(exposure_trail, previous_exposure))
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=(
            f"GEX by strike · ±{window_pct * 100:.1f}% · walls · last {trail_count} γ snapshots",
            "Cumulative PnL (wall trades)",
        ),
    )

    window = _chart_exposure_window(
        exposure if isinstance(exposure, pd.Series) else pd.Series(dtype=float),
        spot_val,
        window_pct=window_pct,
        max_strikes=max_strikes,
    )
    prior = _prior_exposure_series(
        _normalize_exposure_trail(exposure_trail, previous_exposure),
        previous_exposure,
    )
    x_range = _symmetric_x_range([float(v) for v in window.values]) if not window.empty else [-1.0, 1.0]
    if not window.empty and spot_val > 0:
        _add_gamma_bars(fig, window=window, prior=prior, row=1, col=1)
        _decorate_gex_strike_panel(
            fig,
            window=window,
            spot_val=spot_val,
            x_range=x_range,
            levels=levels,
            exposure_trail=exposure_trail,
            previous_exposure=previous_exposure,
            wall_mode=True,
            window_pct=window_pct,
            ticker=state.get("ticker"),
        )
        fig.add_vline(
            x=0,
            line=dict(color="rgba(226, 232, 240, 0.55)", width=1.5),
            row=1,
            col=1,
        )

    if spot_val > 0:
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=[spot_val, spot_val],
                mode="lines",
                line=dict(color=_AMBER, width=2.5, dash="dot"),
                name="Spot",
                hovertemplate="Spot %{y:.2f}<extra></extra>",
                meta={"layer": "spot"},
            ),
            row=1,
            col=1,
        )

    for label, key, color, symbol in (
        ("Low wall", "min_gamma_strike", _RED, "diamond"),
        ("High wall", "max_gamma_strike", _GREEN, "diamond"),
        ("Entry", "recommended", _AMBER, "star"),
    ):
        sig = signals.get(key) or {}
        strike = safe_float(sig.get("strike"), 0.0)
        if strike > 0:
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[strike],
                    mode="markers+text",
                    marker=dict(size=12, color=color, symbol=symbol),
                    text=[label],
                    textposition="middle right",
                    name=label,
                    hovertemplate=f"{label}<br>%{{y:.0f}} {sig.get('option_type', '')}<extra></extra>",
                    meta={"layer": "walls"},
                ),
                row=1,
                col=1,
            )

    for pos in open_positions:
        strike = safe_float(pos.get("strike"), 0.0)
        if strike <= 0:
            continue
        pnl = pos.get("pnl_pct")
        pnl_txt = f"{pnl:+.1%}" if pnl is not None else "open"
        color = _GREEN if (pnl or 0) >= 0 else _RED
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[strike],
                mode="markers",
                marker=dict(size=14, color=color, symbol="x", line=dict(width=2, color=_TEXT)),
                name=f"Open {pos.get('option_type')}",
                hovertemplate=f"OPEN {str(pos.get('option_type')).upper()} %{y:.0f}<br>PnL {pnl_txt}<extra></extra>",
                meta={"layer": "positions"},
            ),
            row=1,
            col=1,
        )

    closed = [t for t in reversed(recent) if t.get("status") == "closed" or t.get("exit_reason")]
    cum = 0.0
    xs, ys, colors = [], [], []
    for i, trade in enumerate(closed):
        cum += safe_float(trade.get("pnl_usd"), 0.0)
        xs.append(i + 1)
        ys.append(cum)
        colors.append(_GREEN if safe_float(trade.get("pnl_usd"), 0) >= 0 else _RED)

    if xs:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                line=dict(color=_BLUE, width=2),
                marker=dict(color=colors, size=7),
                name="Equity",
                hovertemplate="Trade #%{x}<br>Cum PnL $%{y:,.0f}<extra></extra>",
                meta={"layer": "equity"},
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dot", line_color=_MUTED, row=2, col=1)
    else:
        fig.add_annotation(
            text="No closed wall trades yet",
            xref="x2",
            yref="y2",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=_MUTED, size=12),
        )

    total_pnl = safe_float(performance.get("total_pnl_usd"), 0.0)
    win_rate = safe_float(performance.get("win_rate"), 0.0)
    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_PANEL,
        font=dict(family="Inter, system-ui, sans-serif", size=11, color=_TEXT),
        margin=dict(l=68, r=24, t=42, b=44),
        height=700,
        showlegend=False,
        clickmode="event+select",
        title=dict(
            text=f"Wall GEX · WR {win_rate:.0%} · ${total_pnl:,.0f} · click strike → trade",
            x=0.01,
            y=0.99,
            font=dict(size=11, color=_MUTED),
        ),
    )
    if not window.empty:
        strike_axis = _strike_axis_layout([float(s) for s in window.index], spot_val, axis="y")
        fig.update_yaxes(**strike_axis, row=1, col=1)
    else:
        fig.update_yaxes(title_text="Strike", row=1, col=1, gridcolor="rgba(148,163,184,0.06)")
    fig.update_xaxes(
        title_text="Net gamma (Bn)",
        row=1,
        col=1,
        range=x_range,
        gridcolor="rgba(148,163,184,0.06)",
        zeroline=True,
        zerolinecolor="rgba(255,255,255,0.25)",
    )
    fig.update_xaxes(title_text="Trade #", row=2, col=1, gridcolor="rgba(148,163,184,0.06)")
    fig.update_yaxes(title_text="USD", row=2, col=1, gridcolor="rgba(148,163,184,0.06)")
    return fig


def build_wall_strategy_dashboard(
    *,
    ticker: str,
    spot: float | None,
    exposure: pd.Series | None,
    snapshot: dict[str, Any] | None = None,
    window_pct: float = 0.01,
    max_strikes: int = 40,
    exposure_trail: list[dict[str, Any]] | None = None,
    previous_exposure: pd.Series | None = None,
) -> dict[str, Any]:
    state = build_wall_strategy_state(
        ticker=ticker,
        spot=spot,
        exposure=exposure,
        snapshot=snapshot,
        window_pct=window_pct,
    )
    fig = build_wall_strategy_chart(
        spot=spot,
        exposure=exposure,
        state=state,
        window_pct=window_pct,
        max_strikes=max_strikes,
        exposure_trail=exposure_trail,
        previous_exposure=previous_exposure,
    )
    return {
        "state": state,
        "chart_json": _encode(fig),
        "window_pct": window_pct,
        "strategy_mode": "wall",
    }
