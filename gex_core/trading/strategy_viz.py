"""Dashboard visualization for the gamma auto-trader strategy."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder

from gex_core.features import safe_float
from gex_core.trading.advisor import _rule_based_advice
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


def _encode(fig: go.Figure) -> str:
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def _atm_window(series: pd.Series, spot: float, window_pct: float = 0.012) -> pd.Series:
    if series is None or series.empty or spot <= 0:
        return pd.Series(dtype=float)
    s = pd.Series(series, dtype=float).sort_index()
    lo, hi = spot * (1 - window_pct), spot * (1 + window_pct)
    return s[(s.index >= lo) & (s.index <= hi)]


def build_strategy_state(
    *,
    ticker: str,
    spot: float | None,
    exposure: pd.Series | None,
    previous_exposure: pd.Series | None,
    snapshot: dict[str, Any] | None = None,
    prev_spot: float | None = None,
) -> dict[str, Any]:
    """Current signals, filter checklist, and open-position marks for the dashboard."""
    spot_val = safe_float(spot, 0.0)
    snap = snapshot or {}
    if spot_val <= 0:
        spot_val = safe_float(snap.get("spot"), 0.0)

    signals = compute_gamma_signals(exposure, previous_exposure, spot=spot_val)
    market = MarketContext(
        spot=spot_val,
        prev_spot=prev_spot,
        gamma_flip=safe_float(snap.get("gamma_flip"), 0.0) or None,
        regime=str(snap.get("regime") or ""),
        is_cpi_day=bool(snap.get("is_cpi_day")),
        is_nfp_day=bool(snap.get("is_nfp_day")),
        is_fomc_week=bool(snap.get("is_fomc_week")),
        flow_net_delta_gex_bn=safe_float(snap.get("flow_net_delta_gex_bn"), 0.0) or None,
    )

    memory = {"performance": get_performance_summary(ticker)}
    advice = {"approve": False, "reason": "No signal"}
    filters = {"approve": False, "reason": "No signal"}
    if signals.get("available"):
        filters = evaluate_entry_filters(signals, market=market)
        advice = _rule_based_advice(signals, memory, market=market)

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
        "spot": spot_val,
        "signals": signals,
        "filters": filters,
        "advice": advice,
        "open_positions": marked_positions,
        "performance": memory["performance"],
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
) -> go.Figure:
    """Two-row dashboard: magnet map + cumulative PnL."""
    spot_val = safe_float(spot or state.get("spot"), 0.0)
    signals = state.get("signals") or {}
    performance = state.get("performance") or {}
    recent = state.get("recent_trades") or []
    levels = state.get("levels") or {}
    open_positions = state.get("open_positions") or []

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.06,
        subplot_titles=("Magnet map · entry levels & open positions", "Cumulative PnL (recent trades)"),
    )

    window = _atm_window(exposure if isinstance(exposure, pd.Series) else pd.Series(dtype=float), spot_val)
    if not window.empty and spot_val > 0:
        colors = [_GREEN if v >= 0 else _RED for v in window.values]
        fig.add_trace(
            go.Bar(
                x=window.values,
                y=window.index.astype(float),
                orientation="h",
                marker_color=colors,
                opacity=0.75,
                name="Gamma",
                hovertemplate="Strike %{y:.0f}<br>γ %{x:+.3f} Bn<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if spot_val > 0:
        x_span = list(window.values) if not window.empty else [0]
        pad = max(0.05, max((abs(v) for v in x_span), default=0.1) * 1.2)
        fig.add_trace(
            go.Scatter(
                x=[-pad, pad],
                y=[spot_val, spot_val],
                mode="lines",
                line=dict(color=_AMBER, width=3),
                name="Spot",
                hovertemplate="Spot %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    flip = safe_float(levels.get("gamma_flip"), 0.0)
    if flip > 0:
        fig.add_hline(y=flip, line_dash="dot", line_color=_BLUE, opacity=0.7, row=1, col=1)
        fig.add_annotation(x=0, y=flip, text=" γ-flip", showarrow=False, xanchor="left", font=dict(size=10, color=_BLUE), row=1, col=1)

    for label, key, color, symbol in (
        ("Max +γ", "max_positive_gamma", _GREEN, "diamond"),
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
        margin=dict(l=60, r=24, t=48, b=36),
        height=720,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
        title=dict(
            text=f"Gamma Auto-Trader · WR {win_rate:.0%} · ${total_pnl:,.0f} total",
            x=0.02,
            font=dict(size=13, color=_MUTED),
        ),
    )
    fig.update_yaxes(title_text="Strike", row=1, col=1, gridcolor="rgba(148,163,184,0.08)")
    fig.update_xaxes(title_text="Net gamma (Bn)", row=1, col=1, gridcolor="rgba(148,163,184,0.08)", zeroline=True, zerolinecolor="rgba(255,255,255,0.2)")
    fig.update_xaxes(title_text="Trade #", row=2, col=1, gridcolor="rgba(148,163,184,0.08)")
    fig.update_yaxes(title_text="USD", row=2, col=1, gridcolor="rgba(148,163,184,0.08)")
    return fig


def build_strategy_dashboard(
    *,
    ticker: str,
    spot: float | None,
    exposure: pd.Series | None,
    previous_exposure: pd.Series | None,
    snapshot: dict[str, Any] | None = None,
    prev_spot: float | None = None,
) -> dict[str, Any]:
    state = build_strategy_state(
        ticker=ticker,
        spot=spot,
        exposure=exposure,
        previous_exposure=previous_exposure,
        snapshot=snapshot,
        prev_spot=prev_spot,
    )
    fig = build_strategy_chart(spot=spot, exposure=exposure, state=state)
    return {"state": state, "chart_json": _encode(fig)}
