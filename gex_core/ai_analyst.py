"""
Dealer Gamma Intelligence Analyst.

Converts the raw GEX data from Unusual Whales into structured signals,
scenario predictions, and a plain-language narrative paragraph.

Design
------
* Fully rule-based core — no external AI API required.
* Optional OpenAI narrative polish when OPENAI_API_KEY is set.
* Output is a single GammaAnalysis dataclass consumed by Flask,
  Streamlit, and the CLI.

Key dealer-gamma concepts used
-------------------------------
LONG gamma regime (total GEX > 0):
    Dealers are net long gamma — they delta-hedge by buying dips and selling
    rallies, which dampens volatility and keeps price in a range.

SHORT gamma regime (total GEX < 0):
    Dealers are net short gamma — they chase moves (buy rallies, sell dips),
    amplifying price swings and increasing volatility.

Gamma flip:
    The strike where cumulative GEX crosses zero.  Below the flip: dealers
    are locally short gamma (volatile).  Above the flip: locally long gamma
    (stabilising).

Call wall:
    Strike with the largest positive net GEX.  Dealers must sell the
    underlying as price rallies into this zone — strong resistance.

Put wall:
    Strike with the largest negative net GEX.  Dealers must buy the
    underlying as price falls into this zone — potential support, but thin
    negative gamma can also amplify down-moves.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GammaSignal:
    """A single labelled GEX signal with a human-readable interpretation."""
    label: str
    value: str
    detail: str
    sentiment: str  # "bullish" | "bearish" | "neutral" | "caution"


@dataclass
class GammaAnalysis:
    """Full output of the gamma intelligence analysis."""
    ticker: str
    spot: float

    # Regime
    regime: str          # "LONG gamma" | "SHORT gamma"
    regime_detail: str   # one-sentence explanation
    bias: str            # "bullish" | "bearish" | "neutral"
    confidence: float    # 0–1

    # Key levels
    gamma_flip: float | None
    flip_distance_pct: float | None   # positive = spot above flip
    call_wall: float | None
    put_wall: float | None
    wall_range: float | None          # call_wall − put_wall
    dominant_strike: float | None     # strike with highest |GEX|

    # Structured signals
    signals: list[GammaSignal] = field(default_factory=list)

    # Plain-language outputs
    predictions: list[str] = field(default_factory=list)
    narrative: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "spot": self.spot,
            "regime": self.regime,
            "regime_detail": self.regime_detail,
            "bias": self.bias,
            "confidence": round(self.confidence, 3),
            "gamma_flip": round(self.gamma_flip, 2) if self.gamma_flip else None,
            "flip_distance_pct": round(self.flip_distance_pct * 100, 2) if self.flip_distance_pct is not None else None,
            "call_wall": self.call_wall,
            "put_wall": self.put_wall,
            "wall_range": round(self.wall_range, 0) if self.wall_range else None,
            "dominant_strike": self.dominant_strike,
            "signals": [
                {"label": s.label, "value": s.value, "detail": s.detail, "sentiment": s.sentiment}
                for s in self.signals
            ],
            "predictions": self.predictions,
            "narrative": self.narrative,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pct(spot: float, level: float) -> float:
    """Percentage distance from spot to level (positive = level above spot)."""
    return (level - spot) / spot * 100.0


def _regime_label(total_gex: float) -> tuple[str, str]:
    """Return (regime, detail) for a given total GEX."""
    if total_gex > 5:
        return (
            "LONG gamma",
            "Dealers are heavily net long gamma — expect range-bound, low-volatility price action."
        )
    elif total_gex > 0:
        return (
            "LONG gamma",
            "Dealers are moderately net long gamma — volatility slightly suppressed, dips likely bought."
        )
    elif total_gex > -5:
        return (
            "SHORT gamma",
            "Dealers are slightly net short gamma — directional moves possible, volatility may pick up."
        )
    else:
        return (
            "SHORT gamma",
            "Dealers are heavily net short gamma — expect elevated volatility and momentum-driven price action."
        )


def _flip_signal(spot: float, flip: float | None) -> GammaSignal | None:
    if flip is None:
        return None
    dist = _pct(spot, flip)
    if abs(dist) < 0.3:
        return GammaSignal(
            label="Gamma Flip",
            value=f"{flip:.0f}",
            detail=f"Spot is within 0.3% of the gamma flip — critical inflection zone. Dealer hedging behaviour will shift sharply here.",
            sentiment="caution",
        )
    elif spot > flip:
        return GammaSignal(
            label="Gamma Flip",
            value=f"{flip:.0f}",
            detail=f"Spot is {abs(dist):.1f}% above the flip — dealers locally long gamma, a stabilising force.",
            sentiment="bullish",
        )
    else:
        return GammaSignal(
            label="Gamma Flip",
            value=f"{flip:.0f}",
            detail=f"Spot is {abs(dist):.1f}% below the flip — dealers locally short gamma, expect choppier price action and possible acceleration toward {flip:.0f}.",
            sentiment="bearish",
        )


def _wall_signal(spot: float, wall: float | None, gex: float | None, kind: str) -> GammaSignal | None:
    if wall is None:
        return None
    dist = _pct(spot, wall)
    if kind == "call":
        if abs(dist) < 0.5:
            return GammaSignal(
                label="Call Wall",
                value=f"{wall:.0f}",
                detail=f"Spot is near the call wall ({wall:.0f}, {gex:+.2f} Bn$). Dealers are selling here to hedge — expect strong resistance.",
                sentiment="bearish",
            )
        else:
            return GammaSignal(
                label="Call Wall",
                value=f"{wall:.0f}",
                detail=f"{wall:.0f} ({gex:+.2f} Bn$) is {dist:+.1f}% away — key resistance ceiling. Dealer selling intensifies as price approaches.",
                sentiment="neutral",
            )
    else:
        if abs(dist) < 0.5:
            return GammaSignal(
                label="Put Wall",
                value=f"{wall:.0f}",
                detail=f"Spot is near the put wall ({wall:.0f}, {gex:+.2f} Bn$) — dealers forced to buy here, providing floor support.",
                sentiment="bullish",
            )
        else:
            neg_str = "" if gex is None else f" ({gex:+.2f} Bn$)"
            return GammaSignal(
                label="Put Wall",
                value=f"{wall:.0f}",
                detail=f"{wall:.0f}{neg_str} is {dist:+.1f}% away. Strong dealer buying expected if price reaches this level.",
                sentiment="neutral",
            )


def _concentration_signal(gex_by_strike: pd.Series) -> GammaSignal:
    """Score how concentrated the gamma is around a few strikes."""
    if gex_by_strike.empty:
        return GammaSignal("Concentration", "N/A", "Insufficient data.", "neutral")
    abs_vals = gex_by_strike.abs()
    top5 = abs_vals.nlargest(5).sum()
    total = abs_vals.sum()
    conc = top5 / total if total > 0 else 0.0
    if conc > 0.60:
        detail = f"Top 5 strikes hold {conc*100:.0f}% of total gamma — highly concentrated. Expect strong pinning near the dominant strike."
        sentiment = "caution"
    elif conc > 0.40:
        detail = f"Top 5 strikes hold {conc*100:.0f}% of total gamma — moderately concentrated. A few key levels will drive price behaviour."
        sentiment = "neutral"
    else:
        detail = f"Gamma is diffuse across many strikes ({conc*100:.0f}% in top 5). Price is less likely to pin to a single level."
        sentiment = "neutral"
    return GammaSignal("Concentration", f"{conc*100:.0f}%", detail, sentiment)


def _asymmetry_signal(gex_by_strike: pd.Series) -> GammaSignal:
    """Bull/bear skew of the gamma profile."""
    pos = float(gex_by_strike[gex_by_strike > 0].sum())
    neg = float(gex_by_strike[gex_by_strike < 0].sum())
    total_abs = pos + abs(neg)
    if total_abs == 0:
        return GammaSignal("Skew", "Flat", "No notable skew.", "neutral")
    bull_pct = pos / total_abs * 100
    if bull_pct > 65:
        detail = f"Gamma skewed {bull_pct:.0f}% positive — dealers structurally long upside. Upward moves are damped, downward moves may be sharper."
        sentiment = "bearish"
    elif bull_pct > 55:
        detail = f"Slight bullish gamma skew ({bull_pct:.0f}% positive). Modestly favours upside stability."
        sentiment = "neutral"
    elif bull_pct < 35:
        detail = f"Gamma heavily skewed negative ({100-bull_pct:.0f}% put-side). Dealers short puts — downside moves can accelerate."
        sentiment = "bearish"
    else:
        detail = f"Balanced gamma profile ({bull_pct:.0f}% positive) — no strong directional skew."
        sentiment = "neutral"
    return GammaSignal("Gamma Skew", f"{bull_pct:.0f}% calls", detail, sentiment)


def _pinning_signal(spot: float, gex_by_strike: pd.Series) -> GammaSignal:
    """Identify the most likely strike gravity centre."""
    if gex_by_strike.empty:
        return GammaSignal("Pin Level", "N/A", "No data.", "neutral")
    # Positive GEX strikes only — these create the strongest pinning
    pos = gex_by_strike[gex_by_strike > 0]
    if pos.empty:
        dominant = float(gex_by_strike.abs().idxmax())
    else:
        dominant = float(pos.idxmax())
    dist = _pct(spot, dominant)
    detail = (
        f"{dominant:.0f} has the strongest positive gamma ({gex_by_strike[dominant]:+.2f} Bn$). "
        f"Dealers will hedge toward this strike, creating a gravitational pull "
        f"{'upward' if dist > 0 else 'downward'} ({abs(dist):.1f}% from spot)."
    )
    sentiment = "bullish" if dist > 0 else "bearish"
    return GammaSignal("Pin / Gravity", f"{dominant:.0f}", detail, sentiment)


def _momentum_signal(history: list[dict] | None) -> GammaSignal | None:
    """Detect trend in total GEX from recent snapshots."""
    if not history or len(history) < 3:
        return None
    totals = [float(h.get("total_gex", 0) or 0) for h in history[-6:]]
    delta = totals[-1] - totals[0]
    if abs(delta) < 0.5:
        return GammaSignal(
            "GEX Momentum", "Flat",
            "Total GEX has been stable over recent snapshots.",
            "neutral",
        )
    elif delta > 0:
        return GammaSignal(
            "GEX Momentum", f"+{delta:.2f} Bn$",
            f"GEX has been building (+{delta:.2f} Bn$) — dealers are accumulating long gamma. This suppresses volatility further.",
            "bullish",
        )
    else:
        return GammaSignal(
            "GEX Momentum", f"{delta:.2f} Bn$",
            f"GEX has been declining ({delta:.2f} Bn$) — dealers reducing long gamma exposure. Volatility may increase.",
            "bearish",
        )


def _build_predictions(
    spot: float,
    total_gex: float,
    gamma_flip: float | None,
    call_wall: float | None,
    put_wall: float | None,
    gex_by_strike: pd.Series,
) -> list[str]:
    preds = []

    # 1. Regime-based range expectation
    if total_gex > 5:
        preds.append(
            f"LONG gamma: expect {spot:.0f} to stay range-bound. Dealers will buy dips and sell rallies, "
            f"limiting both upside and downside moves."
        )
    elif total_gex > 0:
        preds.append(
            f"Mild LONG gamma: price may drift but large moves are dampened. Watch for mean-reversion from extremes."
        )
    else:
        preds.append(
            f"SHORT gamma: directional moves may accelerate. Volatility is likely to expand — both up and down moves can feed on themselves."
        )

    # 2. Gamma flip scenario
    if gamma_flip is not None:
        dist = _pct(spot, gamma_flip)
        if abs(dist) < 1.0:
            preds.append(
                f"Gamma flip at {gamma_flip:.0f} is very close ({abs(dist):.1f}% from current spot). "
                f"A breach could shift dealer hedging dynamics abruptly — watch for a volatility spike around this level."
            )
        elif spot < gamma_flip:
            preds.append(
                f"Spot ({spot:.0f}) is {abs(dist):.1f}% below the gamma flip ({gamma_flip:.0f}). "
                f"In this zone dealers are locally short gamma. "
                f"A sustained rally through {gamma_flip:.0f} would move us into dealer-long territory and likely dampen further moves."
            )
        else:
            preds.append(
                f"Spot ({spot:.0f}) is {dist:.1f}% above the gamma flip ({gamma_flip:.0f}). "
                f"Dealers are locally long gamma — stabilising force. "
                f"A drop below {gamma_flip:.0f} would flip the hedging dynamic and could accelerate selling."
            )

    # 3. Call wall / put wall corridor
    if call_wall and put_wall:
        rng = abs(call_wall - put_wall)
        preds.append(
            f"Primary dealer corridor: {put_wall:.0f} – {call_wall:.0f} "
            f"({rng:.0f} pts wide). This is the range where dealer hedging flows are most active. "
            f"Expect the market to oscillate within this band unless a catalyst forces a break."
        )
    elif call_wall:
        preds.append(
            f"Call wall at {call_wall:.0f} is the dominant ceiling — dealer selling will intensify near this level."
        )

    # 4. Dominant strike gravity
    pos = gex_by_strike[gex_by_strike > 0]
    if not pos.empty:
        top = float(pos.idxmax())
        if abs(_pct(spot, top)) < 3.0:
            preds.append(
                f"The {top:.0f} strike has the highest positive gamma concentration — acts as a gravitational anchor. "
                f"Near-term price may gravitate toward or pin near {top:.0f}."
            )

    return preds


def _build_narrative(
    ticker: str,
    spot: float,
    total_gex: float,
    regime: str,
    gamma_flip: float | None,
    call_wall: float | None,
    put_wall: float | None,
    bias: str,
    signals: list[GammaSignal],
) -> str:
    """Compose a concise narrative paragraph."""
    flip_str = f"The gamma flip sits at {gamma_flip:.0f}" if gamma_flip else "No clear gamma flip identified"
    wall_str = ""
    if call_wall and put_wall:
        wall_str = f" with the call wall at {call_wall:.0f} and put wall at {put_wall:.0f}"
    elif call_wall:
        wall_str = f" with the call wall at {call_wall:.0f}"

    above_below = "above" if (gamma_flip and spot > gamma_flip) else "below"
    local_regime = "long gamma (stabilising)" if (gamma_flip and spot > gamma_flip) else "short gamma (trending/volatile)"

    concentration = next((s for s in signals if s.label == "Concentration"), None)
    conc_note = f" Gamma is {concentration.value} concentrated." if concentration else ""

    narrative = (
        f"{ticker} is in a {regime} environment with {total_gex:+.1f} Bn$ net dealer exposure "
        f"at the current spot of {spot:,.0f}.{wall_str}. "
        f"{flip_str}, and spot is currently {above_below} the flip, placing dealers in a "
        f"{local_regime} position locally.{conc_note} "
        f"The overall bias is {bias} — "
    )

    if bias == "bullish":
        narrative += (
            "dealer hedging flows are expected to support price on dips and resist large downside moves. "
            "Upside rallies are capped by the call wall."
        )
    elif bias == "bearish":
        narrative += (
            "dealer hedging flows may amplify downside moves and offer limited resistance to selling. "
            "Any rally will face structural headwinds from dealer short-gamma positioning."
        )
    else:
        narrative += (
            "dealer hedging flows are balanced with no strong directional tilt. "
            "Price is likely to remain in the dealer corridor until a catalyst triggers a break."
        )

    return narrative


def _openai_narrative(base_narrative: str, analysis_dict: dict) -> str:
    """Optionally enrich the narrative with an OpenAI call."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return base_narrative
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        prompt = (
            "You are an expert options market analyst specialising in dealer gamma exposure (GEX). "
            "Rewrite the following draft analysis into a crisp, confident 2-paragraph summary that a "
            "professional trader would find actionable. Use specific numbers. Avoid hedging phrases like "
            "'may' or 'might' when the signal is clear. Include a clear directional call.\n\n"
            f"Draft: {base_narrative}\n\nKey data: {analysis_dict}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return base_narrative


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def analyze_dealer_gamma(
    ticker: str,
    spot: float,
    gex_by_strike: pd.Series,
    cumulative_gex: pd.Series,
    total_gex_bn: float,
    gamma_flip: float | None = None,
    history: list[dict] | None = None,
    use_openai: bool = True,
) -> GammaAnalysis:
    """
    Analyse dealer gamma positioning and generate predictions.

    Parameters
    ----------
    ticker : str
        Underlying symbol.
    spot : float
        Current underlying price.
    gex_by_strike : pd.Series
        Net GEX per strike (Bn$), indexed by strike price.
    cumulative_gex : pd.Series
        Running sum of gex_by_strike (Bn$).
    total_gex_bn : float
        Total net GEX across all strikes (Bn$).
    gamma_flip : float | None
        Pre-computed gamma flip strike; computed here if not provided.
    history : list[dict] | None
        Recent snapshot dicts for momentum analysis.
    use_openai : bool
        Whether to call OpenAI for narrative polish (requires OPENAI_API_KEY).

    Returns
    -------
    GammaAnalysis
    """
    from gex_core.features import estimate_gamma_flip as _estimate_flip

    # ── Key levels ──────────────────────────────────────────────────────────
    if gamma_flip is None and not cumulative_gex.empty:
        gamma_flip = _estimate_flip(cumulative_gex)

    call_wall: float | None = None
    call_wall_gex: float | None = None
    put_wall: float | None = None
    put_wall_gex: float | None = None
    dominant_strike: float | None = None

    if not gex_by_strike.empty:
        call_wall = float(gex_by_strike.idxmax())
        call_wall_gex = float(gex_by_strike.max())
        put_wall = float(gex_by_strike.idxmin())
        put_wall_gex = float(gex_by_strike.min())
        dominant_strike = float(gex_by_strike.abs().idxmax())

    wall_range = abs(call_wall - put_wall) if call_wall and put_wall else None
    flip_dist_pct = (spot - gamma_flip) / spot if gamma_flip and spot > 0 else None

    # ── Regime ─────────────────────────────────────────────────────────────
    regime, regime_detail = _regime_label(total_gex_bn)

    # ── Bias ───────────────────────────────────────────────────────────────
    # Bias is bullish when: LONG gamma AND spot above flip (or no flip)
    # Bias is bearish when: SHORT gamma OR spot below flip
    long_gamma = total_gex_bn >= 0
    above_flip = gamma_flip is None or spot >= gamma_flip

    if long_gamma and above_flip:
        bias = "bullish"
        confidence = min(0.75, 0.5 + abs(total_gex_bn) * 0.01)
    elif not long_gamma and not above_flip:
        bias = "bearish"
        confidence = min(0.75, 0.5 + abs(total_gex_bn) * 0.01)
    elif long_gamma and not above_flip:
        bias = "neutral"  # mixed: overall long but locally short near spot
        confidence = 0.45
    else:
        bias = "neutral"
        confidence = 0.40

    # ── Build signals ───────────────────────────────────────────────────────
    signals: list[GammaSignal] = []

    # 1. Regime
    signals.append(GammaSignal(
        label="Regime",
        value=regime,
        detail=regime_detail,
        sentiment="bullish" if long_gamma else "bearish",
    ))

    # 2. Gamma flip
    flip_sig = _flip_signal(spot, gamma_flip)
    if flip_sig:
        signals.append(flip_sig)

    # 3. Call wall
    cw_sig = _wall_signal(spot, call_wall, call_wall_gex, "call")
    if cw_sig:
        signals.append(cw_sig)

    # 4. Put wall
    pw_sig = _wall_signal(spot, put_wall, put_wall_gex, "put")
    if pw_sig:
        signals.append(pw_sig)

    # 5. Gamma skew
    if not gex_by_strike.empty:
        signals.append(_asymmetry_signal(gex_by_strike))

    # 6. Concentration
    if not gex_by_strike.empty:
        signals.append(_concentration_signal(gex_by_strike))

    # 7. Pinning gravity
    if not gex_by_strike.empty:
        signals.append(_pinning_signal(spot, gex_by_strike))

    # 8. GEX momentum (requires history)
    mom = _momentum_signal(history)
    if mom:
        signals.append(mom)

    # ── Predictions ─────────────────────────────────────────────────────────
    predictions = _build_predictions(
        spot, total_gex_bn, gamma_flip, call_wall, put_wall, gex_by_strike
    )

    # ── Narrative ───────────────────────────────────────────────────────────
    narrative = _build_narrative(
        ticker, spot, total_gex_bn, regime, gamma_flip, call_wall, put_wall, bias, signals
    )
    if use_openai:
        analysis_dict = {
            "total_gex_bn": round(total_gex_bn, 2),
            "call_wall": call_wall,
            "put_wall": put_wall,
            "gamma_flip": round(gamma_flip, 0) if gamma_flip else None,
            "bias": bias,
        }
        narrative = _openai_narrative(narrative, analysis_dict)

    return GammaAnalysis(
        ticker=ticker,
        spot=spot,
        regime=regime,
        regime_detail=regime_detail,
        bias=bias,
        confidence=round(confidence, 3),
        gamma_flip=gamma_flip,
        flip_distance_pct=flip_dist_pct,
        call_wall=call_wall,
        put_wall=put_wall,
        wall_range=wall_range,
        dominant_strike=dominant_strike,
        signals=signals,
        predictions=predictions,
        narrative=narrative,
    )
