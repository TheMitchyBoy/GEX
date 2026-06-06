"""
Market exposure AI agent — integrates gex-llm-patterns PatternLibrary with live GEX data.

Vendors the pattern library from https://github.com/iAmGiG/gex-llm-patterns (AGPL-3.0).
Uses rule-based dealer gamma analysis when no LLM key is configured.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from gex_core.ai_analyst import analyze_dealer_gamma
from gex_core.features import estimate_gamma_flip

logger = logging.getLogger(__name__)

_VENDOR_ROOT = Path(__file__).resolve().parent.parent / "vendor" / "gex-llm-patterns"
if _VENDOR_ROOT.is_dir() and str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))


def _load_pattern_library():
    try:
        from src.analysis.pattern_library import PatternLibrary

        return PatternLibrary()
    except Exception as exc:
        logger.warning("gex-llm-patterns PatternLibrary unavailable: %s", exc)
        return None


def _market_data_for_patterns(
    *,
    ticker: str,
    spot: float,
    total_gex_bn: float,
    gamma_flip: float | None,
    gex_by_strike: pd.Series,
) -> dict[str, Any]:
    call_wall = float(gex_by_strike.idxmax()) if not gex_by_strike.empty else spot
    return {
        "options_flow": True,
        "gex_metrics": True,
        "strike_distribution": True,
        "net_gex": total_gex_bn * 1e9,
        "strikes": call_wall,
        "call_wall": call_wall,
        "gamma_concentration": (
            float(gex_by_strike.abs().nlargest(5).sum() / gex_by_strike.abs().sum() * 100)
            if not gex_by_strike.empty and gex_by_strike.abs().sum() > 0
            else 0.0
        ),
        "gamma_flip": gamma_flip or spot,
        "spot": spot,
        "ticker": ticker,
        "vix": 0.0,
        "compression": "unknown",
    }


def _llm_exposure_analysis(prompt: str) -> str | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        import openai

        client = openai.OpenAI(api_key=api_key)
        model = os.environ.get("GEX_AGENT_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a market mechanics interpreter specializing in dealer gamma exposure. "
                        "Use the WHO → WHOM → WHAT framework: identify who is constrained, whom they affect, "
                        "and what forced action results. Be concise and actionable for intraday SPX trading."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.35,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("LLM exposure analysis failed: %s", exc)
        return None


def analyze_market_exposure(
    *,
    ticker: str,
    spot: float,
    gex_by_strike: pd.Series,
    cumulative_gex: pd.Series | None = None,
    total_gex_bn: float,
    gamma_flip: float | None = None,
    history: list[dict] | None = None,
    exposure_type: str = "gamma",
) -> dict[str, Any]:
    """
    Run the market exposure agent on current positioning.

    Returns structured predictions, matched patterns, and an optional LLM narrative.
    """
    if cumulative_gex is None or cumulative_gex.empty:
        cumulative_gex = gex_by_strike.cumsum() if not gex_by_strike.empty else pd.Series(dtype=float)
    if gamma_flip is None and not cumulative_gex.empty:
        gamma_flip = estimate_gamma_flip(cumulative_gex)

    base = analyze_dealer_gamma(
        ticker=ticker,
        spot=spot,
        gex_by_strike=gex_by_strike,
        cumulative_gex=cumulative_gex,
        total_gex_bn=total_gex_bn,
        gamma_flip=gamma_flip,
        history=history,
        use_openai=False,
    )

    pattern_library = _load_pattern_library()
    pattern_matches: list[dict[str, Any]] = []
    if pattern_library is not None:
        market_data = _market_data_for_patterns(
            ticker=ticker,
            spot=spot,
            total_gex_bn=total_gex_bn,
            gamma_flip=gamma_flip,
            gex_by_strike=gex_by_strike,
        )
        for match in pattern_library.match_patterns(market_data)[:5]:
            pattern = match.get("pattern")
            pattern_matches.append(
                {
                    "name": match.get("pattern_name"),
                    "category": match.get("category"),
                    "confidence": round(float(match.get("confidence", 0.0)), 3),
                    "who": getattr(pattern, "who", None) if pattern else None,
                    "whom": getattr(pattern, "whom", None) if pattern else None,
                    "what": getattr(pattern, "what", None) if pattern else None,
                    "description": getattr(pattern, "mechanics_description", None) if pattern else None,
                }
            )

    who = "Dealers / market makers"
    whom = "Directional traders"
    what = (
        "Buy dips and sell rallies (long gamma)"
        if total_gex_bn >= 0
        else "Chase moves and amplify volatility (short gamma)"
    )
    if pattern_matches:
        top = pattern_matches[0]
        who = top.get("who") or who
        whom = top.get("whom") or whom
        what = top.get("what") or what

    prompt = (
        f"Analyze SPX dealer {exposure_type} exposure at spot {spot:,.0f}.\n"
        f"Net GEX: {total_gex_bn:+.2f} Bn$ / %, regime: {base.regime}, bias: {base.bias}.\n"
        f"Gamma flip: {gamma_flip}, call wall: {base.call_wall}, put wall: {base.put_wall}.\n"
        f"Top pattern match: {pattern_matches[0]['name'] if pattern_matches else 'none'}.\n"
        f"Provide WHO → WHOM → WHAT, a 1-sentence trade bias, and key levels to watch."
    )
    llm_narrative = _llm_exposure_analysis(prompt)
    narrative = llm_narrative or base.narrative

    trading_notes = []
    if pattern_matches:
        lib = pattern_library
        if lib is not None:
            rules = getattr(lib.get_pattern(pattern_matches[0]["name"]), "trading_rules", None)
            if rules:
                trading_notes = [f"{k}: {v}" for k, v in rules.items()]

    return {
        "ticker": ticker,
        "exposure_type": exposure_type,
        "regime": base.regime,
        "bias": base.bias,
        "confidence": base.confidence,
        "who": who,
        "whom": whom,
        "what": what,
        "gamma_flip": base.gamma_flip,
        "call_wall": base.call_wall,
        "put_wall": base.put_wall,
        "predictions": base.predictions,
        "signals": [s.label for s in base.signals[:6]],
        "pattern_matches": pattern_matches,
        "trading_notes": trading_notes,
        "narrative": narrative,
        "llm_enhanced": llm_narrative is not None,
        "agent_source": "gex-llm-patterns + gex_core",
    }
