"""
Market exposure AI agent — integrates Nous Research Hermes Agent with live GEX data.

Vendors https://github.com/NousResearch/hermes-agent (MIT). Uses rule-based
dealer gamma analysis when Hermes is not installed or no LLM key is configured.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

from gex_core.ai_analyst import analyze_dealer_gamma
from gex_core.features import estimate_gamma_flip

logger = logging.getLogger(__name__)

_HERMES_SYSTEM_PROMPT = (
    "You are a market mechanics interpreter specializing in dealer gamma exposure. "
    "Use the WHO → WHOM → WHAT framework: identify who is constrained, whom they affect, "
    "and what forced action results. Be concise and actionable for intraday SPX trading. "
    "End with one sentence trade bias and 2-3 key levels."
)

# Nonexistent toolset → zero tools (Hermes resolves unknown toolsets to []).
_HERMES_EMPTY_TOOLSET = "__gex_no_tools__"


def _resolve_hermes_llm_config() -> tuple[str, str, str, str] | None:
    """Return (provider, api_key, model, base_url) when an LLM backend is configured."""
    explicit_provider = os.environ.get("GEX_HERMES_PROVIDER", "").strip().lower()
    model = os.environ.get("GEX_AGENT_MODEL", "").strip()

    candidates: list[tuple[str, str, str, str]] = []
    if explicit_provider == "openai" or (not explicit_provider and os.environ.get("OPENAI_API_KEY")):
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if key:
            candidates.append(
                ("openai", key, model or "gpt-4o-mini", "https://api.openai.com/v1"),
            )
    if explicit_provider == "openrouter" or (not explicit_provider and os.environ.get("OPENROUTER_API_KEY")):
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if key:
            candidates.append(
                (
                    "openrouter",
                    key,
                    model or "openai/gpt-4o-mini",
                    "https://openrouter.ai/api/v1",
                ),
            )

    if explicit_provider and explicit_provider not in {"openai", "openrouter"}:
        key = (
            os.environ.get("OPENAI_API_KEY", "").strip()
            or os.environ.get("OPENROUTER_API_KEY", "").strip()
        )
        if key:
            base = os.environ.get("GEX_HERMES_BASE_URL", "https://openrouter.ai/api/v1").strip()
            candidates.append((explicit_provider, key, model or "openai/gpt-4o-mini", base))

    return candidates[0] if candidates else None


def _hermes_analyze(user_prompt: str) -> str | None:
    """Run a single-turn Hermes analysis (no tools)."""
    cfg = _resolve_hermes_llm_config()
    if not cfg:
        return None

    try:
        from run_agent import AIAgent
    except ImportError:
        logger.warning("hermes-agent not installed — run scripts/install_agent.sh")
        return None

    provider, api_key, model, base_url = cfg
    try:
        agent = AIAgent(
            model=model,
            api_key=api_key,
            provider=provider,
            base_url=base_url,
            enabled_toolsets=[_HERMES_EMPTY_TOOLSET],
            skip_context_files=True,
            skip_memory=True,
            load_soul_identity=False,
            quiet_mode=True,
            max_iterations=1,
            reasoning_config={"enabled": False},
            ephemeral_system_prompt=_HERMES_SYSTEM_PROMPT,
        )
        response = agent.chat(user_prompt)
        return response.strip() if response else None
    except Exception as exc:
        logger.warning("Hermes analysis failed: %s", exc)
        return None


def _default_who_what(total_gex_bn: float) -> tuple[str, str, str]:
    who = "Dealers / market makers"
    whom = "Directional traders"
    what = (
        "Buy dips and sell rallies (long gamma)"
        if total_gex_bn >= 0
        else "Chase moves and amplify volatility (short gamma)"
    )
    return who, whom, what


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

    Returns structured predictions and an optional Hermes LLM narrative.
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

    who, whom, what = _default_who_what(total_gex_bn)

    prompt = (
        f"Analyze {ticker} dealer {exposure_type} exposure at spot {spot:,.0f}.\n"
        f"Net GEX: {total_gex_bn:+.2f} Bn$ / %, regime: {base.regime}, bias: {base.bias}.\n"
        f"Gamma flip: {gamma_flip}, call wall: {base.call_wall}, put wall: {base.put_wall}.\n"
        f"Signals: {', '.join(s.label for s in base.signals[:4]) or 'none'}.\n"
        "Provide WHO → WHOM → WHAT, a 1-sentence trade bias, and key levels to watch."
    )
    hermes_narrative = _hermes_analyze(prompt)
    narrative = hermes_narrative or base.narrative

    hermes_active = hermes_narrative is not None
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
        "pattern_matches": [],
        "trading_notes": [],
        "narrative": narrative,
        "hermes_enhanced": hermes_active,
        "llm_enhanced": hermes_active,
        "agent_source": "hermes-agent + gex_core" if hermes_active else "gex_core",
    }
