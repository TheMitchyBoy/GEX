"""
Market exposure AI agent — integrates Nous Research Hermes Agent with live GEX data.

Vendors https://github.com/NousResearch/hermes-agent (MIT). Uses rule-based
dealer gamma analysis when Hermes is not installed or no LLM key is configured.

When ``agg`` (full UW fetch result) is provided, the agent feeds the LLM every
available Unusual Whales data point via :mod:`gex_core.uw_context_bundle`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

from gex_core.ai_analyst import analyze_dealer_gamma
from gex_core.features import resolve_gamma_flip
from gex_core.uw_context_bundle import build_uw_context_bundle, bundle_to_prompt_json
from gex_core.uw_ai_predictor import predict_from_uw_data

logger = logging.getLogger(__name__)

_HERMES_SYSTEM_PROMPT = (
    "You are a market mechanics interpreter specializing in dealer gamma exposure. "
    "Use the WHO → WHOM → WHAT framework: identify who is constrained, whom they affect, "
    "and what forced action results. Be concise and actionable for intraday SPX trading. "
    "End with one sentence trade bias and 2-3 key levels."
)

_HERMES_FULL_DATA_PROMPT = (
    "You are a market mechanics interpreter specializing in dealer gamma exposure. "
    "You receive a comprehensive Unusual Whales JSON data bundle with strike-level greeks, "
    "spot exposures, term structure, intraday bars, extended features, and history.\n"
    "Use ALL data points — not just summary stats. Apply WHO → WHOM → WHAT, cite specific "
    "strikes/levels from the data, give a 1-sentence trade bias, and list 2-3 key levels."
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


def _hermes_analyze(user_prompt: str, *, system_prompt: str | None = None) -> str | None:
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
            ephemeral_system_prompt=system_prompt or _HERMES_SYSTEM_PROMPT,
        )
        response = agent.chat(user_prompt)
        return response.strip() if response else None
    except Exception as exc:
        logger.warning("Hermes analysis failed: %s", exc)
        return None


def _pattern_matches_from_learning(uw_bundle: dict[str, Any] | None) -> list[str]:
    learning = (uw_bundle or {}).get("daily_learning") or {}
    out: list[str] = []
    for lesson in learning.get("recent_lessons") or []:
        text = lesson.get("lesson")
        if text:
            out.append(f"{lesson.get('market_date', '?')}: {text}")
    return out[:5]


def _trading_notes_from_strategy(uw_bundle: dict[str, Any] | None) -> list[str]:
    strategy = ((uw_bundle or {}).get("daily_learning") or {}).get("today_strategy") or {}
    notes: list[str] = []
    if strategy.get("summary"):
        notes.append(str(strategy["summary"]))
    for play in strategy.get("plays") or []:
        if isinstance(play, dict) and play.get("name"):
            notes.append(
                f"{play['name']}: {play.get('trigger', '')} → {play.get('target', '')}".strip()
            )
    notes.extend(str(n) for n in (strategy.get("risk_notes") or [])[:3])
    return notes[:6]


def _default_who_what(total_gex_bn: float) -> tuple[str, str, str]:
    who = "Dealers / market makers"
    whom = "Directional traders"
    what = (
        "Buy dips and sell rallies (long gamma)"
        if total_gex_bn >= 0
        else "Chase moves and amplify volatility (short gamma)"
    )
    return who, whom, what


def _build_llm_prompt(
    *,
    ticker: str,
    spot: float,
    total_gex_bn: float,
    base,
    gamma_flip: float | None,
    exposure_type: str,
    uw_bundle: dict[str, Any] | None,
) -> str:
    if uw_bundle:
        return (
            f"Analyze {ticker} dealer {exposure_type} exposure using the full Unusual Whales "
            f"data bundle below. Spot={spot:,.0f}, net GEX={total_gex_bn:+.2f} Bn$/%, "
            f"regime={base.regime}, bias={base.bias}, gamma_flip={gamma_flip}.\n\n"
            f"Unusual Whales data:\n{bundle_to_prompt_json(uw_bundle)}"
        )
    return (
        f"Analyze {ticker} dealer {exposure_type} exposure at spot {spot:,.0f}.\n"
        f"Net GEX: {total_gex_bn:+.2f} Bn$ / %, regime: {base.regime}, bias: {base.bias}.\n"
        f"Gamma flip: {gamma_flip}, call wall: {base.call_wall}, put wall: {base.put_wall}.\n"
        f"Signals: {', '.join(s.label for s in base.signals[:4]) or 'none'}.\n"
        "Provide WHO → WHOM → WHAT, a 1-sentence trade bias, and key levels to watch."
    )


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
    agg=None,
    spot_gamma_bn: float | None = None,
    api_key: str | None = None,
    knn_prediction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run the market exposure agent on current positioning.

    When ``agg`` (GexAggregates from ``fetch_uw_gex``) is supplied, the LLM
    receives the full Unusual Whales context bundle instead of a short summary.

    Returns structured predictions and an optional Hermes LLM narrative.
    """
    if cumulative_gex is None or cumulative_gex.empty:
        cumulative_gex = gex_by_strike.cumsum() if not gex_by_strike.empty else pd.Series(dtype=float)
    greek_df = None
    if agg is not None and hasattr(agg.gex_by_strike, "attrs"):
        raw = agg.gex_by_strike.attrs.get("greek_exposure_df")
        if isinstance(raw, pd.DataFrame) and not raw.empty:
            greek_df = raw
    if gamma_flip is None:
        gamma_flip = resolve_gamma_flip(
            spot=spot,
            gex_by_strike=gex_by_strike,
            cumulative_gex=cumulative_gex,
            greek_exposure_df=greek_df,
        )

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

    uw_bundle: dict[str, Any] | None = None
    if agg is not None:
        uw_bundle = build_uw_context_bundle(
            ticker=ticker,
            spot=spot,
            agg=agg,
            gamma_flip=gamma_flip,
            spot_gamma_bn=spot_gamma_bn,
            history=history,
            knn_prediction=knn_prediction,
            api_key=api_key,
            fetch_extras=bool(api_key),
        )

    who, whom, what = _default_who_what(total_gex_bn)
    prompt = _build_llm_prompt(
        ticker=ticker,
        spot=spot,
        total_gex_bn=total_gex_bn,
        base=base,
        gamma_flip=gamma_flip,
        exposure_type=exposure_type,
        uw_bundle=uw_bundle,
    )
    system = _HERMES_FULL_DATA_PROMPT if uw_bundle else _HERMES_SYSTEM_PROMPT
    hermes_narrative = _hermes_analyze(prompt, system_prompt=system)
    narrative = hermes_narrative or base.narrative

    hermes_active = hermes_narrative is not None
    result: dict[str, Any] = {
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
        "pattern_matches": _pattern_matches_from_learning(uw_bundle),
        "trading_notes": _trading_notes_from_strategy(uw_bundle),
        "narrative": narrative,
        "hermes_enhanced": hermes_active,
        "llm_enhanced": hermes_active,
        "agent_source": "hermes-agent + gex_core" if hermes_active else "gex_core",
        "uw_data_fed": uw_bundle is not None,
    }
    if uw_bundle:
        from gex_core.uw_context_bundle import bundle_token_estimate

        result["context_summary"] = {
            "strike_rows": len(uw_bundle.get("greek_exposure_by_strike", [])),
            "spot_exposure_rows": len(uw_bundle.get("spot_exposures_by_strike", [])),
            "estimated_tokens": bundle_token_estimate(uw_bundle),
        }
    return result


def predict_market_exposure(
    *,
    ticker: str,
    spot: float,
    gex_by_strike: pd.Series,
    cumulative_gex: pd.Series | None = None,
    total_gex_bn: float,
    agg,
    gamma_flip: float | None = None,
    spot_gamma_bn: float | None = None,
    history: list[dict] | None = None,
    knn_prediction: dict[str, Any] | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Feed all Unusual Whales data to the AI and return structured predictions.

    Uses OpenAI JSON mode when available, Hermes as fallback, rule-based
    analysis when no LLM is configured.
    """
    if cumulative_gex is None or cumulative_gex.empty:
        cumulative_gex = gex_by_strike.cumsum() if not gex_by_strike.empty else pd.Series(dtype=float)
    greek_df = None
    if hasattr(agg.gex_by_strike, "attrs"):
        raw = agg.gex_by_strike.attrs.get("greek_exposure_df")
        if isinstance(raw, pd.DataFrame) and not raw.empty:
            greek_df = raw
    if gamma_flip is None:
        gamma_flip = resolve_gamma_flip(
            spot=spot,
            gex_by_strike=gex_by_strike,
            cumulative_gex=cumulative_gex,
            greek_exposure_df=greek_df,
        )

    return predict_from_uw_data(
        ticker=ticker,
        spot=spot,
        gex_by_strike=gex_by_strike,
        cumulative_gex=cumulative_gex,
        total_gex_bn=total_gex_bn,
        agg=agg,
        gamma_flip=gamma_flip,
        spot_gamma_bn=spot_gamma_bn,
        history=history,
        knn_prediction=knn_prediction,
        api_key=api_key,
        fetch_extras=bool(api_key),
    )
