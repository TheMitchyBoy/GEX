"""
LLM predictions from comprehensive Unusual Whales context bundles.

Uses OpenAI (preferred) or Hermes when configured; falls back to rule-based
predictions from ``ai_analyst`` when no LLM is available.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from gex_core.ai_analyst import GammaAnalysis, analyze_dealer_gamma
from gex_core.uw_context_bundle import build_uw_context_bundle, bundle_to_prompt_json

logger = logging.getLogger(__name__)

_PREDICTION_SCHEMA = {
    "predicted_regime": "LONG gamma | SHORT gamma",
    "predicted_delta_gex_bn": "float — expected change in total GEX next snapshot",
    "predicted_total_gex_bn": "float — expected total GEX next snapshot",
    "spot_bias": "bullish | bearish | neutral",
    "confidence": "0.0–1.0",
    "gamma_flip": "float strike or null",
    "key_levels": {
        "support": ["float strikes"],
        "resistance": ["float strikes"],
        "pin": "float strike or null",
    },
    "scenarios": [
        {"label": "str", "probability": "0.0–1.0", "description": "str"}
    ],
    "predictions": ["list of 3–5 actionable prediction strings"],
    "reasoning": "2–3 sentence explanation citing specific data points",
}

_SYSTEM_PROMPT = (
    "You are an expert options market analyst specializing in dealer gamma exposure (GEX). "
    "You receive a comprehensive JSON bundle of Unusual Whales data: strike-level greeks, "
    "spot exposures, expiration term structure, intraday minute bars, extended features "
    "(charm, vanna, delta, VIX, events), snapshot history, and optional KNN forecast.\n\n"
    "Analyze ALL provided data points holistically. Weight:\n"
    "- Net GEX regime and gamma flip proximity\n"
    "- Call/put walls and strike concentration\n"
    "- Charm/vanna/delta exposures and spot-exposure OI vs volume\n"
    "- Intraday gamma trend and price action\n"
    "- Term structure (0DTE vs near-term ratios from expiration data)\n"
    "- Snapshot momentum and KNN forecast when present\n"
    "- Vol regime and event calendar flags\n\n"
    "Respond with ONLY valid JSON matching this schema:\n"
    f"{json.dumps(_PREDICTION_SCHEMA, indent=2)}"
)


def _resolve_openai_config() -> tuple[str, str] | None:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    model = os.environ.get("GEX_AGENT_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    return key, model


def _openai_predict(bundle: dict[str, Any]) -> dict[str, Any] | None:
    cfg = _resolve_openai_config()
    if not cfg:
        return None
    api_key, model = cfg
    try:
        import openai

        client = openai.OpenAI(api_key=api_key)
        user_content = (
            "Using every data point in this Unusual Whales context bundle, "
            "produce structured market predictions for the next snapshot interval.\n\n"
            f"{bundle_to_prompt_json(bundle)}"
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            max_tokens=int(os.environ.get("GEX_AI_MAX_TOKENS", "1200")),
            temperature=float(os.environ.get("GEX_AI_TEMPERATURE", "0.3")),
        )
        raw = resp.choices[0].message.content
        return _parse_prediction_json(raw)
    except Exception as exc:
        logger.warning("OpenAI UW prediction failed: %s", exc)
        return None


def _hermes_predict(bundle: dict[str, Any]) -> dict[str, Any] | None:
    from gex_core.market_exposure_agent import _hermes_analyze, _resolve_hermes_llm_config

    if not _resolve_hermes_llm_config():
        return None
    prompt = (
        "Analyze this Unusual Whales GEX data bundle and respond with ONLY valid JSON "
        f"matching this schema:\n{json.dumps(_PREDICTION_SCHEMA)}\n\n"
        f"Data:\n{bundle_to_prompt_json(bundle)}"
    )
    raw = _hermes_analyze(prompt)
    if not raw:
        return None
    return _parse_prediction_json(raw)


def _parse_prediction_json(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    text = raw.strip()
    # Strip markdown fences if the model wrapped JSON
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        parsed = json.loads(text)
        return _normalize_prediction(parsed)
    except json.JSONDecodeError:
        # Try to extract first JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return _normalize_prediction(json.loads(text[start : end + 1]))
            except json.JSONDecodeError:
                pass
    logger.warning("Failed to parse LLM prediction JSON")
    return None


def _normalize_prediction(parsed: dict[str, Any]) -> dict[str, Any]:
    """Ensure required fields exist with sane defaults."""
    out: dict[str, Any] = {
        "predicted_regime": str(parsed.get("predicted_regime", "neutral")),
        "predicted_delta_gex_bn": _safe_float(parsed.get("predicted_delta_gex_bn")),
        "predicted_total_gex_bn": _safe_float(parsed.get("predicted_total_gex_bn")),
        "spot_bias": str(parsed.get("spot_bias", parsed.get("predicted_spot_bias", "neutral"))).lower(),
        "confidence": min(1.0, max(0.0, _safe_float(parsed.get("confidence"), 0.5))),
        "gamma_flip": _safe_float(parsed.get("gamma_flip")) or None,
        "key_levels": parsed.get("key_levels") or {"support": [], "resistance": [], "pin": None},
        "scenarios": parsed.get("scenarios") or [],
        "predictions": parsed.get("predictions") or [],
        "reasoning": str(parsed.get("reasoning", "")),
        "llm_enhanced": True,
        "prediction_source": parsed.get("prediction_source", "llm"),
    }
    if not out["predictions"] and out["reasoning"]:
        out["predictions"] = [out["reasoning"]]
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rule_based_predictions(base: GammaAnalysis, bundle: dict[str, Any]) -> dict[str, Any]:
    summary = bundle.get("summary") or {}
    total = _safe_float(summary.get("total_gex_bn"))
    delta = 0.0
    knn = bundle.get("knn_forecast") or {}
    if knn:
        delta = _safe_float(knn.get("predicted_delta_gex_bn"))
        total = _safe_float(knn.get("predicted_total_gex_bn"), total + delta)
    return {
        "predicted_regime": "LONG gamma" if total >= 0 else "SHORT gamma",
        "predicted_delta_gex_bn": round(delta, 4),
        "predicted_total_gex_bn": round(total, 4),
        "spot_bias": base.bias,
        "confidence": base.confidence,
        "gamma_flip": base.gamma_flip,
        "key_levels": {
            "support": [base.put_wall] if base.put_wall else [],
            "resistance": [base.call_wall] if base.call_wall else [],
            "pin": base.dominant_strike,
        },
        "scenarios": [],
        "predictions": base.predictions,
        "reasoning": base.narrative,
        "llm_enhanced": False,
        "prediction_source": "rule_based",
    }


def predict_from_uw_data(
    *,
    ticker: str,
    spot: float,
    gex_by_strike,
    cumulative_gex,
    total_gex_bn: float,
    agg,
    gamma_flip: float | None = None,
    spot_gamma_bn: float | None = None,
    history: list[dict] | None = None,
    knn_prediction: dict[str, Any] | None = None,
    api_key: str | None = None,
    fetch_extras: bool = True,
) -> dict[str, Any]:
    """
    Build full UW context and produce AI predictions.

    Returns structured predictions plus the context bundle metadata.
    """
    bundle = build_uw_context_bundle(
        ticker=ticker,
        spot=spot,
        agg=agg,
        gamma_flip=gamma_flip,
        spot_gamma_bn=spot_gamma_bn,
        history=history,
        knn_prediction=knn_prediction,
        api_key=api_key,
        fetch_extras=fetch_extras,
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

    llm_result = _openai_predict(bundle)
    if llm_result is None:
        llm_result = _hermes_predict(bundle)
    if llm_result is not None:
        llm_result = _normalize_prediction(llm_result)
    else:
        llm_result = _rule_based_predictions(base, bundle)

    from gex_core.prediction_log import calibrated_llm_confidence, log_llm_prediction
    from gex_core.uw_context_bundle import bundle_token_estimate

    anchor_ts = None
    if history:
        anchor_ts = history[-1].get("ts")
    source = "llm" if llm_result.get("llm_enhanced") else str(llm_result.get("prediction_source", "rule_based"))
    log_llm_prediction(
        ticker=ticker,
        source=source,
        prediction=llm_result,
        snapshot_ts=anchor_ts,
        market_date=bundle.get("market_date"),
    )
    if llm_result.get("confidence") is not None:
        llm_result = dict(llm_result)
        llm_result["raw_confidence"] = llm_result["confidence"]
        llm_result["confidence"] = calibrated_llm_confidence(
            float(llm_result["confidence"]),
            ticker,
            source=source if source == "llm" else None,
        )

    return {
        **llm_result,
        "ticker": ticker.upper(),
        "spot": spot,
        "regime": base.regime,
        "bias": base.bias,
        "narrative": llm_result.get("reasoning") or base.narrative,
        "signals": [s.label for s in base.signals[:8]],
        "context_summary": {
            "strike_rows": len(bundle.get("greek_exposure_by_strike", [])),
            "spot_exposure_rows": len(bundle.get("spot_exposures_by_strike", [])),
            "expiration_rows": len(bundle.get("gex_by_expiration", [])),
            "intraday_bars": len((bundle.get("intraday") or {}).get("minute_bars", [])),
            "snapshot_history_rows": len(bundle.get("snapshot_history", [])),
            "extended_feature_count": len(bundle.get("extended_features") or {}),
            "has_knn_forecast": bool(bundle.get("knn_forecast")),
            "estimated_tokens": bundle_token_estimate(bundle),
        },
    }
