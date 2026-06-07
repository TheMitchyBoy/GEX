"""AI advisor for auto-trader entries — uses trade memory and UW context."""

from __future__ import annotations

import json
import logging
from typing import Any

from gex_core.gex_chatbot import _openai_chat, _resolve_openai_config
from gex_core.trading.config import (
    advisor_context_max_chars,
    clear_all_filters,
    min_entry_confidence,
    min_gamma_delta,
)
from gex_core.trading.filters import MarketContext, evaluate_entry_filters
from gex_core.trading.journal import get_trade_memory_for_ai
from gex_core.uw_context_bundle import bundle_to_prompt_json

logger = logging.getLogger(__name__)

_ADVISOR_SYSTEM = (
    "You are the GEX Auto-Trader advisor. You review gamma-based option entry signals and "
    "decide whether to approve a paper trade. You receive the full candidate signal pack, "
    "trade memory, and Unusual Whales context when available.\n\n"
    "Rules:\n"
    "- Prefer trades aligned with dealer gamma magnets (max positive gamma strikes).\n"
    "- Require spot momentum toward the magnet and regime alignment.\n"
    "- Reject when gamma at the magnet is declining (negative gamma_delta on max +γ).\n"
    "- Down-weight signal types with poor historical win rate in trade_memory.\n"
    "- confidence 0.0-1.0: approve only when edge is clear (typically >= 0.55).\n"
    "- If uw_context is present, cite specific strikes, flow, and intraday trends.\n"
    "- Output ONLY valid JSON with keys: approve (bool), confidence (float), option_type (call|put), "
    "reason (string), suggestions (array of strings).\n"
    "- This is paper trading analysis, not financial advice.\n"
)


def _build_advisor_context(
    *,
    signals: dict[str, Any],
    memory: dict[str, Any],
    market: MarketContext | None,
    uw_bundle: dict[str, Any] | None,
) -> str:
    payload: dict[str, Any] = {
        "candidate_signals": signals,
        "trade_memory": memory,
    }
    if market is not None:
        payload["market"] = {
            "spot": market.spot,
            "prev_spot": market.prev_spot,
            "regime": market.regime,
            "gamma_flip": market.gamma_flip,
            "flow_net_delta_gex_bn": market.flow_net_delta_gex_bn,
        }
    if uw_bundle:
        payload["uw_context"] = uw_bundle
    text = bundle_to_prompt_json(payload)
    cap = advisor_context_max_chars()
    if len(text) > cap:
        compact = {
            "candidate_signals": signals,
            "trade_memory": memory,
            "uw_summary": (uw_bundle or {}).get("summary"),
            "uw_strikes_near_spot": (uw_bundle or {}).get("strikes_near_spot"),
            "knn_forecast": (uw_bundle or {}).get("knn_forecast"),
        }
        text = bundle_to_prompt_json(compact)
        if len(text) > cap:
            text = text[:cap]
    return text


def _apply_advisor_gates(
    parsed: dict[str, Any],
    *,
    signals: dict[str, Any],
    market: MarketContext | None,
    uw_bundle: dict[str, Any] | None,
    memory: dict[str, Any],
) -> dict[str, Any]:
    parsed["confidence"] = float(parsed.get("confidence", 0.5))
    parsed["approve"] = bool(parsed.get("approve"))
    parsed["option_type"] = str(
        parsed.get("option_type", signals.get("recommended", {}).get("option_type", "call"))
    )
    parsed["reason"] = str(parsed.get("reason", ""))
    parsed["suggestions"] = parsed.get("suggestions") or memory.get("performance", {}).get("lessons", [])

    floor = min_entry_confidence()
    if parsed["approve"] and floor > 0 and parsed["confidence"] < floor:
        parsed["approve"] = False
        parsed["reason"] = (
            f"Confidence {parsed['confidence']:.2f} below minimum {floor:.2f} — "
            + (parsed["reason"] or "wait for clearer edge.")
        )

    if parsed["approve"]:
        filter_result = evaluate_entry_filters(signals, market=market, uw_bundle=uw_bundle)
        if not filter_result.get("approve"):
            parsed["approve"] = False
            parsed["reason"] = filter_result.get("reason", parsed["reason"])
        parsed["size_multiplier"] = float(filter_result.get("size_multiplier") or 1.0)
        if filter_result.get("filter"):
            parsed["filter"] = filter_result.get("filter")
    else:
        parsed.setdefault("size_multiplier", 0.0)

    return parsed


def _rule_based_advice(
    signals: dict[str, Any],
    memory: dict[str, Any],
    *,
    market: MarketContext | None = None,
    uw_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rec = signals.get("recommended") or {}
    perf = memory.get("performance") or {}
    by_signal = perf.get("by_signal") or {}
    sig_type = rec.get("signal_type", "unknown")
    hist = by_signal.get(sig_type, {})
    hist_wr = float(hist.get("win_rate", 0.55))
    hist_avg = float(hist.get("avg_pnl_pct", 0.02))

    score = float(rec.get("score", 0))
    delta = float(rec.get("gamma_delta", 0))
    confidence = min(0.95, 0.45 + min(score, 2.0) * 0.15 + min(max(delta, 0), 1.0) * 0.2 + hist_wr * 0.15)
    if hist.get("count", 0) >= 5 and hist_avg < -0.03:
        confidence *= 0.7

    filter_result = evaluate_entry_filters(signals, market=market, uw_bundle=uw_bundle)
    if not filter_result.get("approve"):
        return {
            "approve": False,
            "confidence": round(confidence * 0.8, 3),
            "option_type": rec.get("option_type", "call"),
            "reason": filter_result.get("reason", "Entry filters rejected trade."),
            "suggestions": [filter_result.get("reason", "Wait for cleaner setup.")],
            "source": "rule_based",
            "filter": filter_result.get("filter"),
            "size_multiplier": float(filter_result.get("size_multiplier") or 0.0),
        }

    if hist.get("count", 0) >= 3 and float(hist.get("avg_pnl_pct", 0)) < -0.03:
        confidence *= 0.75

    min_delta = min_gamma_delta()
    score_ok = min_delta <= 0 or score > min_delta
    delta_ok = min_delta <= 0 or delta >= min_delta
    approve = True if clear_all_filters() else (score_ok and delta_ok)
    suggestions = list(perf.get("lessons") or [])[:3]
    if delta > 0.08:
        suggestions.insert(0, "Strong gamma acceleration — momentum entry favored.")
    if not approve:
        suggestions.insert(0, "Wait for stronger gamma edge or better signal-type track record.")

    verdict = {
        "approve": approve,
        "confidence": round(confidence, 3),
        "option_type": rec.get("option_type", "call"),
        "reason": rec.get("rationale", "Rule-based gamma signal review."),
        "suggestions": suggestions[:5],
        "source": "rule_based",
        "size_multiplier": float(filter_result.get("size_multiplier") or 1.0),
    }
    return _apply_advisor_gates(
        verdict,
        signals=signals,
        market=market,
        uw_bundle=uw_bundle,
        memory=memory,
    )


def advise_entry(
    *,
    ticker: str,
    signals: dict[str, Any],
    uw_bundle: dict[str, Any] | None = None,
    market: MarketContext | None = None,
) -> dict[str, Any]:
    """Return AI or rule-based entry verdict with suggestions."""
    memory = get_trade_memory_for_ai(ticker)
    context = _build_advisor_context(
        signals=signals,
        memory=memory,
        market=market,
        uw_bundle=uw_bundle,
    )

    if _resolve_openai_config():
        system = _ADVISOR_SYSTEM + "\n\nContext:\n" + context
        prompt = (
            "Review the recommended gamma signal, trade memory, and market context. "
            "Should we open a paper option trade? Return JSON only."
        )
        reply, err = _openai_chat(system, [], prompt, json_mode=True, temperature=0.2, max_tokens=600)
        if reply:
            try:
                parsed = json.loads(reply)
                parsed["source"] = "openai"
                return _apply_advisor_gates(
                    parsed,
                    signals=signals,
                    market=market,
                    uw_bundle=uw_bundle,
                    memory=memory,
                )
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.warning("AI advisor JSON parse failed: %s", exc)
        if err:
            logger.info("AI advisor falling back to rules: %s", err)

    return _rule_based_advice(signals, memory, market=market, uw_bundle=uw_bundle)


def build_suggestions(ticker: str) -> list[str]:
    """Actionable suggestions from performance memory."""
    memory = get_trade_memory_for_ai(ticker)
    perf = memory.get("performance") or {}
    out = list(perf.get("lessons") or [])
    open_count = len(memory.get("open_positions") or [])
    if open_count >= 2:
        out.insert(0, f"{open_count} open positions — consider tighter entries until exits clear.")
    if perf.get("total_trades", 0) < 5:
        out.append("Limited trade history — AI confidence will improve as more paper trades complete.")
    wr = float(perf.get("win_rate") or 0)
    if wr > 0.6 and perf.get("total_trades", 0) >= 10:
        out.append(f"Win rate {wr:.0%} — current gamma entry rules are performing well.")
    elif wr < 0.4 and perf.get("total_trades", 0) >= 10:
        out.append(f"Win rate {wr:.0%} — AI is down-weighting weak signal types automatically.")
    return out[:6]
