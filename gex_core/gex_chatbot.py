"""
GEX chatbot — conversational assistant backed by full Unusual Whales context.

Maintains in-memory session history and answers user questions using OpenAI
(preferred), Hermes, or rule-based dealer gamma analysis as fallback.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from gex_core.ai_analyst import analyze_dealer_gamma
from gex_core.features import estimate_gamma_flip
from gex_core.market_exposure_agent import _hermes_analyze, _resolve_hermes_llm_config
from gex_core.uw_context_bundle import build_uw_context_bundle, bundle_to_prompt_json

logger = logging.getLogger(__name__)

_CHAT_SYSTEM = (
    "You are the GEX Assistant — an expert in dealer gamma exposure (GEX) for SPX intraday trading. "
    "You have access to a comprehensive Unusual Whales data bundle (strike greeks, spot exposures, "
    "term structure, intraday bars, charm/vanna/delta, VIX, events, snapshot history, KNN forecast).\n\n"
    "Rules:\n"
    "- Answer the user's question directly using specific numbers and strikes from the data.\n"
    "- Be concise (2–5 sentences unless they ask for detail).\n"
    "- When asked for predictions, cite regime, gamma flip, walls, and intraday trends.\n"
    "- If data is missing, say so — do not invent levels.\n"
    "- This is market analysis, not financial advice.\n"
    "- When trade_memory is present in the data bundle, use it to refine predictions and "
    "reference past paper-trade performance.\n"
    "- When trader_backtest is present, it is a walk-forward simulation using the CURRENT "
    "live auto-trader parameters (risk %, stops, filters). Cite those numbers when asked "
    "about backtests or strategy performance.\n"
)

_MAX_SESSIONS = int(os.environ.get("GEX_CHAT_MAX_SESSIONS", "200"))
_MAX_MESSAGES = int(os.environ.get("GEX_CHAT_MAX_MESSAGES", "40"))
_SESSION_TTL_SECONDS = int(os.environ.get("GEX_CHAT_SESSION_TTL_SECONDS", "7200"))


@dataclass
class ChatSession:
    session_id: str
    messages: list[dict[str, str]] = field(default_factory=list)
    updated_at: float = field(default_factory=time.monotonic)


_sessions: dict[str, ChatSession] = {}
_lock = threading.Lock()


def _resolve_openai_config() -> tuple[str, str] | None:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    model = os.environ.get("GEX_AGENT_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    return key, model


def _prune_sessions() -> None:
    now = time.monotonic()
    stale = [sid for sid, s in _sessions.items() if now - s.updated_at > _SESSION_TTL_SECONDS]
    for sid in stale:
        _sessions.pop(sid, None)
    if len(_sessions) <= _MAX_SESSIONS:
        return
    ordered = sorted(_sessions.items(), key=lambda item: item[1].updated_at)
    for sid, _ in ordered[: len(_sessions) - _MAX_SESSIONS]:
        _sessions.pop(sid, None)


def get_or_create_session(session_id: str | None) -> ChatSession:
    with _lock:
        _prune_sessions()
        if session_id and session_id in _sessions:
            session = _sessions[session_id]
            session.updated_at = time.monotonic()
            return session
        new_id = session_id or uuid.uuid4().hex
        session = ChatSession(session_id=new_id)
        _sessions[new_id] = session
        return session


def reset_session(session_id: str) -> None:
    with _lock:
        _sessions.pop(session_id, None)


def build_welcome_message(
    *,
    ticker: str,
    spot: float | None,
    regime: str,
    total_gex: float,
    gamma_flip: float | None,
    exposure: str,
) -> str:
    spot_str = f"${spot:,.0f}" if spot else "N/A"
    flip_str = f"{gamma_flip:.0f}" if gamma_flip else "N/A"
    return (
        f"Hi — I'm your **GEX assistant** for {ticker}. "
        f"I have Unusual Whales {exposure} exposure for this snapshot "
        f"(spot {spot_str}, {regime}, net GEX {total_gex:+.2f} Bn$/%, flip {flip_str}).\n\n"
        "Ask me about regime, key levels, predictions, charm/vanna, or trade setups. "
        "Ask me to **backtest** the strategy to simulate current auto-trader settings on history."
    )


def _build_system_message(
    *,
    ticker: str,
    spot: float,
    total_gex_bn: float,
    exposure: str,
    uw_bundle: dict[str, Any] | None,
    base_analysis,
    trader_backtest: dict[str, Any] | None = None,
) -> str:
    parts = [_CHAT_SYSTEM, f"Current ticker: {ticker}, exposure type: {exposure}, spot: {spot:,.0f}."]
    if uw_bundle:
        parts.append(f"Unusual Whales data bundle:\n{bundle_to_prompt_json(uw_bundle)}")
    elif not trader_backtest:
        parts.append(
            f"Summary (limited data): regime={base_analysis.regime}, "
            f"net GEX={total_gex_bn:+.2f} Bn$/%, bias={base_analysis.bias}, "
            f"flip={base_analysis.gamma_flip}, call wall={base_analysis.call_wall}, "
            f"put wall={base_analysis.put_wall}."
        )
    if trader_backtest:
        parts.append(f"Trader walk-forward backtest (current parameters):\n{bundle_to_prompt_json(trader_backtest)}")
    return "\n\n".join(parts)


def _classify_llm_error(exc: Exception) -> str:
    """Map an LLM exception to a short user-facing reason."""
    text = str(exc).lower()
    if "insufficient_quota" in text or ("429" in text and "quota" in text):
        return "OpenAI quota exceeded — add billing credits at platform.openai.com"
    if "invalid_api_key" in text or "incorrect api key" in text or "401" in text:
        return "OpenAI API key is invalid — check OPENAI_API_KEY"
    if "rate_limit" in text or ("429" in text and "quota" not in text):
        return "OpenAI rate limit hit — try again shortly"
    if "model" in text and ("not found" in text or "does not exist" in text):
        return "OpenAI model unavailable — check GEX_AGENT_MODEL"
    return "LLM request failed — see server logs for details"


def _openai_chat(system: str, history: list[dict[str, str]], user_message: str) -> tuple[str | None, str | None]:
    """Return (reply, user_error). user_error is set when the call fails."""
    cfg = _resolve_openai_config()
    if not cfg:
        return None, None
    api_key, model = cfg
    try:
        import openai

        client = openai.OpenAI(api_key=api_key)
        messages = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=int(os.environ.get("GEX_AI_MAX_TOKENS", "800")),
            temperature=float(os.environ.get("GEX_AI_TEMPERATURE", "0.4")),
        )
        content = resp.choices[0].message.content
        return (content.strip() if content else None), None
    except Exception as exc:
        logger.warning("OpenAI chat failed: %s", exc)
        return None, _classify_llm_error(exc)


def _hermes_chat(system: str, history: list[dict[str, str]], user_message: str) -> tuple[str | None, str | None]:
    if not _resolve_hermes_llm_config():
        return None, None
    transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history[-12:])
    prompt = (
        f"{transcript}\nUSER: {user_message}\n\n"
        "Reply as the GEX assistant. Be concise and cite specific data."
    )
    reply = _hermes_analyze(prompt, system_prompt=system)
    if reply:
        return reply, None
    if _resolve_openai_config():
        return None, None
    return None, "Hermes/OpenRouter LLM unavailable — install hermes-agent or set OPENAI_API_KEY"


def _fallback_notice(llm_errors: list[str]) -> str:
    if llm_errors:
        return f"\n*(Rule-based reply — {llm_errors[0]})*"
    if not _resolve_openai_config() and not _resolve_hermes_llm_config():
        return "\n*(Rule-based reply — set OPENAI_API_KEY for full conversational AI with all UW data.)*"
    return "\n*(Rule-based reply — LLM backends unavailable.)*"


def _rule_based_reply(user_message: str, base_analysis, *, llm_errors: list[str] | None = None) -> str:
    msg = user_message.lower()
    parts: list[str] = []

    if any(w in msg for w in ("regime", "gamma", "gex", "environment")):
        parts.append(base_analysis.regime_detail)
    if any(w in msg for w in ("flip", "inflection", "cross")):
        flip = base_analysis.gamma_flip
        parts.append(
            f"Gamma flip at {flip:.0f}." if flip else "No clear gamma flip in current data."
        )
    if any(w in msg for w in ("wall", "level", "support", "resistance", "target")):
        cw, pw = base_analysis.call_wall, base_analysis.put_wall
        if cw and pw:
            parts.append(f"Call wall {cw:.0f}, put wall {pw:.0f}.")
        elif cw:
            parts.append(f"Call wall at {cw:.0f}.")
    if any(w in msg for w in ("predict", "forecast", "expect", "outlook")):
        parts.extend(base_analysis.predictions[:2])

    if not parts:
        parts.append(base_analysis.narrative)

    parts.append(_fallback_notice(llm_errors or []))
    return " ".join(parts)


def chat_reply(
    *,
    session_id: str | None,
    user_message: str,
    ticker: str,
    spot: float,
    gex_by_strike: pd.Series,
    cumulative_gex: pd.Series | None,
    total_gex_bn: float,
    gamma_flip: float | None = None,
    exposure_type: str = "gamma",
    agg=None,
    spot_gamma_bn: float | None = None,
    history: list[dict] | None = None,
    knn_prediction: dict[str, Any] | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Process a user message and return assistant reply with session state."""
    user_message = (user_message or "").strip()
    if not user_message:
        return {"error": "Message cannot be empty"}

    session = get_or_create_session(session_id)

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

    uw_bundle = None
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

    trader_backtest = None
    try:
        from gex_core.trading.backtest_agent import run_agent_backtest, user_wants_backtest

        if user_wants_backtest(user_message):
            trader_backtest = run_agent_backtest(ticker)
    except Exception:
        logger.exception("Agent backtest failed for %s", ticker)

    system = _build_system_message(
        ticker=ticker,
        spot=spot,
        total_gex_bn=total_gex_bn,
        exposure=exposure_type,
        uw_bundle=uw_bundle,
        base_analysis=base,
        trader_backtest=trader_backtest,
    )

    prior = session.messages[-_MAX_MESSAGES:]
    llm_errors: list[str] = []
    reply, openai_err = _openai_chat(system, prior, user_message)
    llm_source = "openai"
    if openai_err:
        llm_errors.append(openai_err)
    if reply is None:
        reply, hermes_err = _hermes_chat(system, prior, user_message)
        llm_source = "hermes"
        if hermes_err:
            llm_errors.append(hermes_err)
    if reply is None:
        if trader_backtest is not None:
            from gex_core.trading.backtest_agent import format_backtest_reply

            reply = format_backtest_reply(trader_backtest) + _fallback_notice(llm_errors)
            llm_source = "rule_based"
        else:
            reply = _rule_based_reply(user_message, base, llm_errors=llm_errors)
            llm_source = "rule_based"

    with _lock:
        session.messages.append({"role": "user", "content": user_message})
        session.messages.append({"role": "assistant", "content": reply})
        if len(session.messages) > _MAX_MESSAGES:
            session.messages = session.messages[-_MAX_MESSAGES:]
        session.updated_at = time.monotonic()

    return {
        "session_id": session.session_id,
        "reply": reply,
        "llm_source": llm_source,
        "llm_error": llm_errors[0] if llm_errors and llm_source == "rule_based" else None,
        "openai_configured": _resolve_openai_config() is not None,
        "uw_data_fed": uw_bundle is not None,
        "trader_backtest": trader_backtest,
        "messages": list(session.messages),
    }
