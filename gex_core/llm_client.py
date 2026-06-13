"""Shared OpenAI client helpers for advisor and daily-learning features."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def resolve_openai_config() -> tuple[str, str] | None:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    model = os.environ.get("GEX_AGENT_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    return key, model


def _classify_llm_error(exc: Exception) -> str:
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


def openai_chat(
    system: str,
    history: list[dict[str, str]],
    user_message: str,
    *,
    json_mode: bool = False,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> tuple[str | None, str | None]:
    """Return (reply, user_error). user_error is set when the call fails."""
    cfg = resolve_openai_config()
    if not cfg:
        return None, None
    api_key, model = cfg
    try:
        import openai

        client = openai.OpenAI(api_key=api_key)
        messages = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else int(os.environ.get("GEX_AI_MAX_TOKENS", "1200")),
            "temperature": temperature if temperature is not None else float(os.environ.get("GEX_AI_TEMPERATURE", "0.35")),
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content
        return (content.strip() if content else None), None
    except Exception as exc:
        logger.warning("OpenAI chat failed: %s", exc)
        return None, _classify_llm_error(exc)
