"""Tests for GEX chatbot."""

from unittest.mock import patch

import pandas as pd

from gex_core.gex_chatbot import (
    build_welcome_message,
    chat_reply,
    get_or_create_session,
    reset_session,
)


def test_build_welcome_message():
    msg = build_welcome_message(
        ticker="SPX",
        spot=5050.0,
        regime="LONG gamma",
        total_gex=4.2,
        gamma_flip=5000.0,
        exposure="gamma",
    )
    assert "GEX assistant" in msg
    assert "$5,050" in msg
    assert "LONG gamma" in msg


def test_chat_reply_rule_based_without_llm():
    strikes = pd.Series({5000: 2.0, 5050: -1.0, 5100: 3.0})
    with patch("gex_core.gex_chatbot._openai_chat", return_value=None), patch(
        "gex_core.gex_chatbot._hermes_chat", return_value=None
    ):
        result = chat_reply(
            session_id=None,
            user_message="What is the gamma regime?",
            ticker="SPX",
            spot=5025.0,
            gex_by_strike=strikes,
            cumulative_gex=strikes.cumsum(),
            total_gex_bn=4.0,
            gamma_flip=5000.0,
        )
    assert result["reply"]
    assert result["llm_source"] == "rule_based"
    assert result["session_id"]
    assert len(result["messages"]) == 2


def test_chat_reply_openai_when_available():
    strikes = pd.Series({5000: 2.0, 5050: -1.0})
    with patch("gex_core.gex_chatbot._openai_chat", return_value="Dealers are long gamma near 5050."):
        result = chat_reply(
            session_id=None,
            user_message="Where is the pin?",
            ticker="SPX",
            spot=5025.0,
            gex_by_strike=strikes,
            cumulative_gex=strikes.cumsum(),
            total_gex_bn=1.0,
        )
    assert result["llm_source"] == "openai"
    assert "5050" in result["reply"]


def test_session_reset():
    session = get_or_create_session(None)
    session.messages.append({"role": "user", "content": "hi"})
    reset_session(session.session_id)
    fresh = get_or_create_session(session.session_id)
    assert fresh.messages == []
