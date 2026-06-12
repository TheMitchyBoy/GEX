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
    with patch("gex_core.gex_chatbot._openai_chat", return_value=(None, None)), patch(
        "gex_core.gex_chatbot._hermes_chat", return_value=(None, None)
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
    with patch("gex_core.gex_chatbot._openai_chat", return_value=("Dealers are long gamma near 5050.", None)):
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


def test_session_reset(tmp_path, monkeypatch):
    monkeypatch.setenv("GEX_TRADING_DB", str(tmp_path / "journal.db"))
    session = get_or_create_session(None, ticker="SPX")
    session.messages.append({"role": "user", "content": "hi"})
    reset_session(session.session_id)
    fresh = get_or_create_session(session.session_id, ticker="SPX")
    assert fresh.messages == []


def test_session_persists_to_sqlite(tmp_path, monkeypatch):
    monkeypatch.setenv("GEX_TRADING_DB", str(tmp_path / "journal.db"))
    with monkeypatch.context() as m:
        m.setattr("gex_core.gex_chatbot._openai_chat", lambda *a, **k: ("Persisted reply.", None))
        first = chat_reply(
            session_id=None,
            user_message="What is the regime?",
            ticker="SPX",
            spot=5000.0,
            gex_by_strike=__import__("pandas").Series({5000: 1.0}),
            cumulative_gex=__import__("pandas").Series({5000: 1.0}),
            total_gex_bn=1.0,
        )
    sid = first["session_id"]
    from gex_core.chat_store import load_session

    stored = load_session(sid)
    assert stored is not None
    assert len(stored["messages"]) == 2
