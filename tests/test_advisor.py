"""Tests for the auto-trader AI advisor."""

from __future__ import annotations

import json

import pytest

from gex_core.trading.advisor import _apply_advisor_gates, _build_advisor_context, advise_entry
from gex_core.trading.config import min_entry_confidence
from gex_core.trading.filters import MarketContext


@pytest.fixture(autouse=True)
def strict_filters(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_CLEAR_FILTERS", "0")
    monkeypatch.setenv("GEX_TRADER_STRICT_FILTERS", "0")
    monkeypatch.setenv("GEX_TRADER_MIN_GAMMA_DELTA", "0")
    monkeypatch.setenv("GEX_TRADER_REQUIRE_MOMENTUM", "0")
    monkeypatch.setenv("GEX_TRADER_REQUIRE_FLIP_SIDE", "0")


def test_build_advisor_context_includes_uw_bundle():
    signals = {"available": True, "recommended": {"strike": 5010.0, "option_type": "call"}}
    memory = {"performance": {"total_trades": 2}}
    uw = {"summary": {"spot": 5000}, "strikes_near_spot": [{"strike": 5010}]}
    text = _build_advisor_context(signals=signals, memory=memory, market=None, uw_bundle=uw)
    assert "uw_context" in text
    assert "5010" in text


def test_apply_advisor_gates_rejects_low_confidence(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_MIN_ENTRY_CONFIDENCE", "0.60")
    signals = {
        "available": True,
        "recommended": {"strike": 5010.0, "option_type": "call", "score": 1.0, "gamma_delta": 0.1},
    }
    parsed = _apply_advisor_gates(
        {"approve": True, "confidence": 0.45, "reason": "weak"},
        signals=signals,
        market=MarketContext(spot=5000.0, regime="LONG gamma"),
        uw_bundle=None,
        memory={"performance": {}},
    )
    assert not parsed["approve"]
    assert "0.60" in parsed["reason"]


def test_advise_entry_uses_openai_json(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_MIN_ENTRY_CONFIDENCE", "0.0")

    def fake_openai(system, history, prompt, **kwargs):
        assert kwargs.get("json_mode") is True
        return (
            json.dumps(
                {
                    "approve": True,
                    "confidence": 0.82,
                    "option_type": "call",
                    "reason": "Strong magnet alignment.",
                    "suggestions": ["Momentum favors calls."],
                }
            ),
            None,
        )

    monkeypatch.setattr("gex_core.trading.advisor._openai_chat", fake_openai)
    monkeypatch.setattr("gex_core.trading.advisor._resolve_openai_config", lambda: ("key", "gpt-4o-mini"))

    signals = {
        "available": True,
        "recommended": {
            "signal_type": "max_positive_gamma",
            "strike": 5010.0,
            "option_type": "call",
            "score": 1.5,
            "gamma_delta": 0.12,
            "rationale": "Max positive gamma magnet",
        },
    }
    advice = advise_entry(
        ticker="SPX",
        signals=signals,
        market=MarketContext(spot=5000.0, prev_spot=4995.0, regime="LONG gamma"),
    )
    assert advice["source"] == "openai"
    assert advice["approve"] is True
    assert advice["confidence"] == 0.82


def test_min_entry_confidence_default_when_filters_on():
    assert min_entry_confidence() == 0.55
