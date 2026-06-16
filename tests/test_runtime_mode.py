"""Tests for processor runtime mode flags."""

from gex_core.runtime_mode import (
    is_processor_mode,
    lightweight_market_context_enabled,
    reconcile_predictions_enabled,
    summary_market_features_enabled,
)


def test_processor_mode_env(monkeypatch):
    monkeypatch.setenv("GEX_PROCESSOR_MODE", "1")
    assert is_processor_mode() is True
    assert summary_market_features_enabled() is False
    assert lightweight_market_context_enabled() is True
    assert reconcile_predictions_enabled() is True

    monkeypatch.setenv("GEX_SUMMARY_MARKET_FEATURES", "1")
    assert summary_market_features_enabled() is True

    monkeypatch.setenv("GEX_LIGHTWEIGHT_MARKET_CONTEXT", "0")
    assert lightweight_market_context_enabled() is False


def test_web_mode_defaults(monkeypatch):
    monkeypatch.delenv("GEX_PROCESSOR_MODE", raising=False)
    monkeypatch.delenv("GEX_SUMMARY_MARKET_FEATURES", raising=False)
    assert is_processor_mode() is False
    assert summary_market_features_enabled() is True
