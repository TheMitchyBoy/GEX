"""Tests for GEX_TRADER_CLEAR_FILTERS master switch."""

from __future__ import annotations

import pandas as pd

from gex_core.trading.advisor import _rule_based_advice
from gex_core.trading.config import (
    clear_all_filters,
    max_strike_distance_pct,
    min_gamma_delta,
    require_spot_momentum,
    strict_entry_filters,
)
from gex_core.trading.filters import MarketContext, evaluate_entry_filters
from gex_core.trading.signals import compute_entry_candidates


def test_clear_all_filters_default_off():
    assert not clear_all_filters()


def test_clear_all_filters_disables_strict_gates(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_CLEAR_FILTERS", "1")
    assert clear_all_filters()
    assert not strict_entry_filters()
    assert min_gamma_delta() == 0.0
    assert not require_spot_momentum()
    assert max_strike_distance_pct() == 1.0


def test_clear_all_filters_allows_declining_gamma_magnet(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_CLEAR_FILTERS", "1")
    monkeypatch.setenv("GEX_TRADER_MAX_GAMMA_ONLY", "1")
    cur = pd.Series({4990: 0.2, 5000: 0.4, 5010: 2.0})
    prev = pd.Series({4990: 0.2, 5000: 0.4, 5010: 2.5})
    out = compute_entry_candidates(cur, prev, spot=5000.0)
    assert out["available"] is True


def test_strict_filters_when_clear_off(monkeypatch):
    monkeypatch.delenv("GEX_TRADER_CLEAR_FILTERS", raising=False)
    monkeypatch.delenv("GEX_TRADER_STRICT_FILTERS", raising=False)

    assert not clear_all_filters()
    assert not strict_entry_filters()


def test_clear_all_filters_bypasses_advisor_score_floor(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_CLEAR_FILTERS", "1")
    signals = {
        "spot": 5000.0,
        "recommended": {
            "signal_type": "max_positive_gamma",
            "strike": 5010.0,
            "option_type": "call",
            "score": 0.0,
            "gamma_delta": -0.05,
        },
    }
    advice = _rule_based_advice(signals, {"performance": {}}, market=MarketContext(spot=5000.0))
    assert advice["approve"] is True

    filt = evaluate_entry_filters(signals, market=MarketContext(spot=5000.0))
    assert filt["approve"] is True
