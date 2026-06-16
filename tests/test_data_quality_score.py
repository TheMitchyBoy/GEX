"""Tests for data quality scoring helpers."""

import pandas as pd

from gex_core.data_quality_score import (
    compute_quality_score,
    regime_consistent,
    regime_from_total_gex,
    should_hard_reject_total_gex_mismatch,
    strike_profile_confidence,
    total_gex_tolerance_bn,
)


def test_regime_from_total_gex():
    assert regime_from_total_gex(1.0) == "LONG gamma"
    assert regime_from_total_gex(-0.1) == "SHORT gamma"


def test_regime_consistent_matches_slope():
    assert regime_consistent(1.0, 0.5) is True
    assert regime_consistent(-1.0, -0.5) is True
    assert regime_consistent(1.0, -0.5) is False


def test_compute_quality_score_high_when_clean():
    score = compute_quality_score(
        brackets_spot=True,
        spot_disagreement_pct=0.0,
        total_gex_consistent=True,
        data_lag_sec=30.0,
        strike_profile_source="live_spot_exposures",
        strike_count=25,
        validation_ok=True,
    )
    assert score >= 0.9


def test_compute_quality_score_lower_when_misaligned():
    good = compute_quality_score(
        brackets_spot=True,
        spot_disagreement_pct=0.0,
        total_gex_consistent=True,
        data_lag_sec=30.0,
        strike_profile_source="live_spot_exposures",
        strike_count=25,
        validation_ok=True,
    )
    bad = compute_quality_score(
        brackets_spot=False,
        spot_disagreement_pct=0.02,
        total_gex_consistent=False,
        data_lag_sec=5000.0,
        strike_profile_source="eod_scaled",
        strike_count=3,
        validation_ok=False,
    )
    assert bad < good


def test_strike_profile_confidence_mapping():
    assert strike_profile_confidence("live_spot_exposures") == "high"
    assert strike_profile_confidence("eod_scaled") == "low"


def test_total_gex_tolerance_default():
    assert total_gex_tolerance_bn() == 0.05


def test_should_hard_reject_respects_env(monkeypatch):
    monkeypatch.setenv("GEX_HARD_REJECT_TOTAL_GEX_MISMATCH", "0")
    assert should_hard_reject_total_gex_mismatch(mismatch=True) is False
