"""Tests for snapshot processing pipeline."""

import pandas as pd

from gex_core.snapshot_processing import (
    derive_snapshot_features,
    export_ts_from_uw_time,
    prepare_snapshot_for_storage,
    strike_profile_hash,
    strikes_equal,
    validate_snapshot,
)


def test_strike_profile_hash_stable():
    strike = pd.Series({6000.0: 1.0, 6050.0: -0.5})
    assert strike_profile_hash(strike) == strike_profile_hash(strike.copy())


def test_strikes_equal_detects_duplicate():
    left = pd.Series({6000.0: 1.0, 6050.0: -0.5})
    right = pd.Series({6000.0: 1.0, 6050.0: -0.5})
    different = pd.Series({6000.0: 2.0, 6050.0: -0.5})
    assert strikes_equal(left, right)
    assert not strikes_equal(left, different)


def test_export_ts_from_uw_time_buckets_to_interval():
    ts = export_ts_from_uw_time("2026-06-15T14:37:12+00:00", interval_minutes=10)
    assert ts.endswith("_143000")


def test_derive_snapshot_features_includes_walls():
    strike = pd.Series({6000.0: 1.5, 6050.0: -2.0})
    cumulative = strike.cumsum()
    summary = {"spot": 6025.0, "total_gex_bn_per_pct": -0.5, "net_gamma_regime": "SHORT gamma"}
    features = derive_snapshot_features(
        ticker="SPX",
        ts="2026-06-15_143000",
        gex_by_strike=strike,
        cumulative_gex=cumulative,
        gex_by_expiration=pd.Series(dtype=float),
        summary=summary,
        prior=None,
    )
    assert features["call_wall"] == 6000.0
    assert features["put_wall"] == 6050.0
    assert features["strike_count"] == 2
    assert isinstance(features["surface_vector"], list)


def test_validate_snapshot_rejects_empty_profile():
    result = validate_snapshot(
        ticker="SPX",
        ts="2026-06-15_143000",
        gex_by_strike=pd.Series(dtype=float),
        summary={"spot": 0.0, "total_gex_bn_per_pct": 0.0},
        prior=None,
    )
    assert not result.ok
    assert result.status == "rejected"


def test_prepare_snapshot_skips_duplicate(monkeypatch):
    monkeypatch.setenv("GEX_SKIP_DUPLICATE_SNAPSHOTS", "1")
    strike = pd.Series({6000.0: 1.5, 6050.0: -2.0})
    cumulative = strike.cumsum()

    def _fake_prior(ticker: str):
        return {
            "ts": "2026-06-15_142000",
            "spot": 6025.0,
            "total_gex": -0.5,
            "regime": "SHORT gamma",
            "strike_count": 2,
            "strike": strike.copy(),
        }

    monkeypatch.setattr("gex_core.snapshot_processing.fetch_prior_snapshot", _fake_prior)
    prepared = prepare_snapshot_for_storage(
        "SPX",
        gex_by_strike=strike,
        cumulative_gex=cumulative,
        gex_by_expiration=pd.Series(dtype=float),
        summary={"spot": 6025.0, "total_gex_bn_per_pct": -0.5, "net_gamma_regime": "SHORT gamma"},
        timestamp="2026-06-15_143000",
    )
    assert prepared.skipped_duplicate
    assert prepared.summary.get("call_wall") == 6000.0
