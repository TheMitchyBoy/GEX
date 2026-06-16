"""Tests for UW spot consensus."""

from gex_core.spot_consensus import build_spot_consensus


def test_build_spot_consensus_uses_chosen_spot():
    result = build_spot_consensus(
        stock_state=6000.0,
        spot_exposure_price=6001.0,
        intraday_price=6000.5,
        chosen=6000.5,
    )
    assert result["spot"] == 6000.5
    assert result["spot_disagreement"] is False


def test_build_spot_consensus_flags_disagreement(monkeypatch):
    monkeypatch.setenv("GEX_SPOT_DISAGREEMENT_TOLERANCE_PCT", "0.001")
    result = build_spot_consensus(
        stock_state=6000.0,
        spot_exposure_price=6100.0,
        chosen=6000.0,
    )
    assert result["spot_disagreement"] is True
    assert result["spot_disagreement_pct"] > 0.01
