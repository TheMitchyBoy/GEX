"""Tests for UW AI predictor."""

from unittest.mock import patch

import pandas as pd

from gex_core.pipeline import GexAggregates
from gex_core.uw_ai_predictor import _parse_prediction_json, predict_from_uw_data


def _minimal_agg() -> GexAggregates:
    gex = pd.Series({5000.0: 2.0, 5050.0: -1.0, 5100.0: 3.0})
    greek = pd.DataFrame({"strike": gex.index, "net_gex": gex.values, "call_gex": gex.values, "put_gex": 0.0})
    gex.attrs["greek_exposure_df"] = greek
    return GexAggregates(
        gex_by_strike=gex,
        gex_by_expiration=pd.Series(dtype=float),
        cumulative_gex=gex.cumsum(),
        surface_data=pd.DataFrame(),
        total_gex_bn=float(gex.sum()),
    )


def test_parse_prediction_json_handles_fenced_json():
    raw = '```json\n{"predictions": ["test"], "confidence": 0.8, "spot_bias": "bullish"}\n```'
    parsed = _parse_prediction_json(raw)
    assert parsed is not None
    assert parsed["predictions"] == ["test"]
    assert parsed["confidence"] == 0.8
    assert parsed["spot_bias"] == "bullish"
    assert parsed["llm_enhanced"] is True


def test_predict_from_uw_data_rule_based_fallback():
    agg = _minimal_agg()
    with patch("gex_core.uw_ai_predictor._openai_predict", return_value=None), patch(
        "gex_core.uw_ai_predictor._hermes_predict", return_value=None
    ):
        result = predict_from_uw_data(
            ticker="SPX",
            spot=5025.0,
            gex_by_strike=agg.gex_by_strike,
            cumulative_gex=agg.cumulative_gex,
            total_gex_bn=agg.total_gex_bn,
            agg=agg,
            fetch_extras=False,
        )
    assert result["prediction_source"] == "rule_based"
    assert result["llm_enhanced"] is False
    assert result["predictions"]
    assert result["context_summary"]["strike_rows"] >= 0


def test_predict_from_uw_data_uses_openai_when_available():
    agg = _minimal_agg()
    llm_out = {
        "predicted_regime": "LONG gamma",
        "predicted_delta_gex_bn": 0.5,
        "predicted_total_gex_bn": 4.5,
        "spot_bias": "bullish",
        "confidence": 0.75,
        "predictions": ["Expect range-bound action near 5050 pin."],
        "reasoning": "Dealers long gamma above flip.",
    }
    with patch("gex_core.uw_ai_predictor._openai_predict", return_value=llm_out):
        result = predict_from_uw_data(
            ticker="SPX",
            spot=5025.0,
            gex_by_strike=agg.gex_by_strike,
            cumulative_gex=agg.cumulative_gex,
            total_gex_bn=agg.total_gex_bn,
            agg=agg,
            fetch_extras=False,
        )
    assert result["llm_enhanced"] is True
    assert result["predicted_regime"] == "LONG gamma"
    assert "5050" in result["predictions"][0] or result["predictions"]
