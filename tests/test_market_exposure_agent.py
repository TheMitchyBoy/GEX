import os
from unittest.mock import patch

import pandas as pd

from gex_core.market_exposure_agent import (
    _resolve_hermes_llm_config,
    analyze_market_exposure,
)


def test_resolve_hermes_llm_config_prefers_openai(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GEX_AGENT_MODEL", "gpt-4o-mini")
    cfg = _resolve_hermes_llm_config()
    assert cfg is not None
    provider, api_key, model, base_url = cfg
    assert provider == "openai"
    assert api_key == "sk-test"
    assert model == "gpt-4o-mini"
    assert "openai.com" in base_url


def test_analyze_market_exposure_uses_full_uw_bundle_when_agg_provided():
    strikes = pd.Series({5000: 2.0, 5050: -1.0, 5100: 3.0})
    greek = pd.DataFrame({"strike": strikes.index, "net_gex": strikes.values, "call_gex": strikes.values, "put_gex": 0.0})
    strikes.attrs["greek_exposure_df"] = greek
    from gex_core.pipeline import GexAggregates

    agg = GexAggregates(
        gex_by_strike=strikes,
        gex_by_expiration=pd.Series(dtype=float),
        cumulative_gex=strikes.cumsum(),
        surface_data=pd.DataFrame(),
        total_gex_bn=float(strikes.sum()),
    )
    with patch(
        "gex_core.market_exposure_agent._hermes_analyze",
        return_value="Hermes: full UW bundle analyzed.",
    ) as mock_hermes:
        result = analyze_market_exposure(
            ticker="SPX",
            spot=5025.0,
            gex_by_strike=strikes,
            total_gex_bn=4.0,
            gamma_flip=5000.0,
            agg=agg,
        )
    assert result["uw_data_fed"] is True
    assert result["context_summary"]["strike_rows"] >= 0
    mock_hermes.assert_called_once()
    prompt = mock_hermes.call_args[0][0]
    assert "greek_exposure_by_strike" in prompt or "Unusual Whales" in prompt


def test_analyze_market_exposure_uses_hermes_when_available():
    strikes = pd.Series({5000: 2.0, 5050: -1.0, 5100: 3.0})
    with patch(
        "gex_core.market_exposure_agent._hermes_analyze",
        return_value="Hermes: dealers are long gamma; fade extremes.",
    ):
        result = analyze_market_exposure(
            ticker="SPX",
            spot=5025.0,
            gex_by_strike=strikes,
            total_gex_bn=4.0,
            gamma_flip=5000.0,
        )
    assert result["hermes_enhanced"] is True
    assert "Hermes" in result["narrative"]
    assert result["agent_source"] == "hermes-agent + gex_core"
    assert result["who"]
    assert result["what"]


def test_analyze_market_exposure_rule_based_without_hermes():
    strikes = pd.Series({5000: 2.0, 5050: -1.0})
    with patch("gex_core.market_exposure_agent._hermes_analyze", return_value=None):
        result = analyze_market_exposure(
            ticker="SPX",
            spot=5025.0,
            gex_by_strike=strikes,
            total_gex_bn=-1.0,
            gamma_flip=5000.0,
        )
    assert result["hermes_enhanced"] is False
    assert result["agent_source"] == "gex_core"
    assert result["narrative"]
