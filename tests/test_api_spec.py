"""api_spec.yaml must stay aligned with gex_core.uw_loader."""

from pathlib import Path

import pytest

from gex_core.api_spec import gex_scaling, implementation_rules, load_api_spec, uw_endpoint_paths
from gex_core.uw_loader import _BASE_URL, _CLIENT_ID, _GEX_SCALE, _RETRYABLE_STATUS


def test_api_spec_file_exists():
    path = Path(__file__).resolve().parent.parent / "api_spec.yaml"
    assert path.is_file()


def test_spec_lists_primary_uw_endpoints():
    paths = uw_endpoint_paths()
    assert "/api/stock/{ticker}/greek-exposure/strike" in paths
    assert "/api/stock/{ticker}/spot-exposures/strike" in paths
    assert "/api/stock/{ticker}/spot-exposures" in paths
    assert "/api/stock/{ticker}/stock-state" in paths


def test_greek_exposure_scaling_matches_loader():
    scaling = gex_scaling("greek_exposure")
    assert scaling["divisor"] == _GEX_SCALE
    assert scaling["api_unit"] == "millions_usd"


def test_spot_exposure_scaling_matches_modules():
    from gex_core.intraday_backfill import _RAW_GAMMA_SCALE
    from gex_core.spot_exposure import RAW_SCALE

    strike_scaling = gex_scaling("spot_exposure_strike")
    intraday_scaling = gex_scaling("spot_exposure_intraday")
    assert strike_scaling["divisor"] == RAW_SCALE == _RAW_GAMMA_SCALE
    assert intraday_scaling["divisor"] == _RAW_GAMMA_SCALE


def test_implementation_rules_match_loader():
    rules = implementation_rules()
    assert rules["loader_module"] == "gex_core.uw_loader"
    assert rules["client_id"] == _CLIENT_ID
    assert rules["chart_gamma_source"] == "spot-exposures/strike"
    assert rules["gamma_flip_source"] == "spot-exposures/strike"
    assert rules["gamma_flip_method"] == "magnet_profile_atm_window"
    assert set(rules["retryable_http_status"]) == set(_RETRYABLE_STATUS)


def test_spec_server_matches_loader_base_url():
    spec = load_api_spec()
    servers = spec.get("servers") or []
    assert servers
    assert servers[0]["url"] == _BASE_URL


def test_snapshot_gamma_flip_prefers_spot_strike():
    import pandas as pd

    from gex_core.features import enrich_snapshot_metrics

    spot_oi = pd.Series([-5.0, 2.0, 3.0], index=[7440.0, 7450.0, 7460.0])
    greek = pd.Series([-1.0, 2.5, 1.0], index=[7440.0, 7450.0, 7460.0])
    metrics = enrich_snapshot_metrics(
        {
            "strike": spot_oi,
            "greek_strike": greek,
            "cumulative": spot_oi.cumsum(),
            "spot": 7460.0,
            "call_wall": 7460.0,
            "put_wall": 7440.0,
        }
    )
    flip = metrics["gamma_flip"]
    assert flip is not None
    spot_flip = enrich_snapshot_metrics(
        {
            "strike": spot_oi,
            "cumulative": spot_oi.cumsum(),
            "spot": 7460.0,
            "call_wall": 7460.0,
            "put_wall": 7440.0,
        }
    )["gamma_flip"]
    assert float(flip) == float(spot_flip)
    greek_flip = enrich_snapshot_metrics(
        {
            "strike": pd.Series(dtype=float),
            "greek_strike": greek,
            "cumulative": greek.cumsum(),
            "spot": 7460.0,
            "call_wall": 7460.0,
            "put_wall": 7440.0,
        }
    )["gamma_flip"]
    assert float(flip) != float(greek_flip)
