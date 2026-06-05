from unittest.mock import patch

import pandas as pd

from gex_core.uw_loader import _normalize_net_exposure, fetch_uw_gex, fetch_uw_greek_exposure


@patch("gex_core.uw_loader.fetch_uw_spot", return_value=5000.0)
@patch("gex_core.uw_loader.fetch_uw_greek_exposure")
def test_fetch_uw_gex_aggregates(mock_greek, mock_spot):
    mock_greek.return_value = pd.DataFrame(
        {
            "strike": [4900.0, 5000.0],
            "call_gex": [1.0, 2.0],
            "put_gex": [-0.5, -1.0],
            "net_gex": [0.5, 1.0],
        }
    )
    spot, agg = fetch_uw_gex("SPX", api_key="test-key")
    assert spot == 5000.0
    assert abs(agg.total_gex_bn - 1.5) < 1e-6
    assert len(agg.gex_by_strike) == 2


def test_normalize_net_exposure_prefers_explicit_net_column():
    frame = pd.DataFrame(
        {
            "call_gex": [5.0, 7.0],
            "put_gex": [2.0, 3.0],  # unsigned puts
            "net_gex": [3.0, 4.0],
        }
    )
    result = _normalize_net_exposure(
        frame,
        call_col="call_gex",
        put_col="put_gex",
        net_col="net_gex",
    )
    assert list(result) == [3.0, 4.0]


def test_normalize_net_exposure_uses_signed_put_sum_when_puts_negative():
    frame = pd.DataFrame(
        {
            "call_gamma_oi": [10.0, 8.0, 4.0],
            "put_gamma_oi": [-6.0, -3.0, -1.0],
        }
    )
    result = _normalize_net_exposure(
        frame,
        call_col="call_gamma_oi",
        put_col="put_gamma_oi",
    )
    assert list(result) == [4.0, 5.0, 3.0]


@patch("gex_core.uw_loader._get")
def test_fetch_uw_greek_exposure_passes_historical_date(mock_get):
    mock_get.return_value = [
        {
            "date": "2026-06-03",
            "strike": "5000",
            "call_gex": "2000",
            "put_gex": "-1000",
        }
    ]

    df = fetch_uw_greek_exposure("SPX", api_key="test-key", date="2026-06-03")

    mock_get.assert_called_once_with(
        "/api/stock/SPX/greek-exposure/strike",
        api_key="test-key",
        date="2026-06-03",
    )
    assert df.attrs["market_date"] == "2026-06-03"
