from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from gex_core.uw_loader import (
    _get,
    normalize_net_exposure,
    fetch_uw_best_spot_price,
    fetch_uw_gex,
    fetch_uw_greek_exposure,
)


def _resp(status: int, json_payload=None, headers=None):
    r = MagicMock(spec=requests.Response)
    r.status_code = status
    r.headers = headers or {}
    r.json.return_value = json_payload if json_payload is not None else {"data": []}

    def _raise():
        if status >= 400:
            raise requests.HTTPError(f"{status} error")

    r.raise_for_status.side_effect = _raise
    return r


@patch("gex_core.uw_loader.fetch_uw_best_spot_price", return_value=5000.0)
@patch("gex_core.uw_loader.fetch_uw_greek_exposure_by_expiration", return_value=pd.Series(dtype=float))
@patch(
    "gex_core.uw_loader.fetch_uw_spot_exposures",
    return_value=pd.DataFrame({"price": [5000.0], "strike": [5000.0]}),
)
@patch("gex_core.uw_loader.fetch_uw_greek_exposure")
def test_fetch_uw_gex_aggregates(mock_greek, mock_spot_df, mock_exp, mock_best_spot):
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
    result = normalize_net_exposure(
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
    result = normalize_net_exposure(
        frame,
        call_col="call_gamma_oi",
        put_col="put_gamma_oi",
    )
    assert list(result) == [4.0, 5.0, 3.0]


@patch("gex_core.uw_loader.fetch_uw_spot", return_value=7355.0)
@patch("gex_core.uw_loader.fetch_uw_spot_exposures", return_value=pd.DataFrame({"price": [7355.0]}))
@patch(
    "gex_core.uw_loader.fetch_uw_spot_exposures_intraday",
    return_value=pd.DataFrame({"price": [7383.85]}),
)
@patch("gex_core.uw_loader.fetch_uw_stock_state_price", return_value=7383.85)
def test_fetch_uw_best_spot_price_prefers_stock_state_live(
    _mock_state,
    _mock_intraday,
    _mock_spot_strike,
    _mock_spot,
):
    assert fetch_uw_best_spot_price("SPX", api_key="test-key") == 7383.85


@patch("gex_core.uw_loader.fetch_uw_spot", return_value=7355.0)
@patch("gex_core.uw_loader.fetch_uw_spot_exposures", return_value=pd.DataFrame({"price": [7355.0]}))
@patch(
    "gex_core.uw_loader.fetch_uw_spot_exposures_intraday",
    return_value=pd.DataFrame({"price": [7385.0]}),
)
@patch("gex_core.uw_loader.fetch_uw_stock_state_price", return_value=7383.85)
def test_fetch_uw_best_spot_price_uses_intraday_for_historical_date(
    _mock_state,
    _mock_intraday,
    _mock_spot_strike,
    _mock_spot,
):
    assert fetch_uw_best_spot_price("SPX", api_key="test-key", date="2026-06-05") == 7385.0


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


@patch("gex_core.uw_loader.time.sleep", return_value=None)
@patch("gex_core.uw_loader.requests.get")
def test_get_retries_on_429_then_succeeds(mock_get, _sleep):
    mock_get.side_effect = [
        _resp(429, headers={"Retry-After": "0"}),
        _resp(200, {"data": [{"strike": "1"}]}),
    ]
    data = _get("/api/x", api_key="k")
    assert data == [{"strike": "1"}]
    assert mock_get.call_count == 2


@patch("gex_core.uw_loader.time.sleep", return_value=None)
@patch("gex_core.uw_loader.requests.get")
def test_get_retries_on_timeout_then_succeeds(mock_get, _sleep):
    mock_get.side_effect = [
        requests.Timeout("slow"),
        _resp(200, {"data": [{"ok": 1}]}),
    ]
    data = _get("/api/x", api_key="k")
    assert data == [{"ok": 1}]
    assert mock_get.call_count == 2


@patch("gex_core.uw_loader.time.sleep", return_value=None)
@patch("gex_core.uw_loader.requests.get")
def test_get_does_not_retry_on_403(mock_get, _sleep):
    mock_get.return_value = _resp(403)
    with pytest.raises(requests.HTTPError):
        _get("/api/x", api_key="k")
    assert mock_get.call_count == 1


@patch("gex_core.uw_loader.time.sleep", return_value=None)
@patch("gex_core.uw_loader.requests.get")
def test_get_raises_after_exhausting_retries(mock_get, _sleep):
    mock_get.return_value = _resp(503)
    with pytest.raises(requests.HTTPError):
        _get("/api/x", api_key="k")
    # initial attempt + _MAX_RETRIES
    from gex_core.uw_loader import _MAX_RETRIES
    assert mock_get.call_count == _MAX_RETRIES + 1
