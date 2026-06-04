from unittest.mock import patch

import pandas as pd

from gex_core.uw_loader import fetch_uw_gex


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
