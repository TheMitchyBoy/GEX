import pandas as pd

from gex_core.spot_exposure import spot_exposure_mm_positions, spot_exposure_net_series


def test_spot_exposure_net_series_gamma_from_oi_columns():
    df = pd.DataFrame(
        {
            "strike": [6000.0, 6100.0],
            "call_gamma_oi": [2.0e9, 1.0e9],
            "put_gamma_oi": [-1.5e9, -0.5e9],
        }
    )
    series = spot_exposure_net_series(df, "gamma")
    assert len(series) == 2
    assert float(series.loc[6000.0]) == 0.5
    assert float(series.loc[6100.0]) == 0.5


def test_spot_exposure_net_series_vanna():
    df = pd.DataFrame(
        {
            "strike": [6000.0],
            "call_vanna_oi": [3.0e9],
            "put_vanna_oi": [-1.0e9],
        }
    )
    series = spot_exposure_net_series(df, "vanna")
    assert float(series.iloc[0]) == 2.0


def test_spot_exposure_mm_positions_sums_oi_columns():
    df = pd.DataFrame(
        {
            "call_delta_oi": [1.0e9, 2.0e9],
            "put_delta_oi": [-0.5e9, -0.5e9],
            "call_gamma_oi": [3.0e9, 1.0e9],
            "put_gamma_oi": [-1.0e9, -2.0e9],
        }
    )
    positions = spot_exposure_mm_positions(df)
    assert positions["net_call_delta_bn"] == 3.0
    assert positions["net_put_delta_bn"] == -1.0
    assert positions["net_call_gex_bn"] == 4.0
    assert positions["net_put_gex_bn"] == -3.0
