import pandas as pd

from gex_core.ai_analyst import _concentration_signal


def test_concentration_signal_ignores_series_attrs_with_dataframes():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=[100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    series.attrs["greek_exposure_df"] = pd.DataFrame({"strike": [100.0], "net_gex": [1.0]})
    series.attrs["spot_exposures_df"] = pd.DataFrame({"price": [100.0]})

    signal = _concentration_signal(series)

    assert signal.label == "Concentration"
    assert signal.value != "N/A"
