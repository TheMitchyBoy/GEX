import pandas as pd

from gex_core.data_quality import clean_option_data, DataQualityConfig


def test_clean_option_data_filters_low_oi():
    df = pd.DataFrame(
        {
            "option": ["SPX260620C04800000", "SPX260620C04900000"],
            "gamma": [0.01, 0.02],
            "open_interest": [0, 100],
            "iv": [0.2, 0.2],
            "bid": [1.0, 1.0],
            "ask": [1.1, 1.1],
        }
    )
    cfg = DataQualityConfig(enabled=True, min_open_interest=10)
    cleaned, report = clean_option_data(df, spot=4800.0, config=cfg)
    assert len(cleaned) == 1
    assert report.rows_out == 1
