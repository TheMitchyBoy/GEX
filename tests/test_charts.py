import json

import pandas as pd

from gex_core.charts import make_0dte_movement_chart, make_gex_profile_chart


def test_gex_profile_chart_uses_tight_spot_window_and_bar_width():
    strikes = pd.Series(
        [1.0, -0.5, 2.0, -1.0, 0.7],
        index=[4700, 4800, 4900, 5000, 5100],
        dtype=float,
    )

    payload = json.loads(make_gex_profile_chart(strikes, "SPX", spot=4900.0))

    bar = payload["data"][0]
    assert bar["width"] > 1
    assert payload["layout"]["xaxis"]["range"][0] >= 4700
    assert payload["layout"]["bargap"] == 0.0


def test_0dte_movement_chart_compares_same_day_snapshots():
    previous = {
        "ts": "2026-06-05_140000",
        "ts_label": "2026-06-05 14:00:00",
        "spot": 5000,
        "strike": pd.Series({4975.0: 1.0, 5000.0: -2.0, 5025.0: 0.5}),
    }
    current = {
        "ts": "2026-06-05_141000",
        "ts_label": "2026-06-05 14:10:00",
        "spot": 5005,
        "strike": pd.Series({4975.0: 1.5, 5000.0: -1.0, 5025.0: -0.5}),
    }

    payload = json.loads(make_0dte_movement_chart(current, previous, "SPX", spot=5005))

    bar = payload["data"][0]
    assert bar["name"] == "ΔGEX since prior same-day snapshot"
    assert bar["y"] == [0.5, 1.0, -1.0]
    assert bar["width"] > 1
