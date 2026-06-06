import pandas as pd

from gex_core.periscope import build_periscope_context
from gex_core.market_exposure_agent import analyze_market_exposure
from gex_core.charts import make_periscope_exposure_chart, make_mm_positions_chart


def test_build_periscope_context_from_history():
    strikes = pd.Series({5000: 1.0, 5050: -0.5, 5100: 0.3})
    history = [
        {
            "ts": "2026-06-05_021908",
            "ts_label": "2026-06-05 02:19",
            "spot": 5025.0,
            "total_gex": 0.8,
            "regime": "LONG gamma",
            "gamma_flip": 5000.0,
            "call_wall": 5100.0,
            "put_wall": 4950.0,
            "strike": strikes,
        }
    ]
    ctx = build_periscope_context(ticker="SPX", history=history)
    assert ctx["spot"] == 5025.0
    assert ctx["regime"] == "LONG gamma"
    assert not ctx["exposure_series"].empty


def test_market_exposure_agent_returns_who_what():
    strikes = pd.Series({5000: 2.0, 5050: -1.0, 5100: 3.0})
    result = analyze_market_exposure(
        ticker="SPX",
        spot=5025.0,
        gex_by_strike=strikes,
        total_gex_bn=4.0,
        gamma_flip=5000.0,
    )
    assert result["who"]
    assert result["whom"]
    assert result["what"]
    assert result["narrative"]


def test_periscope_charts_render_json():
    strikes = pd.Series({5000: 1.0, 5050: -0.5})
    chart = make_periscope_exposure_chart(strikes, spot=5025.0, exposure_type="gamma")
    assert chart is not None
    assert "data" in chart
    pos = make_mm_positions_chart(
        {"net_call_delta_bn": 1.0, "net_put_delta_bn": -0.5, "net_call_gex_bn": 2.0, "net_put_gex_bn": -1.0}
    )
    assert pos is not None
