from pathlib import Path

from gex_core.history import list_tickers
from gex_core.intelligence import build_gamma_analysis_panel
from gex_core.refresh import refresh_tickers
from gex_core.tickers import PRIMARY_TICKER, is_supported_ticker, supported_tickers


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("strike,gex\n5000,1.0\n", encoding="utf-8")


def test_supported_tickers_are_spx_only():
    assert PRIMARY_TICKER == "SPX"
    assert supported_tickers() == ["SPX"]
    assert is_supported_ticker("spx")
    assert not is_supported_ticker("SPY")


def test_list_tickers_filters_non_spx_exports(tmp_path):
    _touch(tmp_path / "SPX_gex_by_strike_2026-06-04_032228.csv")
    _touch(tmp_path / "SPY_gex_by_strike_2026-06-04_032229.csv")
    _touch(tmp_path / "NVDA_gex_by_strike_2026-06-04_032229.csv")

    assert list_tickers(tmp_path) == ["SPX"]


def test_refresh_tickers_falls_back_to_spx(monkeypatch):
    calls = []

    def fake_refresh_ticker(ticker, force=False):
        calls.append((ticker, force))
        return True

    monkeypatch.setattr("gex_core.refresh.refresh_ticker", fake_refresh_ticker)

    assert refresh_tickers(["SPY", "NVDA"], force=True) == {"SPX": True}
    assert calls == [("SPX", True)]


def test_gamma_analysis_panel_computes_key_spx_distances():
    panel = build_gamma_analysis_panel(
        {
            "regime": "SHORT gamma",
            "spot": 5000,
            "total_gex": -4.0,
            "pos_gex": 6.0,
            "neg_gex": -10.0,
            "gamma_flip": 4980,
            "call_wall": 5050,
            "put_wall": 4950,
            "near_term_ratio": 0.5,
            "gex_concentration": 0.25,
            "cum_slope_at_spot": -0.0123,
        }
    )

    assert panel["flip"]["distance_pts"] == -20
    assert panel["spot_minus_flip_pts"] == 20
    assert panel["nearest_wall_name"] == "Call wall"
    assert panel["nearest_wall"]["abs_distance_pts"] == 50
    assert panel["net_ratio"] == -0.25
    assert panel["risk_label"] in {"moderate", "high"}
