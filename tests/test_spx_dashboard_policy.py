from pathlib import Path

from gex_core.history import list_tickers
from gex_core.intelligence import (
    build_gamma_analysis_panel,
    build_model_accountability_panel,
    build_term_structure_panel,
)
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

    def fake_refresh_ticker(ticker, force=False, market_date=None):
        calls.append((ticker, force, market_date))
        return True

    monkeypatch.setattr("gex_core.refresh.refresh_ticker", fake_refresh_ticker)

    assert refresh_tickers(["SPY", "NVDA"], force=True) == {"SPX": True}
    assert calls == [("SPX", True, None)]


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


def test_term_structure_panel_labels_zero_dte_heavy_setup():
    panel = build_term_structure_panel(
        {
            "zero_dte_gex_bn": 4.0,
            "zero_dte_ratio": 0.6,
            "near_term_gex_bn": 5.0,
            "near_term_ratio": 0.75,
            "back_term_gex_bn": 1.0,
            "back_term_ratio": 0.15,
            "term_curvature": 4.0,
            "expiration_count": 4,
        },
        {"predicted_zero_dte_ratio": 0.7, "predicted_near_term_ratio": 0.8, "predicted_term_curvature": 4.5},
    )

    assert panel["concentration_label"] == "0DTE-heavy"
    assert abs(panel["zero_dte_ratio_delta_forecast"] - 0.1) < 1e-9
    assert panel["term_curvature_delta_forecast"] == 0.5


def test_model_accountability_warns_on_thin_history():
    panel = build_model_accountability_panel(
        "SPX",
        {
            "training_snapshot_count": 5,
            "training_window_days": 7,
            "confidence": 0.4,
            "raw_confidence": 0.8,
            "confidence_breakdown": {"sample_factor": 0.1},
        },
        {"n": 2, "accuracy": 0.5, "confidence_accuracy_gap": 0.1},
    )

    assert panel["training_snapshot_count"] == 5
    assert any("thin" in warning for warning in panel["warnings"])
