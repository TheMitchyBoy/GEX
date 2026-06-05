import pandas as pd

from gex_core.predict import predict_next_snapshot, select_recent_history


def _snapshot(ts: str, total_gex: float, spot: float = 5000.0) -> dict:
    strikes = pd.Series(
        [total_gex * 0.25, total_gex * 0.75],
        index=[spot - 50, spot + 50],
        dtype=float,
    )
    return {
        "ts": ts,
        "ts_label": ts,
        "ticker": "SPX",
        "strike": strikes,
        "cumulative": strikes.cumsum(),
        "total_gex": float(total_gex),
        "pos_gex": float(strikes[strikes > 0].sum()),
        "neg_gex": float(strikes[strikes < 0].sum()),
        "gex_std": float(strikes.std()),
        "near_term_ratio": 0.0,
        "surface_peak": 0.0,
        "call_wall": float(strikes.idxmax()),
        "put_wall": float(strikes.idxmin()),
        "gamma_flip": None,
        "regime": "LONG gamma",
        "abs_mean": float(strikes.abs().mean()),
        "spot": spot,
    }


def test_select_recent_history_uses_latest_snapshot_window():
    history = [
        _snapshot("2026-05-28_000000", 10.0),
        _snapshot("2026-05-29_000000", 11.0),
        _snapshot("2026-06-02_000000", 12.0),
        _snapshot("2026-06-05_000000", 13.0),
    ]

    recent = select_recent_history(history, lookback_days=7)

    assert [row["ts"] for row in recent] == [
        "2026-05-29_000000",
        "2026-06-02_000000",
        "2026-06-05_000000",
    ]


def test_predict_next_snapshot_reports_recent_training_window():
    history = [
        _snapshot("2026-05-28_000000", 10.0),
        _snapshot("2026-05-29_000000", 11.0),
        _snapshot("2026-06-02_000000", 12.0),
        _snapshot("2026-06-03_000000", 13.0),
        _snapshot("2026-06-04_000000", 14.0),
        _snapshot("2026-06-05_000000", 15.0),
    ]

    pred = predict_next_snapshot(history, lookback_days=7)

    assert pred is not None
    assert pred["training_snapshot_count"] == 5
    assert pred["training_window_days"] == 7
    assert pred["predicted_total_gex"] == history[-1]["total_gex"] + pred["predicted_delta_gex"]
    assert pred["forecast_horizon"] == "next_snapshot"
    assert 0.0 <= pred["confidence"] <= 1.0
    assert 0.0 <= pred["raw_confidence"] <= 1.0
    assert pred["confidence_breakdown"]["training_rows"] >= 3
    assert "term_structure" in pred
