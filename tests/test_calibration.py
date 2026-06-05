"""Tests for fitted forecast relationships replacing hard-coded scalars."""

from gex_core.calibration import (
    DEFAULT_MOVE_PER_DELTA_GEX,
    calibrate_confidence,
    fit_close_above_flip_rate,
    fit_move_per_delta_gex,
)


def _hist(spots, totals, flips=None):
    rows = []
    for i, (s, t) in enumerate(zip(spots, totals)):
        row = {"spot": s, "total_gex": t}
        if flips is not None:
            row["gamma_flip"] = flips[i]
        rows.append(row)
    return rows


def test_move_fit_falls_back_when_undersampled():
    fit = fit_move_per_delta_gex(_hist([5000, 5010], [1.0, 2.0]))
    assert fit["fitted"] is False
    assert fit["slope"] == DEFAULT_MOVE_PER_DELTA_GEX


def test_move_fit_recovers_known_slope():
    # Construct forward returns that are exactly 0.001 * delta_gex.
    spots = [5000.0]
    totals = [0.0]
    deltas = [5, -3, 8, -6, 4, -2, 7]
    for d in deltas:
        ret = 0.001 * d
        spots.append(spots[-1] * (1 + ret))
        totals.append(totals[-1] + d)
    fit = fit_move_per_delta_gex(_hist(spots, totals))
    assert fit["fitted"] is True
    assert abs(fit["slope"] - 0.001) < 1e-4


def test_close_above_flip_rate():
    rate = fit_close_above_flip_rate(
        _hist(
            [100, 100, 100, 100, 100, 100],
            [1, 1, 1, 1, 1, 1],
            flips=[90, 90, 110, 90, 110, 90],
        )
    )
    # 4 of 6 snapshots have spot >= flip.
    assert abs(rate - (4 / 6)) < 1e-9


def test_calibrate_confidence_shrinks_with_small_n():
    # With many samples and a strong hit-rate, calibrated confidence rises.
    high_n = calibrate_confidence(0.2, hit_rate=0.9, n=50)
    low_n = calibrate_confidence(0.2, hit_rate=0.9, n=1)
    assert high_n > low_n
    assert calibrate_confidence(0.5, hit_rate=None, n=0) == 0.5
