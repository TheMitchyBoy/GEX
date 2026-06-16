"""Tests for gamma wall detection."""

import pandas as pd

from gex_core.wall_detection import detect_walls


def test_detect_walls_ignores_dust():
    strike = pd.Series(
        {
            5900.0: 1e-12,
            6000.0: 2.0,
            6050.0: -3.0,
            6100.0: 1e-12,
        }
    )
    walls = detect_walls(strike, spot=6025.0)
    assert walls["call_wall"] == 6000.0
    assert walls["put_wall"] == 6050.0


def test_detect_walls_empty_profile():
    walls = detect_walls(pd.Series(dtype=float), spot=6000.0)
    assert walls["call_wall"] is None
    assert walls["put_wall"] is None
