"""Low-GEX trader engine (dry run and paper execute)."""

from unittest.mock import patch

import pandas as pd

from gex_core.trading.low_gex_engine import run_low_gex_trade


def test_run_low_gex_trade_signal_only():
    exposure = pd.Series([-2.0, 0.5, 1.0], index=[7440.0, 7460.0, 7480.0])
    with patch("gex_core.trading.low_gex_engine.is_trader_session_active", return_value=True):
        result = run_low_gex_trade(
            ticker="SPX",
            spot=7460.0,
            exposure=exposure,
            execute=False,
        )
    assert result["action"] == "signal_only"
    assert result["signal"]["recommended"]["option_type"] == "put"


def test_run_low_gex_trade_paper_execute():
    exposure = pd.Series([-2.0, 0.5, 1.0], index=[7440.0, 7460.0, 7480.0])
    with (
        patch("gex_core.trading.low_gex_engine.is_trader_session_active", return_value=True),
        patch("gex_core.trading.low_gex_engine.list_open_trades", return_value=[]),
        patch("gex_core.trading.low_gex_engine.open_trade", return_value=42) as mock_open,
    ):
        result = run_low_gex_trade(
            ticker="SPX",
            spot=7460.0,
            exposure=exposure,
            execute=True,
        )
    assert result["action"] == "opened"
    assert result["trade_id"] == 42
    mock_open.assert_called_once()
