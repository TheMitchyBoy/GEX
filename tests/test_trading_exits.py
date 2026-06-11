from gex_core.trading.config import stop_loss_pct, take_profit_pct
from gex_core.trading.exits import ExitProfile, ExitState, build_exit_profile, evaluate_exit, spot_progress_toward_strike


def test_gamma_magnet_default_sl_tp(monkeypatch):
    monkeypatch.delenv("GEX_TRADER_STOP_LOSS_PCT", raising=False)
    monkeypatch.delenv("GEX_TRADER_TAKE_PROFIT_PCT", raising=False)
    assert stop_loss_pct() == 0.06
    assert take_profit_pct() == 0.28


def test_spot_progress_toward_call_strike():
    progress = spot_progress_toward_strike(
        entry_spot=5000.0,
        current_spot=5025.0,
        strike=5050.0,
        option_type="call",
    )
    assert progress == 0.5


def test_strong_setup_holds_for_full_target():
    profile = build_exit_profile(
        ai_confidence=0.85,
        gamma_delta=0.12,
        regime="LONG gamma",
        entry_spot=5000.0,
        strike=5010.0,
    )
    assert profile.hold_for_target
    assert profile.partial_take_profit is None

    state = ExitState()
    reason, pnl = evaluate_exit(
        0.20,
        state=state,
        bars_held=3,
        entry_spot=5000.0,
        strike=5010.0,
        current_spot=5020.0,
        option_type="call",
        profile=profile,
        magnet_strike=5050.0,
        magnet_primary=False,
    )
    assert reason is None

    tp = take_profit_pct()
    reason, pnl = evaluate_exit(
        tp + 0.01,
        state=state,
        bars_held=8,
        entry_spot=5000.0,
        strike=5010.0,
        current_spot=5035.0,
        option_type="call",
        profile=profile,
        magnet_strike=5050.0,
        magnet_primary=False,
    )
    assert reason == "take_profit"
    assert pnl == tp


def test_max_hold_exits_at_bar_limit(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_MAX_HOLD_MINUTES", "30")
    monkeypatch.setenv("GEX_TRADER_BAR_MINUTES", "2")
    profile = ExitProfile(hold_for_target=True, full_take_profit=0.60, time_stop_bars=15)
    state = ExitState()
    reason, pnl = evaluate_exit(
        0.12,
        state=state,
        bars_held=15,
        entry_spot=5000.0,
        strike=5010.0,
        current_spot=5020.0,
        option_type="call",
        profile=profile,
    )
    assert reason == "max_hold"
    assert pnl == 0.12


def test_magnet_touch_requires_min_pnl(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_MAGNET_TOUCH_EXIT", "1")
    monkeypatch.setenv("GEX_TRADER_MAGNET_TOUCH_MIN_PNL_PCT", "0.08")
    profile = ExitProfile(hold_for_target=True, full_take_profit=0.35)
    state = ExitState()
    reason, _ = evaluate_exit(
        0.03,
        state=state,
        bars_held=3,
        entry_spot=5000.0,
        strike=5010.0,
        current_spot=5050.0,
        option_type="call",
        profile=profile,
        magnet_strike=5050.0,
        magnet_primary=True,
    )
    assert reason is None


def test_time_stop_skipped_when_moving_toward_magnet():
    profile = ExitProfile(time_stop_bars=10)
    state = ExitState()
    reason, _ = evaluate_exit(
        0.01,
        state=state,
        bars_held=6,
        entry_spot=5000.0,
        strike=5050.0,
        current_spot=5025.0,
        option_type="call",
        profile=profile,
    )
    assert reason is None
