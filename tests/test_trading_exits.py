from gex_core.trading.exits import ExitProfile, ExitState, build_exit_profile, evaluate_exit, spot_progress_toward_strike


def test_spot_progress_toward_call_strike():
    progress = spot_progress_toward_strike(
        entry_spot=5000.0,
        current_spot=5025.0,
        strike=5050.0,
        option_type="call",
    )
    assert progress == 0.5


def test_take_profit_at_thirty_percent():
    profile = ExitProfile(full_take_profit=0.30)
    state = ExitState()
    reason, pnl = evaluate_exit(
        0.31,
        state=state,
        bars_held=3,
        entry_spot=5000.0,
        strike=5010.0,
        current_spot=5020.0,
        option_type="call",
        profile=profile,
        entry_positive_gamma_strike=5010.0,
        current_positive_gamma_strike=5010.0,
    )
    assert reason == "take_profit"
    assert pnl == 0.30


def test_holds_below_take_profit_when_gamma_strike_unchanged():
    profile = ExitProfile(full_take_profit=0.30)
    state = ExitState()
    reason, _ = evaluate_exit(
        0.20,
        state=state,
        bars_held=15,
        entry_spot=5000.0,
        strike=5010.0,
        current_spot=5020.0,
        option_type="call",
        profile=profile,
        entry_positive_gamma_strike=5010.0,
        current_positive_gamma_strike=5010.0,
    )
    assert reason is None


def test_no_stop_loss_on_large_drawdown():
    profile = ExitProfile(full_take_profit=0.30)
    state = ExitState()
    reason, _ = evaluate_exit(
        -0.25,
        state=state,
        bars_held=3,
        entry_spot=5000.0,
        strike=5010.0,
        current_spot=4975.0,
        option_type="call",
        profile=profile,
        entry_positive_gamma_strike=5010.0,
        current_positive_gamma_strike=5010.0,
    )
    assert reason is None


def test_gamma_strike_change_triggers_exit():
    profile = ExitProfile(full_take_profit=0.30)
    state = ExitState()
    reason, pnl = evaluate_exit(
        0.05,
        state=state,
        bars_held=3,
        entry_spot=5000.0,
        strike=5010.0,
        current_spot=5020.0,
        option_type="call",
        profile=profile,
        entry_positive_gamma_strike=5010.0,
        current_positive_gamma_strike=5020.0,
    )
    assert reason == "gamma_strike_change"
    assert pnl == 0.05


def test_build_exit_profile_uses_take_profit_default(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_TAKE_PROFIT_PCT", "0.30")
    profile = build_exit_profile(
        ai_confidence=0.85,
        gamma_delta=0.12,
        regime="LONG gamma",
        entry_spot=5000.0,
        strike=5010.0,
    )
    assert profile.full_take_profit == 0.30
