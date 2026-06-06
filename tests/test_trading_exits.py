from gex_core.trading.exits import ExitProfile, ExitState, build_exit_profile, evaluate_exit, spot_progress_toward_strike


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
    )
    assert reason is None

    reason, pnl = evaluate_exit(
        0.36,
        state=state,
        bars_held=8,
        entry_spot=5000.0,
        strike=5010.0,
        current_spot=5035.0,
        option_type="call",
        profile=profile,
    )
    assert reason == "take_profit"
    assert pnl == 0.35


def test_time_stop_skipped_when_moving_toward_magnet():
    profile = ExitProfile(time_stop_bars=6)
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
