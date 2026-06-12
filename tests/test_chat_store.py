from gex_core.chat_store import delete_session, load_session, prune_old_sessions, save_session


def test_chat_session_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("GEX_TRADING_DB", str(tmp_path / "journal.db"))
    save_session(
        session_id="abc123",
        ticker="SPX",
        messages=[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
    )
    loaded = load_session("abc123")
    assert loaded is not None
    assert len(loaded["messages"]) == 2
    delete_session("abc123")
    assert load_session("abc123") is None
    assert prune_old_sessions() >= 0
