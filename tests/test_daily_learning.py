import json
from datetime import datetime, timedelta

import pandas as pd

from gex_core.market_time import MARKET_TZ

from gex_core.daily_learning import (
    attach_learning_to_bundle,
    daily_learning_enabled,
    format_strategy_brief,
    generate_lesson_for_date,
    get_insight,
    get_or_create_today_strategy,
    get_today_strategy_for_context,
    list_recent_lessons,
)
from gex_core.uw_context_bundle import build_context_bundle_from_snapshot


def test_daily_learning_disabled_by_default(monkeypatch):
    monkeypatch.delenv("GEX_DAILY_LEARNING", raising=False)
    assert daily_learning_enabled() is False


def test_attach_learning_to_bundle_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("GEX_DAILY_LEARNING", raising=False)
    bundle = {"summary": {"regime": "LONG gamma"}}
    result = attach_learning_to_bundle(bundle, "SPX")
    assert result is bundle
    assert "daily_learning" not in bundle


def test_rule_based_strategy_from_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("GEX_TRADING_DB", str(tmp_path / "journal.db"))
    snapshot = {
        "ts": "2026-06-05_120000",
        "market_date": "2026-06-05",
        "spot": 6000.0,
        "total_gex": 2.5,
        "regime": "LONG gamma",
        "gamma_flip": 5980.0,
        "call_wall": 6050.0,
        "put_wall": 5920.0,
        "strike": pd.Series({6000: 1.0, 6050: 2.0, 5920: -1.0}),
        "cumulative": pd.Series({6000: 1.0, 6050: 3.0, 5920: 2.0}),
    }
    bundle = build_context_bundle_from_snapshot(
        ticker="SPX",
        snapshot=snapshot,
        fetch_extras=False,
    )
    strategy = get_or_create_today_strategy(ticker="SPX", uw_bundle=bundle, force_refresh=True)
    assert strategy["bias"] == "mean_reversion"
    assert strategy["plays"]
    assert "LONG gamma" in strategy["summary"]


def test_lesson_persisted_for_export_day(tmp_path, monkeypatch):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    db = tmp_path / "journal.db"
    index_db = tmp_path / "index.db"
    monkeypatch.setenv("GEX_TRADING_DB", str(db))
    monkeypatch.setenv("GEX_INDEX_DB", str(index_db))
    monkeypatch.setattr("gex_core.exports.EXPORT_DIR", export_dir)
    monkeypatch.setattr("gex_core.history.EXPORT_DIR", export_dir)

    lesson_day = (datetime.now(MARKET_TZ).date() - timedelta(days=2)).isoformat()
    today = (datetime.now(MARKET_TZ).date() - timedelta(days=1)).isoformat()
    ts = f"{lesson_day}_120000"
    (export_dir / f"SPX_gex_by_strike_{ts}.csv").write_text(
        "strike,gex_bn_per_pct\n6000,1.0\n", encoding="utf-8"
    )
    (export_dir / f"SPX_cumulative_gex_{ts}.csv").write_text(
        "strike,cumulative_gex_bn_per_pct\n6000,1.0\n", encoding="utf-8"
    )
    summary = {
        "spot": 6000,
        "total_gex_bn_per_pct": 1.0,
        "net_gamma_regime": "LONG gamma",
        "market_date": lesson_day,
    }
    (export_dir / f"SPX_summary_{ts}.json").write_text(json.dumps(summary), encoding="utf-8")

    from gex_core.storage import sync_ticker_exports

    sync_ticker_exports("SPX", export_dir, force=True)

    with monkeypatch.context() as m:
        m.setattr("gex_core.daily_learning.market_today", lambda: today)
        lesson = generate_lesson_for_date("SPX", lesson_day, use_llm=False)

    assert lesson is not None
    assert lesson["market_date"] == lesson_day
    assert get_insight("SPX", lesson_day, "lesson") is not None
    assert list_recent_lessons("SPX")


def test_format_strategy_brief():
    text = format_strategy_brief({"bias": "momentum", "confidence": 0.6, "summary": "Trade breakouts."})
    assert "Today's plan" in text
    assert "momentum" in text
    assert "Trade breakouts" in text


def test_get_today_strategy_for_context_is_rule_based_without_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GEX_TRADING_DB", str(tmp_path / "journal.db"))
    snapshot = {
        "ts": "2026-06-05_120000",
        "market_date": "2026-06-05",
        "spot": 6000.0,
        "total_gex": 2.5,
        "regime": "LONG gamma",
        "gamma_flip": 5980.0,
        "call_wall": 6050.0,
        "put_wall": 5920.0,
        "strike": pd.Series({6000: 1.0, 6050: 2.0, 5920: -1.0}),
        "cumulative": pd.Series({6000: 1.0, 6050: 3.0, 5920: 2.0}),
    }
    bundle = build_context_bundle_from_snapshot(ticker="SPX", snapshot=snapshot, fetch_extras=False)
    strategy = get_today_strategy_for_context(ticker="SPX", uw_bundle=bundle)
    assert strategy["source"] == "rule_based"
    assert strategy["plays"]


def test_attach_learning_to_bundle_skips_llm_generation(tmp_path, monkeypatch):
    monkeypatch.setenv("GEX_DAILY_LEARNING", "1")
    monkeypatch.setenv("GEX_TRADING_DB", str(tmp_path / "journal.db"))
    calls = {"n": 0}

    def _track(**_kwargs):
        calls["n"] += 1
        return {"summary": "cached"}

    monkeypatch.setattr("gex_core.daily_learning.get_or_create_today_strategy", _track)
    monkeypatch.setattr("gex_core.daily_learning.market_today", lambda: "2099-01-01")
    bundle = {"summary": {"regime": "LONG gamma", "total_gex_bn": 1.0}}
    attach_learning_to_bundle(bundle, "SPX")
    assert calls["n"] == 0
    assert bundle["daily_learning"]["today_strategy"]["source"] == "rule_based"
