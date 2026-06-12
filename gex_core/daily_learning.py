"""
Daily learning loop for the GEX AI agent.

Persists one *lesson* per completed trading day (what worked / failed from UW
data + paper trades) and a fresh *strategy* for the current session. Both are
injected into LLM context so the agent improves incrementally instead of
starting from scratch each morning.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from gex_core.market_time import MARKET_TZ, market_today
from gex_core.uw_context_bundle import bundle_to_prompt_json

logger = logging.getLogger(__name__)

_INSIGHT_SCHEMA = """
        CREATE TABLE IF NOT EXISTS daily_insights (
            ticker TEXT NOT NULL,
            market_date TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (ticker, market_date, kind)
        );
        CREATE INDEX IF NOT EXISTS idx_daily_insights_ticker_date
            ON daily_insights (ticker, market_date DESC);
        """


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect():
    from gex_core.trading.journal import db_path
    from gex_core.sqlite_util import connect_sqlite

    return connect_sqlite(db_path(), schema_sql=_INSIGHT_SCHEMA)


def _save_insight(ticker: str, market_date: str, kind: str, payload: dict[str, Any]) -> None:
    ticker = ticker.upper()
    market_date = market_date[:10]
    now = _now_iso()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO daily_insights (ticker, market_date, kind, payload_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, market_date, kind) DO UPDATE SET
                payload_json = excluded.payload_json,
                updated_at = excluded.updated_at
            """,
            (ticker, market_date, kind, json.dumps(payload), now, now),
        )
        conn.commit()


def get_insight(ticker: str, market_date: str, kind: str) -> dict[str, Any] | None:
    ticker = ticker.upper()
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT payload_json FROM daily_insights
            WHERE ticker = ? AND market_date = ? AND kind = ?
            """,
            (ticker, market_date[:10], kind),
        ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row["payload_json"])
    except json.JSONDecodeError:
        return None


def list_recent_lessons(ticker: str, *, days: int | None = None) -> list[dict[str, Any]]:
    ticker = ticker.upper()
    days = days if days is not None else int(os.environ.get("GEX_AI_LESSON_LOOKBACK_DAYS", "7"))
    cutoff = (datetime.now(MARKET_TZ).date() - timedelta(days=max(1, days))).isoformat()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT market_date, payload_json FROM daily_insights
            WHERE ticker = ? AND kind = 'lesson' AND market_date >= ?
            ORDER BY market_date DESC
            """,
            (ticker, cutoff),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            payload = json.loads(row["payload_json"])
            payload.setdefault("market_date", row["market_date"])
            out.append(payload)
        except json.JSONDecodeError:
            continue
    return out


def _trades_on_market_date(ticker: str, market_date: str) -> list[dict[str, Any]]:
    from gex_core.trading.journal import list_recent_trades

    market_date = market_date[:10]
    closed = [t for t in list_recent_trades(limit=200, ticker=ticker) if t.get("status") == "closed"]
    matched: list[dict[str, Any]] = []
    for trade in closed:
        exit_ts = trade.get("exit_ts") or trade.get("entry_ts")
        if not exit_ts:
            continue
        try:
            exit_day = datetime.fromisoformat(str(exit_ts).replace("Z", "+00:00")).astimezone(MARKET_TZ).date().isoformat()
        except ValueError:
            continue
        if exit_day == market_date:
            matched.append(trade)
    return matched


def _day_snapshot_stats(ticker: str, market_date: str) -> dict[str, Any]:
    from gex_core.history import load_snapshot_at_ts
    from gex_core.storage import list_indexed_timestamps_for_date

    timestamps = list_indexed_timestamps_for_date(ticker, market_date)
    if not timestamps:
        return {"snapshot_count": 0}

    first = load_snapshot_at_ts(ticker, timestamps[0])
    last = load_snapshot_at_ts(ticker, timestamps[-1])
    if not first or not last:
        return {"snapshot_count": len(timestamps)}

    spots = []
    regimes = []
    for ts in timestamps[:: max(1, len(timestamps) // 12)]:
        snap = load_snapshot_at_ts(ticker, ts)
        if not snap:
            continue
        spot = snap.get("spot")
        if spot:
            spots.append(float(spot))
        regime = snap.get("regime")
        if regime:
            regimes.append(str(regime))

    return {
        "snapshot_count": len(timestamps),
        "open_spot": float(first.get("spot") or 0),
        "close_spot": float(last.get("spot") or 0),
        "spot_low": min(spots) if spots else None,
        "spot_high": max(spots) if spots else None,
        "regime_open": first.get("regime"),
        "regime_close": last.get("regime"),
        "gamma_flip_close": last.get("gamma_flip"),
        "call_wall_close": last.get("call_wall"),
        "put_wall_close": last.get("put_wall"),
        "total_gex_close": last.get("total_gex"),
        "dominant_regime": max(set(regimes), key=regimes.count) if regimes else last.get("regime"),
    }


def _trade_stats(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {"count": 0, "win_rate": 0.0, "total_pnl_usd": 0.0, "by_signal": {}}
    wins = [t for t in trades if float(t.get("pnl_pct") or 0) > 0]
    by_signal: dict[str, dict[str, Any]] = {}
    for trade in trades:
        sig = trade.get("signal_type") or "unknown"
        bucket = by_signal.setdefault(sig, {"count": 0, "wins": 0, "pnl_usd": 0.0})
        bucket["count"] += 1
        bucket["pnl_usd"] += float(trade.get("pnl_usd") or 0)
        if float(trade.get("pnl_pct") or 0) > 0:
            bucket["wins"] += 1
    for bucket in by_signal.values():
        bucket["win_rate"] = bucket["wins"] / bucket["count"] if bucket["count"] else 0.0
    return {
        "count": len(trades),
        "win_rate": len(wins) / len(trades),
        "total_pnl_usd": sum(float(t.get("pnl_usd") or 0) for t in trades),
        "by_signal": by_signal,
        "exit_reasons": {
            reason: sum(1 for t in trades if t.get("exit_reason") == reason)
            for reason in sorted({t.get("exit_reason") for t in trades if t.get("exit_reason")})
        },
    }


def _rule_based_lesson(
    *,
    ticker: str,
    market_date: str,
    day_stats: dict[str, Any],
    trade_stats: dict[str, Any],
) -> dict[str, Any]:
    takeaways: list[str] = []
    lesson_parts: list[str] = []

    regime = day_stats.get("regime_close") or day_stats.get("dominant_regime") or "unknown"
    if day_stats.get("snapshot_count", 0) >= 2:
        open_spot = day_stats.get("open_spot") or 0
        close_spot = day_stats.get("close_spot") or 0
        if open_spot and close_spot:
            move_pct = (close_spot - open_spot) / open_spot * 100
            lesson_parts.append(f"Spot moved {move_pct:+.2f}% in a {regime} session.")
            if "LONG gamma" in str(regime) and abs(move_pct) < 0.35:
                takeaways.append("Tight range in long gamma — favor fade setups at walls over breakouts.")
            if "SHORT gamma" in str(regime) and abs(move_pct) > 0.6:
                takeaways.append("Trend day in short gamma — momentum toward magnets outperformed fades.")

    if trade_stats.get("count", 0) > 0:
        wr = trade_stats.get("win_rate", 0)
        lesson_parts.append(
            f"Paper trades: {trade_stats['count']} closed, {wr:.0%} win rate, "
            f"${trade_stats.get('total_pnl_usd', 0):+.0f} PnL."
        )
        best = max(
            trade_stats.get("by_signal", {}).items(),
            key=lambda kv: kv[1].get("pnl_usd", -999),
            default=(None, None),
        )
        if best[0] and best[1].get("count", 0) >= 1:
            takeaways.append(f"Best signal type today: {best[0]} ({best[1].get('pnl_usd', 0):+.0f} USD).")
        if wr < 0.4:
            takeaways.append("Low win rate — tighten entry filters and wait for stronger gamma acceleration.")

    if not lesson_parts:
        lesson_parts.append(f"Indexed {day_stats.get('snapshot_count', 0)} GEX snapshots for {market_date}.")

    return {
        "ticker": ticker.upper(),
        "market_date": market_date,
        "regime": regime,
        "day_stats": day_stats,
        "trade_stats": trade_stats,
        "lesson": " ".join(lesson_parts),
        "takeaways": takeaways or ["Review wall proximity before 0DTE entries."],
        "source": "rule_based",
    }


def _llm_json(prompt: str, system: str) -> dict[str, Any] | None:
    try:
        from gex_core.gex_chatbot import _openai_chat

        raw, _err = _openai_chat(system, [], prompt, json_mode=True, temperature=0.25)
        if not raw:
            from gex_core.market_exposure_agent import _hermes_analyze

            raw = _hermes_analyze(prompt, system_prompt=system)
        if not raw:
            return None
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as exc:
        logger.debug("Daily learning LLM failed: %s", exc)
        return None


def _llm_enhance_lesson(base: dict[str, Any]) -> dict[str, Any]:
    system = (
        "You distill one trading day into a concise lesson for a GEX auto-trader. "
        "Output ONLY JSON with keys: lesson (string, 1-2 sentences), takeaways (array of 2-4 strings)."
    )
    prompt = f"Day statistics and trades:\n{bundle_to_prompt_json(base)}"
    parsed = _llm_json(prompt, system)
    if not parsed:
        return base
    base = dict(base)
    if parsed.get("lesson"):
        base["lesson"] = str(parsed["lesson"])
    if parsed.get("takeaways"):
        base["takeaways"] = [str(t) for t in parsed["takeaways"][:6]]
    base["source"] = "openai"
    return base


def generate_lesson_for_date(ticker: str, market_date: str, *, use_llm: bool = True) -> dict[str, Any] | None:
    """Build and persist a lesson for one completed market date."""
    market_date = market_date[:10]
    if market_date >= market_today():
        return None

    existing = get_insight(ticker, market_date, "lesson")
    if existing:
        return existing

    day_stats = _day_snapshot_stats(ticker, market_date)
    if day_stats.get("snapshot_count", 0) < 1:
        return None

    trades = _trades_on_market_date(ticker, market_date)
    trade_stats = _trade_stats(trades)
    lesson = _rule_based_lesson(
        ticker=ticker,
        market_date=market_date,
        day_stats=day_stats,
        trade_stats=trade_stats,
    )
    if use_llm and os.environ.get("OPENAI_API_KEY", "").strip():
        lesson = _llm_enhance_lesson(lesson)
    _save_insight(ticker, market_date, "lesson", lesson)
    return lesson


def finalize_pending_lessons(ticker: str, *, max_days: int | None = None) -> list[dict[str, Any]]:
    """Backfill lessons for recent dates that have exports but no lesson row."""
    from gex_core.storage import list_indexed_dates

    ticker = ticker.upper()
    today = market_today()
    max_days = max_days if max_days is not None else int(os.environ.get("GEX_AI_LESSON_BACKFILL_DAYS", "5"))
    dates = [d for d in list_indexed_dates(ticker) if d < today][-max_days:]
    created: list[dict[str, Any]] = []
    for market_date in dates:
        if get_insight(ticker, market_date, "lesson"):
            continue
        lesson = generate_lesson_for_date(ticker, market_date)
        if lesson:
            created.append(lesson)
    return created


def _rule_based_strategy(
    *,
    ticker: str,
    market_date: str,
    uw_bundle: dict[str, Any] | None,
    recent_lessons: list[dict[str, Any]],
    trade_memory: dict[str, Any] | None,
) -> dict[str, Any]:
    summary = (uw_bundle or {}).get("summary") or {}
    extended = (uw_bundle or {}).get("extended_features") or {}
    regime = summary.get("regime") or (
        "LONG gamma" if float(summary.get("total_gex_bn") or 0) >= 0 else "SHORT gamma"
    )
    flip = summary.get("gamma_flip")
    call_wall = summary.get("call_wall")
    put_wall = summary.get("put_wall")
    spot = (uw_bundle or {}).get("spot") or summary.get("spot")

    bias = "neutral"
    if "LONG gamma" in str(regime):
        bias = "mean_reversion"
    elif "SHORT gamma" in str(regime):
        bias = "momentum"

    plays: list[dict[str, Any]] = []
    if call_wall and spot:
        plays.append(
            {
                "name": "fade_rally_to_call_wall",
                "direction": "short",
                "trigger": f"spot within 0.3% of call wall {call_wall:.0f}",
                "target": f"gamma flip {flip:.0f}" if flip else "prior session VWAP",
                "confidence": 0.58 if "LONG gamma" in str(regime) else 0.45,
            }
        )
    if put_wall and spot:
        plays.append(
            {
                "name": "fade_dip_to_put_wall",
                "direction": "long",
                "trigger": f"spot within 0.3% of put wall {put_wall:.0f}",
                "target": f"gamma flip {flip:.0f}" if flip else "session mid",
                "confidence": 0.58 if "LONG gamma" in str(regime) else 0.45,
            }
        )
    if "SHORT gamma" in str(regime) and flip and spot:
        plays.append(
            {
                "name": "momentum_through_flip",
                "direction": "long" if float(spot) > float(flip) else "short",
                "trigger": f"sustained hold {'above' if float(spot) > float(flip) else 'below'} flip {flip:.0f}",
                "target": f"call wall {call_wall:.0f}" if call_wall else "next magnet",
                "confidence": 0.52,
            }
        )

    risk_notes: list[str] = []
    if extended.get("is_fomc_week"):
        risk_notes.append("FOMC week — reduce size and widen invalidation.")
    if extended.get("is_nfp_day") or extended.get("is_cpi_day"):
        risk_notes.append("Macro event day — wait for post-release structure.")
    perf = (trade_memory or {}).get("performance") or {}
    for lesson_text in perf.get("lessons") or []:
        risk_notes.append(str(lesson_text))

    lesson_hint = recent_lessons[0].get("lesson") if recent_lessons else None
    summary_text = (
        f"{regime} session — bias {bias.replace('_', ' ')}. "
        f"Watch flip {flip}, walls {put_wall}/{call_wall}."
    )
    if lesson_hint:
        summary_text += f" Yesterday: {lesson_hint}"

    return {
        "ticker": ticker.upper(),
        "market_date": market_date,
        "regime": regime,
        "bias": bias,
        "confidence": 0.55 if plays else 0.4,
        "summary": summary_text,
        "plays": plays[:4],
        "levels": {
            "support": [put_wall] if put_wall else [],
            "resistance": [call_wall] if call_wall else [],
            "pin": flip,
        },
        "risk_notes": risk_notes[:6],
        "recent_lessons_applied": [l.get("market_date") for l in recent_lessons[:3]],
        "source": "rule_based",
    }


def _llm_enhance_strategy(base: dict[str, Any], uw_bundle: dict[str, Any] | None) -> dict[str, Any]:
    system = (
        "You are a GEX intraday strategist. Using the Unusual Whales bundle, trade memory, "
        "and recent daily lessons, output ONLY JSON with keys: "
        "summary (string), bias (mean_reversion|momentum|neutral), confidence (0-1), "
        "plays (array of {name, direction, trigger, target, confidence}), "
        "levels ({support[], resistance[], pin}), risk_notes (string array), "
        "key_insight (one new thing learned for today)."
    )
    prompt = (
        f"Draft strategy base:\n{bundle_to_prompt_json(base)}\n\n"
        f"Full market context:\n{bundle_to_prompt_json(uw_bundle or {})}"
    )
    parsed = _llm_json(prompt, system)
    if not parsed:
        return base
    merged = dict(base)
    for key in ("summary", "bias", "confidence", "plays", "levels", "risk_notes", "key_insight"):
        if parsed.get(key) is not None:
            merged[key] = parsed[key]
    merged["source"] = "openai"
    return merged


def _strategy_cache_ttl_sec() -> float:
    try:
        return max(300.0, float(os.environ.get("GEX_AI_STRATEGY_CACHE_SEC", "7200")))
    except (TypeError, ValueError):
        return 7200.0


def _strategy_is_fresh(payload: dict[str, Any]) -> bool:
    updated = payload.get("updated_at")
    if not updated:
        return False
    try:
        ts = datetime.fromisoformat(str(updated).replace("Z", "+00:00"))
        age = datetime.now(timezone.utc) - ts.astimezone(timezone.utc)
        return age.total_seconds() < _strategy_cache_ttl_sec()
    except ValueError:
        return False


def get_or_create_today_strategy(
    *,
    ticker: str,
    uw_bundle: dict[str, Any] | None = None,
    trade_memory: dict[str, Any] | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Return today's trading strategy, generating and persisting if needed."""
    ticker = ticker.upper()
    today = market_today()

    if not force_refresh:
        cached = get_insight(ticker, today, "strategy")
        if cached and _strategy_is_fresh(cached):
            return cached

    recent_lessons = list_recent_lessons(ticker)
    if trade_memory is None:
        try:
            from gex_core.trading.journal import get_trade_memory_for_ai

            trade_memory = get_trade_memory_for_ai(ticker)
        except Exception:
            trade_memory = {}

    strategy = _rule_based_strategy(
        ticker=ticker,
        market_date=today,
        uw_bundle=uw_bundle,
        recent_lessons=recent_lessons,
        trade_memory=trade_memory,
    )
    if uw_bundle and os.environ.get("OPENAI_API_KEY", "").strip():
        strategy = _llm_enhance_strategy(strategy, uw_bundle)

    strategy["updated_at"] = _now_iso()
    _save_insight(ticker, today, "strategy", strategy)
    return strategy


def run_daily_learning_cycle(
    ticker: str,
    *,
    uw_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Finalize yesterday's lesson and refresh today's strategy."""
    ticker = ticker.upper()
    lessons = finalize_pending_lessons(ticker)
    strategy = get_or_create_today_strategy(ticker=ticker, uw_bundle=uw_bundle)
    return {
        "lessons_created": len(lessons),
        "recent_lessons": list_recent_lessons(ticker),
        "today_strategy": strategy,
    }


def attach_learning_to_bundle(bundle: dict[str, Any], ticker: str) -> dict[str, Any]:
    """Mutate bundle with cached daily lessons + today's strategy for LLM context."""
    if not bundle:
        return bundle
    bundle["daily_learning"] = {
        "recent_lessons": list_recent_lessons(ticker),
        "today_strategy": get_or_create_today_strategy(ticker=ticker, uw_bundle=bundle),
    }
    return bundle


def format_strategy_brief(strategy: dict[str, Any]) -> str:
    """Human-readable one-liner for dashboard welcome text."""
    summary = strategy.get("summary") or "Review gamma regime and key walls before trading."
    bias = strategy.get("bias") or "neutral"
    conf = strategy.get("confidence")
    conf_str = f" ({float(conf):.0%} confidence)" if conf is not None else ""
    return f"**Today's plan** ({bias.replace('_', ' ')}{conf_str}): {summary}"
