"""SQLite trade journal and performance memory for the auto-trader."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = _REPO_ROOT / "data" / "trading_journal.db"


def db_path() -> Path:
    raw = Path(os.environ.get("GEX_TRADING_DB", str(DEFAULT_DB)))
    return _REPO_ROOT / raw if not raw.is_absolute() else raw


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            option_type TEXT NOT NULL,
            strike REAL NOT NULL,
            qty REAL NOT NULL DEFAULT 1,
            entry_ts TEXT NOT NULL,
            exit_ts TEXT,
            entry_spot REAL NOT NULL,
            exit_spot REAL,
            entry_premium REAL NOT NULL,
            exit_premium REAL,
            pnl_pct REAL,
            pnl_usd REAL,
            exit_reason TEXT,
            signal_type TEXT,
            signal_strike REAL,
            signal_gamma REAL,
            gamma_delta REAL,
            ai_confidence REAL,
            ai_reason TEXT,
            meta_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_trades_status ON trades (status, entry_ts DESC);
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            ticker TEXT NOT NULL,
            action TEXT NOT NULL,
            payload_json TEXT,
            ai_verdict TEXT,
            ai_notes TEXT
        );
        CREATE TABLE IF NOT EXISTS trader_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    return conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_trade(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    if d.get("meta_json"):
        try:
            d["meta"] = json.loads(d["meta_json"])
        except json.JSONDecodeError:
            d["meta"] = {}
    return d


def is_trader_armed() -> bool:
    with _connect() as conn:
        row = conn.execute("SELECT value FROM trader_state WHERE key = 'armed'").fetchone()
    return bool(row and row["value"] == "1")


def set_trader_armed(armed: bool) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO trader_state (key, value, updated_at) VALUES ('armed', ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            ("1" if armed else "0", _now_iso()),
        )
        conn.commit()


def record_decision(
    *,
    ticker: str,
    action: str,
    payload: dict[str, Any] | None = None,
    ai_verdict: str | None = None,
    ai_notes: str | None = None,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO decisions (ts, ticker, action, payload_json, ai_verdict, ai_notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (_now_iso(), ticker.upper(), action, json.dumps(payload or {}), ai_verdict, ai_notes),
        )
        conn.commit()


def patch_trade_meta(trade_id: int, meta: dict[str, Any]) -> None:
    with _connect() as conn:
        row = conn.execute("SELECT meta_json FROM trades WHERE id = ?", (trade_id,)).fetchone()
        current: dict[str, Any] = {}
        if row and row["meta_json"]:
            try:
                current = json.loads(row["meta_json"])
            except json.JSONDecodeError:
                current = {}
        current.update(meta)
        conn.execute("UPDATE trades SET meta_json = ? WHERE id = ?", (json.dumps(current), trade_id))
        conn.commit()


def open_trade(
    *,
    ticker: str,
    option_type: str,
    strike: float,
    entry_spot: float,
    entry_premium: float,
    signal_type: str,
    signal_strike: float,
    signal_gamma: float,
    gamma_delta: float,
    ai_confidence: float,
    ai_reason: str,
    meta: dict[str, Any] | None = None,
    qty: float = 1.0,
) -> int:
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO trades (
                ticker, status, option_type, strike, qty, entry_ts, entry_spot,
                entry_premium, signal_type, signal_strike, signal_gamma, gamma_delta,
                ai_confidence, ai_reason, meta_json
            ) VALUES (?, 'open', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker.upper(),
                option_type.lower(),
                float(strike),
                float(qty),
                _now_iso(),
                float(entry_spot),
                float(entry_premium),
                signal_type,
                float(signal_strike),
                float(signal_gamma),
                float(gamma_delta),
                float(ai_confidence),
                ai_reason[:2000],
                json.dumps(meta or {}),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def close_trade(
    trade_id: int,
    *,
    exit_spot: float,
    exit_premium: float,
    pnl_pct: float,
    pnl_usd: float,
    exit_reason: str,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE trades SET
                status = 'closed',
                exit_ts = ?,
                exit_spot = ?,
                exit_premium = ?,
                pnl_pct = ?,
                pnl_usd = ?,
                exit_reason = ?
            WHERE id = ? AND status = 'open'
            """,
            (_now_iso(), float(exit_spot), float(exit_premium), float(pnl_pct), float(pnl_usd), exit_reason, trade_id),
        )
        conn.commit()


def reduce_trade_qty(trade_id: int, new_qty: float) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE trades SET qty = ? WHERE id = ? AND status = 'open'",
            (max(0.0, float(new_qty)), trade_id),
        )
        conn.commit()


def list_open_trades(ticker: str | None = None) -> list[dict[str, Any]]:
    with _connect() as conn:
        if ticker:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'open' AND ticker = ? ORDER BY entry_ts DESC",
                (ticker.upper(),),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'open' ORDER BY entry_ts DESC"
            ).fetchall()
    return [_row_to_trade(r) for r in rows]


def list_recent_trades(limit: int = 30, ticker: str | None = None) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 200))
    with _connect() as conn:
        if ticker:
            rows = conn.execute(
                "SELECT * FROM trades WHERE ticker = ? ORDER BY entry_ts DESC LIMIT ?",
                (ticker.upper(), limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY entry_ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [_row_to_trade(r) for r in rows]


def strike_stop_cooldown_active(
    ticker: str,
    strike: float,
    option_type: str,
    *,
    lookback: int = 5,
) -> bool:
    """True if this strike was recently stopped out (avoid immediate re-entry)."""
    opt = option_type.lower()
    for trade in list_recent_trades(limit=lookback, ticker=ticker):
        if trade.get("status") != "closed" and trade.get("exit_reason") is None:
            continue
        if str(trade.get("exit_reason", "")) != "stop_loss":
            continue
        if float(trade.get("signal_strike") or trade.get("strike") or 0) != float(strike):
            continue
        if str(trade.get("option_type", "")).lower() != opt:
            continue
        return True
    return False


def get_account_equity() -> float:
    from gex_core.trading.config import account_equity_usd, live_trading_allowed, use_webull_account_equity

    if live_trading_allowed() and use_webull_account_equity():
        from gex_core.trading.webull_broker import fetch_total_account_value

        live_equity = fetch_total_account_value()
        if live_equity is not None:
            return live_equity

    perf = get_performance_summary()
    return account_equity_usd() + float(perf.get("total_pnl_usd") or 0.0)


def get_account_equity_source() -> str:
    from gex_core.trading.config import (
        live_trading_allowed,
        paper_trading_only,
        use_webull_account_equity,
        webull_configured,
    )

    if live_trading_allowed() and use_webull_account_equity() and webull_configured():
        from gex_core.trading.webull_broker import fetch_total_account_value

        if fetch_total_account_value() is not None:
            return "webull_live"
        return "configured_fallback"

    if not paper_trading_only():
        return "configured"

    perf = get_performance_summary()
    if float(perf.get("total_pnl_usd") or 0.0) != 0.0:
        return "paper_journal"
    return "configured"


def get_performance_summary(ticker: str | None = None) -> dict[str, Any]:
    with _connect() as conn:
        if ticker:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'closed' AND ticker = ? ORDER BY exit_ts DESC",
                (ticker.upper(),),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'closed' ORDER BY exit_ts DESC"
            ).fetchall()

    closed = [_row_to_trade(r) for r in rows]
    if not closed:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_pnl_pct": 0.0,
            "total_pnl_usd": 0.0,
            "by_signal": {},
            "lessons": [],
        }

    wins = [t for t in closed if (t.get("pnl_pct") or 0) > 0]
    by_signal: dict[str, dict[str, Any]] = {}
    for t in closed:
        sig = t.get("signal_type") or "unknown"
        bucket = by_signal.setdefault(sig, {"count": 0, "wins": 0, "pnl_sum": 0.0})
        bucket["count"] += 1
        bucket["pnl_sum"] += float(t.get("pnl_pct") or 0)
        if (t.get("pnl_pct") or 0) > 0:
            bucket["wins"] += 1

    for sig, bucket in by_signal.items():
        bucket["win_rate"] = bucket["wins"] / bucket["count"] if bucket["count"] else 0.0
        bucket["avg_pnl_pct"] = bucket["pnl_sum"] / bucket["count"] if bucket["count"] else 0.0

    lessons = _derive_lessons(closed, by_signal)
    return {
        "total_trades": len(closed),
        "win_rate": len(wins) / len(closed),
        "avg_pnl_pct": sum(float(t.get("pnl_pct") or 0) for t in closed) / len(closed),
        "total_pnl_usd": sum(float(t.get("pnl_usd") or 0) for t in closed),
        "by_signal": by_signal,
        "lessons": lessons,
        "recent_closed": closed[:8],
    }


def _derive_lessons(closed: list[dict[str, Any]], by_signal: dict[str, dict[str, Any]]) -> list[str]:
    lessons: list[str] = []
    if not closed:
        return lessons

    stop_exits = sum(1 for t in closed if t.get("exit_reason") == "stop_loss")
    tp_exits = sum(1 for t in closed if t.get("exit_reason") == "take_profit")
    if stop_exits > len(closed) * 0.4:
        lessons.append("Stop-loss exits are frequent — consider requiring stronger gamma acceleration before entry.")
    if tp_exits > len(closed) * 0.35:
        lessons.append("Take-profit targets are working well — current target fits this gamma regime.")

    eod = sum(1 for t in closed if t.get("exit_reason") in {"eod_flatten", "session_gap"})
    if eod > len(closed) * 0.3:
        lessons.append("Many session-end exits — entries may be too late in the day for 0DTE.")

    magnet = sum(1 for t in closed if t.get("exit_reason") == "magnet_touch")
    if magnet > 0:
        lessons.append(f"Magnet-touch exits working ({magnet} trades) — keep magnet proximity targets.")

    best_sig = max(by_signal.items(), key=lambda kv: kv[1].get("avg_pnl_pct", -999), default=(None, None))
    worst_sig = min(by_signal.items(), key=lambda kv: kv[1].get("avg_pnl_pct", 999), default=(None, None))
    if best_sig[0] and best_sig[1]["count"] >= 3:
        lessons.append(
            f"Best performer: {best_sig[0]} signals ({best_sig[1]['win_rate']:.0%} win rate, "
            f"{best_sig[1]['avg_pnl_pct']:+.1%} avg)."
        )
    if worst_sig[0] and worst_sig[1]["count"] >= 3 and worst_sig[1]["avg_pnl_pct"] < 0:
        lessons.append(
            f"Weakest performer: {worst_sig[0]} signals ({worst_sig[1]['avg_pnl_pct']:+.1%} avg) — AI should down-weight."
        )
    return lessons


def get_trade_memory_for_ai(ticker: str | None = None) -> dict[str, Any]:
    """Compact trade history for LLM context."""
    perf = get_performance_summary(ticker)
    open_positions = list_open_trades(ticker)
    return {
        "armed": is_trader_armed(),
        "open_positions": [
            {
                "id": t["id"],
                "option_type": t["option_type"],
                "strike": t["strike"],
                "entry_spot": t["entry_spot"],
                "signal_type": t["signal_type"],
                "signal_gamma": t["signal_gamma"],
                "entry_ts": t["entry_ts"],
            }
            for t in open_positions
        ],
        "performance": {
            "total_trades": perf["total_trades"],
            "win_rate": round(perf["win_rate"], 3),
            "avg_pnl_pct": round(perf["avg_pnl_pct"], 4),
            "total_pnl_usd": round(perf["total_pnl_usd"], 2),
            "by_signal": perf["by_signal"],
            "lessons": perf["lessons"],
        },
        "recent_trades": [
            {
                "option_type": t["option_type"],
                "strike": t["strike"],
                "signal_type": t["signal_type"],
                "pnl_pct": t.get("pnl_pct"),
                "exit_reason": t.get("exit_reason"),
            }
            for t in perf.get("recent_closed", [])
        ],
    }
