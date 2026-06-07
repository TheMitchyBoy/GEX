"""Unusual Whales option marks for SPY 0DTE backtests.

Uses ``GET /api/option-contract/{symbol}/intraday`` for historical minute
bars (``close`` as mark) and ``GET /api/stock/{ticker}/option-contracts`` as
a same-day chain fallback when intraday is unavailable.

Intraday marks align with export snapshot times (UTC keys, market session date
for 0DTE expiry). When UW data is missing, callers should fall back to the
synthetic leverage model in ``paper_broker``.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from gex_core.market_time import ts_market_date
from gex_core.trading.config import execution_ticker
from gex_core.trading.execution import build_webull_option_symbol
from gex_core.uw_loader import _get

logger = logging.getLogger(__name__)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def contract_mid_from_row(row: dict[str, Any]) -> float | None:
    """NBBO mid, else last/close/avg."""
    bid = _safe_float(row.get("nbbo_bid"))
    ask = _safe_float(row.get("nbbo_ask"))
    if bid is not None and ask is not None and ask > 0:
        if bid > 0:
            return (bid + ask) / 2.0
        return ask
    for key in ("last_price", "close", "last", "avg_price"):
        px = _safe_float(row.get(key))
        if px is not None and px > 0:
            return px
    return None


def option_pnl_pct_from_mids(*, entry_mid: float, current_mid: float) -> float:
    if entry_mid <= 0:
        return 0.0
    return (current_mid - entry_mid) / entry_mid


def fetch_uw_option_intraday(
    option_symbol: str,
    *,
    market_date: str,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Minute bars for one contract on a trading day."""
    rows = _get(
        f"/api/option-contract/{option_symbol}/intraday",
        api_key=api_key,
        date=market_date,
    )
    return [row for row in rows if isinstance(row, dict)]


def fetch_uw_option_contracts(
    ticker: str,
    *,
    expiry: str,
    option_type: str | None = None,
    api_key: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Same-day chain snapshot (no historical date param on this endpoint)."""
    params: dict[str, Any] = {"expiry": expiry, "limit": limit}
    if option_type:
        params["option_type"] = option_type
    rows = _get(f"/api/stock/{ticker.upper()}/option-contracts", api_key=api_key, **params)
    return [row for row in rows if isinstance(row, dict)]


def _intraday_rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        ts_raw = row.get("start_time") or row.get("end_time")
        mid = contract_mid_from_row(row)
        if not ts_raw or mid is None or mid <= 0:
            continue
        instant = pd.to_datetime(ts_raw, utc=True, errors="coerce")
        if pd.isna(instant):
            continue
        records.append({"time": instant.to_pydatetime(), "mid": float(mid)})
    if not records:
        return pd.DataFrame(columns=["time", "mid"])
    frame = pd.DataFrame(records).sort_values("time").reset_index(drop=True)
    return frame


def _cache_dir() -> Path:
    from gex_core.data_root import resolve_data_root

    root = resolve_data_root()
    path = root / "uw_option_cache" / "intraday"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_path(option_symbol: str, market_date: str) -> Path:
    safe_symbol = option_symbol.replace("/", "_")
    return _cache_dir() / f"{safe_symbol}_{market_date}.json"


class UwOptionMarkProvider:
    """Lazy UW intraday marks with memory + disk cache."""

    def __init__(self, *, api_key: str | None = None, use_disk_cache: bool = True) -> None:
        self._api_key = api_key
        self._use_disk_cache = use_disk_cache
        self._intraday: dict[tuple[str, str], pd.DataFrame] = {}
        self._chain_day: dict[tuple[str, str], dict[tuple[float, str], float]] = {}
        self.hits = 0
        self.misses = 0
        self.fallbacks = 0

    @property
    def underlying(self) -> str:
        return execution_ticker().upper()

    def option_symbol(self, *, strike: float, option_type: str, market_date: str) -> str:
        return build_webull_option_symbol(
            underlying=self.underlying,
            expire_date=market_date,
            option_type=option_type,
            strike=strike,
        )

    def _read_disk_cache(self, option_symbol: str, market_date: str) -> pd.DataFrame | None:
        if not self._use_disk_cache:
            return None
        path = _cache_path(option_symbol, market_date)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("rows") if isinstance(payload, dict) else payload
            if not isinstance(rows, list):
                return None
            return _intraday_rows_to_frame(rows)
        except (OSError, json.JSONDecodeError, TypeError):
            return None

    def _write_disk_cache(self, option_symbol: str, market_date: str, rows: list[dict[str, Any]]) -> None:
        if not self._use_disk_cache or not rows:
            return
        path = _cache_path(option_symbol, market_date)
        try:
            path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
        except OSError as exc:
            logger.debug("UW option cache write failed for %s: %s", path, exc)

    def _load_intraday(self, option_symbol: str, market_date: str) -> pd.DataFrame:
        key = (option_symbol, market_date)
        if key in self._intraday:
            return self._intraday[key]

        cached = self._read_disk_cache(option_symbol, market_date)
        if cached is not None and not cached.empty:
            self._intraday[key] = cached
            return cached

        rows: list[dict[str, Any]] = []
        try:
            rows = fetch_uw_option_intraday(option_symbol, market_date=market_date, api_key=self._api_key)
        except Exception as exc:
            logger.debug("UW intraday fetch failed for %s on %s: %s", option_symbol, market_date, exc)

        frame = _intraday_rows_to_frame(rows)
        if frame.empty:
            frame = self._chain_fallback_frame(option_symbol=option_symbol, market_date=market_date)
        else:
            self._write_disk_cache(option_symbol, market_date, rows)

        self._intraday[key] = frame
        return frame

    def _chain_fallback_frame(self, *, option_symbol: str, market_date: str) -> pd.DataFrame:
        """Use option-contracts NBBO when intraday is empty (typically same-day only)."""
        chain_key = (self.underlying, market_date)
        if chain_key not in self._chain_day:
            try:
                rows = fetch_uw_option_contracts(
                    self.underlying,
                    expiry=market_date,
                    api_key=self._api_key,
                )
            except Exception as exc:
                logger.debug("UW option-contracts fallback failed for %s: %s", market_date, exc)
                rows = []
            by_contract: dict[tuple[float, str], float] = {}
            for row in rows:
                sym = str(row.get("option_symbol") or "")
                mid = contract_mid_from_row(row)
                if not sym or mid is None:
                    continue
                parsed = _parse_option_symbol(sym)
                if parsed is None:
                    continue
                strike, opt_type = parsed
                by_contract[(strike, opt_type)] = mid
            self._chain_day[chain_key] = by_contract

        parsed = _parse_option_symbol(option_symbol)
        if parsed is None:
            return pd.DataFrame(columns=["time", "mid"])
        strike, opt_type = parsed
        mid = self._chain_day[chain_key].get((strike, opt_type))
        if mid is None:
            return pd.DataFrame(columns=["time", "mid"])
        # Flat mark for the session when only chain snapshot exists.
        session_open = datetime.fromisoformat(f"{market_date}T14:30:00+00:00")
        return pd.DataFrame([{"time": session_open, "mid": float(mid)}])

    def mid_at(self, *, ts: str, strike: float, option_type: str) -> float | None:
        market_date = ts_market_date(ts)
        symbol = self.option_symbol(strike=strike, option_type=option_type, market_date=market_date)
        frame = self._load_intraday(symbol, market_date)
        if frame.empty:
            self.misses += 1
            return None
        instant = _parse_export_instant(ts)
        mid = _mid_at_instant(frame, instant)
        if mid is not None:
            self.hits += 1
        else:
            self.misses += 1
        return mid

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "fallbacks": self.fallbacks}


def _parse_export_instant(ts: str) -> datetime:
    from gex_core.exports import parse_timestamp

    return parse_timestamp(ts).replace(tzinfo=timezone.utc)


def _mid_at_instant(frame: pd.DataFrame, instant: datetime) -> float | None:
    if frame.empty:
        return None
    eligible = frame[frame["time"] <= instant]
    if eligible.empty:
        return float(frame.iloc[0]["mid"]) if frame.iloc[0]["mid"] > 0 else None
    value = float(eligible.iloc[-1]["mid"])
    return value if value > 0 else None


def _parse_option_symbol(symbol: str) -> tuple[float, str] | None:
    """Parse OSI symbol into (strike, option_type)."""
    symbol = symbol.strip().upper()
    if len(symbol) < 15:
        return None
    cp = symbol[-9]
    if cp not in {"C", "P"}:
        return None
    opt_type = "call" if cp == "C" else "put"
    try:
        strike = int(symbol[-8:]) / 1000.0
    except ValueError:
        return None
    return strike, opt_type


def create_uw_mark_provider_if_enabled() -> UwOptionMarkProvider | None:
    from gex_core.trading.config import uw_option_marks_disk_cache, uw_option_marks_enabled

    if not uw_option_marks_enabled():
        return None
    api_key = os.environ.get("UW_API_KEY", "").strip() or None
    if not api_key:
        logger.warning("GEX_TRADER_UW_OPTION_MARKS=1 but UW_API_KEY is unset; using synthetic marks")
        return None
    return UwOptionMarkProvider(
        api_key=api_key,
        use_disk_cache=uw_option_marks_disk_cache(),
    )
