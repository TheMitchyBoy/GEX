"""Background Unusual Whales websocket feed for live ticker prices.

Subscribes to ``price:<TICKER>`` on ``wss://api.unusualwhales.com/socket`` and
keeps an in-memory ring buffer for dashboard charts. The receive loop runs in a
daemon thread so Flask page renders never block on the socket.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_WS_BASE = "wss://api.unusualwhales.com/socket"
_DEFAULT_TICKERS = ("SPX",)
_RECONNECT_BASE = float(os.environ.get("GEX_UW_PRICE_WS_RECONNECT_SEC", "3"))
_RECONNECT_MAX = float(os.environ.get("GEX_UW_PRICE_WS_RECONNECT_MAX_SEC", "60"))


def _recv_timeout_sec() -> float:
    try:
        return max(15.0, float(os.environ.get("GEX_UW_PRICE_WS_RECV_TIMEOUT_SEC", "45")))
    except (TypeError, ValueError):
        return 45.0


def _is_expected_disconnect(exc: BaseException) -> bool:
    """True when UW or the network dropped the socket without a close handshake."""
    name = type(exc).__name__
    if name in {"ConnectionClosed", "ConnectionClosedOK", "ConnectionClosedError"}:
        return True
    if isinstance(exc, ConnectionResetError):
        return True
    message = str(exc).lower()
    return "no close frame" in message or "connection closed" in message


def _max_points() -> int:
    try:
        return max(100, int(os.environ.get("GEX_UW_PRICE_WS_MAX_POINTS", "2500")))
    except (TypeError, ValueError):
        return 2500


def _ws_enabled() -> bool:
    flag = os.environ.get("GEX_UW_PRICE_WS", "1").strip().lower()
    return flag not in {"0", "false", "no", "off"}


def _price_channel(ticker: str) -> str:
    return f"price:{ticker.upper()}"


def _ms_to_iso(ts_ms: int | float | str) -> str:
    value = float(ts_ms)
    if value > 1e12:
        value /= 1000.0
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def _parse_price_message(channel: str, payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    close = payload.get("close")
    ts = payload.get("time")
    if close is None or ts is None:
        return None
    try:
        price = float(close)
    except (TypeError, ValueError):
        return None
    if price <= 0:
        return None
    ticker = channel.split(":", 1)[-1].upper() if ":" in channel else ""
    return {
        "ticker": ticker,
        "ts": _ms_to_iso(ts),
        "close": price,
        "vol": payload.get("vol"),
        "channel": channel,
        "received_at": time.time(),
    }


class UWPriceStream:
    """Thread-safe in-memory cache of UW websocket price ticks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._points: dict[str, deque[dict[str, Any]]] = {}
        self._latest: dict[str, dict[str, Any]] = {}
        self._status: dict[str, str] = {}
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws: Any | None = None
        self._stop = threading.Event()
        self._tickers: tuple[str, ...] = _DEFAULT_TICKERS
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self, tickers: list[str] | tuple[str, ...] | None = None) -> None:
        if not _ws_enabled():
            logger.info("UW price websocket disabled via GEX_UW_PRICE_WS")
            return
        from gex_core.env_bootstrap import uw_api_configured

        if not uw_api_configured():
            logger.info("UW price websocket not started — UW_API_KEY missing")
            return
        if tickers:
            self._tickers = tuple(t.upper() for t in tickers if t)
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._thread_main,
            name="uw-price-stream",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        loop = self._loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(self._request_close)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _request_close(self) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            asyncio.create_task(ws.close())
        except RuntimeError:
            pass

    def status(self, ticker: str) -> str:
        with self._lock:
            return self._status.get(ticker.upper(), "idle")

    def get_latest_price(self, ticker: str) -> float:
        with self._lock:
            row = self._latest.get(ticker.upper())
        if not row:
            return 0.0
        return float(row.get("close") or 0.0)

    def get_price_points(
        self,
        ticker: str,
        *,
        max_points: int | None = None,
    ) -> list[dict[str, Any]]:
        ticker = ticker.upper()
        cap = max_points or _max_points()
        with self._lock:
            rows = list(self._points.get(ticker, ()))
        if len(rows) > cap:
            rows = rows[-cap:]
        return [{"ts": row["ts"], "close": float(row["close"])} for row in rows]

    def ingest_point(self, ticker: str, point: dict[str, Any]) -> None:
        """Test hook / manual injection."""
        ticker = ticker.upper()
        with self._lock:
            bucket = self._points.setdefault(ticker, deque(maxlen=_max_points()))
            bucket.append(point)
            self._latest[ticker] = point
            self._status[ticker] = "live"

    def _record(self, point: dict[str, Any]) -> None:
        ticker = point["ticker"]
        with self._lock:
            bucket = self._points.setdefault(ticker, deque(maxlen=_max_points()))
            bucket.append(point)
            self._latest[ticker] = point
            self._status[ticker] = "live"

    def _set_status(self, ticker: str, status: str) -> None:
        with self._lock:
            self._status[ticker.upper()] = status

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_forever())
        finally:
            loop.close()
            self._loop = None
            self._connected = False

    async def _run_forever(self) -> None:
        from gex_core.env_bootstrap import uw_api_key

        api_key = uw_api_key()
        if not api_key:
            return

        attempt = 0
        while not self._stop.is_set():
            try:
                await self._connect_and_stream(api_key)
                attempt = 0
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self._stop.is_set():
                    break
                attempt += 1
                delay = min(_RECONNECT_BASE * (2 ** max(0, attempt - 1)), _RECONNECT_MAX)
                if _is_expected_disconnect(exc):
                    logger.info(
                        "UW price websocket disconnected (%s); reconnecting in %.1fs",
                        exc,
                        delay,
                    )
                else:
                    logger.warning(
                        "UW price websocket error (%s); reconnecting in %.1fs",
                        exc,
                        delay,
                    )
                for ticker in self._tickers:
                    self._set_status(ticker, "reconnecting")
                self._connected = False
                await asyncio.sleep(delay)

    async def _connect_and_stream(self, api_key: str) -> None:
        import websockets
        from websockets.exceptions import ConnectionClosed

        uri = f"{_WS_BASE}?token={api_key}"
        recv_timeout = _recv_timeout_sec()
        ws = await websockets.connect(uri, open_timeout=20, close_timeout=2)
        self._ws = ws
        try:
            for ticker in self._tickers:
                channel = _price_channel(ticker)
                await ws.send(json.dumps({"channel": channel, "msg_type": "join"}))
                self._set_status(ticker, "subscribed")
            self._connected = True
            logger.info("UW price websocket connected for %s", ", ".join(self._tickers))

            while not self._stop.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=recv_timeout)
                except asyncio.TimeoutError:
                    try:
                        pong = await ws.ping()
                        await asyncio.wait_for(pong, timeout=10.0)
                    except Exception as exc:
                        raise ConnectionClosed(None, None) from exc
                    continue
                except ConnectionClosed:
                    break
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(message, list) or len(message) != 2:
                    continue
                channel, payload = message
                if not isinstance(channel, str) or not channel.startswith("price:"):
                    continue
                if isinstance(payload, dict) and payload.get("status") == "ok":
                    continue
                point = _parse_price_message(channel, payload)
                if point:
                    self._record(point)
        finally:
            self._connected = False
            self._ws = None
            try:
                await ws.close()
            except Exception:
                pass


_STREAM: UWPriceStream | None = None
_STREAM_LOCK = threading.Lock()


def get_uw_price_stream() -> UWPriceStream:
    global _STREAM
    with _STREAM_LOCK:
        if _STREAM is None:
            _STREAM = UWPriceStream()
        return _STREAM


def start_uw_price_stream(tickers: list[str] | tuple[str, ...] | None = None) -> None:
    get_uw_price_stream().start(tickers=tickers)


def stop_uw_price_stream() -> None:
    global _STREAM
    with _STREAM_LOCK:
        if _STREAM is not None:
            _STREAM.stop()
