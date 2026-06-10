"""Webull OpenAPI live options execution."""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from gex_core.trading.config import (
    live_trading_allowed,
    webull_account_id,
    webull_app_key,
    webull_app_secret,
    webull_configured,
    webull_data_endpoint,
    webull_equity_cache_seconds,
    webull_fill_poll_sec,
    webull_fill_timeout_sec,
    webull_limit_buffer_pct,
    webull_option_category,
    webull_region,
    webull_trade_endpoint,
    webull_use_uat,
)
from gex_core.trading.execution import build_webull_option_symbol
from gex_core.trading.paper_broker import estimate_entry_premium, estimate_option_pnl_pct

logger = logging.getLogger(__name__)

_client = None
_trade_client = None
_data_client = None

_FILLED = frozenset({"FILLED", "PARTIAL FILLED", "PARTIAL_FILLED", 4, 5, "4", "5"})
_TERMINAL_BAD = frozenset({"CANCELLED", "FAILED", "REJECTED", 2, 3, "2", "3"})
_EQUITY_CACHE: dict[str, tuple[float, float]] = {}
_LAST_WEBULL_ERROR: dict[str, Any] = {}
_RATE_LIMIT_COOLDOWN_SEC = 45


def clear_webull_equity_cache() -> None:
    """Drop cached Webull net-liquidation values (for tests)."""
    _EQUITY_CACHE.clear()


def clear_webull_error_state() -> None:
    """Reset cached Webull auth/rate-limit state (for tests)."""
    _LAST_WEBULL_ERROR.clear()


def reset_webull_clients() -> None:
    """Drop cached SDK clients so the next call re-initializes auth."""
    global _client, _trade_client, _data_client
    _client = None
    _trade_client = None
    _data_client = None


def invalidate_local_webull_token() -> bool:
    """Remove a stale on-disk token so Webull can issue a fresh one."""
    path = _token_file_path()
    try:
        if path.is_file():
            path.unlink()
            logger.info("Removed stale Webull token file at %s", path)
            return True
    except OSError as exc:
        logger.warning("Failed to remove Webull token file %s: %s", path, exc)
    return False


def _token_file_path() -> Path:
    env_dir = os.environ.get("WEBULL_OPENAPI_TOKEN_DIR", "").strip()
    base = Path(env_dir).expanduser() if env_dir else Path("conf")
    return base.resolve() / "token.txt"


def _is_rate_limited_message(message: str) -> bool:
    upper = message.upper()
    return "429" in upper or "TOO_MANY_REQUESTS" in upper or "TOO MANY REQUESTS" in upper


def _is_invalid_token_message(message: str) -> bool:
    upper = message.upper()
    return (
        "INVALID_TOKEN" in upper
        or ("HTTP STATUS: 401" in upper and "UNAUTHORIZED" in upper)
        or ("CODE: INVALID_TOKEN" in upper)
        or ("PERMISSION DENIED" in upper and "401" in upper)
    )


def _classify_webull_error(exc: Exception | str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(exc, dict):
        code = str(exc.get("code") or exc.get("error_code") or "").upper()
        msg = str(exc.get("msg") or exc.get("message") or exc.get("error_msg") or "")
        message = f"Code: {code}, Msg: {msg}".strip(", ")
    else:
        message = str(exc)
        code = ""
        msg = message
        error_code = getattr(exc, "error_code", None)
        if error_code:
            code = str(error_code).upper()
        error_msg = getattr(exc, "error_msg", None)
        if error_msg:
            msg = str(error_msg)
        http_status = getattr(exc, "http_status", None)
        if http_status and not message.startswith("HTTP Status:"):
            message = f"HTTP Status: {http_status}, Code: {code}, Msg: {msg}"

    return {
        "message": message,
        "rate_limited": _is_rate_limited_message(message),
        "invalid_token": _is_invalid_token_message(message) or code == "INVALID_TOKEN",
    }


def note_webull_error(exc: Exception | str | dict[str, Any]) -> None:
    """Remember recent Webull failures for dashboard auth/rate-limit banners."""
    flags = _classify_webull_error(exc)
    if flags["invalid_token"]:
        invalidate_local_webull_token()
        reset_webull_clients()
    _LAST_WEBULL_ERROR.update({"at": time.monotonic(), **flags})


def webull_api_paused() -> bool:
    """Skip live Webull calls while we are in a 429 cooldown."""
    return bool(webull_auth_status().get("pause_api"))


def webull_auth_status() -> dict[str, Any]:
    """Dashboard payload for Webull OpenAPI auth and rate-limit guidance."""
    if not webull_configured():
        return {"required": False}
    if not live_trading_allowed():
        return {"required": False, "paper_mode": True}

    recent = _LAST_WEBULL_ERROR
    recent_age = time.monotonic() - float(recent.get("at") or 0.0) if recent.get("at") else None
    recent_active = recent_age is not None and recent_age < 600
    rate_limited = (
        bool(recent.get("rate_limited"))
        and recent_age is not None
        and recent_age < _RATE_LIMIT_COOLDOWN_SEC
    )
    invalid_token = bool(recent.get("invalid_token")) and recent_active

    if webull_use_uat():
        return {
            "required": False,
            "uat": True,
            "message": "UAT sandbox — use GEX_WEBULL_USE_UAT=1 with sandbox credentials.",
        }

    show_banner = rate_limited or invalid_token
    if invalid_token:
        headline = "Webull token rejected (401)"
        detail = (
            "Webull returned INVALID_TOKEN / permission denied. Any stale token file was cleared. "
            "Disarm, wait ~30 seconds, confirm GEX_WEBULL_APP_KEY, GEX_WEBULL_APP_SECRET, and "
            "GEX_WEBULL_ACCOUNT_ID match your production OpenAPI app, then retry."
        )
    elif rate_limited:
        headline = "Webull rate limit (429)"
        detail = "Disarm the trader and wait about 30 seconds before retrying."
    else:
        headline = ""
        detail = ""

    return {
        "required": True,
        "show_banner": show_banner,
        "pause_api": rate_limited,
        "invalid_token": invalid_token,
        "rate_limited": rate_limited,
        "headline": headline,
        "message": detail,
        "last_error": recent.get("message") if recent_active else None,
    }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _balance_payload(body: dict[str, Any]) -> dict[str, Any]:
    payload = _unwrap_data(body)
    if isinstance(payload, list):
        return payload[0] if payload and isinstance(payload[0], dict) else {}
    return payload if isinstance(payload, dict) else {}


def parse_total_account_value(balance: dict[str, Any]) -> float | None:
    """Extract net account value from a Webull balance payload."""
    for key in (
        "total_net_liquidation_value",
        "totalNetLiquidationValue",
        "net_liquidation_value",
        "total_asset",
        "totalAsset",
    ):
        val = _safe_float(balance.get(key))
        if val is not None:
            return val

    assets = balance.get("account_currency_assets") or balance.get("accountCurrencyAssets") or []
    if isinstance(assets, dict):
        assets = assets.get("items") or assets.get("data") or [assets]
    for row in assets:
        if not isinstance(row, dict):
            continue
        currency = str(row.get("currency") or "USD").upper()
        if currency and currency != "USD":
            continue
        val = _safe_float(row.get("net_liquidation_value") or row.get("netLiquidationValue"))
        if val is not None:
            return val
    return None


def fetch_account_balance(*, account_id: str | None = None) -> dict[str, Any]:
    """Fetch raw Webull account balance for the configured account."""
    trade = _ensure_client()
    aid = (account_id or webull_account_id()).strip()
    if not aid:
        raise EnvironmentError("GEX_WEBULL_ACCOUNT_ID is not set")
    return _response_body(trade.account_v2.get_account_balance(aid))


def fetch_total_account_value(*, force_refresh: bool = False) -> float | None:
    """Return Webull net liquidation value for risk-based position sizing."""
    aid = webull_account_id().strip()
    if not aid:
        return None

    cache_ttl = webull_equity_cache_seconds()
    now = time.monotonic()
    cached = _EQUITY_CACHE.get(aid)
    if not force_refresh and cached and (now - cached[0]) < cache_ttl:
        return cached[1]

    try:
        body = fetch_account_balance(account_id=aid)
        if not _is_ok(body):
            note_webull_error(body)
            logger.warning("Webull balance request rejected: %s", body)
            return cached[1] if cached else None
        value = parse_total_account_value(_balance_payload(body))
        if value is not None:
            _EQUITY_CACHE[aid] = (now, value)
            return value
    except Exception as exc:
        note_webull_error(exc)
        logger.warning("Webull account balance fetch failed: %s", _format_webull_error(exc))
    return cached[1] if cached else None


def _format_webull_error(exc: Exception) -> str:
    msg = str(exc)
    host = webull_trade_endpoint()
    if "Name or service not known" in msg or "NameResolutionError" in msg or "NXDOMAIN" in msg:
        return (
            f"Cannot resolve Webull API host '{host}'. "
            "Set GEX_WEBULL_ENDPOINT=api.webull.com (production) or "
            "GEX_WEBULL_USE_UAT=1 for sandbox. Use GEX_TRADER_PAPER=1 to skip Webull."
        )
    if "Failed to establish a new connection" in msg or "Max retries exceeded" in msg:
        return f"Cannot connect to Webull API at {host}: {msg}"
    return msg


def _configure_api_client(client, *, region: str) -> None:
    from webull.core.common import api_type

    trade_host = webull_trade_endpoint()
    data_host = webull_data_endpoint()
    client.add_endpoint(region, trade_host, api_type=api_type.DEFAULT)
    if data_host and data_host != trade_host:
        client.add_endpoint(region, data_host, api_type=api_type.QUOTES)


def _ensure_client():
    global _client, _trade_client
    if _trade_client is not None:
        return _trade_client
    if webull_api_paused():
        auth = webull_auth_status()
        raise EnvironmentError(auth.get("headline") or "Webull API paused — rate limited.")
    from webull.core.client import ApiClient
    from webull.trade.trade_client import TradeClient

    key = webull_app_key()
    secret = webull_app_secret()
    region = webull_region()
    try:
        _client = ApiClient(key, secret, region)
        _configure_api_client(_client, region=region)
        _trade_client = TradeClient(_client)
    except Exception as exc:
        note_webull_error(exc)
        raise EnvironmentError(_format_webull_error(exc)) from exc
    return _trade_client


def _ensure_data_client():
    global _data_client
    if _data_client is not None:
        return _data_client
    from webull.data.data_client import DataClient

    _ensure_client()
    _data_client = DataClient(_client)
    return _data_client


def _response_body(response: Any) -> dict[str, Any]:
    if response is None:
        return {}
    if hasattr(response, "json"):
        try:
            data = response.json()
            return data if isinstance(data, dict) else {"data": data}
        except Exception:
            pass
    if isinstance(response, dict):
        return response
    return {"raw": str(response)}


def _unwrap_data(body: dict[str, Any]) -> Any:
    if not isinstance(body, dict):
        return body
    if "data" in body:
        return body["data"]
    return body


def _is_ok(body: dict[str, Any]) -> bool:
    code = body.get("code") or body.get("status")
    if code in (0, "0", 200, "200", "SUCCESS", "success"):
        return True
    if body.get("success") is True:
        return True
    return not body.get("error") and not body.get("msg")


def _order_status(order: dict[str, Any]) -> str:
    status = order.get("status") or order.get("order_status") or order.get("orderStatus") or ""
    return str(status).upper().replace("_", " ")


def _order_filled_qty(order: dict[str, Any]) -> int:
    for key in ("filled_quantity", "filled_qty", "filledQty", "quantity_filled", "filledQuantity"):
        val = order.get(key)
        if val is not None:
            try:
                return max(0, int(float(val)))
            except (TypeError, ValueError):
                continue
    status = _order_status(order)
    if status == "FILLED":
        for key in ("quantity", "qty", "entrust_qty"):
            val = order.get(key)
            if val is not None:
                try:
                    return max(1, int(float(val)))
                except (TypeError, ValueError):
                    pass
    return 0


def _order_avg_price(order: dict[str, Any]) -> float | None:
    for key in ("avg_fill_price", "average_price", "filled_price", "price", "limit_price"):
        val = order.get(key)
        if val is not None:
            try:
                px = float(val)
                if px > 0:
                    return px
            except (TypeError, ValueError):
                continue
    return None


def _extract_quote(snapshot: dict[str, Any]) -> dict[str, float | None]:
    ask = bid = last = None
    for key, target in (
        ("ask", "ask"),
        ("ask_price", "ask"),
        ("best_ask", "ask"),
        ("bid", "bid"),
        ("bid_price", "bid"),
        ("best_bid", "bid"),
        ("latest_price", "last"),
        ("last_price", "last"),
        ("close", "last"),
        ("trade_price", "last"),
    ):
        val = snapshot.get(key)
        if val is None:
            continue
        try:
            px = float(val)
        except (TypeError, ValueError):
            continue
        if px <= 0:
            continue
        if target == "ask":
            ask = px
        elif target == "bid":
            bid = px
        else:
            last = px
    return {"ask": ask, "bid": bid, "last": last}


def fetch_option_quote(
    *,
    underlying: str,
    option_type: str,
    strike: float,
    expire_date: str,
) -> dict[str, float | None]:
    """Fetch bid/ask/last for a single option contract."""
    symbol = build_webull_option_symbol(
        underlying=underlying,
        expire_date=expire_date,
        option_type=option_type,
        strike=strike,
    )
    try:
        data = _ensure_data_client()
        resp = _response_body(
            data.option_market_data.get_option_snapshot(symbol, webull_option_category())
        )
        payload = _unwrap_data(resp)
        rows: list[Any]
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = payload.get("items") or payload.get("snapshots") or payload.get("data") or [payload]
        else:
            rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_symbol = str(row.get("symbol") or row.get("ticker") or "")
            if row_symbol and row_symbol != symbol:
                continue
            quote = _extract_quote(row)
            if any(quote.values()):
                quote["symbol"] = symbol
                return quote
        if isinstance(payload, dict) and payload:
            quote = _extract_quote(payload)
            quote["symbol"] = symbol
            return quote
    except Exception as exc:
        logger.debug("Webull option quote failed for %s: %s", symbol, exc)
    return {"ask": None, "bid": None, "last": None, "symbol": symbol}


def limit_price_for_buy(
    spot: float,
    strike: float,
    *,
    side: str = "buy",
    underlying: str | None = None,
    option_type: str = "call",
    expire_date: str | None = None,
) -> float:
    """Quote-aware limit price with paper-estimate fallback."""
    from gex_core.trading.config import live_trading_allowed

    buf = webull_limit_buffer_pct()
    quote: dict[str, float | None] = {}
    if live_trading_allowed() and underlying and expire_date:
        quote = fetch_option_quote(
            underlying=underlying,
            option_type=option_type,
            strike=strike,
            expire_date=expire_date,
        )
    if side == "buy":
        ref = quote.get("ask") or quote.get("last")
        if ref and ref > 0:
            return round(float(ref) * (1.0 + buf), 2)
        est = estimate_entry_premium(spot, strike)
        return round(max(0.05, est * (1.0 + buf)), 2)
    ref = quote.get("bid") or quote.get("last")
    if ref and ref > 0:
        return round(max(0.01, float(ref) * (1.0 - buf * 0.5)), 2)
    est = estimate_entry_premium(spot, strike)
    return round(max(0.01, est * (1.0 - buf * 0.5)), 2)


def wait_for_order_fill(
    client_order_id: str,
    *,
    timeout_sec: float | None = None,
    poll_sec: float | None = None,
    cancel_on_timeout: bool = True,
) -> dict[str, Any] | None:
    """Poll order detail until filled, failed, or timeout."""
    timeout_sec = timeout_sec if timeout_sec is not None else webull_fill_timeout_sec()
    poll_sec = poll_sec if poll_sec is not None else webull_fill_poll_sec()
    trade = _ensure_client()
    account_id = webull_account_id()
    deadline = time.time() + timeout_sec
    last_detail: dict[str, Any] = {}

    while time.time() < deadline:
        detail = _response_body(trade.order_v2.get_order_detail(account_id, client_order_id))
        if not _is_ok(detail):
            time.sleep(poll_sec)
            continue
        order_raw = _unwrap_data(detail)
        order = order_raw if isinstance(order_raw, dict) else {}
        if isinstance(order_raw, list) and order_raw:
            order = order_raw[0] if isinstance(order_raw[0], dict) else {}
        last_detail = order
        status = _order_status(order)
        filled_qty = _order_filled_qty(order)
        if status in _FILLED or filled_qty > 0:
            avg = _order_avg_price(order)
            return {
                "status": status,
                "filled_qty": filled_qty or int(float(order.get("quantity") or order.get("qty") or 1)),
                "filled_premium": avg,
                "order": order,
                "detail": detail,
            }
        if status in _TERMINAL_BAD:
            return None
        time.sleep(poll_sec)

    if cancel_on_timeout:
        try:
            trade.order_v2.cancel_option(account_id, client_order_id)
        except Exception as exc:
            logger.warning("Webull cancel after timeout failed for %s: %s", client_order_id, exc)
    return None


def build_option_order(
    *,
    client_order_id: str,
    symbol: str,
    strike: float,
    option_type: str,
    expire_date: str,
    side: str,
    quantity: int,
    limit_price: float,
) -> dict[str, Any]:
    side = side.upper()
    opt = option_type.upper()
    qty = str(max(1, int(quantity)))
    return {
        "client_order_id": client_order_id[:32],
        "combo_type": "NORMAL",
        "order_type": "LIMIT",
        "limit_price": f"{limit_price:.2f}",
        "quantity": qty,
        "option_strategy": "SINGLE",
        "side": side,
        "time_in_force": "DAY",
        "entrust_type": "QTY",
        "instrument_type": "OPTION",
        "market": "US",
        "symbol": symbol.upper(),
        "legs": [
            {
                "side": side,
                "quantity": qty,
                "symbol": symbol.upper(),
                "strike_price": f"{strike:.2f}",
                "option_expire_date": expire_date,
                "instrument_type": "OPTION",
                "option_type": opt,
                "market": "US",
            }
        ],
    }


class WebullBroker:
    name = "webull"

    def __init__(self) -> None:
        self._account_id = webull_account_id()

    def _account(self) -> str:
        if not self._account_id:
            raise EnvironmentError("GEX_WEBULL_ACCOUNT_ID is not set")
        return self._account_id

    def buy_option(
        self,
        *,
        underlying: str,
        option_type: str,
        strike: float,
        expire_date: str,
        quantity: int,
        limit_price: float,
        spot: float,
    ) -> dict[str, Any]:
        del spot  # quote path uses option chain; spot kept for interface compatibility
        client_order_id = uuid.uuid4().hex
        order = build_option_order(
            client_order_id=client_order_id,
            symbol=underlying,
            strike=strike,
            option_type=option_type,
            expire_date=expire_date,
            side="BUY",
            quantity=quantity,
            limit_price=limit_price,
        )
        trade = _ensure_client()
        account_id = self._account()
        preview = _response_body(trade.order_v2.preview_option(account_id, [order]))
        if not _is_ok(preview):
            logger.warning("Webull preview rejected: %s", preview)
            return {"ok": False, "stage": "preview", "response": preview, "client_order_id": client_order_id}

        placed = _response_body(trade.order_v2.place_option(account_id, [order]))
        if not _is_ok(placed):
            logger.error("Webull place_option failed: %s", placed)
            return {"ok": False, "stage": "place", "response": placed, "client_order_id": client_order_id}

        fill = wait_for_order_fill(client_order_id)
        if not fill:
            return {
                "ok": False,
                "stage": "fill_timeout",
                "client_order_id": client_order_id,
                "limit_price": limit_price,
                "preview": preview,
                "response": placed,
                "broker": self.name,
            }

        filled_premium = float(fill.get("filled_premium") or limit_price)
        filled_qty = int(fill.get("filled_qty") or quantity)
        return {
            "ok": True,
            "stage": "filled",
            "client_order_id": client_order_id,
            "limit_price": limit_price,
            "filled_premium": filled_premium,
            "filled_qty": filled_qty,
            "fill_status": fill.get("status"),
            "preview": preview,
            "response": placed,
            "fill_detail": fill,
            "broker": self.name,
        }

    def sell_option(
        self,
        *,
        underlying: str,
        option_type: str,
        strike: float,
        expire_date: str,
        quantity: int,
        limit_price: float,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        del client_order_id  # always use a fresh id for sell orders
        cid = uuid.uuid4().hex[:32]
        order = build_option_order(
            client_order_id=cid,
            symbol=underlying,
            strike=strike,
            option_type=option_type,
            expire_date=expire_date,
            side="SELL",
            quantity=quantity,
            limit_price=max(0.01, limit_price),
        )
        trade = _ensure_client()
        account_id = self._account()
        placed = _response_body(trade.order_v2.place_option(account_id, [order]))
        if not _is_ok(placed):
            return {"ok": False, "stage": "place", "client_order_id": cid, "response": placed, "broker": self.name}

        fill = wait_for_order_fill(client_order_id=cid, cancel_on_timeout=False)
        if fill:
            filled_premium = float(fill.get("filled_premium") or limit_price)
            filled_qty = int(fill.get("filled_qty") or quantity)
            return {
                "ok": True,
                "stage": "filled",
                "client_order_id": cid,
                "limit_price": limit_price,
                "filled_premium": filled_premium,
                "filled_qty": filled_qty,
                "response": placed,
                "fill_detail": fill,
                "broker": self.name,
            }
        return {
            "ok": True,
            "stage": "submitted",
            "client_order_id": cid,
            "limit_price": limit_price,
            "response": placed,
            "broker": self.name,
        }

    def position_pnl_pct(self, trade: dict[str, Any], *, spot: float) -> float | None:
        meta = trade.get("meta") or {}
        entry = float(trade.get("entry_premium") or 0)
        if entry <= 0:
            return estimate_option_pnl_pct(
                trade["option_type"],
                entry_spot=float(trade["entry_spot"]),
                current_spot=spot,
                strike=float(trade["strike"]),
            )
        try:
            trade_client = _ensure_client()
            resp = _response_body(trade_client.account_v2.get_account_position(self._account()))
            positions = resp.get("data") or resp.get("positions") or resp.get("items") or []
            if isinstance(positions, dict):
                positions = positions.get("items") or positions.get("positions") or []
            for pos in positions:
                if not isinstance(pos, dict):
                    continue
                strike = float(pos.get("strike_price") or pos.get("strike") or 0)
                if abs(strike - float(trade["strike"])) > 0.01:
                    continue
                opt_type = str(pos.get("option_type") or pos.get("put_call") or "").lower()
                if opt_type and opt_type not in str(trade["option_type"]).lower():
                    continue
                mv = float(pos.get("market_value") or pos.get("current_value") or 0)
                cost = float(pos.get("cost") or pos.get("average_cost") or pos.get("cost_basis") or 0)
                if cost > 0 and mv > 0:
                    return (mv - cost) / cost
                last = float(pos.get("last_price") or pos.get("market_price") or 0)
                if last > 0 and entry > 0:
                    return (last - entry) / entry
        except Exception as exc:
            logger.debug("Webull position lookup failed: %s", exc)
        return estimate_option_pnl_pct(
            trade["option_type"],
            entry_spot=float(trade["entry_spot"]),
            current_spot=spot,
            strike=float(trade["strike"]),
        )
