"""Webull OpenAPI live options execution."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from gex_core.trading.config import (
    webull_account_id,
    webull_app_key,
    webull_app_secret,
    webull_endpoint,
    webull_limit_buffer_pct,
    webull_region,
)
from gex_core.trading.paper_broker import estimate_entry_premium, estimate_option_pnl_pct

logger = logging.getLogger(__name__)

_client = None
_trade_client = None


def _ensure_client():
    global _client, _trade_client
    if _trade_client is not None:
        return _trade_client
    from webull.core.client import ApiClient
    from webull.trade.trade_client import TradeClient

    key = webull_app_key()
    secret = webull_app_secret()
    region = webull_region()
    _client = ApiClient(key, secret, region)
    _client.add_endpoint(region, webull_endpoint())
    _trade_client = TradeClient(_client)
    return _trade_client


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


def _is_ok(body: dict[str, Any]) -> bool:
    code = body.get("code") or body.get("status")
    if code in (0, "0", 200, "200", "SUCCESS", "success"):
        return True
    if body.get("success") is True:
        return True
    return not body.get("error") and not body.get("msg")


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
        ok = _is_ok(placed)
        if not ok:
            logger.error("Webull place_option failed: %s", placed)
        return {
            "ok": ok,
            "stage": "place",
            "client_order_id": client_order_id,
            "limit_price": limit_price,
            "preview": preview,
            "response": placed,
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
        cid = (client_order_id or uuid.uuid4().hex)[:32]
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
        placed = _response_body(trade.order_v2.place_option(self._account(), [order]))
        ok = _is_ok(placed)
        return {
            "ok": ok,
            "stage": "sell",
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


def limit_price_for_buy(spot: float, strike: float, *, side: str = "buy") -> float:
    est = estimate_entry_premium(spot, strike)
    buf = webull_limit_buffer_pct()
    if side == "buy":
        return round(est * (1.0 + buf), 2)
    return round(max(0.01, est * (1.0 - buf * 0.5)), 2)
