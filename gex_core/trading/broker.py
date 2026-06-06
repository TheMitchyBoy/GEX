"""Broker abstraction — paper simulation or live Webull execution."""

from __future__ import annotations

from typing import Any, Protocol

from gex_core.trading.config import paper_trading_only, webull_configured


class Broker(Protocol):
    name: str

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
    ) -> dict[str, Any]: ...

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
    ) -> dict[str, Any]: ...

    def position_pnl_pct(self, trade: dict[str, Any], *, spot: float) -> float | None: ...


def get_broker() -> Broker:
    if paper_trading_only():
        from gex_core.trading.paper_broker import PaperBroker

        return PaperBroker()
    if webull_configured():
        from gex_core.trading.webull_broker import WebullBroker

        return WebullBroker()
    from gex_core.trading.paper_broker import PaperBroker

    return PaperBroker()


def broker_mode_label() -> str:
    if paper_trading_only():
        return "paper"
    if webull_configured():
        return "webull_live"
    return "paper"
