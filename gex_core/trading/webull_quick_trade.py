"""Webull quick-trade helpers: quote analysis, entry/exit prices, and order execution."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

from gex_core.market_time import market_today
from gex_core.trading.config import (
    execution_ticker,
    live_trading_allowed,
    paper_trading_only,
    signal_ticker,
    stop_loss_pct,
    take_profit_pct,
    webull_configured,
    webull_limit_buffer_pct,
)
from gex_core.trading.execution import (
    build_webull_option_symbol,
    map_execution_strike,
    resolve_execution_spot,
    sync_execution_context,
)
from gex_core.trading.journal import close_trade, list_open_trades, open_trade, record_decision
from gex_core.trading.paper_broker import estimate_entry_premium
from gex_core.trading.webull_broker import (
    WebullBroker,
    fetch_option_quote,
    fetch_total_account_value,
    webull_auth_status,
)

logger = logging.getLogger(__name__)

PriceStyle = Literal["passive", "mid", "smart", "aggressive"]

_POSITION_CACHE: tuple[float, list[dict[str, Any]]] | None = None


@dataclass(frozen=True)
class QuoteAnalysis:
    symbol: str
    bid: float | None
    ask: float | None
    last: float | None
    mid: float | None
    spread: float | None
    spread_pct: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "bid": self.bid,
            "ask": self.ask,
            "last": self.last,
            "mid": self.mid,
            "spread": self.spread,
            "spread_pct": self.spread_pct,
        }


def _tick(price: float) -> float:
    if price < 3.0:
        return 0.01
    return 0.05


def estimate_option_quote(
    *,
    symbol: str,
    spot: float,
    strike: float,
    spread_pct: float = 0.06,
) -> dict[str, float | str | None]:
    """Synthetic NBBO when Webull is unavailable or the contract has no live quote."""
    mid = estimate_entry_premium(spot, strike)
    half = mid * spread_pct / 2.0
    bid = round(max(0.01, mid - half), 2)
    ask = round(max(0.01, mid + half), 2)
    return {
        "bid": bid,
        "ask": ask,
        "last": round(mid, 2),
        "symbol": symbol,
        "source": "estimated",
    }


def _resolve_quote_spot(
    spot: float | None,
    strategy_trade: dict[str, Any] | None,
) -> float | None:
    if spot and spot > 0:
        return float(spot)
    if strategy_trade:
        exec_spot = strategy_trade.get("execution_spot")
        if exec_spot and float(exec_spot) > 0:
            return float(exec_spot)
        signal_spot = strategy_trade.get("signal_spot")
        if signal_spot and float(signal_spot) > 0:
            mapped = resolve_execution_spot(signal_spot=float(signal_spot))
            if mapped and mapped > 0:
                return float(mapped)
    mapped = resolve_execution_spot()
    return float(mapped) if mapped and mapped > 0 else None


def _maybe_map_strike_for_execution(
    underlying: str,
    strike: float,
    *,
    strategy_trade: dict[str, Any] | None = None,
    spot: float | None = None,
) -> float:
    """Map SPX-scale strikes to SPY when the trade desk underlying is SPY."""
    und = underlying.upper()
    exec_sym = execution_ticker().upper()
    sig_sym = signal_ticker().upper()
    if und != exec_sym or und == sig_sym or strike <= 500:
        return strike

    if strategy_trade and strategy_trade.get("execution_strike"):
        signal_strike = float(strategy_trade.get("signal_strike") or 0.0)
        if abs(strike - signal_strike) < 1.0 or strike > 500:
            return float(strategy_trade["execution_strike"])

    signal_spot = float((strategy_trade or {}).get("signal_spot") or spot or 0.0)
    exec_spot = float((strategy_trade or {}).get("execution_spot") or 0.0)
    if exec_spot <= 0 and signal_spot > 0:
        resolved = resolve_execution_spot(signal_spot=signal_spot)
        exec_spot = float(resolved) if resolved else 0.0
    if signal_spot > 0 and exec_spot > 0:
        return map_execution_strike(strike, signal_spot=signal_spot, execution_spot=exec_spot)
    return strike


def analyze_quote(quote: dict[str, float | None]) -> QuoteAnalysis:
    bid = quote.get("bid")
    ask = quote.get("ask")
    last = quote.get("last")
    symbol = str(quote.get("symbol") or "")
    mid = None
    spread = None
    spread_pct = None
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        mid = round((bid + ask) / 2.0, 2)
        spread = round(ask - bid, 2)
        if mid > 0:
            spread_pct = round(spread / mid, 4)
    elif last is not None and last > 0:
        mid = round(last, 2)
    return QuoteAnalysis(
        symbol=symbol,
        bid=bid,
        ask=ask,
        last=last,
        mid=mid,
        spread=spread,
        spread_pct=spread_pct,
    )


def _round_limit(price: float) -> float:
    return round(max(0.01, price), 2)


def entry_limit_price(
    analysis: QuoteAnalysis,
    *,
    style: PriceStyle = "smart",
    buffer_pct: float | None = None,
    spot: float = 0.0,
    strike: float = 0.0,
) -> float:
    """Suggest a buy limit — lower is cheaper; mid/smart balance fill vs price."""
    buf = webull_limit_buffer_pct() if buffer_pct is None else buffer_pct
    bid = analysis.bid
    ask = analysis.ask
    mid = analysis.mid

    if style == "passive" and bid and bid > 0:
        return _round_limit(bid)
    if style == "aggressive":
        ref = ask or analysis.last or mid
        if ref and ref > 0:
            return _round_limit(ref * (1.0 + buf))
    if style == "mid" and mid and mid > 0:
        return _round_limit(mid)
    if style == "smart":
        if bid and ask and bid > 0 and ask > 0:
            if analysis.spread_pct is not None and analysis.spread_pct <= 0.08:
                return _round_limit(mid or (bid + ask) / 2.0)
            return _round_limit(bid + _tick(bid))
        ref = ask or analysis.last or mid
        if ref and ref > 0:
            return _round_limit(ref * (1.0 + buf * 0.5))
    if spot > 0 and strike > 0:
        return _round_limit(estimate_entry_premium(spot, strike) * (1.0 + buf))
    return _round_limit(mid or 0.05)


def exit_limit_price(
    analysis: QuoteAnalysis,
    *,
    style: PriceStyle = "smart",
    buffer_pct: float | None = None,
    entry_premium: float | None = None,
) -> float:
    """Suggest a sell limit — higher captures more premium."""
    buf = webull_limit_buffer_pct() if buffer_pct is None else buffer_pct
    bid = analysis.bid
    ask = analysis.ask
    mid = analysis.mid

    if style == "passive":
        ref = ask or mid or analysis.last
        if ref and ref > 0:
            return _round_limit(ref)
    if style == "aggressive":
        ref = bid or analysis.last or mid
        if ref and ref > 0:
            return _round_limit(max(0.01, ref * (1.0 - buf * 0.5)))
    if style == "mid" and mid and mid > 0:
        return _round_limit(mid)
    if style == "smart":
        if bid and ask and ask > 0:
            if analysis.spread_pct is not None and analysis.spread_pct <= 0.08:
                return _round_limit(mid or (bid + ask) / 2.0)
            return _round_limit(ask - _tick(ask))
        ref = bid or analysis.last
        if ref and ref > 0:
            return _round_limit(max(0.01, ref * (1.0 - buf * 0.25)))
    if entry_premium and entry_premium > 0:
        return _round_limit(entry_premium)
    return _round_limit(mid or 0.01)


def entry_conditions(analysis: QuoteAnalysis, *, quote_source: str = "live") -> dict[str, Any]:
    """When to enter — favors tight spreads and two-sided quotes."""
    if not analysis.bid and not analysis.ask and not analysis.last:
        return {
            "action": "wait",
            "signal": "no_quote",
            "summary": "No live option quote — check Webull connection or contract.",
            "score": 0.0,
            "quote_source": quote_source,
        }

    score = 0.5
    reasons: list[str] = []
    if quote_source == "estimated":
        reasons.append("Paper estimate — connect Webull for live NBBO before trading")
        score = 0.48
    if analysis.bid and analysis.ask and analysis.bid > 0 and analysis.ask > 0:
        score += 0.25
        reasons.append("Two-sided NBBO available")
    else:
        reasons.append("One-sided quote — use mid/last with caution")

    if analysis.spread_pct is not None:
        if analysis.spread_pct <= 0.05:
            score += 0.2
            reasons.append(f"Tight spread ({analysis.spread_pct:.1%}) — good for limit at mid")
        elif analysis.spread_pct <= 0.12:
            score += 0.05
            reasons.append(f"Moderate spread ({analysis.spread_pct:.1%}) — prefer bid+tick entry")
        else:
            score -= 0.15
            reasons.append(f"Wide spread ({analysis.spread_pct:.1%}) — wait or bid only")

    action = "go" if score >= 0.65 else "wait" if score >= 0.45 else "avoid"
    if quote_source == "estimated" and action == "go":
        action = "wait"
    return {
        "action": action,
        "signal": "spread_quality",
        "summary": "; ".join(reasons),
        "score": round(min(1.0, max(0.0, score)), 2),
        "quote_source": quote_source,
    }


def exit_conditions(
    analysis: QuoteAnalysis,
    *,
    entry_premium: float | None = None,
    peak_premium: float | None = None,
) -> dict[str, Any]:
    """When to exit — target take-profit, stop-loss, or spread-aware sell."""
    if entry_premium is None or entry_premium <= 0:
        return {
            "action": "hold",
            "signal": "no_entry",
            "summary": "Set entry premium to evaluate exit targets.",
            "score": 0.0,
        }

    mark = analysis.mid or analysis.bid or analysis.last
    if not mark or mark <= 0:
        return {
            "action": "hold",
            "signal": "no_quote",
            "summary": "No mark for PnL — wait for quote.",
            "score": 0.0,
        }

    pnl_pct = (mark - entry_premium) / entry_premium
    tp = take_profit_pct()
    sl = stop_loss_pct()
    reasons: list[str] = []
    action = "hold"
    score = 0.5

    if pnl_pct >= tp:
        action = "sell"
        score = 0.95
        reasons.append(f"At/above take-profit target ({pnl_pct:+.1%} vs {tp:.0%})")
    elif pnl_pct <= -sl:
        action = "sell"
        score = 0.9
        reasons.append(f"At/below stop-loss ({pnl_pct:+.1%} vs -{sl:.0%})")
    else:
        reasons.append(f"Mark PnL {pnl_pct:+.1%} (TP {tp:.0%}, SL -{sl:.0%})")

    if peak_premium and peak_premium > entry_premium:
        peak_pnl = (peak_premium - entry_premium) / entry_premium
        retrace = (peak_premium - mark) / peak_premium if peak_premium > 0 else 0.0
        if peak_pnl >= 0.08 and retrace >= 0.25:
            action = "sell"
            score = max(score, 0.85)
            reasons.append(f"Trailing: gave back {retrace:.0%} from session peak")

    if action == "hold" and analysis.spread_pct is not None and analysis.spread_pct <= 0.06:
        reasons.append("Tight spread — passive sell at ask may maximize exit")

    return {
        "action": action,
        "signal": "pnl_targets",
        "summary": "; ".join(reasons),
        "score": round(score, 2),
        "pnl_pct": round(pnl_pct, 4),
        "mark": round(mark, 2),
    }


def price_ladder(analysis: QuoteAnalysis, *, entry_premium: float | None = None) -> dict[str, Any]:
    """All suggested limit prices for UI buttons."""
    return {
        "entry": {
            "passive": entry_limit_price(analysis, style="passive"),
            "mid": entry_limit_price(analysis, style="mid"),
            "smart": entry_limit_price(analysis, style="smart"),
            "aggressive": entry_limit_price(analysis, style="aggressive"),
        },
        "exit": {
            "passive": exit_limit_price(analysis, style="passive", entry_premium=entry_premium),
            "mid": exit_limit_price(analysis, style="mid", entry_premium=entry_premium),
            "smart": exit_limit_price(analysis, style="smart", entry_premium=entry_premium),
            "aggressive": exit_limit_price(analysis, style="aggressive", entry_premium=entry_premium),
        },
    }


def fetch_broker_option_positions(*, force_refresh: bool = False) -> list[dict[str, Any]]:
    """Best-effort open option positions from Webull account API."""
    global _POSITION_CACHE
    if not webull_configured():
        return []
    from gex_core.trading.config import webull_position_cache_seconds
    from gex_core.trading.webull_broker import webull_api_paused

    now = time.monotonic()
    if (
        not force_refresh
        and _POSITION_CACHE is not None
        and (now - _POSITION_CACHE[0]) < webull_position_cache_seconds()
    ):
        return list(_POSITION_CACHE[1])
    if webull_api_paused() and _POSITION_CACHE is not None:
        return list(_POSITION_CACHE[1])
    try:
        from gex_core.trading.webull_broker import _ensure_client, _response_body, webull_account_id

        trade = _ensure_client()
        aid = webull_account_id()
        if not aid:
            return []
        resp = _response_body(trade.account_v2.get_account_position(aid))
        positions = resp.get("data") or resp.get("positions") or resp.get("items") or []
        if isinstance(positions, dict):
            positions = positions.get("items") or positions.get("positions") or []
        out: list[dict[str, Any]] = []
        for pos in positions:
            if not isinstance(pos, dict):
                continue
            inst = str(pos.get("instrument_type") or pos.get("asset_type") or "").upper()
            sym = str(pos.get("symbol") or pos.get("ticker") or "")
            if inst and "OPTION" not in inst and len(sym) < 15:
                continue
            qty = pos.get("quantity") or pos.get("qty") or pos.get("position")
            try:
                qty_f = float(qty)
            except (TypeError, ValueError):
                qty_f = 0.0
            if qty_f <= 0:
                continue
            out.append(
                {
                    "symbol": sym,
                    "strike": float(pos.get("strike_price") or pos.get("strike") or 0),
                    "option_type": str(pos.get("option_type") or pos.get("put_call") or "").lower(),
                    "qty": qty_f,
                    "cost": float(pos.get("cost") or pos.get("average_cost") or pos.get("cost_basis") or 0),
                    "market_value": float(pos.get("market_value") or pos.get("current_value") or 0),
                    "last_price": float(pos.get("last_price") or pos.get("market_price") or 0),
                    "unrealized_pnl": float(pos.get("unrealized_profit_loss") or pos.get("unrealized_pnl") or 0),
                }
            )
        _POSITION_CACHE = (time.monotonic(), out)
        return list(out)
    except Exception as exc:
        logger.warning("Webull positions fetch failed: %s", exc)
        if _POSITION_CACHE is not None:
            return list(_POSITION_CACHE[1])
        return []


def build_recommended_trade(*, strategy_state: dict[str, Any]) -> dict[str, Any]:
    """Map gamma strategy recommendation to an execution contract for the trade desk."""
    signals = strategy_state.get("signals") or {}
    filters = strategy_state.get("filters") or {}
    advice = strategy_state.get("advice") or {}
    rec = signals.get("recommended") or {}
    signal_spot = float(strategy_state.get("spot") or 0.0)

    if not signals.get("available"):
        return {
            "available": False,
            "reason": signals.get("reason") or signals.get("skip_reason") or "No gamma signal",
            "filters": filters,
            "advice": advice,
        }
    if not rec.get("strike") or signal_spot <= 0:
        return {
            "available": False,
            "reason": "Recommended strike or spot unavailable",
            "filters": filters,
            "advice": advice,
            "recommended": rec,
        }

    signal_strike = float(rec["strike"])
    option_type = str(rec.get("option_type") or "call").lower()
    mapped = map_signal_strike_to_execution(signal_strike, signal_spot=signal_spot)
    if mapped.get("error"):
        return {
            "available": False,
            "reason": mapped["error"],
            "recommended": rec,
            "filters": filters,
            "advice": advice,
        }

    exec_strike = float(mapped["execution_strike"])
    expire_date = market_today()
    und = execution_ticker().upper()
    filters_ok = bool(filters.get("approve"))
    advice_ok = bool(advice.get("approve"))
    return {
        "available": True,
        "signal_ticker": signal_ticker().upper(),
        "execution_ticker": und,
        "signal_strike": signal_strike,
        "execution_strike": exec_strike,
        "option_type": option_type,
        "magnet_strike": rec.get("magnet_strike"),
        "signal_type": rec.get("signal_type"),
        "rationale": rec.get("rationale"),
        "gamma_delta": rec.get("gamma_delta"),
        "symbol": build_webull_option_symbol(
            underlying=und,
            expire_date=expire_date,
            option_type=option_type,
            strike=exec_strike,
        ),
        "expire_date": expire_date,
        "signal_spot": signal_spot,
        "execution_spot": mapped.get("execution_spot"),
        "spot_ratio": mapped.get("spot_ratio"),
        "filters": filters,
        "advice": advice,
        "strategy_ready": filters_ok and advice_ok,
        "recommended": rec,
    }


def combined_entry_guidance(
    quote_entry: dict[str, Any],
    *,
    strategy_trade: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Blend spread-quality entry with gamma filter/advice state."""
    out = dict(quote_entry)
    if not strategy_trade or not strategy_trade.get("available"):
        out["strategy"] = None
        return out

    filters = strategy_trade.get("filters") or {}
    advice = strategy_trade.get("advice") or {}
    filters_ok = bool(filters.get("approve"))
    advice_ok = bool(advice.get("approve"))
    strategy_ready = bool(strategy_trade.get("strategy_ready"))

    reasons: list[str] = [quote_entry.get("summary") or ""]
    if strategy_ready:
        reasons.append(f"Strategy: {advice.get('reason') or filters.get('reason') or 'filters pass'}")
    elif filters_ok:
        reasons.append(f"Filters pass; advisor: {advice.get('reason') or 'wait'}")
    else:
        reasons.append(f"Strategy blocked: {filters.get('reason') or advice.get('reason') or 'filters fail'}")

    score = float(quote_entry.get("score") or 0.0)
    if strategy_ready:
        score = min(1.0, score + 0.2)
    elif not filters_ok:
        score = max(0.0, score - 0.25)

    action = quote_entry.get("action") or "wait"
    if strategy_ready and action == "go":
        action = "go"
    elif not filters_ok and action == "go":
        action = "wait"
    elif strategy_ready and action == "wait" and score >= 0.65:
        action = "go"

    out.update(
        {
            "action": action,
            "score": round(min(1.0, max(0.0, score)), 2),
            "summary": "; ".join(r for r in reasons if r),
            "strategy": {
                "ready": strategy_ready,
                "filters_ok": filters_ok,
                "advice_ok": advice_ok,
                "rationale": strategy_trade.get("rationale"),
                "signal_type": strategy_trade.get("signal_type"),
            },
        }
    )
    return out


def quote_payload(
    *,
    underlying: str,
    option_type: str,
    strike: float,
    expire_date: str | None = None,
    entry_premium: float | None = None,
    peak_premium: float | None = None,
    spot: float | None = None,
    strategy_trade: dict[str, Any] | None = None,
) -> dict[str, Any]:
    underlying = underlying.upper()
    expire_date = expire_date or market_today()
    spot_val = _resolve_quote_spot(spot, strategy_trade)
    strike = _maybe_map_strike_for_execution(
        underlying,
        strike,
        strategy_trade=strategy_trade,
        spot=spot_val,
    )
    option_symbol = build_webull_option_symbol(
        underlying=underlying,
        expire_date=expire_date,
        option_type=option_type,
        strike=strike,
    )
    quote_source = "live"
    quote: dict[str, float | str | None] = {
        "bid": None,
        "ask": None,
        "last": None,
        "symbol": option_symbol,
    }
    if webull_configured():
        quote = fetch_option_quote(
            underlying=underlying,
            option_type=option_type,
            strike=strike,
            expire_date=expire_date,
        )
    if not any(quote.get(k) for k in ("bid", "ask", "last")) and spot_val and spot_val > 0:
        quote = estimate_option_quote(symbol=option_symbol, spot=spot_val, strike=strike)
        quote_source = "estimated"
    analysis = analyze_quote(quote)
    entry = entry_conditions(analysis, quote_source=quote_source)
    if strategy_trade:
        entry = combined_entry_guidance(entry, strategy_trade=strategy_trade)
    mark = analysis.mid or analysis.bid or analysis.last
    unrealized_pnl_pct = None
    if entry_premium and entry_premium > 0 and mark and mark > 0:
        unrealized_pnl_pct = round((mark - entry_premium) / entry_premium, 4)
    return {
        "underlying": underlying,
        "option_type": option_type.lower(),
        "strike": strike,
        "expire_date": expire_date,
        "option_symbol": option_symbol,
        "quote_source": quote_source,
        "spot": spot_val,
        "unrealized_pnl_pct": unrealized_pnl_pct,
        "quote": analysis.to_dict(),
        "entry_conditions": entry,
        "exit_conditions": exit_conditions(
            analysis,
            entry_premium=entry_premium,
            peak_premium=peak_premium,
        ),
        "prices": price_ladder(analysis, entry_premium=entry_premium),
        "limit_price_for_buy": entry_limit_price(
            analysis,
            style="smart",
            spot=spot_val or 0.0,
            strike=strike,
        ),
        "limit_price_for_sell": exit_limit_price(
            analysis,
            style="smart",
            entry_premium=entry_premium,
        ),
        "webull_auth": webull_auth_status(),
    }


def dashboard_state(*, signal_ticker_arg: str | None = None) -> dict[str, Any]:
    """Full payload for the quick-trade dashboard."""
    sig = (signal_ticker_arg or signal_ticker()).upper()
    und = execution_ticker().upper()
    live = live_trading_allowed()
    paper = paper_trading_only()
    from gex_core.trading.config import use_webull_account_equity

    equity = (
        fetch_total_account_value()
        if live and use_webull_account_equity() and webull_configured()
        else None
    )
    exec_spot = resolve_execution_spot()
    signal_spot = None
    try:
        from gex_core.uw_price_stream import get_uw_price_stream

        signal_spot = get_uw_price_stream().get_latest_price(sig)
    except Exception:
        pass

    return {
        "signal_ticker": sig,
        "execution_ticker": und,
        "webull_configured": webull_configured(),
        "live_trading_allowed": live,
        "paper_mode": paper,
        "webull_auth": webull_auth_status(),
        "account_equity": equity,
        "execution_spot": exec_spot,
        "signal_spot": signal_spot,
        "default_expire": market_today(),
        "stop_loss_pct": stop_loss_pct(),
        "take_profit_pct": take_profit_pct(),
        "journal_positions": list_open_trades(sig),
        "broker_positions": fetch_broker_option_positions(),
    }


def map_signal_strike_to_execution(
    signal_strike: float,
    *,
    signal_spot: float,
    execution_spot: float | None = None,
) -> dict[str, Any]:
    exec_spot = execution_spot or resolve_execution_spot(signal_spot=signal_spot)
    if not exec_spot or signal_spot <= 0:
        return {"error": "execution spot unavailable"}
    exec_strike = map_execution_strike(signal_strike, signal_spot=signal_spot, execution_spot=exec_spot)
    ctx = sync_execution_context(signal_spot=signal_spot)
    return {
        "signal_strike": signal_strike,
        "execution_strike": exec_strike,
        "signal_spot": signal_spot,
        "execution_spot": exec_spot,
        "spot_ratio": ctx.get("spot_ratio"),
        "symbol": build_webull_option_symbol(
            underlying=execution_ticker(),
            expire_date=market_today(),
            option_type="call",
            strike=exec_strike,
        ),
    }


def execute_buy(
    *,
    underlying: str,
    option_type: str,
    strike: float,
    quantity: int,
    limit_price: float | None = None,
    expire_date: str | None = None,
    spot: float = 0.0,
    ticker: str | None = None,
    price_style: PriceStyle = "smart",
    journal: bool = True,
) -> dict[str, Any]:
    if not live_trading_allowed():
        return {"ok": False, "error": "Live Webull trading not enabled (set GEX_TRADER_PAPER=0 and Webull credentials)."}

    expire_date = expire_date or market_today()
    quote = fetch_option_quote(
        underlying=underlying,
        option_type=option_type,
        strike=strike,
        expire_date=expire_date,
    )
    analysis = analyze_quote(quote)
    limit = limit_price if limit_price and limit_price > 0 else entry_limit_price(
        analysis, style=price_style, spot=spot, strike=strike
    )

    broker = WebullBroker()
    result = broker.buy_option(
        underlying=underlying,
        option_type=option_type,
        strike=strike,
        expire_date=expire_date,
        quantity=quantity,
        limit_price=limit,
        spot=spot,
    )
    result["quote"] = analysis.to_dict()
    result["entry_conditions"] = entry_conditions(analysis)
    result["limit_used"] = limit

    if result.get("ok") and journal:
        trade_id = open_trade(
            ticker=ticker or signal_ticker(),
            option_type=option_type,
            strike=strike,
            qty=float(result.get("filled_qty") or quantity),
            entry_spot=spot,
            entry_premium=float(result.get("filled_premium") or limit),
            signal_type="manual_webull",
            signal_strike=strike,
            signal_gamma=0.0,
            gamma_delta=0.0,
            ai_confidence=1.0,
            ai_reason="Webull quick-trade buy",
            meta={"client_order_id": result.get("client_order_id"), "source": "webull_quick_trade"},
        )
        result["journal_trade_id"] = trade_id
        record_decision(
            ticker=ticker or signal_ticker(),
            action="quick_buy",
            payload=result,
            ai_verdict="filled" if result.get("ok") else "rejected",
        )
    return result


def execute_sell(
    *,
    underlying: str,
    option_type: str,
    strike: float,
    quantity: int,
    limit_price: float | None = None,
    expire_date: str | None = None,
    entry_premium: float | None = None,
    spot: float = 0.0,
    ticker: str | None = None,
    trade_id: int | None = None,
    price_style: PriceStyle = "smart",
    journal: bool = True,
) -> dict[str, Any]:
    if not live_trading_allowed():
        return {"ok": False, "error": "Live Webull trading not enabled."}

    expire_date = expire_date or market_today()
    quote = fetch_option_quote(
        underlying=underlying,
        option_type=option_type,
        strike=strike,
        expire_date=expire_date,
    )
    analysis = analyze_quote(quote)
    limit = limit_price if limit_price and limit_price > 0 else exit_limit_price(
        analysis, style=price_style, entry_premium=entry_premium
    )

    broker = WebullBroker()
    result = broker.sell_option(
        underlying=underlying,
        option_type=option_type,
        strike=strike,
        expire_date=expire_date,
        quantity=quantity,
        limit_price=limit,
    )
    result["quote"] = analysis.to_dict()
    result["exit_conditions"] = exit_conditions(analysis, entry_premium=entry_premium)
    result["limit_used"] = limit

    if result.get("ok") and journal and trade_id and entry_premium and entry_premium > 0:
        exit_prem = float(result.get("filled_premium") or limit)
        pnl_pct = (exit_prem - entry_premium) / entry_premium
        pnl_usd = (exit_prem - entry_premium) * 100.0 * quantity
        close_trade(
            trade_id,
            exit_spot=spot,
            exit_premium=exit_prem,
            pnl_pct=pnl_pct,
            pnl_usd=pnl_usd,
            exit_reason="manual_webull",
        )
        result["journal_trade_id"] = trade_id
        record_decision(ticker=ticker or signal_ticker(), action="quick_sell", payload=result)
    return result
