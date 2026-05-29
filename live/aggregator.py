"""Simple aggregator to compute GEX deltas from option flow events.

Event format (JSON lines):
{
  "option": "SPX260620C04800000",
  "gamma": 0.00012,        # gamma per contract (optional but recommended)
  "quantity": 50,          # number of contracts (positive int)
  "side": "buy",         # 'buy' or 'sell' (affects dealer exposure interpretation)
  "spot": 4800.0           # current spot price (optional, can be provided per-run)
}

This module estimates how a trade flow event changes notional GEX by strike.
"""
from typing import Dict, Any, List, Tuple
import re
import numpy as np

contract_size = 100


def parse_option_symbol(symbol: str) -> Dict[str, Any]:
    """Extract type and strike from standard option symbol like SPX260620C04800000.
    Returns dict with 'type' ('C' or 'P') and numeric 'strike'."""
    m_type = re.search(r"\d([A-Z])\d", symbol)
    m_strike = re.search(r"\d[A-Z](\d+)\d\d\d$", symbol)
    result = {"type": None, "strike": None}
    if m_type:
        result["type"] = m_type.group(1)
    if m_strike:
        try:
            result["strike"] = int(m_strike.group(1))
        except Exception:
            result["strike"] = None
    return result


class GEXAggregator:
    def __init__(self, spot: float = None):
        self.spot = spot
        # store per-strike current GEX estimate (notional change)
        self.gex_by_strike = {}

    def ingest_event(self, event: Dict[str, Any]) -> Tuple[int, float]:
        """Process a single flow event and return (strike, delta_gex).

        Expects event to contain 'option', 'quantity', and either 'gamma' or will skip.
        """
        symbol = event.get("option")
        if symbol is None:
            raise ValueError("Event missing 'option' field")

        parsed = parse_option_symbol(symbol)
        strike = parsed.get("strike")
        opt_type = parsed.get("type")
        if strike is None or opt_type is None:
            raise ValueError(f"Could not parse option symbol: {symbol}")

        gamma = event.get("gamma")
        if gamma is None:
            # can't compute impact without gamma per contract
            raise ValueError("Event missing 'gamma' — cannot compute GEX delta")

        qty = int(event.get("quantity", 0))
        if qty == 0:
            return strike, 0.0

        # Interpret side: buys from customer increase dealer short exposure; however
        # for notional delta, we treat quantity as change in open interest magnitude.
        side = event.get("side", "buy").lower()
        multiplier = 1
        # For calls dealer long gamma => positive; puts negative
        type_mul = -1 if opt_type == "P" else 1

        # Spot: prefer event-provided spot, else aggregator spot, else must be provided
        spot = float(event.get("spot")) if event.get("spot") is not None else self.spot
        if spot is None:
            raise ValueError("Spot price unknown; provide 'spot' in event or aggregator")

        # delta in open interest approximation: assume quantity increases OI if side is buy
        delta_oi = qty if side in ("buy", "open") else -qty

        # GEX formula used in main.py: spot * gamma * open_interest * contract_size * spot * 0.01
        delta_gex = spot * float(gamma) * delta_oi * contract_size * spot * 0.01
        delta_gex = delta_gex * type_mul

        # accumulate
        prev = self.gex_by_strike.get(strike, 0.0)
        self.gex_by_strike[strike] = prev + delta_gex

        return strike, float(delta_gex)

    def top_movers(self, top_n: int = 5) -> List[Tuple[int, float]]:
        items = sorted(self.gex_by_strike.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return items[:top_n]


class EnhancedGEXAggregator(GEXAggregator):
    """Enhanced aggregator with time-decay, flow-imbalance, aggressiveness and scoring.

    Maintains per-strike rolling metrics and produces a normalized signal score.
    """
    def __init__(self, spot: float = None, decay_half_life_sec: float = 300.0):
        super().__init__(spot=spot)
        # per-strike state: { strike: {weighted_gex, buy_qty, sell_qty, agg_score, ts}} 
        self.state = {}
        # decay parameter (half-life in seconds)
        self.decay_half_life = decay_half_life_sec

    def _decay_factor(self, dt_seconds: float) -> float:
        # exponential decay factor for dt seconds
        if dt_seconds <= 0:
            return 1.0
        # half-life decay: factor = 0.5 ** (dt / half_life)
        return 0.5 ** (dt_seconds / max(self.decay_half_life, 1e-9))

    def ingest_event(self, event: Dict[str, Any], timestamp: float = None) -> Dict[str, Any]:
        """Process event and update enhanced per-strike metrics.

        Returns a dict with strike, delta_gex, and current signal for that strike.
        """
        import time

        ts = timestamp if timestamp is not None else time.time()
        strike, delta_gex = super().ingest_event(event)

        entry = self.state.get(strike, {
            "weighted_gex": 0.0,
            "buy_qty": 0,
            "sell_qty": 0,
            "agg_score": 0.0,
            "last_ts": ts,
            "agg_count": 0,
        })

        # apply decay to weighted_gex and agg_score based on time since last update
        dt = ts - entry["last_ts"]
        f = self._decay_factor(dt)
        entry["weighted_gex"] *= f
        entry["agg_score"] *= f

        # Update quantities and weighted_gex
        side = event.get("side", "buy").lower()
        qty = int(event.get("quantity", 0))
        if side in ("buy", "open"):
            entry["buy_qty"] += qty
        else:
            entry["sell_qty"] += qty

        # Aggressiveness: optional field or inferred from trade_price vs mid
        aggress = event.get("aggressiveness")
        if aggress is None:
            # try to infer from bid/ask/trade_price if present
            bid = event.get("bid")
            ask = event.get("ask")
            tp = event.get("trade_price")
            if bid is not None and ask is not None and tp is not None and ask > bid:
                mid = 0.5 * (bid + ask)
                spread = max(ask - bid, 1e-9)
                aggress = (tp - mid) / spread
            else:
                aggress = 0.0

        # update aggregate score (simple EWMA)
        entry["agg_score"] += float(aggress)
        entry["weighted_gex"] += float(delta_gex)
        entry["last_ts"] = ts
        entry["agg_count"] += 1

        self.state[strike] = entry

        signal = self.compute_signal(strike)
        return {"strike": strike, "delta_gex": delta_gex, "signal": signal}

    def compute_signal(self, strike: int) -> Dict[str, Any]:
        """Compute normalized signal for a strike based on current state across strikes.

        Returns dict with score (-1..1), direction, recent_gex, flow_imbalance, avg_aggressiveness.
        """
        import math

        if strike not in self.state:
            return {"score": 0.0, "direction": "neutral", "recent_gex": 0.0, "flow_imbalance": 0.0, "avg_aggressiveness": 0.0}

        entry = self.state[strike]
        recent_gex = entry["weighted_gex"]
        buy = entry["buy_qty"]
        sell = entry["sell_qty"]
        total = buy + sell if (buy + sell) > 0 else 1
        flow_imbalance = (buy - sell) / total
        avg_aggress = entry["agg_score"]/max(entry["agg_count"],1)

        # Normalize recent_gex across all strikes using mean/std
        values = [abs(v["weighted_gex"]) for v in self.state.values()]
        mean = np.mean(values) if values else 0.0
        std = np.std(values) if values else 1.0
        if std < 1e-9:
            z = 0.0
        else:
            z = (recent_gex - mean)/std

        # Combine signals: heavier weight to signed recent_gex and flow imbalance and aggressiveness
        raw_score = 0.6 * np.tanh(recent_gex / max(std,1e-9)) + 0.3 * flow_imbalance + 0.1 * np.tanh(avg_aggress)
        # Clamp to [-1,1]
        score = max(-1.0, min(1.0, float(raw_score)))
        if score > 0.15:
            direction = "up"
        elif score < -0.15:
            direction = "down"
        else:
            direction = "neutral"

        return {
            "score": score,
            "direction": direction,
            "recent_gex": recent_gex,
            "flow_imbalance": flow_imbalance,
            "avg_aggressiveness": avg_aggress,
        }

    def top_signals(self, top_n: int = 10) -> List[Tuple[int, Dict[str, Any]]]:
        scored = []
        for s in self.state.keys():
            sig = self.compute_signal(s)
            scored.append((s, sig))
        scored.sort(key=lambda kv: abs(kv[1]["score"]), reverse=True)
        return scored[:top_n]
