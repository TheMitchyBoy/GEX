"""Entry signals toward min or max net gamma strikes (spot-exposures/strike)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from gex_core.features import select_atm_strike_series
from gex_core.trading.config import max_strike_distance_pct
from gex_core.trading.signals import _clean, _option_type_for_strike

WallTarget = Literal["min", "max"]


@dataclass(frozen=True)
class WallGexSignal:
    signal_type: str
    strike: float
    gamma_bn: float
    option_type: str
    spot: float
    wall_strike: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "strike": self.strike,
            "gamma_bn": self.gamma_bn,
            "option_type": self.option_type,
            "spot": self.spot,
            "wall_strike": self.wall_strike,
            "rationale": self.rationale,
            "direction": self.option_type,
        }


def compute_wall_gex_signal(
    exposure: pd.Series | None,
    *,
    spot: float | None,
    target: WallTarget = "min",
    window_pct: float = 0.12,
) -> dict[str, Any]:
    """Pick call/put toward the min or max net GEX strike near spot.

    Wall below spot → puts; wall above spot → calls.
    """
    cur = _clean(exposure)
    spot_val = float(spot or 0.0)
    if cur.empty:
        return {"available": False, "reason": "No gamma exposure data"}
    if spot_val <= 0:
        return {"available": False, "reason": "No spot price"}

    search = select_atm_strike_series(cur, spot_val, window_pct=window_pct, min_strikes=5)
    if search.empty:
        search = cur

    if target == "max":
        positive = search[search > 0]
        if not positive.empty:
            search = positive
        wall_strike = float(search.idxmax())
        wall_gamma = float(search.max())
        label = "Highest"
        signal_type = "max_gamma_strike"
    else:
        wall_strike = float(search.idxmin())
        wall_gamma = float(search.min())
        label = "Lowest"
        signal_type = "min_gamma_strike"

    option_type = _option_type_for_strike(wall_strike, spot_val)
    dist = abs(wall_strike - spot_val) / spot_val
    max_dist = max_strike_distance_pct()

    if dist > max_dist:
        return {
            "available": False,
            "reason": (
                f"{label} GEX strike {wall_strike:.0f} is {dist:.1%} from spot "
                f"(max {max_dist:.1%})"
            ),
            "spot": spot_val,
            "wall_strike": wall_strike,
            "gamma_bn": wall_gamma,
            "target": target,
        }

    sig = WallGexSignal(
        signal_type=signal_type,
        strike=wall_strike,
        gamma_bn=wall_gamma,
        option_type=option_type,
        spot=spot_val,
        wall_strike=wall_strike,
        rationale=(
            f"{label} GEX {wall_gamma:+.3f} Bn at {wall_strike:.0f} "
            f"→ buy {option_type} toward wall"
        ),
    )
    out = {
        "available": True,
        "spot": spot_val,
        "target": target,
        "recommended": sig.to_dict(),
        "master_direction": option_type,
    }
    if target == "max":
        out["max_gamma_strike"] = sig.to_dict()
    else:
        out["min_gamma_strike"] = sig.to_dict()
    return out


def compute_low_gex_signal(
    exposure: pd.Series | None,
    *,
    spot: float | None,
    window_pct: float = 0.12,
) -> dict[str, Any]:
    return compute_wall_gex_signal(exposure, spot=spot, target="min", window_pct=window_pct)


def compute_high_gex_signal(
    exposure: pd.Series | None,
    *,
    spot: float | None,
    window_pct: float = 0.12,
) -> dict[str, Any]:
    return compute_wall_gex_signal(exposure, spot=spot, target="max", window_pct=window_pct)
