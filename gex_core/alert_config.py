"""Environment-driven thresholds for the alert engine."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AlertConfig:
    wall_shift_threshold: float = 20.0
    near_term_spike_threshold: float = 0.12
    regime_flip_prob_threshold: float = 0.55
    large_delta_gex_bn: float = 3.0
    large_delta_gex_ratio: float = 0.25
    auto_dispatch: bool = False
    dispatch_cooldown_minutes: int = 15
    min_severity_for_auto: str = "high"


def load_alert_config() -> AlertConfig:
    return AlertConfig(
        wall_shift_threshold=_env_float("GEX_ALERT_WALL_SHIFT_PTS", 20.0),
        near_term_spike_threshold=_env_float("GEX_ALERT_NEAR_TERM_SPIKE", 0.12),
        regime_flip_prob_threshold=_env_float("GEX_ALERT_REGIME_FLIP_PROB", 0.55),
        large_delta_gex_bn=_env_float("GEX_ALERT_LARGE_DELTA_BN", 3.0),
        large_delta_gex_ratio=_env_float("GEX_ALERT_LARGE_DELTA_RATIO", 0.25),
        auto_dispatch=_env_bool("GEX_ALERT_AUTO_DISPATCH", False),
        dispatch_cooldown_minutes=_env_int("GEX_ALERT_DISPATCH_COOLDOWN_MINUTES", 15),
        min_severity_for_auto=os.environ.get("GEX_ALERT_AUTO_MIN_SEVERITY", "high"),
    )
