"""Option chain parsing and data-quality filters."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd

_SYMBOL_PATTERN = re.compile(
    r"^(?P<root>[A-Z]+)(?P<expiration>\d{6})(?P<type>[CP])(?P<strike_raw>\d{8})$"
)

_NUMERIC_COLS = ("gamma", "open_interest", "iv", "bid", "ask", "volume", "charm")


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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


@dataclass(frozen=True)
class DataQualityConfig:
    """Filter thresholds from environment variables."""

    enabled: bool = field(default_factory=lambda: _env_bool("GEX_DATA_FILTERS", True))
    min_open_interest: int = field(default_factory=lambda: _env_int("GEX_MIN_OPEN_INTEREST", 1))
    min_gamma: float = field(default_factory=lambda: _env_float("GEX_MIN_GAMMA", 0.0))
    max_iv: float = field(default_factory=lambda: _env_float("GEX_MAX_IV", 6.0))
    max_bid_ask_spread_pct: float = field(
        default_factory=lambda: _env_float("GEX_MAX_BID_ASK_SPREAD_PCT", 1.0)
    )
    max_strike_distance_pct: float = field(
        default_factory=lambda: _env_float("GEX_MAX_STRIKE_DISTANCE_PCT", 0.35)
    )
    dedupe_symbols: bool = field(default_factory=lambda: _env_bool("GEX_DEDUPE_SYMBOLS", True))


@dataclass
class DataQualityReport:
    rows_in: int = 0
    rows_out: int = 0
    filters_enabled: bool = True
    removed: dict[str, int] = field(default_factory=dict)

    @property
    def rows_removed(self) -> int:
        return max(0, self.rows_in - self.rows_out)

    def record(self, step: str, before: int, after: int) -> None:
        dropped = max(0, before - after)
        if dropped:
            self.removed[step] = self.removed.get(step, 0) + dropped

    def to_dict(self) -> dict:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "filters_enabled": self.filters_enabled,
            "removed": dict(self.removed),
        }

    def summary_line(self) -> str:
        kept = f"{self.rows_out}/{self.rows_in} contracts kept"
        if not self.filters_enabled:
            return f"Data quality: {kept} (filters off, symbols parsed only)."
        if not self.removed:
            return f"Data quality: {kept} (all filters passed)."
        breakdown = ", ".join(f"{name} -{n}" for name, n in self.removed.items())
        return f"Data quality: {kept} ({self.rows_removed} removed: {breakdown})."


def parse_option_symbols(data: pd.DataFrame) -> pd.DataFrame:
    if "option" not in data.columns:
        raise ValueError("Option payload is missing 'option' symbols.")

    frame = data.copy()
    parsed = frame["option"].astype(str).str.strip().str.extract(_SYMBOL_PATTERN)
    frame = frame.join(parsed)
    frame["strike"] = pd.to_numeric(frame["strike_raw"], errors="coerce") / 1000.0
    frame["expiration"] = pd.to_datetime(frame["expiration"], format="%y%m%d", errors="coerce")

    for col in _NUMERIC_COLS:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    return frame


def _mask_valid_symbol(frame: pd.DataFrame) -> pd.Series:
    required = frame[["type", "strike", "gamma", "open_interest", "expiration"]]
    return required.notna().all(axis=1)


def _mask_quote_quality(frame: pd.DataFrame, max_spread_pct: float) -> pd.Series:
    mid = (frame["bid"] + frame["ask"]) / 2.0
    spread = (frame["ask"] - frame["bid"]).abs()
    has_quote = (frame["bid"] > 0) & (frame["ask"] > 0) & (frame["ask"] >= frame["bid"])
    spread_ok = spread <= (mid.abs() * max_spread_pct).clip(lower=0.01)
    return (~has_quote) | (has_quote & spread_ok)


def _build_filter_steps(
    frame: pd.DataFrame, spot: float | None, config: DataQualityConfig
) -> list[tuple[str, pd.Series]]:
    today = pd.Timestamp(datetime.today().date())
    steps: list[tuple[str, pd.Series]] = [
        ("invalid_symbol", _mask_valid_symbol(frame)),
        ("low_oi", frame["open_interest"] >= config.min_open_interest),
        ("non_positive_gamma", frame["gamma"] > config.min_gamma),
        ("expired", frame["expiration"].dt.normalize() >= today),
    ]

    if spot is not None and config.max_strike_distance_pct > 0:
        lower = spot * (1 - config.max_strike_distance_pct)
        upper = spot * (1 + config.max_strike_distance_pct)
        steps.append(("far_otm", (frame["strike"] >= lower) & (frame["strike"] <= upper)))

    if "iv" in frame.columns:
        steps.append(
            (
                "iv_outlier",
                frame["iv"].isna() | ((frame["iv"] > 0) & (frame["iv"] <= config.max_iv)),
            )
        )

    if {"bid", "ask"}.issubset(frame.columns):
        steps.append(("wide_or_crossed_spread", _mask_quote_quality(frame, config.max_bid_ask_spread_pct)))

    return steps


def apply_quality_filters(
    frame: pd.DataFrame,
    spot: float | None,
    config: DataQualityConfig,
    report: DataQualityReport,
) -> pd.DataFrame:
    report.filters_enabled = config.enabled
    if not config.enabled:
        return frame

    for step_name, mask in _build_filter_steps(frame, spot, config):
        before = len(frame)
        frame = frame.loc[mask]
        report.record(step_name, before, len(frame))

    if config.dedupe_symbols and "option" in frame.columns:
        before = len(frame)
        frame = frame.sort_values("open_interest", ascending=False).drop_duplicates(
            subset=["option"], keep="first"
        )
        report.record("duplicate_symbol", before, len(frame))

    return frame


def clean_option_data(
    data: pd.DataFrame,
    spot: float | None = None,
    config: DataQualityConfig | None = None,
) -> tuple[pd.DataFrame, DataQualityReport]:
    if "gamma" not in data.columns or "open_interest" not in data.columns:
        raise ValueError("Option payload is missing required columns: gamma/open_interest.")

    cfg = config or DataQualityConfig()
    report = DataQualityReport(rows_in=len(data), filters_enabled=cfg.enabled)

    frame = parse_option_symbols(data)
    frame = apply_quality_filters(frame, spot=spot, config=cfg, report=report)

    if "charm" in frame.columns:
        frame = frame.copy()
        frame["charm"] = frame["charm"].fillna(0.0)

    if frame.empty:
        raise RuntimeError("No valid option rows remained after data cleaning.")

    report.rows_out = len(frame)
    return frame.reset_index(drop=True), report
