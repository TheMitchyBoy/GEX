"""Option chain parsing and data-quality filters."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

_SYMBOL_PATTERN = re.compile(
    r"^(?P<root>[A-Z]+)(?P<expiration>\d{6})(?P<type>[CP])(?P<strike_raw>\d{8})$"
)


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
    """Env-tunable filter thresholds. Set GEX_DATA_FILTERS=0 to disable all."""

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
    removed: dict[str, int] = field(default_factory=dict)

    @property
    def rows_removed(self) -> int:
        return max(0, self.rows_in - self.rows_out)

    def record(self, step: str, before: int, after: int) -> None:
        dropped = max(0, before - after)
        if dropped:
            self.removed[step] = self.removed.get(step, 0) + dropped

    def summary_line(self) -> str:
        if not self.removed:
            return f"Data quality: {self.rows_out}/{self.rows_in} contracts kept (no filters applied)."
        parts = ", ".join(f"{name} -{count}" for name, count in self.removed.items())
        return (
            f"Data quality: {self.rows_out}/{self.rows_in} contracts kept "
            f"({self.rows_removed} removed: {parts})."
        )


def parse_option_symbols(data: pd.DataFrame) -> pd.DataFrame:
    if "option" not in data.columns:
        raise ValueError("Option payload is missing 'option' symbols.")

    frame = data.copy()
    symbols = frame["option"].astype(str).str.strip()
    parsed = symbols.str.extract(_SYMBOL_PATTERN)
    frame = frame.join(parsed)

    frame["strike"] = pd.to_numeric(frame["strike_raw"], errors="coerce") / 1000.0
    frame["gamma"] = pd.to_numeric(frame["gamma"], errors="coerce")
    frame["open_interest"] = pd.to_numeric(frame["open_interest"], errors="coerce")
    frame["expiration"] = pd.to_datetime(frame["expiration"], format="%y%m%d", errors="coerce")

    for col in ("iv", "bid", "ask", "volume", "charm"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    return frame


def _apply_mask(frame: pd.DataFrame, mask: pd.Series, step: str, report: DataQualityReport) -> pd.DataFrame:
    before = len(frame)
    out = frame.loc[mask]
    report.record(step, before, len(out))
    return out


def apply_quality_filters(
    frame: pd.DataFrame,
    spot: float | None,
    config: DataQualityConfig,
    report: DataQualityReport,
) -> pd.DataFrame:
    if not config.enabled:
        report.rows_out = len(frame)
        return frame

    # Required fields after symbol parse
    frame = _apply_mask(
        frame,
        frame[["type", "strike", "gamma", "open_interest", "expiration"]].notna().all(axis=1),
        "invalid_symbol",
        report,
    )

    frame = _apply_mask(frame, frame["open_interest"] >= config.min_open_interest, "low_oi", report)
    frame = _apply_mask(frame, frame["gamma"] > config.min_gamma, "non_positive_gamma", report)

    today = pd.Timestamp(datetime.today().date())
    frame = _apply_mask(frame, frame["expiration"].dt.normalize() >= today, "expired", report)

    if spot is not None and config.max_strike_distance_pct > 0:
        lower = spot * (1 - config.max_strike_distance_pct)
        upper = spot * (1 + config.max_strike_distance_pct)
        in_range = (frame["strike"] >= lower) & (frame["strike"] <= upper)
        frame = _apply_mask(frame, in_range, "far_otm", report)

    if "iv" in frame.columns:
        iv_ok = frame["iv"].isna() | ((frame["iv"] > 0) & (frame["iv"] <= config.max_iv))
        frame = _apply_mask(frame, iv_ok, "iv_outlier", report)

    if {"bid", "ask"}.issubset(frame.columns):
        mid = (frame["bid"] + frame["ask"]) / 2.0
        spread = (frame["ask"] - frame["bid"]).abs()
        has_quote = (frame["bid"] > 0) & (frame["ask"] > 0) & (frame["ask"] >= frame["bid"])
        spread_ok = spread <= (mid.abs() * config.max_bid_ask_spread_pct).clip(lower=0.01)
        quote_ok = (~has_quote) | (has_quote & spread_ok)
        frame = _apply_mask(frame, quote_ok, "wide_or_crossed_spread", report)

    if config.dedupe_symbols and "option" in frame.columns:
        before = len(frame)
        frame = frame.sort_values("open_interest", ascending=False).drop_duplicates(
            subset=["option"], keep="first"
        )
        report.record("duplicate_symbol", before, len(frame))

    return frame


def finalize_option_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "charm" in frame.columns:
        frame = frame.copy()
        frame["charm"] = frame["charm"].fillna(0.0)
    if frame.empty:
        raise RuntimeError("No valid option rows remained after data cleaning.")
    return frame.reset_index(drop=True)


def clean_option_data(
    data: pd.DataFrame,
    spot: float | None = None,
    config: DataQualityConfig | None = None,
) -> tuple[pd.DataFrame, DataQualityReport]:
    if "gamma" not in data.columns or "open_interest" not in data.columns:
        raise ValueError("Option payload is missing required columns: gamma/open_interest.")

    cfg = config or DataQualityConfig()
    report = DataQualityReport(rows_in=len(data))

    parsed = parse_option_symbols(data)
    filtered = apply_quality_filters(parsed, spot=spot, config=cfg, report=report)
    cleaned = finalize_option_frame(filtered)

    report.rows_out = len(cleaned)
    return cleaned, report
