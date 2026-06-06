"""Periscope-style market maker exposure dashboard data assembly."""

from __future__ import annotations

from typing import Any

import pandas as pd

from gex_core.charts import safe_float
from gex_core.features import estimate_gamma_flip
from gex_core.history import build_history, list_timestamps, load_snapshot_at_ts
from gex_core.tickers import PRIMARY_TICKER

EXPOSURE_TYPES = ("gamma", "vanna", "charm")


def _strike_window(series: pd.Series, spot: float, window_pct: float = 0.04) -> pd.Series:
    if series.empty or spot <= 0:
        return series.sort_index()
    lo, hi = spot * (1 - window_pct), spot * (1 + window_pct)
    window = series.loc[(series.index >= lo) & (series.index <= hi)]
    return window.sort_index() if len(window) >= 5 else series.sort_index()


def _exposure_series(
    greek_df: pd.DataFrame | None,
    strike_series: pd.Series | None,
    exposure: str,
) -> pd.Series:
    exposure = exposure.lower()
    if greek_df is not None and not greek_df.empty:
        df = greek_df.set_index("strike") if "strike" in greek_df.columns else greek_df
        if exposure == "gamma":
            if "net_gex" in df.columns:
                return pd.to_numeric(df["net_gex"], errors="coerce").dropna()
            if "call_gex" in df.columns and "put_gex" in df.columns:
                return pd.to_numeric(df["call_gex"], errors="coerce").fillna(0.0) + pd.to_numeric(
                    df["put_gex"], errors="coerce"
                ).fillna(0.0)
        elif exposure == "vanna" and "call_vanna" in df.columns:
            return pd.to_numeric(df["call_vanna"], errors="coerce").fillna(0.0) + pd.to_numeric(
                df["put_vanna"], errors="coerce"
            ).fillna(0.0)
        elif exposure == "charm" and "call_charm" in df.columns:
            return pd.to_numeric(df["call_charm"], errors="coerce").fillna(0.0) + pd.to_numeric(
                df["put_charm"], errors="coerce"
            ).fillna(0.0)
    if exposure == "gamma" and strike_series is not None:
        return pd.Series(strike_series, dtype=float)
    return pd.Series(dtype=float)


def _mm_positions(greek_df: pd.DataFrame | None) -> dict[str, float]:
    """Summarize net dealer call/put positioning from UW greek table."""
    out = {
        "net_call_delta_bn": 0.0,
        "net_put_delta_bn": 0.0,
        "net_call_gex_bn": 0.0,
        "net_put_gex_bn": 0.0,
    }
    if greek_df is None or greek_df.empty:
        return out
    df = greek_df
    if "call_delta" in df.columns:
        out["net_call_delta_bn"] = float(pd.to_numeric(df["call_delta"], errors="coerce").fillna(0.0).sum())
    if "put_delta" in df.columns:
        out["net_put_delta_bn"] = float(pd.to_numeric(df["put_delta"], errors="coerce").fillna(0.0).sum())
    if "call_gex" in df.columns:
        out["net_call_gex_bn"] = float(pd.to_numeric(df["call_gex"], errors="coerce").fillna(0.0).sum())
    if "put_gex" in df.columns:
        out["net_put_gex_bn"] = float(pd.to_numeric(df["put_gex"], errors="coerce").fillna(0.0).sum())
    return out


def _snapshot_row(ticker: str, ts: str | None, history: list[dict]) -> dict:
    if not history:
        return {}
    if ts:
        for row in history:
            if row.get("ts") == ts:
                return row
        loaded = load_snapshot_at_ts(ticker, ts)
        if loaded:
            return loaded
    return history[-1]


def build_periscope_context(
    *,
    ticker: str = PRIMARY_TICKER,
    selected_ts: str | None = None,
    exposure: str = "gamma",
    uw_entry: dict | None = None,
    history: list[dict] | None = None,
    price_points: list[dict] | None = None,
    previous_snapshot: dict | None = None,
) -> dict[str, Any]:
    """Assemble template/API payload for the Periscope exposure view."""
    exposure = exposure.lower() if exposure.lower() in EXPOSURE_TYPES else "gamma"
    history = history or build_history(ticker)
    timestamps = list_timestamps(ticker)
    selected = _snapshot_row(ticker, selected_ts, history)

    spot = safe_float(selected.get("spot"), 0.0)
    if uw_entry and uw_entry.get("spot"):
        spot = safe_float(uw_entry["spot"], spot)

    strike_series = selected.get("strike")
    if isinstance(strike_series, pd.Series):
        gex_series = strike_series
    elif strike_series is not None:
        gex_series = pd.Series(strike_series, dtype=float)
    else:
        gex_series = pd.Series(dtype=float)

    greek_df = None
    if uw_entry and uw_entry.get("agg") is not None:
        greek_df = uw_entry["agg"].gex_by_strike.attrs.get("greek_exposure_df")
        if greek_df is None and uw_entry["agg"].surface_data is not None and not uw_entry["agg"].surface_data.empty:
            greek_df = uw_entry["agg"].surface_data

    current_exposure = _exposure_series(greek_df, gex_series, exposure)
    previous_exposure = pd.Series(dtype=float)
    if previous_snapshot:
        prev_strike = previous_snapshot.get("strike")
        if isinstance(prev_strike, pd.Series):
            previous_exposure = _exposure_series(None, prev_strike, exposure)

    gamma_flip = selected.get("gamma_flip")
    if gamma_flip is None and not gex_series.empty:
        cumulative = selected.get("cumulative")
        if isinstance(cumulative, pd.Series) and not cumulative.empty:
            gamma_flip = estimate_gamma_flip(cumulative)

    regime = selected.get("regime", "N/A")
    total_gex = safe_float(selected.get("total_gex"), 0.0)
    if uw_entry and uw_entry.get("agg") is not None:
        total_gex = safe_float(uw_entry["agg"].total_gex_bn, total_gex)
        regime = "LONG gamma" if total_gex >= 0 else "SHORT gamma"

    replay_index = max(0, timestamps.index(selected["ts"])) if selected.get("ts") in timestamps else max(0, len(timestamps) - 1)

    return {
        "ticker": ticker,
        "exposure": exposure,
        "spot": spot or None,
        "regime": regime,
        "total_gex": total_gex,
        "gamma_flip": gamma_flip,
        "call_wall": selected.get("call_wall"),
        "put_wall": selected.get("put_wall"),
        "selected_ts": selected.get("ts"),
        "selected_label": selected.get("ts_label", "Latest"),
        "timestamps": timestamps,
        "replay_index": replay_index,
        "prev_ts": timestamps[replay_index - 1] if replay_index > 0 else None,
        "next_ts": timestamps[replay_index + 1] if replay_index + 1 < len(timestamps) else None,
        "has_history": bool(history),
        "exposure_series": current_exposure,
        "previous_exposure": previous_exposure,
        "exposure_window": _strike_window(current_exposure, spot or 0.0, window_pct=0.035),
        "exposure_extended": _strike_window(current_exposure, spot or 0.0, window_pct=0.08),
        "mm_positions": _mm_positions(greek_df if isinstance(greek_df, pd.DataFrame) else None),
        "price_points": price_points or [],
        "history": history,
        "selected": selected,
        "uw_fetched_at": uw_entry.get("fetched_at") if uw_entry else None,
        "vanna_charm_available": bool(
            isinstance(greek_df, pd.DataFrame)
            and {"call_vanna", "put_vanna", "call_charm", "put_charm"}.intersection(greek_df.columns)
        ),
    }
