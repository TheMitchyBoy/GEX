"""
Build a comprehensive Unusual Whales context bundle for AI prediction.

Aggregates every UW data point available at snapshot time — greek exposure by
strike, spot exposures, expiration term structure, intraday minute series,
daily greek history, extended scalars, and optional KNN forecast — into a
token-budgeted JSON structure suitable for LLM consumption.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from gex_core.ai_analyst import _clean_strike_series
from gex_core.extended_features import merge_extended_features
from gex_core.features import resolve_gamma_flip, safe_float
from gex_core.market_features import attach_market_features, fetch_cross_asset_returns, fetch_vol_regime
from gex_core.pipeline import GexAggregates
from gex_core.spot_exposure import spot_exposure_mm_positions, spot_exposure_net_series

logger = logging.getLogger(__name__)


def _max_strikes() -> int:
    try:
        return max(10, int(os.environ.get("GEX_AI_CONTEXT_MAX_STRIKES", "48")))
    except ValueError:
        return 48


def _intraday_tail() -> int:
    try:
        return max(5, int(os.environ.get("GEX_AI_INTRADAY_TAIL", "30")))
    except ValueError:
        return 30


def _history_tail() -> int:
    try:
        return max(5, int(os.environ.get("GEX_AI_HISTORY_TAIL", "20")))
    except ValueError:
        return 20


def _round(value: Any, digits: int = 4) -> Any:
    if value is None:
        return None
    try:
        f = float(value)
        if not np.isfinite(f):
            return None
        return round(f, digits)
    except (TypeError, ValueError):
        return value


def _strike_window_df(
    df: pd.DataFrame,
    spot: float,
    *,
    strike_col: str = "strike",
    max_rows: int | None = None,
    sort_by_abs: str | None = None,
) -> pd.DataFrame:
    """Return strikes near spot, optionally sorted by absolute exposure."""
    if df is None or df.empty or strike_col not in df.columns:
        return pd.DataFrame()
    max_rows = max_rows or _max_strikes()
    frame = df.copy()
    frame[strike_col] = pd.to_numeric(frame[strike_col], errors="coerce")
    frame = frame.dropna(subset=[strike_col])
    if frame.empty:
        return frame
    if spot > 0:
        frame["_dist"] = (frame[strike_col] - spot).abs()
        frame = frame.sort_values("_dist").head(max_rows * 2)
    if sort_by_abs and sort_by_abs in frame.columns:
        frame["_abs"] = pd.to_numeric(frame[sort_by_abs], errors="coerce").abs()
        frame = frame.sort_values("_abs", ascending=False).head(max_rows)
    else:
        frame = frame.head(max_rows)
    return frame.drop(columns=[c for c in ("_dist", "_abs") if c in frame.columns], errors="ignore")


def _df_to_records(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    round_digits: int = 4,
) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    cols = columns or [c for c in df.columns if c not in ("date", "ticker", "time")]
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        record: dict[str, Any] = {}
        for col in cols:
            if col not in df.columns:
                continue
            val = row[col]
            if isinstance(val, (int, float, np.floating, np.integer)):
                record[col] = _round(val, round_digits)
            elif pd.isna(val):
                continue
            else:
                record[col] = str(val)
        if record:
            rows.append(record)
    return rows


def _series_to_records(series: pd.Series, *, value_name: str = "value") -> list[dict[str, Any]]:
    if series is None or series.empty:
        return []
    out: list[dict[str, Any]] = []
    for idx, val in series.items():
        out.append({"key": _round(idx, 2), value_name: _round(val, 4)})
    return out


def _summarize_gex_profile(
    gex_by_strike: pd.Series,
    spot: float,
    *,
    greek_exposure_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    gex_by_strike = _clean_strike_series(gex_by_strike)
    if gex_by_strike.empty:
        return {}
    total = float(gex_by_strike.sum())
    abs_vals = gex_by_strike.abs()
    top5_share = float(abs_vals.nlargest(5).sum() / abs_vals.sum()) if abs_vals.sum() else 0.0
    pos = float(gex_by_strike[gex_by_strike > 0].sum())
    neg = float(gex_by_strike[gex_by_strike < 0].sum())
    call_wall = float(gex_by_strike.idxmax())
    put_wall = float(gex_by_strike.idxmin())
    dominant = float(abs_vals.idxmax())
    cumulative = gex_by_strike.cumsum()
    flip = resolve_gamma_flip(
        spot=spot,
        gex_by_strike=gex_by_strike,
        cumulative_gex=cumulative,
        greek_exposure_df=greek_exposure_df,
    )
    return {
        "total_gex_bn": _round(total, 3),
        "positive_gex_bn": _round(pos, 3),
        "negative_gex_bn": _round(neg, 3),
        "top5_concentration_pct": _round(top5_share * 100, 1),
        "call_wall": _round(call_wall, 0),
        "put_wall": _round(put_wall, 0),
        "dominant_strike": _round(dominant, 0),
        "gamma_flip": _round(flip, 0) if flip else None,
        "spot_vs_flip_pct": _round((spot - flip) / spot * 100, 2) if flip and spot > 0 else None,
        "strike_count": len(gex_by_strike),
    }


def _summarize_intraday(minute_df: pd.DataFrame) -> dict[str, Any]:
    if minute_df is None or minute_df.empty:
        return {}
    tail = minute_df.tail(_intraday_tail())
    cols = [
        c
        for c in (
            "time",
            "price",
            "gamma_per_one_percent_move_oi",
            "call_gamma_oi",
            "put_gamma_oi",
            "charm_per_one_percent_move_oi",
            "vanna_per_one_percent_move_oi",
        )
        if c in tail.columns
    ]
    records = _df_to_records(tail, cols)
    summary: dict[str, Any] = {"minute_bars": records, "bar_count": len(records)}
    if "gamma_per_one_percent_move_oi" in minute_df.columns:
        gamma = pd.to_numeric(minute_df["gamma_per_one_percent_move_oi"], errors="coerce").dropna()
        if len(gamma) >= 2:
            summary["gamma_trend_bn"] = _round(float(gamma.iloc[-1] - gamma.iloc[0]) / 1e9, 4)
            summary["latest_gamma_bn"] = _round(float(gamma.iloc[-1]) / 1e9, 4)
    if "price" in minute_df.columns:
        prices = pd.to_numeric(minute_df["price"], errors="coerce").dropna()
        if len(prices) >= 2:
            summary["intraday_return_pct"] = _round(float((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100), 3)
    return summary


def _summarize_snapshot_history(history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not history:
        return []
    tail = history[-_history_tail():]
    rows: list[dict[str, Any]] = []
    for row in tail:
        rows.append(
            {
                "ts": row.get("ts_label") or row.get("ts"),
                "spot": _round(row.get("spot"), 1),
                "total_gex_bn": _round(row.get("total_gex"), 3),
                "regime": row.get("regime"),
                "gamma_flip": _round(row.get("gamma_flip"), 0),
                "call_wall": _round(row.get("call_wall"), 0),
                "put_wall": _round(row.get("put_wall"), 0),
            }
        )
    return rows


def build_uw_context_bundle(
    *,
    ticker: str,
    spot: float,
    agg: GexAggregates,
    gamma_flip: float | None = None,
    spot_gamma_bn: float | None = None,
    history: list[dict] | None = None,
    knn_prediction: dict[str, Any] | None = None,
    api_key: str | None = None,
    fetch_extras: bool = True,
) -> dict[str, Any]:
    """
    Assemble all Unusual Whales data points into a structured context bundle.

    Parameters
    ----------
    ticker, spot, agg
        Core UW fetch result from ``fetch_uw_gex``.
    gamma_flip, spot_gamma_bn
        Optional precomputed levels from spot-exposures intraday.
    history
        Recent export snapshots for momentum / KNN context.
    knn_prediction
        Output of ``predict_next_snapshot`` when available.
    api_key
        UW API key for optional extra fetches (intraday, daily history).
    fetch_extras
        When False, skip network calls and use only data already in ``agg``.
    """
    greek_df = agg.gex_by_strike.attrs.get("greek_exposure_df")
    spot_df = agg.gex_by_strike.attrs.get("spot_exposures_df")
    market_date = None
    if isinstance(greek_df, pd.DataFrame) and greek_df.attrs.get("market_date"):
        market_date = greek_df.attrs["market_date"]
    elif isinstance(spot_df, pd.DataFrame) and spot_df.attrs.get("market_date"):
        market_date = spot_df.attrs["market_date"]

    gex_summary = _summarize_gex_profile(
        agg.gex_by_strike,
        spot,
        greek_exposure_df=greek_df if isinstance(greek_df, pd.DataFrame) else None,
    )
    if gamma_flip is None:
        gamma_flip = gex_summary.get("gamma_flip")

    # Strike tables — near-spot window for token efficiency
    greek_cols = [
        "strike",
        "call_gex",
        "put_gex",
        "net_gex",
        "call_delta",
        "put_delta",
        "call_charm",
        "put_charm",
        "call_vanna",
        "put_vanna",
    ]
    greek_strikes: list[dict[str, Any]] = []
    if isinstance(greek_df, pd.DataFrame) and not greek_df.empty:
        window = _strike_window_df(greek_df, spot, sort_by_abs="net_gex")
        greek_strikes = _df_to_records(window, [c for c in greek_cols if c in window.columns])

    spot_cols = [
        "strike",
        "price",
        "net_gamma_oi_bn",
        "call_gamma_oi",
        "put_gamma_oi",
        "call_gamma_vol",
        "put_gamma_vol",
        "call_gamma_bid",
        "put_gamma_bid",
    ]
    spot_strikes: list[dict[str, Any]] = []
    spot_gamma_series = pd.Series(dtype=float)
    mm_positions: dict[str, float] = {}
    if isinstance(spot_df, pd.DataFrame) and not spot_df.empty:
        window = _strike_window_df(spot_df, spot, sort_by_abs="net_gamma_oi_bn" if "net_gamma_oi_bn" in spot_df.columns else None)
        spot_strikes = _df_to_records(window, [c for c in spot_cols if c in window.columns])
        spot_gamma_series = spot_exposure_net_series(spot_df, "gamma")
        mm_positions = spot_exposure_mm_positions(spot_df)

    # Expiration term structure
    expiration_records = _series_to_records(agg.gex_by_expiration, value_name="gex_bn")

    # Extended scalars (charm, vanna, flow, events, vol)
    metrics: dict[str, Any] = {"spot": spot, "ts": market_date}
    merge_extended_features(
        metrics,
        greek_df=greek_df if isinstance(greek_df, pd.DataFrame) else None,
        spot_exposures_df=spot_df if isinstance(spot_df, pd.DataFrame) else None,
        market_date=market_date,
        vol_regime=fetch_vol_regime() if fetch_extras else None,
        cross_asset=fetch_cross_asset_returns() if fetch_extras else None,
    )

    # Optional extra UW fetches
    intraday_summary: dict[str, Any] = {}
    daily_greek_history: list[dict[str, Any]] = []
    if fetch_extras and api_key:
        try:
            from gex_core.uw_loader import fetch_uw_greek_exposure_history, fetch_uw_spot_exposures_intraday
            from gex_core.market_time import market_today

            minute_df = fetch_uw_spot_exposures_intraday(ticker, api_key=api_key, date=market_date or market_today())
            intraday_summary = _summarize_intraday(minute_df)

            hist_df = fetch_uw_greek_exposure_history(ticker, api_key=api_key)
            if not hist_df.empty:
                tail = hist_df.tail(_history_tail())
                daily_greek_history = _df_to_records(tail)
        except Exception:
            pass

    # Surface data (charm/vanna by strike when available)
    surface_records: list[dict[str, Any]] = []
    if not agg.surface_data.empty:
        surface = agg.surface_data.copy()
        if "net_gex" not in surface.columns and "GEX" in surface.columns:
            surface = surface.rename(columns={"GEX": "net_gex"})
        surface_window = _strike_window_df(surface, spot, sort_by_abs="net_gex")
        surface_cols = [c for c in ("strike", "net_gex", "charm", "vanna") if c in surface_window.columns]
        surface_records = _df_to_records(surface_window, surface_cols)

    snapshot_history = _summarize_snapshot_history(history)
    if history:
        enriched = [dict(h) for h in history[-_history_tail():]]
        attach_market_features(enriched)

    bundle: dict[str, Any] = {
        "ticker": ticker.upper(),
        "market_date": market_date,
        "spot": _round(spot, 2),
        "data_source": "unusual_whales",
        "summary": {
            **gex_summary,
            "spot_gamma_intraday_bn": _round(spot_gamma_bn, 4) if spot_gamma_bn is not None else None,
            "gamma_flip": _round(gamma_flip, 0) if gamma_flip else gex_summary.get("gamma_flip"),
            "total_gex_bn": _round(agg.total_gex_bn, 3),
        },
        "greek_exposure_by_strike": greek_strikes,
        "spot_exposures_by_strike": spot_strikes,
        "spot_exposure_mm_positions": {k: _round(v, 4) for k, v in mm_positions.items()},
        "spot_gamma_profile": _series_to_records(spot_gamma_series, value_name="gamma_bn"),
        "gex_by_expiration": expiration_records,
        "surface_charm_vanna": surface_records,
        "extended_features": {k: _round(v, 4) for k, v in metrics.items() if k not in ("spot", "ts", "extended_features")},
        "intraday": intraday_summary,
        "daily_greek_history": daily_greek_history,
        "snapshot_history": snapshot_history,
    }

    if knn_prediction:
        bundle["knn_forecast"] = {
            k: _round(v, 4) if isinstance(v, (int, float)) else v
            for k, v in knn_prediction.items()
            if k not in {"predicted_strike", "knn_strike", "flow_strike"}
        }

    try:
        from gex_core.trading.journal import get_trade_memory_for_ai

        bundle["trade_memory"] = get_trade_memory_for_ai(ticker)
    except Exception:
        pass

    try:
        from gex_core.daily_learning import attach_learning_to_bundle, daily_learning_enabled

        if daily_learning_enabled():
            attach_learning_to_bundle(bundle, ticker)
    except Exception:
        logger.debug("Daily learning attachment skipped", exc_info=True)

    return bundle


def build_context_bundle_from_snapshot(
    *,
    ticker: str,
    snapshot: dict[str, Any],
    history: list[dict] | None = None,
    knn_prediction: dict[str, Any] | None = None,
    api_key: str | None = None,
    fetch_extras: bool = True,
) -> dict[str, Any]:
    """Build an LLM context bundle from a periscope/export snapshot (no live agg)."""
    spot = safe_float(snapshot.get("spot"), 0.0)
    strike = snapshot.get("strike")
    if not isinstance(strike, pd.Series):
        strike = pd.Series(dtype=float)
    cumulative = snapshot.get("cumulative")
    if not isinstance(cumulative, pd.Series):
        cumulative = strike.cumsum() if not strike.empty else pd.Series(dtype=float)

    greek_df = snapshot.get("greek_exposure_df")
    spot_df = snapshot.get("spot_exposures_df")
    market_date = snapshot.get("market_date") or snapshot.get("ts", "")[:10]

    class _SnapshotAgg:
        def __init__(self) -> None:
            self.gex_by_strike = strike.rename("gex_bn_per_pct") if not strike.empty else pd.Series(dtype=float)
            self.gex_by_strike.attrs = {
                "greek_exposure_df": greek_df if isinstance(greek_df, pd.DataFrame) else None,
                "spot_exposures_df": spot_df if isinstance(spot_df, pd.DataFrame) else None,
            }
            self.cumulative_gex = cumulative
            self.gex_by_expiration = snapshot.get("expiration") or pd.Series(dtype=float)
            self.surface_data = snapshot.get("surface_df") if isinstance(snapshot.get("surface_df"), pd.DataFrame) else pd.DataFrame()
            self.total_gex_bn = safe_float(snapshot.get("total_gex"), float(strike.sum()) if not strike.empty else 0.0)

    agg = _SnapshotAgg()
    return build_uw_context_bundle(
        ticker=ticker,
        spot=spot,
        agg=agg,
        gamma_flip=snapshot.get("gamma_flip"),
        spot_gamma_bn=None,
        history=history,
        knn_prediction=knn_prediction,
        api_key=api_key,
        fetch_extras=fetch_extras and bool(api_key),
    )


def try_build_uw_bundle_from_entry(
    *,
    ticker: str,
    spot: float,
    uw_entry: dict[str, Any] | None,
    gamma_flip: float | None = None,
    history: list[dict[str, Any]] | None = None,
    knn_prediction: dict[str, Any] | None = None,
    api_key: str | None = None,
    fetch_extras: bool = False,
) -> dict[str, Any] | None:
    """Build a UW context bundle when live aggregation is available."""
    if not uw_entry or uw_entry.get("agg") is None or spot <= 0:
        return None
    return build_uw_context_bundle(
        ticker=ticker,
        spot=float(spot),
        agg=uw_entry["agg"],
        gamma_flip=gamma_flip,
        spot_gamma_bn=uw_entry.get("spot_gamma_bn"),
        history=history,
        knn_prediction=knn_prediction,
        api_key=api_key,
        fetch_extras=fetch_extras,
    )


def bundle_to_prompt_json(bundle: dict[str, Any]) -> str:
    """Serialize the context bundle for LLM consumption."""
    return json.dumps(bundle, indent=None, default=str)


def bundle_token_estimate(bundle: dict[str, Any]) -> int:
    """Rough character-based token estimate (~4 chars per token)."""
    return len(bundle_to_prompt_json(bundle)) // 4
