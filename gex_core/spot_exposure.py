"""UW Periscope spot-exposure strike profiles (spot-exposures/strike endpoint)."""

from __future__ import annotations

import pandas as pd

from gex_core.uw_loader import normalize_net_exposure

# spot-exposures/strike values are raw dollars per 1% move (or greek unit).
RAW_SCALE = 1e9

EXPOSURE_OI_COLUMNS: dict[str, tuple[str, str]] = {
    "gamma": ("call_gamma_oi", "put_gamma_oi"),
    "vanna": ("call_vanna_oi", "put_vanna_oi"),
    "charm": ("call_charm_oi", "put_charm_oi"),
}


def spot_exposure_net_series(
    spot_df: pd.DataFrame | None,
    exposure: str = "gamma",
) -> pd.Series:
    """Per-strike net exposure in Bn$ from UW spot-exposures/strike (OI-based)."""
    if spot_df is None or spot_df.empty or "strike" not in spot_df.columns:
        return pd.Series(dtype=float)

    exposure = exposure.lower()
    if exposure == "gamma" and "net_gamma_oi_bn" in spot_df.columns:
        series = pd.Series(
            pd.to_numeric(spot_df["net_gamma_oi_bn"], errors="coerce").values,
            index=pd.to_numeric(spot_df["strike"], errors="coerce"),
            dtype=float,
        )
        return series.dropna().sort_index()

    call_col, put_col = EXPOSURE_OI_COLUMNS.get(exposure, EXPOSURE_OI_COLUMNS["gamma"])
    if call_col not in spot_df.columns or put_col not in spot_df.columns:
        return pd.Series(dtype=float)

    net = normalize_net_exposure(spot_df, call_col=call_col, put_col=put_col)
    series = pd.Series(
        (net / RAW_SCALE).values,
        index=pd.to_numeric(spot_df["strike"], errors="coerce"),
        dtype=float,
    )
    return series.dropna().sort_index()


def spot_exposure_mm_positions(spot_df: pd.DataFrame | None) -> dict[str, float]:
    """Net dealer call/put delta and gamma totals from spot-exposures/strike."""
    out = {
        "net_call_delta_bn": 0.0,
        "net_put_delta_bn": 0.0,
        "net_call_gex_bn": 0.0,
        "net_put_gex_bn": 0.0,
    }
    if spot_df is None or spot_df.empty:
        return out

    mapping = {
        "net_call_delta_bn": "call_delta_oi",
        "net_put_delta_bn": "put_delta_oi",
        "net_call_gex_bn": "call_gamma_oi",
        "net_put_gex_bn": "put_gamma_oi",
    }
    for key, col in mapping.items():
        if col in spot_df.columns:
            out[key] = float(pd.to_numeric(spot_df[col], errors="coerce").fillna(0.0).sum()) / RAW_SCALE
    return out


def spot_exposure_gamma_flip(
    strike_series: pd.Series,
    spot: float | None = None,
) -> float | None:
    """Gamma flip from a spot-exposure strike profile (ATM-local window)."""
    if strike_series is None or strike_series.empty:
        return None
    from gex_core.features import resolve_gamma_flip

    return resolve_gamma_flip(
        spot=spot,
        gex_by_strike=strike_series,
        cumulative_gex=strike_series.cumsum(),
    )


def spot_exposure_surface_df(
    spot_df: pd.DataFrame | None,
    exposure: str = "gamma",
) -> pd.DataFrame:
    """Strike table in pipeline units (Bn$ / %) for charts and CSV export."""
    if spot_df is None or spot_df.empty or "strike" not in spot_df.columns:
        return pd.DataFrame()

    exposure = exposure.lower()
    call_col, put_col = EXPOSURE_OI_COLUMNS.get(exposure, EXPOSURE_OI_COLUMNS["gamma"])
    if call_col not in spot_df.columns or put_col not in spot_df.columns:
        return pd.DataFrame()

    frame = spot_df.copy()
    frame["strike"] = pd.to_numeric(frame["strike"], errors="coerce")
    calls = pd.to_numeric(frame[call_col], errors="coerce").fillna(0.0) / RAW_SCALE
    puts = pd.to_numeric(frame[put_col], errors="coerce").fillna(0.0) / RAW_SCALE
    net = normalize_net_exposure(frame, call_col=call_col, put_col=put_col) / RAW_SCALE
    out = pd.DataFrame(
        {
            "strike": frame["strike"].values,
            "call_gex": calls.values,
            "put_gex": puts.values,
            "net_gex": net.values,
            "GEX": net.values,
        }
    )
    return out.dropna(subset=["strike"]).sort_values("strike").reset_index(drop=True)


def spot_exposure_walls(strike_series: pd.Series) -> tuple[float | None, float | None]:
    """Call/put walls as max/min net gamma strikes."""
    if strike_series is None or strike_series.empty:
        return None, None
    return float(strike_series.idxmax()), float(strike_series.idxmin())
