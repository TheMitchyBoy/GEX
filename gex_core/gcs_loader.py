"""
GCS options-chain loader for the GEX analytics pipeline.

Downloads a CSV from Google Cloud Storage (private or public), auto-detects
the column schema, normalises it to the format expected by gex_core.pipeline,
and computes Black-Scholes gamma for files that carry IV but not greeks.

Authentication
--------------
Relies on Google Application Default Credentials (ADC).  Before running,
authenticate with one of:

  1. Service account key file:
       export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
       python main.py --ticker SPX --gcs-source gs://options_data_gex_analysis/...

  2. Interactive user credentials (local dev):
       gcloud auth application-default login

  3. Workload Identity / metadata server (GCP VMs, Cloud Run, GKE):
       No action needed – credentials are injected automatically.

Supported URL formats
---------------------
  gs://options_data_gex_analysis/option-trades-2026-06-03.csv
  https://storage.googleapis.com/options_data_gex_analysis/option-trades-2026-06-03.csv

Supported CSV schemas
---------------------
Chain format (greeks / OI present):
  option | strike | expiration | type | gamma | open_interest | iv | bid | ask | volume
  — or any column-name variations listed in _COLUMN_ALIASES below.

Trade / flow format (aggregated into a pseudo-chain):
  timestamp | underlying | expiration | strike | option_type | trade_price | trade_size | iv
  — gamma is computed via Black-Scholes if IV + expiration + strike are available.

Example (Python)
----------------
    from gex_core.gcs_loader import load_gcs_options
    spot, options_df = load_gcs_options(
        "gs://options_data_gex_analysis/option-trades-2026-06-03.csv",
        ticker="SPX",
    )
    # options_df is ready to pass directly to attach_signed_gex() / clean_option_data()
"""
from __future__ import annotations

import io
import logging
import re
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GCS URL parsing
# ---------------------------------------------------------------------------

_GCS_RE = re.compile(
    r"^(?:gs://|https://storage\.googleapis\.com/)(?P<bucket>[^/]+)/(?P<object>.+)$"
)


def _parse_gcs_url(url: str) -> tuple[str, str]:
    m = _GCS_RE.match(url.strip())
    if not m:
        raise ValueError(
            f"Not a recognised GCS URL: {url!r}\n"
            "Expected format: gs://bucket/object  or  "
            "https://storage.googleapis.com/bucket/object"
        )
    return m.group("bucket"), m.group("object")


def _fetch_bytes(url: str) -> bytes:
    """Download raw bytes from GCS using Application Default Credentials."""
    try:
        from google.cloud import storage  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "google-cloud-storage is required to load GCS data.\n"
            "Install it with:  pip install google-cloud-storage\n"
            "Then authenticate:  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json"
        ) from exc

    bucket_name, object_name = _parse_gcs_url(url)
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(object_name)
    logger.info("Downloading gs://%s/%s …", bucket_name, object_name)
    return blob.download_as_bytes()


# ---------------------------------------------------------------------------
# Column alias table
# ---------------------------------------------------------------------------
# Maps lowercased / underscore-normalised column names → canonical pipeline name.
# Add new aliases here when a new data vendor is encountered.

_COLUMN_ALIASES: dict[str, str] = {
    # ── OCC option symbol ────────────────────────────────────────────────────
    "option": "option",
    "symbol": "option",
    "option_symbol": "option",
    "contract": "option",
    "contractsymbol": "option",
    "contract_symbol": "option",
    "osisymbol": "option",
    "osi_symbol": "option",

    # ── Underlying ticker ───────────────────────────────────────────────────
    "underlying": "underlying",
    "underlying_symbol": "underlying",
    "root": "underlying",
    "ticker": "underlying",
    "security": "underlying",
    "underlier": "underlying",
    "underlying_ticker": "underlying",

    # ── Underlying spot price ───────────────────────────────────────────────
    "underlying_price": "underlying_price",
    "spot": "underlying_price",
    "spot_price": "underlying_price",
    "index_price": "underlying_price",
    "underprice": "underlying_price",
    "underlying_last": "underlying_price",
    "underlying_close": "underlying_price",
    "last_price_underlying": "underlying_price",

    # ── Strike price ────────────────────────────────────────────────────────
    "strike": "strike",
    "strike_price": "strike",
    "strikeprice": "strike",
    "exercise_price": "strike",
    "k": "strike",

    # ── Expiration date ─────────────────────────────────────────────────────
    "expiration": "expiration",
    "expiration_date": "expiration",
    "expiry": "expiration",
    "expiry_date": "expiration",
    "maturity": "expiration",
    "maturity_date": "expiration",
    "maturitydate": "expiration",
    "exp_date": "expiration",
    "expdt": "expiration",

    # ── Option type (C / P) ─────────────────────────────────────────────────
    "type": "type",
    "option_type": "type",
    "put_call": "type",
    "cp": "type",
    "callput": "type",
    "call_put": "type",
    "flag": "type",
    "right": "type",
    "side": "type",  # NB: may conflict with buy/sell; handled below

    # ── Greeks ──────────────────────────────────────────────────────────────
    "gamma": "gamma",
    "charm": "charm",
    "delta_decay": "charm",
    "delta": "delta",
    "theta": "theta",
    "vega": "vega",
    "vanna": "vanna",

    # ── Open interest ────────────────────────────────────────────────────────
    "open_interest": "open_interest",
    "oi": "open_interest",
    "openinterest": "open_interest",
    "open_int": "open_interest",

    # ── Implied volatility ───────────────────────────────────────────────────
    "iv": "iv",
    "implied_volatility": "iv",
    "impliedvol": "iv",
    "impliedvolatility": "iv",
    "ivol": "iv",
    "imp_vol": "iv",
    "mark_iv": "iv",

    # ── Market quotes ────────────────────────────────────────────────────────
    "bid": "bid",
    "ask": "ask",
    "offer": "ask",
    "last": "last_price",
    "last_price": "last_price",
    "mark": "last_price",
    "mid": "last_price",

    # ── Volume ───────────────────────────────────────────────────────────────
    "volume": "volume",
    "vol": "volume",          # careful: may also mean volatility
    "trade_volume": "volume",
    "daily_volume": "volume",

    # ── Trade-level fields ────────────────────────────────────────────────────
    "trade_price": "trade_price",
    "price": "trade_price",
    "premium": "trade_price",
    "fill_price": "trade_price",
    "execution_price": "trade_price",

    "trade_size": "trade_size",
    "filled_qty": "trade_size",
    "qty": "trade_size",
    "quantity": "trade_size",
    "size": "trade_size",

    # ── Timestamp / date ──────────────────────────────────────────────────────
    "timestamp": "timestamp",
    "trade_timestamp": "timestamp",
    "datetime": "timestamp",
    "time": "timestamp",
    "trade_time": "timestamp",
    "date": "date",
    "trade_date": "date",
}


def _normalise_col_key(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names; skip if canonical is already taken."""
    taken: set[str] = set()
    rename: dict[str, str] = {}
    for col in df.columns:
        canonical = _COLUMN_ALIASES.get(_normalise_col_key(col))
        if canonical and canonical not in taken:
            rename[col] = canonical
            taken.add(canonical)
    return df.rename(columns=rename)


# ---------------------------------------------------------------------------
# Option-type normalisation
# ---------------------------------------------------------------------------

def _normalise_type(series: pd.Series) -> pd.Series:
    """Coerce option-type values to 'C' or 'P'."""
    s = series.astype(str).str.strip().str.upper()
    return s.map(
        lambda x: "C" if x in {"C", "CALL", "CALLS", "1"}
        else ("P" if x in {"P", "PUT", "PUTS", "0", "-1"} else np.nan)
    )


# ---------------------------------------------------------------------------
# OCC symbol builder
# ---------------------------------------------------------------------------

def _build_occ_symbol(expiration, option_type: str, strike: float, root: str = "SPX") -> str | None:
    """Build an OCC-format symbol from components.

    Example: SPX + 2026-06-06 + C + 5600.0  →  "SPX260606C05600000"
    """
    try:
        ts = pd.Timestamp(expiration)
        strike_int = int(round(float(strike) * 1000))
        return f"{root.upper()}{ts.strftime('%y%m%d')}{option_type.upper()[0]}{strike_int:08d}"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Black-Scholes gamma (numpy-only, no scipy)
# ---------------------------------------------------------------------------

def _bs_gamma_vec(
    S: float,
    K: np.ndarray,
    T: np.ndarray,
    sigma: np.ndarray,
    r: float = 0.0,
) -> np.ndarray:
    """Vectorised Black-Scholes gamma.  Returns 0 for degenerate inputs."""
    result = np.zeros(len(K), dtype=float)
    valid = (T > 1e-6) & (sigma > 1e-6) & (K > 0) & np.isfinite(K) & np.isfinite(sigma)
    if not valid.any():
        return result
    sv = np.sqrt(T[valid])
    d1 = (np.log(S / K[valid]) + (r + 0.5 * sigma[valid] ** 2) * T[valid]) / (sigma[valid] * sv)
    pdf_d1 = np.exp(-0.5 * d1 ** 2) / np.sqrt(2.0 * np.pi)
    result[valid] = pdf_d1 / (S * sigma[valid] * sv)
    return result


# ---------------------------------------------------------------------------
# Trade-level → chain aggregation
# ---------------------------------------------------------------------------

def _aggregate_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate individual trade rows into a pseudo option-chain.

    Groups by (expiration, strike, type) and computes:
      - open_interest  = sum of trade_size (contract quantity proxy)
      - trade_price    = volume-weighted average price
      - iv             = mean IV (if present)
    """
    group_keys = [c for c in ("expiration", "strike", "type") if c in df.columns]
    if not group_keys:
        raise ValueError(
            "Trade-level CSV must contain at least: expiration, strike, and type (option_type/put_call) columns."
        )

    agg_spec: dict[str, Any] = {}
    if "trade_size" in df.columns:
        agg_spec["trade_size"] = "sum"
    if "trade_price" in df.columns:
        if "trade_size" in df.columns:
            # volume-weighted average price
            df = df.copy()
            df["_wprice"] = (
                pd.to_numeric(df["trade_price"], errors="coerce").fillna(0)
                * pd.to_numeric(df["trade_size"], errors="coerce").fillna(1)
            )
            agg_spec["_wprice"] = "sum"
        else:
            agg_spec["trade_price"] = "mean"
    if "iv" in df.columns:
        agg_spec["iv"] = "mean"

    chain = df.groupby(group_keys, observed=True).agg(agg_spec).reset_index()

    # Rename aggregated columns to pipeline names
    if "trade_size" in chain.columns:
        chain = chain.rename(columns={"trade_size": "open_interest"})
    if "_wprice" in chain.columns:
        size_sum = df.groupby(group_keys, observed=True)["trade_size"].sum().reset_index()
        chain["trade_price"] = chain["_wprice"] / size_sum["trade_size"].replace(0, np.nan)
        chain = chain.drop(columns=["_wprice"])
    if "trade_price" in chain.columns:
        chain = chain.rename(columns={"trade_price": "bid"})
        chain["ask"] = chain["bid"] * 1.01  # 1% spread placeholder

    return chain


# ---------------------------------------------------------------------------
# Column coercion + gamma derivation
# ---------------------------------------------------------------------------

def _coerce_and_compute(
    df: pd.DataFrame,
    spot: float,
    root: str = "SPX",
    risk_free_rate: float = 0.05,
) -> pd.DataFrame:
    """
    Cast types, normalise option type, and compute gamma if absent.

    After this call the DataFrame will have:
      expiration (datetime64), strike (float64), type ('C'/'P'),
      gamma (float64), open_interest (float64), iv, bid, ask, volume
    """
    df = df.copy()

    # Expiration → datetime
    if "expiration" in df.columns:
        df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce")

    # Strike → float
    if "strike" in df.columns:
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")

    # Option type → C / P
    if "type" in df.columns:
        df["type"] = _normalise_type(df["type"])

    # IV → float (percentage columns → decimal if > 5)
    if "iv" in df.columns:
        df["iv"] = pd.to_numeric(df["iv"], errors="coerce")
        high_iv = df["iv"] > 5
        if high_iv.sum() > len(df) * 0.5:
            df.loc[high_iv, "iv"] = df.loc[high_iv, "iv"] / 100.0

    # Open interest → float
    if "open_interest" in df.columns:
        df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0.0)
    else:
        df["open_interest"] = 1.0

    # Gamma: use CSV value if present, otherwise compute via Black-Scholes
    if "gamma" in df.columns:
        df["gamma"] = pd.to_numeric(df["gamma"], errors="coerce")
    else:
        df["gamma"] = np.nan

    missing_gamma = df["gamma"].isna() | (df["gamma"] <= 0)
    if missing_gamma.any() and "iv" in df.columns and "expiration" in df.columns:
        today = pd.Timestamp(date.today())
        T = ((df.loc[missing_gamma, "expiration"] - today).dt.days.clip(lower=0) / 365.0).to_numpy(float)
        K = df.loc[missing_gamma, "strike"].fillna(0).to_numpy(float)
        sigma = df.loc[missing_gamma, "iv"].fillna(0).to_numpy(float)
        computed = _bs_gamma_vec(spot, K, T, sigma, r=risk_free_rate)
        df.loc[missing_gamma, "gamma"] = computed
        n_computed = int(missing_gamma.sum())
        logger.info(
            "Black-Scholes gamma computed for %d/%d rows (no gamma column in source).",
            n_computed, len(df),
        )

    # Bid / ask / volume defaults
    for col, default in (("bid", 0.0), ("ask", 0.0), ("volume", 0.0)):
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    # OCC option symbol (required by parse_option_symbols in data_quality.py)
    if "option" not in df.columns and all(c in df.columns for c in ("expiration", "strike", "type")):
        df["option"] = [
            _build_occ_symbol(exp, t, k, root=root)
            for exp, t, k in zip(df["expiration"], df["type"], df["strike"])
        ]

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_gcs_options(
    url: str,
    ticker: str = "SPX",
    spot: float | None = None,
    risk_free_rate: float = 0.05,
) -> tuple[float, pd.DataFrame]:
    """
    Download an options CSV from GCS and return pipeline-ready data.

    Parameters
    ----------
    url : str
        GCS object URL in either of:
          gs://options_data_gex_analysis/option-trades-2026-06-03.csv
          https://storage.googleapis.com/options_data_gex_analysis/...
    ticker : str
        Underlying ticker to filter to (default "SPX").
        If the CSV covers multiple underlyings (SPX, SPY, QQQ …), only rows
        matching this ticker are kept.
    spot : float | None
        Spot price override.  If None the loader tries to read it from an
        "underlying_price" / "spot" column in the CSV.  Pass an explicit
        value when neither column exists:
            --spot 5800
    risk_free_rate : float
        Annual risk-free rate used in Black-Scholes gamma calculation
        when gamma is absent from the source file.

    Returns
    -------
    spot_price : float
    options_df : pd.DataFrame
        Columns: option, strike, expiration, type, gamma, open_interest,
                 iv, bid, ask, volume   (same as CBOE payload).

    Raises
    ------
    ImportError
        If google-cloud-storage is not installed.
    google.auth.exceptions.DefaultCredentialsError
        If no Application Default Credentials are configured.
    ValueError
        If spot price cannot be determined or ticker not found.
    """
    raw = _fetch_bytes(url)
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    logger.info("Loaded %d rows from %s", len(df), url)

    df = _normalise_columns(df)

    # ── Filter to requested ticker ──────────────────────────────────────────
    if "underlying" in df.columns:
        df["underlying"] = df["underlying"].astype(str).str.strip().str.upper()
        mask = df["underlying"].str.startswith(ticker.upper())
        if not mask.any():
            available = sorted(df["underlying"].dropna().unique())[:20]
            raise ValueError(
                f"No rows match ticker {ticker!r} in the CSV.\n"
                f"Available underlyings (first 20): {available}"
            )
        before = len(df)
        df = df.loc[mask].copy()
        logger.info("Filtered %d → %d rows for underlying=%s", before, len(df), ticker)

    # ── Resolve spot price ──────────────────────────────────────────────────
    if spot is None or spot <= 0:
        if "underlying_price" in df.columns:
            prices = pd.to_numeric(df["underlying_price"], errors="coerce").dropna()
            if not prices.empty:
                spot = float(prices.median())
                logger.info("Spot price from CSV: %.4f", spot)

    if spot is None or spot <= 0:
        raise ValueError(
            f"Could not determine spot price for {ticker} from the CSV.\n"
            "Add an 'underlying_price' / 'spot' column, or pass --spot=<price> on the CLI."
        )

    # ── Detect chain vs trade-level data ────────────────────────────────────
    has_chain_cols = "gamma" in df.columns or "open_interest" in df.columns
    has_trade_cols = "trade_price" in df.columns or "trade_size" in df.columns

    if has_trade_cols and not has_chain_cols:
        logger.info(
            "Detected trade-level CSV; aggregating to pseudo option-chain by (expiration, strike, type)."
        )
        df = _aggregate_trades(df)

    # ── Coerce types and compute missing gamma ──────────────────────────────
    df = _coerce_and_compute(df, spot=spot, root=ticker, risk_free_rate=risk_free_rate)

    # ── Drop rows that can't be used by the pipeline ─────────────────────────
    required = ["option", "strike", "expiration", "type", "gamma", "open_interest"]
    present_required = [c for c in required if c in df.columns]
    df = df.dropna(subset=present_required)

    if df.empty:
        raise ValueError(
            f"No usable rows remained after normalisation for ticker {ticker!r}.\n"
            "Check that the CSV contains at least: expiration, strike, type, and either "
            "gamma/open_interest (chain) or trade_size (trades)."
        )

    logger.info(
        "GCS load complete: %d option contracts for %s (spot=%.2f).",
        len(df), ticker, spot,
    )
    return float(spot), df.reset_index(drop=True)


def latest_gcs_url(
    bucket: str = "options_data_gex_analysis",
    prefix: str = "option-trades-",
    suffix: str = ".csv",
    as_of: str | None = None,
) -> str:
    """
    Return the gs:// URL for the most recent options file in a GCS bucket.

    Parameters
    ----------
    bucket : str
        GCS bucket name.
    prefix : str
        Object name prefix (e.g. "option-trades-").
    suffix : str
        Object name suffix (e.g. ".csv").
    as_of : str | None
        ISO date string "YYYY-MM-DD".  If None, uses today's date.
        Falls back to yesterday's file if today's is not found.

    Returns
    -------
    str
        Full gs:// URL of the latest matching file.
    """
    try:
        from google.cloud import storage  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("google-cloud-storage is required.") from exc

    client = storage.Client()
    target_date = as_of or date.today().isoformat()

    for delta in (0, 1, 2):
        from datetime import timedelta, date as _date
        d = (_date.fromisoformat(target_date) - timedelta(days=delta)).isoformat()
        name = f"{prefix}{d}{suffix}"
        blob = client.bucket(bucket).blob(name)
        try:
            if blob.exists():
                return f"gs://{bucket}/{name}"
        except Exception:
            continue

    raise FileNotFoundError(
        f"No file matching '{prefix}YYYY-MM-DD{suffix}' found in gs://{bucket}/ "
        f"for {target_date} or the two preceding days."
    )
