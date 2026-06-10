#!/usr/bin/env python3
"""Print .head() of every Unusual Whales endpoint used by the GEX dashboard.

Run locally or on Railway to inspect columns/units when configuring charts::

    python scripts/uw_api_head.py
    python scripts/uw_api_head.py --ticker SPX --rows 5
"""

from __future__ import annotations

import argparse

import pandas as pd

from gex_core.env_bootstrap import bootstrap_env


def _section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump UW API column heads for chart setup")
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--rows", type=int, default=3)
    args = parser.parse_args()

    bootstrap_env()
    from gex_core.uw_loader import (
        _get,
        fetch_uw_greek_exposure,
        fetch_uw_greek_exposure_by_expiration,
        fetch_uw_spot_exposures,
        fetch_uw_spot_exposures_intraday,
        fetch_uw_stock_state_price,
    )

    ticker = args.ticker.upper()
    n = max(1, args.rows)

    raw_greek = _get(f"/api/stock/{ticker}/greek-exposure/strike")
    _section(f"RAW  GET /api/stock/{ticker}/greek-exposure/strike  (API unit: M$)")
    greek_raw = pd.DataFrame(raw_greek)
    print("columns:", list(greek_raw.columns))
    print(greek_raw.head(n).to_string())

    greek = fetch_uw_greek_exposure(ticker)
    _section(f"LOADER  greek-exposure/strike  (pipeline: Bn$ = M$ / 1000)")
    print("columns:", list(greek.columns))
    print(greek.head(n).to_string())

    raw_spot = _get(f"/api/stock/{ticker}/spot-exposures/strike")
    _section(f"RAW  GET /api/stock/{ticker}/spot-exposures/strike  (API unit: raw $)")
    spot_raw = pd.DataFrame(raw_spot)
    print("columns:", list(spot_raw.columns))
    gamma_cols = [c for c in spot_raw.columns if "gamma" in c.lower() or c in ("strike", "price", "date", "time")]
    print("\nGamma-related subset:")
    print(spot_raw[gamma_cols].head(n).to_string())

    spot = fetch_uw_spot_exposures(ticker)
    _section(f"LOADER  spot-exposures/strike  (+ net_gamma_oi, net_gamma_oi_bn)")
    print("columns:", list(spot.columns))
    print(spot[gamma_cols + [c for c in spot.columns if c.startswith("net_gamma")]].head(n).to_string())

    intraday = fetch_uw_spot_exposures_intraday(ticker)
    _section(f"GET /api/stock/{ticker}/spot-exposures  (1-min totals, raw $)")
    if intraday.empty:
        print("(empty)")
    else:
        print("columns:", list(intraday.columns))
        print(intraday.head(n).to_string())

    exp = fetch_uw_greek_exposure_by_expiration(ticker)
    _section(f"GET /api/stock/{ticker}/greek-exposure/expiry  (Bn$ series)")
    if exp.empty:
        print("(empty)")
    else:
        print(exp.head(n).to_string())

    _section("Chart mapping cheat sheet")
    print(
        """
Primary gamma-by-strike chart (use this, not spot-OI):
  endpoint : greek-exposure/strike
  x        : strike
  y        : net_gex  OR magnet(call_gex, put_gex)  [Bn$ / %]
  scale    : divide API values by 1e3 (M$ -> Bn$)

ATM grid / spot price only:
  endpoint : spot-exposures/strike
  x        : strike
  y        : net_gamma_oi = call_gamma_oi + put_gamma_oi  [raw $ -> /1e9 for Bn$]
  spot     : price column

Intraday total GEX banner:
  endpoint : spot-exposures
  x        : time
  y        : gamma_per_one_percent_move_oi  [raw $ -> /1e9 for Bn$]

Term structure:
  endpoint : greek-exposure/expiry
  x        : expiry date
  y        : net gex  [Bn$ after /1e3]
"""
    )
    print(f"stock-state price: {fetch_uw_stock_state_price(ticker):.2f}")


if __name__ == "__main__":
    main()
