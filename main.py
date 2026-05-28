import argparse
import json
from datetime import timedelta, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib import dates

# Set plot style
plt.style.use("seaborn-dark")
for param in ["figure.facecolor", "axes.facecolor", "savefig.facecolor"]:
    plt.rcParams[param] = "#212946"
for param in ["text.color", "axes.labelcolor", "xtick.color", "ytick.color"]:
    plt.rcParams[param] = "0.9"

contract_size = 100
DATA_DIR = Path("data")
DEFAULT_OUTDIR = Path("img")
DEFAULT_EXPORT_DIR = DATA_DIR / "exports"
DEFAULT_CACHE_TTL_MINUTES = 15
REQUEST_TIMEOUT_SECONDS = 10


def run(
    ticker,
    refresh=False,
    cache_ttl_minutes=DEFAULT_CACHE_TTL_MINUTES,
    show_plots=True,
    save_plots=True,
    outdir=DEFAULT_OUTDIR,
    top_n=5,
    strike_window_pct=0.15,
    max_dte=365,
    export_csv=True,
    export_dir=DEFAULT_EXPORT_DIR,
):
    spot_price, option_data = scrape_data(
        ticker=ticker, refresh=refresh, cache_ttl_minutes=cache_ttl_minutes
    )
    compute_total_gex(spot_price, option_data)
    print_regime_summary(option_data)
    print_key_gex_levels(option_data, top_n=top_n)
    gex_by_strike, cumulative_gex = compute_gex_by_strike(
        ticker=ticker,
        spot=spot_price,
        data=option_data,
        show_plots=show_plots,
        save_plots=save_plots,
        outdir=outdir,
        top_n=top_n,
        strike_window_pct=strike_window_pct,
    )
    print_gamma_flip_estimate(cumulative_gex)
    gex_by_expiration = compute_gex_by_expiration(
        ticker=ticker,
        data=option_data,
        show_plots=show_plots,
        save_plots=save_plots,
        outdir=outdir,
        max_dte=max_dte,
    )
    surface_data = print_gex_surface(
        ticker=ticker,
        spot=spot_price,
        data=option_data,
        show_plots=show_plots,
        save_plots=save_plots,
        outdir=outdir,
        max_dte=max_dte,
        strike_window_pct=strike_window_pct,
    )
    if export_csv:
        export_analytics_csv(
            ticker=ticker,
            gex_by_strike=gex_by_strike,
            cumulative_gex=cumulative_gex,
            gex_by_expiration=gex_by_expiration,
            surface_data=surface_data,
            export_dir=export_dir,
        )


def is_cache_fresh(cache_file, cache_ttl_minutes):
    if not cache_file.exists():
        return False
    max_age = timedelta(minutes=cache_ttl_minutes)
    modified_at = datetime.fromtimestamp(cache_file.stat().st_mtime)
    return datetime.now() - modified_at <= max_age


def fetch_options_payload(ticker):
    endpoints = [
        f"https://cdn.cboe.com/api/global/delayed_quotes/options/_{ticker}.json",
        f"https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker}.json",
    ]

    last_error = None
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as err:
            last_error = err

    raise RuntimeError(f"Could not fetch options data for {ticker}: {last_error}")


def parse_payload(payload):
    if "data" not in payload:
        raise ValueError("Unexpected response format: missing 'data' field.")
    if "current_price" not in payload["data"] or "options" not in payload["data"]:
        raise ValueError("Unexpected response format: missing current price or options.")
    spot_price = float(payload["data"]["current_price"])
    option_data = pd.DataFrame(payload["data"]["options"])
    return spot_price, option_data


def scrape_data(ticker, refresh=False, cache_ttl_minutes=DEFAULT_CACHE_TTL_MINUTES):
    """Scrape data from CBOE website"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = DATA_DIR / f"{ticker}.json"

    if not refresh and is_cache_fresh(cache_file, cache_ttl_minutes):
        with cache_file.open() as f:
            payload = json.load(f)
    else:
        payload = fetch_options_payload(ticker)
        with cache_file.open("w") as f:
            json.dump(payload, f)

    spot_price, option_data = parse_payload(payload)
    return spot_price, fix_option_data(option_data)


def fix_option_data(data):
    """
    Fix option data columns.

    From the name of the option derive type of option, expiration and strike price
    """
    data["type"] = data.option.str.extract(r"\d([A-Z])\d")
    data["strike"] = data.option.str.extract(r"\d[A-Z](\d+)\d\d\d").astype(int)
    data["expiration"] = data.option.str.extract(r"[A-Z](\d+)").astype(str)
    # Convert expiration to datetime format
    data["expiration"] = pd.to_datetime(data["expiration"], format="%y%m%d")
    return data


def compute_total_gex(spot, data):
    """Compute dealers' total GEX"""
    # Compute gamma exposure for each option
    data["GEX"] = spot * data.gamma * data.open_interest * contract_size * spot * 0.01

    # For puts we assume negative gamma, i.e. dealers sell puts and buy calls.
    type_multiplier = np.where(data["type"] == "P", -1, 1)
    data["GEX"] = data["GEX"] * type_multiplier
    print(f"Total notional GEX: ${round(data.GEX.sum() / 10 ** 9, 4)} Bn")


def print_regime_summary(data):
    gex_by_strike = data.groupby("strike")["GEX"].sum().sort_values(ascending=False) / 10**9
    net_gex = gex_by_strike.sum()
    regime = "LONG gamma" if net_gex >= 0 else "SHORT gamma"

    print(f"Net gamma regime: {regime} ({net_gex:.3f} Bn$ / %)")

    if not gex_by_strike.empty:
        call_wall_strike = gex_by_strike.idxmax()
        put_wall_strike = gex_by_strike.idxmin()
        print(f"Estimated call wall: strike {call_wall_strike} ({gex_by_strike.max():.3f})")
        print(f"Estimated put wall: strike {put_wall_strike} ({gex_by_strike.min():.3f})")


def print_key_gex_levels(data, top_n=5):
    gex_by_strike = data.groupby("strike")["GEX"].sum() / 10**9
    positive = gex_by_strike[gex_by_strike > 0].sort_values(ascending=False).head(top_n)
    negative = gex_by_strike[gex_by_strike < 0].sort_values(ascending=True).head(top_n)

    print(f"\nTop {top_n} positive GEX strikes (Bn$ / %):")
    if positive.empty:
        print("  None")
    else:
        for strike, gex in positive.items():
            print(f"  Strike {strike}: {gex:.3f}")

    print(f"\nTop {top_n} negative GEX strikes (Bn$ / %):")
    if negative.empty:
        print("  None")
    else:
        for strike, gex in negative.items():
            print(f"  Strike {strike}: {gex:.3f}")


def print_gamma_flip_estimate(cumulative):
    if cumulative.empty:
        print("\nGamma flip estimate: unavailable (no strike data).")
        return

    signs = np.sign(cumulative.values)
    change_points = np.where(np.diff(signs) != 0)[0]

    if len(change_points) == 0:
        print("\nGamma flip estimate: no zero-crossing in cumulative strike GEX.")
        return

    idx = change_points[0]
    x0 = cumulative.index[idx]
    x1 = cumulative.index[idx + 1]
    y0 = cumulative.iloc[idx]
    y1 = cumulative.iloc[idx + 1]

    if y1 == y0:
        flip_estimate = float(x0)
    else:
        flip_estimate = float(x0 - y0 * (x1 - x0) / (y1 - y0))

    local_slope = abs(y1 - y0) / max(abs(x1 - x0), 1e-9)
    if local_slope >= 0.10:
        confidence = "high"
    elif local_slope >= 0.03:
        confidence = "medium"
    else:
        confidence = "low"
    print(f"\nEstimated gamma flip strike: {flip_estimate:.2f} (confidence: {confidence})")


def build_output_path(base_dir, ticker, plot_name, suffix):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return Path(base_dir) / f"{ticker}_{plot_name}_{timestamp}.{suffix}"


def finalize_plot(fig, output_path, show_plots, save_plots):
    if save_plots:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {output_path}")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def compute_gex_by_strike(
    ticker, spot, data, show_plots, save_plots, outdir, top_n=5, strike_window_pct=0.15
):
    """Compute and plot GEX by strike"""
    # Compute total GEX by strike
    gex_by_strike = data.groupby("strike")["GEX"].sum() / 10**9

    # Limit data to a configurable strike window around spot.
    lower = spot * (1 - strike_window_pct)
    upper = spot * (1 + strike_window_pct)
    limit_criteria = (gex_by_strike.index > lower) & (gex_by_strike.index < upper)
    selected = gex_by_strike.loc[limit_criteria]

    # Plot GEX by strike
    fig, ax = plt.subplots()
    colors = np.where(selected.values >= 0, "#38E07A", "#FE53BB")
    bars = ax.bar(selected.index, selected.values, color=colors, alpha=0.7)
    ax.grid(color="#2A3459")
    ax.tick_params(axis="x")
    ax.tick_params(axis="y")
    ax.set_xlabel("Strike", fontweight="heavy")
    ax.set_ylabel("Gamma Exposure (Bn$ / %)", fontweight="heavy")
    ax.set_title(f"{ticker} GEX by strike", fontweight="heavy")
    if not selected.empty:
        labels_to_annotate = selected.abs().sort_values(ascending=False).head(top_n)
        selected_index_values = selected.index.to_numpy()
        for strike in labels_to_annotate.index:
            bar_idx = int(np.where(selected_index_values == strike)[0][0])
            bar = bars[bar_idx]
            ax.annotate(
                f"{int(strike)}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3 if bar.get_height() >= 0 else -12),
                textcoords="offset points",
                ha="center",
                va="bottom" if bar.get_height() >= 0 else "top",
                fontsize=8,
            )
    finalize_plot(
        fig,
        build_output_path(outdir, ticker, "gex_by_strike", "png"),
        show_plots,
        save_plots,
    )
    cumulative = gex_by_strike.sort_index().cumsum()
    return gex_by_strike, cumulative


def compute_gex_by_expiration(ticker, data, show_plots, save_plots, outdir, max_dte=365):
    """Compute and plot GEX by expiration"""
    # Limit data to configurable days-to-expiration.
    selected_date = datetime.today() + timedelta(days=max_dte)
    data = data.loc[data.expiration < selected_date]

    # Compute GEX by expiration date
    gex_by_expiration = data.groupby("expiration")["GEX"].sum() / 10**9

    # Plot GEX by expiration
    fig, ax = plt.subplots()
    ax.bar(
        gex_by_expiration.index,
        gex_by_expiration.values,
        color="#FE53BB",
        alpha=0.5,
    )
    ax.grid(color="#2A3459")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y")
    ax.set_xlabel("Expiration date", fontweight="heavy")
    ax.set_ylabel("Gamma Exposure (Bn$ / %)", fontweight="heavy")
    ax.set_title(f"{ticker} GEX by expiration", fontweight="heavy")
    finalize_plot(
        fig,
        build_output_path(outdir, ticker, "gex_by_expiration", "png"),
        show_plots,
        save_plots,
    )
    return gex_by_expiration


def print_gex_surface(
    ticker, spot, data, show_plots, save_plots, outdir, max_dte=365, strike_window_pct=0.15
):
    """Plot 3D surface"""
    # Limit data to configurable DTE and strike window around spot.
    selected_date = datetime.today() + timedelta(days=max_dte)
    lower = spot * (1 - strike_window_pct)
    upper = spot * (1 + strike_window_pct)
    limit_criteria = (
        (data.expiration < selected_date)
        & (data.strike > lower)
        & (data.strike < upper)
    )
    data = data.loc[limit_criteria]

    # Compute GEX by expiration and strike
    data = data.groupby(["expiration", "strike"])["GEX"].sum() / 10**6
    data = data.reset_index()

    # Plot 3D surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        data["strike"],
        dates.date2num(data["expiration"]),
        data["GEX"],
        cmap="seismic_r",
    )
    ax.yaxis.set_major_formatter(dates.AutoDateFormatter(ax.xaxis.get_major_locator()))
    ax.set_ylabel("Expiration date", fontweight="heavy")
    ax.set_xlabel("Strike Price", fontweight="heavy")
    ax.set_zlabel("Gamma (M$ / %)", fontweight="heavy")
    finalize_plot(
        fig,
        build_output_path(outdir, ticker, "gex_surface", "png"),
        show_plots,
        save_plots,
    )
    return data


def export_analytics_csv(
    ticker, gex_by_strike, cumulative_gex, gex_by_expiration, surface_data, export_dir
):
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    gex_by_strike.rename("gex_bn_per_pct").to_csv(
        export_dir / f"{ticker}_gex_by_strike_{timestamp}.csv"
    )
    cumulative_gex.rename("cumulative_gex_bn_per_pct").to_csv(
        export_dir / f"{ticker}_cumulative_gex_{timestamp}.csv"
    )
    gex_by_expiration.rename("gex_bn_per_pct").to_csv(
        export_dir / f"{ticker}_gex_by_expiration_{timestamp}.csv"
    )
    surface_data.to_csv(export_dir / f"{ticker}_gex_surface_{timestamp}.csv", index=False)
    print(f"Saved CSV exports to: {export_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute and plot options gamma exposure.")
    parser.add_argument("--ticker", type=str, help="Underlying ticker symbol, e.g. SPX.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh options data even when cache is fresh.",
    )
    parser.add_argument(
        "--cache-ttl-minutes",
        type=int,
        default=DEFAULT_CACHE_TTL_MINUTES,
        help=f"Cache freshness threshold in minutes (default: {DEFAULT_CACHE_TTL_MINUTES}).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open plot windows.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save plots to disk.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(DEFAULT_OUTDIR),
        help=f"Directory where plots are saved (default: {DEFAULT_OUTDIR}).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top positive/negative GEX strikes to print (default: 5).",
    )
    parser.add_argument(
        "--strike-window-pct",
        type=float,
        default=0.15,
        help="Strike window around spot for charts (default: 0.15 means +-15%%).",
    )
    parser.add_argument(
        "--max-dte",
        type=int,
        default=365,
        help="Maximum days-to-expiration to include in term/surface charts (default: 365).",
    )
    parser.add_argument(
        "--no-export-csv",
        action="store_true",
        help="Disable CSV analytics exports.",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=str(DEFAULT_EXPORT_DIR),
        help=f"Directory where CSV exports are saved (default: {DEFAULT_EXPORT_DIR}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ticker = args.ticker.upper() if args.ticker else input("Enter desired ticker:").upper()
    run(
        ticker=ticker,
        refresh=args.refresh,
        cache_ttl_minutes=args.cache_ttl_minutes,
        show_plots=not args.no_show,
        save_plots=not args.no_save,
        outdir=args.outdir,
        top_n=max(1, args.top_n),
        strike_window_pct=max(0.01, min(1.0, args.strike_window_pct)),
        max_dte=max(1, args.max_dte),
        export_csv=not args.no_export_csv,
        export_dir=args.export_dir,
    )
