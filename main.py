"""
Gamma Exposure Tracker (GEX)

This module scrapes option data from the CBOE website and analyzes the dealers' 
notional gamma exposure (GEX) by computing, visualizing, and exporting gamma metrics.

GEX represents the total notional gamma exposure across all options, which helps traders 
understand market dynamics and potential gamma flip levels where dealer behavior changes.

Key metrics computed:
- Total notional GEX: aggregate gamma exposure across all options
- GEX by strike: gamma exposure distribution across strike prices
- GEX by expiration: gamma exposure distribution across expiration dates
- Gamma flip levels: estimated price levels where cumulative GEX crosses zero
- 3D GEX surface: visual representation of gamma exposure across strikes and expiration dates

Usage:
    python main.py SPX                    # Analyze SPX with defaults
    python main.py SPX --refresh          # Force refresh cached data
    python main.py SPX --max-dte 60       # Only include options expiring within 60 days
    python main.py SPX --no-show          # Generate plots without displaying them
"""

import argparse
import json
import sys
from datetime import timedelta, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib import dates

# ============================================================================
# CONFIGURATION & STYLING
# ============================================================================

# Set matplotlib plot style to a dark theme for better visualization
plt.style.use("dark_background")
plt.rcParams.update(
    {
        "figure.facecolor": "#0F172A",
        "axes.facecolor": "#111827",
        "savefig.facecolor": "#0F172A",
        "text.color": "#E2E8F0",
        "axes.labelcolor": "#E2E8F0",
        "xtick.color": "#CBD5E1",
        "ytick.color": "#CBD5E1",
        "axes.edgecolor": "#475569",
        "grid.color": "#334155",
        "grid.linestyle": "-",
        "grid.alpha": 0.25,
        "font.size": 11,
        "figure.figsize": (12, 7),
        "legend.frameon": False,
        "lines.linewidth": 2,
        "patch.edgecolor": "none",
    }
)

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_CYAN = "\033[96m"
ANSI_GREEN = "\033[92m"
ANSI_MAGENTA = "\033[95m"
ANSI_YELLOW = "\033[93m"
ANSI_RED = "\033[91m"


def supports_color():
    return sys.stdout.isatty()


def color_text(text, style):
    return f"{style}{text}{ANSI_RESET}" if supports_color() else text


def print_section_header(title):
    label = f" {title} "
    border = "═" * len(label)
    print()
    print(color_text(border, ANSI_MAGENTA))
    print(color_text(label, ANSI_BOLD + ANSI_MAGENTA))
    print(color_text(border, ANSI_MAGENTA))


def print_banner(ticker):
    title = f"GEX Tracker | {ticker}"
    border = "═" * len(title)
    print()
    print(color_text(border, ANSI_CYAN))
    print(color_text(title, ANSI_BOLD + ANSI_CYAN))
    print(color_text(border, ANSI_CYAN))
    print(color_text("Gamma vibes incoming — let the market stories unfold.", ANSI_DIM))
    print(color_text("Sit back, sip something nice, and watch the gamma dance.", ANSI_DIM))
    print()

# Standard contract multiplier for equity options (100 shares per contract)
contract_size = 100

# Default directories
DATA_DIR = Path("data")
DEFAULT_OUTDIR = Path("img")
DEFAULT_EXPORT_DIR = DATA_DIR / "exports"

# Cache configuration
DEFAULT_CACHE_TTL_MINUTES = 15  # How long to keep cached option data fresh (in minutes)
REQUEST_TIMEOUT_SECONDS = 10     # API request timeout


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

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
    """
    Execute the complete GEX analysis workflow for a given ticker.
    
    This is the main orchestration function that:
    1. Fetches and caches option data from CBOE
    2. Computes gamma exposure metrics (total, by strike, by expiration)
    3. Generates visualizations (bar charts, 3D surface)
    4. Prints regime summary and key levels
    5. Optionally exports data as CSV
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'SPX', 'XES')
        refresh (bool): If True, ignore cache and fetch fresh data from CBOE
        cache_ttl_minutes (int): Cache validity duration in minutes
        show_plots (bool): If True, display plots in matplotlib windows
        save_plots (bool): If True, save plots to disk as PNG files
        outdir (Path/str): Directory where PNG plots are saved
        top_n (int): Number of top positive/negative GEX strikes to highlight and print
        strike_window_pct (float): Percentage window around spot price for strike charts
                                   (e.g., 0.15 = ±15% around current price)
        max_dte (int): Maximum days-to-expiration to include in analysis (filters long-dated options)
        export_csv (bool): If True, export computed metrics as CSV files
        export_dir (Path/str): Directory where CSV files are saved
    
    Returns:
        None (prints output and generates files as side effects)
    
    Example:
        >>> run("SPX", refresh=True, show_plots=False, strike_window_pct=0.20)
    """
    
    print_banner(ticker)
    if refresh:
        print(color_text("Refreshing data from CBOE...", ANSI_DIM))
    
    # Step 1: Fetch and parse option data
    spot_price, option_data = scrape_data(
        ticker=ticker, refresh=refresh, cache_ttl_minutes=cache_ttl_minutes
    )
    print(
        f"{color_text('Spot', ANSI_YELLOW)}: {spot_price:.2f}   "
        f"{color_text('As of', ANSI_DIM)}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    
    # Step 2: Compute aggregate gamma exposure and print summary statistics
    compute_total_gex(spot_price, option_data)
    compute_total_charm(option_data)
    print_regime_summary(option_data)
    print_key_gex_levels(option_data, top_n=top_n)
    print_key_charm_levels(option_data, top_n=top_n)
    print_gamma_position_signal(spot_price, option_data, gamma_threshold=5000)
    print_future_gex_forecast(spot_price, option_data)
    
    # Step 3: Analyze gamma exposure across strike prices
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
    
    # Step 4: Estimate where gamma exposure crosses zero (gamma flip level)
    print_gamma_flip_estimate(cumulative_gex)
    
    # Step 5: Analyze gamma exposure across expiration dates
    gex_by_expiration = compute_gex_by_expiration(
        ticker=ticker,
        data=option_data,
        show_plots=show_plots,
        save_plots=save_plots,
        outdir=outdir,
        max_dte=max_dte,
    )
    
    # Step 6: Create 3D visualization of GEX surface (strike vs expiration)
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
    
    # Step 7: Export all computed metrics to CSV for further analysis
    if export_csv:
        export_analytics_csv(
            ticker=ticker,
            gex_by_strike=gex_by_strike,
            cumulative_gex=cumulative_gex,
            gex_by_expiration=gex_by_expiration,
            surface_data=surface_data,
            export_dir=export_dir,
        )


# ============================================================================
# DATA FETCHING & CACHING
# ============================================================================

def is_cache_fresh(cache_file, cache_ttl_minutes):
    """
    Check if a cached file is still valid based on age.
    
    Compares the file's modification time against the current time and 
    cache TTL (time-to-live) to determine if the cached data should be reused.
    
    Args:
        cache_file (Path): Path to the cache file to check
        cache_ttl_minutes (int): How many minutes the cache is considered fresh
    
    Returns:
        bool: True if cache exists and is newer than TTL, False otherwise
    
    Example:
        >>> fresh = is_cache_fresh(Path("data/SPX.json"), 15)
        >>> if fresh:
        ...     # Use cached data
        ... else:
        ...     # Fetch new data
    """
    if not cache_file.exists():
        return False
    
    max_age = timedelta(minutes=cache_ttl_minutes)
    modified_at = datetime.fromtimestamp(cache_file.stat().st_mtime)
    return datetime.now() - modified_at <= max_age


def fetch_options_payload(ticker):
    """
    Fetch raw options data from CBOE API.
    
    Attempts to fetch JSON data from CBOE's delayed quotes API using two possible
    endpoint formats. The API returns current price and all active options data.
    
    CBOE Endpoints:
    - https://cdn.cboe.com/api/global/delayed_quotes/options/_{ticker}.json
    - https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker}.json
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'SPX', 'XES')
    
    Returns:
        dict: Raw JSON payload containing current price and options list
    
    Raises:
        RuntimeError: If both endpoints fail to return data
    
    Example:
        >>> data = fetch_options_payload("SPX")
        >>> print(data["data"]["current_price"])
        4800.25
    """
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
    """
    Parse raw CBOE API response into structured format.
    
    Extracts spot price and option data from the nested JSON structure returned
    by CBOE, converting options to a pandas DataFrame for easier analysis.
    
    Expected payload structure:
    {
        "data": {
            "current_price": 4800.25,
            "options": [
                {"option": "SPX  260620C04800000", "gamma": 0.00012, ...},
                ...
            ]
        }
    }
    
    Args:
        payload (dict): Raw JSON response from CBOE API
    
    Returns:
        tuple: (spot_price: float, option_data: DataFrame)
               - spot_price: Current underlying price
               - option_data: DataFrame with all options and their Greeks
    
    Raises:
        ValueError: If payload structure is unexpected or missing required fields
    
    Example:
        >>> spot, options_df = parse_payload(raw_json)
        >>> print(f"Spot: {spot}, Options: {len(options_df)}")
    """
    if "data" not in payload:
        raise ValueError("Unexpected response format: missing 'data' field.")
    if "current_price" not in payload["data"] or "options" not in payload["data"]:
        raise ValueError("Unexpected response format: missing current price or options.")
    
    spot_price = float(payload["data"]["current_price"])
    raw_options = payload["data"].get("options", [])
    if raw_options is None:
        raw_options = []
    option_data = pd.DataFrame(raw_options)
    return spot_price, option_data


def scrape_data(ticker, refresh=False, cache_ttl_minutes=DEFAULT_CACHE_TTL_MINUTES):
    """
    Fetch and cache options data from CBOE website.
    
    Implements a caching strategy to minimize API calls:
    1. If refresh=False and cache is fresh, loads from disk
    2. Otherwise, fetches from CBOE and saves to cache
    3. Parses the JSON payload and fixes data types
    
    Cache file location: data/{ticker}.json
    
    Args:
        ticker (str): Stock ticker symbol
        refresh (bool): Force fresh fetch even if cache is valid
        cache_ttl_minutes (int): Cache validity duration in minutes
    
    Returns:
        tuple: (spot_price: float, option_data: DataFrame)
               Returns cleaned option data with derived fields (type, strike, expiration)
    
    Example:
        >>> spot, options_df = scrape_data("SPX", refresh=False)
        >>> print(len(options_df), "options available")
        1248 options available
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = DATA_DIR / f"{ticker}.json"

    # Load from cache if valid, otherwise fetch fresh
    if not refresh and is_cache_fresh(cache_file, cache_ttl_minutes):
        with cache_file.open() as f:
            payload = json.load(f)
    else:
        payload = fetch_options_payload(ticker)
        with cache_file.open("w") as f:
            json.dump(payload, f)

    spot_price, option_data = parse_payload(payload)
    option_data = fix_option_data(option_data)
    if option_data is None or not isinstance(option_data, pd.DataFrame):
        raise RuntimeError("Failed to parse option data from CBOE payload.")
    if option_data.empty:
        raise RuntimeError("Option data was loaded but contains no rows. Check the data source and payload structure.")
    return spot_price, option_data


# ============================================================================
# DATA PROCESSING
# ============================================================================

def fix_option_data(data):
    """
    Parse and enrich option data with derived fields.
    
    Option symbols follow a standard format: SPX  260620C04800000
    This function extracts:
    - Type (C=Call, P=Put): the middle letter
    - Strike price: the numeric value after type
    - Expiration date: the first 6 digits (YYMMDD format)
    
    The derived fields are needed for grouping and analysis:
    - Calls vs Puts have opposite gamma signs
    - Strikes are grouped for GEX aggregation
    - Expiration dates define the term structure
    
    Args:
        data (DataFrame): Raw option data from CBOE with 'option' column
    
    Returns:
        DataFrame: Enhanced data with added columns:
                   - 'type': 'C' for calls, 'P' for puts
                   - 'strike': integer strike price
                   - 'expiration': datetime of expiration
    
    Example:
        >>> data = fix_option_data(raw_df)
        >>> print(data[['option', 'type', 'strike', 'expiration']].head())
        option          type strike expiration
        SPX  260620C...  C    4800   2026-06-20
    """
    # Extract option type (C or P) using regex on the option symbol
    # Pattern: any digits, then [A-Z] (the type), then digits
    data["type"] = data["option"].str.extract(r"\d([A-Z])\d")
    
    # Extract strike price (numeric value after type letter)
    # Pattern: digit, [A-Z], then capture digits (strike), then 3 more digits (cents)
    data["strike"] = data["option"].str.extract(r"\d[A-Z](\d+)\d\d\d").astype(int)
    
    # Extract expiration date (first 6 digits in YYMMDD format)
    # Pattern: [A-Z], then capture 6 digits
    data["expiration"] = data["option"].str.extract(r"[A-Z](\d+)").astype(str)
    
    # Convert YYMMDD string to datetime for sorting and analysis
    data["expiration"] = pd.to_datetime(data["expiration"], format="%y%m%d")
    
    # Parse charm if available, keeping the raw values numeric
    if "charm" in data.columns:
        data["charm"] = pd.to_numeric(data["charm"], errors="coerce").fillna(0.0)

    return data


def compute_total_gex(spot, data):
    """
    Calculate dealers' total notional gamma exposure (GEX).
    
    GEX represents the aggregate gamma exposure across all options based on
    the assumption that dealers hedge by buying calls and selling puts.
    
    Formula for each option:
        GEX = Spot × Gamma × Open Interest × Contract Size × Spot × 0.01
        
    The 0.01 factor converts 1% gamma to decimals. 
    For puts, we multiply by -1 because dealers are short puts (negative gamma).
    
    Interpretation:
    - Positive total GEX: Dealers are long gamma (profit from volatility)
    - Negative total GEX: Dealers are short gamma (hurt by volatility)
    - The magnitude indicates how much the dealer's delta changes per 1% spot move
    
    Args:
        spot (float): Current spot price of the underlying
        data (DataFrame): Options data with columns 'gamma', 'open_interest', 'type'
                         (DataFrame is modified in-place)
    
    Returns:
        None (modifies data['GEX'] in-place and prints total)
    
    Example:
        >>> compute_total_gex(4800, options_df)
        Total notional GEX: $-38.1193 Bn
    """
    if data is None:
        raise ValueError("compute_total_gex() received no option data. Verify that scrape_data() returned a valid DataFrame.")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("compute_total_gex() requires a pandas DataFrame for option data.")
    if data.empty:
        raise ValueError("compute_total_gex() received an empty option DataFrame.")

    # Compute gross gamma exposure for each option
    # Units: notional dollar exposure per 1% spot price move
    data["GEX"] = spot * data.gamma * data.open_interest * contract_size * spot * 0.01

    # Apply sign convention: calls are positive (dealers long calls), puts are negative
    # (dealers short puts). This aligns with dealer delta hedging behavior.
    type_multiplier = np.where(data["type"] == "P", -1, 1)
    data["GEX"] = data["GEX"] * type_multiplier
    
    # Print aggregate GEX in billions
    total_gex_bn = round(data.GEX.sum() / 10 ** 9, 4)
    print_section_header("Total GEX")
    print(f"Total notional gamma exposure: {color_text(f'${total_gex_bn:.4f} Bn', ANSI_GREEN)}")


def compute_total_charm(data):
    if "charm" not in data.columns:
        return None

    data["CharmExposure"] = data["charm"] * data.open_interest * contract_size
    total_charm = data["CharmExposure"].sum()

    print_section_header("Charm Exposure")
    print(f"Total notional charm exposure: {color_text(f'{total_charm:,.2f}', ANSI_YELLOW)} per day")
    if total_charm >= 0:
        print(color_text("Charm is net positive, indicating delta is decaying more gently in the current book.", ANSI_DIM))
    else:
        print(color_text("Charm is net negative, indicating delta is decaying faster across the current positions.", ANSI_DIM))

    return data


def print_key_charm_levels(data, top_n=5):
    if "CharmExposure" not in data.columns:
        print_section_header("Charm Levels")
        print(color_text("Charm analysis unavailable: no charm field in the option data.", ANSI_DIM))
        return

    charm_by_strike = data.groupby("strike")["CharmExposure"].sum()
    positive = charm_by_strike[charm_by_strike > 0].sort_values(ascending=False).head(top_n)
    negative = charm_by_strike[charm_by_strike < 0].sort_values().head(top_n)

    print_section_header(f"Top {top_n} Charm Strikes")
    if positive.empty and negative.empty:
        print(color_text("No meaningful charm exposure by strike.", ANSI_DIM))
        return

    if not positive.empty:
        print(color_text("Positive charm strikes:", ANSI_GREEN))
        for strike, charm in positive.items():
            print(f"  Strike {strike}: {charm:.3f}")

    if not negative.empty:
        print(color_text("Negative charm strikes:", ANSI_RED))
        for strike, charm in negative.items():
            print(f"  Strike {strike}: {charm:.3f}")


def print_regime_summary(data):
    """
    Print high-level gamma regime analysis and key price levels.
    
    Identifies:
    1. Net gamma regime: LONG gamma (positive) or SHORT gamma (negative)
    2. Call wall: Strike with maximum positive GEX (dealers long call exposure)
    3. Put wall: Strike with minimum negative GEX (dealers short put exposure)
    
    These price levels represent areas where dealer hedging behavior can impact
    the market (gamma squeezes, gamma floors).
    
    Args:
        data (DataFrame): Options data with computed 'GEX' column
    
    Returns:
        None (prints regime summary)
    
    Example:
        >>> print_regime_summary(options_df)
        Net gamma regime: LONG gamma (12.345 Bn$ / %)
        Estimated call wall: strike 4800 (45.123)
        Estimated put wall: strike 4750 (-38.105)
    """
    # Aggregate GEX by strike (sum across all expirations and option types)
    gex_by_strike = data.groupby("strike")["GEX"].sum().sort_values(ascending=False) / 10**9
    net_gex = gex_by_strike.sum()
    
    # Determine if the market is in a long or short gamma regime
    regime = "LONG gamma" if net_gex >= 0 else "SHORT gamma"
    print_section_header("Gamma Regime")
    print(f"Net gamma regime: {color_text(regime, ANSI_YELLOW)} ({net_gex:.3f} Bn$ / %)")
    print(color_text("Market mood: dealers are either cozy with volatility or braced for it.", ANSI_DIM))

    # Find call wall (max positive GEX) and put wall (min negative GEX)
    if not gex_by_strike.empty:
        call_wall_strike = gex_by_strike.idxmax()
        put_wall_strike = gex_by_strike.idxmin()
        print(f"Estimated call wall: strike {color_text(call_wall_strike, ANSI_GREEN)} ({gex_by_strike.max():.3f})")
        print(f"Estimated put wall: strike {color_text(put_wall_strike, ANSI_RED)} ({gex_by_strike.min():.3f})")


def print_key_gex_levels(data, top_n=5):
    """
    Print the top N positive and negative GEX strikes.
    
    These levels represent the most significant gamma exposure levels where
    dealer hedging activity is concentrated. Key trading levels.
    
    Args:
        data (DataFrame): Options data with computed 'GEX' column
        top_n (int): Number of top strikes to display on each side
    
    Returns:
        None (prints to console)
    
    Example:
        >>> print_key_gex_levels(options_df, top_n=3)
        Top 3 positive GEX strikes (Bn$ / %):
          Strike 4800: 45.123
          Strike 4850: 38.456
          Strike 4750: 32.789
        
        Top 3 negative GEX strikes (Bn$ / %):
          Strike 4700: -42.100
          Strike 4650: -35.200
          Strike 4600: -28.900
    """
    # Aggregate GEX by strike across all expirations
    gex_by_strike = data.groupby("strike")["GEX"].sum() / 10**9
    
    # Separate positive and negative GEX and sort by magnitude
    positive = gex_by_strike[gex_by_strike > 0].sort_values(ascending=False).head(top_n)
    negative = gex_by_strike[gex_by_strike < 0].sort_values(ascending=True).head(top_n)

    print_section_header(f"Top {top_n} GEX Strikes")
    print(color_text("Hot gamma zones — keep an eye on these strikes.", ANSI_DIM))
    print(f"{color_text('Signal', ANSI_BOLD):<12} {color_text('Strike', ANSI_BOLD):<10} {color_text('GEX (Bn$ / %)', ANSI_BOLD)}")
    if positive.empty:
        print(color_text('  None', ANSI_DIM))
    else:
        for strike, gex in positive.items():
            print(f"  {color_text('LONG', ANSI_GREEN):<12} {strike:<10} {gex:.3f}")

    print(color_text('\n' + 'Cautionary gamma levels — these are the downside hotspots.', ANSI_DIM))
    print(f"{color_text('Signal', ANSI_BOLD):<12} {color_text('Strike', ANSI_BOLD):<10} {color_text('GEX (Bn$ / %)', ANSI_BOLD)}")
    if negative.empty:
        print(color_text('  None', ANSI_DIM))
    else:
        for strike, gex in negative.items():
            print(f"  {color_text('SHORT', ANSI_RED):<12} {strike:<10} {gex:.3f}")


def print_gamma_position_signal(spot, data, gamma_threshold=5000):
    """
    Print a buy/sell signal based on the strike with the strongest gamma exposure.

    This function finds the strike with absolute GEX above the threshold and then
    compares it to the current spot price. If the strike is above spot, the signal
    is to buy calls and sell puts. If it is below spot, the signal is to buy puts
    and sell calls.

    Args:
        spot (float): Current underlying spot price.
        data (DataFrame): Options data with computed 'GEX'.
        gamma_threshold (float): Absolute GEX threshold in raw dollars for the
                                  signal strike.

    Returns:
        None (prints a recommendation)
    """
    gex_by_strike = data.groupby("strike")["GEX"].sum()
    high_gamma = gex_by_strike[gex_by_strike.abs() > gamma_threshold]

    if high_gamma.empty:
        print_section_header("Gamma Signal")
        print(color_text(f"No strike has absolute gamma exposure above {gamma_threshold}.", ANSI_DIM))
        print(color_text("No signal generated.", ANSI_DIM))
        return

    signal_strike = high_gamma.abs().idxmax()
    strike_gex = gex_by_strike.loc[signal_strike]
    strike_gex_bn = strike_gex / 10**9

    if signal_strike > spot:
        action = "BUY CALLS and SELL PUTS"
        position = "above"
        action_color = ANSI_GREEN
    elif signal_strike < spot:
        action = "BUY PUTS and SELL CALLS"
        position = "below"
        action_color = ANSI_RED
    else:
        action = "HOLD or consider balanced call/put positioning"
        position = "at"
        action_color = ANSI_YELLOW

    print_section_header("Gamma Signal")
    print(f"{color_text('Signal strike:', ANSI_CYAN)} {signal_strike} is {position} current SPX ({spot}).")
    print(f"{color_text('Total strike GEX:', ANSI_CYAN)} {strike_gex_bn:.3f} Bn$ / %")
    print(f"{color_text('Recommended positioning:', ANSI_CYAN)} {color_text(action, action_color)}")


def print_future_gex_forecast(spot, data, strike_window_pct=0.15, top_n=3):
    """
    Predict future gamma exposure trends from current strike-level GEX growth and decline.

    This function uses the current GEX distribution across strikes to identify:
    - strike ranges where GEX is accelerating (growth)
    - strike ranges where GEX is decelerating (decline)
    - whether those patterns are clustered above or below the current spot

    It then prints a simple directional forecast for future total GEX behavior.
    """
    gex_by_strike = data.groupby("strike")["GEX"].sum() / 10**9
    if gex_by_strike.empty:
        print("\nFuture GEX forecast: unavailable (no strike data).")
        return

    gex_by_strike = gex_by_strike.sort_index()
    strike_diffs = gex_by_strike.index.to_series().diff().replace(0, np.nan)
    slope = gex_by_strike.diff() / strike_diffs
    slope = slope.fillna(0)

    lower = spot * (1 - strike_window_pct)
    upper = spot * (1 + strike_window_pct)
    local_slope = slope[(slope.index >= lower) & (slope.index <= upper)]
    trend_score = local_slope.mean() if not local_slope.empty else slope.mean()

    if trend_score > 0:
        forecast = "likely to increase"
        trend_text = "build further positive gamma or less negative gamma as strikes move"
    elif trend_score < 0:
        forecast = "likely to decline"
        trend_text = "fade positive gamma or add more negative gamma as strikes move"
    else:
        forecast = "appear stable"
        trend_text = "showing little directional change across the strike curve"

    strongest_growth = slope.sort_values(ascending=False).head(top_n)
    strongest_decline = slope.sort_values().head(top_n)

    growth_above = strongest_growth[strongest_growth.index > spot]
    growth_below = strongest_growth[strongest_growth.index < spot]
    decline_above = strongest_decline[strongest_decline.index > spot]
    decline_below = strongest_decline[strongest_decline.index < spot]

    mood = "buoyant" if trend_score > 0.05 else "steady" if abs(trend_score) <= 0.05 else "cautious"
    mood_line = {
        "buoyant": "The gamma curve is humming with positive energy.",
        "steady": "The market feels steady — neither frothy nor fearful.",
        "cautious": "The gamma map is whispering caution; stay alert."
    }[mood]

    print_section_header("Future GEX Forecast")
    print(f"{color_text('Forecast:', ANSI_CYAN)} Based on current strike momentum, total GEX is {color_text(forecast, ANSI_YELLOW)}.")
    print(f"{color_text('Gamma mood:', ANSI_CYAN)} {color_text(mood_line, ANSI_DIM)}")
    print(f"{color_text('Strike slope range:', ANSI_CYAN)} {lower:.0f} to {upper:.0f}")
    print(f"{color_text('Trend score:', ANSI_CYAN)} {trend_score:.4f} Bn$ per strike")
    print(f"{color_text('Interpretation:', ANSI_CYAN)} {trend_text}.")

    if not growth_above.empty:
        strike = int(growth_above.index[0])
        value = growth_above.iat[0]
        print(f"  {color_text('Growth above spot:', ANSI_GREEN)} strike {strike} ({value:.4f} Bn$ per strike step)")
    if not growth_below.empty:
        strike = int(growth_below.index[0])
        value = growth_below.iat[0]
        print(f"  {color_text('Growth below spot:', ANSI_GREEN)} strike {strike} ({value:.4f} Bn$ per strike step)")
    if not decline_above.empty:
        strike = int(decline_above.index[0])
        value = decline_above.iat[0]
        print(f"  {color_text('Decline above spot:', ANSI_RED)} strike {strike} ({value:.4f} Bn$ per strike step)")
    if not decline_below.empty:
        strike = int(decline_below.index[0])
        value = decline_below.iat[0]
        print(f"  {color_text('Decline below spot:', ANSI_RED)} strike {strike} ({value:.4f} Bn$ per strike step)")


def print_gamma_flip_estimate(cumulative):
    """
    Estimate the strike where cumulative GEX crosses zero (gamma flip level).
    
    As price moves up/down, cumulative gamma exposure changes. When it crosses
    zero, it indicates a shift from long gamma to short gamma or vice versa.
    These are critical levels where dealer hedging behavior changes.
    
    The function finds the first zero-crossing in the cumulative GEX curve
    and linearly interpolates to estimate the exact strike level.
    
    Confidence levels based on local slope:
    - High (≥0.10): Sharp transition, reliable estimate
    - Medium (≥0.03): Moderate transition
    - Low (<0.03): Gradual transition, less reliable
    
    Args:
        cumulative (Series): Cumulative GEX sorted by strike (indexed by strike)
    
    Returns:
        None (prints estimated flip level and confidence)
    
    Example:
        >>> print_gamma_flip_estimate(cumulative_gex)
        Estimated gamma flip strike: 4800.50 (confidence: high)
    """
    if cumulative.empty:
        print("\nGamma flip estimate: unavailable (no strike data).")
        return

    # Find where the sign of cumulative GEX changes (zero-crossings)
    signs = np.sign(cumulative.values)
    change_points = np.where(np.diff(signs) != 0)[0]

    if len(change_points) == 0:
        print("\nGamma flip estimate: no zero-crossing in cumulative strike GEX.")
        return

    # Get the first (lowest strike) zero-crossing point
    idx = change_points[0]
    x0 = cumulative.index[idx]
    x1 = cumulative.index[idx + 1]
    y0 = cumulative.iloc[idx]
    y1 = cumulative.iloc[idx + 1]

    # Linear interpolation to estimate the exact strike where cumulative GEX = 0
    if y1 == y0:
        flip_estimate = float(x0)
    else:
        flip_estimate = float(x0 - y0 * (x1 - x0) / (y1 - y0))

    # Measure confidence based on slope steepness
    # Steep slope = sharp transition = high confidence
    local_slope = abs(y1 - y0) / max(abs(x1 - x0), 1e-9)
    if local_slope >= 0.10:
        confidence = "high"
    elif local_slope >= 0.03:
        confidence = "medium"
    else:
        confidence = "low"
    
    print(f"\nEstimated gamma flip strike: {flip_estimate:.2f} (confidence: {confidence})")


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def build_output_path(base_dir, ticker, plot_name, suffix):
    """
    Construct a timestamped output file path for plots.
    
    Format: {base_dir}/{ticker}_{plot_name}_{YYYY-MM-DD_HHMMSS}.{suffix}
    
    The timestamp ensures each run produces unique file names, preventing overwrites.
    
    Args:
        base_dir (Path/str): Output directory
        ticker (str): Stock ticker symbol
        plot_name (str): Descriptive name for the plot (e.g., 'gex_by_strike')
        suffix (str): File extension (e.g., 'png', 'pdf')
    
    Returns:
        Path: Full path to the output file
    
    Example:
        >>> path = build_output_path("img", "SPX", "gex_by_strike", "png")
        >>> print(path)
        img/SPX_gex_by_strike_2025-05-28_143025.png
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return Path(base_dir) / f"{ticker}_{plot_name}_{timestamp}.{suffix}"


def finalize_plot(fig, output_path, show_plots, save_plots):
    """
    Save and/or display a matplotlib figure.
    
    Centralized plot handling to manage multiple output options:
    - Save to disk (PNG)
    - Display plot in window
    - Or both

    Args:
        fig (Figure): Matplotlib figure object to finalize
        output_path (Path): Path where to save the plot
        show_plots (bool): If True, display plot in window
        save_plots (bool): If True, save plot to disk

    Returns:
        None (modifies figure/file system as side effect)

    Example:
        >>> finalize_plot(fig, Path("img/plot.png"), show_plots=True, save_plots=True)
    """
    if save_plots:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {output_path}")

    if show_plots:
        try:
            plt.show(block=True)
        except Exception:
            plt.show()
# ============================================================================

def compute_gex_by_strike(
    ticker, spot, data, show_plots, save_plots, outdir, top_n=5, strike_window_pct=0.15
):
    """
    Analyze and visualize gamma exposure distribution across strike prices.
    
    This function creates two key outputs:
    1. Bar chart of GEX by strike (within configurable window around spot price)
    2. Cumulative GEX curve (useful for identifying gamma flip levels)
    
    The strike window limits the chart to a focused range around the current price
    to avoid over-plotting very distant out-of-the-money options.
    
    The top_n parameter highlights the most significant gamma levels with labels.
    
    Args:
        ticker (str): Stock ticker symbol (for labeling)
        spot (float): Current spot price
        data (DataFrame): Options data with 'strike' and 'GEX' columns
        show_plots (bool): Display chart in window
        save_plots (bool): Save chart to disk
        outdir (Path/str): Output directory for plots
        top_n (int): Number of top strikes to label on chart
        strike_window_pct (float): Percentage window around spot (e.g., 0.15 = ±15%)
    
    Returns:
        tuple: (gex_by_strike: Series, cumulative_gex: Series)
               - gex_by_strike: GEX indexed by strike
               - cumulative_gex: Cumulative GEX sorted by strike
    
    Example:
        >>> gex_by_strike, cumulative = compute_gex_by_strike(
        ...     "SPX", 4800, options_df, True, True, "img", top_n=3, strike_window_pct=0.20
        ... )
        >>> print(f"Strike range: {gex_by_strike.index.min()} to {gex_by_strike.index.max()}")
    """
    
    # Step 1: Aggregate GEX across all expirations by strike
    gex_by_strike = data.groupby("strike")["GEX"].sum() / 10**9

    # Step 2: Filter to a window around the current spot price
    # This focuses the visualization on relevant price levels
    lower = spot * (1 - strike_window_pct)
    upper = spot * (1 + strike_window_pct)
    limit_criteria = (gex_by_strike.index > lower) & (gex_by_strike.index < upper)
    selected = gex_by_strike.loc[limit_criteria]

    # Step 3: Create bar chart with color-coding
    # Green bars = positive GEX (long gamma)
    # Pink bars = negative GEX (short gamma)
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = np.where(selected.values >= 0, "#38E07A", "#FE53BB")
    bars = ax.bar(selected.index, selected.values, color=colors, alpha=0.85, edgecolor="none")
    
    # Styling
    ax.set_facecolor("#0F172A")
    ax.grid(color="#334155", linestyle="-", alpha=0.25)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y")
    ax.set_xlabel("Strike", fontweight="heavy")
    ax.set_ylabel("Gamma Exposure (Bn$ / %)", fontweight="heavy")
    ax.set_title(f"{ticker} GEX by strike", fontweight="heavy")
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Step 4: Label the top_n most significant gamma levels
    if not selected.empty:
        # Find top strikes by absolute GEX magnitude
        labels_to_annotate = selected.abs().sort_values(ascending=False).head(top_n)
        selected_index_values = selected.index.to_numpy()
        
        for strike in labels_to_annotate.index:
            # Find the bar corresponding to this strike
            bar_idx = int(np.where(selected_index_values == strike)[0][0])
            bar = bars[bar_idx]
            
            # Place label above (positive GEX) or below (negative GEX) the bar
            ax.annotate(
                f"{int(strike)}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3 if bar.get_height() >= 0 else -12),
                textcoords="offset points",
                ha="center",
                va="bottom" if bar.get_height() >= 0 else "top",
                fontsize=8,
            )
    
    # Step 5: Save/display the plot
    finalize_plot(
        fig,
        build_output_path(outdir, ticker, "gex_by_strike", "png"),
        show_plots,
        save_plots,
    )
    
    # Step 6: Compute and return cumulative GEX for gamma flip analysis
    cumulative = gex_by_strike.sort_index().cumsum()
    return gex_by_strike, cumulative


# ============================================================================
# ANALYSIS: GAMMA BY EXPIRATION
# ============================================================================

def compute_gex_by_expiration(ticker, data, show_plots, save_plots, outdir, max_dte=365):
    """
    Analyze and visualize gamma exposure distribution across expiration dates.
    
    Shows how gamma exposure is distributed across the term structure of options.
    Near-term expirations typically have more gamma per dollar due to theta decay.
    This helps understand which expiration cycles are most impactful for price moves.
    
    Args:
        ticker (str): Stock ticker symbol (for labeling)
        data (DataFrame): Options data with 'expiration' and 'GEX' columns
        show_plots (bool): Display chart in window
        save_plots (bool): Save chart to disk
        outdir (Path/str): Output directory for plots
        max_dte (int): Maximum days-to-expiration to include (filters distant expirations)
                       Use 0 to include same-day (0DTE) expirations only.
    
    Returns:
        Series: GEX indexed by expiration date
    
    Example:
        >>> gex_by_exp = compute_gex_by_expiration("SPX", options_df, True, True, "img", max_dte=90)
        >>> print(gex_by_exp.head())
    """
    
    # Step 1: Filter to expirations within max_dte
    today = datetime.today().date()
    selected_date = today + timedelta(days=max_dte)
    data = data.loc[
        (data.expiration.dt.date >= today)
        & (data.expiration.dt.date <= selected_date)
    ]

    # Step 2: Aggregate GEX by expiration date across all strikes
    gex_by_expiration = data.groupby("expiration")["GEX"].sum() / 10**9

    # Step 3: Create bar chart showing term structure
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(
        gex_by_expiration.index,
        gex_by_expiration.values,
        color="#82C7FF",
        alpha=0.8,
        edgecolor="none",
    )
    
    # Styling
    ax.set_facecolor("#0F172A")
    ax.grid(color="#334155", linestyle="-", alpha=0.25)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y")
    ax.set_xlabel("Expiration date", fontweight="heavy")
    ax.set_ylabel("Gamma Exposure (Bn$ / %)", fontweight="heavy")
    ax.set_title(f"{ticker} GEX by expiration", fontweight="heavy")
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Step 4: Save/display the plot
    finalize_plot(
        fig,
        build_output_path(outdir, ticker, "gex_by_expiration", "png"),
        show_plots,
        save_plots,
    )
    
    return gex_by_expiration


# ============================================================================
# ANALYSIS: 3D GAMMA SURFACE
# ============================================================================

def print_gex_surface(
    ticker, spot, data, show_plots, save_plots, outdir, max_dte=365, strike_window_pct=0.15
):
    """
    Create and visualize a 3D surface plot of GEX across strikes and expirations.
    
    This visualization shows how gamma exposure varies across two dimensions:
    - X-axis: Strike prices
    - Y-axis: Expiration dates (term structure)
    - Z-axis: GEX magnitude
    
    This provides a comprehensive view of the gamma landscape and can highlight
    clusters of gamma exposure (hot spots) across both dimensions.
    
    Args:
        ticker (str): Stock ticker symbol (for labeling)
        spot (float): Current spot price
        data (DataFrame): Options data with 'strike', 'expiration', and 'GEX' columns
        show_plots (bool): Display plot in window
        save_plots (bool): Save plot to disk
        outdir (Path/str): Output directory for plots
        max_dte (int): Maximum days-to-expiration to include
                       Use 0 to include same-day (0DTE) expirations only.
        strike_window_pct (float): Percentage window around spot for strike axis
    
    Returns:
        DataFrame: Surface data with columns ['expiration', 'strike', 'GEX']
    
    Example:
        >>> surface = print_gex_surface("SPX", 4800, options_df, True, True, "img")
    """
    
    # Step 1: Apply filters (date and strike range)
    today = datetime.today().date()
    selected_date = today + timedelta(days=max_dte)
    limit_criteria = (
        (data.expiration.dt.date >= today)
        & (data.expiration.dt.date <= selected_date)
        & (data.strike > spot * (1 - strike_window_pct))
        & (data.strike < spot * (1 + strike_window_pct))
    )
    data = data.loc[limit_criteria]

    # Step 2: Aggregate GEX by [expiration, strike] pairs
    # Convert to millions for better numerical scale on Z-axis
    data = data.groupby(["expiration", "strike"])["GEX"].sum() / 10**6
    data = data.reset_index()

    # Step 3: Create 3D surface plot
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Convert dates to numeric for 3D plotting
    x = data["strike"].to_numpy()
    y = dates.date2num(data["expiration"])
    z = data["GEX"].to_numpy()

    if data.empty:
        print("No surface data available for the selected filters.")
    elif len(data) < 3:
        ax.scatter(
            x,
            y,
            z,
            c=z,
            cmap="coolwarm",
            depthshade=True,
            s=60,
            edgecolors="k",
            linewidths=0.3,
        )
    else:
        try:
            ax.plot_trisurf(
                x,
                y,
                z,
                cmap="coolwarm",
                linewidth=0.2,
                antialiased=True,
                alpha=0.95,
            )
        except (RuntimeError, ValueError):
            ax.scatter(
                x,
                y,
                z,
                c=z,
                cmap="coolwarm",
                depthshade=True,
                s=60,
                edgecolors="k",
                linewidths=0.3,
            )

    # Format Y-axis (expiration) with proper date labels
    ax.yaxis.set_major_formatter(dates.AutoDateFormatter(ax.xaxis.get_major_locator()))
    ax.view_init(elev=30, azim=-60)
    
    # Labeling
    ax.set_ylabel("Expiration date", fontweight="heavy")
    ax.set_xlabel("Strike Price", fontweight="heavy")
    ax.set_zlabel("Gamma (M$ / %)", fontweight="heavy")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.line.set_color("#475569")
        except Exception:
            pass
    
    # Step 4: Save/display the plot
    finalize_plot(
        fig,
        build_output_path(outdir, ticker, "gex_surface", "png"),
        show_plots,
        save_plots,
    )
    
    return data


# ============================================================================
# EXPORT
# ============================================================================

def export_analytics_csv(
    ticker, gex_by_strike, cumulative_gex, gex_by_expiration, surface_data, export_dir
):
    """
    Export all computed gamma metrics to CSV files for external analysis.
    
    Creates four CSV files in the export directory:
    1. gex_by_strike: GEX aggregated by strike price
    2. cumulative_gex: Cumulative GEX useful for gamma flip analysis
    3. gex_by_expiration: GEX aggregated by expiration date (term structure)
    4. gex_surface: Full 3D surface data (expiration × strike × GEX)
    
    Each file is timestamped to preserve historical runs.
    
    Args:
        ticker (str): Stock ticker symbol
        gex_by_strike (Series): GEX indexed by strike price
        cumulative_gex (Series): Cumulative GEX indexed by strike
        gex_by_expiration (Series): GEX indexed by expiration date
        surface_data (DataFrame): Surface data with columns [expiration, strike, GEX]
        export_dir (Path/str): Directory where CSV files are saved
    
    Returns:
        None (creates files as side effect)
    
    Example:
        >>> export_analytics_csv("SPX", gex_strike, cum_gex, gex_exp, surface, "data/exports")
        Saved CSV exports to: data/exports
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Export each metric to its own CSV
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


# ============================================================================
# CLI ARGUMENT PARSING
# ============================================================================

def parse_args():
    """
    Parse and return command-line arguments.
    
    Provides a user-friendly interface for customizing the analysis:
    - Ticker selection (required)
    - Caching behavior (refresh vs use cache)
    - Plot output (show/save/both)
    - Analysis parameters (strike window, max DTE, etc.)
    - CSV export options
    
    Returns:
        Namespace: Parsed arguments with attributes:
                   - ticker: Stock symbol
                   - refresh: Force data refresh
                   - cache_ttl_minutes: Cache validity
                   - no_show: Suppress plot windows
                   - no_save: Don't save plots to disk
                   - outdir: Plot output directory
                   - top_n: Top strikes to label
                   - strike_window_pct: Strike window percentage
                   - max_dte: Maximum days-to-expiration
                   - no_export_csv: Skip CSV export
                   - export_dir: CSV export directory
    
    Example:
        >>> args = parse_args()
        >>> print(args.ticker)
        SPX
    """
    parser = argparse.ArgumentParser(description="Compute and plot options gamma exposure.")
    
    # Required argument
    parser.add_argument("--ticker", type=str, help="Underlying ticker symbol, e.g. SPX.")
    
    # Caching options
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
    
    # Plot display/save options
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
    
    # Analysis parameters
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
        help="Maximum days-to-expiration to include in term/surface charts (default: 365). Use 0 to include same-day options only.",
    )
    
    # CSV export options
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


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Get ticker from CLI arg or prompt user
    ticker = args.ticker.upper() if args.ticker else input("Enter desired ticker:").upper()
    
    # Execute the analysis with provided/parsed arguments
    run(
        ticker=ticker,
        refresh=args.refresh,
        cache_ttl_minutes=args.cache_ttl_minutes,
        show_plots=not args.no_show,
        save_plots=not args.no_save,
        outdir=args.outdir,
        top_n=max(1, args.top_n),
        strike_window_pct=max(0.01, min(1.0, args.strike_window_pct)),
        max_dte=max(0, args.max_dte),
        export_csv=not args.no_export_csv,
        export_dir=args.export_dir,
    )
