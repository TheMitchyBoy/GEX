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
plt.style.use("seaborn-dark")
for param in ["figure.facecolor", "axes.facecolor", "savefig.facecolor"]:
    plt.rcParams[param] = "#212946"
for param in ["text.color", "axes.labelcolor", "xtick.color", "ytick.color"]:
    plt.rcParams[param] = "0.9"

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
    
    # Step 1: Fetch and parse option data
    spot_price, option_data = scrape_data(
        ticker=ticker, refresh=refresh, cache_ttl_minutes=cache_ttl_minutes
    )
    
    # Step 2: Compute aggregate gamma exposure and print summary statistics
    compute_total_gex(spot_price, option_data)
    print_regime_summary(option_data)
    print_key_gex_levels(option_data, top_n=top_n)
    
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
    option_data = pd.DataFrame(payload["data"]["options"])
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
    return spot_price, fix_option_data(option_data)


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
    data["type"] = data.option.str.extract(r"\d([A-Z])\d")
    
    # Extract strike price (numeric value after type letter)
    # Pattern: digit, [A-Z], then capture digits (strike), then 3 more digits (cents)
    data["strike"] = data.option.str.extract(r"\d[A-Z](\d+)\d\d\d").astype(int)
    
    # Extract expiration date (first 6 digits in YYMMDD format)
    # Pattern: [A-Z], then capture 6 digits
    data["expiration"] = data.option.str.extract(r"[A-Z](\d+)").astype(str)
    
    # Convert YYMMDD string to datetime for sorting and analysis
    data["expiration"] = pd.to_datetime(data["expiration"], format="%y%m%d")
    
    return data


# ============================================================================
# GEX COMPUTATION
# ============================================================================

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
    # Compute gross gamma exposure for each option
    # Units: notional dollar exposure per 1% spot price move
    data["GEX"] = spot * data.gamma * data.open_interest * contract_size * spot * 0.01

    # Apply sign convention: calls are positive (dealers long calls), puts are negative
    # (dealers short puts). This aligns with dealer delta hedging behavior.
    type_multiplier = np.where(data["type"] == "P", -1, 1)
    data["GEX"] = data["GEX"] * type_multiplier
    
    # Print aggregate GEX in billions
    total_gex_bn = round(data.GEX.sum() / 10 ** 9, 4)
    print(f"Total notional GEX: ${total_gex_bn} Bn")


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
    print(f"Net gamma regime: {regime} ({net_gex:.3f} Bn$ / %)")

    # Find call wall (max positive GEX) and put wall (min negative GEX)
    if not gex_by_strike.empty:
        call_wall_strike = gex_by_strike.idxmax()
        put_wall_strike = gex_by_strike.idxmin()
        print(f"Estimated call wall: strike {call_wall_strike} ({gex_by_strike.max():.3f})")
        print(f"Estimated put wall: strike {put_wall_strike} ({gex_by_strike.min():.3f})")


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
    - Display in window
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
        plt.show()
    else:
        plt.close(fig)


# ============================================================================
# ANALYSIS: GAMMA BY STRIKE
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
    fig, ax = plt.subplots()
    colors = np.where(selected.values >= 0, "#38E07A", "#FE53BB")
    bars = ax.bar(selected.index, selected.values, color=colors, alpha=0.7)
    
    # Styling
    ax.grid(color="#2A3459")
    ax.tick_params(axis="x")
    ax.tick_params(axis="y")
    ax.set_xlabel("Strike", fontweight="heavy")
    ax.set_ylabel("Gamma Exposure (Bn$ / %)", fontweight="heavy")
    ax.set_title(f"{ticker} GEX by strike", fontweight="heavy")
    
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
    
    Returns:
        Series: GEX indexed by expiration date
    
    Example:
        >>> gex_by_exp = compute_gex_by_expiration("SPX", options_df, True, True, "img", max_dte=90)
        >>> print(gex_by_exp.head())
    """
    
    # Step 1: Filter to expirations within max_dte
    selected_date = datetime.today() + timedelta(days=max_dte)
    data = data.loc[data.expiration < selected_date]

    # Step 2: Aggregate GEX by expiration date across all strikes
    gex_by_expiration = data.groupby("expiration")["GEX"].sum() / 10**9

    # Step 3: Create bar chart showing term structure
    fig, ax = plt.subplots()
    ax.bar(
        gex_by_expiration.index,
        gex_by_expiration.values,
        color="#FE53BB",
        alpha=0.5,
    )
    
    # Styling
    ax.grid(color="#2A3459")
    ax.tick_params(axis="x", rotation=45)  # Rotate x-axis labels for readability
    ax.tick_params(axis="y")
    ax.set_xlabel("Expiration date", fontweight="heavy")
    ax.set_ylabel("Gamma Exposure (Bn$ / %)", fontweight="heavy")
    ax.set_title(f"{ticker} GEX by expiration", fontweight="heavy")
    
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
        strike_window_pct (float): Percentage window around spot for strike axis
    
    Returns:
        DataFrame: Surface data with columns ['expiration', 'strike', 'GEX']
    
    Example:
        >>> surface = print_gex_surface("SPX", 4800, options_df, True, True, "img")
    """
    
    # Step 1: Apply filters (date and strike range)
    selected_date = datetime.today() + timedelta(days=max_dte)
    lower = spot * (1 - strike_window_pct)
    upper = spot * (1 + strike_window_pct)
    limit_criteria = (
        (data.expiration < selected_date)
        & (data.strike > lower)
        & (data.strike < upper)
    )
    data = data.loc[limit_criteria]

    # Step 2: Aggregate GEX by [expiration, strike] pairs
    # Convert to millions for better numerical scale on Z-axis
    data = data.groupby(["expiration", "strike"])["GEX"].sum() / 10**6
    data = data.reset_index()

    # Step 3: Create 3D surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    
    # Plot triangular surface using the data points
    # seismic_r colormap: red = positive GEX, blue = negative GEX
    ax.plot_trisurf(
        data["strike"],
        dates.date2num(data["expiration"]),  # Convert dates to numeric for 3D plotting
        data["GEX"],
        cmap="seismic_r",
    )
    
    # Format Y-axis (expiration) with proper date labels
    ax.yaxis.set_major_formatter(dates.AutoDateFormatter(ax.xaxis.get_major_locator()))
    
    # Labeling
    ax.set_ylabel("Expiration date", fontweight="heavy")
    ax.set_xlabel("Strike Price", fontweight="heavy")
    ax.set_zlabel("Gamma (M$ / %)", fontweight="heavy")
    
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
        help="Maximum days-to-expiration to include in term/surface charts (default: 365).",
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
        max_dte=max(1, args.max_dte),
        export_csv=not args.no_export_csv,
        export_dir=args.export_dir,
    )
