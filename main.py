"""
Gamma Exposure Tracker (GEX) — CLI entry point.

Fetches dealer gamma exposure from the Unusual Whales API, prints regime
summary and key levels, generates charts, and writes timestamped CSV/JSON
exports under ``data/exports/``.

Architecture
------------
``run()`` delegates to ``gex_core.data_source.fetch_gex_data`` for the UW
fetch, then ``_run_uw()`` handles console output, plotting, and export. The
web dashboard (``web_app.py``) and refresh scheduler (``gex_core.refresh``)
call ``run()`` with ``show_plots=False`` to append new snapshots.

Usage::

    python main.py --ticker SPX
    python main.py --ticker SPX --no-show
    python main.py --ticker SPX --market-date 2026-06-01
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_plt = None
_dates = None


def _matplotlib():
    global _plt, _dates
    if _plt is None:
        import matplotlib.pyplot as plt
        from matplotlib import dates

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
        _plt = plt
        _dates = dates
    return _plt, _dates


ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_CYAN = "\033[96m"
ANSI_GREEN = "\033[92m"
ANSI_MAGENTA = "\033[95m"
ANSI_YELLOW = "\033[93m"
ANSI_RED = "\033[91m"

DATA_DIR = Path("data")
DEFAULT_OUTDIR = Path("img")
DEFAULT_EXPORT_DIR = DATA_DIR / "exports"


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
    print()


def _run_uw(
    ticker,
    show_plots,
    save_plots,
    outdir,
    top_n,
    strike_window_pct,
    export_csv,
    export_dir,
    uw_api_key=None,
    fetched=None,
    market_date=None,
):
    """Print regime summary, charts, and CSV exports from UW aggregates."""
    from gex_core.uw_loader import fetch_uw_gex, fetch_uw_spot_exposures

    if fetched is not None:
        spot_price, agg = fetched.spot, fetched.aggregates
        market_date = fetched.market_date or market_date
    else:
        print(color_text("Fetching data from Unusual Whales API...", ANSI_DIM))
        spot_price, agg = fetch_uw_gex(ticker, api_key=uw_api_key, date=market_date)
        market_date = agg.gex_by_strike.attrs.get("market_date") or market_date
    print(
        f"{color_text('Spot (UW)', ANSI_YELLOW)}: {spot_price:.2f}   "
        f"{color_text('As of', ANSI_DIM)}: {market_date or datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    gex_by_strike = agg.gex_by_strike
    cumulative_gex = agg.cumulative_gex
    gex_by_expiration = agg.gex_by_expiration
    surface_data = agg.surface_data
    total_gex_bn = agg.total_gex_bn

    print_section_header("Total GEX (Unusual Whales)")
    regime = "LONG gamma" if total_gex_bn >= 0 else "SHORT gamma"
    print(f"Total net GEX: {color_text(f'${total_gex_bn:.4f} Bn', ANSI_GREEN)}")

    print_section_header("Gamma Regime")
    print(f"Net gamma regime: {color_text(regime, ANSI_YELLOW)} ({total_gex_bn:.3f} Bn$ / %)")

    if not gex_by_strike.empty:
        call_wall = float(gex_by_strike.idxmax())
        put_wall = float(gex_by_strike.idxmin())
        print(f"Estimated call wall: strike {color_text(str(int(call_wall)), ANSI_GREEN)} ({gex_by_strike.max():.3f})")
        print(f"Estimated put wall:  strike {color_text(str(int(put_wall)), ANSI_RED)} ({gex_by_strike.min():.3f})")

    print_section_header(f"Top {top_n} GEX Strikes (UW)")
    positive = gex_by_strike[gex_by_strike > 0].sort_values(ascending=False).head(top_n)
    negative = gex_by_strike[gex_by_strike < 0].sort_values().head(top_n)
    print(f"{color_text('Signal', ANSI_BOLD):<12} {color_text('Strike', ANSI_BOLD):<10} {color_text('GEX (Bn$ / %)', ANSI_BOLD)}")
    for strike, gex in positive.items():
        print(f"  {color_text('LONG', ANSI_GREEN):<12} {strike:<10.0f} {gex:.3f}")
    for strike, gex in negative.items():
        print(f"  {color_text('SHORT', ANSI_RED):<12} {strike:<10.0f} {gex:.3f}")

    greek_df = gex_by_strike.attrs.get("greek_exposure_df") if hasattr(gex_by_strike, "attrs") else None
    spot_df = gex_by_strike.attrs.get("spot_exposures_df") if hasattr(gex_by_strike, "attrs") else None
    from gex_core.features import estimate_gamma_flip_detailed, resolve_gamma_flip

    gamma_flip_detail = estimate_gamma_flip_detailed(
        spot=spot_price,
        gex_by_strike=gex_by_strike,
        cumulative_gex=cumulative_gex,
        greek_exposure_df=greek_df if isinstance(greek_df, pd.DataFrame) else None,
        spot_exposure_df=spot_df if isinstance(spot_df, pd.DataFrame) else None,
    )
    gamma_flip_strike = resolve_gamma_flip(
        spot=spot_price,
        gex_by_strike=gex_by_strike,
        cumulative_gex=cumulative_gex,
        greek_exposure_df=greek_df if isinstance(greek_df, pd.DataFrame) else None,
        spot_exposure_df=spot_df if isinstance(spot_df, pd.DataFrame) else None,
    )
    print_gamma_flip_estimate(gamma_flip_detail)

    try:
        from gex_core.ai_analyst import analyze_dealer_gamma

        analysis = analyze_dealer_gamma(
            ticker=ticker,
            spot=spot_price,
            gex_by_strike=gex_by_strike,
            cumulative_gex=cumulative_gex,
            total_gex_bn=total_gex_bn,
            gamma_flip=gamma_flip_strike,
        )
        print_section_header("AI Dealer Gamma Analysis")
        bias_color = ANSI_GREEN if analysis.bias == "bullish" else ANSI_RED if analysis.bias == "bearish" else ANSI_YELLOW
        print(color_text(f"Bias: {analysis.bias.upper()}  (confidence {analysis.confidence * 100:.0f}%)", bias_color))
        print()
        print(color_text(analysis.narrative, ANSI_DIM))
        print()
        print(color_text("Predictions:", ANSI_BOLD))
        for i, p in enumerate(analysis.predictions, 1):
            print(f"  {color_text(str(i), ANSI_CYAN)}. {p}")
        print()
        print(color_text("Signals:", ANSI_BOLD))
        for sig in analysis.signals:
            icon = (
                color_text("▲", ANSI_GREEN)
                if sig.sentiment == "bullish"
                else color_text("▼", ANSI_RED)
                if sig.sentiment == "bearish"
                else color_text("◆", ANSI_YELLOW)
            )
            print(f"  {icon} {color_text(sig.label, ANSI_BOLD)}: {sig.value}")
    except Exception:
        pass

    try:
        spot_df = fetch_uw_spot_exposures(ticker, api_key=uw_api_key, date=market_date)
        if not spot_df.empty and "net_gamma_oi" in spot_df.columns:
            print_section_header("Intraday Gamma OI vs Volume (±ATM, relative units)")
            top_idx = spot_df["net_gamma_oi"].abs().sort_values(ascending=False).head(5).index
            best = spot_df.loc[top_idx].sort_values("strike")
            for _, row in best.iterrows():
                net = float(row["net_gamma_oi"])
                c = float(row.get("call_gamma_oi", 0))
                p = float(row.get("put_gamma_oi", 0))
                print(f"  Strike {row['strike']:>8.0f}: net={net:>+14,.0f}  (call={c:>+14,.0f} / put={p:>+14,.0f})")
    except Exception:
        pass

    if show_plots or save_plots:
        plot_gex_profile(
            ticker=ticker,
            gex_by_strike=gex_by_strike,
            show_plots=show_plots,
            save_plots=save_plots,
            outdir=outdir,
            spot=spot_price,
            window_pct=strike_window_pct if strike_window_pct >= 0.05 else 0.10,
        )
        plot_gex_by_strike(
            ticker=ticker,
            spot=spot_price,
            gex_by_strike=gex_by_strike,
            show_plots=show_plots,
            save_plots=save_plots,
            outdir=outdir,
            top_n=top_n,
            strike_window_pct=strike_window_pct,
            cumulative_gex=cumulative_gex,
            gamma_flip=gamma_flip_strike,
        )
        plot_cumulative_gex(
            ticker=ticker,
            cumulative_gex=cumulative_gex,
            show_plots=show_plots,
            save_plots=save_plots,
            outdir=outdir,
            spot=spot_price,
            gamma_flip=gamma_flip_strike,
        )

    if export_csv:
        from gex_core.extended_features import merge_extended_features
        from gex_core.market_features import fetch_cross_asset_returns, fetch_vol_regime

        spot_df = None
        try:
            spot_df = fetch_uw_spot_exposures(ticker, api_key=uw_api_key, date=market_date)
        except Exception:
            spot_df = gex_by_strike.attrs.get("spot_exposures_df") if hasattr(gex_by_strike, "attrs") else None

        extended_payload: dict = {}
        merge_extended_features(
            extended_payload,
            greek_df=greek_df if isinstance(greek_df, pd.DataFrame) else None,
            spot_exposures_df=spot_df if isinstance(spot_df, pd.DataFrame) else None,
            market_date=market_date,
            vol_regime=fetch_vol_regime(),
            cross_asset=fetch_cross_asset_returns(),
        )

        summary = {
            "ticker": ticker.upper(),
            "data_source": "unusual_whales",
            "source": "Unusual Whales API",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "market_date": market_date,
            "spot": float(spot_price),
            "spot_price": float(spot_price),
            "total_gex_bn_per_pct": float(total_gex_bn),
            "net_gamma_regime": regime,
            "extended_features": {k: extended_payload[k] for k in extended_payload},
            "call_wall": {
                "strike": float(gex_by_strike.idxmax()),
                "gex_bn_per_pct": float(gex_by_strike.max()),
            }
            if not gex_by_strike.empty
            else None,
            "put_wall": {
                "strike": float(gex_by_strike.idxmin()),
                "gex_bn_per_pct": float(gex_by_strike.min()),
            }
            if not gex_by_strike.empty
            else None,
            "gamma_flip": gamma_flip_strike,
            "gamma_flip_detail": gamma_flip_detail,
            "uw_endpoint": "spot-exposures/strike",
        }
        from gex_core.snapshot_export import write_snapshot_export

        write_snapshot_export(
            ticker=ticker,
            gex_by_strike=gex_by_strike,
            cumulative_gex=cumulative_gex,
            gex_by_expiration=gex_by_expiration,
            surface_data=surface_data,
            greek_exposure_df=greek_df if isinstance(greek_df, pd.DataFrame) else None,
            summary=summary,
            export_dir=export_dir,
            timestamp=f"{market_date}_000000" if market_date else None,
        )
        print(f"Saved CSV exports to: {export_dir}")


def run(
    ticker,
    show_plots=True,
    save_plots=True,
    outdir=DEFAULT_OUTDIR,
    top_n=5,
    strike_window_pct=0.01,
    export_csv=True,
    export_dir=DEFAULT_EXPORT_DIR,
    uw_api_key=None,
    market_date=None,
    **_ignored,
):
    """
    Fetch UW GEX for *ticker*, print summary, optionally plot and export.

    Extra keyword arguments are accepted for backward compatibility with older
    callers (e.g. ``refresh=True`` from the refresh scheduler) and ignored.
    """
    print_banner(ticker)
    print(color_text("Fetching data from Unusual Whales...", ANSI_DIM))

    from gex_core.data_source import fetch_gex_data

    fetched = fetch_gex_data(ticker, uw_api_key=uw_api_key, market_date=market_date)
    return _run_uw(
        ticker=ticker,
        show_plots=show_plots,
        save_plots=save_plots,
        outdir=outdir,
        top_n=top_n,
        strike_window_pct=strike_window_pct,
        export_csv=export_csv,
        export_dir=export_dir,
        uw_api_key=uw_api_key,
        fetched=fetched,
        market_date=market_date,
    )


def print_gamma_flip_estimate(result):
    """Print the strike where dealer gamma flips near ATM."""
    print_section_header("Gamma Flip")
    if result["flip_strike"] is None:
        print(color_text(f"Gamma flip estimate unavailable: {result['message']}.", ANSI_DIM))
    else:
        print(
            f"Estimated gamma flip strike: {result['flip_strike']:.2f} "
            f"(confidence: {result['confidence']})"
        )
    return result


def build_output_path(base_dir, ticker, plot_name, suffix):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return Path(base_dir) / f"{ticker}_{plot_name}_{timestamp}.{suffix}"


def finalize_plot(fig, output_path, show_plots, save_plots):
    if save_plots:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {output_path}")

    if show_plots:
        plt, _ = _matplotlib()
        try:
            plt.show(block=True)
        except Exception:
            plt.show()


def plot_gex_by_strike(
    ticker,
    spot,
    gex_by_strike,
    show_plots,
    save_plots,
    outdir,
    top_n=5,
    strike_window_pct=0.15,
    cumulative_gex=None,
    gamma_flip=None,
):
    plt, _ = _matplotlib()

    lower = spot * (1 - strike_window_pct)
    upper = spot * (1 + strike_window_pct)
    limit_criteria = (gex_by_strike.index > lower) & (gex_by_strike.index < upper)
    selected = gex_by_strike.loc[limit_criteria]

    fig, ax1 = plt.subplots(figsize=(14, 7))
    colors = np.where(selected.values >= 0, "#00d97e", "#ff4757")
    bars = ax1.bar(selected.index, selected.values, color=colors, alpha=0.88, edgecolor="none", zorder=3)

    ax1.axvline(x=spot, color="#F8D56B", linestyle="--", linewidth=1.5, label=f"Spot {spot:.0f}", zorder=5)

    if gamma_flip is not None:
        ax1.axvline(x=gamma_flip, color="#F97316", linestyle=":", linewidth=1.5, label=f"Flip ~{gamma_flip:.0f}", zorder=5)
        try:
            ax1.annotate(
                f"Flip ~{gamma_flip:.0f}",
                xy=(gamma_flip, ax1.get_ylim()[1] * 0.85),
                xytext=(8, 0),
                textcoords="offset points",
                color="#F97316",
                fontsize=8,
                ha="left",
            )
        except Exception:
            pass

    if cumulative_gex is not None and not cumulative_gex.empty:
        cum_window = cumulative_gex.loc[(cumulative_gex.index >= lower) & (cumulative_gex.index <= upper)]
        if not cum_window.empty:
            ax2 = ax1.twinx()
            ax2.plot(
                cum_window.index,
                cum_window.values,
                color="#60A5FA",
                linewidth=2.5,
                alpha=0.9,
                label="Cumulative GEX",
                zorder=4,
            )
            ax2.axhline(y=0, color="#60A5FA", linestyle=":", linewidth=0.8, alpha=0.4)
            ax2.set_ylabel("Cumulative GEX (Bn$ / %)", color="#60A5FA", fontweight="heavy")
            ax2.tick_params(axis="y", colors="#60A5FA")
            for spine in ax2.spines.values():
                spine.set_visible(False)
            ax2.legend(loc="upper right", fontsize=9)

    ax1.set_facecolor("#0F172A")
    ax1.grid(color="#334155", linestyle="-", alpha=0.25, zorder=0)
    ax1.tick_params(axis="x", rotation=45)
    ax1.set_xlabel("Strike", fontweight="heavy")
    ax1.set_ylabel("Gamma Exposure (Bn$ / %)", fontweight="heavy")
    ax1.set_title(f"{ticker} GEX by Strike", fontweight="heavy")
    ax1.legend(loc="upper left", fontsize=9)
    for spine in ax1.spines.values():
        spine.set_visible(False)

    if not selected.empty:
        labels_to_annotate = selected.abs().sort_values(ascending=False).head(top_n)
        selected_index_values = selected.index.to_numpy()
        for strike in labels_to_annotate.index:
            bar_idx = int(np.where(selected_index_values == strike)[0][0])
            bar = bars[bar_idx]
            ax1.annotate(
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


def plot_gex_profile(
    ticker,
    gex_by_strike,
    show_plots,
    save_plots,
    outdir,
    spot=None,
    window_pct=0.10,
    max_bars=60,
):
    """Periscope-style horizontal bar profile of GEX by strike."""
    plt, _ = _matplotlib()

    if spot is not None and spot > 0:
        lo, hi = spot * (1 - window_pct), spot * (1 + window_pct)
        window = gex_by_strike.loc[(gex_by_strike.index >= lo) & (gex_by_strike.index <= hi)]
        if len(window) < 5:
            window = gex_by_strike
    else:
        window = gex_by_strike

    window = window.sort_index(ascending=True)
    if len(window) > max_bars:
        keep = window.abs().sort_values(ascending=False).head(max_bars).index
        window = window.loc[keep].sort_index(ascending=True)

    y_labels = [str(int(s)) for s in window.index]
    x_vals = window.values.astype(float)
    colors = ["#00d97e" if v >= 0 else "#ff4757" for v in x_vals]

    fig, ax = plt.subplots(figsize=(9, max(6, len(window) * 0.22)))
    ax.barh(y_labels, x_vals, color=colors, edgecolor="none", height=0.7)
    ax.axvline(x=0, color="rgba(255,255,255,0.25)", linewidth=0.8)

    if spot is not None and spot > 0:
        spot_label = str(int(spot))
        if spot_label in y_labels:
            idx = y_labels.index(spot_label)
            ax.axhline(y=idx, color="#f59e0b", linestyle="--", linewidth=1.2, label=f"Spot {int(spot)}")
            ax.legend(fontsize=8, loc="lower right")

    ax.set_facecolor("#07090f")
    fig.patch.set_facecolor("#07090f")
    ax.grid(axis="x", color="rgba(255,255,255,0.05)", linewidth=0.6)
    ax.tick_params(colors="#6e7681", labelsize=9)
    ax.set_xlabel("GEX (Bn$ / %)", color="#c9d1d9", fontsize=10)
    ax.set_ylabel("Strike", color="#c9d1d9", fontsize=10)
    ax.set_title(f"{ticker} Dealer Gamma Profile", color="#c9d1d9", fontsize=12, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    finalize_plot(
        fig,
        build_output_path(outdir, ticker, "gex_profile", "png"),
        show_plots,
        save_plots,
    )


def plot_cumulative_gex(
    ticker,
    cumulative_gex,
    show_plots,
    save_plots,
    outdir,
    spot=None,
    gamma_flip=None,
):
    if cumulative_gex is None or cumulative_gex.empty:
        return

    plt, _ = _matplotlib()
    fig, ax = plt.subplots(figsize=(14, 5))

    x = cumulative_gex.index.to_numpy(dtype=float)
    y = cumulative_gex.values.astype(float)

    ax.fill_between(x, y, 0, where=(y >= 0), color="#00d97e", alpha=0.20, label="Long gamma zone")
    ax.fill_between(x, y, 0, where=(y < 0), color="#ff4757", alpha=0.20, label="Short gamma zone")
    ax.plot(x, y, color="#60A5FA", linewidth=2.5, zorder=4)
    ax.axhline(y=0, color="#94a3b8", linestyle="--", linewidth=0.9)

    if spot is not None:
        ax.axvline(x=spot, color="#F8D56B", linestyle="--", linewidth=1.5, label=f"Spot {spot:.0f}", zorder=5)

    if gamma_flip is not None:
        ax.axvline(x=gamma_flip, color="#F97316", linestyle=":", linewidth=1.5, label=f"Flip ~{gamma_flip:.0f}", zorder=5)
        ax.annotate(
            f"Flip ~{gamma_flip:.0f}",
            xy=(gamma_flip, 0),
            xytext=(10, 18),
            textcoords="offset points",
            color="#F97316",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#F97316", lw=1.2),
        )

    ax.set_facecolor("#0F172A")
    ax.grid(color="#334155", linestyle="-", alpha=0.25)
    ax.set_xlabel("Strike", fontweight="heavy")
    ax.set_ylabel("Cumulative GEX (Bn$ / %)", fontweight="heavy")
    ax.set_title(f"{ticker} Cumulative GEX", fontweight="heavy")
    ax.legend(fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    finalize_plot(
        fig,
        build_output_path(outdir, ticker, "cumulative_gex", "png"),
        show_plots,
        save_plots,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Compute and plot options gamma exposure from Unusual Whales.")
    parser.add_argument("--ticker", type=str, help="Underlying ticker symbol, e.g. SPX.")
    parser.add_argument("--no-show", action="store_true", help="Do not open plot windows.")
    parser.add_argument("--no-save", action="store_true", help="Do not save plots to disk.")
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR), help=f"Plot output directory (default: {DEFAULT_OUTDIR}).")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top GEX strikes to print (default: 5).")
    parser.add_argument(
        "--strike-window-pct",
        type=float,
        default=0.01,
        help="Strike window around spot for charts (default: 0.01 = ±1%%). Maximum 0.50.",
    )
    parser.add_argument("--no-export-csv", action="store_true", help="Disable CSV analytics exports.")
    parser.add_argument(
        "--export-dir",
        type=str,
        default=str(DEFAULT_EXPORT_DIR),
        help=f"CSV export directory (default: {DEFAULT_EXPORT_DIR}).",
    )
    parser.add_argument(
        "--uw-key",
        type=str,
        default=None,
        metavar="KEY",
        help="Unusual Whales API key (overrides UW_API_KEY env var).",
    )
    parser.add_argument(
        "--market-date",
        type=str,
        default=None,
        help="Historical UW market date to fetch in YYYY-MM-DD format.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ticker = args.ticker.upper() if args.ticker else input("Enter desired ticker:").upper()
    run(
        ticker=ticker,
        show_plots=not args.no_show,
        save_plots=not args.no_save,
        outdir=args.outdir,
        top_n=max(1, args.top_n),
        strike_window_pct=min(max(0.0, args.strike_window_pct), 0.50),
        export_csv=not args.no_export_csv,
        export_dir=args.export_dir,
        uw_api_key=args.uw_key,
        market_date=args.market_date,
    )
