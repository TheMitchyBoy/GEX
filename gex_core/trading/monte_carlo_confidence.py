"""Monte Carlo sweep over AI advisor confidence thresholds for ROI optimization."""

from __future__ import annotations

import os
from typing import Any

from gex_core.exports import EXPORT_DIR
from gex_core.history import _build_history_impl
from gex_core.trading.config import min_entry_confidence, strong_entry_confidence
from gex_core.trading.monte_carlo_search import TraderConfig, run_config_trial


def _confidence_grid(
    *,
    min_conf_start: float = 0.35,
    min_conf_stop: float = 0.90,
    min_conf_step: float = 0.05,
    strong_levels: list[float] | None = None,
) -> list[TraderConfig]:
    """Build a grid of min-entry and strong-confidence advisor settings."""
    strong_levels = strong_levels or [0.70, 0.75, 0.80, 0.85, 0.90]
    configs: list[TraderConfig] = []

    configs.append(
        TraderConfig(
            name="production",
            env={
                "GEX_TRADER_MIN_ENTRY_CONFIDENCE": f"{min_entry_confidence():.3f}",
                "GEX_TRADER_STRONG_CONFIDENCE": f"{strong_entry_confidence():.3f}",
            },
        )
    )

    level = min_conf_start
    while level <= min_conf_stop + 1e-9:
        for strong in strong_levels:
            configs.append(
                TraderConfig(
                    name=f"min_{level:.2f}_strong_{strong:.2f}",
                    env={
                        "GEX_TRADER_MIN_ENTRY_CONFIDENCE": f"{level:.3f}",
                        "GEX_TRADER_STRONG_CONFIDENCE": f"{strong:.3f}",
                    },
                )
            )
        level = round(level + min_conf_step, 2)

    return configs


def run_confidence_monte_carlo(
    *,
    ticker: str = "SPX",
    lookback_days: int | None = None,
    max_snapshots: int | None = None,
    starting_capital: float | None = None,
    min_conf_start: float = 0.35,
    min_conf_stop: float = 0.90,
    min_conf_step: float = 0.05,
    strong_levels: list[float] | None = None,
) -> dict[str, Any]:
    """Sweep advisor confidence floors and rank by walk-forward ROI."""
    ticker = ticker.upper()
    lookback = lookback_days if lookback_days is not None else int(os.environ.get("GEX_BACKTEST_LOOKBACK_DAYS", "14"))
    max_snaps = max_snapshots if max_snapshots is not None else int(os.environ.get("GEX_BACKTEST_MAX_SNAPSHOTS", "500"))
    capital = starting_capital if starting_capital is not None else float(os.environ.get("GEX_TRADER_ACCOUNT_EQUITY", "500"))

    history = _build_history_impl(
        ticker,
        EXPORT_DIR,
        lookback_days=lookback,
        max_snapshots=max_snaps,
        dedupe_identical_strikes=False,
    )
    if len(history) < 2:
        return {
            "ticker": ticker,
            "message": "Not enough export history for confidence Monte Carlo",
            "snapshots": len(history),
            "trials": [],
        }

    configs = _confidence_grid(
        min_conf_start=min_conf_start,
        min_conf_stop=min_conf_stop,
        min_conf_step=min_conf_step,
        strong_levels=strong_levels,
    )
    results = [
        run_config_trial(cfg, ticker=ticker, history=history, starting_capital=capital)
        for cfg in configs
    ]
    ranked = sorted(results, key=lambda row: row["score"], reverse=True)
    with_trades = [r for r in ranked if r["total_trades"] > 0]
    profitable = [r for r in ranked if float(r.get("return_pct") or 0) > 0]

    best_roi = max(with_trades, key=lambda r: float(r.get("return_pct") or -999), default=None)

    return {
        "ticker": ticker,
        "lookback_days": lookback,
        "starting_capital": capital,
        "snapshots": len(history),
        "date_from": history[0]["ts"],
        "date_to": history[-1]["ts"],
        "trials_run": len(results),
        "trials_with_trades": len(with_trades),
        "trials_profitable": len(profitable),
        "production": next((r for r in results if r["name"] == "production"), None),
        "best": ranked[0] if ranked else None,
        "best_roi": best_roi,
        "best_profitable": profitable[0] if profitable else None,
        "top": ranked[: min(15, len(ranked))],
        "grid": {
            "min_conf_start": min_conf_start,
            "min_conf_stop": min_conf_stop,
            "min_conf_step": min_conf_step,
            "strong_levels": strong_levels or [0.70, 0.75, 0.80, 0.85, 0.90],
        },
    }


def summarize_confidence_monte_carlo(summary: dict[str, Any]) -> dict[str, Any]:
    """Compact summary for LLM / API consumers."""
    best = summary.get("best_roi") or summary.get("best_profitable") or summary.get("best")
    prod = summary.get("production")
    return {
        "ticker": summary.get("ticker"),
        "snapshots": summary.get("snapshots"),
        "window": {"from": summary.get("date_from"), "to": summary.get("date_to")},
        "trials_run": summary.get("trials_run"),
        "trials_with_trades": summary.get("trials_with_trades"),
        "trials_profitable": summary.get("trials_profitable"),
        "grid": summary.get("grid"),
        "production": _trial_row(prod),
        "best_roi": _trial_row(best),
        "top": [_trial_row(row) for row in (summary.get("top") or [])[:8]],
        "message": summary.get("message"),
    }


def _trial_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    env = (row.get("config") or {}).get("env") or {}
    return {
        "name": row.get("name"),
        "min_entry_confidence": float(env.get("GEX_TRADER_MIN_ENTRY_CONFIDENCE", 0)),
        "strong_confidence": float(env.get("GEX_TRADER_STRONG_CONFIDENCE", 0)),
        "return_pct": row.get("return_pct"),
        "total_pnl_usd": row.get("total_pnl_usd"),
        "win_rate": row.get("win_rate"),
        "total_trades": row.get("total_trades"),
        "max_drawdown_pct": row.get("max_drawdown_pct"),
        "skipped_low_confidence": row.get("skipped_low_confidence"),
        "score": row.get("score"),
    }
