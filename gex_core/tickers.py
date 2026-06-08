"""Ticker policy for dashboard-facing workflows."""

from __future__ import annotations

from pathlib import Path

PRIMARY_TICKER = "SPX"
SUPPORTED_TICKERS = (PRIMARY_TICKER,)


def supported_tickers() -> list[str]:
    """Return dashboard-supported tickers in display order."""
    return list(SUPPORTED_TICKERS)


def is_supported_ticker(ticker: str | None) -> bool:
    return (ticker or "").upper() in SUPPORTED_TICKERS


def normalize_ticker(ticker: str | None = None) -> str:
    """Normalize any dashboard ticker request to the SPX dashboard symbol."""
    return PRIMARY_TICKER


def find_available_tickers(export_dir: Path | None = None) -> list[str]:
    """Tickers with export history, falling back to supported list."""
    from gex_core.exports import EXPORT_DIR
    from gex_core.history import list_tickers

    root = export_dir or EXPORT_DIR
    tickers = list_tickers(root)
    return tickers if PRIMARY_TICKER in tickers else supported_tickers()
