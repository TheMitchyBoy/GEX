"""Ticker policy for dashboard-facing workflows."""

from __future__ import annotations

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
