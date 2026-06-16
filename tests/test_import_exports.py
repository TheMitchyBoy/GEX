"""Tests for CSV export import into PostgreSQL."""

import json

import pandas as pd
import pytest

from gex_core.import_exports import (
    load_export_snapshot,
    normalize_legacy_summary,
    summarize_import_results,
)
from gex_core.import_exports import ImportResult


def test_normalize_legacy_summary_flattens_nested_fields():
    summary = {
        "gamma_flip": {"flip_strike": 6000.0, "confidence": "high"},
        "call_wall": {"strike": 6050.0, "gex_bn_per_pct": 1.0},
    }
    out = normalize_legacy_summary(summary)
    assert out["gamma_flip"] == 6000.0
    assert out["call_wall"] == 6050.0


def test_load_export_snapshot_reads_matched_files(tmp_path):
    ts = "2026-06-15_143000"
    strike = pd.Series({6000.0: 1.5, 6050.0: -2.0, 6100.0: 0.5})
    strike.to_csv(tmp_path / f"SPX_gex_by_strike_{ts}.csv", header=["gex_bn_per_pct"])
    strike.cumsum().to_csv(tmp_path / f"SPX_cumulative_gex_{ts}.csv", header=["cumulative_gex_bn_per_pct"])
    summary = {
        "spot": 6050.0,
        "total_gex_bn_per_pct": 0.0,
        "net_gamma_regime": "LONG gamma",
        "market_date": "2026-06-15",
    }
    (tmp_path / f"SPX_summary_{ts}.json").write_text(json.dumps(summary), encoding="utf-8")

    payload = load_export_snapshot("SPX", ts, export_dir=tmp_path)
    assert payload is not None
    assert len(payload["gex_by_strike"]) >= 3
    assert payload["summary"]["total_gex_bn_per_pct"] == pytest.approx(float(payload["gex_by_strike"].sum()))


def test_summarize_import_results():
    results = [
        ImportResult("SPX", "a", status="imported", written=True),
        ImportResult("SPX", "b", status="skipped_existing", skipped=True),
        ImportResult("SPX", "c", status="error", error="boom"),
    ]
    counts = summarize_import_results(results)
    assert counts == {"imported": 1, "skipped": 1, "errors": 1, "dry_run": 0}
