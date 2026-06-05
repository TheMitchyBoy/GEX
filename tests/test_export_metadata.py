from gex_core.export_metadata import build_export_metadata, filter_config_hash


def test_filter_config_hash_stable():
    assert len(filter_config_hash()) == 12


def test_build_export_metadata():
    meta = build_export_metadata(
        "SPX",
        market_date="2026-06-01",
        spot=5000.0,
        total_gex_bn=1.2,
        regime="LONG gamma",
    )
    assert meta["export_schema_version"] == 2
    assert meta["filter_config_hash"]
