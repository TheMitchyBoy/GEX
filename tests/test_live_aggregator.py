from live.aggregator import EnhancedGEXAggregator, parse_option_symbol


def test_parse_option_symbol():
    parsed = parse_option_symbol("SPX260620C04800000")
    assert parsed["type"] == "C"
    assert parsed["strike"] == 4800


def test_enhanced_aggregator_ingest():
    agg = EnhancedGEXAggregator(spot=4800.0)
    info = agg.ingest_event(
        {
            "option": "SPX260620C04800000",
            "gamma": 0.0001,
            "quantity": 10,
            "side": "buy",
            "spot": 4800.0,
        },
        timestamp=1000.0,
    )
    assert info["strike"] == 4800
    assert "signal" in info
