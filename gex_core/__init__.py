"""
GEX analytics package.

Layers (bottom to top):

1. **Load** — ``uw_loader`` / ``data_source`` fetch Unusual Whales strike GEX.
2. **Persist** — ``exports`` + ``history`` scan timestamped CSV/JSON snapshots.
3. **Features** — ``features`` enriches each snapshot with flip, concentration, surface vectors.
4. **Forecast** — ``predict`` runs weighted KNN with optional trained-model overlay;
   ``calibration``, ``regime``, and ``structural`` refine confidence and attribution.
5. **Present** — ``charts``, ``intelligence``, and the Flask/Streamlit apps render dashboards.
"""

from gex_core.ai_analyst import GammaAnalysis, GammaSignal, analyze_dealer_gamma
from gex_core.data_quality import DataQualityConfig, DataQualityReport, clean_option_data
from gex_core.uw_loader import fetch_uw_gex, fetch_uw_greek_exposure, fetch_uw_spot
from gex_core.decompose import GexDecomposition, decompose_from_snapshots
from gex_core.exports import (
    EXPORT_DIR,
    find_exports_for_ticker,
    load_strike_series,
    parse_timestamp,
)
from gex_core.features import (
    compute_features_from_exports,
    enrich_snapshot_metrics,
    estimate_gamma_flip,
    extract_surface_vector,
    resolve_gamma_flip,
)
from gex_core.pipeline import GexAggregates
from gex_core.predict import (
    apply_flow_to_prediction,
    load_flow_predictions,
    predict_multi_horizon,
    predict_next_snapshot,
    similar_setups,
)
from gex_core.calibration import (
    calibrate_confidence,
    expected_directional_move_pct,
    fit_move_per_delta_gex,
)
from gex_core.market_features import attach_market_features, realized_volatility
from gex_core.regime import classify_regime, model_blend_weight
from gex_core.structural import attribute_last_move, structural_forward_delta

__all__ = [
    "GammaAnalysis",
    "GammaSignal",
    "analyze_dealer_gamma",
    "DataQualityConfig",
    "DataQualityReport",
    "EXPORT_DIR",
    "GexAggregates",
    "GexDecomposition",
    "apply_flow_to_prediction",
    "attach_market_features",
    "attribute_last_move",
    "calibrate_confidence",
    "classify_regime",
    "expected_directional_move_pct",
    "fit_move_per_delta_gex",
    "model_blend_weight",
    "predict_multi_horizon",
    "realized_volatility",
    "structural_forward_delta",
    "clean_option_data",
    "compute_features_from_exports",
    "decompose_from_snapshots",
    "enrich_snapshot_metrics",
    "estimate_gamma_flip",
    "extract_surface_vector",
    "resolve_gamma_flip",
    "find_exports_for_ticker",
    "fetch_uw_gex",
    "fetch_uw_greek_exposure",
    "fetch_uw_spot",
    "load_flow_predictions",
    "load_strike_series",
    "parse_timestamp",
    "predict_next_snapshot",
    "similar_setups",
]
