"""GEX prediction: KNN, surface similarity, and trained model inference."""

from __future__ import annotations

import json
import math
from datetime import timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from gex_core.features import enrich_snapshot_metrics, safe_float, snapshot_feature_vector
from gex_core.exports import parse_timestamp
from gex_core.models_manifest import load_manifest

MODELS_DIR = Path("models")
DEFAULT_LOOKBACK_DAYS = 7


def _zscore_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (matrix - mean) / std, mean, std


def _weighted_knn_predict(
    train_features: np.ndarray,
    train_targets: dict[str, np.ndarray],
    query: np.ndarray,
    k: int = 4,
    surface_vectors: list[np.ndarray] | None = None,
    query_surface: np.ndarray | None = None,
    surface_weight: float = 0.35,
) -> tuple[dict[str, float], list[int], np.ndarray, float]:
    z_train, mean, std = _zscore_matrix(train_features)
    z_query = (query - mean) / std
    distances = np.linalg.norm(z_train - z_query, axis=1)

    if surface_vectors and query_surface is not None and len(surface_vectors) == len(distances):
        for i, sv in enumerate(surface_vectors):
            denom = max(np.linalg.norm(sv) * np.linalg.norm(query_surface), 1e-12)
            cos_dist = 1.0 - float(np.dot(sv, query_surface) / denom)
            distances[i] = (1 - surface_weight) * distances[i] + surface_weight * cos_dist

    k = min(k, len(distances))
    nn_idx = np.argsort(distances)[:k]
    nn_dist = distances[nn_idx]
    weights = 1.0 / (nn_dist + 1e-6)
    weights = weights / weights.sum()

    predictions = {}
    for key, targets in train_targets.items():
        predictions[key] = float(np.sum(weights * targets[nn_idx]))

    avg_dist = float(nn_dist.mean())
    confidence = max(0.0, min(1.0, 1.0 / (1.0 + avg_dist)))
    return predictions, list(nn_idx), weights, confidence


def _calibrate_confidence(
    raw_confidence: float,
    *,
    train_count: int,
    model_overlay: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Dampen confidence when the recent training window is thin."""
    sample_factor = max(0.0, min(1.0, (train_count - 3) / 25.0))
    model_conf = safe_float(model_overlay.get("confidence"), 0.0) if model_overlay else 0.0
    overlay_factor = 0.15 if model_overlay else 0.0
    calibrated = raw_confidence * (0.45 + 0.40 * sample_factor + overlay_factor)
    if model_conf:
        calibrated = 0.75 * calibrated + 0.25 * model_conf
    calibrated = max(0.0, min(1.0, calibrated))
    return calibrated, {
        "raw_neighbor_confidence": raw_confidence,
        "sample_factor": sample_factor,
        "training_rows": train_count,
        "model_overlay_used": bool(model_overlay),
        "model_overlay_confidence": model_conf or None,
        "method": "neighbor_distance_sample_damped",
    }


def prepare_training_rows(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for i in range(len(history) - 1):
        cur = enrich_snapshot_metrics(history[i].copy())
        nxt = history[i + 1]
        prev = enrich_snapshot_metrics(history[i - 1].copy()) if i > 0 else None
        if prev:
            cur["total_gex_momentum"] = cur["total_gex"] - prev["total_gex"]
            cur["flip_velocity"] = safe_float(cur["gamma_flip"]) - safe_float(prev.get("gamma_flip"), 0.0)
        else:
            cur["total_gex_momentum"] = 0.0
            cur["flip_velocity"] = 0.0

        rows.append(
            {
                "ts": cur["ts"],
                "features": snapshot_feature_vector(cur),
                "surface_vector": cur.get("surface_vector", np.zeros(32)),
                "target_total_gex": nxt["total_gex"],
                "target_delta_gex": nxt["total_gex"] - cur["total_gex"],
                "target_flip": safe_float(nxt.get("gamma_flip"), safe_float(cur.get("gamma_flip"), 0.0)),
                "target_near_term_ratio": nxt["near_term_ratio"],
                "target_zero_dte_ratio": safe_float(nxt.get("zero_dte_ratio"), 0.0),
                "target_term_curvature": safe_float(nxt.get("term_curvature"), 0.0),
                "target_strike": nxt.get("strike"),
                "next_ts": nxt["ts"],
            }
        )
    return rows


def select_recent_history(
    history: list[dict[str, Any]],
    lookback_days: int | None = DEFAULT_LOOKBACK_DAYS,
) -> list[dict[str, Any]]:
    """Return snapshots inside the recent regime window, sorted by timestamp."""
    ordered = sorted(history, key=lambda row: row["ts"])
    if not ordered or not lookback_days or lookback_days <= 0:
        return ordered
    latest = parse_timestamp(ordered[-1]["ts"])
    cutoff = latest - timedelta(days=lookback_days)
    return [row for row in ordered if parse_timestamp(row["ts"]) >= cutoff]


def predict_next_snapshot(
    history: list[dict[str, Any]],
    k: int = 4,
    lookback_days: int | None = DEFAULT_LOOKBACK_DAYS,
) -> dict[str, Any] | None:
    history = select_recent_history(history, lookback_days=lookback_days)
    if len(history) < 4:
        return None

    enriched = [enrich_snapshot_metrics(h.copy()) for h in history]
    train = prepare_training_rows(enriched)
    if len(train) < 3:
        return None

    current = enriched[-1]
    prev = enriched[-2] if len(enriched) > 1 else None
    if prev:
        current["total_gex_momentum"] = current["total_gex"] - prev["total_gex"]
        current["flip_velocity"] = safe_float(current["gamma_flip"]) - safe_float(prev.get("gamma_flip"), 0.0)

    x_train = np.vstack([row["features"] for row in train])
    x_now = snapshot_feature_vector(current)
    surface_vectors = [row["surface_vector"] for row in train]
    query_surface = current.get("surface_vector", np.zeros(32))

    targets = {
        "total_gex": np.array([row["target_total_gex"] for row in train]),
        "delta_gex": np.array([row["target_delta_gex"] for row in train]),
        "flip": np.array([row["target_flip"] for row in train]),
        "near_term_ratio": np.array([row["target_near_term_ratio"] for row in train]),
        "zero_dte_ratio": np.array([row["target_zero_dte_ratio"] for row in train]),
        "term_curvature": np.array([row["target_term_curvature"] for row in train]),
    }

    preds, nn_idx, nn_weights, confidence = _weighted_knn_predict(
        x_train, targets, x_now, k=k,
        surface_vectors=surface_vectors,
        query_surface=query_surface,
    )
    preds["total_gex"] = current["total_gex"] + preds["delta_gex"]

    raw_confidence = confidence
    model_preds = _predict_from_trained_models(current, enriched, lookback_days=lookback_days)
    if model_preds:
        preds["delta_gex"] = 0.5 * preds["delta_gex"] + 0.5 * model_preds.get("delta_gex", preds["delta_gex"])
        preds["total_gex"] = current["total_gex"] + preds["delta_gex"]
        confidence = 0.5 * confidence + 0.5 * model_preds.get("confidence", confidence)

    confidence, confidence_breakdown = _calibrate_confidence(
        confidence,
        train_count=len(train),
        model_overlay=model_preds,
    )

    neighbors = []
    for rank, i in enumerate(nn_idx, start=1):
        src = next(row for row in enriched if row["ts"] == train[i]["ts"])
        neighbors.append(
            {
                "rank": rank,
                "snapshot": src["ts_label"],
                "next_snapshot": _ts_label(train[i]["next_ts"]),
                "distance": float(np.linalg.norm(x_train[i] - x_now)),
                "next_total_gex": float(train[i]["target_total_gex"]),
                "next_delta_gex": float(train[i]["target_delta_gex"]),
            }
        )

    regime_flip_prob = _regime_flip_probability(train, nn_idx, current["total_gex"])
    predicted_strike = pd.Series(dtype=float)
    for weight, idx in zip(nn_weights, nn_idx):
        strike_series = train[idx].get("target_strike")
        if strike_series is None:
            continue
        strike_series = pd.Series(strike_series, dtype=float)
        predicted_strike = predicted_strike.add(strike_series * float(weight), fill_value=0.0)

    return {
        "predicted_total_gex": preds["total_gex"],
        "predicted_delta_gex": preds["delta_gex"],
        "predicted_regime": "LONG gamma" if preds["total_gex"] >= 0 else "SHORT gamma",
        "predicted_flip": preds["flip"],
        "predicted_near_term_ratio": preds["near_term_ratio"],
        "predicted_zero_dte_ratio": preds["zero_dte_ratio"],
        "predicted_term_curvature": preds["term_curvature"],
        "predicted_strike": predicted_strike if not predicted_strike.empty else None,
        "regime_flip_probability": regime_flip_prob,
        "confidence": confidence,
        "raw_confidence": raw_confidence,
        "confidence_breakdown": confidence_breakdown,
        "neighbors": neighbors,
        "model_overlay": model_preds,
        "model_manifest": _model_metadata(current.get("ticker", "SPX")),
        "forecast_horizon": "next_snapshot",
        "term_structure": {
            "current_zero_dte_ratio": safe_float(current.get("zero_dte_ratio"), 0.0),
            "predicted_zero_dte_ratio": preds["zero_dte_ratio"],
            "current_near_term_ratio": safe_float(current.get("near_term_ratio"), 0.0),
            "predicted_near_term_ratio": preds["near_term_ratio"],
            "current_term_curvature": safe_float(current.get("term_curvature"), 0.0),
            "predicted_term_curvature": preds["term_curvature"],
        },
        "training_snapshot_count": len(enriched),
        "training_window_days": lookback_days,
    }


def _ts_label(ts: str) -> str:
    return parse_timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _regime_flip_probability(train: list[dict], nn_idx: list[int], current_total: float) -> float:
    flips = 0
    for i in nn_idx:
        before = train[i]["target_total_gex"] - train[i]["target_delta_gex"]
        after = train[i]["target_total_gex"]
        if (before >= 0) != (after >= 0):
            flips += 1
    base_prob = flips / max(len(nn_idx), 1)
    if current_total != 0 and abs(current_total + np.mean([train[i]["target_delta_gex"] for i in nn_idx])) < abs(current_total) * 0.1:
        base_prob = min(1.0, base_prob + 0.15)
    return float(base_prob)


def similar_setups(
    history: list[dict[str, Any]],
    top_n: int = 5,
    lookback_days: int | None = DEFAULT_LOOKBACK_DAYS,
) -> list[dict[str, Any]]:
    history = select_recent_history(history, lookback_days=lookback_days)
    if len(history) < 3:
        return []

    enriched = [enrich_snapshot_metrics(h.copy()) for h in history]
    current = enriched[-1]
    rows = []
    for i in range(len(enriched) - 1):
        row = enriched[i]
        rows.append((i, row, snapshot_feature_vector(row), row.get("surface_vector", np.zeros(32))))

    current_feat = snapshot_feature_vector(current)
    current_surface = current.get("surface_vector", np.zeros(32))

    matrix = np.vstack([feat for _, _, feat, _ in rows])
    z_matrix, mean, std = _zscore_matrix(matrix)
    z_current = (current_feat - mean) / std
    distances = np.linalg.norm(z_matrix - z_current, axis=1)

    for i, (_, _, _, sv) in enumerate(rows):
        denom = max(np.linalg.norm(sv) * np.linalg.norm(current_surface), 1e-12)
        cos_dist = 1.0 - float(np.dot(sv, current_surface) / denom)
        distances[i] = 0.65 * distances[i] + 0.35 * cos_dist

    idx_sorted = np.argsort(distances)[: min(top_n, len(distances))]
    results = []
    for idx in idx_sorted:
        hist_idx, snap, _, _ = rows[idx]
        next_snap = enriched[hist_idx + 1] if hist_idx + 1 < len(enriched) else None
        delta = (next_snap["total_gex"] - snap["total_gex"]) if next_snap else None
        sim_score = 1.0 / (1.0 + float(distances[idx]))
        results.append(
            {
                "snapshot": snap["ts_label"],
                "distance": float(distances[idx]),
                "similarity": sim_score,
                "regime": snap["regime"],
                "total_gex": snap["total_gex"],
                "next_snapshot": next_snap["ts_label"] if next_snap else None,
                "next_total_gex": next_snap["total_gex"] if next_snap else None,
                "next_delta_gex": delta,
                "next_regime": next_snap["regime"] if next_snap else None,
                "ts": snap["ts"],
            }
        )
    return results


def _manifest_allows_overlay(ticker: str, current_ts: str, lookback_days: int | None) -> bool:
    manifest = load_manifest(ticker)
    if not manifest:
        return False
    training_end = manifest.get("training_end_ts")
    if not training_end or not lookback_days or lookback_days <= 0:
        return bool(training_end)
    try:
        return parse_timestamp(current_ts) - parse_timestamp(training_end) <= timedelta(days=lookback_days)
    except Exception:
        return False


def _predict_from_trained_models(
    current: dict,
    history: list[dict],
    lookback_days: int | None = DEFAULT_LOOKBACK_DAYS,
) -> dict[str, Any] | None:
    ticker = current.get("ticker", "SPX")
    result: dict[str, Any] = {}
    if not _manifest_allows_overlay(ticker, current["ts"], lookback_days):
        return None

    xgb_path = MODELS_DIR / f"{ticker}_gex_delta_model.joblib"
    if xgb_path.exists():
        try:
            bundle = joblib.load(xgb_path)
            feat_cols = bundle["features"]
            row = _build_model_row(current, history, feat_cols)
            if row is not None:
                pred = float(bundle["model"].predict(row)[0])
                result["delta_gex"] = pred
                result["confidence"] = 0.6
        except Exception:
            pass

    lstm_path = MODELS_DIR / f"{ticker}_gex_lstm.keras"
    meta_path = MODELS_DIR / f"{ticker}_gex_lstm_meta.joblib"
    if meta_path.exists() and lstm_path.exists() and len(history) >= 2:
        try:
            import tensorflow as tf
            meta_bundle = joblib.load(meta_path)
            meta = meta_bundle["meta"]
            scaler = meta_bundle["scaler"]
            seq_len = meta["seq_len"]
            feature_cols = meta["features"]
            if len(history) >= seq_len:
                seq_rows = []
                for h in history[-seq_len:]:
                    seq_rows.append(_snapshot_to_feature_dict(enrich_snapshot_metrics(h.copy())))
                seq_df = np.array([[row.get(c, 0.0) for c in feature_cols] for row in seq_rows])
                flat = scaler.transform(seq_df.reshape(-1, len(feature_cols)))
                X = flat.reshape(1, seq_len, len(feature_cols))
                model = tf.keras.models.load_model(str(lstm_path))
                pred = float(model.predict(X, verbose=0)[0][0])
                result["delta_gex"] = pred if "delta_gex" not in result else 0.5 * result["delta_gex"] + 0.5 * pred
                result["confidence"] = max(result.get("confidence", 0.5), 0.55)
        except Exception:
            pass

    return result if result else None


def _model_metadata(ticker: str) -> dict[str, Any] | None:
    manifest = load_manifest(ticker)
    if not manifest:
        return None
    return {
        "ticker": manifest.get("ticker", ticker.upper()),
        "model_type": manifest.get("model_type"),
        "trained_at_utc": manifest.get("trained_at_utc"),
        "feature_version": manifest.get("feature_version"),
        "metrics": manifest.get("metrics", {}),
        "training_start_ts": manifest.get("training_start_ts"),
        "training_end_ts": manifest.get("training_end_ts"),
        "lookback_days": manifest.get("lookback_days"),
        "n_snapshots": manifest.get("n_snapshots"),
        "lag": manifest.get("lag"),
    }


def _snapshot_to_feature_dict(row: dict) -> dict[str, float]:
    strike = row.get("strike", pd.Series(dtype=float))
    mag = strike.abs().sort_values(ascending=False) if len(strike) else pd.Series(dtype=float)
    features = {
        "total_gex_bn": row["total_gex"],
        "pos_gex_bn": row["pos_gex"],
        "neg_gex_bn": row["neg_gex"],
        "gex_mean_bn": row.get("abs_mean", 0.0),
        "gex_std_bn": row["gex_std"],
        "call_wall": safe_float(row.get("call_wall"), 0.0),
        "put_wall": safe_float(row.get("put_wall"), 0.0),
        "wall_spread": safe_float(row.get("wall_spread"), 0.0),
        "gex_concentration": safe_float(row.get("gex_concentration"), 0.0),
        "gex_com": safe_float(row.get("gex_com"), safe_float(row.get("spot"), 0.0)),
        "term_total_gex_bn": row.get("term_total_gex_bn", 0.0),
        "near_term_gex_bn": row.get("near_term_gex_bn", 0.0),
        "near_term_ratio": row["near_term_ratio"],
        "back_term_gex_bn": row.get("back_term_gex_bn", 0.0),
        "zero_dte_gex_bn": row.get("zero_dte_gex_bn", 0.0),
        "zero_dte_ratio": row.get("zero_dte_ratio", 0.0),
        "back_term_ratio": row.get("back_term_ratio", 0.0),
        "term_curvature": row.get("term_curvature", 0.0),
        "surface_mean_m": 0.0,
        "surface_std_m": 0.0,
        "surface_max_m": row.get("surface_peak", 0.0),
        "surface_peak": row.get("surface_peak", 0.0),
        "gamma_flip": safe_float(row.get("gamma_flip"), 0.0),
        "flip_distance_pct": safe_float(row.get("flip_distance_pct"), 0.0),
        "cum_slope_at_spot": safe_float(row.get("cum_slope_at_spot"), 0.0),
        "total_gex_momentum": safe_float(row.get("total_gex_momentum"), 0.0),
        "flip_velocity": safe_float(row.get("flip_velocity"), 0.0),
        "near_term_ratio_delta": safe_float(row.get("near_term_ratio_delta"), 0.0),
        "zero_dte_ratio_delta": safe_float(row.get("zero_dte_ratio_delta"), 0.0),
        "term_curvature_delta": safe_float(row.get("term_curvature_delta"), 0.0),
    }
    for i in range(5):
        features[f"top_gex_{i + 1}"] = float(mag.iloc[i]) if i < len(mag) else 0.0
    return features


def _build_model_row(current: dict, history: list[dict], feat_cols: list[str]) -> np.ndarray | None:
    rows = []
    for h in history[-4:]:
        rows.append(_snapshot_to_feature_dict(enrich_snapshot_metrics(h.copy())))
    if not rows:
        return None
    while len(rows) < 4:
        rows.insert(0, rows[0])
    flat = {}
    for lag, row in enumerate(rows):
        for k, v in row.items():
            flat[f"{k}_lag{lag}"] = v
    try:
        return np.array([[flat[c] for c in feat_cols]])
    except KeyError:
        return None


def load_flow_predictions(feed_path: Path, spot: float, top_n: int = 10) -> dict[str, Any]:
    """Load flow feed and compute predicted GEX delta overlay."""
    from live.aggregator import EnhancedGEXAggregator

    agg = EnhancedGEXAggregator(spot=spot)
    events = []
    if feed_path.exists():
        with feed_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                events.append(json.loads(line))
                try:
                    agg.ingest_event(events[-1])
                except (ValueError, KeyError):
                    continue

    flow_by_strike_bn = {
        float(strike): float(gex) / 1e9 for strike, gex in agg.gex_by_strike.items()
    }
    total_flow_gex = sum(flow_by_strike_bn.values())
    top_signals = agg.top_signals(top_n=top_n)
    return {
        "event_count": len(events),
        "predicted_flow_delta_gex_bn": total_flow_gex,
        "flow_by_strike_bn": flow_by_strike_bn,
        "top_signals": [
            {"strike": s, **sig} for s, sig in top_signals
        ],
    }


def apply_flow_to_prediction(
    prediction: dict[str, Any] | None,
    flow: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Blend option-flow ΔGEX into KNN forecast totals and per-strike gamma profile."""
    if not prediction:
        return prediction
    if not flow or flow.get("event_count", 0) <= 0:
        return prediction

    out = dict(prediction)
    raw_flow_delta = float(flow.get("predicted_flow_delta_gex_bn", 0.0))
    event_count = int(flow.get("event_count", 0))
    flow_weight = min(1.0, math.log1p(event_count) / math.log(101.0))
    flow_delta = raw_flow_delta * flow_weight
    out["base_predicted_delta_gex"] = out["predicted_delta_gex"]
    out["raw_flow_delta_gex"] = raw_flow_delta
    out["flow_blend_weight"] = flow_weight
    out["flow_delta_gex"] = flow_delta
    out["predicted_delta_gex"] = out["predicted_delta_gex"] + flow_delta
    out["predicted_total_gex"] = out["predicted_total_gex"] + flow_delta
    out["predicted_regime"] = "LONG gamma" if out["predicted_total_gex"] >= 0 else "SHORT gamma"
    out["flow_event_count"] = flow["event_count"]
    out["flow_top_signals"] = flow.get("top_signals", [])

    knn_strike = out.get("predicted_strike")
    if knn_strike is not None and not isinstance(knn_strike, pd.Series):
        knn_strike = pd.Series(knn_strike, dtype=float)
    elif knn_strike is None:
        knn_strike = pd.Series(dtype=float)

    flow_strike = pd.Series(flow.get("flow_by_strike_bn", {}), dtype=float) * flow_weight
    if not flow_strike.empty:
        combined = knn_strike.add(flow_strike, fill_value=0.0)
        out["predicted_strike"] = combined if not combined.empty else None
        out["flow_strike"] = flow_strike
        out["knn_strike"] = knn_strike if not knn_strike.empty else None
    else:
        out["flow_strike"] = flow_strike

    return out
