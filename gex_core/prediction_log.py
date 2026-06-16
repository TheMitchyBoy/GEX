"""Log LLM / agent predictions and reconcile against the next realized snapshot."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from gex_core.exports import parse_timestamp
from gex_core.features import safe_float

logger = logging.getLogger(__name__)

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect():
    from gex_core.db import get_connection
    from gex_core.trading.journal import db_path

    return get_connection(group="predictions", sqlite_path=db_path())


def log_llm_prediction(
    *,
    ticker: str,
    source: str,
    prediction: dict[str, Any],
    snapshot_ts: str | None = None,
    market_date: str | None = None,
) -> int | None:
    """Persist one forecast anchored to a snapshot timestamp."""
    ticker = ticker.upper()
    payload = {
        "predicted_regime": prediction.get("predicted_regime") or prediction.get("regime"),
        "predicted_delta_gex_bn": prediction.get("predicted_delta_gex_bn")
        or prediction.get("predicted_delta_gex"),
        "predicted_total_gex_bn": prediction.get("predicted_total_gex_bn"),
        "spot_bias": prediction.get("spot_bias") or prediction.get("bias"),
        "confidence": prediction.get("confidence"),
        "gamma_flip": prediction.get("gamma_flip"),
        "bias": prediction.get("bias"),
        "summary": prediction.get("summary"),
        "plays": prediction.get("plays"),
        "llm_enhanced": prediction.get("llm_enhanced"),
        "prediction_source": prediction.get("prediction_source") or prediction.get("source"),
    }
    try:
        with _connect() as conn:
            from gex_core.db import insert_returning_id

            prediction_id = insert_returning_id(
                conn,
                """
                INSERT INTO llm_predictions (
                    ticker, source, snapshot_ts, market_date, created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker,
                    source,
                    snapshot_ts,
                    (market_date or "")[:10] or None,
                    _now_iso(),
                    json.dumps(payload),
                ),
            )
            conn.commit()
            return prediction_id
    except Exception:
        logger.exception("Failed to log LLM prediction for %s", ticker)
        return None


def _outcome_metrics(predicted: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    pred_delta = safe_float(predicted.get("predicted_delta_gex_bn"), 0.0)
    actual_delta = safe_float(actual.get("delta_gex_bn"), 0.0)
    sign_hit = (pred_delta >= 0) == (actual_delta >= 0) if pred_delta != 0 or actual_delta != 0 else True

    spot0 = safe_float(actual.get("spot_before"), 0.0)
    spot1 = safe_float(actual.get("spot_after"), 0.0)
    spot_move = (spot1 - spot0) / spot0 if spot0 > 0 and spot1 > 0 else 0.0
    bias = str(predicted.get("spot_bias") or predicted.get("bias") or "neutral").lower()
    bias_hit = None
    if bias in {"bullish", "long"} and spot_move != 0:
        bias_hit = spot_move > 0
    elif bias in {"bearish", "short"} and spot_move != 0:
        bias_hit = spot_move < 0
    elif bias in {"neutral", "mean_reversion", "momentum"}:
        bias_hit = abs(spot_move) < 0.004

    pred_regime = str(predicted.get("predicted_regime") or "").upper()
    actual_regime = str(actual.get("regime") or "").upper()
    regime_hit = None
    if pred_regime and actual_regime:
        regime_hit = ("LONG" in pred_regime) == ("LONG" in actual_regime)

    conf = safe_float(predicted.get("confidence"), 0.0)
    return {
        "delta_mae": abs(pred_delta - actual_delta),
        "sign_hit": sign_hit,
        "bias_hit": bias_hit,
        "regime_hit": regime_hit,
        "confidence": conf,
        "spot_move_pct": round(spot_move * 100, 4),
    }


def _next_snapshot_ts(ticker: str, anchor_ts: str) -> str | None:
    from gex_core.history import list_timestamps

    timestamps = list_timestamps(ticker)
    if not timestamps:
        return None
    anchor_dt = parse_timestamp(anchor_ts)
    for ts in timestamps:
        if parse_timestamp(ts) > anchor_dt:
            return ts
    return None


def reconcile_llm_predictions(ticker: str, *, latest_ts: str | None = None) -> int:
    """Resolve open predictions once the next snapshot after the anchor is available."""
    from gex_core.history import get_latest_ts, load_snapshot_at_ts

    ticker = ticker.upper()
    _ = latest_ts or get_latest_ts(ticker)

    resolved = 0
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, snapshot_ts, payload_json FROM llm_predictions
            WHERE ticker = ? AND resolved_at IS NULL
            ORDER BY created_at ASC
            """,
            (ticker,),
        ).fetchall()

        for row in rows:
            anchor_ts = row["snapshot_ts"]
            if not anchor_ts:
                continue
            next_ts = _next_snapshot_ts(ticker, anchor_ts)
            if not next_ts:
                continue

            before = load_snapshot_at_ts(ticker, anchor_ts)
            after = load_snapshot_at_ts(ticker, next_ts)
            if not before or not after:
                continue

            try:
                predicted = json.loads(row["payload_json"])
            except json.JSONDecodeError:
                continue

            actual = {
                "snapshot_ts": next_ts,
                "spot_before": before.get("spot"),
                "spot_after": after.get("spot"),
                "regime": after.get("regime"),
                "total_gex_bn": after.get("total_gex"),
                "delta_gex_bn": safe_float(after.get("total_gex"), 0.0)
                - safe_float(before.get("total_gex"), 0.0),
            }
            outcome = _outcome_metrics(predicted, actual)
            conn.execute(
                """
                UPDATE llm_predictions
                SET resolved_at = ?, actual_json = ?, outcome_json = ?
                WHERE id = ?
                """,
                (_now_iso(), json.dumps(actual), json.dumps(outcome), row["id"]),
            )
            resolved += 1
            market_date = (before.get("market_date") or "")[:10]
            if market_date:
                try:
                    from gex_core.daily_quality import update_prediction_accuracy_daily

                    update_prediction_accuracy_daily(ticker, market_date=market_date, outcome=outcome)
                except Exception:
                    logger.debug("prediction accuracy daily rollup failed", exc_info=True)
        conn.commit()
    return resolved


def get_llm_calibration_stats(
    ticker: str,
    *,
    source: str | None = None,
    min_samples: int | None = None,
) -> dict[str, Any]:
    """Aggregate resolved prediction outcomes for confidence calibration."""
    ticker = ticker.upper()
    min_samples = min_samples if min_samples is not None else int(os.environ.get("GEX_LLM_CALIB_MIN_SAMPLES", "5"))
    query = """
        SELECT outcome_json, payload_json FROM llm_predictions
        WHERE ticker = ? AND resolved_at IS NOT NULL
    """
    params: list[Any] = [ticker]
    if source:
        query += " AND source = ?"
        params.append(source)
    query += " ORDER BY resolved_at DESC LIMIT 500"

    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()

    if not rows:
        return {"n": 0, "accuracy": None, "sign_accuracy": None, "bias_accuracy": None, "avg_confidence": None}

    sign_hits = []
    bias_hits = []
    confidences = []
    for row in rows:
        try:
            outcome = json.loads(row["outcome_json"])
            payload = json.loads(row["payload_json"])
        except json.JSONDecodeError:
            continue
        if outcome.get("sign_hit") is not None:
            sign_hits.append(bool(outcome["sign_hit"]))
        if outcome.get("bias_hit") is not None:
            bias_hits.append(bool(outcome["bias_hit"]))
        confidences.append(safe_float(payload.get("confidence"), safe_float(outcome.get("confidence"), 0.0)))

    n = len(sign_hits)
    empty = {"n": 0, "accuracy": None, "sign_accuracy": None, "bias_accuracy": None, "avg_confidence": None}
    if n < min_samples:
        return {**empty, "n": n}

    sign_accuracy = sum(sign_hits) / n if sign_hits else None
    bias_accuracy = sum(bias_hits) / len(bias_hits) if bias_hits else None
    avg_confidence = sum(confidences) / len(confidences) if confidences else None
    return {
        "n": n,
        "accuracy": sign_accuracy,
        "sign_accuracy": sign_accuracy,
        "bias_accuracy": bias_accuracy,
        "avg_confidence": avg_confidence,
        "confidence_accuracy_gap": (
            abs(avg_confidence - sign_accuracy)
            if avg_confidence is not None and sign_accuracy is not None
            else None
        ),
    }


def calibrated_llm_confidence(raw_confidence: float, ticker: str, *, source: str | None = None) -> float:
    from gex_core.calibration import calibrate_confidence

    stats = get_llm_calibration_stats(ticker, source=source)
    return calibrate_confidence(raw_confidence, stats.get("sign_accuracy"), stats.get("n", 0) or 0)
