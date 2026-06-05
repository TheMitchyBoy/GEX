# Forecasting

Predictions live in `gex_core.predict` and power the dashboard “next snapshot” forecast.

## Pipeline overview

1. **Build history** from exports (`gex_core.history`)  
2. **Enrich** each snapshot (flip distance, term structure, surface vector, market context)  
3. **Weighted KNN** on z-scored feature vectors → ΔGEX point estimate + empirical interval  
4. **Optional overlay** from `models/SPX/` when manifest has enough training rows  
5. **Blend** KNN and overlay using volatility-aware `model_blend_weight`  
6. **Calibrate confidence** using walk-forward sign accuracy (`gex_core.calibration`)  
7. **Structural attribution** — spot vs residual ΔGEX (`gex_core.structural`)  

## KNN details

- **Neighbors:** default `k=4`, exponential recency weighting (`RECENCY_DECAY = 0.92`)  
- **Surface similarity:** cosine distance on normalized near-spot GEX vector (blended into distance)  
- **Intervals:** weighted std of neighbor target ΔGEX (~68% band at `INTERVAL_Z = 1.0`)  
- **Typical error:** `neighbor_typical_abs_error` — mean absolute deviation of neighbors from point forecast  

## Model overlay gates

The trained overlay is **skipped** when:

- Manifest `n_train` &lt; `MIN_OVERLAY_TRAIN_ROWS` (8)  
- Manifest is stale vs lookback window  

Primary signal remains KNN until enough labeled snapshots exist.

## Training

**Linear / XGBoost:**

```bash
pip install -r requirements-ml.txt
python scripts/train_gex_model.py --ticker SPX --lookback-days 7
```

**LSTM:**

```bash
python scripts/train_gex_lstm.py --ticker SPX --seq-len 8 --epochs 50
```

Artifacts: `models/SPX/manifest.json`, joblib/keras files under `models/`.

## Backtesting

Walk-forward evaluation:

```bash
python scripts/backtest_gex_prediction.py --ticker SPX
```

CI runs the same script and enforces a minimum sign-accuracy floor when enough samples exist (`GEX_BACKTEST_MIN_ACCURACY`, default `0.30`).

## Flow overlay

When `GEX_FLOW_FEED` contains events, `apply_flow_to_prediction` blends flow ΔGEX into the forecast. See [Live Flow](Live-Flow).

## Dashboard panels

- **Model Accountability** — training depth, overlay status, walk-forward MAE/sign accuracy  
- **Forecast probabilities** — close above/below flip, regime-flip probability  
- **Similar setups** — nearest historical neighbors  
