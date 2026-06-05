# GEX improvement ideas

A prioritized, grounded backlog produced from a full review of `gex_core/`, the
web/dashboard layer, scripts, and ops config. Each item cites the relevant file
(and line where useful) so it can be picked up directly. Items are grouped by
theme and tagged with rough impact/effort.

Legend: **Impact** = user/operational value, **Effort** = relative size.

---

## 1. Security & hardening (do first)

1. **Disable Flask debug in the dev entrypoint.** `web_app.py:799` runs
   `APP.run(..., debug=True)` and also binds port `8501` while prod/gunicorn uses
   `8080` (`Procfile`, `Dockerfile:17`). Gate debug behind a `FLASK_DEBUG` env var
   and align the dev port with prod. **Impact: high · Effort: low**

2. **Protect state-changing GET routes.** `force_refresh`, `dispatch_alerts`, and
   `/ticker/<t>/bootstrap` (`web_app.py:352`, `:586`, `:651`) mutate state / spend
   UW API quota and are unauthenticated. `GET` side effects can be triggered via
   `<img>`/links/crawlers. Convert dispatch + bootstrap to `POST` with a CSRF/secret
   token, and require a token for forced refresh. **Impact: high · Effort: med**

3. **Don't auto-dispatch alerts on every page render.** With
   `GEX_ALERT_AUTO_DISPATCH=1`, each `ticker_page` render calls
   `maybe_dispatch_alerts` (`web_app.py:586`); only the cooldown/dedupe logic in
   `gex_core/alert_dispatch.py` prevents webhook spam. Move auto-dispatch to the
   scheduler/background job only. **Impact: high · Effort: low**

4. **Validate the webhook URL (SSRF guard).** `dispatch_alerts_to_webhook`
   (`gex_core/intelligence.py:742`) POSTs to `GEX_ALERT_WEBHOOK_URL` with no scheme
   or host checks. Enforce HTTPS, block private/link-local ranges
   (`127.0.0.0/8`, `169.254.0.0/16`, RFC1918), and optionally require a signed
   header. **Impact: med · Effort: low**

5. **Pin front-end CDN assets + add SRI.** `templates/ticker.html:7-8` loads
   `plotly-latest.min.js` (unpinned, deprecated) and Bootstrap without
   subresource-integrity hashes. Pin exact versions and add `integrity`/`crossorigin`.
   **Impact: med · Effort: low**

6. **Harden pickle/model loading.** `predict.py` uses `joblib.load` (≈`:495`,`:510`)
   which deserializes pickle from `models/`. Document/verify that `models/` is only
   writable by trusted CI, and consider signing manifests. **Impact: med · Effort: low**

7. **Avoid leaking config via `/health`.** `gex_core/system_status.py:40-56` exposes
   `index_db` path and webhook/auto-dispatch booleans to unauthenticated callers.
   Trim to non-sensitive fields or require auth. **Impact: low · Effort: low**

---

## 2. Correctness bugs

1. **`front_term_gex_bn` duplicates `zero_dte_gex_bn`.** `features.py:125-126` sets
   `front_term_gex_bn = zero_dte` and `front_term_ratio = zero_dte/denom`, almost
   certainly a copy-paste error (should use a front-month aggregate). Verify intended
   semantics and fix. **Impact: med · Effort: low**

2. **Streamlit renders a duplicate dashboard block.** `streamlit_app.py:648-759`
   repeats the snapshot/heatmap/strike UI already rendered at `:558-646`, so users
   see the controls twice. Delete the duplicate or extract a single shared function.
   **Impact: med · Effort: low**

3. **cwd-relative paths break under gunicorn/docker.** `predict.py:37`
   (`MODELS_DIR = Path("models")`), `models_manifest.py:10`, `decompose.py:27`
   (`Path("data")`), `web_app.py` `IMG_DIR = Path("img")`, and
   `streamlit_app.py` `Path("data/exports")` resolve from the process cwd, unlike
   `gex_core/exports.py` which resolves from `__file__`. Standardize all on
   repo-root resolution. **Impact: med · Effort: low**

4. **Deprecated `datetime.utcnow()`.** Used in `main.py:254` and
   `system_status.py:56`; replace with `datetime.now(timezone.utc)` for
   timezone-aware UTC. **Impact: low · Effort: low**

5. **Naive date math for expiry/DTE filters.** `pipeline.py:61` and
   `data_quality.py:123` use `datetime.today()` with no timezone/market-calendar
   awareness, which can mis-bucket 0DTE near session boundaries. **Impact: low · Effort: med**

---

## 3. Reliability & error handling

1. **Replace silent `except Exception: pass`/swallow patterns with structured
   logging.** Present in `main.py:200,214,652`, `refresh.py:68,78`,
   `system_status.py:37`, `charts.py:313,681`, `live/aggregator.py:32`,
   `scripts/backtest_features.py:72`. Each hides real failures (AI, SQLite upsert,
   chart build, index sync). Log with context; keep optional features non-fatal but
   visible. **Impact: high · Effort: med**

2. **Surface partial-failure states in the UI.** `web_app.py` swallows UW refresh,
   prediction, flow-overlay, and backtest errors (≈`:121`,`:521`,`:528`,`:545`) and
   shows stale data with no indicator. Add an explicit "live feed unavailable / using
   last snapshot" banner. **Impact: med · Effort: med**

3. **Atomic + locked writes for shared state files.** `.alert_dispatch_state.json`
   and `live/state.py` write in place with no locking, racing under concurrent
   refresh + manual dispatch. Use temp-file + `os.replace` and a lock. **Impact: med · Effort: low**

4. **Don't mask missing export files.** `history.py:175` falls back to the strike
   file when cumulative is absent; `decompose.py:169-181` returns a zero
   decomposition when the contract cache is missing — both silently degrade. Log a
   warning and/or flag the snapshot as incomplete. **Impact: low · Effort: low**

5. **Validate scheduler/test isolation.** `web_app.py:795` starts APScheduler at
   import time; CI imports `web_app` without setting `GEX_DISABLE_SCHEDULER=1`, so
   tests may spawn background UW calls. Set it in CI and lazy-start the scheduler.
   **Impact: med · Effort: low**

---

## 4. Performance

1. **Stop re-indexing every export on each sync.** `storage.py:129-169`
   `sync_ticker_exports` re-reads/upserts all exports per call (O(exports × JSON)),
   and it runs from `/health` and history builds. Track an indexed watermark / only
   ingest new timestamps. **Impact: high · Effort: med**

2. **Cache loaded ML models.** `predict.py` reloads joblib/LSTM models on every
   prediction (LSTM ≈`:532`). Memoize by manifest hash so dashboard polling doesn't
   reload TensorFlow each call. **Impact: high · Effort: low**

3. **Reduce redundant UW HTTP calls.** A CLI run fetches greek+spot in
   `fetch_uw_gex`, then `main.py:204` hits `fetch_uw_spot_exposures` again. The two
   UW calls in `uw_loader.py:268-272` are sequential. Skip the redundant call when
   spot is known and/or parallelize. **Impact: med · Effort: med**

4. **Add retries/backoff + rate-limit handling to the UW client.**
   `uw_loader.py:117` is a single `requests.get` with a 15s timeout and no retry —
   one transient failure aborts a refresh. **Impact: med · Effort: low**

5. **Fix O(n²) hot loops.** KNN neighbor lookup uses
   `next(row for row in enriched ...)` inside the neighbor loop (`predict.py:267`);
   `structural_forward_delta` recomputes `attribute_last_move` per step
   (`structural.py:58-62`); `features.py:177-179` bins strikes in a Python loop
   (use `np.digitize`). **Impact: med · Effort: med**

6. **Move blocking work out of request handlers.** `ticker_page`
   (`web_app.py:342-648`, ~306 lines) and `_ticker_api_payload` synchronously run
   UW exports, yfinance calls, prediction, and a 7s webhook POST on the gunicorn
   worker — with only 1 worker / 2 threads (`Procfile`/`Dockerfile`). Offload
   `force_refresh`/dispatch to a background job/queue. **Impact: high · Effort: high**

7. **Avoid per-snapshot VIX blocking I/O.** `market_features.py:136-141` fetches VIX
   over the network in the forecast hot path and stamps one value on all rows. Cache
   it and make it truly per-snapshot. **Impact: low · Effort: med**

---

## 5. Architecture & duplication

1. **Decouple the scheduler from `main.run`.** `refresh.py:56` imports `main.run`,
   dragging matplotlib/argparse into the background refresh path. Extract a thin
   `gex_core.runner.export_snapshot()` with no plotting deps. **Impact: med · Effort: med**

2. **Slim `main.py` (710 lines).** Move plotting/export into `gex_core` (reuse
   `charts.py`/`exports.py`) and delete the duplicate `estimate_gamma_flip`
   (`main.py:326-360` vs `features.py:33-52`). **Impact: med · Effort: med**

3. **De-duplicate shared helpers.** `safe_float` exists in both `charts.py:27` and
   `features.py`; OCC parsing exists in both `data_quality.py` and
   `live/aggregator.py` (the latter weaker); history-building is reimplemented in
   `scripts/backtest_gex_prediction.py:40-80` instead of using
   `gex_core.history.build_history`. **Impact: med · Effort: med**

4. **Converge the two dashboards.** Flask (`web_app.py`) and Streamlit
   (`streamlit_app.py`) duplicate ticker discovery, chart building (Streamlit inlines
   Plotly instead of using `gex_core.charts`), UW fetch, color tokens, and export-dir
   resolution. Either share `gex_core.charts` in both or document which is canonical.
   **Impact: med · Effort: high**

5. **Extract a shared data-assembly layer** used by `ticker_page` and the JSON APIs
   (`_ticker_api_payload`) to eliminate parallel logic. **Impact: med · Effort: med**

6. **Externalize magic constants.** Many tuned values are hardcoded: regime
   thresholds (`ai_analyst.py:122-141`), decomposition 60/40 split
   (`decompose.py:189-191`), blend weights (`predict.py`, `regime.py:67-72`),
   scenario sensitivities (`intelligence.py:29-30`), UW put-sign heuristic `0.55`
   (`uw_loader.py:87-90`). Move to config with documented defaults. **Impact: low · Effort: med**

---

## 6. Testing gaps

Highest-value missing coverage (modules with **no** direct tests):

- `pipeline.py` — the core GEX formula. **(high priority)**
- `main.py` — CLI/export path, `estimate_gamma_flip` confidence logic.
- `decompose.py`, `ai_analyst.py`, `backtest_metrics.py`, `models_manifest.py`.
- `intelligence.generate_alerts` / `simulate_spot_scenario` /
  `dispatch_alerts_to_webhook`.
- `predict.similar_setups`, `_predict_from_trained_models`, multi-horizon edge cases.
- `refresh.is_snapshot_stale` / `refresh_ticker` integration (only mocked today).
- `live/ingest.py`, `live/state.py`.
- Web routes beyond the smoke test: `force_refresh`, `dispatch_alerts`, exports
  route, bootstrap POST.

Add **lint + type checks** to CI (ruff/mypy) and a **Docker build smoke test** —
`ci.yml` currently runs unit tests + backtest gate only. **Impact: high · Effort: med**

---

## 7. Deployment & CI

1. **Add `.dockerignore`.** `Dockerfile:12` `COPY . .` bakes in `.git`,
   `data/exports/*`, fixtures, and PNGs — image bloat + stale data. **Impact: med · Effort: low**
2. **Run the container as non-root** and use a multi-stage build to drop `gcc`
   (`Dockerfile:5-7`). **Impact: med · Effort: low**
3. **Honor `$PORT`** — `Dockerfile:17` hardcodes `8080` while `Procfile` uses
   `${PORT:-8080}`; breaks platforms that inject `$PORT`. **Impact: low · Effort: low**
4. **Add a `HEALTHCHECK`** hitting `/health`. **Impact: low · Effort: low**
5. **Separate the data pipeline from the web container.** The scheduler runs inside
   the HTTP process (`docker-compose.yml`), coupling dashboard availability to the
   refresh job, and the `refresh` profile overlaps it. **Impact: med · Effort: med**
6. **Re-check export commits.** `daily_exports.yml`/`intraday_exports.yml` push with
   `[skip ci]`, so a malformed CSV bypasses test gates; add a lightweight validation
   step. The two workflows also disagree on committing `img/*.png`. **Impact: low · Effort: low**
7. **`publish-wiki.yml` force-pushes** `master`; add a guard/merge strategy so manual
   wiki edits aren't silently overwritten. **Impact: low · Effort: low**

---

## 8. UX / dashboard

1. **Live polling only updates the banner** (`templates/ticker.html:777-794`) —
   charts/cards stay stale until full reload. Patch metrics via the existing JSON API
   or use SSE/htmx. **Impact: med · Effort: med**
2. **Auto-replay does a full page reload every 2s** (`ticker.html:809-815`) — switch
   to client-side/API-driven snapshot stepping. **Impact: med · Effort: med**
3. **Forced refresh has no progress/loading state** — add a skeleton/spinner and
   confirmation. **Impact: low · Effort: low**
4. **Invalid `?ts=` silently shows latest** (`web_app.py:152-159`) — show a
   "snapshot not found" message with a link to latest. **Impact: low · Effort: low**
5. **Widget `theme`/`compact` params are passed but ignored** by
   `templates/widget.html` despite being documented. Wire up the styling.
   **Impact: low · Effort: low**
6. **Human-readable snapshot labels** in the selector (`ticker.html:250-257`) instead
   of raw `YYYY-MM-DD_HHMMSS`. **Impact: low · Effort: low**
7. **Accessibility** — alert severity is color-only; charts lack text alternatives.
   **Impact: low · Effort: med**

---

## Suggested first slice (small, high-value PR)

A low-risk batch that needs no architectural change:

- Disable Flask `debug=True` behind an env flag and fix the dev/prod port mismatch (§1.1).
- Fix the `front_term_gex_bn` copy-paste bug (§2.1) and remove the Streamlit
  duplicate block (§2.2).
- Standardize cwd-relative paths to repo-root resolution (§2.3).
- Cache loaded ML models (§4.2).
- Add `.dockerignore`, non-root user, and a `HEALTHCHECK` (§7.1–7.4).
- Set `GEX_DISABLE_SCHEDULER=1` in CI (§3.5) and add ruff to the test workflow (§6).

Everything above is independently shippable; the section ordering reflects a
reasonable rollout sequence (security → correctness → reliability → performance →
structure → tests/ops/UX).
