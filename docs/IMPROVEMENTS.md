# GEX improvement ideas

A prioritized, grounded backlog produced from a full review of `gex_core/`,
the web/dashboard layer, scripts, and ops config. Each item cites the relevant file
(and line where useful) so it can be picked up directly. Items are grouped by
theme and tagged with rough impact/effort.

Legend: **Impact** = user/operational value, **Effort** = relative size.

---

## Completed (as of 2026-06-12)

These items were addressed in earlier work; kept here so the backlog stays accurate.

| Item | Notes |
|------|-------|
| Flask debug gated behind `FLASK_DEBUG` | `web_app.py` — dev port from `PORT` env (default 8080) |
| `front_term_gex_bn` copy-paste bug | `features.py` — uses front-month aggregate, not zero-DTE |
| Streamlit duplicate dashboard block | `streamlit_app.py` — duplicate section removed |
| `safe_float` / `estimate_gamma_flip` duplication | Centralized in `gex_core.features` |
| Webhook SSRF guard | `gex_core/intelligence.py` — `validate_webhook_url` + tests |
| `.dockerignore`, non-root user, `HEALTHCHECK` | `Dockerfile`, `.dockerignore` |
| `GEX_DISABLE_SCHEDULER` in CI/tests | `ci.yml`, `conftest.py` |
| Widget `theme`/`compact` params | `templates/widget.html` — light/dark + compact layout |
| Tracked `__pycache__` / root CLI PNG artifacts | Removed from git; patterns in `.gitignore` |

**Removed / renamed:** `templates/ticker.html` was replaced by `periscope.html` and
`wall_gex.html`. UX items below that reference `ticker.html` should target those
templates instead.

---

## 1. Security & hardening (do first)

1. ~~**Disable Flask debug in the dev entrypoint.**~~ **Done** — see Completed table.

2. **Protect state-changing GET routes.** `force_refresh`, `dispatch_alerts`, and
   `/ticker/<t>/bootstrap` mutate state / spend UW API quota and are unauthenticated.
   `GET` side effects can be triggered via `<img>`/links/crawlers. Convert dispatch +
   bootstrap to `POST` with a CSRF/secret token, and require a token for forced refresh.
   Partial progress: `POST /refresh` requires a token; verify remaining GET routes.
   **Impact: high · Effort: med**

3. **Don't auto-dispatch alerts on every page render.** With
   `GEX_ALERT_AUTO_DISPATCH=1`, each dashboard render may call `maybe_dispatch_alerts`;
   only the cooldown/dedupe logic in `gex_core/alert_dispatch.py` prevents webhook spam.
   Move auto-dispatch to the scheduler/background job only. **Impact: high · Effort: low**

4. ~~**Validate the webhook URL (SSRF guard).**~~ **Done** — `validate_webhook_url` in
   `gex_core/intelligence.py` with tests in `tests/test_webhook_security.py`.

5. **Pin front-end CDN assets + add SRI.** `templates/periscope.html` and
   `templates/webull_trade.html` load Bootstrap and Plotly with pinned versions but
   without subresource-integrity hashes. Add `integrity`/`crossorigin`.
   **Impact: med · Effort: low**

6. **Harden pickle/model loading.** `predict.py` uses `joblib.load` which deserializes
   pickle from `models/`. Document/verify that `models/` is only writable by trusted CI,
   and consider signing manifests. **Impact: med · Effort: low**

7. **Avoid leaking config via `/health`.** `gex_core/system_status.py` exposes
   `index_db` path and webhook/auto-dispatch booleans to unauthenticated callers.
   Trim to non-sensitive fields or require auth. **Impact: low · Effort: low**

---

## 2. Correctness bugs

1. ~~**`front_term_gex_bn` duplicates `zero_dte_gex_bn`.**~~ **Done** — see Completed.

2. ~~**Streamlit renders a duplicate dashboard block.**~~ **Done** — see Completed.

3. **cwd-relative paths break under gunicorn/docker.** `predict.py`
   (`MODELS_DIR = Path("models")`), `models_manifest.py`, `decompose.py`
   (`Path("data")`), `web_app.py` `IMG_DIR = Path("img")`, and
   `streamlit_app.py` `Path("data/exports")` resolve from the process cwd, unlike
   `gex_core/exports.py` which resolves from `__file__`. Standardize all on
   repo-root resolution. **Impact: med · Effort: low**

4. **Deprecated `datetime.utcnow()`.** Used in `event_calendar.py`, `main.py`, and
   `system_status.py`; replace with `datetime.now(timezone.utc)` for timezone-aware UTC.
   **Impact: low · Effort: low**

5. **Naive date math for expiry/DTE filters.** `pipeline.py` and
   `data_quality.py` use `datetime.today()` with no timezone/market-calendar
   awareness, which can mis-bucket 0DTE near session boundaries. **Impact: low · Effort: med**

---

## 3. Reliability & error handling

1. **Replace silent `except Exception: pass`/swallow patterns with structured
   logging.** Present in `main.py`, `refresh.py`, `system_status.py`, `charts.py`,
   `live/aggregator.py`. Each hides real failures (AI, SQLite upsert, chart build,
   index sync). Log with context; keep optional features non-fatal but visible.
   **Impact: high · Effort: med**

2. **Surface partial-failure states in the UI.** `web_app.py` swallows UW refresh,
   prediction, flow-overlay, and backtest errors and shows stale data with no indicator.
   Add an explicit "live feed unavailable / using last snapshot" banner.
   Partial progress: stale-banner tests exist; verify all code paths.
   **Impact: med · Effort: med**

3. **Atomic + locked writes for shared state files.** `.alert_dispatch_state.json`
   and `live/state.py` write in place with no locking, racing under concurrent
   refresh + manual dispatch. Use temp-file + `os.replace` and a lock.
   **Impact: med · Effort: low**

4. **Don't mask missing export files.** `history.py` falls back to the strike file
   when cumulative is absent; `decompose.py` returns a zero decomposition when the
   contract cache is missing — both silently degrade. Log a warning and/or flag the
   snapshot as incomplete. **Impact: low · Effort: low**

5. ~~**Validate scheduler/test isolation.**~~ **Done** — `GEX_DISABLE_SCHEDULER=1`
   in CI and `conftest.py`.

---

## 4. Performance

1. **Stop re-indexing every export on each sync.** `storage.py`
   `sync_ticker_exports` re-reads/upserts all exports per call (O(exports × JSON)),
   and it runs from `/health` and history builds. Track an indexed watermark / only
   ingest new timestamps. **Impact: high · Effort: med**

2. **Cache loaded ML models.** `predict.py` reloads joblib/LSTM models on every
   prediction. Memoize by manifest hash so dashboard polling doesn't reload
   TensorFlow each call. **Impact: high · Effort: low**

3. **Reduce redundant UW HTTP calls.** A CLI run fetches greek+spot in
   `fetch_uw_gex`, then `main.py` hits `fetch_uw_spot_exposures` again. The two
   UW calls in `uw_loader.py` are sequential. Skip the redundant call when spot is
   known and/or parallelize. **Impact: med · Effort: med**

4. **Add retries/backoff + rate-limit handling to the UW client.**
   `uw_loader.py` is a single `requests.get` with a 15s timeout and no retry —
   one transient failure aborts a refresh. **Impact: med · Effort: low**

5. **Fix O(n²) hot loops.** KNN neighbor lookup, `structural_forward_delta`, and
   strike binning in `features.py` (use `np.digitize`). **Impact: med · Effort: med**

6. **Move blocking work out of request handlers.** Dashboard routes and
   `_ticker_api_payload` synchronously run UW exports, yfinance calls, prediction,
   and webhook POST on the gunicorn worker. Offload refresh/dispatch to a background
   job/queue. **Impact: high · Effort: high**

7. **Avoid per-snapshot VIX blocking I/O.** `market_features.py` fetches VIX over
   the network in the forecast hot path. Cache it and make it truly per-snapshot.
   **Impact: low · Effort: med**

---

## 5. Architecture & duplication

1. **Decouple the scheduler from `main.run`.** `refresh.py` imports `main.run`,
   dragging matplotlib/argparse into the background refresh path. Extract a thin
   `gex_core.runner.export_snapshot()` with no plotting deps. **Impact: med · Effort: med**

2. **Slim `main.py`.** Move plotting/export into `gex_core` (reuse `charts.py`/
   `exports.py`). **Impact: med · Effort: med**

3. **De-duplicate shared helpers.** OCC parsing exists in both `data_quality.py` and
   `live/aggregator.py`; history-building is reimplemented in
   `scripts/backtest_gex_prediction.py` instead of using `gex_core.history.build_history`.
   **Impact: med · Effort: med**

4. **Converge the two dashboards.** Flask (`web_app.py`) and Streamlit
   (`streamlit_app.py`) duplicate ticker discovery, chart building, UW fetch, color
   tokens, and export-dir resolution. Either share `gex_core.charts` in both or
   document which is canonical. **Impact: med · Effort: high**

5. **Extract a shared data-assembly layer** used by dashboard pages and the JSON APIs
   (`_ticker_api_payload`) to eliminate parallel logic. **Impact: med · Effort: med**

6. **Externalize magic constants.** Many tuned values are hardcoded across
   `ai_analyst.py`, `decompose.py`, `predict.py`, `regime.py`, `intelligence.py`,
   `uw_loader.py`. Move to config with documented defaults. **Impact: low · Effort: med**

---

## 6. Testing gaps

Highest-value missing coverage (modules with **no** direct tests):

- `pipeline.py` — the core GEX formula. **(high priority)**
- `main.py` — CLI/export path.
- `decompose.py`, `ai_analyst.py`, `backtest_metrics.py`, `models_manifest.py`.
- `intelligence.generate_alerts` / `simulate_spot_scenario` /
  `dispatch_alerts_to_webhook`.
- `predict.similar_setups`, `_predict_from_trained_models`, multi-horizon edge cases.
- `refresh.is_snapshot_stale` / `refresh_ticker` integration (only mocked today).
- `live/ingest.py`, `live/state.py`.
- Web routes beyond smoke tests: `force_refresh`, `dispatch_alerts`, exports route,
  bootstrap POST.

Add **lint + type checks** to CI (ruff/mypy) and a **Docker build smoke test** —
`ci.yml` currently runs unit tests + backtest gate only. **Impact: high · Effort: med**

---

## 7. Deployment & CI

1. ~~**Add `.dockerignore`.**~~ **Done** — `.dockerignore` present.
2. ~~**Run the container as non-root** and use a multi-stage build.~~ **Done** — non-root
   `appuser` in `Dockerfile`; gcc still installed for some deps.
3. **Honor `$PORT`** — verify `Dockerfile`/`docker-entrypoint.sh` vs `Procfile`.
   **Impact: low · Effort: low**
4. ~~**Add a `HEALTHCHECK`** hitting `/health`.~~ **Done** — `Dockerfile`.
5. **Separate the data pipeline from the web container.** The scheduler runs inside
   the HTTP process (`docker-compose.yml`), coupling dashboard availability to the
   refresh job. **Impact: med · Effort: med**
6. **Re-check export commits.** `daily_exports.yml`/`intraday_exports.yml` push with
   `[skip ci]`; add a lightweight validation step. **Impact: low · Effort: low**
7. **`publish-wiki.yml` force-pushes** `master`; add a guard/merge strategy.
   **Impact: low · Effort: low**

---

## 8. UX / dashboard

1. **Live polling only updates the banner** (`templates/periscope.html`) — charts/cards
   stay stale until full reload. Patch metrics via the existing JSON API or use SSE/htmx.
   **Impact: med · Effort: med**
2. **Auto-replay does a full page reload** — switch to client-side/API-driven snapshot
   stepping. **Impact: med · Effort: med**
3. **Forced refresh has no progress/loading state** — add a skeleton/spinner and
   confirmation. **Impact: low · Effort: low**
4. **Invalid `?ts=` silently shows latest** — show a "snapshot not found" message with
   a link to latest. **Impact: low · Effort: low**
5. ~~**Widget `theme`/`compact` params are passed but ignored.**~~ **Done** — see Completed.
6. **Human-readable snapshot labels** in the selector instead of raw
   `YYYY-MM-DD_HHMMSS`. **Impact: low · Effort: low**
7. **Accessibility** — alert severity is color-only; charts lack text alternatives.
   **Impact: low · Effort: med**

---

## Suggested next slice (small, high-value PR)

A low-risk batch that needs no architectural change:

- Standardize cwd-relative paths to repo-root resolution (§2.3).
- Cache loaded ML models (§4.2).
- Replace `datetime.utcnow()` with timezone-aware UTC (§2.4).
- Add ruff to the CI test workflow (§6).
- Pin CDN assets with SRI in `periscope.html` / `webull_trade.html` (§1.5).

Everything above is independently shippable; the section ordering reflects a
reasonable rollout sequence (security → correctness → reliability → performance →
structure → tests/ops/UX).
