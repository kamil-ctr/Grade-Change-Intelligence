# CHANGELOG

Kept synchronised with the codebase. Newest first.

---

## [0.9.0] — 2026-07-25 — React dashboard: Phase 1 (MVP) complete

### New files

`frontend/` — Vite + plain React, no TypeScript, no component library, no
chart library (every chart is hand-rolled SVG). Production build: 207 KB JS
/ 65 KB gzipped, 8.5 KB CSS. Six panels (`App.jsx` + `src/components/`):
`ForecastCone`, `RecommendationsPanel`, `CorrelationTable`,
`StabilizationBars`, `TrustPanel`, `EconomicsPanel`, each tagged with its
deliverable number in its own header. Design tokens in `styles.css` follow
the dataviz skill's validated reference palette (fixed-order categorical
hues, reserved status colors, thin 2px marks, always-present legends).

### Bug fixes

1. **ROI Assumptions form rendered permanently empty.** `EconomicsPanel`
   initialised its state with `useState(economics || {})`; `economics`
   arrives asynchronously from the parent's fetch, after this component's
   first render, and `useState`'s initial value is consumed exactly once.
   Found by actually loading the page in a browser (`claude-in-chrome`), not
   by code review — the bug was invisible in isolation since every *other*
   panel's data arrived the same way but was rendered directly from props,
   not copied into local state first. Fixed with a `useEffect` syncing local
   state whenever the `economics` prop changes.

### Verified

Loaded against the live API in a real Chrome tab: all six panels render
correct live data; clicked Reject on a real advisory and watched the Trust
panel update immediately (2 surfaced -> 2 responded, 100% -> 50% acceptance,
and the calibration text correctly flipped to "confidence is not yet
separating accepted from rejected advice" once rejected-advisory confidence
exceeded accepted-advisory confidence); zero console errors; clean
production build; clean `oxlint`.

### Milestone

**Phase 1 (MVP) is complete** — all 10 modules built and verified, all 6
graded deliverables have working, demonstrated evidence (not just code).
Remaining choices: Phase 2 modules (What-If Studio, Copilot, trust-score
learning, formal benchmark) vs. moving to the reserved packaging/deck block.

---

## [0.8.0] — 2026-07-25 — FastAPI service: every engine reachable over HTTP

### New files

| File | Lines | Purpose |
|---|---|---|
| `gci/api/datasource.py` | 85 | Connectivity layer: `DataSource` ABC + `SimulatedDataSource` |
| `gci/api/service.py` | 259 | Business logic, HTTP-independent |
| `gci/api/app.py` | 157 | FastAPI wiring |
| `tests/test_api.py` | 204 | 28 tests (17 service-level, 11 through `TestClient`) |

Twelve endpoints live (`whatif`/`copilot` intentionally not stubbed — Phase 2
modules they'd wrap don't exist yet, and this project does not ship
placeholder logic). `SimulatedDataSource` generates a small deterministic
40-event demo corpus at startup and replays it as "live" data, with a
`row_at(event_id, t_min)` cursor that never returns anything past the
requested time — the backward-only feature guarantee (D9) carried through to
the API boundary.

### Bug fixes

1. **`forecast_model.joblib` didn't exist.** `forecast.py` had only ever
   been exercised against synthetic data in its own tests; the real training
   script had never been run. Caught because the API's `/api/live` and
   `/api/recommendations` need it. Run now — see forecast section above for
   results.
2. **`HistGradientBoostingRegressor` fallback had no `early_stopping`.**
   Every classifier factory in `ml/registry.py` sets it; the forecast
   module's regression fallback didn't, so the first training run burned
   several extra minutes on iterations past convergence. Fixed to match the
   existing pattern before the artefact used by the API was persisted.
3. **NaN in a JSON response crashed the endpoint.** Starlette's default
   renderer calls `json.dumps(..., allow_nan=False)`; several legitimately
   NaN values (`settle_min` for a never-settling transition,
   `mean_confidence_accepted` before any feedback) reached it and raised
   `ValueError: Out of range float values are not JSON compliant`. Fixed
   with a `NanSafeJSONResponse` that maps non-finite floats to `null` at the
   wire boundary only.

### Verified

Smoke-tested against a real `uvicorn` process (not just the in-process test
client): `/api/health`, `/api/live`, `/api/recommendations` all correct on a
rushed `LWC-52 -> BRD-120` transition — 100% risk probability with SHAP
explanation, an optimizer recommendation extending the ramp 6.6 -> 25 min
predicted to eliminate a 7.9-minute off-spec period, priced at $304.55
(P10-P90 $213-$411).

### Known issues

1. **`forecast_model.joblib` is 14.3 MB.** Combined with the other `models/`
   artefacts this may approach the 10 MB submission limit; needs a decision
   in the reserved packaging block (slim the persisted bundle, or regenerate
   `models/` at submission time the way `data/` already is).
2. **Forecast coverage runs ~6-7 points under nominal** (see forecast
   section) — a known, disclosed limitation of the fallback estimator, not
   hidden in the reported numbers.

---

## [0.7.0] — 2026-07-25 — Advisory ledger (deliverable 6)

### New files

| File | Lines | Purpose |
|---|---|---|
| `gci/ledger.py` | ~185 | Append-only JSON-Lines accept/reject audit trail + quality evaluation |
| `tests/test_ledger.py` | ~105 | 10 tests |

`record()`/`respond()` log an `Advisory` being surfaced and the operator's
decision; `evaluate()` reports acceptance rate (overall and per source), mean
confidence of accepted vs rejected (a calibration check), and realised
dollar value of accepted advisories. JSON Lines persistence, no database
dependency. **All six graded deliverables now have a working engine** —
remaining work is the API surface and dashboard that make them visible.

---

## [0.6.0] — 2026-07-25 — Provenance and advisory packaging (deliverable 5)

### New files

| File | Lines | Purpose |
|---|---|---|
| `gci/provenance.py` | ~185 | Unified `Advisory` schema, grounded per-source explanations, `AdvisoryPolicy` gate |
| `tests/test_provenance.py` | ~135 | 14 tests |

Folds risk predictions, `discovery.CorrelationResult`,
`optimizer.OptimizationResult` and `stabilization.LoopImpact` into one
`Advisory` shape with a human explanation grounded in each source's own
computation (SHAP drivers, measured lag/correlation, the specific plan
change). `rank_and_gate()` generalises `roi.should_surface` across all
advisory types: confidence floor always, value floor only where a price
exists, sorted priced-first, capped at `AdvisoryPolicy.max_concurrent_suggestions`.
This closes deliverable 5.

---

## [0.5.0] — 2026-07-25 — Stabilization loop impact ranking

### New files

| File | Lines | Purpose |
|---|---|---|
| `gci/stabilization.py` | ~150 | Twin-based sensitivity ranking of plan parameters on `settle_min` |
| `tests/test_stabilization.py` | ~95 | 10 tests |

Perturbs each tunable plan parameter up/down from baseline and measures the
change in `settle_min` via `optimizer.evaluate_plan` — same physics as the
optimizer, not a separate approximation. `tau_c_scale` stands in for all
three PI trim loops (`control.TRIM_PAIRS`) at once. Fixed a first-draft bug
before it shipped: "best direction" was originally chosen only between the
two perturbed candidates, which could recommend a change that was worse than
doing nothing whenever *both* directions hurt. Fixed by including the
baseline itself in the "best" comparison.

---

## [0.4.0] — 2026-07-25 — Correlation discovery

### New files

| File | Lines | Purpose |
|---|---|---|
| `gci/discovery.py` | ~215 | Lagged Pearson correlation + mutual information + novelty vs `KNOWN_LOOPS` |
| `tests/test_discovery.py` | ~150 | 9 tests, including a planted-signal recovery test |

Sweeps every ordered pair among the 19 process tags, pooling `(cause[t],
effect[t+lag])` within events only (never across, per D11). Only non-negative
lags are swept since testing every ordered pair already covers both
directions. Mutual information at the best lag is computed on a capped
subsample via `sklearn.feature_selection.mutual_info_regression`. Results are
classified against `grades.KNOWN_LOOPS` via `is_known_relationship` and
tagged `Source.CORRELATION_DISCOVERY`. `series_by_tag_from_dataset()` reads
the persisted corpus directly rather than re-simulating.

**Smoke test on the real corpus** (150-event subsample, 3 min max lag, 0.35
threshold): 36 correlations found, 26 novel — e.g. `machine_speed -> caliper`
(r=-0.97), a real indirect coupling through basis weight and the empirical
density model, correctly flagged as not in `KNOWN_LOOPS`, alongside confirmed
known loops like `filler_flow -> ash` (r=+0.95).

---

## [0.3.0] — 2026-07-25 — Real hardware: env restore, forecast, ROI, optimizer

Moved off the 45s/3.9GB sandbox onto real hardware. Restored trimmed settings,
regenerated the corpus at 3x scale, retrained, and completed three Phase 1
modules: `forecast.py`, `roi.py`, `optimizer.py`.

### Environment restoration

- Random Forest: 150→**300** trees, depth 10→**14**, min leaf 40→**20**.
- SHAP sample: 600→**3,000** rows. Permutation importance: 1,500 rows/3
  repeats → **6,000 rows/8 repeats**.
- Corpus: 500 events (two chunks) → **1,500 events, one call, 72s**.
- `scripts/train_risk_model.py --perm-repeats` default 3→8.

### New files

| File | Lines | Purpose |
|---|---|---|
| `gci/forecast.py` | 349 | Quantile trajectory forecast cone: +2/+5/+10 min basis-weight deviation at P10/P50/P90 |
| `gci/roi.py` | 187 | Confidence-weighted dollar pricing + `AdvisoryPolicy` surfacing gate |
| `gci/optimizer.py` | 260 | Coordinate-descent ramp/setpoint search over the twin, cached per scenario |
| `scripts/train_forecast_model.py` | 127 | CLI: train/evaluate/persist the forecast model |
| `tests/test_forecast.py` | 218 | 12 tests, leakage-focused |
| `tests/test_roi.py` | 143 | 19 tests |
| `tests/test_optimizer.py` | 126 | 9 tests |

### Bug fixes (found during this session)

1. **`_probe()` false negatives on this machine were actually true negatives.**
   LightGBM and XGBoost both installed via pip but fail at import:
   `Library not loaded: @rpath/libomp.dylib`. Root cause: this machine's
   Homebrew is under the Intel prefix (`/usr/local`, likely via Rosetta) while
   Python and the wheels are native arm64 — the x86_64 `libomp.dylib` cannot
   satisfy an arm64 rpath, and there is no arm64 Homebrew (`/opt/homebrew`)
   present to provide the right one. `xgboost-cpu` additionally has no
   prebuilt wheel for this platform and needs `cmake` (absent) to build from
   source. **User decision: skip both, proceed with the three
   always-available models** — exactly the case `ml/registry.py`'s
   optional-dependency probing (D20) was built for.
2. **`report_markdown()` crashed `pipeline.save()`** on a missing `tabulate`
   dependency (`DataFrame.to_markdown()`). All other artefacts (model,
   metrics, comparisons, SHAP, warning detail) are written before the
   markdown report in `save()`'s ordering, so this only ever silently left a
   stale `evaluation_report.md` rather than corrupting anything load-bearing.
   Fixed by installing `tabulate`.
3. **Grade code typo in a new test** (`test_forecast.py` used `"SC-58"`,
   which doesn't exist — the library has `SC-56`). Caught immediately by the
   test itself failing with a clear `KeyError`.

### Risk model retrained

Selected model changed from LightGBM (500 events, sandbox) to **Histogram
Gradient Boosting** (1,500 events, this machine — LightGBM/XGBoost
unavailable, see above). Test set, scored once:

| Metric | Before (500 ev, LightGBM) | After (1,500 ev, HistGBM) |
|---|---|---|
| PR-AUC | 0.821 | 0.827 |
| Event detection rate | 0.847 | 0.827 |
| Median warning time | 4.50 min | 4.67 min |
| False alarms / clean event | 0.179 | **0.099** |

Detection delta is within the ~0.05 per-event standard error (Assumption 17)
and not a meaningful regression; false alarms improved substantially, which
matters more for an advisory system's credibility than a few points of
detection rate.

### Design decisions added

D22 (forecast targets kept out of `features.py`), D23 (quantile
rearrangement over a joint quantile model), D24 (off-spec tonne priced as
margin + rework), D25 (coordinate descent over a full 3-D grid in the
optimizer). See `PROJECT_LOG.md` §6 for rationale.

### Known issues

1. **LightGBM/XGBoost unusable on this machine** (see above). Both remain
   `pip install`ed in case a future environment (e.g. an arm64 Homebrew, or a
   different machine) resolves the runtime — the registry will pick them up
   automatically with no code change, per D20.
2. **`requirements.txt` still does not pin `lightgbm`/`xgboost`/`shap`**,
   consistent with their optional-dependency status; anyone reproducing this
   run on a working arm64 Homebrew should `pip install lightgbm xgboost shap`
   for the full five-model comparison.

---

## [0.2.0] — 2026-07-25 — ML pipeline + risk prediction model

Phase 1, module 1 complete: a leakage-audited model comparison pipeline and a
trained off-spec risk model.

### New files

| File | Lines | Purpose |
|---|---|---|
| `gci/ml/__init__.py` | 55 | Package contract: the four leakage guarantees, stated as the module docstring |
| `gci/ml/splits.py` | 205 | Event-wise stratified 3-way splitting |
| `gci/ml/metrics.py` | 375 | Classification + early-warning metrics, threshold selection, alarm on-delay |
| `gci/ml/registry.py` | 215 | Extensible model registry with optional-dependency probing |
| `gci/ml/explain.py` | 330 | Tiered SHAP (4 exact paths + labelled fallback), consensus importance |
| `gci/ml/pipeline.py` | 700 | Orchestration, checkpointing, operating-point tuning, artefact persistence |
| `scripts/train_risk_model.py` | 151 | CLI: train, compare, select, explain, persist |
| `tests/test_ml.py` | 430 | 37 tests, leakage-focused |
| `CHANGELOG.md` | — | This file |

### Modified files

- `gci/features.py` — added `downcast_features`, `save_features`, `load_features`. float64 → float32 halves the memory footprint; the frame plus three split copies were the peak.
- `gci/ml/registry.py` — Random Forest trimmed twice (300→200→150 trees, depth 14→10) to fit the environment's per-call time budget without changing its character.
- `scripts/generate_data.py` — always caches the feature frame; writes `validation.json` before the slowest step so an interrupted run still leaves a usable artefact.
- `PROJECT_LOG.md` — updated throughout.

### Models added

Five candidates, all trained and compared:

| Model | Available via | Exact SHAP path |
|---|---|---|
| LightGBM | `lightgbm` | `shap.TreeExplainer` / native `pred_contrib` |
| XGBoost | `xgboost-cpu` | `shap.TreeExplainer` / native `pred_contribs` |
| Random Forest | scikit-learn | `shap.TreeExplainer` |
| Histogram Gradient Boosting | scikit-learn | `shap.TreeExplainer` |
| Logistic Regression | scikit-learn | analytic linear SHAP |

**Selected: LightGBM** — threshold 0.60, alarm on-delay 3 samples (15 s).

Held-out test performance (scored once):

| Metric | Value |
|---|---|
| PR-AUC | 0.821 |
| Precision | 0.802 |
| Recall (row-level) | 0.580 |
| F1 | 0.673 |
| False positive rate | 0.041 |
| **Event detection rate** | **0.847** |
| **Median warning time** | **4.50 min** |
| Mean warning time | 5.28 min |
| False alarms per clean event | 0.179 |

### Performance improvements

- **Feature caching** — training loaded a cached frame in 0.2 s instead of regenerating the corpus for 25 s on every run.
- **Per-model checkpointing** — each candidate persists as soon as it is trained and evaluated; re-runs restore instead of refitting. Turned a fragile 2-minute run into five resumable steps.
- **float32 downcast** — feature frame memory halved (90 MB → 45 MB at 300 events).
- **Removed a redundant downcast** — `save_features` was re-copying an already-downcast frame, an extra ~150 MB copy at 500 events.
- **Corpus grown 300 → 500 events**, which closed a generalisation gap (see below).

### Bug fixes

1. **Stale checkpoints silently restored across datasets.** Validity was checked on feature names and split seed only, so growing the corpus 300 → 500 restored models trained on the *old* data and reported their metrics as current. Fixed by adding a `dataset_fingerprint` (SHA-256 over sorted event ids, row count, feature names, and the spec/horizon config) to checkpoint metadata.
2. **`EventSplit.mask` rejected `"validation"`.** The class used `"val"` internally while the pipeline passed `"validation"`. Fixed with an explicit alias table rather than by picking a winner.
3. **`save()` raised `KeyError: 'sweep'`.** `select_by_operating_point` stores `per_model`, not `sweep`; the serialiser assumed the older shape. Fixed with `.get()`.
4. **`bw_dev_roc_1min` KeyError in feature building.** f-string interpolation of a float produced `bw_dev_roc_1.0min`. Fixed with explicit suffix labels.
5. **Gaussian pulse faults leaked signal before onset.** A Gaussian has infinite tails, so `REFINER_LOAD_SWING` was nonzero before its start time — a subtle form of look-ahead for anything trained on it. Now masked to the active window.

### Known issues

1. **Row-level recall (0.580) reads low and is easy to misquote.** It is the wrong headline: an excursion only has to be caught once to warn the operator, so **event detection rate (0.847)** is the number that matters. Both are reported everywhere.
2. **Per-event metrics remain noisy.** At 500 events the validation split has ~59 breaching events, giving a standard error near 0.05 on detection rate. Validation 0.831 vs test 0.847 is well inside that.
3. **False alarms are 0.179 per clean event** — roughly one nuisance alarm per six clean transitions. Acceptable for an advisory system but the weakest product metric; the ROI gate in `AdvisoryPolicy.min_value_usd_to_surface` will suppress low-value alarms further once the ROI engine lands.
4. **`shap` needed a two-step install** (`--no-deps` then `numba` separately); the default resolution was killed repeatedly on the 58 MB `llvmlite` wheel. The tiered explainer means this is a convenience, not a dependency.
5. **`xgboost` requires `xgboost-cpu`.** The default wheel bundles 131 MB of CUDA libraries and could not be installed here. `requirements.txt` pins `xgboost-cpu`.
6. **Stale artefacts cannot be deleted** in this environment (`data/features.csv.gz`, `data/chunk_*.pkl`). Must be excluded from the submission zip.
7. **Random Forest was trimmed for wall-clock reasons**, not statistical ones. It is not disadvantaged in any way that changes the comparison, but a longer run would use more trees.

---

## [0.1.0] — 2026-07-25 — Phase 0 foundation

### New files

`gci/config.py`, `gci/grades.py`, `gci/twin.py`, `gci/control.py`,
`gci/faults.py`, `gci/events.py`, `gci/features.py`,
`scripts/generate_data.py`, `tests/test_twin.py`, `tests/test_control.py`,
`tests/test_events.py`, `tests/test_features.py`, `README.md`,
`requirements.txt`, `PROJECT_LOG.md`.

### Added

- Physics-based digital twin: mass balance → basis weight and ash, energy balance → moisture, empirical density → caliper, with FOPDT dynamics and an incremental stepper.
- Coordinated grade-change controller: S-curve target trajectory, lead compensation, SIMC-tuned PI trim, slew-limited feasibility floor.
- 7-grade recipe library, 19-tag process dictionary, known control-loop graph.
- Fault library: 11 named failure modes over Ornstein–Uhlenbeck correlated drift.
- Closed-loop event generation with labelling and dataset validation.
- 104 strictly backward-looking features.
- 78 tests.

### Bug fixes

1. **Lead compensation applied only to trimmed actuators.** Machine speed — the dominant basis-weight driver on a large transition — was uncompensated, leaving the sheet trailing 8–11% for the whole ramp.
2. **Linear target ramp was physically unachievable.** It demands a step change in actuator *rate* at onset, which no slew-limited drive can deliver. Replaced with an S-curve (zero rate at both ends); feasibility floor scaled by the 1.5× peak-rate factor.
3. **Events began spuriously off-spec.** Initial conditions used nominal recipe values rather than solving the inverse model at the prevailing disturbance state.

### Known issues

Off-spec rate 61.7% (deliberately high to create a learnable problem);
no CD control; caliper is the least trustworthy quality variable; no real
mill data.
