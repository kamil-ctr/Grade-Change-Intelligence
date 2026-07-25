# PROJECT LOG — Grade Change Intelligence (GCI)

**Honeywell Hackathon — Question 5A: Grade Change Intelligence in Paper Making Process**

This file is the project's source of truth. It is updated after every major
implementation step and kept synchronised with the codebase.

| | |
|---|---|
| **Last updated** | 2026-07-25 — frontend complete; **Phase 1 (MVP) fully done** |
| **Current phase** | Phase 1 complete (10/10 modules); Phase 2 not started |
| **Tests** | 226 passing, 0 failing (Python; frontend has no unit test suite — verified live in-browser instead, see below) |
| **Lines of code** | 7,671 (5,451 source / 1,808 tests / 412 scripts) |
| **Submission deadline** | 2026-07-26, 23:59 |

---

## 1. Status at a glance

| Phase | Scope | Status |
|---|---|---|
| **Phase 0** | Twin, controller, faults, events, features, tests | ✅ **Complete** |
| **Phase 1** | MVP — risk model, forecast, ROI, optimizer, discovery, stabilization, provenance, ledger, API, dashboard | ✅ **Complete (10/10 modules)** |
| **Phase 2** | What-If Studio, AI Copilot, feedback learning, benchmark | ⬜ Pending |
| **Phase 3** | Shadow-mode scoring, public-dataset validation | ⬜ Stretch |
| **Reserved** | Architecture doc, 6-slide deck, packaging, rehearsal | ⬜ Reserved (last 2 hrs) |

---

## 1a. Environment restoration (real hardware, 2026-07-25)

The earlier sandbox's 45s/call, ~3.9 GB limits are gone. Per `CLAUDE.md`'s
environment note, restored:

| Setting | Sandbox value | Restored to |
|---|---|---|
| Random Forest | 150 trees, depth 10, leaf 40 | **300 trees, depth 14, leaf 20** |
| SHAP sample | 600 rows | **3,000 rows** |
| Permutation importance | 1,500 rows, 3 repeats | **6,000 rows, 8 repeats** |
| Corpus | 500 events, two 250-event chunks | **1,500 events, one call, 72s** |

New corpus health: 1,500 events, 504,000 feature rows, off-spec rate 62.5%,
settled rate 88.7%, all 42 grade pairs represented, no validation issues.

**Deviation from `CLAUDE.md`'s known traps, found on this machine, not
predicted by it:** `xgboost-cpu` has no prebuilt wheel for this platform and
needs `cmake` (not installed) to build from source — plain `xgboost` was
substituted for the install attempt, but both it and `lightgbm` ultimately
**fail at import** with `Library not loaded: @rpath/libomp.dylib`. This
machine's Homebrew is installed under the Intel prefix (`/usr/local`,
apparently via Rosetta) while Python and the wheels are native arm64, so the
x86_64 `libomp.dylib` Homebrew provides cannot satisfy the arm64 wheels'
rpath, and there is no `/opt/homebrew` arm64 Homebrew present to provide the
right one. Fix would be installing a parallel arm64 Homebrew — a system-wide
change judged not worth it for two of five candidate models when the
registry's optional-dependency design (D20) exists exactly for this case.
**User decision: skip LightGBM/XGBoost**, proceed with the three
always-available models. Both packages remain `pip install`ed (harmless,
just unusable) in case a future environment resolves the runtime.

**Risk model retrained** on the 1,500-event corpus with the three available
models at restored capacity:

| Model | Val PR-AUC | Val detection | Val false alarms/event |
|---|---|---|---|
| Random Forest | 0.785 | 0.805 | 0.108 |
| **Hist Gradient Boosting** | 0.782 | 0.811 | **0.072** |
| Logistic Regression | 0.759 | 0.822 | 0.315 |

**Selected: Histogram Gradient Boosting** (threshold 0.85, on-delay 1 sample).
Held-out test, scored once:

| Metric | 500-event LightGBM (old) | 1,500-event HistGBM (current) |
|---|---|---|
| PR-AUC | 0.821 | 0.827 |
| Event detection rate | 0.847 | 0.827 |
| Median warning time | 4.50 min | 4.67 min |
| False alarms / clean event | 0.179 | **0.099** (−45%) |

Detection rate is within the ~0.05 per-event standard error noted in
Assumption 17 and is not a meaningful regression; false alarms improved
substantially — the more relevant number for an advisory system, since it is
directly what erodes operator trust. Top consensus features unchanged in
character: `bw_dev_headroom_pct`, `t_since_ramp_min`, `transition_magnitude`,
`plan_trim_enabled`, `plan_ramp_min`.

---

## 2. Completed modules

### `gci/config.py` — global configuration
Spec thresholds (basis weight off-spec at **>2.5%** deviation), sampling
interval, machine geometry and furnish constants, the `Source` provenance tag
enum, the editable `Economics` dataclass, and the `AdvisoryPolicy` governance
posture (advisory-only, no control writeback, with the
ADVISORY → SUPERVISORY → CLOSED_LOOP maturity ladder).

### `gci/grades.py` — recipe and grade library
Seven grades spanning 45–150 g/m² (newsprint, SC, LWC, woodfree ×2,
boxboard ×2) with quality targets, actuator envelopes and per-minute ramp
limits. Defines the 19-tag process dictionary (6 MVs, 4 CVs, 9 DVs) and — 
critically — the **`KNOWN_LOOPS` graph** encoding the control relationships the
existing QCS already models. Phase 1's discovery engine flags anything strong
but absent from this graph as *novel*.

### `gci/twin.py` — digital twin
First-principles steady-state model (mass balance → basis weight and ash,
energy balance → moisture, empirical density → caliper) wrapped in
first-order-plus-dead-time dynamics per quality variable. Provides:
- `steady_state()` — vectorised forward model
- `inverse_solve()` — grade targets → actuator setpoints (keeps recipe book and twin self-consistent)
- `simulate()` — batch open-loop run
- `TwinStepper` — incremental stepping required for closed-loop control

### `gci/control.py` — coordinated grade-change controller
Models the plant's own MD control package:
- **S-curve target trajectory** (`scurve`, smoothstep) — zero rate at both ends
- **Feedforward with lead compensation** — lead = θ + τ per loop, applied to *all* actuators via `MV_LEAD_DRIVER`
- **PI trim** with **SIMC tuning** derived from the identified FOPDT parameters and numerically-obtained process gains, plus conditional-integration anti-windup
- **`min_feasible_ramp_min()`** — slew-limited feasibility floor and the binding actuator

### `gci/faults.py` — fault and disturbance library
Ornstein–Uhlenbeck correlated baseline drift on all 9 disturbance tags, plus
**11 named fault types** (consistency upset, steam header dip, retention loss,
broke surge, wire drainage loss, freeness shift, humidity rise, couch vacuum
loss, press load drift, steam hunting, refiner load swing) with step / ramp /
pulse / oscillation signatures. Every fault carries a ground-truth cause label
so explanations can be scored, not just asserted.

### `gci/events.py` — event generation and labelling
Closed-loop simulation of a complete grade change, plus labelling
(`off_spec`, `off_spec_minutes`, `max_abs_dev_pct`, `first_breach_min`,
`settle_min`), feasibility-aware plan sampling, dataset persistence
(npz + JSON) and `validate_dataset()` health checks.

### `gci/features.py` — feature engineering
104 strictly backward-looking features per sample: phase, transition context,
plan parameters, quality state and dynamics, actuator state with slew
utilisation, saturation headroom, disturbance state, and the
**trend-extrapolation feature** (`bw_dev_projected`) that backs the dashboard's
"future state if the deviation follows the current trajectory" requirement.
Labels are forward-looking breach-within-horizon. Includes `event_wise_split()`.

### `gci/ml/` — machine learning pipeline (Phase 1, module 1)

**`splits.py`** Event-wise stratified 3-way splitting. Stratifies on outcome and
transition-difficulty tertile so class balance and difficulty mix are comparable
across splits.

**`metrics.py`** All required classification metrics plus early-warning analysis
(median/mean warning time, event detection rate, per-clean-event false alarm
rate). Threshold selection by `f1` / `fbeta` / `max_fpr`. `confirm_alarms()`
implements ISA-18.2 alarm on-delay, grouped by event so a persistence window
never spans a boundary.

**`registry.py`** Five models behind a `ModelSpec` indirection. Adding a model is
one entry. Optional dependencies are probed once and unavailable models are
skipped with a recorded reason rather than crashing the run.

**`explain.py`** Tiered SHAP with four *exact* paths — `shap.TreeExplainer`,
LightGBM native `pred_contrib`, XGBoost native `pred_contribs`, and analytic
linear SHAP — plus a clearly-labelled approximate surrogate that is never
reached by any registered model. Also consensus feature importance merging
native gain, permutation (on validation), and mean |SHAP|.

**`pipeline.py`** Orchestration: train all → tune each model's operating point →
select on the product objective → score the winner on test once → explain →
persist. Includes per-model checkpointing with dataset fingerprinting.

### `scripts/generate_data.py`, `scripts/train_risk_model.py`
CLIs for corpus generation and model training.

### `gci/forecast.py` — quantile trajectory forecasting (Phase 1, module 2)
Predicts basis-weight deviation (%) at +2/+5/+10 min, at the 10th/50th/90th
percentiles, for the dashboard's forecast cone. Forward targets are built
*outside* `features.py` (`build_forecast_targets`, keyed by `event_id` +
`sample_idx` against a per-event deviation lookup) so the risk model's
backward-only feature contract is never touched; `forecast_feature_columns()`
defensively re-excludes the new target columns from the input list. Reuses
`ml.splits.event_wise_split_3way` so the forecast and risk models share
identical event membership. One `LGBMRegressor(objective="quantile")` per
(horizon, quantile) — falling back to scikit-learn's
`HistGradientBoostingRegressor(loss="quantile")` if LightGBM is unavailable,
same probe-don't-assume posture as `ml/registry.py`; on this machine that
fallback is what actually runs (see §1a). Predictions are monotone-corrected
by row-wise sorting (quantile rearrangement) since independent quantile
regressors can cross. `scripts/train_forecast_model.py` loads the persisted
corpus directly (`events.load_dataset`) rather than re-simulating, so a
retrain after the risk model is near-instant.

**12 tests**, most exercising the leakage-sensitive plumbing with
hand-checkable expectations (target matches a manual shift, NaN exactly at
the window boundary, corrupting one future sample changes only the rows whose
horizon reaches it) rather than relying on the small synthetic end-to-end
pipeline alone.

**Trained on the real 1,500-event corpus** (initially only unit-tested on
synthetic data — caught and fixed before the API needed the artefact, see
§9 "api/"). Held-out test, scored once:

| Horizon | MAE (median) | P10-P90 coverage (nominal 80%) | Mean interval width |
|---|---|---|---|
| +2 min | 0.71% | 74.8% | 2.01% |
| +5 min | 0.89% | 73.9% | 2.41% |
| +10 min | 0.90% | 72.7% | 2.31% |

**Known issue, stated honestly:** coverage runs ~6-7 points under the
nominal 80% at every horizon — the P10-P90 band is somewhat too narrow, most
likely because the `HistGradientBoostingRegressor` fallback (LightGBM
unavailable, §1a) is less well-suited to quantile loss than LightGBM's
native quantile objective would be. Acceptable for a demo forecast cone
(direction and rough magnitude are right, per the low MAE) but would need
widening — e.g. fit at nominal quantiles further from the target coverage,
or a conformal calibration pass — before being trusted for anything
production-facing. Not silently rounded up in the report above.

**Fixed before this run, not after:** the `HistGradientBoostingRegressor`
fallback had no `early_stopping`, unlike every classifier factory in
`ml/registry.py` — training ran the full 400 iterations x 9 models
regardless of convergence. Caught by watching the first training attempt run
long; fixed by matching the same `early_stopping=True,
validation_fraction=0.15, n_iter_no_change=30` pattern already used
elsewhere, before persisting the artefact used by the API.

### `gci/roi.py` — confidence-weighted value model (Phase 1, module 3)
Prices a recommendation in dollars: avoided off-spec tonnes (production rate
from grade basis weight × machine speed × trim width) times the cost of an
off-spec tonne (`net_margin_per_tonne + rework_cost_per_tonne` — assumes
rework as broke, not discount-sale, the conservative case), weighted by the
calling model's confidence, banded P10-P90 via `Economics.low_multiplier` /
`high_multiplier`. `should_surface()` implements the ROI gate
`AdvisoryPolicy.min_value_usd_to_surface` was reserved for — advice below the
value or confidence floor never reaches the operator. `portfolio_annual_value()`
extrapolates a *representative sample* of per-event values correctly (mean ×
annual transition count), documented against the double-counting trap of
summing each event's own already-annualized figure. **19 tests.**

### `gci/optimizer.py` — bounded setpoint/ramp search (Phase 1, module 4)
Searches `ControlPlan`'s three tunable knobs (`ramp_min`, `lead_scale`,
`tau_c_scale`) for the plan minimising predicted off-spec severity, using
`events.run_event` as the objective — the same closed-loop physics the
training corpus and the risk model's causal structure are built on, not a
separate approximation of it. Coordinate descent (sweep one knob, narrow,
repeat) rather than a full 3-D grid: the three knobs are close to independent
and a full grid at 30-minute-window fidelity would be `n^3` simulations for
no accuracy benefit found in testing. Baseline and every candidate share one
disturbance/fault realisation (`seed`) so the comparison isolates the plan's
effect. Ramp search floor is raised to the transition's own physical
feasibility floor (`control.min_feasible_ramp_min`) — recommending an
impossible ramp would defeat the point. Results memoised per (transition,
fault signature, seed) for demo-speed replay. `OptimizationResult.price()`
ties directly into `roi.price_plan_comparison`, tagged
`Source.PHYSICS_MODEL` at a fixed, documented confidence (0.75 — trust in
the twin's fidelity, not a statistical estimate; see known limitation 1).
**9 tests**, including one that deliberately starts from a rushed
below-floor plan and asserts a *strictly* better plan is found, not just a
tie.

### `gci/discovery.py` — lagged correlation + novelty scoring (Phase 1, module 5)
Sweeps every ordered pair among the 19 process tags for the strongest lagged
Pearson correlation (pairs pooled *within* events only — pooling across
events would correlate independent OU disturbance draws, D11), computes
mutual information at that lag as a nonlinear-robust second measure, and
classifies each against `grades.KNOWN_LOOPS` via `is_known_relationship`.
Only sweeps non-negative lags: testing every *ordered* pair already covers
both directions, so `(B, A)` at a positive lag stands in for what `(A, B)` at
a negative lag would show. `series_by_tag_from_dataset()` reads the persisted
corpus directly (cube + tags + meta) rather than re-simulating, with an
optional subsample for interactive speed — same posture as the SHAP/permutation
sampling elsewhere. Smoke-tested on the real 1,500-event corpus (150-event
subsample, 3 min max lag): 36 correlations above the 0.35 threshold, **26
novel** — including `machine_speed -> caliper` (r=-0.97, not in `KNOWN_LOOPS`,
a real indirect coupling through basis weight and the empirical density
model) surfaced correctly alongside confirmed known loops like
`filler_flow -> ash` (r=+0.95). **9 tests**, including a synthetic-signal
test that plants a known lag and confirms discovery recovers it exactly.

### `gci/stabilization.py` — loop impact ranking on settling time (Phase 1, module 6)
Local sensitivity analysis on the twin, not a data-mined regression: each
tunable plan parameter (`ramp_min`, `lead_scale`, `tau_c_scale`) is perturbed
up and down from a baseline plan and the resulting change in `settle_min` is
measured via the same closed-loop simulation `optimizer.py` uses, so ranking
and recommendation share one physics model. `tau_c_scale` stands in for all
three PI trim loops at once (`control.TRIM_PAIRS`), which is the actual
"loop" the deliverable's language refers to. "Best direction" is chosen
against the baseline too, not just between the two probed points — a
parameter where both directions make things worse now correctly reports "no
change helps" (`best_direction="none"`) instead of recommending whichever
probed direction happened to be less bad. Never-settling outcomes (NaN
`settle_min`) are treated as a fixed worst-case penalty (the window length)
rather than dropped or left to break numeric comparisons. **10 tests.**

### `gci/provenance.py` — source tagging and confidence (Phase 1, module 7)
The common packaging layer every other engine's output passes through before
reaching the operator: a single `Advisory` shape (id, title, source,
confidence, human explanation, optional priced `RecommendationValue`, raw
detail). Does not generate advice — each upstream engine already tags and
scores its own output at the point of computation; this module folds risk
predictions, `discovery.CorrelationResult`, `optimizer.OptimizationResult`
and `stabilization.LoopImpact` into one shape and explains each grounded in
its own actual computation (SHAP drivers for risk predictions, the measured
lag/correlation for discoveries, the specific ramp/lead/trim change for
optimizer recommendations). `rank_and_gate()` applies `AdvisoryPolicy`
uniformly — confidence floor always, value floor only for advisories that
carry a price (unpriced ones, e.g. discovered correlations, are judged on
confidence alone), sorted priced-first, capped at
`max_concurrent_suggestions`. **14 tests.**

### `gci/ledger.py` — accept/reject capture and quality evaluation (Phase 1, module 8)
Append-only JSON-Lines audit trail: `record()` logs an `Advisory` (from
`provenance.py`) being surfaced, `respond()` logs the operator's
accept/reject/ignore decision against it. JSON Lines chosen over a database
for the same reason as elsewhere in this project — no dependency a judge's
clean checkout might not have, and it is trivially replay-able and
diff-friendly, which a real audit trail should be. `evaluate()` gives a
quality summary the shape Phase 2's `learning.py` trust-score reranking will
read directly: overall and per-source acceptance rate, mean confidence of
accepted vs rejected advisories (a calibration signal — if the two are
indistinguishable, confidence isn't actually informing operator trust), and
realised dollar value of accepted advisories. **10 tests.**

### `gci/api/` — FastAPI service (Phase 1, module 9)
Three files, clean separation: `datasource.py` (Connectivity layer — the
`DataSource` ABC plus `SimulatedDataSource`, its shipped implementation),
`service.py` (all business logic, testable with zero HTTP), `app.py` (thin
FastAPI wiring). `SimulatedDataSource` generates a small, fully deterministic
40-event demo corpus at startup (seed `20260726`, distinct from the training
corpus's `20260725`) and replays it as "live" data — `row_at(event_id, t_min)`
never returns a feature row past the requested cursor, carrying the
backward-only guarantee (D9) through to the API boundary. Endpoints:
`/api/health`, `/api/grades`, `/api/events`, `/api/events/{id}`, `/api/live`,
`/api/recommendations` (+ `POST .../feedback`), `/api/correlations`,
`/api/stabilization`, `/api/economics` (GET/PUT), `/api/trust`. `whatif` and
`copilot` are Phase 2 (`whatif.py`/`copilot.py` not yet built) and are
intentionally not stubbed — no placeholder endpoints per the project's own
rule.

**"Demo cannot fail" interpretation for this build:** rather than a frozen
JSON blob, the service degrades gracefully field-by-field — `risk_bundle`/
`forecast_bundle` load best-effort at startup (`GCIService.health()` reports
exactly which loaded and why not, never silently), and `/api/correlations`
falls back from the persisted corpus to the always-present demo events if
`data/events_series.npz` is missing. Every response is always real computed
output, never stale canned data.

**Bug found and fixed before shipping:** Starlette's default JSON renderer
calls `json.dumps(..., allow_nan=False)`, so any NaN in a response (e.g.
`settle_min` for a transition that never settles, `mean_confidence_accepted`
before any operator feedback exists — both legitimate values, not bugs)
crashed the endpoint with `ValueError: Out of range float values are not
JSON compliant`. Fixed with a `NanSafeJSONResponse` that recursively maps
non-finite floats to `null` at the HTTP boundary only — the Python-side
values stay honestly NaN, since a browser's `JSON.parse` would reject a
literal `NaN` token exactly like Starlette does.

Smoke-tested against a real running `uvicorn` server (not just the in-process
test client): a rushed 6.6-minute ramp on `LWC-52 -> BRD-120` correctly
triggers a 100% risk probability (SHAP-explained: `bw_dev_headroom_pct`,
`bw_abs_dev_pct` lead) and an optimizer recommendation to extend the ramp to
25 min, eliminating the predicted 7.9 min off-spec period, priced at
$304.55 (annualized $331k/yr at this scenario's implied frequency).

**Known issue carried forward:** `forecast_model.joblib` is 14.3 MB (9
`HistGradientBoostingRegressor` models — LightGBM unavailable, see §1a).
Combined with `risk_model.joblib` and other artefacts, `models/` alone may
approach the 10 MB submission limit; packaging will need either a slimmer
persisted forecast bundle or regeneration-not-shipping for `models/` too,
decided in the reserved packaging block. **28 tests** (17 service-level,
independent of HTTP, + 11 through FastAPI's `TestClient` against the real
app) in `tests/test_api.py`.

---

### `frontend/` — React dashboard (Phase 1, module 10 — Phase 1 complete)
Vite + plain React (no TypeScript, no component library, no chart library --
every chart is hand-rolled SVG against the dataviz skill's mark specs and the
validated reference palette, so the whole production bundle is 207 KB / 65 KB
gzipped). Six panels, each mapped to exactly one deliverable tag shown in its
header, matching `CLAUDE.md`'s "six simple panels over three beautiful ones"
guidance:

| Panel | Deliverable | Backed by |
|---|---|---|
| Live risk & forecast cone | 3 | `/api/live` |
| Recommendations (priced, provenanced, accept/reject) | 3, 5, 6 | `/api/recommendations` + feedback |
| Correlation explorer | 3 | `/api/correlations` |
| Stabilization impact | 4 | `/api/stabilization` |
| Trust & ledger | 6 | `/api/trust` |
| ROI assumptions (editable) | ROI engine | `/api/economics` |

Header carries the transition selector, a simulated-clock slider (`t_min`,
0-29 min) driving `/api/live` and `/api/recommendations` together, and a
`models live` / `degraded mode` pill sourced directly from `/api/health` --
the frontend surfaces the same honest degradation the backend reports rather
than hiding it.

**Verified in a real browser** (`claude-in-chrome`, not just a build check):
loaded against the live API, scrolled and inspected all six panels, clicked
Reject on a live advisory and watched the Trust panel update in real time
(2 surfaced -> 2 responded, acceptance rate 100% -> 50%, and the mean-confidence
comparison correctly flagged "confidence is not yet separating accepted from
rejected advice" once the rejected advisory's confidence exceeded the
accepted one's) -- the ledger's calibration signal (from `ledger.py`'s own
design intent) working end-to-end through real UI interaction, not just unit
tests. Zero console errors.

**Bug found and fixed in the browser, not by inspection:** `EconomicsPanel`
initialised its form state with `useState(economics || {})`. `economics` is
fetched asynchronously in the parent and arrives *after* this component's
first render, and `useState`'s initial value is only consumed once --
so every field in the ROI Assumptions panel rendered permanently empty
despite the data being correctly loaded and used everywhere else on the
page. Fixed with a `useEffect` that syncs local state whenever the
`economics` prop actually changes. This is exactly the class of bug static
analysis and a build check cannot catch and a real browser load does
immediately -- the reason "start the dev server and use the feature" is a
hard requirement for UI work, not a suggestion.

---

## 3. Pending modules

Phase 1 (MVP) is complete. Remaining work is Phase 2:

| Module | Phase | Purpose | Deliverable |
|---|---|---|---|
| `whatif.py` | 2 | Slider-driven twin replay | 2 |
| `copilot.py` | 2 | Grounded operator assistant | 4, 5 |
| `learning.py` | 2 | Trust scores, reranking, trust evolution | 6 |
| `benchmark.py` | 2 | Baseline vs optimized A/B and business case | 3 |

---

## 4. API endpoints

**Live.** `uvicorn gci.api.app:app --reload`. Frontend should build against
this exact surface:

| Method | Path | Purpose | Status |
|---|---|---|---|
| `GET` | `/api/health` | Liveness, model-loaded status, active `AdvisoryPolicy` | ✅ |
| `GET` | `/api/grades` | Grade library and recipe envelopes | ✅ |
| `GET` | `/api/events` | Historical (demo-corpus) transition index | ✅ |
| `GET` | `/api/events/{id}` | Full trajectory + labels for one transition | ✅ |
| `GET` | `/api/live` | Current transition state, risk, forecast cone | ✅ |
| `GET` | `/api/recommendations` | Ranked, priced, provenanced advice | ✅ |
| `POST` | `/api/recommendations/{id}/feedback` | Accept / reject capture | ✅ |
| `GET` | `/api/correlations` | Discovered relationships with novelty flags | ✅ |
| `GET` | `/api/stabilization` | Loop impact ranking on settling time | ✅ |
| `GET` | `/api/economics` / `PUT` | Read/update ROI assumptions | ✅ |
| `GET` | `/api/trust` | Ledger-derived acceptance/calibration summary | ✅ |
| `POST` | `/api/whatif` | Run the twin under operator-supplied setpoints | ⬜ Phase 2 (`whatif.py` not built) |
| `POST` | `/api/copilot` | Grounded natural-language query | ⬜ Phase 2 (`copilot.py` not built) |

`/api/live`, `/api/recommendations` and `/api/stabilization` all accept an
optional `event_id` query param (defaults to the demo corpus's largest
excursion) and `/api/live`/`/api/recommendations` also accept `t_min` — the
simulated-clock cursor into that transition (default 12.0 min).

---

## 5. Folder structure

```
gci/
├── PROJECT_LOG.md            <- this file (source of truth)
├── README.md
├── requirements.txt
├── gci/                      <- library
│   ├── __init__.py
│   ├── config.py             137 lines   spec, economics, provenance tags, policy
│   ├── grades.py             209 lines   grade library, tag dictionary, known loops
│   ├── twin.py               496 lines   digital twin + incremental stepper
│   ├── control.py            366 lines   S-curve, lead compensation, SIMC PI trim
│   ├── faults.py             428 lines   OU drift + 11 named fault types
│   ├── events.py             495 lines   closed-loop events, labelling, validation
│   ├── features.py           330 lines   104 leak-free features + caching
│   ├── forecast.py           349 lines   quantile forecast cone (+2/+5/+10 min)
│   ├── roi.py                187 lines   confidence-weighted $ pricing + advisory gate
│   ├── optimizer.py          260 lines   coordinate-descent ramp/setpoint search
│   └── ml/                              machine learning pipeline
│       ├── __init__.py        55 lines   pipeline contract + leakage guarantees
│       ├── splits.py         205 lines   event-wise stratified 3-way splits
│       ├── metrics.py        375 lines   metrics + early warning + alarm on-delay
│       ├── registry.py       215 lines   5 models, extensible by one entry
│       ├── explain.py        330 lines   tiered exact SHAP + consensus importance
│       └── pipeline.py       700 lines   orchestration, checkpoints, tuning
├── scripts/
│   ├── generate_data.py         151 lines   generate → validate → save → featurise
│   ├── train_risk_model.py      151 lines   train → compare → select → explain
│   └── train_forecast_model.py  127 lines   train quantile forecasters, evaluate, persist
├── models/                   <- generated artefacts
│   ├── risk_model.joblib     fitted classifier + features + threshold
│   ├── forecast_model.joblib fitted quantile regressors + features + horizons
│   ├── forecast_metrics.json pinball loss, coverage, MAE/RMSE per horizon/split
│   ├── metrics.json          every model, every split, environment record
│   ├── comparison_*.csv      side-by-side model comparison
│   ├── confusion_matrix.json per model, per split
│   ├── feature_importance.csv consensus attribution
│   ├── shap_explanations.json mean |SHAP| + worked local examples
│   ├── warning_detail.json   per-event early-warning outcomes
│   ├── evaluation_report.md  human-readable report
│   └── checkpoints/          per-model resumable checkpoints
├── tests/                   ~2,600 lines  226 unittest tests
│   ├── test_twin.py          physics, mass balance, FOPDT dynamics
│   ├── test_control.py       S-curve, SIMC gains, feasibility, envelope clamps
│   ├── test_events.py        faults, labelling, dataset health, persistence
│   ├── test_features.py      label logic, leakage, splitting
│   ├── test_ml.py            splits, on-delay, warning time, exact SHAP, e2e
│   ├── test_forecast.py      target construction, leakage, quantile monotonicity
│   ├── test_roi.py           pricing formula, band ordering, advisory gate
│   ├── test_optimizer.py     search correctness, feasibility floor, caching
│   ├── test_discovery.py     lag recovery on planted signal, novelty flagging
│   ├── test_stabilization.py sensitivity ranking, never-worse-than-baseline
│   ├── test_provenance.py    grounded explanations, policy gate
│   ├── test_ledger.py        record/respond, persistence, quality evaluation
│   └── test_api.py           28 tests: service layer + real FastAPI TestClient
├── data/                     <- generated, not committed
│   ├── events_series.npz     35.75 MB   trajectory cube (1500, 360, 22)
│   ├── events_meta.json      2.14 MB    plans, faults, labels
│   ├── features.pkl          cached feature frame (regenerable)
│   ├── ledger.jsonl          accept/reject audit trail (grows at runtime)
│   └── validation.json       health report
└── frontend/                 <- React dashboard (Vite, plain JS, no chart lib)
    ├── index.html
    ├── vite.config.js         dev-server proxy: /api -> 127.0.0.1:8000
    └── src/
        ├── App.jsx            header + 6-panel grid layout
        ├── styles.css         design tokens (validated dataviz palette)
        ├── lib/api.js         fetch wrappers, one per endpoint
        └── components/        ForecastCone, RecommendationsPanel,
                                CorrelationTable, StabilizationBars,
                                TrustPanel, EconomicsPanel
```

---

## 6. Design decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | **Physics-informed digital twin as primary data source** | No public dataset of paper-machine grade changes exists. A twin grounded in mass/energy balance is defensible, fully labelled, and lets us prove causality — which historical data alone could not. |
| D2 | **Closed-loop simulation, not open-loop ramps** | An open-loop ramp fails *every* transition (27% deviation), which would destroy the ML problem. Modelling the plant's own controller means failures come from disturbances and bad plans — the real causes. |
| D3 | **S-curve target trajectory** | A linear ramp demands a step change in actuator *rate* at onset, which no slew-limited drive can deliver, so the sheet always trails. Zero-rate endpoints remove the onset transient. Cost: peak rate is 1.5× average, raising the feasibility floor. |
| D4 | **Lead compensation on all actuators, not just trimmed ones** | Machine speed is the dominant basis-weight driver on a large transition (430 m/min vs 11 m³/min stock). Leaving it uncompensated left the sheet trailing 8–11% for the entire ramp. |
| D5 | **Lead = θ + τ exactly** | For a ramp input through FOPDT the output lags by exactly (θ + τ). Using a fraction of it is a tuning error, not a design choice. |
| D6 | **SIMC tuning from numerically-obtained process gains** | Gains are computed by perturbing the twin at the operating point, so one controller works across 45–150 g/m² without retuning — and the tuning is textbook-defensible rather than hand-fitted. |
| D7 | **Feasibility floor drives plan sampling** | Makes long transitions genuinely harder than short ones, matching reality, and yields a high-value zero-uncertainty recommendation ("this ramp is below the physical floor"). |
| D8 | **Inverse-solve initial conditions at the prevailing disturbance state** | Before a grade change, existing controls have already trimmed out disturbance offsets. Without this, events began spuriously off-spec at t=0. |
| D9 | **Strictly backward-looking features, verified by test** | Leakage is the easiest way to produce an impressive but worthless model. `test_no_future_leakage` corrupts the future and asserts past features are bit-identical. |
| D10 | **Event-wise train/test splits** | Rows within a transition are heavily autocorrelated; a random row split would leak near-duplicates across the boundary and inflate every metric. |
| D11 | **Ornstein–Uhlenbeck drift, not white noise** | Real disturbances are time-correlated. White noise would make lagged-correlation discovery a coin flip. |
| D12 | **`unittest`, not `pytest`** | Zero extra dependencies; judges can run `python -m unittest` on a clean machine. |
| D13 | **Data regenerated, not shipped** | Fully seed-determined, rebuilt in 25 s. Keeps the submission under the 10 MB limit. |
| D14 | **Advisory-only, no control writeback** | Mirrors how APC advisory products are actually commissioned; deployable without a safety case. |
| D15 | **Classification metrics exclude rows already off-spec** | Predicting a breach while the sheet is visibly off-spec is trivial and worthless — the operator sees the same trend line. Including those rows inflates every score. |
| D16 | **Event detection rate is the headline, not row recall** | An excursion only has to be caught *once* to warn the operator. Row-level recall (0.580) understates usefulness; event detection (0.847) is what the product delivers. |
| D17 | **Alarm on-delay (ISA-18.2)** | A single noisy sample crossing threshold is not evidence, but it is a nuisance alarm. Because the risk signal is strongly autocorrelated, on-delay removes false alarms at almost no cost in warning time. |
| D18 | **Model selection on the product objective, not PR-AUC** | Ranking by PR-AUC then tuning the winner's threshold optimises two different objectives in sequence. Each model now gets its own tuned operating point, then selection minimises per-event false alarms subject to a detection floor. PR-AUC is still reported as model-quality diagnostic. |
| D19 | **Tiered exact SHAP, with approximation labelled as such** | Four independent exact paths mean every registered model — including the linear baseline, via analytic `coef·(x−E[x])` — gets exact per-prediction attribution. Explainability never has to be traded against accuracy when choosing a model. |
| D20 | **Optional dependencies probed, never assumed** | LightGBM, XGBoost and SHAP are all optional; scikit-learn's histogram boosting is a competitive always-available stand-in. The pipeline yields a usable model on a bare `pip install scikit-learn`. This is the "demo cannot fail" requirement applied to training. |
| D21 | **Per-model checkpointing with dataset fingerprinting** | Training five ensembles is minutes of work that should not be lost to an interruption. Fingerprinting the *data* (not just feature names and seed) is what makes resumption safe — a weaker check silently restored models trained on a different corpus. |
| D22 | **Forecast targets live in `forecast.py`, never in `features.py`** | The risk model's feature contract must stay strictly backward-looking (D9); a forward-looking regression target added to that same frame would be exactly the leakage `test_no_future_leakage` exists to catch. `forecast_feature_columns()` re-excludes the target columns defensively rather than trusting call order. |
| D23 | **Quantile rearrangement, not a joint model** | Three independent per-quantile regressors are simpler and reuse the existing tree-ensemble tooling, but can cross near the tails. Row-wise sorting the predictions (Chernozhukov, Fernandez-Val & Galichon 2010) is the standard, cheap fix rather than a joint quantile model. |
| D24 | **Off-spec tonne priced as margin *plus* rework, not either alone** | The conservative assumption is that excursion product is reworked as broke, not downgraded and shipped — see known limitation 6. So it costs the mill twice: the margin it never earns, and the cost of repulping it. Pricing only one term would understate every recommendation's value. |
| D25 | **Optimizer uses coordinate descent, not a 3-D grid** | The three tunable knobs (ramp, lead, trim aggressiveness) govern close-to-independent effects. A full `n^3` grid at full-fidelity (30-minute closed-loop) simulation cost buys negligible accuracy over sweeping one knob at a time and narrowing — verified by `test_finds_real_improvement_for_a_rushed_plan`, which needs the search to actually beat, not just match, a bad baseline. |

---

## 7. Assumptions

**Process and machine**
1. Single fourdrinier machine, 6.0 m trim, one MD control zone. No CD profile control — the problem statement scopes to MD.
2. Headbox consistency ~0.9%, fiber first-pass retention 0.95, filler slurry 30% consistency at 0.60 retention. Mid-range published values.
3. Transport delays: 25 s wire→scanner for basis weight, 60 s for moisture (dryer section), 35 s ash, 40 s caliper. Within the commonly reported 20–90 s range.
4. Ash content is treated as equal to retained filler fraction (fiber ash neglected).
5. Drying capacity scales as steam pressure^0.85; moisture floor 1.5%.
6. Sheet density is empirical in ash, moisture and press load — the one non-first-principles relationship in the twin.

**Operations**
7. QCS scan and control interval 5 s. Real scanners average over a 20–30 s traverse; 5 s is a reasonable proxy for a fixed-point or averaged reading.
8. Event window 30 min, ramp starting at minute 5.
9. Grade changes occur ~3.2×/day over 340 operating days.
10. About one third of transitions run clean; the remainder carry 1–3 concurrent faults.
11. Roughly a quarter of plans are rushed below the feasibility floor by production pressure.

**Machine learning**
14. Corpus of 1,500 events: 908 train / 296 validation / 296 test, split by event.
15. Prediction target: basis weight deviates >2.5% from setpoint within the next 10 minutes.
16. Operating point tuned for an event detection floor of 0.80.
17. Per-event metrics carry a standard error near 0.05 at 500-event corpus size (smaller, but not zero, at 1,500); differences smaller than ~0.10 in detection rate are not meaningful.

**Economics** (all editable in `config.Economics`, surfaced in the UI)
12. Net margin \$95/tonne, rework \$42/tonne, steam \$9/GJ at 2.4 GJ/tonne.
13. ROI uncertainty band P10–P90 at 0.70×–1.35× the point estimate.

---

## 8. Known limitations

1. **No real mill data.** The twin is calibrated to published ranges, not to a specific machine. Any commissioning would require re-identification against site historian data. Stated openly in the deck rather than hidden.
2. **No CD (cross-direction) control.** Basis weight and moisture profiles across the sheet are out of scope; only machine-direction behaviour is modelled.
3. **Ash model neglects fiber ash** and treats filler retention as independent of shear history.
4. **Sheet density is empirical**, so caliper is the least trustworthy of the four quality variables. It is not used as a primary risk driver.
5. **Off-spec rate is 61.7%**, higher than a well-run mill would see. This is a deliberate consequence of sampling hard transitions and rushed ramps to create a learnable problem; it inflates the *baseline* against which improvement is measured, and the benchmark must report relative improvement, not absolute rates.
6. **The hardest transition (NP-45 → BRD-120, a 2.7× basis weight change) retains ~3.4% deviation** even at generous ramp times. Real mills would stage this through an intermediate grade. Not modelled.
7. **WFU-80 saturates the dryer section** (needs ~1100 kPa against a 1092 kPa ceiling). Emergent from the physics, not designed in — transitions into WFU-80 are genuinely the hardest on this machine.
8. **Faults are injected, not learned.** The catalogue is drawn from reported mill failure modes, but a real deployment would discover site-specific modes.
9. **`data/features.csv.gz` (24 MB) is a stale artefact** from an earlier run before feature persistence was made opt-in. It cannot be deleted from this environment; it must be excluded from the submission zip.
10. **Single-threaded generation**: 500 events in ~56 s (two 250-event chunks). Fine here, but would not scale to tens of thousands of events without parallelisation.
11. **Row-level recall is 0.580** and is easy to misquote out of context. Event detection rate (0.847) is the meaningful figure; both are always reported together.
12. **False alarms at 0.179 per clean event** — about one nuisance alarm per six clean transitions. Acceptable for advisory use and the weakest product metric. The ROI gate (`AdvisoryPolicy.min_value_usd_to_surface`) will suppress low-value alarms further once the ROI engine lands.
13. **Per-event metrics are noisy** at 500 events (~59 breaching events in validation). Validation detection 0.831 vs test 0.847 is well inside sampling error, not evidence of a real difference.
14. **Random Forest is trimmed** (150 trees, depth 10) to fit this environment's per-call wall-clock budget, not for statistical reasons.
15. **`xgboost-cpu` is required** rather than `xgboost`: the default wheel bundles 131 MB of CUDA libraries that could not be installed here.
16. **No hyperparameter search.** All five models use single sensible configurations. A tuned sweep would likely add a little PR-AUC but was judged a poor use of remaining time versus completing the product surface.

---

## 9. Test results

```
Ran 155 tests in 250.2s — OK
```

| Suite | Tests | Covers |
|---|---|---|
| `test_twin.py` | 21 | Round-trip on all 7 grades, mass-balance linearity and inverse-speed scaling, ash bounds, six monotonicity properties, zero steady-state drift over 20 min, dead-time enforcement, first-order response magnitude, convergence to analytic steady state, slew-limit enforcement, seed determinism |
| `test_control.py` | 20 | S-curve endpoints/monotonicity/zero-endpoint-rate/1.5× peak factor, SIMC gain sanity and tuning direction, output limits, anti-windup recovery, feasibility floor ordering, trajectory endpoints, lead precedes setpoint, recipe envelope never violated under absurd measurements, process gain signs |
| `test_events.py` | 22 | Disturbance bounds and coverage, OU autocorrelation > 0.9, all 11 faults produce finite non-zero profiles, nothing before onset, actuator faults routed as overrides, sampler never duplicates, deviation/settle/label maths, events start on old grade and end on new, rushed worse than generous, seed reproducibility, disk round-trip |
| `test_features.py` | 15 | Forward-looking label boundaries, time-to-breach consistency, no NaN/Inf across 104 features, warmup dropping, label exclusion from feature set, manual deviation check, projection formula, **no future leakage**, disjoint event-wise splits, determinism |
| `test_ml.py` | 37 | Registry availability and graceful skipping; 3-way splits disjoint, complete, deterministic, overlap-rejecting; alarm on-delay semantics and **no leakage across event boundaries**; threshold criteria; metric internal consistency; early-warning time measured only from pre-breach in-spec alarms; false-alarm accounting; **exact SHAP for every registered model**; linear SHAP additivity against model logits; pipeline end-to-end, artefact writing, unknown-model tolerance, checkpoint round-trip |
| `test_forecast.py` | 12 | Target construction matches a hand-computed shift; NaN exactly at the window boundary; **forward perturbation changes only the rows a horizon actually reaches**; forecast targets excluded from the model's own feature list; quantile monotonicity after rearrangement; event-wise split shared with the risk model; save/load round-trip |
| `test_roi.py` | 19 | Production-rate arithmetic; margin+rework pricing formula; confidence scaling; band ordering for both positive and (cost-forced) negative point estimates; invalid-confidence rejection; plan-comparison clamps negative improvement to zero; `AdvisoryPolicy` gating on both value and confidence floors; portfolio extrapolation uses the mean, not a sum |
| `test_optimizer.py` | 9 | Deterministic evaluation under a fixed seed; search never regresses the baseline; a deliberately rushed plan is strictly improved, not just tied; ramp search respects the physical feasibility floor; cache hit/clear correctness; ROI pricing hookup |

### Dataset health (1,500 events, current)

| Metric | Value |
|---|---|
| Events | 1,500 (one call, seed 20260725) |
| Generation time | 72 s |
| Off-spec rate | 62.5% |
| Settled within window | 88.7% |
| Distinct grade pairs | 42 of 42 possible |
| Feature samples | 504,000 × 104 features |
| Positive rate (breach within 10 min) | 30.8% |
| Validation issues | none |

### Risk model results (current, restored capacity, 1,500 events)

Three models available on this machine (LightGBM/XGBoost unavailable — see
§1a). **Histogram Gradient Boosting selected** on the product objective;
threshold 0.85, alarm on-delay 1 sample (5 s).

Validation (used for all selection and tuning):

| Model | PR-AUC | Precision | Recall | F1 | FPR | Detection | Med. warning | False alarms/event |
|---|---|---|---|---|---|---|---|---|
| Random Forest | 0.785 | 0.882 | 0.478 | 0.620 | 0.020 | 0.805 | 4.58 min | 0.108 |
| **Hist Gradient Boosting** | 0.782 | 0.881 | 0.443 | 0.590 | 0.019 | 0.811 | 4.42 min | **0.072** |
| Logistic Regression | 0.759 | 0.821 | 0.500 | 0.621 | 0.034 | 0.822 | 4.88 min | 0.315 |

Held-out test, scored once after the winner was fixed:

| Metric | Value |
|---|---|
| PR-AUC | 0.827 |
| Precision | 0.894 |
| Recall (row-level) | 0.494 |
| F1 | 0.636 |
| False positive rate | 0.017 |
| **Event detection rate** | **0.827** |
| **Median warning time** | **4.67 min** |
| False alarms per clean event | **0.099** |

Top consensus features: `bw_dev_headroom_pct`, `t_since_ramp_min`,
`transition_magnitude`, `plan_trim_enabled`, `plan_ramp_min`, `ramp_progress`,
`mv_stock_flow_remaining_frac`, `plan_tau_c_scale` — same causal character as
the 500-event run (plan parameters carry as much signal as current
deviation), computed now on a 3,000-row exact-SHAP sample and 6,000-row/8-repeat
permutation importance instead of 600/1,500-3.

**Historical LightGBM result retained for reference** (500 events, sandbox
capacity): PR-AUC 0.821, detection 0.847, median warning 4.50 min, false
alarms 0.179/event. See §1a for the direct before/after comparison and why
detection moving 0.847→0.827 is within noise while false alarms improving
0.179→0.099 is not.

**Discriminative signal:** off-spec rate is 54.5% for ramps at or above the
feasibility floor versus 79.3% for rushed ramps — predictive but not
deterministic, which is what makes the modelling problem real.

---

## 10. Next steps

**Phase 1 (MVP) is complete.** All ten modules built, tested (251 Python
tests + live in-browser verification), and wired end to end:

1. ~~`gci/ml/` + risk predictor~~ — ✅ Hist Gradient Boosting on this machine, test PR-AUC 0.827, detection 0.827, 4.67 min median warning
2. ~~`forecast.py`~~ — ✅ quantile forecast cone, +2/+5/+10 min, P10/P50/P90
3. ~~`roi.py`~~ — ✅ confidence-weighted dollar pricing + `AdvisoryPolicy` gate
4. ~~`optimizer.py`~~ — ✅ coordinate-descent ramp/setpoint search over the twin, cached for demo speed
5. ~~`discovery.py`~~ — ✅ lagged correlation + mutual information + novelty vs `KNOWN_LOOPS`
6. ~~`stabilization.py`~~ — ✅ loop impact ranking on settle time
7. ~~`provenance.py`~~ — ✅ unified `Advisory` schema, grounded explanations, policy gate
8. ~~`ledger.py`~~ — ✅ accept/reject capture, quality evaluation
9. ~~`api/`~~ — ✅ FastAPI, 12 live endpoints, graceful degradation
10. ~~`frontend/`~~ — ✅ React dashboard, 6 panels, verified live in-browser

**What's next is a choice, not a default next-module:**

- **Option A — Phase 2** (`whatif.py`, `copilot.py`, `learning.py`,
  `benchmark.py`): What-If Studio, AI Copilot, trust-score reranking, formal
  A/B business case. Time-permitting only; every graded deliverable already
  has working evidence without these.
- **Option B — Reserved block now**: architecture doc, 6-slide deck,
  dashboard screenshots, packaging under 10 MB, clean-checkout rehearsal.
  `CLAUDE.md` explicitly reserves the last ~2 hours for this and warns
  against writing the deck in the final ten minutes.

**Packaging item resolved (2026-07-26):** `scripts/package_submission.sh`
builds the zip from `git ls-files`, excluding the two committed `.joblib`
artefacts (16 MB combined — kept in git for the Render deploy, regenerable
via `scripts/train_*.py`) and the stale `HANDOFF.md`. Output is 212 KB.

---

## 11. Deliverable coverage matrix

| # | Honeywell deliverable | Evidence | Status |
|---|---|---|---|
| 1 | Develop a solution to the challenge | Full stack: Phase 0 + all 9 engine modules + live API + React dashboard, verified in a real browser | ✅ Phase 1 complete |
| 2 | Document building blocks and module communication | Architecture doc + this log | ⬜ Reserved |
| 3 | Dashboard: new correlations, their impact, future state on current trend, suggested setpoints | `discovery.py` + Correlation Explorer panel; `forecast.py` cone + Live risk & forecast cone panel; `optimizer.py` setpoints + Recommendations panel — all verified rendering live data in-browser | ✅ Complete |
| 4 | Dashboard: loops/parameters driving stabilization + setpoints to stabilize faster | `stabilization.py` + Stabilization impact panel — verified rendering live data in-browser | ✅ Complete |
| 5 | Tag every suggestion with source of inference | `config.Source`, exact SHAP, `provenance.py` — source tag + confidence + grounded explanation visible on every advisory card in the Recommendations panel | ✅ Complete |
| 6 | Allow accept/reject, record responses to evaluate quality | `ledger.py` + Accept/Reject buttons + Trust & ledger panel — verified end-to-end in-browser (click Reject -> ledger updates -> acceptance rate and calibration text update live). `learning.py` (Phase 2 reranking) still pending | ✅ Complete (core), Phase 2 extends |
| + | Presentation in the provided template (6-slide limit) | Reserved block | ⬜ Reserved |

---

## 12. Dev journal — 2026-07-25

Long one. Started by moving off the sandbox and onto real hardware, which
meant undoing all the corners I'd cut for a 45-second shell limit: Random
Forest back up to full depth and tree count, SHAP and permutation sampling
back up to a size I'd actually trust, corpus regenerated at 1500 events
instead of 500. That alone surfaced a real environment problem — LightGBM and
XGBoost won't load on this machine because of an architecture mismatch
between Homebrew and Python, not something I could just pip-install my way
out of. Decided it wasn't worth chasing a second Homebrew install for two of
five candidate models when the pipeline was already built to degrade
gracefully, so I let it degrade and moved on. Histogram Gradient Boosting
ended up the selected model, and honestly the false-alarm rate improved a
lot even though detection dipped slightly within noise.

After that it was module after module, in the order I'd planned:
forecasting, ROI pricing, the setpoint optimizer, correlation discovery,
stabilization ranking, provenance, the accept/reject ledger, then the API,
then the dashboard. Found and fixed a few real bugs along the way instead of
just writing code and hoping — a grade code typo in a test, a missing
early-stopping flag that made forecast training run way longer than it
needed to, a NaN value that crashed the API's JSON responses, and a React
state bug that left an entire form silently empty because it only read its
props once instead of syncing with them. That last one I only caught because
I actually opened the thing in a browser and clicked around instead of
trusting that a clean build meant a working app.

By the end of the day all six graded deliverables had working code behind
them, not just a plan for them, and I could click through a real dashboard
talking to a real API talking to real trained models. Spent the rest of the
evening on repo hygiene — fixing commit authorship, checking nothing junky
got committed, writing an honest README instead of the placeholder one from
Phase 0. Tomorrow's the deadline, so what's left is a judgment call: spend
remaining time on Phase 2 features, or lock in what's here and put the
effort into packaging and the deck. Leaning toward the latter.

---

## 13. Dev journal — 2026-07-26

Deadline day. Decided against Phase 2 — chose to spend the day making what's
already built provably solid instead of adding more surface area a judge
could poke a hole in the day of the demo.

Went looking for loose ends first rather than assuming everything from
yesterday was tight. Found two real gaps in the provenance work: the
optimizer never actually produced a `RECIPE_LIMIT` advisory even though the
enum existed for exactly that case (a baseline plan below the physical
feasibility floor), so every recommendation was labelled `PHYSICS_MODEL`
regardless of whether the twin did any real optimization or the answer was
just "the actuator can't move that fast." Fixed that at the source in
`optimizer.py`, not by relabelling in the UI. Also found `HISTORICAL_DATA`
and `OPERATOR_PRECEDENT` sitting in the `Source` taxonomy with nothing
behind them — no code path produces either one. Left them defined rather
than deleting or faking a producer just to make the enum look fully used;
an honest gap beats a fabricated one.

Then went after the API's edge behavior on purpose, on the theory that a
judge bringing their own numbers is exactly the scenario a demo built
against one curated storyline hasn't been tested on. Wrote 25 tests that
throw malformed JSON, wrong types, out-of-range values, and unknown IDs at
every endpoint that takes a body or an identifier. One of them caught a real
bug immediately: a bare `NaN` JSON literal crashed FastAPI's own built-in
validation-error handler, because that handler responds with a plain
`JSONResponse` instead of the app's NaN-safe one — the earlier NaN fix only
covered the success path, not the rejection path. Small thing, but exactly
the kind of small thing that turns into a live 500 in front of a judge.

Checked performance before touching anything, instead of assuming the app
was slow — DevTools Network tab, panel by panel. Nothing crossed 2 seconds,
so I left the code alone rather than "optimizing" something that wasn't
broken.

Did one more UI pass, deliberately conservative: real loading skeletons
instead of a panel just going blank while data is in flight, consistent
decimal formatting across panels, a bit more breathing room. No new
components, no new colors, no restructuring — the six-panel layout from
yesterday earns its spot on the slide, it didn't need reinventing.

Built the two things that were still missing real evidence rather than a
plan for evidence: an aggregate benchmark of the optimizer across 100
transitions (57% improve, median 0.25 min / P95 7.84 min off-spec avoided,
$6,372 total priced value) instead of leaning on one cherry-picked demo
transition, and an actual packaging script, because "the repo is probably
under 10 MB" turned out to be wrong by 7 MB once the two trained model
files were counted — the fix was excluding them from the zip (not the git
repo, since Render still needs them) and regenerating from
`scripts/train_*.py` if a judge wants to retrain from scratch.

Full suite is at 251 tests, all green, no regressions from any of today's
changes. What's left is the deck, the screenshots, and one clean-checkout
rehearsal — the part CLAUDE.md said to reserve two hours for, and the part
I intend to actually reserve time for instead of writing at 11pm.
