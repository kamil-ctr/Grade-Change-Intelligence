# PROJECT LOG — Grade Change Intelligence (GCI)

**Honeywell Hackathon — Question 5A: Grade Change Intelligence in Paper Making Process**

This file is the project's source of truth. It is updated after every major
implementation step and kept synchronised with the codebase.

| | |
|---|---|
| **Last updated** | 2026-07-25 — Phase 1 module 1 complete |
| **Current phase** | Phase 1 (MVP), module 1 of 9 done |
| **Tests** | 115 passing, 0 failing |
| **Lines of code** | 5,700 (3,807 source / 1,321 tests / 278 scripts) |
| **Submission deadline** | 2026-07-26, 23:59 |

---

## 1. Status at a glance

| Phase | Scope | Status |
|---|---|---|
| **Phase 0** | Twin, controller, faults, events, features, tests | ✅ **Complete** |
| **Phase 1** | MVP — risk model, ROI, optimizer, discovery, API, dashboard | 🔄 **1/9 modules** |
| **Phase 2** | What-If Studio, AI Copilot, feedback learning, benchmark | ⬜ Pending |
| **Phase 3** | Shadow-mode scoring, public-dataset validation | ⬜ Stretch |
| **Reserved** | Architecture doc, 6-slide deck, packaging, rehearsal | ⬜ Reserved (last 2 hrs) |

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

---

## 3. Pending modules

| Module | Phase | Purpose | Deliverable |
|---|---|---|---|
| `forecast.py` | 1 | Quantile trajectory forecast cone (**next task**) | 1, 3 |
| `roi.py` | 1 | Confidence-weighted value per recommendation | 3 |
| `optimizer.py` | 1 | Bounded setpoint/ramp search over the twin | 2, 4 |
| `discovery.py` | 1 | Lagged correlation + MI + novelty scoring | 3 |
| `stabilization.py` | 1 | Loop impact ranking on settling time | 4 |
| `provenance.py` | 1 | Source tagging and confidence on every suggestion | 5 |
| `ledger.py` | 1 | Accept/reject capture and quality evaluation | 6 |
| `api/` | 1 | FastAPI service + baked demo snapshot | 1–6 |
| `frontend/` | 1 | React dashboard, all required panels | 3, 4 |
| `whatif.py` | 2 | Slider-driven twin replay | 2 |
| `copilot.py` | 2 | Grounded operator assistant | 4, 5 |
| `learning.py` | 2 | Trust scores, reranking, trust evolution | 6 |
| `benchmark.py` | 2 | Baseline vs optimized A/B and business case | 3 |

---

## 4. API endpoints

**None yet** — the FastAPI service is a Phase 1 module. Planned surface,
recorded here so the frontend can be built against a fixed contract:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Liveness and model-loaded status |
| `GET` | `/api/grades` | Grade library and recipe envelopes |
| `GET` | `/api/events` | Historical transition index |
| `GET` | `/api/events/{id}` | Full trajectory + labels for one transition |
| `GET` | `/api/live` | Current transition state, risk, forecast cone |
| `GET` | `/api/recommendations` | Ranked advice with ROI and provenance |
| `POST` | `/api/recommendations/{id}/feedback` | Accept / reject capture |
| `GET` | `/api/correlations` | Discovered relationships with novelty flags |
| `GET` | `/api/stabilization` | Loop impact ranking on settling time |
| `POST` | `/api/whatif` | Run the twin under operator-supplied setpoints |
| `POST` | `/api/copilot` | Grounded natural-language query |
| `GET` | `/api/economics` / `PUT` | Read/update ROI assumptions |
| `GET` | `/api/trust` | Trust scores and acceptance history |

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
│   └── ml/                              machine learning pipeline
│       ├── __init__.py        55 lines   pipeline contract + leakage guarantees
│       ├── splits.py         205 lines   event-wise stratified 3-way splits
│       ├── metrics.py        375 lines   metrics + early warning + alarm on-delay
│       ├── registry.py       215 lines   5 models, extensible by one entry
│       ├── explain.py        330 lines   tiered exact SHAP + consensus importance
│       └── pipeline.py       700 lines   orchestration, checkpoints, tuning
├── scripts/
│   ├── generate_data.py      151 lines   generate → validate → save → featurise
│   └── train_risk_model.py   151 lines   train → compare → select → explain
├── models/                   <- generated artefacts
│   ├── risk_model.joblib     fitted LightGBM + features + threshold
│   ├── metrics.json          every model, every split, environment record
│   ├── comparison_*.csv      side-by-side model comparison
│   ├── confusion_matrix.json per model, per split
│   ├── feature_importance.csv consensus attribution
│   ├── shap_explanations.json mean |SHAP| + worked local examples
│   ├── warning_detail.json   per-event early-warning outcomes
│   ├── evaluation_report.md  human-readable report
│   └── checkpoints/          per-model resumable checkpoints
├── tests/                   1,321 lines  115 unittest tests
│   ├── test_twin.py          physics, mass balance, FOPDT dynamics
│   ├── test_control.py       S-curve, SIMC gains, feasibility, envelope clamps
│   ├── test_events.py        faults, labelling, dataset health, persistence
│   ├── test_features.py      label logic, leakage, splitting
│   └── test_ml.py            splits, on-delay, warning time, exact SHAP, e2e
└── data/                     <- generated, not committed
    ├── events_series.npz     7.1 MB   trajectory cube (300, 360, 22)
    ├── events_meta.json      0.4 MB   plans, faults, labels
    └── validation.json       health report
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
14. Corpus of 500 events: 304 train / 98 validation / 98 test, split by event.
15. Prediction target: basis weight deviates >2.5% from setpoint within the next 10 minutes.
16. Operating point tuned for an event detection floor of 0.80.
17. Per-event metrics carry a standard error near 0.05 at this corpus size; differences smaller than ~0.10 in detection rate are not meaningful.

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
Ran 115 tests in 20.3s — OK
```

| Suite | Tests | Covers |
|---|---|---|
| `test_twin.py` | 21 | Round-trip on all 7 grades, mass-balance linearity and inverse-speed scaling, ash bounds, six monotonicity properties, zero steady-state drift over 20 min, dead-time enforcement, first-order response magnitude, convergence to analytic steady state, slew-limit enforcement, seed determinism |
| `test_control.py` | 20 | S-curve endpoints/monotonicity/zero-endpoint-rate/1.5× peak factor, SIMC gain sanity and tuning direction, output limits, anti-windup recovery, feasibility floor ordering, trajectory endpoints, lead precedes setpoint, recipe envelope never violated under absurd measurements, process gain signs |
| `test_events.py` | 22 | Disturbance bounds and coverage, OU autocorrelation > 0.9, all 11 faults produce finite non-zero profiles, nothing before onset, actuator faults routed as overrides, sampler never duplicates, deviation/settle/label maths, events start on old grade and end on new, rushed worse than generous, seed reproducibility, disk round-trip |
| `test_features.py` | 15 | Forward-looking label boundaries, time-to-breach consistency, no NaN/Inf across 104 features, warmup dropping, label exclusion from feature set, manual deviation check, projection formula, **no future leakage**, disjoint event-wise splits, determinism |
| `test_ml.py` | 37 | Registry availability and graceful skipping; 3-way splits disjoint, complete, deterministic, overlap-rejecting; alarm on-delay semantics and **no leakage across event boundaries**; threshold criteria; metric internal consistency; early-warning time measured only from pre-breach in-spec alarms; false-alarm accounting; **exact SHAP for every registered model**; linear SHAP additivity against model logits; pipeline end-to-end, artefact writing, unknown-model tolerance, checkpoint round-trip |

### Dataset health (500 events)

| Metric | Value |
|---|---|
| Events | 500 (two 250-event chunks, seeds 20260725 / 20260726) |
| Generation time | 56 s total |
| Off-spec rate | 62.6% |
| Settled within window | 89.6% |
| Distinct grade pairs | 42 of 42 possible |
| Feature samples | 168,000 × 104 features |
| Positive rate (breach within 10 min) | 29.5% |
| Validation issues | none |

### Risk model results

Five models compared. **LightGBM selected** on the product objective;
threshold 0.60, alarm on-delay 3 samples (15 s).

Validation (used for all selection and tuning):

| Model | PR-AUC | Precision | Recall | F1 | FPR | Detection | Med. warning | False alarms/event |
|---|---|---|---|---|---|---|---|---|
| XGBoost | 0.782 | 0.824 | 0.518 | 0.636 | 0.033 | 0.831 | 4.58 min | 0.205 |
| **LightGBM** | 0.776 | 0.815 | 0.523 | 0.637 | 0.036 | 0.831 | 4.42 min | **0.128** |
| Hist Gradient Boosting | 0.772 | 0.823 | 0.507 | 0.627 | 0.033 | 0.847 | 4.58 min | 0.179 |
| Random Forest | 0.761 | 0.846 | 0.499 | 0.628 | 0.027 | 0.831 | 4.42 min | 0.231 |
| Logistic Regression | 0.746 | 0.744 | 0.560 | 0.639 | 0.058 | 0.814 | 4.92 min | 0.333 |

Held-out test, scored once after the winner was fixed:

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

SHAP method: `shap.TreeExplainer` (exact). Top consensus features:
`t_since_ramp_min`, `bw_dev_headroom_pct`, `plan_ramp_min`,
`plan_trim_enabled`, `bw_abs_dev_pct`, `ramp_progress`, `plan_lead_scale`.

That ranking is a useful sanity check: the model leans on *how the transition was
planned* (ramp duration, whether trim is enabled, lead scale) as much as on the
current deviation — which is exactly the causal structure the twin encodes, and
exactly what makes prescriptive advice possible rather than just alarms.

**Discriminative signal:** off-spec rate is 54.5% for ramps at or above the
feasibility floor versus 79.3% for rushed ramps — predictive but not
deterministic, which is what makes the modelling problem real.

---

## 10. Next steps

**NEXT IMMEDIATE TASK: `forecast.py` — future trajectory forecasting.**
Quantile regression on basis-weight deviation at +2/+5/+10 min to produce the
dashboard's forecast cone. Reuses the existing splits and leakage controls.

**Then, in order:**
1. ~~`gci/ml/` + risk predictor~~ — ✅ done (LightGBM, test PR-AUC 0.821, 4.5 min median warning)
2. `roi.py` — confidence-weighted value model.
3. `optimizer.py` — bounded ramp/setpoint search over the twin, cached for demo speed.
4. `discovery.py` — lagged correlation + mutual information + novelty vs `KNOWN_LOOPS`.
5. `stabilization.py` — loop impact ranking on settle time.
6. `provenance.py` + `ledger.py` — source tags, confidence, accept/reject capture.
7. `api/` — FastAPI with baked snapshot fallback.
8. `frontend/` — React dashboard, all required panels.

**Before submission:** exclude `data/` from the zip; verify a clean-checkout run;
capture dashboard screenshots for the Artifacts slide.

---

## 11. Deliverable coverage matrix

| # | Honeywell deliverable | Evidence | Status |
|---|---|---|---|
| 1 | Develop a solution to the challenge | Full stack, Phases 0–2 | 🔄 Phase 0 + risk model done |
| 2 | Document building blocks and module communication | Architecture doc + this log | ⬜ Reserved |
| 3 | Dashboard: new correlations, their impact, future state on current trend, suggested setpoints | `discovery.py`, `bw_dev_projected` feature (built), Correlation Explorer panel | 🔄 Feature built |
| 4 | Dashboard: loops/parameters driving stabilization + setpoints to stabilize faster | `stabilization.py`, `settle_min` labels (built) | 🔄 Labels built |
| 5 | Tag every suggestion with source of inference | `config.Source` (built), exact SHAP for local explanations (built), `provenance.py` | 🔄 Tags + SHAP built |
| 6 | Allow accept/reject, record responses to evaluate quality | `ledger.py`, `learning.py` | ⬜ Pending |
| + | Presentation in the provided template (6-slide limit) | Reserved block | ⬜ Reserved |
