# CHANGELOG

Kept synchronised with the codebase. Newest first.

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
