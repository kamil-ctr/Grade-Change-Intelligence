# CLAUDE.md — Grade Change Intelligence (GCI)

Project memory for Claude Code. Read `PROJECT_LOG.md` for full status and
`CHANGELOG.md` for history — those two are the source of truth and **must be
kept updated after every module**.

## What this is

An advisory intelligence layer for automatic grade changes on a paper machine,
built for the **Honeywell Hackathon, Question 5A**. It predicts when basis
weight will go off-spec (>2.5% from setpoint) during a grade transition,
recommends setpoint changes, prices each recommendation in dollars, and learns
from operator accept/reject.

**Submission deadline: 2026-07-26 23:59.** Six graded deliverables — see
§11 of `PROJECT_LOG.md` for the coverage matrix. Every feature must trace to at
least one.

**Advisory only.** The system never writes to a control system. Keep it that way.

## Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Regenerate the corpus (~60s for 500 events; fully seed-determined)
python scripts/generate_data.py --events 500

# Train / compare / select / explain the risk model
python scripts/train_risk_model.py

# Tests — MUST stay green
python -m unittest discover -s tests -t .
```

## Architecture

Seven layers, defined in the original design and **not to be redesigned**
without a critical blocker:

```
Operator Experience   cockpit · AI Copilot · What-If Studio
Decision & Value      ROI engine · advisory ledger · provenance & trust
Intelligence Engine   risk predictor · correlation discovery · setpoint optimizer
Digital Twin Core     machine dynamics · recipe library · event historian
Connectivity          one documented DataSource interface (simulated impl)
Learning & Governance feedback capture · trust scores · audit trail
```

Current code:

| Module | Role |
|---|---|
| `gci/config.py` | Spec thresholds, `Economics`, `Source` provenance tags, `AdvisoryPolicy` |
| `gci/grades.py` | 7 grades, 19-tag dictionary, **`KNOWN_LOOPS`** (novelty baseline) |
| `gci/twin.py` | Mass/energy-balance twin + FOPDT + `TwinStepper` |
| `gci/control.py` | S-curve trajectory, lead compensation, SIMC PI trim, feasibility floor |
| `gci/faults.py` | 11 named faults over Ornstein–Uhlenbeck drift |
| `gci/events.py` | Closed-loop events, labelling, validation |
| `gci/features.py` | 104 backward-looking features, caching |
| `gci/ml/` | Splits, metrics, registry, tiered SHAP, pipeline |

## Non-negotiable rules

1. **Never split rows randomly.** Splits are event-wise only
   (`gci/ml/splits.py`). Samples 5 s apart are near-duplicates.
2. **No future leakage.** Features are strictly backward-looking. There is a
   test that corrupts the future and asserts past features are bit-identical
   (`tests/test_features.py::test_no_future_leakage`). Do not weaken it.
3. **Test split is scored exactly once**, after the winner is fixed. Never tune
   against it.
4. **Every recommendation carries a provenance tag** from `config.Source` and a
   confidence value. No untagged advice reaches the UI.
5. **Event detection rate is the headline metric, not row-level recall.** A
   transition is ~336 samples; an excursion only has to be caught once.
6. **No placeholder logic** where a real implementation is feasible.
7. **Keep the project runnable at every commit.** Tests green before moving on.
8. **Update `PROJECT_LOG.md` and `CHANGELOG.md` after each module.**

## Conventions

- `unittest`, not `pytest` — zero extra dependencies, judges can run it clean.
- Docstrings explain *why*, especially where a choice is non-obvious or where a
  naive approach was wrong. Several existing docstrings record real bugs; keep
  that habit.
- Type hints on public functions. `from __future__ import annotations` at top.
- Optional dependencies are probed, never assumed (`gci/ml/registry.py`).
- Determinism: everything seeded and reproducible.

## Current state

- **Phase 0 complete**: twin, controller, faults, events, features.
- **Phase 1 module 1 complete**: ML pipeline + risk model.
  LightGBM selected — test PR-AUC 0.821, **event detection 0.847, median warning
  4.50 min**, false alarms 0.179 per clean event.
- **115 tests passing.**

### Next task

`gci/forecast.py` — quantile trajectory forecasting for the dashboard forecast
cone. Predict basis-weight deviation at +2/+5/+10 min with prediction intervals.
Reuse `gci/ml/splits.py` and the same leakage controls. Then, in order:
`roi.py` → `optimizer.py` → `discovery.py` → `stabilization.py` →
`provenance.py` → `ledger.py` → `api/` → `frontend/`.

## Environment note — important

The earlier development sandbox had a **45-second limit per shell call** and
~3.9 GB RAM. Several things were trimmed purely to fit it, and **you should
undo them now that you are on real hardware**:

| What was trimmed | Restore to |
|---|---|
| Random Forest: 150 trees, depth 10 | 300 trees, depth 14, `min_samples_leaf=20` |
| SHAP sample: 600 rows | 3000+ rows |
| Permutation importance: 1500 rows, 2 repeats | 6000 rows, 5–10 repeats |
| Corpus: 500 events in two chunks | Generate 1000–2000 in one call; per-event metrics are noisy below ~1000 |
| No hyperparameter search | Consider a small sweep if time allows |

Per-model checkpointing and dataset fingerprinting (`gci/ml/pipeline.py`) were
also added because of that limit — **keep those**, they are good practice.

## Known traps

- **`xgboost-cpu`, not `xgboost`** — the default wheel bundles 131 MB of CUDA.
- **`shap` needs numba/llvmlite.** If it fails, the tiered explainer in
  `gci/ml/explain.py` still gives exact TreeSHAP via LightGBM/XGBoost native
  `pred_contrib`, and analytic SHAP for linear models. Never let it silently
  fall back to the ridge surrogate without labelling it approximate.
- **Off-spec rate is 62.6%** by design (hard transitions, rushed ramps) to make
  the problem learnable. Report *relative* improvement in benchmarks, not
  absolute rates.
- **`data/` is regenerable and must not be committed** or shipped in the
  submission zip (10 MB limit). Stale files exist there: `chunk_*.pkl`,
  `features.csv.gz`.
- **WFU-80 saturates the dryer section** (needs ~1100 kPa vs a 1092 kPa
  ceiling). This is emergent physics, not a bug — surface it as a discovered
  insight.

## Deliverable reminders

- The presentation template (`IDEA_Presentation_Format.pptx`) has a **hard
  six-slide limit** including the title slide, and asks for
  points/diagrams/infographics rather than paragraphs.
- **Slide 5 is "Artifacts" — dashboard screenshots are directly graded.** The UI
  must look good in a static screenshot, not just in motion.
- Two of the six deliverables are *dashboard content* requirements. Prefer six
  simple panels that each map to a bullet over three beautiful ones.
- Reserve the last ~2 hours for the architecture doc, the deck, screenshots,
  packaging under 10 MB, and one clean-checkout rehearsal.
