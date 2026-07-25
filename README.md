# Grade Change Intelligence (GCI)

An advisory intelligence layer for automatic grade changes on a paper machine.
GCI predicts when basis weight is at risk of going off-spec during a grade
transition, recommends setpoint changes to prevent it, prices every
recommendation in dollars, and learns from whether the operator accepts it.

Built for the **Honeywell Hackathon, Question 5A**.

> **Advisory only.** GCI never writes to the control system. It sits beside the
> existing QCS / MD control package and gives the operator guidance, which is
> how advanced-control advisory products are actually commissioned.

---

## Problem

Question 5A asks for an intelligence layer on top of a paper machine's
existing automatic grade-change sequence: predict when a transition is going
to run off-spec, explain why, recommend what to change, price the
recommendation, and let the operator accept or reject it so the system can be
evaluated on real usage rather than on paper. There is no public dataset of
instrumented grade changes to train against, so the first design decision was
to build a physics-grounded digital twin of the machine (mass and energy
balance, FOPDT dynamics) and simulate the corpus rather than fabricate one —
see `PROJECT_LOG.md` §6, decision D1, for the full reasoning.

---

## Architecture

Seven layers, top to bottom:

| Layer | Contents |
|---|---|
| Operator Experience | cockpit dashboard (`frontend/`) · AI Copilot (Phase 2) · What-If Studio (Phase 2) |
| Decision & Value | ROI engine (`gci/roi.py`) · advisory ledger (`gci/ledger.py`) · provenance & trust (`gci/provenance.py`) |
| Intelligence Engine | risk predictor (`gci/ml/`) · forecast cone (`gci/forecast.py`) · correlation discovery (`gci/discovery.py`) · setpoint optimizer (`gci/optimizer.py`) · stabilization ranking (`gci/stabilization.py`) |
| Digital Twin Core | machine dynamics (`gci/twin.py`, `gci/control.py`) · recipe library (`gci/grades.py`) · event historian (`gci/events.py`) |
| Connectivity | one documented `DataSource` interface (`gci/api/datasource.py`), simulated implementation |
| Learning & Governance | feedback capture, trust scores, audit trail (`gci/ledger.py`) |
| API | FastAPI service (`gci/api/`), 12 endpoints wiring every layer above together |

A standalone architecture diagram/doc is reserved for the final packaging
pass (see `PROJECT_LOG.md` §10); until then this table plus the module list
below is the authoritative map.

---

## Quick start

```bash
# Backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate the labelled grade-change corpus (~70 s at 1500 events, fully seed-determined)
python scripts/generate_data.py --events 1500

# Train the risk model and the forecast model
python scripts/train_risk_model.py
python scripts/train_forecast_model.py

# Run the full test suite (226 tests)
python -m unittest discover -s tests -t .

# Serve the API
uvicorn gci.api.app:app --reload
```

```bash
# Frontend (separate terminal)
cd frontend
npm install
npm run dev   # proxies /api to localhost:8000 — see vite.config.js
```

---

## Results

Risk model (Histogram Gradient Boosting — selected on the product objective;
see `PROJECT_LOG.md` §1a for why LightGBM/XGBoost aren't used on this
machine), held-out test set, scored once:

| Metric | Value |
|---|---|
| PR-AUC | 0.827 |
| Event detection rate | **0.827** |
| Median warning time | **4.67 min** |
| False alarms / clean event | **0.099** |

Forecast model (quantile regression, +2/+5/+10 min basis-weight deviation),
held-out test set:

| Horizon | MAE (median) | P10–P90 coverage (nominal 80%) |
|---|---|---|
| +2 min | 0.71% | 74.8% |
| +5 min | 0.89% | 73.9% |
| +10 min | 0.90% | 72.7% |

Coverage running under nominal is a known, disclosed limitation of the
fallback estimator (LightGBM's native quantile objective is unavailable on
this machine) — stated here rather than rounded up. Full numbers, per-model
comparisons and the reasoning behind every modelling choice are in
`models/evaluation_report.md` and `PROJECT_LOG.md` §9.

Corpus: 1,500 simulated grade changes, 504,000 feature rows, 62.5% off-spec
rate (deliberately high — hard transitions and rushed ramps, by design, to
keep the prediction problem learnable; see `PROJECT_LOG.md` known limitation
5). **226 tests passing, 0 failing.**

---

## How each deliverable is met

| # | Honeywell deliverable | Where |
|---|---|---|
| 1 | Develop a solution to the challenge | Full stack: physics twin (`gci/twin.py`, `gci/control.py`) + 9 intelligence/decision modules + FastAPI (`gci/api/`) + React dashboard (`frontend/`) |
| 2 | Document building blocks and module communication | This README's architecture table + `PROJECT_LOG.md` (module-by-module design log); standalone diagram reserved for final packaging |
| 3 | Dashboard: new correlations, their impact, future state on current trend, suggested setpoints | `gci/discovery.py` + `gci/forecast.py` + `gci/optimizer.py`, served at `/api/correlations`, `/api/live`, `/api/recommendations`; rendered in the Correlation Explorer, Live risk & forecast cone, and Recommendations panels |
| 4 | Dashboard: loops/parameters driving stabilization + setpoints to stabilize faster | `gci/stabilization.py`, served at `/api/stabilization`; rendered in the Stabilization impact panel |
| 5 | Tag every suggestion with source of inference | `gci/config.py` (`Source` enum) + exact SHAP (`gci/ml/explain.py`) + `gci/provenance.py`; every advisory card shows its source tag, confidence, and a grounded explanation |
| 6 | Allow accept/reject, record responses to evaluate quality | `gci/ledger.py`, served at `POST /api/recommendations/{id}/feedback` and `GET /api/trust`; rendered in the Trust & ledger panel with live acceptance-rate and confidence-calibration feedback |

---

## What is here

| Module | Purpose |
|---|---|
| `gci/config.py` | Spec thresholds, economics, provenance tags, advisory policy |
| `gci/grades.py` | 7-grade recipe library, 19-tag dictionary, known control-loop graph |
| `gci/twin.py` | Physics-based digital twin (mass + energy balance, FOPDT dynamics) |
| `gci/control.py` | Coordinated grade-change controller: S-curve, lead compensation, SIMC PI trim |
| `gci/faults.py` | 11 named fault types plus correlated disturbance drift |
| `gci/events.py` | Closed-loop event simulation, labelling, dataset validation |
| `gci/features.py` | 104 leak-free features, forward-looking breach labels |
| `gci/ml/` | Model registry, splits, metrics, tiered SHAP, training pipeline |
| `gci/forecast.py` | Quantile trajectory forecasting (+2/+5/+10 min basis-weight deviation) |
| `gci/roi.py` | Confidence-weighted dollar pricing + advisory surfacing policy |
| `gci/optimizer.py` | Coordinate-descent ramp/setpoint search over the twin |
| `gci/discovery.py` | Lagged correlation + mutual information + novelty scoring |
| `gci/stabilization.py` | Twin-based sensitivity ranking of plan parameters on settling time |
| `gci/provenance.py` | Unified advisory schema, grounded explanations, policy gate |
| `gci/ledger.py` | Accept/reject audit trail and quality evaluation |
| `gci/api/` | FastAPI service: connectivity layer, business logic, HTTP wiring |
| `frontend/` | React dashboard — 6 panels, one per deliverable |

See **`PROJECT_LOG.md`** for the full status, every design decision and its
rationale, assumptions, and known limitations. It is the project's source of
truth, updated after every module.

---

## Why the physics is trustworthy

The twin is first-principles where it matters:

```
basis weight = retained dry mass rate / sheet area rate    (mass balance)
ash          = retained filler / total retained solids     (mass balance)
moisture     = water load / drying capacity                (energy balance)
caliper      = basis weight / apparent density              (empirical)
```

Quality variables then reach steady state through first-order-plus-dead-time
dynamics, because the transport delay between an actuator and the QCS scanner
is the fundamental reason grade changes are hard: by the time the scanner sees
a deviation, its cause is already 30–60 seconds in the past.

Verification is by property, not by eyeballing curves — the test suite
asserts mass conservation, response direction, dead-time enforcement,
time-constant magnitude, convergence to the analytic steady state, and
absence of future leakage in the feature set (`test_no_future_leakage`).

---

## Data

The corpus is **regenerated, not shipped**. It is fully determined by its
seed, so `--seed 20260725` reproduces the exact dataset behind the reported
results while keeping the repository small.

| | |
|---|---|
| Events | 1,500 grade changes, 42 distinct grade pairs |
| Window | 30 min at 5 s sampling |
| Off-spec rate | 62.5% |
| Feature samples | 504,000 × 104 |
| Generation time | ~72 s |

---

## Team

Mohammad Kamil — solo entry.

## License

No license has been applied to this repository; all rights reserved by
default. Contact the author for reuse.
