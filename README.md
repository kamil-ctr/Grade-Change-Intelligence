# Grade Change Intelligence (GCI)

We built an advisory layer for automatic grade changes on a paper machine.
It watches a transition in progress, predicts whether basis weight is
headed off-spec, explains why, prices a fix in dollars, and remembers
whether the operator agreed with it. That's the whole idea.

Built for the **Honeywell Hackathon, Question 5A**.

> **Advisory only.** GCI never writes to the control system. It sits beside the
> existing QCS / MD control package and gives the operator guidance, which is
> how advanced-control advisory products are actually commissioned.

---

## Problem

Question 5A asks for an intelligence layer on a paper machine's existing
grade-change sequence: predict off-spec risk, explain why, recommend a
fix, price it, and let the operator accept or reject it so the system gets
judged on real usage, not on paper. No public dataset of instrumented grade
changes exists. So the foundation everything else sits on is a
physics-grounded digital twin — mass and energy balance, FOPDT dynamics —
simulating the corpus instead of faking one. See `PROJECT_LOG.md` §6,
decision D1, for the full reasoning.

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

A standalone architecture diagram is reserved for the final packaging pass
(see `PROJECT_LOG.md` §10). Until then, this table is the map we work from.

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

# Run the full test suite (253 tests)
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

Five candidate models, picked on the product objective, not just PR-AUC.
Held-out test set, scored once:

| Metric | Value |
|---|---|
| PR-AUC | 0.827 |
| Event detection rate | **0.827** |
| Median warning time | **4.67 min** |
| False alarms / clean event | **0.099** |

Two of the five never made it that far. We tried LightGBM and XGBoost
first, since gradient boosting is usually the strongest tabular baseline,
but both failed at import with `Library not loaded: @rpath/libomp.dylib` —
this machine's Homebrew sits under the Intel prefix while Python and the
wheels are native arm64, and no arm64 Homebrew was around to supply what
they needed. Not worth a second Homebrew just for two of five models. We
trained the three that would run. Histogram Gradient Boosting won anyway.

Forecast model (quantile regression, +2/+5/+10 min basis-weight deviation),
held-out test set:

| Horizon | MAE (median) | P10–P90 coverage (nominal 80%) |
|---|---|---|
| +2 min | 0.71% | 74.8% |
| +5 min | 0.89% | 73.9% |
| +10 min | 0.90% | 72.7% |

Same root cause: coverage lands at 72–75% against a nominal 80%, since
LightGBM's quantile objective isn't available here either. Stated plainly,
not rounded up. Full numbers: `models/evaluation_report.md`, `PROJECT_LOG.md` §9.

Setpoint optimizer (`gci/optimizer.py`), benchmarked across a 100-event demo
corpus — every transition re-planned exactly as the dashboard's "Recommended
plan" card does, baseline vs. optimizer-recommended plan on the twin:

| Metric | Value |
|---|---|
| Transitions where the recommendation reduces off-spec time | **57.0%** (57/100) |
| Off-spec minutes avoided, mean / median / P95 | 1.79 / 0.25 / 7.84 min |
| Recommended-plan value, mean / median | $64 / $8 |
| Total priced value across the corpus | $6,372 |

The other 43% aren't failures — many logged baselines are already
near-optimal, so zero improvement means the optimizer found nothing better.
Regenerate with `python scripts/benchmark_stabilization.py`; per-event
detail is in `models/benchmark.json` and `models/benchmark_report.md`.

Corpus: 1,500 simulated grade changes, 504,000 feature rows, 62.5% off-spec
rate — deliberately high, by design, to keep the problem learnable (see
`PROJECT_LOG.md` known limitation 5). **253 tests passing, 0 failing**, 25
of which exercise API robustness against malformed input
(`tests/test_robustness.py`).

---

## How each deliverable is met

| # | Honeywell deliverable | Where |
|---|---|---|
| 1 | Develop a solution to the challenge | Full stack: physics twin (`gci/twin.py`, `gci/control.py`) + 9 intelligence/decision modules + FastAPI (`gci/api/`) + React dashboard (`frontend/`) |
| 2 | Document building blocks and module communication | This README's architecture table + `PROJECT_LOG.md` (module-by-module design log); standalone diagram reserved for final packaging |
| 3 | Dashboard: new correlations, their impact, future state on current trend, suggested setpoints | `gci/discovery.py` + `gci/forecast.py` + `gci/optimizer.py`, served at `/api/correlations`, `/api/live`, `/api/recommendations`; rendered in the Correlation Explorer, Basis-Weight Off-Spec Risk, Basis-Weight Forecast, and Recommendations panels |
| 4 | Dashboard: loops/parameters driving stabilization + setpoints to stabilize faster | `gci/stabilization.py`, served at `/api/stabilization`; rendered in the Stabilization impact panel |
| 5 | Tag every suggestion with source of inference | `gci/config.py` (`Source` enum) + exact SHAP (`gci/ml/explain.py`) + `gci/provenance.py`; every advisory card shows its source tag, confidence, and a grounded explanation |
| 6 | Allow accept/reject, record responses to evaluate quality | `gci/ledger.py`, served at `POST /api/recommendations/{id}/feedback` and `GET /api/trust`; rendered in the Trust & ledger panel with live acceptance-rate and confidence-calibration feedback |

### Recommendations vs. insights

Two kinds of dashboard content, deliberately kept apart. Recommendations —
tagged `Physics / recipe rule`, `Recipe constraint`, `Learned model` — are
actionable: a priced plan change with an accept/reject control. Insights,
like the Correlation Explorer and Stabilization impact panels, are
diagnostic instead: what the data or the twin reveals, with no single
action attached. We don't price or gate insights like recommendations —
that would either under-price a real diagnostic or over-claim certainty for
an exploratory finding.

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
rationale, assumptions, and known limitations — the project's source of
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

Quality variables reach steady state through first-order-plus-dead-time
dynamics. That's why grade changes are hard: by the time the scanner sees a
deviation, its cause is already 30–60 seconds in the past.

Verification is by property, not by eyeballing curves. The test suite
asserts mass conservation, response direction, dead-time enforcement,
time-constant magnitude, convergence to the analytic steady state, and
absence of future leakage in the feature set (`test_no_future_leakage`).

---

## Data

The corpus is regenerated, not shipped. It's fully determined by its seed,
so `--seed 20260725` reproduces the exact dataset behind these results.

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
