# Grade Change Intelligence (GCI)

An advisory intelligence layer for automatic grade changes on a paper machine.
GCI predicts when basis weight is at risk of going off-spec during a grade
transition, recommends setpoint changes to prevent it, prices every
recommendation in dollars, and learns from whether the operator accepts it.

Built for the Honeywell Hackathon, Question 5A.

> **Advisory only.** GCI never writes to the control system. It sits beside the
> existing QCS / MD control package and gives the operator guidance, which is
> how advanced-control advisory products are actually commissioned.

---

## Quick start

```bash
pip install -r requirements.txt

# Generate the labelled grade-change corpus (~25 s, fully seed-determined)
python scripts/generate_data.py --events 300

# Run the test suite
python -m unittest discover -s tests -t .
```

---

## What is here (Phase 0 complete)

| Module | Purpose |
|---|---|
| `gci/config.py` | Spec thresholds, economics, provenance tags, advisory policy |
| `gci/grades.py` | 7-grade recipe library, 19-tag dictionary, known control-loop graph |
| `gci/twin.py` | Physics-based digital twin (mass + energy balance, FOPDT dynamics) |
| `gci/control.py` | Coordinated grade-change controller: S-curve, lead compensation, SIMC PI trim |
| `gci/faults.py` | 11 named fault types plus correlated disturbance drift |
| `gci/events.py` | Closed-loop event simulation, labelling, dataset validation |
| `gci/features.py` | 104 leak-free features, forward-looking breach labels |

See **`PROJECT_LOG.md`** for the full status, design decisions, assumptions and
limitations. It is the project's source of truth.

---

## Why the physics is trustworthy

The twin is first-principles where it matters:

```
basis weight = retained dry mass rate / sheet area rate    (mass balance)
ash          = retained filler / total retained solids     (mass balance)
moisture     = water load / drying capacity                (energy balance)
caliper      = basis weight / apparent density             (empirical)
```

Quality variables then reach steady state through first-order-plus-dead-time
dynamics, because the transport delay between an actuator and the QCS scanner
is the fundamental reason grade changes are hard: by the time the scanner sees
a deviation, its cause is already 30–60 seconds in the past.

Verification is by property, not by eyeballing curves — 78 tests assert mass
conservation, response direction, dead-time enforcement, time-constant
magnitude, convergence to the analytic steady state, and absence of future
leakage in the feature set.

---

## Data

The corpus is **regenerated, not shipped**. It is fully determined by its seed,
so `--seed 20260725` reproduces the exact dataset behind the reported results
while keeping the repository small.

| | |
|---|---|
| Events | 300 grade changes, 42 distinct grade pairs |
| Window | 30 min at 5 s sampling |
| Off-spec rate | 61.7% |
| Feature samples | 100,800 × 104 |
| Generation time | ~25 s |
