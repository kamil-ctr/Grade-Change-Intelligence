# Stabilization / setpoint optimizer benchmark

Aggregate result of running `recommend_plan` against every transition in a
100-event demo corpus (seed 20260726), the same corpus shape
the API serves. Baseline is each event's own logged plan; the optimizer
re-plans it exactly as the dashboard's "Recommended plan" card does.

- **Transitions evaluated:** 100 / 100 (0 skipped on error)
- **Success rate (improvement > 0):** 57.0% (57/100)
- **Off-spec minutes avoided, mean:** 1.79 min
- **Off-spec minutes avoided, median:** 0.25 min
- **Off-spec minutes avoided, P95:** 7.84 min
- **Recommended-plan value, mean:** $64
- **Recommended-plan value, median:** $8
- **Total priced value across corpus:** $6,372
- **Provenance split:** 80 physics-model, 20 recipe-limit
- **Runtime:** 219.6s

Regenerate with `python scripts/benchmark_stabilization.py`. Per-event detail
in `models/benchmark.json`.
