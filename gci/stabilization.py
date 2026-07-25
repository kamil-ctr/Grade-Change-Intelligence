"""
Loop impact ranking on settling time -- which control-plan parameters most
drive how fast a transition stabilises, and what to change to stabilise
faster (deliverable 4).

Method: local sensitivity analysis on the twin, not a data-mined regression.
For a specific transition (and, optionally, its fault context), each tunable
plan parameter (`ramp_min`, `lead_scale`, `tau_c_scale`) is perturbed up and
down from a baseline plan, and the resulting change in `settle_min` is
measured via the same closed-loop simulation `optimizer.py` uses
(`events.run_event`) -- so "which loop matters" and "what actually happens if
you change it" come from the same physics, not two different approximations
of it. This is deliberately twin-grounded (`Source.PHYSICS_MODEL`) and
complements `discovery.py`'s data-mined correlations rather than duplicating
them.

Loops, not just parameters: `tau_c_scale` is the aggressiveness of every PI
trim loop at once (`control.TRIM_PAIRS`: basis_weight/stock_flow,
moisture/steam_pressure, ash/filler_flow), so ranking its impact *is* ranking
"how hard should the trim loops work" -- the mechanism the problem
statement's "loops" language refers to.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import Source
from .control import ControlPlan
from .events import WINDOW_MIN
from .faults import Fault
from .optimizer import evaluate_plan

# (down, up) perturbation from the baseline value, in the parameter's own
# units -- ramp_min in minutes, the two scales as multiplier deltas.
DEFAULT_PERTURBATIONS: Dict[str, Tuple[float, float]] = {
    "ramp_min": (-2.0, 2.0),
    "lead_scale": (-0.25, 0.25),
    "tau_c_scale": (-0.4, 0.4),
}


def _finite_settle(settle_min: float, window_min: float) -> float:
    """
    A transition that never settles inside the window is not "no data" -- it
    is the worst outcome, worse than any that does settle. Standing in a
    fixed penalty (the full window length) keeps the ranking numeric and
    orders "never settles" strictly after every finite settle time, without
    NaN-aware comparisons scattered through the ranking logic.
    """
    return float(settle_min) if np.isfinite(settle_min) else float(window_min)


@dataclass
class LoopImpact:
    """One tunable parameter's measured effect on settling time."""

    parameter: str
    baseline_value: float
    baseline_settle_min: float
    sensitivity_min_per_unit: float   # d(settle_min)/d(parameter), signed
    best_direction: str                # "increase" | "decrease"
    best_delta: float
    best_settle_min: float
    improvement_min: float             # baseline - best (positive = faster)
    source: str = Source.PHYSICS_MODEL

    def to_dict(self) -> dict:
        return {
            "parameter": self.parameter,
            "baseline_value": round(self.baseline_value, 4),
            "baseline_settle_min": round(self.baseline_settle_min, 3),
            "sensitivity_min_per_unit": round(self.sensitivity_min_per_unit, 4),
            "best_direction": self.best_direction,
            "best_delta": round(self.best_delta, 4),
            "best_settle_min": round(self.best_settle_min, 3),
            "improvement_min": round(self.improvement_min, 3),
            "source": self.source,
        }


def rank_loop_impact(
    from_grade: str,
    to_grade: str,
    baseline_plan: ControlPlan,
    faults: Sequence[Fault] = (),
    seed: int = 0,
    perturbations: Dict[str, Tuple[float, float]] = DEFAULT_PERTURBATIONS,
    window_min: float = WINDOW_MIN,
) -> List[LoopImpact]:
    """
    Rank each tunable plan parameter by how much it moves `settle_min`.

    Baseline and every perturbed candidate share one disturbance/fault
    realisation (`seed`), the same experimental-design choice `optimizer.py`
    makes, so measured sensitivity isolates the parameter's effect rather
    than re-sampled noise. Sorted by descending |sensitivity| -- the loop the
    dashboard should lead with is the one where a small change buys the most
    settling-time improvement.
    """
    baseline_labels = evaluate_plan(
        from_grade, to_grade, baseline_plan, faults, seed=seed, window_min=window_min
    )
    baseline_settle = _finite_settle(baseline_labels["settle_min"], window_min)

    impacts: List[LoopImpact] = []
    for name, (down, up) in perturbations.items():
        base_value = getattr(baseline_plan, name)

        candidates: List[Tuple[float, float]] = []
        for delta in (down, up):
            if delta == 0:
                continue
            candidate_plan = replace(baseline_plan, **{name: base_value + delta})
            labels = evaluate_plan(
                from_grade, to_grade, candidate_plan, faults, seed=seed,
                window_min=window_min,
            )
            settle = _finite_settle(labels["settle_min"], window_min)
            candidates.append((delta, settle))

        if not candidates:
            continue

        deltas = np.array([d for d, _ in candidates])
        settles = np.array([s for _, s in candidates])
        slopes = (settles - baseline_settle) / deltas
        steep_idx = int(np.argmax(np.abs(slopes)))

        # "Best" is chosen against the baseline too (delta 0), not just
        # between the two perturbed points -- otherwise a parameter where
        # *both* probed directions make things worse would still report one
        # of them as "best", recommending a change that helps nothing.
        with_baseline = candidates + [(0.0, baseline_settle)]
        best_delta, best_settle = min(with_baseline, key=lambda pair: pair[1])

        impacts.append(
            LoopImpact(
                parameter=name,
                baseline_value=float(base_value),
                baseline_settle_min=baseline_settle,
                sensitivity_min_per_unit=float(slopes[steep_idx]),
                best_direction=(
                    "increase" if best_delta > 0
                    else "decrease" if best_delta < 0
                    else "none"
                ),
                best_delta=float(best_delta),
                best_settle_min=best_settle,
                improvement_min=baseline_settle - best_settle,
            )
        )

    impacts.sort(key=lambda imp: -abs(imp.sensitivity_min_per_unit))
    return impacts
