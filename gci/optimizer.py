"""
Bounded setpoint/ramp search over the twin -- the Intelligence Engine's
setpoint optimizer.

Given a specific grade change (and, optionally, the fault context already
known for it), searches the `ControlPlan` space GCI is allowed to
recommend -- `ramp_min`, `lead_scale`, `tau_c_scale` -- for the plan that
minimises predicted off-spec severity, subject to the recipe's own bounds and
the transition's physical feasibility floor. Every candidate is evaluated by
running the same closed-loop simulation `events.run_event` uses to build the
training corpus, so "what the optimizer recommends" and "what the model was
trained on" are the same physics, not two different approximations of it.

Search, not gradient descent: the objective (off-spec minutes from a
closed-loop nonlinear simulation with a PI trim loop) is neither smooth nor
cheap to differentiate. Coordinate descent -- sweep one knob at a time,
holding the others fixed, then narrow and repeat -- is used instead of a
full 3-D grid: a `n_per_dim^3` grid is the accurate search, but the three
knobs govern close-to-independent effects (ramp speed, trajectory lead,
trim aggressiveness), so coordinate descent finds a comparable optimum for
`3 x n_per_dim` evaluations per round instead of `n_per_dim^3`. Results are
memoised per (transition, fault signature, seed) so a demo replay of the
same scenario is instant after the first search.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from .config import DEFAULT_ECONOMICS, Economics, Source
from .control import ControlPlan, min_feasible_ramp_min, recipe_slew_limits
from .events import WINDOW_MIN, run_event
from .faults import Fault
from .grades import get_grade
from .roi import RecommendationValue, price_plan_comparison
from .twin import PaperMachineTwin

# Trust in the twin's fidelity for a physics-model-sourced recommendation.
# Not a statistical confidence (the twin is calibrated to published ranges,
# not a specific mill -- see PROJECT_LOG known limitations); a fixed,
# documented value rather than a fabricated precision.
PHYSICS_MODEL_CONFIDENCE: float = 0.75


@dataclass
class OptimizationBounds:
    """Search bounds for each tunable plan parameter."""

    ramp_min: Tuple[float, float] = (2.0, 25.0)
    lead_scale: Tuple[float, float] = (0.5, 1.6)
    tau_c_scale: Tuple[float, float] = (0.5, 2.2)


@dataclass
class OptimizationResult:
    """The best plan found and its predicted improvement over the baseline."""

    plan: ControlPlan
    baseline_plan: ControlPlan
    baseline_labels: Dict[str, float]
    recommended_labels: Dict[str, float]
    improvement: Dict[str, float]
    objective: str
    n_evaluations: int
    min_ramp_min: float
    binding_actuator: str
    to_grade: str
    source: str = Source.PHYSICS_MODEL

    def to_dict(self) -> dict:
        return {
            "plan": self.plan.to_dict(),
            "baseline_plan": self.baseline_plan.to_dict(),
            "baseline_labels": self.baseline_labels,
            "recommended_labels": self.recommended_labels,
            "improvement": self.improvement,
            "objective": self.objective,
            "n_evaluations": self.n_evaluations,
            "min_ramp_min": self.min_ramp_min,
            "binding_actuator": self.binding_actuator,
            "source": self.source,
        }

    def price(
        self, economics: Economics = DEFAULT_ECONOMICS,
        confidence: float = PHYSICS_MODEL_CONFIDENCE,
    ) -> RecommendationValue:
        """Dollar-price the improvement via the ROI engine."""
        return price_plan_comparison(
            baseline_off_spec_minutes=self.baseline_labels.get("off_spec_minutes", 0.0),
            improved_off_spec_minutes=self.recommended_labels.get("off_spec_minutes", 0.0),
            grade_to=self.to_grade,
            confidence=confidence,
            source=self.source,
            economics=economics,
        )


def evaluate_plan(
    from_grade: str,
    to_grade: str,
    plan: ControlPlan,
    faults: Sequence[Fault] = (),
    seed: int = 0,
    window_min: float = WINDOW_MIN,
) -> Dict[str, float]:
    """Run one closed-loop simulation of a candidate plan and return its labels."""
    twin = PaperMachineTwin(seed=seed)
    ev = run_event(
        twin, from_grade, to_grade, plan, faults,
        window_min=window_min, event_id=0, seed=seed,
    )
    return ev.labels


def _linspace(lo: float, hi: float, n: int) -> Tuple[float, ...]:
    if hi <= lo:
        return (lo,)
    return tuple(float(x) for x in np.linspace(lo, hi, n))


def _coordinate_descent(
    evaluate: Callable[[ControlPlan], float],
    start: ControlPlan,
    bounds: OptimizationBounds,
    n_per_dim: int = 7,
    n_rounds: int = 2,
) -> Tuple[ControlPlan, float, int]:
    """Sweep each knob in turn, narrowing bounds around the best point found
    each round. See module docstring for why coordinate descent, not a grid."""
    best_plan = start
    best_score = evaluate(start)
    n_eval = 1

    cur_bounds: Dict[str, Tuple[float, float]] = {
        "ramp_min": bounds.ramp_min,
        "lead_scale": bounds.lead_scale,
        "tau_c_scale": bounds.tau_c_scale,
    }

    for _ in range(max(n_rounds, 1)):
        for name, (lo, hi) in cur_bounds.items():
            for value in _linspace(lo, hi, n_per_dim):
                candidate = replace(best_plan, **{name: value})
                score = evaluate(candidate)
                n_eval += 1
                if score < best_score:
                    best_score, best_plan = score, candidate

        shrink = 0.45
        cur_bounds = {
            name: (
                max(lo, getattr(best_plan, name) - (hi - lo) * shrink / 2),
                min(hi, getattr(best_plan, name) + (hi - lo) * shrink / 2),
            )
            for name, (lo, hi) in cur_bounds.items()
        }

    return best_plan, best_score, n_eval


_CACHE: Dict[tuple, OptimizationResult] = {}


def _fault_signature(faults: Sequence[Fault]) -> tuple:
    return tuple(sorted(f.code for f in faults))


def recommend_plan(
    from_grade: str,
    to_grade: str,
    baseline_plan: ControlPlan,
    faults: Sequence[Fault] = (),
    seed: int = 0,
    bounds: OptimizationBounds = OptimizationBounds(),
    objective: str = "off_spec_minutes",
    n_per_dim: int = 7,
    n_rounds: int = 2,
    window_min: float = WINDOW_MIN,
    use_cache: bool = True,
) -> OptimizationResult:
    """
    Search for the plan minimising `objective` (a key into `EventResult.labels`,
    e.g. `"off_spec_minutes"` or `"max_abs_dev_pct"`) for this specific
    transition and fault context, holding the disturbance/fault realisation
    fixed across candidates (same `seed`) so the comparison isolates the
    effect of the plan rather than re-sampling noise.

    The ramp-time search floor is raised to the transition's own physical
    feasibility floor: recommending a ramp the recipe's actuators cannot
    physically deliver would defeat the point (see `control.min_feasible_ramp_min`).
    """
    cache_key = (
        from_grade, to_grade, tuple(baseline_plan.to_dict().items()),
        _fault_signature(faults), seed, objective, n_per_dim, n_rounds, window_min,
    )
    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key]

    g_from, g_to = get_grade(from_grade), get_grade(to_grade)
    probe = PaperMachineTwin(seed=seed)
    floor_min, binding_actuator = min_feasible_ramp_min(
        probe.inverse_solve(g_from), probe.inverse_solve(g_to),
        recipe_slew_limits(g_to),
    )
    ramp_lo = max(bounds.ramp_min[0], floor_min)
    ramp_hi = max(bounds.ramp_min[1], ramp_lo + 1e-6)
    search_bounds = replace(bounds, ramp_min=(ramp_lo, ramp_hi))

    baseline_labels = evaluate_plan(
        from_grade, to_grade, baseline_plan, faults, seed=seed, window_min=window_min
    )

    def score(plan: ControlPlan) -> float:
        labels = evaluate_plan(
            from_grade, to_grade, plan.clipped(g_to), faults, seed=seed,
            window_min=window_min,
        )
        return labels[objective]

    start = baseline_plan.clipped(g_to)
    if start.ramp_min < ramp_lo:
        start = replace(start, ramp_min=ramp_lo)

    best_plan, _, n_eval = _coordinate_descent(
        score, start, search_bounds, n_per_dim=n_per_dim, n_rounds=n_rounds,
    )
    best_plan = best_plan.clipped(g_to)
    recommended_labels = evaluate_plan(
        from_grade, to_grade, best_plan, faults, seed=seed, window_min=window_min
    )

    improvement = {
        k: baseline_labels[k] - recommended_labels[k]
        for k in ("off_spec_minutes", "max_abs_dev_pct", "mean_abs_dev_pct")
        if k in baseline_labels and k in recommended_labels
    }

    result = OptimizationResult(
        plan=best_plan,
        baseline_plan=baseline_plan,
        baseline_labels=baseline_labels,
        recommended_labels=recommended_labels,
        improvement=improvement,
        objective=objective,
        n_evaluations=n_eval,
        min_ramp_min=floor_min,
        binding_actuator=binding_actuator,
        to_grade=to_grade,
    )

    if use_cache:
        _CACHE[cache_key] = result
    return result


def clear_cache() -> None:
    _CACHE.clear()
