"""
Grade-change event generation and labelling.

An "event" is one complete auto grade change: a settling period on the old
grade, the coordinated transition, and the stabilisation period on the new
grade. Each event is simulated closed-loop through the twin with its own
disturbance realisation and fault set, then labelled with the outcomes the
problem statement cares about:

  * off-spec    basis weight deviating more than 2.5% from its setpoint
  * settle time how long until the sheet is stable on the new grade
  * cause       which injected fault (if any) was responsible

This module is the single source of "historical data" for every downstream
engine, which is why labelling lives here rather than being recomputed
inconsistently in three places.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config as C
from .control import (
    ControlPlan,
    GradeChangeController,
    min_feasible_ramp_min,
    recipe_slew_limits,
)
from .faults import (
    Fault,
    apply_faults,
    baseline_disturbances,
    sample_faults,
)
from .grades import (
    CV_TAGS,
    DV_TAGS,
    GRADE_CODES,
    MV_TAGS,
    Grade,
    get_grade,
    transition_magnitude,
)
from .twin import PaperMachineTwin

WINDOW_MIN: float = 30.0
RAMP_START_MIN: float = 5.0

# Series stored per event (order matters -- it defines the array layout)
SERIES_TAGS: Tuple[str, ...] = (
    MV_TAGS
    + CV_TAGS
    + DV_TAGS
    + tuple(f"{cv}_sp" for cv in ("basis_weight", "moisture", "ash"))
)


@dataclass
class EventResult:
    """One simulated grade change: trajectories, metadata and labels."""

    event_id: int
    from_grade: str
    to_grade: str
    plan: Dict[str, float]
    faults: List[dict]
    series: Dict[str, np.ndarray]
    labels: Dict[str, float]
    seed: int
    context: Dict[str, object] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return int(self.series["basis_weight"].shape[0])

    @property
    def t_min(self) -> np.ndarray:
        return np.arange(self.n_steps, dtype=float) * C.DT_S / 60.0

    def to_meta(self) -> dict:
        meta = {
            "event_id": self.event_id,
            "from_grade": self.from_grade,
            "to_grade": self.to_grade,
            "seed": self.seed,
            "n_faults": len(self.faults),
            "fault_codes": "|".join(f["code"] for f in self.faults),
            "primary_cause": self.faults[0]["code"] if self.faults else "NONE",
        }
        meta.update({f"plan_{k}": v for k, v in self.plan.items()})
        meta.update(self.context)
        meta.update(self.labels)
        return meta


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------
def bw_deviation_pct(bw: np.ndarray, bw_sp: np.ndarray) -> np.ndarray:
    """Signed basis weight deviation as a percentage of the active setpoint."""
    return (bw - bw_sp) / np.maximum(np.abs(bw_sp), 1e-9) * 100.0


def compute_settle_time(
    bw: np.ndarray, final_sp: float, ramp_start_min: float
) -> float:
    """
    Minutes from ramp start until basis weight stays within SETTLE_TOL_PCT of
    the final target for SETTLE_DWELL_MIN continuously.

    Returns NaN if the sheet never stabilises inside the window -- which is
    itself a meaningful outcome and is handled explicitly downstream.
    """
    tol = final_sp * C.SETTLE_TOL_PCT / 100.0
    inside = np.abs(bw - final_sp) <= tol
    dwell_steps = int(round(C.SETTLE_DWELL_MIN * 60.0 / C.DT_S))
    start_idx = int(round(ramp_start_min * 60.0 / C.DT_S))

    run = 0
    for k in range(start_idx, inside.shape[0]):
        run = run + 1 if inside[k] else 0
        if run >= dwell_steps:
            settle_idx = k - dwell_steps + 1
            return float(settle_idx * C.DT_S / 60.0 - ramp_start_min)
    return float("nan")


def label_event(
    series: Dict[str, np.ndarray], grade_to: Grade, ramp_start_min: float
) -> Dict[str, float]:
    """Derive every outcome label from the simulated trajectories."""
    bw = series["basis_weight"]
    bw_sp = series["basis_weight_sp"]
    dev = bw_deviation_pct(bw, bw_sp)
    abs_dev = np.abs(dev)

    off_spec_mask = abs_dev > C.BW_SPEC_PCT
    off_spec_steps = int(off_spec_mask.sum())
    off_spec_min = off_spec_steps * C.DT_S / 60.0

    if off_spec_steps > 0:
        first_idx = int(np.argmax(off_spec_mask))
        first_breach_min = float(first_idx * C.DT_S / 60.0)
    else:
        first_breach_min = float("nan")

    settle_min = compute_settle_time(bw, grade_to.basis_weight, ramp_start_min)

    moisture_dev = np.abs(series["moisture"] - series["moisture_sp"])
    ash_dev = np.abs(series["ash"] - series["ash_sp"])

    return {
        "off_spec": float(off_spec_steps > 0),
        "off_spec_minutes": off_spec_min,
        "max_abs_dev_pct": float(abs_dev.max()),
        "mean_abs_dev_pct": float(abs_dev.mean()),
        "first_breach_min": first_breach_min,
        "settle_min": settle_min,
        "settled": float(not np.isnan(settle_min)),
        "moisture_off_spec_min": float(
            (moisture_dev > C.MOISTURE_SPEC_ABS).sum() * C.DT_S / 60.0
        ),
        "ash_off_spec_min": float(
            (ash_dev > C.ASH_SPEC_ABS).sum() * C.DT_S / 60.0
        ),
        "final_bw_error": float(bw[-1] - grade_to.basis_weight),
    }


# ---------------------------------------------------------------------------
# Simulation of a single event
# ---------------------------------------------------------------------------
def run_event(
    twin: PaperMachineTwin,
    from_grade: str,
    to_grade: str,
    plan: ControlPlan,
    faults: Sequence[Fault] = (),
    window_min: float = WINDOW_MIN,
    event_id: int = 0,
    seed: int = 0,
    add_noise: bool = True,
) -> EventResult:
    """Simulate one closed-loop grade change end to end."""
    rng = twin.rng
    n = int(round(window_min * 60.0 / C.DT_S))

    g_from, g_to = get_grade(from_grade), get_grade(to_grade)

    dv_base = baseline_disturbances(rng, n)
    dv, mv_overrides = apply_faults(dv_base, faults)
    dv0 = {k: float(v[0]) for k, v in dv.items()}

    # The machine starts *on* the old grade, not merely at its nominal recipe
    # values: before a grade change the existing controls have already trimmed
    # out whatever disturbance offset is present. Solving the inverse model at
    # the prevailing disturbance state reproduces that, so an event does not
    # begin spuriously off-spec.
    mv_from = twin.inverse_solve(g_from, dv0)
    mv_to = twin.inverse_solve(g_to, dv0)

    controller = GradeChangeController(twin, g_from, g_to, mv_from, mv_to, plan)

    cv0 = twin.steady_state_at(mv_from, dv0)
    stepper = twin.make_stepper(mv_from, cv0)

    series: Dict[str, np.ndarray] = {
        tag: np.empty(n, dtype=float) for tag in SERIES_TAGS
    }

    measured = dict(cv0)
    for k in range(n):
        t_s = k * C.DT_S
        cmd = controller.command(t_s, measured)

        # Actuator-targeted faults (e.g. dosing pump loss) corrupt the command
        # that actually reaches the process, not the controller's intent.
        for tag, profile in mv_overrides.items():
            cmd[tag] = cmd[tag] + float(profile[k])

        dv_k = {tag: float(dv[tag][k]) for tag in dv}
        measured, mv_actual = stepper.step(
            cmd, dv_k, controller.slew_limits, add_noise=add_noise
        )

        sp = controller.setpoints_at(t_s)
        for tag in MV_TAGS:
            series[tag][k] = mv_actual[tag]
        for tag in CV_TAGS:
            series[tag][k] = measured[tag]
        for tag in DV_TAGS:
            series[tag][k] = dv_k[tag]
        for tag in ("basis_weight", "moisture", "ash"):
            series[f"{tag}_sp"][k] = sp[tag]

    labels = label_event(series, g_to, plan.start_min)

    return EventResult(
        event_id=event_id,
        from_grade=from_grade,
        to_grade=to_grade,
        plan=controller.plan.to_dict(),
        faults=[f.to_dict() for f in faults],
        series=series,
        labels=labels,
        seed=seed,
        context={
            "min_ramp_min": controller.min_ramp_min,
            "binding_actuator": controller.binding_actuator,
            "ramp_is_feasible": float(controller.ramp_is_feasible),
            "ramp_deficit_min": float(
                max(controller.min_ramp_min - controller.plan.ramp_min, 0.0)
            ),
            "transition_magnitude": transition_magnitude(from_grade, to_grade),
        },
    )


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
def sample_plan(
    rng: np.random.Generator,
    grade_to: Grade,
    floor_min: float = 0.0,
    aggressive_bias: float = 0.0,
) -> ControlPlan:
    """
    Draw a control plan the way a mill would.

    `floor_min` is the slew-limited feasibility floor for this specific
    transition. Most plans respect it; roughly a quarter are rushed below it
    by production pressure, which is exactly the population GCI needs to warn
    about. Sampling around the floor rather than around a fixed nominal is
    what makes long transitions (large speed changes) harder than short ones,
    as they are in reality.
    """
    base = max(grade_to.nominal_ramp_min, floor_min)

    if rng.random() < 0.24 + aggressive_bias:
        # Rushed: below the physical feasibility floor.
        ramp = max(floor_min, 1.0) * rng.uniform(0.50, 0.92)
    else:
        ramp = base * rng.uniform(1.00, 1.45)

    return ControlPlan(
        ramp_min=float(ramp),
        lead_scale=float(np.clip(rng.normal(1.0, 0.22), 0.25, 1.8)),
        tau_c_scale=float(np.clip(rng.normal(1.15, 0.35), 0.45, 3.0)),
        trim_enabled=bool(rng.random() > 0.06),      # occasional trim in manual
        start_min=RAMP_START_MIN,
    )


def sample_transition(rng: np.random.Generator) -> Tuple[str, str]:
    """Pick a from/to grade pair, favouring the larger, harder transitions."""
    while True:
        a, b = rng.choice(GRADE_CODES, size=2, replace=False)
        mag = transition_magnitude(a, b)
        # Accept with probability rising in magnitude so the dataset is not
        # dominated by trivial 45->52 style moves.
        if rng.random() < min(0.25 + mag * 1.6, 1.0):
            return str(a), str(b)


def generate_dataset(
    n_events: int = 300,
    seed: int = 20260725,
    window_min: float = WINDOW_MIN,
    progress: bool = False,
) -> List[EventResult]:
    """Generate a full labelled corpus of grade-change events."""
    master = np.random.default_rng(seed)
    events: List[EventResult] = []

    for i in range(n_events):
        ev_seed = int(master.integers(0, 2**31 - 1))
        twin = PaperMachineTwin(seed=ev_seed)
        rng = twin.rng

        from_grade, to_grade = sample_transition(rng)
        g_from, g_to = get_grade(from_grade), get_grade(to_grade)

        # Feasibility floor is transition-specific, so it must be computed
        # before the plan is drawn.
        floor_min, _ = min_feasible_ramp_min(
            twin.inverse_solve(g_from),
            twin.inverse_solve(g_to),
            recipe_slew_limits(g_to),
        )
        plan = sample_plan(rng, g_to, floor_min=floor_min)
        faults = sample_faults(rng, window_min, plan.start_min)

        ev = run_event(
            twin,
            from_grade,
            to_grade,
            plan,
            faults,
            window_min=window_min,
            event_id=i,
            seed=ev_seed,
        )
        events.append(ev)

        if progress and (i + 1) % 50 == 0:
            print(f"  generated {i + 1}/{n_events} events")

    return events


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_dataset(events: Sequence[EventResult], out_dir: Path) -> Dict[str, Path]:
    """
    Persist a corpus as a compressed npz (series) plus JSON (metadata).

    npz keeps the payload small enough to regenerate rather than ship, and
    avoids a parquet engine dependency.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_events = len(events)
    n_steps = events[0].n_steps
    n_tags = len(SERIES_TAGS)

    cube = np.empty((n_events, n_steps, n_tags), dtype=np.float32)
    for i, ev in enumerate(events):
        for j, tag in enumerate(SERIES_TAGS):
            cube[i, :, j] = ev.series[tag]

    series_path = out_dir / "events_series.npz"
    np.savez_compressed(
        series_path,
        cube=cube,
        tags=np.array(SERIES_TAGS),
        dt_s=np.array([C.DT_S]),
    )

    meta_path = out_dir / "events_meta.json"
    meta = {
        "n_events": n_events,
        "n_steps": n_steps,
        "dt_s": C.DT_S,
        "window_min": n_steps * C.DT_S / 60.0,
        "tags": list(SERIES_TAGS),
        "events": [ev.to_meta() for ev in events],
        "faults": {
            ev.event_id: ev.faults for ev in events if ev.faults
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=float))

    return {"series": series_path, "meta": meta_path}


def load_dataset(out_dir: Path) -> Tuple[np.ndarray, List[str], dict]:
    """Load a persisted corpus: (cube, tags, metadata)."""
    out_dir = Path(out_dir)
    with np.load(out_dir / "events_series.npz", allow_pickle=False) as z:
        cube = z["cube"]
        tags = [str(t) for t in z["tags"]]
    meta = json.loads((out_dir / "events_meta.json").read_text())
    return cube, tags, meta


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_dataset(events: Sequence[EventResult]) -> Dict[str, object]:
    """
    Sanity-check a generated corpus before anything is trained on it.

    Catches the failure modes that silently ruin a hackathon model: no class
    balance, physically impossible values, or labels that disagree with the
    trajectories they were derived from.
    """
    issues: List[str] = []
    n = len(events)
    off_spec = np.array([ev.labels["off_spec"] for ev in events])
    settled = np.array([ev.labels["settled"] for ev in events])
    settle_times = np.array(
        [ev.labels["settle_min"] for ev in events], dtype=float
    )
    rate = float(off_spec.mean())

    if not (0.15 <= rate <= 0.75):
        issues.append(
            f"off-spec rate {rate:.1%} outside the usable 15-75% band; "
            "the classifier would be degenerate"
        )
    if settled.mean() < 0.70:
        issues.append(
            f"only {settled.mean():.1%} of events stabilise inside the window"
        )

    for ev in events:
        bw = ev.series["basis_weight"]
        if not np.all(np.isfinite(bw)):
            issues.append(f"event {ev.event_id}: non-finite basis weight")
            break
        if bw.min() < 5.0 or bw.max() > 400.0:
            issues.append(
                f"event {ev.event_id}: basis weight out of physical range "
                f"({bw.min():.1f}-{bw.max():.1f} g/m2)"
            )
            break
        m = ev.series["moisture"]
        if m.min() < 0.5 or m.max() > 30.0:
            issues.append(f"event {ev.event_id}: moisture out of range")
            break
        a = ev.series["ash"]
        if a.min() < -0.1 or a.max() > 60.0:
            issues.append(f"event {ev.event_id}: ash out of range")
            break

        # Label consistency: off_spec must agree with the trajectory
        dev = np.abs(bw_deviation_pct(bw, ev.series["basis_weight_sp"]))
        recomputed = float((dev > C.BW_SPEC_PCT).any())
        if recomputed != ev.labels["off_spec"]:
            issues.append(f"event {ev.event_id}: off_spec label inconsistent")
            break

    fault_counts: Dict[str, int] = {}
    for ev in events:
        key = ev.faults[0]["code"] if ev.faults else "NONE"
        fault_counts[key] = fault_counts.get(key, 0) + 1

    grade_pairs = len({(ev.from_grade, ev.to_grade) for ev in events})

    valid_settle = settle_times[np.isfinite(settle_times)]
    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "n_events": n,
        "off_spec_rate": rate,
        "settled_rate": float(settled.mean()),
        "median_settle_min": (
            float(np.median(valid_settle)) if valid_settle.size else float("nan")
        ),
        "mean_off_spec_minutes": float(
            np.mean([ev.labels["off_spec_minutes"] for ev in events])
        ),
        "distinct_grade_pairs": grade_pairs,
        "primary_cause_counts": dict(
            sorted(fault_counts.items(), key=lambda kv: -kv[1])
        ),
    }
