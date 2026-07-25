"""
Fault and disturbance library.

Grade changes do not fail because the ramp maths is wrong -- they fail because
something else moves at the same time. This module injects the upsets that
mill engineers actually report, each with a physically sensible signature so
that the discovery engine has something real to find.

Every fault carries a `ground_truth_cause` so that model explanations can be
scored against what actually happened. That is what makes the accuracy claims
in the deck defensible rather than decorative.

Baseline (no-fault) behaviour is still not constant: all disturbance tags
carry slow correlated drift plus measurement noise, because a perfectly steady
machine would make the prediction problem artificially easy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config as C
from .twin import DV_NOMINAL

# ---------------------------------------------------------------------------
# Disturbance baseline
# ---------------------------------------------------------------------------
# Realistic 1-sigma slow drift for each disturbance tag over an event window.
DV_DRIFT_SIGMA: Dict[str, float] = {
    "headbox_consistency": 0.00022,
    "broke_ratio": 0.020,
    "wire_drainage_index": 0.012,
    "steam_header_kpa": 14.0,
    "white_water_freeness": 9.0,
    "couch_vacuum": 1.1,
    "press_load": 5.0,
    "refiner_sec": 3.0,
    "ambient_humidity": 2.0,
}

DV_BOUNDS: Dict[str, Tuple[float, float]] = {
    "headbox_consistency": (0.0055, 0.0125),
    "broke_ratio": (0.0, 0.45),
    "wire_drainage_index": (0.70, 1.10),
    "steam_header_kpa": (700.0, 1250.0),
    "white_water_freeness": (250.0, 430.0),
    "couch_vacuum": (30.0, 60.0),
    "press_load": (260.0, 400.0),
    "refiner_sec": (60.0, 140.0),
    "ambient_humidity": (25.0, 90.0),
}


def _ou_drift(
    rng: np.random.Generator, n: int, sigma: float, tau_min: float = 8.0
) -> np.ndarray:
    """
    Ornstein-Uhlenbeck drift: mean-reverting coloured noise.

    Real process disturbances are correlated in time, not white. Using OU
    rather than white noise is what makes lagged-correlation discovery a
    meaningful exercise instead of a coin flip.
    """
    theta = C.DT_S / (tau_min * 60.0)
    x = np.zeros(n, dtype=float)
    step_sigma = sigma * np.sqrt(2.0 * theta)
    for k in range(1, n):
        x[k] = x[k - 1] + theta * (0.0 - x[k - 1]) + rng.normal(0.0, step_sigma)
    return x


def baseline_disturbances(
    rng: np.random.Generator, n_steps: int, nominal: Optional[Dict[str, float]] = None
) -> Dict[str, np.ndarray]:
    """Healthy-machine disturbance trajectories: nominal plus correlated drift."""
    base = dict(DV_NOMINAL)
    if nominal:
        base.update(nominal)

    out: Dict[str, np.ndarray] = {}
    for tag, value in base.items():
        drift = _ou_drift(rng, n_steps, DV_DRIFT_SIGMA.get(tag, 0.0))
        # Each machine start has a slightly different operating point.
        offset = rng.normal(0.0, DV_DRIFT_SIGMA.get(tag, 0.0) * 0.6)
        series = value + offset + drift
        lo, hi = DV_BOUNDS.get(tag, (-np.inf, np.inf))
        out[tag] = np.clip(series, lo, hi)
    return out


# ---------------------------------------------------------------------------
# Fault definitions
# ---------------------------------------------------------------------------
@dataclass
class Fault:
    """A single injected upset with a named, physically-grounded cause."""

    code: str
    label: str
    target_tag: str
    shape: str                 # step | ramp | pulse | oscillation
    magnitude: float           # signed, in the target tag's units
    onset_min: float
    duration_min: float
    period_min: float = 2.0    # only used by 'oscillation'
    description: str = ""

    def profile(self, n_steps: int) -> np.ndarray:
        """Additive contribution of this fault over the event window."""
        t_min = np.arange(n_steps, dtype=float) * C.DT_S / 60.0
        out = np.zeros(n_steps, dtype=float)
        start, end = self.onset_min, self.onset_min + self.duration_min
        active = (t_min >= start) & (t_min <= end)

        if not active.any():
            return out

        if self.shape == "step":
            # First-order approach to the step (nothing in a mill is a true step)
            tau = 0.35
            out[active] = self.magnitude * (
                1.0 - np.exp(-(t_min[active] - start) / tau)
            )
            # Decay back after the fault clears
            after = t_min > end
            if after.any():
                tail = self.magnitude * (
                    1.0 - np.exp(-(end - start) / tau)
                )
                out[after] = tail * np.exp(-(t_min[after] - end) / (tau * 3.0))

        elif self.shape == "ramp":
            frac = np.clip((t_min - start) / max(self.duration_min, 1e-6), 0.0, 1.0)
            out = self.magnitude * frac
            out[t_min < start] = 0.0

        elif self.shape == "pulse":
            centre = (start + end) / 2.0
            width = max(self.duration_min / 4.0, 0.2)
            # Masked to the active window: a Gaussian's tails are infinite, but
            # a physical upset has a start, and leaking signal before onset
            # would be a subtle form of look-ahead in any model trained on it.
            out[active] = self.magnitude * np.exp(
                -0.5 * ((t_min[active] - centre) / width) ** 2
            )

        elif self.shape == "oscillation":
            out[active] = self.magnitude * np.sin(
                2.0 * np.pi * (t_min[active] - start) / max(self.period_min, 1e-6)
            )

        else:  # pragma: no cover - guarded by FAULT_CATALOG
            raise ValueError(f"Unknown fault shape '{self.shape}'")

        return out

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "label": self.label,
            "target_tag": self.target_tag,
            "shape": self.shape,
            "magnitude": self.magnitude,
            "onset_min": self.onset_min,
            "duration_min": self.duration_min,
            "description": self.description,
        }


@dataclass(frozen=True)
class FaultSpec:
    """Template describing how to sample a fault of a given kind."""

    code: str
    label: str
    target_tag: str
    shape: str
    mag_range: Tuple[float, float]
    duration_range: Tuple[float, float]
    description: str
    weight: float = 1.0
    signed: bool = False       # if True, magnitude sign is randomised

    def sample(self, rng: np.random.Generator, window_min: float,
               ramp_start_min: float) -> Fault:
        mag = float(rng.uniform(*self.mag_range))
        if self.signed and rng.random() < 0.5:
            mag = -mag
        duration = float(rng.uniform(*self.duration_range))
        # Faults cluster around the transition -- that is when the machine is
        # most disturbed and when operators report the most trouble.
        onset = float(
            np.clip(
                rng.normal(ramp_start_min + 1.5, 2.6),
                0.5,
                max(window_min - duration - 0.5, 1.0),
            )
        )
        return Fault(
            code=self.code,
            label=self.label,
            target_tag=self.target_tag,
            shape=self.shape,
            magnitude=mag,
            onset_min=onset,
            duration_min=duration,
            period_min=float(rng.uniform(1.2, 3.5)),
            description=self.description,
        )


FAULT_CATALOG: Tuple[FaultSpec, ...] = (
    FaultSpec(
        code="CONSISTENCY_UPSET",
        label="Headbox consistency upset",
        target_tag="headbox_consistency",
        shape="step",
        mag_range=(0.00035, 0.00110),
        duration_range=(3.0, 12.0),
        signed=True,
        weight=1.5,
        description=(
            "Stock preparation swing changes headbox consistency, so the same "
            "stock flow delivers a different fibre mass and basis weight moves "
            "with it."
        ),
    ),
    FaultSpec(
        code="STEAM_HEADER_DIP",
        label="Steam header pressure dip",
        target_tag="steam_header_kpa",
        shape="step",
        mag_range=(-260.0, -90.0),
        duration_range=(4.0, 14.0),
        weight=1.2,
        description=(
            "Mill steam demand elsewhere pulls header pressure down, capping "
            "dryer capacity and driving moisture up during the ramp."
        ),
    ),
    FaultSpec(
        code="RETENTION_LOSS",
        label="Retention aid dosing loss",
        target_tag="retention_aid",
        shape="ramp",
        mag_range=(-190.0, -70.0),
        duration_range=(4.0, 15.0),
        weight=1.3,
        description=(
            "Polymer pump degradation lowers first-pass retention, so retained "
            "solids fall and basis weight drifts below target."
        ),
    ),
    FaultSpec(
        code="BROKE_SURGE",
        label="Broke ratio surge",
        target_tag="broke_ratio",
        shape="step",
        mag_range=(0.07, 0.22),
        duration_range=(4.0, 16.0),
        weight=1.1,
        description=(
            "Broke from the previous grade re-enters the furnish, shifting "
            "filler retention and ash away from the new recipe."
        ),
    ),
    FaultSpec(
        code="WIRE_DRAINAGE_LOSS",
        label="Wire drainage degradation",
        target_tag="wire_drainage_index",
        shape="ramp",
        mag_range=(-0.16, -0.06),
        duration_range=(6.0, 18.0),
        weight=0.9,
        description=(
            "Wire wear or plugging reduces drainage, lowering retention and "
            "adding water load to the dryer section."
        ),
    ),
    FaultSpec(
        code="FREENESS_SHIFT",
        label="Refiner freeness shift",
        target_tag="white_water_freeness",
        shape="ramp",
        mag_range=(-55.0, -18.0),
        duration_range=(5.0, 16.0),
        signed=True,
        weight=0.9,
        description=(
            "Refining change alters fibre freeness, which moves retention and "
            "sheet drainage together."
        ),
    ),
    FaultSpec(
        code="HUMIDITY_RISE",
        label="Ambient humidity rise",
        target_tag="ambient_humidity",
        shape="ramp",
        mag_range=(9.0, 24.0),
        duration_range=(8.0, 20.0),
        weight=0.7,
        description=(
            "Hood humidity climbs, reducing evaporation rate for the same "
            "steam pressure and pushing moisture off target."
        ),
    ),
    FaultSpec(
        code="COUCH_VACUUM_LOSS",
        label="Couch vacuum loss",
        target_tag="couch_vacuum",
        shape="step",
        mag_range=(-9.0, -3.5),
        duration_range=(4.0, 12.0),
        weight=0.7,
        description=(
            "Vacuum system fault raises sheet water content entering the press "
            "section, an upstream cause of a downstream moisture excursion."
        ),
    ),
    FaultSpec(
        code="PRESS_LOAD_DRIFT",
        label="Press load drift",
        target_tag="press_load",
        shape="ramp",
        mag_range=(-38.0, -14.0),
        duration_range=(6.0, 18.0),
        signed=True,
        weight=0.6,
        description=(
            "Nip load drifts, changing sheet density and therefore caliper for "
            "an unchanged basis weight."
        ),
    ),
    FaultSpec(
        code="STEAM_HUNTING",
        label="Steam pressure hunting",
        target_tag="steam_header_kpa",
        shape="oscillation",
        mag_range=(35.0, 95.0),
        duration_range=(5.0, 15.0),
        weight=0.6,
        description=(
            "A badly tuned upstream pressure loop oscillates, modulating "
            "drying capacity and producing a cyclic moisture signature."
        ),
    ),
    FaultSpec(
        code="REFINER_LOAD_SWING",
        label="Refiner load swing",
        target_tag="refiner_sec",
        shape="pulse",
        mag_range=(14.0, 34.0),
        duration_range=(4.0, 12.0),
        signed=True,
        weight=0.6,
        description=(
            "Specific edge load swings during furnish changeover, an upstream "
            "driver not represented in the MD control matrix."
        ),
    ),
)

FAULT_BY_CODE: Dict[str, FaultSpec] = {f.code: f for f in FAULT_CATALOG}
FAULT_CODES: Tuple[str, ...] = tuple(f.code for f in FAULT_CATALOG)


def sample_faults(
    rng: np.random.Generator,
    window_min: float,
    ramp_start_min: float,
    n_faults: Optional[int] = None,
    p_none: float = 0.34,
) -> List[Fault]:
    """
    Draw a set of faults for one event.

    About a third of transitions run clean, which matches the reality that
    most grade changes succeed -- the class imbalance is part of the problem.
    """
    if n_faults is None:
        if rng.random() < p_none:
            n_faults = 0
        else:
            n_faults = int(rng.choice([1, 2, 3], p=[0.58, 0.32, 0.10]))

    if n_faults == 0:
        return []

    weights = np.array([f.weight for f in FAULT_CATALOG], dtype=float)
    weights = weights / weights.sum()
    idx = rng.choice(
        len(FAULT_CATALOG), size=min(n_faults, len(FAULT_CATALOG)),
        replace=False, p=weights,
    )
    return [FAULT_CATALOG[i].sample(rng, window_min, ramp_start_min) for i in idx]


def apply_faults(
    dv: Dict[str, np.ndarray], faults: Sequence[Fault]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Add fault profiles onto the baseline disturbance trajectories.

    Returns (disturbed_dv, actuator_overrides). Faults targeting an actuator
    tag (e.g. retention aid dosing) are returned separately because they must
    be applied to the controller output, not to the disturbance vector.
    """
    from .grades import MV_TAGS

    out = {k: np.array(v, dtype=float, copy=True) for k, v in dv.items()}
    overrides: Dict[str, np.ndarray] = {}
    n = len(next(iter(dv.values())))

    for fault in faults:
        profile = fault.profile(n)
        if fault.target_tag in MV_TAGS:
            overrides.setdefault(
                fault.target_tag, np.zeros(n, dtype=float)
            )
            overrides[fault.target_tag] += profile
        elif fault.target_tag in out:
            out[fault.target_tag] = out[fault.target_tag] + profile
            lo, hi = DV_BOUNDS.get(fault.target_tag, (-np.inf, np.inf))
            out[fault.target_tag] = np.clip(out[fault.target_tag], lo, hi)

    return out, overrides
