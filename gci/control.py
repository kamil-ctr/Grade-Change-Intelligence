"""
Coordinated grade-change controller.

This is the *baseline automation* that GCI advises on top of -- deliberately
modelled on how a commercial MD control package executes a grade change:

  1. Target trajectory   The quality setpoints ramp from the old grade to the
                         new grade over a recipe-defined ramp time.

  2. Feedforward         Actuator setpoints are computed by inverting the
                         process model and ramped in coordination. Because the
                         process has dead time and lag, the actuator ramp is
                         started *early* (lead compensation) so the measured
                         quality tracks the target instead of trailing it.

  3. Feedback trim       A PI controller per loop removes the residual error
                         caused by model mismatch and disturbances. Gains are
                         computed with SIMC tuning from the identified FOPDT
                         parameters rather than hand-tuned, so they adapt
                         automatically to the operating point.

The three knobs GCI's optimizer is allowed to move are `ramp_min`,
`lead_scale` and `tau_c_scale` -- all of which are bounded by recipe limits.
Nothing here writes to a real control system; it is a simulation of the
plant's own controller so that "what would have happened" can be evaluated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from . import config as C
from .grades import CV_TAGS, Grade
from .twin import DYNAMICS, PaperMachineTwin

# Which actuator trims which quality variable (the MD control pairing)
TRIM_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("basis_weight", "stock_flow"),
    ("moisture", "steam_pressure"),
    ("ash", "filler_flow"),
)

# Which quality variable's dynamics govern each actuator's lead compensation.
# Every actuator needs this, not just the trimmed ones: machine speed is the
# dominant basis-weight driver on a large transition (a 430 m/min change moves
# basis weight far more than the stock flow change does), so leaving it
# uncompensated makes the sheet trail the target for the whole ramp.
MV_LEAD_DRIVER: Dict[str, str] = {
    "stock_flow": "basis_weight",
    "machine_speed": "basis_weight",
    "retention_aid": "basis_weight",
    "jet_wire_ratio": "basis_weight",
    "filler_flow": "ash",
    "steam_pressure": "moisture",
}


# A linear target ramp demands a step change in actuator *rate* at the moment
# the ramp starts, which no slew-limited drive can deliver -- so the sheet
# always lags at ramp onset. Commercial grade-change packages therefore shape
# the trajectory with an S-curve whose rate is zero at both ends. The cost is
# that peak rate is 1.5x the average, which raises the feasibility floor.
SCURVE_PEAK_RATE_FACTOR: float = 1.5


def scurve(u: np.ndarray | float) -> np.ndarray | float:
    """
    Smoothstep 3u^2 - 2u^3 on [0, 1].

    Zero first derivative at u=0 and u=1, so the transition eases in and out
    instead of demanding an instantaneous rate change from the drives.
    """
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def min_feasible_ramp_min(
    mv_from: Dict[str, float],
    mv_to: Dict[str, float],
    slew_limits: Dict[str, float],
) -> Tuple[float, str]:
    """
    Shortest ramp the recipe's slew limits physically permit, and which
    actuator sets it.

    This matters more than it first appears. A scheduler can ask for a 6-minute
    grade change, but if the drive can only shed 45 m/min of machine speed per
    minute and the transition needs 430 m/min, the sheet *cannot* track the
    target no matter how good the controller is -- basis weight will run low
    for the whole ramp. Detecting that up front is one of GCI's most valuable
    recommendations, and it is pure recipe arithmetic (RECIPE_LIMIT provenance,
    no model uncertainty).
    """
    worst_tag, worst_time = "", 0.0
    for tag, rate in slew_limits.items():
        if tag not in mv_from or tag not in mv_to or rate <= 0:
            continue
        # S-curve peak rate is 1.5x the average, and it is the peak that has
        # to fit under the slew limit.
        required = (
            abs(mv_to[tag] - mv_from[tag]) / rate * SCURVE_PEAK_RATE_FACTOR
        )
        if required > worst_time:
            worst_time, worst_tag = required, tag
    return float(worst_time), worst_tag


def recipe_slew_limits(grade_to: Grade) -> Dict[str, float]:
    """Per-minute actuator slew limits taken from the destination recipe."""
    return {
        "stock_flow": grade_to.max_stock_ramp,
        "filler_flow": grade_to.max_filler_ramp,
        "steam_pressure": grade_to.max_steam_ramp,
        "machine_speed": grade_to.max_speed_ramp,
        "retention_aid": 200.0,
        "jet_wire_ratio": 0.2,
    }


@dataclass
class ControlPlan:
    """
    The tunable part of a grade change. Everything here is bounded by the
    recipe; the GCI setpoint optimizer searches inside these bounds.
    """

    ramp_min: float                    # duration of the coordinated ramp
    lead_scale: float = 1.0            # multiplier on model-derived lead time
    tau_c_scale: float = 1.0           # SIMC closed-loop speed (higher = slower)
    trim_enabled: bool = True
    start_min: float = 5.0             # when the ramp begins in the event window

    def clipped(self, grade_to: Grade) -> "ControlPlan":
        """Enforce recipe bounds -- a plan can never violate the recipe."""
        return ControlPlan(
            ramp_min=float(np.clip(self.ramp_min, 2.0, 25.0)),
            lead_scale=float(np.clip(self.lead_scale, 0.0, 2.5)),
            tau_c_scale=float(np.clip(self.tau_c_scale, 0.3, 4.0)),
            trim_enabled=self.trim_enabled,
            start_min=float(self.start_min),
        )

    def to_dict(self) -> dict:
        return {
            "ramp_min": self.ramp_min,
            "lead_scale": self.lead_scale,
            "tau_c_scale": self.tau_c_scale,
            "trim_enabled": self.trim_enabled,
            "start_min": self.start_min,
        }


class PILoop:
    """
    Velocity-form PI controller with SIMC tuning and anti-windup.

    SIMC (Skogestad IMC) for a FOPDT process g(s) = K e^(-th s) / (tau s + 1):
        Kc = tau / (K * (tau_c + theta))
        Ti = min(tau, 4 * (tau_c + theta))
    with tau_c the desired closed-loop time constant. tau_c = theta gives the
    usual "moderately fast, robust" setting used in mill practice.
    """

    def __init__(
        self,
        process_gain: float,
        tau_s: float,
        dead_time_s: float,
        tau_c_scale: float = 1.0,
        out_limits: Tuple[float, float] = (-np.inf, np.inf),
    ):
        self.K = float(process_gain)
        self.tau = float(tau_s)
        self.theta = float(dead_time_s)
        tau_c = max(self.theta * tau_c_scale, 1e-3)

        denom = self.K * (tau_c + self.theta)
        self.Kc = self.tau / denom if abs(denom) > 1e-12 else 0.0
        self.Ti = min(self.tau, 4.0 * (tau_c + self.theta))
        self.out_limits = out_limits

        self._integral = 0.0
        self.last_error = 0.0

    def reset(self) -> None:
        self._integral = 0.0
        self.last_error = 0.0

    def step(self, error: float) -> float:
        """Return the trim to add to the feedforward actuator value."""
        self.last_error = float(error)
        proportional = self.Kc * error
        # Trapezoidal integration of the integral term
        self._integral += error * C.DT_S
        integral_term = (self.Kc / max(self.Ti, 1e-6)) * self._integral

        raw = proportional + integral_term
        clipped = float(np.clip(raw, *self.out_limits))

        # Conditional integration anti-windup: unwind if we saturated
        if raw != clipped and abs(self._integral) > 0:
            self._integral -= error * C.DT_S
            integral_term = (self.Kc / max(self.Ti, 1e-6)) * self._integral
            clipped = float(np.clip(proportional + integral_term, *self.out_limits))

        return clipped

    @property
    def gains(self) -> Dict[str, float]:
        return {"Kc": self.Kc, "Ti": self.Ti, "K": self.K, "tau": self.tau,
                "theta": self.theta}


def process_gains(
    twin: PaperMachineTwin, mv: Dict[str, float], dv: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Local process gain d(CV)/d(MV) for each trim pair, obtained by numerical
    perturbation of the twin at the current operating point.

    Using the twin rather than a fixed gain matrix is what lets the same
    controller work across a 45-150 g/m2 grade range without retuning.
    """
    base = twin.steady_state_at(mv, dv)
    gains: Dict[str, float] = {}
    for cv_tag, mv_tag in TRIM_PAIRS:
        delta = max(abs(mv[mv_tag]) * 0.02, 1e-4)
        perturbed = dict(mv)
        perturbed[mv_tag] = mv[mv_tag] + delta
        after = twin.steady_state_at(perturbed, dv)
        gains[mv_tag] = (after[cv_tag] - base[cv_tag]) / delta
    return gains


class GradeChangeController:
    """Executes one grade change: target trajectory + feedforward + PI trim."""

    def __init__(
        self,
        twin: PaperMachineTwin,
        grade_from: Grade,
        grade_to: Grade,
        mv_from: Dict[str, float],
        mv_to: Dict[str, float],
        plan: ControlPlan,
    ):
        self.twin = twin
        self.grade_from = grade_from
        self.grade_to = grade_to
        self.mv_from = dict(mv_from)
        self.mv_to = dict(mv_to)
        self.plan = plan.clipped(grade_to)

        self.start_s = self.plan.start_min * 60.0
        self.ramp_s = self.plan.ramp_min * 60.0

        # Lead time per quality loop.
        #
        # For a ramp input through a first-order-plus-dead-time process the
        # output lags the input by exactly (theta + tau) in steady ramp. So the
        # theoretically exact lead is theta + tau -- not a fraction of it.
        # `lead_scale` lets the optimizer detune this either way, which is what
        # a real commissioning engineer would trim by hand.
        self.lead_s: Dict[str, float] = {
            cv: (DYNAMICS[cv]["dead_time_s"] + DYNAMICS[cv]["tau_s"])
            * self.plan.lead_scale
            for cv in CV_TAGS
        }

        # Slew limits come straight from the recipe (RECIPE_LIMIT provenance).
        self.slew_limits: Dict[str, float] = recipe_slew_limits(grade_to)

        # Feasibility of the requested ramp, given those limits.
        self.min_ramp_min, self.binding_actuator = min_feasible_ramp_min(
            self.mv_from, self.mv_to, self.slew_limits
        )
        self.ramp_is_feasible = self.plan.ramp_min >= self.min_ramp_min - 1e-9

        # PI loops tuned at the *destination* operating point
        gains = process_gains(twin, self.mv_to)
        self.loops: Dict[str, PILoop] = {}
        for cv_tag, mv_tag in TRIM_PAIRS:
            span = abs(self.mv_to[mv_tag] - self.mv_from[mv_tag])
            trim_budget = max(0.25 * span, 0.12 * abs(self.mv_to[mv_tag]))
            self.loops[cv_tag] = PILoop(
                process_gain=gains[mv_tag],
                tau_s=DYNAMICS[cv_tag]["tau_s"],
                dead_time_s=DYNAMICS[cv_tag]["dead_time_s"],
                tau_c_scale=self.plan.tau_c_scale,
                out_limits=(-trim_budget, trim_budget),
            )

        self._mv_limits = {
            "stock_flow": grade_to.stock_flow_limits,
            "filler_flow": grade_to.filler_flow_limits,
            "steam_pressure": grade_to.steam_pressure_limits,
            "machine_speed": grade_to.speed_limits,
            "retention_aid": grade_to.retention_aid_limits,
            "jet_wire_ratio": (0.95, 1.05),
        }

    # -- trajectories ------------------------------------------------------
    def _fraction(self, t_s: float, lead_s: float = 0.0) -> float:
        """S-curve ramp progress in [0, 1] at time t, optionally shifted earlier."""
        if self.ramp_s <= 0:
            return 1.0 if t_s >= self.start_s - lead_s else 0.0
        u = (t_s - (self.start_s - lead_s)) / self.ramp_s
        return float(scurve(u))

    def setpoints_at(self, t_s: float) -> Dict[str, float]:
        """Quality target trajectory -- what the sheet is *supposed* to be."""
        w = self._fraction(t_s)
        a, b = self.grade_from, self.grade_to
        return {
            "basis_weight": a.basis_weight + (b.basis_weight - a.basis_weight) * w,
            "moisture": a.moisture + (b.moisture - a.moisture) * w,
            "ash": a.ash + (b.ash - a.ash) * w,
            "caliper": float("nan"),
        }

    def feedforward_at(self, t_s: float) -> Dict[str, float]:
        """Model-inverted actuator trajectory with per-loop lead compensation."""
        out: Dict[str, float] = {}
        for tag in self.mv_from:
            driver = MV_LEAD_DRIVER.get(tag)
            lead = self.lead_s.get(driver, 0.0) if driver else 0.0
            w = self._fraction(t_s, lead_s=lead)
            out[tag] = self.mv_from[tag] + (self.mv_to[tag] - self.mv_from[tag]) * w
        return out

    # -- main entry point --------------------------------------------------
    def command(self, t_s: float, measured: Dict[str, float]) -> Dict[str, float]:
        """Compute the actuator command for this sample."""
        ff = self.feedforward_at(t_s)
        sp = self.setpoints_at(t_s)

        if self.plan.trim_enabled:
            for cv_tag, mv_tag in TRIM_PAIRS:
                error = sp[cv_tag] - measured[cv_tag]
                ff[mv_tag] = ff[mv_tag] + self.loops[cv_tag].step(error)

        # Hard recipe clamp -- the controller can never leave the envelope.
        for tag, limits in self._mv_limits.items():
            if tag in ff:
                ff[tag] = float(np.clip(ff[tag], limits[0], limits[1]))
        return ff

    def reset(self) -> None:
        for loop in self.loops.values():
            loop.reset()

    def describe(self) -> dict:
        """Serialisable description used by the Copilot and provenance layer."""
        return {
            "from_grade": self.grade_from.code,
            "to_grade": self.grade_to.code,
            "plan": self.plan.to_dict(),
            "lead_s": dict(self.lead_s),
            "slew_limits": dict(self.slew_limits),
            "min_ramp_min": self.min_ramp_min,
            "binding_actuator": self.binding_actuator,
            "ramp_is_feasible": self.ramp_is_feasible,
            "pi_gains": {cv: loop.gains for cv, loop in self.loops.items()},
        }
