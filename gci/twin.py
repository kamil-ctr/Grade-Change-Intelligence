"""
Digital twin of a paper machine wet-end + dryer section.

Design notes
------------
The twin is deliberately *first-principles where it matters* and empirical
elsewhere, which is what makes its recommendations explainable:

  basis weight  = retained dry mass rate / sheet area rate      (mass balance)
  ash           = retained filler / total retained solids       (mass balance)
  moisture      = water load / drying capacity                  (energy balance)
  caliper       = basis weight / apparent sheet density         (empirical)

Each quality variable then reaches its steady-state value through a
first-order-plus-dead-time (FOPDT) response, because the physical transport
delay between an actuator and the QCS scanner is the dominant reason grade
changes are hard: by the time the scanner sees a deviation, the cause is
already 30-60 seconds in the past. Delays used here are in the range reported
for commercial machines (wire-to-scanner ~20-40 s, dryer section ~60-90 s).

Performance: a 30-minute event at 5 s sampling is 360 steps. `simulate` runs
in single-digit milliseconds, which is what makes the interactive What-If
Studio possible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from . import config as C
from .grades import CV_TAGS, DV_TAGS, MV_TAGS, Grade

# ---------------------------------------------------------------------------
# FOPDT parameters per quality variable
# ---------------------------------------------------------------------------
#   dead_time_s : transport delay actuator -> QCS scanner
#   tau_s       : first-order time constant of the response
DYNAMICS: Dict[str, Dict[str, float]] = {
    "basis_weight": {"dead_time_s": 25.0, "tau_s": 45.0, "noise": 0.16},
    "ash": {"dead_time_s": 35.0, "tau_s": 60.0, "noise": 0.14},
    "moisture": {"dead_time_s": 60.0, "tau_s": 150.0, "noise": 0.06},
    "caliper": {"dead_time_s": 40.0, "tau_s": 50.0, "noise": 0.7},
}

# Actuator response (valve / drive dynamics) and slew limits per second
ACTUATOR_TAU_S: Dict[str, float] = {
    "stock_flow": 8.0,
    "filler_flow": 10.0,
    "steam_pressure": 20.0,
    "machine_speed": 12.0,
    "retention_aid": 15.0,
    "jet_wire_ratio": 6.0,
}

# Nominal disturbance values (the "healthy machine" operating point)
DV_NOMINAL: Dict[str, float] = {
    "headbox_consistency": C.STOCK_CONSISTENCY,
    "broke_ratio": 0.14,
    "wire_drainage_index": 1.00,
    "steam_header_kpa": 1150.0,
    "white_water_freeness": 340.0,
    "couch_vacuum": 45.0,
    "press_load": 320.0,
    "refiner_sec": 95.0,
    "ambient_humidity": 52.0,
}


@dataclass
class SimResult:
    """Output of a twin run. All arrays are shape (T,) or (T, n)."""

    t_min: np.ndarray                  # time axis in minutes from event start
    mv: Dict[str, np.ndarray]          # realised actuator values
    mv_cmd: Dict[str, np.ndarray]      # commanded actuator values
    cv: Dict[str, np.ndarray]          # measured quality variables (with noise)
    cv_true: Dict[str, np.ndarray]     # noise-free quality variables
    dv: Dict[str, np.ndarray]          # disturbances
    sp: Dict[str, np.ndarray]          # active setpoint trajectory per CV

    @property
    def n_steps(self) -> int:
        return int(self.t_min.shape[0])

    def frame(self) -> Dict[str, np.ndarray]:
        """Flatten to a single tag -> series mapping for feature building."""
        out: Dict[str, np.ndarray] = {"t_min": self.t_min}
        out.update(self.mv)
        out.update(self.cv)
        out.update(self.dv)
        for k, v in self.sp.items():
            out[f"{k}_sp"] = v
        return out


class PaperMachineTwin:
    """Deterministic-given-seed paper machine simulator."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    # -- static / steady-state model ---------------------------------------
    @staticmethod
    def steady_state(
        stock_flow: np.ndarray,
        filler_flow: np.ndarray,
        steam_pressure: np.ndarray,
        machine_speed: np.ndarray,
        retention_aid: np.ndarray,
        headbox_consistency: np.ndarray,
        broke_ratio: np.ndarray,
        wire_drainage_index: np.ndarray,
        white_water_freeness: np.ndarray,
        press_load: np.ndarray,
        ambient_humidity: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Steady-state quality variables for a given operating point.

        Accepts scalars or equal-length arrays (fully vectorised).
        """
        eps = 1e-9

        # Retention aid raises first-pass retention with diminishing returns.
        aid_effect = 1.0 + 0.045 * np.log1p(
            np.maximum(retention_aid, eps) / 250.0
        )
        # Poor drainage / low freeness reduce effective retention.
        drainage_effect = 0.94 + 0.06 * wire_drainage_index
        freeness_effect = 1.0 + 0.00018 * (white_water_freeness - 340.0)

        fiber_ret = np.clip(
            C.FIBER_RETENTION * aid_effect * drainage_effect * freeness_effect,
            0.70,
            0.995,
        )
        # Broke furnish carries fines -> slightly better filler retention.
        filler_ret = np.clip(
            C.FILLER_RETENTION * aid_effect * (1.0 + 0.35 * broke_ratio),
            0.20,
            0.92,
        )

        # --- mass balance -> basis weight --------------------------------
        # kg/min of retained solids
        fiber_mass = stock_flow * 1000.0 * headbox_consistency * fiber_ret
        filler_mass = filler_flow * 1000.0 * C.FILLER_CONSISTENCY * filler_ret
        total_mass = fiber_mass + filler_mass

        area_rate = np.maximum(machine_speed * C.WIRE_WIDTH_M, eps)  # m2/min
        basis_weight = total_mass / area_rate * 1000.0               # g/m2

        # --- ash ---------------------------------------------------------
        ash = filler_mass / np.maximum(total_mass, eps) * 100.0

        # --- energy balance -> moisture ----------------------------------
        drying_demand = basis_weight * machine_speed
        drying_capacity = np.power(
            np.maximum(steam_pressure, 1.0), C.MOISTURE_STEAM_EXP
        )
        humidity_penalty = 1.0 + 0.0035 * (ambient_humidity - 52.0)
        moisture = (
            C.MOISTURE_FLOOR
            + C.MOISTURE_K * drying_demand / np.maximum(drying_capacity, eps)
            * humidity_penalty
        )
        moisture = np.clip(moisture, 1.5, 25.0)

        # --- caliper -----------------------------------------------------
        density = (
            C.BASE_SHEET_DENSITY
            * (1.0 + C.DENSITY_ASH_COEF * (ash - 10.0))
            * (1.0 + C.DENSITY_MOISTURE_COEF * (moisture - 6.0))
            * (1.0 + 0.00035 * (press_load - 320.0))
        )
        density = np.clip(density, 450.0, 1100.0)
        caliper = basis_weight / density * 1000.0  # um

        return {
            "basis_weight": basis_weight,
            "ash": ash,
            "moisture": moisture,
            "caliper": caliper,
        }

    # -- inverse: grade targets -> actuator setpoints ----------------------
    def inverse_solve(
        self, grade: Grade, dv: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Solve for the actuator setpoints that hit a grade's quality targets at
        the healthy operating point. This is how recipe nominal setpoints are
        derived, so the twin and the recipe book are always self-consistent.
        """
        d = dict(DV_NOMINAL)
        if dv:
            d.update(dv)

        speed = grade.machine_speed
        aid = 250.0

        aid_effect = 1.0 + 0.045 * np.log1p(aid / 250.0)
        drainage_effect = 0.94 + 0.06 * d["wire_drainage_index"]
        freeness_effect = 1.0 + 0.00018 * (d["white_water_freeness"] - 340.0)
        fiber_ret = float(
            np.clip(
                C.FIBER_RETENTION * aid_effect * drainage_effect
                * freeness_effect,
                0.70,
                0.995,
            )
        )
        filler_ret = float(
            np.clip(
                C.FILLER_RETENTION * aid_effect
                * (1.0 + 0.35 * d["broke_ratio"]),
                0.20,
                0.92,
            )
        )

        # Required retained mass rate (kg/min) for the target basis weight
        area_rate = speed * C.WIRE_WIDTH_M
        total_mass = grade.basis_weight * area_rate / 1000.0
        filler_mass = grade.ash / 100.0 * total_mass
        fiber_mass = total_mass - filler_mass

        stock_flow = fiber_mass / (
            1000.0 * d["headbox_consistency"] * fiber_ret
        )
        filler_flow = filler_mass / (
            1000.0 * C.FILLER_CONSISTENCY * filler_ret
        )

        # Steam pressure to land on the moisture target
        humidity_penalty = 1.0 + 0.0035 * (d["ambient_humidity"] - 52.0)
        margin = max(grade.moisture - C.MOISTURE_FLOOR, 0.2)
        capacity = C.MOISTURE_K * grade.basis_weight * speed * humidity_penalty
        steam_pressure = float(
            np.power(capacity / margin, 1.0 / C.MOISTURE_STEAM_EXP)
        )

        return {
            "stock_flow": float(
                np.clip(stock_flow, *grade.stock_flow_limits)
            ),
            "filler_flow": float(
                np.clip(filler_flow, *grade.filler_flow_limits)
            ),
            "steam_pressure": float(
                np.clip(steam_pressure, *grade.steam_pressure_limits)
            ),
            "machine_speed": float(speed),
            "retention_aid": aid,
            "jet_wire_ratio": 0.995,
        }

    # -- dynamics ----------------------------------------------------------
    @staticmethod
    def _first_order(u: np.ndarray, tau_s: float, y0: float) -> np.ndarray:
        """Discrete first-order lag: y[k] = a*y[k-1] + (1-a)*u[k]."""
        a = float(np.exp(-C.DT_S / max(tau_s, 1e-6)))
        y = np.empty_like(u, dtype=float)
        prev = float(y0)
        for k in range(u.shape[0]):
            prev = a * prev + (1.0 - a) * float(u[k])
            y[k] = prev
        return y

    @staticmethod
    def _apply_delay(u: np.ndarray, dead_time_s: float) -> np.ndarray:
        """Shift a series forward in time by the transport delay."""
        n = int(round(dead_time_s / C.DT_S))
        if n <= 0:
            return u.copy()
        out = np.empty_like(u, dtype=float)
        out[:n] = u[0]
        out[n:] = u[:-n]
        return out

    def _realise_actuators(
        self,
        mv_cmd: Dict[str, np.ndarray],
        slew_limits: Dict[str, float],
        mv0: Dict[str, float],
    ) -> Dict[str, np.ndarray]:
        """Apply per-minute slew limits then valve/drive lag to commands."""
        realised: Dict[str, np.ndarray] = {}
        for tag, cmd in mv_cmd.items():
            limited = np.empty_like(cmd, dtype=float)
            prev = float(mv0.get(tag, cmd[0]))
            max_step = slew_limits.get(tag, np.inf) / C.STEPS_PER_MIN
            for k in range(cmd.shape[0]):
                delta = float(cmd[k]) - prev
                delta = float(np.clip(delta, -max_step, max_step))
                prev = prev + delta
                limited[k] = prev
            realised[tag] = self._first_order(
                limited, ACTUATOR_TAU_S.get(tag, 10.0), float(mv0.get(tag, limited[0]))
            )
        return realised

    def simulate(
        self,
        mv_cmd: Dict[str, np.ndarray],
        dv: Dict[str, np.ndarray],
        slew_limits: Dict[str, float],
        mv0: Dict[str, float],
        cv0: Dict[str, float],
        sp: Dict[str, np.ndarray],
        add_noise: bool = True,
    ) -> SimResult:
        """
        Run the twin forward.

        Parameters
        ----------
        mv_cmd : commanded actuator trajectories, each shape (T,)
        dv     : disturbance trajectories, each shape (T,)
        mv0/cv0: initial conditions (previous grade's steady state)
        sp     : setpoint trajectory per quality variable (for plotting/labels)
        """
        T = len(next(iter(mv_cmd.values())))
        t_min = np.arange(T, dtype=float) * C.DT_S / 60.0

        mv = self._realise_actuators(mv_cmd, slew_limits, mv0)

        # Steam pressure cannot exceed what the header can supply.
        header = dv.get(
            "steam_header_kpa",
            np.full(T, DV_NOMINAL["steam_header_kpa"]),
        )
        mv["steam_pressure"] = np.minimum(mv["steam_pressure"], header * 0.95)

        ss = self.steady_state(
            stock_flow=mv["stock_flow"],
            filler_flow=mv["filler_flow"],
            steam_pressure=mv["steam_pressure"],
            machine_speed=mv["machine_speed"],
            retention_aid=mv["retention_aid"],
            headbox_consistency=dv["headbox_consistency"],
            broke_ratio=dv["broke_ratio"],
            wire_drainage_index=dv["wire_drainage_index"],
            white_water_freeness=dv["white_water_freeness"],
            press_load=dv["press_load"],
            ambient_humidity=dv["ambient_humidity"],
        )

        cv_true: Dict[str, np.ndarray] = {}
        cv_meas: Dict[str, np.ndarray] = {}
        for tag in CV_TAGS:
            spec = DYNAMICS[tag]
            target = self._apply_delay(ss[tag], spec["dead_time_s"])
            y = self._first_order(target, spec["tau_s"], cv0[tag])
            cv_true[tag] = y
            if add_noise:
                noise = self.rng.normal(0.0, spec["noise"], size=T)
                # QCS scanners average over a scan -> smooth the noise a little
                kernel = np.array([0.25, 0.5, 0.25])
                noise = np.convolve(noise, kernel, mode="same")
                cv_meas[tag] = y + noise
            else:
                cv_meas[tag] = y.copy()

        return SimResult(
            t_min=t_min,
            mv=mv,
            mv_cmd={k: np.asarray(v, dtype=float) for k, v in mv_cmd.items()},
            cv=cv_meas,
            cv_true=cv_true,
            dv={k: np.asarray(v, dtype=float) for k, v in dv.items()},
            sp=sp,
        )

    # -- incremental stepping (for closed-loop control) --------------------
    def make_stepper(
        self, mv0: Dict[str, float], cv0: Dict[str, float]
    ) -> "TwinStepper":
        return TwinStepper(self, mv0, cv0)

    # -- convenience -------------------------------------------------------
    def steady_state_at(
        self, mv: Dict[str, float], dv: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Scalar steady state for a single operating point."""
        d = dict(DV_NOMINAL)
        if dv:
            d.update(dv)
        out = self.steady_state(
            stock_flow=np.array([mv["stock_flow"]]),
            filler_flow=np.array([mv["filler_flow"]]),
            steam_pressure=np.array([mv["steam_pressure"]]),
            machine_speed=np.array([mv["machine_speed"]]),
            retention_aid=np.array([mv.get("retention_aid", 250.0)]),
            headbox_consistency=np.array([d["headbox_consistency"]]),
            broke_ratio=np.array([d["broke_ratio"]]),
            wire_drainage_index=np.array([d["wire_drainage_index"]]),
            white_water_freeness=np.array([d["white_water_freeness"]]),
            press_load=np.array([d["press_load"]]),
            ambient_humidity=np.array([d["ambient_humidity"]]),
        )
        return {k: float(v[0]) for k, v in out.items()}


class TwinStepper:
    """
    Incremental, one-sample-at-a-time view of the twin.

    Needed for closed-loop simulation, where the actuator command at step k
    depends on the measurement at step k-1. Holds the transport-delay buffers
    and first-order lag states that `simulate` computes in batch.
    """

    def __init__(
        self,
        twin: "PaperMachineTwin",
        mv0: Dict[str, float],
        cv0: Dict[str, float],
    ):
        self.twin = twin
        self.mv_actual = dict(mv0)      # post-slew, post-lag actuator values
        self.mv_slewed = dict(mv0)      # post-slew, pre-lag
        self.cv_true = dict(cv0)
        # Transport delay buffers: one FIFO of steady-state values per CV
        self._delay_buf: Dict[str, list] = {}
        for tag in CV_TAGS:
            n = max(int(round(DYNAMICS[tag]["dead_time_s"] / C.DT_S)), 0)
            self._delay_buf[tag] = [cv0[tag]] * max(n, 1)
            self._delay_n = None
        self._alpha_cv = {
            tag: float(np.exp(-C.DT_S / DYNAMICS[tag]["tau_s"]))
            for tag in CV_TAGS
        }
        self._alpha_mv = {
            tag: float(np.exp(-C.DT_S / ACTUATOR_TAU_S.get(tag, 10.0)))
            for tag in MV_TAGS
        }

    def step(
        self,
        mv_cmd: Dict[str, float],
        dv: Dict[str, float],
        slew_limits: Dict[str, float],
        add_noise: bool = True,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Advance one sample. Returns (measured_cv, realised_mv)."""
        # --- actuators: slew limit then first-order lag -------------------
        for tag in MV_TAGS:
            cmd = float(mv_cmd[tag])
            max_step = slew_limits.get(tag, np.inf) / C.STEPS_PER_MIN
            delta = float(np.clip(cmd - self.mv_slewed[tag], -max_step, max_step))
            self.mv_slewed[tag] += delta
            a = self._alpha_mv[tag]
            self.mv_actual[tag] = (
                a * self.mv_actual[tag] + (1.0 - a) * self.mv_slewed[tag]
            )

        # Header pressure physically caps dryer steam pressure.
        header = float(dv.get("steam_header_kpa", DV_NOMINAL["steam_header_kpa"]))
        self.mv_actual["steam_pressure"] = min(
            self.mv_actual["steam_pressure"], header * 0.95
        )

        # --- steady state at this operating point -------------------------
        ss = self.twin.steady_state(
            stock_flow=np.array([self.mv_actual["stock_flow"]]),
            filler_flow=np.array([self.mv_actual["filler_flow"]]),
            steam_pressure=np.array([self.mv_actual["steam_pressure"]]),
            machine_speed=np.array([self.mv_actual["machine_speed"]]),
            retention_aid=np.array([self.mv_actual["retention_aid"]]),
            headbox_consistency=np.array([dv["headbox_consistency"]]),
            broke_ratio=np.array([dv["broke_ratio"]]),
            wire_drainage_index=np.array([dv["wire_drainage_index"]]),
            white_water_freeness=np.array([dv["white_water_freeness"]]),
            press_load=np.array([dv["press_load"]]),
            ambient_humidity=np.array([dv["ambient_humidity"]]),
        )

        # --- transport delay + first-order lag ----------------------------
        measured: Dict[str, float] = {}
        for tag in CV_TAGS:
            buf = self._delay_buf[tag]
            buf.append(float(ss[tag][0]))
            delayed = buf.pop(0)
            a = self._alpha_cv[tag]
            self.cv_true[tag] = a * self.cv_true[tag] + (1.0 - a) * delayed
            if add_noise:
                measured[tag] = self.cv_true[tag] + float(
                    self.twin.rng.normal(0.0, DYNAMICS[tag]["noise"])
                )
            else:
                measured[tag] = self.cv_true[tag]

        return measured, dict(self.mv_actual)
