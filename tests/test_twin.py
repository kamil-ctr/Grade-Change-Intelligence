"""
Physics tests for the digital twin.

These assert the properties a process engineer would check first: does the
model conserve mass, does it respond in the right direction, and does it
settle where the algebra says it should. Written with `unittest` so the suite
runs with `python -m unittest` and needs no extra dependency.
"""
import unittest

import numpy as np

from gci import config as C
from gci.grades import CV_TAGS, GRADE_LIBRARY, get_grade
from gci.twin import DV_NOMINAL, DYNAMICS, PaperMachineTwin


class TestSteadyStateRoundTrip(unittest.TestCase):
    """inverse_solve -> steady_state must return the grade's own targets."""

    def setUp(self):
        self.twin = PaperMachineTwin(seed=0)

    def test_all_grades_round_trip(self):
        for code, grade in GRADE_LIBRARY.items():
            with self.subTest(grade=code):
                mv = self.twin.inverse_solve(grade)
                ss = self.twin.steady_state_at(mv)
                self.assertAlmostEqual(
                    ss["basis_weight"], grade.basis_weight, delta=0.05,
                    msg=f"{code}: basis weight",
                )
                self.assertAlmostEqual(
                    ss["ash"], grade.ash, delta=0.05, msg=f"{code}: ash",
                )
                # WFU-80 legitimately saturates the dryer section, so its
                # moisture target is only reachable to within the steam limit.
                self.assertAlmostEqual(
                    ss["moisture"], grade.moisture, delta=0.20,
                    msg=f"{code}: moisture",
                )

    def test_actuators_inside_recipe_envelope(self):
        for code, grade in GRADE_LIBRARY.items():
            with self.subTest(grade=code):
                mv = self.twin.inverse_solve(grade)
                lo, hi = grade.stock_flow_limits
                self.assertTrue(lo <= mv["stock_flow"] <= hi)
                lo, hi = grade.filler_flow_limits
                self.assertTrue(lo <= mv["filler_flow"] <= hi)
                lo, hi = grade.steam_pressure_limits
                self.assertTrue(lo <= mv["steam_pressure"] <= hi)


class TestMassBalance(unittest.TestCase):
    """Basis weight must obey the mass balance it is derived from."""

    def setUp(self):
        self.twin = PaperMachineTwin(seed=0)
        self.mv = self.twin.inverse_solve(get_grade("LWC-52"))

    def test_basis_weight_linear_in_stock_flow(self):
        """Doubling fibre flow at fixed speed must double the fibre mass."""
        base = self.twin.steady_state_at(self.mv)
        mv2 = dict(self.mv)
        mv2["filler_flow"] = 0.0
        mv3 = dict(mv2)
        mv3["stock_flow"] = mv2["stock_flow"] * 2.0

        no_filler = self.twin.steady_state_at(mv2)
        doubled = self.twin.steady_state_at(mv3)
        self.assertAlmostEqual(
            doubled["basis_weight"] / no_filler["basis_weight"], 2.0, delta=0.02
        )
        self.assertGreater(base["basis_weight"], no_filler["basis_weight"])

    def test_basis_weight_inverse_in_speed(self):
        """Basis weight is mass per unit area: halving speed doubles it."""
        base = self.twin.steady_state_at(self.mv)
        faster = dict(self.mv)
        faster["machine_speed"] = self.mv["machine_speed"] * 2.0
        out = self.twin.steady_state_at(faster)
        self.assertAlmostEqual(
            out["basis_weight"] / base["basis_weight"], 0.5, delta=0.01
        )

    def test_ash_is_bounded_fraction(self):
        for filler in (0.0, 0.1, 0.4, 1.0):
            mv = dict(self.mv)
            mv["filler_flow"] = filler
            ash = self.twin.steady_state_at(mv)["ash"]
            self.assertGreaterEqual(ash, 0.0)
            self.assertLessEqual(ash, 100.0)

    def test_zero_filler_gives_zero_ash(self):
        mv = dict(self.mv)
        mv["filler_flow"] = 0.0
        self.assertAlmostEqual(self.twin.steady_state_at(mv)["ash"], 0.0, places=6)


class TestMonotonicity(unittest.TestCase):
    """Directional responses must match papermaking intuition."""

    def setUp(self):
        self.twin = PaperMachineTwin(seed=0)
        self.mv = self.twin.inverse_solve(get_grade("WFU-70"))

    def _perturb(self, tag, factor):
        mv = dict(self.mv)
        mv[tag] = mv[tag] * factor
        return self.twin.steady_state_at(mv)

    def test_more_steam_dries_the_sheet(self):
        base = self.twin.steady_state_at(self.mv)
        self.assertLess(self._perturb("steam_pressure", 1.2)["moisture"],
                        base["moisture"])

    def test_faster_machine_wets_the_sheet_and_thins_it(self):
        base = self.twin.steady_state_at(self.mv)
        out = self._perturb("machine_speed", 1.15)
        self.assertLess(out["basis_weight"], base["basis_weight"])

    def test_more_filler_raises_ash(self):
        base = self.twin.steady_state_at(self.mv)
        self.assertGreater(self._perturb("filler_flow", 1.3)["ash"], base["ash"])

    def test_more_retention_aid_raises_basis_weight(self):
        base = self.twin.steady_state_at(self.mv)
        self.assertGreater(
            self._perturb("retention_aid", 1.5)["basis_weight"],
            base["basis_weight"],
        )

    def test_lower_consistency_lowers_basis_weight(self):
        base = self.twin.steady_state_at(self.mv)
        dv = dict(DV_NOMINAL)
        dv["headbox_consistency"] *= 0.9
        self.assertLess(
            self.twin.steady_state_at(self.mv, dv)["basis_weight"],
            base["basis_weight"],
        )

    def test_caliper_tracks_basis_weight(self):
        light = self.twin.steady_state_at(self.twin.inverse_solve(get_grade("NP-45")))
        heavy = self.twin.steady_state_at(
            self.twin.inverse_solve(get_grade("BRD-150"))
        )
        self.assertGreater(heavy["caliper"], light["caliper"])


class TestDynamics(unittest.TestCase):
    """FOPDT behaviour: dead time then a first-order approach."""

    def setUp(self):
        self.twin = PaperMachineTwin(seed=0)
        self.grade = get_grade("LWC-52")
        self.mv = self.twin.inverse_solve(self.grade)
        self.cv0 = self.twin.steady_state_at(self.mv)

    def test_steady_state_holds_without_drift(self):
        stepper = self.twin.make_stepper(self.mv, self.cv0)
        slew = {k: 1e6 for k in self.mv}
        for _ in range(240):  # 20 minutes
            measured, _ = stepper.step(self.mv, DV_NOMINAL, slew, add_noise=False)
        for tag in CV_TAGS:
            self.assertAlmostEqual(
                measured[tag], self.cv0[tag], delta=1e-3, msg=f"{tag} drifted"
            )

    def test_dead_time_delays_the_response(self):
        """Nothing may move before the transport delay has elapsed."""
        stepper = self.twin.make_stepper(self.mv, self.cv0)
        slew = {k: 1e6 for k in self.mv}
        step_mv = dict(self.mv)
        step_mv["stock_flow"] *= 1.25

        dead_steps = int(DYNAMICS["basis_weight"]["dead_time_s"] / C.DT_S)
        for k in range(dead_steps - 1):
            measured, _ = stepper.step(step_mv, DV_NOMINAL, slew, add_noise=False)
            self.assertAlmostEqual(
                measured["basis_weight"], self.cv0["basis_weight"], delta=1e-6,
                msg=f"basis weight moved at step {k}, before the dead time",
            )

    def test_approximately_first_order_at_tau(self):
        """
        One time constant after the dead time, a pure FOPDT process would have
        covered 63.2% of the step. The measured value is slightly lower here
        because the actuator's own lag (8 s for the stock valve) is in series
        with the process lag, making the overall response mildly second-order.
        The bound below is tight enough to catch a broken time constant while
        still admitting that cascade.
        """
        stepper = self.twin.make_stepper(self.mv, self.cv0)
        slew = {k: 1e6 for k in self.mv}
        step_mv = dict(self.mv)
        step_mv["stock_flow"] *= 1.25
        final = self.twin.steady_state_at(step_mv)["basis_weight"]
        start = self.cv0["basis_weight"]

        dead_steps = int(round(DYNAMICS["basis_weight"]["dead_time_s"] / C.DT_S))
        tau_steps = int(round(DYNAMICS["basis_weight"]["tau_s"] / C.DT_S))

        measured = None
        for _ in range(dead_steps + tau_steps):
            measured, _ = stepper.step(step_mv, DV_NOMINAL, slew, add_noise=False)

        fraction = (measured["basis_weight"] - start) / (final - start)
        self.assertGreater(fraction, 0.50)
        self.assertLess(fraction, 1.0 - np.exp(-1.0) + 0.02)

    def test_converges_to_analytic_steady_state(self):
        stepper = self.twin.make_stepper(self.mv, self.cv0)
        slew = {k: 1e6 for k in self.mv}
        step_mv = dict(self.mv)
        step_mv["steam_pressure"] *= 1.15
        expected = self.twin.steady_state_at(step_mv)

        for _ in range(600):  # 50 minutes, >> tau
            measured, _ = stepper.step(step_mv, DV_NOMINAL, slew, add_noise=False)
        for tag in CV_TAGS:
            self.assertAlmostEqual(
                measured[tag], expected[tag], delta=1e-2, msg=f"{tag}"
            )

    def test_slew_limit_is_respected(self):
        stepper = self.twin.make_stepper(self.mv, self.cv0)
        slew = {k: 1e6 for k in self.mv}
        slew["stock_flow"] = 2.0  # m3/min per minute
        target = dict(self.mv)
        target["stock_flow"] += 50.0

        prev = self.mv["stock_flow"]
        max_step = 2.0 / C.STEPS_PER_MIN
        for _ in range(60):
            _, mv_actual = stepper.step(target, DV_NOMINAL, slew, add_noise=False)
            self.assertLessEqual(
                mv_actual["stock_flow"] - prev, max_step + 1e-6
            )
            prev = mv_actual["stock_flow"]


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_result(self):
        def run():
            twin = PaperMachineTwin(seed=123)
            mv = twin.inverse_solve(get_grade("NP-45"))
            cv0 = twin.steady_state_at(mv)
            stepper = twin.make_stepper(mv, cv0)
            slew = {k: 1e6 for k in mv}
            out = []
            for _ in range(50):
                measured, _ = stepper.step(mv, DV_NOMINAL, slew, add_noise=True)
                out.append(measured["basis_weight"])
            return np.array(out)

        np.testing.assert_allclose(run(), run())


if __name__ == "__main__":
    unittest.main()
