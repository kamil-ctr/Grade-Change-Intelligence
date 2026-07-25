"""Tests for the coordinated grade-change controller."""
import unittest

import numpy as np

from gci import config as C
from gci.control import (
    SCURVE_PEAK_RATE_FACTOR,
    ControlPlan,
    GradeChangeController,
    PILoop,
    min_feasible_ramp_min,
    process_gains,
    recipe_slew_limits,
    scurve,
)
from gci.grades import get_grade
from gci.twin import PaperMachineTwin


class TestSCurve(unittest.TestCase):
    def test_endpoints_and_midpoint(self):
        self.assertAlmostEqual(float(scurve(0.0)), 0.0)
        self.assertAlmostEqual(float(scurve(1.0)), 1.0)
        self.assertAlmostEqual(float(scurve(0.5)), 0.5)

    def test_clamped_outside_unit_interval(self):
        self.assertAlmostEqual(float(scurve(-3.0)), 0.0)
        self.assertAlmostEqual(float(scurve(7.0)), 1.0)

    def test_monotonic(self):
        u = np.linspace(0.0, 1.0, 200)
        self.assertTrue(np.all(np.diff(scurve(u)) >= -1e-12))

    def test_zero_rate_at_both_ends(self):
        """The whole point: no step demand on the drives at start or finish."""
        u = np.linspace(0.0, 1.0, 2001)
        d = np.diff(scurve(u))
        self.assertLess(d[0], d[len(d) // 2] * 0.01)
        self.assertLess(d[-1], d[len(d) // 2] * 0.01)

    def test_peak_rate_factor_is_correct(self):
        u = np.linspace(0.0, 1.0, 20001)
        d = np.diff(scurve(u)) / np.diff(u)
        self.assertAlmostEqual(float(d.max()), SCURVE_PEAK_RATE_FACTOR, delta=0.01)


class TestPILoop(unittest.TestCase):
    def test_simc_gains_are_positive_and_finite(self):
        loop = PILoop(process_gain=2.5, tau_s=45.0, dead_time_s=25.0)
        self.assertGreater(loop.Kc, 0.0)
        self.assertGreater(loop.Ti, 0.0)
        self.assertTrue(np.isfinite(loop.Kc) and np.isfinite(loop.Ti))

    def test_slower_tuning_lowers_gain(self):
        fast = PILoop(2.5, 45.0, 25.0, tau_c_scale=0.5)
        slow = PILoop(2.5, 45.0, 25.0, tau_c_scale=3.0)
        self.assertGreater(fast.Kc, slow.Kc)

    def test_output_respects_limits(self):
        loop = PILoop(2.5, 45.0, 25.0, out_limits=(-1.0, 1.0))
        for _ in range(200):
            out = loop.step(100.0)
            self.assertLessEqual(out, 1.0 + 1e-9)
        for _ in range(400):
            out = loop.step(-100.0)
            self.assertGreaterEqual(out, -1.0 - 1e-9)

    def test_anti_windup_allows_prompt_recovery(self):
        """After long saturation the loop must come back without a long unwind."""
        loop = PILoop(2.5, 45.0, 25.0, out_limits=(-1.0, 1.0))
        for _ in range(300):
            loop.step(50.0)          # drive hard into saturation
        recovered = [loop.step(0.0) for _ in range(20)]
        self.assertLess(abs(recovered[-1]), 1.0)

    def test_zero_error_zero_output(self):
        loop = PILoop(2.5, 45.0, 25.0)
        self.assertAlmostEqual(loop.step(0.0), 0.0)


class TestFeasibility(unittest.TestCase):
    def setUp(self):
        self.twin = PaperMachineTwin(seed=0)

    def test_floor_is_positive_and_names_an_actuator(self):
        a, b = get_grade("NP-45"), get_grade("BRD-150")
        floor, tag = min_feasible_ramp_min(
            self.twin.inverse_solve(a), self.twin.inverse_solve(b),
            recipe_slew_limits(b),
        )
        self.assertGreater(floor, 0.0)
        self.assertIn(tag, recipe_slew_limits(b))

    def test_bigger_transition_has_a_higher_floor(self):
        small = min_feasible_ramp_min(
            self.twin.inverse_solve(get_grade("BRD-120")),
            self.twin.inverse_solve(get_grade("BRD-150")),
            recipe_slew_limits(get_grade("BRD-150")),
        )[0]
        big = min_feasible_ramp_min(
            self.twin.inverse_solve(get_grade("NP-45")),
            self.twin.inverse_solve(get_grade("BRD-150")),
            recipe_slew_limits(get_grade("BRD-150")),
        )[0]
        self.assertGreater(big, small)

    def test_identical_grades_need_no_time(self):
        g = get_grade("NP-45")
        mv = self.twin.inverse_solve(g)
        floor, _ = min_feasible_ramp_min(mv, mv, recipe_slew_limits(g))
        self.assertAlmostEqual(floor, 0.0)


class TestController(unittest.TestCase):
    def setUp(self):
        self.twin = PaperMachineTwin(seed=0)
        self.a, self.b = get_grade("LWC-52"), get_grade("WFU-70")
        self.mv_a = self.twin.inverse_solve(self.a)
        self.mv_b = self.twin.inverse_solve(self.b)

    def _controller(self, **kw):
        plan = ControlPlan(ramp_min=kw.pop("ramp_min", 10.0), **kw)
        return GradeChangeController(
            self.twin, self.a, self.b, self.mv_a, self.mv_b, plan
        )

    def test_setpoints_start_and_end_on_grade_targets(self):
        c = self._controller()
        start = c.setpoints_at(0.0)
        end = c.setpoints_at(60.0 * 60.0)
        self.assertAlmostEqual(start["basis_weight"], self.a.basis_weight, places=6)
        self.assertAlmostEqual(end["basis_weight"], self.b.basis_weight, places=6)
        self.assertAlmostEqual(start["ash"], self.a.ash, places=6)
        self.assertAlmostEqual(end["ash"], self.b.ash, places=6)

    def test_setpoint_trajectory_is_monotonic(self):
        c = self._controller()
        vals = [c.setpoints_at(t)["basis_weight"] for t in range(0, 1800, 5)]
        diffs = np.diff(vals)
        self.assertTrue(np.all(diffs >= -1e-9))

    def test_feedforward_leads_the_setpoint(self):
        """Actuators must start moving before the target does."""
        c = self._controller(lead_scale=1.0)
        t = c.start_s - 20.0  # 20 s before the target begins to move
        sp = c.setpoints_at(t)
        ff = c.feedforward_at(t)
        self.assertAlmostEqual(sp["basis_weight"], self.a.basis_weight, places=4)
        self.assertNotAlmostEqual(ff["stock_flow"], self.mv_a["stock_flow"], places=4)

    def test_zero_lead_removes_the_lead(self):
        c = self._controller(lead_scale=0.0)
        ff = c.feedforward_at(c.start_s - 20.0)
        self.assertAlmostEqual(ff["stock_flow"], self.mv_a["stock_flow"], places=6)

    def test_commands_never_leave_the_recipe_envelope(self):
        c = self._controller(ramp_min=2.0)   # deliberately aggressive
        measured = {
            "basis_weight": 20.0, "moisture": 20.0,
            "ash": 0.0, "caliper": 50.0,
        }  # absurd measurements to drive the trims hard
        for k in range(400):
            cmd = c.command(k * C.DT_S, measured)
            lo, hi = self.b.stock_flow_limits
            self.assertTrue(lo - 1e-9 <= cmd["stock_flow"] <= hi + 1e-9)
            lo, hi = self.b.steam_pressure_limits
            self.assertTrue(lo - 1e-9 <= cmd["steam_pressure"] <= hi + 1e-9)
            lo, hi = self.b.filler_flow_limits
            self.assertTrue(lo - 1e-9 <= cmd["filler_flow"] <= hi + 1e-9)

    def test_trim_disabled_leaves_pure_feedforward(self):
        c = self._controller(trim_enabled=False)
        measured = {"basis_weight": 0.0, "moisture": 0.0, "ash": 0.0,
                    "caliper": 0.0}
        cmd = c.command(600.0, measured)
        ff = c.feedforward_at(600.0)
        for tag in cmd:
            self.assertAlmostEqual(cmd[tag], ff[tag], places=6)

    def test_process_gains_have_expected_signs(self):
        gains = process_gains(self.twin, self.mv_b)
        self.assertGreater(gains["stock_flow"], 0.0)     # more stock -> heavier
        self.assertLess(gains["steam_pressure"], 0.0)    # more steam -> drier
        self.assertGreater(gains["filler_flow"], 0.0)    # more filler -> ashier

    def test_plan_is_clipped_to_sane_bounds(self):
        plan = ControlPlan(ramp_min=999.0, lead_scale=-5.0, tau_c_scale=99.0)
        clipped = plan.clipped(self.b)
        self.assertLessEqual(clipped.ramp_min, 25.0)
        self.assertGreaterEqual(clipped.lead_scale, 0.0)
        self.assertLessEqual(clipped.tau_c_scale, 4.0)


if __name__ == "__main__":
    unittest.main()
