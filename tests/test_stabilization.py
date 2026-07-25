"""Tests for loop impact ranking on settling time."""
import json
import unittest

import numpy as np

from gci.config import Source
from gci.control import ControlPlan
from gci.stabilization import (
    DEFAULT_PERTURBATIONS,
    _finite_settle,
    rank_loop_impact,
)


class TestFiniteSettle(unittest.TestCase):
    def test_finite_value_passed_through(self):
        self.assertAlmostEqual(_finite_settle(7.5, window_min=30.0), 7.5)

    def test_nan_becomes_window_penalty(self):
        self.assertAlmostEqual(_finite_settle(float("nan"), window_min=30.0), 30.0)


class TestRankLoopImpact(unittest.TestCase):
    def test_all_default_parameters_present(self):
        results = rank_loop_impact(
            "NP-45", "SC-56", ControlPlan(ramp_min=8.0), seed=1,
        )
        params = {r.parameter for r in results}
        self.assertEqual(params, set(DEFAULT_PERTURBATIONS.keys()))

    def test_sorted_by_descending_abs_sensitivity(self):
        results = rank_loop_impact(
            "NP-45", "BRD-120", ControlPlan(ramp_min=10.0), seed=2,
        )
        mags = [abs(r.sensitivity_min_per_unit) for r in results]
        self.assertEqual(mags, sorted(mags, reverse=True))

    def test_best_never_worse_than_baseline(self):
        results = rank_loop_impact(
            "NP-45", "BRD-120", ControlPlan(ramp_min=6.0, lead_scale=0.6), seed=3,
        )
        for r in results:
            self.assertLessEqual(r.best_settle_min, r.baseline_settle_min + 1e-9)
            self.assertGreaterEqual(r.improvement_min, -1e-9)

    def test_zero_delta_perturbation_is_skipped(self):
        results = rank_loop_impact(
            "NP-45", "SC-56", ControlPlan(ramp_min=8.0), seed=4,
            perturbations={"ramp_min": (0.0, 0.0), "lead_scale": (-0.2, 0.2)},
        )
        params = {r.parameter for r in results}
        self.assertNotIn("ramp_min", params)
        self.assertIn("lead_scale", params)

    def test_source_tagged_physics_model(self):
        results = rank_loop_impact(
            "NP-45", "SC-56", ControlPlan(ramp_min=8.0), seed=5,
        )
        for r in results:
            self.assertEqual(r.source, Source.PHYSICS_MODEL)

    def test_deterministic_with_same_seed(self):
        plan = ControlPlan(ramp_min=7.0, tau_c_scale=1.2)
        a = rank_loop_impact("NP-45", "SC-56", plan, seed=9)
        b = rank_loop_impact("NP-45", "SC-56", plan, seed=9)
        self.assertEqual(
            [r.to_dict() for r in a], [r.to_dict() for r in b]
        )

    def test_to_dict_is_serialisable(self):
        results = rank_loop_impact(
            "NP-45", "SC-56", ControlPlan(ramp_min=8.0), seed=6,
        )
        json.dumps([r.to_dict() for r in results])  # must not raise

    def test_best_direction_matches_best_delta_sign(self):
        results = rank_loop_impact(
            "NP-45", "BRD-120", ControlPlan(ramp_min=5.0), seed=7,
        )
        for r in results:
            if r.best_delta > 0:
                self.assertEqual(r.best_direction, "increase")
            elif r.best_delta < 0:
                self.assertEqual(r.best_direction, "decrease")
            else:
                self.assertEqual(r.best_direction, "none")


if __name__ == "__main__":
    unittest.main()
