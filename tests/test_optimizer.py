"""Tests for the bounded setpoint/ramp optimizer."""
import unittest

from gci import optimizer
from gci.config import Source
from gci.control import ControlPlan, min_feasible_ramp_min, recipe_slew_limits
from gci.grades import get_grade
from gci.optimizer import (
    OptimizationBounds,
    PHYSICS_MODEL_CONFIDENCE,
    clear_cache,
    evaluate_plan,
    recommend_plan,
)
from gci.twin import PaperMachineTwin


class TestEvaluatePlan(unittest.TestCase):
    def test_returns_expected_label_keys(self):
        labels = evaluate_plan("NP-45", "SC-56", ControlPlan(ramp_min=8.0), seed=1)
        for key in ("off_spec_minutes", "max_abs_dev_pct", "settle_min"):
            self.assertIn(key, labels)

    def test_deterministic_with_same_seed(self):
        plan = ControlPlan(ramp_min=8.0, lead_scale=1.1)
        a = evaluate_plan("NP-45", "SC-56", plan, seed=7)
        b = evaluate_plan("NP-45", "SC-56", plan, seed=7)
        self.assertEqual(a, b)


class TestRecommendPlan(unittest.TestCase):
    def setUp(self):
        clear_cache()

    def test_respects_feasibility_floor(self):
        result = recommend_plan(
            "NP-45", "BRD-120", ControlPlan(ramp_min=3.0),
            seed=3, n_per_dim=5, n_rounds=1, use_cache=False,
        )
        self.assertGreaterEqual(result.plan.ramp_min, result.min_ramp_min - 1e-6)

    def test_never_worse_than_baseline(self):
        """Coordinate descent starts from the baseline as its first
        candidate, so the search can only match or improve it."""
        baseline = ControlPlan(ramp_min=4.0, lead_scale=0.6, tau_c_scale=1.8)
        result = recommend_plan(
            "NP-45", "BRD-120", baseline, seed=11,
            n_per_dim=5, n_rounds=1, use_cache=False,
        )
        self.assertLessEqual(
            result.recommended_labels["off_spec_minutes"],
            result.baseline_labels["off_spec_minutes"] + 1e-9,
        )

    def test_finds_real_improvement_for_a_rushed_plan(self):
        """A ramp well below the feasibility floor should be a genuinely
        bad baseline; the optimizer must find something better, not just
        tie."""
        g_from, g_to = get_grade("NP-45"), get_grade("BRD-120")
        probe = PaperMachineTwin(seed=21)
        floor_min, _ = min_feasible_ramp_min(
            probe.inverse_solve(g_from), probe.inverse_solve(g_to),
            recipe_slew_limits(g_to),
        )
        rushed = ControlPlan(ramp_min=max(floor_min * 0.5, 2.0))
        result = recommend_plan(
            "NP-45", "BRD-120", rushed, seed=21,
            n_per_dim=6, n_rounds=2, use_cache=False,
        )
        self.assertLess(
            result.recommended_labels["off_spec_minutes"],
            result.baseline_labels["off_spec_minutes"],
        )
        self.assertGreater(result.improvement["off_spec_minutes"], 0.0)

    def test_caching_returns_identical_result(self):
        baseline = ControlPlan(ramp_min=6.0)
        first = recommend_plan(
            "NP-45", "SC-56", baseline, seed=5, n_per_dim=4, n_rounds=1,
        )
        second = recommend_plan(
            "NP-45", "SC-56", baseline, seed=5, n_per_dim=4, n_rounds=1,
        )
        self.assertIs(first, second)

    def test_clear_cache_forces_recompute(self):
        baseline = ControlPlan(ramp_min=6.0)
        first = recommend_plan(
            "NP-45", "SC-56", baseline, seed=5, n_per_dim=4, n_rounds=1,
        )
        clear_cache()
        second = recommend_plan(
            "NP-45", "SC-56", baseline, seed=5, n_per_dim=4, n_rounds=1,
        )
        self.assertIsNot(first, second)
        # Labels can contain NaN (e.g. "never breached"), which is not
        # self-equal, so compare via the plan actually chosen instead of
        # dict equality on the labels.
        self.assertEqual(first.plan.to_dict(), second.plan.to_dict())
        self.assertAlmostEqual(
            first.recommended_labels["off_spec_minutes"],
            second.recommended_labels["off_spec_minutes"],
        )

    def test_price_ties_to_roi_engine(self):
        result = recommend_plan(
            "NP-45", "BRD-120", ControlPlan(ramp_min=3.0),
            seed=3, n_per_dim=5, n_rounds=1, use_cache=False,
        )
        value = result.price()
        # ramp_min=3.0 is below this transition's feasibility floor, so the
        # result is tagged RECIPE_LIMIT (pure recipe arithmetic), not
        # PHYSICS_MODEL -- see recommend_plan's source-selection logic.
        self.assertEqual(result.source, Source.RECIPE_LIMIT)
        self.assertEqual(value.source, Source.RECIPE_LIMIT)
        self.assertEqual(value.grade_code, "BRD-120")
        self.assertEqual(value.confidence, PHYSICS_MODEL_CONFIDENCE)

    def test_to_dict_is_serialisable(self):
        import json

        result = recommend_plan(
            "NP-45", "SC-56", ControlPlan(ramp_min=6.0), seed=5,
            n_per_dim=4, n_rounds=1, use_cache=False,
        )
        json.dumps(result.to_dict())  # must not raise


if __name__ == "__main__":
    unittest.main()
