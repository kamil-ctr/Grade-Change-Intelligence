"""Tests for the ROI / value-pricing engine."""
import unittest

from gci.config import AdvisoryPolicy, Economics, Source
from gci.grades import get_grade
from gci.roi import (
    portfolio_annual_value,
    price_plan_comparison,
    price_recommendation,
    production_rate_tonnes_per_min,
    should_surface,
    usd_per_offspec_tonne,
)


class TestProductionRate(unittest.TestCase):
    def test_matches_manual_calculation(self):
        grade = get_grade("NP-45")
        expected = grade.machine_speed * 6.0 * grade.basis_weight / 1.0e6
        self.assertAlmostEqual(production_rate_tonnes_per_min(grade), expected, places=9)

    def test_heavier_grade_has_higher_rate_per_unit_speed(self):
        light, heavy = get_grade("NP-45"), get_grade("BRD-120")
        # Not a same-speed comparison in general, but basis weight dominance
        # (120 vs 45 g/m2) should show up even after the speed difference.
        self.assertGreater(
            production_rate_tonnes_per_min(heavy) / heavy.machine_speed,
            production_rate_tonnes_per_min(light) / light.machine_speed,
        )


class TestUsdPerOffspecTonne(unittest.TestCase):
    def test_sums_margin_and_rework(self):
        econ = Economics(net_margin_per_tonne=95.0, rework_cost_per_tonne=42.0)
        self.assertAlmostEqual(usd_per_offspec_tonne(econ), 137.0)


class TestPriceRecommendation(unittest.TestCase):
    def test_zero_avoided_minutes_gives_zero_gross_value(self):
        v = price_recommendation(0.0, "NP-45", confidence=0.9)
        self.assertAlmostEqual(v.point_estimate_usd, 0.0)
        self.assertAlmostEqual(v.avoided_off_spec_tonnes, 0.0)

    def test_negative_avoided_minutes_clamped_to_zero(self):
        v = price_recommendation(-5.0, "NP-45", confidence=0.9)
        self.assertAlmostEqual(v.avoided_off_spec_tonnes, 0.0)
        self.assertAlmostEqual(v.point_estimate_usd, 0.0)

    def test_scales_with_confidence(self):
        low = price_recommendation(10.0, "NP-45", confidence=0.2)
        high = price_recommendation(10.0, "NP-45", confidence=0.9)
        self.assertLess(low.point_estimate_usd, high.point_estimate_usd)

    def test_rejects_out_of_range_confidence(self):
        with self.assertRaises(ValueError):
            price_recommendation(10.0, "NP-45", confidence=1.5)
        with self.assertRaises(ValueError):
            price_recommendation(10.0, "NP-45", confidence=-0.1)

    def test_band_ordering_for_positive_point_estimate(self):
        v = price_recommendation(10.0, "NP-45", confidence=0.8)
        self.assertLessEqual(v.low_usd, v.point_estimate_usd)
        self.assertLessEqual(v.point_estimate_usd, v.high_usd)

    def test_band_ordering_for_negative_point_estimate(self):
        # A large implementation cost with no avoided time forces a negative
        # point estimate; the band must still be correctly ordered.
        v = price_recommendation(
            0.0, "NP-45", confidence=0.5, implementation_cost_usd=500.0
        )
        self.assertLess(v.point_estimate_usd, 0.0)
        self.assertLessEqual(v.low_usd, v.point_estimate_usd)
        self.assertLessEqual(v.point_estimate_usd, v.high_usd)

    def test_annualized_uses_economics_factor(self):
        econ = Economics(grade_changes_per_day=2.0, operating_days_per_year=300.0)
        v = price_recommendation(10.0, "NP-45", confidence=1.0, economics=econ)
        self.assertAlmostEqual(
            v.annualized_usd, v.point_estimate_usd * 600.0, places=6
        )

    def test_source_and_grade_tagged(self):
        v = price_recommendation(
            5.0, "WFU-80", confidence=0.7, source=Source.PHYSICS_MODEL
        )
        self.assertEqual(v.source, Source.PHYSICS_MODEL)
        self.assertEqual(v.grade_code, "WFU-80")

    def test_to_dict_round_trips_key_fields(self):
        v = price_recommendation(5.0, "NP-45", confidence=0.6)
        d = v.to_dict()
        self.assertIn("point_estimate_usd", d)
        self.assertEqual(d["grade_code"], "NP-45")


class TestPricePlanComparison(unittest.TestCase):
    def test_positive_improvement_priced_positive(self):
        v = price_plan_comparison(
            baseline_off_spec_minutes=8.0, improved_off_spec_minutes=2.0,
            grade_to="NP-45", confidence=0.9,
        )
        self.assertGreater(v.point_estimate_usd, 0.0)
        self.assertAlmostEqual(v.avoided_off_spec_minutes, 6.0)

    def test_worse_plan_clamped_to_zero_not_negative(self):
        v = price_plan_comparison(
            baseline_off_spec_minutes=2.0, improved_off_spec_minutes=8.0,
            grade_to="NP-45", confidence=0.9,
        )
        self.assertAlmostEqual(v.point_estimate_usd, 0.0)


class TestShouldSurface(unittest.TestCase):
    def test_surfaces_when_above_both_floors(self):
        policy = AdvisoryPolicy(min_value_usd_to_surface=100.0, min_confidence_to_surface=0.3)
        v = price_recommendation(10.0, "NP-45", confidence=0.9)
        self.assertGreaterEqual(v.point_estimate_usd, 100.0)
        self.assertTrue(should_surface(v, policy))

    def test_suppressed_below_value_floor(self):
        policy = AdvisoryPolicy(min_value_usd_to_surface=1e9, min_confidence_to_surface=0.0)
        v = price_recommendation(10.0, "NP-45", confidence=0.9)
        self.assertFalse(should_surface(v, policy))

    def test_suppressed_below_confidence_floor(self):
        policy = AdvisoryPolicy(min_value_usd_to_surface=0.0, min_confidence_to_surface=0.99)
        v = price_recommendation(10.0, "NP-45", confidence=0.5)
        self.assertFalse(should_surface(v, policy))


class TestPortfolioAnnualValue(unittest.TestCase):
    def test_empty_is_zero(self):
        self.assertEqual(portfolio_annual_value([]), 0.0)

    def test_uses_mean_not_sum(self):
        econ = Economics(grade_changes_per_day=1.0, operating_days_per_year=100.0)
        total = portfolio_annual_value([100.0, 200.0, 300.0], economics=econ)
        # mean = 200, annualisation_factor = 100
        self.assertAlmostEqual(total, 200.0 * 100.0)


if __name__ == "__main__":
    unittest.main()
