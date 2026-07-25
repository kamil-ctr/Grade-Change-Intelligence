"""Tests for the provenance / advisory packaging layer."""
import unittest

from gci.config import AdvisoryPolicy, Source
from gci.control import ControlPlan
from gci.discovery import CorrelationResult
from gci.optimizer import OptimizationResult
from gci.provenance import (
    Advisory,
    advisory_from_correlation,
    advisory_from_optimization,
    advisory_from_risk_prediction,
    advisory_from_stabilization,
    explain_correlation,
    explain_optimization,
    explain_risk_prediction,
    explain_stabilization,
    rank_and_gate,
)
from gci.roi import price_recommendation
from gci.stabilization import LoopImpact


def _make_correlation(is_known=False, r=0.6):
    return CorrelationResult(
        cause="dv_tag", effect="basis_weight", best_lag_min=0.5,
        correlation=r, mutual_information=0.3, n_samples=500, is_known=is_known,
    )


def _make_optimization():
    baseline = ControlPlan(ramp_min=4.0)
    plan = ControlPlan(ramp_min=7.0, lead_scale=1.1, tau_c_scale=0.9)
    return OptimizationResult(
        plan=plan, baseline_plan=baseline,
        baseline_labels={"off_spec_minutes": 5.0},
        recommended_labels={"off_spec_minutes": 1.0},
        improvement={"off_spec_minutes": 4.0},
        objective="off_spec_minutes", n_evaluations=30,
        min_ramp_min=6.0, binding_actuator="machine_speed", to_grade="NP-45",
    )


def _make_stabilization(direction="decrease"):
    return LoopImpact(
        parameter="tau_c_scale", baseline_value=1.0, baseline_settle_min=12.0,
        sensitivity_min_per_unit=-3.0, best_direction=direction, best_delta=-0.4,
        best_settle_min=10.4 if direction != "none" else 12.0,
        improvement_min=1.6 if direction != "none" else 0.0,
    )


class TestExplanations(unittest.TestCase):
    def test_risk_prediction_without_shap(self):
        text = explain_risk_prediction(0.72)
        self.assertIn("72%", text)

    def test_risk_prediction_with_shap_lists_drivers(self):
        text = explain_risk_prediction(0.72, [("bw_dev_headroom_pct", -0.5), ("plan_ramp_min", 0.2)])
        self.assertIn("bw_dev_headroom_pct", text)

    def test_correlation_known_vs_novel_wording(self):
        known_text = explain_correlation(_make_correlation(is_known=True))
        novel_text = explain_correlation(_make_correlation(is_known=False))
        self.assertIn("known", known_text)
        self.assertIn("newly discovered", novel_text)

    def test_optimization_mentions_ramp_change(self):
        text = explain_optimization(_make_optimization())
        self.assertIn("4.0", text)
        self.assertIn("7.0", text)

    def test_stabilization_none_direction_message(self):
        text = explain_stabilization(_make_stabilization(direction="none"))
        self.assertIn("neither probed direction", text)

    def test_stabilization_with_direction_mentions_minutes(self):
        text = explain_stabilization(_make_stabilization(direction="decrease"))
        self.assertIn("min", text)


class TestAdvisoryBuilders(unittest.TestCase):
    def test_risk_prediction_tagged_risk_model(self):
        a = advisory_from_risk_prediction("adv-1", 0.6, [("feat", 0.1)])
        self.assertEqual(a.source, Source.RISK_MODEL)
        self.assertEqual(a.confidence, 0.6)

    def test_correlation_confidence_bounded_by_abs_correlation(self):
        a = advisory_from_correlation("adv-2", _make_correlation(r=0.85))
        self.assertAlmostEqual(a.confidence, 0.85)
        a_neg = advisory_from_correlation("adv-3", _make_correlation(r=-0.7))
        self.assertAlmostEqual(a_neg.confidence, 0.7)

    def test_optimization_advisory_carries_value(self):
        value = price_recommendation(4.0, "NP-45", confidence=0.75)
        a = advisory_from_optimization("adv-4", _make_optimization(), confidence=0.75, value=value)
        self.assertIsNotNone(a.value)
        self.assertEqual(a.to_dict()["value"]["grade_code"], "NP-45")

    def test_stabilization_advisory_to_dict(self):
        a = advisory_from_stabilization("adv-5", _make_stabilization(), confidence=0.75)
        d = a.to_dict()
        self.assertEqual(d["source"], Source.PHYSICS_MODEL)
        self.assertIn("source_label", d)


class TestRankAndGate(unittest.TestCase):
    def test_filters_below_confidence_floor(self):
        policy = AdvisoryPolicy(min_confidence_to_surface=0.5, min_value_usd_to_surface=0.0)
        low = Advisory(id="a", title="t", source=Source.RISK_MODEL, confidence=0.2, explanation="x")
        high = Advisory(id="b", title="t", source=Source.RISK_MODEL, confidence=0.9, explanation="x")
        out = rank_and_gate([low, high], policy)
        self.assertEqual([a.id for a in out], ["b"])

    def test_value_floor_applies_only_to_priced_advisories(self):
        policy = AdvisoryPolicy(min_confidence_to_surface=0.0, min_value_usd_to_surface=1000.0)
        cheap_value = price_recommendation(0.1, "NP-45", confidence=0.9)
        priced_low = Advisory(
            id="priced", title="t", source=Source.PHYSICS_MODEL, confidence=0.9,
            explanation="x", value=cheap_value,
        )
        unpriced = Advisory(
            id="unpriced", title="t", source=Source.CORRELATION_DISCOVERY,
            confidence=0.9, explanation="x", value=None,
        )
        out = rank_and_gate([priced_low, unpriced], policy)
        self.assertEqual([a.id for a in out], ["unpriced"])

    def test_caps_at_max_concurrent_suggestions(self):
        policy = AdvisoryPolicy(
            min_confidence_to_surface=0.0, min_value_usd_to_surface=0.0,
            max_concurrent_suggestions=2,
        )
        advisories = [
            Advisory(id=str(i), title="t", source=Source.RISK_MODEL, confidence=0.5 + i * 0.01, explanation="x")
            for i in range(5)
        ]
        out = rank_and_gate(advisories, policy)
        self.assertEqual(len(out), 2)

    def test_priced_sorted_before_unpriced(self):
        policy = AdvisoryPolicy(min_confidence_to_surface=0.0, min_value_usd_to_surface=0.0)
        value = price_recommendation(4.0, "NP-45", confidence=0.9)
        priced = Advisory(id="priced", title="t", source=Source.PHYSICS_MODEL, confidence=0.5, explanation="x", value=value)
        unpriced_high_conf = Advisory(id="unpriced", title="t", source=Source.CORRELATION_DISCOVERY, confidence=0.99, explanation="x")
        out = rank_and_gate([unpriced_high_conf, priced], policy)
        self.assertEqual(out[0].id, "priced")


if __name__ == "__main__":
    unittest.main()
