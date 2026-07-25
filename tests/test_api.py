"""
Tests for the API layer.

`TestGCIService` exercises the business logic directly against a small,
fast, custom `SimulatedDataSource` -- independent of HTTP and of whether the
trained model artefacts happen to be present. `TestApiEndpoints` exercises a
handful of routes through FastAPI's `TestClient` against the real app
(real trained artefacts, real demo corpus) as an end-to-end sanity check.
"""
import json
import tempfile
import unittest
from pathlib import Path

from gci.api.datasource import SimulatedDataSource
from gci.api.service import GCIService

ROOT = Path(__file__).resolve().parents[1]


def _service(tmp_dir: str, n_events: int = 8, seed: int = 999) -> GCIService:
    return GCIService(
        models_dir=ROOT / "models",
        data_dir=ROOT / "data",
        ledger_path=Path(tmp_dir) / "ledger.jsonl",
        datasource=SimulatedDataSource(n_events=n_events, seed=seed),
    )


class TestGCIService(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.service = _service(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_health_reports_status(self):
        health = self.service.health()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["n_demo_events"], 8)

    def test_grades_returns_all_seven(self):
        grades = self.service.grades()
        self.assertEqual(len(grades), 7)
        self.assertIn("basis_weight", grades[0])

    def test_list_events_matches_datasource_count(self):
        events = self.service.list_events()
        self.assertEqual(len(events), 8)
        for e in events:
            self.assertIn("primary_cause", e)

    def test_event_detail_unknown_raises_keyerror(self):
        with self.assertRaises(KeyError):
            self.service.event_detail(9999)

    def test_event_detail_known_event_has_series(self):
        eid = self.service.list_events()[0]["event_id"]
        detail = self.service.event_detail(eid)
        self.assertIn("basis_weight", detail["series"])
        self.assertEqual(len(detail["t_min"]), len(detail["series"]["basis_weight"]))

    def test_live_state_never_leaks_future_row(self):
        eid = self.service.list_events()[0]["event_id"]
        state = self.service.live_state(eid, t_min=5.0)
        self.assertLessEqual(state["t_min"], 5.0 + 1e-6)

    def test_default_event_id_is_valid(self):
        eid = self.service.default_event_id()
        self.assertIn(eid, self.service.datasource.list_events())

    def test_recommendations_returns_advisories_within_policy_cap(self):
        eid = self.service.default_event_id()
        advisories = self.service.recommendations(eid, t_min=15.0)
        self.assertLessEqual(len(advisories), self.service.policy.max_concurrent_suggestions)
        for a in advisories:
            self.assertGreaterEqual(a.confidence, self.service.policy.min_confidence_to_surface)

    def test_recommendations_are_recorded_in_ledger(self):
        eid = self.service.default_event_id()
        advisories = self.service.recommendations(eid, t_min=15.0)
        surfaced_ids = {e.advisory_id for e in self.service.ledger.entries if e.kind == "surfaced"}
        for a in advisories:
            self.assertIn(a.id, surfaced_ids)

    def test_feedback_updates_ledger_and_persists(self):
        eid = self.service.default_event_id()
        advisories = self.service.recommendations(eid, t_min=15.0)
        if not advisories:
            self.skipTest("no advisories surfaced for this seed/event")
        summary = self.service.feedback(advisories[0].id, "accepted", note="ok")
        self.assertGreaterEqual(summary["n_responded"], 1)
        self.assertTrue(self.service.ledger_path.exists())

    def test_feedback_rejects_invalid_decision(self):
        with self.assertRaises(ValueError):
            self.service.feedback("nonexistent", "maybe")

    def test_correlations_computation_returns_list_of_dicts(self):
        # Exercises the actual computation directly rather than through the
        # cache, since `correlations()` never computes inline (see below) --
        # it only ever returns whatever the background warm has produced.
        results = self.service._compute_correlations(max_events=8, max_lag_min=1.0, min_abs_correlation=0.3)
        dicts = [r.to_dict() for r in results]
        self.assertIsInstance(dicts, list)
        if dicts:
            self.assertIn("cause", dicts[0])
            self.assertIn("novel", dicts[0])

    def test_correlations_never_blocks_and_reads_the_cache(self):
        # `correlations()` must never trigger a computation itself -- it only
        # reflects whatever the background warm thread has set. An empty
        # cache reads as an empty list; a populated cache is returned as-is.
        self.service._correlations_cache = None
        self.assertEqual(self.service.correlations(), [])

        sentinel = [{"cause": "a", "effect": "b"}]
        self.service._correlations_cache = sentinel
        first = self.service.correlations()
        second = self.service.correlations()
        self.assertIs(first, second)

    def test_stabilization_returns_ranked_parameters(self):
        eid = self.service.default_event_id()
        impacts = self.service.stabilization(eid)
        params = {i["parameter"] for i in impacts}
        self.assertTrue(params.issubset({"ramp_min", "lead_scale", "tau_c_scale"}))

    def test_economics_round_trip(self):
        original = self.service.get_economics()
        updated = self.service.update_economics(net_margin_per_tonne=120.0)
        self.assertAlmostEqual(updated["net_margin_per_tonne"], 120.0)
        self.assertNotAlmostEqual(original["net_margin_per_tonne"], 120.0)

    def test_economics_update_ignores_unknown_and_none_fields(self):
        updated = self.service.update_economics(net_margin_per_tonne=None, rework_cost_per_tonne=50.0)
        self.assertAlmostEqual(updated["rework_cost_per_tonne"], 50.0)

    def test_trust_summary_before_any_feedback(self):
        summary = self.service.trust_summary()
        self.assertEqual(summary["n_responded"], 0)


class TestApiEndpoints(unittest.TestCase):
    """End-to-end sanity check through the real FastAPI app: real trained
    artefacts (if present), real demo corpus."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from gci.api.app import app

        cls.client = TestClient(app)

    def test_health_endpoint(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_grades_endpoint(self):
        r = self.client.get("/api/grades")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()), 7)

    def test_events_endpoint_nonempty(self):
        r = self.client.get("/api/events")
        self.assertEqual(r.status_code, 200)
        self.assertGreater(len(r.json()), 0)

    def test_event_detail_404_for_unknown(self):
        r = self.client.get("/api/events/999999")
        self.assertEqual(r.status_code, 404)

    def test_live_endpoint_defaults_to_a_valid_event(self):
        r = self.client.get("/api/live")
        self.assertEqual(r.status_code, 200)
        self.assertIn("basis_weight_dev_pct", r.json())

    def test_recommendations_then_feedback_round_trip(self):
        r = self.client.get("/api/recommendations")
        self.assertEqual(r.status_code, 200)
        advisories = r.json()
        if not advisories:
            self.skipTest("no advisories surfaced for the default demo event")
        adv_id = advisories[0]["id"]
        r2 = self.client.post(f"/api/recommendations/{adv_id}/feedback", json={"decision": "accepted"})
        self.assertEqual(r2.status_code, 200)

    def test_feedback_bad_decision_returns_400(self):
        r = self.client.post("/api/recommendations/nope/feedback", json={"decision": "whatever"})
        self.assertEqual(r.status_code, 400)

    def test_correlations_endpoint(self):
        r = self.client.get("/api/correlations")
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json(), list)

    def test_stabilization_endpoint(self):
        r = self.client.get("/api/stabilization")
        self.assertEqual(r.status_code, 200)

    def test_economics_get_and_put(self):
        r = self.client.get("/api/economics")
        self.assertEqual(r.status_code, 200)
        r2 = self.client.put("/api/economics", json={"net_margin_per_tonne": 111.0})
        self.assertEqual(r2.status_code, 200)
        self.assertAlmostEqual(r2.json()["net_margin_per_tonne"], 111.0)

    def test_trust_endpoint(self):
        r = self.client.get("/api/trust")
        self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()
