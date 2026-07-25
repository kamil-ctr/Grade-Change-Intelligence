"""
Robustness tests: the API must never return a 500 for malformed or
out-of-range input, on any endpoint that accepts a real request body or a
user-supplied identifier.

Scope note: the running system only has two endpoints that accept a request
*body* (`POST /recommendations/{id}/feedback`, `PUT /economics`) plus a
handful of GET endpoints keyed by `event_id`/`t_min` into a fixed demo
corpus. There is no file-upload or raw-sensor-reading endpoint to hardened
against custom data -- these tests cover the input surface that actually
exists, not a hypothetical one.

Every test asserts either a clean 4xx (with a body, not a bare stack trace)
or a valid 2xx with a sensible fallback -- never a 500.
"""
import unittest

from fastapi.testclient import TestClient

from gci.api.app import app, service

client = TestClient(app)


def _default_event_id() -> int:
    return service.default_event_id()


class TestFeedbackMalformedInput(unittest.TestCase):
    def test_empty_body_is_rejected_cleanly(self):
        r = client.post("/api/recommendations/whatever/feedback", json={})
        self.assertEqual(r.status_code, 422)
        self.assertNotIn("Traceback", r.text)

    def test_missing_decision_field_is_rejected_cleanly(self):
        r = client.post("/api/recommendations/whatever/feedback", json={"note": "no decision here"})
        self.assertEqual(r.status_code, 422)

    def test_decision_wrong_type_is_rejected_cleanly(self):
        r = client.post("/api/recommendations/whatever/feedback", json={"decision": 12345})
        self.assertEqual(r.status_code, 422)

    def test_invalid_decision_value_returns_400(self):
        r = client.post("/api/recommendations/whatever/feedback", json={"decision": "maybe"})
        self.assertEqual(r.status_code, 400)

    def test_unknown_advisory_id_with_valid_decision_returns_404(self):
        r = client.post(
            "/api/recommendations/this-was-never-surfaced-xyz/feedback",
            json={"decision": "accepted"},
        )
        self.assertEqual(r.status_code, 404)

    def test_oversized_note_is_rejected_cleanly(self):
        r = client.post(
            "/api/recommendations/whatever/feedback",
            json={"decision": "accepted", "note": "x" * 5000},
        )
        self.assertEqual(r.status_code, 422)

    def test_unicode_note_is_handled(self):
        # Requires a real, surfaced advisory -- fetch one first.
        advisories = client.get(f"/api/recommendations?event_id={_default_event_id()}").json()
        if not advisories:
            self.skipTest("no advisories surfaced for the default event at this t_min")
        r = client.post(
            f"/api/recommendations/{advisories[0]['id']}/feedback",
            json={"decision": "accepted", "note": "密度 \U0001F4C8 café тест"},
        )
        self.assertEqual(r.status_code, 200)

    def test_control_characters_in_note_do_not_crash(self):
        advisories = client.get(f"/api/recommendations?event_id={_default_event_id()}").json()
        if not advisories:
            self.skipTest("no advisories surfaced for the default event at this t_min")
        r = client.post(
            f"/api/recommendations/{advisories[0]['id']}/feedback",
            json={"decision": "rejected", "note": "line1\x00line2\ttabbed\nnewline"},
        )
        self.assertEqual(r.status_code, 200)

    def test_malformed_json_body_is_rejected_cleanly(self):
        r = client.post(
            "/api/recommendations/whatever/feedback",
            content=b"{not valid json",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(r.status_code, 422)
        self.assertNotIn("Traceback", r.text)


class TestEconomicsOutOfRangeInput(unittest.TestCase):
    def test_negative_net_margin_is_rejected(self):
        r = client.put("/api/economics", json={"net_margin_per_tonne": -50000.0})
        self.assertEqual(r.status_code, 422)

    def test_non_numeric_value_is_rejected(self):
        r = client.put("/api/economics", json={"net_margin_per_tonne": "abc"})
        self.assertEqual(r.status_code, 422)

    def test_zero_grade_changes_per_day_is_rejected(self):
        r = client.put("/api/economics", json={"grade_changes_per_day": 0})
        self.assertEqual(r.status_code, 422)

    def test_operating_days_beyond_a_year_is_rejected(self):
        r = client.put("/api/economics", json={"operating_days_per_year": 400})
        self.assertEqual(r.status_code, 422)

    def test_empty_body_is_a_no_op_not_an_error(self):
        before = client.get("/api/economics").json()
        r = client.put("/api/economics", json={})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), before)

    def test_unexpected_extra_field_is_ignored_not_fatal(self):
        r = client.put("/api/economics", json={"net_margin_per_tonne": 95.0, "totally_made_up_field": 123})
        self.assertEqual(r.status_code, 200)

    def test_nan_value_is_rejected_not_silently_accepted(self):
        # JSON has no native NaN; float('nan') round-trips through Python's
        # json module as the literal token `NaN`, which some clients send.
        r = client.put(
            "/api/economics",
            content=b'{"net_margin_per_tonne": NaN}',
            headers={"Content-Type": "application/json"},
        )
        self.assertIn(r.status_code, (400, 422))


class TestEventLookupOutOfRangeInput(unittest.TestCase):
    def test_negative_event_id_returns_404_not_500(self):
        r = client.get("/api/events/-1")
        self.assertEqual(r.status_code, 404)

    def test_absurdly_large_event_id_returns_404_not_500(self):
        r = client.get("/api/events/999999999")
        self.assertEqual(r.status_code, 404)

    def test_non_integer_event_id_returns_422_not_500(self):
        r = client.get("/api/events/not-a-number")
        self.assertEqual(r.status_code, 422)

    def test_unknown_event_id_on_live_returns_404(self):
        r = client.get("/api/live?event_id=999999999")
        self.assertEqual(r.status_code, 404)

    def test_unknown_event_id_on_recommendations_returns_404(self):
        r = client.get("/api/recommendations?event_id=999999999")
        self.assertEqual(r.status_code, 404)

    def test_unknown_event_id_on_stabilization_returns_404(self):
        r = client.get("/api/stabilization?event_id=999999999")
        self.assertEqual(r.status_code, 404)

    def test_very_large_t_min_falls_back_gracefully(self):
        r = client.get(f"/api/live?event_id={_default_event_id()}&t_min=1000000")
        self.assertEqual(r.status_code, 200)
        self.assertIn("basis_weight_dev_pct", r.json())

    def test_negative_t_min_falls_back_gracefully(self):
        r = client.get(f"/api/live?event_id={_default_event_id()}&t_min=-500")
        self.assertEqual(r.status_code, 200)
        self.assertIn("basis_weight_dev_pct", r.json())

    def test_non_numeric_t_min_returns_422_not_500(self):
        r = client.get(f"/api/live?event_id={_default_event_id()}&t_min=not-a-number")
        self.assertEqual(r.status_code, 422)


if __name__ == "__main__":
    unittest.main()
