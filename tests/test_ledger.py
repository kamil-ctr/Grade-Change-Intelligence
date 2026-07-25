"""Tests for the advisory ledger."""
import math
import tempfile
import unittest
from pathlib import Path

from gci.config import Source
from gci.ledger import AdvisoryLedger
from gci.provenance import Advisory
from gci.roi import price_recommendation


def _advisory(id_, source=Source.RISK_MODEL, confidence=0.7, value=None):
    return Advisory(
        id=id_, title="t", source=source, confidence=confidence,
        explanation="x", value=value,
    )


class TestRecordAndRespond(unittest.TestCase):
    def test_record_returns_entry_id_and_stores_fields(self):
        ledger = AdvisoryLedger()
        value = price_recommendation(4.0, "NP-45", confidence=0.7)
        entry_id = ledger.record(_advisory("a1", value=value), event_id=42, timestamp=1000.0)
        self.assertEqual(len(ledger.entries), 1)
        e = ledger.entries[0]
        self.assertEqual(e.entry_id, entry_id)
        self.assertEqual(e.advisory_id, "a1")
        self.assertEqual(e.event_id, 42)
        self.assertAlmostEqual(e.value_usd, value.point_estimate_usd)

    def test_respond_rejects_invalid_decision(self):
        ledger = AdvisoryLedger()
        ledger.record(_advisory("a1"), timestamp=1.0)
        with self.assertRaises(ValueError):
            ledger.respond("a1", "maybe", timestamp=2.0)

    def test_respond_accepts_valid_decisions(self):
        ledger = AdvisoryLedger()
        ledger.record(_advisory("a1"), timestamp=1.0)
        ledger.respond("a1", "accepted", timestamp=2.0)
        self.assertEqual(len(ledger.entries), 2)
        self.assertEqual(ledger.entries[1].decision, "accepted")


class TestPersistence(unittest.TestCase):
    def test_save_and_load_round_trip(self):
        ledger = AdvisoryLedger()
        ledger.record(_advisory("a1"), event_id=1, timestamp=1.0)
        ledger.respond("a1", "accepted", note="looks right", timestamp=2.0)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.jsonl"
            ledger.save(path)
            reloaded = AdvisoryLedger.load(path)

        self.assertEqual(len(reloaded.entries), 2)
        self.assertEqual(reloaded.entries[0].advisory_id, "a1")
        self.assertEqual(reloaded.entries[1].decision, "accepted")
        self.assertEqual(reloaded.entries[1].note, "looks right")

    def test_load_missing_file_returns_empty_ledger(self):
        ledger = AdvisoryLedger.load("/nonexistent/path/ledger.jsonl")
        self.assertEqual(ledger.entries, [])


class TestQualityEvaluation(unittest.TestCase):
    def test_acceptance_rate_excludes_unanswered(self):
        ledger = AdvisoryLedger()
        ledger.record(_advisory("a1"), timestamp=1.0)
        ledger.record(_advisory("a2"), timestamp=1.0)
        ledger.respond("a1", "accepted", timestamp=2.0)
        # a2 never responded to
        self.assertAlmostEqual(ledger.acceptance_rate(), 1.0)

    def test_acceptance_rate_nan_when_nothing_responded(self):
        ledger = AdvisoryLedger()
        ledger.record(_advisory("a1"), timestamp=1.0)
        self.assertTrue(math.isnan(ledger.acceptance_rate()))

    def test_acceptance_rate_by_source_filter(self):
        ledger = AdvisoryLedger()
        ledger.record(_advisory("a1", source=Source.RISK_MODEL), timestamp=1.0)
        ledger.record(_advisory("a2", source=Source.PHYSICS_MODEL), timestamp=1.0)
        ledger.respond("a1", "accepted", timestamp=2.0)
        ledger.respond("a2", "rejected", timestamp=2.0)
        self.assertAlmostEqual(ledger.acceptance_rate(Source.RISK_MODEL), 1.0)
        self.assertAlmostEqual(ledger.acceptance_rate(Source.PHYSICS_MODEL), 0.0)

    def test_evaluate_summary_fields(self):
        ledger = AdvisoryLedger()
        v1 = price_recommendation(5.0, "NP-45", confidence=0.8)
        v2 = price_recommendation(1.0, "NP-45", confidence=0.4)
        ledger.record(_advisory("a1", confidence=0.8, value=v1), timestamp=1.0)
        ledger.record(_advisory("a2", confidence=0.4, value=v2), timestamp=1.0)
        ledger.respond("a1", "accepted", timestamp=2.0)
        ledger.respond("a2", "rejected", timestamp=2.0)

        summary = ledger.evaluate()
        self.assertEqual(summary["n_surfaced"], 2)
        self.assertEqual(summary["n_responded"], 2)
        self.assertAlmostEqual(summary["acceptance_rate_overall"], 0.5)
        self.assertAlmostEqual(summary["mean_confidence_accepted"], 0.8)
        self.assertAlmostEqual(summary["mean_confidence_rejected"], 0.4)
        self.assertAlmostEqual(summary["realized_value_usd_accepted"], v1.point_estimate_usd)

    def test_last_response_wins_on_repeated_response(self):
        ledger = AdvisoryLedger()
        ledger.record(_advisory("a1"), timestamp=1.0)
        ledger.respond("a1", "rejected", timestamp=2.0)
        ledger.respond("a1", "accepted", timestamp=3.0)
        self.assertAlmostEqual(ledger.acceptance_rate(), 1.0)


if __name__ == "__main__":
    unittest.main()
