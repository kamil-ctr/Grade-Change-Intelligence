"""
Tests for feature generation.

The most important test here is `test_no_future_leakage`. Leakage is the
single easiest way to produce an impressive-looking but worthless model, so it
is verified directly rather than assumed.
"""
import unittest

import numpy as np
import pandas as pd

from gci import config as C
from gci.control import ControlPlan
from gci.events import generate_dataset, run_event
from gci.features import (
    WARMUP_STEPS,
    build_dataset_features,
    build_event_features,
    event_wise_split,
    feature_columns,
    future_breach_label,
    time_to_breach,
)
from gci.twin import PaperMachineTwin


class TestLabelFunctions(unittest.TestCase):
    def test_breach_label_is_forward_looking_only(self):
        n = 60
        dev = np.zeros(n)
        dev[40] = 99.0                       # single breach at sample 40
        y = future_breach_label(dev, horizon_min=1.0)   # 12 samples
        self.assertEqual(y[39], 1)           # just before -> positive
        self.assertEqual(y[40], 0)           # at the breach itself -> not future
        self.assertEqual(y[27], 0)           # 13 samples before -> outside horizon
        self.assertEqual(y[28], 1)           # 12 samples before -> inside horizon

    def test_no_breach_gives_all_zeros(self):
        y = future_breach_label(np.zeros(100), horizon_min=5.0)
        self.assertEqual(int(y.sum()), 0)

    def test_time_to_breach_matches_the_label(self):
        n = 60
        dev = np.zeros(n)
        dev[40] = 99.0
        y = future_breach_label(dev, horizon_min=1.0)
        ttb = time_to_breach(dev, horizon_min=1.0)
        self.assertTrue(np.all(np.isfinite(ttb[y == 1])))
        self.assertTrue(np.all(np.isnan(ttb[y == 0])))
        self.assertAlmostEqual(ttb[39], C.DT_S / 60.0, places=6)


class TestFeatureTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        twin = PaperMachineTwin(seed=42)
        cls.event = run_event(
            twin, "NP-45", "BRD-120", ControlPlan(ramp_min=6.0), [],
            event_id=0, seed=42,
        )
        cls.df = build_event_features(cls.event)

    def test_no_nan_or_inf_in_features(self):
        cols = feature_columns(self.df)
        block = self.df[cols].to_numpy(dtype=float)
        self.assertFalse(np.isnan(block).any(), "NaN in feature matrix")
        self.assertFalse(np.isinf(block).any(), "Inf in feature matrix")

    def test_warmup_rows_are_dropped(self):
        self.assertEqual(len(self.df), self.event.n_steps - WARMUP_STEPS)

    def test_identifier_columns_present(self):
        for col in ("event_id", "from_grade", "to_grade", "primary_cause",
                    "y_breach"):
            self.assertIn(col, self.df.columns)

    def test_labels_are_binary(self):
        self.assertTrue(set(self.df["y_breach"].unique()) <= {0, 1})

    def test_feature_columns_exclude_labels(self):
        cols = feature_columns(self.df)
        for banned in ("y_breach", "y_time_to_breach_min", "event_id",
                       "sample_idx", "t_min"):
            self.assertNotIn(banned, cols)

    def test_deviation_feature_matches_manual_calculation(self):
        s = self.event.series
        expected = (
            (s["basis_weight"] - s["basis_weight_sp"]) / s["basis_weight_sp"] * 100.0
        )[WARMUP_STEPS:]
        np.testing.assert_allclose(
            self.df["bw_dev_pct"].to_numpy(), expected, rtol=1e-9
        )

    def test_projection_feature_extrapolates_the_trend(self):
        proj = (
            self.df["bw_dev_pct"]
            + self.df["bw_dev_roc_1min"] * C.RISK_HORIZON_MIN
        )
        np.testing.assert_allclose(
            self.df["bw_dev_projected"].to_numpy(), proj.to_numpy(), rtol=1e-9
        )

    def test_reasonable_feature_count(self):
        self.assertGreater(len(feature_columns(self.df)), 60)


class TestNoLeakage(unittest.TestCase):
    def test_no_future_leakage(self):
        """
        Corrupt the second half of an event's trajectories and confirm that
        features in the first half are bit-identical. Any backward-looking
        feature must be blind to the future by construction.
        """
        twin = PaperMachineTwin(seed=5)
        ev = run_event(
            twin, "LWC-52", "WFU-80", ControlPlan(ramp_min=7.0), [],
            event_id=1, seed=5,
        )
        base = build_event_features(ev)

        cut = ev.n_steps // 2
        corrupted = type(ev)(
            event_id=ev.event_id, from_grade=ev.from_grade, to_grade=ev.to_grade,
            plan=ev.plan, faults=ev.faults,
            series={k: v.copy() for k, v in ev.series.items()},
            labels=ev.labels, seed=ev.seed, context=ev.context,
        )
        rng = np.random.default_rng(0)
        for tag, arr in corrupted.series.items():
            arr[cut:] += rng.normal(0.0, 5.0, size=arr[cut:].shape)

        after = build_event_features(corrupted)

        # Compare only rows strictly before the corruption point.
        n_rows = cut - WARMUP_STEPS
        cols = feature_columns(base)
        np.testing.assert_allclose(
            base[cols].to_numpy(dtype=float)[:n_rows],
            after[cols].to_numpy(dtype=float)[:n_rows],
            rtol=1e-12,
            atol=1e-12,
            err_msg="a feature reads from the future",
        )


class TestSplitting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = generate_dataset(n_events=12, seed=99)
        cls.df = build_dataset_features(cls.events)

    def test_split_is_disjoint_by_event(self):
        train, test = event_wise_split(self.df, test_frac=0.25, seed=1)
        self.assertEqual(
            set(train["event_id"]) & set(test["event_id"]), set()
        )
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)

    def test_split_preserves_every_row(self):
        train, test = event_wise_split(self.df, test_frac=0.25, seed=1)
        self.assertEqual(len(train) + len(test), len(self.df))

    def test_split_is_deterministic(self):
        a, _ = event_wise_split(self.df, seed=7)
        b, _ = event_wise_split(self.df, seed=7)
        self.assertEqual(list(a["event_id"]), list(b["event_id"]))

    def test_dataset_features_have_both_classes(self):
        self.assertEqual(set(self.df["y_breach"].unique()), {0, 1})


if __name__ == "__main__":
    unittest.main()
