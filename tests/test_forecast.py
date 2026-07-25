"""
Tests for the quantile forecast module.

`test_forward_perturbation_changes_only_reachable_targets` is the forecast
analogue of `test_features.py::test_no_future_leakage`: instead of proving
features are blind to the future, it proves the *targets* are correctly
reading the future -- exactly the samples a given horizon should reach, and
no others.
"""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gci import config as C
from gci.control import ControlPlan
from gci.events import generate_dataset, run_event
from gci.features import build_dataset_features, build_event_features
from gci.forecast import (
    FORECAST_HORIZONS_MIN,
    FORECAST_QUANTILES,
    FORECAST_TARGET_COLS,
    ForecastPipeline,
    build_forecast_targets,
    dev_lookup_from_events,
    forecast_feature_columns,
    pinball_loss,
    target_column,
)
from gci.twin import PaperMachineTwin


class TestTargetConstruction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        twin = PaperMachineTwin(seed=11)
        cls.event = run_event(
            twin, "NP-45", "SC-56", ControlPlan(ramp_min=6.0), [],
            event_id=0, seed=11,
        )
        cls.features = build_event_features(cls.event)
        cls.dev_lookup = dev_lookup_from_events([cls.event])
        cls.targets = build_forecast_targets(cls.dev_lookup, cls.features)

    def test_target_columns_present(self):
        for h in FORECAST_HORIZONS_MIN:
            self.assertIn(target_column(h), self.targets.columns)

    def test_target_matches_manual_shift(self):
        """The target at row i, horizon h, must equal the raw deviation
        exactly `h` minutes later in the same event -- no off-by-one, no
        smoothing."""
        dev = self.dev_lookup[self.event.event_id]
        row = 5
        sample_idx = int(self.targets["sample_idx"].iloc[row])
        for h in FORECAST_HORIZONS_MIN:
            steps = int(round(h * C.STEPS_PER_MIN))
            expected_idx = sample_idx + steps
            got = self.targets[target_column(h)].iloc[row]
            if expected_idx < len(dev):
                self.assertAlmostEqual(got, dev[expected_idx], places=8)
            else:
                self.assertTrue(np.isnan(got))

    def test_target_nan_near_window_end(self):
        """The last rows of an event cannot see far enough ahead for the
        largest horizon; they must be NaN, not a fabricated value."""
        col = target_column(max(FORECAST_HORIZONS_MIN))
        n = self.event.n_steps
        steps = int(round(max(FORECAST_HORIZONS_MIN) * C.STEPS_PER_MIN))
        last_row = self.targets.iloc[-1]
        self.assertGreaterEqual(int(last_row["sample_idx"]) + steps, n)
        self.assertTrue(np.isnan(last_row[col]))

    def test_forward_perturbation_changes_only_reachable_targets(self):
        """Corrupting a single future sample must change the target of every
        row whose horizon reaches it, and no others -- proof the lookup reads
        exactly the intended future sample, not something looser."""
        dev = self.dev_lookup[self.event.event_id].copy()
        corrupt_at = len(dev) - 5
        dev_corrupted = dev.copy()
        dev_corrupted[corrupt_at] += 1000.0

        targets_before = build_forecast_targets(
            {self.event.event_id: dev}, self.features
        )
        targets_after = build_forecast_targets(
            {self.event.event_id: dev_corrupted}, self.features
        )

        h = FORECAST_HORIZONS_MIN[0]
        steps = int(round(h * C.STEPS_PER_MIN))
        col = target_column(h)
        reaches = (
            self.features["sample_idx"].to_numpy() + steps == corrupt_at
        )
        changed = ~np.isclose(
            targets_before[col].to_numpy(), targets_after[col].to_numpy(),
            equal_nan=True,
        )
        np.testing.assert_array_equal(changed, reaches)

    def test_features_exclude_forecast_targets(self):
        cols = forecast_feature_columns(self.targets)
        for c in FORECAST_TARGET_COLS:
            self.assertNotIn(c, cols)

    def test_features_unaffected_by_target_construction(self):
        """Adding forecast targets must not alter any backward-looking
        feature column already present in the frame."""
        base_cols = [c for c in self.features.columns if c not in FORECAST_TARGET_COLS]
        pd.testing.assert_frame_equal(
            self.targets[base_cols], self.features[base_cols]
        )


class TestPinballLoss(unittest.TestCase):
    def test_zero_for_perfect_prediction(self):
        y = np.array([1.0, -2.0, 3.5])
        self.assertAlmostEqual(pinball_loss(y, y, 0.5), 0.0)
        self.assertAlmostEqual(pinball_loss(y, y, 0.1), 0.0)
        self.assertAlmostEqual(pinball_loss(y, y, 0.9), 0.0)

    def test_asymmetric_for_biased_prediction(self):
        y_true = np.zeros(10)
        y_pred = np.ones(10)  # always over-predicts by 1
        # A high quantile (0.9) should be penalised less for over-prediction
        # than a low quantile (0.1).
        self.assertLess(pinball_loss(y_true, y_pred, 0.9), pinball_loss(y_true, y_pred, 0.1))


class TestForecastPipeline(unittest.TestCase):
    """
    A small end-to-end run. Kept deliberately tiny (12 events) so the suite
    stays fast; correctness of the leakage-sensitive plumbing is covered by
    `TestTargetConstruction` above using exact, hand-checkable expectations.
    """

    @classmethod
    def setUpClass(cls):
        cls.events = generate_dataset(n_events=12, seed=909)
        cls.df = build_dataset_features(cls.events)
        cls.dev_lookup = dev_lookup_from_events(cls.events)

    def test_fit_predict_evaluate_round_trip(self):
        pipeline = ForecastPipeline(
            self.df, self.dev_lookup, val_frac=0.25, test_frac=0.25,
            seed=1, verbose=False,
        )
        pipeline.fit_all()
        val = pipeline.evaluate("validation")

        for h in FORECAST_HORIZONS_MIN:
            key = f"{h:g}min"
            self.assertIn(key, val)
            m = val[key]
            self.assertGreater(m["n"], 0)
            self.assertTrue(np.isfinite(m["mae_median"]))
            self.assertGreaterEqual(m["interval_coverage"], 0.0)
            self.assertLessEqual(m["interval_coverage"], 1.0)
            for q in FORECAST_QUANTILES:
                self.assertIn(str(q), m["pinball_loss"])
                self.assertGreaterEqual(m["pinball_loss"][str(q)], 0.0)

    def test_quantile_predictions_are_monotone(self):
        pipeline = ForecastPipeline(
            self.df, self.dev_lookup, val_frac=0.25, test_frac=0.25,
            seed=1, verbose=False,
        )
        pipeline.fit_all()
        X, _ = pipeline.xy("validation", FORECAST_HORIZONS_MIN[0])
        cone = pipeline.predict_cone(X)
        for h in FORECAST_HORIZONS_MIN:
            lo = cone[h][FORECAST_QUANTILES[0]]
            mid = cone[h][FORECAST_QUANTILES[1]]
            hi = cone[h][FORECAST_QUANTILES[-1]]
            self.assertTrue(np.all(lo <= mid + 1e-9))
            self.assertTrue(np.all(mid <= hi + 1e-9))

    def test_save_writes_artifacts(self):
        pipeline = ForecastPipeline(
            self.df, self.dev_lookup, val_frac=0.25, test_frac=0.25,
            seed=1, verbose=False,
        )
        pipeline.fit_all()
        pipeline.evaluate("validation")

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            written = pipeline.save(Path(tmp))
            self.assertTrue(written["model"].exists())
            self.assertTrue(written["metrics"].exists())

            from gci.forecast import forecast_cone_from_bundle, load_forecast_bundle

            bundle = load_forecast_bundle(written["model"])
            X, _ = pipeline.xy("validation", FORECAST_HORIZONS_MIN[0])
            cone = forecast_cone_from_bundle(bundle, X)
            self.assertEqual(set(cone.keys()), set(FORECAST_HORIZONS_MIN))

    def test_uses_event_wise_split(self):
        """No row from a validation/test event may appear in train -- the
        same event-wise guarantee the risk model relies on."""
        pipeline = ForecastPipeline(
            self.df, self.dev_lookup, val_frac=0.25, test_frac=0.25,
            seed=1, verbose=False,
        )
        train_events = set(pipeline.frame("train")["event_id"].unique())
        val_events = set(pipeline.frame("validation")["event_id"].unique())
        test_events = set(pipeline.frame("test")["event_id"].unique())
        self.assertEqual(train_events & val_events, set())
        self.assertEqual(train_events & test_events, set())
        self.assertEqual(val_events & test_events, set())


if __name__ == "__main__":
    unittest.main()
