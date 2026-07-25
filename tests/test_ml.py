"""
Tests for the ML pipeline.

The tests that matter most here are the leakage guards. A model that looks
excellent because of a subtle split or threshold leak is worse than no model at
all, so the properties are asserted rather than trusted:

* splits are disjoint by event and cover every row
* threshold selection and metrics never see the future
* alarm confirmation cannot span an event boundary
* early-warning time is measured only from alarms raised *before* a breach and
  *while still in spec*
"""
import unittest

import numpy as np
import pandas as pd

from gci import config as C
from gci.control import ControlPlan
from gci.events import generate_dataset
from gci.features import build_dataset_features, downcast_features
from gci.ml.explain import (
    ShapResult,
    _predict_proba,
    _shap_via_linear,
    compute_shap,
    native_importance,
)
from gci.ml.metrics import (
    classification_metrics,
    confirm_alarms,
    early_warning_analysis,
    evaluate_model,
    pick_threshold,
)
from gci.ml.pipeline import RiskModelPipeline
from gci.ml.registry import MODEL_REGISTRY, available_models, registry_report
from gci.ml.splits import event_wise_split_3way, split_summary


class TestRegistry(unittest.TestCase):
    def test_required_models_are_registered(self):
        for name in ("lightgbm", "xgboost", "random_forest"):
            self.assertIn(name, MODEL_REGISTRY)

    def test_at_least_one_model_is_always_available(self):
        """Must hold on a bare scikit-learn install."""
        self.assertGreaterEqual(len(available_models()), 1)
        self.assertTrue(MODEL_REGISTRY["random_forest"].available)
        self.assertTrue(MODEL_REGISTRY["hist_gradient_boosting"].available)

    def test_unavailable_model_raises_on_build_not_import(self):
        spec = MODEL_REGISTRY["lightgbm"]
        if spec.available:
            self.assertIsNotNone(spec.build())
        else:
            with self.assertRaises(ImportError):
                spec.build()

    def test_registry_report_is_serialisable(self):
        import json

        json.dumps(registry_report())


class TestSplits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = generate_dataset(n_events=24, seed=555)
        cls.df = build_dataset_features(cls.events)

    def test_three_way_split_is_disjoint(self):
        split = event_wise_split_3way(self.df, seed=1)
        a = set(split.train_events.tolist())
        b = set(split.val_events.tolist())
        c = set(split.test_events.tolist())
        self.assertEqual(a & b, set())
        self.assertEqual(a & c, set())
        self.assertEqual(b & c, set())

    def test_split_covers_every_row_exactly_once(self):
        split = event_wise_split_3way(self.df, seed=1)
        total = sum(
            int(split.mask(self.df, w).sum()) for w in ("train", "val", "test")
        )
        self.assertEqual(total, len(self.df))

    def test_no_event_appears_in_two_frames(self):
        split = event_wise_split_3way(self.df, seed=1)
        frames = {w: split.frame(self.df, w) for w in ("train", "val", "test")}
        ids = {w: set(f["event_id"]) for w, f in frames.items()}
        self.assertEqual(ids["train"] & ids["val"], set())
        self.assertEqual(ids["val"] & ids["test"], set())

    def test_aliases_resolve(self):
        split = event_wise_split_3way(self.df, seed=1)
        np.testing.assert_array_equal(
            split.events_for("val"), split.events_for("validation")
        )
        with self.assertRaises(KeyError):
            split.events_for("nonsense")

    def test_deterministic_given_seed(self):
        a = event_wise_split_3way(self.df, seed=7)
        b = event_wise_split_3way(self.df, seed=7)
        np.testing.assert_array_equal(a.test_events, b.test_events)

    def test_different_seed_gives_different_split(self):
        a = event_wise_split_3way(self.df, seed=1)
        b = event_wise_split_3way(self.df, seed=99)
        self.assertNotEqual(
            set(a.test_events.tolist()), set(b.test_events.tolist())
        )

    def test_overlap_is_rejected(self):
        from gci.ml.splits import EventSplit

        with self.assertRaises(ValueError):
            EventSplit(
                train_events=np.array([1, 2]), val_events=np.array([2, 3]),
                test_events=np.array([4]), seed=0,
            )

    def test_summary_reports_all_splits(self):
        split = event_wise_split_3way(self.df, seed=1)
        table = split_summary(self.df, split)
        self.assertEqual(set(table["split"]), {"train", "val", "test"})
        self.assertTrue((table["rows"] > 0).all())


class TestAlarmConfirmation(unittest.TestCase):
    def test_on_delay_requires_consecutive_samples(self):
        raw = np.array([0, 1, 0, 1, 1, 1, 0], dtype=bool)
        out = confirm_alarms(raw, persistence_samples=3)
        # Only index 5 has three consecutive True ending there
        np.testing.assert_array_equal(
            out, np.array([0, 0, 0, 0, 0, 1, 0], dtype=bool)
        )

    def test_delay_of_one_is_a_passthrough(self):
        raw = np.array([0, 1, 1, 0], dtype=bool)
        np.testing.assert_array_equal(confirm_alarms(raw, 1), raw)

    def test_persistence_never_spans_event_boundary(self):
        """Two events each with two trailing alarms must not combine."""
        raw = np.array([0, 1, 1, 1, 1, 0], dtype=bool)
        group = np.array([1, 1, 1, 2, 2, 2])
        out = confirm_alarms(raw, persistence_samples=3, group=group)
        self.assertTrue(out[3] == False)  # noqa: E712 - only 1 sample into event 2
        self.assertTrue(out[4] == False)  # noqa: E712 - only 2 samples in
        # Event 1 has exactly three consecutive at index 3 if it leaked; it must not
        self.assertEqual(int(out.sum()), 0)

    def test_reduces_or_preserves_alarm_count(self):
        rng = np.random.default_rng(0)
        raw = rng.random(500) > 0.6
        for delay in (1, 2, 5, 10):
            self.assertLessEqual(
                int(confirm_alarms(raw, delay).sum()), int(raw.sum())
            )


class TestThresholdAndMetrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = (rng.random(1000) > 0.7).astype(int)
        # Informative but imperfect score
        self.p = np.clip(
            0.25 + 0.5 * self.y + rng.normal(0, 0.22, 1000), 0.0, 1.0
        )

    def test_threshold_is_in_range(self):
        thr, info = pick_threshold(self.y, self.p)
        self.assertGreaterEqual(thr, 0.0)
        self.assertLessEqual(thr, 1.0)
        self.assertIn("criterion", info)

    def test_all_criteria_supported(self):
        for criterion in ("f1", "fbeta", "max_fpr"):
            thr, _ = pick_threshold(self.y, self.p, criterion=criterion)
            self.assertTrue(np.isfinite(thr))

    def test_fbeta_is_more_conservative_than_f1(self):
        """Precision-weighted tuning should not pick a lower threshold."""
        f1, _ = pick_threshold(self.y, self.p, criterion="f1")
        fb, _ = pick_threshold(self.y, self.p, criterion="fbeta", beta=0.3)
        self.assertGreaterEqual(fb, f1 - 1e-9)

    def test_degenerate_labels_do_not_crash(self):
        thr, info = pick_threshold(np.zeros(50, dtype=int), np.full(50, 0.4))
        self.assertEqual(thr, 0.5)
        self.assertIn("note", info)

    def test_all_required_metrics_present(self):
        m = classification_metrics(self.y, self.p, 0.5)
        for key in (
            "pr_auc", "precision", "recall", "f1", "false_positive_rate",
            "tp", "fp", "tn", "fn",
        ):
            self.assertIn(key, m)

    def test_metrics_are_internally_consistent(self):
        m = classification_metrics(self.y, self.p, 0.5)
        tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]
        self.assertAlmostEqual(m["precision"], tp / (tp + fp), places=9)
        self.assertAlmostEqual(m["recall"], tp / (tp + fn), places=9)
        self.assertAlmostEqual(m["false_positive_rate"], fp / (fp + tn), places=9)
        self.assertEqual(tp + fp + fn + tn, m["n_samples"])

    def test_perfect_predictions_score_perfectly(self):
        m = classification_metrics(self.y, self.y.astype(float), 0.5)
        self.assertAlmostEqual(m["precision"], 1.0)
        self.assertAlmostEqual(m["recall"], 1.0)
        self.assertAlmostEqual(m["false_positive_rate"], 0.0)


class TestEarlyWarning(unittest.TestCase):
    def _frame(self, off_spec, proba, event_id=1):
        n = len(off_spec)
        return pd.DataFrame(
            {
                "event_id": np.full(n, event_id),
                "t_min": np.arange(n) * C.DT_S / 60.0,
                "currently_off_spec": np.asarray(off_spec, dtype=float),
                "y_breach": np.zeros(n, dtype=int),
            }
        ), np.asarray(proba, dtype=float)

    def test_warning_time_is_gap_from_alarm_to_breach(self):
        off = [0] * 20 + [1] * 10
        p = [0.0] * 8 + [0.9] * 12 + [0.9] * 10
        df, proba = self._frame(off, p)
        out = early_warning_analysis(df, proba, threshold=0.5)
        self.assertEqual(out["n_breaching_events"], 1)
        self.assertEqual(out["n_warned_events"], 1)
        # alarm at index 8, breach at index 20 -> 12 samples
        self.assertAlmostEqual(
            out["median_warning_min"], 12 * C.DT_S / 60.0, places=6
        )

    def test_alarm_after_breach_does_not_count_as_a_warning(self):
        off = [0] * 10 + [1] * 10
        p = [0.0] * 10 + [0.99] * 10       # only alarms once already off-spec
        df, proba = self._frame(off, p)
        out = early_warning_analysis(df, proba, threshold=0.5)
        self.assertEqual(out["n_warned_events"], 0)
        self.assertEqual(out["detection_rate"], 0.0)

    def test_clean_event_with_alarm_is_a_false_alarm(self):
        df, proba = self._frame([0] * 20, [0.0] * 10 + [0.9] * 10)
        out = early_warning_analysis(df, proba, threshold=0.5)
        self.assertEqual(out["n_breaching_events"], 0)
        self.assertEqual(out["false_alarm_event_rate"], 1.0)

    def test_clean_event_without_alarm_is_not_a_false_alarm(self):
        df, proba = self._frame([0] * 20, [0.1] * 20)
        out = early_warning_analysis(df, proba, threshold=0.5)
        self.assertEqual(out["false_alarm_event_rate"], 0.0)

    def test_on_delay_cannot_increase_false_alarms(self):
        rng = np.random.default_rng(3)
        df, proba = self._frame(
            [0] * 300, np.clip(rng.normal(0.45, 0.2, 300), 0, 1)
        )
        base = early_warning_analysis(df, proba, 0.5, persistence_samples=1)
        delayed = early_warning_analysis(df, proba, 0.5, persistence_samples=6)
        self.assertLessEqual(
            delayed["false_alarm_event_rate"], base["false_alarm_event_rate"]
        )

    def test_length_mismatch_is_rejected(self):
        df, proba = self._frame([0] * 10, [0.1] * 10)
        with self.assertRaises(ValueError):
            early_warning_analysis(df, proba[:5], 0.5)


class TestExplainability(unittest.TestCase):
    def test_linear_shap_is_exact_and_additive(self):
        """
        The additivity property is the definition of a correct SHAP
        decomposition: base + sum(phi) must reproduce the model output.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(200, 6)),
                         columns=[f"f{i}" for i in range(6)])
        y = (X["f0"] + 0.5 * X["f1"] > 0).astype(int)
        model = Pipeline(
            [("scale", StandardScaler()),
             ("clf", LogisticRegression(max_iter=500))]
        ).fit(X, y)

        result = _shap_via_linear(model, X)
        self.assertIsNotNone(result)
        self.assertTrue(result.exact)

        logits = model.decision_function(X)
        reconstructed = result.base_value + result.values.sum(axis=1)
        np.testing.assert_allclose(reconstructed, logits, atol=1e-8)

    def test_shap_available_for_every_registered_model(self):
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.normal(size=(150, 5)),
                         columns=[f"f{i}" for i in range(5)])
        y = (X["f0"] > 0).astype(int)
        for name, spec in available_models().items():
            with self.subTest(model=name):
                model = spec.build()
                model.fit(X, y)
                result = compute_shap(model, X, max_rows=60)
                self.assertEqual(result.values.shape[1], X.shape[1])
                self.assertTrue(
                    result.exact,
                    msg=f"{name} fell back to an approximate explainer",
                )

    def test_local_explanation_is_ranked_by_magnitude(self):
        result = ShapResult(
            values=np.array([[0.1, -0.5, 0.3]]), base_value=0.0,
            method="test", exact=True, feature_names=["a", "b", "c"],
        )
        local = result.local(0, top_k=3)
        self.assertEqual([c["feature"] for c in local], ["b", "c", "a"])
        self.assertEqual(local[0]["direction"], "reduces risk")

    def test_mean_abs_shap_is_sorted(self):
        result = ShapResult(
            values=np.array([[0.1, -0.5, 0.3], [0.2, -0.4, 0.1]]),
            base_value=0.0, method="test", exact=True,
            feature_names=["a", "b", "c"],
        )
        table = result.mean_abs()
        self.assertEqual(table.iloc[0]["feature"], "b")
        self.assertTrue(
            (table["mean_abs_shap"].diff().dropna() <= 1e-12).all()
        )


class TestPipelineEndToEnd(unittest.TestCase):
    """A small but complete run, to catch integration breakage."""

    @classmethod
    def setUpClass(cls):
        events = generate_dataset(n_events=30, seed=2024)
        cls.df = downcast_features(build_dataset_features(events))

    def test_pipeline_trains_selects_and_explains(self):
        pipe = RiskModelPipeline(
            self.df, models=["random_forest", "logistic_regression"],
            verbose=False, seed=3,
        )
        pipe.fit_all()
        self.assertEqual(len(pipe.trained), 2)
        self.assertIsNotNone(pipe.best)

        for model in pipe.trained.values():
            for split in ("validation", "test"):
                clf = model.results[split]["classification"]
                self.assertIn("pr_auc", clf)
                self.assertGreaterEqual(clf["pr_auc"], 0.0)
                self.assertLessEqual(clf["pr_auc"], 1.0)

        pipe.explain_best(n_repeats=1, shap_max_rows=100)
        self.assertIsNotNone(pipe.shap)
        self.assertIsNotNone(pipe.importance)
        self.assertGreater(len(pipe.importance), 0)

    def test_report_and_artifacts_are_written(self):
        import tempfile
        from pathlib import Path

        pipe = RiskModelPipeline(
            self.df, models=["logistic_regression"], verbose=False, seed=3
        )
        pipe.fit_all()
        pipe.explain_best(n_repeats=1, shap_max_rows=100)

        with tempfile.TemporaryDirectory() as tmp:
            written = pipe.save(Path(tmp))
            for key in (
                "metrics", "confusion_matrix", "feature_importance",
                "shap", "report", "warning_detail",
            ):
                self.assertIn(key, written)
                self.assertTrue(written[key].exists())
            report = written["report"].read_text()
            self.assertIn("Leakage controls", report)
            self.assertIn("PR-AUC", report)

    def test_unknown_model_is_skipped_not_fatal(self):
        pipe = RiskModelPipeline(
            self.df, models=["logistic_regression", "does_not_exist"],
            verbose=False, seed=3,
        )
        self.assertIn("does_not_exist", pipe.skipped)
        pipe.fit_all()
        self.assertIn("logistic_regression", pipe.trained)

    def test_checkpoint_round_trip(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            first = RiskModelPipeline(
                self.df, models=["logistic_regression"], verbose=False, seed=3
            )
            first.fit_all(checkpoint_dir=ckpt)

            second = RiskModelPipeline(
                self.df, models=["logistic_regression"], verbose=False, seed=3
            )
            second.fit_all(checkpoint_dir=ckpt, resume=True)
            # Restored, so no time was spent refitting
            self.assertEqual(
                second.trained["logistic_regression"].results[
                    "validation"
                ]["classification"]["pr_auc"],
                first.trained["logistic_regression"].results[
                    "validation"
                ]["classification"]["pr_auc"],
            )


if __name__ == "__main__":
    unittest.main()
