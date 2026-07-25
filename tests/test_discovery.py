"""Tests for lagged correlation discovery."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gci import config as C
from gci.config import Source
from gci.discovery import (
    CorrelationResult,
    _lagged_pairs,
    discover_correlations,
    novel_correlations,
    series_by_tag_from_dataset,
    series_by_tag_from_events,
)
from gci.events import generate_dataset, save_dataset, load_dataset


def _synthetic_series(n_events: int, n_steps: int, lag_steps: int, noise: float, seed: int):
    """cause is independent noise per event; effect = cause shifted forward by
    `lag_steps` (i.e. cause LEADS effect), plus small noise. So the true
    relationship is (cause, effect) at best_lag == lag_steps."""
    rng = np.random.default_rng(seed)
    cause_arrays, effect_arrays = [], []
    for _ in range(n_events):
        cause = rng.normal(0.0, 1.0, size=n_steps)
        effect = np.empty(n_steps)
        effect[:lag_steps] = rng.normal(0.0, 1.0, size=lag_steps)
        effect[lag_steps:] = cause[: n_steps - lag_steps] + rng.normal(0.0, noise, size=n_steps - lag_steps)
        cause_arrays.append(cause)
        effect_arrays.append(effect)
    return cause_arrays, effect_arrays


class TestLaggedPairs(unittest.TestCase):
    def test_respects_event_boundaries(self):
        c = [np.array([1.0, 2.0, 3.0, 4.0]), np.array([10.0, 20.0, 30.0])]
        e = [np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.5, 0.6, 0.7])]
        x, y = _lagged_pairs(c, e, lag_steps=1)
        # event 1: cause[:3] vs effect[1:4] ; event 2: cause[:2] vs effect[1:3]
        np.testing.assert_array_equal(x, np.array([1.0, 2.0, 3.0, 10.0, 20.0]))
        np.testing.assert_array_equal(y, np.array([0.2, 0.3, 0.4, 0.6, 0.7]))

    def test_zero_lag_is_full_alignment(self):
        c = [np.array([1.0, 2.0, 3.0])]
        e = [np.array([9.0, 8.0, 7.0])]
        x, y = _lagged_pairs(c, e, lag_steps=0)
        np.testing.assert_array_equal(x, c[0])
        np.testing.assert_array_equal(y, e[0])


class TestDiscoverCorrelations(unittest.TestCase):
    def test_finds_the_true_lag_and_sign(self):
        lag_steps = 6  # 0.5 min at DT_S=5
        cause, effect = _synthetic_series(
            n_events=20, n_steps=120, lag_steps=lag_steps, noise=0.05, seed=1
        )
        series = {"cause_tag": cause, "effect_tag": effect}
        results = discover_correlations(
            series, max_lag_min=2.0, min_abs_correlation=0.3, mi_max_samples=1000,
        )
        by_pair = {(r.cause, r.effect): r for r in results}
        self.assertIn(("cause_tag", "effect_tag"), by_pair)
        found = by_pair[("cause_tag", "effect_tag")]
        self.assertAlmostEqual(
            found.best_lag_min, lag_steps / C.STEPS_PER_MIN, places=6
        )
        self.assertGreater(found.correlation, 0.9)
        self.assertGreater(found.mutual_information, 0.0)

    def test_weak_relationship_excluded_by_threshold(self):
        rng = np.random.default_rng(2)
        n_events, n_steps = 10, 100
        a = [rng.normal(size=n_steps) for _ in range(n_events)]
        b = [rng.normal(size=n_steps) for _ in range(n_events)]
        results = discover_correlations(
            {"a": a, "b": b}, max_lag_min=1.0, min_abs_correlation=0.3,
        )
        self.assertEqual(results, [])

    def test_is_known_flag_matches_grades_known_loops(self):
        lag_steps = 3
        cause, effect = _synthetic_series(
            n_events=15, n_steps=100, lag_steps=lag_steps, noise=0.05, seed=3
        )
        # "stock_flow" -> "basis_weight" IS in KNOWN_LOOPS; a made-up pair
        # name is not.
        series = {"stock_flow": cause, "basis_weight": effect, "not_a_real_tag": effect}
        results = discover_correlations(
            series, max_lag_min=1.0, min_abs_correlation=0.3, mi_max_samples=500,
        )
        by_pair = {(r.cause, r.effect): r for r in results}
        known = by_pair[("stock_flow", "basis_weight")]
        self.assertTrue(known.is_known)
        self.assertEqual(known.source, Source.CORRELATION_DISCOVERY)

        novel = by_pair[("not_a_real_tag", "basis_weight")]
        self.assertFalse(novel.is_known)
        self.assertIn(novel, novel_correlations(results))
        self.assertNotIn(known, novel_correlations(results))

    def test_results_sorted_by_descending_abs_correlation(self):
        strong_c, strong_e = _synthetic_series(12, 100, 2, noise=0.02, seed=4)
        weak_c, weak_e = _synthetic_series(12, 100, 2, noise=0.9, seed=5)
        series = {
            "strong_cause": strong_c, "strong_effect": strong_e,
            "weak_cause": weak_c, "weak_effect": weak_e,
        }
        results = discover_correlations(series, max_lag_min=1.0, min_abs_correlation=0.05)
        mags = [abs(r.correlation) for r in results]
        self.assertEqual(mags, sorted(mags, reverse=True))

    def test_to_dict_round_trips(self):
        r = CorrelationResult(
            cause="a", effect="b", best_lag_min=0.5, correlation=0.6,
            mutual_information=0.2, n_samples=100, is_known=False,
        )
        d = r.to_dict()
        self.assertTrue(d["novel"])
        self.assertFalse(d["is_known"])
        self.assertEqual(d["cause"], "a")


class TestSeriesAdapters(unittest.TestCase):
    def test_series_by_tag_from_events_shapes(self):
        events = generate_dataset(n_events=6, seed=555)
        series = series_by_tag_from_events(events, tags=("basis_weight", "stock_flow"))
        self.assertEqual(set(series.keys()), {"basis_weight", "stock_flow"})
        self.assertEqual(len(series["basis_weight"]), 6)
        for arr, ev in zip(series["basis_weight"], events):
            self.assertEqual(len(arr), ev.n_steps)

    def test_series_by_tag_from_dataset_matches_events(self):
        events = generate_dataset(n_events=5, seed=556)
        from_events = series_by_tag_from_events(events, tags=("basis_weight", "moisture"))

        with tempfile.TemporaryDirectory() as tmp:
            save_dataset(events, Path(tmp))
            cube, tags, meta = load_dataset(Path(tmp))
            from_dataset = series_by_tag_from_dataset(
                cube, tags, meta, tags=("basis_weight", "moisture"),
            )

        for tag in ("basis_weight", "moisture"):
            for a, b in zip(from_events[tag], from_dataset[tag]):
                np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
