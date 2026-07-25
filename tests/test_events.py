"""Tests for fault injection, event simulation and labelling."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gci import config as C
from gci.control import ControlPlan
from gci.events import (
    bw_deviation_pct,
    compute_settle_time,
    generate_dataset,
    label_event,
    load_dataset,
    run_event,
    save_dataset,
    validate_dataset,
)
from gci.faults import (
    DV_BOUNDS,
    FAULT_CATALOG,
    apply_faults,
    baseline_disturbances,
    sample_faults,
)
from gci.grades import CV_TAGS, DV_TAGS, MV_TAGS, get_grade
from gci.twin import DV_NOMINAL, PaperMachineTwin


class TestDisturbances(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.n = 360

    def test_baseline_covers_every_tag_and_stays_in_bounds(self):
        dv = baseline_disturbances(self.rng, self.n)
        for tag in DV_TAGS:
            self.assertIn(tag, dv)
            self.assertEqual(dv[tag].shape[0], self.n)
            lo, hi = DV_BOUNDS[tag]
            self.assertGreaterEqual(dv[tag].min(), lo - 1e-9)
            self.assertLessEqual(dv[tag].max(), hi + 1e-9)

    def test_baseline_is_near_nominal_on_average(self):
        dv = baseline_disturbances(np.random.default_rng(7), 2000)
        for tag in DV_TAGS:
            rel = abs(dv[tag].mean() - DV_NOMINAL[tag]) / max(
                abs(DV_NOMINAL[tag]), 1e-9
            )
            self.assertLess(rel, 0.10, msg=f"{tag} baseline drifted too far")

    def test_drift_is_correlated_not_white(self):
        """OU drift must have high lag-1 autocorrelation; white noise would not."""
        dv = baseline_disturbances(np.random.default_rng(3), 2000)
        x = dv["steam_header_kpa"]
        x = x - x.mean()
        acf1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        self.assertGreater(acf1, 0.9)


class TestFaults(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.n = 360

    def test_every_catalog_entry_produces_a_finite_profile(self):
        for spec in FAULT_CATALOG:
            with self.subTest(fault=spec.code):
                fault = spec.sample(self.rng, 30.0, 5.0)
                profile = fault.profile(self.n)
                self.assertEqual(profile.shape[0], self.n)
                self.assertTrue(np.all(np.isfinite(profile)))
                self.assertGreater(np.abs(profile).max(), 0.0)

    def test_nothing_happens_before_onset(self):
        for spec in FAULT_CATALOG:
            with self.subTest(fault=spec.code):
                fault = spec.sample(self.rng, 30.0, 5.0)
                profile = fault.profile(self.n)
                onset_idx = int(fault.onset_min * 60.0 / C.DT_S)
                if onset_idx > 1:
                    self.assertAlmostEqual(
                        float(np.abs(profile[: onset_idx - 1]).max()), 0.0,
                        delta=1e-9,
                    )

    def test_fault_moves_its_target_tag(self):
        dv = baseline_disturbances(self.rng, self.n)
        for spec in FAULT_CATALOG:
            if spec.target_tag not in DV_TAGS:
                continue
            with self.subTest(fault=spec.code):
                fault = spec.sample(self.rng, 30.0, 5.0)
                out, _ = apply_faults(dv, [fault])
                self.assertGreater(
                    float(np.abs(out[spec.target_tag] - dv[spec.target_tag]).max()),
                    0.0,
                )

    def test_actuator_faults_are_returned_as_overrides(self):
        dv = baseline_disturbances(self.rng, self.n)
        aid_fault = [s for s in FAULT_CATALOG if s.target_tag in MV_TAGS][0]
        fault = aid_fault.sample(self.rng, 30.0, 5.0)
        _, overrides = apply_faults(dv, [fault])
        self.assertIn(fault.target_tag, overrides)

    def test_no_faults_leaves_disturbances_untouched(self):
        dv = baseline_disturbances(self.rng, self.n)
        out, overrides = apply_faults(dv, [])
        self.assertEqual(overrides, {})
        for tag in dv:
            np.testing.assert_allclose(out[tag], dv[tag])

    def test_sampler_can_return_none_and_never_duplicates(self):
        rng = np.random.default_rng(11)
        saw_empty = False
        for _ in range(300):
            faults = sample_faults(rng, 30.0, 5.0)
            codes = [f.code for f in faults]
            self.assertEqual(len(codes), len(set(codes)))
            saw_empty = saw_empty or not codes
        self.assertTrue(saw_empty)


class TestLabelling(unittest.TestCase):
    def test_deviation_is_relative_to_setpoint(self):
        bw = np.array([102.0, 98.0])
        sp = np.array([100.0, 100.0])
        np.testing.assert_allclose(bw_deviation_pct(bw, sp), [2.0, -2.0])

    def test_settle_time_detects_a_clean_settle(self):
        n = 360
        bw = np.full(n, 100.0)
        bw[: int(8 * 60 / C.DT_S)] = 90.0          # off target for 8 minutes
        settle = compute_settle_time(bw, 100.0, ramp_start_min=5.0)
        self.assertAlmostEqual(settle, 3.0, delta=0.2)

    def test_settle_time_is_nan_when_never_stable(self):
        n = 360
        bw = 100.0 + 20.0 * np.sin(np.linspace(0, 20, n))
        self.assertTrue(np.isnan(compute_settle_time(bw, 100.0, 5.0)))

    def test_labels_agree_with_the_trajectory(self):
        n = 200
        series = {
            "basis_weight": np.full(n, 100.0),
            "basis_weight_sp": np.full(n, 100.0),
            "moisture": np.full(n, 6.0),
            "moisture_sp": np.full(n, 6.0),
            "ash": np.full(n, 12.0),
            "ash_sp": np.full(n, 12.0),
        }
        labels = label_event(series, get_grade("BRD-120"), 5.0)
        self.assertEqual(labels["off_spec"], 0.0)
        self.assertAlmostEqual(labels["off_spec_minutes"], 0.0)

        series["basis_weight"][50:70] = 110.0    # 10% excursion
        labels = label_event(series, get_grade("BRD-120"), 5.0)
        self.assertEqual(labels["off_spec"], 1.0)
        self.assertGreater(labels["off_spec_minutes"], 0.0)
        self.assertAlmostEqual(labels["max_abs_dev_pct"], 10.0, delta=0.01)


class TestEventSimulation(unittest.TestCase):
    def test_event_has_complete_finite_series(self):
        twin = PaperMachineTwin(seed=4)
        ev = run_event(twin, "NP-45", "LWC-52", ControlPlan(ramp_min=6.0), [])
        for tag in MV_TAGS + CV_TAGS + DV_TAGS:
            self.assertIn(tag, ev.series)
            self.assertTrue(np.all(np.isfinite(ev.series[tag])), msg=tag)

    def test_event_starts_on_the_old_grade(self):
        """A transition must not begin already off-spec."""
        twin = PaperMachineTwin(seed=4)
        ev = run_event(
            twin, "NP-45", "LWC-52", ControlPlan(ramp_min=6.0), [], add_noise=False
        )
        self.assertAlmostEqual(
            ev.series["basis_weight"][0], get_grade("NP-45").basis_weight, delta=0.3
        )

    def test_event_ends_on_the_new_grade(self):
        twin = PaperMachineTwin(seed=4)
        ev = run_event(
            twin, "NP-45", "LWC-52", ControlPlan(ramp_min=6.0), [], add_noise=False
        )
        self.assertAlmostEqual(
            ev.series["basis_weight"][-1], get_grade("LWC-52").basis_weight, delta=1.5
        )

    def test_a_rushed_ramp_is_worse_than_a_generous_one(self):
        """The feasibility floor must actually matter."""
        rushed = run_event(
            PaperMachineTwin(seed=9), "NP-45", "BRD-120",
            ControlPlan(ramp_min=4.0), [], seed=9, add_noise=False,
        )
        generous = run_event(
            PaperMachineTwin(seed=9), "NP-45", "BRD-120",
            ControlPlan(ramp_min=16.0), [], seed=9, add_noise=False,
        )
        self.assertGreater(
            rushed.labels["max_abs_dev_pct"], generous.labels["max_abs_dev_pct"]
        )

    def test_context_reports_feasibility(self):
        ev = run_event(
            PaperMachineTwin(seed=2), "NP-45", "BRD-150",
            ControlPlan(ramp_min=3.0), [], seed=2,
        )
        self.assertEqual(ev.context["ramp_is_feasible"], 0.0)
        self.assertGreater(ev.context["ramp_deficit_min"], 0.0)

    def test_reproducible_from_seed(self):
        a = run_event(PaperMachineTwin(seed=77), "SC-56", "WFU-70",
                      ControlPlan(ramp_min=8.0), [], seed=77)
        b = run_event(PaperMachineTwin(seed=77), "SC-56", "WFU-70",
                      ControlPlan(ramp_min=8.0), [], seed=77)
        np.testing.assert_allclose(
            a.series["basis_weight"], b.series["basis_weight"]
        )


class TestDatasetGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = generate_dataset(n_events=24, seed=1234)

    def test_validation_passes(self):
        report = validate_dataset(self.events)
        self.assertTrue(report["ok"], msg=str(report["issues"]))

    def test_both_classes_present(self):
        outcomes = {ev.labels["off_spec"] for ev in self.events}
        self.assertEqual(outcomes, {0.0, 1.0})

    def test_transitions_are_varied(self):
        pairs = {(ev.from_grade, ev.to_grade) for ev in self.events}
        self.assertGreater(len(pairs), 8)
        for a, b in pairs:
            self.assertNotEqual(a, b)

    def test_round_trip_through_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_dataset(self.events, Path(tmp))
            self.assertTrue(paths["series"].exists())
            cube, tags, meta = load_dataset(Path(tmp))
            self.assertEqual(cube.shape[0], len(self.events))
            self.assertEqual(meta["n_events"], len(self.events))
            np.testing.assert_allclose(
                cube[0, :, tags.index("basis_weight")],
                self.events[0].series["basis_weight"].astype(np.float32),
                rtol=1e-5,
            )


if __name__ == "__main__":
    unittest.main()
