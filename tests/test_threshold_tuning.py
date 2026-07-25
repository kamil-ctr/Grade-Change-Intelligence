"""
Regression test for the threshold-tuning pass (`scripts/tune_risk_threshold.py`).

The pass swept the operating threshold from 0.05 to 0.95 on the validation
split and found that the currently deployed threshold already minimises
false-alarm-per-clean-event subject to detection staying within 0.02 of
itself and median warning time not dropping -- i.e. no change was warranted.
This test locks that finding in: if `models/threshold_sweep.csv` or
`risk_model.joblib`'s stored threshold is ever regenerated, the constrained
selection computed here must still land on the threshold actually deployed,
so a silent drift between "what the sweep says is best" and "what the code
uses" gets caught instead of shipped.
"""
import unittest
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DETECTION_TOLERANCE = 0.02


class TestThresholdTuning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sweep_path = MODELS_DIR / "threshold_sweep.csv"
        bundle_path = MODELS_DIR / "risk_model.joblib"
        if not sweep_path.exists() or not bundle_path.exists():
            raise unittest.SkipTest("models/ artefacts not present in this checkout")
        cls.sweep = pd.read_csv(sweep_path)
        cls.deployed_threshold = float(joblib.load(bundle_path)["threshold"])

    def test_sweep_covers_the_requested_grid(self):
        expected = [round(0.05 + 0.05 * i, 2) for i in range(19)]
        self.assertEqual(sorted(self.sweep["threshold"].round(2).tolist()), expected)

    def test_deployed_threshold_is_the_constrained_optimum(self):
        sweep = self.sweep
        current = sweep.iloc[(sweep["threshold"] - self.deployed_threshold).abs().idxmin()]

        min_detection = current["detection_rate"] - DETECTION_TOLERANCE
        min_warning = current["median_warning_min"]

        candidates = sweep[
            (sweep["detection_rate"] >= min_detection)
            & (sweep["median_warning_min"] >= min_warning)
        ]
        best = candidates.loc[candidates["false_alarm_event_rate"].idxmin()]

        self.assertAlmostEqual(
            float(best["threshold"]), self.deployed_threshold, places=6,
            msg=(
                "the sweep's constrained-optimum threshold no longer matches "
                "the threshold stored in risk_model.joblib -- either the model "
                "was retrained without re-running the threshold sweep, or the "
                "sweep was regenerated without updating the deployed threshold"
            ),
        )


if __name__ == "__main__":
    unittest.main()
