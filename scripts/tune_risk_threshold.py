"""
Threshold-tuning-only pass on the already-trained risk model: sweep the
probability threshold on the VALIDATION split, pick a candidate under an
explicit constraint, then check it against TEST exactly once. No retraining,
no feature changes, no hyperparameter changes -- this only ever calls
`_predict_proba` on the estimator already persisted in risk_model.joblib and
re-derives classification/early-warning metrics at different thresholds via
the existing `gci.ml.metrics.evaluate_model`.

Reuses the exact split (`event_wise_split_3way`, seed=42, val_frac=test_frac
=0.20 -- the same defaults `RiskModelPipeline` used at training time) against
the cached feature matrix at data/features.pkl, so validation and test here
are bit-identical to the sets the stored model was originally scored on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gci.features import load_features  # noqa: E402
from gci.ml.explain import _predict_proba  # noqa: E402
from gci.ml.metrics import evaluate_model  # noqa: E402
from gci.ml.splits import event_wise_split_3way  # noqa: E402

MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)
DETECTION_TOLERANCE = 0.02


def main() -> None:
    bundle = joblib.load(MODELS_DIR / "risk_model.joblib")
    estimator = bundle["estimator"]
    features = bundle["features"]
    persistence_samples = bundle["persistence_samples"]
    current_threshold = bundle["threshold"]

    df = load_features(DATA_DIR / "features.pkl")
    split = event_wise_split_3way(df, val_frac=0.20, test_frac=0.20, seed=42)
    val_df = split.frame(df, "val").reset_index(drop=True)
    test_df = split.frame(df, "test").reset_index(drop=True)

    proba_val = _predict_proba(estimator, val_df[features])
    proba_test = _predict_proba(estimator, test_df[features])

    rows = []
    for thr in THRESHOLDS:
        res = evaluate_model(val_df, proba_val, float(thr), persistence_samples=persistence_samples)
        clf, warn = res["classification"], res["early_warning"]
        rows.append({
            "threshold": float(thr),
            "precision": clf["precision"],
            "recall": clf["recall"],
            "f1": clf["f1"],
            "detection_rate": warn["detection_rate"],
            "median_warning_min": warn["median_warning_min"],
            "false_alarm_event_rate": warn["false_alarm_event_rate"],
            "n_warned_events": warn["n_warned_events"],
            "n_breaching_events": warn["n_breaching_events"],
            "n_clean_events": warn["n_clean_events"],
        })
    sweep = pd.DataFrame(rows)
    sweep.to_csv(MODELS_DIR / "threshold_sweep.csv", index=False)

    current_row = sweep.iloc[(sweep["threshold"] - current_threshold).abs().idxmin()]
    print("=== validation sweep (0.05..0.95 step 0.05) ===")
    print(sweep.to_string(index=False))
    print(f"\ncurrent threshold in use: {current_threshold}")
    print(f"current row (validation):\n{current_row}")

    min_detection = current_row["detection_rate"] - DETECTION_TOLERANCE
    min_warning = current_row["median_warning_min"]

    candidates = sweep[
        (sweep["detection_rate"] >= min_detection)
        & (sweep["median_warning_min"] >= min_warning)
        & (sweep["threshold"] != current_row["threshold"])
    ]

    print(f"\nconstraint: detection_rate >= {min_detection:.4f} AND median_warning_min >= {min_warning:.4f}")
    print(f"candidates passing constraint (excluding current):\n{candidates.to_string(index=False)}")

    if candidates.empty:
        print("\nNO IMPROVEMENT FOUND on validation -- keeping current threshold. Stopping here.")
        return

    best = candidates.loc[candidates["false_alarm_event_rate"].idxmin()]
    if best["false_alarm_event_rate"] >= current_row["false_alarm_event_rate"]:
        print("\nNO IMPROVEMENT FOUND (no candidate beats current false-alarm rate) -- keeping current threshold. Stopping here.")
        return

    chosen_threshold = float(best["threshold"])
    print(f"\ncandidate threshold from validation: {chosen_threshold}")
    print(f"validation false_alarm_event_rate: {current_row['false_alarm_event_rate']:.4f} -> {best['false_alarm_event_rate']:.4f}")

    # --- one-shot test-set check -------------------------------------------------
    current_test = evaluate_model(test_df, proba_test, float(current_threshold), persistence_samples=persistence_samples)
    candidate_test = evaluate_model(test_df, proba_test, chosen_threshold, persistence_samples=persistence_samples)

    def _row(res, thr):
        return {
            "threshold": thr,
            "pr_auc": res["classification"]["pr_auc"],
            "detection_rate": res["early_warning"]["detection_rate"],
            "median_warning_min": res["early_warning"]["median_warning_min"],
            "false_alarm_event_rate": res["early_warning"]["false_alarm_event_rate"],
        }

    before = _row(current_test, current_threshold)
    after = _row(candidate_test, chosen_threshold)
    print("\n=== TEST set, one-shot ===")
    print("before:", json.dumps(before, indent=2))
    print("after: ", json.dumps(after, indent=2))

    improved = (
        after["false_alarm_event_rate"] < before["false_alarm_event_rate"]
        and after["detection_rate"] >= before["detection_rate"] - DETECTION_TOLERANCE
        and after["median_warning_min"] >= before["median_warning_min"]
    )
    if not improved:
        print("\nTEST SET DID NOT CONFIRM the validation improvement -- REVERTING, keeping current threshold.")
        return

    print(f"\nTEST SET CONFIRMS improvement. New threshold: {chosen_threshold}")


if __name__ == "__main__":
    main()
