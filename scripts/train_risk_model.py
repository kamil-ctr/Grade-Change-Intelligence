#!/usr/bin/env python3
"""
Train, compare and persist the off-spec risk model.

Usage:
    python scripts/train_risk_model.py [--events 300] [--seed 42]

Trains every available model in the registry, selects the best on validation
PR-AUC, scores it once on the held-out test split, explains it, and writes all
artefacts to models/.

Artefacts written:
    risk_model.joblib          fitted estimator + feature list + threshold
    metrics.json               every model, every split, full environment record
    comparison_validation.csv  side-by-side model comparison
    comparison_test.csv
    confusion_matrix.json      per model, per split
    feature_importance.csv     consensus of native / permutation / SHAP
    shap_explanations.json     mean |SHAP| plus worked local examples
    warning_detail.json        per-event early-warning outcomes
    evaluation_report.md       human-readable report
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gci.events import generate_dataset  # noqa: E402
from gci.features import (  # noqa: E402
    build_dataset_features,
    downcast_features,
    load_features,
    save_features,
)
from gci.ml.pipeline import RiskModelPipeline  # noqa: E402
from gci.ml.registry import registry_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=300)
    parser.add_argument("--data-seed", type=int, default=20260725)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "models")
    parser.add_argument(
        "--threshold-criterion", default="f1", choices=["f1", "fbeta", "max_fpr"]
    )
    parser.add_argument("--perm-repeats", type=int, default=3)
    parser.add_argument(
        "--regenerate", action="store_true",
        help="ignore the feature cache and rebuild the corpus",
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="restrict to these registry models (default: all available)",
    )
    parser.add_argument(
        "--min-detection", type=float, default=0.80,
        help="event detection rate floor for operating-point tuning",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="retrain every model instead of restoring checkpoints",
    )
    parser.add_argument(
        "--skip-explain", action="store_true",
        help="skip SHAP and permutation importance (run explain_risk_model.py "
             "afterwards instead)",
    )
    args = parser.parse_args()

    print("Model registry:")
    for entry in registry_report():
        mark = "available" if entry["available"] else (
            f"unavailable ({', '.join(entry['missing'])})"
        )
        print(f"  {entry['label']:32s} {mark}")
    print()

    cache = ROOT / "data" / "features.pkl"
    if cache.exists() and not args.regenerate:
        print(f"Loading cached features from {cache.relative_to(ROOT)} ...")
        t0 = time.perf_counter()
        df = load_features(cache)
        print(f"  {len(df):,} rows x {df.shape[1]} columns in "
              f"{time.perf_counter() - t0:.1f}s")
    else:
        print(f"Generating {args.events} events (seed={args.data_seed}) ...")
        t0 = time.perf_counter()
        events = generate_dataset(n_events=args.events, seed=args.data_seed)
        print(f"  {time.perf_counter() - t0:.1f}s")

        print("Building features ...")
        t0 = time.perf_counter()
        df = downcast_features(build_dataset_features(events))
        print(f"  {len(df):,} rows x {df.shape[1]} columns in "
              f"{time.perf_counter() - t0:.1f}s")
        save_features(df, cache)
        print(f"  cached to {cache.relative_to(ROOT)}")
    print()

    pipeline = RiskModelPipeline(
        df, seed=args.seed, threshold_criterion=args.threshold_criterion,
        models=args.models,
    )
    print("Split composition:")
    print(pipeline.split_table.to_string(index=False))
    print()

    pipeline.fit_all(
        checkpoint_dir=args.out / "checkpoints", resume=not args.no_resume
    )
    print("\nTuning operating point per model (validation only):")
    pipeline.select_by_operating_point(min_detection_rate=args.min_detection)

    if not args.skip_explain:
        pipeline.explain_best(n_repeats=args.perm_repeats)

    print("\nValidation comparison:")
    print(pipeline.comparison_table("validation").to_string(index=False))
    print("\nTest comparison (winner is the only honest number here):")
    print(pipeline.comparison_table("test").to_string(index=False))

    written = pipeline.save(args.out)
    print("\nArtefacts:")
    for name, path in written.items():
        print(f"  {name:24s} {path.name} "
              f"({path.stat().st_size / 1024:.1f} KB)")

    best = pipeline.best
    assert best is not None
    test_clf = best.results["test"]["classification"]
    test_warn = best.results["test"]["early_warning"]
    print(
        f"\nHEADLINE  {best.label}: "
        f"PR-AUC {test_clf['pr_auc']:.3f}, "
        f"recall {test_clf['recall']:.3f}, "
        f"FPR {test_clf['false_positive_rate']:.3f}, "
        f"median warning {test_warn['median_warning_min']:.1f} min "
        f"(test split)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
