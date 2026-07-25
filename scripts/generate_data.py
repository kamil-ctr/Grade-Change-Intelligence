#!/usr/bin/env python3
"""
Generate, validate and featurise the grade-change corpus.

Usage:
    python scripts/generate_data.py [--events 300] [--seed 20260725]

Writes to data/:
    events_series.npz   trajectory cube (n_events, n_steps, n_tags)
    events_meta.json    per-event metadata, plan, faults and labels
    features.parquet    per-sample model-ready feature table (or .csv.gz)
    validation.json     dataset health report

The corpus is regenerated rather than shipped: it is fully determined by the
seed, so `--seed 20260725` reproduces the exact dataset used for the reported
results while keeping the repository small.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gci.events import (  # noqa: E402
    generate_dataset,
    save_dataset,
    validate_dataset,
)
from gci.features import build_dataset_features, feature_columns  # noqa: E402


def write_features(df, out_dir: Path) -> Path:
    """Parquet if an engine is available, otherwise gzipped CSV."""
    try:
        path = out_dir / "features.parquet"
        df.to_parquet(path, index=False)
        return path
    except Exception:
        path = out_dir / "features.csv.gz"
        df.to_csv(path, index=False, compression="gzip")
        return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--save-features",
        action="store_true",
        help="also persist the feature table (large; rebuilt in seconds "
             "from the trajectories, so off by default)",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.events} grade-change events (seed={args.seed}) ...")
    t0 = time.perf_counter()
    events = generate_dataset(
        n_events=args.events, seed=args.seed, progress=True
    )
    gen_s = time.perf_counter() - t0
    print(f"  done in {gen_s:.1f}s ({gen_s / args.events * 1000:.0f} ms/event)")

    print("Validating ...")
    report = validate_dataset(events)
    for key in (
        "n_events", "off_spec_rate", "settled_rate", "median_settle_min",
        "mean_off_spec_minutes", "distinct_grade_pairs",
    ):
        value = report[key]
        print(f"  {key:24s} {value:.3f}" if isinstance(value, float)
              else f"  {key:24s} {value}")
    if not report["ok"]:
        print("  ISSUES:")
        for issue in report["issues"]:
            print(f"    - {issue}")
    else:
        print("  no issues")

    print("Saving trajectories ...")
    paths = save_dataset(events, args.out)
    for name, path in paths.items():
        print(f"  {name:8s} {path.name}  ({path.stat().st_size / 1e6:.2f} MB)")

    # Write the health report as soon as it exists, before the slowest step,
    # so an interrupted run still leaves a usable artefact behind.
    report["generation_seconds"] = gen_s
    validation_path = args.out / "validation.json"
    validation_path.write_text(json.dumps(report, indent=2, default=float))

    print("Building features ...")
    t0 = time.perf_counter()
    df = build_dataset_features(events, progress=True)
    feat_s = time.perf_counter() - t0
    cols = feature_columns(df)
    print(
        f"  {len(df):,} samples x {len(cols)} features in {feat_s:.1f}s"
        f"  |  positive rate {df['y_breach'].mean():.1%}"
    )

    # Always cache the feature frame: training loads it instead of spending
    # 25 seconds regenerating the corpus on every run.
    from gci.features import save_features

    cache_path = args.out / "features.pkl"
    save_features(df, cache_path)
    print(f"  cached {cache_path.name} "
          f"({cache_path.stat().st_size / 1e6:.2f} MB)")

    if args.save_features:
        feature_path = write_features(df, args.out)
        print(f"  also wrote {feature_path.name} "
              f"({feature_path.stat().st_size / 1e6:.2f} MB)")

    report["n_samples"] = int(len(df))
    report["n_features"] = int(len(cols))
    report["positive_rate"] = float(df["y_breach"].mean())
    report["feature_seconds"] = feat_s
    validation_path.write_text(json.dumps(report, indent=2, default=float))

    print("\nDone." if report["ok"] else "\nDone, with validation issues.")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
