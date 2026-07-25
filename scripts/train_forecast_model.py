#!/usr/bin/env python3
"""
Train the quantile forecast model (dashboard forecast cone).

Usage:
    python scripts/train_forecast_model.py [--events 1500] [--seed 42]

Loads the cached feature frame and persisted event trajectories written by
`scripts/generate_data.py` (regenerating them if absent), builds forward
basis-weight-deviation targets at +2/+5/+10 min, trains 10th/50th/90th
percentile regressors per horizon, evaluates on validation then test (once),
and writes:

    forecast_model.joblib    estimators + feature list + horizons + quantiles
    forecast_metrics.json    pinball loss, coverage, MAE/RMSE per horizon/split
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gci.events import generate_dataset, load_dataset, save_dataset  # noqa: E402
from gci.features import (  # noqa: E402
    build_dataset_features,
    downcast_features,
    load_features,
    save_features,
)
from gci.forecast import (  # noqa: E402
    ForecastPipeline,
    dev_lookup_from_dataset,
    dev_lookup_from_events,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=1500)
    parser.add_argument("--data-seed", type=int, default=20260725)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "models")
    parser.add_argument(
        "--regenerate", action="store_true",
        help="ignore cached corpus/features and rebuild both",
    )
    args = parser.parse_args()

    data_dir = ROOT / "data"
    cache = data_dir / "features.pkl"
    have_cache = cache.exists() and (data_dir / "events_series.npz").exists()

    if have_cache and not args.regenerate:
        print(f"Loading cached features from {cache.relative_to(ROOT)} ...")
        t0 = time.perf_counter()
        df = load_features(cache)
        print(f"  {len(df):,} rows x {df.shape[1]} columns in "
              f"{time.perf_counter() - t0:.1f}s")

        print("Loading persisted trajectories ...")
        cube, tags, meta = load_dataset(data_dir)
        dev_lookup = dev_lookup_from_dataset(cube, tags, meta)
        print(f"  {len(dev_lookup)} events")
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
        save_dataset(events, data_dir)
        dev_lookup = dev_lookup_from_events(events)
    print()

    pipeline = ForecastPipeline(df, dev_lookup, seed=args.seed)
    print(f"Features: {len(pipeline.features)}  "
          f"Horizons: {pipeline.horizons}  Quantiles: {pipeline.quantiles}")
    print("Split composition:")
    for which in ("train", "validation", "test"):
        print(f"  {which:11s} {len(pipeline.split.events_for(which))} events")
    print()

    print("Training quantile models ...")
    t0 = time.perf_counter()
    pipeline.fit_all()
    print(f"  {time.perf_counter() - t0:.1f}s total\n")

    print("Validation:")
    val = pipeline.evaluate("validation")
    for h, m in val.items():
        print(
            f"  {h:>6s}  n={m['n']:6d}  MAE {m['mae_median']:.3f}%  "
            f"coverage {m['interval_coverage']:.3f} "
            f"(nominal {m['nominal_coverage']:.2f})  "
            f"width {m['mean_interval_width']:.3f}%"
        )

    print("\nTest (scored once):")
    test = pipeline.evaluate("test")
    for h, m in test.items():
        print(
            f"  {h:>6s}  n={m['n']:6d}  MAE {m['mae_median']:.3f}%  "
            f"coverage {m['interval_coverage']:.3f} "
            f"(nominal {m['nominal_coverage']:.2f})  "
            f"width {m['mean_interval_width']:.3f}%"
        )

    written = pipeline.save(args.out)
    print("\nArtefacts:")
    for name, path in written.items():
        print(f"  {name:10s} {path.name} ({path.stat().st_size / 1024:.1f} KB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
