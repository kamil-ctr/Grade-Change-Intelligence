"""
Feature engineering for off-spec risk prediction.

Design rule enforced throughout this module: **a feature at sample k may only
use information available at or before sample k**. Every rolling statistic is
backward-looking and every rate of change is a backward difference. There is a
unit test (`test_features.py::test_no_future_leakage`) that verifies this by
perturbing the future of a trajectory and asserting the features do not move.

The target is the one the problem statement defines: will basis weight deviate
by more than 2.5% from its active setpoint at any point within the next
RISK_HORIZON_MIN minutes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config as C
from .control import recipe_slew_limits
from .events import EventResult, bw_deviation_pct
from .grades import CV_TAGS, DV_TAGS, MV_TAGS, get_grade
from .twin import DV_NOMINAL

# Samples of history required before a row is usable
WARMUP_STEPS: int = int(round(2.0 * 60.0 / C.DT_S))  # 2 minutes


# ---------------------------------------------------------------------------
# Backward-looking primitives
# ---------------------------------------------------------------------------
def _shift(x: np.ndarray, n: int) -> np.ndarray:
    """Shift forward in time by n samples, padding with the first value."""
    if n <= 0:
        return x.copy()
    out = np.empty_like(x)
    out[:n] = x[0]
    out[n:] = x[:-n]
    return out


def _roc_per_min(x: np.ndarray, minutes: float) -> np.ndarray:
    """Backward rate of change per minute over a `minutes` window."""
    n = max(int(round(minutes * 60.0 / C.DT_S)), 1)
    return (x - _shift(x, n)) / minutes


def _rolling(x: np.ndarray, minutes: float, fn: str) -> np.ndarray:
    """Backward-looking rolling statistic (includes the current sample)."""
    n = max(int(round(minutes * 60.0 / C.DT_S)), 1)
    s = pd.Series(x)
    r = s.rolling(window=n, min_periods=1)
    return getattr(r, fn)().to_numpy()


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def future_breach_label(
    abs_dev_pct: np.ndarray, horizon_min: float = C.RISK_HORIZON_MIN
) -> np.ndarray:
    """1 if |deviation| exceeds spec at any sample in (k, k + horizon]."""
    h = max(int(round(horizon_min * 60.0 / C.DT_S)), 1)
    breach = abs_dev_pct > C.BW_SPEC_PCT
    n = breach.shape[0]
    out = np.zeros(n, dtype=np.int8)
    for k in range(n):
        end = min(k + 1 + h, n)
        if k + 1 < end and breach[k + 1 : end].any():
            out[k] = 1
    return out


def time_to_breach(
    abs_dev_pct: np.ndarray, horizon_min: float = C.RISK_HORIZON_MIN
) -> np.ndarray:
    """Minutes until the next breach, NaN if none inside the horizon."""
    h = max(int(round(horizon_min * 60.0 / C.DT_S)), 1)
    breach = abs_dev_pct > C.BW_SPEC_PCT
    n = breach.shape[0]
    out = np.full(n, np.nan, dtype=float)
    for k in range(n):
        end = min(k + 1 + h, n)
        window = breach[k + 1 : end]
        if window.size and window.any():
            out[k] = (int(np.argmax(window)) + 1) * C.DT_S / 60.0
    return out


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
def build_event_features(
    ev: EventResult, horizon_min: float = C.RISK_HORIZON_MIN
) -> pd.DataFrame:
    """Build the per-sample feature table for one grade-change event."""
    s = ev.series
    n = ev.n_steps
    t_min = ev.t_min

    g_from, g_to = get_grade(ev.from_grade), get_grade(ev.to_grade)
    slew = recipe_slew_limits(g_to)

    bw, bw_sp = s["basis_weight"], s["basis_weight_sp"]
    dev = bw_deviation_pct(bw, bw_sp)
    abs_dev = np.abs(dev)

    f: Dict[str, np.ndarray] = {}

    # -- time / phase ------------------------------------------------------
    ramp_start = float(ev.plan["start_min"])
    ramp_min = float(ev.plan["ramp_min"])
    f["t_min"] = t_min
    f["t_since_ramp_min"] = t_min - ramp_start
    f["ramp_progress"] = np.clip((t_min - ramp_start) / max(ramp_min, 1e-6), 0.0, 1.0)
    f["in_ramp"] = (
        (t_min >= ramp_start) & (t_min <= ramp_start + ramp_min)
    ).astype(float)
    f["post_ramp"] = (t_min > ramp_start + ramp_min).astype(float)

    # -- transition context (RECIPE_LIMIT provenance) ----------------------
    f["bw_from"] = np.full(n, g_from.basis_weight)
    f["bw_to"] = np.full(n, g_to.basis_weight)
    f["bw_change_pct"] = np.full(
        n, (g_to.basis_weight - g_from.basis_weight) / g_from.basis_weight * 100.0
    )
    f["ash_change"] = np.full(n, g_to.ash - g_from.ash)
    f["moisture_change"] = np.full(n, g_to.moisture - g_from.moisture)
    f["speed_change_pct"] = np.full(
        n,
        (g_to.machine_speed - g_from.machine_speed) / g_from.machine_speed * 100.0,
    )
    f["transition_magnitude"] = np.full(
        n, float(ev.context.get("transition_magnitude", 0.0))
    )

    # -- plan features -----------------------------------------------------
    f["plan_ramp_min"] = np.full(n, ramp_min)
    f["plan_lead_scale"] = np.full(n, float(ev.plan["lead_scale"]))
    f["plan_tau_c_scale"] = np.full(n, float(ev.plan["tau_c_scale"]))
    f["plan_trim_enabled"] = np.full(n, float(ev.plan["trim_enabled"]))
    f["min_ramp_min"] = np.full(n, float(ev.context.get("min_ramp_min", 0.0)))
    f["ramp_deficit_min"] = np.full(
        n, float(ev.context.get("ramp_deficit_min", 0.0))
    )
    f["ramp_is_feasible"] = np.full(
        n, float(ev.context.get("ramp_is_feasible", 1.0))
    )

    # -- current quality state --------------------------------------------
    f["bw_dev_pct"] = dev
    f["bw_abs_dev_pct"] = abs_dev
    f["currently_off_spec"] = (abs_dev > C.BW_SPEC_PCT).astype(float)
    f["bw_dev_headroom_pct"] = C.BW_SPEC_PCT - abs_dev
    f["moisture_dev"] = s["moisture"] - s["moisture_sp"]
    f["ash_dev"] = s["ash"] - s["ash_sp"]
    f["caliper"] = s["caliper"]

    # -- quality dynamics --------------------------------------------------
    for minutes, suffix in ((0.5, "0_5min"), (1.0, "1min"), (2.0, "2min")):
        f[f"bw_dev_roc_{suffix}"] = _roc_per_min(dev, minutes)
    f["bw_roc_1min"] = _roc_per_min(bw, 1.0)
    f["bw_sp_roc_1min"] = _roc_per_min(bw_sp, 1.0)
    f["moisture_roc_1min"] = _roc_per_min(s["moisture"], 1.0)
    f["ash_roc_1min"] = _roc_per_min(s["ash"], 1.0)
    f["bw_dev_max_2min"] = _rolling(abs_dev, 2.0, "max")
    f["bw_dev_std_2min"] = np.nan_to_num(_rolling(dev, 2.0, "std"))
    f["bw_dev_mean_2min"] = _rolling(dev, 2.0, "mean")

    # Linear extrapolation of the current trend to the horizon: the
    # "future state if the deviation follows the current trajectory" the
    # problem statement asks the dashboard to show.
    f["bw_dev_projected"] = dev + f["bw_dev_roc_1min"] * horizon_min
    f["bw_dev_projected_abs"] = np.abs(f["bw_dev_projected"])
    f["projected_breach"] = (
        f["bw_dev_projected_abs"] > C.BW_SPEC_PCT
    ).astype(float)

    # -- actuator state ----------------------------------------------------
    for tag in MV_TAGS:
        x = s[tag]
        f[f"mv_{tag}"] = x
        f[f"mv_{tag}_roc_1min"] = _roc_per_min(x, 1.0)
        # Slew utilisation: how much of the permitted rate is being used.
        rate_limit = slew.get(tag, np.inf)
        f[f"mv_{tag}_slew_util"] = np.abs(_roc_per_min(x, 0.5)) / max(rate_limit, 1e-9)

    # Distance still to travel to the destination operating point
    from .twin import PaperMachineTwin

    tw = PaperMachineTwin(seed=0)
    mv_to = tw.inverse_solve(g_to)
    for tag in ("stock_flow", "filler_flow", "steam_pressure", "machine_speed"):
        remaining = mv_to[tag] - s[tag]
        f[f"mv_{tag}_remaining"] = remaining
        f[f"mv_{tag}_remaining_frac"] = remaining / max(abs(mv_to[tag]), 1e-9)

    # -- saturation / headroom (physical constraints) ----------------------
    steam_ceiling = s["steam_header_kpa"] * 0.95
    f["steam_headroom_kpa"] = steam_ceiling - s["steam_pressure"]
    f["steam_saturated"] = (f["steam_headroom_kpa"] < 15.0).astype(float)
    f["stock_headroom"] = g_to.stock_flow_limits[1] - s["stock_flow"]
    f["speed_at_limit"] = (
        np.abs(_roc_per_min(s["machine_speed"], 0.5)) > 0.95 * g_to.max_speed_ramp
    ).astype(float)

    # -- disturbance state -------------------------------------------------
    for tag in DV_TAGS:
        x = s[tag]
        nominal = DV_NOMINAL[tag]
        f[f"dv_{tag}"] = x
        f[f"dv_{tag}_dev"] = x - nominal
        f[f"dv_{tag}_dev_norm"] = (x - nominal) / max(abs(nominal), 1e-9)
        f[f"dv_{tag}_roc_2min"] = _roc_per_min(x, 2.0)

    # -- labels ------------------------------------------------------------
    y = future_breach_label(abs_dev, horizon_min)
    ttb = time_to_breach(abs_dev, horizon_min)

    df = pd.DataFrame(f)
    df["y_breach"] = y
    df["y_time_to_breach_min"] = ttb
    df["event_id"] = ev.event_id
    df["from_grade"] = ev.from_grade
    df["to_grade"] = ev.to_grade
    df["primary_cause"] = ev.faults[0]["code"] if ev.faults else "NONE"
    df["sample_idx"] = np.arange(n)

    # Drop warmup rows: rolling statistics are not yet meaningful there.
    df = df.iloc[WARMUP_STEPS:].reset_index(drop=True)
    return df


def build_dataset_features(
    events: Sequence[EventResult],
    horizon_min: float = C.RISK_HORIZON_MIN,
    progress: bool = False,
) -> pd.DataFrame:
    """Concatenate per-event feature tables into one training frame."""
    frames: List[pd.DataFrame] = []
    for i, ev in enumerate(events):
        frames.append(build_event_features(ev, horizon_min))
        if progress and (i + 1) % 50 == 0:
            print(f"  featurised {i + 1}/{len(events)} events")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------
NON_FEATURE_COLS: Tuple[str, ...] = (
    "y_breach",
    "y_time_to_breach_min",
    "event_id",
    "from_grade",
    "to_grade",
    "primary_cause",
    "sample_idx",
    "t_min",
)


def feature_columns(df: pd.DataFrame) -> List[str]:
    """Numeric model inputs, excluding labels and identifiers."""
    return [
        c
        for c in df.columns
        if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def downcast_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Halve the memory footprint: float64 -> float32 for feature columns,
    strings -> categorical.

    float32 has ~7 significant digits, far more than any process measurement
    carries, so nothing meaningful is lost. This matters because the training
    pipeline holds the full frame plus three split copies simultaneously.
    """
    out = df.copy()
    for col in out.columns:
        if col in ("event_id", "sample_idx", "y_breach"):
            continue
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].astype(np.float32)
        elif pd.api.types.is_object_dtype(out[col]):
            out[col] = out[col].astype("category")
    return out


def save_features(df: pd.DataFrame, path) -> None:
    """
    Cache the feature frame so training need not regenerate the corpus.

    Uses pandas' native pickle: no optional engine required, loads in under a
    second, and preserves dtypes exactly (which parquet round-trips would not
    for the categorical columns).
    """
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Only downcast if the caller has not already done so: `downcast_features`
    # copies the whole frame, and at 500 events that copy is ~150 MB.
    already = all(
        not pd.api.types.is_float_dtype(df[c]) or df[c].dtype == np.float32
        for c in df.columns
    )
    (df if already else downcast_features(df)).to_pickle(
        path, compression="infer"
    )


def load_features(path) -> pd.DataFrame:
    """Load a cached feature frame."""
    return pd.read_pickle(path)


def event_wise_split(
    df: pd.DataFrame, test_frac: float = 0.25, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by *event*, never by row.

    Rows within one transition are heavily autocorrelated; a random row split
    would leak almost-identical samples across the boundary and inflate every
    metric. Splitting whole events is the only honest option.
    """
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(df["event_id"].unique()))
    rng.shuffle(ids)
    n_test = max(int(round(len(ids) * test_frac)), 1)
    test_ids = set(ids[:n_test].tolist())

    mask = df["event_id"].isin(test_ids)
    return df.loc[~mask].reset_index(drop=True), df.loc[mask].reset_index(drop=True)
