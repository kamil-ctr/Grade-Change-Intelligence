"""
Quantile trajectory forecasting -- the dashboard forecast cone.

Predicts basis-weight deviation (% of setpoint) at +2 / +5 / +10 minutes, at
the 10th / 50th / 90th percentiles, from the same backward-looking features
the risk model uses. This is what backs "future state if the deviation
follows its current trajectory": not a single point but a cone, because the
whole point of surfacing uncertainty is to let the operator judge how much to
trust it.

Targets here are forward-looking by construction -- the value later on the
same trajectory -- so they are computed in this module, never inside
`features.py`. That module's feature contract must stay strictly
backward-looking (`test_no_future_leakage` in `tests/test_features.py`
enforces it), and mixing a forward target into that frame would be exactly
the leakage that test exists to catch. `forecast_feature_columns()` below is
the defensive boundary: it takes the risk model's feature list and explicitly
strips out anything this module adds on top.

Reuses `gci.ml.splits.event_wise_split_3way` for the train/validation/test
partition, so the forecast and risk models are scored against identical event
membership -- the same non-negotiable (never split rows randomly) applies
here as much as it does to the classifier.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config as C
from .events import EventResult, bw_deviation_pct
from .features import feature_columns
from .ml.registry import RANDOM_STATE
from .ml.splits import EventSplit, event_wise_split_3way

FORECAST_HORIZONS_MIN: Tuple[float, ...] = (2.0, 5.0, 10.0)
FORECAST_QUANTILES: Tuple[float, ...] = (0.1, 0.5, 0.9)


def target_column(horizon_min: float) -> str:
    """Column name for the forward basis-weight-deviation target at a horizon."""
    return f"y_bw_dev_fwd_{horizon_min:g}min"


FORECAST_TARGET_COLS: Tuple[str, ...] = tuple(
    target_column(h) for h in FORECAST_HORIZONS_MIN
)


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
def dev_lookup_from_events(events: Sequence[EventResult]) -> Dict[int, np.ndarray]:
    """Per-event signed basis-weight-deviation trajectory, keyed by event_id."""
    return {
        ev.event_id: bw_deviation_pct(
            ev.series["basis_weight"], ev.series["basis_weight_sp"]
        )
        for ev in events
    }


def dev_lookup_from_dataset(
    cube: np.ndarray, tags: Sequence[str], meta: dict
) -> Dict[int, np.ndarray]:
    """
    The same per-event lookup built directly from a persisted dataset
    (`events.load_dataset`'s return), without re-simulating. Training scripts
    should prefer this: the corpus is already on disk and re-running
    `generate_dataset` costs the better part of a minute at 1000+ events.
    """
    bw_idx = list(tags).index("basis_weight")
    sp_idx = list(tags).index("basis_weight_sp")
    out: Dict[int, np.ndarray] = {}
    for i, ev_meta in enumerate(meta["events"]):
        bw = cube[i, :, bw_idx].astype(np.float64)
        bw_sp = cube[i, :, sp_idx].astype(np.float64)
        out[int(ev_meta["event_id"])] = bw_deviation_pct(bw, bw_sp)
    return out


def build_forecast_targets(
    dev_lookup: Dict[int, np.ndarray],
    df: pd.DataFrame,
    horizons_min: Sequence[float] = FORECAST_HORIZONS_MIN,
) -> pd.DataFrame:
    """
    Add forward basis-weight-deviation targets to a feature frame.

    `df` must carry `event_id` and `sample_idx` columns from
    `features.build_dataset_features` (or a row subset of it) built on the
    same events `dev_lookup` was derived from. For a row at absolute sample
    `sample_idx` within its event, the target at horizon `h` is the *signed*
    deviation `h` minutes later in that same event's trajectory.

    Rows near the end of the 30-minute window have no valid future sample for
    the larger horizons -- those get NaN (the simulation ended, not "no
    risk") rather than a fabricated value. Callers drop NaN rows per horizon
    at fit/eval time instead of filling them.
    """
    series = [
        pd.Series(
            dev, index=pd.MultiIndex.from_product([[eid], np.arange(len(dev))])
        )
        for eid, dev in dev_lookup.items()
    ]
    lookup = pd.concat(series)
    lookup.index.names = ["event_id", "step"]

    out = df.copy()
    event_id = out["event_id"].to_numpy()
    sample_idx = out["sample_idx"].to_numpy()
    for h in horizons_min:
        steps = int(round(h * C.STEPS_PER_MIN))
        idx = pd.MultiIndex.from_arrays([event_id, sample_idx + steps])
        out[target_column(h)] = lookup.reindex(idx).to_numpy()
    return out


def forecast_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Backward-looking model inputs: the risk model's feature list, with any
    forecast target columns explicitly re-excluded. Defensive rather than
    redundant -- `features.feature_columns` only knows about `NON_FEATURE_COLS`
    defined in `features.py`, not about columns this module layers on top of
    the same frame, so it cannot see the targets on its own.
    """
    excluded = set(FORECAST_TARGET_COLS)
    return [c for c in feature_columns(df) if c not in excluded]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def _quantile_estimator(quantile: float):
    """
    LightGBM's native quantile objective if available; otherwise scikit-learn's
    `HistGradientBoostingRegressor` with the same loss. Same "optional deps
    probed, never assumed" posture as `ml/registry.py` -- the pipeline still
    yields a usable forecaster on a bare `pip install scikit-learn`.
    """
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=-1,
        ), "lightgbm"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            max_iter=400,
            learning_rate=0.06,
            max_depth=7,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=RANDOM_STATE,
        ), "hist_gradient_boosting"


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Quantile (pinball) loss -- the proper scoring rule for a quantile forecast."""
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


@dataclass
class ForecastResult:
    horizon_min: float
    quantile: float
    estimator: object
    method: str


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class ForecastPipeline:
    """
    Trains one quantile regressor per (horizon, quantile) pair -- by default
    3 horizons x 3 quantiles = 9 small models -- on the event-wise split
    shared with the risk model.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dev_lookup: Dict[int, np.ndarray],
        val_frac: float = 0.20,
        test_frac: float = 0.20,
        seed: int = 42,
        horizons: Sequence[float] = FORECAST_HORIZONS_MIN,
        quantiles: Sequence[float] = FORECAST_QUANTILES,
        verbose: bool = True,
    ):
        self.horizons = tuple(horizons)
        self.quantiles = tuple(sorted(quantiles))
        self.verbose = verbose
        self.features = forecast_feature_columns(df)
        self.split: EventSplit = event_wise_split_3way(
            df, val_frac=val_frac, test_frac=test_frac, seed=seed
        )
        self.frame_ = build_forecast_targets(dev_lookup, df, self.horizons)
        self.models: Dict[Tuple[float, float], ForecastResult] = {}
        self.metrics_: Dict[str, dict] = {}

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def frame(self, which: str) -> pd.DataFrame:
        return self.split.frame(self.frame_, which)

    def xy(self, which: str, horizon_min: float) -> Tuple[pd.DataFrame, np.ndarray]:
        part = self.frame(which)
        col = target_column(horizon_min)
        valid = part[col].notna()
        return part.loc[valid, self.features], part.loc[valid, col].to_numpy()

    def fit_all(self) -> "ForecastPipeline":
        for h in self.horizons:
            X, y = self.xy("train", h)
            for q in self.quantiles:
                est, method = _quantile_estimator(q)
                est.fit(X, y)
                self.models[(h, q)] = ForecastResult(h, q, est, method)
            self._log(
                f"  horizon {h:g} min: trained {len(self.quantiles)} quantile "
                f"models ({self.models[(h, self.quantiles[0])].method}) on "
                f"{len(X):,} rows"
            )
        return self

    def predict_cone(self, X: pd.DataFrame) -> Dict[float, Dict[float, np.ndarray]]:
        """
        Predicted quantiles per horizon, monotone-corrected: independent
        quantile regressors can cross near the tails, so predictions are
        sorted row-wise (quantile rearrangement -- the standard, cheap fix;
        see Chernozhukov, Fernandez-Val & Galichon 2010) before being handed
        to the caller as a usable cone.
        """
        out: Dict[float, Dict[float, np.ndarray]] = {}
        for h in self.horizons:
            preds = np.column_stack(
                [self.models[(h, q)].estimator.predict(X[self.features])
                 for q in self.quantiles]
            )
            preds = np.sort(preds, axis=1)
            out[h] = {q: preds[:, i] for i, q in enumerate(self.quantiles)}
        return out

    def evaluate(self, which: str = "validation") -> Dict[str, dict]:
        part = self.frame(which)
        lo_q, hi_q = self.quantiles[0], self.quantiles[-1]
        median_q = self.quantiles[len(self.quantiles) // 2]
        results: Dict[str, dict] = {}
        for h in self.horizons:
            col = target_column(h)
            valid = part[col].notna()
            X = part.loc[valid, self.features]
            y_true = part.loc[valid, col].to_numpy()
            cone = self.predict_cone(X)[h]

            per_q = {
                str(q): pinball_loss(y_true, cone[q], q) for q in self.quantiles
            }
            lo, hi = cone[lo_q], cone[hi_q]
            coverage = float(np.mean((y_true >= lo) & (y_true <= hi))) if len(y_true) else float("nan")
            median = cone[median_q]

            results[f"{h:g}min"] = {
                "n": int(valid.sum()),
                "mae_median": float(np.mean(np.abs(y_true - median))) if len(y_true) else float("nan"),
                "rmse_median": float(np.sqrt(np.mean((y_true - median) ** 2))) if len(y_true) else float("nan"),
                "interval_coverage": coverage,
                "nominal_coverage": float(hi_q - lo_q),
                "mean_interval_width": float(np.mean(hi - lo)) if len(y_true) else float("nan"),
                "pinball_loss": per_q,
            }
        self.metrics_[which] = results
        return results

    def save(self, out_dir: Path) -> Dict[str, Path]:
        import joblib

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: Dict[str, Path] = {}

        bundle = {
            "estimators": {k: m.estimator for k, m in self.models.items()},
            "method": {k: m.method for k, m in self.models.items()},
            "features": self.features,
            "horizons_min": self.horizons,
            "quantiles": self.quantiles,
        }
        model_path = out_dir / "forecast_model.joblib"
        joblib.dump(bundle, model_path)
        written["model"] = model_path

        metrics_path = out_dir / "forecast_metrics.json"
        metrics_path.write_text(json.dumps(self.metrics_, indent=2, default=float))
        written["metrics"] = metrics_path

        return written


# ---------------------------------------------------------------------------
# Serving-time helpers (used by the API layer)
# ---------------------------------------------------------------------------
def load_forecast_bundle(path: Path) -> dict:
    import joblib

    return joblib.load(path)


def forecast_cone_from_bundle(
    bundle: dict, X: pd.DataFrame
) -> Dict[float, Dict[float, np.ndarray]]:
    """Predict a monotone quantile cone from a saved bundle, for serving."""
    features = bundle["features"]
    quantiles = bundle["quantiles"]
    out: Dict[float, Dict[float, np.ndarray]] = {}
    for h in bundle["horizons_min"]:
        preds = np.column_stack(
            [bundle["estimators"][(h, q)].predict(X[features]) for q in quantiles]
        )
        preds = np.sort(preds, axis=1)
        out[h] = {q: preds[:, i] for i, q in enumerate(quantiles)}
    return out
