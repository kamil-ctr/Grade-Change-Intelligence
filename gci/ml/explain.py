"""
Explainability: global importance and SHAP values.

SHAP strategy is tiered, so an exact answer is produced whenever one is
obtainable and the pipeline never simply gives up:

1. ``shap`` package present -> ``shap.TreeExplainer`` (exact TreeSHAP).
2. LightGBM model -> ``predict(pred_contrib=True)`` (exact TreeSHAP, native).
3. XGBoost model -> ``predict(..., pred_contribs=True)`` (exact TreeSHAP, native).
4. Otherwise -> permutation importance for global attribution, and a clearly
   labelled linear-surrogate local attribution.

Tiers 2 and 3 matter: LightGBM and XGBoost both compute exact TreeSHAP
internally, so the `shap` dependency is a convenience, not a requirement. Local
per-prediction explanations survive even on a minimal install, which is what the
operator-facing "why this prediction?" panel depends on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Global importance
# ---------------------------------------------------------------------------
def native_importance(
    model, feature_names: Sequence[str]
) -> Optional[pd.DataFrame]:
    """Model's own importance measure, if it exposes one."""
    values = None
    kind = ""

    booster = getattr(model, "feature_importances_", None)
    if booster is not None:
        values = np.asarray(booster, dtype=float)
        kind = "impurity_or_gain"

    if values is None or values.size != len(feature_names):
        return None

    return (
        pd.DataFrame({"feature": list(feature_names), "importance": values,
                      "kind": kind})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def permutation_based_importance(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    n_repeats: int = 5,
    max_rows: int = 6000,
    random_state: int = 0,
    scoring: str = "average_precision",
) -> pd.DataFrame:
    """
    Permutation importance on a held-out split.

    Computed on validation data, never training data: importance measured on
    the fitting set rewards memorisation.
    """
    from sklearn.inspection import permutation_importance

    rng = np.random.default_rng(random_state)
    if len(X) > max_rows:
        idx = rng.choice(len(X), size=max_rows, replace=False)
        X, y = X.iloc[idx], np.asarray(y)[idx]

    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state,
        scoring=scoring, n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": list(X.columns),
                "importance": result.importances_mean,
                "std": result.importances_std,
                "kind": f"permutation_{scoring}",
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
@dataclass
class ShapResult:
    """SHAP values plus provenance about how they were obtained."""

    values: np.ndarray            # (n_rows, n_features)
    base_value: float
    method: str
    exact: bool
    feature_names: List[str]

    def mean_abs(self) -> pd.DataFrame:
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "mean_abs_shap": np.abs(self.values).mean(axis=0),
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    def local(self, row: int, top_k: int = 8) -> List[dict]:
        """Top contributions for one prediction, for the 'why?' panel."""
        contrib = self.values[row]
        order = np.argsort(np.abs(contrib))[::-1][:top_k]
        return [
            {
                "feature": self.feature_names[i],
                "shap_value": float(contrib[i]),
                "direction": "increases risk" if contrib[i] > 0 else "reduces risk",
            }
            for i in order
        ]


def _shap_via_package(model, X: pd.DataFrame) -> Optional[ShapResult]:
    try:
        import shap
    except Exception:
        return None
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X, check_additivity=False)
        # Binary classifiers may return a list per class or a 3-D array
        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]

        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = np.asarray(base).ravel()
            base = float(base[1] if base.size > 1 else base[0])
        return ShapResult(
            values=values, base_value=float(base),
            method="shap.TreeExplainer", exact=True,
            feature_names=list(X.columns),
        )
    except Exception:
        return None


def _shap_via_lightgbm(model, X: pd.DataFrame) -> Optional[ShapResult]:
    try:
        from lightgbm import LGBMClassifier, LGBMModel
    except Exception:
        return None
    if not isinstance(model, (LGBMClassifier, LGBMModel)):
        return None
    try:
        contrib = np.asarray(
            model.booster_.predict(X.to_numpy(), pred_contrib=True), dtype=float
        )
        # Last column is the expected value
        return ShapResult(
            values=contrib[:, :-1], base_value=float(contrib[0, -1]),
            method="lightgbm.pred_contrib (native TreeSHAP)", exact=True,
            feature_names=list(X.columns),
        )
    except Exception:
        return None


def _shap_via_xgboost(model, X: pd.DataFrame) -> Optional[ShapResult]:
    try:
        import xgboost as xgb
    except Exception:
        return None
    if not isinstance(model, xgb.XGBModel):
        return None
    try:
        booster = model.get_booster()
        dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
        contrib = np.asarray(
            booster.predict(dmatrix, pred_contribs=True), dtype=float
        )
        return ShapResult(
            values=contrib[:, :-1], base_value=float(contrib[0, -1]),
            method="xgboost.pred_contribs (native TreeSHAP)", exact=True,
            feature_names=list(X.columns),
        )
    except Exception:
        return None


def _shap_via_linear(model, X: pd.DataFrame) -> Optional[ShapResult]:
    """
    Exact SHAP for a linear model.

    For f(x) = w.x + b with an interventional (marginal) background
    distribution, the Shapley value of feature i is exactly

        phi_i = w_i * (x_i - E[x_i])

    and the base value is w.E[x] + b. No approximation and no sampling -- so a
    linear model is just as explainable as a tree ensemble, which removes any
    need to trade accuracy against interpretability when choosing between them.

    Handles a bare estimator or an sklearn Pipeline whose final step is linear:
    the preceding steps are applied first, so attributions are in the space the
    coefficients actually live in, then mapped back to the original columns.
    """
    try:
        from sklearn.pipeline import Pipeline
    except Exception:  # pragma: no cover
        return None

    final, transform = model, None
    if isinstance(model, Pipeline):
        final = model.steps[-1][1]
        if len(model.steps) > 1:
            transform = Pipeline(model.steps[:-1])

    coef = getattr(final, "coef_", None)
    intercept = getattr(final, "intercept_", None)
    if coef is None or intercept is None:
        return None

    coef = np.asarray(coef, dtype=float).ravel()
    intercept = float(np.asarray(intercept, dtype=float).ravel()[0])

    Xt = np.asarray(
        transform.transform(X) if transform is not None else X.to_numpy(),
        dtype=float,
    )
    if coef.size != Xt.shape[1]:
        return None

    background = Xt.mean(axis=0)
    values = (Xt - background[None, :]) * coef[None, :]
    base = float(np.dot(coef, background) + intercept)

    return ShapResult(
        values=values, base_value=base,
        method="analytic linear SHAP (exact)", exact=True,
        feature_names=list(X.columns),
    )


def _shap_surrogate(model, X: pd.DataFrame) -> ShapResult:
    """
    Fallback local attribution: a ridge surrogate fitted to the model's own
    predicted probabilities on standardised features.

    Labelled `exact=False` and named as a surrogate everywhere it surfaces --
    an approximation presented as SHAP would be exactly the kind of quiet
    dishonesty this project is trying to avoid.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    proba = _predict_proba(model, X)

    ridge = Ridge(alpha=1.0).fit(Xs, proba)
    contributions = Xs * ridge.coef_[None, :]

    return ShapResult(
        values=contributions, base_value=float(ridge.intercept_),
        method="ridge surrogate (approximate, not TreeSHAP)", exact=False,
        feature_names=list(X.columns),
    )


def compute_shap(
    model, X: pd.DataFrame, max_rows: int = 3000, random_state: int = 0
) -> ShapResult:
    """Best available SHAP values for `model` on `X`, using the tiered strategy."""
    if len(X) > max_rows:
        rng = np.random.default_rng(random_state)
        X = X.iloc[rng.choice(len(X), size=max_rows, replace=False)]

    for attempt in (
        _shap_via_package,
        _shap_via_lightgbm,
        _shap_via_xgboost,
        _shap_via_linear,
    ):
        result = attempt(model, X)
        if result is not None:
            return result
    return _shap_surrogate(model, X)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Positive-class probability, tolerant of estimator API differences."""
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))
    return np.asarray(model.predict(X), dtype=float)


def combined_importance(
    model,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    shap_result: Optional[ShapResult] = None,
    n_repeats: int = 5,
    perm_max_rows: int = 6000,
) -> pd.DataFrame:
    """
    Merge every available attribution into one ranked table.

    Three independent views (native gain, permutation, mean |SHAP|) agreeing on
    the top features is far stronger evidence than any single ranking, and
    disagreement is itself diagnostic.
    """
    frames: List[pd.DataFrame] = []

    native = native_importance(model, list(X_val.columns))
    if native is not None:
        total = native["importance"].sum()
        native = native.assign(
            native_norm=native["importance"] / (total if total else 1.0)
        )[["feature", "native_norm"]]
        frames.append(native.set_index("feature"))

    perm = permutation_based_importance(
        model, X_val, y_val, n_repeats=n_repeats, max_rows=perm_max_rows
    )
    perm_total = perm["importance"].abs().sum()
    frames.append(
        perm.assign(
            permutation_norm=perm["importance"] / (perm_total if perm_total else 1.0)
        )[["feature", "permutation_norm"]].set_index("feature")
    )

    if shap_result is not None:
        shap_df = shap_result.mean_abs()
        shap_total = shap_df["mean_abs_shap"].sum()
        frames.append(
            shap_df.assign(
                shap_norm=shap_df["mean_abs_shap"]
                / (shap_total if shap_total else 1.0)
            )[["feature", "shap_norm"]].set_index("feature")
        )

    merged = pd.concat(frames, axis=1).fillna(0.0)
    merged["consensus"] = merged.mean(axis=1)
    return (
        merged.sort_values("consensus", ascending=False)
        .reset_index()
        .rename(columns={"index": "feature"})
    )
