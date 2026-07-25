"""
Model registry.

Adding a model means appending one `ModelSpec` -- no changes to the pipeline,
metrics or reporting. That is the whole point of the indirection.

Optional dependencies are handled by design rather than by hope. LightGBM,
XGBoost and SHAP are all optional: each spec declares what it needs, the
registry probes for it once, and unavailable models are skipped with a recorded
reason instead of crashing the run. scikit-learn's histogram gradient boosting
is always present and is a genuinely competitive stand-in for LightGBM, so the
pipeline produces a usable model on a bare `pip install scikit-learn`.

This is not defensive padding -- it is the "demo cannot fail" requirement
applied to the training stage.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

RANDOM_STATE = 20260725


def _probe(module: str) -> bool:
    """Is an optional dependency importable?"""
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


@dataclass
class ModelSpec:
    """Everything the pipeline needs to know about one candidate model."""

    name: str
    label: str
    family: str                       # tree_ensemble | linear
    factory: Callable[[], object]
    requires: Tuple[str, ...] = ()    # importable module names
    needs_scaling: bool = False
    supports_native_shap: bool = False
    notes: str = ""

    @property
    def available(self) -> bool:
        return all(_probe(m) for m in self.requires)

    @property
    def missing(self) -> List[str]:
        return [m for m in self.requires if not _probe(m)]

    def build(self):
        if not self.available:
            raise ImportError(
                f"model '{self.name}' needs missing package(s): "
                f"{', '.join(self.missing)}"
            )
        return self.factory()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "family": self.family,
            "available": self.available,
            "missing": self.missing,
            "needs_scaling": self.needs_scaling,
            "supports_native_shap": self.supports_native_shap,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Factories (imports are deferred so an absent package cannot break import)
# ---------------------------------------------------------------------------
def _random_forest():
    from sklearn.ensemble import RandomForestClassifier

    # 150 trees is well past the point where added trees change the ranking on
    # 100k rows; depth 10 with a leaf floor of 40 keeps it from memorising the
    # heavy autocorrelation between consecutive samples within an event.
    return RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=40,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def _hist_gradient_boosting():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.06,
        max_depth=7,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=RANDOM_STATE,
    )


def _lightgbm():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=48,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )


def _xgboost():
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def _logistic_regression():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # The scaler lives inside the estimator, so it is fitted on training folds
    # only and can never see validation or test statistics.
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000, C=0.5, random_state=RANDOM_STATE
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        ModelSpec(
            name="lightgbm",
            label="LightGBM",
            family="tree_ensemble",
            factory=_lightgbm,
            requires=("lightgbm",),
            supports_native_shap=True,
            notes="Gradient boosting with exact TreeSHAP via pred_contrib.",
        ),
        ModelSpec(
            name="xgboost",
            label="XGBoost",
            family="tree_ensemble",
            factory=_xgboost,
            requires=("xgboost",),
            supports_native_shap=True,
            notes="Gradient boosting with exact TreeSHAP via pred_contribs.",
        ),
        ModelSpec(
            name="random_forest",
            label="Random Forest",
            family="tree_ensemble",
            factory=_random_forest,
            notes="Bagged trees; robust baseline, no tuning sensitivity.",
        ),
        ModelSpec(
            name="hist_gradient_boosting",
            label="Histogram Gradient Boosting",
            family="tree_ensemble",
            factory=_hist_gradient_boosting,
            notes=(
                "scikit-learn's LightGBM-equivalent. Always available, so the "
                "pipeline still yields a strong model with no optional deps."
            ),
        ),
        ModelSpec(
            name="logistic_regression",
            label="Logistic Regression",
            family="linear",
            factory=_logistic_regression,
            needs_scaling=True,
            notes=(
                "Deliberate sanity baseline: if the tree ensembles cannot beat "
                "a linear model, the features are doing the work, not the model."
            ),
        ),
    )
}


def available_models() -> Dict[str, ModelSpec]:
    return {n: s for n, s in MODEL_REGISTRY.items() if s.available}


def unavailable_models() -> Dict[str, ModelSpec]:
    return {n: s for n, s in MODEL_REGISTRY.items() if not s.available}


def registry_report() -> List[dict]:
    return [spec.to_dict() for spec in MODEL_REGISTRY.values()]
