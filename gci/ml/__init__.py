"""
Machine learning pipeline for off-spec risk prediction.

Pipeline contract
-----------------
1. **Event-wise splitting only.** Train / validation / test are disjoint sets of
   whole grade-change events, stratified by outcome. Rows are never split
   randomly: samples 5 seconds apart inside one transition are almost
   identical, so a row split leaks near-duplicates across the boundary and
   inflates every metric.

2. **No future leakage anywhere.** Features are backward-looking by
   construction (verified by `tests/test_features.py::test_no_future_leakage`).
   The pipeline adds three further guarantees: any preprocessing is fitted on
   train only, the decision threshold is chosen on validation only, and the
   test split is scored exactly once at the very end.

3. **Evaluation excludes rows that are already off-spec.** Predicting a breach
   while the sheet is *currently* off-spec is trivial and worthless -- the
   operator can already see it. Every headline metric is computed on rows where
   basis weight is still inside the 2.5% band, which is the only population
   where a warning has value.

4. **The headline metric is early warning time.** PR-AUC ranks models, but what
   an operator buys is minutes of notice. Both are reported.

Modules
-------
`splits`    event-wise stratified three-way splitting
`metrics`   classification metrics plus early-warning-time analysis
`registry`  model registry -- add a model by appending one entry
`explain`   global importance and SHAP (exact TreeSHAP where available)
`pipeline`  orchestration: train all, select best, persist artefacts
"""

from .metrics import (  # noqa: F401
    classification_metrics,
    early_warning_analysis,
    evaluate_model,
    pick_threshold,
)
from .registry import MODEL_REGISTRY, ModelSpec, available_models  # noqa: F401
from .splits import EventSplit, event_wise_split_3way  # noqa: F401

__all__ = [
    "EventSplit",
    "event_wise_split_3way",
    "classification_metrics",
    "early_warning_analysis",
    "evaluate_model",
    "pick_threshold",
    "MODEL_REGISTRY",
    "ModelSpec",
    "available_models",
]
