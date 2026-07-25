"""
Evaluation metrics for off-spec risk prediction.

Two families of metric, and the distinction matters:

**Classification metrics** (PR-AUC, precision, recall, F1, false positive rate)
answer "is the model statistically sound?".

**Early warning metrics** (median and mean warning time, detection rate, false
alarm rate per event) answer "does this help an operator?". An operator does not
buy PR-AUC; they buy minutes of notice before the sheet goes off-spec. Warning
time is therefore reported as the headline number.

Evaluation population
---------------------
By default all metrics are computed on rows where basis weight is **still inside
the 2.5% band** (`currently_off_spec == 0`). Predicting a breach while the sheet
is already off-spec is trivial -- the operator is looking at the same trend line
-- and including those rows would inflate every score. This exclusion is applied
consistently and is stated in the evaluation report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from .. import config as C


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------
def pick_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    criterion: str = "f1",
    beta: float = 0.5,
    max_fpr: float = 0.10,
) -> Tuple[float, Dict[str, float]]:
    """
    Choose an operating threshold. **Validation data only** -- never test.

    Criteria
    --------
    ``f1``     balanced operating point; the default.
    ``fbeta``  precision-weighted (beta < 1). Fewer nuisance alarms, in the
               spirit of alarm rationalisation: an advisory system that cries
               wolf gets ignored and then switched off.
    ``max_fpr``highest recall achievable while holding the false positive rate
               at or below ``max_fpr``.
    """
    if y_true.sum() == 0 or y_true.sum() == y_true.size:
        return 0.5, {"note": "degenerate labels; threshold defaulted to 0.5"}

    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision_recall_curve returns one more point than thresholds
    precision, recall = precision[:-1], recall[:-1]

    if criterion == "f1":
        with np.errstate(divide="ignore", invalid="ignore"):
            score = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    elif criterion == "fbeta":
        b2 = beta * beta
        with np.errstate(divide="ignore", invalid="ignore"):
            score = (
                (1 + b2) * precision * recall
                / np.maximum(b2 * precision + recall, 1e-12)
            )
    elif criterion == "max_fpr":
        n_neg = float((y_true == 0).sum())
        score = np.empty_like(thresholds, dtype=float)
        for i, thr in enumerate(thresholds):
            pred = (proba >= thr).astype(int)
            fp = float(((pred == 1) & (y_true == 0)).sum())
            fpr = fp / max(n_neg, 1.0)
            score[i] = recall[i] if fpr <= max_fpr else -1.0
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown criterion '{criterion}'")

    score = np.nan_to_num(score, nan=-1.0)
    best = int(np.argmax(score))
    return float(thresholds[best]), {
        "criterion": criterion,
        "score": float(score[best]),
        "precision_at_threshold": float(precision[best]),
        "recall_at_threshold": float(recall[best]),
    }


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------
def confirm_alarms(
    raw: np.ndarray, persistence_samples: int, group: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Alarm on-delay: require `persistence_samples` consecutive samples above
    threshold before an alarm is confirmed.

    This is standard alarm-management practice (ISA-18.2 on-delay) and it exists
    for exactly the reason it is needed here: a single noisy sample crossing a
    threshold is not evidence of anything, but it does generate a nuisance
    alarm, and an advisory system that raises nuisance alarms gets ignored and
    then switched off. Because the risk signal is strongly autocorrelated, a
    genuine excursion stays above threshold for many consecutive samples, so
    on-delay removes false alarms at almost no cost in warning time.

    `group` (event ids) prevents the persistence window from spanning the
    boundary between two events.
    """
    raw = np.asarray(raw).astype(bool)
    if persistence_samples <= 1:
        return raw

    series = pd.Series(raw.astype(float))
    if group is None:
        rolled = series.rolling(persistence_samples, min_periods=persistence_samples).min()
    else:
        rolled = series.groupby(pd.Series(group)).transform(
            lambda s: s.rolling(
                persistence_samples, min_periods=persistence_samples
            ).min()
        )
    return rolled.fillna(0.0).to_numpy() >= 1.0


def classification_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    persistence_samples: int = 1,
    group: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Every classification metric required by the evaluation spec."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = confirm_alarms(
        proba >= threshold, persistence_samples, group
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    out = {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
        "specificity": float(specificity),
        "accuracy": float((tp + tn) / max(y_true.size, 1)),
        "brier_score": float(brier_score_loss(y_true, proba)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "n_samples": int(y_true.size),
        "positive_rate": float(y_true.mean()),
    }
    # ROC-AUC is undefined on a single-class set
    if 0 < y_true.sum() < y_true.size:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
    return out


def confusion_matrix_dict(
    y_true: np.ndarray, proba: np.ndarray, threshold: float
) -> Dict[str, object]:
    """Confusion matrix in a JSON-serialisable, UI-friendly shape."""
    pred = (np.asarray(proba, dtype=float) >= threshold).astype(int)
    cm = confusion_matrix(np.asarray(y_true).astype(int), pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": float(threshold),
        "labels": ["in-spec (no breach ahead)", "breach within horizon"],
        "matrix": cm.tolist(),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


# ---------------------------------------------------------------------------
# Early warning analysis -- the headline metric
# ---------------------------------------------------------------------------
@dataclass
class WarningResult:
    """Per-event early-warning outcome."""

    event_id: int
    breached: bool
    warned: bool
    warning_min: float = float("nan")
    breach_min: float = float("nan")
    alarm_min: float = float("nan")
    false_alarm: bool = False

    def to_dict(self) -> dict:
        return {
            "event_id": int(self.event_id),
            "breached": bool(self.breached),
            "warned": bool(self.warned),
            "warning_min": float(self.warning_min),
            "breach_min": float(self.breach_min),
            "alarm_min": float(self.alarm_min),
            "false_alarm": bool(self.false_alarm),
        }


def early_warning_analysis(
    df: pd.DataFrame,
    proba: np.ndarray,
    threshold: float,
    time_col: str = "t_min",
    persistence_samples: int = 1,
) -> Dict[str, object]:
    """
    How much notice does the model actually give?

    For every event:

    * Find the first sample where basis weight leaves the 2.5% band -- the
      breach.
    * Find the first sample where the model raises an alarm (p >= threshold)
      **while the sheet is still in spec** and before that breach.
    * Warning time is the gap between them.

    Events that never breach contribute to the false alarm rate instead. Only
    warned-and-breached events contribute to the warning-time statistics, so
    the median is not silently diluted by misses -- the detection rate reports
    those separately.
    """
    if len(df) != len(proba):
        raise ValueError("df and proba must be the same length")

    work = df[["event_id", time_col, "currently_off_spec"]].copy()
    work["proba"] = np.asarray(proba, dtype=float)
    work["alarm"] = confirm_alarms(
        work["proba"].to_numpy() >= threshold,
        persistence_samples,
        group=work["event_id"].to_numpy(),
    )

    results: List[WarningResult] = []

    for event_id, group in work.groupby("event_id", sort=True):
        group = group.sort_values(time_col)
        t = group[time_col].to_numpy()
        off_spec = group["currently_off_spec"].to_numpy().astype(bool)
        alarm = group["alarm"].to_numpy().astype(bool)

        if off_spec.any():
            breach_idx = int(np.argmax(off_spec))
            breach_t = float(t[breach_idx])
            # Alarms that are early (before the breach) and non-trivial
            # (raised while still inside spec).
            eligible = alarm & ~off_spec
            eligible[breach_idx:] = False
            if eligible.any():
                alarm_idx = int(np.argmax(eligible))
                results.append(
                    WarningResult(
                        event_id=int(event_id), breached=True, warned=True,
                        warning_min=breach_t - float(t[alarm_idx]),
                        breach_min=breach_t, alarm_min=float(t[alarm_idx]),
                    )
                )
            else:
                results.append(
                    WarningResult(
                        event_id=int(event_id), breached=True, warned=False,
                        breach_min=breach_t,
                    )
                )
        else:
            fired = bool((alarm & ~off_spec).any())
            alarm_t = (
                float(t[int(np.argmax(alarm & ~off_spec))]) if fired
                else float("nan")
            )
            results.append(
                WarningResult(
                    event_id=int(event_id), breached=False, warned=False,
                    false_alarm=fired, alarm_min=alarm_t,
                )
            )

    breached = [r for r in results if r.breached]
    warned = [r for r in breached if r.warned]
    clean = [r for r in results if not r.breached]
    times = np.array([r.warning_min for r in warned], dtype=float)

    return {
        "n_events": len(results),
        "n_breaching_events": len(breached),
        "n_clean_events": len(clean),
        "n_warned_events": len(warned),
        "detection_rate": (len(warned) / len(breached)) if breached else float("nan"),
        "median_warning_min": float(np.median(times)) if times.size else float("nan"),
        "mean_warning_min": float(np.mean(times)) if times.size else float("nan"),
        "p25_warning_min": float(np.percentile(times, 25)) if times.size else float("nan"),
        "p75_warning_min": float(np.percentile(times, 75)) if times.size else float("nan"),
        "max_warning_min": float(times.max()) if times.size else float("nan"),
        "false_alarm_event_rate": (
            sum(r.false_alarm for r in clean) / len(clean) if clean else float("nan")
        ),
        "horizon_min": float(C.RISK_HORIZON_MIN),
        "persistence_samples": int(persistence_samples),
        "persistence_seconds": float(persistence_samples * C.DT_S),
        "per_event": [r.to_dict() for r in results],
    }


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    df: pd.DataFrame,
    proba: np.ndarray,
    threshold: float,
    exclude_currently_off_spec: bool = True,
    persistence_samples: int = 1,
) -> Dict[str, object]:
    """
    Full evaluation on one split.

    Classification metrics use the in-spec population only (see module
    docstring). Early-warning analysis necessarily uses all rows, because it
    needs to see the breach itself to measure the gap to it.
    """
    y_all = df["y_breach"].to_numpy().astype(int)
    proba = np.asarray(proba, dtype=float)

    if exclude_currently_off_spec:
        keep = (df["currently_off_spec"].to_numpy() == 0)
    else:
        keep = np.ones(len(df), dtype=bool)

    # Alarm confirmation is applied on the full series, grouped by event, then
    # subset -- so the persistence window never straddles an event boundary or
    # a gap created by the in-spec filter.
    confirmed = confirm_alarms(
        proba >= threshold, persistence_samples, group=df["event_id"].to_numpy()
    )
    effective_proba = np.where(confirmed, np.maximum(proba, threshold),
                               np.minimum(proba, threshold - 1e-9))

    metrics = classification_metrics(
        y_all[keep], effective_proba[keep], threshold
    )
    # PR-AUC must be measured on the raw scores: it is threshold-independent and
    # debouncing is a threshold-stage decision, not a change to the ranking.
    metrics["pr_auc"] = float(average_precision_score(y_all[keep], proba[keep]))
    metrics["evaluated_on"] = (
        "in-spec rows only" if exclude_currently_off_spec else "all rows"
    )
    metrics["n_rows_excluded"] = int((~keep).sum())
    metrics["persistence_samples"] = int(persistence_samples)

    warning = early_warning_analysis(
        df, proba, threshold, persistence_samples=persistence_samples
    )

    return {
        "classification": metrics,
        "early_warning": {
            k: v for k, v in warning.items() if k != "per_event"
        },
        "per_event_warning": warning["per_event"],
        "confusion_matrix": confusion_matrix_dict(
            y_all[keep], effective_proba[keep], threshold
        ),
    }


def metrics_table(results: Dict[str, dict], split: str = "validation") -> pd.DataFrame:
    """Side-by-side comparison table across models for the report and UI."""
    rows = []
    for name, res in results.items():
        clf = res[split]["classification"]
        warn = res[split]["early_warning"]
        rows.append(
            {
                "model": name,
                "pr_auc": clf["pr_auc"],
                "precision": clf["precision"],
                "recall": clf["recall"],
                "f1": clf["f1"],
                "fpr": clf["false_positive_rate"],
                "median_warning_min": warn["median_warning_min"],
                "mean_warning_min": warn["mean_warning_min"],
                "detection_rate": warn["detection_rate"],
                "false_alarm_event_rate": warn["false_alarm_event_rate"],
            }
        )
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(
        drop=True
    )
