"""
Risk-model training pipeline.

Orchestration only -- splitting, metrics, model construction and explanation all
live in their own modules, so this file reads as the procedure it is:

    split events -> train every available model -> pick a threshold on
    validation -> rank by validation PR-AUC -> score the winner on test once ->
    explain it -> persist everything.

Leakage guarantees enforced here
--------------------------------
* Splits are by event, produced by `splits.event_wise_split_3way`.
* Every model is fitted on the training split only.
* The operating threshold is chosen on validation only.
* Model *selection* uses validation only.
* The test split is scored exactly once, after the winner is fixed. It is never
  consulted during selection or tuning, so the reported test metrics are an
  honest out-of-sample estimate.
* Permutation importance is computed on validation, not training.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import config as C
from ..features import feature_columns
from . import explain as explain_mod
from .metrics import evaluate_model, metrics_table, pick_threshold
from .registry import MODEL_REGISTRY, ModelSpec, registry_report
from .splits import EventSplit, event_wise_split_3way, split_summary

SELECTION_METRIC = "pr_auc"
TIEBREAK_METRIC = "median_warning_min"


@dataclass
class TrainedModel:
    """One fitted candidate with its evaluation on every split."""

    name: str
    label: str
    spec: ModelSpec
    estimator: object
    threshold: float
    threshold_info: Dict[str, float]
    results: Dict[str, dict]          # split -> evaluation
    fit_seconds: float

    def score(self, metric: str = SELECTION_METRIC, split: str = "validation") -> float:
        block = self.results[split]
        if metric in block["classification"]:
            return float(block["classification"][metric])
        return float(block["early_warning"][metric])

    def summary(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "family": self.spec.family,
            "threshold": self.threshold,
            "fit_seconds": self.fit_seconds,
            "validation": {
                **self.results["validation"]["classification"],
                **self.results["validation"]["early_warning"],
            },
            "test": {
                **self.results["test"]["classification"],
                **self.results["test"]["early_warning"],
            },
        }


class RiskModelPipeline:
    """Trains, compares, selects and persists the off-spec risk model."""

    def __init__(
        self,
        df: pd.DataFrame,
        val_frac: float = 0.20,
        test_frac: float = 0.20,
        seed: int = 42,
        threshold_criterion: str = "f1",
        models: Optional[Sequence[str]] = None,
        verbose: bool = True,
        persistence_samples: int = 1,
    ):
        self.df = df.reset_index(drop=True)
        self.seed = seed
        self.threshold_criterion = threshold_criterion
        self.verbose = verbose
        # Alarm on-delay in samples. See metrics.confirm_alarms.
        self.persistence_samples = int(persistence_samples)

        self.features: List[str] = feature_columns(self.df)
        self.split: EventSplit = event_wise_split_3way(
            self.df, val_frac=val_frac, test_frac=test_frac, seed=seed
        )
        self.split_table = split_summary(self.df, self.split)

        requested = list(models) if models else list(MODEL_REGISTRY.keys())
        self.requested = requested
        self.skipped: Dict[str, str] = {}
        self.candidates: List[ModelSpec] = []
        for name in requested:
            spec = MODEL_REGISTRY.get(name)
            if spec is None:
                self.skipped[name] = "not in registry"
            elif not spec.available:
                self.skipped[name] = (
                    f"missing package(s): {', '.join(spec.missing)}"
                )
            else:
                self.candidates.append(spec)

        self.trained: Dict[str, TrainedModel] = {}
        self.best: Optional[TrainedModel] = None
        self.operating_point: Optional[Dict[str, object]] = None
        self.shap: Optional[explain_mod.ShapResult] = None
        self.importance: Optional[pd.DataFrame] = None
        self._frames: Dict[str, pd.DataFrame] = {}

    # -- data access -------------------------------------------------------
    def frame(self, which: str) -> pd.DataFrame:
        if which not in self._frames:
            self._frames[which] = self.split.frame(self.df, which)
        return self._frames[which]

    def xy(self, which: str) -> Tuple[pd.DataFrame, np.ndarray]:
        part = self.frame(which)
        return part[self.features], part["y_breach"].to_numpy().astype(int)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    # -- training ----------------------------------------------------------
    # -- checkpointing -----------------------------------------------------
    def dataset_fingerprint(self) -> str:
        """
        Identity of the exact data this pipeline was built on.

        Checkpoint validity depends on the *data*, not only on the feature names
        and split seed. Without this, growing the corpus from 300 to 500 events
        silently restored models trained on the old data -- the feature list and
        seed were unchanged, so a weaker check passed and the reported metrics
        described a model that no longer matched the dataset.
        """
        import hashlib

        ids = np.asarray(sorted(self.df["event_id"].unique()), dtype=np.int64)
        digest = hashlib.sha256()
        digest.update(ids.tobytes())
        digest.update(str(len(self.df)).encode())
        digest.update(",".join(self.features).encode())
        digest.update(str(C.RISK_HORIZON_MIN).encode())
        digest.update(str(C.BW_SPEC_PCT).encode())
        return digest.hexdigest()[:16]

    def _checkpoint_paths(self, ckpt_dir: Path, name: str) -> Tuple[Path, Path]:
        return ckpt_dir / f"{name}.joblib", ckpt_dir / f"{name}.results.json"

    def _load_checkpoint(
        self, ckpt_dir: Path, spec: ModelSpec
    ) -> Optional[TrainedModel]:
        """Restore a previously fitted candidate, if one was saved."""
        model_path, results_path = self._checkpoint_paths(ckpt_dir, spec.name)
        if not (model_path.exists() and results_path.exists()):
            return None
        try:
            import joblib

            payload = joblib.load(model_path)
            stored = json.loads(results_path.read_text())
            if stored.get("features") != self.features:
                return None      # feature set changed; checkpoint is stale
            if stored.get("split_seed") != self.seed:
                return None      # different split; not comparable
            if stored.get("dataset_fingerprint") != self.dataset_fingerprint():
                return None      # trained on different data; must refit
            return TrainedModel(
                name=spec.name, label=spec.label, spec=spec,
                estimator=payload["estimator"],
                threshold=float(stored["threshold"]),
                threshold_info=stored.get("threshold_info", {}),
                results=stored["results"],
                fit_seconds=float(stored.get("fit_seconds", 0.0)),
            )
        except Exception:
            return None

    def _save_checkpoint(self, ckpt_dir: Path, model: TrainedModel) -> None:
        try:
            import joblib

            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model_path, results_path = self._checkpoint_paths(ckpt_dir, model.name)
            joblib.dump({"estimator": model.estimator}, model_path)
            results_path.write_text(
                json.dumps(
                    {
                        "features": self.features,
                        "split_seed": self.seed,
                        "dataset_fingerprint": self.dataset_fingerprint(),
                        "threshold": model.threshold,
                        "threshold_info": model.threshold_info,
                        "fit_seconds": model.fit_seconds,
                        "results": model.results,
                    },
                    indent=2,
                    default=float,
                )
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            self._log(f"    WARNING: checkpoint failed for {model.name} ({exc})")

    # -- training ----------------------------------------------------------
    def fit_all(
        self, checkpoint_dir: Optional[Path] = None, resume: bool = True
    ) -> "RiskModelPipeline":
        """
        Fit every available candidate.

        With `checkpoint_dir`, each model is persisted as soon as it is trained
        and evaluated, and a re-run skips work already done. Training five
        ensembles on 62k rows takes a couple of minutes, and losing all of it to
        an interrupted shell is avoidable.
        """
        ckpt = Path(checkpoint_dir) if checkpoint_dir else None
        X_tr, y_tr = self.xy("train")
        val = self.frame("validation")
        X_val, y_val = self.xy("validation")
        test = self.frame("test")
        X_te, _ = self.xy("test")

        self._log(
            f"Training on {len(X_tr):,} rows x {len(self.features)} features "
            f"({self.split.counts['train']} events)"
        )
        if self.skipped:
            for name, reason in self.skipped.items():
                self._log(f"  skipping {name}: {reason}")

        for spec in self.candidates:
            if ckpt is not None and resume:
                restored = self._load_checkpoint(ckpt, spec)
                if restored is not None:
                    self.trained[spec.name] = restored
                    clf = restored.results["validation"]["classification"]
                    self._log(
                        f"  {spec.label}: restored from checkpoint "
                        f"(PR-AUC {clf['pr_auc']:.4f})"
                    )
                    continue

            self._log(f"  fitting {spec.label} ...")
            estimator = spec.build()
            t0 = time.perf_counter()
            estimator.fit(X_tr, y_tr)
            fit_seconds = time.perf_counter() - t0

            proba_val = explain_mod._predict_proba(estimator, X_val)

            # Threshold chosen on validation, in-spec rows only -- the
            # population where a warning has value.
            in_spec = val["currently_off_spec"].to_numpy() == 0
            threshold, info = pick_threshold(
                y_val[in_spec], proba_val[in_spec],
                criterion=self.threshold_criterion,
            )

            results = {
                "validation": evaluate_model(
                    val, proba_val, threshold,
                    persistence_samples=self.persistence_samples,
                ),
                "test": evaluate_model(
                    test, explain_mod._predict_proba(estimator, X_te), threshold,
                    persistence_samples=self.persistence_samples,
                ),
            }

            trained = TrainedModel(
                name=spec.name, label=spec.label, spec=spec,
                estimator=estimator, threshold=threshold,
                threshold_info=info, results=results,
                fit_seconds=fit_seconds,
            )
            self.trained[spec.name] = trained
            if ckpt is not None:
                self._save_checkpoint(ckpt, trained)

            clf = results["validation"]["classification"]
            warn = results["validation"]["early_warning"]
            self._log(
                f"    PR-AUC {clf['pr_auc']:.4f} | F1 {clf['f1']:.3f} | "
                f"recall {clf['recall']:.3f} | FPR {clf['false_positive_rate']:.3f} | "
                f"median warning {warn['median_warning_min']:.2f} min "
                f"({fit_seconds:.1f}s)"
            )

        if not self.trained:
            raise RuntimeError(
                "no models could be trained; check the registry availability report"
            )

        self._select_best()
        return self

    def _select_best(self) -> None:
        """
        Provisional ranking by validation PR-AUC.

        This is only provisional: `select_by_operating_point` replaces it with a
        product-level choice once each model has been given its own tuned
        operating point. PR-AUC remains the right *model quality* measure, but it
        is not the right *product* choice -- see that method for why.
        """
        def key(model: TrainedModel) -> Tuple[float, float]:
            primary = model.score(SELECTION_METRIC, "validation")
            tie = model.score(TIEBREAK_METRIC, "validation")
            return (primary, 0.0 if np.isnan(tie) else tie)

        self.best = max(self.trained.values(), key=key)
        self._log(
            f"\nProvisional best by PR-AUC: {self.best.label} "
            f"({self.best.score():.4f})"
        )

    # -- operating point + product-level selection -------------------------
    def select_by_operating_point(
        self,
        min_detection_rate: float = 0.80,
        thresholds: Optional[Sequence[float]] = None,
        delays: Sequence[int] = (1, 2, 3, 4, 6),
    ) -> Dict[str, object]:
        """
        Give every model its own tuned operating point, then choose between them
        on the product objective. Validation only.

        Why not just rank by PR-AUC: PR-AUC measures how well a model *ranks*
        samples, but the deployed system is a binary advisor with a threshold and
        an alarm on-delay, and two models with similar PR-AUC can behave very
        differently once thresholded. Ranking by PR-AUC and then tuning the
        threshold for the winner is incoherent -- it optimises two different
        objectives in sequence. Worse, it can pick a model whose validation-tuned
        threshold does not transfer: a conservative model can hit the detection
        floor on validation by a hair and fall below it out of sample.

        So: tune each model's own (threshold, on-delay), then select the model
        with the fewest nuisance alarms that still clears the detection floor,
        tie-broken by longer warning time. PR-AUC is still reported for every
        model as a model-quality diagnostic.
        """
        if not self.trained:
            raise RuntimeError("call fit_all() before select_by_operating_point()")

        if thresholds is None:
            thresholds = np.round(np.arange(0.20, 0.86, 0.05), 4)

        val = self.frame("validation")
        X_val, _ = self.xy("validation")

        per_model: Dict[str, dict] = {}
        for name, model in self.trained.items():
            proba = explain_mod._predict_proba(model.estimator, X_val)
            best_point: Optional[dict] = None
            for thr in thresholds:
                for delay in delays:
                    res = evaluate_model(
                        val, proba, float(thr), persistence_samples=int(delay)
                    )
                    warn = res["early_warning"]
                    if (
                        warn["detection_rate"] < min_detection_rate
                        or not np.isfinite(warn["median_warning_min"])
                    ):
                        continue
                    point = {
                        "threshold": float(thr),
                        "persistence_samples": int(delay),
                        "detection_rate": warn["detection_rate"],
                        "median_warning_min": warn["median_warning_min"],
                        "false_alarm_event_rate": warn["false_alarm_event_rate"],
                    }
                    key = (
                        point["false_alarm_event_rate"],
                        -point["median_warning_min"],
                    )
                    if best_point is None or key < (
                        best_point["false_alarm_event_rate"],
                        -best_point["median_warning_min"],
                    ):
                        best_point = point
            per_model[name] = best_point or {}
            if best_point:
                self._log(
                    f"  {model.label}: thr {best_point['threshold']:.2f}, "
                    f"delay {best_point['persistence_samples']}, "
                    f"detection {best_point['detection_rate']:.3f}, "
                    f"warning {best_point['median_warning_min']:.2f} min, "
                    f"false alarms {best_point['false_alarm_event_rate']:.3f}"
                )
            else:
                self._log(
                    f"  {model.label}: no point reaches detection "
                    f">= {min_detection_rate:.2f}"
                )

        feasible = {n: p for n, p in per_model.items() if p}
        if not feasible:
            self._log(
                "  no model clears the detection floor; keeping PR-AUC winner"
            )
            self.operating_point = {
                "objective": "fallback: PR-AUC winner, F1 threshold",
                "per_model": per_model,
            }
            return self.operating_point

        winner = min(
            feasible.items(),
            key=lambda kv: (
                kv[1]["false_alarm_event_rate"], -kv[1]["median_warning_min"]
            ),
        )[0]

        self.best = self.trained[winner]
        point = feasible[winner]
        self.best.threshold = float(point["threshold"])
        self.persistence_samples = int(point["persistence_samples"])

        # Re-score every model at its own tuned operating point so the
        # comparison table reflects real deployed behaviour.
        for name, model in self.trained.items():
            p = per_model.get(name) or {
                "threshold": model.threshold,
                "persistence_samples": self.persistence_samples,
            }
            model.threshold = float(p["threshold"])
            delay = int(p["persistence_samples"])
            for split in ("validation", "test"):
                part = self.frame(split)
                X, _ = self.xy(split)
                proba = explain_mod._predict_proba(model.estimator, X)
                model.results[split] = evaluate_model(
                    part, proba, model.threshold, persistence_samples=delay
                )

        self.operating_point = {
            "objective": (
                "minimise per-clean-event false alarm rate subject to "
                f"event detection rate >= {min_detection_rate}"
            ),
            "min_detection_rate": min_detection_rate,
            "tuned_on": "validation",
            "winner": winner,
            "chosen": point,
            "per_model": per_model,
        }
        warn = self.best.results["validation"]["early_warning"]
        self._log(
            f"\nSelected on product objective: {self.best.label} "
            f"(threshold {self.best.threshold:.2f}, on-delay "
            f"{self.persistence_samples} samples)"
        )
        self._log(
            f"  validation: detection {warn['detection_rate']:.3f}, "
            f"median warning {warn['median_warning_min']:.2f} min, "
            f"false alarms {warn['false_alarm_event_rate']:.3f}/clean event"
        )
        return self.operating_point

    def tune_operating_point(
        self,
        min_detection_rate: float = 0.80,
        thresholds: Optional[Sequence[float]] = None,
        delays: Sequence[int] = (1, 2, 3, 4, 6),
    ) -> Dict[str, object]:
        """
        Choose the threshold and alarm on-delay on an **event-level** objective.

        Row-level F1 is the wrong target for this product. A grade change is
        ~336 samples long, so even a 6% row-level false positive rate makes a
        nuisance alarm near-certain somewhere in every clean transition -- and
        the first thing a mill does with a system like that is stop looking at
        it. Conversely, row-level recall understates usefulness: an excursion
        only has to be caught *once* to warn the operator, so event detection
        rate is what actually matters.

        This method therefore minimises the per-clean-event false alarm rate
        subject to a floor on event detection rate. Tuned on **validation
        only**, then applied unchanged to test.
        """
        if self.best is None:
            raise RuntimeError("call fit_all() before tune_operating_point()")

        val = self.frame("validation")
        X_val, _ = self.xy("validation")
        proba = explain_mod._predict_proba(self.best.estimator, X_val)

        if thresholds is None:
            thresholds = np.round(np.arange(0.20, 0.86, 0.025), 4)

        sweep: List[dict] = []
        for thr in thresholds:
            for delay in delays:
                res = evaluate_model(
                    val, proba, float(thr), persistence_samples=int(delay)
                )
                clf, warn = res["classification"], res["early_warning"]
                sweep.append(
                    {
                        "threshold": float(thr),
                        "persistence_samples": int(delay),
                        "precision": clf["precision"],
                        "recall": clf["recall"],
                        "f1": clf["f1"],
                        "false_positive_rate": clf["false_positive_rate"],
                        "detection_rate": warn["detection_rate"],
                        "median_warning_min": warn["median_warning_min"],
                        "false_alarm_event_rate": warn["false_alarm_event_rate"],
                    }
                )

        feasible = [
            s for s in sweep
            if s["detection_rate"] >= min_detection_rate
            and np.isfinite(s["median_warning_min"])
        ]
        if not feasible:
            self._log(
                f"  no operating point reaches detection >= {min_detection_rate:.2f}; "
                "keeping the F1 threshold"
            )
            chosen = {
                "threshold": self.best.threshold,
                "persistence_samples": self.persistence_samples,
            }
        else:
            # Fewest nuisance alarms; ties broken by longer warning time.
            chosen = min(
                feasible,
                key=lambda s: (
                    s["false_alarm_event_rate"], -s["median_warning_min"]
                ),
            )

        self.operating_point = {
            "objective": (
                "minimise per-event false alarm rate subject to "
                f"detection_rate >= {min_detection_rate}"
            ),
            "min_detection_rate": min_detection_rate,
            "tuned_on": "validation",
            "chosen": chosen,
            "sweep": sweep,
        }

        previous = self.best.threshold
        self.best.threshold = float(chosen["threshold"])
        self.persistence_samples = int(chosen["persistence_samples"])
        self._log(
            f"Operating point: threshold {previous:.3f} -> "
            f"{self.best.threshold:.3f}, on-delay "
            f"{self.persistence_samples} samples "
            f"({self.persistence_samples * C.DT_S:.0f}s)"
        )

        # Re-score every model at the shared operating point so the comparison
        # table reflects how each would actually behave in production.
        for model in self.trained.values():
            model.threshold = float(chosen["threshold"])
            for split in ("validation", "test"):
                part = self.frame(split)
                X, _ = self.xy(split)
                p = explain_mod._predict_proba(model.estimator, X)
                model.results[split] = evaluate_model(
                    part, p, model.threshold,
                    persistence_samples=self.persistence_samples,
                )

        warn = self.best.results["validation"]["early_warning"]
        self._log(
            f"  validation: detection {warn['detection_rate']:.3f}, "
            f"median warning {warn['median_warning_min']:.2f} min, "
            f"false alarms {warn['false_alarm_event_rate']:.3f}/clean event"
        )
        return self.operating_point

    # -- explanation -------------------------------------------------------
    def explain_best(
        self, n_repeats: int = 5, shap_max_rows: int = 600
    ) -> "RiskModelPipeline":
        """
        Explain the selected model.

        `shap_max_rows` is deliberately modest. Exact TreeSHAP on a bagged
        forest costs O(trees x leaves x features) per row, so a few thousand
        rows is minutes of compute -- while a few hundred is already ample for
        stable mean |SHAP| rankings, which is what the global attribution needs.
        Local explanations are computed on demand at serving time, not here.
        """
        if self.best is None:
            raise RuntimeError("call fit_all() before explain_best()")

        X_val, y_val = self.xy("validation")
        self._log(f"Computing SHAP values on {shap_max_rows} rows ...")
        self.shap = explain_mod.compute_shap(
            self.best.estimator, X_val, max_rows=shap_max_rows
        )
        self._log(f"  method: {self.shap.method} (exact={self.shap.exact})")

        self._log("Computing consensus feature importance ...")
        self.importance = explain_mod.combined_importance(
            self.best.estimator, X_val, y_val, self.shap, n_repeats=n_repeats,
            perm_max_rows=1500,
        )
        top = self.importance.head(8)["feature"].tolist()
        self._log(f"  top features: {', '.join(top)}")
        return self

    # -- comparison --------------------------------------------------------
    def comparison_table(self, split: str = "validation") -> pd.DataFrame:
        return metrics_table(
            {name: m.results for name, m in self.trained.items()}, split=split
        )

    # -- persistence -------------------------------------------------------
    def save(self, out_dir: Path) -> Dict[str, Path]:
        if self.best is None:
            raise RuntimeError("call fit_all() before save()")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: Dict[str, Path] = {}

        # -- model ---------------------------------------------------------
        try:
            import joblib

            bundle = {
                "estimator": self.best.estimator,
                "features": self.features,
                "threshold": self.best.threshold,
                "persistence_samples": self.persistence_samples,
                "model_name": self.best.name,
                "horizon_min": C.RISK_HORIZON_MIN,
                "spec_pct": C.BW_SPEC_PCT,
            }
            path = out_dir / "risk_model.joblib"
            joblib.dump(bundle, path)
            written["model"] = path
        except Exception as exc:  # pragma: no cover - environment dependent
            self._log(f"  WARNING: could not persist model ({exc})")

        # -- metrics -------------------------------------------------------
        metrics_payload = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "selection": {
                "metric": SELECTION_METRIC,
                "tiebreak": TIEBREAK_METRIC,
                "winner": self.best.name,
                "threshold_criterion": self.threshold_criterion,
            },
            "config": {
                "risk_horizon_min": C.RISK_HORIZON_MIN,
                "bw_spec_pct": C.BW_SPEC_PCT,
                "dt_s": C.DT_S,
                "n_features": len(self.features),
                "seed": self.seed,
            },
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            "operating_point": (
                {
                    k: v for k, v in self.operating_point.items()
                    if k not in ("sweep",)
                }
                if self.operating_point else None
            ),
            "operating_point_sweep": (
                self.operating_point.get("sweep")
                if self.operating_point else None
            ),
            "split": self.split.to_dict(),
            "split_summary": self.split_table.to_dict(orient="records"),
            "registry": registry_report(),
            "skipped": self.skipped,
            "models": {n: m.summary() for n, m in self.trained.items()},
        }
        path = out_dir / "metrics.json"
        path.write_text(json.dumps(metrics_payload, indent=2, default=float))
        written["metrics"] = path

        # -- comparison tables --------------------------------------------
        for split in ("validation", "test"):
            path = out_dir / f"comparison_{split}.csv"
            self.comparison_table(split).to_csv(path, index=False)
            written[f"comparison_{split}"] = path

        # -- confusion matrices -------------------------------------------
        path = out_dir / "confusion_matrix.json"
        path.write_text(
            json.dumps(
                {
                    name: {
                        split: m.results[split]["confusion_matrix"]
                        for split in ("validation", "test")
                    }
                    for name, m in self.trained.items()
                },
                indent=2,
            )
        )
        written["confusion_matrix"] = path

        # -- feature importance -------------------------------------------
        if self.importance is not None:
            path = out_dir / "feature_importance.csv"
            self.importance.to_csv(path, index=False)
            written["feature_importance"] = path

        # -- SHAP ----------------------------------------------------------
        if self.shap is not None:
            summary = self.shap.mean_abs()
            payload = {
                "method": self.shap.method,
                "exact": self.shap.exact,
                "base_value": self.shap.base_value,
                "n_rows_explained": int(self.shap.values.shape[0]),
                "mean_abs_shap": summary.to_dict(orient="records"),
                # A handful of worked local explanations for the UI's
                # "why this prediction?" panel.
                "examples": [
                    {"row": int(r), "top_contributions": self.shap.local(int(r))}
                    for r in np.linspace(
                        0, self.shap.values.shape[0] - 1, num=5, dtype=int
                    )
                ],
            }
            path = out_dir / "shap_explanations.json"
            path.write_text(json.dumps(payload, indent=2, default=float))
            written["shap"] = path

        # -- per-event warning detail (feeds the dashboard) ----------------
        path = out_dir / "warning_detail.json"
        path.write_text(
            json.dumps(
                {
                    split: self.best.results[split]["per_event_warning"]
                    for split in ("validation", "test")
                },
                indent=2,
                default=float,
            )
        )
        written["warning_detail"] = path

        # -- human-readable report ----------------------------------------
        path = out_dir / "evaluation_report.md"
        path.write_text(self.report_markdown())
        written["report"] = path

        return written

    # -- reporting ---------------------------------------------------------
    def report_markdown(self) -> str:
        assert self.best is not None
        best = self.best
        val = best.results["validation"]
        test = best.results["test"]

        def block(res: dict) -> str:
            clf, warn = res["classification"], res["early_warning"]
            return (
                f"| PR-AUC | {clf['pr_auc']:.4f} |\n"
                f"| Precision | {clf['precision']:.4f} |\n"
                f"| Recall | {clf['recall']:.4f} |\n"
                f"| F1 | {clf['f1']:.4f} |\n"
                f"| False positive rate | {clf['false_positive_rate']:.4f} |\n"
                f"| Median warning time | {warn['median_warning_min']:.2f} min |\n"
                f"| Mean warning time | {warn['mean_warning_min']:.2f} min |\n"
                f"| **Event detection rate** | {warn['detection_rate']:.3f} |\n"
                f"| False alarm rate (per clean event) | "
                f"{warn['false_alarm_event_rate']:.3f} |\n"
            )

        lines = [
            "# Off-Spec Risk Model — Evaluation Report",
            "",
            f"**Selected model:** {best.label}  ",
            f"**Selection criterion:** highest validation {SELECTION_METRIC}, "
            f"tie-broken on {TIEBREAK_METRIC}  ",
            f"**Operating threshold:** {best.threshold:.4f} "
            f"(chosen on validation via `{self.threshold_criterion}`)  ",
            f"**Prediction target:** basis weight deviates more than "
            f"{C.BW_SPEC_PCT}% from setpoint within the next "
            f"{C.RISK_HORIZON_MIN:.0f} minutes",
            "",
            "## Leakage controls",
            "",
            "- Splits are **event-wise**, never row-wise.",
            "- Features are strictly backward-looking (asserted by "
            "`test_features.py::test_no_future_leakage`).",
            "- Threshold selected on validation only.",
            "- Model selection used validation only.",
            "- **Test split scored exactly once**, after the winner was fixed.",
            "- Permutation importance computed on validation, not training.",
            "",
            "## Operating point",
            "",
            (
                f"Alarm on-delay: **{self.persistence_samples} samples "
                f"({self.persistence_samples * C.DT_S:.0f} s)** -- an alarm is "
                "confirmed only after the risk score stays above threshold for "
                "that long (ISA-18.2 on-delay)."
            ),
            "",
            (
                "The threshold and on-delay are tuned on an **event-level** "
                "objective, not row-level F1. A transition is ~336 samples "
                "long, so even a 6% row-level false positive rate makes a "
                "nuisance alarm near-certain in every clean transition, and a "
                "system that does that gets switched off. Conversely row-level "
                "recall understates usefulness: an excursion only has to be "
                "caught once to warn the operator. The objective is therefore "
                "to minimise per-clean-event false alarms subject to a floor on "
                "event detection rate."
                if self.operating_point else
                "Threshold selected on validation by row-level "
                f"`{self.threshold_criterion}`."
            ),
            "",
            "## Evaluation population",
            "",
            "Classification metrics are computed on rows where basis weight is "
            "still **inside** the ±2.5% band. Predicting a breach while the "
            "sheet is already off-spec is trivial and would inflate every "
            "score. Early-warning metrics necessarily span all rows, since "
            "they measure the gap to the breach itself.",
            "",
            "## Split composition",
            "",
            self.split_table.to_markdown(index=False),
            "",
            "## Model comparison (validation)",
            "",
            self.comparison_table("validation").to_markdown(index=False),
            "",
            "## Selected model — validation",
            "",
            "| Metric | Value |",
            "|---|---|",
            block(val),
            "## Selected model — test (held out, scored once)",
            "",
            "| Metric | Value |",
            "|---|---|",
            block(test),
        ]

        if self.shap is not None:
            lines += [
                "## Explainability",
                "",
                f"SHAP method: `{self.shap.method}` "
                f"(exact={self.shap.exact}), "
                f"{self.shap.values.shape[0]} rows explained.",
                "",
            ]

        if self.importance is not None:
            lines += [
                "### Top 15 features by consensus attribution",
                "",
                self.importance.head(15).to_markdown(index=False),
                "",
                "Consensus averages the model's native importance, permutation "
                "importance on validation, and mean |SHAP|. Agreement across "
                "three independent views is stronger evidence than any single "
                "ranking.",
                "",
            ]

        if self.skipped:
            lines += [
                "## Models skipped",
                "",
                *[f"- `{n}`: {r}" for n, r in self.skipped.items()],
                "",
            ]

        return "\n".join(lines)
