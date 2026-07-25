"""
Lagged correlation discovery + novelty scoring.

Surfaces relationships between process tags directly from the historical
corpus, without being told in advance which loops matter -- then classifies
each against `grades.KNOWN_LOOPS`: a relationship already in that graph is
what the existing QCS already models (not news to an operator); one that
isn't is a genuine discovery, tagged `Source.CORRELATION_DISCOVERY` and
surfaced for the dashboard's Correlation Explorer panel (deliverable 3).

Method
------
For each ordered tag pair `(cause, effect)`, sweep non-negative lags (cause
leads effect by `lag` samples) and compute, at each lag, the Pearson
correlation between `cause[t]` and `effect[t + lag]`, pooled across many
events. Sweeping only non-negative lags is not a limitation: testing every
*ordered* pair already covers both directions -- what a negative lag on
`(A, B)` would show up as is exactly the positive-lag result for `(B, A)`.

Pairs are pooled *within* each event, never across event boundaries: pairing
`cause[t]` from one event with `effect[t + lag]` from a different event would
correlate two unrelated disturbance realisations (each event's
Ornstein-Uhlenbeck drift is its own independent draw, D11) and manufacture
signal that is not there.

The best-|correlation| lag is kept, plus mutual information at that lag
(`sklearn.feature_selection.mutual_info_regression`) as a second, nonlinear-
relationship-robust measure -- a real but non-monotonic coupling should not
be dismissed just because Pearson r is near zero. MI is computed on a capped
subsample (`mi_max_samples`) since its k-NN estimator does not scale to the
full pooled sample the way a Pearson sweep does -- the same "a few thousand
rows is ample for a stable estimate" posture used for SHAP and permutation
importance elsewhere in this codebase.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config as C
from .config import Source
from .events import EventResult
from .grades import ALL_TAGS, is_known_relationship

DEFAULT_MAX_LAG_MIN: float = 4.0
DEFAULT_MIN_ABS_CORRELATION: float = 0.30
DEFAULT_MI_MAX_SAMPLES: int = 4000


SeriesByTag = Dict[str, List[np.ndarray]]


def series_by_tag_from_events(
    events: Sequence[EventResult], tags: Sequence[str] = ALL_TAGS
) -> SeriesByTag:
    """Per-event trajectories, keyed by tag -- the input `discover_correlations`
    expects. Kept as a list of per-event arrays (not concatenated) so lag
    pairing can respect event boundaries."""
    return {tag: [ev.series[tag] for ev in events] for tag in tags}


def series_by_tag_from_dataset(
    cube: np.ndarray,
    tags_in_cube: Sequence[str],
    meta: dict,
    tags: Sequence[str] = ALL_TAGS,
    max_events: Optional[int] = None,
    seed: int = 42,
) -> SeriesByTag:
    """
    The same per-tag structure, built directly from a persisted dataset
    (`events.load_dataset`'s return) without re-simulating -- the training
    corpus is already on disk, so discovery should read it rather than pay to
    regenerate it.

    `max_events` subsamples for interactive speed, exactly like the SHAP and
    permutation-importance sampling in `ml/pipeline.py`: correlation discovery
    over the full multi-thousand-event corpus is unnecessary precision at the
    cost of an unresponsive dashboard.
    """
    n_events = cube.shape[0]
    idx = np.arange(n_events)
    if max_events is not None and max_events < n_events:
        idx = np.random.default_rng(seed).choice(n_events, size=max_events, replace=False)
        idx.sort()

    col = {tag: tags_in_cube.index(tag) for tag in tags}
    return {
        tag: [cube[i, :, col[tag]].astype(np.float64) for i in idx]
        for tag in tags
    }


def _lagged_pairs(
    cause_arrays: Sequence[np.ndarray], effect_arrays: Sequence[np.ndarray], lag_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Pool `(cause[t], effect[t + lag])` across events, never crossing an
    event boundary."""
    cs: List[np.ndarray] = []
    es: List[np.ndarray] = []
    for c, e in zip(cause_arrays, effect_arrays):
        n = min(len(c), len(e))
        if n - lag_steps <= 1:
            continue
        cs.append(c[: n - lag_steps])
        es.append(e[lag_steps:n])
    if not cs:
        return np.array([]), np.array([])
    return np.concatenate(cs), np.concatenate(es)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    return float(r) if np.isfinite(r) else 0.0


def _mutual_information(
    x: np.ndarray, y: np.ndarray, max_samples: int, seed: int
) -> float:
    if x.size < 10:
        return 0.0
    from sklearn.feature_selection import mutual_info_regression

    if x.size > max_samples:
        idx = np.random.default_rng(seed).choice(x.size, size=max_samples, replace=False)
        x, y = x[idx], y[idx]
    mi = mutual_info_regression(
        x.reshape(-1, 1), y, random_state=seed, n_neighbors=3
    )
    return float(mi[0])


@dataclass
class CorrelationResult:
    """One discovered (or confirmed-known) tag relationship."""

    cause: str
    effect: str
    best_lag_min: float
    correlation: float          # signed Pearson r at the best lag
    mutual_information: float   # at the best lag
    n_samples: int
    is_known: bool
    source: str = Source.CORRELATION_DISCOVERY

    def to_dict(self) -> dict:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "best_lag_min": round(self.best_lag_min, 3),
            "correlation": round(self.correlation, 4),
            "mutual_information": round(self.mutual_information, 4),
            "n_samples": self.n_samples,
            "is_known": self.is_known,
            "novel": not self.is_known,
            "source": self.source,
        }


def discover_correlations(
    series: SeriesByTag,
    tags: Optional[Sequence[str]] = None,
    max_lag_min: float = DEFAULT_MAX_LAG_MIN,
    min_abs_correlation: float = DEFAULT_MIN_ABS_CORRELATION,
    mi_max_samples: int = DEFAULT_MI_MAX_SAMPLES,
    seed: int = 42,
) -> List[CorrelationResult]:
    """
    Sweep every ordered tag pair for the strongest lagged Pearson correlation,
    compute mutual information at that lag, and classify against
    `grades.KNOWN_LOOPS`.

    Only pairs reaching `min_abs_correlation` are returned, sorted by
    descending |correlation| -- an unfiltered 19x18 pair sweep is mostly
    noise, and the dashboard's Correlation Explorer needs signal, not a
    census.
    """
    tags = list(tags) if tags is not None else list(series.keys())
    max_lag_steps = max(int(round(max_lag_min * C.STEPS_PER_MIN)), 0)

    results: List[CorrelationResult] = []
    for cause in tags:
        for effect in tags:
            if cause == effect:
                continue
            best_lag, best_r, best_n = 0, 0.0, 0
            for lag in range(0, max_lag_steps + 1):
                x, y = _lagged_pairs(series[cause], series[effect], lag)
                if x.size == 0:
                    continue
                r = _pearson(x, y)
                if abs(r) > abs(best_r):
                    best_lag, best_r, best_n = lag, r, x.size

            if abs(best_r) < min_abs_correlation or best_n == 0:
                continue

            x, y = _lagged_pairs(series[cause], series[effect], best_lag)
            mi = _mutual_information(x, y, mi_max_samples, seed)

            results.append(
                CorrelationResult(
                    cause=cause,
                    effect=effect,
                    best_lag_min=best_lag / C.STEPS_PER_MIN,
                    correlation=best_r,
                    mutual_information=mi,
                    n_samples=best_n,
                    is_known=is_known_relationship(cause, effect),
                )
            )

    results.sort(key=lambda r: -abs(r.correlation))
    return results


def novel_correlations(results: Sequence[CorrelationResult]) -> List[CorrelationResult]:
    """The subset of a discovery run that is *not* already in `KNOWN_LOOPS` --
    the actual news for the Correlation Explorer panel."""
    return [r for r in results if not r.is_known]
