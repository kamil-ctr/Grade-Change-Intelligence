"""
Event-wise stratified splitting.

Why not a random row split
--------------------------
A grade-change event is 360 samples taken 5 seconds apart. Consecutive samples
are almost identical, and the label ("will basis weight breach in the next 10
minutes") is shared across long runs of them. Splitting rows at random puts
near-duplicate samples on both sides of the boundary, so the model is scored on
data it has effectively already seen. Reported PR-AUC then approaches 1.0 and
means nothing.

Splitting whole events is the only honest option, and it is what this module
provides. Stratification keeps the class balance and the mix of easy/hard
transitions comparable across splits, which matters at 300 events where a
naive shuffle can easily land most failures in one fold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class EventSplit:
    """Disjoint event-id sets plus convenience row masks."""

    train_events: np.ndarray
    val_events: np.ndarray
    test_events: np.ndarray
    seed: int
    strata: Dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        overlaps = (
            set(self.train_events) & set(self.val_events),
            set(self.train_events) & set(self.test_events),
            set(self.val_events) & set(self.test_events),
        )
        for overlap in overlaps:
            if overlap:
                raise ValueError(f"event split overlap: {sorted(overlap)[:5]}")

    @property
    def counts(self) -> Dict[str, int]:
        return {
            "train": int(self.train_events.size),
            "val": int(self.val_events.size),
            "test": int(self.test_events.size),
        }

    # "val" and "validation" are both accepted so callers need not remember
    # which spelling this class uses.
    _ALIASES = {
        "train": "train", "training": "train",
        "val": "val", "validation": "val", "valid": "val",
        "test": "test", "holdout": "test",
    }

    def events_for(self, which: str) -> np.ndarray:
        key = self._ALIASES.get(which.lower())
        if key is None:
            raise KeyError(
                f"unknown split '{which}'; expected one of "
                f"{sorted(set(self._ALIASES))}"
            )
        return {
            "train": self.train_events,
            "val": self.val_events,
            "test": self.test_events,
        }[key]

    def mask(self, df: pd.DataFrame, which: str) -> np.ndarray:
        events = self.events_for(which)
        return df["event_id"].isin(set(events.tolist())).to_numpy()

    def frame(self, df: pd.DataFrame, which: str) -> pd.DataFrame:
        return df.loc[self.mask(df, which)].reset_index(drop=True)

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "counts": self.counts,
            "train_events": self.train_events.tolist(),
            "val_events": self.val_events.tolist(),
            "test_events": self.test_events.tolist(),
        }


def _event_strata(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per event describing what it should be stratified on.

    Two axes matter:
      * whether the transition ever went off-spec (the class balance)
      * how large the transition was (difficulty)
    """
    grouped = df.groupby("event_id", sort=True)
    table = pd.DataFrame(
        {
            "event_id": np.array(sorted(df["event_id"].unique())),
        }
    ).set_index("event_id")

    table["ever_off_spec"] = grouped["currently_off_spec"].max().astype(int)
    table["any_positive"] = grouped["y_breach"].max().astype(int)
    if "transition_magnitude" in df.columns:
        table["magnitude"] = grouped["transition_magnitude"].first()
    else:  # pragma: no cover - defensive
        table["magnitude"] = 0.0

    # Difficulty tertiles; qcut with duplicate edges collapses gracefully.
    try:
        table["mag_band"] = pd.qcut(
            table["magnitude"], q=3, labels=["low", "mid", "high"],
            duplicates="drop",
        ).astype(str)
    except (ValueError, IndexError):  # pragma: no cover - tiny datasets
        table["mag_band"] = "all"

    table["stratum"] = (
        table["ever_off_spec"].astype(str) + "|" + table["mag_band"]
    )
    return table.reset_index()


def event_wise_split_3way(
    df: pd.DataFrame,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
    seed: int = 42,
) -> EventSplit:
    """
    Split events into train / validation / test, stratified by outcome and
    transition difficulty.

    Allocation inside each stratum is deterministic given `seed`: events are
    shuffled, then dealt out so that small strata still contribute to every
    split instead of landing entirely in one.
    """
    if not 0.0 < val_frac < 1.0 or not 0.0 < test_frac < 1.0:
        raise ValueError("val_frac and test_frac must be in (0, 1)")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must leave room for training")

    strata = _event_strata(df)
    rng = np.random.default_rng(seed)

    train: List[int] = []
    val: List[int] = []
    test: List[int] = []

    for _, group in strata.groupby("stratum", sort=True):
        ids = group["event_id"].to_numpy()
        ids = ids[rng.permutation(ids.size)]
        n = ids.size

        n_test = int(np.floor(n * test_frac))
        n_val = int(np.floor(n * val_frac))
        # Guarantee representation in val/test for strata big enough to spare it
        if n >= 3:
            n_test = max(n_test, 1)
            n_val = max(n_val, 1)
        if n_test + n_val >= n:
            n_test = min(n_test, max(n - 2, 0))
            n_val = min(n_val, max(n - 1 - n_test, 0))

        test.extend(ids[:n_test].tolist())
        val.extend(ids[n_test : n_test + n_val].tolist())
        train.extend(ids[n_test + n_val :].tolist())

    return EventSplit(
        train_events=np.array(sorted(train)),
        val_events=np.array(sorted(val)),
        test_events=np.array(sorted(test)),
        seed=seed,
        strata=dict(zip(strata["event_id"], strata["stratum"])),
    )


def split_summary(df: pd.DataFrame, split: EventSplit) -> pd.DataFrame:
    """Human-readable table confirming the splits are comparable."""
    rows = []
    for which in ("train", "val", "test"):
        part = split.frame(df, which)
        rows.append(
            {
                "split": which,
                "events": int(part["event_id"].nunique()),
                "rows": int(len(part)),
                "positive_rate": float(part["y_breach"].mean()),
                "in_spec_rows": int((part["currently_off_spec"] == 0).sum()),
                "in_spec_positive_rate": float(
                    part.loc[part["currently_off_spec"] == 0, "y_breach"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)
