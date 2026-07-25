"""
Connectivity layer -- the one documented interface every engine above it
talks to (see `CLAUDE.md`'s seven-layer architecture table).
`SimulatedDataSource` is the shipped implementation: a small, freshly
generated, fully deterministic demo corpus replayed as "live" data. A future
site-historian connector (OPC-UA, a plant historian) would be a second
implementation of this same three-method interface -- nothing above this
layer would need to change.

The `t_min` cursor on `row_at` is the connectivity contract's core guarantee:
callers only ever see data at or before the requested time, mirroring what a
live scanner feed would actually provide and preserving the backward-only
feature guarantee (D9) all the way from simulated "plant" through to the API.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

import pandas as pd

from ..events import EventResult, generate_dataset
from ..features import build_event_features

DEFAULT_DEMO_EVENT_COUNT = 40
DEFAULT_DEMO_SEED = 20260726  # distinct from the training corpus seed (20260725)


class DataSource(ABC):
    """What every engine above this layer is allowed to ask for."""

    @abstractmethod
    def list_events(self) -> List[int]: ...

    @abstractmethod
    def get_event(self, event_id: int) -> EventResult: ...

    @abstractmethod
    def row_at(self, event_id: int, t_min: float) -> pd.Series:
        """Backward-looking feature row at or before `t_min` -- never after."""
        ...


class SimulatedDataSource(DataSource):
    """Replays a small, deterministic, freshly generated corpus as the
    "live" feed -- the shipped implementation of `DataSource`."""

    def __init__(
        self,
        n_events: int = DEFAULT_DEMO_EVENT_COUNT,
        seed: int = DEFAULT_DEMO_SEED,
        events: Optional[Sequence[EventResult]] = None,
    ):
        events = events if events is not None else generate_dataset(n_events=n_events, seed=seed)
        self._events: Dict[int, EventResult] = {ev.event_id: ev for ev in events}
        self._feature_cache: Dict[int, pd.DataFrame] = {}

    def list_events(self) -> List[int]:
        return sorted(self._events.keys())

    def get_event(self, event_id: int) -> EventResult:
        if event_id not in self._events:
            raise KeyError(event_id)
        return self._events[event_id]

    def _features(self, event_id: int) -> pd.DataFrame:
        if event_id not in self._feature_cache:
            self._feature_cache[event_id] = build_event_features(self.get_event(event_id))
        return self._feature_cache[event_id]

    def row_at(self, event_id: int, t_min: float) -> pd.Series:
        df = self._features(event_id)
        eligible = df[df["t_min"] <= t_min]
        if eligible.empty:
            eligible = df.iloc[[0]]
        return eligible.iloc[-1]

    def default_event_id(self) -> int:
        """The most demo-worthy event -- the largest excursion -- so a caller
        that doesn't specify one still sees something interesting."""
        return max(self._events.values(), key=lambda ev: ev.labels["max_abs_dev_pct"]).event_id

    @property
    def events(self) -> Dict[int, EventResult]:
        return self._events
