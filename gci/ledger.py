"""
Advisory ledger -- accept/reject capture and quality evaluation
(deliverable 6).

Every `Advisory` GCI surfaces (see `provenance.py`) is recorded here when
shown, and again when the operator responds. The ledger is the audit trail
Learning & Governance is built on: Phase 2's trust-score reranking
(`learning.py`, not yet built) will read straight off this history rather
than a parallel store, so its shape is decided now even though nothing
downstream consumes it yet.

Persistence is append-only JSON Lines, one event per line -- the natural
shape for an audit trail (immutable, trivially replay-able, diff-friendly)
and avoids a database dependency this hackathon build does not need.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .provenance import Advisory

DECISIONS: Tuple[str, ...] = ("accepted", "rejected", "ignored")


@dataclass
class LedgerEntry:
    """One row of the audit trail: either an advisory being surfaced, or an
    operator's response to one already surfaced."""

    entry_id: str
    kind: str                          # "surfaced" | "response"
    advisory_id: str
    timestamp: float
    source: Optional[str] = None
    confidence: Optional[float] = None
    value_usd: Optional[float] = None
    decision: Optional[str] = None
    note: Optional[str] = None
    event_id: Optional[int] = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class AdvisoryLedger:
    """In-memory log of surfaced advisories and operator responses, with
    JSON-Lines persistence."""

    def __init__(self) -> None:
        self.entries: List[LedgerEntry] = []

    # -- capture -------------------------------------------------------
    def record(
        self, advisory: Advisory, event_id: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> str:
        """Log that an advisory was surfaced to the operator."""
        entry = LedgerEntry(
            entry_id=str(uuid.uuid4()),
            kind="surfaced",
            advisory_id=advisory.id,
            timestamp=timestamp if timestamp is not None else time.time(),
            source=advisory.source,
            confidence=advisory.confidence,
            value_usd=advisory.value.point_estimate_usd if advisory.value else None,
            event_id=event_id,
        )
        self.entries.append(entry)
        return entry.entry_id

    def was_surfaced(self, advisory_id: str) -> bool:
        """Has this advisory ever actually been shown to the operator? A
        request forging an advisory_id that was never surfaced is malformed
        input, not a real accept/reject -- callers use this to reject it
        with a clean not-found rather than silently logging a response that
        can never be paired with anything in `evaluate()`."""
        return advisory_id in self._surfaced()

    def respond(
        self, advisory_id: str, decision: str, note: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> str:
        """Record the operator's accept/reject/ignore decision on a
        previously surfaced advisory."""
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}, got {decision!r}")
        if not self.was_surfaced(advisory_id):
            raise KeyError(f"advisory_id {advisory_id!r} was never surfaced")
        entry = LedgerEntry(
            entry_id=str(uuid.uuid4()),
            kind="response",
            advisory_id=advisory_id,
            timestamp=timestamp if timestamp is not None else time.time(),
            decision=decision,
            note=note,
        )
        self.entries.append(entry)
        return entry.entry_id

    # -- persistence -----------------------------------------------------
    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for entry in self.entries:
                f.write(json.dumps(entry.to_dict()) + "\n")
        return path

    @classmethod
    def load(cls, path) -> "AdvisoryLedger":
        ledger = cls()
        path = Path(path)
        if not path.exists():
            return ledger
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ledger.entries.append(LedgerEntry(**json.loads(line)))
        return ledger

    # -- quality evaluation ------------------------------------------------
    def _surfaced(self) -> Dict[str, LedgerEntry]:
        return {e.advisory_id: e for e in self.entries if e.kind == "surfaced"}

    def _responses(self) -> Dict[str, LedgerEntry]:
        """Latest response per advisory -- if the UI ever lets an operator
        change their mind, the last response wins."""
        out: Dict[str, LedgerEntry] = {}
        for e in self.entries:
            if e.kind == "response":
                out[e.advisory_id] = e
        return out

    def acceptance_rate(self, source: Optional[str] = None) -> float:
        """
        Fraction of *responded* advisories accepted, optionally restricted to
        one source. Advisories still awaiting a response are excluded rather
        than counted against -- an unanswered suggestion says nothing about
        its quality.
        """
        surfaced, responses = self._surfaced(), self._responses()
        relevant = [
            r for aid, r in responses.items()
            if aid in surfaced and (source is None or surfaced[aid].source == source)
        ]
        if not relevant:
            return float("nan")
        accepted = sum(1 for r in relevant if r.decision == "accepted")
        return accepted / len(relevant)

    def evaluate(self) -> Dict[str, object]:
        """
        Quality summary: overall and per-source acceptance rate, mean
        confidence of accepted vs rejected advisories (a calibration signal
        -- if the two are indistinguishable, confidence is not actually
        informing operator trust), and realised dollar value of accepted,
        priced advisories.
        """
        surfaced, responses = self._surfaced(), self._responses()
        paired = [(surfaced[aid], r) for aid, r in responses.items() if aid in surfaced]

        by_source: Dict[str, List[Tuple[LedgerEntry, LedgerEntry]]] = {}
        for s, r in paired:
            by_source.setdefault(s.source, []).append((s, r))

        accepted_conf = [
            s.confidence for s, r in paired
            if r.decision == "accepted" and s.confidence is not None
        ]
        rejected_conf = [
            s.confidence for s, r in paired
            if r.decision == "rejected" and s.confidence is not None
        ]
        accepted_value = sum(
            s.value_usd for s, r in paired
            if r.decision == "accepted" and s.value_usd is not None
        )

        return {
            "n_surfaced": len(surfaced),
            "n_responded": len(paired),
            "acceptance_rate_overall": self.acceptance_rate(),
            "acceptance_rate_by_source": {
                src: sum(1 for s, r in items if r.decision == "accepted") / len(items)
                for src, items in by_source.items()
            },
            "mean_confidence_accepted": (
                float(np.mean(accepted_conf)) if accepted_conf else float("nan")
            ),
            "mean_confidence_rejected": (
                float(np.mean(rejected_conf)) if rejected_conf else float("nan")
            ),
            "realized_value_usd_accepted": float(accepted_value),
        }
