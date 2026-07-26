"""
Render `data/ledger.jsonl` as a readable table for the terminal -- a
read-only view for submission screenshots. Does not touch the ledger file
or its format; `gci/ledger.py` is unchanged.

Usage:
    python scripts/render_ledger.py [--limit N] [--path data/ledger.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from tabulate import tabulate

ROOT = Path(__file__).resolve().parent.parent


def _truncate(value, width: int) -> str:
    text = "" if value is None else str(value)
    # Notes can contain raw control characters (a robustness test deliberately
    # sent some, including a NUL byte) -- strip them so one ledger entry
    # stays one table row instead of corrupting the terminal layout.
    text = "".join(c if c.isprintable() else " " for c in text)
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def load_rows(path: Path, limit: int) -> list[list[str]]:
    lines = path.read_text().splitlines()
    if limit:
        lines = lines[-limit:]

    rows = []
    for line in lines:
        entry = json.loads(line)
        ts = datetime.fromtimestamp(entry["timestamp"], tz=timezone.utc)
        rows.append([
            ts.strftime("%Y-%m-%d %H:%M:%S"),
            _truncate(entry.get("advisory_id"), 18),
            entry.get("decision") or "-",
            entry.get("source") or "-",
            _truncate(entry.get("note"), 24),
        ])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=ROOT / "data" / "ledger.jsonl")
    parser.add_argument("--limit", type=int, default=30, help="0 = show every entry")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"no ledger found at {args.path}", file=sys.stderr)
        raise SystemExit(1)

    rows = load_rows(args.path, args.limit)
    headers = ["timestamp (UTC)", "advisory_id", "decision", "source", "note"]
    print(tabulate(rows, headers=headers, tablefmt="simple"))
    print(f"\n{len(rows)} of {sum(1 for _ in args.path.open())} total ledger entries shown")


if __name__ == "__main__":
    main()
