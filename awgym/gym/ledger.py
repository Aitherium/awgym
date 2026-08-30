"""Run-level ledger — the audit trail the ratchet and the portal read.

One line per episode appended to ``ledger.jsonl`` under the data root. The
ledger is the gym's single source of truth for "did anything happen and did
it get better": run_id, game, steps, score, RHAE (when scored), surprise
before/after, wm_train_steps, config sha.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..config import data_root

LEDGER_NAME = "ledger.jsonl"


def _ledger_path() -> Path:
    return data_root() / LEDGER_NAME


def append_run(row: dict[str, Any]) -> dict[str, Any]:
    """Append one episode row, returning it with run_id + ts filled in."""
    entry = dict(row)
    entry.setdefault("run_id", _new_run_id())
    entry.setdefault("ts", datetime.now(timezone.utc).isoformat())
    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def _new_run_id() -> str:
    import uuid
    return f"gym-{uuid.uuid4().hex[:12]}"


def read_ledger(limit: Optional[int] = None) -> list[dict[str, Any]]:
    """Newest-first run rows (whole-file read; the ledger stays small)."""
    path = _ledger_path()
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(rows[-limit:])) if limit else list(reversed(rows))
