"""Solver run workspace — the league's file truth.

One run = one directory under ``<data_root>/games/<game_id>/runs/<ts>/``.
The workspace files are the coordination protocol (the vendored dream-team
DECISION-block convention, adapted): every role's output lands here and the
leader's DECISION blocks are the commit points. The workspace is also what
check_awgym_run_hygiene asserts over: the directory must contain ZERO
agent-written ``.py`` files (the LeWM API is the only execution surface).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..config import data_root

# Tolerant on purpose: the leader may emit the block as its own paragraphs
# (``DECISION:`` then indented lines) or inline after prose on the same line
# (``... so: DECISION: game_id: x``). Either way, the block is everything up
# to the next blank line or the end of the reply.
_DECISION_RE = re.compile(
    r"DECISION:\s*(?P<body>[^\n]*(?:\n(?!\n)[^\n]*)*)")


@dataclass
class RunWorkspace:
    """Per-run directory + the DECISION block parser."""
    game_id: str
    run_id: str
    root: Path = field(init=False)

    def __post_init__(self) -> None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.root = (data_root() / "games" / self.game_id
                     / "runs" / f"{ts}-{self.run_id}")
        self.root.mkdir(parents=True, exist_ok=True)

    # ── the DECISION block ────────────────────────────────────────────

    @staticmethod
    def parse_decision(text: str) -> Optional[dict[str, Any]]:
        """Extract the leader's ``DECISION:`` block from a model reply.

        Returns a dict with the block's fields keyed by their first token
        (``game_id:`` -> ``game_id``), or None when no block is present.
        Handles both the multiline form (one field per line) and the
        single-line form the model actually emits (``DECISION: game_id: x
        goal: y ...`` — measured 2026-08-30: a line-pair split turned the
        whole block into one value and the round played nothing).
        """
        m = _DECISION_RE.search(text)
        if not m:
            return None
        body = m.group("body")
        out: dict[str, Any] = {}
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            # Single-line blocks: split on every ``key: value`` pair.
            if ":" in line:
                for part in re.split(r"\s+(?=\w+:\s)", line):
                    if ":" in part:
                        key, _, value = part.partition(":")
                        out[key.strip()] = value.strip()
        return out or None

    # ── files ─────────────────────────────────────────────────────────

    def append_jsonl(self, name: str, row: dict[str, Any]) -> None:
        with (self.root / name).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def append_text(self, name: str, text: str) -> None:
        with (self.root / name).open("a", encoding="utf-8") as fh:
            fh.write(text.rstrip() + "\n\n")

    def log_round(self, round_no: int, decision: dict[str, Any]) -> None:
        self.append_jsonl("DECISIONS.jsonl",
                          {"round": round_no, **decision})

    def log_observation(self, step: int, grid: list,
                        action: int, next_grid: list,
                        surprise: Optional[float]) -> None:
        self.append_jsonl("observations.jsonl",
                          {"step": step, "action": action,
                           "surprise": surprise,
                           "grid": grid, "next_grid": next_grid})

    def summary(self) -> dict[str, Any]:
        """The run's file inventory (for the ledger row + findings card)."""
        files = sorted(p.name for p in self.root.iterdir())
        return {"run_dir": str(self.root), "files": files}
