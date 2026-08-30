"""Recording writer — byte-compatible with the live lane's recordings.

The live ARC exhibit (arc.aitherium.com) stores per-run jsonl files at
``recordings/<game>.…recording.jsonl`` whose action lines carry this envelope
(measured 2026-08-30 from the live corpus):

    {"timestamp": ISO8601, "data": {"game_id": …, "frame": [grid_64x64],
     "state": "NOT_FINISHED", "levels_completed": N, "win_levels": N,
     "action_input": {…}, "guid": …, "full_reset": bool,
     "available_actions": […]}}

The frame is wrapped in an extra list — [grid] — which the live recorder
writes and its consumers expect. awgym matches the envelope exactly so the
live lane's readers (arc_wm_eval.py and friends) can ingest gym recordings
unchanged.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from .orchestrator import Transition


class Recorder:
    """Appends transition lines in the live-lane envelope."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO | None = None

    def __enter__(self) -> "Recorder":
        self._fh = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def write_transition(self, t: Transition, available_actions: list[int] | None = None,
                         action_input: dict | None = None) -> None:
        if self._fh is None:
            raise RuntimeError("Recorder not open (use it as a context manager)")
        data = {
            "game_id": t.game_id,
            "frame": [t.next_grid],          # the live-lane wrapper
            "state": "NOT_FINISHED" if not t.done else "FINISHED",
            "levels_completed": t.level or 0,
            "win_levels": 0,
            "action_input": action_input or {
                "id": t.action,
                "data": {"game_id": t.game_id},
                "reasoning": "gym-scripted",
            },
            "guid": str(uuid.uuid4()),
            "full_reset": False,
            "available_actions": available_actions or [1, 2, 3, 4, 5, 6, 7],
        }
        line = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        self._fh.write(json.dumps(line) + "\n")
        self._fh.flush()
