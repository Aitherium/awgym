"""The awgym game pool — the ONLY environments the gym ever executes.

Trust boundary (from the awgym plan, security section): the pool is built by
scanning directories for metadata.json files (the SDK's own scan contract).
Only these PINNED, vendored games are ever executed — never agent-written
environment code. The `game_creator` path (which executes generated python
upstream) is deliberately absent.

Sources (all scanned recursively for metadata.json):
  * ARC_GYM_ENV_DIRS — env var, ';'-separated list of extra roots
  * the vendored dream-team environment_files_generated/ (25 generated games)
  * the live lane's environment_files/ when present (E: fork) — optional
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from ..config import dream_team_root

ENV_DIRS_ENV = "ARC_GYM_ENV_DIRS"
_LIVE_LANE_ENVS = r"E:\AitherOS-Data\arc-agi-3\ARC-AGI-3-Agents\environment_files"


@dataclass(frozen=True)
class GameInfo:
    game_id: str
    title: str
    metadata_path: str
    baseline_actions: int | None = None
    tags: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "title": self.title,
            "baseline_actions": self.baseline_actions,
            "tags": list(self.tags),
        }


def _scan_root(root: Path, out: dict[str, GameInfo]) -> None:
    if not root.is_dir():
        return
    for meta in root.rglob("metadata.json"):
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        gid = str(data.get("game_id") or "").strip()
        if not gid or gid in out:
            continue
        tags = data.get("tags") or []
        out[gid] = GameInfo(
            game_id=gid,
            title=str(data.get("title") or data.get("name") or gid),
            metadata_path=str(meta),
            baseline_actions=data.get("baseline_actions"),
            tags=tuple(str(t) for t in tags) if isinstance(tags, list) else (),
        )


def game_pool() -> dict[str, GameInfo]:
    """Scan all configured roots; the vendored generated set is always last
    (so an explicit ARC_GYM_ENV_DIRS root wins on id collision)."""
    out: dict[str, GameInfo] = {}
    extra = os.environ.get(ENV_DIRS_ENV, "")
    for root in (p.strip() for p in extra.split(";") if p.strip()):
        _scan_root(Path(root), out)
    # the live lane's downloaded games, when this host carries them
    _scan_root(Path(_LIVE_LANE_ENVS), out)
    # the vendored generated set — the reliable offline default
    _scan_root(Path(dream_team_root()) / "environment_files_generated", out)
    return out


def list_games() -> list[GameInfo]:
    return sorted(game_pool().values(), key=lambda g: g.game_id)


def pick_game(game_id: str | None = None) -> GameInfo:
    pool = game_pool()
    if game_id:
        if game_id not in pool:
            raise KeyError(
                f"game {game_id!r} not in the pool ({len(pool)} games scanned; "
                f"set ARC_GYM_ENV_DIRS to add roots)")
        return pool[game_id]
    if not pool:
        raise RuntimeError(
            "no ARC games scanned — set ARC_GYM_ENV_DIRS to a directory tree "
            "of metadata.json game files (e.g. the dream-team "
            "environment_files_generated)")
    # rotation default: round-robin would need state; first-by-id is
    # deterministic and fine for the training loop (the orchestrator rotates
    # across the pool itself).
    return min(pool.values(), key=lambda g: g.game_id)
