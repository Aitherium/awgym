"""The eval slice — transitions LeWM never trains on.

The training pool and the eval slice are DISJOINT BY CONSTRUCTION: a fixed set
of games (ARC_GYM_EVAL_GAMES, default 5 of the vendored generated set) is
NEVER passed to /observe, and the scorer plays those games to measure
retrodiction on unseen grids. This is the same holdout discipline as the live
lane's arc_wm_eval.py and the UNSEEN-bucket lesson: aggregate surprise is
gamed by memorizing seen pairs, so the disjoint slice is the only place a
learned model can earn anything.
"""

from __future__ import annotations

import os

DEFAULT_EVAL_GAMES = [
    # All five are in the vendored dream-team generated pool (measured
    # 2026-08-30) — the container bakes ONLY that set, and resolve_eval_games
    # fails LOUD on an absent id, so an SDK-game id here would kill every
    # in-container `awgym score` run.
    "cc2048-f080c4af",
    "cg1842-e7600c9f",
    "cl0426-c5dfa3ac",
    "df4821-818c2c5e",
    "dl4827-b8768101",
]

EVAL_GAMES_ENV = "ARC_GYM_EVAL_GAMES"


def eval_games() -> list[str]:
    raw = os.environ.get(EVAL_GAMES_ENV)
    if raw:
        return [g.strip() for g in raw.split(";") if g.strip()]
    return list(DEFAULT_EVAL_GAMES)


def resolve_eval_games(pool: dict) -> list[str]:
    """Resolve the configured eval ids against the real pool. Any id absent
    from the pool FAILS LOUD — a silently-shrunken eval slice is a
    vacuous-pass trap (the gate-coverage lesson)."""
    wanted = eval_games()
    missing = [g for g in wanted if g not in pool]
    if missing:
        raise KeyError(
            f"eval-slice games missing from the pool: {missing} — set "
            f"ARC_GYM_EVAL_GAMES to ids that exist (pool: "
            f"{sorted(pool)[:5]}...)")
    return wanted
