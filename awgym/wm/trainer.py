"""The training loop — play, observe, burst-train, score, ledger.

The gym's loop is ADDITIVE to the live LeWM service: /observe writes go into
the SAME 50k reservoir the live rotation runner feeds (no conflict), bursts
are throttled (ARC_GYM_MIN_BURST_INTERVAL_S, default 60) so the live lane's
heartbeat never starves, and every burst is followed by a surprise read over
the DISJOINT eval slice. CPU-only by default — GPU training of the unfrozen
model OOMs on this host's SIGReg (measured); the env knobs exist for a box
that can.

This module is the mutating surface of the ratchet trial: ARC_GYM_* values
for the pool, burst schedule and eval seed are what a trial may change.
"""

from __future__ import annotations

import os
import time
from typing import Callable

from ..envs.games import game_pool
from ..gym.ledger import append_run
from ..gym.orchestrator import GameSession, Transition
from .eval_slice import resolve_eval_games
from .lewm_client import LeWMClient

MIN_BURST_INTERVAL_S = float(os.environ.get("ARC_GYM_MIN_BURST_INTERVAL_S", "60"))
BURST_STEPS_ENV = "ARC_GYM_BURST_STEPS"
EPISODES_ENV = "ARC_GYM_EPISODES"


class Trainer:
    """Owns the observe→train→score cycle. The policy callable is the only
    plug point a Phase-2 solver or a scripted explorer supplies."""

    def __init__(self, client: LeWMClient | None = None,
                 policy: Callable[[list, GameSession], int] | None = None,
                 burst_steps: int | None = None):
        self.client = client or LeWMClient()
        self.policy = policy or self._curious_policy
        self.burst_steps = burst_steps or int(
            os.environ.get(BURST_STEPS_ENV) or "200")
        self._last_burst_ts = 0.0

    # -- policies --------------------------------------------------------
    @staticmethod
    def _random_policy(grid: list, session: GameSession) -> int:
        import random
        return random.randint(1, 7)

    def _curious_policy(self, grid: list, session: GameSession) -> int:
        """plan_curious-driven exploration: ask LeWM for the intrinsic-reward
        action sequence and take its first action. Doubles as an
        intrinsic-exploration data generator (the plan's goal)."""
        try:
            out = self.client._post("/plan_curious", {
                "grid": grid, "horizon": 3, "iters": 8,
                "game": session.game.game_id})
            actions = out.get("actions") or []
            if actions:
                return int(actions[0])
        except Exception:
            pass  # LeWM down/err — fall back to a legal random action
        return self._random_policy(grid, session)

    # -- the loop --------------------------------------------------------
    def run_episodes(self, n: int = 5, max_steps: int = 60,
                     game_ids: list[str] | None = None) -> list[dict]:
        """Play n episodes (rotating through the pool or the given ids),
        observe every transition into LeWM, then run one throttled burst.
        Returns the ledger rows (one per episode)."""
        pool = game_pool()
        if game_ids:
            missing = [g for g in game_ids if g not in pool]
            if missing:
                raise KeyError(f"games not in pool: {missing}")
            games = [pool[g] for g in game_ids]
        else:
            # The eval slice is DISJOINT BY CONSTRUCTION: never observe its
            # games, or the holdout is contaminated and the metric is gamed by
            # memorization (the eval_slice docstring's whole point). Explicit
            # --games overrides on purpose — a caller naming an eval game is
            # making a deliberate choice.
            eval_ids = set(resolve_eval_games(pool))
            games = [g for g in sorted(pool.values(),
                                       key=lambda g: g.game_id)
                     if g.game_id not in eval_ids]
            if not games:
                raise KeyError("no trainable games outside the eval slice — "
                               "pool is entirely eval games")
        rows: list[dict] = []
        observed = 0
        for i in range(n):
            game = games[i % len(games)]
            session = GameSession(game=game, policy=self.policy,
                                  max_steps=max_steps)
            transitions = session.play()
            observed += self._observe_transitions(transitions)
            row = append_run({
                "game_id": game.game_id,
                "steps": len(transitions),
                "score": 0,  # Phase 2 solver fills this
                "policy": type(self.policy).__name__,
            })
            rows.append(row)
        if observed:
            self._burst()
        return rows

    def _observe_transitions(self, transitions: list[Transition]) -> int:
        ok = 0
        for t in transitions:
            try:
                self.client.observe(
                    grid=t.grid, action=t.action, next_grid=t.next_grid,
                    game=t.game_id, source="awgym")
                ok += 1
            except Exception:
                # one bad transition must not kill the episode — the WM is
                # additive; the next /health read shows whether writes landed
                continue
        return ok

    def _burst(self) -> dict:
        """One throttled /train burst, then a surprise read over the eval
        slice. Returns the burst result dict."""
        now = time.time()
        wait = self._last_burst_ts + MIN_BURST_INTERVAL_S - now
        if wait > 0:
            time.sleep(wait)
        out = self.client.train(steps=self.burst_steps)
        self._last_burst_ts = time.time()
        try:
            pool = game_pool()
            eval_ids = resolve_eval_games(pool)
            surprise = self._eval_surprise(eval_ids[:2], max_steps=10)
        except Exception:
            surprise = None
        return {"train": out.get("result") or out, "eval_surprise": surprise}

    def _eval_surprise(self, game_ids: list[str], max_steps: int = 10,
                       policy: Callable | None = None) -> float:
        """Mean surprise over the eval slice (never trained on)."""
        pool = game_pool()
        pol = policy or self._random_policy
        total, count = 0.0, 0
        for gid in game_ids:
            session = GameSession(game=pool[gid], policy=pol,
                                  max_steps=max_steps)
            for t in session.play():
                s = self.client.surprise(t.grid, t.action, t.next_grid,
                                         ctx=t.game_id)
                if s is not None:
                    total += s
                    count += 1
        return (total / count) if count else float("nan")
