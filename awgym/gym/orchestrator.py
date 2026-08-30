"""GameSession — the reset/step loop that turns a game into transitions.

A policy is any callable taking the current Grid and returning an action
(0-7, the ARC-AGI-3 action space). The session captures every transition in
the canonical shape the LeWM adapter and the recorder consume:

    {grid, action, next_grid, reward, done, game_id, step, level}
"""

from __future__ import annotations

import os
import queue
import random
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from ..envs.games import GameInfo

Policy = Callable[[Any, Any], int]  # (grid, session) -> action 0..7

MAX_STEPS_ENV = "ARC_GYM_MAX_STEPS"
DEFAULT_MAX_STEPS = 60
STEP_TIMEOUT_ENV = "ARC_GYM_STEP_TIMEOUT_S"
DEFAULT_STEP_TIMEOUT_S = 90.0


@dataclass
class Transition:
    game_id: str
    step: int
    grid: list  # 64x64 int grid (the Grid.data shape)
    action: int
    next_grid: list
    reward: float
    done: bool
    level: int | None = None

    def as_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "step": self.step,
            "grid": self.grid,
            "action": self.action,
            "next_grid": self.next_grid,
            "reward": self.reward,
            "done": self.done,
            "level": self.level,
        }


@dataclass
class GameSession:
    """Plays one game to the step cap through a policy, capturing every
    transition. The env is constructed per-session so each run starts from a
    fresh level sequence (ARCAGI3Env is stateful)."""

    game: GameInfo
    policy: Policy
    max_steps: int = field(default_factory=lambda: int(
        os.environ.get(MAX_STEPS_ENV) or DEFAULT_MAX_STEPS))
    env: Any = None  # ARCAGI3Env instance (created lazily on reset)

    def __post_init__(self) -> None:
        if self.env is None:
            self.env = self._make_env()

    def _make_env(self) -> Any:
        from ..vendor.dream_team import ARCAGI3Env  # deferred: needs the tree
        # The SDK scans ENVIRONMENTS_DIR (recursively) at Arcade construction.
        # Point it at the game's own directory (its parent tree) — cheaper and
        # exact. MUST be an overwrite, never setdefault: the SDK builds one
        # registry per Arcade, and a poisoned first-game root makes every
        # later game in a multi-game run return "SDK returned None
        # environment" (measured 2026-08-30: episode 2 of `awgym train` died
        # on fl5273-6a25a72a while the al7306 root was still set).
        env_dir = os.path.dirname(os.path.dirname(self.game.metadata_path))
        os.environ["ENVIRONMENTS_DIR"] = env_dir
        return ARCAGI3Env(game_id=self.game.game_id, operation_mode="offline")

    def _step(self, action: int, data: dict | None) -> tuple:
        """One env.step with a hard timeout, run on a daemon thread.

        The vendored _step_with_retry treats ANY SDK error as transient and
        retries with exponential backoff up to ~10 minutes (measured
        2026-08-30: a permanent KeyError cost 10 consecutive backoffs). The
        training loop must not inherit that: one broken step ends the episode
        (the daemon thread eventually finishes its bounded retries and is
        discarded with the session).
        """
        timeout = float(os.environ.get(STEP_TIMEOUT_ENV,
                                       DEFAULT_STEP_TIMEOUT_S))
        q: queue.Queue = queue.Queue(maxsize=1)

        def _run() -> None:
            try:
                q.put(self.env.step(action, data=data))
            except Exception as exc:  # noqa: BLE001 — any env error ends the step
                q.put(exc)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        try:
            result = q.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(
                f"env.step(action={action}) exceeded {timeout:g}s — "
                "the vendored retry loop is backlogging a permanent error")
        if isinstance(result, Exception):
            raise result
        return result

    def play(self) -> list[Transition]:
        obs = self.env.reset()
        grid = obs.data if hasattr(obs, "data") else obs
        transitions: list[Transition] = []
        for step in range(self.max_steps):
            action = int(self.policy(grid, self))
            # CLICK (6) is coordinate-bearing. The arcengine adapter clamps
            # game-space coords into the logical grid (measured 2026-08-30:
            # x = min(max((int(data["x"]) - x_offset) // scale, 0), width-1)),
            # so uniform random coords in a wide range are always a valid
            # cell. A bare int CLICK raises KeyError 'x' inside the adapter —
            # the exact permanent error the vendored retry loop backlogs.
            # Phase 1 policies are int-only; coords are policy-free filler.
            data = None
            if action == 6:
                data = {"x": random.randint(0, 127),
                        "y": random.randint(0, 127)}
            # env.step returns (obs, reward, done, info) RL-style per the
            # vendored ARCAGI3Env contract.
            try:
                obs, reward, done, info = self._step(action, data)
            except Exception:
                break  # one bad step must not cost the whole episode
            next_grid = obs.data if hasattr(obs, "data") else obs
            transitions.append(Transition(
                game_id=self.game.game_id,
                step=step,
                grid=grid,
                action=action,
                next_grid=next_grid,
                reward=float(reward or 0.0),
                done=bool(done),
                level=getattr(info, "levels_completed", None),
            ))
            grid = next_grid
            if done:
                break
        return transitions
