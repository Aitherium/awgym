"""GameSession — the reset/step loop that turns a game into transitions.

A policy is any callable taking the current Grid and returning an action
(0-7, the ARC-AGI-3 action space). The session captures every transition in
the canonical shape the LeWM adapter and the recorder consume:

    {grid, action, next_grid, reward, done, game_id, step, level}
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

from ..envs.games import GameInfo

Policy = Callable[[Any, Any], int]  # (grid, session) -> action 0..7

MAX_STEPS_ENV = "ARC_GYM_MAX_STEPS"
DEFAULT_MAX_STEPS = 60


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
        # The SDK scans ENVIRONMENTS_DIR at Arcade construction; the pool scan
        # above already proves the metadata exists, so point the SDK at the
        # game's own directory (its parent tree) — cheaper and exact.
        env_dir = os.path.dirname(os.path.dirname(self.game.metadata_path))
        os.environ.setdefault("ENVIRONMENTS_DIR", env_dir)
        return ARCAGI3Env(game_id=self.game.game_id, operation_mode="offline")

    def play(self) -> list[Transition]:
        obs = self.env.reset()
        grid = obs.data if hasattr(obs, "data") else obs
        transitions: list[Transition] = []
        for step in range(self.max_steps):
            action = int(self.policy(grid, self))
            # env.step returns (obs, reward, done, info) RL-style per the
            # vendored ARCAGI3Env contract.
            obs, reward, done, info = self.env.step(action)
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
