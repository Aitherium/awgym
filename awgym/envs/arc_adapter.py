"""ARC-AGI-3 as ONE EnvironmentAdapter — the reference domain of the general solver.

    from awgym.envs.arc_adapter import ArcEnvAdapter, ArcLevelScorer
    adapter = ArcEnvAdapter(game_id="ls20")      # observe / actions / step / reset / domain
    scorer  = ArcLevelScorer()                   # Score(levels=<max levels_completed>) | Refusal

`GameSession` (awgym/gym/orchestrator.py) stays the ARC-native loop the LeWM trainer,
recorder and latent-gate scorer consume; this adapter is the same env behind the
domain-neutral `awpredict.contracts.EnvironmentAdapter` so `ProblemSession` can play
it exactly as it plays a kernel harness or a fleet defect. It COMPOSES a GameSession
for the SDK construction and the bounded threaded `_step` (the vendored retry loop
backlogs a permanent error for ~10 minutes; one broken step must end the episode).

Observation is the grid as a list of lists (hashable via `tuple(map(tuple, grid))` if a
policy needs a key). Actions are the ARC ints 1..7; CLICK (6) is coordinate-bearing and
the adapter fills coordinates the way GameSession does, so int-only policies work.
`info` carries `levels_completed` and `state`, which is what the scorer reads.

The scorer is deliberately the SIMPLEST honest one: max `levels_completed` seen. It
REFUSES an episode with no transitions (nothing was played) rather than scoring 0 —
"the solver scored zero" and "the solver never ran" are different facts and the ARC
record (467 EMPTY of 1,128 runs) is what happens when they are conflated.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from awpredict.contracts import Refusal, Score, Transition

ACTIONS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)  # legal non-RESET ARC actions


def _grid_of(obs: Any) -> Any:
    data = obs.data if hasattr(obs, "data") else obs
    try:
        return [list(row) for row in data]
    except TypeError:
        return data


class ArcEnvAdapter:
    """EnvironmentAdapter over the offline ARC-AGI-3 SDK (via awgym's vendored env)."""

    def __init__(self, game_id: str, seed: Optional[int] = None,
                 max_steps: int = 80) -> None:
        from ..gym.orchestrator import GameSession
        from .games import pick_game

        self.game = pick_game(game_id)
        self.domain = f"arc:{self.game.game_id}"
        self._rng = random.Random(seed)
        # policy is never called: ProblemSession drives step() directly.
        self._session = GameSession(game=self.game, policy=lambda g, s: 0,
                                    max_steps=max_steps)
        self._levels = 0
        self._state = "NOT_FINISHED"

    # -- EnvironmentAdapter --------------------------------------------------
    def reset(self) -> Any:
        obs = self._session.env.reset()
        self._levels, self._state = 0, "NOT_FINISHED"
        return obs

    def observe(self, env_state: Any) -> Any:
        return _grid_of(env_state)

    def actions(self) -> Sequence[Any]:
        return list(ACTIONS)

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        a = int(action)
        data = None
        if a == 6:
            data = {"x": self._rng.randint(0, 127), "y": self._rng.randint(0, 127)}
        obs, reward, done, info = self._session._step(a, data)
        levels = getattr(info, "levels_completed", None)
        if levels is None and isinstance(info, dict):
            levels = info.get("levels_completed")
        if levels is not None:
            self._levels = max(self._levels, int(levels))
        state = getattr(info, "state", None) or (info.get("state") if isinstance(info, dict)
                                                 else None)
        if state:
            self._state = str(state)
        return obs, float(reward or 0.0), bool(done), {
            "levels_completed": self._levels, "state": self._state}


class ArcLevelScorer:
    """Score = the most levels completed in the episode. Refuses an unplayed episode."""

    metric = "levels"
    minimize = False

    def score(self, episode: List[Transition]) -> "Score | Refusal":
        if not episode:
            return Refusal("no transitions — the game was never played")
        levels = 0
        for t in episode:
            meta = _meta_of(t)
            try:
                levels = max(levels, int(meta.get("levels_completed", 0)))
            except (TypeError, ValueError):
                continue
        return Score(metric=self.metric, value=float(levels),
                     evidence=f"{len(episode)} transitions; final state "
                              f"{_meta_of(episode[-1]).get('state', '?')}")


def _meta_of(t: Any) -> Dict[str, Any]:
    """A Transition, a journaled dict, or any object carrying `.meta` — never raise."""
    meta = getattr(t, "meta", None)
    if meta is None and isinstance(t, dict):
        meta = t.get("meta")
    return meta if isinstance(meta, dict) else {}
