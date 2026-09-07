"""Toy environment + scorers the ProblemSession tests load BY REF ("_toyworld:CounterEnv").

Lives beside the tests, not in the package: fixtures do not ship to PyPI.
"""

from __future__ import annotations

from awpredict.contracts import Refusal, Score


class CounterEnv:
    """State is an int; +1/-1 moves it; done at >= goal."""

    domain = "toy"

    def __init__(self, goal: int = 3) -> None:
        self.goal = goal
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def observe(self, env_state):
        return int(env_state)

    def actions(self):
        return [-1, 1]

    def step(self, action):
        self.state += int(action)
        return self.state, float(self.state), self.state >= self.goal, {"goal": self.goal}


class ReachScorer:
    metric = "reached"
    minimize = False

    def score(self, episode):
        if not episode:
            return Refusal("no transitions")
        return Score(self.metric, float(max(t.next_obs for t in episode)))


class NoneScorer:
    """The defect shape: 'could not judge' expressed as nothing."""

    metric = "reached"
    minimize = False

    def score(self, episode):
        return None


class CrashScorer:
    metric = "reached"
    minimize = False

    def score(self, episode):
        raise RuntimeError("judge exploded")
