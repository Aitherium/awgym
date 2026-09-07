"""ProblemSession — play ONE problem through ANY EnvironmentAdapter, judged by a Scorer
that may refuse, with every record journaled and replayable.

    from awgym.gym.problem import ProblemSession, load_ref
    spec = ProblemSpec(problem_id="arc:ls20", domain="arc",
                       adapter_ref="awgym.envs.arc_adapter:ArcEnvAdapter",
                       scorer_ref="awgym.envs.arc_adapter:ArcLevelScorer",
                       adapter_kwargs={"game_id": "ls20"})
    session = ProblemSession(spec, policy=random_policy(seed=7), baseline=0.0)
    episode = session.run()
    episode.outcome            # Outcome(score=Score(levels=1.0) | refusal=..., kept=True|False)

    python -m awgym.gym.problem spec.json --baseline 0 --journal-root /data/solve

This is `GameSession` (awgym/gym/orchestrator.py) with the ARC removed: the world is an
`awpredict.contracts.EnvironmentAdapter` (observe / actions / step), the judge is a
`Scorer`, the stop is a `Budget`, and the four records every domain emits — Transition,
Attempt, Fact, Outcome — are the ones the contract names. The ARC game becomes ONE adapter
(`awgym.envs.arc_adapter`), a kernel harness another, a fleet defect a third. Nothing in
here knows what a grid is.

Rules this module enforces, because the ARC lane paid for each of them:
  * The spec, the adapter and the scorer are VALIDATED before the first step.
    A run over a half-shaped adapter used to read as "the policy is bad".
  * A scorer that returns None/0.0/garbage is recorded as a REFUSAL naming the
    defect — never as a score of zero (`check_score`).
  * `kept` is only ever decided against an explicit `baseline`.
  * Every record is journaled in a hash chain BEFORE the next step is taken, so a
    crashed run leaves a verifiable partial journal, not nothing.
  * The policy is the only thing that decides an action. No planner is switched on
    by an env flag: `spec.planner` is data the policy factory reads.

Import boundary: this package ships to PyPI. `awpredict` is a declared sibling brick;
nothing from the monorepo (`lib.*`, `services.*`) may be imported here.
"""

from __future__ import annotations

import importlib
import json
import random
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from awpredict.contracts import (
    Attempt,
    EnvironmentAdapter,
    Fact,
    Outcome,
    ProblemSpec,
    Refusal,
    Score,
    Scorer,
    Transition,
    check_score,
    conforms,
)

from .journal import EpisodeJournal

Policy = Callable[[Any, "ProblemSession"], Any]  # (obs, session) -> action


class ProblemError(ValueError):
    """The problem could not be set up honestly — refused before the first step."""


# ------------------------------------------------------------------ loading refs
def load_ref(ref: str, kwargs: Optional[Dict[str, Any]] = None) -> Any:
    """Instantiate ``"package.module:ClassName"`` with kwargs. Raises loudly."""
    mod_name, sep, cls_name = (ref or "").partition(":")
    if not sep or not mod_name or not cls_name:
        raise ProblemError(f"ref must be 'package.module:ClassName', got {ref!r}")
    module = importlib.import_module(mod_name)
    try:
        cls = getattr(module, cls_name)
    except AttributeError as exc:
        raise ProblemError(f"{mod_name} has no attribute {cls_name!r}") from exc
    return cls(**(kwargs or {}))


# --------------------------------------------------------------------- policies
def random_policy(seed: Optional[int] = None) -> Policy:
    rng = random.Random(seed)

    def policy(obs: Any, session: "ProblemSession") -> Any:  # noqa: ARG001
        acts = list(session.adapter.actions())
        if not acts:
            raise ProblemError("adapter.actions() returned no actions")
        return rng.choice(acts)

    return policy


def const_policy(action: Any) -> Policy:
    def policy(obs: Any, session: "ProblemSession") -> Any:  # noqa: ARG001
        return action

    return policy


# ---------------------------------------------------------------------- episode
@dataclass
class Episode:
    spec: ProblemSpec
    episode_id: str
    transitions: List[Transition] = field(default_factory=list)
    facts: List[Fact] = field(default_factory=list)
    attempt: Optional[Attempt] = None
    outcome: Optional[Outcome] = None
    journal_dir: Optional[Path] = None
    stop_reason: str = ""

    def summary(self) -> str:
        o = self.outcome
        if o is None:
            return f"{self.spec.problem_id}: no outcome ({self.stop_reason or 'unfinished'})"
        if o.score is not None:
            verdict = f"{o.score.metric}={o.score.value:g}"
            kept = "kept" if o.kept else ("reverted" if o.kept is False else "unjudged")
        else:
            verdict, kept = f"REFUSED: {o.refusal.reason}", "unjudged"
        return (f"{self.spec.problem_id}: {verdict} baseline={o.baseline} {kept} "
                f"steps={len(self.transitions)} facts={len(self.facts)} "
                f"stop={self.stop_reason} journal={self.journal_dir}")


# ---------------------------------------------------------------------- session
class ProblemSession:
    """One problem, one adapter, one scorer, one budget, one journal."""

    def __init__(
        self,
        spec: ProblemSpec,
        policy: Policy,
        *,
        adapter: Any = None,
        scorer: Any = None,
        baseline: Optional[float] = None,
        journal_root: Optional[str | Path] = None,
        episode_id: Optional[str] = None,
        clock: Callable[[], float] = time.monotonic,
        model_used: str = "",
    ):
        problems = spec.problems()
        if problems:
            raise ProblemError("spec is not well-formed: " + "; ".join(problems))
        self.spec = spec
        self.policy = policy
        self.adapter = adapter if adapter is not None else load_ref(
            spec.adapter_ref, spec.adapter_kwargs)
        self.scorer = scorer if scorer is not None else load_ref(
            spec.scorer_ref, spec.scorer_kwargs)
        missing = conforms(self.adapter, EnvironmentAdapter)
        if missing:
            raise ProblemError(f"adapter {type(self.adapter).__name__} is missing "
                               f"{missing} — not an EnvironmentAdapter")
        missing = conforms(self.scorer, Scorer)
        if missing:
            raise ProblemError(f"scorer {type(self.scorer).__name__} is missing "
                               f"{missing} — not a Scorer")
        self.baseline = baseline
        self.clock = clock
        self.model_used = model_used
        self.episode_id = episode_id or f"ep-{uuid.uuid4().hex[:12]}"
        self.journal: Optional[EpisodeJournal] = None
        if journal_root is not None:
            self.journal = EpisodeJournal(Path(journal_root) / _safe(spec.problem_id)
                                          / self.episode_id)
        self.episode = Episode(spec=spec, episode_id=self.episode_id,
                               journal_dir=self.journal.dir if self.journal else None)

    # ---------------------------------------------------------------- learning
    def learn(self, relation: str, fact: str, confidence: float = 0.7) -> Fact:
        """A policy's way to write something down. Scoped by the spec, journaled."""
        f = Fact(memory_scope=self.spec.memory_scope, relation=relation, fact=fact,
                 confidence=confidence, source_episode=self.episode_id)
        problems = f.problems()
        if problems:
            raise ProblemError("fact is not well-formed: " + "; ".join(problems))
        self.episode.facts.append(f)
        self._journal("fact", f.as_dict())
        return f

    # -------------------------------------------------------------------- run
    def run(self, initial_state: Any = None) -> Episode:
        if self.journal is not None:
            self.journal.init(self.spec.to_dict(), self.episode_id)
        started = self.clock()
        budget = self.spec.budget
        ep = self.episode

        state = initial_state
        if state is None and hasattr(self.adapter, "reset"):
            state = self.adapter.reset()
        obs = self.adapter.observe(state)
        first_effect: Optional[int] = None
        opening: List[Any] = []
        trajectory: List[float] = []
        step = 0
        stop = ""
        while True:
            if budget.steps and step >= budget.steps:
                stop = "budget.steps"
                break
            if budget.seconds and (self.clock() - started) >= budget.seconds:
                stop = "budget.seconds"
                break
            try:
                action = self.policy(obs, self)
            except Exception as exc:  # noqa: BLE001 - a policy crash ends the episode
                stop = f"policy_error:{type(exc).__name__}"
                self._journal("error", {"where": "policy", "step": step, "error": str(exc)})
                break
            try:
                state, reward, done, info = self.adapter.step(action)
            except Exception as exc:  # noqa: BLE001 - one broken step ends the episode
                stop = f"step_error:{type(exc).__name__}"
                self._journal("error", {"where": "step", "step": step, "error": str(exc)})
                break
            next_obs = self.adapter.observe(state)
            if first_effect is None and next_obs != obs:
                first_effect = step
            if len(opening) < 8:
                opening.append(action)
            trajectory.append(float(reward or 0.0))
            t = Transition(domain=self.spec.domain, episode=self.episode_id, step=step,
                           obs=obs, action=action, next_obs=next_obs,
                           reward=float(reward or 0.0), done=bool(done),
                           meta=dict(info) if isinstance(info, dict) else {})
            ep.transitions.append(t)
            self._journal("transition", t.as_dict())
            obs = next_obs
            step += 1
            if done:
                stop = "done"
                break

        ep.stop_reason = stop
        verdict = self._judge(ep.transitions)
        score = verdict if isinstance(verdict, Score) else None
        refusal = verdict if isinstance(verdict, Refusal) else None
        ep.attempt = Attempt(domain=self.spec.domain, problem_id=self.spec.problem_id,
                             episode=self.episode_id, steps=step, score=score,
                             refusal=refusal, first_effect_step=first_effect,
                             opening_actions=opening, trajectory=trajectory)
        self._journal("attempt", ep.attempt.as_dict())
        kept: Optional[bool] = None
        if score is not None and self.baseline is not None:
            kept = score.better_than(self.baseline)
        ep.outcome = Outcome(problem_id=self.spec.problem_id, domain=self.spec.domain,
                             episode=self.episode_id, score=score, refusal=refusal,
                             baseline=self.baseline, kept=kept,
                             transitions_n=len(ep.transitions), facts_n=len(ep.facts),
                             duration_ms=(self.clock() - started) * 1000.0,
                             model_used=self.model_used,
                             meta={"stop_reason": stop, "planner": self.spec.planner})
        problems = ep.outcome.problems()
        if problems:  # cannot happen by construction; if it does, say so in the record
            ep.outcome.meta["outcome_problems"] = problems
        self._journal("outcome", ep.outcome.as_dict())
        if self.journal is not None:
            self.journal.close(self.episode_id)
        # An adapter that holds real state (a harness's original bytes, an SDK session)
        # gets told the episode is over. Its failure is journaled, never raised: the
        # verdict above is already recorded and must not be lost to cleanup.
        close = getattr(self.adapter, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # noqa: BLE001
                self._journal("error", {"where": "close", "error": str(exc)})
        return ep

    # ---------------------------------------------------------------- helpers
    def _judge(self, transitions: List[Transition]) -> Score | Refusal:
        try:
            result = self.scorer.score(transitions)
        except Exception as exc:  # noqa: BLE001 - a crashing judge is a refusal, not a zero
            return Refusal(f"scorer raised {type(exc).__name__}: {exc}")
        problems = check_score(result)
        if problems:
            return Refusal("scorer defect: " + "; ".join(problems))
        return result

    def _journal(self, kind: str, payload: dict) -> None:
        if self.journal is not None:
            self.journal.append(kind, payload)


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:80] or "problem"


# ------------------------------------------------------------------------- CLI
def _load_spec(path: str) -> ProblemSpec:
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # optional; JSON needs nothing
        except ImportError as exc:
            raise ProblemError("a .yaml spec needs pyyaml; use .json") from exc
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    return ProblemSpec.from_dict(raw)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m awgym.gym.problem",
                                description="play one problem spec, judged and journaled")
    p.add_argument("spec", help="ProblemSpec as .json or .yaml")
    p.add_argument("--baseline", type=float, default=None,
                   help="score to beat; without it the outcome is unjudged (kept=None)")
    p.add_argument("--journal-root", default=None, help="dir to write <problem>/<episode>/")
    p.add_argument("--policy", default="random", choices=["random", "const"])
    p.add_argument("--const-action", default="0")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--json", action="store_true", help="print the Outcome as JSON")
    args = p.parse_args(argv)
    # The vendored ARC env logs emoji through loguru; on a cp1252 console that is a
    # UnicodeEncodeError inside the logging machinery on every step. Not our
    # record, not our verdict — but it drowns both, so widen the stream.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                continue  # stream already usable as-is; the widening is best-effort

    spec = _load_spec(args.spec)
    if args.policy == "const":
        act: Any = args.const_action
        try:
            act = int(act)
        except ValueError:
            act = args.const_action  # non-numeric action passes through as-is
        policy = const_policy(act)
    else:
        policy = random_policy(args.seed)
    try:
        session = ProblemSession(spec, policy, baseline=args.baseline,
                                 journal_root=args.journal_root)
    except ProblemError as exc:
        print(f"REFUSED: {exc}")
        return 2
    ep = session.run()
    if args.json:
        print(json.dumps(ep.outcome.as_dict(), indent=1, default=str))
    else:
        print(ep.summary())
    if ep.outcome is None or ep.outcome.refusal is not None:
        return 2
    return 0 if ep.outcome.kept in (True, None) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
