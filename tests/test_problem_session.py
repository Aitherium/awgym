"""ProblemSession: the domain-neutral solving loop, judged by a scorer that can refuse.

No ARC SDK here — a toy counter environment is enough to pin every rule the loop
enforces: validation before the first step, budget stops, a scorer defect recorded as a
REFUSAL (never a zero), `kept` only against a baseline, facts scoped and journaled, and a
hash-chained journal that detects tampering.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from awgym.gym.journal import EpisodeJournal, JournalError
from awgym.gym.problem import (
    ProblemError,
    ProblemSession,
    const_policy,
    load_ref,
    main,
    random_policy,
)
from awpredict.contracts import Budget, ProblemSpec, Transition

# toy world lives in tests/_toyworld.py so load_ref can import it by name
sys.path.insert(0, str(Path(__file__).parent))
from _toyworld import CounterEnv, CrashScorer, NoneScorer, ReachScorer  # noqa: E402


def _spec(**over) -> ProblemSpec:
    base = dict(problem_id="toy:counter", domain="toy",
                adapter_ref="_toyworld:CounterEnv",
                scorer_ref="_toyworld:ReachScorer",
                budget=Budget(steps=10))
    base.update(over)
    return ProblemSpec(**base)


# ------------------------------------------------------------------- the loop
def test_a_winning_episode_is_kept_against_its_baseline(tmp_path: Path):
    s = ProblemSession(_spec(), const_policy(1), adapter=CounterEnv(), scorer=ReachScorer(),
                       baseline=1.0, journal_root=tmp_path)
    ep = s.run()
    assert ep.stop_reason == "done"
    assert len(ep.transitions) == 3
    assert ep.outcome.score.value == 3.0 and ep.outcome.kept is True
    assert ep.outcome.transitions_n == 3 and ep.outcome.problems() == []
    assert ep.attempt.first_effect_step == 0 and ep.attempt.opening_actions == [1, 1, 1]
    assert isinstance(ep.transitions[0], Transition) and ep.transitions[0].domain == "toy"


def test_budget_steps_stops_an_endless_walk():
    s = ProblemSession(_spec(budget=Budget(steps=4)), const_policy(-1),
                       adapter=CounterEnv(), scorer=ReachScorer(), baseline=0.0)
    ep = s.run()
    assert ep.stop_reason == "budget.steps" and len(ep.transitions) == 4
    assert ep.outcome.kept is False  # -1 is not better than 0


def test_budget_seconds_stops_via_the_injected_clock():
    ticks = iter([0.0, 0.0, 0.5, 1.5, 2.0, 9.0])
    s = ProblemSession(_spec(budget=Budget(steps=0, seconds=1.0)), const_policy(-1),
                       adapter=CounterEnv(), scorer=ReachScorer(),
                       clock=lambda: next(ticks))
    ep = s.run()
    assert ep.stop_reason == "budget.seconds"


def test_without_a_baseline_kept_is_unjudged_not_false():
    ep = ProblemSession(_spec(), const_policy(1), adapter=CounterEnv(),
                        scorer=ReachScorer()).run()
    assert ep.outcome.score is not None and ep.outcome.kept is None
    assert ep.outcome.problems() == []


# -------------------------------------------------------------- the scorer arm
def test_scorer_returning_none_is_a_refusal_naming_the_defect_never_a_zero():
    ep = ProblemSession(_spec(), const_policy(1), adapter=CounterEnv(), scorer=NoneScorer(),
                        baseline=0.0).run()
    assert ep.outcome.score is None
    assert "scorer defect" in ep.outcome.refusal.reason and "None" in ep.outcome.refusal.reason
    assert ep.outcome.kept is None
    assert "REFUSED" in ep.summary()


def test_crashing_scorer_is_a_refusal_not_an_exception():
    ep = ProblemSession(_spec(), const_policy(1), adapter=CounterEnv(),
                        scorer=CrashScorer()).run()
    assert "judge exploded" in ep.outcome.refusal.reason


def test_empty_episode_is_refused_by_the_scorer_itself():
    def broken_policy(obs, session):
        raise KeyError("no idea")

    ep = ProblemSession(_spec(), broken_policy, adapter=CounterEnv(),
                        scorer=ReachScorer()).run()
    assert ep.stop_reason.startswith("policy_error")
    assert ep.outcome.refusal is not None and "no transitions" in ep.outcome.refusal.reason


def test_a_step_error_ends_the_episode_but_keeps_what_was_played():
    class Flaky(CounterEnv):
        def step(self, action):
            if self.state >= 2:
                raise OSError("env died")
            return super().step(action)

    ep = ProblemSession(_spec(), const_policy(1), adapter=Flaky(), scorer=ReachScorer(),
                        baseline=0.0).run()
    assert ep.stop_reason == "step_error:OSError"
    assert len(ep.transitions) == 2 and ep.outcome.score.value == 2.0


# ------------------------------------------------------------ validation arms
def test_malformed_spec_and_half_adapters_are_refused_before_any_step():
    with pytest.raises(ProblemError):
        ProblemSession(_spec(memory_scope="nope"), const_policy(1),
                       adapter=CounterEnv(), scorer=ReachScorer())

    class HalfEnv:
        domain = "x"

        def observe(self, s):
            return s

    with pytest.raises(ProblemError, match="actions"):
        ProblemSession(_spec(), const_policy(1), adapter=HalfEnv(), scorer=ReachScorer())

    class HalfScorer:
        metric = "m"

    with pytest.raises(ProblemError, match="Scorer"):
        ProblemSession(_spec(), const_policy(1), adapter=CounterEnv(), scorer=HalfScorer())


def test_load_ref_refuses_bad_refs_and_loads_good_ones():
    with pytest.raises(ProblemError):
        load_ref("awgym.gym.problem")
    with pytest.raises(ProblemError):
        load_ref("awgym.gym.problem:NoSuchThing")
    assert callable(load_ref("awgym.gym.problem:const_policy", {"action": 1}))


# ------------------------------------------------------------------- learning
def test_facts_are_scoped_by_the_spec_and_journaled(tmp_path: Path):
    def curious(obs, session):
        if obs == 1:
            session.learn("direction", "+1 raises the counter", 0.9)
        return 1

    s = ProblemSession(_spec(memory_scope="acme:alice:toy"), curious, adapter=CounterEnv(),
                       scorer=ReachScorer(), baseline=0.0, journal_root=tmp_path)
    ep = s.run()
    assert len(ep.facts) == 1 and ep.facts[0].memory_scope == "acme:alice:toy"
    assert ep.outcome.facts_n == 1
    kinds = EpisodeJournal(ep.journal_dir).kinds()
    assert kinds["fact"] == 1 and kinds["transition"] == 3
    assert kinds["attempt"] == 1 and kinds["outcome"] == 1


# ------------------------------------------------------------------- journal
def test_journal_verifies_and_detects_tampering(tmp_path: Path):
    ep = ProblemSession(_spec(), const_policy(1), adapter=CounterEnv(), scorer=ReachScorer(),
                        baseline=0.0, journal_root=tmp_path).run()
    j = EpisodeJournal(ep.journal_dir)
    ok, n, head = j.verify()
    assert ok and n == 5  # 3 transitions + attempt + outcome
    assert j.state()["hash"] == head and j.state()["rows"] == 5
    lines = j.journal_path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace('"reward":2.0', '"reward":9.0')
    j.journal_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    bad, at, _ = j.verify()
    assert not bad and at == 1


def test_journal_refuses_to_reuse_a_dir_and_verifies_empty_as_clean(tmp_path: Path):
    j = EpisodeJournal(tmp_path / "e")
    j.init({"problem_id": "p"}, "e")
    assert j.verify() == (True, 0, "0" * 64)
    j.append("transition", {"step": 0})
    with pytest.raises(JournalError):
        j.init({"problem_id": "p"}, "e")
    with pytest.raises(JournalError):
        EpisodeJournal(tmp_path / "missing").verify()


# ----------------------------------------------------------------------- CLI
def test_cli_plays_a_json_spec_and_reports_exit_codes(tmp_path: Path):
    spec = _spec().to_dict()
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    assert main([str(p), "--policy", "const", "--const-action", "1", "--baseline", "0",
                 "--journal-root", str(tmp_path / "j")]) == 0
    assert main([str(p), "--policy", "const", "--const-action", "-1", "--baseline", "0"]) == 1
    bad = dict(spec, scorer_ref="_toyworld:NoneScorer")
    p.write_text(json.dumps(bad), encoding="utf-8")
    assert main([str(p), "--policy", "const", "--const-action", "1"]) == 2
    worse = dict(spec, memory_scope="broken")
    p.write_text(json.dumps(worse), encoding="utf-8")
    assert main([str(p)]) == 2


def test_random_policy_is_seeded_and_refuses_an_empty_action_set():
    class NoActs(CounterEnv):
        def actions(self):
            return []

    s = ProblemSession(_spec(), random_policy(seed=1), adapter=NoActs(), scorer=ReachScorer())
    ep = s.run()
    assert ep.stop_reason.startswith("policy_error:ProblemError")
    a = ProblemSession(_spec(), random_policy(seed=3), adapter=CounterEnv(),
                       scorer=ReachScorer()).run()
    b = ProblemSession(_spec(), random_policy(seed=3), adapter=CounterEnv(),
                       scorer=ReachScorer()).run()
    assert [t.action for t in a.transitions] == [t.action for t in b.transitions]
