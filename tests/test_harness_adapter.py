"""HarnessEnv: a keep-or-revert harness played through the same ProblemSession as ARC.

A toy harness — a file holding one integer, a scorer that prints METRIC value=<int> and exits
non-zero for anything that is not an int — is enough to pin the loop's rules: a refused
proposal is REVERTED and never becomes a number; a worse proposal is reverted; a better one is
kept and becomes the new baseline; direction is honoured; the scorer refuses when nothing was
ever believed; the original bytes come back when nothing was kept.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from awpredict.contracts import Budget, ProblemSpec, Refusal, Score

pytest.importorskip("awevolve")
from awgym.envs.harness_adapter import HarnessEnv, HarnessScorer  # noqa: E402
from awgym.gym.journal import EpisodeJournal  # noqa: E402
from awgym.gym.problem import ProblemSession  # noqa: E402

SCORER_PY = """\
import pathlib, sys
raw = pathlib.Path("value.txt").read_text(encoding="utf-8").strip()
n = int(raw)          # not an int -> ValueError -> non-zero exit: the candidate is WRONG
print(f"METRIC value={n}")
"""


def _harness(tmp_path: Path, *, minimize: bool = False, start: str = "3\n") -> Path:
    (tmp_path / "value.txt").write_text(start, encoding="utf-8")
    (tmp_path / "score.py").write_text(SCORER_PY, encoding="utf-8")
    h = {"name": "toy-int", "mutable_file": str(tmp_path / "value.txt"),
         "eval_command": f'"{sys.executable}" score.py', "metric_regex": r"METRIC value=(-?\d+)",
         "metric_name": "value", "minimize": minimize, "time_budget_s": 60,
         "base_dir": str(tmp_path)}
    p = tmp_path / "harness.json"
    p.write_text(json.dumps(h), encoding="utf-8")
    return p


def _spec(hp: Path, **over) -> ProblemSpec:
    base = dict(problem_id="harness:toy-int", domain="harness",
                adapter_ref="awgym.envs.harness_adapter:HarnessEnv",
                scorer_ref="awgym.envs.harness_adapter:HarnessScorer",
                adapter_kwargs={"harness": str(hp)}, scorer_kwargs={"harness": str(hp)},
                budget=Budget(steps=6))
    base.update(over)
    return ProblemSpec(**base)


def _scripted(*proposals):
    it = iter(proposals)

    def policy(obs, session):
        return next(it)

    return policy


# ------------------------------------------------------------------ the env
def test_reset_measures_the_baseline_and_exposes_the_file(tmp_path: Path):
    env = HarnessEnv(harness=_harness(tmp_path))
    st = env.reset()
    assert st["score"] == 3.0 and st["last"]["baseline_ok"]
    obs = env.observe(st)
    assert obs["content"] == "3\n" and obs["score"] == 3.0 and env.actions() == ["propose"]
    assert env.domain == "harness:toy-int"


def test_better_is_kept_worse_and_refused_are_reverted(tmp_path: Path):
    hp = _harness(tmp_path)
    env = HarnessEnv(harness=hp)
    env.reset()
    _, r1, _, i1 = env.step("5\n")
    assert i1["kept"] and r1 == 2.0 and (tmp_path / "value.txt").read_text() == "5\n"
    _, r2, _, i2 = env.step("4\n")
    assert not i2["kept"] and r2 == -1.0 and (tmp_path / "value.txt").read_text() == "5\n"
    _, r3, _, i3 = env.step("abc\n")
    assert i3["strict_ok"] is False and i3["score"] is None and r3 == 0.0
    assert (tmp_path / "value.txt").read_text() == "5\n"
    assert i3["current_score"] == 5.0


def test_minimize_direction_is_honoured(tmp_path: Path):
    env = HarnessEnv(harness=_harness(tmp_path, minimize=True, start="10\n"))
    env.reset()
    _, r, _, info = env.step("7\n")
    assert info["kept"] and r == 3.0
    _, r2, _, info2 = env.step("9\n")
    assert not info2["kept"] and r2 == -2.0 and (tmp_path / "value.txt").read_text() == "7\n"


def test_a_noop_proposal_costs_nothing_and_changes_nothing(tmp_path: Path):
    env = HarnessEnv(harness=_harness(tmp_path))
    env.reset()
    _, r, _, info = env.step("3\n")
    assert r == 0.0 and info["reason"] == "no-op proposal" and info["current_score"] == 3.0


def test_close_restores_original_only_when_nothing_was_kept(tmp_path: Path):
    hp = _harness(tmp_path)
    env = HarnessEnv(harness=hp)
    env.reset()
    env.step("abc\n")
    env.close()
    assert (tmp_path / "value.txt").read_text() == "3\n"
    env2 = HarnessEnv(harness=hp)
    env2.reset()
    env2.step("8\n")
    env2.close()
    assert (tmp_path / "value.txt").read_text() == "8\n"  # a kept file IS the result


def test_non_string_action_is_refused_loudly(tmp_path: Path):
    env = HarnessEnv(harness=_harness(tmp_path))
    env.reset()
    with pytest.raises(ValueError):
        env.step(42)


def test_unavailable_harness_is_refused_at_construction(tmp_path: Path):
    hp = _harness(tmp_path)
    (tmp_path / "value.txt").unlink()
    with pytest.raises(Exception, match="not available"):
        HarnessEnv(harness=hp)


# --------------------------------------------------------------- the scorer
def test_scorer_reports_current_believed_score_and_refuses_when_none(tmp_path: Path):
    hp = _harness(tmp_path)
    scorer = HarnessScorer(harness=hp)
    assert scorer.metric == "value" and scorer.minimize is False
    assert isinstance(scorer.score([]), Refusal)

    class T:  # a Transition stand-in carrying only meta
        def __init__(self, meta):
            self.meta = meta

    out = scorer.score([T({"strict_ok": False, "current_score": None})])
    assert isinstance(out, Refusal) and "no trial produced" in out.reason
    ok = scorer.score([T({"strict_ok": True, "current_score": 5.0}),
                       T({"strict_ok": False, "current_score": 5.0})])
    assert isinstance(ok, Score) and ok.value == 5.0 and "some refused" in ok.evidence


# --------------------------------------------- through the general solver loop
def test_problem_session_plays_a_harness_end_to_end(tmp_path: Path):
    hp = _harness(tmp_path)
    policy = _scripted("5\n", "abc\n", "4\n", "9\n")
    s = ProblemSession(_spec(hp, budget=Budget(steps=4)), policy, baseline=3.0,
                       journal_root=tmp_path / "j")
    ep = s.run()
    assert ep.stop_reason == "budget.steps" and len(ep.transitions) == 4
    assert ep.outcome.score.value == 9.0 and ep.outcome.kept is True
    assert (tmp_path / "value.txt").read_text() == "9\n"
    metas = [t.meta for t in ep.transitions]
    assert [m["kept"] for m in metas] == [True, False, False, True]
    assert metas[1]["strict_ok"] is False
    ok, n, _ = EpisodeJournal(ep.journal_dir).verify()
    assert ok and n == 6  # 4 transitions + attempt + outcome
    assert ep.transitions[0].domain == "harness"


def test_episode_of_only_refusals_is_reverted_not_scored_as_zero(tmp_path: Path):
    hp = _harness(tmp_path)
    s = ProblemSession(_spec(hp, budget=Budget(steps=2)), _scripted("x\n", "y\n"),
                       baseline=3.0)
    ep = s.run()
    # the baseline WAS believed at reset, so the current score is the baseline: not better
    assert ep.outcome.score.value == 3.0 and ep.outcome.kept is False
    assert (tmp_path / "value.txt").read_text() == "3\n"


def test_unscorable_baseline_and_only_refusals_is_a_refusal(tmp_path: Path):
    hp = _harness(tmp_path, start="garbage\n")
    s = ProblemSession(_spec(hp, budget=Budget(steps=1)), _scripted("nope\n"), baseline=0.0)
    ep = s.run()
    assert ep.outcome.refusal is not None and "no trial produced" in ep.outcome.refusal.reason
    assert ep.outcome.kept is None
