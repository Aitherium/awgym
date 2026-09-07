"""awrun `solve` kind: a queued ProblemSpec is claimed, played, journaled; the exit code is JOB
health (0 played-and-judged, 2 refused/crashed) and the verdict (kept/reverted/unjudged) is in
the message. A malformed item is a code-2 failure with the reason, never a silently skipped
queue entry."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("awrun")
from awgym.gym.awrun_solve import SOLVE_KIND, run_solve  # noqa: E402
from awgym.gym.journal import EpisodeJournal  # noqa: E402
from awrun.dispatcher import dispatch_once  # noqa: E402
from awrun.store import KINDS, RunStore  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))


def _toy_spec(**over):
    spec = {"problem_id": "toy:counter", "domain": "toy",
            "adapter_ref": "_toyworld:CounterEnv", "scorer_ref": "_toyworld:ReachScorer",
            "budget": {"steps": 10}, "memory_scope": "platform:*:*", "planner": "none"}
    spec.update(over)
    return spec


def test_solve_is_a_registered_kind():
    assert SOLVE_KIND in KINDS


def test_queued_solve_item_is_claimed_played_and_journaled(tmp_path: Path):
    store = RunStore(tmp_path / "awrun")
    item = store.submit(SOLVE_KIND, {"spec": _toy_spec(), "baseline": 0.0,
                                     "policy": "const", "const_action": 1,
                                     "journal_root": str(tmp_path / "solve")})
    done = dispatch_once(store, worker_id="t", run_fns={SOLVE_KIND: run_solve})
    assert done is not None and done.id == item.id
    assert done.status == "done", done.result
    eps = list((tmp_path / "solve").glob("*/ep-*/journal.jsonl"))
    assert len(eps) == 1
    ok, n, _ = EpisodeJournal(eps[0].parent).verify()
    assert ok and n == 6  # policy provenance + 3 transitions + attempt + outcome
    kinds = EpisodeJournal(eps[0].parent).kinds()
    assert kinds.get("policy") == 1  # the journal says WHICH policy played
    # state.json must agree with the chain AFTER the provenance row, or the gate
    # (check_solver_loop_closes SLC001) reads an honest episode as tampered.
    ok, n, head = EpisodeJournal(eps[0].parent).verify()
    state = json.loads((eps[0].parent / "state.json").read_text(encoding="utf-8"))
    assert state["rows"] == n and state["hash"] == head, state


def test_reverted_and_refused_map_to_exit_codes(tmp_path: Path):
    class Item:
        def __init__(self, spec):
            self.id, self.kind, self.spec = "r-test", SOLVE_KIND, spec

    root = tmp_path / "solve"
    code, msg = run_solve(Item({"spec": _toy_spec(), "baseline": 0.0, "policy": "const",
                                "const_action": -1, "journal_root": str(root)}))
    assert code == 0 and "reverted" in msg  # job health 0; the verdict is in the message
    code, msg = run_solve(Item({"spec": _toy_spec(scorer_ref="_toyworld:NoneScorer"),
                                "policy": "const", "const_action": 1,
                                "journal_root": str(root)}))
    assert code == 2 and "REFUSED" in msg
    code, msg = run_solve(Item({"spec": _toy_spec(), "policy": "const", "const_action": 1,
                                "journal_root": str(root)}))
    assert code == 0 and "unjudged" in msg  # no baseline: kept is None, not False


def test_malformed_items_fail_loudly_with_a_reason():
    class Item:
        id, kind = "r-bad", SOLVE_KIND

        def __init__(self, spec):
            self.spec = spec

    assert run_solve(Item({}))[0] == 2
    assert run_solve(Item({"spec": {"problem_id": "x"}}))[0] == 2
    code, msg = run_solve(Item({"spec": _toy_spec(memory_scope="nope")}))
    assert code == 2 and "memory_scope" in msg
    code, msg = run_solve(Item({"spec": _toy_spec(), "policy": "quantum"}))
    assert code == 2 and "unknown policy" in msg
    code, msg = run_solve(Item({"spec": _toy_spec(adapter_ref="no.such.module:X")}))
    assert code == 2 and "setup failed" in msg
