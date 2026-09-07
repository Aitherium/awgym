"""Phase 2b solver-league unit tests — the DECISION parser, the sequence
policy, the workspace layout, and the A/B stub backend."""

from __future__ import annotations

import json

from awgym.solver.leader import _parse_actions, _SequencePolicy, _StubWM
from awgym.solver.workspace import RunWorkspace


class _FakeGame:
    game_id = "testgame"
    tags = "t1"
    baseline_actions = 2


class _FakeSession:
    pass


def test_parse_decision_extracts_block() -> None:
    reply = (
        "Some analysis prose.\n\n"
        "DECISION:\n"
        "  game_id: testgame\n"
        "  goal: [VALIDATE] the green dot's motion rule\n"
        "  hypothesis: dot moves one cell per step\n"
        "  action_sequence: [1, 2, 3]\n"
        "  confidence: 0.6\n"
    )
    d = RunWorkspace.parse_decision(reply)
    assert d is not None
    assert d["game_id"] == "testgame"
    assert d["action_sequence"] == "[1, 2, 3]"
    assert d["confidence"] == "0.6"


def test_parse_decision_absent() -> None:
    assert RunWorkspace.parse_decision("no block here") is None


def test_parse_actions_list_and_string() -> None:
    assert _parse_actions({"action_sequence": "[1, 2, 3]"}) == [1, 2, 3]
    assert _parse_actions({"action_sequence": "1 2 3"}) == [1, 2, 3]
    assert _parse_actions({"action_sequence": []}) == []


def test_sequence_policy_plays_then_falls_back() -> None:
    policy = _SequencePolicy([3, 4, 5])
    session = _FakeSession()
    grid: list = []
    got = [policy(grid, session) for _ in range(5)]
    assert got[:3] == [3, 4, 5]
    assert all(0 <= a <= 7 for a in got[3:])


def test_workspace_layout_and_files() -> None:
    ws = RunWorkspace(game_id="testgame", run_id="testrun")
    assert ws.root.name.endswith("testrun")
    assert ws.root.parent.parent.name == "testgame"
    ws.log_round(1, {"goal": "x"})
    row = json.loads((ws.root / "DECISIONS.jsonl").read_text(encoding="utf-8"))
    assert row["round"] == 1
    ws.append_text("critique.md", "#feedback @leader [LOW] check that")
    assert "check that" in (ws.root / "critique.md").read_text(encoding="utf-8")
    s = ws.summary()
    assert "DECISIONS.jsonl" in s["files"]


def test_stub_backend_answers_identity() -> None:
    stub = _StubWM()
    assert stub.health()["device"] == "stub"
    assert stub.observe([], 0, [])["ok"] is True
    assert stub.surprise([], 0, []) == 0.0


def test_parse_decision_single_line() -> None:
    """The shape the model actually emits (measured 2026-08-30): the whole
    DECISION block on ONE line. The first line-pair split turned the block
    into one value and the round played nothing."""
    reply = ("DECISION: game_id: g1 goal: [GOAL] hypothesis: h "
             "action_sequence: [1, 2] steps: 8 confidence: 0.5")
    d = RunWorkspace.parse_decision(reply)
    assert d is not None
    assert d["game_id"] == "g1"
    assert d["goal"] == "[GOAL]"
    assert d["action_sequence"] == "[1, 2]"
    assert d["confidence"] == "0.5"
