"""Gym core — GameSession produces transitions, Recorder writes the
live-lane envelope, Ledger appends run rows. These run against the REAL
vendored env (offline, no network)."""

import json
import os

import pytest

from awgym.envs.games import game_pool, pick_game
from awgym.gym.orchestrator import GameSession
from awgym.gym.recording import Recorder
from awgym.gym.ledger import append_run, read_ledger

ENVS = os.path.join(
    os.environ.get("ARC_GYM_DREAMTEAM_ROOT")
    or r"E:\AitherOS-Data\arc-agi-3\dream-team",
    "environment_files_generated")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(ENVS),
    reason="vendored generated environments not present on this host")


def _const_policy(action: int):
    def policy(grid, session):  # noqa: ARG001
        return action
    return policy


def test_pool_scans_generated_games():
    pool = game_pool()
    assert len(pool) >= 25, f"expected the 25 generated games, got {len(pool)}"
    g = pool.get("al7306-000afcc7")
    assert g is not None and g.baseline_actions is not None


def test_session_plays_and_captures_transitions():
    game = pick_game("al7306-000afcc7")
    session = GameSession(game=game, policy=_const_policy(1), max_steps=8)
    transitions = session.play()
    assert len(transitions) == 8  # never hits done in 8 steps of action 1
    t = transitions[0]
    assert t.game_id == "al7306-000afcc7"
    assert len(t.grid) == 64 and len(t.next_grid) == 64
    assert 0 <= t.action <= 7 and t.step == 0
    # grids change or not — but shapes hold
    assert all(len(row) == 64 for row in t.next_grid)


def test_recorder_matches_live_lane_envelope(tmp_path):
    game = pick_game("al7306-000afcc7")
    session = GameSession(game=game, policy=_const_policy(2), max_steps=3)
    transitions = session.play()
    rec_path = tmp_path / "run.recording.jsonl"
    with Recorder(rec_path) as rec:
        for t in transitions:
            rec.write_transition(t)
    lines = rec_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    line = json.loads(lines[0])
    assert "timestamp" in line and "data" in line
    data = line["data"]
    # the live-lane envelope: frame wrapped in an extra list, key names exact
    assert data["game_id"] == "al7306-000afcc7"
    assert isinstance(data["frame"], list) and len(data["frame"]) == 1
    assert len(data["frame"][0]) == 64
    assert set(data) >= {"state", "levels_completed", "win_levels",
                         "action_input", "guid", "full_reset",
                         "available_actions"}


def test_ledger_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_GYM_DATA_ROOT", str(tmp_path))
    row = append_run({"game_id": "al7306-000afcc7", "steps": 3, "score": 0})
    assert row["run_id"].startswith("gym-")
    rows = read_ledger()
    assert len(rows) == 1 and rows[0]["game_id"] == "al7306-000afcc7"
