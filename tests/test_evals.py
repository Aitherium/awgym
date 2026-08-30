"""Eval layer — the latent gate's skill arithmetic (no LeWM needed) and the
scorer's DEAD contract (real run against an unreachable base)."""

import os


from awgym.evals.latent_gate import gate_skill, identity_surprise
from awgym.gym.orchestrator import Transition

ENVS = os.path.join(
    os.environ.get("ARC_GYM_DREAMTEAM_ROOT")
    or r"E:\AitherOS-Data\arc-agi-3\dream-team",
    "environment_files_generated")


def _t(diff: int) -> Transition:
    row = [0] * 64
    grid = [row[:] for _ in range(64)]
    nxt = [row[:] for _ in range(64)]
    for r in range(diff):
        nxt[r][0] = 1
    return Transition(game_id="al7306-000afcc7", step=0, grid=grid,
                      action=1, next_grid=nxt, reward=0.0, done=False)


def test_identity_surprise_is_latent_space_l2():
    # identity = ||encode(grid) - encode(next)||^2 — latent space, the same
    # units as /surprise (a grid-space proxy would mix scales: measured bug,
    # fixed 2026-08-30)
    calls = []

    class FakeClient:
        def encode(self, grid):
            calls.append(1)
            return [1.0] * 4 if grid[0][0] == 1 else [0.0] * 4

    t = _t(1)
    t.next_grid[0][0] = 1
    ident = identity_surprise(t, FakeClient())
    assert ident == 4.0  # (1-0)^2 * 4 dims
    assert len(calls) == 2


def test_gate_skill_beats_identity():
    class FakeClient:
        def encode(self, grid):
            return [1.0] * 4 if grid[0][0] == 1 else [0.0] * 4

        def surprise(self, grid, action, next_grid, ctx=None):
            return 1.0  # half the identity's 4.0 -> skill 0.75

    t = _t(1)
    t.next_grid[0][0] = 1
    res = gate_skill([t], FakeClient())
    assert res is not None
    assert abs(res["skill"] - 0.75) < 1e-9
    assert res["n"] == 1


def test_gate_skill_none_when_no_surprises():
    class DeadClient:
        def encode(self, grid):
            return [0.0] * 4

        def surprise(self, grid, action, next_grid, ctx=None):
            return None

    assert gate_skill([_t(4)], DeadClient()) is None


def test_gate_skill_degenerate_identity_is_none():
    class ZeroClient:
        def encode(self, grid):
            return [0.0] * 4  # identical grids -> identity 0

        def surprise(self, grid, action, next_grid, ctx=None):
            return 0.0

    # no change at all -> identity 0 -> degenerate, must be None (never 0/0)
    assert gate_skill([_t(0)], ZeroClient()) is None


def test_scorer_dead_when_lewm_unreachable():
    # real run, real subprocess, unreachable base -> exit 2, never 0
    import subprocess
    import sys
    p = subprocess.run(
        [sys.executable, "-m", "awgym.evals.score_awgym",
         "--base", "https://127.0.0.1:9", "--ca", os.devnull],
        capture_output=True, text=True, timeout=60)
    assert p.returncode == 2
    assert "DEAD" in p.stdout
