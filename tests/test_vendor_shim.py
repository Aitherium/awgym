"""The vendor shim is the load-bearing seam — these prove both directions.

Positive: with the vendored tree present, the clean seams resolve.
Negative: with a bogus ARC_GYM_DREAMTEAM_ROOT, the shim raises RuntimeError
with the fix spelled out — a silent fallback would read as "gym works" while
every env import died.
"""

import os

import pytest

from awgym.config import dream_team_root

# Import the shim ONLY when the tree exists: the vendored tree is a RUNTIME
# input (documented in the README), and a publish runner has none — the
# module-level import raised RuntimeError at COLLECTION there, killing the
# whole suite (measured 2026-08-30, drain run). The tree-dependent arms
# below skip; collection must survive.
if os.path.isdir(dream_team_root()):
    from awgym.vendor import dream_team
else:
    dream_team = None


@pytest.mark.skipif(
    not os.path.isdir(dream_team_root()),
    reason="vendored dream-team tree not extracted on this host")
def test_root_resolves_to_a_real_tree():
    root = dream_team_root()
    assert os.path.isdir(root), (
        f"vendored dream-team root {root!r} missing — extract the zip first; "
        f"the positive arms of this suite skip until it exists")


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(dream_team_root(), "arc_agi_3")),
    reason="vendored dream-team tree not extracted on this host")
def test_clean_seams_resolve():
    # ARCAGI3Env is a class and rhae exposes the RHAE scorer functions.
    assert callable(dream_team.ARCAGI3Env)
    assert hasattr(dream_team.rhae, "rhae_level_score")
    assert hasattr(dream_team.rhae, "rhae_environment_score")


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(dream_team_root(), "arc_agi_3")),
    reason="vendored dream-team tree not extracted on this host")
def test_env_constructs_and_resets_offline(monkeypatch):
    # The SDK scans recursively for metadata.json under ENVIRONMENTS_DIR;
    # the generated games live at environment_files_generated/<prefix>/<id>/.
    gen_dir = os.path.join(dream_team_root(), "environment_files_generated")
    if not os.path.isdir(gen_dir):
        pytest.skip("generated environments not present in the vendored tree")
    monkeypatch.setenv("ENVIRONMENTS_DIR", gen_dir)
    env = dream_team.ARCAGI3Env(game_id="al7306-000afcc7", operation_mode="offline")
    obs = env.reset()
    # reset() returns a Grid dataclass; its .data is the 64x64 int grid.
    data = obs.data if hasattr(obs, "data") else obs
    assert len(data) == 64 and all(len(row) == 64 for row in data)
    assert all(0 <= v <= 15 for row in data for v in row)


@pytest.mark.skipif(dream_team is None,
                    reason="vendored tree absent — shim not imported")
def test_bogus_root_fails_loud(monkeypatch):
    monkeypatch.setenv("ARC_GYM_DREAMTEAM_ROOT", r"C:\definitely-not-a-tree")
    with pytest.raises(RuntimeError, match="ARC_GYM_DREAMTEAM_ROOT"):
        dream_team.load()
