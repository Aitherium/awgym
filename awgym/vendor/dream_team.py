"""Import shim for the vendored NVIDIA dream-team ARC-AGI-3 solver (Apache-2.0).

awgym never vendors dream-team's code into this package. It imports the CLEAN
seams — ``ARCAGI3Env`` (the RL-style env over the official arc-agi SDK) and the
``rhae`` RHAE scorer — from the tree at ``ARC_GYM_DREAMTEAM_ROOT`` at import
time. The monorepo stays free of the vendored tree (license files, beam/,
tests, run dirs), and every awgym module gets ONE import surface.

Only the two library seams are exposed. Nothing here touches run dirs, the
viewer (:8420), or ``game_creator`` (which executes generated python upstream
— awgym deliberately never uses it; see the awgym plan, security section).

Failure is LOUD on purpose: a missing root means every env-using command must
die with the fix spelled out, not degrade into "no games available".
"""

from __future__ import annotations

import os
import sys
from typing import Any, Tuple

from ..config import dream_team_root

__all__ = ["ARCAGI3Env", "rhae", "dream_team_root", "load"]


def _check_root(root: str) -> None:
    if not os.path.isdir(os.path.join(root, "arc_agi_3")):
        raise RuntimeError(
            f"dream-team vendored source not found at {root!r}. "
            f"Set ARC_GYM_DREAMTEAM_ROOT to the extracted NVIDIA/dream-team "
            f"tree (Apache-2.0). Extract it with: "
            f"unzip dream-team-main.zip -d <parent> && mv <parent>/dream-team-main "
            f"<parent>/dream-team")


def _install_langgraph_stub() -> None:
    """Provide a minimal ``add_messages`` without the langgraph stack.

    Why: dream-team's ``containers.py`` imports ``langgraph.graph.message``
    ONLY as the ``StepRecord.messages`` field reducer (``Annotated[list,
    add_messages]``). The real langgraph chain (langgraph -> langchain_core
    -> uuid_utils) dies on this host: uuid_utils' native DLL is blocked by
    Windows Application Control (same class as the @swc/core SAC block,
    measured 2026-08-27 — see memory sac-blocks-swc-native-bindings). A
    reducer that appends preserves the append-only history semantics the env
    uses; the solver's LLM-message machinery (which needs real langgraph
    semantics) is deliberately NOT part of the gym's seams.
    """
    if "langgraph" in sys.modules:
        return
    import types

    mg = types.ModuleType("langgraph")
    gg = types.ModuleType("langgraph.graph")
    gm = types.ModuleType("langgraph.graph.message")

    def add_messages(left: Any, right: Any) -> list:
        merged = list(left or [])
        for item in right or []:
            merged.append(item)
        return merged

    gm.add_messages = add_messages
    gg.message = gm
    mg.graph = gg
    sys.modules["langgraph"] = mg
    sys.modules["langgraph.graph"] = gg
    sys.modules["langgraph.graph.message"] = gm


def load() -> Tuple[Any, Any]:
    """Import and return (ARCAGI3Env, rhae) from the vendored tree."""
    root = dream_team_root()
    _check_root(root)
    sys.path.insert(0, root)
    _install_langgraph_stub()
    try:
        from arc_agi_3.environment import ARCAGI3Env  # type: ignore[import-not-found]
        from arc_agi_3 import rhae  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover — any import failure is fatal
        raise RuntimeError(f"dream-team import failed from {root!r}: {exc}") from exc
    return ARCAGI3Env, rhae


# Eager at module import: `from awgym.vendor.dream_team import ARCAGI3Env` is
# the documented surface, and a gym command without the envs is a misconfigured
# command. The negative case is pinned by tests/test_vendor_shim.py.
ARCAGI3Env, rhae = load()
