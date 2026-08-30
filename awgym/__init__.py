"""awgym — an ARC training gym: a game a world model can watch.

Phase 1: ARC-AGI-3 environments (ARCAGI3Env over the official arc-agi SDK,
RHAE scoring — both re-exported from the vendored NVIDIA dream-team tree via
``awgym.vendor.dream_team``) plus a LeWM training loop: play, observe every
transition, burst-train, score on a disjoint eval slice.

Phase 2: the six DreamTeam solver roles reimplemented as awdk role agents
with LeWM as the SIMULATOR (neural predictions replace the agent-written
executable world model).

The vendored dream-team source (Apache-2.0, NOTICE retained) lives OUTSIDE
this package at ``ARC_GYM_DREAMTEAM_ROOT`` (default
``E:\\AitherOS-Data\\arc-agi-3\\dream-team``); this package never carries it.
"""

__version__ = "0.1.0"
