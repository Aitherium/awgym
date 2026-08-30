"""The awgym harness scorer — the ratchet's measuring stick.

Contract (harnesses.py): prints `METRIC <name>=<float>` lines, exits non-zero
when the metric could not be computed (--strict), and is a SEPARATE file from
the mutable trial surface (AVO006 — the scorer cannot edit the thing it
scores). Exit codes: 0 = metric printed; 2 = DEAD (LeWM unreachable / eval
slice unplayable / no metric) — never 0 on silence.

The metric is the latent-gate skill over the DISJOINT eval slice: LeWM's
prediction surprise vs the identity baseline on transitions from games the
training loop never observes. skill > 0 = beats "nothing changes".
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from ..envs.games import game_pool
from ..gym.orchestrator import GameSession
from ..wm.eval_slice import resolve_eval_games
from ..wm.lewm_client import LeWMClient
from .latent_gate import gate_skill

EVAL_EPISODES_ENV = "ARC_GYM_EVAL_EPISODES"
EVAL_STEPS_ENV = "ARC_GYM_EVAL_STEPS"


def _const_policy(action: int):
    def policy(grid, session):  # noqa: ARG001
        return action
    return policy


def compute_skill(client: LeWMClient) -> Optional[dict]:
    pool = game_pool()
    eval_ids = resolve_eval_games(pool)
    episodes = int(os.environ.get(EVAL_EPISODES_ENV) or "2")
    steps = int(os.environ.get(EVAL_STEPS_ENV) or "10")
    transitions = []
    for gid in eval_ids:
        for _ in range(episodes):
            session = GameSession(game=pool[gid],
                                  policy=_const_policy(3),
                                  max_steps=steps)
            transitions.extend(session.play())
    return gate_skill(transitions, client)


def main(argv: list[str] | None = None) -> int:
    """argv param: the CLI's `awgym score` subparser owns its own flags, so
    cmd_score re-invokes this entry point with a rebuilt argv (measured
    2026-08-30: passing the argparse.Namespace through blew up with
    "main() takes 0 positional arguments")."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero when no metric could be computed")
    ap.add_argument("--base", default=os.environ.get("ARC_GYM_LEWM_BASE"))
    ap.add_argument("--ca", default=os.environ.get("ARC_GYM_LEWM_CA"))
    args = ap.parse_args(argv)

    client = LeWMClient(base=args.base, ca=args.ca, timeout=30.0)
    try:
        h = client.health()
        if not h.get("ok"):
            raise RuntimeError(f"WM /health not ok: {h}")
    except Exception as e:
        print(f"DEAD: LeWM unreachable — {type(e).__name__}: {str(e)[:120]}")
        return 2

    try:
        result = compute_skill(client)
    except Exception as e:
        print(f"DEAD: eval failed — {type(e).__name__}: {str(e)[:120]}")
        return 2
    if result is None:
        print("DEAD: no transitions produced both surprises (slice empty?)")
        return 2

    skill = result["skill"]
    print(f"METRIC awgym_skill={skill:.6f}")
    print("METRIC awgym_skill_floor=0.0")
    print(f"# n={result['n']} mean_surprise_lewm="
          f"{result['mean_surprise_lewm']:.4f} "
          f"mean_surprise_identity={result['mean_surprise_identity']:.4f}")
    # --strict = the harness contract: a measured metric with exit 1 means
    # "the trial was refused" (below the identity floor), exit 0 = accepted.
    # The metric line is ALWAYS printed so the ratchet can record the number
    # either way; DEAD (no metric) stays exit 2.
    if args.strict and skill <= 0.0:
        print("awgym_skill below the identity floor (0.0) — trial refused")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
