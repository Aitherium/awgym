"""DECISION-block mirror to the adk decisions daemon — human steer.

The leader's DECISION blocks are the workspace-file truth (the league's
coordination protocol). This module mirrors each round's DECISION to the
decision-card plane via ``adk.decisions.agent_tools.raise_card`` so a human
can steer the running league (the plan's aw* composition: "adk decisions
daemon for human steer"). Best-effort on purpose: ``raise_card`` reports the
daemon unreachable as skipped, never as an outage — the workspace files
carry the truth either way.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

log = logging.getLogger("awgym.solver.decisions")


def mirror_decision(round_no: int, game_id: str,
                    decision: dict[str, Any],
                    agent: str = "awgym-league") -> Optional[Any]:
    """Raise a decision card for one round's DECISION block."""
    try:
        from adk.decisions.agent_tools import raise_card
    except ImportError:
        log.warning("adk not importable — DECISION mirror skipped (no daemon)")
        return None
    sequence = decision.get("action_sequence", "")
    summary = (f"awgym round {round_no} on {game_id}: "
               f"{decision.get('goal', 'no goal')} — "
               f"actions [{sequence}] confidence "
               f"{decision.get('confidence', '?')}")
    try:
        return raise_card(
            f"awgym round {round_no} — {game_id}",
            summary=summary,
            detail=(f"hypothesis: {decision.get('hypothesis', '?')}\n"
                    f"checkpoints: {decision.get('checkpoints', '?')}\n"
                    f"critical_flags: {decision.get('critical_flags', 'none')}"),
            options=["continue", "pivot", "abort"],
            recommend="continue",
            default="continue",
            kind="decision",
            urgency="normal",
            agent=agent,
        )
    except Exception as exc:  # noqa: BLE001 — the mirror must never break the loop
        log.warning("DECISION mirror failed: %s", exc)
        return None
