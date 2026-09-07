"""End-of-run findings card on the relay — one card per solve.

The plan's coordination protocol: "awrelay carries only end-of-run findings
cards." The relay is the human-readable channel (a human can read it, unlike
the transcript); the run workspace + ledger carry the data.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

log = logging.getLogger("awgym.solver.findings")

_CHANNEL = "#awgym"


def post_findings(game_id: str, run_id: str,
                  outcome: dict[str, Any]) -> Optional[Any]:
    """Post one findings card for a finished solve run.

    ``outcome`` carries the measured numbers: steps, mean surprise, the
    latent-gate skill, and the leader's final hypothesis. Any transport
    failure (relay down, credentials missing) is reported as skipped — the
    ledger row remains the record of truth.
    """
    try:
        from awrelay.client import Envelope, RelayClient
    except ImportError:
        log.warning("awrelay not importable — findings card skipped")
        return None
    try:
        client = RelayClient()
        envelope = Envelope(
            sender="awgym-league",
            kind="finding",
            body=(
                f"**awgym solve {game_id} ({run_id})**\n"
                f"- steps: {outcome.get('steps', '?')}\n"
                f"- mean surprise: {outcome.get('mean_surprise', '?'):.2f}\n"
                f"- latent-gate skill: {outcome.get('skill', '?'):.4f}\n"
                f"- final hypothesis: {outcome.get('hypothesis', 'none')}\n"
                f"- backend: {outcome.get('sim_backend', 'lewm')}"
            ),
        )
        return client.send(_CHANNEL, envelope)
    except Exception as exc:  # noqa: BLE001 — never break the loop on a card
        log.warning("findings card skipped: %s", exc)
        return None
