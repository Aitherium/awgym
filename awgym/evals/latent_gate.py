"""The latent gate — LeWM's prediction skill vs the identity baseline.

Ported from the live lane's wm_latent_gate.py algorithm (the promotion gate
the ARC exhibit uses), NOT imported from the E: fork — awgym treats that tree
as read-only. skill = 1 - surprise_lewm / surprise_identity over a transition
set; positive skill means the world model predicts better than "nothing
changes".

BOTH quantities are latent-space L2 squared — the same units, so the ratio
is meaningful:
  * surprise_lewm  = /surprise(grid, action, next) — ||z_hat - z_tp1||^2
  * surprise_ident = ||encode(grid) - encode(next_grid)||^2 — the no-op
    baseline (exactly the d_id term lewm.py's train_step uses)
A grid-space proxy for identity would compare different scales and make the
gate meaningless (measured in tests 2026-08-30).

The eval set must be the DISJOINT eval slice — on seen pairs, memorization
gives a skill that evaporates on anything new.
"""

from __future__ import annotations

import statistics
from typing import Callable, Optional

from ..gym.orchestrator import Transition
from ..wm.lewm_client import LeWMClient


def identity_surprise(t: Transition, client: LeWMClient) -> Optional[float]:
    """Latent-space L2 squared of the no-op baseline (two /encode calls)."""
    try:
        z_t = client.encode(t.grid)
        z_tp1 = client.encode(t.next_grid)
    except Exception:
        return None
    return sum((a - b) ** 2 for a, b in zip(z_t, z_tp1))


def gate_skill(transitions: list[Transition],
               client: LeWMClient,
               surprise_fn: Optional[Callable[[Transition], Optional[float]]] = None
               ) -> Optional[dict]:
    """skill = 1 - mean(surprise_lewm) / mean(surprise_identity).

    Returns None when no transition produced both surprises (LeWM down /
    empty set / degenerate identity). The dict carries the components so the
    scorer can print evidence, not just the number.
    """
    lewm_vals: list[float] = []
    id_vals: list[float] = []
    for t in transitions:
        s = surprise_fn(t) if surprise_fn else client.surprise(
            t.grid, t.action, t.next_grid, ctx=t.game_id)
        if s is None:
            continue
        ident = identity_surprise(t, client)
        if ident is None:
            continue
        lewm_vals.append(float(s))
        id_vals.append(float(ident))
    if not lewm_vals:
        return None
    m_lewm = statistics.fmean(lewm_vals)
    m_id = statistics.fmean(id_vals)
    if m_id <= 1e-12:
        return None  # degenerate: nothing changed at all in the slice
    return {
        "skill": 1.0 - m_lewm / m_id,
        "n": len(lewm_vals),
        "mean_surprise_lewm": m_lewm,
        "mean_surprise_identity": m_id,
    }
