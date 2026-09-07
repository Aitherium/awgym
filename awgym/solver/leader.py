"""The league's leader loop — one awdk-style agent process, role-switched.

Faithful to the vendored dream-team single-process league: the leader
invokes role prompts against a shared workspace and commits to one
DECISION: block per round. The simulator is the LeWM world model (the
``wm_*`` surface through the gym service); NO role writes or executes code.
Each round: build the context -> the leader proposes an action sequence ->
the gym plays it -> the transitions are observed to LeWM -> the workspace
and ledger record the outcome.

The leader's LLM routes through MicroScheduler only (the standing rule —
never bypass it). The gym container carries the internal key + CA from its
env, so the call is a plain authenticated POST.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import httpx

from ..envs.games import pick_game
from ..gym.ledger import append_run
from ..gym.orchestrator import GameSession
from ..wm.lewm_client import LeWMClient
from .decisions_channel import mirror_decision
from .workspace import RunWorkspace

log = logging.getLogger("awgym.solver")

_MS_URL = os.environ.get(
    "ARC_GYM_LEADER_LLM",
    "https://aitheros-microscheduler:8150/v1/chat/completions")
_LEADER_MODEL = os.environ.get("ARC_GYM_LEADER_MODEL", "aither-orchestrator")
_INTERNAL_KEY = os.environ.get("AITHER_INTERNAL_SECRET") or os.environ.get(
    "AITHER_INTERNAL_SECRET_PREVIOUS") or ""


def _ca() -> Optional[str]:
    for cand in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        p = os.environ.get(cand)
        if p and os.path.exists(p):
            return p
    for p in ("/etc/tls/ca-bundle.pem", "/certs/ca-chain.pem"):
        if os.path.exists(p):
            return p
    return None


def _leader_call(system: str, context: str, timeout: float = 300.0) -> str:
    """One leader LLM call through MicroScheduler (fail-closed)."""
    if not _INTERNAL_KEY:
        raise RuntimeError("AITHER_INTERNAL_SECRET unset — leader LLM refused")
    payload = {
        "model": _LEADER_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": context},
        ],
        "temperature": 0.4,
        "max_tokens": 1024,
    }
    r = httpx.post(_MS_URL, json=payload, headers={"X-Internal-Key": _INTERNAL_KEY},
                   verify=_ca(), timeout=timeout)
    r.raise_for_status()
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"leader LLM reply shape unexpected: {exc}") from exc


class _SequencePolicy:
    """Plays the leader's chosen action sequence, then falls back to random."""

    def __init__(self, actions: list[int]) -> None:
        self._actions = list(actions)
        self._i = 0

    def __call__(self, grid: list, session: GameSession) -> int:
        if self._i < len(self._actions):
            action = self._actions[self._i]
            self._i += 1
            return action
        import random
        return random.randint(0, 7)


def _parse_actions(decision: dict[str, Any]) -> list[int]:
    """Parse the DECISION's action_sequence field into ints 0-7."""
    raw = decision.get("action_sequence", "")
    if isinstance(raw, list):
        return [int(a) for a in raw if str(a).isdigit()]
    import re
    return [int(a) for a in re.findall(r"\d+", str(raw)) if int(a) <= 7]


class _StubWM:
    """The A/B control: identity predictions, zero surprise, no training.

    The league loop against the stub produces the same ledger rows with a
    no-knowledge baseline, so the neural simulator's contribution is the
    difference between the two runs (the plan's accept (d))."""

    def __init__(self, base: Optional[str] = None,
                 ca: Optional[str] = None, token: Optional[str] = None,
                 timeout: float = 30.0) -> None:
        pass

    def health(self) -> dict:
        return {"ok": True, "device": "stub", "train_steps": 0}

    def observe(self, grid: list, action: int, next_grid: list) -> dict:
        return {"ok": True, "backend": "stub"}

    def surprise(self, grid: list, action: int,
                 next_grid: list) -> Optional[float]:
        return 0.0


def _make_wm(simulator: str, client: Optional[LeWMClient]) -> Any:
    """The simulator backend (the A/B lever): lewm -> the real client."""
    if simulator == "stub":
        return _StubWM()
    return client or LeWMClient()


def run_solve(game_id: str, rounds: int = 5,
              max_steps: int = 60,
              simulator: str = "lewm",
              client: Optional[LeWMClient] = None) -> dict[str, Any]:
    """Run one league solve on a game: leader rounds -> played steps -> WM.

    Returns the run outcome for the ledger: rounds, steps, surprise
    summary, the leader's final hypothesis, and the workspace inventory.
    ``simulator`` selects the backend (lewm | stub) — the A/B lever.
    """
    game = pick_game(game_id)
    game_id = game.game_id  # the resolved id — pick_game(None) chooses the first
    wm = _make_wm(simulator, client)
    ws = RunWorkspace(game_id=game_id, run_id="solve")
    sys_prompt = _LEADER_SYSTEM
    outcome: dict[str, Any] = {
        "game_id": game_id, "run_id": ws.run_id, "simulator": simulator,
        "rounds": 0, "steps": 0, "hypothesis": None, "skill": None,
    }
    total_steps = 0
    all_surprises: list[float] = []

    for round_no in range(1, rounds + 1):
        context = _build_context(game, ws, round_no, total_steps, outcome)
        try:
            reply = _leader_call(sys_prompt, context)
        except Exception as exc:  # noqa: BLE001 — a failed round is recorded
            log.warning("round %d leader call failed: %s", round_no, exc)
            outcome["error"] = f"round {round_no}: {type(exc).__name__}: {exc}"
            break
        decision = RunWorkspace.parse_decision(reply)
        if not decision:
            log.warning("round %d produced no DECISION block", round_no)
            ws.append_jsonl("rounds_raw.jsonl",
                            {"round": round_no, "reply": reply[:2000]})
            continue
        ws.log_round(round_no, decision)
        mirror_decision(round_no, game_id, decision)

        actions = _parse_actions(decision)
        if not actions:
            actions = []  # random fallback — never -1 (the env rejects it)
        session = GameSession(game=game,
                              policy=_SequencePolicy(actions),
                              max_steps=max_steps)
        transitions = session.play()
        total_steps += len(transitions)
        outcome["steps"] = total_steps
        outcome["hypothesis"] = decision.get("hypothesis")
        outcome["goal"] = decision.get("goal")

        # Observe the played transitions to LeWM (the league's training data).
        for t in transitions[-8:]:  # the round's tail — throttled by design
            s: Optional[float] = None
            try:
                wm.observe(t.grid, t.action, t.next_grid)
                s = wm.surprise(t.grid, t.action, t.next_grid)
                if s is not None:
                    all_surprises.append(float(s))
            except Exception as exc:  # noqa: BLE001 — observe must not kill the loop
                log.warning("observe failed at step %s: %s", t.step, exc)
            ws.log_observation(t.step, t.grid, t.action, t.next_grid, s)

        ws.append_jsonl("rounds.jsonl", {
            "round": round_no, "actions": actions,
            "transitions": len(transitions),
            "surprise": (all_surprises[-1] if all_surprises else None),
        })
        outcome["rounds"] = round_no
        log.info("round %d: %d transitions, surprise %s",
                 round_no, len(transitions),
                 all_surprises[-1] if all_surprises else "n/a")

    if all_surprises:
        outcome["mean_surprise"] = sum(all_surprises) / len(all_surprises)
    # The league's solution record — json, never .py (the run-hygiene gate
    # asserts zero agent-written .py in a run dir by construction).
    ws.root.joinpath("solution.json").write_text(
        json.dumps({
            "game_id": game_id, "simulator": simulator,
            "steps": total_steps, "rounds": outcome["rounds"],
            "hypothesis": outcome.get("hypothesis"),
            "goal": outcome.get("goal"),
            "mean_surprise": outcome.get("mean_surprise"),
            "files": sorted(p.name for p in ws.root.iterdir()),
        }, indent=1), encoding="utf-8")
    outcome["workspace"] = ws.summary()
    append_run({"run_id": ws.run_id, "game_id": game_id, "kind": "solve",
                "steps": total_steps, "rounds": outcome["rounds"],
                "simulator": simulator, "score": 0,
                "mean_surprise": outcome.get("mean_surprise"),
                "hypothesis": outcome.get("hypothesis"),
                "run_dir": str(ws.root)})
    return outcome


_LEADER_SYSTEM = """You are the Team Leader of an ARC-AGI-3 solver league.
The simulator is the LeWM neural world model; the league NEVER writes or
executes code. Each round you commit to ONE execution choice in this exact
format:

DECISION:
  game_id: <id>
  goal: <[GOAL]|[DISCOVER]|[VALIDATE]|[REFUTE] one line>
  hypothesis: <the active hypothesis>
  action_sequence: [<integers 0-7>]
  steps: <count>
  checkpoints: <continue/stop conditions>
  teammate_proposals: <+endorse / -reject each>
  critical_flags: <open [CRITICAL] flags + how this round addresses them>
  confidence: <0.0-1.0>

Rules: verify strategy with the tools you have (surprise numbers, grid
changes); resolve every open [CRITICAL] flag; prefer actions that add a
distinction to the active hypothesis; keep sequences within the step
budget. End each reply with exactly one DECISION: block."""


def _build_context(game: Any, ws: RunWorkspace, round_no: int,
                   steps: int, outcome: dict[str, Any]) -> str:
    """The compact round context — summaries, never raw 64x64 grids."""
    lines = [
        f"# League round {round_no}",
        f"game: {game.game_id} (tags: {getattr(game, 'tags', '') or ''}, "
        f"baseline actions: {getattr(game, 'baseline_actions', '?')})",
        f"steps so far: {steps}",
        f"last round: {outcome.get('rounds', 0)} rounds, "
        f"hypothesis: {outcome.get('hypothesis') or 'none'}",
    ]
    for name in ("critique.md", "hypotheses.jsonl", "policies.jsonl",
                 "predictions.jsonl"):
        p = ws.root / name
        if p.is_file():
            tail = p.read_text(encoding="utf-8").strip().splitlines()[-3:]
            lines.append(f"--- {name} (tail) ---")
            lines.extend(tail)
    return "\n".join(lines)
