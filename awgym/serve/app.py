"""The gym service — the only write surface of the loop (:8199).

A platform gateway proxies verbatim to this service (the idle-job proxy
pattern: the loop never runs inside the gateway's request process). The
service is container-hosted on the shared network; it reaches LeWM
in-network (its service name on the shared network) with the internal CA.
It carries ARC_GYM_LEWM_TOKEN from its env for writes.

Endpoints mirror what the portal panel and MCP tools consume:
  GET  /gym/games          pool listing
  POST /gym/runs           start a play/train run (background task)
  GET  /gym/runs/{id}      run status + recording tail
  GET  /gym/score          latest latent-gate skill (or the ledger tail)
  POST /gym/train          one throttled burst
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException

from ..config import data_root
from ..envs.games import game_pool
from ..gym.ledger import append_run, read_ledger
from ..gym.orchestrator import GameSession
from ..gym.recording import Recorder
from ..wm.lewm_client import LeWMClient
from ..wm.trainer import Trainer

_RUNS: dict[str, dict[str, Any]] = {}

# Write-path auth: the WM's own /observe /train /save are token-enforced
# (provisioned 2026-08-30, fail-open fixed). The gym service must NOT inherit
# that fail-open class on its own write surfaces: /gym/runs (spawns game
# sessions — CPU/LLM cost) and /gym/train (drives the shared WM) require the
# fleet internal key (AITHER_INTERNAL_SECRET, sent as X-Internal-Key /
# X-Internal-Token — the genesis proxy forwards both). Unset = FAIL-CLOSED:
# writes are refused with 503, never silently accepted.
_INTERNAL_KEY = os.environ.get("AITHER_INTERNAL_SECRET") or ""


def _require_internal_key(x_internal_key: Optional[str] = None,
                          x_internal_token: Optional[str] = None) -> None:
    if not _INTERNAL_KEY:
        raise HTTPException(status_code=503, detail="gym write auth unconfigured")
    sent = x_internal_key or x_internal_token
    if not sent or sent != _INTERNAL_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing internal key")


def create_app(base: Optional[str] = None, ca: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="awgym", version="0.1.0")
    client = LeWMClient(base=base, ca=ca)
    trainer = Trainer(client=client)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "service": "awgym"}

    @app.get("/gym/games")
    def games() -> dict:
        pool = game_pool()
        return {"games": [g.as_dict() for g in
                          sorted(pool.values(), key=lambda g: g.game_id)],
                "count": len(pool)}

    @app.post("/gym/runs")
    async def start_run(payload: dict,
                        x_internal_key: Optional[str] = Header(default=None),
                        x_internal_token: Optional[str] = Header(default=None)) -> dict:
        _require_internal_key(x_internal_key, x_internal_token)
        game_id = payload.get("game_id")
        steps = int(payload.get("steps") or 60)
        policy_name = payload.get("policy") or "random"
        from ..envs.games import pick_game
        game = pick_game(game_id)
        run_id = f"run-{uuid.uuid4().hex[:10]}"
        policy = trainer._curious_policy if policy_name == "curious" \
            else trainer._random_policy

        async def _run() -> None:
            session = GameSession(game=game, policy=policy, max_steps=steps)
            transitions = await asyncio.to_thread(session.play)
            out = str(data_root() / "recordings" / f"{run_id}.recording.jsonl")
            with Recorder(out) as rec:
                for t in transitions:
                    rec.write_transition(t)
            observed = trainer._observe_transitions(transitions)
            row = append_run({"run_id": run_id, "game_id": game.game_id,
                              "steps": len(transitions), "score": 0,
                              "observed": observed, "policy": policy_name,
                              "recording": out})
            _RUNS[run_id] = {"status": "done", "row": row}

        _RUNS[run_id] = {"status": "running", "game_id": game.game_id}
        asyncio.create_task(_run())
        return {"run_id": run_id, "status": "running",
                "game_id": game.game_id}

    @app.get("/gym/runs/{run_id}")
    def run_status(run_id: str) -> dict:
        run = _RUNS.get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"no such run {run_id}")
        return run

    @app.get("/gym/score")
    def score() -> dict:
        rows = read_ledger(limit=10)
        return {"recent_runs": rows}

    @app.post("/gym/train")
    def train(payload: dict,
              x_internal_key: Optional[str] = Header(default=None),
              x_internal_token: Optional[str] = Header(default=None)) -> dict:
        _require_internal_key(x_internal_key, x_internal_token)
        steps = int(payload.get("steps") or trainer.burst_steps)
        return {"burst": trainer.client.train(steps=steps).get("result")}

    @app.post("/gym/score-run")
    def score_run(x_internal_key: Optional[str] = Header(default=None),
                  x_internal_token: Optional[str] = Header(default=None)) -> dict:
        """The harness scorer, run where the runtime is (this container holds
        the vendored tree + in-network WM access; the ratchet's host has
        neither — measured 2026-08-30). Mirrors score_awgym --strict:
        returns the metric + `refused` so the harness command can reproduce
        the CLI's stdout/exit contract (exit 1 below the identity floor)."""
        _require_internal_key(x_internal_key, x_internal_token)
        from ..evals.score_awgym import compute_skill
        result = compute_skill(client)
        if result is None:
            raise HTTPException(status_code=503, detail="no metric — slice empty")
        return {
            "skill": result["skill"],
            "n": result["n"],
            "mean_surprise_lewm": result["mean_surprise_lewm"],
            "mean_surprise_identity": result["mean_surprise_identity"],
            "refused": result["skill"] <= 0.0,
        }

    return app
