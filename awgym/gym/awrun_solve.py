"""awrun `solve` kind — a ProblemSpec queued as a job, played by a ProblemSession.

    # host side (aitherd / adk daemon): register the runner, then dispatch as usual
    from awgym.gym.awrun_solve import run_solve
    dispatch_once(store, worker_id="aitherd-1", run_fns={**default_fns, "solve": run_solve})

    # or run a dedicated worker
    python -m awgym.gym.awrun_solve --root <awrun-root> --journal-root <data>/solve

    # submit
    awrun submit --kind solve --spec '{"spec": {...ProblemSpec...}, "baseline": 0.0}'

The queue knows nothing about solving — like `render` and `artpack` it only carries the
kind so any host with awgym installed can claim `solve` items. This module is the RunFn:
item.spec -> ProblemSpec (+ optional baseline / policy / seed / journal_root) ->
ProblemSession.run() -> (exit code, one-line summary). The exit code is JOB HEALTH:
0 the problem was played and judged (kept, reverted or unjudged — read the message),
2 refused / malformed / crashed. Unlike `awgym.gym.problem.main`'s 0/1/2 for humans, a
reverted play is NOT a non-zero here: awrun marks any non-zero job `failed`, and "the
loop judged a candidate worse" is the loop working, not a job to page on.

The journal root defaults to $AWGYM_SOLVE_ROOT or <awrun root>/solve so the
`check_solver_loop_closes.py --root` gate has one place to look. A malformed spec is a
code-2 FAILURE with the reason in the message — never a silently skipped item.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Optional, Tuple

from awpredict.contracts import ProblemSpec

from .problem import ProblemError, ProblemSession, const_policy, random_policy

SOLVE_KIND = "solve"
SOLVE_ROOT_ENV = "AWGYM_SOLVE_ROOT"


def _journal_root(item_spec: dict, run_root: Optional[Path]) -> Path:
    explicit = item_spec.get("journal_root") or os.environ.get(SOLVE_ROOT_ENV)
    if explicit:
        return Path(explicit)
    if run_root is not None:
        return Path(run_root) / "solve"
    return Path.home() / ".aither" / "solve"


def _policy_from(item_spec: dict):
    name = str(item_spec.get("policy") or "random")
    if name == "const":
        return const_policy(item_spec.get("const_action", 0))
    if name == "random":
        seed = item_spec.get("seed")
        return random_policy(int(seed) if seed is not None else None)
    if name == "layered":
        # ARC-middle layers as policies: {"kind": "layered", "layers": [...],
        # "council": {...}, "fallback": {...}} -- see awgym.gym.policies.
        # Imported lazily so a malformed description REFUSES the item (code 2)
        # rather than failing the worker at boot.
        from .policies import load_policy

        desc = item_spec.get("policy_spec")
        if not isinstance(desc, dict):
            raise ProblemError("layered policy needs item_spec.policy_spec = desc")
        try:
            return load_policy(desc)
        except ValueError as exc:
            raise ProblemError(f"layered policy malformed: {exc}") from exc
    raise ProblemError(f"unknown policy {name!r} (random|const|layered)")


def run_solve(item: Any, *, run_root: Optional[Path] = None) -> Tuple[int, str]:
    """The awrun RunFn for kind `solve`. Never raises: a defect is a code-2 message."""
    raw = getattr(item, "spec", None) or {}
    if not isinstance(raw, dict) or not isinstance(raw.get("spec"), dict):
        return 2, "solve item needs spec.spec = ProblemSpec dict"
    try:
        spec = ProblemSpec.from_dict(raw["spec"])
    except (TypeError, ValueError) as exc:
        return 2, f"ProblemSpec.from_dict failed: {exc}"
    baseline = raw.get("baseline")
    try:
        session = ProblemSession(
            spec, _policy_from(raw),
            baseline=float(baseline) if baseline is not None else None,
            journal_root=_journal_root(raw, run_root),
            # item id + a per-attempt suffix: a retried/bumped item must not collide
            # with its own earlier journal (append-only dirs refuse a second init).
            episode_id=f"ep-{getattr(item, 'id', 'awrun')}-{uuid.uuid4().hex[:6]}",
            model_used=str(raw.get("model_used") or ""),
        )
    except ProblemError as exc:
        return 2, f"REFUSED: {exc}"
    except Exception as exc:  # noqa: BLE001 - a broken adapter import is a failed item
        return 2, f"setup failed: {type(exc).__name__}: {exc}"
    try:
        ep = session.run()
    except Exception as exc:  # noqa: BLE001 - the loop itself guards steps; this is the rest
        return 2, f"run crashed: {type(exc).__name__}: {exc}"
    # Policy provenance: the journal must be able to answer "which policy played
    # this episode?" -- a layered run and a random run that both scored zero are
    # different facts. Appended after the run (the journal inits at run start).
    try:
        provenance = {"policy": str(raw.get("policy") or "random")}
        if isinstance(raw.get("policy_spec"), dict):
            provenance["spec"] = {k: v for k, v in raw["policy_spec"].items()
                                  if k != "wm"}
        session.journal.append("policy", provenance)
        # session.run() already closed the journal (state.json = rows/head BEFORE this
        # row). Re-close, or every awrun-played episode reads as "truncated or edited"
        # to SLC001 -- measured 2026-09-06 on 2 of 4 seed episodes.
        session.journal.close(session.episode_id)
    except Exception as exc:  # noqa: BLE001 - an unwritable journal is a defect
        return 2, f"policy provenance failed: {type(exc).__name__}: {exc}"
    out = ep.outcome
    if out is None or out.refusal is not None:
        return 2, ep.summary()
    # JOB health, not verdict: a reverted play is the loop working exactly as designed
    # (awrun marks any non-zero code `failed`, and a failed job is what a human is paged
    # for). The verdict — kept / reverted / unjudged — travels in the message, which is
    # what the kernel's P6 parses. Only a refusal or a crash is a non-zero job.
    return 0, ep.summary()


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m awgym.gym.awrun_solve",
                                 description="claim and run awrun `solve` items")
    ap.add_argument("--root", default=None, help="awrun root (default: awrun's own)")
    ap.add_argument("--journal-root", default=None)
    ap.add_argument("--worker-id", default=f"awgym-solve-{os.getpid()}")
    ap.add_argument("--once", action="store_true", help="claim at most one item, then exit")
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--pidfile", default=None,
                    help="write this process id here (the hidden-task launcher guards on it)")
    args = ap.parse_args(argv)
    try:
        from awrun.dispatcher import _RUN_FNS, dispatch_once, run_forever
        from awrun.store import RunStore
    except ImportError as exc:
        print(f"DEAD: awrun not importable ({exc}); pip install awrun", file=sys.stderr)
        return 2
    if args.journal_root:
        os.environ[SOLVE_ROOT_ENV] = args.journal_root
    store = RunStore(Path(args.root)) if args.root else RunStore()
    root = Path(args.root) if args.root else None
    fns = dict(_RUN_FNS)
    fns[SOLVE_KIND] = lambda item: run_solve(item, run_root=root)
    if args.pidfile:
        Path(args.pidfile).parent.mkdir(parents=True, exist_ok=True)
        Path(args.pidfile).write_text(str(os.getpid()), encoding="utf-8")
    if args.once:
        item = dispatch_once(store, worker_id=args.worker_id, run_fns=fns)
        print(json.dumps({"claimed": bool(item), "id": getattr(item, "id", None),
                          "status": getattr(item, "status", None)}))
        return 0
    run_forever(store, worker_id=args.worker_id, run_fns=fns, poll_interval=args.poll)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
