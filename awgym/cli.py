"""awgym CLI — get_parser() + cmd_* per the awdk cli.py convention.

Commands: play (one game, recorded), train (episodes + burst), score
(--strict, the harness scorer), status (health + ledger tail), serve
(the :8199 gym service).
"""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__

_CA_HINT = ("(set ARC_GYM_LEWM_CA to the internal CA bundle on fleet hosts; "
            "reads work with the system store)")


def _client_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--base", default=os.environ.get("ARC_GYM_LEWM_BASE"),
                   help="LeWM base URL (default ARC_GYM_LEWM_BASE)")
    p.add_argument("--ca", default=os.environ.get("ARC_GYM_LEWM_CA"),
                   help="internal CA bundle path for TLS to LeWM")


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="awgym",
        description="ARC training gym — play ARC-AGI-3 games with the LeWM "
                    "world model watching, train it on the transitions, and "
                    "score it on grids it never saw.")
    p.add_argument("--version", action="version", version=f"awgym {__version__}")
    sub = p.add_subparsers(dest="command")

    sp = sub.add_parser("play", help="play one game to the step cap, recorded")
    sp.add_argument("--game", help="game id (default: first by id in the pool)")
    sp.add_argument("--steps", type=int, default=60)
    sp.add_argument("--out", default=None,
                    help="recording path (default: data root recordings/)")
    sp.add_argument("--policy", choices=["random", "curious"], default="random")
    sp.set_defaults(func=cmd_play)

    sp = sub.add_parser("train", help="episodes -> observe -> burst train -> ledger")
    sp.add_argument("--episodes", type=int, default=5)
    sp.add_argument("--max-steps", type=int, default=60)
    sp.add_argument("--burst", type=int, default=None, help="burst steps")
    sp.add_argument("--games", default=None,
                    help="';'-separated game ids (default: rotate the pool)")
    _client_args(sp)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("score", help="latent-gate skill over the eval slice")
    sp.add_argument("--strict", action="store_true")
    _client_args(sp)
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("status", help="LeWM health + recent ledger rows")
    _client_args(sp)
    sp.add_argument("--rows", type=int, default=5)
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("serve", help="run the :8199 gym service")
    sp.add_argument("--port", type=int, default=8199)
    _client_args(sp)
    sp.set_defaults(func=cmd_serve)
    return p


def cmd_play(args: argparse.Namespace) -> int:
    from .envs.games import pick_game
    from .gym.orchestrator import GameSession
    from .gym.recording import Recorder
    from .config import data_root
    _quiet_import_noise()  # the vendored import chain adds loguru sinks — re-silence

    game = pick_game(args.game)
    from .wm.lewm_client import LeWMClient
    from .wm.trainer import Trainer
    if args.policy == "curious":
        t = Trainer(client=LeWMClient(base=args.base, ca=args.ca))
        policy = t._curious_policy
    else:
        policy = Trainer._random_policy
    session = GameSession(game=game, policy=policy, max_steps=args.steps)
    transitions = session.play()
    out = args.out or str(data_root() / "recordings"
                          / f"{game.game_id}.awgym-play.recording.jsonl")
    with Recorder(out) as rec:
        for t in transitions:
            rec.write_transition(t)
    print(f"played {game.game_id}: {len(transitions)} steps -> {out}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .wm.lewm_client import LeWMClient
    from .wm.trainer import Trainer

    games = args.games.split(";") if args.games else None
    t = Trainer(client=LeWMClient(base=args.base, ca=args.ca),
                burst_steps=args.burst)
    rows = t.run_episodes(n=args.episodes, max_steps=args.max_steps,
                          game_ids=games)
    for r in rows:
        print(f"ledger {r['run_id']}: {r['game_id']} {r['steps']} steps")
    print(f"observed+burst done ({len(rows)} episodes)")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from .evals.score_awgym import main as score_main
    return score_main(["--strict"] if args.strict else [])


def cmd_status(args: argparse.Namespace) -> int:
    from .gym.ledger import read_ledger
    from .wm.lewm_client import LeWMClient

    c = LeWMClient(base=args.base, ca=args.ca, timeout=15.0)
    try:
        h = c.health()
        lt = h.get("last_train") or {}
        print(f"WM ok={h.get('ok')} device={h.get('device')} "
              f"steps={h.get('train_steps')} "
              f"recon={lt.get('recon')} z_std={lt.get('z_std')}")
    except Exception as e:
        print(f"WM unreachable: {type(e).__name__}: {str(e)[:100]} {_CA_HINT}")
    rows = read_ledger(limit=args.rows)
    print(f"ledger: {len(rows)} recent rows")
    for r in rows:
        print(f"  {r.get('run_id')}: {r.get('game_id')} "
              f"{r.get('steps')} steps ts={r.get('ts', '')[:19]}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn
    from .serve.app import create_app

    # In-network convention: every service speaks TLS with the shared internal
    # CA (the platform proxy targets the service's in-network name). The
    # container mounts its leaf+key at /certs/{cert,key}.pem; unset here =
    # plain HTTP for a local dev serve.
    ssl_certfile = os.environ.get("ARC_GYM_TLS_CERT") or None
    ssl_keyfile = os.environ.get("ARC_GYM_TLS_KEY") or None
    if bool(ssl_certfile) != bool(ssl_keyfile):
        print("ARC_GYM_TLS_CERT and ARC_GYM_TLS_KEY must be set together",
              file=sys.stderr)
        return 2
    uvicorn.run(create_app(base=args.base, ca=args.ca),
                host="0.0.0.0", port=args.port,
                ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)
    return 0


def _quiet_import_noise() -> None:
    """Silence the vendored import chain's console noise so awgym commands
    are clean WITHOUT a PYTHONIOENCODING/grep wrapper (the wrapper was the
    toil shape the automation gate flagged 2026-08-30). The noise: loguru's
    emoji console sink (crashes on cp1252), torch's pynvml deprecation
    warning, and arc_agi's INFO scorecard lines."""
    import logging
    import warnings

    warnings.filterwarnings("ignore", message=".*pynvml.*")
    logging.getLogger("arc_agi").setLevel(logging.WARNING)
    logging.getLogger("arc_agi_3").setLevel(logging.WARNING)
    try:
        from loguru import logger
        logger.remove()
    except Exception:
        pass  # loguru absent — nothing to silence


def main() -> int:
    _quiet_import_noise()
    args = get_parser().parse_args()
    if not getattr(args, "func", None):
        get_parser().print_help()
        return 0
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
