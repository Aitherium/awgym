"""awgym CLI — get_parser() + cmd_* per the awdk cli.py convention.

Subcommands are registered as their Phase-1 modules land: play/train/score/
status/serve. Until then the parser answers --version and usage, so a broken
install is loud and a working one is quiet.
"""

from __future__ import annotations

import argparse

from . import __version__


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="awgym",
        description="ARC training gym — play ARC-AGI-3 games with the LeWM "
                    "world model watching, train it on the transitions, and "
                    "score it on grids it never saw.")
    p.add_argument("--version", action="version", version=f"awgym {__version__}")
    # Phase 1 registers: play, train, score, status, serve.
    p.add_subparsers(dest="command")
    return p


def main() -> int:
    args = get_parser().parse_args()
    if args.command is None:
        get_parser().print_help()
        return 0
    handler = globals().get(f"cmd_{args.command}")
    if handler is None:
        print(f"awgym: command '{args.command}' is registered in Phase 1 "
              f"(not built yet) — see the awgym plan")
        return 2
    return int(handler(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
