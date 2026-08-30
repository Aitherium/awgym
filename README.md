# awgym

An ARC training gym — a game a world model can watch.

`pip install awgym` · [The Aither World](https://aitherium.github.io/)

A world model is trained on transcripts of tasks nobody runs. awgym makes the
task the teacher: one environment, six solver roles playing through it, and a
world model that learns from the play.

**Phase 1 — ARC-AGI-3 environments + a LeWM training loop.** The environments
are `ARCAGI3Env` over the official `arc-agi` SDK with RHAE scoring, both
re-exported from the vendored NVIDIA dream-team tree via
`awgym.vendor.dream_team`. The loop plays, observes every transition,
burst-trains, and scores on a disjoint eval slice.

**Phase 2 — the six DreamTeam solver roles as agent roles.** The six roles are
reimplemented as awdk role agents with LeWM as the SIMULATOR — neural
predictions replace the agent-written executable world model.

```bash
awgym --help        # the CLI
awgym serve         # the gym server (FastAPI)
```

## The vendored tree

The dream-team source (Apache-2.0, NOTICE retained) lives OUTSIDE this package
at `ARC_GYM_DREAMTEAM_ROOT` (default `E:\AitherOS-Data\arc-agi-3\dream-team`);
this package never carries it.

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
```

## Licence

Apache-2.0.
