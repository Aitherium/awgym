"""The harness scorer, run where the runtime is — the gym SERVICE container.

The ratchet's host has neither the vendored tree nor in-network WM access
(measured 2026-08-30 — a host-side score_awgym dies DEAD against its own
loopback), so the awgym harness eval_command drives THIS module, which POSTs
/gym/score-run on the gym service and reproduces score_awgym --strict's
stdout/exit contract for the harness regex:

    METRIC awgym_skill=<float>
    METRIC awgym_skill_floor=0.0
    exit 1 when the skill is at/below the identity floor (trial refused)
    exit 2 when the gym service is unreachable (DEAD, never a refusal)

The credential is the fleet internal key from the driving container's env;
the CA is the first existing bundle in the standard fallback chain (the
fleet template mounts the combined bundle for every service).
"""

from __future__ import annotations

import os
import sys

import httpx

_GYM_URL = os.environ.get("ARC_GYM_SCORE_URL",
                          "http://127.0.0.1:8199/gym/score-run")
_CA_CANDIDATES = (
    "SSL_CERT_FILE",
    "/etc/tls/ca-bundle.pem",
    "/certs/ca-chain.pem",
)


def _ca() -> str | None:
    for cand in _CA_CANDIDATES:
        path = os.environ.get(cand) if cand == "SSL_CERT_FILE" else cand
        if path and os.path.exists(path):
            return path
    return None


def main() -> int:
    key = os.environ.get("AITHER_INTERNAL_SECRET") or os.environ.get(
        "AITHER_INTERNAL_SECRET_PREVIOUS") or ""
    if not key:
        print("DEAD: no AITHER_INTERNAL_SECRET in the driving container's env")
        return 2
    try:
        r = httpx.post(_GYM_URL,
                       headers={"X-Internal-Token": key},
                       verify=_ca(), timeout=180)
        r.raise_for_status()
    except Exception as exc:  # noqa: BLE001 — any transport failure is DEAD
        print(f"DEAD: gym service unreachable — {type(exc).__name__}: "
              f"{str(exc)[:120]}")
        return 2
    d = r.json()
    print(f"METRIC awgym_skill={d['skill']:.6f}")
    print("METRIC awgym_skill_floor=0.0")
    print(f"# n={d['n']} refused={d['refused']}")
    if d["refused"]:
        print("awgym_skill at/below the identity floor — trial refused")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
