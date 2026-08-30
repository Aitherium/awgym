"""awgym configuration — env-driven, mirroring the ARC_WM_* conventions.

Every path is overridable via env so a host with a different layout never
needs a code edit (the writable-state-placement and storage-topology rules).
"""

import os
from pathlib import Path

# The vendored NVIDIA dream-team tree (Apache-2.0). The shim fails LOUDLY
# when this is missing — a silent fallback would read as "gym works" while
# every env import dies.
DREAMTEAM_ROOT_ENV = "ARC_GYM_DREAMTEAM_ROOT"
_DEFAULT_DREAMTEAM_ROOT = r"E:\AitherOS-Data\arc-agi-3\dream-team"

# The LeWM world-model service (HTTPS in-network; the gym service runs on the
# same host, so the loopback URL + internal CA apply — see lewm_client).
LEWM_BASE_ENV = "ARC_GYM_LEWM_BASE"
_DEFAULT_LEWM_BASE = "https://127.0.0.1:8197"

# Heavy outputs (recordings, runs, ledgers, checkpoints) stay off the C: repo.
DATA_ROOT_ENV = "ARC_GYM_DATA_ROOT"
_DEFAULT_DATA_ROOT = r"E:\AitherOS-Data\arc-agi-3\awgym"

# X-WM-Token for the LeWM WRITE paths (/observe /train /save). Reads work
# without it; writes 401 when the WM enforces auth. Never default a secret.
LEWM_TOKEN_ENV = "ARC_GYM_LEWM_TOKEN"

# The shared internal CA bundle for TLS to LeWM. Deployment-specific --
# never default an internal path (the moat refuses one); empty means the
# caller either set the env var or accepts the system trust store.
LEWM_CA_ENV = "ARC_GYM_LEWM_CA"


def dream_team_root() -> str:
    root = os.environ.get(DREAMTEAM_ROOT_ENV) or _DEFAULT_DREAMTEAM_ROOT
    return root


def lewm_base() -> str:
    return os.environ.get(LEWM_BASE_ENV) or _DEFAULT_LEWM_BASE


def data_root() -> Path:
    root = Path(os.environ.get(DATA_ROOT_ENV) or _DEFAULT_DATA_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    return root


def lewm_token() -> str:
    return os.environ.get(LEWM_TOKEN_ENV) or ""


def internal_ca() -> str:
    """The shared internal CA the LeWM service presents.

    Deployment-specific: ARC_GYM_LEWM_CA overrides it. The default is empty
    so a stranger's install never inherits an internal path -- TLS against a
    custom CA requires the env var (see lewm_client).
    """
    return os.environ.get(LEWM_CA_ENV) or ""
