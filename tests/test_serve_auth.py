"""The gym service's write-path auth — the fail-open class the WM carried
until 2026-08-30 must not be reborn on the gym service.

/ gym/runs and /gym/train are the write surfaces (spawn game sessions, drive
the shared WM); they require the fleet internal key. The auth is read at
module import, so these tests reload serve.app under a fresh env.
"""

import importlib


KEY = "test-internal-key-123"


def _app(monkeypatch, secret: str | None):
    if secret is None:
        monkeypatch.delenv("AITHER_INTERNAL_SECRET", raising=False)
    else:
        monkeypatch.setenv("AITHER_INTERNAL_SECRET", secret)
    mod = importlib.import_module("awgym.serve.app")
    importlib.reload(mod)
    return mod.create_app()


def test_writes_fail_closed_when_unconfigured(monkeypatch):
    from fastapi.testclient import TestClient
    app = _app(monkeypatch, None)
    with TestClient(app) as c:
        r = c.post("/gym/train", json={"steps": 1})
        assert r.status_code == 503  # unconfigured = refused, never accepted


def test_writes_401_on_wrong_key(monkeypatch):
    from fastapi.testclient import TestClient
    app = _app(monkeypatch, KEY)
    with TestClient(app) as c:
        r = c.post("/gym/train", json={"steps": 1},
                   headers={"X-Internal-Key": "wrong"})
        assert r.status_code == 401


def test_writes_pass_with_the_key(monkeypatch):
    from fastapi.testclient import TestClient
    app = _app(monkeypatch, KEY)
    with TestClient(app, raise_server_exceptions=False) as c:
        # the gate passes before the trainer's LeWM call fails on the
        # unreachable base — 500 (not 401/503) proves the gate let it through
        r = c.post("/gym/train", json={"steps": 1},
                   headers={"X-Internal-Key": KEY})
        assert r.status_code == 500


def test_reads_are_open(monkeypatch):
    from fastapi.testclient import TestClient
    app = _app(monkeypatch, KEY)
    with TestClient(app) as c:
        r = c.get("/gym/games")
        assert r.status_code == 200
        assert "games" in r.json()
