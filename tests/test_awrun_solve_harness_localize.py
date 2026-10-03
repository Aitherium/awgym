"""awrun_solve: a `harness` problem's file is made readable where THIS worker runs.

The solve problem catalog names fleet paths (`/data/solve/...`); a worker outside the
fleet cannot read them, and the file is only ever written by an exporter on first use.
"""
from __future__ import annotations

import json
import textwrap

from awgym.gym import awrun_solve as rs

FLEET = "/data/solve/harnesses/synthetic.json"


def _spec():
    return {"domain": "harness", "adapter_kwargs": {"harness": FLEET},
            "scorer_kwargs": {"harness": FLEET}}


def _exporter(tmp_path, monkeypatch, body="Path(path).write_text(json.dumps({'name': name}))"):
    name = "fake_exporter_" + tmp_path.name.replace("-", "_")
    mod = tmp_path / f"{name}.py"
    mod.write_text(textwrap.dedent(f"""
        import json
        from pathlib import Path
        CALLS = []
        def export(name, path):
            CALLS.append(name)
            {body}
    """), encoding="utf-8")
    monkeypatch.setenv(rs.HARNESS_EXPORTER_ENV, f"{name}:export")
    monkeypatch.setenv(rs.HARNESS_EXPORTER_PATH_ENV, str(tmp_path))


def test_fleet_path_maps_under_the_solve_root_and_is_exported(tmp_path, monkeypatch):
    _exporter(tmp_path, monkeypatch)
    root = tmp_path / "solve"
    spec = _spec()
    assert rs._localize_harness(spec, root) is None
    local = root / "harnesses" / "synthetic.json"
    assert json.loads(local.read_text())["name"] == "synthetic"
    assert spec["adapter_kwargs"]["harness"] == str(local)
    assert spec["scorer_kwargs"]["harness"] == str(local)


def test_no_exporter_is_a_refusal_naming_the_path(tmp_path, monkeypatch):
    monkeypatch.delenv(rs.HARNESS_EXPORTER_ENV, raising=False)
    why = rs._localize_harness(_spec(), tmp_path / "solve")
    assert why and FLEET in why and rs.HARNESS_EXPORTER_ENV in why


def test_an_exporter_that_writes_nothing_is_a_refusal(tmp_path, monkeypatch):
    _exporter(tmp_path, monkeypatch, body="pass")
    why = rs._localize_harness(_spec(), tmp_path / "solve")
    assert why and "not written" in why


def test_non_harness_problems_are_untouched(tmp_path):
    spec = {"domain": "arc", "adapter_kwargs": {"game_id": "x"}}
    assert rs._localize_harness(spec, tmp_path) is None
    assert spec == {"domain": "arc", "adapter_kwargs": {"game_id": "x"}}
