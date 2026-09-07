"""watch — the public research page must not be able to lie about a record.

Seeds journals directly with EpisodeJournal, then asserts the presenter and the
request handler keep every honesty rule: a refusal is never a score, a tampered or
truncated journal never shows "chain verified", an unreadable journal is shown as
unreadable rather than dropped, journal bytes are HTML-escaped, and the ?d= path
cannot escape the root or leak a non-public episode.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from awgym.gym.journal import EpisodeJournal  # noqa: E402
from awgym.gym.watch import (  # noqa: E402
    handle,
    iter_episodes,
    outcome_of,
    read_episode,
    render_episode,
    render_index,
    verdict_word,
)

SPEC_ARC = {"problem_id": "arc:al7306-000afcc7", "domain": "arc",
            "adapter_ref": "awgym.envs.arc_adapter:ArcEnvAdapter"}
SPEC_INTERNAL = {"problem_id": "fleet:defect-17", "domain": "fleet"}


def _seed(root: Path, slug: str, ep_id: str, rows: list[tuple[str, dict]]) -> Path:
    """Create <root>/<slug>/<ep_id>/ with journal rows; returns the episode dir."""
    d = root / slug / ep_id
    j = EpisodeJournal(d)
    j.init({**SPEC_ARC, "problem_id": f"{slug}:{ep_id}"}, ep_id)
    for kind, payload in rows:
        j.append(kind, payload)
    j.close(ep_id)
    return d


def _fixture(root: Path) -> dict[str, Path]:
    kept = _seed(root, "arc", "ep-kept", [
        ("transition", {"step": 0, "action": 1, "reward": 1.0}),
        ("attempt", {"steps": 2, "opening_actions": [1]}),
        ("fact", {"relation": "direction", "fact": "up raises", "confidence": 0.9}),
        ("outcome", {"score": {"metric": "levels", "value": 3.0}, "baseline": 1.0,
                     "kept": True, "refusal": None})])
    reverted = _seed(root, "arc", "ep-reverted", [
        ("transition", {"step": 0, "action": 2, "reward": 0.0}),
        ("outcome", {"score": {"metric": "levels", "value": 0.0}, "baseline": 0.0,
                     "kept": False, "refusal": None})])
    refused = _seed(root, "arc", "ep-refused", [
        ("outcome", {"score": None, "baseline": 0.0, "kept": None,
                     "refusal": {"reason": "scorer defect: None"}})])
    empty = _seed(root, "arc", "ep-empty", [])
    # tampered: edit one payload value after close
    tampered = _seed(root, "arc", "ep-tampered", [
        ("transition", {"step": 0, "action": 1, "reward": 1.0}),
        ("outcome", {"score": {"metric": "levels", "value": 9.0}, "baseline": 1.0,
                     "kept": True, "refusal": None})])
    _tamper(tampered, '"action":1', '"action":7')
    # truncated: drop the final (outcome) line so state.json disagrees with the chain
    truncated = _seed(root, "arc", "ep-truncated", [
        ("transition", {"step": 0, "action": 1, "reward": 1.0}),
        ("outcome", {"score": {"metric": "levels", "value": 2.0}, "baseline": 1.0,
                     "kept": True, "refusal": None})])
    lines = (truncated / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    (truncated / "journal.jsonl").write_text(
        "\n".join(lines[:-1]) + "\n", encoding="utf-8")
    # unreadable: a non-JSON line appended after close
    unreadable = _seed(root, "arc", "ep-unreadable", [
        ("outcome", {"score": {"metric": "levels", "value": 1.0}, "baseline": 1.0,
                     "kept": True, "refusal": None})])
    with (unreadable / "journal.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    # xss: fact text that must never reach the page unescaped
    xss = _seed(root, "arc", "ep-xss", [
        ("fact", {"relation": "r", "fact": "<script>alert(1)</script>",
                  "confidence": 0.5}),
        ("outcome", {"score": {"metric": "m", "value": 1.0}, "baseline": 1.0,
                     "kept": True, "refusal": None})])
    internal = _seed(root, "fleet", "ep-internal", [
        ("outcome", {"score": {"metric": "m", "value": 1.0}, "baseline": 0.0,
                     "kept": True, "refusal": None})])
    return {"kept": kept, "reverted": reverted, "refused": refused, "empty": empty,
            "tampered": tampered, "truncated": truncated, "unreadable": unreadable,
            "xss": xss, "internal": internal}


def _tamper(d: Path, old: str, new: str) -> None:
    """Edit one payload value in the CANONICAL (compact, no-space) journal text."""
    p = d / "journal.jsonl"
    text = p.read_text(encoding="utf-8")
    assert old in text, f"tamper target {old!r} not found in {d.name}"
    p.write_text(text.replace(old, new), encoding="utf-8")


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    _fixture(tmp_path)
    return tmp_path


def _episode_by_slug(eps, slug: str):
    for d in eps:
        if d.name == slug:
            return d
    return None


# ---------------------------------------------------------------- the reader
def test_iter_episodes_finds_only_real_dirs_newest_first(root: Path):
    eps = iter_episodes(root)
    names = [d.name for d in eps]
    assert names  # not silently empty
    for wanted in ("ep-kept", "ep-reverted", "ep-refused", "ep-empty",
                   "ep-tampered", "ep-truncated", "ep-unreadable", "ep-xss"):
        assert wanted in names  # unreadable and tampered are NOT dropped
    assert "ep-internal" in names  # all shown without a public filter


def test_public_prefixes_hide_other_problems(root: Path):
    eps = iter_episodes(root, ("arc:",))
    assert _episode_by_slug(eps, "ep-kept") is not None
    assert _episode_by_slug(eps, "ep-internal") is None


def test_read_episode_reports_tampered_and_truncated_honestly(root: Path):
    tampered = read_episode(root / "arc" / "ep-tampered")
    assert tampered["verify"]["ok"] is False and tampered["tampered"] is True
    truncated = read_episode(root / "arc" / "ep-truncated")
    assert truncated["verify"]["ok"] is True and truncated["truncated"] is True
    assert truncated["tampered"] is True  # a cut tail is a broken record
    assert truncated["counts"].get("outcome", 0) == 0  # the cut line was the outcome


def test_read_episode_marks_unreadable_never_raises(root: Path):
    ep = read_episode(root / "arc" / "ep-unreadable")
    assert isinstance(ep["unreadable"], str) and ep["unreadable"]
    assert ep["tampered"] is True and ep["verify"]["ok"] is False


def test_read_episode_tolerates_an_empty_journal(root: Path):
    ep = read_episode(root / "arc" / "ep-empty")
    assert ep["unreadable"] is None and ep["verify"]["ok"] is True
    assert ep["verify"]["rows"] == 0 and ep["counts"] == {}


def test_verdict_words_and_outcome_lookup(root: Path):
    assert verdict_word(outcome_of(root / "arc" / "ep-kept")) == "KEPT"
    assert verdict_word(outcome_of(root / "arc" / "ep-reverted")) == "REVERTED"
    refused = read_episode(root / "arc" / "ep-refused")
    assert verdict_word(outcome_of(root / "arc" / "ep-refused")) == "REFUSED"
    assert refused["counts"].get("fact", 0) == 0
    assert verdict_word(outcome_of(root / "arc" / "ep-empty")) == "NO OUTCOME"


# ---------------------------------------------------------------- the renderer
def test_index_shows_all_verdicts_and_never_a_false_chain(root: Path):
    page = render_index(root)
    assert "Aither General Solver" in page
    assert "ep-kept" in page and "KEPT" in page
    assert "REVERTED" in page and "REFUSED" in page
    assert "TAMPERED at row" in page          # the edited journal is called out
    assert "TRUNCATED" in page                # the cut journal is called out
    assert "UNREADABLE" in page               # the corrupt journal is called out
    # every intact episode's card shows a verified chain, not just one of them
    assert page.count("chain verified") >= 4


def test_xss_is_escaped_on_index_and_detail(root: Path):
    for page in (render_index(root),
                 render_episode(root, "arc/ep-xss")):
        assert "<script>alert(1)</script>" not in page
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_empty_root_renders_honestly(tmp_path: Path):
    page = render_index(tmp_path)
    assert "No episodes yet" in page and "chain verified" not in page


# ---------------------------------------------------------------- the handler
def test_healthz_and_index_routes(root: Path):
    status, body, ctype = handle(root, "/healthz", {})
    assert status == 200 and body == b"ok"
    status, body, ctype = handle(root, "/", {})
    assert status == 200 and "text/html" in ctype
    assert b"Aither General Solver" in body


def test_api_episodes_carries_summaries_not_rows(root: Path):
    import json
    status, body, ctype = handle(root, "/api/episodes", {})
    assert status == 200 and ctype == "application/json"
    eps = json.loads(body)
    assert any(e["dir"] == "ep-kept" for e in eps)
    assert all("rows" not in e for e in eps)
    assert all("verify" in e for e in eps)


def test_verify_endpoint_and_raw_route(root: Path):
    import json
    status, body, _ = handle(root, "/api/verify", {"d": "arc/ep-kept"})
    assert status == 200 and json.loads(body)["ok"] is True
    status, body, _ = handle(root, "/api/verify", {"d": "arc/ep-tampered"})
    assert status == 200 and json.loads(body)["ok"] is False
    status, body, ctype = handle(root, "/raw", {"d": "arc/ep-kept"})
    assert status == 200 and body.startswith(b"{") and b'"kind"' in body
    assert "x-ndjson" in ctype


def test_public_filter_applies_to_raw_and_verify_routes(root: Path):
    status, _, _ = handle(root, "/api/episode",
                          {"d": "fleet/ep-internal"}, ("arc:",))
    assert status == 404          # hidden episode never leaks
    status, _, _ = handle(root, "/raw", {"d": "fleet/ep-internal"}, ("arc:",))
    assert status == 404
    status, _, _ = handle(root, "/api/verify", {"d": "fleet/ep-internal"}, ("arc:",))
    assert status == 404
    status, _, _ = handle(root, "/api/episode", {"d": "fleet/ep-internal"})
    assert status == 200          # visible without a filter


def test_traversal_and_bad_inputs_are_404(root: Path):
    for bad in ("../ep-kept", "..%2Fep-kept", "../../etc/passwd",
                "/etc/passwd", "arc/../arc/ep-kept", "", "no/such/ep"):
        status, _, _ = handle(root, "/api/episode", {"d": bad})
        assert status == 404, bad
    status, _, _ = handle(root, "/api/episode", {"d": "arc\\..\\ep-kept"})
    assert status == 404          # Windows backslash spellings die too
    status, _, _ = handle(root, "/nope", {})
    assert status == 404
    status, _, _ = handle(root, "/api/episode", {})
    assert status == 404


def test_handler_never_raises_on_garbage(root: Path):
    import json
    status, body, _ = handle(root, "/api/episode", {"d": "arc/ep-unreadable"})
    assert status == 200
    assert json.loads(body)["unreadable"]  # the corrupt one is still served, marked
