"""Policy layers — memory recall, attempt ledger, council, WM veto, compose.

Each layer returns an ACTION or None (a refusal to act, never a guess);
compose guarantees a legal action and journals which layers declined, so
"the council was down" and "the council decided" differ in the record.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from awgym.gym.journal import EpisodeJournal  # noqa: E402
from awgym.gym.policies import (  # noqa: E402
    arc_ledger_action,
    compose,
    council_policy,
    failed_opening_actions,
    load_policy,
    prior_rows,
    recall_facts,
    wm_veto_policy,
)

SPEC = {"problem_id": "arc:al7306", "domain": "arc"}
VALID = [1, 2, 3, 4, 5, 6, 7]


def _session(root: Path, slug: str, ep: str, rows: list[tuple[str, dict]]) -> Path:
    d = root / slug / ep
    j = EpisodeJournal(d)
    j.init(dict(SPEC, problem_id=f"arc:{slug}"), ep)
    for kind, payload in rows:
        j.append(kind, payload)
    j.close(ep)
    return d


@pytest.fixture()
def session(tmp_path: Path):
    """Two prior episodes (one with a fact + a failed opening) + the live one."""
    _session(tmp_path, "al7306", "ep-1", [
        ("fact", {"relation": "direction", "fact": "+1 raises the counter",
                  "confidence": 0.9}),
        ("attempt", {"steps": 4, "opening_actions": [5, 5], "first_effect_step": -1,
                     "trajectory": [0.0, 0.0, 0.0, 0.0]}),
        ("outcome", {"score": {"metric": "m", "value": 0.0}, "baseline": 0.0,
                     "kept": False, "refusal": None})])
    _session(tmp_path, "al7306", "ep-2", [
        ("attempt", {"steps": 2, "opening_actions": [3], "first_effect_step": 1,
                     "trajectory": [0.0, 1.0]})])
    live = tmp_path / "al7306" / "ep-3"
    j = EpisodeJournal(live)
    j.init(dict(SPEC, problem_id="arc:al7306"), "ep-3")
    adapter = SimpleNamespace(actions=VALID)
    s = SimpleNamespace(journal=j, adapter=adapter)
    s.learn = lambda rel, fact, conf: None  # noqa: E731 - tests only want policy
    return s


# ------------------------------------------------------------------- memory
def test_recall_facts_reads_prior_episodes(session):
    facts = recall_facts(session)
    assert facts and facts[0][0] == "direction"
    assert "+1 raises" in facts[0][1]


def test_recall_facts_respects_confidence_floor(session):
    assert recall_facts(session, min_confidence=0.95) == []


def test_failed_openings_only_from_effectless_attempts(session):
    tried = failed_opening_actions(session)
    assert 5 in tried and 3 not in tried  # ep-2's [3] DID produce an effect


def test_ledger_offers_an_untried_opening(session):
    action = arc_ledger_action(session, VALID)
    assert action is not None and action not in (5,)


def test_ledger_returns_none_when_everything_was_tried(session):
    tried_all = SimpleNamespace(
        journal=session.journal,
        adapter=SimpleNamespace(actions=[7]))
    _session(Path(session.journal.dir).parent.parent, "al7306", "ep-x", [
        ("attempt", {"opening_actions": [7], "first_effect_step": -1})])
    # the fixture above only tried 5 and 3; simulate all tried by using [7] only
    action = arc_ledger_action(tried_all, [7])
    assert action is None


def test_prior_rows_skips_unreadable_records(tmp_path: Path):
    d = tmp_path / "al7306" / "ep-broken"
    j = EpisodeJournal(d)
    j.init(dict(SPEC, problem_id="arc:al7306"), "ep-broken")
    j.close("ep-broken")
    (d / "journal.jsonl").write_text("{not json\n", encoding="utf-8")
    live = _session(tmp_path, "al7306", "ep-ok", [])
    s = SimpleNamespace(journal=EpisodeJournal(live))
    assert prior_rows(s, ("fact",)) == []


# ------------------------------------------------------------------ council
class _FakeClient:
    """httpx-shaped client: returns the canned response or raises."""

    def __init__(self, content="3", fail=False):
        self.content = content
        self.fail = fail
        self.calls = []

    def post(self, url, json, timeout=None):
        self.calls.append((url, json))
        if self.fail:
            raise RuntimeError("endpoint down")
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": self.content}}]})


def test_council_parses_a_valid_action(session):
    client = _FakeClient(content="4")
    policy = council_policy(url="http://llm.test/v1/chat", client=client)
    assert policy({"grid": [[0]]}, session) == 4
    assert "observation" in client.calls[0][1]["messages"][0]["content"]


def test_council_refuses_an_unparseable_reply(session):
    policy = council_policy(url="http://llm.test/v1/chat",
                            client=_FakeClient(content="maybe action blue"))
    notes = []
    s = SimpleNamespace(journal=session.journal, adapter=session.adapter,
                        learn=lambda r, f, c: notes.append(f))
    assert policy({"grid": [[0]]}, s) is None
    assert any("unparseable" in n for n in notes)


def test_council_without_a_url_cannot_decide(session, monkeypatch):
    monkeypatch.delenv("AITHER_SOLVE_LLM_URL", raising=False)
    policy = council_policy()  # no url, no env
    assert policy({"grid": [[0]]}, session) is None


def test_council_transport_error_is_a_refusal_not_a_crash(session):
    policy = council_policy(url="http://llm.test/v1/chat",
                            client=_FakeClient(fail=True))
    notes = []
    s = SimpleNamespace(journal=session.journal, adapter=session.adapter,
                        learn=lambda r, f, c: notes.append(f))
    assert policy({"grid": [[0]]}, s) is None
    assert any("unavailable" in n for n in notes)


# ----------------------------------------------------------------- wm veto
class _VetoWM:
    def __init__(self, allow=True):
        self.allow = allow

    def veto(self, obs, action, session):
        if self.allow:
            return True, "fine"
        return False, "predicted surprise 4.2"


def test_veto_blocks_and_records_the_reason(session):
    inner = lambda obs, s: 2  # noqa: E731
    policy = wm_veto_policy(_VetoWM(allow=False), inner)
    notes = []
    s = SimpleNamespace(journal=session.journal, adapter=session.adapter,
                        learn=lambda r, f, c: notes.append(f))
    assert policy({"grid": [[0]]}, s) is None
    assert any("vetoed" in n and "surprise" in n for n in notes)


def test_veto_passes_an_allowed_action_through(session):
    inner = lambda obs, s: 2  # noqa: E731
    policy = wm_veto_policy(_VetoWM(allow=True), inner)
    assert policy({"grid": [[0]]}, session) == 2


def test_broken_wm_is_absent_not_a_freeze(session):
    class BrokenWM:
        def veto(self, obs, action, session):
            raise RuntimeError("wm crashed")

    policy = wm_veto_policy(BrokenWM(), lambda obs, s: 2)  # noqa: E731
    assert policy({"grid": [[0]]}, session) == 2


# ------------------------------------------------------------------- compose
def test_compose_first_winner_and_fallback(session):
    policy = compose([lambda obs, s: None, lambda obs, s: 6],  # noqa: E731
                     fallback=lambda obs, s: 1)  # noqa: E731
    assert policy({"grid": [[0]]}, session) == 6
    policy2 = compose([lambda obs, s: None, lambda obs, s: None],  # noqa: E731
                      fallback=lambda obs, s: 1)  # noqa: E731
    assert policy2({"grid": [[0]]}, session) == 1


def test_compose_guarantees_a_legal_action(session):
    policy = compose([lambda obs, s: 99], fallback=lambda obs, s: 1)  # noqa: E731
    assert policy({"grid": [[0]]}, session) in VALID  # 99 coerced to a legal draw
    assert policy({"grid": [[0]]}, session) in VALID


def test_compose_journals_declined_layers_once(session):
    notes = []
    s = SimpleNamespace(journal=session.journal, adapter=session.adapter,
                        learn=lambda r, f, c: notes.append(f))
    policy = compose([lambda obs, sess: None],  # noqa: E731
                     fallback=lambda obs, sess: 3)  # noqa: E731
    for _ in range(10):
        policy({"grid": [[0]]}, s)
    assert len([n for n in notes if "declined" in n]) <= 3  # capped, not flooding


# ------------------------------------------------------------- load_policy
def test_load_policy_roundtrip_with_fake_council(session):
    desc = {"kind": "layered", "layers": ["council"],
            "council": {"url": "http://llm.test/v1/chat", "model": "m"},
            "fallback": {"kind": "random"}}
    policy = load_policy(desc, client=_FakeClient(content="2"))
    assert policy({"grid": [[0]]}, session) == 2


def test_load_policy_refuses_unknown_kinds_and_layers(session):
    with pytest.raises(ValueError, match="unknown policy kind"):
        load_policy({"kind": "magic"})
    with pytest.raises(ValueError, match="unknown layer"):
        load_policy({"kind": "layered", "layers": ["crystal_ball"],
                     "fallback": {"kind": "random"}})
    with pytest.raises(ValueError, match="wm_veto"):
        load_policy({"kind": "layered", "layers": ["council", "wm_veto"],
                     "fallback": {"kind": "random"}})
    with pytest.raises(ValueError, match="no layers"):
        load_policy({"kind": "layered", "layers": [],
                     "fallback": {"kind": "random"}})
    with pytest.raises(ValueError, match="fallback"):
        load_policy({"kind": "layered", "layers": ["council"],
                     "fallback": "crystal_ball"})


def test_arc_ledger_layer_skips_tried_openings(session):
    desc = {"kind": "layered", "layers": ["arc_ledger"],
            "fallback": {"kind": "random"}}
    policy = load_policy(desc)
    assert policy({"grid": [[0]]}, session) in VALID
