"""Policy layers — the ARC solver's domain-agnostic middle, as composable policies.

The ARC fork proved a layered solver (memory -> hypothesis council -> model-based
plan -> action gate -> attempt ledger) beats the bare-LLM baseline on the same
games and budget. Every one of those layers is domain-agnostic; only the
observation encoding is grid-specific. This module ports the middle onto the
ProblemSession policy shape -- (obs, session) -> action -- with two conventions:

  * A layer returns an ACTION, or None when it cannot decide (endpoint down,
    no memory, everything tried). None is a REFUSAL to act, never a guess.
  * `compose()` runs layers in order and guarantees a legal action: the first
    layer that answers wins; if none do, the fallback policy acts and the
    refusal is recorded as a fact, so "the council was down" and "the council
    decided" are distinguishable in the journal.

Memory is the journal itself: recall reads `fact` rows from EARLIER episodes of
the same problem (sibling dirs of the current episode), and the ledger reads
their `attempt` rows. Nothing here imports the monorepo -- awgym ships to PyPI
(gate 1zi), so the awm write/read seam stays in the kernel's P6, not here.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .journal import EpisodeJournal, JournalError

Policy = Callable[[Any, Any], Any]
LLM_URL_ENV = "AITHER_SOLVE_LLM_URL"


# ------------------------------------------------------------------- memory


def _prior_episode_dirs(session: Any) -> List[Path]:
    """Sibling episode dirs of the CURRENT episode (same problem), oldest first."""
    journal = getattr(session, "journal", None)
    cur = Path(getattr(journal, "dir", ""))
    if not cur.is_dir():
        return []
    out = []
    for d in sorted(cur.parent.iterdir()):
        if d == cur or not d.is_dir():
            continue
        if not (d / "episode.json").is_file():
            continue
        out.append(d)
    return out


def prior_rows(session: Any, kinds: Tuple[str, ...]) -> List[dict]:
    """Payloads of journal rows of the given kinds from prior episodes."""
    out: List[dict] = []
    for d in _prior_episode_dirs(session):
        j = EpisodeJournal(d)
        try:
            rows = list(j.rows())
        except JournalError:
            continue  # a broken prior record is not this episode's memory
        for row in rows:
            if row.get("kind") in kinds and isinstance(row.get("payload"), dict):
                out.append(row["payload"])
    return out


def recall_facts(session: Any, max_facts: int = 12,
                 min_confidence: float = 0.0) -> List[Tuple[str, str, float]]:
    """Facts earlier episodes of this problem recorded (relation, fact, conf)."""
    out = []
    for p in prior_rows(session, ("fact",)):
        conf = p.get("confidence")
        if isinstance(conf, (int, float)) and conf < min_confidence:
            continue
        rel = str(p.get("relation") or "?")
        fact = str(p.get("fact") or "")
        if fact:
            out.append((rel, fact, float(conf) if isinstance(conf, (int, float))
                        else 0.0))
        if len(out) >= max_facts:
            break
    return out


def failed_opening_actions(session: Any) -> List[Any]:
    """Actions earlier episodes OPENED with and got no first effect from."""
    tried = []
    for p in prior_rows(session, ("attempt",)):
        if p.get("first_effect_step") in (None, -1) and p.get("opening_actions"):
            tried.append(p["opening_actions"][0])
    return tried


def episode_rewards(session: Any, limit: int = 10) -> List[float]:
    """Reward history of the CURRENT episode (already journaled steps)."""
    out = []
    j = getattr(session, "journal", None)
    if j is None:
        return out
    try:
        rows = list(j.rows())
    except JournalError:
        return out
    for row in rows:
        if row.get("kind") != "transition":
            continue
        p = row.get("payload") or {}
        r = p.get("reward")
        if isinstance(r, (int, float)):
            out.append(float(r))
    return out[-limit:]


# -------------------------------------------------------------------- layers


def arc_ledger_action(session: Any, valid: Sequence[Any]) -> Optional[Any]:
    """An action not yet tried as an opening that failed to produce an effect.

    Returns None when every legal action has already been tried that way --
    the ledger has nothing new to offer, so the caller should fall back.
    """
    tried = set(failed_opening_actions(session))
    for a in valid:
        if a not in tried:
            return a
    return None


def recall_context(session: Any, max_facts: int = 12) -> str:
    """Prior facts rendered as compact prompt context, or "" when none."""
    facts = recall_facts(session, max_facts=max_facts)
    if not facts:
        return ""
    return "\n".join(f"- ({rel}) {fact} [conf {conf:g}]" for rel, fact, conf in facts)


def council_policy(*, url: Optional[str] = None, model: str = "default",
                   client: Any = None, timeout_s: float = 25.0,
                   max_tokens: int = 300) -> Policy:
    """Ask a chat-completing endpoint for the next action.

    url defaults to $AITHER_SOLVE_LLM_URL; NO URL means this layer can never
    decide and returns None immediately (a council that is not configured is
    not a council that guessed). The prompt is built from the current
    observation, the reward history of this episode and recalled facts from
    prior episodes. The reply is parsed as an action; anything unparseable is
    a refusal. A transport error is a refusal, logged via session.learn.
    """
    if not url:
        url = os.getenv(LLM_URL_ENV) or ""

    def _post(payload: dict) -> Any:
        if client is not None:
            r = client.post(url, json=payload, timeout=timeout_s)
        else:
            import httpx
            with httpx.Client(timeout=timeout_s) as c:
                r = c.post(url, json=payload)
        r.raise_for_status()
        return r.json()

    def policy(obs: Any, session: Any) -> Any:
        if not url:
            return None
        try:
            obs_txt = json.dumps(obs, default=str)[:4000]
            history = ", ".join(f"{r:g}" for r in episode_rewards(session)) or "none"
            context = recall_context(session)
            facts = f"\nWhat prior episodes learned:\n{context}" if context else ""
            prompt = (
                f"You are the planner for a solver playing a puzzle nobody explained.\n"
                f"Choose ONE action to take next from the legal set "
                f"{json.dumps(_valid_actions(session))}.\n"
                f"Rewards so far this episode: {history}.\n"
                f"Current observation:\n{obs_txt}{facts}\n"
                f"Reply with ONLY the action value.")
            body = _post({"model": model, "messages": [{"role": "user",
                                                        "content": prompt}],
                          "max_tokens": max_tokens,
                          "temperature": 0.2})
            content = str(((body.get("choices") or [{}])[0]
                           .get("message") or {}).get("content") or "").strip()
            action = _parse_action(content, _valid_actions(session))
            if action is None:
                session.learn("policy", f"council reply unparseable: {content!r}", 0.5)
                return None
            return action
        except Exception as exc:  # noqa: BLE001 - any transport defect = refusal
            session.learn("policy", f"council unavailable ({type(exc).__name__})", 0.5)
            return None

    return policy


def _parse_action(content: str, valid: Sequence[Any]) -> Optional[Any]:
    for line in content.splitlines():
        line = line.strip().strip("`").strip()
        if line.isdigit():
            n = int(line)
            if n in valid:
                return n
    return None


def _valid_actions(session: Any) -> List[Any]:
    actions = getattr(getattr(session, "adapter", None), "actions", ())
    if callable(actions):
        try:
            actions = actions()
        except Exception:
            return []
    if not isinstance(actions, (list, tuple)):
        return []
    return list(actions)


def wm_veto_policy(wm: Any, inner: Policy) -> Policy:
    """Run `inner`, then ask the world model to veto the resulting state.

    The world model duck: `wm.veto(obs, action, session) -> (allow: bool,
    reason: str)`. A veto means the layer returns None (the caller falls
    back) and the veto reason is recorded as a fact -- a veto is a decision,
    not a silence. A world model that raises is treated as ABSENT (no veto),
    because a broken auxiliary must not freeze the solver.
    """

    def policy(obs: Any, session: Any) -> Any:
        action = inner(obs, session)
        if action is None:
            return None
        try:
            allow, reason = wm.veto(obs, action, session)
        except Exception:
            return action
        if not allow:
            session.learn("policy", f"wm vetoed {action!r}: {reason}", 0.7)
            return None
        return action

    return policy


def compose(steps: List[Callable[..., Any]], fallback: Policy,
            valid: Optional[Sequence[Any]] = None) -> Policy:
    """First layer that answers wins; otherwise the fallback acts.

    The final action is guaranteed legal: if it is outside the adapter's
    action set the fallback draws randomly from it instead. Each layer that
    declines is recorded as a fact (capped so a disabled council does not
    flood the journal), keeping "council down" distinct from "council said no".
    """
    if not steps:
        raise ValueError("compose needs at least one layer or a fallback")
    counts: Dict[str, int] = {}

    def _note(session: Any, name: str, why: str) -> None:
        counts[name] = counts.get(name, 0) + 1
        if counts[name] <= 3:
            session.learn("policy", f"{name} declined: {why}", 0.5)

    def policy(obs: Any, session: Any) -> Any:
        actions = list(valid) if valid is not None else _valid_actions(session)
        for step in steps:
            name = getattr(step, "__name__", type(step).__name__)
            try:
                action = step(obs, session)
            except Exception as exc:  # noqa: BLE001 - a broken layer = a refusal
                _note(session, name, f"raised {type(exc).__name__}")
                continue
            if action is None:
                _note(session, name, "no decision")
                continue
            if actions and action not in actions:
                _note(session, name, f"{action!r} not in legal set {actions}")
                continue
            return action
        try:
            action = fallback(obs, session)
        except Exception:
            action = None
        if action is None or (actions and action not in actions):
            action = random.choice(actions) if actions else None
        return action

    policy.__name__ = "composed"  # type: ignore[attr-defined]
    return policy


def load_policy(desc: dict, *, client: Any = None) -> Policy:
    """Build a policy from a queue-entry description (never free text).

    {"kind": "layered",
     "layers": ["arc_ledger", "council", "wm_veto"],   # in order, first wins
     "council": {"url": ..., "model": ...},
     "wm": <object or None>,                            # operator-provided
     "fallback": "random" | "const" | {"const": <action>}}

    Unknown kinds/layers raise ValueError -- a malformed policy description
    must refuse the item (awrun code 2), never silently become a guess.
    """
    kind = desc.get("kind")
    if kind == "random":
        return _random_fallback(int(desc["seed"]) if desc.get("seed") is not None
                                else None)
    if kind == "const":
        return _const_fallback(desc.get("action", 0))
    if kind != "layered":
        raise ValueError(f"unknown policy kind {kind!r} (random|const|layered)")
    steps: List[Callable[..., Any]] = []
    for name in desc.get("layers") or []:
        if name == "arc_ledger":
            steps.append(lambda obs, s, _n=name: arc_ledger_action(s, _valid_actions(s)))
        elif name == "council":
            c = desc.get("council") or {}
            steps.append(council_policy(url=c.get("url"), model=c.get("model", "default"),
                                        client=client))
        elif name == "wm_veto":
            wm = desc.get("wm")
            if wm is None:
                raise ValueError("wm_veto layer needs a wm object in the desc")
            inner = steps.pop() if steps else None
            if inner is None:
                raise ValueError("wm_veto must follow a layer it can veto")
            steps.append(wm_veto_policy(wm, inner))
        else:
            raise ValueError(f"unknown layer {name!r} (arc_ledger|council|wm_veto)")
    fb = desc.get("fallback") or {}
    if isinstance(fb, str):
        fallback = (_random_fallback(None) if fb == "random"
                    else _const_fallback(0) if fb == "const"
                    else _raise_unknown(fb))
    elif isinstance(fb, dict):
        fallback = load_policy(fb)
    else:
        raise ValueError(f"fallback must be a dict or name, got {type(fb).__name__}")
    if not steps:
        raise ValueError("layered policy with no layers is a fallback in disguise")
    return compose(steps, fallback)


def _raise_unknown(name: str) -> Policy:
    raise ValueError(f"unknown fallback {name!r} (random|const)")


def _random_fallback(seed: Optional[int]) -> Policy:
    rng = random.Random(seed)

    def policy(obs: Any, session: Any) -> Any:
        actions = _valid_actions(session)
        return rng.choice(actions) if actions else None

    policy.__name__ = "random_fallback"  # type: ignore[attr-defined]
    return policy


def _const_fallback(action: Any) -> Policy:
    def policy(obs: Any, session: Any) -> Any:
        return action

    policy.__name__ = "const_fallback"  # type: ignore[attr-defined]
    return policy
