"""A keep-or-revert HARNESS as an EnvironmentAdapter — the second domain of the general solver.

    from awgym.envs.harness_adapter import HarnessEnv, HarnessScorer
    env    = HarnessEnv(harness="harness.json")     # awevolve Harness: ONE file, ONE strict scorer
    scorer = HarnessScorer(harness="harness.json")  # Score(<metric>=current believed) | Refusal

    spec = ProblemSpec(problem_id="harness:triton_kernels", domain="harness",
                       adapter_ref="awgym.envs.harness_adapter:HarnessEnv",
                       scorer_ref="awgym.envs.harness_adapter:HarnessScorer",
                       adapter_kwargs={"harness": "…/harness.json"},
                       scorer_kwargs={"harness": "…/harness.json"},
                       budget=Budget(steps=6, seconds=1800), planner="none")

The world is a FILE and a COMMAND THAT SCORES IT (awevolve's `Harness`; the monorepo's
`lib/research/harnesses.Harness` is the same shape and is loaded through awevolve's
`load_harness` when exported as JSON). One step = one PROPOSAL: the action is the file's new
content (a str, or {"content": str}); the adapter writes it, runs the scorer, and KEEPS the
change only if the scorer believed it AND it beat the current score — otherwise the file is
REVERTED byte-for-byte. That is `ratchet_loop`'s keep-or-revert, expressed as `step()`, so the
same `ProblemSession` that plays an ARC game plays a kernel benchmark.

What the ARC lane taught, applied here:
  * The scorer's NON-ZERO EXIT means the candidate is WRONG, not merely slower. A refused
    trial is reverted and carries `strict_ok=False`; its reward is 0, never a number parsed
    out of a broken build's stdout.
  * Direction is stated (`minimize`), never assumed. Reward is `delta` in the IMPROVING
    direction, so a minimised harness ratchets the right way.
  * `HarnessScorer` scores the file's CURRENT believed score (the last kept measurement),
    and REFUSES when no measurement was ever believed — an episode where every proposal was
    rejected and the baseline itself failed to score is not a zero.
  * `actions()` is an open vocabulary: it returns the one action KIND ("propose"); what to
    propose is the policy's whole job (an LLM, awevolve's proposer, a mutation operator).

Safety: only `mutable_file` is ever written; awevolve's `check_scorer_boundary` refuses a
harness whose mutable file could rewrite what judges it. The original bytes are restored on
every refused/worse trial and on `close()`, so a crashed episode does not leave a losing
candidate on disk as if it had been kept.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from awpredict.contracts import Refusal, Score, Transition

try:
    from awevolve.harness import Harness, HarnessError, load_harness
    from awevolve.scorer import ScoreError
    from awevolve.scorer import Scorer as _EvolveScorer
except ImportError as _exc:  # pragma: no cover - loud, never a silent fallback
    raise ImportError("awgym.envs.harness_adapter needs awevolve (pip install awevolve): "
                      f"{_exc}") from _exc

HarnessLike = Union[str, Path, Dict[str, Any], Harness]
ACTION_KINDS: Tuple[str, ...] = ("propose",)


def _load(harness: HarnessLike) -> Harness:
    if isinstance(harness, Harness):
        return harness
    return load_harness(harness)


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _meta_of(t: Any) -> Dict[str, Any]:
    """A Transition, a journaled dict, or any object carrying `.meta` — never raise."""
    meta = getattr(t, "meta", None)
    if meta is None and isinstance(t, dict):
        meta = t.get("meta")
    return meta if isinstance(meta, dict) else {}


class HarnessEnv:
    """EnvironmentAdapter over one awevolve Harness: propose → score → keep or revert."""

    def __init__(self, harness: HarnessLike, keep_or_revert: bool = True,
                 max_obs_chars: int = 4000) -> None:
        self.harness = _load(harness)
        ok, why = self.harness.is_available()
        if not ok:
            raise HarnessError(f"harness {self.harness.name!r} is not available: {why}")
        self.domain = f"harness:{self.harness.name}"
        self.keep_or_revert = keep_or_revert
        self.max_obs_chars = max_obs_chars
        self._scorer = _EvolveScorer(self.harness)
        self._path = self.harness.mutable_file
        self._original: Optional[str] = None
        self._current_score: Optional[float] = None
        self._trials = 0
        self._kept = 0
        self._refused = 0
        self._history: List[Dict[str, Any]] = []

    # -- EnvironmentAdapter --------------------------------------------------
    def reset(self) -> Dict[str, Any]:
        """Remember the original bytes and measure the baseline once."""
        text = self._path.read_text(encoding="utf-8")
        self._original = text
        self._trials = self._kept = self._refused = 0
        self._history = []
        try:
            self._current_score = self._scorer.score_once()
            baseline_ok, why = True, ""
        except ScoreError as exc:
            self._current_score, baseline_ok, why = None, False, str(exc)
        return self._state(text, last={"baseline_ok": baseline_ok, "reason": why})

    def observe(self, env_state: Any) -> Any:
        if not isinstance(env_state, dict):
            return env_state
        return {"file": self._path.name, "sha": env_state.get("sha"),
                "score": env_state.get("score"), "trials": env_state.get("trials"),
                "content": (env_state.get("content") or "")[: self.max_obs_chars]}

    def actions(self) -> Sequence[Any]:
        return list(ACTION_KINDS)

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        content = action.get("content") if isinstance(action, dict) else action
        if not isinstance(content, str):
            raise ValueError("a harness action is the proposed file content (str or "
                             "{'content': str}); got " + type(action).__name__)
        before = self._path.read_text(encoding="utf-8")
        if self._original is None:
            self._original = before
        self._trials += 1
        if content == before:
            info = {"kept": False, "strict_ok": True, "score": self._current_score,
                    "current_score": self._current_score, "reason": "no-op proposal"}
            return self._state(before, last=info), 0.0, False, info
        _atomic_write(self._path, content)
        try:
            score = self._scorer.score_once()
        except ScoreError as exc:
            self._refused += 1
            _atomic_write(self._path, before)
            info = {"kept": False, "strict_ok": False, "score": None,
                    "current_score": self._current_score, "reason": str(exc)[:400]}
            self._history.append(info)
            return self._state(before, last=info), 0.0, False, info
        prev = self._current_score
        improved = prev is None or self.harness.better(score, prev)
        if prev is None:
            delta = 0.0
        else:
            delta = (prev - score) if self.harness.minimize else (score - prev)
        keep = improved or not self.keep_or_revert
        if keep:
            self._kept += 1
            self._current_score = score
            text = content
        else:
            _atomic_write(self._path, before)
            text = before
        info = {"kept": keep, "strict_ok": True, "score": score,
                "current_score": self._current_score, "delta": delta,
                "reason": "improved" if improved else "not better — reverted"}
        self._history.append(info)
        return self._state(text, last=info), float(delta), False, info

    # -- housekeeping --------------------------------------------------------
    def close(self) -> None:
        """Restore the original bytes unless a trial was KEPT (a kept file is the result)."""
        if self._original is not None and self._kept == 0:
            _atomic_write(self._path, self._original)

    def _state(self, text: str, last: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"content": text, "sha": _sha(text), "score": self._current_score,
                "trials": self._trials, "kept": self._kept, "refused": self._refused,
                "last": last or {}}


class HarnessScorer:
    """Score = the file's CURRENT believed score after the episode. Refuses when none exists."""

    def __init__(self, harness: HarnessLike) -> None:
        h = _load(harness)
        self.metric = h.metric_name
        self.minimize = bool(h.minimize)
        self._name = h.name

    def score(self, episode: List[Transition]) -> "Score | Refusal":
        if not episode:
            return Refusal(f"harness {self._name}: no trials were made")
        current: Optional[float] = None
        strict_all = True
        believed = 0
        for t in episode:
            meta = _meta_of(t)
            if meta.get("strict_ok") is False:
                strict_all = False
            if meta.get("current_score") is not None:
                current = float(meta["current_score"])
                believed += 1
        if current is None:
            return Refusal(f"harness {self._name}: no trial produced a believed score "
                           f"({len(episode)} trials, scorer refused or never ran)")
        return Score(metric=self.metric, value=current, minimize=self.minimize,
                     strict_ok=True,
                     evidence=f"{len(episode)} trials, {believed} believed"
                              f"{'' if strict_all else ', some refused'}")
