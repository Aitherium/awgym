"""EpisodeJournal — the hash-chained record one solving episode writes, and replays from.

    from awgym.gym.journal import EpisodeJournal

    j = EpisodeJournal(episode_dir)
    j.init(spec_dict, episode_id)
    j.append("transition", {...})
    j.append("outcome", {...})
    ok, rows, head = j.verify()          # recompute the chain from row 0

Why a CHAIN and not a log: the ARC lane learned that a run which learned nothing and a
run which explored honestly and found nothing print the same lines. A journal whose every
row hashes its predecessor makes an edited, truncated or reordered record DETECTABLE
rather than merely different — the same v1 contract the Dark Matters world runner uses
(`lib/worldrunner/journal.py`), re-stated here in stdlib because awgym ships to PyPI and
may import nothing from the monorepo.

Files in an episode dir:
  episode.json     {"v":1,"episode_id","spec":{...},"created"}
  journal.jsonl    one row per record; row = {v, seq, ts, kind, payload, prev, hash}
  state.json       {"v":1,"episode_id","rows","hash","last_ts"}

The journal is append-only. `verify()` is the replay: it is what a gate grades.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

JOURNAL_V = 1
GENESIS_PREV = "0" * 64


class JournalError(Exception):
    """A journal that cannot be trusted — never read as an empty one."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=_json_default)


def _json_default(o: Any) -> Any:
    # dataclasses arrive already as dicts (callers pass .as_dict()); anything else
    # that is not JSON-native is stringified LOUDLY rather than dropped.
    return f"<{type(o).__name__}:{o!r}>"


def row_hash(prev: str, row_wo_hash: dict) -> str:
    return hashlib.sha256((prev + canonical(row_wo_hash)).encode("utf-8")).hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class EpisodeJournal:
    """Append-only, hash-chained record of one episode."""

    def __init__(self, episode_dir: str | Path):
        self.dir = Path(episode_dir)
        self.journal_path = self.dir / "journal.jsonl"
        self.episode_path = self.dir / "episode.json"
        self.state_path = self.dir / "state.json"
        self._seq = 0
        self._head = GENESIS_PREV

    # ------------------------------------------------------------------ write
    def init(self, spec: dict, episode_id: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.journal_path.exists() and self.journal_path.stat().st_size > 0:
            raise JournalError(f"{self.journal_path} already holds rows — a journal is "
                               f"append-only; start a new episode dir")
        _atomic_write(self.episode_path, canonical(
            {"v": JOURNAL_V, "episode_id": episode_id, "spec": spec,
             "created": _now_iso()}) + "\n")
        self.journal_path.write_text("", encoding="utf-8")
        self._seq, self._head = 0, GENESIS_PREV
        self._write_state(episode_id)

    def append(self, kind: str, payload: dict) -> dict:
        if not kind:
            raise JournalError("a journal row needs a kind")
        row = {"v": JOURNAL_V, "seq": self._seq, "ts": _now_iso(), "kind": kind,
               "payload": payload, "prev": self._head}
        row["hash"] = row_hash(self._head, row)
        with self.journal_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(canonical(row) + "\n")
        self._seq += 1
        self._head = row["hash"]
        return row

    def _write_state(self, episode_id: str) -> None:
        _atomic_write(self.state_path, canonical(
            {"v": JOURNAL_V, "episode_id": episode_id, "rows": self._seq,
             "hash": self._head, "last_ts": _now_iso()}) + "\n")

    def close(self, episode_id: str) -> None:
        self._write_state(episode_id)

    # ------------------------------------------------------------------- read
    def rows(self) -> Iterator[dict]:
        if not self.journal_path.exists():
            raise JournalError(f"{self.journal_path} does not exist")
        with self.journal_path.open("r", encoding="utf-8") as fh:
            for n, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise JournalError(f"row {n} is not JSON: {exc}") from exc

    def verify(self) -> tuple[bool, int, str]:
        """Recompute the chain from row 0. Returns (ok, rows_checked, head_hash).

        `ok` is False on the FIRST break — a wrong prev, a wrong hash, a seq gap.
        A journal with zero rows verifies as (True, 0, GENESIS_PREV): empty is a
        legitimate state; UNREADABLE is not, and raises.
        """
        prev, n = GENESIS_PREV, 0
        for row in self.rows():
            if row.get("seq") != n or row.get("prev") != prev:
                return False, n, prev
            claimed = row.get("hash")
            body = {k: v for k, v in row.items() if k != "hash"}
            if row_hash(prev, body) != claimed:
                return False, n, prev
            prev, n = claimed, n + 1
        return True, n, prev

    def kinds(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in self.rows():
            out[row.get("kind", "?")] = out.get(row.get("kind", "?"), 0) + 1
        return out

    def state(self) -> Optional[dict]:
        if not self.state_path.exists():
            return None
        return json.loads(self.state_path.read_text(encoding="utf-8"))


def _self_test() -> int:
    import shutil

    ok = True
    d = Path(tempfile.mkdtemp(prefix="awgym-journal-"))
    try:
        j = EpisodeJournal(d)
        j.init({"problem_id": "toy"}, "e1")
        j.append("transition", {"step": 0})
        j.append("outcome", {"score": 1.0})
        j.close("e1")
        good, n, head = j.verify()
        ok &= good and n == 2 and j.state()["hash"] == head
        # tamper: edit a payload -> chain breaks at that row
        lines = j.journal_path.read_text(encoding="utf-8").splitlines()
        lines[0] = lines[0].replace('"step":0', '"step":9')
        j.journal_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        bad, at, _ = j.verify()
        ok &= (not bad) and at == 0
        # truncate: drop the last row -> verifies but head != state hash
        j.journal_path.write_text(lines[0].replace('"step":9', '"step":0') + "\n",
                                  encoding="utf-8")
        good2, n2, head2 = j.verify()
        ok &= good2 and n2 == 1 and head2 != j.state()["hash"]
        # re-init on a non-empty journal is refused — the refusal is the assertion
        reinit_refused = False
        try:
            j.init({}, "e1")
        except JournalError:
            reinit_refused = True
        ok &= reinit_refused
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("[ok] journal self-test" if ok else "[FAIL] journal self-test")
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_self_test())
