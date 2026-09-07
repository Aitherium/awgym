"""watch — the public live-research page for the General Solver.

"A mind, learning in the open": renders hash-chained solve-episode journals and lets
any visitor re-verify the chain. The page's entire value is that it CANNOT lie — an
edited journal shows TAMPERED, a refused outcome never renders as a score, and an
unreadable journal shows as unreadable rather than being skipped silently.

Records are read through the EpisodeJournal contract (awgym.gym.journal), never
parsed by hand. Stdlib only, zero external assets, everything journal-derived is
HTML-escaped. Run it anywhere the journals live:

    python -m awgym.gym.watch --root ~/.aither/solve --port 8655
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from .journal import GENESIS_PREV, EpisodeJournal, JournalError

PAGE_TITLE = "Aither General Solver — a mind, learning in the open"

_esc = html.escape
_GREEN = "#0c7a3d"
_GREEN_BG = "#e7f6ec"
_RED = "#b3261e"
_RED_BG = "#fdecea"
_AMBER = "#9a6700"
_AMBER_BG = "#fff4d6"
_GREY = "#5f6368"
_GREY_BG = "#f1f3f4"


# --------------------------------------------------------------------- reading


def _episode_meta(episode_dir: Path) -> Optional[dict]:
    """episode.json as a dict, or None when missing/unreadable/not a dict."""
    try:
        meta = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    return meta if isinstance(meta, dict) else None


def _spec_of(meta: Optional[dict]) -> dict:
    if not isinstance(meta, dict):
        return {}
    spec = meta.get("spec")
    return spec if isinstance(spec, dict) else {}


def _is_public(spec: dict, public_prefixes: tuple) -> bool:
    """Public when the spec names a problem_id under a public prefix.

    No problem_id -> never public: an unlabelled episode must not appear on a
    public page by accident.
    """
    if not public_prefixes:
        return True
    pid = str(spec.get("problem_id") or "")
    return bool(pid) and any(pid.startswith(p) for p in public_prefixes)


def iter_episodes(root: Path, public_prefixes: tuple = ()) -> list:
    """Episode dirs under root, newest-created first; stray dirs tolerated."""
    root = Path(root)
    found = []
    if not root.is_dir():
        return found
    for ep_path in sorted(root.rglob("episode.json")):
        d = ep_path.parent
        meta = _episode_meta(d)
        if meta is None or not _is_public(_spec_of(meta), public_prefixes):
            continue
        created = meta.get("created")
        key = str(created) if isinstance(created, str) else ""
        found.append((key, d))
    found.sort(key=lambda pair: (pair[0], str(pair[1])), reverse=True)
    return [d for _, d in found]


def read_episode(episode_dir) -> dict:
    """Everything one episode dir says, verified. Never raises.

    tampered: the chain does not verify, or state.json disagrees with it.
    truncated: the chain verifies but state.json records MORE rows than remain
        (the journal's tail was cut after close()).
    unreadable: journal.jsonl exists but cannot be parsed — the record exists and
        the page must say so, never drop the episode.
    """
    d = Path(episode_dir)
    out = {"dir": d.name, "path": str(d), "episode": _episode_meta(d)}
    j = EpisodeJournal(d)
    rows: list = []
    unreadable: Optional[str] = None
    try:
        for row in j.rows():
            rows.append(row)
    except JournalError as exc:
        unreadable = str(exc)
    out["rows"] = rows
    out["unreadable"] = None
    counts: dict = {}
    for row in rows:
        kind = str(row.get("kind") or "?")
        counts[kind] = counts.get(kind, 0) + 1
    out["counts"] = counts
    state: Optional[dict] = None
    if unreadable is None:
        try:
            ok, n, head = j.verify()
        except JournalError as exc:
            unreadable = str(exc)
            ok, n, head = False, len(rows), GENESIS_PREV
        if unreadable is None:
            try:
                state = j.state()
            except Exception:
                state = None
            out["verify"] = {"ok": ok, "rows": n, "head": head}
            out["state"] = state
            state_matches = None if state is None else bool(state.get("hash") == head)
            out["state_matches"] = state_matches
            out["tampered"] = bool(not ok or state_matches is False)
            out["truncated"] = bool(ok and state_matches is False)
            return out
    out["unreadable"] = unreadable
    out["verify"] = {"ok": False, "rows": len(rows), "head": GENESIS_PREV}
    out["state"] = None
    out["state_matches"] = None
    out["tampered"] = True
    out["truncated"] = False
    return out


def outcome_of(episode_dir) -> Optional[dict]:
    """The payload of the kind=outcome row, or None."""
    for row in read_episode(episode_dir).get("rows", []):
        if row.get("kind") == "outcome" and isinstance(row.get("payload"), dict):
            return row["payload"]
    return None


def verdict_word(outcome: Optional[dict]) -> str:
    if outcome is None:
        return "NO OUTCOME"
    if outcome.get("refusal"):
        return "REFUSED"
    kept = outcome.get("kept")
    if kept is True:
        return "KEPT"
    if kept is False:
        return "REVERTED"
    return "UNJUDGED"


def episode_summary(episode_dir) -> dict:
    """The index-card view: everything except the full rows."""
    ep = read_episode(episode_dir)
    ep.pop("rows", None)
    return ep


# ------------------------------------------------------------------ rendering


def _verdict_style(word: str) -> tuple:
    if word in ("KEPT",):
        return _GREEN, _GREEN_BG
    if word == "REVERTED":
        return _RED, _RED_BG
    if word == "REFUSED":
        return _AMBER, _AMBER_BG
    return _GREY, _GREY_BG


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _badge(ep: dict) -> str:
    if ep.get("unreadable"):
        return (f'<span class="badge" style="color:{_RED};background:{_RED_BG}">'
                f'UNREADABLE — {_esc(str(ep["unreadable"]))}</span>')
    if ep.get("tampered"):
        if ep.get("truncated"):
            word = "TRUNCATED"
        else:
            at = ep.get("verify", {}).get("rows", "?")
            word = f"TAMPERED at row {at}"
        return (f'<span class="badge" style="color:{_RED};background:{_RED_BG}">'
                f"{word}</span>")
    n = ep.get("verify", {}).get("rows", 0)
    return (f'<span class="badge" style="color:{_GREEN};background:{_GREEN_BG}">'
            f"chain verified — {n} rows</span>")


def _verdict_banner(ep: dict, outcome: Optional[dict]) -> str:
    word = verdict_word(outcome)
    broken = bool(ep.get("unreadable") or ep.get("tampered"))
    if broken:
        word = "TRUNCATED" if ep.get("truncated") else "TAMPERED"
    color, bg = _verdict_style(word)
    if broken:
        color, bg = _RED, _RED_BG
    return (f'<span class="verdict" style="color:{color};background:{bg}">'
            f"{_esc(word)}</span>")


def _outcome_line(ep: dict, outcome: Optional[dict]) -> str:
    if ep.get("unreadable") or not isinstance(outcome, dict):
        return ""
    parts = []
    score = outcome.get("score")
    if isinstance(score, dict) and score.get("value") is not None:
        metric = _esc(str(score.get("metric") or "score"))
        parts.append(f"{metric}={_esc(_fmt(score.get('value')))}")
    if outcome.get("baseline") is not None:
        parts.append(f"baseline={_esc(_fmt(outcome.get('baseline')))}")
    refusal = outcome.get("refusal")
    if refusal:
        reason = refusal.get("reason") if isinstance(refusal, dict) else str(refusal)
        parts.append(f"refusal: {_esc(str(reason))}")
    counts = ep.get("counts", {})
    parts.append(f"{counts.get('transition', 0)} steps")
    parts.append(f"{counts.get('fact', 0)} facts")
    return " · ".join(parts)


def _kv(payload: dict) -> str:
    rows = []
    for k, v in payload.items():
        if v is None:
            continue
        rows.append(f"<tr><th>{_esc(str(k))}</th><td>{_esc(_fmt(v))}</td></tr>")
    if not rows:
        return "<em>no fields</em>"
    return "<table>" + "".join(rows) + "</table>"


def _row_html(row: dict) -> str:
    kind = _esc(str(row.get("kind") or "?"))
    payload = row.get("payload")
    body = _kv(payload) if isinstance(payload, dict) else _esc(str(payload))
    head = _esc(str(row.get("hash") or ""))[:16]
    ts = _esc(str(row.get("ts") or ""))
    return (f'<details class="row {kind}"><summary><code>{kind}</code>'
            f'<span class="ts">{ts}</span><span class="h">{head}…</span></summary>'
            f"{body}</details>")


def _facts_html(ep: dict) -> str:
    facts = [r for r in ep.get("rows", []) if r.get("kind") == "fact"]
    if not facts:
        return "<p class='muted'>no facts recorded this episode</p>"
    items = []
    for row in facts:
        p = row.get("payload")
        if isinstance(p, dict):
            rel = _esc(str(p.get("relation") or ""))
            fact = _esc(str(p.get("fact") or ""))
            conf = _esc(str(p.get("confidence") or ""))
            items.append(f"<li>{rel} — {fact}<span class='muted'> (conf {conf})</span></li>")
        else:
            items.append(f"<li>{_esc(str(p))}</li>")
    return "<ul class='facts'>" + "".join(items) + "</ul>"


def _rel_of(episode_dir: Path, root: Path) -> str:
    """The d= address of an episode dir: <problem_slug>/<episode_dir>."""
    return str(Path(episode_dir).resolve().relative_to(Path(root).resolve()).as_posix())


def _card(d: Path, root: Path) -> str:
    ep = read_episode(d)
    meta = ep.get("episode") or {}
    spec = _spec_of(meta)
    outcome = outcome_of(d)
    rel = _rel_of(d, root)
    created = _esc(str(meta.get("created") or ""))
    eid = _esc(str(meta.get("episode_id") or d.name))
    line = _outcome_line(ep, outcome)
    line_html = f"<p>{line}</p>" if line else ""
    pid = _esc(str(spec.get("problem_id") or "?"))
    domain = _esc(str(spec.get("domain") or "?"))
    facts = _facts_html(ep)
    preview = "".join(_row_html(r) for r in ep.get("rows", [])[-4:])
    links = (f"<a href='/episode?d={_esc(rel)}'>detail</a> · "
             f"<a href='/raw?d={_esc(rel)}'>raw journal</a> · "
             f"<a href='/api/verify?d={_esc(rel)}'>verify</a>")
    return (f'<div class="card"><div class="topline">{_verdict_banner(ep, outcome)}'
            f"{_badge(ep)}<span class='pill'>{domain}</span></div>"
            f"<p class='pid'>{pid}</p>"
            f"<p class='muted'>{eid} · {created}</p>{line_html}"
            f"<h4>Facts learned</h4>{facts}"
            f"<h4>Record tail</h4>{preview}"
            f"<p class='muted'>{links}</p></div>")


_CSS = ("<style>"
        "body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;"
        "max-width:960px;margin:0 auto;padding:1.4rem;color:#1f2328;line-height:1.45}"
        "h1{font-size:1.5rem;margin-bottom:.3rem}h2{font-size:1.1rem;margin:1.6rem 0 .4rem}"
        "h4{margin:.7rem 0 .2rem;font-size:.85rem;color:#57606a;text-transform:uppercase;"
        "letter-spacing:.04em}"
        "table{border-collapse:collapse;width:100%;margin:.35rem 0}"
        "th,td{border:1px solid #d0d7de;padding:.2rem .5rem;text-align:left;"
        "vertical-align:top;font-size:.85rem;word-break:break-word}"
        "th{background:#f6f8fa;width:9rem}"
        ".card{border:1px solid #d0d7de;border-radius:8px;padding:.8rem 1rem;margin:.7rem 0}"
        ".topline{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;margin-bottom:.3rem}"
        ".verdict{display:inline-block;font-weight:700;padding:.15rem .6rem;"
        "border-radius:999px;font-size:.8rem}"
        ".badge{display:inline-block;padding:.1rem .5rem;border-radius:999px;"
        "font-size:.75rem}.pill{display:inline-block;background:#f6f8fa;border-radius:999px;"
        "padding:.1rem .6rem;font-size:.75rem}.pid{font-weight:600;margin:.15rem 0}"
        ".muted{color:#656d76;font-size:.85rem}.row{margin:.3rem 0;font-size:.85rem}"
        ".ts,.h{color:#656d76;margin-left:.7rem;font-size:.75rem}"
        "code{background:#f6f8fa;padding:0 .2rem;border-radius:4px}"
        ".facts{margin:.2rem 0 .2rem 1.1rem;padding:0}.facts li{margin:.12rem 0}"
        "footer{margin-top:2.4rem;color:#656d76;font-size:.85rem;border-top:1px solid "
        "#d0d7de;padding-top:.8rem}</style>")


def _page(body: str, poll: bool, newest: str = "") -> str:
    meta = '<meta http-equiv="refresh" content="30">' if poll else ""
    data = f' data-ep="{_esc(newest)}"' if newest else ""
    js = ("<script>"
          "async function tick(){try{const r=await fetch('/api/episodes');"
          "const a=await r.json();if(!a||!a.length){return}"
          "const cur=document.body.dataset.ep||'';"
          "const top=a[0];const now=(top.dir||'')+':'+((top.counts||{}).transition||0);"
          "if(cur&&now!==cur){location.reload()}}catch(e){}}"
          "setInterval(tick,5000);</script>" if poll else "")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{PAGE_TITLE}</title>{_CSS}{meta}</head>"
            f"<body{data}>{body}{js}</body></html>")


def render_index(root: Path, limit: int = 40, public_prefixes: tuple = ()) -> str:
    eps = iter_episodes(root, public_prefixes)[:limit]
    head = (f"<h1>{PAGE_TITLE}</h1>"
            f"<p>Every solve is a hash-chained journal: each row carries the hash of "
            f"the row before it, so an edited, truncated or reordered record is "
            f"detectable rather than merely different. Kept <em>and</em> reverted runs "
            f"are shown — a run judged worse than its baseline is the loop working, "
            f"not a failure. A refused run is a refusal, never a score. Re-verify any "
            f"record with <code>/api/verify</code>, or download the raw journal "
            f"(<code>/raw</code>) and check the chain yourself.</p>")
    if not eps:
        return _page(head + "<h2>No episodes yet</h2>"
                     "<p class='muted'>The solver has not finished a run in this root."
                     "</p>", poll=True)
    groups: dict = {}
    order: list = []
    for d in eps:
        spec = _spec_of(_episode_meta(d))
        pid = str(spec.get("problem_id") or "(unknown problem)")
        if pid not in groups:
            groups[pid] = []
            order.append(pid)
        groups[pid].append(d)
    cards = []
    for pid in order:
        cards.append(f"<h2>{_esc(pid)}</h2>")
        cards.extend(_card(d, root) for d in groups[pid])
    newest = _rel_of(eps[0], root)
    return _page(head + "".join(cards), poll=True, newest=newest)


def render_episode(root: Path, rel: str, public_prefixes: tuple = ()) -> str:
    d = _resolve(root, rel)
    meta = _episode_meta(d) if d is not None else None
    if d is None or not _is_public(_spec_of(meta), public_prefixes):
        return _page(f"<h1>Not found</h1><p>No episode at <code>{_esc(rel)}</code>."
                     "</p>", poll=False)
    ep = read_episode(d)
    outcome = outcome_of(d)
    spec = _spec_of(meta)
    created = _esc(str((meta or {}).get("created") or ""))
    line = _outcome_line(ep, outcome)
    line_html = f"<p>{line}</p>" if line else ""
    rows_html = "".join(_row_html(r) for r in ep.get("rows", []))
    body = (f"<h1>{_esc(str((meta or {}).get('episode_id') or d.name))}</h1>"
            f"<p class='muted'>{created}</p>"
            f"<p class='topline'>{_verdict_banner(ep, outcome)}{_badge(ep)}"
            f"<span class='pill'>{_esc(str(spec.get('domain') or '?'))}</span></p>"
            f"{line_html}"
            f"<h2>Problem</h2>{_kv(spec) if spec else '<em>no spec recorded</em>'}"
            f"<h2>Facts learned</h2>{_facts_html(ep)}"
            f"<h2>Journal</h2><p class='muted'>{ep.get('counts', {}).get('transition', 0)} "
            f"transitions · {ep.get('verify', {}).get('rows', 0)} rows</p>{rows_html}"
            f"<p class='muted'><a href='/'>back to index</a> · "
            f"<a href='/raw?d={_esc(rel)}'>raw journal</a> · "
            f"<a href='/api/episode?d={_esc(rel)}'>JSON</a></p>")
    return _page(body, poll=False)


# ------------------------------------------------------------------ http plane


def _resolve(root: Path, rel: Optional[str]) -> Optional[Path]:
    if not rel or not isinstance(rel, str):
        return None
    if rel.startswith("/") or "\\" in rel or ".." in rel.split("/"):
        return None
    base = Path(root).resolve()
    cand = (base / rel).resolve()
    if not cand.is_relative_to(base):
        return None
    if not cand.is_dir() or not (cand / "episode.json").is_file():
        return None
    return cand


def _json(body: Any, status: int = 200):
    return (status, json.dumps(body).encode("utf-8"), "application/json")


def handle(root: Path, path: str, query: dict, public_prefixes: tuple = ()):
    """Pure request handler: (status, body bytes, content type). Never raises."""
    try:
        rel = query.get("d") if isinstance(query, dict) else None
        d = _resolve(root, rel)
        public = d is not None and _is_public(_spec_of(_episode_meta(d)),
                                              public_prefixes)
        if path == "/healthz":
            return (200, b"ok", "text/plain")
        if path == "/api/verify":
            if not public:
                return _json({"error": "no such episode"}, 404)
            ep = read_episode(d)
            return _json({"dir": rel, "ok": ep.get("verify", {}).get("ok"),
                          "rows": ep.get("verify", {}).get("rows"),
                          "head": ep.get("verify", {}).get("head"),
                          "state_matches": ep.get("state_matches"),
                          "tampered": ep.get("tampered"),
                          "truncated": ep.get("truncated"),
                          "unreadable": ep.get("unreadable")})
        if path == "/api/episode":
            if not public:
                return _json({"error": "no such episode"}, 404)
            return _json(read_episode(d))
        if path == "/api/episodes":
            return _json([episode_summary(p) for p in
                          iter_episodes(root, public_prefixes)])
        if path == "/raw":
            if not public:
                return (404, b"no such episode", "text/plain")
            try:
                raw = (d / "journal.jsonl").read_bytes()
            except OSError:
                return (404, b"journal missing", "text/plain")
            return (200, raw, "application/x-ndjson")
        if path == "/episode":
            return (200, render_episode(root, rel or "", public_prefixes)
                    .encode("utf-8"), "text/html; charset=utf-8")
        if path == "/":
            return (200, render_index(root, public_prefixes=public_prefixes)
                    .encode("utf-8"), "text/html; charset=utf-8")
        return _json({"error": f"no route {path}"}, 404)
    except Exception as exc:  # noqa: BLE001 - a defect answers 500, never a crash
        return _json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def _make_handler(root: Path, public_prefixes: tuple):
    class Handler(BaseHTTPRequestHandler):
        server_version = "awgym-watch/1"

        def do_GET(self):  # noqa: N802 - http.server API
            parsed = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            status, body, ctype = handle(root, parsed.path, query, public_prefixes)
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except OSError:
                pass

        def log_message(self, fmt, *args):  # noqa: A002 - stdlib signature
            sys.stderr.write("watch: %s\n" % (fmt % args))

    return Handler


def serve(root: Path, host: str = "127.0.0.1", port: int = 8655,
          public_prefixes: tuple = ()) -> None:
    httpd = ThreadingHTTPServer((host, port), _make_handler(root, public_prefixes))
    print(f"watching {root} on http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("watch: stopped")
    finally:
        httpd.server_close()


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m awgym.gym.watch",
                                 description="serve General Solver episode journals")
    ap.add_argument("--root", default=str(Path.home() / ".aither" / "solve"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8655)
    ap.add_argument("--public-prefixes", default="",
                    help="comma list of problem_id prefixes to show (default: all)")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                continue  # stream already usable; the widening is best-effort
    prefixes = tuple(p for p in args.public_prefixes.split(",") if p)
    serve(Path(args.root), host=args.host, port=args.port, public_prefixes=prefixes)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
