#!/usr/bin/env python3
"""Capture ONE live web response per DISTINCT query used by the front harness.

This is the capture half of the frozen-snapshot web arm. It runs the real
``measure_fronts_web`` probes against the live provider, but monkeypatches the
single provider boundary (``SearchEngine.search``) with a cache-on-first-use
wrapper: the first occurrence of a query performs one live lookup, every later
occurrence is served from the in-memory cache. The result is a deterministic
JSON fixture bundle (query -> hits) that the frozen harness replays offline.

What is captured:
  * ``queries``      -- normalized query -> live search hits (title/url/content/...)
  * ``kb_describes`` -- normalized concept -> live Wikipedia summary extract
                        (a secondary retrieval path used by kb_describe; freezing
                        it too is what lets the frozen harness make ZERO network
                        calls while still executing the web-enabled code paths)

Run:
    uv run python scripts/capture_web_fixtures.py

The committed instrument, its data files and ravana/src are never edited: the
harness module's DATA_PATH/SUMMARY_PATH/GATE_LOG_PATH are redirected to
``scratch/front_probes_web_capture*`` before main() is called.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, PROJ, os.path.join(PROJ, "ravana", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FIXTURES_DIR = os.path.join(PROJ, "tests", "fixtures", "web_snapshots")
FIXTURES_PATH = os.path.join(FIXTURES_DIR, "web_fixtures.json")
CAPTURED_AT = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
PROVIDER = "intentforge(localhost:4000) -> local_api -> duckduckgo -> oxiverse"

os.makedirs(FIXTURES_DIR, exist_ok=True)

_LOCK = threading.RLock()
_FIX = {
    "schema": "ravana-web-fixtures-v1",
    "captured_at": CAPTURED_AT,
    "provider": PROVIDER,
    "captured_by": "scripts/capture_web_fixtures.py",
    "queries": {},
    "kb_describes": {},
    "tool_web_search": {},
}
# Resumable: keep already-captured hits so a second pass only queries new inputs.
if os.path.exists(FIXTURES_PATH):
    with open(FIXTURES_PATH, encoding="utf-8") as _fh:
        _old = json.load(_fh)
    if _old.get("schema") == _FIX["schema"]:
        _FIX["queries"].update(_old.get("queries", {}))
        _FIX["kb_describes"].update(_old.get("kb_describes", {}))
        _FIX["tool_web_search"].update(_old.get("tool_web_search", {}))
        print(f"[capture] resumed with {len(_FIX['queries'])} existing query fixtures")
_QUERY_CACHE = {k: v.get("hits", []) for k, v in _FIX["queries"].items()}
_KB_CACHE = {k: v.get("extract") for k, v in _FIX["kb_describes"].items()}
_TOOL_CACHE = {k: v.get("response") for k, v in _FIX["tool_web_search"].items()}
_N_LIVE = 0


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _flush() -> None:
    """Atomic, deterministic write (sorted keys) so the fixture is reproducible."""
    tmp = FIXTURES_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_FIX, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, FIXTURES_PATH)


def _install_provider_stubs() -> tuple:
    """Wrap the two live retrieval boundaries, caching each distinct input once."""
    from ravana.web.learner import SearchEngine
    from ravana.chat.engine import CognitiveChatEngine
    from ravana.agent import tool_registry

    real_search = SearchEngine.search
    real_kb = CognitiveChatEngine.kb_describe
    real_tool = tool_registry._web_search_via_intentforge

    def capture_search(self, query, max_results=10, local_only=False):
        global _N_LIVE
        key = _norm(query)
        with _LOCK:
            if key in _QUERY_CACHE:
                return _QUERY_CACHE[key]
        # First occurrence: ONE live provider call.
        hits = real_search(self, query, max_results=max_results,
                           local_only=local_only)
        if not isinstance(hits, list):
            hits = list(hits or [])
        with _LOCK:
            _QUERY_CACHE[key] = hits
            if key not in _FIX["queries"]:
                _FIX["queries"][key] = {
                    "query": query,
                    "hits": hits,
                    "max_results": int(max_results),
                    "local_only": bool(local_only),
                }
                _N_LIVE += 1
                print(f"[capture] query {_N_LIVE:4d} {key!r} -> {len(hits)} hits",
                      flush=True)
                _flush()
        return hits

    def capture_kb(self, concept, timeout=6.0):
        key = _norm(concept)
        with _LOCK:
            if key in _KB_CACHE:
                return _KB_CACHE[key]
        text = real_kb(self, concept, timeout=timeout)
        with _LOCK:
            _KB_CACHE[key] = text
            if key not in _FIX["kb_describes"]:
                _FIX["kb_describes"][key] = {"concept": concept, "extract": text}
                print(f"[capture] kb_describe {key!r} -> "
                      f"{'hit' if text else 'miss'}", flush=True)
                _flush()
        return text

    def capture_tool(query):
        key = _norm(query)
        with _LOCK:
            if key in _TOOL_CACHE:
                return _TOOL_CACHE[key]
        text = real_tool(query)
        with _LOCK:
            _TOOL_CACHE[key] = text
            if key not in _FIX["tool_web_search"]:
                _FIX["tool_web_search"][key] = {"query": query, "response": text}
                print(f"[capture] tool_web_search {key!r} -> {len(text or '')} chars",
                      flush=True)
                _flush()
        return text

    SearchEngine.search = capture_search
    CognitiveChatEngine.kb_describe = capture_kb
    tool_registry._web_search_via_intentforge = capture_tool
    return real_search, real_kb, real_tool


def top_up(missing_path):
    """Capture provider hits for queries a frozen replay requested but the
    live capture pass never reached (the live 12 s answer deadline truncates
    variant loops early). One live call per distinct missing query."""
    from ravana.web.learner import SearchEngine

    with open(missing_path, encoding="utf-8") as fh:
        missing = json.load(fh)
    se = SearchEngine()
    added = 0
    for key in sorted(missing):
        if key in _FIX["queries"]:
            continue
        hits = se.search(key, max_results=6, local_only=False)
        if not isinstance(hits, list):
            hits = []
        _FIX["queries"][key] = {"query": key, "hits": hits, "max_results": 6,
                                "local_only": False, "top_up": True}
        added += 1
        print(f"[capture] top-up {key!r} -> {len(hits)} hits", flush=True)
        _flush()
    print(f"[capture] top-up added={added} total_queries={len(_FIX['queries'])}")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if "--top-up" in argv:
        return top_up(argv[argv.index("--top-up") + 1])
    print(f"[capture] fixtures -> {FIXTURES_PATH}")
    print(f"[capture] captured_at={CAPTURED_AT}\n[capture] provider={PROVIDER}")
    _flush()
    _install_provider_stubs()

    import measure_fronts_web as mfw
    # Never touch the committed instrument's data files.
    mfw.DATA_PATH = os.path.join(PROJ, "scratch", "front_probes_web_capture.jsonl")
    mfw.SUMMARY_PATH = os.path.join(PROJ, "scratch", "front_probes_web_capture_summary.json")
    mfw.GATE_LOG_PATH = os.path.join(PROJ, "scratch", "front_probes_web_capture_gates.jsonl")

    started = time.time()
    rc = mfw.main([])
    _flush()
    print(f"\n[capture] complete rc={rc} distinct_queries={len(_FIX['queries'])} "
          f"kb_describes={len(_FIX['kb_describes'])} "
          f"tool_queries={len(_FIX['tool_web_search'])} "
          f"elapsed_s={time.time()-started:.1f}")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
