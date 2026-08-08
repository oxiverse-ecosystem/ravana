"""
Live-web smoke: the REAL web-read -> C-lite path, no network stub.

Unlike the offline soak (which injects facts via the C-lite writer directly),
this test drives the actual production path a deployed agent uses:

    learn_from_web(topic)
      -> SearchEngine.search(local_only=True)   # hits localhost:4000/search?q=
      -> snippets fed into _learn_from_text()   # the branch that runs C-lite
      -> WebToGraph.learn_text() writes typed edges into engine.graph

It requires a live local search engine at http://localhost:4000/search?q=.
If that endpoint is unreachable, the test SKIPS (it is a deployment smoke,
not a unit test) rather than failing — so it stays honest in CI.

This closes the last caveat from the soak: confirming the live background
loop actually invokes _learn_from_text with real fetched text, so C-lite
fires in a real run, not just when facts are injected directly.
"""

import os
import sys
import socket

import numpy as np
import pytest

_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in [
    os.path.join(_proj_root, "ravana_ml", "src"),
    os.path.join(_proj_root, "ravana", "src"),
    os.path.join(_proj_root, "ravana-v2", "src"),
    _proj_root,
]:
    if p not in sys.path:
        sys.path.insert(0, p)

pytestmark = [pytest.mark.integration]

GLOVE_CACHE = os.path.join(_proj_root, "data", "ravana_glove_cache.npz")
SEARCH_URL = "http://localhost:4000/search?q="


def _offline_gate() -> bool:
    """True when RAVANA_OFFLINE=1 — the global web gate that makes
    SearchEngine.search() return [] for every query (no network). Under this
    gate the live-web-read path is intentionally disabled, so this is an
    HONEST skip, not a silent pass: the test verifies the REAL web-read path,
    which cannot run when web access is globally disabled. We skip with a
    true reason instead of failing on a misleading ``assert len(results) > 0``.
    """
    return os.environ.get("RAVANA_OFFLINE") == "1"


def _search_engine_up() -> bool:
    """Return True only when the LIVE web path can genuinely run.

    The live-web smoke drives the real production path (learn_from_web ->
    SearchEngine -> localhost:4000). It must SKIP — never FAIL — whenever
    that path cannot actually succeed. Two preconditions gate it:

    (a) RAVANA_OFFLINE must be off. The engine's GLOBAL offline gate
        short-circuits every search() to [] when RAVANA_OFFLINE=1, so a
        "live web" test can never observe real results offline. Running it
        offline would FAIL on `assert len(results) > 0` for a reason
        unrelated to the endpoint — so we skip. (This also keeps the full
        integration suite green under the repo's default offline conftest.)

    (b) A REAL IntentForge server must be serving results. A bare TCP-open
        check is fragile: any process that merely holds port 4000 open
        (e.g. Docker Desktop on a dev machine) passes the socket test, but
        the endpoint serves nothing (HTTP 000), so the smoke would FAIL
        instead of SKIP. We therefore verify the contract the test depends
        on: TCP reachable AND GET /search?q=water returns HTTP 2xx AND the
        JSON body parses to a non-empty `results` list with at least one
        entry carrying a usable `url` (the exact shape the SearchEngine
        intentforge parser requires). Anything short of that — a zombie
        listener, an empty body, a non-JSON response, a non-2xx status — is
        "not up" and triggers SKIP, not FAIL.
    """
    import json as _json
    import urllib.request as _ureq

    # (a) Offline mode: the live path cannot run -> skip, don't fail.
    if os.environ.get("RAVANA_OFFLINE") == "1":
        return False

    # (b-1) TCP reachability (fast path; rules out the common "nothing there").
    try:
        s = socket.create_connection(("localhost", 4000), timeout=2.0)
        s.close()
    except OSError:
        return False

    # (b-2) Require a real HTTP response that actually serves results.
    try:
        req = _ureq.Request(SEARCH_URL + "water", headers={"User-Agent": "ravana-live-web-smoke"})
        with _ureq.urlopen(req, timeout=4.0) as resp:
            if resp.status < 200 or resp.status >= 300:
                return False
            body = resp.read().decode("utf-8", errors="replace")
        try:
            data = _json.loads(body)
        except ValueError:
            return False
        results = data.get("results", []) if isinstance(data, dict) else []
        if not isinstance(results, list) or not results:
            return False
        return any(
            isinstance(r, dict) and r.get("url")
            for r in results
        )
    except OSError:
        # Connection refused / reset / HTTP 000 from a non-server listener.
        return False


def _typed_edge_count(graph) -> int:
    n = 0
    for (src, tgt), e in getattr(graph, "edges", {}).items():
        if e is not None and e.relation_type in (
                "is_a", "has_property", "causes", "located_in", "part_of"):
            n += 1
    return n


@pytest.mark.skipif(
    (not _search_engine_up()) or _offline_gate(),
    reason="local search engine not up at localhost:4000, or RAVANA_OFFLINE=1 (web-read globally disabled — live-web path cannot run)",
)
class TestLiveWebCLiteSmoke:
    def test_live_search_returns_results(self):
        """The live endpoint responds and returns parsed results."""
        # Defensive runtime guard: the class-level skipif is evaluated at
        # collection time, but under xdist the controller's view of the port
        # can race with a worker. Re-check here so the test never fails when
        # the local search engine is genuinely down (it is, in CI).
        if not _search_engine_up():
            pytest.skip("local search engine not up at localhost:4000")
        if _offline_gate():
            pytest.skip("RAVANA_OFFLINE=1 — live web-read path disabled")
        from ravana.chat.engine import CognitiveChatEngine
        eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True)
        results = eng.search_engine.search("water", max_results=3, local_only=True)
        assert len(results) > 0, "live localhost:4000/search must return results"
        assert any(r.get("content") or r.get("title") for r in results), \
            "results must carry content/title for C-lite to extract facts"

    def test_live_web_read_writes_c_lite_facts(self):
        """The REAL path: learn_from_web -> live search -> _learn_from_text
        -> C-lite writes typed edges into engine.graph."""
        from ravana.chat.engine import CognitiveChatEngine
        if not _search_engine_up():
            pytest.skip("local search engine not up at localhost:4000")
        if _offline_gate():
            pytest.skip("RAVANA_OFFLINE=1 — live web-read path disabled")
        eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True)
        # load glove so the run mirrors production (and N2 would work too)
        if os.path.exists(GLOVE_CACHE):
            d = np.load(GLOVE_CACHE, allow_pickle=True)
            eng._glove_vecs = {str(w).lower(): v for w, v in zip(d["words"].tolist(), d["vecs"])}
            eng._glove_proj = d["proj"].astype(np.float32)
            eng._glove_dim = int(d["proj"].shape[1])

        # Force the LOCAL branch (the one that hits localhost:4000 and feeds
        # snippets into _learn_from_text, where C-lite now lives).
        eng._network_available = False

        edges_before = _typed_edge_count(eng.graph)
        summary, _ = eng.learn_from_web("water", max_results=3, train_decoder=False)
        edges_after = _typed_edge_count(eng.graph)

        # C-lite must have written facts via the REAL web-read path
        assert eng._get_web_to_graph() is not None
        assert eng._get_web_to_graph().fact_count() > 0, \
            "live web-read must write C-lite facts into the graph"
        assert edges_after > edges_before, \
            "live web-read must add typed edges to the graph"
        print(f"\n  [smoke] learn_from_web('water') -> {summary}")
        print(f"  [smoke] C-lite facts written: {eng._get_web_to_graph().fact_count()}")
        print(f"  [smoke] typed edges: {edges_before} -> {edges_after}")
