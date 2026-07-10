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


def _search_engine_up() -> bool:
    try:
        s = socket.create_connection(("localhost", 4000), timeout=2.0)
        s.close()
        return True
    except OSError:
        return False


def _typed_edge_count(graph) -> int:
    n = 0
    for (src, tgt), e in getattr(graph, "edges", {}).items():
        if e is not None and e.relation_type in (
                "is_a", "has_property", "causes", "located_in", "part_of"):
            n += 1
    return n


@pytest.mark.skipif(not _search_engine_up(), reason="local search engine not up at localhost:4000")
class TestLiveWebCLiteSmoke:
    def test_live_search_returns_results(self):
        """The live endpoint responds and returns parsed results."""
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
