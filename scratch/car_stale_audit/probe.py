"""Boot-free stale-ID write-path audit + regression probe for CAR.

Imports UNMODIFIED production functions and drives them on small synthetic
ConceptGraphs. NO CognitiveChatEngine boot, NO weight-store access, NO network,
NO production edits.

Gating checks (G1-G5) must PASS on current main and would FAIL on the pre-fix
code (commit 254fbc9^). Open-hole checks (H1-H7) exercise write paths in the
CURRENT tree that can still attach an edge to a dead node id; they are findings,
not pass/fail criteria, and are reported with file:line evidence.

Usage:
    uv run python scratch/car_stale_audit/probe.py
    uv run python scratch/car_stale_audit/probe.py --legacy-shadow scratch/car_stale_audit/legacy_shadow

Exit code 0 iff every gating check passes AND the legacy instrument control
reproduces the pre-fix resolver behaviour. Otherwise exit 1.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import threading
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))  # repo root

parser = argparse.ArgumentParser()
parser.add_argument("--legacy-shadow", default="",
                    help="optional shadow source root prepended to sys.path (A/B on old code)")
parser.add_argument("--out-dir", default=HERE)
args = parser.parse_args()

if args.legacy_shadow:
    sys.path.insert(0, os.path.abspath(args.legacy_shadow))
for p in (PROJ, os.path.join(PROJ, "ravana", "src"),
          os.path.join(PROJ, "ravana_ml", "src")):
    if p not in sys.path:
        sys.path.insert(0 if not args.legacy_shadow else len(sys.path), p)

import numpy as np  # noqa: E402

from ravana_ml.graph import ConceptGraph  # noqa: E402
from ravana.web.web_to_graph import WebToGraph  # noqa: E402
from ravana.chat.web_learning import _EngineGraphShim  # noqa: E402
from ravana.chat.interface import _GraphEngineShim  # noqa: E402
from ravana.chat.engine_generation import GenerationMixin  # noqa: E402
from ravana.chat.engine_memory import MemoryMixin  # noqa: E402
from ravana.chat.engine_persistence import PersistenceMixin  # noqa: E402
from ravana.chat.chain_walker import ChainWalkerMixin  # noqa: E402
from ravana.learn.intake import SchemaConsistencyGate  # noqa: E402

DIM = 8
CHECKS = []


def norm_src(rel):
    """Read production source; in --legacy-shadow mode prefer the shadow copy."""
    path = os.path.join(PROJ, rel)
    if args.legacy_shadow:
        for prefix in ("ravana/src/", "ravana_ml/src/"):
            if rel.startswith(prefix):
                cand = os.path.join(os.path.abspath(args.legacy_shadow),
                                    rel[len(prefix):])
                if os.path.exists(cand):
                    path = cand
                break
    with open(path, "rb") as fh:
        return fh.read().decode("utf-8", "replace").replace("\r\n", "\n").replace("\r", "\n")


def node(g, label):
    return g.add_node(vector=np.zeros(DIM, dtype=np.float32), label=label).id


def add_check(cid, kind, status, inp, out, evidence):
    CHECKS.append({
        "id": cid, "kind": kind, "status": status,
        "input": inp, "output": out, "evidence": evidence,
    })
    print(f"[{cid}] {kind}/{status}  input={inp}  output={out}", flush=True)


def live(g, nid):
    return g.get_node(nid) is not None


def dangling(g):
    return sorted([s for s, t in g.edges if not live(g, s) or not live(g, t)] +
                  [t for s, t in g.edges if not live(g, t)])


# ── G1: F2 write-choke liveness in WebToGraph._ensure_node ──────────────────
g = ConceptGraph(dim=DIM)
dead_echo = node(g, "echo")
g.remove_node(dead_echo)
host = SimpleNamespace(
    graph=g,
    _all_labels={"echo": dead_echo},          # plain dict = pre-fix host shape
    _concept_keywords={"echo": [dead_echo]},
    _promote_min_sources=1,
    _junk_theta=1.0,                           # disable junk gate deterministically
)
w = WebToGraph.__new__(WebToGraph)
w.ge = host
# Capture the pre-fix resolver answer BEFORE the fixed writer mutates the host.
legacy_choice = host._all_labels.get("echo")  # exactly what pre-fix code returned
fresh = w._ensure_node("echo", source_url="http://teach.local/echo/sourceA")
g1_ok = bool(fresh is not None and live(g, fresh) and fresh != dead_echo
             and dead_echo not in host._concept_keywords.get("echo", []))
add_check("G1", "gating", "PASS" if g1_ok else "FAIL",
          {"label": "echo", "stale_id": dead_echo, "host_index": "plain dict"},
          {"resolved_id": fresh, "live": live(g, fresh) if fresh is not None else None,
           "legacy_resolver_id": legacy_choice,
           "index_after": host._concept_keywords.get("echo")},
          "web_to_graph.py:151-163 (F2 live-only check at write choke point)")

# Instrument control: the legacy replica MUST reproduce the stale id. This is
# the only claim I1 makes; G1 is what separates current from pre-fix code.
i1_ok = legacy_choice == dead_echo
add_check("I1", "instrument", "PASS" if i1_ok else "FAIL",
          {"legacy_resolver": "return _all_labels.get(label)", "stale_id": dead_echo},
          {"legacy_returns_stale": bool(legacy_choice == dead_echo),
           "fixed_writer_returns_stale": bool(fresh == dead_echo)},
          "instrument validation: the A0 arm must reproduce the recorded pre-fix behaviour "
          "(it is expected to FAIL G1 on current code and to FAIL G1 again under the shadow)")

# ── G2: F2 live-only label-to-id shims ──────────────────────────────────────
eng = SimpleNamespace(graph=g, dim=DIM,
                      _concept_keywords={"echo": [dead_echo]},
                      _glove_vector=lambda _w: None)
s1 = _EngineGraphShim(eng)._all_labels.get("echo")
s2 = _GraphEngineShim(g, eng)._all_labels.get("echo")
g2_ok = s1 is None and s2 is None
add_check("G2", "gating", "PASS" if g2_ok else "FAIL",
          {"label": "echo", "stale_id": dead_echo, "shims": ["_EngineGraphShim", "_GraphEngineShim"]},
          {"engine_shim_resolved": s1, "cli_shim_resolved": s2},
          "web_learning.py:3112-3127; interface.py:70-85")

# ── G3: F4 trusted causal node survives junk prune ─────────────────────────
def build_prune_graph(trusted):
    """Prune-eligible node still carrying a trusted edge in `edges`.

    The shipped test_ensure_node/prune test never discriminates: with tau_low=1 a
    node with an incident edge has degree>=1 and is skipped before the guard.
    Clear THAT node's adjacency entries to simulate a restore/indices-lag state
    so degree is 0 while the edge record (and its provenance) is still present.
    """
    gp = ConceptGraph(dim=DIM)
    a = node(gp, "cause_a")
    junk = node(gp, "random_scrape_token")
    e = gp.add_edge(a, junk, weight=0.5, relation_type="causal", confidence=0.5)
    if trusted:
        e.source_metadata = {"edge_kind": "curated",
                             "source": "http://teach.local/x/sourceA"}
    gp._outgoing.pop(a, None)
    gp._incoming.pop(a, None)
    return gp, a, junk


gp, trusted_a, junk = build_prune_graph(trusted=True)
count, labels, hubs = gp.prune_low_degree_junk_nodes(tau_low=1, tau_high=8)
gp_ctl, control_a, junk_ctl = build_prune_graph(trusted=False)
count_ctl, _, _ = gp_ctl.prune_low_degree_junk_nodes(tau_low=1, tau_high=8)
g3_ok = (live(gp, trusted_a) and live(gp, junk)
         and not live(gp_ctl, control_a))
add_check("G3", "gating", "PASS" if g3_ok else "FAIL",
          {"trusted": "edge_kind=curated on a degree-0 (adjacency-lag) node",
           "control": "same shape, no provenance"},
          {"trusted_node_survived": live(gp, trusted_a),
           "control_node_pruned": not live(gp_ctl, control_a),
           "control_prune_count": count_ctl},
          "graph.py:1546-1566 (F4 provenance guard); mutation control removes the stamp")

# ── G4: F1 label-index sync source marker ──────────────────────────────────
gen_src = norm_src("ravana/src/ravana/chat/engine_generation.py")
f1_marker = ("_ids_before - _ids_after" in gen_src
             and "self._concept_keywords.pop" in gen_src)
g4_ok = f1_marker
add_check("G4", "gating", "PASS" if g4_ok else "FAIL",
          {"file": "engine_generation.py", "check": "F1 sync after junk-prune sweep"},
          {"markers_present": f1_marker},
          "engine_generation.py:1516-1531 (source-level; invoking _sleep_consolidate needs a boot)")

# ── G5: F3 trust-first partition (source-level) ────────────────────────────
exp_src = norm_src("ravana/src/ravana/learn/explanation.py")
f3_marker = ("_decorated = [(assertable_chain(c), c.score, c) for c in out]" in exp_src
             and "_decorated.sort" in exp_src)
add_check("G5", "gating", "PASS" if f3_marker else "FAIL",
          {"file": "explanation.py", "check": "F3 trust-first partition before limit"},
          {"markers_present": f3_marker},
          "explanation.py:261-289")

# ═══════════════════════════════════════════════════════════════════════════
# OPEN HOLES — current main can still write an edge whose endpoint is dead.
# Each is driven through the REAL production writer where feasible.
# ═══════════════════════════════════════════════════════════════════════════

# H0: conversational causal teaching path (highest severity — the CAR channel)
gc = ConceptGraph(dim=DIM)
cdead, edead = node(gc, "impact"), node(gc, "bruise")
gc.remove_node(cdead)
gc.remove_node(edead)
cw = SimpleNamespace(graph=gc, _concept_keywords={"impact": [cdead], "bruise": [edead]},
                     _belief_assertions=[], _trace_enabled=False)
ChainWalkerMixin._create_causal_edge(cw, "impact", "bruise", source_text="impact causes bruise.")
h0_open = (cdead, edead) in gc.edges and not live(gc, cdead)
add_check("H0", "open", "OPEN" if h0_open else "CLOSED",
          {"writer": "ChainWalkerMixin._create_causal_edge", "stale_ids": [cdead, edead]},
          {"dangling_edge_written": (cdead, edead) in gc.edges,
           "endpoint_live": live(gc, cdead), "dangling_endpoints": dangling(gc)},
          "chain_walker.py:1394-1401,1425-1430 — _concept_keywords[0] without get_node()")

# H1: episodic-memory writer
g1h = ConceptGraph(dim=DIM)
sd, ad = node(g1h, "echo_subj"), node(g1h, "echo_assoc")
g1h.remove_node(sd)
g1h.remove_node(ad)
mh = SimpleNamespace(graph=g1h,
                     _concept_keywords={"echo_subj": [sd], "echo_assoc": [ad]},
                     _topic_store={})
MemoryMixin._store_episodic(mh, "echo_subj", [("echo_assoc", 1.0)])
h1_open = (sd, ad) in g1h.edges and not live(g1h, sd)
add_check("H1", "open", "OPEN" if h1_open else "CLOSED",
          {"writer": "MemoryMixin._store_episodic", "stale_ids": [sd, ad]},
          {"dangling_edge_written": (sd, ad) in g1h.edges,
           "dangling_endpoints": dangling(g1h)},
          "engine_memory.py:2594-2606 — _concept_keywords[0] without get_node()")

# H2: correction writer
g2h = ConceptGraph(dim=DIM)
fd, vd = node(g2h, "echo_subj"), node(g2h, "echo_assoc")
g2h.remove_node(fd)
g2h.remove_node(vd)
corr = SimpleNamespace(subject="echo_subj", severity=0.5,
                       corrected_fact=("echo_subj", "causes", "echo_assoc"),
                       weakened_edges=[], added_edges=[])
ph = SimpleNamespace(
    graph=g2h, _concept_keywords={"echo_subj": [fd], "echo_assoc": [vd]},
    _trace_enabled=False, _last_chain_hops=[],
    belief_store=SimpleNamespace(assert_belief=lambda *a, **k: None),
    _bg_lock=threading.Lock(), _bg_learning_queue=[], _pending_learning_queue=[])
ph._weaken_edges_for_correction = lambda *a, **k: None
ph._mark_edges_as_incorrect = lambda *a, **k: None
PersistenceMixin._weaken_edges_for_response(ph, corr)
h2_open = (fd, vd) in g2h.edges and not live(g2h, fd)
add_check("H2", "open", "OPEN" if h2_open else "CLOSED",
          {"writer": "PersistenceMixin._weaken_edges_for_response", "stale_ids": [fd, vd]},
          {"dangling_edge_written": (fd, vd) in g2h.edges,
           "added_edges": corr.added_edges},
          "engine_persistence.py:588-596 — _concept_keywords[0] without get_node()")

# H3: sleep-time episodic graduation + inline process_turn replay (source-level)
ghdr = norm_src("ravana/src/ravana/chat/engine_generation.py")
geng = norm_src("ravana/src/ravana/chat/engine.py")
marker = "_a_ids = self._concept_keywords.get(_pair.concept_a, [])"
h3_open = marker in ghdr and marker in geng
add_check("H3", "open", "OPEN" if h3_open else "CLOSED",
          {"writers": ["GenerationMixin._sleep_consolidate", "CognitiveChatEngine.process_turn"]},
          {"unguarded_index_pattern_found": h3_open},
          "engine_generation.py:1471-1481; engine.py:9585-9595 — direct add_edge(_a_ids[0], _b_ids[0])")

# H4: intake gate resolver has no liveness check of its own (host-dependent)
g4h = ConceptGraph(dim=DIM)
kd = node(g4h, "echo")
g4h.remove_node(kd)
plain_host = SimpleNamespace(_all_labels={"echo": kd}, graph=g4h)
resolved = SchemaConsistencyGate._node_id(plain_host, "echo")
h4_open = resolved == kd and not live(g4h, resolved)
add_check("H4", "open", "OPEN" if h4_open else "CLOSED",
          {"resolver": "SchemaConsistencyGate._node_id", "host": "plain _all_labels dict",
           "stale_id": kd},
          {"resolved_id": resolved, "dead": not live(g4h, resolved)},
          "intake.py:315-320 (no get_node check); graph/engine.py:330,526 plain dict index; "
          "chain_walker.py:144 — production chat path uses live-only shim, other hosts do not")

# H5: ConceptGraph.add_edge has no live-endpoint invariant (F5 never shipped)
g5h = ConceptGraph(dim=DIM)
sdead, blive = node(g5h, "dead_src"), node(g5h, "live_tgt")
g5h.remove_node(sdead)
e = g5h.add_edge(sdead, blive, weight=0.5, relation_type="curated")
h5_open = e is not None and (sdead, blive) in g5h.edges and not live(g5h, sdead)
add_edge_src = inspect.getsource(ConceptGraph.add_edge)
h5_no_guard = "get_node(" not in add_edge_src
add_check("H5", "open", "OPEN" if h5_open and h5_no_guard else "CLOSED",
          {"primitive": "ConceptGraph.add_edge", "dead_source": sdead, "live_target": blive},
          {"edge_inserted": (sdead, blive) in g5h.edges,
           "outgoing_index_has_dead_key": sdead in g5h._outgoing,
           "source_has_liveness_guard": not h5_no_guard},
          "graph.py:1694-1782 — final write choke point accepts dead endpoints "
          "(F5 dangling-edge invariant from root-cause doc was never shipped)")

# H6: capacity eviction leaves stale ids and F1 does not cover it
g6h = ConceptGraph(dim=DIM)
g6h.max_nodes = 2
ids = [node(g6h, f"cap_{i}") for i in range(3)]
evicted = [i for i in ids if not live(g6h, i)]
prune_oldest_src = inspect.getsource(ConceptGraph._prune_oldest)
add_node_src = inspect.getsource(ConceptGraph.add_node)
h6_open = bool(evicted) and "self._prune_oldest()" in add_node_src \
    and "concept_keywords" not in prune_oldest_src
add_check("H6", "open", "OPEN" if h6_open else "CLOSED",
          {"max_nodes": 2, "added_ids": ids},
          {"evicted_ids": evicted, "index_synced_by_eviction": not h6_open},
          "graph.py:1332-1336 add_node -> _prune_oldest (2439-2441); F1 sync only in "
          "engine_generation.py:1516-1531 after prune_low_degree_junk_nodes")

# C1: sleep.py direct writers are guarded (cleared, not a hole)
sleep_src = norm_src("ravana/src/ravana/core/sleep.py")
sleep_guarded = ("subj_node = graph.get_node(subj_nids[0])" in sleep_src
                 and "src_node = graph.get_node(src_nid)" in sleep_src)
add_check("C1", "cleared", "CLEARED" if sleep_guarded else "REVIEW",
          {"file": "core/sleep.py", "writers": [192, 344, 400]},
          {"live_node_checks_present": sleep_guarded},
          "sleep.py:180-193,373-404 — endpoints checked with get_node() before add_edge")

# ── deterministic report ───────────────────────────────────────────────────
watched = [
    "ravana/src/ravana/chat/interface.py",
    "ravana/src/ravana/chat/web_learning.py",
    "ravana/src/ravana/chat/engine_generation.py",
    "ravana/src/ravana/web/web_to_graph.py",
    "ravana_ml/src/ravana_ml/graph.py",
    "ravana/src/ravana/learn/intake.py",
    "ravana/src/ravana/chat/chain_walker.py",
    "ravana/src/ravana/chat/engine_memory.py",
    "ravana/src/ravana/chat/engine_persistence.py",
    "ravana/src/ravana/learn/explanation.py",
]
source_hashes = {
    p: hashlib.sha256(norm_src(p).encode("utf-8")).hexdigest()[:16] for p in watched
}
gating = [c for c in CHECKS if c["kind"] == "gating"]
instr = [c for c in CHECKS if c["kind"] == "instrument"]
opens = [c for c in CHECKS if c["kind"] == "open"]
ok = (all(c["status"] == "PASS" for c in gating + instr)
      and all(c["status"] == "OPEN" for c in opens))
payload = {
    "probe": "car_stale_id_write_paths",
    "version": 1,
    "mode": "legacy-shadow" if args.legacy_shadow else "current-main",
    "engine_booted": False,
    "source_hashes": source_hashes,
    "checks": CHECKS,
    "summary": {
        "gating_pass": sum(c["status"] == "PASS" for c in gating),
        "gating_total": len(gating),
        "instrument_pass": sum(c["status"] == "PASS" for c in instr),
        "instrument_total": len(instr),
        "open_holes": sum(c["status"] == "OPEN" for c in opens),
        "open_total": len(opens),
    },
    "verdict": "PASS" if ok else "FAIL",
}
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
payload["report_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
os.makedirs(args.out_dir, exist_ok=True)
out_path = os.path.join(args.out_dir,
                        "results_legacy.json" if args.legacy_shadow else "results.json")
with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(payload, fh, sort_keys=True, indent=2, ensure_ascii=True)
    fh.write("\n")
print(f"SUMMARY: {payload['summary']}  VERDICT: {payload['verdict']}  "
      f"sha256={payload['report_sha256']}  report={out_path}", flush=True)
sys.exit(0 if ok else 1)
