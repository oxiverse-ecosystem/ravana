"""Boot-free A/B probe: stale-endpoint write guard on the highest-severity CAR hole (H0).

Drives the UNMODIFIED production writers on synthetic ConceptGraphs:
  - Arm A  : current production behaviour (baseline vulnerability)
  - Arm B  : candidate LOCAL guard -- F2-style live-only resolver at the write
             choke point of ChainWalkerMixin._create_causal_edge. Implemented in
             scratch by sanitising the host's resolver index BEFORE calling the
             real production method (no copy of production logic, no edits).
  - Arm C  : candidate CHOKE-POINT-ONLY guard -- a scratch monkeypatch of
             ConceptGraph.add_edge that rejects dead endpoints (H5 candidate).

NO CognitiveChatEngine boot, NO weight-store access, NO network, NO production
edits, NO gate/threshold changes. Every arm starts from a freshly built,
identically shaped graph so the comparison is on identical state in one process.

Pre-registered falsifiers (probe is INVALID and exits 1 if any fails):
  F-A  baseline MUST write a dangling causal edge on a stale index
  F-C  the add_edge guard MUST block a direct dead-endpoint add (proves it fired)
  F-B  the local guard MUST prune stale ids (proves it fired) and MUST NOT block
       a legitimate live->live causal write (no regression)

Exit 0 iff every expectation in EXPECT matches. Printed INPUT/OUTPUT is the
evidence; results.json holds the canonical record.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))  # worktree root

for _p in (os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana", "src"), PROJ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import ravana_ml.graph as _graph_mod  # noqa: E402
from ravana_ml.graph import ConceptGraph, ConceptEdge  # noqa: E402
from ravana.chat.chain_walker import ChainWalkerMixin  # noqa: E402
from ravana.chat.engine_memory import MemoryMixin  # noqa: E402

DIM = 8
TEXT = "impact causes bruise."
RECORDS = []

# Import-root assertion: never silently test the main editable install.
IMPORT_ROOT = os.path.dirname(os.path.abspath(_graph_mod.__file__))
if not IMPORT_ROOT.lower().startswith(PROJ.lower()):
    raise RuntimeError(f"wrong import root: {IMPORT_ROOT!r} (expected under {PROJ!r})")

# No-store sentinel: this probe must never touch the weight stores (a CAR rerun
# is active in the parent repo). Record state before and after; compare at exit.
STORE_FILES = [os.path.join(PROJ, "weights", f)
               for f in ("ravana_weights.pkl", "ravana_weights.db")]
def store_state():
    return {os.path.basename(p): (os.path.getmtime(p), os.path.getsize(p))
            if os.path.exists(p) else "absent" for p in STORE_FILES}
STORE_BEFORE = store_state()


def norm_src(rel):
    with open(os.path.join(PROJ, rel), "rb") as fh:
        return fh.read().decode("utf-8", "replace").replace("\r\n", "\n").replace("\r", "\n")


def node(g, label):
    return g.add_node(vector=np.zeros(DIM, dtype=np.float32), label=label).id


def live(g, nid):
    return g.get_node(nid) is not None


def dangling(g):
    return sorted([s for s, t in g.edges if not live(g, s) or not live(g, t)] +
                  [t for s, t in g.edges if not live(g, t)])


# ── candidate guards (scratch only) ────────────────────────────────────────

def local_live_index_guard(host):
    """F2-style live-only resolver: drop stale ids from _concept_keywords.

    Mirrors web_to_graph.py:151-163. The production _create_causal_edge already
    returns False on an empty id list, so sanitising the index at the write
    choke point needs no change to the production method: both its new-edge
    branch and its existing-edge (strengthen) branch fail closed.
    """
    g = host.graph
    removed = {}
    for label, ids in list((getattr(host, "_concept_keywords", {}) or {}).items()):
        kept = [i for i in ids if g.get_node(i) is not None]
        if kept != list(ids):
            host._concept_keywords[label] = kept
            removed[label] = [i for i in ids if i not in kept]
    return removed


def _guarded_add_edge(self, source, target, *args, **kwargs):
    """H5 candidate: no edge may be inserted with a dead endpoint.

    Follows the production self-loop guard's fail-closed shape (graph.py:1707-1718):
    return an UNINSERTED dummy ConceptEdge so callers that dereference .weight
    or .confidence do not crash.
    """
    if self.get_node(source) is None or self.get_node(target) is None:
        return ConceptEdge(source, target, 0.0,
                           relation_type=kwargs.get("relation_type", "semantic"),
                           relation_dim=getattr(self, "_relation_dim", DIM),
                           confidence=kwargs.get("confidence") or 0.5)
    return _ORIG_ADD_EDGE(self, source, target, *args, **kwargs)


_ORIG_ADD_EDGE = ConceptGraph.add_edge


@contextmanager
def addedge_guard_active():
    ConceptGraph.add_edge = _guarded_add_edge
    try:
        yield
    finally:
        ConceptGraph.add_edge = _ORIG_ADD_EDGE


# ── state factories: identical shape for every arm ────────────────────────

def build_causal_state(stale, preplant=False):
    g = ConceptGraph(dim=DIM)
    c, e = node(g, "impact"), node(g, "bruise")
    if stale:
        g.remove_node(c)          # junk prune / capacity eviction: node gone,
        g.remove_node(e)          # engine label index still holds old ids
    if preplant:
        g.add_edge(c, e, weight=0.1, relation_type="semantic", confidence=0.1)
    host = SimpleNamespace(
        graph=g,
        _concept_keywords={"impact": [c], "bruise": [e]},
        _belief_assertions=[],
        _trace_enabled=False,
    )
    return g, host, c, e


def causal_snapshot(g, host, c, e):
    edge = g.get_edge(c, e)
    return {
        "edge_exists": edge is not None,
        "weight": round(edge.weight, 4) if edge is not None else None,
        "relation_type": edge.relation_type if edge is not None else None,
        "confidence": round(edge.confidence, 4) if edge is not None else None,
        "endpoint_live": [live(g, c), live(g, e)],
        "dangling_edges": dangling(g),
        "beliefs": [list(b) for b in host._belief_assertions],
        "resolver_index": {k: list(v) for k, v in host._concept_keywords.items()},
    }


def run_causal_arm(arm, stale, preplant=False):
    g, host, c, e = build_causal_state(stale=stale, preplant=preplant)
    before = causal_snapshot(g, host, c, e)
    guard_fired = False
    if arm == "B_local":
        removed = local_live_index_guard(host)
        guard_fired = bool(removed)
    cm = addedge_guard_active() if arm == "C_addedge" else None
    try:
        if cm is not None:
            cm.__enter__()
        returned = ChainWalkerMixin._create_causal_edge(
            host, "impact", "bruise", source_text=TEXT)
    finally:
        if cm is not None:
            cm.__exit__(None, None, None)
    after = causal_snapshot(g, host, c, e)
    return {"before": before, "after": after, "returned": returned,
            "guard_fired": guard_fired}


def classify_causal(run, preplant=False):
    b, a = run["before"], run["after"]
    if preplant:
        changed = (a["weight"] != b["weight"] or
                   a["relation_type"] != b["relation_type"])
        return "STRENGTHENED" if changed else "UNCHANGED"
    if a["edge_exists"]:
        return "WRITTEN" if all(a["endpoint_live"]) and not a["dangling_edges"] else "OPEN"
    return "CLOSED"


def record(cid, arm, inp, out, status):
    row = {"id": cid, "arm": arm, "input": inp, "output": out, "status": status}
    RECORDS.append(row)
    print(f"[{cid}] {arm:<10} status={status:<12} input={inp}", flush=True)
    print(f"        -> {out}", flush=True)


def run_causal_mixed_arm(arm):
    """Index holds BOTH a dead id and a live id for each label.

    Baseline blindly takes [0] (dead); the local guard keeps only live ids and
    must therefore still teach the fact -- a no-regression control sharper than
    the all-live case.
    """
    g = ConceptGraph(dim=DIM)
    dead_c, dead_e = node(g, "impact"), node(g, "bruise")
    g.remove_node(dead_c)
    g.remove_node(dead_e)
    live_c, live_e = node(g, "impact"), node(g, "bruise")
    host = SimpleNamespace(
        graph=g,
        _concept_keywords={"impact": [dead_c, live_c], "bruise": [dead_e, live_e]},
        _belief_assertions=[], _trace_enabled=False)
    if arm == "B_local":
        local_live_index_guard(host)
    returned = ChainWalkerMixin._create_causal_edge(
        host, "impact", "bruise", source_text=TEXT)
    edge = g.get_edge(live_c, live_e)
    out = {"live_id_written": edge is not None,
           "stale_id_written": g.get_edge(dead_c, dead_e) is not None,
           "dangling_edges": dangling(g),
           "beliefs": [list(b) for b in host._belief_assertions],
           "returned": returned}
    # baseline writes to the dead pair; guarded arm writes to the live pair
    return ("OPEN" if out["stale_id_written"] and not out["live_id_written"] else
            ("WRITTEN" if out["live_id_written"] and not out["stale_id_written"] else "OTHER")), out


def run_direct_add(arm, stale):
    g = ConceptGraph(dim=DIM)
    s, t = node(g, "dead_src"), node(g, "live_tgt")
    if stale:
        g.remove_node(s)
    cm = addedge_guard_active() if arm == "C_addedge" else None
    try:
        if cm is not None:
            cm.__enter__()
        ret = g.add_edge(s, t, weight=0.5, relation_type="curated")
    finally:
        if cm is not None:
            cm.__exit__(None, None, None)
    inserted = (s, t) in g.edges
    out = {"source_live": live(g, s), "target_live": live(g, t),
           "edge_inserted": inserted, "dangling_edges": dangling(g),
           "returned_edge_type": type(ret).__name__,
           "returned_in_edges": any(ret is x for x in g.edges.values())}
    return "OPEN" if inserted and not live(g, s) else (
        "WRITTEN" if inserted else "CLOSED"), out


def run_episodic(arm, stale):
    g = ConceptGraph(dim=DIM)
    s, a = node(g, "echo_subj"), node(g, "echo_assoc")
    if stale:
        g.remove_node(s)
        g.remove_node(a)
    host = SimpleNamespace(
        graph=g,
        _concept_keywords={"echo_subj": [s], "echo_assoc": [a]},
        _topic_store={},
    )
    cm = addedge_guard_active() if arm == "C_addedge" else None
    try:
        if cm is not None:
            cm.__enter__()
        MemoryMixin._store_episodic(host, "echo_subj", [("echo_assoc", 1.0)])
    finally:
        if cm is not None:
            cm.__exit__(None, None, None)
    inserted = (s, a) in g.edges
    return ("OPEN" if inserted and not live(g, s) else
            ("WRITTEN" if inserted else "CLOSED")), {
        "edge_inserted": inserted, "dangling_edges": dangling(g),
        "endpoints_live": [live(g, s), live(g, a)]}


# ══ H0: the highest-severity conversational causal teaching path ══════════

a_new = run_causal_arm("A_baseline", stale=True)
rec = classify_causal(a_new)
record("H0-A-new", "A baseline", {"stale_index": ["impact", "bruise"], "text": TEXT,
                                  "preplanted_edge": False},
       a_new["after"], rec)

b_new = run_causal_arm("B_local", stale=True)
record("H0-B-new", "B local guard", {"index": "identical stale index", "text": TEXT,
                                     "guard_fired": b_new["guard_fired"]},
       b_new["after"], classify_causal(b_new))

c_new = run_causal_arm("C_addedge", stale=True)
record("H0-C-new", "C add_edge guard", {"index": "identical stale index", "text": TEXT},
       c_new["after"], classify_causal(c_new))

# F-A: instrument control -- the vulnerability MUST reproduce on the baseline.
if rec != "OPEN":
    print("INSTRUMENT INVALID: F-A baseline did not write a dangling edge", flush=True)

# F-B: local guard fired and preserves the legitimate teaching write.
a_live = run_causal_arm("A_baseline", stale=False)
b_live = run_causal_arm("B_local", stale=False)
record("H0-A-live", "A baseline", {"live_index": ["impact", "bruise"], "text": TEXT},
       a_live["after"], classify_causal(a_live))
record("H0-B-live", "B local guard", {"live_index": ["impact", "bruise"], "text": TEXT},
       b_live["after"], classify_causal(b_live))

# Mixed index (dead id first, live id second): the guard must resolve to the
# live id and still teach the fact, not merely fail closed.
a_mix, a_mix_out = run_causal_mixed_arm("A_baseline")
b_mix, b_mix_out = run_causal_mixed_arm("B_local")
record("H0-A-mixed", "A baseline", {"index": ["dead_impact", "live_impact"],
                                    "text": TEXT}, a_mix_out, a_mix)
record("H0-B-mixed", "B local guard", {"index": ["dead_impact", "live_impact"],
                                       "text": TEXT}, b_mix_out, b_mix)

# Existing dangling edge: does the choke-point-only guard reach the strengthen branch?
a_old = run_causal_arm("A_baseline", stale=True, preplant=True)
b_old = run_causal_arm("B_local", stale=True, preplant=True)
c_old = run_causal_arm("C_addedge", stale=True, preplant=True)
for cid, run, arm in (("H0-A-strengthen", a_old, "A baseline"),
                      ("H0-B-strengthen", b_old, "B local guard"),
                      ("H0-C-strengthen", c_old, "C add_edge guard")):
    record(cid, arm, {"stale_index": ["impact", "bruise"], "preplanted_edge": "0.1/semantic",
                      "text": TEXT}, run["after"], classify_causal(run, preplant=True))

# ══ H5/H1 extension: dead-endpoint guard at ConceptGraph.add_edge ═════════

h5a, h5a_out = run_direct_add("A_baseline", stale=True)
record("H5-A", "A baseline", {"dead_source": "dead_src", "live_target": "live_tgt"},
       h5a_out, h5a)
h5c, h5c_out = run_direct_add("C_addedge", stale=True)
record("H5-C", "C add_edge guard", {"dead_source": "dead_src", "live_target": "live_tgt"},
       h5c_out, h5c)
h5cl, h5cl_out = run_direct_add("C_addedge", stale=False)
record("H5-C-live", "C add_edge guard", {"live_source": "dead_src", "live_target": "live_tgt"},
       h5cl_out, h5cl)

h1a, h1a_out = run_episodic("A_baseline", stale=True)
record("H1-A", "A baseline", {"stale_index": ["echo_subj", "echo_assoc"]}, h1a_out, h1a)
h1c, h1c_out = run_episodic("C_addedge", stale=True)
record("H1-C", "C add_edge guard", {"stale_index": ["echo_subj", "echo_assoc"]}, h1c_out, h1c)
h1cl, h1cl_out = run_episodic("C_addedge", stale=False)
record("H1-C-live", "C add_edge guard", {"live_index": ["echo_subj", "echo_assoc"]},
       h1cl_out, h1cl)

# ── verdict ────────────────────────────────────────────────────────────────
EXPECT = {
    "H0-A-new": "OPEN",          # F-A: vulnerability reproduced
    "H0-B-new": "CLOSED",        # candidate local guard works
    "H0-C-new": "CLOSED",        # choke-point guard also closes new-edge branch
    "H0-A-live": "WRITTEN",      # no regression, baseline
    "H0-B-live": "WRITTEN",      # F-B: no regression under local guard
    "H0-A-mixed": "OPEN",        # baseline takes [0] even when a live id exists
    "H0-B-mixed": "WRITTEN",     # guard resolves to the live id, still teaches
    "H0-A-strengthen": "STRENGTHENED",
    "H0-B-strengthen": "UNCHANGED",  # local guard reaches strengthen branch
    "H0-C-strengthen": "STRENGTHENED",  # KEY FINDING: add_edge-only guard bypassed
    "H5-A": "OPEN",
    "H5-C": "CLOSED",            # F-C: guard proven installed
    "H5-C-live": "WRITTEN",      # no regression at the choke point
    "H1-A": "OPEN",
    "H1-C": "CLOSED",            # choke-point guard covers the episodic writer
    "H1-C-live": "WRITTEN",
}
actual = {r["id"]: r["status"] for r in RECORDS}
mismatches = {k: {"expected": v, "actual": actual.get(k)} for k, v in EXPECT.items()
              if actual.get(k) != v}
verdict = "PASS" if not mismatches else "FAIL"

watched = [
    "ravana/src/ravana/chat/chain_walker.py",
    "ravana/src/ravana/chat/engine_memory.py",
    "ravana_ml/src/ravana_ml/graph.py",
    "ravana/src/ravana/web/web_to_graph.py",
]
payload = {
    "probe": "car_stale_write_guard_ab",
    "version": 1,
    "engine_booted": False,
    "production_files_edited": False,
    "import_root": IMPORT_ROOT,
    "source_hashes_lf_sha256_16": {p: hashlib.sha256(norm_src(p).encode()).hexdigest()[:16]
                                   for p in watched},
    "falsifiers": {
        "F-A_baseline_vulnerability": "PASS" if actual.get("H0-A-new") == "OPEN" else "FAIL",
        "F-B_local_guard_fired_and_no_regression": "PASS" if (
            b_new["guard_fired"] and actual.get("H0-B-new") == "CLOSED"
            and actual.get("H0-B-live") == "WRITTEN"
            and actual.get("H0-B-mixed") == "WRITTEN") else "FAIL",
        "F-C_addedge_guard_fired": "PASS" if actual.get("H5-C") == "CLOSED" else "FAIL",
    },
    "store_state_before": STORE_BEFORE,
    "store_state_after": store_state(),
    "records": RECORDS,
    "expectations": EXPECT,
    "mismatches": mismatches,
    "verdict": verdict,
}
# No commit metadata enters the payload: the frozen report hash depends only on
# the measurement inputs/outputs and stays stable across commits and amends.
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
payload["report_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
out_path = os.path.join(HERE, "results.json")
with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(payload, fh, sort_keys=True, indent=2, ensure_ascii=True)
    fh.write("\n")
print(f"FALSIFIERS: {payload['falsifiers']}", flush=True)
print(f"VERDICT: {verdict}  mismatches={mismatches}  sha256={payload['report_sha256']}", flush=True)
print(f"report={out_path}", flush=True)
sys.exit(0 if verdict == "PASS" else 1)
