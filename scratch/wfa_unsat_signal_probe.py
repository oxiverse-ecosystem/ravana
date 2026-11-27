#!/usr/bin/env python3
"""Boot-free signal census: what graded values does staged WFA provenance carry?

Read-only: imports production functions UNMODIFIED; builds in-memory ConceptGraph
fixtures only; never boots CognitiveChatEngine, never opens a weight store.
"""
from __future__ import annotations
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
for p in (os.path.join(PROJ, "scripts"), os.path.join(PROJ, "ravana", "src"), PROJ):
    if p not in sys.path:
        sys.path.insert(0, p)
import measure_learn5_wfa as w

summary_path = os.path.join(PROJ, "..", "ravana", "scratch", "learn5_wfa_summary.json")
summary = json.load(open(summary_path, encoding="utf-8"))
rows = summary["pd_audit"]
items = {it["id"]: it for it in w.build_items()}

class E:
    def __init__(self):
        self.graph = w.ConceptGraph(dim=8) if hasattr(w, "ConceptGraph") else None

# reuse production graph class the way the prior scratch does
from ravana_ml.graph import ConceptGraph
class EG:
    def __init__(self):
        self.graph = ConceptGraph(dim=8)

def stage(item, kind, arm):
    eng = EG()
    claim = item["claim"]
    cred = w.is_source_id(claim, "credible")
    disc = w.is_source_id(claim, "discredited")
    if arm in ("inclusion",):
        w._write_asserted_wfa_fact(eng, claim, disc if kind == "decoy" else cred)
    elif arm in ("exclusion", "forced_choice"):
        if kind == "gap":
            w._write_asserted_wfa_fact(eng, claim, cred)
        else:
            w._write_asserted_wfa_fact(eng, claim, disc)
            w.apply_source_invalidation(eng, disc)
    elif arm == "invalidation":
        w._write_asserted_wfa_fact(eng, claim, cred)
        w.apply_source_invalidation(eng, cred)
    return eng, (disc if kind == "decoy" else cred)

out = []
for r in rows[:12]:
    it = items[r["id"]]
    rec = {"id": r["id"], "kind": r["kind"], "pool": r["pool"],
           "claim": it["claim"], "arms": {}}
    for arm in ("inclusion", "exclusion", "invalidation", "forced_choice"):
        eng, sid = stage(it, r["kind"], arm)
        pr = w.graph_provenance_for_claim(eng, it["claim"], sid)
        ctx, audit = w.context_from_graph_provenance(eng, it["claim"], sid)
        rec["arms"][arm] = {
            "n_edges": len(pr),
            "rows": [{k: v for k, v in m.items() if k != "claim_label_overlap"} for m in pr],
            "ctx_rel": ctx.source.reliability,
            "ctx_familiarity": ctx.familiarity,
            "trust": ctx.source.trust_class.value,
            "audit": {k: v for k, v in audit.items() if k != "fingerprint"},
        }
    out.append(rec)
print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
