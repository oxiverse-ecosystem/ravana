#!/usr/bin/env python3
"""Read-only dump of raw ConceptEdges for two far subjects (no engine boot)."""
from __future__ import annotations
import os, pickle, sys
sys.path.insert(0, "C:/Users/Likhith/Documents/projects/ravana_trak_far_rootcause/ravana/src")
sys.path.insert(0, "C:/Users/Likhith/Documents/projects/ravana_trak_far_rootcause/scripts")
sys.path.insert(0, "C:/Users/Likhith/Documents/projects/ravana_trak_far_rootcause/scratch")
from ravana.chat.schema_induction import typed_slot_views_from_graph  # noqa: F401

ROOT = "C:/Users/Likhith/Documents/projects/ravana_m123_run/scratch/m123_data/trak_m123"
for item, subj in (("tra_k_0000", "market"), ("tra_k_0020", "guild"),
                   ("tra_k_0020", "blizzard")):
    p = os.path.join(ROOT, item, "ravana_weights.pkl")
    st = pickle.load(open(p, "rb"))
    g = st.get("graph")
    print("=====", item, subj, "type", type(g).__name__)
    nid = None
    for k, n in g.nodes.items():
        if str(getattr(n, "label", "")).strip().casefold() == subj:
            nid = k
            print(" node", k, repr(getattr(n, "label", "")), "type",
                  getattr(n, "node_type", None))
            break
    if nid is None:
        print(" NO NODE")
        continue
    edges = []
    for eid, e in getattr(g, "edges", {}).items():
        src = getattr(e, "source", None)
        tgt = getattr(e, "target", None)
        if src == nid or tgt == nid:
            edges.append((eid, e))
    print(" incident edges:", len(edges))
    for eid, e in edges[:40]:
        other = getattr(e, "target", None) if getattr(e, "source", None) == nid else getattr(e, "source", None)
        on = g.get_node(other)
        print("  id=%s dir=%s other=%r rel=%r conf=%s meta=%s" % (
            eid, "out" if getattr(e, "source", None) == nid else "in",
            getattr(on, "label", None), getattr(e, "relation_type", None),
            getattr(e, "confidence", None),
            getattr(e, "source_metadata", None)))
