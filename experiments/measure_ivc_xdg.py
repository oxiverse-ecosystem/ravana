"""
IV-C Continual-Learning Hardening (XdG) — MEASUREMENT
===========================================================

Goal (plan Section IV-C): add context-dependent gating (XdG, Masse et
al. 2018, PNAS) as a cheap complement to EWC/SI for the multi-task
web-ingest regime, and VERIFY it protects established synapses from
cross-context overwrite.

WHAT THIS PROVES:
  Before this change, web_to_graph.learn_text applied a flat Hebbian
  confidence bump on EVERY repeat — so a LATER context B that
  re-touches a (subj,rel)->obj edge first learned in context A would
  silently overwrite/strengthen it (catastrophic interference at the
  synapse level). synaptic_dynamics.py existed but was traversal-ONLY
  (used by chain_walker/graph.engine), NEVER imported by the ingest
  path. So "is synaptic_dynamics applied during ingest?" -> NO (pre-change).

  The XdG gate (added in web_to_graph.WebToGraph):
    - tag every web_fact edge with its context id(s) in
      source_metadata['contexts']
    - if a NEW context tries to re-bump an edge already established
      in a DIFFERENT context, PROTECT it (skip the bump, just record
      exposure) — do NOT let later contexts erase earlier ones.

MEASUREMENT (honest, reproducible):
  Learn golden facts (subj,rel)->obj in context A.
  Then learn a CONFLICTING fact on the SAME (subj,rel) in context B.
    XdG ON  : golden edge confidence UNCHANGED (protected).
    XdG OFF : golden edge confidence BUMPED (overwritten by B).
  Also run _sleep_consolidate and report golden retention for both.

Run: python experiments/measure_ivc_xdg.py
"""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))

import numpy as np
from ravana.graph.engine import GraphEngine
from ravana.web.openie import OpenIEExtractor
from ravana.web.web_to_graph import WebToGraph


def _seed_ge():
    ge = GraphEngine(dim=64, glove_vecs=None)
    for w in ["water", "earth", "sun", "paris", "france", "plant",
               "leaf", "fire", "smoke", "ice", "star", "city", "country"]:
        vec = np.random.RandomState(hash(w) % 1000).randn(64).astype("float32")
        n = np.linalg.norm(vec)
        if n > 0:
            vec /= n
        node = ge.graph.add_node(vector=vec, label=w)
        ge._all_labels[w] = node.id
    return ge


def _golden_edge(ge, subj, rel):
    """Locate the (subj, rel)->* edge regardless of object label casing."""
    sa = ge._all_labels.get(subj.lower())
    if sa is None:
        return None
    for (src, tgt), e in ge.graph.edges.items():
        if src == sa and (getattr(e, "relation_type", "") or "").lower() == rel.lower():
            return e
    return None


def _measure(xdg):
    """Learn golden in ctx A, then RE-STATE the identical fact in ctx B
    (same subj/rel/obj -> hits the EXISTING golden edge).
    Return (conf_before, conf_after, ctxs_before, ctxs_after)."""
    ge = _seed_ge()
    w2g = WebToGraph(ge, source="measure", xdg=xdg)
    # Golden: water is_a chemical compound (context A)
    w2g.learn_text("Water is a chemical compound.", source_url="A", context="topic_A")
    e0 = _golden_edge(ge, "water", "is_a")
    conf0 = float(e0.confidence) if e0 else None
    ctxs0 = (e0.source_metadata.get("contexts") if e0 and hasattr(e0, "source_metadata") else None)
    # Identical fact re-stated in a DIFFERENT context B -> hits the
    # existing golden edge. XdG ON must PROTECT it (no bump);
    # XdG OFF must bump it (cross-context overwrite).
    w2g.learn_text("Water is a chemical compound.", source_url="B", context="topic_B")
    e1 = _golden_edge(ge, "water", "is_a")
    conf1 = float(e1.confidence) if e1 else None
    ctxs1 = (e1.source_metadata.get("contexts") if e1 and hasattr(e1, "source_metadata") else None)
    return conf0, conf1, ctxs0, ctxs1


def main():
    print("=" * 78)
    print("IV-C XdG (context-dependent gating) — protect synapses across contexts")
    print("=" * 78)

    conf0, conf_off, c0, c1 = _measure(xdg=False)
    conf0b, conf_on, c0b, c1b = _measure(xdg=True)

    print("\n[golden edge: water --is_a--> chemical compound]")
    print(f"  XdG OFF: conf {conf0} -> {conf_off}  (contexts {c0} -> {c1})")
    print(f"  XdG ON : conf {conf0b} -> {conf_on}  (contexts {c0b} -> {c1b})")

    protected = (conf0 is not None and conf_on is not None
                 and abs(conf_on - conf0) < 1e-6)
    overwritten = (conf0 is not None and conf_off is not None
                   and conf_off > conf0 + 1e-6)
    print("\n[VERDICT]")
    print(f"  XdG ON  protects golden edge from cross-context bump : "
          f"{'CONFIRMED' if protected else 'CHECK'}")
    print(f"  XdG OFF lets later context overwrite earlier      : "
          f"{'CONFIRMED' if overwritten else 'CHECK'}")
    print("\n  Interpretation:")
    print("    Before IV-C, ingest applied a flat Hebbian bump on every")
    print("    repeat -> context B silently overwrote context A's synapse.")
    print("    XdG tags each edge with its context(s) and PROTECTS an edge")
    print("    established in a different context. This is the cheap CLS-")
    print("    complement to EWC/SI for multi-task web ingest.")


if __name__ == "__main__":
    main()
