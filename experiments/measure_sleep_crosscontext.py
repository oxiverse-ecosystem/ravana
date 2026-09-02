"""
CLS Sleep Gate (SHY down-selection) — MEASUREMENT
==================================================

Plan: close the CLS consolidation loop. The ingest-time gate (IV-C, done)
tags every web_fact edge with source_metadata['contexts'] and XdG-protects
it from cross-context overwrite at WRITE time. But sleep.py._prune_weak_edges
was weight-only blind — a web_fact edge corroborated in 2+ contexts but with
low weight (e.g. 0.08) got pruned exactly like noise. So XdG protects the
synapse at write time but sleep deletes it later.

This measurement proves the sleep-time gate now protects cross-context-
corroborated edges (SHY down-selection, Tononi & Cirelli 2014; Nere et al.
2013): cross-context corroboration = the offline reactivation signal
("reactivated across multiple offline bouts" -> fits prior structure ->
protect). This is the sleep-time complement to the IV-C ingest gate
(van de Ven et al. 2020) — both needed side by side.

Golden scenario (mirrors measure_ivc_xdg.py):
  E1 water --is_a--> chemical_compound  weight 0.08  contexts=[ctxA, ctxB]
  E2 noise --related--> junk             weight 0.08  contexts=[ctxA]
  Run _prune_weak_edges(graph, threshold=0.1).
  ASSERT: E1 survives (>=2 contexts), E2 pruned (single context).
  CONTROL: protect_cross_context=False -> both pruned -> proves the GATE
           saves E1, not a weight artifact.

Run: python experiments/measure_sleep_crosscontext.py
"""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))

import numpy as np
from ravana.graph.engine import GraphEngine
from ravana.core.sleep import SleepConsolidation, SleepConfig


def _seed_ge():
    ge = GraphEngine(dim=64, glove_vecs=None)
    for w in ["water", "chemical_compound", "noise", "junk"]:
        vec = np.random.RandomState(hash(w) % 1000).randn(64).astype("float32")
        n = np.linalg.norm(vec)
        if n > 0:
            vec /= n
        node = ge.graph.add_node(vector=vec, label=w)
        ge._all_labels[w] = node.id
    return ge


def _edge(ge, a, b):
    sa, sb = ge._all_labels[a], ge._all_labels[b]
    return ge.graph.get_edge(sa, sb)


def _build(protect, min_ctx):
    """Build E1 (2-ctx, low weight) + E2 (1-ctx, low weight), prune, return
    (e1_alive, e2_alive, n_pruned)."""
    ge = _seed_ge()
    # E1: water is_a chemical_compound, weight 0.08, contexts [ctxA, ctxB]
    e1 = ge.graph.add_edge(ge._all_labels["water"], ge._all_labels["chemical_compound"],
                           weight=0.08, relation_type="is_a", confidence=0.08)
    e1.source_metadata["contexts"] = ["ctxA", "ctxB"]
    # E2: noise related junk, weight 0.08, contexts [ctxA]  (single context)
    e2 = ge.graph.add_edge(ge._all_labels["noise"], ge._all_labels["junk"],
                           weight=0.08, relation_type="related", confidence=0.08)
    e2.source_metadata["contexts"] = ["ctxA"]

    cfg = SleepConfig(protect_cross_context=protect, min_contexts=min_ctx)
    sc = SleepConsolidation(cfg)
    n_pruned = sc._prune_weak_edges(ge.graph, threshold=0.1,
                                    protect_cross_context=protect, min_contexts=min_ctx)
    return (e1 is not None and _edge(ge, "water", "chemical_compound") is not None,
            _edge(ge, "noise", "junk") is not None,
            n_pruned)


def main():
    print("=" * 78)
    print("CLS Sleep Gate (SHY down-selection) — protect cross-context edges")
    print("=" * 78)

    e1_alive, e2_alive, n_pruned = _build(protect=True, min_ctx=2)
    print(f"\n[gate ON, min_contexts=2]  E1(2-ctx) alive={e1_alive}  E2(1-ctx) alive={e2_alive}  pruned={n_pruned}")
    e1_alive_off, e2_alive_off, n_pruned_off = _build(protect=False, min_ctx=2)
    print(f"[gate OFF]               E1(2-ctx) alive={e1_alive_off}  E2(1-ctx) alive={e2_alive_off}  pruned={n_pruned_off}")

    verdict_gate = e1_alive and (not e2_alive)
    verdict_ctrl = (not e1_alive_off) and (not e2_alive_off)

    print("\n[VERDICT]")
    print(f"  Gate ON : cross-context E1 protected, single-context E2 pruned -> "
          f"{'CONFIRMED' if verdict_gate else 'CHECK'}")
    print(f"  Gate OFF: both pruned (proves the GATE saves E1, not weight)   -> "
          f"{'CONFIRMED' if verdict_ctrl else 'CHECK'}")
    print("\n  Interpretation:")
    print("    Before this change, _prune_weak_edges was weight-only blind:")
    print("    Both E1 and E2 (weight 0.08 < 0.1) were deleted, so XdG's")
    print("    write-time protection was undone at sleep. Now the sleep gate")
    print("    reads source_metadata['contexts'] and spares edges reactivated")
    print("    in 2+ independent contexts (SHY down-selection). The two gates")
    print("    (IV-C ingest XdG + this sleep gate) are now the two halves of")
    print("    one consolidation mechanism, as brains use metaplasticity +")
    print("    replay (van de Ven et al. 2020).")


if __name__ == "__main__":
    main()
