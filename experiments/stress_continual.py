"""Continual-learning stress test for the RAVANA graph (Work 3).

Brain basis (per the plan):
  - McClelland 1995 (CLS): interleaved replay, never sequential overwrite.
  - Fusi & Benna cascade consolidation: labile -> resistant; resistant =
    pruning-protected.
  - Tse 2007: schema-fit fast-tracks keeping.
  - Tononi & Cirelli 2014 (SHY): protect reactivated+schema-consistent, prune
    isolated -> raises S/N.
  - Kirkpatrick 2017 EWC: blackout at saturation -> MONITOR CAPACITY.
  - Diaz-Rodriguez 2018: report ACC, BWT (backward transfer), FWT (forward
    transfer), forgetting.

What this validates: before building compositional reasoning (Work A0) on top
of the graph, prove the foundation is STABLE under lifelong ingest — golden
facts survive, noise is pruned, and there is NO EWC-style blackout (catastrophic
forgetting of important edges).

Design:
  1. Build CognitiveChatEngine.
  2. Inject "golden facts" = verified edges between EXISTING seeded concepts,
     tagged edge_kind='web_fact' (schema-consistent, so SHY/Tse protect them).
     Snapshot golden (src,tgt) keys + golden node core_vectors.
  3. Bulk ingest large + adversarial/conflicting text via the local
     WebLearningMixin._learn_from_text (NO network — pure parser). This is the
     "lifelong learning" pressure.
  4. Interference: a chunk of conflicting text AFTER golden facts.
  5. Set sweep params: SleepConfig.prune_threshold + graph_engine
     _pattern_separation / _ps_top_k.
  6. engine._sleep_consolidate(golden_edge_keys=...) -> returns
     important_facts_retained / important_facts_drifted / important_facts_blackout
     / retention_rate (engine.py extension from Work 3).
  7. Measure:
     - retention_rate (BWT analog): golden facts kept after ingest+interference.
       Higher = less forgetting.
     - core_vector_drift: mean cosine distance of golden node vectors vs the
       pre-sleep snapshot (Friston: structural stability).
     - noise_pruned: total edges pruned in the cycle (S/N up if high while
       golden retained).
     - blackout: golden edges fully gone (EWC blackout at saturation) -> must
       be 0 for success.
     - FWT analog: did pre-injecting golden facts stabilize the graph so the
       adversarial ingest produced FEWER surviving off-frame edges? Measured as
       noise_survived = total edges after - (golden + expected); lower = better
       forward stabilization.
  Success: core probes preserved or improved (S/N up), noise pruned, NO blackout.

Run:
    python -m experiments.stress_continual                 # default sweep
    python -m experiments.stress_continual --docs 40 --epochs 3
"""
import os
import sys
import time
import argparse
from typing import Dict, List, Tuple, Any, Set

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana_ml", "src"))

import numpy as np  # noqa: E402


# ── Golden facts: verified edges between EXISTING seeded concepts ──
# These are "important" — schema-consistent, tagged web_fact so SHY/Tse protect.
GOLDEN_FACTS: List[Tuple[str, str, str]] = [
    ("photosynthesis", "is", "process"),
    ("gravity", "is", "force"),
    ("whale", "is", "mammal"),
    ("memory", "is", "capacity"),
    ("light", "is", "wave"),
    ("trust", "is", "bond"),
    ("tree", "is", "plant"),
    ("water", "is", "liquid"),
]


def _golden_keys(engine) -> Set[Tuple[int, int]]:
    """Resolve golden fact labels -> (src,tgt) edge keys that exist in the graph."""
    kw = engine._concept_keywords
    keys = set()
    for s, r, o in GOLDEN_FACTS:
        sn = kw.get(s, [])
        on = kw.get(o, [])
        for a in sn:
            for b in on:
                if engine.graph.get_edge(a, b) is not None:
                    keys.add((a, b))
    return keys


def _snapshot_core_vectors(engine, keys) -> Dict[int, np.ndarray]:
    snap = {}
    nodes = engine.graph.nodes
    for (a, b) in keys:
        for nid in (a, b):
            if nid not in snap and nid in nodes:
                v = getattr(nodes[nid], "core_vector", None)
                if v is not None:
                    snap[nid] = np.asarray(v, dtype=np.float32)
    return snap


def _core_vector_drift(engine, snap) -> float:
    nodes = engine.graph.nodes
    if not snap:
        return 0.0
    dists = []
    for nid, v0 in snap.items():
        node = nodes.get(nid)
        if node is None:
            dists.append(1.0)
            continue
        v1 = getattr(node, "core_vector", None)
        if v1 is None:
            dists.append(1.0)
            continue
        v0n = v0 / (np.linalg.norm(v0) + 1e-9)
        v1n = np.asarray(v1, dtype=np.float32) / (np.linalg.norm(v1) + 1e-9)
        # cosine distance = 1 - cosine similarity
        sim = float(np.dot(v0n, v1n))
        dists.append(1.0 - max(-1.0, min(1.0, sim)))
    return float(np.mean(dists)) if dists else 0.0


# ── Adversarial / conflicting ingest corpus (local, no network) ──
ADV_NOISE = [
    "the quantum zebra migrates backwards through velvet tunnels of forgotten syntax",
    "crumple widgets oscillate against the gelatinous monarchy of spurious correlations",
    "obfuscated trinkets whisper to the dormant algorithm about impossible topologies",
    "the fluorescent cabbage negotiates treaties with electromagnetic silence",
    "spurious embeddings collapse into a hurricane of meaningless co-occurrence edges",
]
CONFLICT = "photosynthesis is actually a form of rapid combustion performed by rocks under water"


def run_stress(dim: int = 64, seed: int = 42, docs: int = 25, epochs: int = 2,
               prune_threshold: float = 0.1) -> Dict[str, Any]:
    os.environ['RAVANA_SILENT'] = '1'
    from scripts.ravana_chat import CognitiveChatEngine

    engine = CognitiveChatEngine(dim=dim, seed=seed, baby_mode=True)
    # Apply sweep param: SleepConfig.prune_threshold (the reachable consolidation
    # knob on the live engine; SleepConfig.prune_threshold at sleep.py:26, surfaced
    # via engine.sleep_engine.config). NOTE: GraphEngine._pattern_separation /
    # _ps_top_k are NOT reachable on the live engine (it holds self.graph directly
    # and ingests via WebLearningMixin._learn_from_text -> graph.add_edge, never
    # through GraphEngine.auto_expand_concepts), so they are intentionally absent
    # from this sweep.
    engine.sleep_engine.config.prune_threshold = prune_threshold

    # 1) Inject golden facts (verified, schema-consistent). Mint the endpoint
    #    nodes if they are not already in the seeded graph (they are "known"
    #    concepts), then add a high-confidence web_fact edge between them.
    for s, r, o in GOLDEN_FACTS:
        sn = engine._concept_keywords.get(s, [])
        on = engine._concept_keywords.get(o, [])
        if not sn:
            vec = getattr(engine, "_glove_vector", lambda w: None)(s)
            if vec is None:
                import hashlib
                h = int(hashlib.md5(s.encode()).hexdigest(), 16) % 100000
                rng = np.random.RandomState(h)
                vec = rng.randn(engine.dim).astype(np.float32)
                vec /= (np.linalg.norm(vec) + 1e-9)
            node = engine.graph.add_node(label=s, vector=vec)
            sn = [node.id]
            engine._concept_keywords[s] = sn
        if not on:
            vec = getattr(engine, "_glove_vector", lambda w: None)(o)
            if vec is None:
                import hashlib
                h = int(hashlib.md5(o.encode()).hexdigest(), 16) % 100000
                rng = np.random.RandomState(h)
                vec = rng.randn(engine.dim).astype(np.float32)
                vec /= (np.linalg.norm(vec) + 1e-9)
            node = engine.graph.add_node(label=o, vector=vec)
            on = [node.id]
            engine._concept_keywords[o] = on
        e = engine.graph.add_edge(sn[0], on[0], weight=0.9, relation_type=r,
                                  confidence=0.9)
        if e is not None and hasattr(e, "source_metadata"):
            e.source_metadata.update({"source": "golden", "edge_kind": "web_fact"})

    golden_keys = _golden_keys(engine)
    golden_snap = _snapshot_core_vectors(engine, golden_keys)
    n_edges_before = len(engine.graph.edges)
    n_nodes_before = len(engine.graph.nodes)

    # 2) Bulk ingest large + adversarial text (local parser, no network).
    corpus = []
    for i in range(docs):
        # synthetic "article" with a topic + body; alternate benign/adversarial
        corpus.append((f"topic {i}", f"topic {i} relates to concept {i} and "
                                    f"structure {i} via mechanism {i}. " + ADV_NOISE[i % len(ADV_NOISE)]))
    for _ in range(epochs):
        for topic, text in corpus:
            try:
                engine._learn_from_text(text, topic)
            except Exception:
                pass
    # 3) Interference: conflicting text injected AFTER golden facts.
    try:
        engine._learn_from_text(CONFLICT, "photosynthesis")
    except Exception:
        pass

    n_edges_after_ingest = len(engine.graph.edges)
    noise_added = n_edges_after_ingest - n_edges_before

    # 4) Sleep + consolidation with retention measurement.
    t0 = time.time()
    result = engine._sleep_consolidate(golden_edge_keys=golden_keys)
    sleep_s = time.time() - t0

    drift = _core_vector_drift(engine, golden_snap)
    n_edges_final = len(engine.graph.edges)
    noise_survived = n_edges_final - len(golden_keys)

    return {
        "params": {"prune_threshold": prune_threshold,
                   "docs": docs, "epochs": epochs},
        "nodes_before": n_nodes_before, "edges_before": n_edges_before,
        "noise_added": noise_added, "edges_final": n_edges_final,
        "sleep_s": round(sleep_s, 2),
        "edges_pruned": result.get("edges_pruned", 0),
        "retention_rate": result.get("retention_rate", None),
        "important_facts_total": result.get("important_facts_total", None),
        "important_facts_retained": result.get("important_facts_retained", None),
        "important_facts_drifted": result.get("important_facts_drifted", None),
        "important_facts_blackout": result.get("important_facts_blackout", None),
        "core_vector_drift": round(drift, 4),
        "noise_survived": noise_survived,
    }


def run_sweep(dim: int = 64, seed: int = 42, docs: int = 25, epochs: int = 2):
    print("=" * 78)
    print("CONTINUAL-LEARNING STRESS TEST (Work 3) — validate the foundation")
    print("=" * 78)
    # Sweep prune_threshold (the reachable consolidation knob on the live
    # engine). pattern_separation/_ps_top_k are GraphEngine-only and not on the
    # live ingest path, so they are excluded (see run_stress note).
    configs = [0.05, 0.1, 0.2, 0.3]
    rows = []
    for pt in configs:
        print(f"\n--- prune_threshold={pt} ---")
        r = run_stress(dim=dim, seed=seed, docs=docs, epochs=epochs,
                       prune_threshold=pt)
        rows.append(r)
        rt = r["retention_rate"]
        print(f"  retention_rate     : {rt:.3f}" if rt is not None else "  retention_rate     : n/a")
        print(f"  important_retained : {r['important_facts_retained']}/{r['important_facts_total']}")
        print(f"  blackout (forgot)  : {r['important_facts_blackout']}")
        print(f"  core_vector_drift  : {r['core_vector_drift']}")
        print(f"  noise_added        : {r['noise_added']}  noise_survived: {r['noise_survived']}  edges_pruned: {r['edges_pruned']}")

    # Summary verdict across configs.
    print("\n" + "=" * 78)
    print("VERDICT (per config: golden retained, noise pruned, NO blackout)")
    print("=" * 78)
    for r in rows:
        ok = (r["important_facts_blackout"] == 0
              and r["retention_rate"] is not None and r["retention_rate"] >= 0.8
              and r["edges_pruned"] > 0)
        tag = "STABLE" if ok else "CHECK"
        print(f"  [{tag}] pt={r['params']['prune_threshold']} | retain={r['retention_rate']} "
              f"blackout={r['important_facts_blackout']} pruned={r['edges_pruned']}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Continual-learning stress test")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--docs", type=int, default=25, help="adversarial docs per epoch")
    ap.add_argument("--epochs", type=int, default=2, help="ingest epochs")
    args = ap.parse_args()
    run_sweep(dim=args.dim, seed=args.seed, docs=args.docs, epochs=args.epochs)


if __name__ == "__main__":
    main()
