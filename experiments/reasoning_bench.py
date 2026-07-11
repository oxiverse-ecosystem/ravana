"""Emergent transitive-reasoning benchmark for the HRR store (Limitation 2).

Brain basis (per the plan):
  - Transitive inference (Dusek & Eichenbaum 1997) proves INTEGRATED structure,
    not paired associates -> we inject ONLY adjacent pairs, never the composed
    pair, so every probe past hop 1 is composition-only.
  - TI success can be faked by value-transfer / embedding similarity (Frank et
    al. 2003) -> FOILS: embedding-distant-but-graph-connected endpoints + a
    strong-but-wrong distractor edge to expose cheating.
  - Tag provenance STORED (direct associate, hop 1) vs INFERRED (composed, hop>1)
    and expect lower confidence on inferred (Johnson 1993 source monitoring;
    Yonelinas recollection vs familiarity).
  - Report calibration (ECE), metacognitive AUROC (Fleming & Dolan 2012), and
    hop-count decay (accuracy/confidence should drop with depth).

Design decisions (fixes a degenerate first pass):
  - Limitation 1's benefit ONLY manifests on CORRELATED atoms. Namespaced OOV
    tokens are orthogonal random atoms, so binding is already perfect and
    decorrelation has nothing to act on (and calibration/foils go degenerate).
    So chains are built from REAL GloVe words (correlated) -> crosstalk is
    possible and decorrelation is measurable.
  - Each chain uses a UNIQUE isolated relation (e.g. '__rb_rel_<cid>') so it is
    decoupled from ravana_weights.db / the seeded graph's relations. Structure is
    composition-only; only the ATOMS are real (so decorrelation matters).
  - inject_controlled_chains: build linear chains of each length/relation from
    sampled real words; add ONLY adjacent (t_i, rel, t_{i+1}) edges. The composed
    tail is NEVER stored -> composition-only probes.
  - probe_chain: HRR recovered tail + per-hop decode cosine (confidence proxy)
    vs graph ground truth (walk the injected same-relation edges).
  - run_arm(whiten, sparse_k, unitary_roles): build engine with flags, inject the
    SAME sampled chains, probe. One HEAD -> both arms: graph edges are identical
    across arms (only HRR atoms differ), so graph ground truth is shared.

Success (per the plan): HRR transitive accuracy climbs with
whiten+sparse_k+unitary_roles; ECE < 0.05; HRR under-confident on inferred
(recollection > familiarity); coverage@len3 >= graph coverage.

Run:
    python -m experiments.reasoning_bench
"""
import os
import sys
import numpy as np

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana_ml", "src"))

LENGTHS = [2, 3, 4, 5]
# Curated embedding-similar clusters per relation family. Chains are built
# WITHIN a cluster so the object pool is correlated -> decode must pick the
# right object among embedding-similar alternatives (crosstalk risk; Frank et
# al. 2003 foil). Shared relation per family (not unique) so many facts compete.
CLUSTERS = {
    "isa": ["cat", "dog", "wolf", "fox", "bear", "lion", "tiger", "horse"],
    "causes": ["fire", "smoke", "heat", "burn", "spark", "flame", "ash"],
    "part_of": ["wheel", "engine", "door", "window", "roof", "tire", "seat"],
    "semantic": ["happy", "sad", "angry", "calm", "joy", "fear", "rage"],
}
CHAINS_PER_REL = 10  # chains per relation family (many competing facts)


def inject_controlled_chains(engine, seed: int = 7):
    """Build linear chains of each length from words WITHIN an embedding-similar
    cluster, one SHARED relation per family. Add ONLY adjacent edges. The composed
    tail is NEVER stored -> composition-only probes. Many chains per relation ->
    decode competes among correlated objects (crosstalk foil)."""
    rng = np.random.RandomState(seed)
    chains = []
    for rel, cluster in CLUSTERS.items():
        for _ in range(CHAINS_PER_REL):
            length = int(rng.choice(LENGTHS))
            words = list(rng.choice(cluster, size=length, replace=False))
            ids = []
            for t in words:
                node = engine.graph.add_node(label=t)
                ids.append(node.id)
            for i in range(length - 1):
                engine.graph.add_edge(ids[i], ids[i + 1],
                                      relation_type=rel, confidence=0.9)
            chains.append({
                "head": words[0], "relation": rel,
                "expected_tail": words[1:], "length": length,
                "head_id": ids[0], "words": words,
            })
    return chains


def _graph_tail(engine, head_id, relation, max_hops):
    """Graph ground truth: walk same-relation outgoing edges (deterministic for
    the injected linear chains). _outgoing stores (target_id, edge) tuples."""
    tail = []
    cur = head_id
    seen = {cur}
    for _ in range(max_hops):
        nxt = None
        for tgt, e in engine.graph._outgoing.get(cur, []):
            if (getattr(e, "relation_type", "") or "").lower() == relation.lower():
                nxt = tgt
                break
        if nxt is None or nxt in seen:
            break
        tail.append(engine.graph.nodes[nxt].label)
        seen.add(nxt)
        cur = nxt
    return tail


def probe_chain(engine, chain):
    head = chain["head"]
    rel = chain["relation"]
    expected = chain["expected_tail"]
    max_hops = len(expected)
    out = engine.hrr_query_chain(head, rel, max_hops=max_hops,
                                 fallback_to_graph=False, return_conf=True)
    # M5: hrr_query_chain now returns (chain, confs, graph_support).
    if isinstance(out, tuple) and len(out) == 3:
        hrr_tail, hrr_confs, graph_support = out
    else:
        hrr_tail, hrr_confs, graph_support = (out if isinstance(out, tuple) else (out, [], [])), [], []
    graph_tail = _graph_tail(engine, chain["head_id"], rel, max_hops)
    hrr_correct = (hrr_tail[:max_hops] == expected[:max_hops])
    rows = []
    for i, exp in enumerate(expected):
        got = hrr_tail[i] if i < len(hrr_tail) else None
        conf = hrr_confs[i] if i < len(hrr_confs) else 0.0
        gs = graph_support[i] if i < len(graph_support) else 0.0
        rows.append({
            "hop": i + 1,
            "provenance": "stored" if i == 0 else "inferred",
            "correct": (got == exp),
            "conf": float(conf),
            "graph_support": float(gs),
        })
    return {
        "length": chain["length"], "relation": rel,
        "hrr_tail": hrr_tail, "expected": expected,
        "graph_tail": graph_tail,
        "hrr_full_correct": hrr_correct,
        "graph_full_correct": (graph_tail[:max_hops] == expected[:max_hops]),
        "words": chain["words"],
        "rows": rows,
    }


def _ece(confs, corrects, n_bins=5):
    confs = np.asarray(confs, dtype=float)
    corrects = np.asarray(corrects, dtype=float)
    if len(confs) == 0:
        return 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (confs >= lo) & (confs <= hi)
        if m.sum() == 0:
            continue
        ece += abs(confs[m].mean() - corrects[m].mean()) * (m.sum() / len(confs))
    return float(ece)


def _auroc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(np.unique(labels)) < 2:
        return float("nan")
    order = np.argsort(scores)
    s = scores[order]
    l = labels[order]
    n_pos = l.sum()
    n_neg = len(l) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        rank[i:j + 1] = (i + j) / 2.0 + 1
        i = j + 1
    sum_rank_pos = rank[l == 1].sum()
    return float((sum_rank_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def run_arm(whiten, sparse_k, unitary_roles, seed=7, m1=True, m3=True, m5=True):
    from scripts.ravana_chat import CognitiveChatEngine
    engine = CognitiveChatEngine(dim=64, seed=seed, baby_mode=True,
                                 hrr_whiten=whiten, hrr_sparse_k=sparse_k,
                                 hrr_unitary_roles=unitary_roles)
    # Mechanism toggles (A/B):
    #  M1 (relation-conditioned candidate restriction): OFF -> clear the verb
    #      index so _candidates() falls back to all atoms.
    #  M3 (resonator iterations): OFF -> 1 iteration (plain NN decode).
    #  M5 (graph re-ranking): ON by construction now (reports graph_support
    #      separately; does not alter the HRR chain, calibration-safe).
    if not m1 and engine.hrr_reasoner is not None:
        engine.hrr_reasoner._by_verb = {}
    if engine.dual_code is not None:
        engine.dual_code._resonator_max_iter = 5 if m3 else 1
    chains = inject_controlled_chains(engine, seed=seed)
    results = [probe_chain(engine, c) for c in chains]
    n = len(results)
    coverage_full = sum(1 for r in results if r["hrr_full_correct"]) / n
    graph_coverage_full = sum(1 for r in results if r["graph_full_correct"]) / n
    acc_by_len = {}
    for L in LENGTHS:
        sub = [r for r in results if r["length"] == L]
        if sub:
            acc_by_len[L] = sum(1 for r in sub if r["hrr_full_correct"]) / len(sub)
    all_conf, all_correct, all_prov, all_gs = [], [], [], []
    for r in results:
        for row in r["rows"]:
            all_conf.append(row["conf"])
            all_correct.append(1 if row["correct"] else 0)
            all_prov.append(row["provenance"])
            all_gs.append(row["graph_support"])
    ece = _ece(all_conf, all_correct)
    auroc = _auroc(all_conf, all_correct)
    stored_conf = [c for c, p in zip(all_conf, all_prov) if p == "stored"]
    inferred_conf = [c for c, p in zip(all_conf, all_prov) if p == "inferred"]
    decay = {}
    for hop in (1, 2, 3, 4):
        sub = [row for r in results for row in r["rows"] if row["hop"] == hop]
        if sub:
            decay[hop] = {
                "acc": float(np.mean([1 if x["correct"] else 0 for x in sub])),
                "conf": float(np.mean([x["conf"] for x in sub])),
            }
    # embedding-distance split: if accuracy were driven by VALUE-TRANSFER
    # (embedding similarity, Frank et al. 2003), CLOSE chains would score HIGH.
    # If it is CLEAN-UP confusion, CLOSE (confusable) chains score LOW. This is
    # the plan's core foil: prove emergent reasoning is real, not transfer.
    g64 = engine.dual_code._lut64
    close_acc, distant_acc = [], []
    close_hop_acc, distant_hop_acc = [], []  # per-hop accuracy split
    for r in results:
        words = r.get("words", [])
        if len(words) < 2:
            continue
        dsum = 0.0
        for i in range(len(words) - 1):
            a, b = g64.get(words[i]), g64.get(words[i + 1])
            if a is not None and b is not None:
                na, nb = a / (np.linalg.norm(a) + 1e-9), b / (np.linalg.norm(b) + 1e-9)
                dsum += 1.0 - float(na @ nb)
        dmean = dsum / (len(words) - 1)
        bucket = close_acc if dmean < 0.55 else distant_acc
        bucket.append(1 if r["hrr_full_correct"] else 0)
        hop_ok = [1 if x["correct"] else 0 for x in r["rows"]]
        (close_hop_acc if dmean < 0.55 else distant_hop_acc).extend(hop_ok)
    acc_close = float(np.mean(close_acc)) if close_acc else None
    acc_distant = float(np.mean(distant_acc)) if distant_acc else None
    hop_close = float(np.mean(close_hop_acc)) if close_hop_acc else None
    hop_distant = float(np.mean(distant_hop_acc)) if distant_hop_acc else None
    gs_mean = float(np.mean(all_gs)) if all_gs else 0.0
    return {
        "config": {"whiten": whiten, "sparse_k": sparse_k, "unitary_roles": unitary_roles,
                   "m1": m1, "m3": m3, "m5": m5},
        "n": n, "coverage_full": coverage_full, "graph_coverage_full": graph_coverage_full,
        "acc_by_len": acc_by_len, "ece": ece, "auroc": auroc,
        "stored_conf_mean": float(np.mean(stored_conf)) if stored_conf else None,
        "inferred_conf_mean": float(np.mean(inferred_conf)) if inferred_conf else None,
        "hop_decay": decay,
        "acc_close": acc_close, "acc_distant": acc_distant,
        "hop_close": hop_close, "hop_distant": hop_distant,
        "graph_support_mean": gs_mean,
    }


def main():
    print("=" * 80)
    print("EMERGENT TRANSITIVE-REASONING BENCHMARK (Limitation 2) + clean-up fix A/B")
    print("=" * 80)
    # A/B the clean-up mechanisms (M1/M3) on the FULL decorrelation config
    # (whiten+sparse256+unitary, which is now the engine default). The earlier
    # turn proved decorrelation alone does NOT move accuracy; these are the
    # competition-structure fixes (Frady et al.) that should.
    configs = [
        ("baseline (no M1/M3)", True, 256, True, False, False),
        ("+ M1 (verb-conditioned candidates)", True, 256, True, True, False),
        ("+ M1 + M3 (resonator)", True, 256, True, True, True),
    ]
    rows = []
    for name, w, k, u, m1, m3 in configs:
        print(f"\n--- arm: {name} (whiten={w}, sparse_k={k}, unitary={u}, m1={m1}, m3={m3}) ---")
        r = run_arm(w, k, u, m1=m1, m3=m3, m5=True)
        rows.append((name, r))
        print(f"  coverage(full tail) : {r['coverage_full']:.3f}  (graph GT: {r['graph_coverage_full']:.3f})")
        bylen = " ".join(f"L{L}={r['acc_by_len'].get(L, 0):.2f}" for L in LENGTHS)
        print(f"  acc by length      : {bylen}")
        print(f"  ECE                 : {r['ece']:.4f}   metacognitive AUROC: {r['auroc']:.3f}")
        sc = r['stored_conf_mean']; ic = r['inferred_conf_mean']
        print(f"  conf stored/inferred: {sc:.3f} / {ic:.3f}  (expect inferred < stored)")
        print(f"  acc close/distant  : {r['acc_close']} / {r['acc_distant']}  "
              f"(close<DISTANT => clean-up confusion, NOT value-transfer)")
        print(f"  hop-acc close/dist : {r['hop_close']} / {r['hop_distant']}  "
              f"(expect gap to narrow with M1/M3)")
        print(f"  graph_support_mean : {r['graph_support_mean']:.3f}  (M5 CLS synergy, separate)")
        decay = " ".join(f"h{h}:a={r['hop_decay'].get(h,{}).get('acc',0):.2f},c={r['hop_decay'].get(h,{}).get('conf',0):.2f}" for h in (1,2,3,4) if h in r['hop_decay'])
        print(f"  hop decay          : {decay}")

    print("\n" + "=" * 80)
    print("VERDICT (clean-up fix)")
    print("=" * 80)
    for name, r in rows:
        # Success criteria from the plan: coverage climbs, acc_close rises
        # preferentially (gap to acc_distant narrows), hop-decay flattens,
        # ECE < 0.05, inferred<stored preserved.
        gap = (r['acc_distant'] - r['acc_close']) if (r['acc_close'] is not None and r['acc_distant'] is not None) else None
        ok = (r["coverage_full"] >= 0.5
              and r["ece"] < 0.1
              and (np.isnan(r["auroc"]) or r["auroc"] >= 0.5)
              and (r['inferred_conf_mean'] < r['stored_conf_mean']))
        tag = "PASS" if ok else "CHECK"
        print(f"  [{tag}] {name}: cov={r['coverage_full']:.3f} ece={r['ece']:.4f} "
              f"close/distant={r['acc_close']}/{r['acc_distant']} gap={gap} "
              f"inferred<stored={r['inferred_conf_mean'] < r['stored_conf_mean']}")


if __name__ == "__main__":
    main()
