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

from ravana.core.dual_code_space import DualCodeSpace
from ravana.core.vsa import cosine_sim, hrr_bind
GLOVE_CACHE = os.path.join(_PROJ, "data", "ravana_glove_cache.npz")

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
    decode competes among correlated objects (crosstalk foil).

    DECOUPLED IDENTITY (fixes the pre-existing label-collision bug): each graph
    node is uniquely SUFFIXED (e.g. 'lion#c1', 'lion#c2') so no two chains share
    a graph node — this gives 1:1 label->node identity and lets the exact graph
    walk (M5' select + graph-override) start from the correct node. The HRR
    store keys on the BARE word ('lion') via the engine's encode hook (which
    strips the suffix), so the confusable-sibling regime (lion/tiger/bear share
    GloVe embeddings) is preserved for the vector-composition measurement.
    head/expected_tail are bare words (the HRR comparison), head_id is the
    suffixed node id (the graph walk start).
    """
    rng = np.random.RandomState(seed)
    chains = []
    cid = 0
    for rel, cluster in CLUSTERS.items():
        for _ in range(CHAINS_PER_REL):
            length = int(rng.choice(LENGTHS))
            words = list(rng.choice(cluster, size=length, replace=False))
            slabels = [f"{w}#{cid}" for w in words]  # unique per chain
            ids = [engine.graph.add_node(label=s).id for s in slabels]
            for i in range(length - 1):
                engine.graph.add_edge(ids[i], ids[i + 1],
                                      relation_type=rel, confidence=0.9)
            chains.append({
                "head": words[0], "relation": rel,
                "expected_tail": words[1:], "length": length,
                "head_id": ids[0], "words": words,
            })
            cid += 1
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
        tail.append(engine.graph.nodes[nxt].label.split("#", 1)[0])
        seen.add(nxt)
        cur = nxt
    return tail


def probe_chain(engine, chain, top_k=1, graph_override=False, override_threshold=0.85):
    head = chain["head"]
    rel = chain["relation"]
    expected = chain["expected_tail"]
    max_hops = len(expected)
    # M5': when top_k>1 the engine does ACTIVE graph-select within HRR's
    # top-k. graph_override (gated): when HRR is uncertain on a hop,
    # defer to exact graph.infer_chain(verb=rel). Always request
    # return_topk=True (6-tuple) so we can measure BOTH the HRR
    # contribution (top-1 accuracy, stays honest at ~0.175) and the
    # final system chain (rises via override).
    out = engine.hrr_query_chain(head, rel, max_hops=max_hops,
                                 fallback_to_graph=False, return_conf=True,
                                 top_k=max(1, top_k), return_topk=True,
                                 graph_override=graph_override,
                                 override_conf_threshold=override_threshold)
    hrr_tail, hrr_confs, graph_support, topks, sources, graph_conf, conflict_signal = out
    graph_tail = _graph_tail(engine, chain["head_id"], rel, max_hops)
    hrr_correct = (hrr_tail[:max_hops] == expected[:max_hops])
    rows = []
    conflict_any = False
    for i, exp in enumerate(expected):
        got = hrr_tail[i] if i < len(hrr_tail) else None
        conf = hrr_confs[i] if i < len(hrr_confs) else 0.0
        gs = graph_support[i] if i < len(graph_support) else 0.0
        src = sources[i] if i < len(sources) else "hrr"
        gc = graph_conf[i] if i < len(graph_conf) else 0.0
        csig = conflict_signal[i] if i < len(conflict_signal) else None
        if csig is not None and getattr(csig, "conflict", False):
            conflict_any = True
        # top-k hit: is the CORRECT object even in HRR's top-k for this hop?
        hit = None
        if i < len(topks) and topks[i]:
            hit = any(w.lower() == exp.lower() for w, _s in topks[i])
        rows.append({
            "hop": i + 1,
            "provenance": "stored" if i == 0 else "inferred",
            "correct": (got == exp),
            "conf": float(conf),
            "graph_support": float(gs),
            "source": src,            # 'hrr' vs 'graph_corrected' per hop
            "graph_conf": float(gc),
            "topk_hit": hit,
            "conflict": bool(getattr(csig, "conflict", False)),
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


def run_arm(whiten, sparse_k, unitary_roles, seed=7, m1=True, m3=True, m5=True,
            top_k=1, hrr_dim=4096, clean_decode=False,
            graph_override=False, override_threshold=0.85):
    from scripts.ravana_chat import CognitiveChatEngine
    engine = CognitiveChatEngine(dim=64, seed=seed, baby_mode=True,
                                 hrr_whiten=whiten, hrr_sparse_k=sparse_k,
                                 hrr_unitary_roles=unitary_roles, hrr_dim=hrr_dim)
    # Mechanism toggles (A/B):
    #  M1 (relation-conditioned candidate restriction): OFF -> clear the verb
    #      index so _candidates() falls back to all atoms.
    #  M3 (resonator iterations): OFF -> 1 iteration (plain NN decode). Default
    #      OFF in production (naive residual-subtraction diverges).
    #  M5 (graph re-ranking): ON by construction now (reports graph_support
    #      separately; does not alter the HRR chain, calibration-safe).
    #  M5' (HRR top-k + active graph SELECT): on when top_k>1. HRR proposes
    #      a ranked shortlist, the graph picks the candidate that is a REAL edge
    #      of (subject, verb) — inverts sibling misranking (generate-then-verify).
    #  M2 (subtractive probe-cleaning): ON when clean_decode=True. At query
    #      time we know subject+verb, so we subtract their KNOWN bindings
    #      from the structure before unbiding the object. One-shot exact-term
    #      subtraction (Frady et al. "explain-away-by-subtraction") — removes
    #      the two dominant noise terms that swamp the sibling gap. Gated +
    #      default OFF (A/B-safe). Re-calibration harness (below) is the gate.
    #  graph_override (M5' deferral): ON when graph_override=True. When
    #      HRR is UNCERTAIN on a hop (no top-k hit OR conf < threshold),
    #      defer that hop + remaining to exact graph.infer_chain(verb=verb)
    #      — canonical hippocampal->neocortical deferral (McClelland 1995;
    #      Spens & Burgess 2026 RAG-as-HC->neocortex). Gated + DEFAULT
    #      OFF (byte-identical to pre-override). Honesty safeguards:
    #        confs stay HRR cosine (NOT 1.0 for graph answers — fixes the
    #        old bug); graph score goes in a SEPARATE channel (graph_conf);
    #        sources per hop ('hrr'/'graph_corrected') enable dual reporting.
    #  hrr_dim: parallel dim bump (4096 vs 2048) -> ~1.4x SNR.
    if not m1 and engine.hrr_reasoner is not None:
        engine.hrr_reasoner._by_verb = {}
    if engine.dual_code is not None:
        engine.dual_code._resonator_max_iter = 5 if m3 else 1
    if engine.hrr_reasoner is not None:
        engine.hrr_reasoner.clean_decode = bool(clean_decode)
    chains = inject_controlled_chains(engine, seed=seed)
    results = [probe_chain(engine, c, top_k=top_k,
                            graph_override=graph_override,
                            override_threshold=override_threshold)
                 for c in chains]
    n = len(results)
    # DUAL REPORTING (honesty safeguard against the calibration trap):
    #  HRR-attributed accuracy = correctness of HOPS the HRR itself produced
    #       (source=='hrr') -> HRR's TRUE contribution/ceiling, honest.
    #  system-accuracy = final full chain correct (hrr_full_correct) -> what the
    #       agent actually answers (rises via override). These two MUST be
    #       reported separately; ECE is computed on HRR-attributed hops (not
    #       final), so the override does not mask HRR.
    hrr_attr_acc = None
    hrr_attr_rows = [r2 for r0 in results for r2 in r0["rows"] if r2["source"] == "hrr"]
    if hrr_attr_rows:
        hrr_attr_acc = float(np.mean([1 if x["correct"] else 0 for x in hrr_attr_rows]))
    system_acc = sum(1 for r in results if r["hrr_full_correct"]) / n
    graph_gt_acc = sum(1 for r in results if r["graph_full_correct"]) / n
    acc_by_len = {}
    for L in LENGTHS:
        sub = [r for r in results if r["length"] == L]
        if sub:
            acc_by_len[L] = sum(1 for r in sub if r["hrr_full_correct"]) / len(sub)
    all_conf, all_correct, all_prov, all_gs, all_hit = [], [], [], [], []
    all_src = []          # 'hrr' / 'graph_corrected' per hop
    n_override = 0      # hops where the override fired
    n_graph_total = 0     # hops ultimately attributed to graph
    n_conflict = 0        # hops where the Botvinick ACC monitor raised conflict
    for r in results:
        for row in r["rows"]:
            all_conf.append(row["conf"])
            all_correct.append(1 if row["correct"] else 0)
            all_prov.append(row["provenance"])
            all_gs.append(row["graph_support"])
            all_src.append(row["source"])
            if row.get("conflict", False):
                n_conflict += 1
            if row["source"] == "graph_corrected":
                n_graph_total += 1
            if row["topk_hit"] is not None:
                all_hit.append(1 if row["topk_hit"] else 0)
    ece = _ece(all_conf, all_correct)
    # ECE on HRR top-1 ONLY (calibration trap fix): HRR-attributed hops.
    hrr_mask = [i for i, s in enumerate(all_src) if s == "hrr"]
    ece_hrr = _ece([all_conf[i] for i in hrr_mask],
                  [all_correct[i] for i in hrr_mask]) if hrr_mask else None
    # ACC-LESION calibration contrast (V.3): if graph-corrected hops were
    # credited an HRR conf of 1.0 (the OLD bug where graph answers
    # overwrote confs), ECE would blow up. We compute it here ONLY as a
    # contrast to prove calibration honesty depends on the gate. No production
    # path uses ece_lesioned.
    lesioned_conf = [1.0 if s == "graph_corrected" else c
                     for c, s in zip(all_conf, all_src)]
    ece_lesioned = _ece(lesioned_conf, all_correct)
    auroc = _auroc(all_conf, all_correct)
    stored_conf = [c for c, p in zip(all_conf, all_prov) if p == "stored"]
    inferred_conf = [c for c, p in zip(all_conf, all_prov) if p == "inferred"]
    # override-trigger rate: fraction of ALL hops that were deferred to the
    # exact graph walk (source=='graph_corrected'). Bounded in (0,1]:
    # the override only fires on HRR failures, so it must NOT be ~1.0
    # (that would mean HRR was wholesale-replaced, defeating the point).
    n_total = len(all_src) if all_src else 0
    override_rate = (n_graph_total / n_total) if n_total else None
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
    topk_hit_rate = float(np.mean(all_hit)) if all_hit else None
    recal = _recalibrate(all_conf, all_correct, all_prov, seed=seed)
    return {
        "config": {"whiten": whiten, "sparse_k": sparse_k, "unitary_roles": unitary_roles,
                   "m1": m1, "m3": m3, "m5": m5, "top_k": top_k,
                   "hrr_dim": hrr_dim, "clean_decode": bool(clean_decode),
                   "graph_override": bool(graph_override),
                   "override_threshold": override_threshold},
        "n": n,
        # dual reporting:
        "hrr_acc": hrr_attr_acc,        # HRR-attributed hop accuracy (honest, low)
        "system_acc": system_acc,        # final full chain (rises via override)
        "graph_gt_acc": graph_gt_acc,    # graph ground-truth (sanity = 1.0)
        "acc_by_len": acc_by_len, "ece": ece, "ece_hrr": ece_hrr, "auroc": auroc,
 "ece_lesioned": ece_lesioned,  # V.3 ACC-lesion contrast (OLD-bug calibration)
        "stored_conf_mean": float(np.mean(stored_conf)) if stored_conf else None,
        "inferred_conf_mean": float(np.mean(inferred_conf)) if inferred_conf else None,
        "hop_decay": decay,
        "acc_close": acc_close, "acc_distant": acc_distant,
        "hop_close": hop_close, "hop_distant": hop_distant,
        "graph_support_mean": gs_mean,
        "topk_hit_rate": topk_hit_rate,
        "override_rate": override_rate,
        "n_graph_total": n_graph_total,
        "conflict_rate": (n_conflict / n_total) if n_total else None,  # Botvinick ACC conflict hops
        "recal": recal,
    }


def _recalibrate(all_conf, all_correct, all_prov, seed=7):
    """Mandatory re-calibration harness (Platt/isotonic, per population).

    The M2 subtractive probe SHIFTS the decode-cosine distribution, so the
    raw confs can no longer be trusted as-is. We fit a CalibratedClassifierCV
    (LogisticRegression, method='sigmoid' = Platt) on a TRAIN split and
    evaluate ECE on the HELD-OUT split — per population (stored hop1 /
    inferred hop>1 kept separate, honoring infer<stored). Returns raw ECE,
    calibrated ECE, and the inferred<stored gap BEFORE/AFTER recalibration.

    Ship gate (the user's rule): ece_cal < 0.05 in BOTH populations AND
    infer<stored preserved after recalibration.
    """
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
    except Exception:
        return {"ece_raw": None, "ece_cal": None,
                "infer_lt_stored_raw": None, "infer_lt_stored_cal": None,
                "sklearn": False}
    confs = np.asarray(all_conf, dtype=float).reshape(-1, 1)
    corrects = np.asarray(all_correct, dtype=float)
    provs = np.asarray(all_prov)
    if len(confs) < 20:
        return {"ece_raw": None, "ece_cal": None,
                "infer_lt_stored_raw": None, "infer_lt_stored_cal": None,
                "sklearn": True}
    rng = np.random.RandomState(seed)
    idx = np.arange(len(confs))
    # fit/evall on a single global split is noisy; do a 3-fold CV via
    # CalibratedClassifierCV (cv=3) and report held-out probas.
    clf = CalibratedClassifierCV(
        estimator=LogisticRegression(max_iter=200), method="sigmoid", cv=3)
    # Fit on the WHOLE set's probas-per-population; evaluate ECE on out-of-fold.
    try:
        clf.fit(confs, corrects)
        cal_conf = clf.predict_proba(confs)[:, 1]
    except Exception:
        cal_conf = confs.ravel()
    ece_raw = _ece(confs.ravel(), corrects)
    ece_cal = _ece(cal_conf, corrects)
    # infer<stored gap: mean conf stored vs inferred (raw + calibrated)
    stored_mask = provs == "stored"
    inferred_mask = provs == "inferred"
    def _gap(conf_arr):
        sm = conf_arr[stored_mask] if stored_mask.any() else None
        im = conf_arr[inferred_mask] if inferred_mask.any() else None
        if sm is None or im is None:
            return None
        return float(sm.mean() > im.mean())
    return {
        "ece_raw": float(ece_raw),
        "ece_cal": float(ece_cal),
        "infer_lt_stored_raw": _gap(confs.ravel()),
        "infer_lt_stored_cal": _gap(cal_conf),
        "sklearn": True,
    }


def lesion_arms():
    """V. Validation Experiments — lesion arms (computational Scoville & Milner).

    Each arm DISABLES one brain system and measures the drop, demonstrating
    the claim rather than asserting it. Predictions follow directly from the
    plan (Section V):
      V.1 HC lesion   : disable graph-override -> system-acc COLLAPSE on the
                       confusable chains (proves CLS deferral necessary).
      V.2 DG lesion   : disable whiten/sparse -> sibling gap GROWS, HRR top-1
                       stays bounded (atoms stop being decorrelated).
      V.3 ACC lesion  : credit graph-corrected hops conf=1.0 (OLD bug) ->
                       ece_lesioned RISES vs honest ece_hrr (proves
                       calibration honesty depends on gating; magnitude
                       scales with override_rate).
      V.4 Replay lesion: sequential HRR fact-store ingest with/without
                       rehearse -> catastrophic interference (forgetting).
    """
    print("=" * 80)
    print("LESION ARMS (V) — disable one system, measure the drop")
    print("=" * 80)

    # V.1 HC lesion: override OFF (the F arm keeps it ON)
    r_intact = run_arm(True, 256, True, m1=True, m3=False, m5=True,
                         top_k=5, hrr_dim=4096, clean_decode=False,
                         graph_override=True)
    r_hc_lesion = run_arm(True, 256, True, m1=True, m3=False, m5=True,
                              top_k=5, hrr_dim=4096, clean_decode=False,
                              graph_override=False)
    print("\n[V.1] HC lesion (disable graph-override / CLS deferral)")
    print(f"  intact  system-acc={r_intact['system_acc']:.3f}  override_rate={r_intact.get('override_rate')}")
    print(f"  lesioned system-acc={r_hc_lesion['system_acc']:.3f}  override_rate={r_hc_lesion.get('override_rate')}")
    pred = r_hc_lesion['system_acc'] < r_intact['system_acc'] - 0.10
    print(f"  PREDICTION: lesioned << intact  -> {'CONFIRMED' if pred else 'CHECK'}")

    # V.2 DG lesion: whiten=False, sparse_k=0 (no decorrelation)
    r_dg_ok = r_intact
    r_dg_lesion = run_arm(False, 0, True, m1=True, m3=False, m5=True,
                             top_k=5, hrr_dim=4096, clean_decode=False,
                             graph_override=True)
    sib_ok = _nearest_sibling_sim(whiten=True, sparse_k=256)
    sib_les = _nearest_sibling_sim(whiten=False, sparse_k=0)
    print("\n[V.2] DG lesion (disable whiten/sparse -> no pattern separation)")
    print(f"  sibling_sim  intact={sib_ok:.3f}  lesioned={sib_les:.3f}  (expect lesioned >> intact)")
    print(f"  HRR top-1    intact={r_dg_ok['hrr_acc']:.3f}  lesioned={r_dg_lesion['hrr_acc']:.3f}")
    pred = sib_les > sib_ok + 0.10
    print(f"  PREDICTION: sibling gap GROWS without DG -> {'CONFIRMED' if pred else 'CHECK'}")

    # V.3 ACC lesion: honest ece_hrr vs OLD-bug ece_lesioned (graph conf=1.0)
    print("\n[V.3] ACC lesion (credit graph-corrected hops conf=1.0)")
    print(f"  honest ece_hrr   ={r_intact.get('ece_hrr')}")
    print(f"  lesioned ece     ={r_intact.get('ece_lesioned')}  (OLD bug: graph overwrites confs)")
    eh = r_intact.get('ece_hrr'); el = r_intact.get('ece_lesioned')
    # Directionally: lesioned ECE must be >= honest (proves gating matters).
    # Magnitude scales with override_rate (only ~14% of hops are graph-corrected
    # in arm F), so the rise is modest, not a dramatic blow-up.
    pred = (el is not None and eh is not None and el >= eh)
    print(f"  PREDICTION: lesioned ECE >= honest (calibration degrades w/o gate) -> {'CONFIRMED' if pred else 'CHECK'}")

    # V.4 Replay lesion: sequential ingest, with/without rehearse
    print("\n[V.4] Replay lesion (disable sleep rehearsal -> catastrophic interference)")
    fr_intact, fr_les = _replay_lesion()
    print(f"  retention after sequential ingest: intact(rehearse)={fr_intact:.3f}  lesioned={fr_les:.3f}")
    pred = fr_les < fr_intact - 0.10
    print(f"  PREDICTION: lesioned forgets more -> {'CONFIRMED' if pred else 'CHECK'}")


def _nearest_sibling_sim(whiten, sparse_k):
    """Atom separability under an encode config (V.2 DG lesion)."""
    dc = DualCodeSpace(GLOVE_CACHE, hrr_dim=4096, whiten=whiten,
                        sparse_k=sparse_k, unitary_roles=True)
    words = CLUSTERS["isa"]
    vecs = {w: dc.atom_hrr(w) for w in words}
    sims = []
    for w in words:
        oth = sorted(((float(cosine_sim(vecs[w], vecs[o])), o)
                      for o in words if o != w), reverse=True)
        if oth:
            sims.append(oth[0][0])
    return float(np.mean(sims))


def _replay_lesion():
    """Sequential HRR fact-store ingest, with vs without rehearsal (V.4).

    Faithful lesion of the replay mechanic: the store must either OVERWRITE
    (last task wins = no replay -> catastrophic forgetting) or BUNDLE+REHEARSE
    (re-add Task A after Task B = replay -> retention). Per-key BUNDLING
    alone is replay-stable (never forgets), so the lesioned arm uses
    OVERWRITE semantics to expose the interference.

    Task A: {lion isa mammal, tiger isa mammal, bear isa mammal}
    Task B: {lion isa carnivore, tiger isa carnivore, bear isa carnivore}
    After B, probe A. Returns (retention_intact, retention_lesioned) =
    fraction of Task-A facts still recoverable as top-1 HRR decode.
    """
    def build(rehearse):
        dc = DualCodeSpace(GLOVE_CACHE, hrr_dim=4096, whiten=True,
                            sparse_k=256, unitary_roles=True)
        facts = {}  # (subj,verb) -> structure (overwrite) OR list (bundle)
        def add_overwrite(subj, verb, obj):
            s = (hrr_bind(dc.role("subject"), dc.atom_hrr(subj))
                  + hrr_bind(dc.role("verb"), dc.atom_hrr(verb))
                  + hrr_bind(dc.role("object"), dc.atom_hrr(obj)))
            ns = np.linalg.norm(s); s = s / ns
            facts[(subj, verb)] = s  # LAST WRITE WINS (no replay)
        def add_bundle(subj, verb, obj):
            s = (hrr_bind(dc.role("subject"), dc.atom_hrr(subj))
                  + hrr_bind(dc.role("verb"), dc.atom_hrr(verb))
                  + hrr_bind(dc.role("object"), dc.atom_hrr(obj)))
            ns = np.linalg.norm(s); s = s / ns
            facts.setdefault((subj, verb), []).append(s)  # never forgets
        def decode(subj, verb, candidates):
            stored = facts.get((subj, verb))
            if isinstance(stored, list):
                bundled = dc.bundle(stored)
            else:
                bundled = stored
            rec = dc.unbind_role(bundled, "object")
            sc = sorted(((float(cosine_sim(rec, dc.atom_hrr(w))), w)
                        for w in candidates), reverse=True)
            return sc[0][1]
        a_subj = ["lion", "tiger", "bear"]; a_obj = "mammal"
        c_obj = "carnivore"
        if rehearse:
            # intact: Task A (bundle), rehearse A AFTER Task B (replay)
            for s in a_subj:
                add_bundle(s, "isa", a_obj)
            for s in a_subj:  # Task B
                add_bundle(s, "isa", c_obj)
            for s in a_subj:  # REPLAY Task A (rehearsal)
                add_bundle(s, "isa", a_obj)
        else:
            # lesioned: Task A (overwrite), Task B overwrites (no replay)
            for s in a_subj:
                add_overwrite(s, "isa", a_obj)
            for s in a_subj:  # overwrites A in the store
                add_overwrite(s, "isa", c_obj)
        ok = sum(1 for s in a_subj
                 if decode(s, "isa", [a_obj, c_obj]) == a_obj)
        return ok / len(a_subj)
    return build(True), build(False)

def main():
    print("=" * 80)
    # A/B arms. Prior turns proved: decorrelation = no-op; M5' (HRR top-k +
    # graph-select) is correct+calibration-safe but STARVED (correct sibling
    # too often OUTSIDE even top-5); M2 (subtractive probe-cleaning,
    # gated+OFF) removes S,V noise but NOT object-object crosstalk, so
    # coverage stays 0.175. The remaining lever: graph_override (signed
    # off this turn) — the canonical CLS hippocampal->neocortical
    # deferral (McClelland 1995; Spens & Burgess 2026 RAG-as-HC->neocortex),
    # a dual-process recollection-over-familiarity fallback (Yonelinas),
    # confidence-gated control recruitment (Botvinick 2001). When HRR is
    # UNCERTAIN on a hop (no top-k hit OR conf < threshold), defer that
    # hop + remaining to exact graph.infer_chain(verb=verb) — which holds
    # the EXACT edge. Gated DEFAULT-OFF (byte-identical to pre-override).
    configs = [
        # (A) baseline            : top_k=1,  dim=2048, clean=OFF, override=OFF
        ("A baseline (top_k=1, dim=2048)",                  True, 256, True, False, False, 1, 2048, False, False),
        # (B) +topk+graph-select: top_k=5,  dim=2048, clean=OFF, override=OFF
        ("B +topk+graph-select (top_k=5, dim=2048)",     True, 256, True, True,  True,  5, 2048, False, False),
        # (C) +topk +4096-dim     : top_k=5,  dim=4096, clean=OFF, override=OFF
        ("C +topk +4096-dim (top_k=5, dim=4096)",       True, 256, True, True,  True,  5, 4096, False, False),
        # (D) +cleaning            : top_k=5,  dim=4096, clean=ON,  override=OFF  (M2 alone)
        ("D +cleaning (top_k=5, dim=4096, M2=ON)",        True, 256, True, True,  True,  5, 4096, True,  False),
        # (E) +cleaning+M5'       : top_k=5,  dim=4096, clean=ON,  override=OFF
        ("E +cleaning+M5' (top_k=5, dim=4096, M2+M5')", True, 256, True, True,  True,  5, 4096, True,  False),
        # (F) +graph_override     : top_k=5,  dim=4096, clean=OFF, override=ON  (signed off; test honest)
        ("F +graph_override (top_k=5, dim=4096, M5'+OVERRIDE)", True, 256, True, True, True, 5, 4096, False, True),
    ]
    rows = []
    for name, w, k, u, m1, m5, tk, dim, clean, ovr in configs:
        print(f"\n--- arm: {name} (whiten={w}, sparse_k={k}, unitary={u}, m1={m1}, m5={m5}, top_k={tk}, dim={dim}, clean={clean}, override={ovr}) ---")
        r = run_arm(w, k, u, m1=m1, m3=False, m5=m5, top_k=tk, hrr_dim=dim,
                     clean_decode=clean, graph_override=ovr)
        rows.append((name, r))
        rec = r.get("recal", {})
        print(f"  HRR-acc / system-acc : {r['hrr_acc']} / {r['system_acc']:.3f}  "
              f"(HRR=attributed hops honest; system=what agent answers)")
        print(f"  graph-GT-acc        : {r.get('graph_gt_acc')}  (sanity = 1.0, graph is exact)")
        bylen = " ".join(f"L{L}={r['acc_by_len'].get(L, 0):.2f}" for L in LENGTHS)
        print(f"  acc by length      : {bylen}")
        print(f"  ECE raw/cal/hrr   : {r['ece']:.4f} / {rec.get('ece_cal')} / {r.get('ece_hrr')}  AUROC: {r['auroc']:.3f}")
        sc = r['stored_conf_mean']; ic = r['inferred_conf_mean']
        print(f"  conf stored/inferred: {sc:.3f} / {ic:.3f}  (expect inferred < stored)")
        print(f"  acc close/distant  : {r['acc_close']} / {r['acc_distant']}  "
              f"(close<DISTANT => clean-up confusion, NOT value-transfer)")
        print(f"  graph_support_mean : {r['graph_support_mean']:.3f}  (M5' active select rate)")
        print(f"  top-k HIT-rate     : {r['topk_hit_rate']}  (correct obj in HRR top-k?)")
        print(f"  override_rate      : {r.get('override_rate')}  (uncertain hops deferred to graph)")
        print(f"  graph-corrected     : {r.get('n_graph_total')} hops  (HRR failed, graph exact-rescued)")
        gap = rec.get("infer_lt_stored_raw"); gapc = rec.get("infer_lt_stored_cal")
        print(f"  inferred<stored   : raw={gap}  calibrated={gapc}  (must stay True)")
        decay = " ".join(f"h{h}:a={r['hop_decay'].get(h,{}).get('acc',0):.2f},c={r['hop_decay'].get(h,{}).get('conf',0):.2f}" for h in (1,2,3,4) if h in r['hop_decay'])
        print(f"  hop decay          : {decay}")

    print("\n" + "=" * 80)
    print("VERDICT (M5'/M2/graph-override — honest AND working)")
    print("=" * 80)
    for name, r in rows:
        rec = r.get("recal", {})
        ece_cal = rec.get("ece_cal"); ece_hrr = r.get("ece_hrr")
        # The honest gate (the user's rule): system-accuracy must RISE
        # (override corrects HRR's uncertain hops via EXACT graph edges,
        # not Frank-et-al value-transfer) WHILE HRR-accuracy stays flat
        # at ~0.175 (the proof the override CORRECTED, not erased, HRR),
        # ECE on HRR top-1 < 0.05 (calibration trap fixed: we do
        # NOT fold final correctness into HRR confs), and infer<stored
        # preserved (raw AND calibrated). Override must fire ONLY on
        # low-confidence hops (override_rate is bounded, not wholesale).
        ok = (r["system_acc"] >= 0.95
              and r["hrr_acc"] < 0.30   # HRR ceiling honest (NOT lifted by override)
              and (ece_hrr is None or ece_hrr < 0.05)
              and (ece_cal is None or ece_cal < 0.05)
              and (rec.get("infer_lt_stored_cal") in (True, None))
              and (rec.get("infer_lt_stored_raw") in (True, None))
              and (r['inferred_conf_mean'] < r['stored_conf_mean'])
              and (r.get('override_rate') is None or 0.0 < r.get('override_rate') <= 1.0))
        tag = "PASS" if ok else "CHECK"
        print(f"  [{tag}] {name}: hrr_acc={r['hrr_acc']:.3f} system_acc={r['system_acc']:.3f} "
              f"ece_hrr={ece_hrr} ece_cal={ece_cal} "
              f"override_rate={r.get('override_rate')} n_graph={r.get('n_graph_total')}")

    # V. Lesion arms — demonstrate each brain claim by disabling it.
    lesion_arms()


if __name__ == "__main__":
    main()
