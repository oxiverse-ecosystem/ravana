"""
Compositional Diagnostics & Ablation Experiment
===============================================
This script performs a rigorous diagnostic audit of the hybrid compositional pathway:
1. Ablation comparison: Graph-Only vs. Encoder-Only vs. Hybrid (Standard & Gated).
2. Traversal Depth (Hop) Sweep: max_depth = 1, 2, 3, 4.
3. Seeding Gating Analysis: Margin-based seeding (best vs. runner-up).
4. Failure Class Separation: Semantic Ambiguity vs. Graph Drift.
"""

import os
import sys
import pickle
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import WordTokenizer

# ──────────────────────────────────────────────────────────────────────────
# Facts and Test Cases
# ──────────────────────────────────────────────────────────────────────────

FACTS = [
    # Chain 1: warmth -> kindness -> trust -> cooperation
    ("kindness", "causes", "trust"),
    ("trust", "causes", "cooperation"),
    
    # Chain 2: light -> hope -> courage -> victory
    ("hope", "causes", "courage"),
    ("courage", "causes", "victory"),
    
    # Chain 3: combustion -> resentment -> hostility -> conflict
    ("resentment", "causes", "hostility"),
    ("hostility", "causes", "conflict"),
    
    # Chain 4: gravity -> loyalty -> support -> stability
    ("loyalty", "causes", "support"),
    ("support", "causes", "stability"),
    
    # Distractor/noise edges
    ("trust", "causes", "vulnerability"),
    ("courage", "causes", "danger"),
    ("hostility", "causes", "isolation"),
    ("support", "causes", "obligation"),
    
    # Additional noise facts
    ("cat", "has", "tail"),
    ("dog", "has", "tail"),
    ("cat", "has", "fur"),
    ("dog", "has", "fur"),
    ("virus", "causes", "illness"),
    ("illness", "causes", "absence"),
    ("bug", "causes", "crash"),
    ("crash", "causes", "outage"),
]

TEST_CASES = [
    {
        "query": "warmth causes",
        "expected": "cooperation",
        "analog_pair": ("warmth", "kindness"),
        "expected_chain": ["kindness", "trust", "cooperation"],
        "category": "High Sim (Analogy)"
    },
    {
        "query": "light causes",
        "expected": "victory",
        "analog_pair": ("light", "hope"),
        "expected_chain": ["hope", "courage", "victory"],
        "category": "High-Mod Sim Boundary"
    },
    {
        "query": "combustion causes",
        "expected": "conflict",
        "analog_pair": ("combustion", "resentment"),
        "expected_chain": ["resentment", "hostility", "conflict"],
        "category": "Mod-Low Sim Boundary"
    },
    {
        "query": "gravity causes",
        "expected": "stability",
        "analog_pair": ("gravity", "loyalty"),
        "expected_chain": ["loyalty", "support", "stability"],
        "category": "Low Sim Boundary"
    }
]

# ──────────────────────────────────────────────────────────────────────────
# Vector Math & Initialization Helpers
# ──────────────────────────────────────────────────────────────────────────

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def proto(model: RLMv2, tok: WordTokenizer, word: str) -> np.ndarray:
    tid = tok.word_to_id.get(word)
    if tid is None:
        raise KeyError(word)
    emb = model.token_embed.weight.data[tid]
    lat, *_ = model._encoder_forward_full(emb)
    return lat


def make_similar_vectors(dim: int, cos_theta: float) -> tuple[np.ndarray, np.ndarray]:
    a = np.random.randn(dim).astype(np.float32)
    a /= np.linalg.norm(a)
    
    b_orth = np.random.randn(dim).astype(np.float32)
    b_orth -= np.dot(b_orth, a) * a
    b_orth /= np.linalg.norm(b_orth)
    
    b = cos_theta * a + np.sqrt(1.0 - cos_theta**2) * b_orth
    return a, b


def expand_vocabulary_and_embeddings(model: RLMv2, tok: WordTokenizer, words: list[str]):
    for w in words:
        tok.encode(w)
    vocab_size = tok.vocab_size
    dim = model.embed_dim
    old_weight = model.token_embed.weight.data
    old_size = old_weight.shape[0]
    
    if vocab_size > old_size:
        new_weight = np.random.randn(vocab_size, dim).astype(np.float32) * np.sqrt(2.0 / dim)
        new_weight[:old_size] = old_weight
        
        from ravana_ml.embedder import LearnedEmbedder
        embedder = LearnedEmbedder(dim=dim)
        all_words = list(tok.word_to_id.keys())
        try:
            embedder.fit(all_words)
        except Exception:
            pass
            
        for w in words:
            tid = tok.word_to_id[w]
            if tid >= old_size:
                new_weight[tid] = embedder.encode(w)
                
        from ravana_ml.tensor import StateTensor, Parameter
        model.token_embed.weight = Parameter(StateTensor(new_weight))
        model.token_embed.num_embeddings = vocab_size
        model.token_embed._rebuild_raw_cache()
        model.vocab_size = vocab_size
        model._token_embed_norms = None


def inject_precise_embeddings(model: RLMv2, tok: WordTokenizer):
    dim = model.embed_dim
    targets = {
        ("warmth", "kindness"): 0.85,
        ("light", "hope"): 0.70,
        ("combustion", "resentment"): 0.45,
        ("gravity", "loyalty"): 0.20,
    }
    for (w1, w2), cos_theta in targets.items():
        tid1 = tok.word_to_id[w1]
        tid2 = tok.word_to_id[w2]
        v1, v2 = make_similar_vectors(dim, cos_theta)
        model.token_embed.weight.data[tid1] = v1
        model.token_embed.weight.data[tid2] = v2
    model._token_embed_norms = None


# ──────────────────────────────────────────────────────────────────────────
# Ablation Algorithms
# ──────────────────────────────────────────────────────────────────────────

def graph_only_predict(model: RLMv2, tok: WordTokenizer, query: str, max_depth: int = 3) -> list[tuple[str, float]]:
    """Pure Graph Traversal: start only if subject has a direct concept node in the graph."""
    parts = query.split()
    subj_word, rel_word = parts[0], parts[1]
    
    causal = {"causes", "cause", "leads", "produces", "creates"}
    rel_type = "causal" if rel_word in causal else None
    
    subj_tid = tok.word_to_id.get(subj_word)
    if subj_tid is None:
        return []
    
    bindings = model.binding_map.get_concepts(subj_tid, min_confidence=0.1)
    if not bindings:
        return []  # Subject has no concept node in the graph
        
    subject_cid = bindings[0].concept_id
    
    activations = {subject_cid: 1.0}
    frontier = [subject_cid]
    
    for depth in range(1, max_depth + 1):
        next_frontier = []
        for nid in frontier:
            act = activations[nid]
            for tgt_id, edge in model.graph.get_outgoing(nid):
                if rel_type and edge.relation_type != rel_type:
                    continue
                prop = act * edge.weight
                if tgt_id not in activations or prop > activations[tgt_id]:
                    activations[tgt_id] = prop
                    next_frontier.append(tgt_id)
        frontier = next_frontier
        
    results = []
    for cid, act in activations.items():
        if cid == subject_cid:
            continue
        tokens = model.binding_map.get_tokens(cid, 0.0)
        for b in tokens:
            word = tok.decode([b.token_id])
            if word not in (subj_word, rel_word) and not word.startswith("?"):
                results.append((word, act))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def encoder_only_predict(model: RLMv2, tok: WordTokenizer, query: str, facts: list) -> list[tuple[str, float]]:
    """Pure Encoder Similarity: find nearest subject in 32d space with same relation, return its object (1-hop)."""
    parts = query.split()
    subj_word, rel_word = parts[0], parts[1]
    
    subj_tid = tok.word_to_id.get(subj_word)
    if subj_tid is None:
        return []
    
    lat_query = proto(model, tok, subj_word)
    
    # Collect all direct facts matching this relation
    pairs = [(s, o) for s, r, o in facts if r == rel_word]
    
    scored = []
    for s, o in pairs:
        s_tid = tok.word_to_id.get(s)
        if s_tid is None:
            continue
        lat_s = proto(model, tok, s)
        sim = cosine(lat_query, lat_s)
        scored.append((o, sim))
        
    # Sort and deduplicate expected targets
    scored.sort(key=lambda x: x[1], reverse=True)
    seen = set()
    results = []
    for o, sim in scored:
        if o not in seen:
            seen.add(o)
            results.append((o, sim))
    return results


def hybrid_predict_gated(
    model: RLMv2,
    tok: WordTokenizer,
    query: str,
    k_neighbors: int = 3,
    max_depth: int = 3,
    gate_mode: str = "standard"  # "standard", "strict_margin", "relative_threshold", "weighted", "margin_multi"
) -> list[tuple[str, float]]:
    """Hybrid traversal with different vector seeding gating strategies."""
    parts = query.split()
    subj_word, rel_word = parts[0], parts[1]
    
    causal = {"causes", "cause", "leads", "produces", "creates"}
    rel_type = "causal" if rel_word in causal else None
    
    subj_tid = tok.word_to_id.get(subj_word)
    if subj_tid is None:
        return []
        
    lat_query = proto(model, tok, subj_word)
    
    # 1. Gather all candidates in graph with 32d similarities
    scored_neighbors = []
    for word, tid in tok.word_to_id.items():
        if word == subj_word:
            continue
        bindings = model.binding_map.get_concepts(tid, min_confidence=0.1)
        if not bindings:
            continue
        cid = bindings[0].concept_id
        lat_word = proto(model, tok, word)
        sim = cosine(lat_query, lat_word)
        scored_neighbors.append((cid, sim, word))
        
    scored_neighbors.sort(key=lambda x: x[1], reverse=True)
    
    # 2. Apply Gating
    seeds = []
    if scored_neighbors:
        best_sim = scored_neighbors[0][1]
        
        if gate_mode == "strict_margin":
            # Only seed if best neighbor is significantly closer than second best
            if len(scored_neighbors) > 1:
                margin = best_sim - scored_neighbors[1][1]
            else:
                margin = best_sim
            
            # If margin is wide, seed only the best. Otherwise, seed none (too ambiguous)
            if margin >= 0.15 and best_sim >= 0.50:
                seeds = [scored_neighbors[0]]
            else:
                seeds = []
                
        elif gate_mode == "relative_threshold":
            # Only seed from neighbors within 85% similarity of the best neighbor
            threshold = 0.85 * best_sim
            seeds = [n for n in scored_neighbors if n[1] >= threshold][:k_neighbors]
            
        elif gate_mode == "weighted":
            # Seed top-5 neighbors to allow alternative path exploration
            seeds = scored_neighbors[:5]
            
        elif gate_mode == "margin_multi":
            # Confidence-aware gating using similarity margins:
            # If margin to runner-up is large, we have a clear winner (seed only top-1).
            # Otherwise, seed top-5 because of high ambiguity.
            if len(scored_neighbors) > 1:
                margin = best_sim - scored_neighbors[1][1]
            else:
                margin = best_sim
                
            if margin >= 0.15 and best_sim >= 0.50:
                seeds = [scored_neighbors[0]]
            else:
                seeds = scored_neighbors[:5]
            
        else:  # "standard" top-k
            seeds = scored_neighbors[:k_neighbors]
            
    # 3. Traversal BFS
    activations = {}
    if gate_mode in ("weighted", "margin_multi"):
        # Apply softmax weighting over similarities (with temperature = 0.15)
        if seeds:
            sims = np.array([n[1] for n in seeds])
            temp = 0.15
            # Shift similarities for numerical stability
            exp_sims = np.exp((sims - np.max(sims)) / temp)
            weights = exp_sims / np.sum(exp_sims)
            for (cid, _, _), w in zip(seeds, weights):
                activations[cid] = float(w)
        else:
            pass
    else:
        for cid, sim, _ in seeds:
            activations[cid] = sim
        
    frontier = [cid for cid, _, _ in seeds]
    for depth in range(1, max_depth + 1):
        next_frontier = []
        for nid in frontier:
            act = activations[nid]
            for tgt_id, edge in model.graph.get_outgoing(nid):
                if rel_type and edge.relation_type != rel_type:
                    continue
                prop = act * edge.weight
                if tgt_id not in activations or prop > activations[tgt_id]:
                    activations[tgt_id] = prop
                    next_frontier.append(tgt_id)
        frontier = next_frontier
        
    # 4. Decode results
    seed_words = {n[2] for n in seeds}
    results = []
    for cid, act in activations.items():
        tokens = model.binding_map.get_tokens(cid, 0.0)
        for b in tokens:
            word = tok.decode([b.token_id])
            if word not in seed_words and not word.startswith("?"):
                results.append((word, act))
                
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ──────────────────────────────────────────────────────────────────────────
# Main Diagnostics Evaluation
# ──────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("RLMv2 HYBRID PATHWAY SCIENTIFIC DIAGNOSTICS")
    print("=" * 80)
    
    checkpoint_path = os.path.join(SCRIPT_DIR, "experiment_results", "encoder_32d_fixed.pkl")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
        
    with open(checkpoint_path, 'rb') as f:
        state = pickle.load(f)
        
    model = RLMv2(
        vocab_size=state["vocab_size"],
        embed_dim=state["embed_dim"],
        concept_dim=state["concept_dim"],
        n_concepts=state["n_concepts"],
        latent_dim=32,
        hidden_dim=48,
        gate_concept_creation=False
    )
    model.load(checkpoint_path)
    tok = model._tokenizer
    
    # Build benchmark words & expand
    all_benchmark_words = []
    for s, r, o in FACTS:
        all_benchmark_words.extend([s, r, o])
    for tc in TEST_CASES:
        all_benchmark_words.append(tc["query"].split()[0])
        all_benchmark_words.extend(tc["analog_pair"])
    all_benchmark_words = list(set(all_benchmark_words))
    
    expand_vocabulary_and_embeddings(model, tok, all_benchmark_words)
    inject_precise_embeddings(model, tok)
    
    # Train graph
    for epoch in range(5):
        for s, r, o in FACTS:
            ids = tok.encode(f"{s} {r} {o}")
            if len(ids) < 3:
                continue
            ctx = np.array([ids[:-1]], dtype=np.int64)
            tgt = np.array([[ids[-1]]], dtype=np.int64)
            model.learn(ctx, tgt)
            
    # ======================================================================
    # STUDY 1: COMPARISON OF METHODS
    # ======================================================================
    print("\n" + "-" * 80)
    print("STUDY 1: ABLATION OF PATHWAY COMPONENTS (Rank / Score of Target)")
    print("-" * 80)
    print(f"{'Test Case':<22s} | {'Graph-Only':<12s} | {'Encoder-Only':<12s} | {'Hybrid (Std)':<12s} | {'Hybrid (Gated)':<12s} | {'Hybrid (Wtd)':<12s} | {'Hybrid (Margin)':<12s}")
    print("-" * 80)
    
    for tc in TEST_CASES:
        q = tc["query"]
        expected = tc["expected"]
        
        # 1. Graph Only
        go = graph_only_predict(model, tok, q)
        go_rank = next((i+1 for i, x in enumerate(go) if x[0] == expected), "N/A")
        go_score = next((x[1] for x in go if x[0] == expected), 0.0)
        go_str = f"R{go_rank} ({go_score:.3f})" if go_rank != "N/A" else "N/A"
        
        # 2. Encoder Only (1-hop retrieval)
        eo = encoder_only_predict(model, tok, q, FACTS)
        eo_rank = next((i+1 for i, x in enumerate(eo) if x[0] == expected), "N/A")
        eo_score = next((x[1] for x in eo if x[0] == expected), 0.0)
        eo_str = f"R{eo_rank} ({eo_score:.3f})" if eo_rank != "N/A" else "N/A"
        
        # 3. Hybrid Standard
        hs = hybrid_predict_gated(model, tok, q, gate_mode="standard")
        hs_rank = next((i+1 for i, x in enumerate(hs) if x[0] == expected), "N/A")
        hs_score = next((x[1] for x in hs if x[0] == expected), 0.0)
        hs_str = f"R{hs_rank} ({hs_score:.3f})" if hs_rank != "N/A" else "N/A"
        
        # 4. Hybrid Relative-Threshold Gated (sim >= 0.85 * best_sim)
        hg = hybrid_predict_gated(model, tok, q, gate_mode="relative_threshold")
        hg_rank = next((i+1 for i, x in enumerate(hg) if x[0] == expected), "N/A")
        hg_score = next((x[1] for x in hg if x[0] == expected), 0.0)
        hg_str = f"R{hg_rank} ({hg_score:.3f})" if hg_rank != "N/A" else "N/A"
        
        # 5. Hybrid Softmax-Weighted (top-5 weighted seeds)
        hw = hybrid_predict_gated(model, tok, q, gate_mode="weighted")
        hw_rank = next((i+1 for i, x in enumerate(hw) if x[0] == expected), "N/A")
        hw_score = next((x[1] for x in hw if x[0] == expected), 0.0)
        hw_str = f"R{hw_rank} ({hw_score:.3f})" if hw_rank != "N/A" else "N/A"
        
        # 6. Hybrid Margin-Gated Multi-Seed
        hm = hybrid_predict_gated(model, tok, q, gate_mode="margin_multi")
        hm_rank = next((i+1 for i, x in enumerate(hm) if x[0] == expected), "N/A")
        hm_score = next((x[1] for x in hm if x[0] == expected), 0.0)
        hm_str = f"R{hm_rank} ({hm_score:.3f})" if hm_rank != "N/A" else "N/A"
        
        print(f"{tc['category']:<22s} | {go_str:<12s} | {eo_str:<12s} | {hs_str:<12s} | {hg_str:<12s} | {hw_str:<12s} | {hm_str:<12s}")
        
    # ======================================================================
    # STUDY 2: HOP LIMIT SWEEP ON HYBRID
    # ======================================================================
    print("\n" + "-" * 80)
    print("STUDY 2: TRAVERSAL DEPTH (HOP) SWEEP (Target Rank & Score)")
    print("-" * 80)
    print(f"{'Category':<22s} | {'1 Hop':<12s} | {'2 Hops':<12s} | {'3 Hops':<12s} | {'4 Hops':<12s}")
    print("-" * 80)
    
    for tc in TEST_CASES:
        q = tc["query"]
        expected = tc["expected"]
        
        hop_strs = []
        for hops in [1, 2, 3, 4]:
            res = hybrid_predict_gated(model, tok, q, max_depth=hops, gate_mode="standard")
            rank = next((i+1 for i, x in enumerate(res) if x[0] == expected), "N/A")
            score = next((x[1] for x in res if x[0] == expected), 0.0)
            hop_strs.append(f"R{rank} ({score:.3f})" if rank != "N/A" else "N/A")
            
        print(f"{tc['category']:<22s} | {hop_strs[0]:<12s} | {hop_strs[1]:<12s} | {hop_strs[2]:<12s} | {hop_strs[3]:<12s}")

    # ======================================================================
    # STUDY 3: DIAGNOSTIC SEPARATION OF FAILURE CLASSES
    # ======================================================================
    print("\n" + "-" * 80)
    print("STUDY 3: DIAGNOSTIC DETAILED TRACE & FAILURE ANALYSIS")
    print("-" * 80)
    
    for tc in TEST_CASES:
        q = tc["query"]
        expected = tc["expected"]
        analogs = tc["analog_pair"]
        chain = tc["expected_chain"]
        
        print(f"\n[Trace] Query: '{q}' (Target: '{expected}')")
        
        # Latent neighbors
        lat_q = proto(model, tok, analogs[0])
        sims = []
        for word in tok.word_to_id.keys():
            bindings = model.binding_map.get_concepts(tok.word_to_id[word], min_confidence=0.1)
            if bindings:
                sims.append((word, cosine(lat_q, proto(model, tok, word))))
        sims.sort(key=lambda x: x[1], reverse=True)
        
        print("  1. Nearest Seeding Neighbors in Latent Space (32d):")
        for word, sim in sims[:4]:
            print(f"     - '{word}': similarity={sim:.4f}")
            
        # Check direct similarity to expected target (Encoder only)
        expected_sim = cosine(lat_q, proto(model, tok, expected))
        print(f"  2. Direct Query-to-Target Encoder Similarity: {expected_sim:.4f}")
        
        # Analyze failure class
        best_neighbor = sims[0][0]
        correct_seed = analogs[1]
        
        # Check direct similarity to correct seed
        correct_seed_sim = cosine(lat_q, proto(model, tok, correct_seed))
        print(f"  3. Direct Query-to-Expected-Seed Encoder Similarity: {correct_seed_sim:.4f}")
        
        if best_neighbor != correct_seed:
            print("  4. FAILURE CLASS: SEMANTIC AMBIGUITY (Wrong Seed)")
            print(f"     - Reason: The encoder mapped query '{analogs[0]}' closest to '{best_neighbor}' instead of expected '{correct_seed}'.")
            
            # Check if multi-seed weighted or margin_multi resolves it
            hw_res = hybrid_predict_gated(model, tok, q, gate_mode="weighted")
            hw_reached = any(x[0] == expected for x in hw_res)
            hm_res = hybrid_predict_gated(model, tok, q, gate_mode="margin_multi")
            hm_reached = any(x[0] == expected for x in hm_res)
            
            if hw_reached or hm_reached:
                print("     - RESOLUTION:")
                if hw_reached:
                    hw_rank = next((i+1 for i, x in enumerate(hw_res) if x[0] == expected))
                    hw_score = next((x[1] for x in hw_res if x[0] == expected))
                    print(f"       * Multi-Seed Weighted RESOLVED this failure! Target reached at Rank {hw_rank} (score={hw_score:.3f}).")
                if hm_reached:
                    hm_rank = next((i+1 for i, x in enumerate(hm_res) if x[0] == expected))
                    hm_score = next((x[1] for x in hm_res if x[0] == expected))
                    print(f"       * Margin-Gated Multi-Seed RESOLVED this failure! Target reached at Rank {hm_rank} (score={hm_score:.3f}).")
            else:
                print("       * Neither Multi-Seed nor Margin-Gated Multi-Seed resolved this failure (correct seed not in top-5).")
        else:
            # Seed is correct, check graph traversal results at different hops
            go_res = graph_only_predict(model, tok, f"{correct_seed} causes", max_depth=3)
            reached = any(x[0] == expected for x in go_res)
            if not reached:
                print("  4. FAILURE CLASS: GRAPH BREAK (Disconnected Topology)")
                print(f"     - Reason: Even from correct seed '{correct_seed}', the expected target '{expected}' was unreachable in the graph.")
            else:
                print("  4. DIAGNOSIS: CORRECT COMPOSITION PATHWAY")
                print(f"     - Path: {analogs[0]} -> {correct_seed} (seeding) -> {' -> '.join(chain[1:])} (traversal)")
                
                # Check for graph drift (did performance drop at depth 4?)
                res_3 = hybrid_predict_gated(model, tok, q, max_depth=3, gate_mode="standard")
                res_4 = hybrid_predict_gated(model, tok, q, max_depth=4, gate_mode="standard")
                rank_3 = next((i+1 for i, x in enumerate(res_3) if x[0] == expected), 99)
                rank_4 = next((i+1 for i, x in enumerate(res_4) if x[0] == expected), 99)
                if rank_4 > rank_3:
                    print("     - WARNING: GRAPH DRIFT detected! Target rank degraded at hop depth 4.")
                else:
                    print("     - Stable traversal across depths.")
                    
    print("\n" + "=" * 80)
    print("END OF DIAGNOSTIC REPORT")
    print("=" * 80)


if __name__ == "__main__":
    main()
