"""
Compositional Hybrid Benchmark using Trained Checkpoint
======================================================
Tests whether traversal and embeddings can COOPERATE on tasks
neither can solve alone, utilizing the trained Phase 4 encoder.

The key test cases and boundary-case pairs:
1. warmth ≈ kindness (high similarity ~0.85 in 64d)
   kindness → trust → cooperation (expected: cooperation)
   
2. light ≈ hope (high-moderate similarity ~0.70 in 64d)
   hope → courage → victory (expected: victory)
   
3. combustion ≈ resentment (moderate-low similarity ~0.45 in 64d)
   resentment → hostility → conflict (expected: conflict)
   
4. gravity ≈ loyalty (low/borderline similarity ~0.20 in 64d)
   loyalty → support → stability (expected: stability)
"""

import os
import sys
import json
import random
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
# Configuration & Constants
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
    
    # Distractor/noise edges to test error propagation
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
        "chain": "warmth ≈ kindness -> trust -> cooperation",
        "category": "High Sim (Analogy)"
    },
    {
        "query": "light causes",
        "expected": "victory",
        "analog_pair": ("light", "hope"),
        "chain": "light ≈ hope -> courage -> victory",
        "category": "High-Mod Sim Boundary"
    },
    {
        "query": "combustion causes",
        "expected": "conflict",
        "analog_pair": ("combustion", "resentment"),
        "chain": "combustion ≈ resentment -> hostility -> conflict",
        "category": "Mod-Low Sim Boundary"
    },
    {
        "query": "gravity causes",
        "expected": "stability",
        "analog_pair": ("gravity", "loyalty"),
        "chain": "gravity ≈ loyalty -> support -> stability",
        "category": "Low Sim Boundary"
    },
    # Negative control
    {
        "query": "bug causes",
        "expected": "outage",
        "analog_pair": None,
        "chain": "bug -> crash -> outage",
        "category": "Direct Traversal"
    }
]

# ──────────────────────────────────────────────────────────────────────────
# Helpers
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
    """Generate two unit vectors in dim-space with exactly cos_theta similarity."""
    a = np.random.randn(dim).astype(np.float32)
    a /= np.linalg.norm(a)
    
    b_orth = np.random.randn(dim).astype(np.float32)
    b_orth -= np.dot(b_orth, a) * a
    b_orth /= np.linalg.norm(b_orth)
    
    b = cos_theta * a + np.sqrt(1.0 - cos_theta**2) * b_orth
    return a, b


def expand_vocabulary_and_embeddings(model: RLMv2, tok: WordTokenizer, words: list[str]):
    """Dynamically expand tokenizer vocabulary and model's token embeddings."""
    for w in words:
        tok.encode(w)
        
    vocab_size = tok.vocab_size
    dim = model.embed_dim
    old_weight = model.token_embed.weight.data
    old_size = old_weight.shape[0]
    
    if vocab_size > old_size:
        # Expand embedding weights
        new_weight = np.random.randn(vocab_size, dim).astype(np.float32) * np.sqrt(2.0 / dim)
        new_weight[:old_size] = old_weight
        
        # Initialize new words with LearnedEmbedder
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
    """Inject controlled semantic similarities into the 64-dim embedding space."""
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
# Vector-Space Seeding and Multi-Hop Traversal Logic
# ──────────────────────────────────────────────────────────────────────────

def hybrid_compositional_predict(
    model: RLMv2,
    tok: WordTokenizer,
    query: str,
    k_neighbors: int = 3,
    similarity_threshold: float = 0.1,
    max_depth: int = 3
) -> list[dict]:
    """The hybrid: seeds graph traversal using 32d latent similarities, then walks Hebbian edges."""
    parts = query.split()
    if len(parts) < 2:
        return []
    subj_word, rel_word = parts[0], parts[1]

    # Map relation word to relation type
    causal_verbs = {"causes", "cause", "leads", "produces", "creates"}
    possessive_verbs = {"has", "have", "contains", "includes"}
    rel_type = None
    if rel_word in causal_verbs:
        rel_type = "causal"
    elif rel_word in possessive_verbs:
        rel_type = "possessive"

    # Step 1: Project subject into 32d latent space
    subj_tid = tok.word_to_id.get(subj_word)
    if subj_tid is None:
        return []
    
    # Check if subject itself has a concept node in the graph
    direct_cid = None
    direct_bindings = model.binding_map.get_concepts(subj_tid, min_confidence=0.1)
    if direct_bindings:
        direct_cid = direct_bindings[0].concept_id
        
    lat_query = proto(model, tok, subj_word)
    
    # Step 2: Vector-space seeding (via cosine similarity in 32d latent space)
    seeds = []  # entries: (concept_id, sim, word)
    if direct_cid is not None:
        seeds.append((direct_cid, 1.0, subj_word))
        
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
        if sim > similarity_threshold:
            scored_neighbors.append((cid, sim, word))
            
    scored_neighbors.sort(key=lambda x: x[1], reverse=True)
    seeds.extend(scored_neighbors[:k_neighbors])
    
    # Remove duplicate concept IDs in seeds, keeping the highest similarity
    seen_cids = set()
    unique_seeds = []
    for cid, sim, word in seeds:
        if cid not in seen_cids:
            seen_cids.add(cid)
            unique_seeds.append((cid, sim, word))
            
    # Step 3: Multi-hop propagation
    activations = {}
    path_sources = {}  # maps node_id -> path list of (prev_node_id, edge_weight, rel_type)
    
    for cid, sim, word in unique_seeds:
        activations[cid] = sim
        path_sources[cid] = [("seed", sim, word)]
        
    # BFS propagation up to max_depth
    frontier = [cid for cid, _, _ in unique_seeds]
    
    for depth in range(1, max_depth + 1):
        next_frontier = []
        for nid in frontier:
            current_act = activations[nid]
            for tgt_id, edge in model.graph.get_outgoing(nid):
                # Filter by relation type
                if rel_type and edge.relation_type != rel_type:
                    continue
                # Propagate activation: src_activation * edge_weight
                prop_act = current_act * edge.weight
                
                if tgt_id not in activations or prop_act > activations[tgt_id]:
                    activations[tgt_id] = prop_act
                    path_sources[tgt_id] = path_sources[nid] + [(nid, edge.weight, edge.relation_type)]
                    next_frontier.append(tgt_id)
        frontier = next_frontier
        
    # Step 4: Decode activations back to words
    results = []
    seed_words = {word for _, _, word in unique_seeds}
    for cid, act in activations.items():
        tokens = model.binding_map.get_tokens(cid, 0.0)
        for b in tokens:
            word = tok.decode([b.token_id])
            if word not in seed_words and not word.startswith("?"):
                path_desc = []
                for step in path_sources[cid]:
                    if step[0] == "seed":
                        path_desc.append(f"{step[2]} (seed, sim={step[1]:.2f})")
                    else:
                        prev_word = tok.decode([model.binding_map.get_tokens(step[0], 0.0)[0].token_id])
                        path_desc.append(f"-[{step[2]} w={step[1]:.2f}]-> {word}")
                results.append({
                    "word": word,
                    "score": act,
                    "concept_id": cid,
                    "path": " ".join(path_desc)
                })
                
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ──────────────────────────────────────────────────────────────────────────
# Main Execution Pipeline
# ──────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("COMPOSITIONAL HYBRID BENCHMARK WITH TRAINED CHECKPOINT")
    print("=" * 70)
    
    checkpoint_path = os.path.join(SCRIPT_DIR, "experiment_results", "encoder_32d_fixed.pkl")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please run experiments/phase4_option3_fixed.py first.")
        sys.exit(1)
        
    print(f"Loading pre-trained encoder checkpoint from {checkpoint_path}...")
    with open(checkpoint_path, 'rb') as f:
        state = pickle.load(f)
        
    # Instantiate model matching checkpoint parameters
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
    
    # 1. Expand vocabulary for all concepts/words in our composition facts
    all_benchmark_words = []
    for s, r, o in FACTS:
        all_benchmark_words.extend([s, r, o])
    for tc in TEST_CASES:
        all_benchmark_words.append(tc["query"].split()[0])
        if tc["analog_pair"]:
            all_benchmark_words.extend(tc["analog_pair"])
            
    # Deduplicate and expand
    all_benchmark_words = list(set(all_benchmark_words))
    expand_vocabulary_and_embeddings(model, tok, all_benchmark_words)
    
    # 2. Inject controlled semantic similarities (embeddings are frozen after injection)
    inject_precise_embeddings(model, tok)
    
    # Print out mapping check
    print("\nVerified latent space similarities (mapped by encoder to 32d):")
    for tc in TEST_CASES:
        pair = tc["analog_pair"]
        if pair:
            w1, w2 = pair
            sim_32d = cosine(proto(model, tok, w1), proto(model, tok, w2))
            print(f"  Analog: {w1:10s} <-> {w2:10s} | 32d Cos Sim = {sim_32d:.4f}")

    # 3. Train Hebbian edges in graph
    print("\nTraining graph edges on facts (embeddings are frozen)...")
    for epoch in range(5):
        for s, r, o in FACTS:
            ids = tok.encode(f"{s} {r} {o}")
            if len(ids) < 3:
                continue
            ctx = np.array([ids[:-1]], dtype=np.int64)
            tgt = np.array([[ids[-1]]], dtype=np.int64)
            model.learn(ctx, tgt)
            
    print(f"Graph constructed: {len(model.graph.nodes)} nodes, {len(model.graph.edges)} edges.")
    
    # 4. Evaluate Test Cases
    print("\n" + "=" * 70)
    print("HYBRID COMPOSITIONAL INFERENCE BENCHMARK RUN")
    print("=" * 70)
    
    results_summary = []
    
    for tc in TEST_CASES:
        query = tc["query"]
        expected = tc["expected"]
        category = tc["category"]
        pair = tc["analog_pair"]
        
        print(f"\n[Category: {category}] Query: '{query}'")
        
        # Compute latent similarity to show in table
        seeding_sim = 1.0
        if pair:
            seeding_sim = cosine(proto(model, tok, pair[0]), proto(model, tok, pair[1]))
            print(f"  Vector Seeding: {pair[0]} ≈ {pair[1]} (32d Sim: {seeding_sim:.4f})")
            
        preds = hybrid_compositional_predict(model, tok, query, k_neighbors=3, similarity_threshold=0.1)
        
        # Rank of expected target
        rank = None
        target_score = 0.0
        for idx, pred in enumerate(preds):
            if pred["word"] == expected:
                rank = idx + 1
                target_score = pred["score"]
                break
                
        # Signal to Noise Ratio (SNR)
        other_scores = [p["score"] for p in preds if p["word"] != expected]
        sum_others = sum(other_scores)
        snr = target_score / sum_others if sum_others > 0 else (99.0 if target_score > 0 else 0.0)
        
        # Path Coherence: Target Score / Seeding Sim (measures decay down the chain)
        coherence = target_score / seeding_sim if seeding_sim > 0 else 0.0
        
        # Print top predictions
        print("  Top Predictions:")
        if not preds:
            print("    (None)")
        for idx, pred in enumerate(preds[:3]):
            marker = "  * " if pred["word"] == expected else "    "
            print(f"{marker}Rank {idx+1}: {pred['word']:15s} | Score: {pred['score']:.4f} | Path: {pred['path']}")
            
        results_summary.append({
            "category": category,
            "query": query,
            "expected": expected,
            "seeding_sim": seeding_sim,
            "rank": rank if rank is not None else "N/A",
            "score": target_score,
            "coherence": coherence,
            "snr": snr
        })
        
    # 5. Output Summary Table
    print("\n" + "=" * 70)
    print("CAPABILITY & METRICS MATRIX")
    print("=" * 70)
    print(f"{'Category':<22s} | {'Query':<15s} | {'Target':<12s} | {'32d Sim':<7s} | {'Rank':<4s} | {'Score':<6s} | {'Coherence':<9s} | {'SNR':<5s}")
    print("-" * 96)
    for r in results_summary:
        rank_str = str(r["rank"])
        sim_str = f"{r['seeding_sim']:.2f}" if r["seeding_sim"] < 1.0 else "1.00"
        snr_str = f"{r['snr']:.2f}" if r["snr"] < 99.0 else "Inf"
        print(f"{r['category']:<22s} | {r['query']:<15s} | {r['expected']:<12s} | {sim_str:<7s} | {rank_str:<4s} | {r['score']:.4f} | {r['coherence']:.4f}    | {snr_str:<5s}")
        
    print()
    print("Interpretation:")
    print("1. If Coherence is close to edge weight decay (e.g. w1 * w2 ≈ 0.3 * 0.3 ≈ 0.09 times Sim), the path is highly coherent.")
    print("2. High SNR indicates the graph traversal successfully filters out noise and prevents error propagation.")
    print("3. Compare the boundary cases: Do the lower 32d similarities propagate errors (low SNR) or smooth out (high rank & high SNR)?")
    print("=" * 70)


if __name__ == "__main__":
    main()
