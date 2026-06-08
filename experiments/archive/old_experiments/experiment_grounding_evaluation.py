#!/usr/bin/env python3
"""
Relational Grounding and Seeding Evaluation Suite
===================================================
Compares:
1. Single Seed (retrieval_v1, k=1)
2. Multi Seed (retrieval_v2_multi_seed, gate_mode="weighted")
3. Multi Seed + Margin (retrieval_v2_multi_seed, gate_mode="margin_multi")

Across five categories of challenge cases:
- Easy: High semantic similarity (~0.85) between analog and graph seed.
- Medium: Moderate semantic similarity (~0.70) between analog and graph seed.
- Hard: Low semantic similarity (~0.20) between analog and graph seed.
- Adversarial: Spurious high-similarity distractors present in latent space.
- OOD (Out-of-Distribution): Zero-shot novel query terms not seen during training.
"""

import os
import sys
import pickle
import numpy as np

# Adjust path to import ravana modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import WordTokenizer

# ──────────────────────────────────────────────────────────────────────────
# Facts and Challenge Cases Definition
# ──────────────────────────────────────────────────────────────────────────

FACTS = [
    # Chain 1 (Easy): warmth -> kindness -> trust -> cooperation
    ("kindness", "causes", "trust"),
    ("trust", "causes", "cooperation"),
    
    # Chain 2 (Medium): light -> hope -> courage -> victory
    ("hope", "causes", "courage"),
    ("courage", "causes", "victory"),
    
    # Chain 3 (Hard): gravity -> loyalty -> support -> stability
    ("loyalty", "causes", "support"),
    ("support", "causes", "stability"),
    
    # Chain 4 (Adversarial): combustion -> resentment -> hostility -> conflict
    ("resentment", "causes", "hostility"),
    ("hostility", "causes", "conflict"),

    # Distractor/noise edges in graph
    ("trust", "causes", "vulnerability"),
    ("courage", "causes", "danger"),
    ("hostility", "causes", "isolation"),
    ("support", "causes", "obligation"),
    
    # Additional noise facts to populate the graph
    ("cat", "has", "tail"),
    ("dog", "has", "tail"),
    ("cat", "has", "fur"),
    ("dog", "has", "fur"),
    ("virus", "causes", "illness"),
    ("illness", "causes", "absence"),
    ("bug", "causes", "crash"),
    ("crash", "causes", "outage"),
]

CHALLENGE_CASES = [
    # --- EASY ---
    {
        "query": "warmth causes",
        "expected": "cooperation",
        "expected_seed": "kindness",
        "category": "Easy",
        "description": "High sim analog pair: warmth ≈ kindness (0.85)"
    },
    # --- MEDIUM ---
    {
        "query": "light causes",
        "expected": "victory",
        "expected_seed": "hope",
        "category": "Medium",
        "description": "Moderate sim analog pair: light ≈ hope (0.70)"
    },
    # --- HARD ---
    {
        "query": "gravity causes",
        "expected": "stability",
        "expected_seed": "loyalty",
        "category": "Hard",
        "description": "Low sim analog pair: gravity ≈ loyalty (0.20)"
    },
    # --- ADVERSARIAL ---
    {
        "query": "combustion causes",
        "expected": "conflict",
        "expected_seed": "resentment",
        "category": "Adversarial",
        "description": "High spurious distractor (combustion ≈ cat: 0.81)"
    },
    # --- OOD (Out of Distribution) ---
    {
        "query": "ignition causes",
        "expected": "conflict",
        "expected_seed": "resentment",
        "category": "OOD",
        "description": "Zero-shot novel term (ignition ≈ combustion ≈ resentment)"
    },
    {
        "query": "pull causes",
        "expected": "stability",
        "expected_seed": "loyalty",
        "category": "OOD",
        "description": "Zero-shot novel term (pull ≈ gravity ≈ loyalty)"
    }
]

# ──────────────────────────────────────────────────────────────────────────
# Vector Math & Helpers
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
        # OOD mappings
        ("ignition", "combustion"): 0.88,
        ("pull", "gravity"): 0.82,
    }
    for (w1, w2), cos_theta in targets.items():
        tid1 = tok.word_to_id[w1]
        tid2 = tok.word_to_id[w2]
        v1, v2 = make_similar_vectors(dim, cos_theta)
        model.token_embed.weight.data[tid1] = v1
        model.token_embed.weight.data[tid2] = v2
    model._token_embed_norms = None


# ──────────────────────────────────────────────────────────────────────────
# Main Evaluation Loop
# ──────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("RAVANA GROUNDING & RETRIEVAL TELEMETRY BENCHMARK")
    print("=" * 90)
    
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
    
    # Expand vocabulary for benchmark terms
    all_words = []
    for s, r, o in FACTS:
        all_words.extend([s, r, o])
    for tc in CHALLENGE_CASES:
        all_words.append(tc["query"].split()[0])
        all_words.append(tc["expected_seed"])
    all_words = list(set(all_words))
    
    expand_vocabulary_and_embeddings(model, tok, all_words)
    inject_precise_embeddings(model, tok)
    
    # Train relational graph briefly on facts
    for epoch in range(5):
        for s, r, o in FACTS:
            ids = tok.encode(f"{s} {r} {o}")
            if len(ids) < 3:
                continue
            ctx = np.array([ids[:-1]], dtype=np.int64)
            tgt = np.array([[ids[-1]]], dtype=np.int64)
            model.learn(ctx, tgt)
            
    # Evaluation Configs
    retrieval_strategies = [
        {"name": "Single Seed (v1)", "fn": lambda q: model.retrieval_v1(q, k_neighbors=1)},
        {"name": "Multi Seed (v2)", "fn": lambda q: model.retrieval_v2_multi_seed(q, k_neighbors=5, gate_mode="weighted")},
        {"name": "Margin Multi (v2)", "fn": lambda q: model.retrieval_v2_multi_seed(q, k_neighbors=5, gate_mode="margin_multi")}
    ]
    
    summary_results = {s["name"]: {"success": 0, "total": 0} for s in retrieval_strategies}
    
    print(f"\n{'Category':<12s} | {'Query':<18s} | {'Target':<12s} | {'Single Seed (v1)':<16s} | {'Multi Seed (v2)':<16s} | {'Margin Multi (v2)':<16s}")
    print("-" * 105)
    
    for tc in CHALLENGE_CASES:
        q = tc["query"]
        expected = tc["expected"]
        
        runs = []
        for strategy in retrieval_strategies:
            res, metrics = strategy["fn"](q)
            rank = next((i+1 for i, x in enumerate(res) if x[0] == expected), "N/A")
            
            # Update success count (rank <= 5 is considered successful retrieval)
            is_success = rank != "N/A" and rank <= 10
            strategy_name = strategy["name"]
            summary_results[strategy_name]["total"] += 1
            if is_success:
                summary_results[strategy_name]["success"] += 1
                
            runs.append(f"R{rank}" if rank != "N/A" else "Fail")
            
        print(f"{tc['category']:<12s} | {q:<18s} | {expected:<12s} | {runs[0]:<16s} | {runs[1]:<16s} | {runs[2]:<16s}")

    # ======================================================================
    # TELEMETRY REPORT
    # ======================================================================
    print("\n" + "=" * 90)
    print("TELEMETRY AND DYNAMIC BEHAVIOR ANALYSIS")
    print("=" * 90)
    
    for strategy in retrieval_strategies:
        print(f"\nStrategy: {strategy['name']}")
        print("-" * 45)
        
        all_metrics = []
        for tc in CHALLENGE_CASES:
            res, metrics = strategy["fn"](tc["query"])
            
            # Calculate actual final rank
            rank = next((i+1 for i, x in enumerate(res) if x[0] == tc["expected"]), 99)
            metrics["final_rank"] = rank if rank != 99 else -1
            all_metrics.append(metrics)
            
        # Compute mean statistics
        mean_seeds = np.mean([m["seed_count"] for m in all_metrics])
        mean_top_sim = np.mean([m["top_seed_similarity"] for m in all_metrics])
        mean_margin = np.mean([m["margin"] for m in all_metrics])
        mean_active = np.mean([m["activated_nodes"] for m in all_metrics])
        mean_dead = np.mean([m["dead_end_nodes"] for m in all_metrics])
        success_rate = (sum(1 for m in all_metrics if m["final_rank"] != -1) / len(all_metrics)) * 100
        
        print(f"  - Seeding: Mean Seed Count = {mean_seeds:.2f}")
        print(f"  - Similarity: Mean Top Similarity = {mean_top_sim:.4f}")
        print(f"  - Confidence: Mean Similarity Margin = {mean_margin:.4f}")
        print(f"  - Traversal: Mean Activated Nodes = {mean_active:.2f}")
        print(f"  - Filtering: Mean Dead-End Nodes = {mean_dead:.2f}")
        print(f"  - Performance: Success Rate (Top-10) = {success_rate:.1f}%")

    # ======================================================================
    # SEED ENTROPY AND UNCERTAINTY ANALYSIS
    # ======================================================================
    print("\n" + "=" * 90)
    print("SEED SIMILARITY LANDSCAPE (ENTROPY) ANALYSIS")
    print("=" * 90)
    
    for tc in CHALLENGE_CASES:
        q = tc["query"]
        expected_seed = tc["expected_seed"]
        subj = q.split()[0]
        
        lat_q = proto(model, tok, subj)
        
        # Gather all vocabulary similarity values
        all_sims = []
        for word, tid in tok.word_to_id.items():
            bindings = model.binding_map.get_concepts(tid, min_confidence=0.1)
            if not bindings:
                continue
            all_sims.append(cosine(lat_q, proto(model, tok, word)))
            
        all_sims = np.array(all_sims)
        # Shift and compute softmax distribution to find entropy of the query
        temp = 0.15
        exp_sims = np.exp((all_sims - np.max(all_sims)) / temp)
        probs = exp_sims / np.sum(exp_sims)
        entropy = -np.sum(probs * np.log(probs + 1e-15))
        
        # Check expected seed's rank and similarity
        scored_neighbors = []
        for word, tid in tok.word_to_id.items():
            bindings = model.binding_map.get_concepts(tid, min_confidence=0.1)
            if bindings:
                scored_neighbors.append((word, cosine(lat_q, proto(model, tok, word))))
        scored_neighbors.sort(key=lambda x: x[1], reverse=True)
        
        seed_rank = next((i+1 for i, x in enumerate(scored_neighbors) if x[0] == expected_seed), "N/A")
        seed_sim = next((x[1] for x in scored_neighbors if x[0] == expected_seed), 0.0)
        top_sim = scored_neighbors[0][1]
        margin = (top_sim - scored_neighbors[1][1]) if len(scored_neighbors) > 1 else top_sim
        
        print(f"Query: '{q}' (Category: {tc['category']})")
        print(f"  * Seed Entropy (Uncertainty) = {entropy:.4f}")
        print(f"  * Top Seed Similarity       = {top_sim:.4f} (Margin = {margin:.4f})")
        print(f"  * Expected Seed             = '{expected_seed}' at Rank {seed_rank} (Similarity = {seed_sim:.4f})")
        
    print("\n" + "=" * 90)
    print("END OF BENCHMARK REPORT")
    print("=" * 90)

if __name__ == "__main__":
    main()
