#!/usr/bin/env python3
"""
Recall@K and Traversal Accuracy Sweep
======================================
This script sweeps the K (number of neighbors) parameter in retrieval_v2_multi_seed
across K = 1, 3, 5, 10, 15, 20 to determine:
1. Expected Seed Recall@K: Is the correct grounding concept in the top-K candidates?
2. Traversal Accuracy@K: Does the target get successfully retrieved at depth?
3. Telemetry: How do activated nodes and dead-end nodes scale with K?
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
from experiment_grounding_evaluation import FACTS, CHALLENGE_CASES, expand_vocabulary_and_embeddings, inject_precise_embeddings, cosine, proto

def main():
    print("=" * 90)
    print("RAVANA RECALL@K AND TRAVERSAL ACCURACY SWEEP")
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
            
    # Sweep configurations
    K_values = [1, 3, 5, 10, 15, 20]
    
    # Tables to print out
    print(f"\n1. SWEEP METRICS OVER K (Weighted Multi-Seed)")
    print("-" * 90)
    print(f"{'K Value':<10s} | {'Seed Recall@K':<15s} | {'Traversal Acc':<15s} | {'Mean Activated':<16s} | {'Mean Dead Ends':<16s}")
    print("-" * 90)
    
    for k in K_values:
        seed_recalls = []
        traversal_successes = []
        activated_counts = []
        dead_end_counts = []
        
        for tc in CHALLENGE_CASES:
            q = tc["query"]
            expected = tc["expected"]
            expected_seed = tc["expected_seed"]
            subj = q.split()[0]
            
            # Retrieve with current K using retrieval_v2_multi_seed (weighted mode)
            res, metrics = model.retrieval_v2_multi_seed(q, k_neighbors=k, gate_mode="weighted")
            
            # A. Calculate Seed Recall: Is the expected seed in the top K nearest neighbors?
            lat_q = proto(model, tok, subj)
            scored_neighbors = []
            for word, tid in tok.word_to_id.items():
                bindings = model.binding_map.get_concepts(tid, min_confidence=0.1)
                if bindings:
                    scored_neighbors.append((word, cosine(lat_q, proto(model, tok, word))))
            scored_neighbors.sort(key=lambda x: x[1], reverse=True)
            top_k_words = {w for w, _ in scored_neighbors[:k]}
            
            seed_recalls.append(1 if expected_seed in top_k_words else 0)
            
            # B. Calculate Traversal Accuracy: Is the expected target reached in top 10 retrieval?
            rank = next((i+1 for i, x in enumerate(res) if x[0] == expected), 99)
            traversal_successes.append(1 if rank <= 10 else 0)
            
            # C. Collect Telemetry
            activated_counts.append(metrics["activated_nodes"])
            dead_end_counts.append(metrics["dead_end_nodes"])
            
        mean_recall = np.mean(seed_recalls) * 100
        mean_acc = np.mean(traversal_successes) * 100
        mean_active = np.mean(activated_counts)
        mean_dead = np.mean(dead_end_counts)
        
        print(f"K = {k:<5d} | {mean_recall:.1f}%        | {mean_acc:.1f}%        | {mean_active:<16.2f} | {mean_dead:<16.2f}")
        
    print("\n2. PER-QUERY RANK TRACE OF EXPECTED SEED")
    print("-" * 90)
    print(f"{'Category':<12s} | {'Query':<18s} | {'Expected Seed':<15s} | {'Rank in Latent Space':<20s} | {'Similarity':<10s}")
    print("-" * 90)
    
    for tc in CHALLENGE_CASES:
        q = tc["query"]
        expected_seed = tc["expected_seed"]
        subj = q.split()[0]
        
        lat_q = proto(model, tok, subj)
        scored_neighbors = []
        for word, tid in tok.word_to_id.items():
            bindings = model.binding_map.get_concepts(tid, min_confidence=0.1)
            if bindings:
                scored_neighbors.append((word, cosine(lat_q, proto(model, tok, word))))
        scored_neighbors.sort(key=lambda x: x[1], reverse=True)
        
        rank = next((i+1 for i, x in enumerate(scored_neighbors) if x[0] == expected_seed), "N/A")
        sim = next((x[1] for x in scored_neighbors if x[0] == expected_seed), 0.0)
        
        print(f"{tc['category']:<12s} | {q:<18s} | {expected_seed:<15s} | {f'Rank {rank}':<20s} | {sim:.4f}")
        
    print("\n" + "=" * 90)
    print("END OF SWEEP REPORT")
    print("=" * 90)

if __name__ == "__main__":
    main()
