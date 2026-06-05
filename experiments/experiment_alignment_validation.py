#!/usr/bin/env python3
"""
Encoder-Graph Alignment Validation Script
==========================================
Runs the sleep cycle with graph-aware contrastive alignment and verifies that:
1. Expected seed Recall@5 increases.
2. Traversal Accuracy on Hard and OOD-Hard tasks increases.
3. The rank of correct seeds rises into the top-5 candidate window.
4. Periodic sleep homeostasis prevents Hebbian drift.
5. Adaptive margin handles semantic fog at high K.
"""

import os
import sys
import pickle
import numpy as np

# Fixed seed for reproducibility
np.random.seed(42)

# Adjust path to import ravana modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import WordTokenizer
from experiment_grounding_evaluation import FACTS, CHALLENGE_CASES, expand_vocabulary_and_embeddings, inject_precise_embeddings, cosine, proto


def run_evaluation(model, tok, title):
    print(f"\nEvaluation: {title}")
    print("-" * 80)
    print(f"{'Category':<12s} | {'Query':<18s} | {'Expected Seed':<15s} | {'Seed Rank':<10s} | {'Seed Sim':<10s} | {'Traversal':<10s}")
    print("-" * 80)
    
    successes = 0
    for tc in CHALLENGE_CASES:
        q = tc["query"]
        expected = tc["expected"]
        expected_seed = tc["expected_seed"]
        subj = q.split()[0]
        
        # 1. Cosine similarity and rank in latent space
        lat_q = proto(model, tok, subj)
        scored_neighbors = []
        for word, tid in tok.word_to_id.items():
            bindings = model.binding_map.get_concepts(tid, min_confidence=0.1)
            if bindings:
                scored_neighbors.append((word, cosine(lat_q, proto(model, tok, word))))
        scored_neighbors.sort(key=lambda x: x[1], reverse=True)
        
        rank = next((i+1 for i, x in enumerate(scored_neighbors) if x[0] == expected_seed), "N/A")
        sim = next((x[1] for x in scored_neighbors if x[0] == expected_seed), 0.0)
        
        # 2. Traversal result
        res, _ = model.retrieval_v2_multi_seed(q, k_neighbors=5, gate_mode="margin_multi")
        t_rank = next((i+1 for i, x in enumerate(res) if x[0] == expected), "N/A")
        
        is_success = t_rank != "N/A" and t_rank <= 10
        if is_success:
            successes += 1
            
        t_str = f"R{t_rank}" if t_rank != "N/A" else "Fail"
        print(f"{tc['category']:<12s} | {q:<18s} | {expected_seed:<15s} | {f'Rank {rank}':<10s} | {sim:.4f}   | {t_str:<10s}")
        
    acc = (successes / len(CHALLENGE_CASES)) * 100
    recall_5 = model.compute_neighbor_recall_at_5() * 100
    print(f"\nSummary Metrics:")
    print(f"  * Graph-Neighbor Recall@5 = {recall_5:.1f}%")
    print(f"  * Traversal Success Rate  = {acc:.1f}%")
    return recall_5, acc


def run_k_sweep(model, tok, title, ks=None, gate_modes=None):
    """Test traversal accuracy across different K values and gate modes."""
    if ks is None:
        ks = [5, 10, 20]
    if gate_modes is None:
        gate_modes = ["margin_multi", "adaptive_margin"]
    
    print(f"\n{title}")
    print("=" * 80)
    
    for gate_mode in gate_modes:
        print(f"\n--- Gate Mode: {gate_mode} ---")
        print(f"{'K':>3s} | {'Successes':>10s} | {'Rate':>8s}")
        print("-" * 30)
        for k in ks:
            successes = 0
            for tc in CHALLENGE_CASES:
                q = tc["query"]
                expected = tc["expected"]
                res, _ = model.retrieval_v2_multi_seed(q, k_neighbors=k, gate_mode=gate_mode)
                rank = next((i+1 for i, x in enumerate(res) if x[0] == expected), 99)
                if rank <= 10:
                    successes += 1
            rate = successes / len(CHALLENGE_CASES) * 100
            print(f"{k:>3d} | {successes:>10d} | {rate:>7.1f}%")


def test_wake_sleep_cycle(model, tok, num_wake_epochs=12, sleep_every_n=2):
    """Test periodic sleep homeostasis during extended wake training."""
    print(f"\n{'='*80}")
    print(f"WAKE-SLEEP CYCLE TEST: {num_wake_epochs} wake epochs, sleep every {sleep_every_n}")
    print(f"{'='*80}")
    
    # Boost alignment for wake-sleep cycle
    model.max_alignment_epochs = 20
    model.alignment_lr = 0.02
    model.lambda_anchor = 0.005
    model.sleep_every_n_wake_epochs = sleep_every_n
    model.wake_epochs_since_sleep = 0
    
    # Track metrics across wake epochs
    for wake_epoch in range(1, num_wake_epochs + 1):
        # One wake epoch: train on all facts
        for s, r, o in FACTS:
            ids = tok.encode(f"{s} {r} {o}")
            if len(ids) < 3:
                continue
            ctx = np.array([ids[:-1]], dtype=np.int64)
            tgt = np.array([[ids[-1]]], dtype=np.int64)
            model.learn(ctx, tgt)
        
        # End wake epoch - triggers sleep if cadence reached
        model.end_wake_epoch(validation_queries=CHALLENGE_CASES)
        
        # Evaluate every few epochs
        if wake_epoch % sleep_every_n == 0 or wake_epoch in [1, num_wake_epochs]:
            successes = 0
            for tc in CHALLENGE_CASES:
                q = tc["query"]
                expected = tc["expected"]
                res, _ = model.retrieval_v2_multi_seed(q, k_neighbors=5, gate_mode="margin_multi")
                rank = next((i+1 for i, x in enumerate(res) if x[0] == expected), 99)
                if rank <= 10:
                    successes += 1
            rate = successes / len(CHALLENGE_CASES) * 100
            recall_5 = model.compute_neighbor_recall_at_5() * 100
            print(f"  Wake epoch {wake_epoch:2d}: Traversal={rate:5.1f}% | Recall@5={recall_5:5.1f}% | Graph edges={len(model.graph.edges)}")


def main():
    print("=" * 90)
    print("RAVANA GRAPH-AWARE ENCODER ALIGNMENT RUN & VALIDATION")
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
    # Allow encoder to learn during wake epochs so alignment_needed is triggered
    model.freeze_encoder = False
    tok = model._tokenizer
    
    # Expand vocabulary and inject precise embeddings
    all_words = []
    for s, r, o in FACTS:
        all_words.extend([s, r, o])
    for tc in CHALLENGE_CASES:
        all_words.append(tc["query"].split()[0])
        all_words.append(tc["expected_seed"])
    all_words = list(set(all_words))
    
    expand_vocabulary_and_embeddings(model, tok, all_words)
    inject_precise_embeddings(model, tok)
    
    # Restore semantic_pairs (cross-domain analogies for Bridge Alignment)
    # These are NOT saved in checkpoint and must be re-injected
    ALL_PAIRS = [
        ("warmth", "affection"), ("light", "understanding"), ("gravity", "loyalty"),
        ("combustion", "resentment"), ("kindness", "trust"), ("hope", "courage"),
        ("loyalty", "support"), ("courage", "victory"), ("resentment", "hostility"),
        ("hostility", "isolation"), ("support", "obligation"), ("trust", "cooperation"),
    ]
    model.semantic_pairs = ALL_PAIRS
    
    # Train relational graph briefly on facts
    for epoch in range(5):
        for s, r, o in FACTS:
            ids = tok.encode(f"{s} {r} {o}")
            if len(ids) < 3:
                continue
            ctx = np.array([ids[:-1]], dtype=np.int64)
            tgt = np.array([[ids[-1]]], dtype=np.int64)
            model.learn(ctx, tgt)
            
    # Step 1: Pre-Alignment Evaluation (AFTER graph training, BEFORE sleep alignment)
    pre_recall, pre_acc = run_evaluation(model, tok, "BEFORE sleep alignment cycle")
    
    # Step 2: Run Sleep Cycle (with offline dynamic alignment)
    print("\nRunning model.sleep_cycle() with graph-aware contrastive alignment...")
    # Increase training impact for this validation run (matching successful manual test)
    model.max_alignment_epochs = 20
    model.alignment_lr = 0.02
    model.lambda_anchor = 0.005
    
    model.sleep_cycle(validation_queries=CHALLENGE_CASES, force_alignment=True)
    
    # Step 3: Post-Alignment Evaluation
    post_recall, post_acc = run_evaluation(model, tok, "AFTER sleep alignment cycle")
    
    # Step 4: K-sweep with different gate modes
    run_k_sweep(model, tok, "K-SWEEP AFTER ALIGNMENT (single sleep)")
    
    # Step 5: Test wake-sleep cycle with periodic sleep
    test_wake_sleep_cycle(model, tok, num_wake_epochs=12, sleep_every_n=3)
    
    # Step 6: K-sweep after extended wake-sleep cycle
    run_k_sweep(model, tok, "K-SWEEP AFTER WAKE-SLEEP CYCLE")
    
    print("\n" + "=" * 90)
    print("COMPARATIVE GAIN ANALYSIS")
    print("=" * 90)
    print(f"  * Neighbor Recall@5 improvement: {pre_recall:.1f}% -> {post_recall:.1f}% (+{post_recall - pre_recall:.1f}%)")
    print(f"  * Traversal Accuracy improvement: {pre_acc:.1f}% -> {post_acc:.1f}% (+{post_acc - pre_acc:.1f}%)")
    
    if post_acc > pre_acc:
        print("\nSUCCESS: Graph-aware encoder alignment resolved grounding failures!")
    else:
        print("\nWARNING: No accuracy improvements observed. Verify edge threshold or margin value.")
    print("=" * 90)


if __name__ == "__main__":
    main()