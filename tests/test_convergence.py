#!/usr/bin/env python3
"""Precision test: verify RLM learns specific distant causal transitions."""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
np.random.seed(42)

print("=" * 60)
print("RAVANA RLM Precision Convergence Test")
print("=" * 60)

import ravana as torch
from ravana import nn
from ravana import StateTensor

# ── Simple deterministic sequence with WIDE semantic gaps ──────────────
# Train: 1 → 50 → 25 → 10 → 40 (all steps are FAR apart on the unit circle)
# This tests whether the model learns EDGES, not geometric coincidence.

TRAIN_SEQ = [1, 50, 25, 10, 40, 1, 50, 25, 10, 40]
N_REPEATS = 20  # 200 training steps total

rlm = nn.RLM(
    vocab_size=64, embed_dim=32, concept_dim=32, n_concepts=128,
    n_hidden=32, n_layers=1, max_seq_len=16,
    free_energy_threshold=5.0, sleep_interval=15,
)

print(f"\nInitial: {rlm}")

# Verify the transitions are genuinely distant
print(f"\nSemantic distances between consecutive tokens in sequence:")
for i in range(len(TRAIN_SEQ) - 1):
    a, b = TRAIN_SEQ[i], TRAIN_SEQ[i + 1]
    a_e = rlm.token_embed(StateTensor(np.array([a]))).data[0]
    b_e = rlm.token_embed(StateTensor(np.array([b]))).data[0]
    sim = float(np.dot(a_e, b_e) / (np.linalg.norm(a_e) * np.linalg.norm(b_e) + 1e-15))
    dist = 1.0 - sim
    print(f"  {a} → {b}:  cosine_dist={dist:.4f}  (sim={sim:.4f})")

# Verify input concepts DON'T overlap with output concepts for these pairs
print(f"\nConcept overlap WITHOUT edges (should be low for distant pairs):")
for i in range(len(TRAIN_SEQ) - 1):
    a, b = TRAIN_SEQ[i], TRAIN_SEQ[i + 1]
    a_e = rlm.token_embed(StateTensor(np.array([a]))).data[0]
    b_e = rlm.token_embed(StateTensor(np.array([b]))).data[0]
    a_concepts = set(rlm.graph.find_similar(a_e, k=5))
    b_concepts = set(rlm.graph.find_similar(b_e, k=5))
    overlap = len(a_concepts & b_concepts)
    print(f"  {a}→{b}: input∩output={overlap}/5")

# ── Train ─────────────────────────────────────────────────────────────
seq = []
for _ in range(N_REPEATS):
    seq.extend(TRAIN_SEQ)

print(f"\nTraining on {len(seq)} steps...")
errors = []

for i in range(len(seq) - 1):
    inp = np.array([seq[i]], dtype=np.int64)
    nxt = np.array([seq[i + 1]], dtype=np.int64)
    err = rlm.learn(inp, nxt)
    errors.append(err)

# ── Verify edges match the causal transitions ─────────────────────────
print(f"\n{'='*60}")
print(f"Edge verification after training:")
print(f"  Total edges: {len(rlm.graph.edges)}")

edges_matched = 0
total_transitions = len(TRAIN_SEQ) - 1
for i in range(total_transitions):
    src, tgt = TRAIN_SEQ[i], TRAIN_SEQ[i + 1]
    src_concept = rlm._nearest_concept(
        rlm.token_embed(StateTensor(np.array([src]))).data[0])
    tgt_concept = rlm._nearest_concept(
        rlm.token_embed(StateTensor(np.array([tgt]))).data[0])
    edge = rlm.graph.get_edge(src_concept, tgt_concept)
    exists = edge is not None
    weight = edge.weight if edge else 0.0
    print(f"  {src}→{tgt} (c{src_concept}→c{tgt_concept}): edge={exists} w={weight:.3f}")
    if exists:
        edges_matched += 1

print(f"\n  Causal edges learned: {edges_matched}/{total_transitions}")

# ── Test prediction: does forward() correctly predict the NEXT concept? ──
print(f"\nPrediction test (after training):")
correct = 0
for i in range(len(TRAIN_SEQ) - 1):
    src, tgt = TRAIN_SEQ[i], TRAIN_SEQ[i + 1]
    inp = np.array([src], dtype=np.int64)
    rlm.forward(inp)
    predicted_set = set(rlm._last_predicted_concepts)
    tgt_concept = rlm._nearest_concept(
        rlm.token_embed(StateTensor(np.array([tgt]))).data[0])
    hit = tgt_concept in predicted_set
    print(f"  {src}→{tgt}: target_concept={tgt_concept} in predicted={predicted_set} → {hit}")
    if hit:
        correct += 1

print(f"\n  Concept prediction accuracy: {correct}/{total_transitions}")

# ── Test teacher-forced prediction (greedy, ground-truth input) ──────
print(f"\nTeacher-forced prediction (greedy):")
forced_correct = 0
for i in range(len(TRAIN_SEQ) - 1):
    src, tgt = TRAIN_SEQ[i], TRAIN_SEQ[i + 1]
    logits = rlm.forward(np.array([src], dtype=np.int64))
    pred_id = int(np.argmax(logits.data))
    hit = pred_id == tgt
    print(f"  {src} → predicted={pred_id}, expected={tgt} → {hit}")
    if hit:
        forced_correct += 1
print(f"  Teacher-forced accuracy: {forced_correct}/{total_transitions}")

# ── Test open-loop generation (greedy) ──────────────────────────────
print(f"\nGreedy generation test:")
gen = [TRAIN_SEQ[0]]
for _ in range(5):
    logits = rlm.forward(np.array([gen], dtype=np.int64))
    next_id = int(np.argmax(logits.data))
    gen.append(next_id)
true_next = TRAIN_SEQ[1:6]
print(f"  Prompt [{TRAIN_SEQ[0]}] → {gen[1:]}")
print(f"  Expected:                 {true_next}")
hits = sum(1 for i, g in enumerate(gen[1:]) if i < len(true_next) and g == true_next[i])
print(f"  Exact match: {hits}/5")

converged = edges_matched >= total_transitions * 0.75
print(f"\n  Converged (≥75% edges learned): {converged}")
print(f"\nFinal: {rlm}")
print("=" * 60)
