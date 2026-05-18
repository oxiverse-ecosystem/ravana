#!/usr/bin/env python3
"""
Experiment: Hierarchical Ambiguity — context specificity and override

Tests whether RLM can handle nested or conflicting hierarchical contexts:
  - Base context: scientist → laboratory → mouse → rodent
  - Overriding context: fixed → driver → mouse → device
  - Test case: scientist → laboratory → fixed → driver → mouse → ?

Does the system correctly prioritize the more specific/recent 'fixed/driver'
context over the general 'scientist/laboratory' context?
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
np.random.seed(42)

from ravana.lab import ConceptLab, ExperimentPhase, Snapshot
from ravana.tensor import StateTensor

TOKENS = {
    'mouse': 0, 'rodent': 1, 'device': 2,
    'scientist': 3, 'laboratory': 4,
    'programmer': 5, 'terminal': 6,
    'fixed': 7, 'driver': 8, 'software': 9,
    'biology': 10, 'computer': 11
}
REV = {v: k for k, v in TOKENS.items()}

def tok(name):
    return TOKENS[name]

def concept_name(lab, cid):
    tid = lab.concept_token_map(cid)
    return REV.get(tid, f"c{cid}")

print("=" * 70)
print("CONCEPT PHYSICS LAB — Experiment: Hierarchical Ambiguity")
print("=" * 70)

config = dict(
    vocab_size=12, embed_dim=32, concept_dim=32, n_concepts=24,
    n_hidden=32, n_layers=1, max_seq_len=12,
    pressure_threshold=5.0, sleep_interval=10,
)

lab = ConceptLab(config, name="hierarchical_ambiguity")
rlm = lab.rlm

# ──────────────────────────────────────────────────────────────────────
# PHASE 1 & 2: Mixed Training
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("PHASE 1 & 2: Mixed Training")
print("─" * 70)

for rep in range(100):
    # Scientist path
    rlm.learn(np.array([tok('scientist'), tok('laboratory'), tok('mouse')], dtype=np.int64),
              np.array([tok('rodent')], dtype=np.int64))
    rlm.learn(np.array([tok('scientist')], dtype=np.int64),
              np.array([tok('laboratory')], dtype=np.int64))
    rlm.learn(np.array([tok('laboratory')], dtype=np.int64),
              np.array([tok('mouse')], dtype=np.int64))
    
    # Programmer path
    rlm.learn(np.array([tok('programmer'), tok('terminal'), tok('mouse')], dtype=np.int64),
              np.array([tok('device')], dtype=np.int64))
    rlm.learn(np.array([tok('programmer')], dtype=np.int64),
              np.array([tok('terminal')], dtype=np.int64))
    rlm.learn(np.array([tok('terminal')], dtype=np.int64),
              np.array([tok('mouse')], dtype=np.int64))

    # Fixed/Driver path
    rlm.learn(np.array([tok('fixed'), tok('driver'), tok('mouse')], dtype=np.int64),
              np.array([tok('device')], dtype=np.int64))
    rlm.learn(np.array([tok('fixed')], dtype=np.int64),
              np.array([tok('driver')], dtype=np.int64))
    rlm.learn(np.array([tok('driver')], dtype=np.int64),
              np.array([tok('mouse')], dtype=np.int64))

# ──────────────────────────────────────────────────────────────────────
# PHASE 3: Hierarchical Conflict Test
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("PHASE 3: Hierarchical Conflict Test")
print("─" * 70)

# Check edges from mouse
mouse_concept = lab.token_concept_map(tok('mouse'))
print(f"  Edges from mouse (c{mouse_concept}):")
for (s, t), e in rlm.graph.edges.items():
    if s == mouse_concept:
        target_name = REV.get(lab.concept_token_map(t), f"c{t}")
        print(f"    → {target_name} w={e.weight:.3f} shortcut={e.shortcut}")

# The Test Case: scientist + laboratory (general) vs fixed + driver (specific)
hierarchical_ctx = [tok('scientist'), tok('laboratory'), tok('fixed'), tok('driver'), tok('mouse')]

print(f"  Testing hierarchy (Specific Late): {[REV[c] for c in hierarchical_ctx]}")
result = lab.probe_with_context(hierarchical_ctx)
pred_name = REV.get(result['predicted'], f"c{result['predicted']}")
print(f"    → Predicted: {pred_name}")

# Reversed hierarchy: fixed + driver (general) vs scientist + laboratory (specific)
reversed_ctx = [tok('fixed'), tok('driver'), tok('scientist'), tok('laboratory'), tok('mouse')]
print(f"\n  Testing hierarchy (General Late): {[REV[c] for c in reversed_ctx]}")
result = lab.probe_with_context(reversed_ctx)
pred_name = REV.get(result['predicted'], f"c{result['predicted']}")
print(f"    → Predicted: {pred_name}")

# ──────────────────────────────────────────────────────────────────────
# PHASE 4: Semantic Wormhole Check
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("PHASE 4: Semantic Wormhole Check")
print("─" * 70)
# Check if scientist ... driver jumps incorrectly
wormhole_ctx = [tok('scientist'), tok('driver')]
print(f"  Testing wormhole: {[REV[c] for c in wormhole_ctx]} (should not necessarily predict 'device' or 'rodent' strongly)")
result = lab.probe_with_context(wormhole_ctx)
pred_name = REV.get(result['predicted'], f"c{result['predicted']}")
print(f"    → Predicted: {pred_name}")

print(f"\nFinal: {rlm}")
print("=" * 70)
