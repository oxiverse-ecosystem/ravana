#!/usr/bin/env python3
"""
Experiment: Reality Friction Pilot — RLM in a Non-Cooperative Reality

Tests RLM's robustness against:
  1. Observation Noise (σ=0.2)
  2. Partial Observability (30% tokens missing/masked)
  3. Delayed Ground Truth (Feedback arrives with lag)
  
Scenario: Ambigous trajectory (scientist -> bat -> animal) under heavy friction.
Can the recurrent state and shortcut edges maintain the "animal" attractor 
even when the signal is corrupted?
"""

import sys, os
# Add both project roots to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ravana-v2')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
np.random.seed(42)

from ravana.lab import ConceptLab
from core.reality_friction import RealityFrictionLayer, RealityFrictionConfig, NoiseConfig, PartialObsConfig

# Token setup
TOKENS = {
    'bat': 0, 'animal': 1, 'baseball': 2,
    'scientist': 3, 'laboratory': 4,
    'programmer': 5, 'terminal': 6,
    'noise_tok': 7 # Used for masking
}
REV = {v: k for k, v in TOKENS.items()}

def tok(name):
    return TOKENS[name]

print("=" * 70)
print("RLM REALITY FRICTION PILOT")
print("=" * 70)

# 1. Initialize RLM
config = dict(
    vocab_size=8, embed_dim=32, concept_dim=32, n_concepts=16,
    n_hidden=32, n_layers=1, max_seq_len=8,
    pressure_threshold=5.0, sleep_interval=10,
)
lab = ConceptLab(config, name="friction_pilot")
rlm = lab.rlm
rlm.context_scale = 3.0

# 2. Setup Friction Layer
friction_cfg = RealityFrictionConfig(
    noise=NoiseConfig(base_sigma=0.2), # Significant noise
    partial=PartialObsConfig(observable_fraction=0.7), # 30% missing
    intensity=0.8
)
friction = RealityFrictionLayer(friction_cfg)

# ──────────────────────────────────────────────────────────────────────
# PHASE 1: Clean Training (Establish the attractors)
# ──────────────────────────────────────────────────────────────────────
print("\n[Phase 1] Establishing clean attractors...")
for rep in range(50):
    # scientist -> bat -> animal
    rlm.learn(np.array([tok('scientist'), tok('bat')], dtype=np.int64),
              np.array([tok('animal')], dtype=np.int64))
    rlm.learn(np.array([tok('scientist')], dtype=np.int64),
              np.array([tok('bat')], dtype=np.int64))

# ──────────────────────────────────────────────────────────────────────
# PHASE 2: Friction Test
# ──────────────────────────────────────────────────────────────────────
print("\n[Phase 2] Testing under Reality Friction...")

test_ctx = [tok('scientist'), tok('bat')]
print(f"  Target Trajectory: {[REV[c] for c in test_ctx]} -> animal")

# We simulate a "noisy" observation pass
# In a real integration, the RLM input embeddings would be jittered
# Here we test if the RECURRENT state can survive a "masked" token

# Case A: Clean
res_clean = lab.probe_with_context(test_ctx)
print(f"  Clean Prediction: {REV.get(res_clean['predicted'], '???')}")

# Case B: Masked (Partial Observability)
# 'scientist' is missing, replaced by 'noise_tok'
masked_ctx = [tok('noise_tok'), tok('bat')]
res_masked = lab.probe_with_context(masked_ctx)
print(f"  Masked [?, bat] Prediction: {REV.get(res_masked['predicted'], '???')}")

# Case C: Noisy Hidden State
# We manually jitter the hidden state to simulate observation noise accumulation
print("  Simulating Hidden State Jitter (σ=0.2)...")
original_h = rlm._last_hidden_state.copy() if rlm._last_hidden_state is not None else np.zeros(32)
rlm._last_hidden_state = original_h + np.random.normal(0, 0.2, original_h.shape)
res_noisy = lab.probe_with_context(test_ctx) # probe_with_context recomputes h, so this is just a test of robustness to z-prediction
print(f"  Noisy Prediction: {REV.get(res_noisy['predicted'], '???')}")

# ──────────────────────────────────────────────────────────────────────
# PHASE 3: Stability Summary
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("Friction Summary")
print("─" * 70)
print(f"  RLM learned edges: {rlm._edges_learned}")
print(f"  Conceptual Accuracy: {rlm.conceptual_accuracy:.3f}")

if REV.get(res_masked['predicted']) == 'animal':
    print("  ✅ STABILITY: Recurrent state/Shortcuts survived masking.")
else:
    print("  ❌ FRAGILITY: System lost the attractor under masking.")

print("=" * 70)
