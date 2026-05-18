#!/usr/bin/env python3
"""
Experiment: Ambiguity Resolution — bank→money vs bank→river

Tests whether RLM can maintain context-dependent meaning:
  Phase 1: Simple ambiguity — bank→money AND bank→river (same input, two outputs)
  Phase 2: Context priming — cash→bank→money vs water→bank→river
  Phase 3: Delayed context — boat→water→bank→river vs paycheck→cash→bank→money

Key question: does upstream context bias downstream prediction
WITHOUT corrupting core concept identity?
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
np.random.seed(42)

from ravana.lab import ConceptLab, ExperimentPhase, Snapshot
from ravana.tensor import StateTensor

TOKENS = {
    'bank': 0, 'money': 1, 'river': 2,
    'cash': 3, 'water': 4, 'boat': 5,
    'paycheck': 6, 'deposit': 7, 'canoe': 8,
}
REV = {v: k for k, v in TOKENS.items()}

def tok(name):
    return TOKENS[name]

def concept_name(lab, cid):
    tid = lab.concept_token_map(cid)
    return REV.get(tid, f"c{cid}")

print("=" * 70)
print("CONCEPT PHYSICS LAB — Experiment: Ambiguity Resolution")
print("=" * 70)

config = dict(
    vocab_size=9, embed_dim=32, concept_dim=32, n_concepts=18,
    n_hidden=32, n_layers=1, max_seq_len=8,
    pressure_threshold=5.0, sleep_interval=5,
)

lab = ConceptLab(config, name="ambiguity_resolution")
rlm = lab.rlm

# ──────────────────────────────────────────────────────────────────────
# PHASE 1: Simple Ambiguity — bank→money AND bank→river
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("PHASE 1: Simple Ambiguity")
print("─" * 70)

lab.run_phase(ExperimentPhase("ambiguous", [
    (tok('bank'), tok('money')),
    (tok('bank'), tok('river')),
], n_repeats=15))

bank_concept = lab.token_concept_map(tok('bank'))
money_concept = lab.token_concept_map(tok('money'))
river_concept = lab.token_concept_map(tok('river'))

print(f"\n  bank concept: c{bank_concept} → nearest concept to 'money': c{money_concept}, 'river': c{river_concept}")

# Probe: bank → ?
probe = lab.probe(tok('bank'))
print(f"  Probe bank → predicted={REV.get(probe['predicted'], probe['predicted'])} "
      f"(confidence={probe['confidence']:.3f})")

# Check outgoing edges from bank
print(f"  Edges from c{bank_concept}:")
for (s, t), e in rlm.graph.edges.items():
    if s == bank_concept:
        print(f"    → c{t} ({concept_name(lab, t)}) w={e.weight:.3f}")

# ──────────────────────────────────────────────────────────────────────
# PHASE 2: Context Priming
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("PHASE 2: Context Priming")
print("─" * 70)

# Train contextual transitions with full-sequence context
# Interleave both contexts so no edge decays more than the other
for rep in range(80):
    # cash → bank → money
    inp = np.array([tok('cash'), tok('bank')], dtype=np.int64)
    nxt = np.array([tok('money')], dtype=np.int64)
    rlm.learn(inp, nxt)
    rlm.learn(np.array([tok('cash')], dtype=np.int64), np.array([tok('bank')], dtype=np.int64))
    rlm.learn(np.array([tok('bank')], dtype=np.int64), np.array([tok('money')], dtype=np.int64))
    # water → bank → river
    inp = np.array([tok('water'), tok('bank')], dtype=np.int64)
    nxt = np.array([tok('river')], dtype=np.int64)
    rlm.learn(inp, nxt)
    rlm.learn(np.array([tok('water')], dtype=np.int64), np.array([tok('bank')], dtype=np.int64))
    rlm.learn(np.array([tok('bank')], dtype=np.int64), np.array([tok('river')], dtype=np.int64))

lab.snapshots.append(Snapshot(rlm, "after_context_priming"))

# Enable context path for probing
rlm.context_scale = 3.0

# Probe with context: cash bank → ? should prefer money
print(f"\n  Probe with context [cash, bank] → ", end="")
ctx = [tok('cash'), tok('bank')]
result = lab.probe_with_context(ctx)
print(f"predicted={REV.get(result['predicted'], result['predicted'])} "
      f"(confidence={result['confidence']:.3f})")

# Probe with context: water bank → ? should prefer river
print(f"  Probe with context [water, bank] → ", end="")
ctx = [tok('water'), tok('bank')]
result = lab.probe_with_context(ctx)
print(f"predicted={REV.get(result['predicted'], result['predicted'])} "
      f"(confidence={result['confidence']:.3f})")

# Direct probe: bank alone (no context)
rlm.context_scale = 0.0
probe_alone = lab.probe(tok('bank'))
rlm.context_scale = 3.0
print(f"  Probe bank alone → predicted={REV.get(probe_alone['predicted'], probe_alone['predicted'])}")

# Check edges from bank concept
print(f"  Edges from c{bank_concept}:")
for (s, t), e in rlm.graph.edges.items():
    if s == bank_concept:
        print(f"    → c{t} ({concept_name(lab, t)}) w={e.weight:.3f} conf={e.confidence:.3f}")

# ──────────────────────────────────────────────────────────────────────
# PHASE 3: Delayed Context
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("PHASE 3: Delayed Context (4-step chains)")
print("─" * 70)

# Train full chains with sequence context (interleaved)
for rep in range(80):
    # boat→water→bank→river
    inp = np.array([tok('boat'), tok('water'), tok('bank')], dtype=np.int64)
    nxt = np.array([tok('river')], dtype=np.int64)
    rlm.learn(inp, nxt)
    rlm.learn(np.array([tok('boat')], dtype=np.int64), np.array([tok('water')], dtype=np.int64))
    rlm.learn(np.array([tok('water')], dtype=np.int64), np.array([tok('bank')], dtype=np.int64))
    rlm.learn(np.array([tok('bank')], dtype=np.int64), np.array([tok('river')], dtype=np.int64))
    # paycheck→cash→bank→money
    inp = np.array([tok('paycheck'), tok('cash'), tok('bank')], dtype=np.int64)
    nxt = np.array([tok('money')], dtype=np.int64)
    rlm.learn(inp, nxt)
    rlm.learn(np.array([tok('paycheck')], dtype=np.int64), np.array([tok('cash')], dtype=np.int64))
    rlm.learn(np.array([tok('cash')], dtype=np.int64), np.array([tok('bank')], dtype=np.int64))
    rlm.learn(np.array([tok('bank')], dtype=np.int64), np.array([tok('money')], dtype=np.int64))

lab.snapshots.append(Snapshot(rlm, "after_delayed_context"))

# Test long context chains
print(f"\n  Probe: [boat, water, bank] → ", end="")
result = lab.probe_with_context([tok('boat'), tok('water'), tok('bank')])
print(f"predicted={REV.get(result['predicted'], result['predicted'])} "
      f"(expected=river)")

print(f"  Probe: [paycheck, cash, bank] → ", end="")
result = lab.probe_with_context([tok('paycheck'), tok('cash'), tok('bank')])
print(f"predicted={REV.get(result['predicted'], result['predicted'])} "
      f"(expected=money)")

print(f"  Probe: bank alone → predicted=", end="")
rlm.context_scale = 0.0
probe_alone = lab.probe(tok('bank'))
rlm.context_scale = 3.0
print(f"{REV.get(probe_alone['predicted'], probe_alone['predicted'])}")

# ──────────────────────────────────────────────────────────────────────
# Analysis: does context change anything?
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("CONTEXT ANALYSIS")
print("─" * 70)

print(f"\n  Is [cash,bank] prediction different from bank-alone? ", end="")
rlm.context_scale = 3.0
p1 = lab.probe_with_context([tok('cash'), tok('bank')])
rlm.context_scale = 0.0
p0 = lab.probe(tok('bank'))
same = p1['predicted'] == p0['predicted']
print(f"{'NO — context ignored' if same else 'YES — context matters'}")

print(f"  Is [water,bank] prediction different from bank-alone? ", end="")
rlm.context_scale = 3.0
p2 = lab.probe_with_context([tok('water'), tok('bank')])
same = p2['predicted'] == p0['predicted']
print(f"{'NO — context ignored' if same else 'YES — context matters'}")

# Check if hidden layers contribute
print(f"\n  Context path uses hidden state from recurrent encoding of full sequence")
print(f"  context_scale={rlm.context_scale}")

# ── Metrics ──
print(f"\n── Edge topology ──")
topo = lab.edge_topology_summary(-1)
print(f"  Total edges: {topo['n_edges']}, weight≥1.0: {topo['n_edges_weight_1']}")

loc = lab.pressure_localization(-1)
print(f"  Pressure localization: entropy={loc['normalized_entropy']:.3f}, hotspots={loc['hotspots']}")

drift = lab.attractor_preservation(before_idx=0, after_idx=-1)
print(f"  Mean attractor drift: L2={drift['mean_drift']:.4f}")

print(f"\nFinal: {rlm}")
print("=" * 70)
