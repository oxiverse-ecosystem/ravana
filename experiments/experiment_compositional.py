#!/usr/bin/env python3
"""
Experiment: Compositional Generalization in RLM

Tests whether concepts recombine under pressure:
1. Stable concepts: fire→hot/burn, ice→cold/freeze, water→wet
2. Ambiguous probe: what does "fire" predict? (hot vs burn)
3. Contradiction: fire→cold (opposes fire→hot)
4. Measure: attractor drift, pressure localization, edge topology change

Architecture note: current RLM binds from the LAST token only (no sequence
context), so context-dependent transitions (blue_fire→cold) require
combining hidden-layer state with concept output — future work.
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
np.random.seed(42)

import ravana as torch
from ravana import nn
from ravana.lab import ConceptLab, ExperimentPhase
from ravana.tensor import StateTensor

print("=" * 70)
print("CONCEPT PHYSICS LAB — Experiment: Compositional Generalization")
print("=" * 70)

# ── Build the Lab ─────────────────────────────────────────────────────
config = dict(
    vocab_size=12, embed_dim=32, concept_dim=32, n_concepts=24,
    n_hidden=32, n_layers=1, max_seq_len=8,
    pressure_threshold=5.0, sleep_interval=10,
)

lab = ConceptLab(config, name="compositional_generalization")
rlm = lab.rlm

# ── Token/Concept Mapping ─────────────────────────────────────────────
TOKENS = {
    'fire': 0, 'hot': 1, 'burn': 2,
    'ice': 3, 'cold': 4, 'freeze': 5,
    'water': 6, 'wet': 7, 'steam': 8,
    'blue': 9, 'magic': 10, 'empty': 11,
}
REV = {v: k for k, v in TOKENS.items()}

def tok(name):
    return TOKENS[name]

def concept_name(lab, cid):
    tid = lab.concept_token_map(cid)
    return REV.get(tid, f"c{cid}")

def name_concepts(l):
    r = l.rlm
    for tok_name, tok_id in TOKENS.items():
        cid = lab.token_concept_map(tok_id)
        node = rlm.graph.get_node(cid)
        if node and node.label.startswith('c'):
            node.label = f"{tok_name}_c{cid}"
    for tok_name, tok_id in TOKENS.items():
        emb = rlm.token_embed(StateTensor(np.array([tok_id]))).data[0]
        for k, (cid, sim) in enumerate(rlm.graph.find_similar(emb, k=3)):
            node = rlm.graph.get_node(cid)
            if node and node.label.startswith('c'):
                node.label = f"near_{tok_name}_{k}_{cid}"

# ──────────────────────────────────────────────────────────────────────
# PHASE 1: Stable concepts
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("PHASE 1: Stable Concept Training")
print("─" * 70)

stable_phase = ExperimentPhase("stable", [
    (tok('fire'), tok('hot')),
    (tok('fire'), tok('burn')),
    (tok('ice'), tok('cold')),
    (tok('ice'), tok('freeze')),
    (tok('water'), tok('wet')),
], n_repeats=15)

snap1 = lab.run_phase(stable_phase)
name_concepts(lab)

print(f"\nAfter phase 1: {rlm}")
print(f"  Accuracy: {rlm.conceptual_accuracy:.3f}")

# ── Probe: fire → ? (should be hot OR burn) ──
p = lab.probe(tok('fire'))
fire_concept = lab.token_concept_map(tok('fire'))
print(f"\n  Probe fire → predicted={REV.get(p['predicted'], p['predicted'])} "
      f"(confidence={p['confidence']:.3f}, entropy={p['entropy']:.3f})")
print(f"  Fire's concept: c{fire_concept}")

# Check edges from fire's concept
print(f"  Edges from c{fire_concept}:")
for (s, t), e in rlm.graph.edges.items():
    if s == fire_concept:
        target_name = concept_name(lab, t)
        print(f"    → c{t} ({target_name}) w={e.weight:.3f} conf={e.confidence:.3f}")

# ──────────────────────────────────────────────────────────────────────
# PHASE 2: Contradiction
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("PHASE 2: Contradiction Injection — fire→cold")
print("─" * 70)

contra_phase = ExperimentPhase("contradiction", [
    (tok('fire'), tok('cold')),  # contradicts fire→hot
], n_repeats=20)

snap2 = lab.run_phase(contra_phase)

print(f"\nAfter phase 2: {rlm}")

# ── Measure: what changed? ──
p2 = lab.probe(tok('fire'))
print(f"\n  Probe fire → predicted={REV.get(p2['predicted'], p2['predicted'])} "
      f"(confidence={p2['confidence']:.3f}, entropy={p2['entropy']:.3f})")

# Check edges from fire's concept
print(f"  Edges from c{fire_concept}:")
for (s, t), e in rlm.graph.edges.items():
    if s == fire_concept:
        target_name = concept_name(lab, t)
        print(f"    → c{t} ({target_name}) w={e.weight:.3f} conf={e.confidence:.3f}")

# ── Metrics ──
print(f"\n── Metrics ──")
loc1 = lab.free_energy_localization(0)
loc2 = lab.free_energy_localization(1)
print(f"Pressure localization: before={loc1['normalized_entropy']:.3f} "
      f"after={loc2['normalized_entropy']:.3f} "
      f"hotspots={loc2['hotspots']}")

drift = lab.attractor_drift(fire_concept, 0, 1)
print(f"Fire attractor drift: L2={drift['l2_distance']:.4f} "
      f"cos={drift['cosine_similarity']:.4f}")

preservation = lab.attractor_preservation(before_idx=0, after_idx=1)
print(f"Mean attractor drift (all concepts): L2={preservation['mean_drift']:.4f} "
      f"max={preservation['max_drift']:.4f}")

branches = lab.branch_detection(fire_concept, radius=0.2)
print(f"Branches around fire: {branches['n_neighbors']} nearby concepts")
for nb in branches['neighbors'][:5]:
    print(f"  c{nb['id']} sim={nb['similarity']:.3f} conf={nb['confidence']:.3f} "
          f"act={nb['activation']:.3f}")

eff = lab.sleep_efficiency()
print(f"Sleep efficiency: mean_drop={eff['mean_drop']:.3f} "
      f"final_pressure={eff['final_pressure']:.3f}")

topo = lab.edge_topology_summary(0)
topo2 = lab.edge_topology_summary(1)
print(f"Edge topology: {topo['n_edges']}→{topo2['n_edges']} edges, "
      f"weight≥1.0: {topo2['n_edges_weight_1']}")

# ──────────────────────────────────────────────────────────────────────
# PHASE 3: Long-horizon chaining test
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("PHASE 3: Long-horizon chaining — fire→hot→steam")
print("─" * 70)

chain_phase = ExperimentPhase("chaining", [
    (tok('hot'), tok('steam')),
    (tok('steam'), tok('water')),
], n_repeats=10)

snap3 = lab.run_phase(chain_phase)

print(f"\nAfter phase 3: {rlm}")
p = lab.probe(tok('fire'))
print(f"  Probe fire → predicted={REV.get(p['predicted'], p['predicted'])}")
p = lab.probe(tok('hot'))
print(f"  Probe hot → predicted={REV.get(p['predicted'], p['predicted'])}")
p = lab.probe(tok('steam'))
print(f"  Probe steam → predicted={REV.get(p['predicted'], p['predicted'])}")

# ──────────────────────────────────────────────────────────────────────
# Full Report
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(lab.report())
print("=" * 70)
