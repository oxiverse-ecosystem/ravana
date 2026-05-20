"""
RAVANA - Full Architecture Test

End-to-end test of the RLM cognitive architecture:
  - Predictive coding settle loop (no backprop)
  - Contradiction detection and resolution (inhibitory edges)
  - ConceptBindingMap (ambiguity, disambiguation)
  - context_scale (dual-path prediction)
  - Sleep cycle (structural plasticity, homeostasis, binding maintenance)
  - Save/load persistence

Usage:
    python test_rlm_full.py
"""

import sys
import os
import time
import tempfile
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ravana-v2'))

from ravana_ml.nn.rlm import RLM
from ravana_ml.nn import functional as F
from ravana_ml.tensor import StateTensor


# ============================================================
# Vocabulary
# ============================================================

VOCAB = [
    '<start>',     # 0
    'fire',        # 1
    'hot',         # 2
    'cold',        # 3
    'bird',        # 4
    'fly',         # 5
    'swim',        # 6
    'apple',       # 7
    'fruit',       # 8
    'tech',        # 9
    'sun',         # 10
    'bright',      # 11
    'dark',        # 12
    'water',       # 13
    'liquid',      # 14
    'pie',         # 15
    'company',     # 16
    'sky',         # 17
    'ocean',       # 18
    'night',       # 19
]
VOCAB_SIZE = len(VOCAB)
W = {w: i for i, w in enumerate(VOCAB)}


def header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def section(title):
    print(f"\n  --- {title} ---")


# ============================================================
# Test Phases
# ============================================================

def phase_1_learning(model):
    """Phase 1: Establish consistent associations."""
    header("Phase 1: Learning Consistent Associations")

    pairs = [
        ('<start>', 'fire',  'hot'),
        ('<start>', 'bird',  'fly'),
        ('<start>', 'sun',   'bright'),
        ('<start>', 'water', 'liquid'),
        ('<start>', 'apple', 'fruit'),
    ]

    errors = []
    for epoch in range(50):
        np.random.shuffle(pairs)
        for ctx, trigger, target in pairs:
            err = model.learn(np.array([[W[ctx], W[trigger]]]), np.array([[W[target]]]))
            errors.append(err)

    avg_first = np.mean(errors[:10])
    avg_last = np.mean(errors[-10:])
    print(f"  Training pairs: {len(pairs)}")
    print(f"  Epochs: 50")
    print(f"  Avg error (first 10): {avg_first:.4f}")
    print(f"  Avg error (last 10):  {avg_last:.4f}")
    print(f"  Edges formed: {model._edges_learned}")
    print(f"  Bindings: {len(model.binding_map)}")
    print(f"  Conceptual accuracy: {model.conceptual_accuracy:.3f}")

    section("Graph Structure")
    for (s, t), e in sorted(model.graph.edges.items()):
        print(f"    {s}->{t}: w={e.weight:.3f} conf={e.confidence:.3f} type={e.edge_type}")

    return errors


def phase_2_contradiction(model):
    """Phase 2: Inject contradictions and observe detection."""
    header("Phase 2: Contradiction Injection")

    # Same triggers, opposite targets
    contradictions = [
        ('<start>', 'fire',  'cold'),   # contradicts fire->hot
        ('<start>', 'bird',  'swim'),   # contradicts bird->fly
        ('<start>', 'sun',   'dark'),   # contradicts sun->bright
    ]

    errors = []
    for epoch in range(30):
        for ctx, trigger, target in contradictions:
            err = model.learn(np.array([[W[ctx], W[trigger]]]), np.array([[W[target]]]))
            errors.append(err)

    section("Contradiction Detection")
    print(f"  Contradiction triples: {len(contradictions)}")
    print(f"  Hotspots: {model.graph.contradiction_hotspots}")
    for nid in model.graph.contradiction_hotspots:
        node = model.graph.get_node(nid)
        if node:
            print(f"    Node {nid}: contradiction_count={node.contradiction_count}, free_energy={node.prediction_free_energy:.1f}")

    section("Competing Edges")
    source_targets = defaultdict(list)
    for (s, t), e in model.graph.edges.items():
        source_targets[s].append((t, e.weight, e.edge_type))
    for src, targets in source_targets.items():
        if len(targets) > 1:
            parts = [f"{t} (w={w:.3f}, {et})" for t, w, et in targets]
            print(f"    {src} -> {' | '.join(parts)}")

    section("Edge Counts")
    excitatory = sum(1 for e in model.graph.edges.values() if e.edge_type == 'excitatory')
    inhibitory = sum(1 for e in model.graph.edges.values() if e.edge_type == 'inhibitory')
    print(f"  Excitatory: {excitatory}")
    print(f"  Inhibitory: {inhibitory}")

    return errors


def phase_3_sleep(model):
    """Phase 3: Sleep cycle — contradiction resolution."""
    header("Phase 3: Sleep Cycle (Contradiction Resolution)")

    inhibitory_before = sum(1 for e in model.graph.edges.values() if e.edge_type == 'inhibitory')
    hotspots_before = len(model.graph.contradiction_hotspots)

    model.sleep_cycle()

    inhibitory_after = sum(1 for e in model.graph.edges.values() if e.edge_type == 'inhibitory')
    hotspots_after = len(model.graph.contradiction_hotspots)

    section("Resolution Results")
    print(f"  Hotspots: {hotspots_before} -> {hotspots_after}")
    print(f"  Inhibitory edges: {inhibitory_before} -> {inhibitory_after}")
    print(f"  Sleep cycles completed: {model.sleep_cycles_completed}")
    print(f"  Bindings after sleep: {len(model.binding_map)}")

    section("Inhibitory Edges Formed")
    for (s, t), e in model.graph.edges.items():
        if e.edge_type == 'inhibitory':
            print(f"    {s}->{t}: w={e.weight:.3f} (suppresses competing activation)")

    section("Binding Status")
    ambiguous_tokens = []
    for tok_id in range(VOCAB_SIZE):
        if model.binding_map.is_ambiguous(tok_id):
            score = model.binding_map.ambiguity_score(tok_id)
            ambiguous_tokens.append((tok_id, score))
    if ambiguous_tokens:
        for tok_id, score in ambiguous_tokens:
            concepts = model.binding_map.get_concepts(tok_id)
            concept_ids = [b.concept_id for b in concepts]
            print(f"    {VOCAB[tok_id]}: ambiguous (score={score:.3f}), concepts={concept_ids}")
    else:
        print(f"    No ambiguous tokens detected")


def phase_4_disambiguation(model):
    """Phase 4: Test context-dependent disambiguation."""
    header("Phase 4: Context-Dependent Disambiguation")

    # Train ambiguous associations
    section("Training Ambiguous Associations")
    ambiguous_pairs = [
        ('<start>', 'apple', 'fruit'),    # apple as food
        ('<start>', 'apple', 'tech'),     # apple as company
        ('<start>', 'apple', 'pie'),      # reinforces food meaning
    ]
    for ctx, trigger, target in ambiguous_pairs:
        model.learn(np.array([[W[ctx], W[trigger]]]), np.array([[W[target]]]))
        print(f"    Trained: {trigger} -> {target}")

    # Check ambiguity
    section("Ambiguity Detection")
    for token in ['fire', 'bird', 'apple', 'sun']:
        tok_id = W[token]
        is_amb = model.binding_map.is_ambiguous(tok_id)
        score = model.binding_map.ambiguity_score(tok_id)
        concepts = model.binding_map.get_concepts(tok_id)
        concept_ids = [b.concept_id for b in concepts]
        status = "AMBIGUOUS" if is_amb else "unambiguous"
        print(f"    {token}: {status} (score={score:.3f}), concepts={concept_ids}")

    # Test disambiguation with different contexts
    section("Disambiguation with Context")
    ctx_vec = model.graph.temporal_context
    if np.linalg.norm(ctx_vec) == 0:
        ctx_vec = model.concept_predictor(
            StateTensor(model._last_hidden_state[np.newaxis, :])
        ).data[0]

    for token in ['apple', 'fire', 'bird']:
        tok_id = W[token]
        if model.binding_map.is_ambiguous(tok_id):
            winner = model.binding_map.disambiguate(tok_id, ctx_vec, model.graph)
            print(f"    {token} -> concept {winner} (context-dependent)")
        else:
            best = model.binding_map.best_concept(tok_id)
            print(f"    {token} -> concept {best} (single meaning)")


def phase_5_dual_path(model):
    """Phase 5: Verify dual-path prediction (concept + context)."""
    header("Phase 5: Dual-Path Prediction")

    section("context_scale Verification")
    print(f"  context_scale: {model.context_scale}")
    print(f"  Concept path: graph activation * 15.0")
    print(f"  Context path: hidden state -> Linear -> logits * {model.context_scale}")

    # Forward pass and verify context path contributes
    tokens = np.array([[W['<start>'], W['fire']]])
    out = model.forward(tokens)

    ctx_logits = model._last_ctx_logits
    ctx_range = float(np.max(ctx_logits) - np.min(ctx_logits))
    out_range = float(np.max(out.data) - np.min(out.data))

    section("Path Contributions")
    print(f"  Context logits range: {ctx_range:.2f}")
    print(f"  Combined logits range: {out_range:.2f}")
    print(f"  context_scale contribution: {ctx_range * model.context_scale:.2f}")

    if ctx_range > 0.1:
        print(f"  PASS: Context path is active (range={ctx_range:.2f})")
    else:
        print(f"  WARN: Context path may be inactive (range={ctx_range:.2f})")


def phase_6_free_energy(model):
    """Phase 6: Pressure / free energy tracking."""
    header("Phase 6: Pressure & Free Energy")

    section("System Pressure")
    print(f"  Total free energy accumulated: {model.total_free_energy:.3f}")
    print(f"  Model free energy: {model._free_energy:.3f}")

    section("Node Pressure Distribution")
    high_free_energy = []
    for nid, node in model.graph.nodes.items():
        if node.prediction_free_energy > 1.0:
            high_free_energy.append((nid, node.prediction_free_energy, node.contradiction_count))
    high_free_energy.sort(key=lambda x: -x[1])
    for nid, p, cc in high_free_energy[:5]:
        print(f"    Node {nid}: free_energy={p:.2f}, contradiction_count={cc}")

    section("Settle Loop Energy")
    if hasattr(model, '_running_avg_states') and model._running_avg_states is not None:
        for i, avg in enumerate(model._running_avg_states):
            print(f"    Layer {i} running avg norm: {np.linalg.norm(avg):.4f}")
    else:
        print(f"    Running average not yet initialized")


def phase_7_persistence(model):
    """Phase 7: Save/load roundtrip."""
    header("Phase 7: Persistence (Save/Load)")

    # Snapshot state
    state_before = {
        'step_counter': model._step_counter,
        'edges_learned': model._edges_learned,
        'sleep_cycles': model.sleep_cycles_completed,
        'context_scale': model.context_scale,
        'bindings': len(model.binding_map),
        'nodes': len(model.graph.nodes),
        'edges': len(model.graph.edges),
        'total_free_energy': model.total_free_energy,
        'conceptual_accuracy': model.conceptual_accuracy,
    }

    section("State Before Save")
    for k, v in state_before.items():
        print(f"    {k}: {v}")

    # Save as zip
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
        zip_path = f.name
    model.save_zip(zip_path)

    # Load
    loaded = RLM.load_zip(zip_path)
    os.unlink(zip_path)

    section("State After Load")
    state_after = {
        'step_counter': loaded._step_counter,
        'edges_learned': loaded._edges_learned,
        'sleep_cycles': loaded.sleep_cycles_completed,
        'context_scale': loaded.context_scale,
        'bindings': len(loaded.binding_map),
        'nodes': len(loaded.graph.nodes),
        'edges': len(loaded.graph.edges),
        'total_free_energy': loaded.total_free_energy,
        'conceptual_accuracy': loaded.conceptual_accuracy,
    }
    for k, v in state_after.items():
        print(f"    {k}: {v}")

    section("Roundtrip Verification")
    mismatches = 0
    for k in state_before:
        before = state_before[k]
        after = state_after[k]
        if isinstance(before, float):
            match = abs(before - after) < 0.01
        else:
            match = before == after
        status = "OK" if match else "MISMATCH"
        if not match:
            mismatches += 1
            print(f"    {k}: {before} -> {after} [{status}]")
    if mismatches == 0:
        print(f"    All {len(state_before)} fields match")

    # Verify loaded model works
    section("Loaded Model Functionality")
    err = loaded.learn(np.array([[W['<start>'], W['fire']]]), np.array([[W['hot']]]))
    print(f"    Learn after load: error={err:.4f}")
    out = loaded.forward(np.array([[W['<start>'], W['fire']]]))
    print(f"    Forward after load: logits shape={out.data.shape}")
    print(f"    Bindings survived: {len(loaded.binding_map)}")


def phase_8_summary(model, errors_phase1, errors_phase2):
    """Final summary."""
    header("Architecture Summary")

    print(f"  Model: RLM(vocab={VOCAB_SIZE}, concepts={len(model.graph.nodes)})")
    print(f"  Steps: {model._step_counter}")
    print(f"  Edges learned: {model._edges_learned}")
    print(f"  Sleep cycles: {model.sleep_cycles_completed}")
    print(f"  Conceptual accuracy: {model.conceptual_accuracy:.3f}")
    print(f"  Total free energy: {model.total_free_energy:.3f}")

    section("Graph Topology")
    excitatory = sum(1 for e in model.graph.edges.values() if e.edge_type == 'excitatory')
    inhibitory = sum(1 for e in model.graph.edges.values() if e.edge_type == 'inhibitory')
    print(f"  Nodes: {len(model.graph.nodes)}")
    print(f"  Excitatory edges: {excitatory}")
    print(f"  Inhibitory edges: {inhibitory}")
    print(f"  Total edges: {len(model.graph.edges)}")

    section("Binding Map")
    print(f"  Total bindings: {len(model.binding_map)}")
    ambiguous = sum(1 for t in range(VOCAB_SIZE) if model.binding_map.is_ambiguous(t))
    print(f"  Ambiguous tokens: {ambiguous}")

    section("Learning Rule")
    print(f"  Predictive coding: YES (no backprop)")
    print(f"  Settle steps: {model.settle_steps}")
    print(f"  Settle lr: {model.settle_lr}")
    print(f"  context_scale: {model.context_scale}")
    print(f"  Stabilizers: residual norm, noise injection (sigma={model.noise_sigma}), anti-collapse")

    section("Cognitive Loop")
    print(f"  Sense: apply_prediction_error -> contradiction_hotspots")
    print(f"  Accumulate: free energy on nodes, tracking")
    print(f"  Resolve: form_inhibitory_edges, should_split, homeostatic_downscale")
    print(f"  Suppress: inhibitory edges in spread_activation, binding decay/prune")

    section("Contradiction Resolution")
    hotspots = len(model.graph.contradiction_hotspots)
    print(f"  Active hotspots: {hotspots}")
    if inhibitory > 0:
        print(f"  Inhibitory edges formed: {inhibitory}")
        print(f"  Status: ACTIVE - system suppresses competing concepts")
    else:
        print(f"  Status: DORMANT - no contradictions triggered resolution")

    print(f"\n{'='*60}")
    print(f"  RAVANA Full Architecture Test Complete")
    print(f"{'='*60}")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    header("RAVANA - Full Architecture Test")
    print(f"  Date: {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Vocab: {VOCAB_SIZE} tokens")
    print(f"  Architecture: Predictive coding + contradiction resolution + binding map")

    np.random.seed(42)

    model = RLM(
        vocab_size=VOCAB_SIZE,
        embed_dim=16,
        concept_dim=16,
        n_concepts=30,
        n_hidden=32,
        n_layers=2,
        sleep_interval=999,  # manual sleep control
    )

    # Run all phases
    errors_p1 = phase_1_learning(model)
    errors_p2 = phase_2_contradiction(model)
    phase_3_sleep(model)
    phase_4_disambiguation(model)
    phase_5_dual_path(model)
    phase_6_free_energy(model)
    phase_7_persistence(model)
    phase_8_summary(model, errors_p1, errors_p2)
