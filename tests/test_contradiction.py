"""
RAVANA — Contradictory Concepts Experiment

The most important experiment for RLM: inject contradictions and observe
whether the system bifurcates, oscillates, collapses, or stabilizes ambiguity.

NOT perplexity. NOT accuracy. This tells you whether RLM has real cognitive
dynamics or is just a lookup table with extra steps.

Three conditions:
  1. Normal: consistent associations (fire→hot, bird→fly)
  2. Contradictory: reversed associations (fire→cold, bird→swim)
  3. Mixed: both normal AND contradictory (fire→hot AND fire→cold)

Measures:
  - Pressure trajectories (Σ|e_i|² = free energy)
  - Settle convergence (does prediction error decrease over settle steps?)
  - Per-layer error norms (which layers carry the contradiction)
  - Activation patterns (which concepts light up for ambiguous inputs)
  - Edge evolution (do inhibitory edges form between contradictions?)

Usage:
    python test_contradiction.py
"""

import sys
import os
import numpy as np
from collections import defaultdict

# Ensure imports work regardless of how script is invoked
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'ravana-v2'))

from ravana_ml.nn.rlm import RLM
from ravana_ml.nn import functional as F
from ravana_ml.tensor import StateTensor


# ──────────────────────────────────────────────────────────────
# Vocabulary & Associations
# ──────────────────────────────────────────────────────────────

VOCAB = [
    '<start>',  # 0 — sequence start token
    'fire',     # 1
    'hot',      # 2
    'cold',     # 3
    'bird',     # 4
    'fly',      # 5
    'swim',     # 6
    'apple',    # 7
    'fruit',    # 8
    'tech',     # 9
    'sun',      # 10
    'bright',   # 11
    'dark',     # 12
    'water',    # 13
    'liquid',   # 14
]

VOCAB_SIZE = len(VOCAB)
WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}

def w(word):
    """Word to token ID."""
    return WORD_TO_ID[word]


# Training format: (context_word, trigger_word, target_word)
# Input sequence: [context, trigger] → predict target
# This ensures T=2 so the settle loop runs.

# Normal: consistent associations
NORMAL_TRIPLES = [
    ('<start>', 'fire',  'hot'),
    ('<start>', 'bird',  'fly'),
    ('<start>', 'sun',   'bright'),
    ('<start>', 'water', 'liquid'),
]

# Contradictory: reversed associations for same triggers
CONTRADICTORY_TRIPLES = [
    ('<start>', 'fire',  'cold'),
    ('<start>', 'bird',  'swim'),
    ('<start>', 'sun',   'dark'),
]

# Ambiguous: same trigger, different targets
AMBIGUOUS_TRIPLES = [
    ('<start>', 'apple', 'fruit'),
    ('<start>', 'apple', 'tech'),
]

# Context-dependent: different context changes the target
CONTEXT_TRIPLES = [
    ('fire',  'apple', 'tech'),    # fire context + apple → tech (company)
    ('fruit', 'apple', 'fruit'),   # fruit context + apple → fruit (food)
    ('bird',  'fire',  'hot'),     # bird context + fire → hot
    ('cold',  'fire',  'cold'),    # cold context + fire → cold (reinforced)
]


# ──────────────────────────────────────────────────────────────
# Instrumented RLM
# ──────────────────────────────────────────────────────────────

class InstrumentedRLM(RLM):
    """RLM that records settle loop internals for analysis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.settle_log = []
        self.edge_history = []  # track edge weight changes

    def learn_and_record(self, token_ids, next_token_ids):
        """Call parent learn() to get all contradiction tracking, then record settle dynamics."""
        # Snapshot settle log length before
        n_before = len(self.settle_log)

        # Call parent learn() — this has all the contradiction tracking, hotspot detection, etc.
        err = super().learn(token_ids, next_token_ids)

        # If settle ran (T > 1), the last entry in settle_log was added by _settle_predictive
        # But parent learn() doesn't add to settle_log, so we need to instrument it
        # by checking if a settle happened and recording the dynamics ourselves

        if token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, :]
        next_id = int(next_token_ids[0]) if next_token_ids.ndim == 1 else int(next_token_ids[0, 0])
        last_input_id = int(token_ids[0, -1])
        T = token_ids.shape[1]

        if T > 1 and self._last_hidden_state is not None:
            # Re-run settle to capture dynamics (cheap — just state copies)
            target_onehot = np.zeros(self.vocab_size, dtype=np.float32)
            target_onehot[next_id] = 1.0

            h_states = [self._last_hidden_state]
            h_temp = self._last_hidden_state
            for layer in self.hidden_layers:
                h_temp = layer(StateTensor(h_temp[np.newaxis, :])).data[0]
                h_temp = np.tanh(h_temp)
                h_states.append(h_temp)

            _, local_errors = self._settle_predictive(h_states, target_onehot)

            ctx_err = local_errors[-1]
            self.settle_log.append({
                'input': last_input_id,
                'target': next_id,
                'input_word': VOCAB[last_input_id] if last_input_id < len(VOCAB) else '?',
                'target_word': VOCAB[next_id] if next_id < len(VOCAB) else '?',
                'final_error_norms': [float(np.linalg.norm(e)) for e in local_errors],
                'total_energy': float(sum(np.linalg.norm(e)**2 for e in local_errors)),
                'ctx_error_norm': float(np.linalg.norm(ctx_err)),
                'conceptual_error': err,
                'step': self._step_counter,
            })

        return err


# ──────────────────────────────────────────────────────────────
# Experiment Runner
# ──────────────────────────────────────────────────────────────

def run_condition(condition_name, train_triples, n_epochs=30, seed=42):
    """Run one experimental condition and return model + metrics."""
    np.random.seed(seed)

    model = InstrumentedRLM(
        vocab_size=VOCAB_SIZE,
        embed_dim=16,
        concept_dim=16,
        n_concepts=30,
        n_hidden=32,
        n_layers=2,
        sleep_interval=20,  # sleep every 20 steps to trigger contradiction resolution
    )

    print(f"\n{'='*60}")
    print(f"  Condition: {condition_name}")
    print(f"  Training triples: {len(train_triples)}")
    print(f"  Epochs: {n_epochs}")
    print(f"{'='*60}")

    for epoch in range(n_epochs):
        np.random.shuffle(train_triples)
        epoch_errors = []
        for ctx, trigger, target in train_triples:
            err = model.learn_and_record(
                np.array([[w(ctx), w(trigger)]]),
                np.array([[w(target)]])
            )
            epoch_errors.append(err)

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            avg_err = np.mean(epoch_errors)
            n_settle = len(model.settle_log)
            print(f"  Epoch {epoch:3d}: avg_error={avg_err:.4f}, "
                  f"edges={model._edges_learned}, "
                  f"settle_records={n_settle}")

    # ──────────────────────────────────────────────────────────
    # Analysis
    # ──────────────────────────────────────────────────────────

    log = model.settle_log
    metrics = {}

    if not log:
        print("  WARNING: No settle records! Settle loop may not be running.")
        metrics['energy_trajectory'] = []
        metrics['energy_first10'] = 0.0
        metrics['energy_last10'] = 0.0
        metrics['word_errors'] = {}
        metrics['input_variance'] = {}
        metrics['inhibitory_edges'] = 0
        metrics['total_edges'] = len(model.graph.edges)
        metrics['max_drift'] = 0.0
        return model, metrics

    # 1. Pressure trajectory
    energies = [r['total_energy'] for r in log]
    metrics['energy_trajectory'] = energies
    n10 = min(10, len(energies))
    metrics['energy_first10'] = float(np.mean(energies[:n10]))
    metrics['energy_last10'] = float(np.mean(energies[-n10:]))

    # 2. Per-word error profiles
    word_errors = defaultdict(list)
    for r in log:
        word_errors[r['input_word']].append(r['ctx_error_norm'])
    metrics['word_errors'] = {wd: float(np.mean(errs)) for wd, errs in word_errors.items()}

    # 3. Contradiction detection: same input, high-variance errors
    input_variance = {}
    for word, errs in word_errors.items():
        if len(errs) > 1:
            input_variance[word] = float(np.var(errs))
    metrics['input_variance'] = input_variance

    # 4. Graph structure
    inhibitory_count = sum(
        1 for e in model.graph.edges.values()
        if e.edge_type == 'inhibitory'
    )
    metrics['inhibitory_edges'] = inhibitory_count
    metrics['total_edges'] = len(model.graph.edges)

    # 5. Edge weight analysis for contradictions
    # Find edges that go from the same source to different targets
    source_targets = defaultdict(list)
    for (s, t), e in model.graph.edges.items():
        source_targets[s].append((t, e.weight, e.edge_type))
    competing_edges = {
        s: targets for s, targets in source_targets.items()
        if len(targets) > 1
    }
    metrics['competing_edges'] = len(competing_edges)

    # 6. Concept drift
    drifts = []
    for nid, node in model.graph.nodes.items():
        if hasattr(node, 'drift_magnitude'):
            drifts.append(node.drift_magnitude)
    metrics['max_drift'] = float(max(drifts)) if drifts else 0.0

    return model, metrics


def probe_ambiguity(model, probe_words):
    """Probe the model with inputs and see what activates."""
    print(f"\n  --- Ambiguity Probes ---")
    for word in probe_words:
        if w(word) >= model.vocab_size:
            continue

        model.graph.reset_activation()
        tokens = np.array([[w('<start>'), w(word)]])
        out = model.forward(tokens)

        active = [(nid, node.activation)
                  for nid, node in model.graph.nodes.items()
                  if node.activation > 0.1]
        active.sort(key=lambda x: -x[1])

        logits = out.data
        # Clip extreme values for display
        logits_clipped = np.clip(logits, -100, 100)
        top3 = np.argsort(logits_clipped)[-3:][::-1]
        predictions = [(VOCAB[tid], float(logits[tid])) for tid in top3
                       if tid < len(VOCAB)]

        print(f"\n  '{word}':")
        print(f"    Active concepts: {len(active)} "
              f"({'CONFLICT' if len(active) > 2 else 'focused'})")
        for nid, act in active[:3]:
            print(f"      concept {nid}: activation={act:.3f}")
        print(f"    Top predictions: {predictions}")


def analyze_edge_competition(model, label):
    """Analyze edges that compete (same source, different targets)."""
    print(f"\n  --- Edge Competition ({label}) ---")
    source_targets = defaultdict(list)
    for (s, t), e in model.graph.edges.items():
        source_targets[s].append((t, e.weight, e.edge_type, e.prediction_count))

    for source, targets in source_targets.items():
        if len(targets) > 1:
            print(f"  Concept {source} -> ", end="")
            parts = []
            for t, w, etype, pc in targets:
                parts.append(f"{t} (w={w:.3f}, {etype}, n={pc})")
            print(" | ".join(parts))


# ──────────────────────────────────────────────────────────────
# Main Experiment
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  RAVANA - Contradictory Concepts Experiment")
    print("  The test that tells you more than 100 benchmarks")
    print("=" * 60)

    results = {}

    # Condition 1: Normal (baseline)
    model_normal, metrics_normal = run_condition(
        "Normal (consistent associations)",
        NORMAL_TRIPLES,
        n_epochs=30,
        seed=42,
    )
    results['normal'] = metrics_normal

    # Condition 2: Contradictory (reversed associations)
    model_contra, metrics_contra = run_condition(
        "Contradictory (reversed associations)",
        CONTRADICTORY_TRIPLES,
        n_epochs=30,
        seed=42,
    )
    results['contradictory'] = metrics_contra

    # Condition 3: Mixed (normal + contradictory + ambiguous + context)
    mixed_triples = NORMAL_TRIPLES + CONTRADICTORY_TRIPLES + AMBIGUOUS_TRIPLES + CONTEXT_TRIPLES
    model_mixed, metrics_mixed = run_condition(
        "Mixed (all types)",
        mixed_triples,
        n_epochs=30,
        seed=42,
    )
    results['mixed'] = metrics_mixed

    # ──────────────────────────────────────────────────────────
    # Comparative Analysis
    # ──────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("  COMPARATIVE ANALYSIS")
    print("=" * 60)

    print("\n  Energy (free energy) trajectory:")
    for cond in ['normal', 'contradictory', 'mixed']:
        m = results[cond]
        if m['energy_trajectory']:
            print(f"    {cond:15s}: first10={m['energy_first10']:.4f}, "
                  f"last10={m['energy_last10']:.4f}, "
                  f"delta={m['energy_first10'] - m['energy_last10']:.4f}")
        else:
            print(f"    {cond:15s}: NO DATA (settle loop didn't run)")

    print("\n  Per-word error profiles:")
    all_words = set()
    for m in results.values():
        all_words.update(m.get('word_errors', {}).keys())
    for word in sorted(all_words):
        vals = [results[c].get('word_errors', {}).get(word, 0)
                for c in ['normal', 'contradictory', 'mixed']]
        print(f"    {word:10s}: normal={vals[0]:.4f}, contra={vals[1]:.4f}, mixed={vals[2]:.4f}")

    print("\n  Input variance (high = contradictory signal):")
    for cond in ['normal', 'contradictory', 'mixed']:
        variance = results[cond].get('input_variance', {})
        if variance:
            max_var_word = max(variance, key=variance.get)
            print(f"    {cond:15s}: max_variance_word='{max_var_word}' "
                  f"({variance[max_var_word]:.4f})")
        else:
            print(f"    {cond:15s}: no variance data")

    print("\n  Graph structure:")
    for cond in ['normal', 'contradictory', 'mixed']:
        m = results[cond]
        print(f"    {cond:15s}: edges={m['total_edges']}, "
              f"inhibitory={m['inhibitory_edges']}, "
              f"competing={m.get('competing_edges', 0)}, "
              f"max_drift={m['max_drift']:.4f}")

    # Edge competition analysis
    analyze_edge_competition(model_mixed, "Mixed")

    # Probe ambiguity
    probe_ambiguity(model_mixed, ['fire', 'bird', 'apple', 'sun', 'water'])

    # ──────────────────────────────────────────────────────────
    # Verdict
    # ──────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("  VERDICT")
    print("=" * 60)

    has_data = all(results[c].get('energy_trajectory') for c in ['normal', 'mixed'])

    if not has_data:
        print("\n  Settle loop did not produce data.")
        print("  Check that T > 1 in training sequences.")
    else:
        mixed_last = results['mixed']['energy_last10']
        normal_last = results['normal']['energy_last10']

        if mixed_last > normal_last * 1.5:
            print("\n  MIXED model has HIGHER persistent energy than NORMAL.")
            print("  Contradictions create lasting dissonance.")
            print("  GOOD: This is what cognitive architecture should do.")
        elif mixed_last < normal_last * 0.8:
            print("\n  MIXED model has LOWER energy than NORMAL.")
            print("  Contradictions resolved too easily - possible collapse.")
            print("  CONCERN: Check stabilizers.")
        else:
            print("\n  MIXED and NORMAL have similar energy levels.")
            print("  Contradictions processed without significant dissonance.")

    if results['mixed'].get('inhibitory_edges', 0) > 0:
        print(f"\n  {results['mixed']['inhibitory_edges']} inhibitory edges formed.")
        print("  System learning to suppress contradictions structurally.")
    else:
        print("\n  No inhibitory edges formed.")
        print("  May need more training or stronger contradiction signals.")

    if results['mixed'].get('competing_edges', 0) > 0:
        print(f"\n  {results['mixed']['competing_edges']} competing edge groups found.")
        print("  Same source concepts connecting to multiple targets.")

    print("\n" + "=" * 60)
    print("  Experiment complete.")
    print("=" * 60)
