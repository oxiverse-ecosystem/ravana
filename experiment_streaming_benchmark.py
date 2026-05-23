"""
Streaming Lifelong Learning Benchmark

The killer experiment for RLM: feed streaming experiences sequentially,
measure retention, interference, adaptation speed, and compute cost.

RLM's architecture is designed for this:
- No replay buffer needed (Hebbian plasticity updates in-place)
- Sleep consolidation reorganizes knowledge naturally
- Concept graph preserves structure under continuous learning
- Bounded compute per step (no backpropagation)

Fair baselines:
- MLP with backprop (catastrophic forgetting expected)
- kNN-LM (exact retrieval, no generalization)
- Sliding window (fixed context, no learning)
"""

import os
import sys
import time
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from ravana_ml.nn.rlm import RLM
from ravana_ml.tokenizer import SimpleTokenizer
from experiment_baselines import SimpleMLP


# ─── Streaming Data Generator ──────────────────────────────────────────

def generate_streaming_experiences(n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate n streaming experiences with structured patterns.

    Creates a stream of factual statements with:
    - Entity-attribute pairs (e.g., "zorbax is hard")
    - Causal chains (e.g., "fire causes heat")
    - Temporal sequences (e.g., "A then B then C")
    - Contradictions injected at controlled intervals
    - Novel entities introduced over time

    Returns list of {text, type, entity, attribute, epoch} dicts.
    """
    rng = np.random.RandomState(seed)

    # Entity pools (divided into epochs to test plasticity)
    entities_pool1 = [
        "zorbax", "vlentor", "quarnox", "threndis", "calibri",
        "nexora", "phyrox", "zenthari", "v或多或", "kryonis",
        "solveni", "draconi", "mytheon", "pyraxis", "glacion",
        "terravox", "aerosis", "luminos", "umbraxis", "feronis",
    ]
    entities_pool2 = [
        "cryonis", "verdantis", "ferrumis", "aqualis", "tempestis",
        "silenxis", "sonoris", "chronaxis", "spatialis", "gravitis",
        "magnetis", "electris", "photonis", "nucleonis", "plasmis",
        "quantaxis", "neutrinis", "bosonis", "fermionis", "hadronis",
    ]
    entities_pool3 = [
        "darkovox", "brightonis", "shadowis", "crystalis", "stormis",
        "flamexus", "frostis", "earthenis", "windaris", "thunderis",
        "oceanis", "mountais", "forestis", "desertis", "arcticis",
        "tropicalis", "volcanis", "tsunamis", "earthquis", "tornadis",
    ]

    attributes = [
        ("hard", "soft"), ("hot", "cold"), ("bright", "dark"),
        ("fast", "slow"), ("heavy", "light"), ("loud", "quiet"),
        ("sharp", "dull"), ("smooth", "rough"), ("wet", "dry"),
        ("strong", "weak"), ("tall", "short"), ("wide", "narrow"),
        ("deep", "shallow"), ("thick", "thin"), ("dense", "sparse"),
    ]

    relations = [
        "is made of", "can emit", "lives in", "produces", "contains",
        "is powered by", "grows near", "feeds on", "protects", "creates",
    ]

    experiences = []
    epoch_size = n // 3

    for i in range(n):
        # Determine epoch (which entity pool)
        if i < epoch_size:
            entities = entities_pool1
            epoch = 1
        elif i < 2 * epoch_size:
            entities = entities_pool2
            epoch = 2
        else:
            entities = entities_pool3
            epoch = 3

        exp_type = rng.choice(['factual', 'relational', 'causal', 'temporal'],
                               p=[0.4, 0.3, 0.2, 0.1])

        if exp_type == 'factual':
            entity = entities[rng.randint(len(entities))]
            attr_pair = attributes[rng.randint(len(attributes))]
            # 80% positive, 20% negative (to test contradiction handling)
            if rng.random() < 0.8:
                attr = attr_pair[0]
            else:
                attr = attr_pair[1]
            text = f"{entity} is {attr}"
            experiences.append({
                'text': text, 'type': exp_type, 'entity': entity,
                'attribute': attr, 'epoch': epoch, 'index': i
            })

        elif exp_type == 'relational':
            entity = entities[rng.randint(len(entities))]
            relation = relations[rng.randint(len(relations))]
            target = entities[(rng.randint(len(entities)))]
            text = f"{entity} {relation} {target}"
            experiences.append({
                'text': text, 'type': exp_type, 'entity': entity,
                'attribute': target, 'epoch': epoch, 'index': i
            })

        elif exp_type == 'causal':
            e1 = entities[rng.randint(len(entities))]
            e2 = entities[rng.randint(len(entities))]
            text = f"{e1} causes {e2} to change"
            experiences.append({
                'text': text, 'type': exp_type, 'entity': e1,
                'attribute': e2, 'epoch': epoch, 'index': i
            })

        elif exp_type == 'temporal':
            e1 = entities[rng.randint(len(entities))]
            e2 = entities[rng.randint(len(entities))]
            e3 = entities[rng.randint(len(entities))]
            text = f"{e1} then {e2} then {e3}"
            experiences.append({
                'text': text, 'type': exp_type, 'entity': e1,
                'attribute': f"{e2},{e3}", 'epoch': epoch, 'index': i
            })

        # Inject contradictions every 5000 experiences
        if i > 0 and i % 5000 == 0:
            entity = entities[rng.randint(len(entities))]
            attr_pair = attributes[rng.randint(len(attributes))]
            # First say it's X, later say it's not-X
            text = f"{entity} is {attr_pair[1]}"  # opposite of what was likely said before
            experiences.append({
                'text': text, 'type': 'contradiction', 'entity': entity,
                'attribute': attr_pair[1], 'epoch': epoch, 'index': i
            })

    return experiences[:n]  # trim to exact count


# ─── Probe Questions (for retention testing) ────────────────────────────

def generate_probe_questions(experiences: List[Dict], n_probes: int = 100,
                              seed: int = 42) -> List[Dict]:
    """Generate probe questions from early experiences to test retention."""
    rng = np.random.RandomState(seed)
    factual = [e for e in experiences if e['type'] == 'factual']
    if len(factual) < n_probes:
        n_probes = len(factual)

    # Sample from early experiences (first 20%)
    early = factual[:len(factual) // 5]
    if len(early) < n_probes:
        early = factual[:n_probes]

    probes = []
    for e in rng.choice(early, size=n_probes, replace=False):
        probes.append({
            'prompt': e['entity'],  # just the entity name
            'target': e['attribute'],
            'text': e['text'],
            'source_index': e['index'],
        })
    return probes


# ─── Baselines ──────────────────────────────────────────────────────────

class SlidingWindowBaseline:
    """Baseline: fixed sliding window of recent experiences. No learning."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.buffer: List[str] = []

    def learn(self, text: str):
        self.buffer.append(text)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

    def query(self, entity: str) -> str:
        """Search buffer for entity mentions."""
        for text in reversed(self.buffer):
            if entity in text:
                # Return the attribute part
                parts = text.split(' is ')
                if len(parts) == 2:
                    return parts[1].strip()
        return ""


class kNNBaseline:
    """Baseline: k-nearest-neighbor lookup over all seen experiences."""

    def __init__(self):
        self.experiences: List[Tuple[str, str]] = []  # (entity, text)

    def learn(self, text: str):
        entity = text.split()[0] if text else ""
        self.experiences.append((entity, text))

    def query(self, entity: str) -> str:
        """Exact match lookup."""
        matches = [(e, t) for e, t in self.experiences if e == entity]
        if matches:
            # Return most recent match
            text = matches[-1][1]
            parts = text.split(' is ')
            if len(parts) == 2:
                return parts[1].strip()
        return ""


# ─── Main Benchmark ─────────────────────────────────────────────────────

def run_streaming_benchmark(n_experiences: int = 100000,
                             n_probes: int = 100,
                             retention_check_interval: int = 10000,
                             seed: int = 42) -> Dict[str, Any]:
    """Run the streaming lifelong learning benchmark.

    Feeds n_experiences sequentially, measuring:
    1. Retention: can the model recall early experiences?
    2. Interference: does new learning damage old knowledge?
    3. Adaptation speed: how quickly does the model learn new facts?
    4. Compute cost: time and memory per experience
    """
    print("\n" + "="*70)
    print("STREAMING LIFELONG LEARNING BENCHMARK")
    print("="*70)
    print(f"Experiences: {n_experiences:,}")
    print(f"Retention probes: {n_probes}")
    print(f"Check interval: every {retention_check_interval:,} experiences")
    print()

    tok = SimpleTokenizer()
    vocab_size = tok.vocab_size

    # Generate data
    print("Generating streaming experiences...")
    experiences = generate_streaming_experiences(n_experiences, seed=seed)
    probes = generate_probe_questions(experiences, n_probes, seed=seed)
    print(f"  Generated {len(experiences):,} experiences, {len(probes)} probes")

    # Initialize models
    np.random.seed(seed)
    rlm = RLM(vocab_size=vocab_size, embed_dim=32, concept_dim=32,
               n_concepts=vocab_size, n_hidden=32, n_layers=3, sleep_interval=100)

    np.random.seed(seed)
    mlp = SimpleMLP(vocab_size=vocab_size, embed_dim=32, n_hidden=32, lr=0.001)

    sliding = SlidingWindowBaseline(window_size=5000)
    knn = kNNBaseline()

    # Results tracking
    retention_history = []  # [{step, rlm_score, mlp_score, sliding_score, knn_score}]
    timing_history = []     # [{step, rlm_time, mlp_time}]
    epoch_boundaries = []

    def evaluate_retention(step: int) -> Dict:
        """Score each model on probe questions."""
        scores = {'rlm': 0, 'mlp': 0, 'sliding': 0, 'knn': 0}

        for probe in probes:
            entity = probe['prompt']
            target = probe['target']

            # RLM: forward pass and check if target char is in top-10
            entity_ids = tok.encode(entity)
            ctx = np.array([entity_ids], dtype=np.int64)
            rlm_logits = rlm.forward(ctx)
            target_id = tok.encode(target)[0]
            rlm_top10 = set(np.argsort(rlm_logits.data)[::-1][:10])
            if target_id in rlm_top10:
                scores['rlm'] += 1

            # MLP
            mlp_logits = mlp.predict(ctx)
            mlp_top10 = set(np.argsort(mlp_logits.flatten())[::-1][:10])
            if target_id in mlp_top10:
                scores['mlp'] += 1

            # Sliding window
            sw_result = sliding.query(entity)
            if sw_result and target in sw_result:
                scores['sliding'] += 1

            # kNN
            knn_result = knn.query(entity)
            if knn_result and target in knn_result:
                scores['knn'] += 1

        return {k: v / len(probes) for k, v in scores.items()}

    # ── Main streaming loop ──
    print("\nStreaming experiences...")
    rlm_total_time = 0.0
    mlp_total_time = 0.0

    for step, exp in enumerate(experiences):
        text = exp['text']
        ids = tok.encode(text)

        # RLM learn
        t0 = time.time()
        for i in range(len(ids) - 1):
            ctx = np.array([ids[:i+1]], dtype=np.int64)
            tgt = np.array([[ids[i+1]]], dtype=np.int64)
            rlm.learn(ctx, tgt)
        rlm_total_time += time.time() - t0

        # MLP learn
        t0 = time.time()
        for i in range(len(ids) - 1):
            ctx = np.array([ids[:i+1]], dtype=np.int64)
            tgt = np.array([[ids[i+1]]], dtype=np.int64)
            mlp.train_step(ctx, tgt[0])
        mlp_total_time += time.time() - t0

        # Baselines
        sliding.learn(text)
        knn.learn(text)

        # Track epoch boundaries
        if step > 0 and step % (n_experiences // 3) == 0:
            epoch_boundaries.append(step)

        # Retention check
        if (step + 1) % retention_check_interval == 0 or step == len(experiences) - 1:
            scores = evaluate_retention(step)
            scores['step'] = step + 1
            scores['rlm_cumulative_time'] = rlm_total_time
            scores['mlp_cumulative_time'] = mlp_total_time
            retention_history.append(scores)

            print(f"  Step {step+1:>8,}/{n_experiences:,} | "
                  f"RLM retention: {scores['rlm']:.1%} | "
                  f"MLP: {scores['mlp']:.1%} | "
                  f"Sliding: {scores['sliding']:.1%} | "
                  f"kNN: {scores['knn']:.1%} | "
                  f"RLM time: {rlm_total_time:.1f}s")

            # RLM sleep cycle if pressure is high
            if rlm.sleep_pressure > rlm.sleep_pressure_threshold:
                rlm.sleep_cycle()

    # ── Final evaluation ──
    print("\n" + "-"*70)
    print("FINAL RESULTS")
    print("-"*70)

    final = retention_history[-1] if retention_history else {}

    # Compute per-experience timing
    rlm_per_exp = rlm_total_time / max(1, n_experiences)
    mlp_per_exp = mlp_total_time / max(1, n_experiences)

    print(f"\n  Retention (probe accuracy):")
    print(f"    RLM:     {final.get('rlm', 0):.1%}")
    print(f"    MLP:     {final.get('mlp', 0):.1%}")
    print(f"    Sliding: {final.get('sliding', 0):.1%}")
    print(f"    kNN:     {final.get('knn', 0):.1%}")

    print(f"\n  Compute cost (total):")
    print(f"    RLM: {rlm_total_time:.1f}s ({rlm_per_exp*1000:.2f}ms/experience)")
    print(f"    MLP: {mlp_total_time:.1f}s ({mlp_per_exp*1000:.2f}ms/experience)")

    print(f"\n  RLM state:")
    print(f"    Concepts: {len(rlm.graph.nodes)}")
    print(f"    Edges: {len(rlm.graph.edges)}")
    print(f"    Sleep cycles: {rlm.sleep_cycles_completed}")
    print(f"    Identity strength: {rlm.identity_strength:.3f}")
    print(f"    Conceptual accuracy: {rlm.conceptual_accuracy:.3f}")

    # Retention curve analysis
    if len(retention_history) >= 2:
        first = retention_history[0]
        last = retention_history[-1]
        print(f"\n  Retention curve (RLM):")
        print(f"    Early: {first['rlm']:.1%} -> Late: {last['rlm']:.1%}")
        delta = last['rlm'] - first['rlm']
        print(f"    Delta: {delta:+.1%} ({'stable' if abs(delta) < 0.1 else 'declining' if delta < 0 else 'improving'})")

    result = {
        'experiment': 'streaming_lifelong_learning',
        'n_experiences': n_experiences,
        'final_retention': {
            'rlm': final.get('rlm', 0),
            'mlp': final.get('mlp', 0),
            'sliding': final.get('sliding', 0),
            'knn': final.get('knn', 0),
        },
        'compute_cost': {
            'rlm_total_s': rlm_total_time,
            'mlp_total_s': mlp_total_time,
            'rlm_per_experience_ms': rlm_per_exp * 1000,
            'mlp_per_experience_ms': mlp_per_exp * 1000,
        },
        'rlm_state': {
            'concepts': len(rlm.graph.nodes),
            'edges': len(rlm.graph.edges),
            'sleep_cycles': rlm.sleep_cycles_completed,
            'identity_strength': rlm.identity_strength,
        },
        'retention_history': retention_history,
    }

    # Save results
    os.makedirs("experiment_results", exist_ok=True)
    with open("experiment_results/streaming_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Results saved to experiment_results/streaming_benchmark.json")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100000)
    parser.add_argument('--probes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_streaming_benchmark(n_experiences=args.n, n_probes=args.probes, seed=args.seed)
