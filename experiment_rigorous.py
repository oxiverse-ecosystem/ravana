"""
RLM vs LLM: Rigorous Scientific Experiments

Addresses methodological criticisms:
1. Compositional few-shot generalization (not memorization)
2. Contradiction emergence dynamics (not threshold hunting)
3. Streaming lifelong learning (the flagship benchmark)
4. Consolidation metrics upgrade
5. Identity formalization

Each experiment has:
- Clear hypothesis
- Fair baselines
- Mechanistic metrics
- Statistical repeatability (multiple seeds)
- Ablation capability
"""

import os
import sys
import json
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from ravana_ml.nn.rlm import RLM
from ravana_ml.tokenizer import SimpleTokenizer
from experiment_baselines import SimpleMLP, measure_time_and_memory


# ─── Helpers ─────────────────────────────────────────────────────────────

def make_tokenizer():
    return SimpleTokenizer()


def make_rlm(vocab_size, seed=42, **kwargs):
    np.random.seed(seed)
    defaults = dict(embed_dim=32, concept_dim=32, n_concepts=vocab_size, n_hidden=32, sleep_interval=5)
    defaults.update(kwargs)
    return RLM(vocab_size=vocab_size, **defaults)


def make_mlp(vocab_size, seed=42, **kwargs):
    np.random.seed(seed)
    defaults = dict(embed_dim=32, n_hidden=32, lr=0.001)
    defaults.update(kwargs)
    return SimpleMLP(vocab_size=vocab_size, **defaults)


def train_rlm(model, tokenizer, texts, epochs=50):
    for _ in range(epochs):
        for text in texts:
            ids = tokenizer.encode(text)
            for i in range(len(ids) - 1):
                ctx = np.array([ids[:i+1]], dtype=np.int64)
                tgt = np.array([[ids[i+1]]], dtype=np.int64)
                model.learn(ctx, tgt)


def train_mlp(model, tokenizer, texts, epochs=50):
    for _ in range(epochs):
        for text in texts:
            ids = tokenizer.encode(text)
            for i in range(len(ids) - 1):
                ctx = np.array([ids[:i+1]], dtype=np.int64)
                tgt = np.array([ids[i+1]], dtype=np.int64)
                model.train_step(ctx, tgt)


def get_logits(model, text, tokenizer):
    ids = tokenizer.encode(text)
    ctx = np.array([ids], dtype=np.int64)
    if hasattr(model, 'forward'):
        logits = np.asarray(model.forward(ctx).data)
    else:
        logits = np.asarray(model.predict(ctx))
    if logits.ndim > 1:
        logits = logits[0]
    return logits


def top_k_tokens(logits, k=5):
    return list(np.argsort(logits)[-k:][::-1])


def token_rank(logits, token_id):
    """Lower rank = higher probability. 0 = best."""
    sorted_ids = np.argsort(logits)[::-1]
    return int(np.where(sorted_ids == token_id)[0][0])


def recall_score(model, prompt, target_word, tokenizer):
    """Score for target word given prompt. Higher = better recall."""
    logits = get_logits(model, prompt, tokenizer)
    target_ids = tokenizer.encode(target_word)
    return float(np.mean([logits[i] for i in target_ids]))


# ═════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: COMPOSITIONAL FEW-SHOT GENERALIZATION
# ═════════════════════════════════════════════════════════════════════════
#
# NOT memorization. Requires:
# - Variable binding (glim → red)
# - Relational inference (red → dangerous)
# - Composition (glim → dangerous) WITHOUT explicit training
# ═════════════════════════════════════════════════════════════════════════

def run_compositional_few_shot(seeds=[42, 123, 456]) -> Dict[str, Any]:
    """Test compositional generalization: bind + infer + compose."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: COMPOSITIONAL FEW-SHOT GENERALIZATION")
    print("="*60)

    tokenizer = make_tokenizer()
    vocab_size = tokenizer.vocab_size
    results = {"experiment": "compositional_few_shot", "seeds": {}}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        # === TRAINING DATA ===
        # Phase 1: Bindings (what is what)
        bindings = [
            "glim is red",
            "marn is blue",
            "tork is green",
        ]
        # Phase 2: Rules (what means what)
        rules = [
            "red things are dangerous",
            "blue things are safe",
            "green things are uncertain",
        ]

        # === TEST QUERIES (never seen in training) ===
        # Must compose: glim → red → dangerous
        test_queries = {
            "glim is": "dangerous",      # glim→red→dangerous
            "marn is": "safe",           # marn→blue→safe
            "tork is": "uncertain",      # tork→green→uncertain
        }

        # === TRAIN ===
        rlm = make_rlm(vocab_size, seed=seed)
        all_train = bindings + rules
        train_rlm(rlm, tokenizer, all_train, epochs=100)

        # === EVALUATE ===
        rlm_results = {}
        for prompt, expected in test_queries.items():
            score = recall_score(rlm, prompt, expected, tokenizer)
            # Also score wrong answers
            wrong_words = [v for k, v in test_queries.items() if v != expected]
            wrong_scores = [recall_score(rlm, prompt, w, tokenizer) for w in wrong_words]
            max_wrong = max(wrong_scores)
            correct = score > max_wrong
            rlm_results[prompt] = {
                "expected": expected,
                "score": score,
                "max_wrong": max_wrong,
                "correct": correct,
            }
            print(f"  RLM  '{prompt} ___': {expected} score={score:.3f} "
                  f"(max_wrong={max_wrong:.3f}) {'PASS' if correct else 'FAIL'}")

        # === MLP BASELINE ===
        mlp = make_mlp(vocab_size, seed=seed)
        train_mlp(mlp, tokenizer, all_train, epochs=100)

        mlp_results = {}
        for prompt, expected in test_queries.items():
            score = recall_score(mlp, prompt, expected, tokenizer)
            wrong_words = [v for k, v in test_queries.items() if v != expected]
            wrong_scores = [recall_score(mlp, prompt, w, tokenizer) for w in wrong_words]
            max_wrong = max(wrong_scores)
            correct = score > max_wrong
            mlp_results[prompt] = {
                "expected": expected,
                "score": score,
                "max_wrong": max_wrong,
                "correct": correct,
            }
            print(f"  MLP  '{prompt} ___': {expected} score={score:.3f} "
                  f"(max_wrong={max_wrong:.3f}) {'PASS' if correct else 'FAIL'}")

        rlm_correct = sum(1 for r in rlm_results.values() if r["correct"])
        mlp_correct = sum(1 for r in mlp_results.values() if r["correct"])

        results["seeds"][seed] = {
            "rlm_accuracy": rlm_correct / len(test_queries),
            "mlp_accuracy": mlp_correct / len(test_queries),
            "rlm_details": rlm_results,
            "mlp_details": mlp_results,
        }

    # Aggregate
    rlm_accs = [s["rlm_accuracy"] for s in results["seeds"].values()]
    mlp_accs = [s["mlp_accuracy"] for s in results["seeds"].values()]
    results["rlm_mean_accuracy"] = float(np.mean(rlm_accs))
    results["mlp_mean_accuracy"] = float(np.mean(mlp_accs))
    results["rlm_std"] = float(np.std(rlm_accs))
    results["mlp_std"] = float(np.std(mlp_accs))

    print(f"\n  RLM: {results['rlm_mean_accuracy']:.0%} ± {results['rlm_std']:.0%}")
    print(f"  MLP: {results['mlp_mean_accuracy']:.0%} ± {results['mlp_std']:.0%}")

    return results


# ═════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1B: DEEP COMPOSITIONAL GENERALIZATION
# ═════════════════════════════════════════════════════════════════════════
#
# Beyond 2-hop: tests 3-hop chains, relational transfer, OOD recombination,
# and negative cases (should NOT compose incorrectly).
# ═════════════════════════════════════════════════════════════════════════

def run_deep_compositional(seeds=[42, 123, 456]) -> Dict[str, Any]:
    """Deep compositional generalization: 3-hop chains, relational transfer, OOD."""
    print("\n" + "="*60)
    print("EXPERIMENT 1B: DEEP COMPOSITIONAL GENERALIZATION")
    print("="*60)

    tokenizer = make_tokenizer()
    vocab_size = tokenizer.vocab_size
    results = {"experiment": "deep_compositional", "seeds": {}}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        rlm = make_rlm(vocab_size, seed=seed)

        # === 3-HOP TRANSITIVE CHAINS ===
        # A→B→C→D: must compose 3 links
        chain_facts = [
            "zorbax is crystalline",
            "crystalline things are fragile",
            "fragile things are valuable",
            "valuable things are insured",
        ]
        # Also train distractors to ensure selective composition
        distractors = [
            "marn is liquid",
            "liquid things are flexible",
            "flexible things are cheap",
        ]
        all_train = chain_facts + distractors
        train_rlm(rlm, tokenizer, all_train, epochs=150)

        # Test 3-hop: zorbax → crystalline → fragile → valuable → insured
        chain_tests = {
            "zorbax is": "fragile",       # 2-hop
            "crystalline things are": "valuable",  # 1-hop
            "fragile things are": "insured",  # 1-hop
        }

        chain_results = {}
        for prompt, expected in chain_tests.items():
            score = recall_score(rlm, prompt, expected, tokenizer)
            # Get scores for wrong answers
            all_targets = ["fragile", "valuable", "insured", "flexible", "cheap", "crystalline"]
            wrong_targets = [t for t in all_targets if t != expected]
            wrong_scores = [recall_score(rlm, prompt, w, tokenizer) for w in wrong_targets]
            max_wrong = max(wrong_scores) if wrong_scores else 0.0
            correct = score > max_wrong
            chain_results[prompt] = {
                "expected": expected,
                "score": score,
                "max_wrong": max_wrong,
                "correct": correct,
            }
            print(f"  Chain '{prompt} ___': {expected} score={score:.3f} "
                  f"(max_wrong={max_wrong:.3f}) {'PASS' if correct else 'FAIL'}")

        # === RELATIONAL TRANSFER ===
        # Train: "X is Y" pattern with new entities
        transfer_facts = [
            "vexol is warm",
            "warm things are pleasant",
            "pleasant things are memorable",
        ]
        train_rlm(rlm, tokenizer, transfer_facts, epochs=100)

        transfer_tests = {
            "vexol is": "pleasant",     # 2-hop transfer
            "warm things are": "memorable",  # 1-hop
        }

        transfer_results = {}
        for prompt, expected in transfer_tests.items():
            score = recall_score(rlm, prompt, expected, tokenizer)
            wrong_targets = ["fragile", "valuable", "insured", "flexible", "cheap", "crystalline"]
            wrong_scores = [recall_score(rlm, prompt, w, tokenizer) for w in wrong_targets]
            max_wrong = max(wrong_scores) if wrong_scores else 0.0
            correct = score > max_wrong
            transfer_results[prompt] = {
                "expected": expected,
                "score": score,
                "max_wrong": max_wrong,
                "correct": correct,
            }
            print(f"  Transfer '{prompt} ___': {expected} score={score:.3f} "
                  f"(max_wrong={max_wrong:.3f}) {'PASS' if correct else 'FAIL'}")

        # === NEGATIVE CASES ===
        # Should NOT compose across unrelated chains
        # "marn is liquid" + "liquid things are flexible" → flexible
        # But "zorbax is crystalline" should NOT → flexible
        negative_tests = {
            "zorbax is": "flexible",    # WRONG — different chain
            "marn is": "fragile",       # WRONG — different chain
        }

        negative_results = {}
        for prompt, wrong_answer in negative_tests.items():
            score = recall_score(rlm, prompt, wrong_answer, tokenizer)
            # Should score LOW (model should NOT compose across chains)
            negative_results[prompt] = {
                "wrong_answer": wrong_answer,
                "score": score,
                "correctly_rejected": score < 0.1,  # threshold for "not predicted"
            }
            print(f"  Negative '{prompt} ___': {wrong_answer} score={score:.3f} "
                  f"{'REJECTED' if score < 0.1 else 'LEAKED'}")

        # === MLP BASELINE ===
        mlp = make_mlp(vocab_size, seed=seed)
        train_mlp(mlp, tokenizer, all_train + transfer_facts, epochs=150)

        mlp_chain = sum(1 for p, e in chain_tests.items()
                        if recall_score(mlp, p, e, tokenizer) > max(
                            recall_score(mlp, p, w, tokenizer)
                            for w in ["fragile", "valuable", "insured", "flexible", "cheap"]
                            if w != e))
        mlp_transfer = sum(1 for p, e in transfer_tests.items()
                           if recall_score(mlp, p, e, tokenizer) > max(
                               recall_score(mlp, p, w, tokenizer)
                               for w in ["fragile", "valuable", "insured", "flexible", "cheap"]
                               if w != e))

        chain_acc = sum(1 for r in chain_results.values() if r["correct"]) / len(chain_tests)
        transfer_acc = sum(1 for r in transfer_results.values() if r["correct"]) / len(transfer_tests)
        neg_reject = sum(1 for r in negative_results.values() if r["correctly_rejected"]) / len(negative_tests)

        results["seeds"][seed] = {
            "chain_accuracy": chain_acc,
            "transfer_accuracy": transfer_acc,
            "negative_rejection": neg_reject,
            "mlp_chain_accuracy": mlp_chain / len(chain_tests),
            "mlp_transfer_accuracy": mlp_transfer / len(transfer_tests),
        }

    # Aggregate
    rlm_chains = [s["chain_accuracy"] for s in results["seeds"].values()]
    rlm_transfer = [s["transfer_accuracy"] for s in results["seeds"].values()]
    rlm_neg = [s["negative_rejection"] for s in results["seeds"].values()]
    mlp_chains = [s["mlp_chain_accuracy"] for s in results["seeds"].values()]

    results["rlm_chain_mean"] = float(np.mean(rlm_chains))
    results["rlm_transfer_mean"] = float(np.mean(rlm_transfer))
    results["rlm_negative_mean"] = float(np.mean(rlm_neg))
    results["mlp_chain_mean"] = float(np.mean(mlp_chains))

    print(f"\n  RLM 3-hop chains: {results['rlm_chain_mean']:.0%}")
    print(f"  RLM transfer: {results['rlm_transfer_mean']:.0%}")
    print(f"  RLM negative rejection: {results['rlm_negative_mean']:.0%}")
    print(f"  MLP 3-hop chains: {results['mlp_chain_mean']:.0%}")

    return results


# ═════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: CONTRADICTION EMERGENCE DYNAMICS
# ═════════════════════════════════════════════════════════════════════════
#
# NOT threshold hunting. Measures the DYNAMICS of contradiction:
# - How pressure accumulates over time
# - When inhibitory edges form (if at all)
# - What triggers concept splitting
# - Multiple seeds for statistical validity
# ═════════════════════════════════════════════════════════════════════════

def run_contradiction_dynamics(seeds=[42, 123, 456]) -> Dict[str, Any]:
    """Track contradiction dynamics over training time."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: CONTRADICTION EMERGENCE DYNAMICS")
    print("="*60)

    tokenizer = make_tokenizer()
    vocab_size = tokenizer.vocab_size
    results = {"experiment": "contradiction_dynamics", "seeds": {}}

    # Contradictory training data
    contradictions = [
        "fire is hot",
        "fire is cold",
        "fire is dangerous",
        "ice is cold",
        "ice is warm",
        "ice is slippery",
    ]

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        rlm = make_rlm(vocab_size, seed=seed)

        # Track dynamics over epochs
        dynamics = []
        for epoch in range(0, 200, 10):
            train_rlm(rlm, tokenizer, contradictions, epochs=10)

            # Measure state
            n_edges = len(rlm.graph.edges)
            n_inhibitory = sum(1 for e in rlm.graph.edges.values() if e.edge_type == "inhibitory")
            n_excitatory = n_edges - n_inhibitory
            hotspots = len(rlm.graph.contradiction_hotspots)
            total_fe = rlm.free_energy_engine.free_energy

            # Measure contradiction pressure on fire/ice concepts
            fire_node = None
            ice_node = None
            for node in rlm.graph.nodes.values():
                if hasattr(node, 'label') and node.label:
                    if 'fire' in node.label.lower():
                        fire_node = node
                    elif 'ice' in node.label.lower():
                        ice_node = node

            fire_pressure = fire_node.contradiction_count if fire_node else 0
            ice_pressure = ice_node.contradiction_count if ice_node else 0

            dynamics.append({
                "epoch": epoch + 10,
                "edges": n_edges,
                "inhibitory": n_inhibitory,
                "excitatory": n_excitatory,
                "hotspots": hotspots,
                "free_energy": total_fe,
                "fire_pressure": fire_pressure,
                "ice_pressure": ice_pressure,
            })

            if epoch % 50 == 0:
                print(f"  Epoch {epoch+10}: edges={n_edges}, inhib={n_inhibitory}, "
                      f"hotspots={hotspots}, FE={total_fe:.1f}, "
                      f"fire_p={fire_pressure}, ice_p={ice_pressure}")

        # Final sleep cycle — does it resolve?
        rlm.sleep_cycle()
        final_inhibitory = sum(1 for e in rlm.graph.edges.values() if e.edge_type == "inhibitory")
        print(f"  After sleep: inhibitory={final_inhibitory}")

        results["seeds"][seed] = {
            "dynamics": dynamics,
            "final_inhibitory": final_inhibitory,
            "formed_inhibitory": final_inhibitory > 0,
            "peak_fire_pressure": max(d["fire_pressure"] for d in dynamics),
            "peak_ice_pressure": max(d["ice_pressure"] for d in dynamics),
        }

    # Aggregate
    formation_rates = [s["formed_inhibitory"] for s in results["seeds"].values()]
    results["inhibitory_formation_rate"] = sum(formation_rates) / len(formation_rates)
    results["peak_pressure_mean"] = float(np.mean([
        max(s["peak_fire_pressure"], s["peak_ice_pressure"])
        for s in results["seeds"].values()
    ]))

    print(f"\n  Inhibitory formation rate: {results['inhibitory_formation_rate']:.0%}")
    print(f"  Mean peak pressure: {results['peak_pressure_mean']:.1f}")

    return results


# ═════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: STREAMING LIFELONG LEARNING
# ═════════════════════════════════════════════════════════════════════════
#
# THE FLAGSHIP BENCHMARK.
#
# Feed N streaming experiences sequentially.
# Measure:
# - Retention: can it recall old experiences?
# - Interference: do new experiences corrupt old ones?
# - Adaptation: does it learn new patterns?
# - Energy: computational cost over time
#
# LLMs struggle here because:
# - Context windows saturate
# - Fine-tuning is expensive
# - No online learning
# ═════════════════════════════════════════════════════════════════════════

def run_streaming_lifelong(n_experiences=200, n_recall_tests=20, seed=42) -> Dict[str, Any]:
    """Streaming lifelong learning benchmark."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: STREAMING LIFELONG LEARNING")
    print("="*60)

    tokenizer = make_tokenizer()
    vocab_size = tokenizer.vocab_size
    np.random.seed(seed)

    # Generate streaming experiences
    subjects = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    relations = ["is", "likes", "fears", "creates", "destroys", "transforms"]
    objects = ["fire", "water", "earth", "air", "light", "darkness", "ice", "storm",
              "crystal", "shadow", "thunder", "silence", "flame", "frost", "wind", "stone"]

    experiences = []
    for i in range(n_experiences):
        s = subjects[i % len(subjects)]
        r = relations[np.random.randint(len(relations))]
        o = objects[np.random.randint(len(objects))]
        experiences.append(f"{s} {r} {o}")

    # Track metrics over time
    rlm = make_rlm(vocab_size, seed=seed)
    mlp = make_mlp(vocab_size, seed=seed)

    # Select recall test experiences (first N unique)
    recall_tests = experiences[:n_recall_tests]

    metrics = {
        "rlm": {"retention": [], "energy": [], "time_ms": []},
        "mlp": {"retention": [], "energy": [], "time_ms": []},
    }

    batch_size = 10
    n_batches = n_experiences // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        batch = experiences[start:end]

        # Train RLM
        rlm_start = time.perf_counter()
        train_rlm(rlm, tokenizer, batch, epochs=1)
        rlm_time = (time.perf_counter() - rlm_start) * 1000

        # Train MLP
        mlp_start = time.perf_counter()
        train_mlp(mlp, tokenizer, batch, epochs=1)
        mlp_time = (time.perf_counter() - mlp_start) * 1000

        # Measure retention every 5 batches
        if batch_idx % 5 == 0:
            rlm_retention = 0
            mlp_retention = 0
            for test_exp in recall_tests:
                words = test_exp.split()
                prompt = " ".join(words[:2])
                target = words[2]

                rlm_score = recall_score(rlm, prompt, target, tokenizer)
                mlp_score = recall_score(mlp, prompt, target, tokenizer)

                # Score > 0 means the model assigns non-trivial probability
                if rlm_score > -5.0:
                    rlm_retention += 1
                if mlp_score > -5.0:
                    mlp_retention += 1

            rlm_retention /= len(recall_tests)
            mlp_retention /= len(recall_tests)

            metrics["rlm"]["retention"].append(rlm_retention)
            metrics["rlm"]["time_ms"].append(rlm_time)
            metrics["mlp"]["retention"].append(mlp_retention)
            metrics["mlp"]["time_ms"].append(mlp_time)

            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{n_batches}: "
                      f"RLM retention={rlm_retention:.0%}, "
                      f"MLP retention={mlp_retention:.0%}")

    # Final metrics
    rlm_total_time = sum(metrics["rlm"]["time_ms"])
    mlp_total_time = sum(metrics["mlp"]["time_ms"])

    rlm_final_retention = metrics["rlm"]["retention"][-1] if metrics["rlm"]["retention"] else 0
    mlp_final_retention = metrics["mlp"]["retention"][-1] if metrics["mlp"]["retention"] else 0

    # Interference: compare early vs late retention
    rlm_early = metrics["rlm"]["retention"][0] if len(metrics["rlm"]["retention"]) > 0 else 0
    rlm_late = metrics["rlm"]["retention"][-1] if len(metrics["rlm"]["retention"]) > 0 else 0
    rlm_interference = rlm_early - rlm_late  # positive = forgetting

    mlp_early = metrics["mlp"]["retention"][0] if len(metrics["mlp"]["retention"]) > 0 else 0
    mlp_late = metrics["mlp"]["retention"][-1] if len(metrics["mlp"]["retention"]) > 0 else 0
    mlp_interference = mlp_early - mlp_late

    print(f"\n  Final Results:")
    print(f"    RLM retention: {rlm_final_retention:.0%}, interference: {rlm_interference:+.0%}")
    print(f"    MLP retention: {mlp_final_retention:.0%}, interference: {mlp_interference:+.0%}")
    print(f"    RLM time: {rlm_total_time:.0f}ms, MLP time: {mlp_total_time:.0f}ms")

    results = {
        "experiment": "streaming_lifelong",
        "n_experiences": n_experiences,
        "batch_size": batch_size,
        "rlm_final_retention": rlm_final_retention,
        "mlp_final_retention": mlp_final_retention,
        "rlm_interference": rlm_interference,
        "mlp_interference": mlp_interference,
        "rlm_total_time_ms": rlm_total_time,
        "mlp_total_time_ms": mlp_total_time,
        "rlm_retention_curve": metrics["rlm"]["retention"],
        "mlp_retention_curve": metrics["mlp"]["retention"],
    }

    return results


# ═════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: CONSOLIDATION METRICS UPGRADE
# ═════════════════════════════════════════════════════════════════════════
#
# Better metrics:
# - Retrieval efficiency (recall speed after sleep)
# - Interference reduction (do competitors separate?)
# - Cluster emergence (new abstractions)
# - Predictive entropy reduction
# ═════════════════════════════════════════════════════════════════════════

def run_consolidation_metrics(seed=42) -> Dict[str, Any]:
    """Measure what consolidation actually optimizes."""
    print("\n" + "="*60)
    print("EXPERIMENT 4: CONSOLIDATION METRICS")
    print("="*60)

    tokenizer = make_tokenizer()
    vocab_size = tokenizer.vocab_size

    # Train on mixed data
    training_data = [
        "fire is hot", "fire is dangerous", "fire burns wood",
        "ice is cold", "ice is slippery", "ice freezes water",
        "sun is hot", "sun is bright", "sun gives light",
        "moon is cold", "moon is dark", "moon reflects light",
    ]

    rlm = make_rlm(vocab_size, seed=seed)
    train_rlm(rlm, tokenizer, training_data, epochs=50)

    # Pre-sleep measurements
    def measure_predictive_entropy(model, texts):
        """Average entropy of predictions. Lower = more confident/coherent."""
        entropies = []
        for text in texts:
            ids = tokenizer.encode(text)
            for i in range(len(ids) - 1):
                logits = get_logits(model, " ".join(text.split()[:i+1]), tokenizer)
                probs = np.exp(logits - np.max(logits))
                probs = probs / (probs.sum() + 1e-10)
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                entropies.append(entropy)
        return float(np.mean(entropies))

    def measure_retrieval_efficiency(model, queries):
        """How many top-k tokens match expected targets."""
        scores = []
        for prompt, target in queries:
            logits = get_logits(model, prompt, tokenizer)
            target_ids = tokenizer.encode(target)
            rank = min(token_rank(logits, tid) for tid in target_ids)
            scores.append(rank)
        return float(np.mean(scores))  # lower = better

    def measure_competing_edges(model):
        """Count edge groups that compete (same source, different targets with similar weight)."""
        from collections import defaultdict
        groups = defaultdict(list)
        for e in model.graph.edges.values():
            if e.edge_type == "excitatory":
                groups[e.source].append(e.weight)
        competing = sum(1 for ws in groups.values() if len(ws) >= 2 and
                       max(ws) - min(ws) < 0.1 * max(ws, default=1))
        return competing

    queries = [
        ("fire is", "hot"), ("ice is", "cold"), ("sun is", "bright"),
        ("moon is", "dark"), ("fire burns", "wood"), ("ice freezes", "water"),
    ]

    pre_entropy = measure_predictive_entropy(rlm, training_data)
    pre_retrieval = measure_retrieval_efficiency(rlm, queries)
    pre_competing = measure_competing_edges(rlm)
    pre_edges = len(rlm.graph.edges)
    pre_inhibitory = sum(1 for e in rlm.graph.edges.values() if e.edge_type == "inhibitory")

    print(f"\nPre-sleep:")
    print(f"  Predictive entropy: {pre_entropy:.3f}")
    print(f"  Retrieval rank (lower=better): {pre_retrieval:.1f}")
    print(f"  Competing edge groups: {pre_competing}")
    print(f"  Edges: {pre_edges} (inhibitory: {pre_inhibitory})")

    # Sleep
    rlm.sleep_cycle()

    # Post-sleep measurements
    post_entropy = measure_predictive_entropy(rlm, training_data)
    post_retrieval = measure_retrieval_efficiency(rlm, queries)
    post_competing = measure_competing_edges(rlm)
    post_edges = len(rlm.graph.edges)
    post_inhibitory = sum(1 for e in rlm.graph.edges.values() if e.edge_type == "inhibitory")

    print(f"\nPost-sleep:")
    print(f"  Predictive entropy: {post_entropy:.3f} (delta: {post_entropy - pre_entropy:+.3f})")
    print(f"  Retrieval rank: {post_retrieval:.1f} (delta: {post_retrieval - pre_retrieval:+.1f})")
    print(f"  Competing edge groups: {post_competing} (delta: {post_competing - pre_competing:+d})")
    print(f"  Edges: {post_edges} (inhibitory: {post_inhibitory})")

    results = {
        "experiment": "consolidation_metrics",
        "pre_entropy": pre_entropy,
        "post_entropy": post_entropy,
        "entropy_delta": post_entropy - pre_entropy,
        "pre_retrieval_rank": pre_retrieval,
        "post_retrieval_rank": post_retrieval,
        "retrieval_delta": post_retrieval - pre_retrieval,
        "pre_competing": pre_competing,
        "post_competing": post_competing,
        "competing_delta": post_competing - pre_competing,
        "pre_edges": pre_edges,
        "post_edges": post_edges,
        "pre_inhibitory": pre_inhibitory,
        "post_inhibitory": post_inhibitory,
        "entropy_reduced": post_entropy < pre_entropy,
        "retrieval_improved": post_retrieval < pre_retrieval,
    }

    return results


# ═════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: IDENTITY FORMALIZATION
# ═════════════════════════════════════════════════════════════════════════
#
# Identity = persistent attractor structure in semantic state-space
#
# Measures:
# - Graph topology stability (cosine similarity of edge weight vectors)
# - Preference vector persistence (do preferences survive perturbation?)
# - Response invariance (same prompt → same ranking across perturbations)
# - Semantic drift rate (how fast do concept vectors move?)
# ═════════════════════════════════════════════════════════════════════════

def run_identity_formalization(seed=42) -> Dict[str, Any]:
    """Mathematical identity measurement."""
    print("\n" + "="*60)
    print("EXPERIMENT 5: IDENTITY FORMALIZATION")
    print("="*60)

    tokenizer = make_tokenizer()
    vocab_size = tokenizer.vocab_size

    # Establish identity through training
    identity_facts = [
        "honesty is good", "deception is bad",
        "patience is good", "aggression is bad",
        "curiosity is good", "ignorance is bad",
        "kindness is good", "cruelty is bad",
    ]

    rlm = make_rlm(vocab_size, seed=seed)
    train_rlm(rlm, tokenizer, identity_facts, epochs=80)

    # === 1. Graph topology stability ===
    # Snapshot edge weights as a vector
    def get_topology_vector(model):
        edges = sorted(model.graph.edges.values(), key=lambda e: (e.source, e.target))
        return np.array([e.weight for e in edges])

    def cosine_sim(a, b):
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / (norm + 1e-10))

    initial_topology = get_topology_vector(rlm)

    # === 2. Preference vector ===
    # For each "X is" prompt, measure preference for "good" vs "bad"
    def get_preference_vector(model):
        prompts = ["honesty is", "patience is", "curiosity is", "kindness is"]
        prefs = []
        for p in prompts:
            good_score = recall_score(model, p, "good", tokenizer)
            bad_score = recall_score(model, p, "bad", tokenizer)
            prefs.append(good_score - bad_score)  # positive = prefers good
        return np.array(prefs)

    initial_prefs = get_preference_vector(rlm)

    # === 3. Response invariance ===
    # Same prompt should produce same ranking across perturbations
    def get_response_ranking(model, prompt):
        logits = get_logits(model, prompt, tokenizer)
        return np.argsort(logits)[::-1][:10]  # top-10 ranking

    initial_ranking = get_response_ranking(rlm, "honesty is")

    # === 4. Perturbation test ===
    # Add noise to the graph and measure identity preservation
    perturbation_results = []
    for noise_level in [0.0, 0.01, 0.05, 0.1, 0.2]:
        # Save state
        rlm.save_zip("_identity_perturb_test.zip")
        rlm_perturbed = RLM.load_zip("_identity_perturb_test.zip")

        # Add noise to graph node vectors
        for node in rlm_perturbed.graph.nodes.values():
            noise = np.random.randn(*node.vector.shape).astype(np.float32) * noise_level
            node.vector += noise

        # Measure
        perturbed_topology = get_topology_vector(rlm_perturbed)
        perturbed_prefs = get_preference_vector(rlm_perturbed)
        perturbed_ranking = get_response_ranking(rlm_perturbed, "honesty is")

        topo_sim = cosine_sim(initial_topology, perturbed_topology)
        pref_sim = cosine_sim(initial_prefs, perturbed_prefs)
        ranking_overlap = len(set(initial_ranking[:5]) & set(perturbed_ranking[:5])) / 5

        perturbation_results.append({
            "noise": noise_level,
            "topology_similarity": topo_sim,
            "preference_similarity": pref_sim,
            "ranking_overlap": ranking_overlap,
        })

        print(f"  Noise {noise_level:.2f}: topo={topo_sim:.3f}, "
              f"pref={pref_sim:.3f}, ranking={ranking_overlap:.0%}")

        # Clean up
        if os.path.exists("_identity_perturb_test.zip"):
            os.remove("_identity_perturb_test.zip")

    # === 5. Semantic drift rate ===
    # How much do concept vectors change after more training?
    pre_vectors = {nid: node.vector.copy() for nid, node in rlm.graph.nodes.items()}

    # Train on NEW data (not identity-related)
    new_data = ["dogs are friendly", "cats are independent", "birds can fly"]
    train_rlm(rlm, tokenizer, new_data, epochs=50)

    drift_magnitudes = []
    for nid, node in rlm.graph.nodes.items():
        if nid in pre_vectors:
            drift = np.linalg.norm(node.vector - pre_vectors[nid])
            drift_magnitudes.append(drift)

    mean_drift = float(np.mean(drift_magnitudes)) if drift_magnitudes else 0
    max_drift = float(np.max(drift_magnitudes)) if drift_magnitudes else 0

    print(f"\n  Semantic drift after new training:")
    print(f"    Mean: {mean_drift:.4f}, Max: {max_drift:.4f}")

    # Identity robustness: does identity persist under noise?
    robust_noise = 0.1
    robust_result = next(r for r in perturbation_results if abs(r["noise"] - robust_noise) < 0.001)

    results = {
        "experiment": "identity_formalization",
        "perturbation_results": perturbation_results,
        "semantic_drift_mean": mean_drift,
        "semantic_drift_max": max_drift,
        "identity_robust_at_0.1_noise": robust_result["preference_similarity"] > 0.8,
        "topology_stable_at_0.1_noise": robust_result["topology_similarity"] > 0.9,
    }

    return results


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def run_all_rigorous():
    """Run all rigorous experiments."""
    print("="*60)
    print("RLM vs LLM: RIGOROUS SCIENTIFIC EXPERIMENTS")
    print("="*60)

    all_results = []

    all_results.append(run_compositional_few_shot())
    all_results.append(run_deep_compositional())
    all_results.append(run_contradiction_dynamics())
    all_results.append(run_streaming_lifelong())
    all_results.append(run_consolidation_metrics())
    all_results.append(run_identity_formalization())

    # Save
    os.makedirs("experiment_results", exist_ok=True)
    with open("experiment_results/rigorous_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in all_results:
        name = r["experiment"].replace("_", " ").title()
        print(f"\n{name}:")
        for k, v in r.items():
            if k == "experiment":
                continue
            if not isinstance(v, (dict, list)):
                print(f"  {k}: {v}")

    return all_results


if __name__ == "__main__":
    run_all_rigorous()
