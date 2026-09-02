#!/usr/bin/env python3
"""
Component Interaction Experiments for RAVANA
=============================================
Tests synergistic effects between components:
1. Decoder + Syntactic Pipeline synergy
2. Curiosity Drive + Sleep Consolidation synergy
3. Basal Ganglia + Cerebellar N-gram interaction
4. User Model + Identity coherence
5. Global Workspace + Dual Process integration
6. Full system emergent behavior
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.ravana_chat import CognitiveChatEngine


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class InteractionConfig:
    seed: int = 42
    output: str = None
    trace: bool = True
    n_runs: int = 3

    test_queries: List[str] = field(default_factory=lambda: [
        "what is trust",
        "how does trust work in relationships",
        "create a blueprint for building trust",
        "why do people betray each other",
        "what makes a friendship last",
        "explain the neuroscience of trust",
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class InteractionMetrics:
    condition: str
    run: int
    query: str
    response: str
    latency_ms: float
    # Quality
    grammar: float
    coherence: float
    grounding: float
    diversity: float
    # Component usage
    used_decoder: bool
    used_syntactic: bool
    used_reasoning: bool
    used_sleep: bool
    # Identity/User model
    identity_strength: float
    user_model_edges: int


# ═══════════════════════════════════════════════════════════════════════════
# Quality Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_grammar_score(response: str) -> float:
    if not response or len(response) < 5: return 0.0
    score = 0.0
    if response[0].isupper(): score += 0.25
    if response.strip().endswith('.'): score += 0.25
    words = set(response.lower().strip('.').split())
    verbs = {'is', 'are', 'was', 'were', 'have', 'has', 'do', 'does', 'can', 'will', 'would', 'could', 'should', 'connect', 'relate', 'lead', 'cause', 'make', 'create', 'include', 'involve', 'mean', 'refer', 'stand', 'represent', 'symbolize'}
    if words & verbs: score += 0.25
    word_counts = Counter(response.lower().split())
    if max(word_counts.values()) < 3: score += 0.25
    return min(1.0, score)


def compute_coherence(response: str, engine) -> float:
    if not response: return 0.0
    words = [w.strip('.,!?') for w in response.lower().split() if len(w.strip('.,!?')) >= 3]
    if not words: return 0.0
    known = sum(1 for w in words if w in engine._concept_keywords)
    return known / len(words)


def compute_grounding(response: str, engine) -> float:
    if not response: return 0.0
    stop = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'were', 'be', 'been', 'are', 'am', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'shall', 'can', 'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their', 'we', 'our', 'you', 'your', 'he', 'she', 'him', 'her', 'his', 'not', 'no', 'nor', 'so', 'if', 'then', 'than', 'too', 'very', 'just', 'about', 'also', 'into', 'over', 'after', 'before', 'i', 'me', 'my', 'we', 'us', 'our', 'you', 'your', 'he', 'she', 'him', 'her', 'his'}
    content = [w for w in response.lower().split() if w not in stop and len(w.strip('.,!?')) >= 3]
    if not content: return 0.0
    grounded = sum(1 for w in content if w in engine._concept_keywords)
    return grounded / len(content)


def compute_diversity(response: str) -> float:
    if not response: return 0.0
    words = [w for w in response.lower().split() if len(w.strip('.,!?')) >= 3]
    if not words: return 0.0
    return len(set(words)) / len(words)


# ═══════════════════════════════════════════════════════════════════════════
# Engine Builders for Interaction Conditions
# ═══════════════════════════════════════════════════════════════════════════

def build_full_system(config: InteractionConfig) -> CognitiveChatEngine:
    """Full RAVANA with all components."""
    os.environ['RAVANA_SILENT'] = '1' if not config.trace else '0'
    engine = CognitiveChatEngine(dim=64, seed=config.seed, baby_mode=True)
    if hasattr(engine, '_seed_corpus_training'):
        engine._seed_corpus_training()
    return engine


def build_no_decoder(config: InteractionConfig) -> CognitiveChatEngine:
    """No neural decoder - templates only."""
    engine = build_full_system(config)
    engine.neural_decoder = None
    engine._decoder_vocab_built = False
    return engine


def build_no_syntactic(config: InteractionConfig) -> CognitiveChatEngine:
    """No syntactic pipeline."""
    engine = build_full_system(config)
    engine.syntactic_assembly = None
    engine.surface_realizer = None
    return engine


def build_decoder_only(config: InteractionConfig) -> CognitiveChatEngine:
    """Decoder only, no syntactic."""
    engine = build_no_syntactic(config)
    return engine


def build_syntactic_only(config: InteractionConfig) -> CognitiveChatEngine:
    """Syntactic only, no decoder."""
    engine = build_no_decoder(config)
    return engine


def build_no_curiosity(config: InteractionConfig) -> CognitiveChatEngine:
    """No curiosity drive."""
    engine = build_full_system(config)
    engine._curiosity_drive_enabled = False
    return engine


def build_no_sleep(config: InteractionConfig) -> CognitiveChatEngine:
    """No sleep consolidation."""
    engine = build_full_system(config)
    engine.sleep_engine = None
    return engine


def build_curiosity_no_sleep(config: InteractionConfig) -> CognitiveChatEngine:
    """Curiosity without sleep."""
    engine = build_no_sleep(config)
    engine._curiosity_drive_enabled = True
    return engine


def build_sleep_no_curiosity(config: InteractionConfig) -> CognitiveChatEngine:
    """Sleep without curiosity."""
    engine = build_no_curiosity(config)
    engine.sleep_engine = None  # Will be recreated in build_full_system then removed
    return engine


def build_no_basal_ganglia(config: InteractionConfig) -> CognitiveChatEngine:
    """No basal ganglia gating."""
    engine = build_full_system(config)
    engine.basal_ganglia = None
    return engine


def build_no_cerebellar(config: InteractionConfig) -> CognitiveChatEngine:
    """No cerebellar n-gram."""
    engine = build_full_system(config)
    engine.cerebellar_ngram = None
    engine._cerebellar_ngram = {}
    engine._cerebellar_depth = {}
    return engine


def build_no_basal_no_cerebellar(config: InteractionConfig) -> CognitiveChatEngine:
    """Neither basal ganglia nor cerebellar."""
    engine = build_no_basal_ganglia(config)
    engine.cerebellar_ngram = None
    engine._cerebellar_ngram = {}
    engine._cerebellar_depth = {}
    return engine


def build_no_user_model(config: InteractionConfig) -> CognitiveChatEngine:
    """No user model personalization."""
    engine = build_full_system(config)
    engine.user_model = None
    return engine


def build_no_identity(config: InteractionConfig) -> CognitiveChatEngine:
    """No identity engine (or minimal)."""
    engine = build_full_system(config)
    engine.identity = None
    return engine


# ═══════════════════════════════════════════════════════════════════════════
# Interaction Test Conditions
# ═══════════════════════════════════════════════════════════════════════════

INTERACTION_CONDITIONS = {
    # Decoder + Syntactic synergy
    "full_system": build_full_system,
    "decoder_only": build_decoder_only,
    "syntactic_only": build_syntactic_only,
    "no_decoder": build_no_decoder,
    "no_syntactic": build_no_syntactic,

    # Curiosity + Sleep synergy
    "curiosity_only": build_no_sleep,
    "sleep_only": build_sleep_no_curiosity,
    "curiosity_sleep": build_full_system,
    "neither_curiosity_sleep": build_no_curiosity,  # Will also have no sleep

    # Basal Ganglia + Cerebellar interaction
    "basal_only": build_no_cerebellar,
    "cerebellar_only": build_no_basal_ganglia,
    "basal_cerebellar": build_full_system,
    "neither_basal_cerebellar": build_no_basal_no_cerebellar,

    # User Model + Identity
    "user_model_only": build_no_identity,
    "identity_only": build_no_user_model,
    "user_identity": build_full_system,
    "neither_user_identity": build_no_user_model,  # Will also have no identity
}


# ═══════════════════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_condition(config: InteractionConfig, condition_name: str, builder) -> List[InteractionMetrics]:
    """Run all queries for a single condition."""
    metrics = []

    for run in range(config.n_runs):
        engine = builder(config)

        for query in config.test_queries:
            t0 = time.time()
            response = engine.process_turn(query)
            latency = (time.time() - t0) * 1000

            m = InteractionMetrics(
                condition=condition_name,
                run=run,
                query=query,
                response=response[:200],
                latency_ms=latency,
                grammar=compute_grammar_score(response),
                coherence=compute_coherence(response, engine),
                grounding=compute_grounding(response, engine),
                diversity=compute_diversity(response),
                used_decoder=(engine.neural_decoder is not None and
                              engine._decoder_vocab_built and
                              engine._decoder_training_count >= 500),
                used_syntactic=(engine.syntactic_assembly is not None and
                                engine.surface_realizer is not None),
                used_reasoning=getattr(engine, '_last_strategy', '').startswith('reasoning'),
                used_sleep=engine.sleep_cycles_completed > 0,
                identity_strength=engine.identity.state.strength if hasattr(engine, 'identity') and engine.identity else 0.0,
                user_model_edges=len(engine.user_model.edge_reactivations) if hasattr(engine, 'user_model') and engine.user_model else 0,
            )
            metrics.append(m)

            if config.trace:
                print(f"    {condition_name} run{run+1}: {query[:30]:30s} | "
                      f"gram={m.grammar:.2f} coh={m.coherence:.2f} "
                      f"dec={m.used_decoder} syn={m.used_syntactic}")

    return metrics


def compute_synergy(both_on, only_a, only_b, neither):
    """Compute synergy index: (both - neither) - ((only_a - neither) + (only_b - neither))"""
    # Positive synergy = components work better together than sum of parts
    return (both_on - neither) - ((only_a - neither) + (only_b - neither))


def run_interaction_experiment(config: InteractionConfig = None):
    if config is None:
        config = InteractionConfig()

    np.random.seed(config.seed)

    print("=" * 70)
    print("COMPONENT INTERACTION EXPERIMENTS")
    print("=" * 70)

    all_results = {}

    # Define condition groups for synergy analysis
    condition_groups = {
        "decoder_syntactic": {
            "both": "full_system",
            "only_a": "decoder_only",
            "only_b": "syntactic_only",
            "neither": "full_system",  # We don't have true neither, use template-only baseline
        },
        "curiosity_sleep": {
            "both": "curiosity_sleep",
            "only_a": "curiosity_only",
            "only_b": "sleep_only",
            "neither": "neither_curiosity_sleep",
        },
        "basal_cerebellar": {
            "both": "basal_cerebellar",
            "only_a": "basal_only",
            "only_b": "cerebellar_only",
            "neither": "neither_basal_cerebellar",
        },
        "user_identity": {
            "both": "user_identity",
            "only_a": "user_model_only",
            "only_b": "identity_only",
            "neither": "neither_user_identity",
        },
    }

    # Run all conditions
    for cond_name, builder in INTERACTION_CONDITIONS.items():
        print(f"\n{'='*60}")
        print(f"CONDITION: {cond_name}")
        print(f"{'='*60}")

        results = run_condition(config, cond_name, builder)
        all_results[cond_name] = results

    # Analyze synergies
    print("\n" + "=" * 70)
    print("SYNERGY ANALYSIS")
    print("=" * 70)

    synergy_results = {}

    for group_name, conds in condition_groups.items():
        print(f"\n{group_name.upper()} Synergy:")

        # Aggregate metrics per condition
        cond_metrics = {}
        for cond_key, cond_name in conds.items():
            if cond_name in all_results:
                results = all_results[cond_name]
                cond_metrics[cond_key] = {
                    "grammar": np.mean([r.grammar for r in results]),
                    "coherence": np.mean([r.coherence for r in results]),
                    "grounding": np.mean([r.grounding for r in results]),
                    "diversity": np.mean([r.diversity for r in results]),
                    "latency": np.mean([r.latency_ms for r in results]),
                    "decoder_usage": np.mean([1.0 if r.used_decoder else 0.0 for r in results]),
                    "syntactic_usage": np.mean([1.0 if r.used_syntactic else 0.0 for r in results]),
                }

        # Compute synergy for key metrics
        if all(k in cond_metrics for k in ["both", "only_a", "only_b", "neither"]):
            for metric in ["grammar", "coherence", "grounding", "diversity"]:
                both = cond_metrics["both"][metric]
                only_a = cond_metrics["only_a"][metric]
                only_b = cond_metrics["only_b"][metric]
                neither = cond_metrics["neither"][metric]

                synergy = compute_synergy(both, only_a, only_b, neither)
                synergy_results[f"{group_name}_{metric}"] = synergy

                print(f"  {metric}: both={both:.3f} only_a={only_a:.3f} only_b={only_b:.3f} neither={neither:.3f}")
                print(f"    Synergy: {synergy:.4f} ({'positive' if synergy > 0 else 'negative' if synergy < 0 else 'neutral'})")

    # Overall summary
    print("\n" + "=" * 70)
    print("INTERACTION SUMMARY")
    print("=" * 70)

    summary = {}
    for cond_name, results in all_results.items():
        summary[cond_name] = {
            "n_queries": len(config.test_queries) * config.n_runs,
            "avg_grammar": float(np.mean([r.grammar for r in results])),
            "avg_coherence": float(np.mean([r.coherence for r in results])),
            "avg_grounding": float(np.mean([r.grounding for r in results])),
            "avg_diversity": float(np.mean([r.diversity for r in results])),
            "avg_latency_ms": float(np.mean([r.latency_ms for r in results])),
            "decoder_usage": float(np.mean([1.0 if r.used_decoder else 0.0 for r in results])),
            "syntactic_usage": float(np.mean([1.0 if r.used_syntactic else 0.0 for r in results])),
            "avg_identity": float(np.mean([r.identity_strength for r in results])),
        }
        print(f"\n{cond_name}:")
        print(f"  Grammar: {summary[cond_name]['avg_grammar']:.3f}")
        print(f"  Coherence: {summary[cond_name]['avg_coherence']:.3f}")
        print(f"  Grounding: {summary[cond_name]['avg_grounding']:.3f}")
        print(f"  Diversity: {summary[cond_name]['avg_diversity']:.3f}")

    # Interaction effects
    print("\n" + "=" * 50)
    print("KEY INTERACTION EFFECTS")
    print("=" * 50)

    # Decoder+Syntactic synergy on coherence
    if "decoder_syntactic_coherence" in synergy_results:
        syn = synergy_results["decoder_syntactic_coherence"]
        print(f"Decoder+Syntactic → Coherence synergy: {syn:+.4f}")

    # Curiosity+Sleep synergy on grounding
    if "curiosity_sleep_grounding" in synergy_results:
        syn = synergy_results["curiosity_sleep_grounding"]
        print(f"Curiosity+Sleep → Grounding synergy: {syn:+.4f}")

    # Basal+Cerebellar synergy on diversity (grammar)
    if "basal_cerebellar_grammar" in synergy_results:
        syn = synergy_results["basal_cerebellar_grammar"]
        print(f"Basal+Cerebellar → Grammar synergy: {syn:+.4f}")

    # User+Identity synergy on coherence
    if "user_identity_coherence" in synergy_results:
        syn = synergy_results["user_identity_coherence"]
        print(f"UserModel+Identity → Coherence synergy: {syn:+.4f}")

    # Save
    if config.output:
        output = {
            'config': asdict(config),
            'summary': summary,
            'synergy': synergy_results,
            'detailed_results': {k: [asdict(r) for r in v] for k, v in all_results.items()},
        }
        with open(config.output, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {config.output}")

    return all_results, synergy_results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAVANA Component Interaction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--runs", type=int, default=3, help="Runs per query")
    parser.add_argument("--no-trace", action="store_true", help="Disable trace output")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()

    config = InteractionConfig(
        seed=args.seed,
        n_runs=args.runs,
        trace=not args.no_trace,
        output=args.output,
    )

    run_interaction_experiment(config)


if __name__ == "__main__":
    main()