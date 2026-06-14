#!/usr/bin/env python3
"""
Ablation Studies for RAVANA Cognitive Architecture
===================================================
Tests the contribution of each major component by removing it one at a time:
1. Neural Decoder (neural language generation)
2. Syntactic Pipeline (SyntacticCellAssembly + SurfaceRealizer)
3. Curiosity Drive (autonomous background learning)
4. Sleep Consolidation (offline memory consolidation)
5. User Model (personalization/topic preferences)
6. Basal Ganglia Gate (action selection gating)
7. Cerebellar N-gram (sequence learning for grammar)

Each ablation runs the same benchmark suite and compares against full model.
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
class AblationConfig:
    """Configuration for ablation experiments."""
    seed: int = 42
    dim: int = 64
    test_queries: List[str] = field(default_factory=lambda: [
        "what is trust",
        "what is friendship",
        "how does trust work",
        "create a blueprint for trust",
        "why do people betray each other",
        "what makes a friendship last",
        "explain the concept of love",
        "how does memory work",
    ])
    n_runs: int = 3
    output: str = None
    trace: bool = True

    # Ablation toggles
    ablate_decoder: bool = False
    ablate_syntactic: bool = False
    ablate_curiosity: bool = False
    ablate_sleep: bool = False
    ablate_user_model: bool = False
    ablate_basal_ganglia: bool = False
    ablate_cerebellar: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AblationMetrics:
    """Metrics collected per ablation condition."""
    ablation_name: str
    run: int
    query: str
    response: str
    response_length: int
    strategy: str
    concepts_activated: List[str]
    concepts_in_response: List[str]
    unique_concepts: int
    grammar_score: float
    concept_coherence: float
    factual_grounding: float
    diversity: float
    latency_ms: float
    used_decoder: bool
    identity_strength: float
    trace: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Quality Metrics (from experiment_chat_quality.py)
# ═══════════════════════════════════════════════════════════════════════════

def compute_grammar_score(response: str) -> float:
    if not response or len(response) < 5:
        return 0.0
    score = 0.0
    if response[0].isupper():
        score += 0.25
    if response.strip().endswith('.'):
        score += 0.25
    words = set(response.lower().strip('.').split())
    verbs = {'is', 'are', 'was', 'were', 'have', 'has', 'do', 'does', 'can', 'will', 'would', 'could', 'should', 'connect', 'relate', 'lead', 'cause', 'make', 'create', 'include', 'involve', 'mean', 'refer', 'stand', 'represent', 'symbolize'}
    if words & verbs:
        score += 0.25
    from collections import Counter
    word_counts = Counter(response.lower().split())
    if max(word_counts.values()) < 3:
        score += 0.25
    return min(1.0, score)


def compute_concept_coherence(response: str, engine: CognitiveChatEngine) -> float:
    if not response:
        return 0.0
    words = [w.strip('.,!?') for w in response.lower().split() if len(w.strip('.,!?')) >= 3]
    if not words:
        return 0.0
    known = sum(1 for w in words if w in engine._concept_keywords)
    return known / len(words)


def compute_factual_grounding(response: str, engine: CognitiveChatEngine) -> float:
    if not response:
        return 0.0
    stop = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'were', 'be', 'been', 'are', 'am', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'shall', 'can', 'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their', 'we', 'our', 'you', 'your', 'he', 'she', 'him', 'her', 'his', 'not', 'no', 'nor', 'so', 'if', 'then', 'than', 'too', 'very', 'just', 'about', 'also', 'into', 'over', 'after', 'before', 'i', 'me', 'my', 'we', 'us', 'our', 'you', 'your', 'he', 'she', 'him', 'her', 'his'}
    content = [w for w in response.lower().split() if w not in stop and len(w.strip('.,!?')) >= 3]
    if not content:
        return 0.0
    grounded = sum(1 for w in content if w in engine._concept_keywords)
    return grounded / len(content)


def compute_diversity(response: str) -> float:
    if not response:
        return 0.0
    words = [w for w in response.lower().split() if len(w.strip('.,!?')) >= 3]
    if not words:
        return 0.0
    return len(set(words)) / len(words)


# ═══════════════════════════════════════════════════════════════════════════
# Engine Builders (with ablation toggles)
# ═══════════════════════════════════════════════════════════════════════════

def build_engine(config: AblationConfig, ablation_name: str) -> CognitiveChatEngine:
    """Build engine with specified ablation applied."""
    os.environ['RAVANA_SILENT'] = '1' if not config.trace else '0'
    engine = CognitiveChatEngine(dim=config.dim, seed=config.seed, baby_mode=True)

    # Apply ablation
    if ablation_name == "decoder":
        engine.neural_decoder = None
        engine._decoder_vocab_built = False
    elif ablation_name == "syntactic":
        engine.syntactic_assembly = None
        engine.surface_realizer = None
    elif ablation_name == "curiosity":
        engine._curiosity_drive_enabled = False
    elif ablation_name == "sleep":
        engine.sleep_engine = None
    elif ablation_name == "user_model":
        engine.user_model = None
    elif ablation_name == "basal_ganglia":
        engine.basal_ganglia = None
    elif ablation_name == "cerebellar":
        engine.cerebellar_ngram = None
        engine._cerebellar_ngram = {}
        engine._cerebellar_depth = {}

    # Re-train decoder from scratch if not ablated (ensures fair comparison)
    if ablation_name != "decoder":
        _retrain_decoder(engine)

    return engine


def _retrain_decoder(engine: CognitiveChatEngine):
    """Quick retrain decoder for fair comparison across ablations."""
    # Use the same seed corpus training
    if hasattr(engine, '_seed_corpus_training'):
        engine._seed_corpus_training()
    if engine.neural_decoder and not engine._decoder_vocab_built:
        engine._build_decoder_vocab()


# ═══════════════════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_ablation_experiment(config: AblationConfig = None):
    if config is None:
        config = AblationConfig()

    np.random.seed(config.seed)

    # Define ablation conditions
    ablation_conditions = [
        "full_model",           # Baseline: all components active
        "decoder",              # No neural decoder
        "syntactic",            # No syntactic pipeline
        "curiosity",            # No curiosity drive
        "sleep",                # No sleep consolidation
        "user_model",           # No user model
        "basal_ganglia",        # No basal ganglia gate
        "cerebellar",           # No cerebellar n-gram
    ]

    print("=" * 70)
    print("ABLATION STUDIES EXPERIMENT")
    print("=" * 70)
    print(f"Conditions: {len(ablation_conditions)}")
    print(f"Queries per condition: {len(config.test_queries)}")
    print(f"Runs per query: {config.n_runs}")
    print()

    all_results = {}

    # Run full model first as baseline
    for ablation_name in ablation_conditions:
        print(f"\n{'='*70}")
        print(f"ABLATION: {ablation_name.upper()}")
        print(f"{'='*70}")

        condition_results = []

        for run in range(config.n_runs):
            print(f"\n  Run {run+1}/{config.n_runs}")
            engine = build_engine(config, ablation_name)

            for query in config.test_queries:
                t0 = time.time()
                try:
                    response = engine.process_turn(query)
                    latency = (time.time() - t0) * 1000

                    # Extract metrics
                    concepts_in_response = [w for w in response.lower().split()
                                           if len(w.strip('.,!?')) >= 3 and w in engine._concept_keywords]

                    # Get activated concepts
                    pre_concepts = set()
                    for nid in getattr(engine, '_last_activated_ids', []):
                        node = engine.graph.nodes.get(nid)
                        if node and node.label:
                            pre_concepts.add(node.label.lower())

                    metrics = AblationMetrics(
                        ablation_name=ablation_name,
                        run=run,
                        query=query,
                        response=response[:500],
                        response_length=len(response),
                        strategy=getattr(engine, '_last_strategy', 'unknown'),
                        concepts_activated=list(pre_concepts),
                        concepts_in_response=concepts_in_response,
                        unique_concepts=len(set(concepts_in_response)),
                        grammar_score=compute_grammar_score(response),
                        concept_coherence=compute_concept_coherence(response, engine),
                        factual_grounding=compute_factual_grounding(response, engine),
                        diversity=compute_diversity(response),
                        latency_ms=latency,
                        used_decoder=engine.neural_decoder is not None and engine._decoder_vocab_built and engine._decoder_training_count >= 500,
                        identity_strength=engine.identity.state.strength if hasattr(engine, 'identity') else 0.0,
                    )
                    condition_results.append(metrics)

                    if config.trace:
                        print(f"    Query: {query}")
                        print(f"    Response: {response[:100]}...")
                        print(f"    Strategy: {metrics.strategy}, Concepts: {metrics.unique_concepts}, "
                              f"Grammar: {metrics.grammar_score:.2f}, Latency: {metrics.latency_ms:.1f}ms")

                except Exception as e:
                    print(f"    ERROR on query '{query}': {e}")
                    # Add failed result
                    metrics = AblationMetrics(
                        ablation_name=ablation_name,
                        run=run,
                        query=query,
                        response="ERROR",
                        response_length=0,
                        strategy="error",
                        concepts_activated=[],
                        concepts_in_response=[],
                        unique_concepts=0,
                        grammar_score=0.0,
                        concept_coherence=0.0,
                        factual_grounding=0.0,
                        diversity=0.0,
                        latency_ms=0.0,
                        used_decoder=False,
                        identity_strength=0.0,
                        trace=str(e),
                    )
                    condition_results.append(metrics)

        all_results[ablation_name] = condition_results

    # Summary
    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY")
    print("=" * 70)

    summary = {}
    for ablation_name, results in all_results.items():
        valid = [r for r in results if r.strategy != "error"]
        if not valid:
            summary[ablation_name] = {"error": "All runs failed"}
            continue

        n = len(valid)
        summary[ablation_name] = {
            "n_samples": n,
            "avg_response_length": np.mean([r.response_length for r in valid]),
            "avg_unique_concepts": np.mean([r.unique_concepts for r in valid]),
            "avg_grammar": np.mean([r.grammar_score for r in valid]),
            "avg_concept_coherence": np.mean([r.concept_coherence for r in valid]),
            "avg_factual_grounding": np.mean([r.factual_grounding for r in valid]),
            "avg_diversity": np.mean([r.diversity for r in valid]),
            "avg_latency_ms": np.mean([r.latency_ms for r in valid]),
            "decoder_usage_rate": np.mean([1.0 if r.used_decoder else 0.0 for r in valid]),
            "avg_identity_strength": np.mean([r.identity_strength for r in valid]),
            "strategy_distribution": dict(Counter(r.strategy for r in valid)),
        }

        # Compute deltas from full model baseline
        if ablation_name != "full_model" and "full_model" in summary:
            base = summary["full_model"]
            for key in ["avg_grammar", "avg_concept_coherence", "avg_factual_grounding",
                        "avg_diversity", "avg_unique_concepts"]:
                base_val = base.get(key, 0)
                abl_val = summary[ablation_name].get(key, 0)
                if base_val != 0:
                    summary[ablation_name][f"delta_{key}"] = abl_val - base_val
                    summary[ablation_name][f"pct_change_{key}"] = (abl_val - base_val) / base_val * 100

        print(f"\n{ablation_name}:")
        for key, val in summary[ablation_name].items():
            if isinstance(val, float):
                print(f"  {key}: {val:.4f}")
            elif isinstance(val, dict):
                print(f"  {key}: {val}")

    # Save detailed results
    if config.output:
        output = {
            'config': asdict(config),
            'summary': summary,
            'detailed_results': {k: [asdict(r) for r in v] for k, v in all_results.items()},
        }
        with open(config.output, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {config.output}")

    return all_results, summary


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAVANA Ablation Studies")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dim", type=int, default=64, help="Graph dimension")
    parser.add_argument("--runs", type=int, default=3, help="Runs per query")
    parser.add_argument("--no-trace", action="store_true", help="Disable trace output")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--only", type=str, help="Run single ablation (e.g., 'decoder')")
    args = parser.parse_args()

    config = AblationConfig(
        seed=args.seed,
        dim=args.dim,
        n_runs=args.runs,
        trace=not args.no_trace,
        output=args.output,
    )

    if args.only:
        # Override to run only specific ablation
        config.__dict__[f"ablate_{args.only}"] = True

    run_ablation_experiment(config)


if __name__ == "__main__":
    main()