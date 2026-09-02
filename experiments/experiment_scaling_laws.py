#!/usr/bin/env python3
"""
Scaling Laws Experiments for RAVANA
====================================
Measures performance vs:
1. Graph size (1K → 100K concepts)
2. Decoder capacity (64 → 512 hidden dim)
3. Training data volume (100 → 100K articles)

Each dimension is swept independently while holding others constant.
Outputs power-law or log-linear fits for extrapolation.
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
from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import WordTokenizer


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScalingConfig:
    seed: int = 42
    output: str = None
    trace: bool = True

    # Sweep configurations (use small for quick runs, large for full)
    # Graph sizes (concepts)
    graph_sizes: List[int] = field(default_factory=lambda: [1000, 5000, 10000, 25000, 50000])
    # Decoder hidden dimensions
    decoder_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    # Training article counts
    training_articles: List[int] = field(default_factory=lambda: [100, 500, 1000, 5000, 10000])

    # Evaluation
    test_queries: List[str] = field(default_factory=lambda: [
        "what is trust", "what is friendship", "how does memory work",
        "explain quantum mechanics", "what causes gravity",
    ])
    n_runs: int = 2

    # Quick mode: fewer points
    quick: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScalingMetrics:
    sweep_type: str  # "graph", "decoder", "data"
    sweep_value: int
    run: int
    query: str
    response: str
    response_length: int
    concepts_activated: int
    concepts_in_response: int
    grammar_score: float
    concept_coherence: float
    factual_grounding: float
    diversity: float
    latency_ms: float
    memory_mb: float
    used_decoder: bool
    identity_strength: float


# ═══════════════════════════════════════════════════════════════════════════
# Quality Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_grammar_score(response: str) -> float:
    if not response or len(response) < 5:
        return 0.0
    score = 0.0
    if response[0].isupper(): score += 0.25
    if response.strip().endswith('.'): score += 0.25
    words = set(response.lower().strip('.').split())
    verbs = {'is', 'are', 'was', 'were', 'have', 'has', 'do', 'does', 'can', 'will', 'would', 'could', 'should', 'connect', 'relate', 'lead', 'cause', 'make', 'create', 'include', 'involve', 'mean', 'refer', 'stand', 'represent', 'symbolize'}
    if words & verbs: score += 0.25
    word_counts = Counter(response.lower().split())
    if max(word_counts.values()) < 3: score += 0.25
    return min(1.0, score)


def compute_concept_coherence(response: str, engine) -> float:
    if not response: return 0.0
    words = [w.strip('.,!?') for w in response.lower().split() if len(w.strip('.,!?')) >= 3]
    if not words: return 0.0
    known = sum(1 for w in words if w in engine._concept_keywords)
    return known / len(words)


def compute_factual_grounding(response: str, engine) -> float:
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


def get_memory_mb() -> float:
    """Get current process memory in MB."""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


# ═══════════════════════════════════════════════════════════════════════════
# Article Generation (synthetic for controllable scaling)
# ═══════════════════════════════════════════════════════════════════════════

SYNTHETIC_ARTICLES = [
    "Trust is the foundation of all relationships. Trust means relying on someone to act in your best interest.",
    "Friendship is a bond between people based on mutual affection and shared experiences.",
    "Love is a deep emotional attachment. Love involves care, commitment, and vulnerability.",
    "Neural networks are computing systems inspired by biological brains with interconnected neurons.",
    "Quantum mechanics describes nature at atomic scales where particles exhibit wave-particle duality.",
    "Gravity is a fundamental force that attracts objects with mass toward each other.",
    "Memory consolidation transfers information from short-term to long-term storage during sleep.",
    "Photosynthesis converts light energy into chemical energy in plants and algae.",
    "Evolution by natural selection explains the diversity of life through variation and selection.",
    "The brain processes information through neural circuits that encode and retrieve memories.",
    "Language acquisition in children follows predictable developmental stages across cultures.",
    "Cognitive dissonance occurs when beliefs conflict, motivating attitude change.",
    "Working memory holds limited information temporarily for reasoning and comprehension.",
    "Attention mechanisms allow models to focus on relevant parts of input sequences.",
    "Transfer learning applies knowledge from one domain to improve performance in another.",
    "Probabilistic reasoning updates beliefs based on evidence using Bayes theorem.",
    "Causal inference distinguishes correlation from causation using counterfactual reasoning.",
    "Metacognition is thinking about thinking, monitoring one's own cognitive processes.",
    "Analogical reasoning transfers structure from known domains to solve novel problems.",
    "Concept formation organizes experience into categories with shared properties.",
]

def generate_articles(n: int) -> List[str]:
    """Generate n articles by cycling and varying synthetic corpus."""
    if n <= len(SYNTHETIC_ARTICLES):
        return SYNTHETIC_ARTICLES[:n]
    articles = []
    for i in range(n):
        base = SYNTHETIC_ARTICLES[i % len(SYNTHETIC_ARTICLES)]
        variant = f"{base} Article {i+1} discusses this concept in detail."
        articles.append(variant)
    return articles


# ═══════════════════════════════════════════════════════════════════════════
# Engine Builders for Each Sweep
# ═══════════════════════════════════════════════════════════════════════════

def build_engine_for_graph_size(config: ScalingConfig, graph_size: int) -> CognitiveChatEngine:
    """Build engine with specific graph capacity."""
    os.environ['RAVANA_SILENT'] = '1' if not config.trace else '0'
    # Note: ConceptGraph max_nodes is set at init
    engine = CognitiveChatEngine(dim=64, seed=config.seed, baby_mode=True)
    # Artificially populate graph to target size for testing
    _populate_graph_to_size(engine.graph, graph_size)
    _retrain_decoder(engine)
    return engine


def build_engine_for_decoder_dim(config: ScalingConfig, decoder_dim: int) -> CognitiveChatEngine:
    """Build engine with specific decoder capacity."""
    os.environ['RAVANA_SILENT'] = '1' if not config.trace else '0'
    engine = CognitiveChatEngine(dim=decoder_dim, seed=config.seed, baby_mode=True)
    _retrain_decoder(engine)
    return engine


def build_engine_for_training_data(config: ScalingConfig, n_articles: int) -> CognitiveChatEngine:
    """Build engine with specific amount of web training data."""
    os.environ['RAVANA_SILENT'] = '1' if not config.trace else '0'
    engine = CognitiveChatEngine(dim=64, seed=config.seed, baby_mode=True)
    engine._network_available = False  # Use synthetic articles instead

    articles = generate_articles(n_articles)
    if engine.neural_decoder and engine._decoder_vocab_built:
        for article in articles:
            article_words = article.lower().split()
            cond_embs = engine._build_conditioning_for_text("topic", article_words)
            engine.neural_decoder.train_on_text(
                article, engine._decoder_word_to_embed, engine._decoder_word_to_idx,
                conditioning_embs=cond_embs
            )
            engine._decoder_web_training_count += max(1, len(article.split('.')))
            engine.neural_decoder.sleep_cycle()
        engine._decoder_training_count = engine._decoder_web_training_count + engine._decoder_training_count
    return engine


def _populate_graph_to_size(graph, target_size: int):
    """Populate graph with synthetic concepts to reach target size."""
    current = len(graph.nodes)
    if current >= target_size:
        return

    label_to_id = {}
    to_add = target_size - current
    # Add synthetic concept nodes
    for i in range(to_add):
        label = f"concept_{i}"
        vector = np.random.randn(64).astype(np.float32)
        vector /= np.linalg.norm(vector) + 1e-8
        node = graph.add_node(label=label, vector=vector)
        label_to_id[label] = node.id

    # Add some random edges for connectivity
    all_labels = list(label_to_id.keys())
    for i in range(min(target_size // 2, 5000)):
        src = all_labels[np.random.randint(0, len(all_labels))]
        tgt = all_labels[np.random.randint(0, len(all_labels))]
        if src != tgt:
            src_id = label_to_id[src]
            tgt_id = label_to_id[tgt]
            if src_id in graph.nodes and tgt_id in graph.nodes:
                graph.add_edge(src_id, tgt_id, relation_type="semantic", weight=1.0)


def _retrain_decoder(engine: CognitiveChatEngine):
    """Quick decoder retraining for fair comparison."""
    if engine.neural_decoder and not engine._decoder_vocab_built:
        engine._build_decoder_vocab()
    if hasattr(engine, '_seed_corpus_training'):
        engine._seed_corpus_training()


# ═══════════════════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_sweep(config: ScalingConfig, sweep_type: str, values: List[int],
              builder_fn, sweep_name: str) -> Tuple[List[ScalingMetrics], Dict]:
    """Run a single scaling sweep."""
    all_metrics = []

    print(f"\n{'='*70}")
    print(f"SCALING SWEEP: {sweep_name.upper()} ({sweep_type})")
    print(f"Values: {values}")
    print(f"{'='*70}")

    for sweep_val in values:
        print(f"\n  Testing {sweep_type} = {sweep_val}")

        for run in range(config.n_runs):
            engine = builder_fn(config, sweep_val)

            for query in config.test_queries:
                mem_before = get_memory_mb()
                t0 = time.time()

                try:
                    response = engine.process_turn(query)
                    latency = (time.time() - t0) * 1000
                    mem_after = get_memory_mb()

                    concepts_in_response = [w for w in response.lower().split()
                                           if len(w.strip('.,!?')) >= 3 and w in engine._concept_keywords]
                    pre_concepts = set()
                    for nid in getattr(engine, '_last_activated_ids', []):
                        node = engine.graph.nodes.get(nid)
                        if node and node.label:
                            pre_concepts.add(node.label.lower())

                    m = ScalingMetrics(
                        sweep_type=sweep_type,
                        sweep_value=sweep_val,
                        run=run,
                        query=query,
                        response=response[:200],
                        response_length=len(response),
                        concepts_activated=len(pre_concepts),
                        concepts_in_response=len(concepts_in_response),
                        grammar_score=compute_grammar_score(response),
                        concept_coherence=compute_concept_coherence(response, engine),
                        factual_grounding=compute_factual_grounding(response, engine),
                        diversity=compute_diversity(response),
                        latency_ms=latency,
                        memory_mb=mem_after - mem_before if mem_after > mem_before else mem_after,
                        used_decoder=(engine.neural_decoder is not None and
                                     engine._decoder_vocab_built and
                                     engine._decoder_training_count >= 500),
                        identity_strength=engine.identity.state.strength if hasattr(engine, 'identity') else 0.0,
                    )
                    all_metrics.append(m)

                    if config.trace:
                        print(f"    {query[:30]:30s} | {m.latency_ms:6.1f}ms | "
                              f"gram={m.grammar_score:.2f} coh={m.concept_coherence:.2f}")

                except Exception as e:
                    print(f"    ERROR: {e}")
                    all_metrics.append(ScalingMetrics(
                        sweep_type=sweep_type, sweep_value=sweep_val, run=run,
                        query=query, response="ERROR", response_length=0,
                        concepts_activated=0, concepts_in_response=0,
                        grammar_score=0.0, concept_coherence=0.0,
                        factual_grounding=0.0, diversity=0.0,
                        latency_ms=0.0, memory_mb=0.0,
                        used_decoder=False, identity_strength=0.0,
                    ))

    # Aggregate summary
    summary = {}
    for val in values:
        val_metrics = [m for m in all_metrics if m.sweep_value == val and m.latency_ms > 0]
        if val_metrics:
            summary[val] = {
                "n": len(val_metrics),
                "avg_latency_ms": np.mean([m.latency_ms for m in val_metrics]),
                "avg_memory_mb": np.mean([m.memory_mb for m in val_metrics]),
                "avg_grammar": np.mean([m.grammar_score for m in val_metrics]),
                "avg_coherence": np.mean([m.concept_coherence for m in val_metrics]),
                "avg_grounding": np.mean([m.factual_grounding for m in val_metrics]),
                "avg_diversity": np.mean([m.diversity for m in val_metrics]),
                "avg_concepts": np.mean([m.concepts_in_response for m in val_metrics]),
                "decoder_rate": np.mean([1.0 if m.used_decoder else 0.0 for m in val_metrics]),
            }

    return all_metrics, summary


def fit_scaling_law(summary: Dict, sweep_type: str) -> Dict:
    """Fit power law or log-linear scaling relationship."""
    if len(summary) < 3:
        return {"error": "Need 3+ points for fitting"}

    xs = np.array(sorted(summary.keys()), dtype=float)
    # Fit latency vs scale
    ys_latency = np.array([summary[x]["avg_latency_ms"] for x in xs])

    # Try power law: y = a * x^b  -> log y = log a + b log x
    log_x = np.log(xs + 1)
    log_y = np.log(ys_latency + 1)

    # Linear regression on log-log
    A = np.vstack([log_x, np.ones_like(log_x)]).T
    b, log_a = np.linalg.lstsq(A, log_y, rcond=None)[0]
    a = np.exp(log_a)

    power_law = {"a": float(a), "b": float(b), "form": "y = a * x^b"}

    # Also linear on raw
    A_raw = np.vstack([xs, np.ones_like(xs)]).T
    m, c = np.linalg.lstsq(A_raw, ys_latency, rcond=None)[0]
    linear = {"slope": float(m), "intercept": float(c), "form": "y = m*x + c"}

    return {
        "power_law": power_law,
        "linear": linear,
        "data_points": len(xs),
    }


def run_scaling_experiment(config: ScalingConfig = None):
    if config is None:
        config = ScalingConfig()

    # Quick mode: fewer points
    if config.quick:
        config.graph_sizes = [1000, 5000, 10000]
        config.decoder_dims = [64, 128, 256]
        config.training_articles = [100, 500, 1000]

    np.random.seed(config.seed)

    print("=" * 70)
    print("SCALING LAWS EXPERIMENT")
    print("=" * 70)

    all_results = {}
    all_summaries = {}

    # Sweep 1: Graph size
    metrics1, summary1 = run_sweep(
        config, "graph", config.graph_sizes,
        build_engine_for_graph_size, "Graph Size"
    )
    all_results["graph"] = metrics1
    all_summaries["graph"] = summary1

    # Sweep 2: Decoder dimension
    metrics2, summary2 = run_sweep(
        config, "decoder", config.decoder_dims,
        build_engine_for_decoder_dim, "Decoder Dim"
    )
    all_results["decoder"] = metrics2
    all_summaries["decoder"] = summary2

    # Sweep 3: Training data
    metrics3, summary3 = run_sweep(
        config, "data", config.training_articles,
        build_engine_for_training_data, "Training Articles"
    )
    all_results["data"] = metrics3
    all_summaries["data"] = summary3

    # Fit scaling laws
    scaling_laws = {}
    for sweep_type in ["graph", "decoder", "data"]:
        scaling_laws[sweep_type] = fit_scaling_law(all_summaries[sweep_type], sweep_type)

    # Summary
    print("\n" + "=" * 70)
    print("SCALING LAWS SUMMARY")
    print("=" * 70)

    for sweep_type in ["graph", "decoder", "data"]:
        print(f"\n{sweep_type.upper()} SWEEP:")
        summary = all_summaries[sweep_type]
        for val, stats in sorted(summary.items()):
            print(f"  {val:>8}: latency={stats['avg_latency_ms']:6.1f}ms  "
                  f"mem={stats['avg_memory_mb']:5.1f}MB  "
                  f"gram={stats['avg_grammar']:.3f}  coh={stats['avg_coherence']:.3f}")

        law = scaling_laws[sweep_type]
        if "error" not in law:
            pl = law["power_law"]
            print(f"  Power law fit: y = {pl['a']:.4f} * x^{pl['b']:.4f}")

    # Save
    if config.output:
        output = {
            'config': asdict(config),
            'summaries': all_summaries,
            'scaling_laws': scaling_laws,
            'detailed_results': {k: [asdict(m) for m in v] for k, v in all_results.items()},
        }
        with open(config.output, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {config.output}")

    return all_results, all_summaries, scaling_laws


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAVANA Scaling Laws Experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--runs", type=int, default=2, help="Runs per point")
    parser.add_argument("--no-trace", action="store_true", help="Disable trace output")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--quick", action="store_true", help="Quick mode with fewer points")
    parser.add_argument("--sweep", type=str, choices=["graph", "decoder", "data"],
                        help="Run only specific sweep")
    args = parser.parse_args()

    config = ScalingConfig(
        seed=args.seed,
        n_runs=args.runs,
        trace=not args.no_trace,
        output=args.output,
        quick=args.quick,
    )

    if args.sweep:
        # Run single sweep
        if args.sweep == "graph":
            run_sweep(config, "graph", config.graph_sizes, build_engine_for_graph_size, "Graph Size")
        elif args.sweep == "decoder":
            run_sweep(config, "decoder", config.decoder_dims, build_engine_for_decoder_dim, "Decoder Dim")
        elif args.sweep == "data":
            run_sweep(config, "data", config.training_articles, build_engine_for_training_data, "Training Articles")
    else:
        run_scaling_experiment(config)


if __name__ == "__main__":
    main()