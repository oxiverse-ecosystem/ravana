#!/usr/bin/env python3
"""
Long-term Persistence Experiments for RAVANA
=============================================
Tests multi-session persistence, concept drift, and catastrophic forgetting:
1. Multi-session continuity (days/weeks simulation)
2. Concept drift detection over time
3. Catastrophic forgetting over 10K+ turns
4. Knowledge retention after sleep cycles
5. User model stability across sessions
"""

import os
import sys
import time
import json
import numpy as np
import random
import pickle
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
class PersistenceConfig:
    seed: int = 42
    output: str = None
    trace: bool = True

    # Session parameters
    n_sessions: int = 10          # Number of simulated "days"
    turns_per_session: int = 1000  # Turns per session
    save_between_sessions: bool = True  # Save/load weights between sessions

    # Evaluation
    eval_interval: int = 100      # Evaluate every N turns
    probe_concepts: List[str] = field(default_factory=lambda: [
        "trust", "friendship", "love", "fear", "hope", "courage",
        "betrayal", "loyalty", "memory", "learning", "sleep", "dream"
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PersistenceMetrics:
    session: int
    turn: int
    global_turn: int
    query: str
    response: str
    concepts_in_response: List[str]
    unique_concepts: int
    grammar_score: float
    concept_coherence: float
    factual_grounding: float
    diversity: float
    latency_ms: float
    identity_strength: float
    user_model_edges: int
    sleep_cycles: int
    graph_size: int


@dataclass
class ForgettingMetrics:
    concept: str
    session_first_seen: int
    session_last_seen: int
    total_mentions: int
    sessions_absent: int
    forgotten: bool


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


# ═══════════════════════════════════════════════════════════════════════════
# Query Generation
# ═══════════════════════════════════════════════════════════════════════════

QUERY_POOLS = {
    "core": [
        "what is trust", "what is friendship", "what is love",
        "what is fear", "what is hope", "what is courage",
        "what is betrayal", "what is loyalty", "what is memory",
    ],
    "followup": [
        "tell me more", "what else", "why is that",
        "how does that work", "give me an example",
        "can you explain further", "what do you mean",
    ],
    "complex": [
        "how does trust work in relationships",
        "create a blueprint for building trust",
        "why do people betray each other",
        "what makes a friendship last",
        "how does memory shape identity",
        "what is the role of sleep in learning",
    ],
    "personal": [
        "do you have friends", "what do you think about betrayal",
        "have you ever been hurt", "what matters most to you",
        "do you dream", "what are you afraid of",
    ],
    "novel": [  # New concepts to track learning
        "what is quantum entanglement", "explain blockchain",
        "how does photosynthesis work", "what is relativity",
        "define machine learning", "what is consciousness",
    ],
}


def generate_session_queries(config: PersistenceConfig, session: int) -> List[str]:
    """Generate queries for a session with increasing novelty over time."""
    queries = []

    # Always start with core concepts
    for q in QUERY_POOLS["core"]:
        queries.append(q)

    # Mix pools - more novel queries in later sessions
    weights = {
        "core": max(1, 4 - session // 3),
        "followup": 3,
        "complex": 2,
        "personal": 1,
        "novel": min(3, session // 2),  # Increase novelty over time
    }

    pool = []
    for category, weight in weights.items():
        pool.extend(QUERY_POOLS[category] * weight)

    # Fill remaining turns
    rng = random.Random(config.seed + session * 1000)
    while len(queries) < config.turns_per_session:
        queries.append(rng.choice(pool))

    return queries[:config.turns_per_session]


# ═══════════════════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_session(config: PersistenceConfig, session: int, engine: CognitiveChatEngine,
                global_turn_offset: int, save_path: str) -> Tuple[List[PersistenceMetrics], int]:
    """Run a single session and return metrics."""
    queries = generate_session_queries(config, session)
    metrics = []

    print(f"\n{'='*60}")
    print(f"SESSION {session + 1}/{config.n_sessions} ({len(queries)} turns)")
    print(f"{'='*60}")

    for turn_idx, query in enumerate(queries):
        turn_num = turn_idx + 1
        global_turn = global_turn_offset + turn_num

        # Get pre-turn state
        pre_concepts = set()
        for nid in getattr(engine, '_last_activated_ids', []):
            node = engine.graph.nodes.get(nid)
            if node and node.label:
                pre_concepts.add(node.label.lower())

        # Process turn
        t0 = time.time()
        response = engine.process_turn(query)
        latency = (time.time() - t0) * 1000

        # Extract concepts in response
        words = [w.strip('.,!?') for w in response.lower().split() if len(w.strip('.,!?')) >= 3]
        concepts_in_response = [w for w in words if w in engine._concept_keywords]

        m = PersistenceMetrics(
            session=session,
            turn=turn_num,
            global_turn=global_turn,
            query=query,
            response=response[:200],
            concepts_in_response=concepts_in_response,
            unique_concepts=len(set(concepts_in_response)),
            grammar_score=compute_grammar_score(response),
            concept_coherence=compute_concept_coherence(response, engine),
            factual_grounding=compute_factual_grounding(response, engine),
            diversity=compute_diversity(response),
            latency_ms=latency,
            identity_strength=engine.identity.state.strength if hasattr(engine, 'identity') else 0.0,
            user_model_edges=len(engine.user_model.edge_reactivations) if hasattr(engine, 'user_model') else 0,
            sleep_cycles=engine.sleep_cycles_completed if hasattr(engine, 'sleep_cycles_completed') else 0,
            graph_size=len(engine.graph.nodes),
        )
        metrics.append(m)

        # Progress logging
        if config.trace and (turn_num % 200 == 0 or turn_num <= 10):
            print(f"  Turn {turn_num:4d} (global {global_turn:5d}): {query[:40]:40s} | "
                  f"strategy={getattr(engine, '_last_strategy', '?'):20s} | "
                  f"concepts={m.unique_concepts:2d} | id={m.identity_strength:.3f} | "
                  f"sleep={m.sleep_cycles:2d} | graph={m.graph_size:4d}")

        # Checkpoint
        if config.save_between_sessions and turn_num % config.eval_interval == 0:
            engine.save()

    # Final save for session
    if config.save_between_sessions:
        save_data = engine.save()
        if config.trace:
            print(f"  [Session {session} saved]")

    return metrics, global_turn


def analyze_forgetting(all_metrics: List[PersistenceMetrics], probe_concepts: List[str]) -> List[ForgettingMetrics]:
    """Analyze concept forgetting across sessions."""
    concept_stats = {}

    # Track per concept
    for concept in probe_concepts:
        sessions_seen = set()
        total_mentions = 0
        first_session = None
        last_session = None

        for m in all_metrics:
            if concept in m.concepts_in_response:
                sessions_seen.add(m.session)
                total_mentions += 1
                if first_session is None:
                    first_session = m.session
                last_session = m.session

        # Check forgetting: not mentioned in last 3 sessions
        max_session = max(m.session for m in all_metrics) if all_metrics else 0
        sessions_absent = max_session - (last_session if last_session is not None else -1)
        forgotten = sessions_absent >= 3 and total_mentions > 0

        concept_stats[concept] = ForgettingMetrics(
            concept=concept,
            session_first_seen=first_session if first_session is not None else -1,
            session_last_seen=last_session if last_session is not None else -1,
            total_mentions=total_mentions,
            sessions_absent=sessions_absent,
            forgotten=forgotten,
        )

    return list(concept_stats.values())


def run_persistence_experiment(config: PersistenceConfig = None):
    if config is None:
        config = PersistenceConfig()

    np.random.seed(config.seed)
    random.seed(config.seed)

    print("=" * 70)
    print("LONG-TERM PERSISTENCE EXPERIMENT")
    print("=" * 70)
    print(f"Sessions: {config.n_sessions}, Turns/session: {config.turns_per_session}")
    print(f"Total turns: {config.n_sessions * config.turns_per_session}")
    print(f"Probe concepts: {config.probe_concepts}")
    print()

    # Setup save path
    save_dir = Path("experiments/experiment_results/persistence")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"persistence_seed{config.seed}.pkl"

    # Initialize engine
    os.environ['RAVANA_SILENT'] = '1' if not config.trace else '0'
    engine = CognitiveChatEngine(dim=64, seed=config.seed, baby_mode=True)

    # Retrain decoder if needed
    if hasattr(engine, '_seed_corpus_training'):
        engine._seed_corpus_training()

    all_metrics = []
    global_turn = 0

    # Run sessions
    for session in range(config.n_sessions):
        session_metrics, global_turn = run_session(
            config, session, engine, global_turn, str(save_path)
        )
        all_metrics.extend(session_metrics)

    # Analyze forgetting
    print("\n" + "=" * 70)
    print("FORGETTING ANALYSIS")
    print("=" * 70)

    forgetting_metrics = analyze_forgetting(all_metrics, config.probe_concepts)

    for fm in forgetting_metrics:
        status = "FORGOTTEN" if fm.forgotten else "retained"
        print(f"  {fm.concept:20s}: first={fm.session_first_seen:2d} last={fm.session_last_seen:2d} "
              f"mentions={fm.total_mentions:3d} absent={fm.sessions_absent:2d} -> {status}")

    # Overall statistics
    total_forgotten = sum(1 for fm in forgetting_metrics if fm.forgotten)
    print(f"\nTotal concepts: {len(forgetting_metrics)}")
    print(f"Forgotten: {total_forgotten}")
    print(f"Retained: {len(forgetting_metrics) - total_forgotten}")

    # Concept drift analysis: track how responses to same query change
    print("\n" + "=" * 70)
    print("CONCEPT DRIFT ANALYSIS (Same query across sessions)")
    print("=" * 70)

    # Pick a few core queries
    core_queries = ["what is trust", "what is friendship", "what is love"]
    for query in core_queries:
        responses = [m.response for m in all_metrics if m.query == query]
        if responses:
            # Compare first vs last
            print(f"\n  Query: '{query}'")
            print(f"    First:  {responses[0][:100]}")
            print(f"    Last:   {responses[-1][:100]}")

            # Diversity over time
            diversities = [compute_diversity(r) for r in responses]
            gramms = [compute_grammar_score(r) for r in responses]
            print(f"    Diversity trend: {np.mean(diversities[:5]):.3f} -> {np.mean(diversities[-5:]):.3f}")
            print(f"    Grammar trend:   {np.mean(gramms[:5]):.3f} -> {np.mean(gramms[-5:]):.3f}")

    # Catastrophic forgetting check: performance on early concepts after many turns
    print("\n" + "=" * 70)
    print("CATASTROPHIC FORGETTING CHECK")
    print("=" * 70)

    # Test probe concepts at the end
    for concept in config.probe_concepts:
        test_query = f"what is {concept}"
        t0 = time.time()
        response = engine.process_turn(test_query)
        latency = (time.time() - t0) * 1000

        concepts_in_response = [w for w in response.lower().split()
                               if len(w.strip('.,!?')) >= 3 and w in engine._concept_keywords]
        coherence = compute_concept_coherence(response, engine)
        grounding = compute_factual_grounding(response, engine)

        print(f"  {concept:20s}: coherence={coherence:.3f} grounding={grounding:.3f} "
              f"concepts={len(concepts_in_response)} latency={latency:.1f}ms")

    # Summary
    print("\n" + "=" * 70)
    print("PERSISTENCE SUMMARY")
    print("=" * 70)

    total_turns = len(all_metrics)
    avg_grammar = np.mean([m.grammar_score for m in all_metrics])
    avg_coherence = np.mean([m.concept_coherence for m in all_metrics])
    avg_grounding = np.mean([m.factual_grounding for m in all_metrics])
    final_graph_size = all_metrics[-1].graph_size if all_metrics else 0
    final_sleep = all_metrics[-1].sleep_cycles if all_metrics else 0

    print(f"Total turns completed: {total_turns}")
    print(f"Final graph size: {final_graph_size} concepts")
    print(f"Total sleep cycles: {final_sleep}")
    print(f"Avg grammar: {avg_grammar:.3f}")
    print(f"Avg coherence: {avg_coherence:.3f}")
    print(f"Avg grounding: {avg_grounding:.3f}")
    print(f"Concepts forgotten: {total_forgotten}/{len(forgetting_metrics)}")

    # Save detailed results
    if config.output:
        output = {
            'config': asdict(config),
            'turn_metrics': [asdict(m) for m in all_metrics],
            'forgetting_analysis': [asdict(fm) for fm in forgetting_metrics],
            'summary': {
                'total_turns': total_turns,
                'final_graph_size': final_graph_size,
                'total_sleep_cycles': final_sleep,
                'avg_grammar': avg_grammar,
                'avg_coherence': avg_coherence,
                'avg_grounding': avg_grounding,
                'forgotten_count': total_forgotten,
                'total_probe_concepts': len(forgetting_metrics),
            }
        }
        with open(config.output, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {config.output}")

    return all_metrics, forgetting_metrics


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAVANA Long-term Persistence")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sessions", type=int, default=10, help="Number of sessions")
    parser.add_argument("--turns", type=int, default=1000, help="Turns per session")
    parser.add_argument("--no-trace", action="store_true", help="Disable trace output")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()

    config = PersistenceConfig(
        seed=args.seed,
        n_sessions=args.sessions,
        turns_per_session=args.turns,
        trace=not args.no_trace,
        output=args.output,
    )

    run_persistence_experiment(config)


if __name__ == "__main__":
    main()