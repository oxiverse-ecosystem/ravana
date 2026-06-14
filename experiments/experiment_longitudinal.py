#!/usr/bin/env python3
"""
Longitudinal Chat Experiment for RAVANA
========================================
Runs extended chat sessions (500+ turns) to measure:
1. Concept drift / stability over time
2. Forgetting curve (Ebbinghaus)
3. User model personalization quality
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.ravana_chat import CognitiveChatEngine


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LongitudinalConfig:
    n_turns: int = 500
    save_interval: int = 50
    seed: int = 42
    trace: bool = True
    output: str = None
    # Query patterns to simulate realistic conversation
    query_sets: Dict[str, List[str]] = field(default_factory=lambda: {
        "core": [
            "what is trust", "what is friendship", "what is love",
            "what is fear", "what is hope", "what is courage",
        ],
        "followup": [
            "tell me more", "what else", "why is that",
            "how does that work", "give me an example",
        ],
        "complex": [
            "how does trust work in relationships",
            "create a blueprint for building trust",
            "why do people betray each other",
            "what makes a friendship last",
        ],
        "personal": [
            "do you have friends", "what do you think about betrayal",
            "have you ever been hurt", "what matters most to you",
        ],
    })


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TurnMetrics:
    turn: int
    query: str
    response: str
    strategy: str
    concepts_activated: List[str]
    concepts_in_response: List[str]
    response_length: int
    unique_concepts: int
    repetition_score: float
    user_model_size: int
    edge_reactivations: int
    identity_strength: float
    sleep_cycles: int
    timestamp: float


# ═══════════════════════════════════════════════════════════════════════════

def generate_query_sequence(config: LongitudinalConfig, n_turns: int) -> List[str]:
    """Generate a realistic query sequence."""
    queries = []
    
    # Start with core concepts
    for q in config.query_sets["core"]:
        queries.append(q)
    
    # Mix of patterns
    query_pool = (
        config.query_sets["core"] * 2 +
        config.query_sets["followup"] * 3 +
        config.query_sets["complex"] * 2 +
        config.query_sets["personal"] * 1
    )
    
    # Add variety
    while len(queries) < n_turns:
        queries.append(np.random.choice(query_pool))
    
    return queries[:n_turns]


def extract_concepts_from_response(response: str, engine: CognitiveChatEngine) -> List[str]:
    """Extract graph concepts mentioned in response."""
    words = [w.strip('.,!?') for w in response.lower().split() if len(w.strip('.,!?')) >= 3]
    return [w for w in words if w in engine._concept_keywords]


def compute_repetition_score(responses: List[str], window: int = 10) -> float:
    """Compute repetition in recent responses."""
    if len(responses) < 2:
        return 0.0
    recent = " ".join(responses[-window:]).lower().split()
    if not recent:
        return 0.0
    counts = Counter(recent)
    max_count = max(counts.values())
    return max_count / len(recent)


def run_longitudinal_experiment(config: LongitudinalConfig = None):
    if config is None:
        config = LongitudinalConfig()
    
    np.random.seed(config.seed)
    
    print("=" * 70)
    print("LONGITUDINAL CHAT EXPERIMENT")
    print("=" * 70)
    print(f"Turns: {config.n_turns}")
    print()
    
    # Create engine with pre-trained weights if available
    os.environ['RAVANA_SILENT'] = '1' if not config.trace else '0'
    engine = CognitiveChatEngine(dim=64, seed=config.seed, baby_mode=True)
    
    if config.trace:
        engine._trace_enabled = True
    
    print(f"Initial: {len(engine.graph.nodes)} concepts, {len(engine.graph.edges)} edges")
    print(f"Decoder: {engine._decoder_web_training_count} web sentences")
    print()
    
    # Generate query sequence
    query_sequence = generate_query_sequence(config, config.n_turns)
    
    all_metrics = []
    responses_history = []
    
    # Pre-load some concepts to avoid cold start
    for q in config.query_sets["core"][:3]:
        engine.process_turn(q)
    
    print("Starting longitudinal session...\n")
    
    for turn_idx, query in enumerate(query_sequence):
        turn_num = turn_idx + 1
        
        # Get pre-turn state
        pre_concepts = set()
        for nid in engine._last_activated_ids:
            node = engine.graph.nodes.get(nid)
            if node and node.label:
                pre_concepts.add(node.label.lower())
        
        # Process turn
        t0 = time.time()
        response = engine.process_turn(query)
        elapsed = time.time() - t0
        
        # Extract metrics
        concepts_in_response = extract_concepts_from_response(response, engine)
        user_model_size = len(engine.user_model.edge_reactivations)
        edge_reactivations = sum(engine.user_model.edge_reactivations.values())
        
        metrics = TurnMetrics(
            turn=turn_num,
            query=query,
            response=response[:200],
            strategy=engine._last_strategy,
            concepts_activated=list(pre_concepts),
            concepts_in_response=concepts_in_response,
            response_length=len(response),
            unique_concepts=len(set(concepts_in_response)),
            repetition_score=compute_repetition_score(responses_history + [response]),
            user_model_size=user_model_size,
            edge_reactivations=edge_reactivations,
            identity_strength=engine.identity.state.strength,
            sleep_cycles=engine.sleep_cycles_completed,
            timestamp=time.time(),
        )
        all_metrics.append(metrics)
        responses_history.append(response)
        
        # Progress
        if config.trace and (turn_num % 25 == 0 or turn_num <= 10):
            print(f"Turn {turn_num:4d}: {query[:50]:50s} | "
                  f"strategy={metrics.strategy:20s} | "
                  f"concepts={len(metrics.concepts_in_response):2d} | "
                  f"rep={metrics.repetition_score:.2f} | "
                  f"id={metrics.identity_strength:.3f} | "
                  f"user_model={metrics.user_model_size:4d} | "
                  f"time={elapsed:.2f}s")
        
        # Checkpoint
        if turn_num % config.save_interval == 0:
            engine.save()
            print(f"  [Checkpoint] Turn {turn_num} saved")
    
    # Final save
    result = engine.save()
    print(f"\n{result}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("LONGITUDINAL ANALYSIS")
    print("=" * 70)
    
    # 1. Concept drift / stability
    print("\n1. CONCEPT STABILITY")
    concept_first_seen = {}
    concept_last_seen = {}
    concept_counts = Counter()
    
    for m in all_metrics:
        for c in m.concepts_in_response:
            if c not in concept_first_seen:
                concept_first_seen[c] = m.turn
            concept_last_seen[c] = m.turn
            concept_counts[c] += 1
    
    # Concepts that appeared then disappeared (forgotten)
    current_turn = all_metrics[-1].turn
    forgotten = [c for c, last in concept_last_seen.items() if current_turn - last > 50]
    stable = [c for c, count in concept_counts.items() if count > 20]
    
    print(f"  Total unique concepts in responses: {len(concept_first_seen)}")
    print(f"  Stable concepts (>20 occurrences): {len(stable)}")
    print(f"  Forgotten concepts (>50 turns absent): {len(forgotten)}")
    if forgotten:
        print(f"  Forgotten: {forgotten[:10]}")
    
    # 2. Forgetting curve (Ebbinghaus-style)
    print("\n2. FORGETTING CURVE ANALYSIS")
    # Track how often concepts reappear after gaps
    for concept in list(concept_first_seen.keys())[:20]:
        appearances = [m.turn for m in all_metrics if concept in m.concepts_in_response]
        if len(appearances) >= 3:
            gaps = [appearances[i+1] - appearances[i] for i in range(len(appearances)-1)]
            avg_gap = np.mean(gaps)
            print(f"  {concept}: {len(appearances)} appearances, avg gap={avg_gap:.1f} turns")
    
    # 3. User model personalization
    print("\n3. USER MODEL PERSONALIZATION")
    print(f"  Edge reactivations tracked: {engine.user_model.edge_reactivations}")
    print(f"  Total reactivations: {sum(engine.user_model.edge_reactivations.values())}")
    print(f"  Unique edges visited: {len(engine.user_model.edge_reactivations)}")
    
    # Top preferred edges
    top_edges = sorted(engine.user_model.edge_reactivations.items(), key=lambda x: -x[1])[:10]
    print(f"  Top preferred edges:")
    for (frm, to), count in top_edges:
        print(f"    {frm} -> {to}: {count} visits")
    
    # 4. Strategy evolution
    print("\n4. STRATEGY EVOLUTION")
    strategy_counts = Counter(m.strategy for m in all_metrics)
    for strategy, count in strategy_counts.most_common():
        print(f"  {strategy}: {count} ({count/len(all_metrics)*100:.1f}%)")
    
    # 5. Identity & coherence
    print("\n5. IDENTITY & COHERENCE")
    identity_over_time = [(m.turn, m.identity_strength) for m in all_metrics]
    print(f"  Identity range: {min(v for _,v in identity_over_time):.3f} - {max(v for _,v in identity_over_time):.3f}")
    print(f"  Final identity: {identity_over_time[-1][1]:.3f}")
    
    # Repetition over time
    rep_over_time = [(m.turn, m.repetition_score) for m in all_metrics]
    avg_rep_early = np.mean([v for t,v in rep_over_time if t < 100])
    avg_rep_late = np.mean([v for t,v in rep_over_time if t > config.n_turns - 100])
    print(f"  Avg repetition (first 100): {avg_rep_early:.3f}")
    print(f"  Avg repetition (last 100): {avg_rep_late:.3f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total turns: {len(all_metrics)}")
    print(f"Unique concepts used: {len(concept_first_seen)}")
    print(f"Stable concepts: {len(stable)}")
    print(f"Forgotten concepts: {len(forgotten)}")
    print(f"User model edges: {len(engine.user_model.edge_reactivations)}")
    print(f"Identity strength: {all_metrics[-1].identity_strength:.3f}")
    print(f"Sleep cycles: {all_metrics[-1].sleep_cycles}")
    
    # Save
    if config.output:
        output = {
            'config': asdict(config),
            'metrics': [asdict(m) for m in all_metrics],
            'analysis': {
                'unique_concepts': len(concept_first_seen),
                'stable_concepts': len(stable),
                'forgotten_concepts': len(forgotten),
                'forgotten_list': forgotten[:20],
                'strategy_distribution': dict(strategy_counts),
            }
        }
        with open(config.output, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {config.output}")
    
    return all_metrics


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Longitudinal Chat Experiment")
    parser.add_argument("--turns", type=int, default=500, help="Number of conversation turns")
    parser.add_argument("--save-interval", type=int, default=50, help="Save every N turns")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-trace", action="store_true", help="Disable trace output")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()
    
    config = LongitudinalConfig(
        n_turns=args.turns,
        save_interval=args.save_interval,
        seed=args.seed,
        trace=not args.no_trace,
        output=args.output,
    )
    
    run_longitudinal_experiment(config)


if __name__ == "__main__":
    main()