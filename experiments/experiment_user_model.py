#!/usr/bin/env python3
"""
User Model Experiment for RAVANA
=================================
Tests user adaptation and personalization:
1. Personalization after N interactions
2. Edge reactivation -> response bias
3. Follow-up coherence
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


# User Personas
USER_PERSONAS = {
    "curious": {
        "queries": [
            "what is trust", "what is friendship", "how does trust work",
            "why do people betray", "what makes love last", "explain empathy",
            "what is courage", "how to build confidence", "what is wisdom",
            "why do we need friends", "what makes us happy", "explain love",
        ],
        "style": "inquisitive",
        "followup_rate": 0.7,
    },
    "skeptical": {
        "queries": [
            "is trust real", "can friendship be trusted", "why trust anyone",
            "is love just chemicals", "why do people lie", "can you prove it",
            "is courage just stupidity", "why believe in hope", "what is truth",
            "does loyalty exist", "is wisdom a myth", "why care about others",
        ],
        "style": "challenging",
        "followup_rate": 0.5,
    },
    "practical": {
        "queries": [
            "how to build trust", "steps to make friends", "ways to show love",
            "how to overcome fear", "how to be brave", "how to be a good friend",
            "how to gain confidence", "how to learn faster", "how to be wise",
            "how to keep promises", "how to help others", "how to communicate",
        ],
        "style": "action-oriented",
        "followup_rate": 0.4,
    },
}


@dataclass
class InteractionMetrics:
    interaction: int
    user: str
    query: str
    response: str
    strategy: str
    edge_boost_used: bool
    activation_boost_keys: List[str]
    user_model_size: int
    edge_reactivations: int
    concepts_in_response: List[str]
    unique_concepts: int
    response_length: int
    is_followup: bool
    followup_coherence: float
    topic_consistency: float
    identity_strength: float
    preferred_edges: List[Tuple[str, str, int]]
    timestamp: float


def generate_user_session(persona: Dict, n_interactions: int) -> List[str]:
    queries = []
    queries.extend(persona["queries"][:10])
    followup_templates = [
        "tell me more", "what else", "why is that",
        "how does that work", "give me an example",
        "what do you think", "can you explain",
    ]
    while len(queries) < n_interactions:
        if np.random.random() < persona["followup_rate"] and len(queries) > 1:
            queries.append(np.random.choice(followup_templates))
        else:
            queries.append(np.random.choice(persona["queries"]))
    return queries[:n_interactions]


def extract_concepts_from_response(response: str, engine: CognitiveChatEngine) -> List[str]:
    words = [w.strip('.,!?') for w in response.lower().split() if len(w.strip('.,!?')) >= 3]
    return [w for w in words if w in engine._concept_keywords]


def compute_followup_coherence(prev_query: str, curr_query: str,
                                prev_response: str, curr_response: str) -> float:
    if not prev_query or not curr_query:
        return 0.0
    prev_words = set(prev_query.lower().split() + prev_response.lower().split())
    curr_words = set(curr_query.lower().split() + curr_response.lower().split())
    overlap = len(prev_words & curr_words)
    total = len(prev_words | curr_words)
    return overlap / max(1, total)


def compute_topic_consistency(response: str, engine: CognitiveChatEngine) -> float:
    words = [w for w in response.lower().split() if len(w) >= 3]
    if not words:
        return 0.0
    in_model = sum(1 for w in words if w in engine.user_model.query_concepts)
    return in_model / len(words)


@dataclass
class UserModelConfig:
    n_interactions: int = 200
    n_users: int = 3
    seed: int = 42
    trace: bool = True
    output: str = None


def run_user_model_experiment(config: UserModelConfig = None):
    if config is None:
        config = UserModelConfig()
    
    np.random.seed(config.seed)
    
    print("=" * 70)
    print("USER MODEL EXPERIMENT")
    print("=" * 70)
    print(f"Interactions: {config.n_interactions}")
    print(f"Users: {list(USER_PERSONAS.keys())}")
    print()
    
    all_user_results = {}
    
    for user_name, persona in USER_PERSONAS.items():
        print(f"\n{'='*70}")
        print(f"USER: {user_name} ({persona['style']})")
        print(f"{'='*70}")
        
        os.environ['RAVANA_SILENT'] = '1'
        engine = CognitiveChatEngine(dim=64, seed=config.seed, baby_mode=True)
        engine.start_background_learning()
        
        if config.trace:
            engine._trace_enabled = True
        
        print(f"Initial: {len(engine.graph.nodes)} concepts, {len(engine.graph.edges)} edges")
        
        query_sequence = generate_user_session(persona, config.n_interactions)
        
        user_metrics = []
        prev_query = ""
        prev_response = ""
        
        print("\nStarting interactions...")
        
        for i, query in enumerate(query_sequence):
            turn_num = i + 1
            is_followup = i > 0 and any(w in query.lower() for w in 
                ["more", "else", "another", "also", "further", "why", "how", "what else"])
            
            response = engine.process_turn(query)
            
            edge_boost_used = engine._activation_boost is not None and len(engine._activation_boost) > 0
            activation_boost_keys = list(engine._activation_boost.keys()) if engine._activation_boost else []
            
            user_model_size = len(engine.user_model.edge_reactivations)
            edge_reactivations = sum(engine.user_model.edge_reactivations.values())
            
            concepts_in_response = extract_concepts_from_response(response, engine)
            unique_concepts = len(set(concepts_in_response))
            
            followup_coherence = 0.0
            topic_consistency = 0.0
            if is_followup:
                followup_coherence = compute_followup_coherence(
                    prev_query, query, prev_response, response)
                topic_consistency = compute_topic_consistency(response, engine)
            
            top_edges = sorted(engine.user_model.edge_reactivations.items(), 
                              key=lambda x: -x[1])[:5]
            preferred = [(f, t, c) for (f, t), c in top_edges]
            
            metrics = InteractionMetrics(
                interaction=turn_num,
                user=user_name,
                query=query,
                response=response[:150],
                strategy=engine._last_strategy,
                edge_boost_used=edge_boost_used,
                activation_boost_keys=activation_boost_keys,
                user_model_size=user_model_size,
                edge_reactivations=edge_reactivations,
                concepts_in_response=concepts_in_response,
                unique_concepts=len(set(concepts_in_response)),
                response_length=len(response),
                is_followup=is_followup,
                followup_coherence=followup_coherence,
                topic_consistency=topic_consistency,
                identity_strength=engine.identity.state.strength,
                preferred_edges=preferred,
                timestamp=time.time(),
            )
            
            user_metrics.append(metrics)
            
            if config.trace and (turn_num % 25 == 0 or turn_num <= 5):
                print(f"  {turn_num:3d}: {query[:50]:50s} | "
                      f"strat={metrics.strategy:15s} | "
                      f"concepts={metrics.unique_concepts:2d} | "
                      f"followup={is_followup} | "
                      f"coh={metrics.followup_coherence:.2f} | "
                      f"topic_cons={metrics.topic_consistency:.2f} | "
                      f"edge_boost={edge_boost_used} | "
                      f"user_model={metrics.user_model_size:3d} | "
                      f"id={metrics.identity_strength:.3f}")
            
            prev_query = query
            prev_response = response
        
        print(f"\n  Analysis for {user_name}:")
        user_model_growth = [m.user_model_size for m in user_metrics]
        print(f"  User model growth: {user_model_growth[0]} -> {user_model_growth[-1]}")
        
        followup_metrics = [m for m in user_metrics if m.is_followup]
        if followup_metrics:
            avg_coherence = np.mean([m.followup_coherence for m in followup_metrics])
            avg_topic_consistency = np.mean([m.topic_consistency for m in followup_metrics])
            print(f"  Follow-up coherence: {avg_coherence:.3f}")
            print(f"  Topic consistency: {avg_topic_consistency:.3f}")
        else:
            print(f"  No follow-ups detected")
        
        edge_boost_count = sum(1 for m in user_metrics if m.edge_boost_used)
        print(f"  Edge boost used: {edge_boost_count}/{len(user_metrics)}")
        
        if user_metrics:
            top = user_metrics[-1].preferred_edges
            print(f"  Top preferred edges: {top[:3]}")
        
        all_user_results[user_name] = user_metrics
        
        engine.stop_background_learning()
        engine.save()
    
    # Cross-user comparison
    print("\n" + "=" * 70)
    print("CROSS-USER COMPARISON")
    print("=" * 70)
    
    for user_name in USER_PERSONAS:
        metrics = all_user_results[user_name]
        if not metrics:
            continue
        
        print(f"\n{user_name}:")
        print(f"  Model size: {metrics[-1].user_model_size}")
        print(f"  Edge reactivations: {metrics[-1].edge_reactivations}")
        print(f"  Identity: {metrics[-1].identity_strength:.3f}")
        
        followups = [m for m in metrics if m.is_followup]
        if followups:
            print(f"  Follow-up coherence: {np.mean([m.followup_coherence for m in followups]):.3f}")
        
        print(f"  Identity final: {metrics[-1].identity_strength:.3f}")
        
        all_preferred = set()
        for m in metrics:
            for f, t, c in m.preferred_edges:
                all_preferred.add((f, t))
        print(f"  Unique preferred edges: {len(all_preferred)}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nPersonalization effectiveness:")
    for user_name in USER_PERSONAS:
        m = all_user_results[user_name]
        if m:
            initial_model = m[0].user_model_size
            final_model = m[-1].user_model_size
            print(f"  {user_name}: {initial_model} -> {final_model} "
                  f"(+{final_model - initial_model} edges)")
    
    if config.output:
        output = {
            'config': config.__dict__,
            'results': {user: [m.__dict__ for m in metrics] 
                       for user, metrics in all_user_results.items()},
        }
        with open(config.output, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {config.output}")
    
    return all_user_results


@dataclass
class UserModelConfig:
    n_interactions: int = 200
    n_users: int = 3
    seed: int = 42
    trace: bool = True
    output: str = None


# CLI
def main():
    import argparse
    parser = argparse.ArgumentParser(description="User Model Experiment")
    parser.add_argument("--interactions", type=int, default=200, help="Interactions per user")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-trace", action="store_true", help="Disable trace output")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()
    
    config = UserModelConfig(
        n_interactions=args.interactions,
        seed=args.seed,
        trace=not args.no_trace,
        output=args.output,
    )
    
    run_user_model_experiment(config)


if __name__ == "__main__":
    main()