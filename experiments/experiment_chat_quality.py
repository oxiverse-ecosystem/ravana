#!/usr/bin/env python3
"""
Chat Quality Experiment for RAVANA CognitiveChatEngine - Optimized
====================================================================
Tests response quality under different training conditions.
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

import random
random.seed(42)
np.random.seed(42)

from scripts.ravana_chat import CognitiveChatEngine


# ════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ChatExperimentConfig:
    test_queries: List[str] = field(default_factory=lambda: [
        "what is trust",
        "what is friendship", 
        "how does trust work",
        "create a blueprint for trust",
    ])
    
    scenarios: List[Dict] = field(default_factory=lambda: [
        {"name": "baseline_no_web", "web_training": False, "seed_corpus": True, "synthetic": True},
        {"name": "seed_only", "web_training": False, "seed_corpus": True, "synthetic": False},
        {"name": "seed_plus_synthetic", "web_training": False, "seed_corpus": True, "synthetic": True},
        {"name": "web_10", "web_training": True, "seed_corpus": True, "synthetic": True, "web_articles": 10},
        {"name": "web_50", "web_training": True, "seed_corpus": True, "synthetic": True, "web_articles": 50},
    ])
    
    n_runs: int = 2
    seed: int = 42


# ════════════════════════════════════════════════════════════════════════════
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


def compute_concept_coherence(response: str, engine: CognitiveChatEngine) -> float:
    if not response: return 0.0
    words = [w.strip('.,!?') for w in response.lower().split() if len(w.strip('.,!?')) >= 3]
    if not words: return 0.0
    known = sum(1 for w in words if w in engine._concept_keywords)
    return known / len(words)


def compute_factual_grounding(response: str, engine: CognitiveChatEngine) -> float:
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


def compute_all_metrics(response: str, engine: CognitiveChatEngine) -> Dict[str, float]:
    return {
        'grammar': compute_grammar_score(response),
        'concept_coherence': compute_concept_coherence(response, engine),
        'factual_grounding': compute_factual_grounding(response, engine),
        'diversity': compute_diversity(response),
    }


# ════════════════════════════════════════════════════════════════════════════
# Engine Builder (Optimized - builds once per scenario)
# ════════════════════════════════════════════════════════════════════════════

DUMMY_ARTICLES = [
    "Trust is the foundation of all relationships. Trust means relying on someone. Trust takes time to build but can be broken quickly.",
    "Friendship is a bond between people. Friends support each other. True friendship lasts through good and bad times.",
    "Love is a deep affection. Love involves care and commitment. Love can be romantic or platonic.",
    "Neural networks are computing systems inspired by brains. They learn from data through layers of neurons.",
    "Quantum mechanics describes nature at small scales. Particles can be waves and particles at the same time.",
]


def build_engine_for_scenario(scenario: Dict) -> CognitiveChatEngine:
    os.environ['RAVANA_SILENT'] = '1'
    engine = CognitiveChatEngine(dim=64, seed=42, baby_mode=True)
    
    if not scenario.get('web_training', False):
        engine._network_available = False
    
    # Apply scenario-specific decoder training state
    if not scenario.get('seed_corpus', True):
        engine._decoder_seed_training_count = 0
    if not scenario.get('synthetic', True):
        engine._decoder_training_count = engine._decoder_web_training_count
    
    # Add web training
    web_articles = scenario.get('web_articles', 0)
    if web_articles > 0 and scenario.get('web_training', False):
        if engine.neural_decoder and engine._decoder_vocab_built:
            for i in range(min(web_articles, len(DUMMY_ARTICLES))):
                article = DUMMY_ARTICLES[i]
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


# ════════════════════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_chat_experiment(config: ChatExperimentConfig = None):
    if config is None:
        config = ChatExperimentConfig()
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    print("=" * 70)
    print("RAVANA Chat Quality Experiment")
    print("=" * 70)
    print(f"Queries: {len(config.test_queries)}, Scenarios: {len(config.scenarios)}, Runs: {config.n_runs}")
    print()
    
    all_results = {}
    
    for scenario in config.scenarios:
        name = scenario['name']
        print(f"\n{'='*70}")
        print(f"SCENARIO: {name}")
        print(f"{'='*70}")
        
        # Build engine ONCE per scenario
        engine = build_engine_for_scenario(scenario)
        print(f"  Decoder: {engine._decoder_training_count} total ({engine._decoder_web_training_count} web, {engine._decoder_seed_training_count} seed)")
        
        scenario_results = []
        
        for query in config.test_queries:
            print(f"\n  Query: '{query}'")
            query_results = []
            
            for run in range(config.n_runs):
                try:
                    response = engine.process_turn(query)
                    metrics = compute_all_metrics(response, engine)
                    metrics['response'] = response[:100]
                    metrics['run'] = run
                    query_results.append(metrics)
                    
                    print(f"    Run {run+1}: {response[:80]}...")
                    print(f"      grammar={metrics['grammar']:.2f}, coherence={metrics['concept_coherence']:.2f}, grounding={metrics['factual_grounding']:.2f}, diversity={metrics['diversity']:.2f}")
                    
                except Exception as e:
                    print(f"    Run {run+1}: ERROR - {e}")
                    query_results.append({'error': str(e)})
            
            scenario_results.append({
                'query': query,
                'runs': query_results
            })
        
        all_results[name] = scenario_results
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for scenario_name, results in all_results.items():
        print(f"\n{scenario_name}:")
        
        all_metrics = {'grammar': [], 'concept_coherence': [], 'factual_grounding': [], 'diversity': []}
        
        for query_result in results:
            for run in query_result['runs']:
                if 'error' not in run:
                    for k in all_metrics:
                        all_metrics[k].append(run[k])
        
        if all_metrics['grammar']:
            for k in all_metrics:
                vals = all_metrics[k]
                print(f"  {k:20s}: {np.mean(vals):.3f} ± {np.std(vals):.3f} (n={len(vals)})")
    
    return all_results


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAVANA Chat Quality Experiment")
    parser.add_argument("--quick", action="store_true", help="Quick run (1 run, 2 queries)")
    parser.add_argument("--scenario", type=str, help="Run single scenario by name")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()
    
    config = ChatExperimentConfig()
    
    if args.quick:
        config.test_queries = config.test_queries[:2]
        config.n_runs = 1
        config.scenarios = config.scenarios[:3]  # baseline, seed_only, seed+synthetic
    
    if args.scenario:
        config.scenarios = [s for s in config.scenarios if s['name'] == args.scenario]
        if not config.scenarios:
            print(f"Unknown scenario: {args.scenario}")
            return
    
    results = run_chat_experiment(config)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()