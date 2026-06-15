#!/usr/bin/env python3
"""
Curiosity Drive Experiment for RAVANA
======================================
Validates the autonomous curiosity drive:
1. Does high PE -> relevant searches?
2. Does contradiction -> resolution search?
3. Novelty-seeking vs exploitation balance
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
class CuriosityConfig:
    n_cycles: int = 30
    delay_seconds: float = 5.0
    dim: int = 64
    seed: int = 42
    trace: bool = True
    output: str = None


# ════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CuriosityCycleMetrics:
    cycle: int
    curiosity_urgency: float
    
    # Signal sources
    pe_signals: int
    contradiction_signals: int
    novelty_signals: int
    serendipity_signals: int
    
    # Selected topics by source
    pe_topics: List[str]
    contradiction_topics: List[str]
    novelty_topics: List[str]
    serendipity_topics: List[str]
    
    # Search quality
    searches_performed: int
    search_relevance: Dict[str, bool]  # topic -> was search relevant
    
    # Balance
    exploitation_ratio: float  # re-selecting known topics
    novelty_ratio: float       # exploring new topics
    
    timestamp: float


# ═══════════════════════════════════════════════════════════════════════════

def analyze_curiosity_signals(engine: CognitiveChatEngine) -> Dict:
    """Analyze what curiosity signals are active."""
    signals = {
        'pe_signals': 0,
        'contradiction_signals': 0,
        'novelty_signals': 0,
        'serendipity_signals': 0,
    }
    
    # High PE concepts
    high_pe = []
    for nid, node in engine.graph.nodes.items():
        if node.label and getattr(node, 'prediction_free_energy', 0) > 0.3:
            high_pe.append(node.label.lower())
            signals['pe_signals'] += 1
    
    # Contradiction pairs
    contradiction_count = 0
    for concept, antonyms in engine._contradiction_map.items():
        contradiction_count += len(antonyms)
    signals['contradiction_signals'] = contradiction_count
    
    # Dormant edges (novelty)
    dormant_count = 0
    if hasattr(engine, '_dormant_edges') and engine._dormant_edges:
        for src, tgt in engine._dormant_edges:
            sn = engine.graph.nodes.get(src)
            tn = engine.graph.nodes.get(tgt)
            if sn and sn.label or tn and tn.label:
                dormant_count += 1
    signals['novelty_signals'] = dormant_count
    
    # High-degree hubs (serendipity)
    if len(engine.graph.nodes) > 0:
        degrees = {}
        for nid in engine.graph.nodes:
            out = len(list(engine.graph.get_outgoing(nid)))
            degrees[nid] = out
        top_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        signals['serendipity_signals'] = len(top_hubs)
    
    return signals


def evaluate_search_relevance(engine: CognitiveChatEngine, topic: str, result: str) -> Dict:
    """Evaluate if a search was relevant to the curiosity signal."""
    # Heuristic: relevant if it learned new concepts
    is_relevant = 'learned' in result.lower() and 'new' in result.lower()
    
    # Check if search topic relates to known curiosity signals
    relates_to_pe = any(topic.lower() in engine._concept_keywords for topic in [topic])
    
    return {
        'topic': topic,
        'result': result,
        'is_relevant': is_relevant,
        'new_concepts': 'learned' in result.lower() and 'new' in result.lower(),
    }


def run_curiosity_experiment(config: CuriosityConfig = None):
    if config is None:
        config = CuriosityConfig()
    
    np.random.seed(config.seed)
    
    print("=" * 70)
    print("CURIOSITY DRIVE EXPERIMENT")
    print("=" * 70)
    print(f"Cycles: {config.n_cycles}, Delay: {config.delay_seconds}s")
    print()
    
    os.environ['RAVANA_SILENT'] = '1' if not config.trace else '0'
    engine = CognitiveChatEngine(dim=config.dim, seed=config.seed, baby_mode=True)
    
    if config.trace:
        engine._trace_enabled = True
    
    print(f"Initial: {len(engine.graph.nodes)} concepts, {len(engine.graph.edges)} edges")
    print()
    
    # Start background learning
    engine.start_background_learning()
    
    # Bootstrap curiosity
    from scripts.ravana_learn import _seed_from_graph_curiosity
    initial_queued = _seed_from_graph_curiosity(engine, max_topics=8)
    print(f"Curiosity-bootstrapped {initial_queued} initial topics")
    print()
    
    all_metrics = []
    topic_history = []
    
    try:
        for cycle in range(1, config.n_cycles + 1):
            print(f"\n{'='*50}")
            print(f"CURIOSITY CYCLE {cycle}/{config.n_cycles}")
            print(f"{'='*50}")
            
            time.sleep(config.delay_seconds)
            
            # Analyze current curiosity signals
            signals = analyze_curiosity_signals(engine)
            
            # Get curiosity urgency
            engine._compute_curiosity_urgency()
            urgency = engine._curiosity_urgency
            
            # Check what was queued this cycle
            with engine._bg_lock:
                queued_topics = list(engine._bg_learning_queue)
                queue_size = len(queued_topics)
            
            # Analyze topic sources
            pe_topics = []
            contradiction_topics = []
            novelty_topics = []
            serendipity_topics = []
            
            # Check high PE
            for nid, node in engine.graph.nodes.items():
                if node.label and getattr(node, 'prediction_free_energy', 0) > 0.3:
                    pe_topics.append(node.label.lower())
            
            # Contradictions
            for concept, antonyms in engine._contradiction_map.items():
                for ant in antonyms:
                    if concept != ant:
                        contradiction_topics.append(f"{concept} vs {ant}")
            
            # Novelty (dormant edges)
            novelty_topics = []
            if hasattr(engine, '_dormant_edges') and engine._dormant_edges:
                dormant_counts = defaultdict(int)
                for src, tgt in engine._dormant_edges:
                    sn = engine.graph.nodes.get(src)
                    tn = engine.graph.nodes.get(tgt)
                    if sn and sn.label:
                        dormant_counts[sn.label.lower()] += 1
                    if tn and tn.label:
                        dormant_counts[tn.label.lower()] += 1
                novelty_topics = sorted(dormant_counts.keys(), key=lambda x: -dormant_counts[x])[:5]
            
            # Serendipity (hub walks)
            serendipity_topics = []
            if len(engine.graph.nodes) > 0:
                degrees = {}
                for nid in engine.graph.nodes:
                    out = len(list(engine.graph.get_outgoing(nid)))
                    degrees[nid] = out
                top_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
                for nid, _ in top_hubs:
                    node = engine.graph.get_node(nid)
                    if node and node.label:
                        serendipity_topics.append(node.label.lower())
            
            # Track topic history for exploitation/novelty ratio
            new_topics = [t for t in queued_topics if t not in topic_history]
            topic_history.extend(new_topics)
            
            novelty_ratio = len(new_topics) / max(1, len(queued_topics))
            exploitation_ratio = 1 - novelty_ratio
            
            # Record search quality (approximate)
            search_relevance = {}
            if engine._bg_search_count > 0:
                for topic in queued_topics[:5]:
                    search_relevance[topic] = topic in engine._concept_keywords
            
            metrics = CuriosityCycleMetrics(
                cycle=cycle,
                curiosity_urgency=urgency,
                pe_signals=signals['pe_signals'],
                contradiction_signals=signals['contradiction_signals'],
                novelty_signals=signals['novelty_signals'],
                serendipity_signals=signals['serendipity_signals'],
                pe_topics=pe_topics[:5],
                contradiction_topics=contradiction_topics[:5],
                novelty_topics=novelty_topics[:5],
                serendipity_topics=serendipity_topics[:5],
                searches_performed=engine._bg_search_count,
                search_relevance=search_relevance,
                exploitation_ratio=exploitation_ratio,
                novelty_ratio=novelty_ratio,
                timestamp=time.time(),
            )
            
            print(f"  Curiosity urgency: {urgency:.3f}")
            print(f"  High PE topics: {pe_topics[:3]}")
            print(f"  Contradictions: {contradiction_topics[:3]}")
            print(f"  Novelty topics: {novelty_topics[:3]}")
            print(f"  Serendipity topics: {serendipity_topics[:3]}")
            print(f"  Queue size: {queue_size}")
            print(f"  Novelty ratio: {novelty_ratio:.2f}")
            print(f"  BG searches: {engine._bg_search_count}")
            
            # Force reselection if queue low
            if queue_size <= 1:
                queued = _seed_from_graph_curiosity(engine, max_topics=4)
                if queued > 0 and config.trace:
                    print(f"  -> Reselected {queued} topics")
            
            all_metrics.append(metrics)
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    finally:
        engine.stop_background_learning()
        result = engine.save()
        print(f"\n{result}")
    
    # Summary
    print("\n" + "=" * 70)
    print("CURIOSITY DRIVE SUMMARY")
    print("=" * 70)
    
    if all_metrics:
        print(f"Cycles completed: {len(all_metrics)}")
        print(f"Total BG searches: {all_metrics[-1].searches_performed}")
        print(f"Final curiosity urgency: {all_metrics[-1].curiosity_urgency:.3f}")
        
        # Signal balance
        avg_pe = np.mean([m.pe_signals for m in all_metrics])
        avg_contradiction = np.mean([m.contradiction_signals for m in all_metrics])
        avg_novelty = np.mean([m.novelty_signals for m in all_metrics])
        avg_serendipity = np.mean([m.serendipity_signals for m in all_metrics])
        
        print(f"\nAverage signal strengths:")
        print(f"  PE signals: {avg_pe:.1f}")
        print(f"  Contradiction signals: {avg_contradiction:.1f}")
        print(f"  Novelty signals: {avg_novelty:.1f}")
        print(f"  Serendipity signals: {avg_serendipity:.1f}")
        
        # Topic diversity
        all_pe_topics = set()
        all_contradiction = set()
        all_novelty = set()
        for m in all_metrics:
            all_pe_topics.update(m.pe_topics)
            all_contradiction.update(m.contradiction_topics)
            all_novelty.update(m.novelty_topics)
        
        print(f"\nUnique topics discovered:")
        print(f"  High PE: {len(all_pe_topics)}")
        print(f"  Contradictions: {len(all_contradiction)}")
        print(f"  Novelty: {len(all_novelty)}")
        
        # Novelty vs exploitation
        avg_novelty_ratio = np.mean([m.novelty_ratio for m in all_metrics])
        print(f"\nAverage novelty ratio: {avg_novelty_ratio:.2f}")
        print(f"Average exploitation ratio: {1-avg_novelty_ratio:.2f}")
        
        # Search relevance
        relevant_searches = sum(1 for m in all_metrics 
                               for topic, rel in m.search_relevance.items() if rel)
        total_searches = sum(len(m.search_relevance) for m in all_metrics)
        if total_searches > 0:
            print(f"Search relevance: {relevant_searches}/{total_searches} = {relevant_searches/total_searches:.2f}")
    
    # Save
    if config.output:
        output = {
            'config': config.__dict__,
            'metrics': [m.__dict__ for m in all_metrics],
        }
        with open(config.output, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {config.output}")
    
    return all_metrics


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CuriosityConfig:
    n_cycles: int = 30
    delay_seconds: float = 5.0
    dim: int = 64
    seed: int = 42
    trace: bool = True
    output: str = None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Curiosity Drive Experiment")
    parser.add_argument("--cycles", type=int, default=30, help="Number of curiosity cycles")
    parser.add_argument("--delay", type=float, default=5.0, help="Seconds between cycles")
    parser.add_argument("--dim", type=int, default=64, help="Graph dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-trace", action="store_true", help="Disable trace output")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()
    
    config = CuriosityConfig(
        n_cycles=args.cycles,
        delay_seconds=args.delay,
        dim=args.dim,
        seed=args.seed,
        trace=not args.no_trace,
        output=args.output,
    )
    
    run_curiosity_experiment(config)


if __name__ == "__main__":
    main()