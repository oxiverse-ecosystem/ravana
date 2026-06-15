#!/usr/bin/env python3
"""
Background Learner Experiment for RAVANA
=========================================
Validates the autonomous curiosity-driven background learning system.
Measures:
1. Curiosity signal quality (PE/contradiction → relevant searches)
2. Knowledge accumulation rate (concepts, edges per cycle)
3. Decoder improvement from autonomous web training
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.ravana_chat import CognitiveChatEngine


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BackgroundLearnerConfig:
    n_cycles: int = 20
    delay_seconds: float = 10.0
    dim: int = 64
    seed: int = 42
    trace: bool = True
    output: str = None


# ═══════════════════════════════════════════════════════════════════════════
# Metrics Collection
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CycleMetrics:
    cycle: int
    concepts: int
    edges: int
    new_concepts: int
    new_edges: int
    web_searches: int
    decoder_total: int
    decoder_web: int
    curiosity_urgency: float
    curiosity_queued: int
    queue_size: int
    high_pe_topics: List[str]
    contradiction_topics: List[str]
    novelty_topics: List[str]
    search_quality: Dict[str, Any]  # topic -> {source, new_concepts, confidence}
    timestamp: float


def analyze_search_quality(engine: CognitiveChatEngine, topic: str, result: str) -> Dict:
    """Analyze quality of a single web search result."""
    return {
        'topic': topic,
        'result': result,
        'has_new_concepts': 'learned' in result.lower() and 'new' in result.lower(),
        'is_offline': 'offline' in result.lower(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_background_learner_experiment(config: BackgroundLearnerConfig = None):
    if config is None:
        config = BackgroundLearnerConfig()
    
    np.random.seed(config.seed)
    
    print("=" * 70)
    print("BACKGROUND LEARNER EXPERIMENT")
    print("=" * 70)
    print(f"Cycles: {config.n_cycles}, Delay: {config.delay_seconds}s")
    print()
    
    # Create engine with background learning enabled
    os.environ['RAVANA_SILENT'] = '1' if not config.trace else '0'
    engine = CognitiveChatEngine(dim=config.dim, seed=config.seed, baby_mode=True)
    
    if config.trace:
        engine._trace_enabled = True
    
    print(f"Initial state:")
    print(f"  Concepts: {len(engine.graph.nodes)}")
    print(f"  Edges: {len(engine.graph.edges)}")
    print(f"  Decoder web training: {engine._decoder_web_training_count}")
    print()
    
    # Start background learning
    engine.start_background_learning()
    
    # Bootstrap curiosity topics
    from scripts.ravana_learn import _seed_from_graph_curiosity
    initial_queued = _seed_from_graph_curiosity(engine, max_topics=8)
    print(f"Curiosity-bootstrapped {initial_queued} initial topics")
    print()
    
    all_metrics = []
    last_concepts = len(engine.graph.nodes)
    last_edges = len(engine.graph.edges)
    
    try:
        for cycle in range(1, config.n_cycles + 1):
            print(f"\n{'='*50}")
            print(f"CYCLE {cycle}/{config.n_cycles}")
            print(f"{'='*50}")
            
            time.sleep(config.delay_seconds)
            
            # Collect metrics
            current_concepts = len(engine.graph.nodes)
            current_edges = len(engine.graph.edges)
            new_concepts = current_concepts - last_concepts
            new_edges = current_edges - last_edges
            
            # Get curiosity state
            with engine._bg_lock:
                queue_size = len(engine._bg_learning_queue)
            
            # Analyze curiosity signal sources
            high_pe = []
            for nid, node in engine.graph.nodes.items():
                if node.label and getattr(node, 'prediction_free_energy', 0) > 0.3:
                    high_pe.append(node.label.lower())
            
            contradiction_topics = list(engine._contradiction_map.keys())[:5]
            
            dormant_counts = defaultdict(int)
            if hasattr(engine, '_dormant_edges') and engine._dormant_edges:
                for src, tgt in engine._dormant_edges:
                    sn = engine.graph.nodes.get(src)
                    tn = engine.graph.nodes.get(tgt)
                    if sn and sn.label:
                        dormant_counts[sn.label.lower()] += 1
                    if tn and tn.label:
                        dormant_counts[tn.label.lower()] += 1
            novelty_topics = sorted(dormant_counts.keys(), key=lambda x: -dormant_counts[x])[:5]
            
            # Record search quality if any searches occurred
            search_quality = {}
            if engine._bg_search_count > 0 and config.trace:
                # Can't easily get individual search results, but we can note
                search_quality['total_searches'] = engine._bg_search_count
            
            metrics = CycleMetrics(
                cycle=cycle,
                concepts=current_concepts,
                edges=current_edges,
                new_concepts=new_concepts,
                new_edges=new_edges,
                web_searches=engine._bg_search_count,
                decoder_total=engine._decoder_training_count,
                decoder_web=engine._decoder_web_training_count,
                curiosity_urgency=engine._curiosity_urgency,
                curiosity_queued=0,  # would need to track from _seed_from_graph_curiosity
                queue_size=queue_size,
                high_pe_topics=high_pe[:5],
                contradiction_topics=contradiction_topics,
                novelty_topics=novelty_topics,
                search_quality=search_quality,
                timestamp=time.time(),
            )
            all_metrics.append(metrics)
            
            print(f"  Concepts: {current_concepts} (+{new_concepts})")
            print(f"  Edges: {current_edges} (+{new_edges})")
            print(f"  Web searches: {engine._bg_search_count}")
            print(f"  Decoder: {engine._decoder_training_count} total ({engine._decoder_web_training_count} web)")
            print(f"  Curiosity urgency: {engine._curiosity_urgency:.3f}")
            print(f"  Queue size: {queue_size}")
            print(f"  High PE topics: {high_pe[:3]}")
            print(f"  Contradictions: {contradiction_topics[:3]}")
            print(f"  Novelty topics: {novelty_topics[:3]}")
            
            last_concepts = current_concepts
            last_edges = current_edges
            
            # Periodically force curiosity reselection if queue low
            if queue_size <= 1:
                queued = _seed_from_graph_curiosity(engine, max_topics=4)
                if queued > 0 and config.trace:
                    print(f"  -> Curiosity reselected {queued} topics")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        engine.stop_background_learning()
        result = engine.save()
        print(f"\n{result}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if all_metrics:
        total_concepts = all_metrics[-1].concepts - all_metrics[0].concepts
        total_edges = all_metrics[-1].edges - all_metrics[0].edges
        total_searches = all_metrics[-1].web_searches
        decoder_improvement = all_metrics[-1].decoder_web - all_metrics[0].decoder_web
        
        print(f"Cycles completed: {len(all_metrics)}")
        print(f"Concepts gained: {total_concepts}")
        print(f"Edges gained: {total_edges}")
        print(f"Total web searches: {total_searches}")
        print(f"Decoder web training improvement: +{decoder_improvement} sentences")
        
        # Rates
        if config.n_cycles > 0:
            print(f"\nRates per cycle:")
            print(f"  Concepts/cycle: {total_concepts / len(all_metrics):.2f}")
            print(f"  Edges/cycle: {total_edges / len(all_metrics):.2f}")
            print(f"  Searches/cycle: {total_searches / len(all_metrics):.2f}")
            print(f"  Decoder web training/cycle: {decoder_improvement / len(all_metrics):.1f}")
        
        # Curiosity signal analysis
        print(f"\nCuriosity signal sources (final cycle):")
        final = all_metrics[-1]
        print(f"  High PE topics: {final.high_pe_topics[:5]}")
        print(f"  Contradiction topics: {final.contradiction_topics[:5]}")
        print(f"  Novelty topics: {final.novelty_topics[:5]}")
    
    # Save results
    if config.output:
        output = {
            'config': asdict(config),
            'metrics': [asdict(m) for m in all_metrics],
            'summary': {
                'total_concepts_gained': total_concepts if all_metrics else 0,
                'total_edges_gained': total_edges if all_metrics else 0,
                'total_searches': total_searches if all_metrics else 0,
                'decoder_web_improvement': decoder_improvement if all_metrics else 0,
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
    parser = argparse.ArgumentParser(description="Background Learner Experiment")
    parser.add_argument("--cycles", type=int, default=20, help="Number of learning cycles")
    parser.add_argument("--delay", type=float, default=10.0, help="Seconds between cycles")
    parser.add_argument("--dim", type=int, default=64, help="Graph dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-trace", action="store_true", help="Disable trace output")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()
    
    config = BackgroundLearnerConfig(
        n_cycles=args.cycles,
        delay_seconds=args.delay,
        dim=args.dim,
        seed=42,
        trace=not args.no_trace,
        output=args.output,
    )
    
    run_background_learner_experiment(config)


if __name__ == "__main__":
    main()