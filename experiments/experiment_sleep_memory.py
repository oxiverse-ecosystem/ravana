#!/usr/bin/env python3
"""
Sleep & Memory Experiment for RAVANA
=====================================
Tests sleep cycle effects on memory:
1. Episodic→semantic consolidation rate
2. Counterfactual dreaming quality
3. Memory interference reduction
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
class SleepMemoryConfig:
    n_sleep_cycles: int = 10
    sleep_interval_turns: int = 8
    seed: int = 42
    trace: bool = True
    output: str = None


# ════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SleepMetrics:
    cycle: int
    turn: int
    episodic_edges_before: int
    semantic_edges_before: int
    episodic_edges_after: int
    semantic_edges_after: int
    consolidated: int
    pruned: int
    contradiction_resolved: int
    counterfactuals_generated: int
    emotional_flips: int
    failure_oversampled: int
    memory_interference: float  # avg PE across edges
    mean_pe: float
    timestamp: float


# ═══════════════════════════════════════════════════════════════════════════

def count_edges(engine: CognitiveChatEngine) -> Tuple[int, int]:
    """Count episodic vs semantic edges."""
    episodic = len(engine._episodic_edges) if hasattr(engine, '_episodic_edges') else 0
    semantic = len(engine._semantic_edges) if hasattr(engine, '_semantic_edges') else 0
    return episodic, semantic


def compute_memory_interference(engine: CognitiveChatEngine) -> float:
    """Compute average prediction error across edges as interference measure."""
    pes = []
    for (src, tgt), edge in engine.graph.edges.items():
        pe = getattr(edge, 'prediction_free_energy', 0.0)
        pes.append(pe)
    return np.mean(pes) if pes else 0.0


def analyze_consolidation_quality(engine: CognitiveChatEngine) -> Dict:
    """Analyze quality of consolidated edges."""
    if not hasattr(engine, '_episodic_edges') or not hasattr(engine, '_semantic_edges'):
        return {'consolidated': 0, 'quality': []}
    
    results = []
    consolidated = 0
    
    for (src, tgt), epi_edge in engine._episodic_edges.items():
        sem_key = (src, tgt)
        if sem_key in engine._semantic_edges:
            sem_edge = engine._semantic_edges[sem_key]
            # Quality: how well does semantic weight match episodic?
            weight_diff = abs(sem_edge.weight - epi_edge.weight)
            confidence_avg = (sem_edge.confidence + epi_edge.confidence) / 2
            results.append({
                'src': src, 'tgt': tgt,
                'episodic_weight': epi_edge.weight,
                'semantic_weight': sem_edge.weight,
                'weight_diff': weight_diff,
                'confidence': confidence_avg,
            })
            consolidated += 1
    
    return {
        'consolidated': consolidated,
        'quality': results,
    }


def run_sleep_memory_experiment(config: 'SleepMemoryConfig' = None):
    if config is None:
        config = SleepMemoryConfig()
    
    np.random.seed(config.seed)
    
    print("=" * 70)
    print("SLEEP & MEMORY EXPERIMENT")
    print("=" * 70)
    print(f"Sleep cycles: {config.n_sleep_cycles}")
    print(f"Sleep interval: every {config.sleep_interval_turns} turns")
    print()
    
    os.environ['RAVANA_SILENT'] = '1'
    engine = CognitiveChatEngine(dim=64, seed=config.seed, baby_mode=True)
    
    if config.trace:
        engine._trace_enabled = True
    
    print(f"Initial: {len(engine.graph.nodes)} concepts, {len(engine.graph.edges)} edges")
    print()
    
    # Add some initial knowledge to have something to consolidate
    initial_facts = [
        "what is trust", "what is friendship", "what is love",
        "what is fear", "what is hope", "what is courage",
        "what is betrayal", "what is loyalty", "what is memory",
        "what is learning", "what is knowledge", "what is wisdom",
    ]
    
    print("Pre-loading initial concepts...")
    for q in initial_facts:
        engine.process_turn(q)
    
    print(f"After pre-load: {len(engine.graph.nodes)} concepts, {len(engine.graph.edges)} edges")
    print()
    
    all_metrics = []
    
    for sleep_cycle in range(1, config.n_sleep_cycles + 1):
        print(f"\n{'='*50}")
        print(f"SLEEP CYCLE {sleep_cycle}/{config.n_sleep_cycles}")
        print(f"{'='*50}")
        
        # Run turns until sleep pressure triggers
        eps_before, sem_before = count_edges(engine)
        pe_before = compute_memory_interference(engine)
        
        # Simulate turns to build up sleep pressure
        turns_run = 0
        while turns_run < config.sleep_interval_turns:
            # Alternate between different query types to create memory pressure
            query_type = turns_run % 4
            if query_type == 0:
                query = "what is trust"
            elif query_type == 1:
                query = "what is friendship"
            elif query_type == 2:
                query = "how does trust work"
            else:
                query = "create a blueprint for trust"
            
            engine.process_turn(query)
            turns_run += 1
        
        # Force sleep consolidation
        print(f"  Running sleep consolidation (turn {engine.turn_count})...")
        engine._sleep_consolidate()
        
        eps_after, sem_after = count_edges(engine)
        pe_after = compute_memory_interference(engine)
        
        # Analyze consolidation
        consolidation_results = analyze_consolidation_quality(engine)
        
        # Count counterfactuals and emotional flips from sleep
        # (these are internal to _sleep_consolidate, we approximate)
        counterfactuals = len(engine._impossible_queries)  # proxy
        
        metrics = SleepMetrics(
            cycle=engine.sleep_cycles_completed,
            turn=engine.turn_count,
            episodic_edges_before=eps_before,
            semantic_edges_before=sem_before,
            episodic_edges_after=len(engine._episodic_edges) if hasattr(engine, '_episodic_edges') else 0,
            semantic_edges_after=len(engine._semantic_edges) if hasattr(engine, '_semantic_edges') else 0,
            consolidated=consolidation_results['consolidated'],
            pruned=eps_before - len(engine._episodic_edges) if hasattr(engine, '_episodic_edges') else 0,
            contradiction_resolved=0,  # would need deeper tracking
            counterfactuals_generated=counterfactuals,
            emotional_flips=0,  # internal to sleep
            failure_oversampled=0,
            memory_interference=pe_after,
            mean_pe=engine._mean_prediction_error,
            timestamp=time.time(),
        )
        
        print(f"  Episodic edges: {eps_before} -> {len(engine._episodic_edges) if hasattr(engine, '_episodic_edges') else 0}")
        print(f"  Semantic edges: {sem_before} -> {len(engine._semantic_edges) if hasattr(engine, '_semantic_edges') else 0}")
        print(f"  Consolidated: {consolidation_results['consolidated']}")
        print(f"  Memory interference (PE): {pe_before:.4f} -> {pe_after:.4f}")
        print(f"  Mean prediction error: {engine._mean_prediction_error:.4f}")
        
        all_metrics.append(metrics)
        
        # Print consolidation quality details
        if consolidation_results['quality']:
            avg_diff = np.mean([q['weight_diff'] for q in consolidation_results['quality']])
            avg_conf = np.mean([q['confidence'] for q in consolidation_results['quality']])
            print(f"  Consolidation quality: {len(consolidation_results['quality'])} edges, "
                  f"avg weight diff={avg_diff:.4f}, avg confidence={avg_conf:.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SLEEP & MEMORY SUMMARY")
    print("=" * 70)
    
    if all_metrics:
        total_consolidated = sum(m.consolidated for m in all_metrics)
        total_pruned = sum(m.pruned for m in all_metrics)
        
        print(f"Sleep cycles completed: {len(all_metrics)}")
        print(f"Total edges consolidated: {total_consolidated}")
        print(f"Total episodic edges pruned: {total_pruned}")
        
        # Interference reduction
        pe_start = all_metrics[0].memory_interference
        pe_end = all_metrics[-1].memory_interference
        print(f"Memory interference: {pe_start:.4f} -> {pe_end:.4f} ({pe_end - pe_start:+.4f})")
        
        # Consolidation quality trend
        print(f"\nConsolidation per cycle:")
        for m in all_metrics:
            print(f"  Cycle {m.cycle}: consolidated={m.consolidated}, "
                  f"interference={m.memory_interference:.4f}")
        
        # Consolidation quality details
        if all_metrics[-1].consolidated > 0:
            final_quality = analyze_consolidation_quality(engine)
            if final_quality['quality']:
                print(f"\nFinal consolidation quality:")
                for q in final_quality['quality'][:10]:
                    src_node = engine.graph.nodes.get(q['src'])
                    tgt_node = engine.graph.nodes.get(q['tgt'])
                    src_label = src_node.label if src_node else f"c{q['src']}"
                    tgt_label = tgt_node.label if tgt_node else f"c{q['tgt']}"
                    print(f"  {src_label} -> {tgt_label}: "
                          f"episodic={q['episodic_weight']:.3f}, "
                          f"semantic={q['semantic_weight']:.3f}, "
                          f"diff={q['weight_diff']:.4f}, "
                          f"conf={q['confidence']:.3f}")
    
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


# Need to import the config class
from dataclasses import dataclass


@dataclass
class SleepMemoryConfig:
    n_sleep_cycles: int = 10
    sleep_interval_turns: int = 8
    seed: int = 42
    trace: bool = True
    output: str = None


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sleep & Memory Experiment")
    parser.add_argument("--cycles", type=int, default=10, help="Number of sleep cycles")
    parser.add_argument("--interval", type=int, default=8, help="Turns between sleep cycles")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-trace", action="store_true", help="Disable trace output")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()
    
    config = SleepMemoryConfig(
        n_sleep_cycles=args.cycles,
        sleep_interval_turns=args.interval,
        seed=args.seed,
        trace=not args.no_trace,
        output=args.output,
    )
    
    run_sleep_memory_experiment(config)


if __name__ == "__main__":
    main()