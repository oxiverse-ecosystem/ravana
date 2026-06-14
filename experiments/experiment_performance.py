#!/usr/bin/env python3
"""
Real-time Performance Benchmarks for RAVANA
============================================
Measures latency, throughput, and memory under load:
1. Latency percentiles (p50, p95, p99, p99.9)
2. Throughput (queries/second)
3. Memory footprint (baseline, peak, growth)
4. Concurrent request handling
5. Cold start vs warm performance
6. Component-level profiling
"""

import os
import sys
import time
import json
import numpy as np
import threading
import queue
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.ravana_chat import CognitiveChatEngine


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PerfConfig:
    seed: int = 42
    output: str = None
    trace: bool = True

    # Benchmark parameters
    warmup_queries: int = 50
    benchmark_queries: int = 500
    concurrent_users: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    query_types: List[str] = field(default_factory=lambda: [
        "simple",      # "what is trust"
        "complex",     # "how does trust work in relationships"
        "creative",    # "create a blueprint for building trust"
        "followup",    # "tell me more"
    ])

    # Profiling
    profile_components: bool = True
    memory_interval_ms: int = 100


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LatencyMetrics:
    query: str
    query_type: str
    latency_ms: float
    component_timings: Dict[str, float]  # component -> ms
    memory_mb: float
    used_decoder: bool
    strategy: str
    success: bool
    error: str = ""


@dataclass
class ThroughputMetrics:
    concurrent_users: int
    duration_sec: float
    total_queries: int
    successful_queries: int
    failed_queries: int
    qps: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    p999_latency: float
    avg_memory_mb: float
    peak_memory_mb: float


# ═══════════════════════════════════════════════════════════════════════════
# Query Templates
# ═══════════════════════════════════════════════════════════════════════════

QUERY_TEMPLATES = {
    "simple": [
        "what is trust", "what is friendship", "what is love",
        "what is fear", "what is hope", "what is courage",
    ],
    "complex": [
        "how does trust work in relationships",
        "create a blueprint for building trust",
        "why do people betray each other",
        "what makes a friendship last",
        "how does memory shape identity",
        "explain the neuroscience of trust",
    ],
    "creative": [
        "write a poem about trust",
        "create a story about friendship",
        "imagine a world without betrayal",
        "design a ritual for building trust",
        "compose a letter to your future self",
    ],
    "followup": [
        "tell me more", "what else", "why is that",
        "how does that work", "give me an example",
        "can you explain further",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# Profiling Helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


# ═══════════════════════════════════════════════════════════════════════════
# Single Query Benchmark (with component profiling)
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_single_query(engine: CognitiveChatEngine, query: str,
                           profile_components: bool = False) -> LatencyMetrics:
    """Run single query with optional component-level timing."""

    mem_before = get_memory_mb()
    component_timings = {}

    # Component-level profiling via monkey-patching (if enabled)
    original_methods = {}
    if profile_components:
        components = [
            ('graph_walk', '_activate_concepts'),
            ('decoder', '_generate_with_decoder'),
            ('syntactic', '_generate_with_syntactic'),
            ('reasoning', '_reasoning_loop'),
            ('sleep_check', '_maybe_sleep'),
        ]
        for name, method_name in components:
            if hasattr(engine, method_name):
                original = getattr(engine, method_name)
                def make_wrapper(orig, comp_name):
                    def wrapper(*args, **kwargs):
                        t0 = time.perf_counter()
                        result = orig(*args, **kwargs)
                        component_timings[comp_name] = (time.perf_counter() - t0) * 1000
                        return result
                    return wrapper
                setattr(engine, method_name, make_wrapper(original, name))

    t0 = time.perf_counter()
    try:
        response = engine.process_turn(query)
        latency = (time.perf_counter() - t0) * 1000
        success = True
        error = ""
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        response = ""
        success = False
        error = str(e)

    # Restore original methods
    if profile_components:
        for name, method_name in components:
            if method_name in original_methods:
                setattr(engine, method_name, original_methods[method_name])

    mem_after = get_memory_mb()

    metrics = LatencyMetrics(
        query=query[:100],
        query_type="",  # filled by caller
        latency_ms=latency,
        component_timings=component_timings,
        memory_mb=mem_after,
        used_decoder=(engine.neural_decoder is not None and
                      engine._decoder_vocab_built and
                      engine._decoder_training_count >= 500),
        strategy=getattr(engine, '_last_strategy', 'unknown'),
        success=success,
        error=error,
    )
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Concurrent Load Test
# ═══════════════════════════════════════════════════════════════════════════

def run_concurrent_load_test(config: PerfConfig, n_users: int, duration_sec: float = 30.0) -> ThroughputMetrics:
    """Run concurrent load test with simulated users."""

    # Create engine per user (isolated state)
    engines = []
    for i in range(n_users):
        os.environ['RAVANA_SILENT'] = '1'
        engine = CognitiveChatEngine(dim=64, seed=config.seed + i, baby_mode=True)
        if hasattr(engine, '_seed_corpus_training'):
            engine._seed_corpus_training()
        engines.append(engine)

    # Warmup
    for engine in engines:
        for _ in range(5):
            engine.process_turn("what is trust")

    all_latencies = []
    all_memory = []
    successful = 0
    failed = 0
    lock = threading.Lock()

    def user_worker(user_id: int, stop_event: threading.Event, results: queue.Queue):
        engine = engines[user_id]
        rng = np.random.RandomState(config.seed + user_id * 1000)

        while not stop_event.is_set():
            query_type = rng.choice(config.query_types)
            query = rng.choice(QUERY_TEMPLATES[query_type])

            t0 = time.perf_counter()
            try:
                engine.process_turn(query)
                latency = (time.perf_counter() - t0) * 1000
                with lock:
                    all_latencies.append(latency)
                    all_memory.append(get_memory_mb())
                    successful += 1
            except Exception:
                with lock:
                    failed += 1

    stop_event = threading.Event()
    results_queue = queue.Queue()

    threads = []
    for i in range(n_users):
        t = threading.Thread(target=user_worker, args=(i, stop_event, results_queue))
        t.start()
        threads.append(t)

    time.sleep(duration_sec)
    stop_event.set()

    for t in threads:
        t.join(timeout=5.0)

    total = successful + failed
    qps = total / duration_sec if duration_sec > 0 else 0

    return ThroughputMetrics(
        concurrent_users=n_users,
        duration_sec=duration_sec,
        total_queries=total,
        successful_queries=successful,
        failed_queries=failed,
        qps=qps,
        p50_latency=percentile(all_latencies, 50),
        p95_latency=percentile(all_latencies, 95),
        p99_latency=percentile(all_latencies, 99),
        p999_latency=percentile(all_latencies, 99.9),
        avg_memory_mb=np.mean(all_memory) if all_memory else 0,
        peak_memory_mb=np.max(all_memory) if all_memory else 0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_performance_experiment(config: PerfConfig = None):
    if config is None:
        config = PerfConfig()

    np.random.seed(config.seed)

    print("=" * 70)
    print("REAL-TIME PERFORMANCE BENCHMARKS")
    print("=" * 70)

    # Initialize engine
    os.environ['RAVANA_SILENT'] = '1' if not config.trace else '0'
    engine = CognitiveChatEngine(dim=64, seed=config.seed, baby_mode=True)
    if hasattr(engine, '_seed_corpus_training'):
        engine._seed_corpus_training()

    print(f"Engine: {len(engine.graph.nodes)} concepts, decoder steps: {engine._decoder_training_count}")
    print(f"Decoder ready: {engine.neural_decoder is not None and engine._decoder_vocab_built}")

    # 1. Warmup
    print(f"\n{'='*50}")
    print(f"WARMUP ({config.warmup_queries} queries)")
    print(f"{'='*50}")

    warmup_queries = []
    for qt in config.query_types:
        warmup_queries.extend(QUERY_TEMPLATES[qt])
    warmup_queries = (warmup_queries * (config.warmup_queries // len(warmup_queries) + 1))[:config.warmup_queries]

    for i, query in enumerate(warmup_queries):
        engine.process_turn(query)
        if config.trace and i % 10 == 0:
            print(f"  Warmed up {i+1}/{config.warmup_queries}")

    # 2. Single-query latency benchmark
    print(f"\n{'='*50}")
    print(f"LATENCY BENCHMARK ({config.benchmark_queries} queries)")
    print(f"{'='*50}")

    all_latency_metrics = []
    query_pool = []
    for qt in config.query_types:
        for q in QUERY_TEMPLATES[qt]:
            query_pool.append((q, qt))

    rng = np.random.RandomState(config.seed)
    for i in range(config.benchmark_queries):
        query, qtype = rng.choice(query_pool)
        m = benchmark_single_query(engine, query, config.profile_components)
        m.query_type = qtype
        all_latency_metrics.append(m)

        if config.trace and i % 50 == 0:
            print(f"  Query {i+1}/{config.benchmark_queries}: {m.latency_ms:.1f}ms")

    # Analyze latency by query type
    print("\nLATENCY BY QUERY TYPE:")
    for qtype in config.query_types:
        type_metrics = [m for m in all_latency_metrics if m.query_type == qtype and m.success]
        if type_metrics:
            latencies = [m.latency_ms for m in type_metrics]
            print(f"  {qtype:10s}: n={len(latencies)}  "
                  f"p50={percentile(latencies,50):.1f}ms  "
                  f"p95={percentile(latencies,95):.1f}ms  "
                  f"p99={percentile(latencies,99):.1f}ms  "
                  f"p99.9={percentile(latencies,99.9):.1f}ms")

    # Component timings
    if config.profile_components:
        print("\nCOMPONENT TIMINGS (avg ms):")
        component_sums = defaultdict(list)
        for m in all_latency_metrics:
            if m.success:
                for comp, timing in m.component_timings.items():
                    component_sums[comp].append(timing)
        for comp, times in component_sums.items():
            print(f"  {comp:20s}: avg={np.mean(times):.2f}ms  "
                  f"p50={percentile(times,50):.2f}ms  "
                  f"p99={percentile(times,99):.2f}ms")

    # 3. Memory profiling
    print(f"\n{'='*50}")
    print(f"MEMORY PROFILE")
    print(f"{'='*50}")

    mem_samples = [m.memory_mb for m in all_latency_metrics if m.success]
    print(f"  Baseline: {mem_samples[0]:.1f}MB")
    print(f"  Average:  {np.mean(mem_samples):.1f}MB")
    print(f"  Peak:     {np.max(mem_samples):.1f}MB")
    print(f"  Growth:   {np.max(mem_samples) - mem_samples[0]:.1f}MB")

    # 4. Concurrent load tests
    print(f"\n{'='*50}")
    print(f"CONCURRENT LOAD TESTS")
    print(f"{'='*50}")

    throughput_results = []
    for n_users in config.concurrent_users:
        print(f"\n  Testing {n_users} concurrent users...")
        result = run_concurrent_load_test(config, n_users, duration_sec=15.0)
        throughput_results.append(result)
        print(f"    QPS: {result.qps:.1f}")
        print(f"    Success: {result.successful_queries}/{result.total_queries}")
        print(f"    p50/p95/p99: {result.p50_latency:.1f}/{result.p95_latency:.1f}/{result.p99_latency:.1f}ms")
        print(f"    Avg/Peak mem: {result.avg_memory_mb:.1f}/{result.peak_memory_mb:.1f}MB")

    # 5. Cold start vs warm
    print(f"\n{'='*50}")
    print(f"COLD START vs WARM")
    print(f"{'='*50}")

    # Fresh engine
    os.environ['RAVANA_SILENT'] = '1'
    cold_engine = CognitiveChatEngine(dim=64, seed=config.seed, baby_mode=True)
    if hasattr(cold_engine, '_seed_corpus_training'):
        cold_engine._seed_corpus_training()

    cold_latencies = []
    for _ in range(20):
        m = benchmark_single_query(cold_engine, "what is trust", False)
        cold_latencies.append(m.latency_ms)

    warm_latencies = [m.latency_ms for m in all_latency_metrics[:20] if m.success]

    print(f"  Cold start (first 20): p50={percentile(cold_latencies,50):.1f}ms  "
          f"p99={percentile(cold_latencies,99):.1f}ms")
    print(f"  Warm (first 20):       p50={percentile(warm_latencies,50):.1f}ms  "
          f"p99={percentile(warm_latencies,99):.1f}ms")

    # Summary
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    all_successful = [m for m in all_latency_metrics if m.success]
    all_latencies = [m.latency_ms for m in all_successful]

    print(f"Total benchmark queries: {len(all_latency_metrics)}")
    print(f"Successful: {len(all_successful)} ({len(all_successful)/len(all_latency_metrics)*100:.1f}%)")
    print(f"\nOverall Latency:")
    print(f"  p50:  {percentile(all_latencies,50):.1f}ms")
    print(f"  p95:  {percentile(all_latencies,95):.1f}ms")
    print(f"  p99:  {percentile(all_latencies,99):.1f}ms")
    print(f"  p99.9:{percentile(all_latencies,99.9):.1f}ms")
    print(f"  Mean: {np.mean(all_latencies):.1f}ms")

    print(f"\nThroughput (single-threaded): ~{1000/np.mean(all_latencies):.1f} QPS")
    print(f"\nConcurrent Throughput:")
    for r in throughput_results:
        print(f"  {r.concurrent_users} users: {r.qps:.1f} QPS (p99={r.p99_latency:.1f}ms)")

    print(f"\nMemory:")
    print(f"  Baseline: {mem_samples[0]:.1f}MB")
    print(f"  Peak:     {np.max(mem_samples):.1f}MB")
    print(f"  Growth:   {np.max(mem_samples) - mem_samples[0]:.1f}MB")

    # Save
    if config.output:
        output = {
            'config': asdict(config),
            'latency_by_type': {
                qtype: {
                    'n': len([m for m in all_latency_metrics if m.query_type == qtype and m.success]),
                    'p50': percentile([m.latency_ms for m in all_latency_metrics if m.query_type == qtype and m.success], 50),
                    'p95': percentile([m.latency_ms for m in all_latency_metrics if m.query_type == qtype and m.success], 95),
                    'p99': percentile([m.latency_ms for m in all_latency_metrics if m.query_type == qtype and m.success], 99),
                    'p999': percentile([m.latency_ms for m in all_latency_metrics if m.query_type == qtype and m.success], 99.9),
                } for qtype in config.query_types
            },
            'component_timings': {
                comp: {'avg': float(np.mean(times)), 'p50': float(percentile(times, 50)), 'p99': float(percentile(times, 99))}
                for comp, times in component_sums.items()
            } if config.profile_components else {},
            'memory': {
                'baseline_mb': float(mem_samples[0]),
                'avg_mb': float(np.mean(mem_samples)),
                'peak_mb': float(np.max(mem_samples)),
                'growth_mb': float(np.max(mem_samples) - mem_samples[0]),
            },
            'throughput': [asdict(r) for r in throughput_results],
            'cold_vs_warm': {
                'cold_p50': float(percentile(cold_latencies, 50)),
                'cold_p99': float(percentile(cold_latencies, 99)),
                'warm_p50': float(percentile(warm_latencies, 50)),
                'warm_p99': float(percentile(warm_latencies, 99)),
            },
        }
        with open(config.output, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {config.output}")

    return all_latency_metrics, throughput_results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAVANA Real-time Performance Benchmarks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--benchmark-queries", type=int, default=500, help="Benchmark queries")
    parser.add_argument("--no-trace", action="store_true", help="Disable trace output")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--no-profile", action="store_true", help="Disable component profiling")
    args = parser.parse_args()

    config = PerfConfig(
        seed=args.seed,
        benchmark_queries=args.benchmark_queries,
        trace=not args.no_trace,
        output=args.output,
        profile_components=not args.no_profile,
    )

    run_performance_experiment(config)


if __name__ == "__main__":
    main()