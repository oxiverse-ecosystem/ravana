"""
Longitudinal Concept Evolution -- Analysis Script

Reads metrics.jsonl from experiment_longitudinal.py and produces:
- Time series summary (text)
- Phase transition detection
- Governor suppression monitoring
- Correlation analysis
- Optional: matplotlib plots (if available)

Usage:
    python analyze_longitudinal.py checkpoints/longitudinal/metrics.jsonl
    python analyze_longitudinal.py checkpoints/longitudinal/metrics.jsonl --plot
"""

import argparse
import json
import sys
from typing import List, Dict, Any

import numpy as np


def load_metrics(path: str) -> List[Dict[str, Any]]:
    """Load metrics from JSONL file."""
    metrics = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))
    return metrics


def time_series_summary(metrics: List[Dict], key: str) -> Dict:
    """Compute summary statistics for a metric over time."""
    values = [m[key] for m in metrics if key in m]
    if not values:
        return {}
    arr = np.array(values)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "first": float(arr[0]),
        "last": float(arr[-1]),
        "trend": float(np.polyfit(range(len(arr)), arr, 1)[0]) if len(arr) > 1 else 0.0,
    }


def detect_phase_transitions(metrics: List[Dict]) -> List[Dict]:
    """Detect phase transitions from the metrics stream."""
    transitions = []
    prev_phase = None
    for m in metrics:
        phase = m.get("phase")
        if phase and phase != prev_phase:
            transitions.append({
                "cycle": m.get("cycle", 0),
                "from": prev_phase,
                "to": phase,
                "confidence": m.get("phase_confidence", 0.0),
                "identity": m.get("identity_strength", 0.0),
                "dissonance": m.get("dissonance_ema", 0.0),
            })
            prev_phase = phase
    return transitions


def governor_suppression_check(metrics: List[Dict]) -> Dict:
    """Check for governor suppression -- rigidification, low exploration, collapse."""
    warnings = []

    # Check entropy trend (declining = rigidifying)
    entropy_summary = time_series_summary(metrics, "graph_entropy")
    if entropy_summary and entropy_summary["trend"] < -0.0001:
        warnings.append(f"ENTROPY DECLINING: trend={entropy_summary['trend']:.6f} -- topology may be rigidifying")

    # Check exploration diversity (phase stuck in one mode)
    phases = [m.get("phase", "unknown") for m in metrics if "phase" in m]
    if phases:
        unique_phases = set(phases)
        if len(unique_phases) == 1:
            warnings.append(f"PHASE LOCKED: stuck in '{phases[0]}' for {len(phases)} samples")

    # Check identity collapse
    id_summary = time_series_summary(metrics, "identity_strength")
    if id_summary and id_summary["last"] < 0.1:
        warnings.append(f"IDENTITY COLLAPSE: strength={id_summary['last']:.3f}")

    # Check meaning stagnation
    meaning_summary = time_series_summary(metrics, "accumulated_meaning")
    if meaning_summary and meaning_summary["trend"] < 0.001 and meaning_summary["count"] > 50:
        warnings.append(f"MEANING STAGNANT: trend={meaning_summary['trend']:.6f}")

    # Check edge growth plateau
    edge_summary = time_series_summary(metrics, "n_edges")
    if edge_summary and edge_summary["trend"] < 0.1 and edge_summary["count"] > 50:
        warnings.append(f"EDGE GROWTH PLATEAU: trend={edge_summary['trend']:.2f} edges/sample")

    return {
        "warnings": warnings,
        "healthy": len(warnings) == 0,
    }


def correlation_analysis(metrics: List[Dict]) -> Dict:
    """Compute correlations between key metrics."""
    keys = ["identity_strength", "valence", "arousal", "dissonance_ema",
            "conceptual_accuracy", "total_free_energy", "n_edges",
            "accumulated_meaning", "sleep_pressure"]

    # Filter to keys present in metrics
    available = [k for k in keys if any(k in m for m in metrics)]
    if len(available) < 2:
        return {}

    # Build matrix
    rows = []
    for m in metrics:
        row = [m.get(k, np.nan) for k in available]
        if not any(np.isnan(row)):
            rows.append(row)

    if len(rows) < 10:
        return {}

    arr = np.array(rows)
    corr = np.corrcoef(arr.T)

    # Find strongest correlations
    strong = []
    for i in range(len(available)):
        for j in range(i+1, len(available)):
            r = corr[i, j]
            if abs(r) > 0.5:
                strong.append({
                    "pair": f"{available[i]} x {available[j]}",
                    "r": float(r),
                })
    strong.sort(key=lambda x: abs(x["r"]), reverse=True)

    return {
        "n_samples": len(rows),
        "strong_correlations": strong[:10],
    }


def analyze(metrics: List[Dict], output_dir: str = None) -> Dict:
    """Run full analysis."""
    if not metrics:
        print("No metrics to analyze.")
        return {}

    print(f"Loaded {len(metrics)} metric samples")
    print(f"Cycle range: {metrics[0].get('cycle', '?')} -> {metrics[-1].get('cycle', '?')}")
    print()

    # Time series summaries
    scalar_keys = ["identity_strength", "valence", "arousal", "dominance",
                   "dissonance_ema", "conceptual_accuracy", "total_free_energy",
                   "n_edges", "n_nodes", "accumulated_meaning", "sleep_pressure",
                   "semantic_memory_count"]

    print("=" * 60)
    print("TIME SERIES SUMMARY")
    print("=" * 60)
    summaries = {}
    for key in scalar_keys:
        s = time_series_summary(metrics, key)
        if s:
            summaries[key] = s
            trend_dir = "^" if s["trend"] > 0.001 else ("v" if s["trend"] < -0.001 else "-")
            print(f"  {key:30s}  {s['first']:8.3f} -> {s['last']:8.3f} {trend_dir}  "
                  f"(mean={s['mean']:.3f}, std={s['std']:.3f})")

    # Phase transitions
    transitions = detect_phase_transitions(metrics)
    print(f"\n{'=' * 60}")
    print(f"PHASE TRANSITIONS ({len(transitions)} detected)")
    print("=" * 60)
    for t in transitions:
        print(f"  cycle {t['cycle']:>8}: {t['from'] or 'N/A':>12} -> {t['to']:<12} "
              f"(conf={t['confidence']:.2f}, id={t['identity']:.3f})")

    # Governor suppression
    gov = governor_suppression_check(metrics)
    print(f"\n{'=' * 60}")
    print("GOVERNOR SUPPRESSION CHECK")
    print("=" * 60)
    if gov["healthy"]:
        print("  HEALTHY -- no suppression indicators detected")
    else:
        for w in gov["warnings"]:
            print(f"  WARNING: {w}")

    # Correlation analysis
    corr = correlation_analysis(metrics)
    if corr:
        print(f"\n{'=' * 60}")
        print(f"STRONG CORRELATIONS (|r| > 0.5, n={corr['n_samples']})")
        print("=" * 60)
        for c in corr["strong_correlations"]:
            bar = "#" * int(abs(c["r"]) * 20)
            sign = "+" if c["r"] > 0 else "-"
            print(f"  {c['pair']:40s}  r={c['r']:+.3f} {sign}{bar}")

    # Optional matplotlib plots
    if output_dir:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            _generate_plots(metrics, output_dir)
            print(f"\nPlots saved to {output_dir}/")
        except ImportError:
            print("\nmatplotlib not available -- skipping plots")

    return {
        "summaries": summaries,
        "transitions": transitions,
        "governor": gov,
        "correlations": corr,
    }


def _generate_plots(metrics: List[Dict], output_dir: str):
    """Generate matplotlib plots."""
    import matplotlib.pyplot as plt
    import os
    os.makedirs(output_dir, exist_ok=True)

    cycles = [m.get("cycle", 0) for m in metrics]

    # 1. Cognitive state
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Cognitive State Evolution")

    pairs = [
        ("identity_strength", "Identity"),
        ("valence", "Valence"),
        ("arousal", "Arousal"),
        ("dissonance_ema", "Dissonance EMA"),
        ("conceptual_accuracy", "Accuracy"),
        ("sleep_pressure", "Sleep Pressure"),
    ]
    for ax, (key, label) in zip(axes.flat, pairs):
        vals = [m.get(key, np.nan) for m in metrics]
        ax.plot(cycles, vals, linewidth=0.5)
        ax.set_ylabel(label)
        ax.set_xlabel("Cycle")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cognitive_state.png"), dpi=150)
    plt.close()

    # 2. Graph topology
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Graph Topology Evolution")

    topo = [
        ("n_edges", "Edges"),
        ("n_nodes", "Nodes"),
        ("graph_entropy", "Graph Entropy"),
        ("clustering_coefficient", "Clustering"),
    ]
    for ax, (key, label) in zip(axes.flat, topo):
        vals = [m.get(key, np.nan) for m in metrics]
        if not all(np.isnan(vals)):
            ax.plot(cycles, vals, linewidth=0.5)
            ax.set_ylabel(label)
            ax.set_xlabel("Cycle")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "topology.png"), dpi=150)
    plt.close()

    # 3. Free energy
    fig, ax = plt.subplots(figsize=(14, 4))
    for key, label in [("semantic_fe", "Semantic"), ("episodic_fe", "Episodic"),
                        ("contradiction_fe", "Contradiction")]:
        vals = [m.get(key, np.nan) for m in metrics]
        ax.plot(cycles, vals, label=label, linewidth=0.5)
    ax.set_ylabel("Free Energy")
    ax.set_xlabel("Cycle")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "free_energy.png"), dpi=150)
    plt.close()

    # 4. Meaning trajectory
    fig, ax = plt.subplots(figsize=(14, 4))
    vals = [m.get("accumulated_meaning", np.nan) for m in metrics]
    ax.plot(cycles, vals, linewidth=0.8, color="purple")
    ax.set_ylabel("Accumulated Meaning")
    ax.set_xlabel("Cycle")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "meaning.png"), dpi=150)
    plt.close()

    print(f"  Generated: cognitive_state.png, topology.png, free_energy.png, meaning.png")


def main():
    parser = argparse.ArgumentParser(description="Analyze longitudinal experiment results")
    parser.add_argument("metrics_path", help="Path to metrics.jsonl")
    parser.add_argument("--plot", action="store_true", help="Generate matplotlib plots")
    parser.add_argument("--output-dir", default=None, help="Output directory for plots")
    args = parser.parse_args()

    metrics = load_metrics(args.metrics_path)
    output_dir = args.output_dir or args.metrics_path.rsplit('/', 1)[0] + '/analysis' if '/' in args.metrics_path else 'analysis'

    results = analyze(metrics, output_dir if args.plot else None)

    # Save analysis results
    analysis_path = args.metrics_path.replace('.jsonl', '_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAnalysis saved to {analysis_path}")


if __name__ == "__main__":
    main()
