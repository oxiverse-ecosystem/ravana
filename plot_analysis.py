"""
Post-hoc analysis for RAVANA experiments.

Generates:
1. Backward-transfer plot — Domain A retention over time as Domain B is learned
2. Concept-drift trajectory — concept vector movement over training
3. Cross-domain transfer summary — bar charts comparing conditions

Reads from experiment_results/ and checkpoints/.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not installed — installing...")
    os.system(f"{sys.executable} -m pip install matplotlib")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True


# ── 1. Backward-Transfer Plot ──────────────────────────────────────────────

def load_snapshots():
    """Load snapshots from checkpoints first, fall back to benchmark json."""
    ckpt_path = "checkpoints/lifelong/snapshots.json"
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            snaps = json.load(f)
        if len(snaps) > 1:
            return snaps
    # Fallback
    results_path = "experiment_results/lifelong_benchmark.json"
    if os.path.exists(results_path):
        with open(results_path) as f:
            data = json.load(f)
        return data.get("snapshots", [])
    return []


def plot_backward_transfer(output_path="experiment_results/backward_transfer.png"):
    """Plot retention curve showing catastrophic forgetting over training."""
    snapshots = load_snapshots()
    if not snapshots:
        print("No snapshots found — run lifelong experiment first.")
        return

    steps = [s["step"] for s in snapshots]
    retention = [s.get("retention_overall", 0) for s in snapshots]
    early_recall = [s.get("early_recall", 0) for s in snapshots]
    late_recall = [s.get("late_recall", 0) for s in snapshots]
    forgetting = [s.get("catastrophic_forgetting", 0) for s in snapshots]
    n_concepts = [s.get("n_concepts", 0) for s in snapshots]
    n_edges = [s.get("n_edges", 0) for s in snapshots]
    sleep_cycles = [s.get("sleep_cycles", 0) for s in snapshots]
    cumulative_time = [s.get("cumulative_time_s", 0) for s in snapshots]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RAVANA Lifelong Learning — 100K Experiences", fontsize=16, fontweight='bold')

    # Panel 1: Retention curves
    ax = axes[0, 0]
    ax.plot(steps, retention, 'b-', linewidth=2, label='Overall Retention')
    ax.plot(steps, early_recall, 'r--', linewidth=1.5, label='Early Knowledge')
    ax.plot(steps, late_recall, 'g:', linewidth=1.5, label='Late Knowledge')
    ax.axhline(y=0.14, color='orange', linestyle='--', alpha=0.5, label='Random Baseline (~14%)')
    ax.set_xlabel('Experience Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Retention Curve')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(steps))

    # Panel 2: Catastrophic forgetting
    ax = axes[0, 1]
    colors = ['red' if f < 0 else 'green' for f in forgetting]
    ax.bar(steps, forgetting, width=max(1, len(steps)//3), color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Experience Step')
    ax.set_ylabel('Forgetting Delta')
    ax.set_title('Catastrophic Forgetting (negative = lost knowledge)')
    ax.grid(True, alpha=0.3)

    # Panel 3: Graph growth
    ax = axes[1, 0]
    ax.plot(steps, n_concepts, 'b-', linewidth=2, label='Concepts')
    ax2 = ax.twinx()
    ax2.plot(steps, n_edges, 'r-', linewidth=1.5, label='Edges')
    ax.set_xlabel('Experience Step')
    ax.set_ylabel('Concept Nodes', color='b')
    ax2.set_ylabel('Edges', color='r')
    ax.set_title('Graph Growth')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Panel 4: Compute cost
    ax = axes[1, 1]
    ax.plot(steps, cumulative_time, 'purple', linewidth=2)
    ax.set_xlabel('Experience Step')
    ax.set_ylabel('Cumulative Time (s)')
    ax.set_title('Compute Cost')
    ax.grid(True, alpha=0.3)
    # Add sleep cycle markers
    if sleep_cycles:
        ax2 = ax.twinx()
        ax2.plot(steps, sleep_cycles, 'orange', linewidth=1, alpha=0.5)
        ax2.set_ylabel('Sleep Cycles', color='orange')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Backward-transfer plot saved to {output_path}")


# ── 2. Concept-Drift Trajectory ────────────────────────────────────────────

def plot_concept_drift(output_path="experiment_results/concept_drift.png"):
    """Plot concept drift from checkpoint snapshots."""
    snapshots = load_snapshots()
    if not snapshots:
        print("No snapshots found.")
        return

    steps = [s["step"] for s in snapshots]
    n_concepts = [s.get("n_concepts", 0) for s in snapshots]
    n_edges = [s.get("n_edges", 0) for s in snapshots]
    n_abstract = [s.get("n_abstract", 0) for s in snapshots]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Concept-Drift Trajectory", fontsize=14, fontweight='bold')

    # Drift: concept count changes
    ax = axes[0]
    concept_deltas = [0] + [n_concepts[i] - n_concepts[i-1] for i in range(1, len(n_concepts))]
    colors = ['green' if d >= 0 else 'red' for d in concept_deltas]
    ax.bar(steps, concept_deltas, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Concepts Added/Removed')
    ax.set_title('Concept Drift (per interval)')
    ax.grid(True, alpha=0.3)

    # Abstract concept emergence
    ax = axes[1]
    if any(n_abstract):
        abstract_frac = [a / max(1, c) for a, c in zip(n_abstract, n_concepts)]
        ax.plot(steps, abstract_frac, 'b-', linewidth=2)
        ax.fill_between(steps, abstract_frac, alpha=0.2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Fraction Abstract')
    ax.set_title('Abstract Concept Emergence')
    ax.grid(True, alpha=0.3)

    # Edge density
    ax = axes[2]
    density = [e / max(1, c * (c - 1) / 2) for e, c in zip(n_edges, n_concepts)]
    ax.plot(steps, density, 'r-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Edge Density')
    ax.set_title('Graph Density Over Time')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Concept-drift plot saved to {output_path}")


# ── 3. Cross-Domain Transfer Summary ───────────────────────────────────────

def plot_cross_domain_summary(output_path="experiment_results/cross_domain_summary.png"):
    """Bar chart comparing cross-domain transfer conditions."""
    # Load both transfer and replay results
    transfer_path = "experiment_results/cross_domain_transfer.json"
    replay_path = "experiment_results/cross_domain_replay.json"

    transfer_data = {}
    replay_data = {}
    if os.path.exists(transfer_path):
        with open(transfer_path) as f:
            transfer_data = json.load(f)
    if os.path.exists(replay_path):
        with open(replay_path) as f:
            replay_data = json.load(f)

    if not transfer_data and not replay_data:
        print("No cross-domain results found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Cross-Domain Transfer Results", fontsize=14, fontweight='bold')

    # Left: Ablation — Domain A retention after Domain B training
    ax = axes[0]
    conditions = []
    a_retention = []
    b_accuracy = []

    if replay_data:
        for cond in ['baseline', 'replay']:
            if cond in replay_data:
                d = replay_data[cond]
                label = 'No replay' if cond == 'baseline' else 'Sleep replay'
                conditions.append(label)
                a_retention.append(d.get('post_b_on_a', {}).get('top10_accuracy', 0) * 100)
                b_accuracy.append(d.get('post_b_on_b', {}).get('top10_accuracy', 0) * 100)
    elif transfer_data:
        rlm = transfer_data.get('rlm', {})
        conditions.append('RLM (no replay)')
        a_retention.append(rlm.get('post_train_b_on_a', {}).get('top10_accuracy', 0) * 100)
        b_accuracy.append(rlm.get('post_train_b_on_b', {}).get('top10_accuracy', 0) * 100)

    if conditions:
        x = np.arange(len(conditions))
        w = 0.35
        bars1 = ax.bar(x - w/2, a_retention, w, label='Domain A retention', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + w/2, b_accuracy, w, label='Domain B accuracy', color='coral', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.set_ylabel('Top-10 Accuracy (%)')
        ax.set_title('Domain A Retention After Domain B Training')
        ax.legend()
        ax.set_ylim(0, 60)
        ax.grid(True, alpha=0.3, axis='y')
        for bar in bars1:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f'{h:.0f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
        for bar in bars2:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f'{h:.0f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

    # Right: Full pipeline with replay
    ax2 = axes[1]
    if replay_data and 'replay' in replay_data:
        rep = replay_data['replay']
        metrics = {
            'A after\nA train': rep.get('post_a_on_a', {}).get('top10_accuracy', 0) * 100,
            'A after\nB train': rep.get('post_b_on_a', {}).get('top10_accuracy', 0) * 100,
            'B after\nB train': rep.get('post_b_on_b', {}).get('top10_accuracy', 0) * 100,
            'A after\nsleep': rep.get('post_sleep_a', {}).get('top10_accuracy', 0) * 100,
        }
        colors = ['#1565C0', '#90CAF9', '#C62828', '#4CAF50']
        bars = ax2.bar(metrics.keys(), metrics.values(), color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax2.set_ylabel('Top-10 Accuracy (%)')
        ax2.set_title('Sleep-Time Replay: Full Pipeline')
        ax2.set_ylim(0, 60)
        ax2.grid(True, alpha=0.3, axis='y')
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax2.annotate(f'{h:.0f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
    else:
        # Fall back to transfer metrics
        rlm = transfer_data.get('rlm', {})
        metrics = data.get("transfer_metrics", {})
        phases = ['After A', 'After B', 'After Sleep']
        edge_counts = []
        for phase_key in ['graph_after_a', 'graph_after_b', 'graph_after_sleep']:
            g = rlm.get(phase_key, {})
            edge_counts.append(g.get("n_edges", 0))
        ax2.bar(phases, edge_counts, color=['steelblue', 'coral', 'green'], alpha=0.8)
        ax2.set_ylabel('Edges')
        ax2.set_title('Graph Growth Across Phases')
        for i, v in enumerate(edge_counts):
            ax2.text(i, v + 10, str(v), ha='center', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Cross-domain summary saved to {output_path}")


# ── 4. Combined Report ─────────────────────────────────────────────────────

def generate_all_plots():
    """Generate all analysis plots."""
    print("=" * 60)
    print("RAVANA ANALYSIS — GENERATING PLOTS")
    print("=" * 60)

    snaps = load_snapshots()
    if snaps:
        print(f"\nLoaded {len(snaps)} snapshots")
        plot_backward_transfer()
        plot_concept_drift()
    else:
        print("Skipping lifelong plots — no snapshots yet")

    if os.path.exists("experiment_results/cross_domain_transfer.json") or \
       os.path.exists("experiment_results/cross_domain_replay.json"):
        plot_cross_domain_summary()
    else:
        print("Skipping cross-domain plots — no results yet")

    print("\nDone.")


if __name__ == "__main__":
    generate_all_plots()
