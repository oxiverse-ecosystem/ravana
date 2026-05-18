#!/usr/bin/env python3
"""
RAVANA v2 — Intelligence Formation Visualizer
Plot learning curves to literally see intelligence forming.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def load_experiments():
    """Load experiment results."""
    project_root = Path(__file__).resolve().parent.parent
    path = project_root / "results" / "experiment_comparison.json"
    if not path.exists():
        print("Run experiments first: python experiments/runner.py")
        return None
    with open(path) as f:
        return json.load(f)


def plot_intelligence_formation(results, save_path=None):
    """
    The key plot: clamp_rate vs dissonance_std over time.
    
    Intelligence emerges when:
    - clamp_rate ↓ (learning to avoid violations)
    - dissonance_std stays ↑ (still exploring)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("🧠 Intelligence Formation Dashboard", fontsize=16, fontweight='bold')
    
    # Note: These are final values - for full curves we'd need episode-by-episode data
    # This shows the comparison across conditions
    
    names = [r['name'] for r in results]
    clamp_rates = [r['final_clamp_rate'] for r in results]
    align_scores = [r['final_alignment_score'] for r in results]
    d_ranges = [r['dissonance_range'][1] - r['dissonance_range'][0] for r in results]
    d_stds = [r['dissonance_std'] for r in results]
    final_clamps = [r['final_clamp_clamps'] for r in results]
    
    # Plot 1: The Intelligence Signature
    ax = axes[0, 0]
    colors = ['#e74c3c', '#3498db', '#f39c12']  # Red, Blue, Orange
    for i, (name, clamp, std) in enumerate(zip(names, clamp_rates, d_stds)):
        ax.scatter(clamp, std, s=200, c=colors[i], label=name, alpha=0.8, edgecolors='black')
    ax.set_xlabel("Clamp Rate (lower = safer)", fontsize=11)
    ax.set_ylabel("Dissonance Std (higher = more exploration)", fontsize=11)
    ax.set_title("🎯 The Intelligence Signature\n(Ideal: bottom-right quadrant)", fontsize=12)
    ax.legend(loc='upper right')
    ax.axhline(y=0.15, color='gray', linestyle='--', alpha=0.5, label='Exploration threshold')
    ax.axvline(x=0.05, color='gray', linestyle='--', alpha=0.5)
    
    # Add quadrant labels
    ax.text(0.02, 0.25, "🧠 INTELLIGENT\n(low clamp, high exploration)", 
            fontsize=10, ha='center', va='center', 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.text(0.15, 0.25, "🚨 RECKLESS\n(high clamp, high exploration)", 
            fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax.text(0.02, 0.05, "⚖️ CAUTIOUS\n(low clamp, low exploration)", 
            fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.text(0.15, 0.05, "💀 DEAD\n(high clamp, no exploration)", 
            fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # Plot 2: Alignment Score
    ax = axes[0, 1]
    bars = ax.barh(names, align_scores, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel("Alignment Score (higher = better thinking)", fontsize=11)
    ax.set_title("🎯 Controller / Clamp Agreement", fontsize=12)
    ax.set_xlim(0, 1)
    for bar, score in zip(bars, align_scores):
        ax.text(score + 0.02, bar.get_y() + bar.get_height()/2, 
                f"{score:.2f}", va='center', fontsize=10)
    
    # Plot 3: Dissonance Range
    ax = axes[1, 0]
    bars = ax.barh(names, d_ranges, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel("Dissonance Range (width of exploration)", fontsize=11)
    ax.set_title("🌊 Exploration Width\n(wider = more cognitive diversity)", fontsize=12)
    for bar, range_val in zip(bars, d_ranges):
        ax.text(range_val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{range_val:.3f}", va='center', fontsize=10)
    
    # Plot 4: Final Clamp Events
    ax = axes[1, 1]
    bars = ax.barh(names, final_clamps, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel("Final Clamp Events (should → 0)", fontsize=11)
    ax.set_title("🚨 Constitutional Violations\n(canary metric)", fontsize=12)
    for bar, clamps in zip(bars, final_clamps):
        ax.text(clamps + 0.5, bar.get_y() + bar.get_height()/2,
                f"{clamps}", va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Dashboard saved to: {save_path}")
    
    return fig


def main():
    """Generate visualization."""
    results = load_experiments()
    if not results:
        return
    
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    
    fig = plot_intelligence_formation(results, save_path=output_dir / "intelligence_formation.png")
    
    # Print key insight
    print("\n" + "="*70)
    print("🧠 INTELLIGENCE FORMATION ANALYSIS")
    print("="*70)
    
    adaptive = [r for r in results if 'Adaptive' in r['name']][0]
    baseline = [r for r in results if 'Baseline' in r['name']][0]
    
    # Calculate improvement
    clamp_improvement = ((baseline['final_clamp_rate'] - adaptive['final_clamp_rate']) 
                        / baseline['final_clamp_rate'] * 100)
    alignment_improvement = ((adaptive['final_alignment_score'] - baseline['final_alignment_score'])
                            / baseline['final_alignment_score'] * 100)
    
    print(f"\nAdaptive vs Baseline:")
    print(f"  Clamp rate improved: {clamp_improvement:.1f}% {'✅' if clamp_improvement > 0 else '❌'}")
    print(f"  Alignment improved: {alignment_improvement:.1f}% {'✅' if alignment_improvement > 0 else '❌'}")
    
    # Check for cowardice
    d_range = adaptive['dissonance_range'][1] - adaptive['dissonance_range'][0]
    if d_range < 0.3 and adaptive['final_clamp_clamps'] < 5:
        print(f"\n🚨 WARNING: Low exploration ({d_range:.3f}) + few clamps")
        print("   System may have become CAUTIOUS, not INTELLIGENT")
    elif d_range > 0.4 and adaptive['final_clamp_clamps'] < 10:
        print(f"\n🧠 SUCCESS: Wide exploration ({d_range:.3f}) + controlled clamps")
        print("   System shows DISCIPLINED CURIOSITY")
    else:
        print(f"\n⚖️ MIXED: Range={d_range:.3f}, Final clamps={adaptive['final_clamp_clamps']}")
    
    print(f"\n📊 See full dashboard: {output_dir / 'intelligence_formation.png'}")


if __name__ == "__main__":
    main()
