"""
RAVANA v2 — Phase B Experimental Framework
Three-way comparison: Baseline vs Adaptive vs Stress Test
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import Governor, GovernorConfig, ResolutionEngine, IdentityEngine, StateManager
from core.adaptation import PolicyTweakLayer, AdaptiveGovernorBridge, AdaptationConfig
from training.pipeline import TrainingPipeline, TrainingConfig


@dataclass
class ExperimentResult:
    """Single experiment outcome."""
    name: str
    episodes: int
    elapsed: float
    
    # Key metrics
    final_clamp_rate: float
    final_alignment_score: float
    dissonance_range: tuple
    dissonance_mean: float
    dissonance_std: float
    
    # Clamp details
    hard_constraint_clamps: int
    final_clamp_clamps: int
    mean_correction: float
    max_correction: float
    
    # Learning (if applicable)
    learning_steps: int = 0
    mean_reward: float = 0.0
    
    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"EXPERIMENT: {self.name}",
            f"{'='*60}",
            f"Final clamp rate: {self.final_clamp_rate:.3f}",
            f"Alignment score: {self.final_alignment_score:.3f}",
            f"Dissonance: {self.dissonance_mean:.3f} ± {self.dissonance_std:.3f}",
            f"Range: [{self.dissonance_range[0]:.3f}, {self.dissonance_range[1]:.3f}]",
            f"Final clamps: {self.final_clamp_clamps}",
        ]
        if self.learning_steps > 0:
            lines.append(f"Learning steps: {self.learning_steps:,}")
            lines.append(f"Mean reward: {self.mean_reward:.4f}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


def run_baseline(episodes: int = 1000) -> ExperimentResult:
    """
    Experiment A: Baseline (no adaptation)
    Records clamp metrics without any learning.
    """
    print("\n" + "="*70)
    print("EXPERIMENT A: BASELINE (No Adaptation)")
    print("="*70)
    
    # Standard components
    governor = Governor(GovernorConfig(
        max_dissonance=0.95, min_dissonance=0.15,
        max_identity=0.95, min_identity=0.10,
        dissonance_target=0.45, identity_target=0.65,
    ))
    resolution = ResolutionEngine(partial_threshold=0.15)
    identity = IdentityEngine(initial_strength=0.5)
    manager = StateManager(governor, resolution, identity)
    
    # Run training
    config = TrainingConfig(total_episodes=episodes, log_interval=100, debug_first_n=0)
    pipeline = TrainingPipeline(manager, config)
    
    start = time.time()
    pipeline.train()
    elapsed = time.time() - start
    
    # Extract metrics
    history = manager.history
    d_values = [h['post_dissonance'] for h in history]
    
    clamp_metrics = governor.get_clamp_metrics()
    
    return ExperimentResult(
        name="Baseline (No Adaptation)",
        episodes=episodes,
        elapsed=elapsed,
        final_clamp_rate=clamp_metrics['clamp_rate'],
        final_alignment_score=clamp_metrics['alignment_score'],
        dissonance_range=(min(d_values), max(d_values)),
        dissonance_mean=sum(d_values) / len(d_values),
        dissonance_std=(sum((d - (sum(d_values)/len(d_values)))**2 for d in d_values) / len(d_values))**0.5,
        hard_constraint_clamps=clamp_metrics['hard_constraint_clamps'],
        final_clamp_clamps=clamp_metrics['final_clamp_clamps'],
        mean_correction=clamp_metrics['mean_correction_magnitude'],
        max_correction=clamp_metrics['max_correction_magnitude']
    )


def run_adaptive(episodes: int = 1000, learning_rate: float = 0.01) -> ExperimentResult:
    """
    Experiment B: With Adaptation
    Compares delta from baseline.
    """
    print("\n" + "="*70)
    print(f"EXPERIMENT B: ADAPTIVE (lr={learning_rate})")
    print("="*70)
    
    # Components with adaptation
    governor = Governor(GovernorConfig(
        max_dissonance=0.95, min_dissonance=0.15,
        max_identity=0.95, min_identity=0.10,
        dissonance_target=0.45, identity_target=0.65,
    ))
    resolution = ResolutionEngine(partial_threshold=0.15)
    identity = IdentityEngine(initial_strength=0.5)
    manager = StateManager(governor, resolution, identity)
    
    # Adaptation layer
    adaptation = PolicyTweakLayer(AdaptationConfig(
        learning_rate=learning_rate,
        momentum=0.9,
        exploration_bonus=0.1,
        clamp_penalty=1.0
    ))
    bridge = AdaptiveGovernorBridge(governor, adaptation)
    
    # Run training
    config = TrainingConfig(total_episodes=episodes, log_interval=100, debug_first_n=0)
    pipeline = TrainingPipeline(manager, config)
    
    start = time.time()
    pipeline.train()
    elapsed = time.time() - start
    
    # Extract metrics
    history = manager.history
    d_values = [h['post_dissonance'] for h in history]
    
    clamp_metrics = governor.get_clamp_metrics()
    adapt_status = adaptation.get_status()
    
    return ExperimentResult(
        name=f"Adaptive (lr={learning_rate})",
        episodes=episodes,
        elapsed=elapsed,
        final_clamp_rate=clamp_metrics['clamp_rate'],
        final_alignment_score=clamp_metrics['alignment_score'],
        dissonance_range=(min(d_values), max(d_values)),
        dissonance_mean=sum(d_values) / len(d_values),
        dissonance_std=(sum((d - (sum(d_values)/len(d_values)))**2 for d in d_values) / len(d_values))**0.5,
        hard_constraint_clamps=clamp_metrics['hard_constraint_clamps'],
        final_clamp_clamps=clamp_metrics['final_clamp_clamps'],
        mean_correction=clamp_metrics['mean_correction_magnitude'],
        max_correction=clamp_metrics['max_correction_magnitude'],
        learning_steps=adapt_status['learning_steps'],
        mean_reward=adapt_status['mean_recent_reward']
    )


def run_stress_test(episodes: int = 500) -> ExperimentResult:
    """
    Experiment C: High Learning Rate (Force Instability)
    Tests whether system is learning or just drifting.
    """
    print("\n" + "="*70)
    print("EXPERIMENT C: STRESS TEST (High LR = 0.1)")
    print("="*70)
    
    return run_adaptive(episodes=episodes, learning_rate=0.1)


def compare_experiments(results: List[ExperimentResult]) -> str:
    """Generate comparison report."""
    lines = [
        "\n" + "="*80,
        "🧠 EXPERIMENTAL COMPARISON",
        "="*80,
        "",
        "Metric                    | Baseline  | Adaptive  | Stress    |",
        "--------------------------|-----------|-----------|-----------|"
    ]
    
    for metric in ['final_clamp_rate', 'final_alignment_score', 'final_clamp_clamps', 'mean_correction']:
        base = getattr(results[0], metric)
        adapt = getattr(results[1], metric)
        stress = getattr(results[2], metric)
        lines.append(f"{metric:<25} | {base:9.4f} | {adapt:9.4f} | {stress:9.4f} |")
    
    # Dissonance analysis
    lines.append("")
    lines.append("DISSONANCE ANALYSIS (🚨 Coward vs Intelligence)")
    for r in results:
        range_width = r.dissonance_range[1] - r.dissonance_range[0]
        status = "🧠 INTELLIGENT" if range_width > 0.4 and r.final_clamp_clamps < 10 else "🚨 COWARD" if range_width < 0.3 else "⚖️ CAUTIOUS"
        lines.append(f"  {r.name:<25}: range={range_width:.3f}, std={r.dissonance_std:.3f} {status}")
    
    lines.append("")
    lines.append("="*80)
    return "\n".join(lines)


def main():
    """Run all three experiments."""
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Run experiments
    baseline = run_baseline(episodes=1000)
    adaptive = run_adaptive(episodes=1000, learning_rate=0.01)
    stress = run_stress_test(episodes=500)
    
    results = [baseline, adaptive, stress]
    
    # Print summaries
    for r in results:
        print(r.summary())
    
    # Comparison
    comparison = compare_experiments(results)
    print(comparison)
    
    # Save results
    with open(results_dir / "experiment_comparison.json", "w") as f:
        json.dump([{
            'name': r.name,
            'episodes': r.episodes,
            'final_clamp_rate': r.final_clamp_rate,
            'final_alignment_score': r.final_alignment_score,
            'dissonance_range': r.dissonance_range,
            'dissonance_mean': r.dissonance_mean,
            'dissonance_std': r.dissonance_std,
            'final_clamp_clamps': r.final_clamp_clamps,
            'learning_steps': r.learning_steps,
            'mean_reward': r.mean_reward
        } for r in results], f, indent=2)
    
    print(f"\n📊 Results saved to: {results_dir / 'experiment_comparison.json'}")


if __name__ == "__main__":
    main()
