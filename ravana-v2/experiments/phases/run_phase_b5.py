#!/usr/bin/env python3
"""
RAVANA v2 — Phase B.5 Training Entry Point
Strategy Layer: Deliberate mode selection + adaptation.
"""

import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core import Governor, GovernorConfig, ResolutionEngine, IdentityEngine, StateManager
from core.adaptation import PolicyTweakLayer, AdaptiveGovernorBridge, AdaptationConfig
from core.strategy import StrategyLayer, StrategyConfig, ExplorationMode
from training.pipeline import TrainingPipeline, TrainingConfig


def main():
    """Execute Phase B.5 training with strategy layer."""
    print("=" * 70)
    print("RAVANA v2 — Phase B.5: STRATEGY LAYER")
    print("=" * 70)
    print("🧠 Deliberate mode selection: choosing HOW to explore")
    print("-" * 70)
    
    # Core components
    governor = Governor(GovernorConfig(
        max_dissonance=0.95,
        min_dissonance=0.15,
        max_identity=0.95,
        min_identity=0.10,
        dissonance_target=0.45,
        identity_target=0.65,
    ))
    
    resolution = ResolutionEngine(partial_threshold=0.15)
    identity = IdentityEngine(initial_strength=0.5)
    
    # State manager
    manager = StateManager(governor, resolution, identity)
    
    # Phase B: Adaptation layer
    adaptation_config = AdaptationConfig(
        learning_rate=0.01,
        exploration_bonus=0.15,  # Slightly higher for strategy exploration
        clamp_penalty=1.0,
    )
    tweak_layer = PolicyTweakLayer(adaptation_config)
    adaptive_bridge = AdaptiveGovernorBridge(governor, tweak_layer)
    
    # Phase B.5: Strategy layer (the new component)
    strategy_config = StrategyConfig(
        crisis_clamp_rate=0.15,
        high_exploration_threshold=0.25,
        boundary_proximity=0.75,
        stability_threshold=0.02,
    )
    strategy = StrategyLayer(strategy_config)
    
    # Training configuration
    config = TrainingConfig(
        total_episodes=2000,  # Longer to see strategy evolution
        log_interval=100,
        debug_first_n=30,
    )
    
    print(f"Episodes: {config.total_episodes:,}")
    print(f"Adaptation: ENABLED (learning_rate={adaptation_config.learning_rate})")
    print(f"Strategy Layer: ENABLED (4 modes)")
    print(f"Modes: EXPLORE_AGGRESSIVE | EXPLORE_SAFE | STABILIZE | RECOVER")
    print("-" * 70)
    
    # Create pipeline
    pipeline = TrainingPipeline(manager, config)
    
    # Inject strategy step into training loop
    original_step = manager.step
    
    def strategy_aware_step(correctness, difficulty, debug=False):
        """Wrap state manager step with strategy layer."""
        # Compute behavioral context
        context = strategy.compute_context(governor, manager)
        
        # Select exploration mode
        selection = strategy.select_mode(context, manager.state.episode)
        
        # Log mode selection (if debug)
        if debug or (manager.state.episode % 100 == 0 and manager.state.episode > 0):
            print(f"  [STRATEGY] Mode: {selection.mode.value:20s} | "
                  f"Confidence: {selection.confidence:.2f} | "
                  f"Reason: {selection.reason}")
        
        # Execute original step
        step_record = original_step(correctness, difficulty, debug)
        
        # Record mode in step record
        step_record['strategy_mode'] = selection.mode.value
        step_record['strategy_confidence'] = selection.confidence
        
        return step_record
    
    # Replace step with strategy-aware version
    manager.step = strategy_aware_step
    
    # Run training
    results = pipeline.train()
    
    # Add strategy analytics to results
    results['strategy_analytics'] = strategy.get_mode_analytics()
    
    # Print strategy summary
    print("\n" + "=" * 70)
    print("STRATEGY LAYER ANALYTICS")
    print("=" * 70)
    analytics = strategy.get_mode_analytics()
    
    print(f"Mode Distribution:")
    for mode, count in analytics['mode_distribution'].items():
        pct = count / sum(analytics['mode_distribution'].values()) * 100
        print(f"  {mode:20s}: {count:4d} episodes ({pct:5.1f}%)")
    
    print(f"\nMode Switches: {analytics['mode_switches']}")
    print(f"Switch Rate: {analytics['switch_rate']:.3f} per episode")
    
    print(f"\nAverage Mode Duration:")
    for mode, duration in analytics['avg_mode_duration'].items():
        if duration > 0:
            print(f"  {mode:20s}: {duration:.1f} episodes")
    
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
