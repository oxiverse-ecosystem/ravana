#!/usr/bin/env python3
"""
RAVANA v2 — Phase D Training Entry Point
Intent Engine: Dynamic objectives that evolve based on outcomes.

This is where the system starts to "want" things.
"""

import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core import (
    Governor, GovernorConfig, ResolutionEngine, IdentityEngine, StateManager
)
from core.adaptation import PolicyTweakLayer, AdaptiveGovernorBridge, AdaptationConfig
from core.strategy_learning import StrategyLearningLayer, ModeOutcome, LearningConfig
from core.strategy import StrategyLayer, StrategyConfig, ExplorationMode, BehavioralContext
from core.intent import IntentEngine, IntentConfig, IntentAwareStrategy
from training.pipeline import TrainingPipeline, TrainingConfig
import numpy as np


def main():
    """Execute Phase D training with full intent engine."""
    # Phase A: Physics
    governor = Governor(GovernorConfig(
        max_dissonance=0.95, min_dissonance=0.15,
        max_identity=0.95, min_identity=0.10,
    ))
    resolution = ResolutionEngine(partial_threshold=0.15)
    identity = IdentityEngine(initial_strength=0.5)
    manager = StateManager(governor, resolution, identity)
    
    # Phase B.5: Strategy
    strategy = StrategyLayer(StrategyConfig())
    
    # Phase C: Learning
    learning = StrategyLearningLayer(LearningConfig())
    
    # 🎯 Phase D: Intent Engine (the new part)
    intent = IntentEngine(IntentConfig())
    
    # Bridge all three
    intent_strategy = IntentAwareStrategy(strategy, learning, intent)
    
    # Training config
    config = TrainingConfig(
        total_episodes=2000,
        log_interval=100,
        debug_first_n=20,
    )
    
    # Modified pipeline with intent
    pipeline = IntentTrainingPipeline(manager, intent_strategy, config)
    results = pipeline.train()
    
    # Print final intent state
    print("\n" + "="*60)
    print("FINAL INTENT STATE")
    print("="*60)
    intent_status = intent.get_current_intent()
    print(f"Dominant objective: {intent_status['dominant_objective']} (weight: {intent_status['dominant_weight']:.2f})")
    print(f"Objective weights: {intent_status['objective_weights']}")
    print(f"Objective trends: {intent_status['objective_trends']}")
    
    return results


class IntentTrainingPipeline(TrainingPipeline):
    """Extended pipeline with intent-driven strategy."""
    
    def __init__(self, manager, intent_strategy, config):
        super().__init__(manager, config)
        self.intent_strategy = intent_strategy
    
    def train(self):
        """Execute training with intent-aware mode selection."""
        print("="*60)
        print("RAVANA v2 — Phase D: Intent Engine")
        print("="*60)
        print(f"Total episodes: {self.config.total_episodes:,}")
        print(f"Governor: CENTRAL")
        print(f"Strategy: ADAPTIVE + LEARNED + INTENT")
        print("="*60)
        
        import time
        start_time = time.time()
        
        for episode in range(self.config.total_episodes):
            # Capture pre-state
            pre_state = {
                'dissonance': self.manager.state.dissonance,
                'identity': self.manager.state.identity,
            }
            
            # Get context for mode selection
            context = self._get_context()
            clamp_events = self._get_recent_clamps(episode)
            
            # 🎯 INTENT-DRIVEN: Select mode with full intelligence
            mode, mode_info = self.intent_strategy.select_mode(context, clamp_events)
            
            # Execute step with this mode's policy bias
            difficulty = self._compute_difficulty(episode)
            correctness = self._simulate_outcome(difficulty, mode)
            
            debug = episode < self.config.debug_first_n
            step_record = self._execute_step(correctness, difficulty, mode, debug)
            
            # Capture post-state
            post_state = {
                'dissonance': self.manager.state.dissonance,
                'identity': self.manager.state.identity,
            }
            
            # Update all layers including intent
            new_clamps = self._extract_clamp_events(step_record)
            self.intent_strategy.update_after_step(
                episode, pre_state, post_state, mode, new_clamps
            )
            
            # Logging
            if (episode + 1) % self.config.log_interval == 0:
                self._log_intent_progress(episode + 1, mode, mode_info)
        
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Intent-driven training complete: {elapsed:.1f}s")
        print(f"Final: D={self.manager.state.dissonance:.3f} I={self.manager.state.identity:.3f}")
        print(f"{'='*60}")
        
        return self._generate_intent_summary(elapsed)
    
    def _get_context(self) -> BehavioralContext:
        """Extract behavioral context for strategy layer."""
        state = self.manager.state
        
        # Compute trend and variance
        history = self.manager.history
        window = 10
        
        if len(history) >= window:
            recent_d = [h['post_dissonance'] for h in history[-window:]]
            dissonance_variance = np.var(recent_d)
            
            # Trend: compare first half to second half
            early_mean = np.mean(recent_d[:5])
            late_mean = np.mean(recent_d[5:])
            dissonance_trend = late_mean - early_mean
        else:
            dissonance_variance = 0.1
            dissonance_trend = 0.0
        
        # Identity drift
        if len(history) >= window:
            recent_i = [h['post_identity'] for h in history[-window:]]
            early_i = np.mean(recent_i[:5])
            late_i = np.mean(recent_i[5:])
            identity_drift = late_i - early_i
        else:
            identity_drift = 0.0
        
        # Clamp rate
        clamp_window = min(20, len(history))
        if clamp_window > 0:
            recent_caps = sum(1 for h in history[-clamp_window:] if h.get('constraint_activated', False))
            clamp_rate = recent_caps / clamp_window
        else:
            clamp_rate = 0.0
        
        return BehavioralContext(
            clamp_rate=clamp_rate,
            dissonance=state.dissonance,
            identity=state.identity,
            dissonance_trend=dissonance_trend,
            identity_drift=identity_drift,
            stability=dissonance_variance,
            dissonance_variance=dissonance_variance
        )
    
    def _compute_recent_variance(self, window=10):
        """Compute recent dissonance variance."""
        import numpy as np
        if len(self.manager.history) < window:
            return 0.0
        recent_d = [h['post_dissonance'] for h in self.manager.history[-window:]]
        return np.var(recent_d)
    
    def _compute_recent_clamp_rate(self, window=20):
        """Compute recent clamp rate."""
        if len(self.manager.history) < window:
            return 0.0
        recent = self.manager.history[-window:]
        clamps = sum(1 for h in recent if h.get('constraint_activated', False))
        return clamps / len(recent)
    
    def _get_recent_clamps(self, episode, window=5):
        """Get recent clamp events for strategy."""
        clamps = []
        for h in self.manager.history[-window:]:
            if h.get('constraint_activated'):
                clamps.append({'episode': h.get('episode', 0)})
        return clamps
    
    def _simulate_outcome(self, difficulty, mode):
        """Simulate with mode-dependent success rates."""
        import random
        
        # Mode affects base success rate
        base_success = 0.7
        mode_modifiers = {
            'explore_aggressive': -0.15,  # Riskier
            'explore_safe': 0.05,         # Safer
            'stabilize': 0.10,            # Most reliable
            'recover': 0.20,              # High success when in recovery
        }
        modifier = mode_modifiers.get(mode.value, 0.0)
        success_rate = base_success + modifier - (difficulty - 0.3) * 0.4
        
        return random.random() < max(0.1, min(0.95, success_rate))
    
    def _execute_step(self, correctness, difficulty, mode, debug):
        """Execute step with mode-specific policy bias."""
        # Mode affects how we interpret correctness
        # This is a placeholder - real implementation would modify the step
        return self.manager.step(correctness, difficulty, debug)
    
    def _extract_clamp_events(self, step_record):
        """Extract clamp events from step record."""
        if step_record.get('constraint_activated'):
            return [{'episode': step_record.get('episode', 0)}]
        return []
    
    def _log_intent_progress(self, episode, mode, mode_info):
        """Log with intent information."""
        state = self.manager.state
        intent = mode_info.get('dominant_objective', 'unknown')
        bias = mode_info.get('intent_bias', 0.0)
        
        print(f"EP{episode:,} | "
              f"D={state.dissonance:.2f} | "
              f"I={state.identity:.2f} | "
              f"Mode:{mode.value[:6]} | "
              f"Wants:{intent[:8]} | "
              f"Bias:{bias:+.2f}")
    
    def _generate_intent_summary(self, elapsed):
        """Generate summary including intent evolution."""
        status = self.intent_strategy.get_full_status()
        
        summary = {
            "phase": "D",
            "total_episodes": self.config.total_episodes,
            "elapsed_seconds": elapsed,
            "final_state": self.manager.get_status()['state'],
            "intent_evolution": self.intent_strategy.intent.weight_history[-10:],
            "final_intent": status['intent'],
            "strategy_analytics": status['strategy'],
            "learning_summary": status['learning'],
        }
        
        # Save
        import json
        from pathlib import Path
        project_root = Path(__file__).resolve().parent.parent.parent
        output_dir = project_root / "results"
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "intent_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary


if __name__ == "__main__":
    main()
