#!/usr/bin/env python3
"""
RAVANA v2 — Phase F: Learned World Model + False World Test

Validates: Is this true intelligence or sophisticated scaffolding?

Test 1: Learned anomaly detection (replacing thresholds)
Test 2: False World Test (resistance to misleading patterns)
"""

import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from core import (
    Governor, GovernorConfig, ResolutionEngine, IdentityEngine,
    StateManager, StrategyLayer, StrategyConfig, BehavioralContext,
    IntentEngine, IntentConfig, IntentAwareStrategy,
    MicroPlanner, PlanningConfig,
    NonStationaryEnvironment, EnvironmentConfig,
    StrategyWithLearning,
)
from core.predictive_world import (
    LearnedWorldModel, WorldModelConfig, FalseWorldTester
)
from core.strategy_learning import LearningConfig


class PhaseFTrainingPipeline:
    """
    Phase F: RAVANA with learned world model.
    
    Key difference: Anomaly detection is learned, not thresholded.
    """
    
    def __init__(self, state_manager, intent_strategy, environment, world_model, config):
        self.manager = state_manager
        self.intent_strategy = intent_strategy
        self.env = environment
        self.world = world_model
        self.config = config
        
        self.false_tester = FalseWorldTester(world_model)
        
        # Tracking
        self.anomalies_old_method = 0  # What threshold would catch
        self.anomalies_new_method = 0  # What learned model catches
        self.learning_events = []
        
        project_root = Path(__file__).resolve().parent.parent.parent
        self.output_dir = project_root / "results"
        self.output_dir.mkdir(exist_ok=True)
    
    def _get_context(self, episode: int) -> BehavioralContext:
        """Extract behavioral context."""
        state = self.manager.state
        history = self.manager.history
        
        # Compute metrics
        clamp_rate = self._compute_clamp_rate(window=20)
        trend = self._compute_dissonance_trend(window=10)
        stability = self._compute_stability(window=10)
        
        # Resolution success rate
        resolution_success = 0.5
        if len(history) >= 10:
            recent = history[-10:]
            successes = sum(1 for h in recent if h['resolution']['full_resolution'])
            resolution_success = successes / len(recent)
        
        return BehavioralContext(
            dissonance=state.dissonance,
            identity=state.identity,
            clamp_rate=clamp_rate,
            dissonance_trend=trend,
            identity_drift=0.0,
            stability=stability,
            dissonance_variance=0.0,
            recent_resolution_success=resolution_success
        )
    
    def _compute_clamp_rate(self, window: int = 20) -> float:
        """Compute recent clamp rate."""
        history = self.manager.history
        if len(history) < window:
            return 0.0
        recent = history[-window:]
        clamps = sum(1 for h in recent if h.get('constraint_activated', False))
        return clamps / window
    
    def _compute_dissonance_trend(self, window: int = 10) -> float:
        """Compute recent dissonance trend."""
        history = self.manager.history
        if len(history) < window:
            return 0.0
        recent = [h['post_dissonance'] for h in history[-window:]]
        if len(recent) < 2:
            return 0.0
        return (recent[-1] - recent[0]) / len(recent)
    
    def _compute_stability(self, window: int = 10) -> float:
        """Compute dissonance stability (inverse of variance)."""
        history = self.manager.history
        if len(history) < window:
            return 0.5
        recent = [h['post_dissonance'] for h in history[-window:]]
        variance = np.var(recent)
        return 1.0 / (1.0 + variance * 10)  # Normalize to [0, 1]
    
    def _extract_clamp_events(self, step_record: Dict) -> List[Dict]:
        """Extract clamp events from step record."""
        events = []
        if step_record.get('constraint_activated', False):
            events.append({
                'episode': step_record.get('episode', 0),
                'reason': step_record.get('reason', 'clamp')
            })
        return events
    
    def train(self) -> Dict[str, Any]:
        """Execute Phase F training with learned world model."""
        print("="*70)
        print("🧠 RAVANA v2 — Phase F: Learned World Model + False World Test")
        print("="*70)
        print(f"Episodes: {self.config.get('total_episodes', 2000)}")
        print(f"Test: Learned surprise detection (replacing thresholds)")
        print(f"Test: False World resistance (can RAVANA resist wrong beliefs?)")
        print("="*70)
        
        for episode in range(self.config.get('total_episodes', 2000)):
            # Environment step (may change boundary, etc.)
            world_state = self.env.step(episode)
            
            # Extract difficulty from world state
            difficulty = world_state.difficulty_level
            
            # Get pre-state
            pre_state = {
                'dissonance': self.manager.state.dissonance,
                'identity': self.manager.state.identity,
                'clamp_rate': self._compute_clamp_rate(20),
            }
            
            # Get context and select mode
            context = self._get_context(episode)
            clamp_events = self._extract_clamp_events({})  # Empty for now
            
            mode, mode_info = self.intent_strategy.select_mode(context, clamp_events)
            
            # Execute step
            correctness = self._simulate_outcome(difficulty)
            
            step_record = self.manager.step(
                correctness=correctness,
                difficulty=difficulty,
                debug=False
            )
            
            # Capture post-state
            post_state = {
                'dissonance': self.manager.state.dissonance,
                'identity': self.manager.state.identity,
                'clamp_rate': self._compute_clamp_rate(20),
            }
            
            # 🔮 LEARNED WORLD MODEL: Observe and learn
            mode_int = self._mode_to_int(mode)
            anomaly = self.world.observe(
                episode, pre_state, mode_int, post_state, 
                self.env.boundary_base
            )
            
            if anomaly:
                self.anomalies_new_method += 1
                if episode % 100 == 0 or episode > 1950:
                    print(f"  EP{episode:04d} | 🚨 LEARNED ANOMALY | "
                          f"Surprise={anomaly.surprise:.3f} | "
                          f"Threshold={self.world.surprise_threshold:.3f}")
            
            # Check what OLD threshold method would catch
            if self._old_threshold_detection(pre_state, post_state, episode):
                self.anomalies_old_method += 1
            
            # Periodically inject false patterns (False World Test)
            if episode in [500, 1000, 1500]:
                resisted = self.false_tester.inject_false_boundary_shift(
                    episode, fake_boundary=0.60
                )
                print(f"  EP{episode:04d} | 🎭 FALSE WORLD TEST | "
                      f"Resisted={resisted}")
            
            # Update intent strategy
            new_clamps = self._extract_clamp_events(step_record)
            self.intent_strategy.update_after_step(
                episode, pre_state, post_state, mode, new_clamps
            )
            
            # Periodic logging
            if (episode + 1) % 200 == 0:
                self._log_progress(episode + 1)
        
        return self._generate_summary()
    
    def _mode_to_int(self, mode) -> int:
        """Convert mode to integer for encoding."""
        from core.strategy import ExplorationMode
        mapping = {
            ExplorationMode.EXPLORE_AGGRESSIVE: 0,
            ExplorationMode.EXPLORE_SAFE: 1,
            ExplorationMode.STABILIZE: 2,
            ExplorationMode.RECOVER: 3,
        }
        return mapping.get(mode, 1)
    
    def _old_threshold_detection(self, pre_state: Dict, post_state: Dict, episode: int) -> bool:
        """
        Threshold-based detection (old method for comparison).
        
        Returns True if old method would flag anomaly.
        """
        # Simple threshold: boundary proximity + unexpected jump
        boundary = 0.95  # Assumed (RAVANA doesn't know this)
        d_jump = abs(post_state['dissonance'] - pre_state['dissonance'])
        
        # Old method: flag if big jump near assumed boundary
        if d_jump > 0.15 and post_state['dissonance'] > boundary * 0.85:
            return True
        
        return False
    
    def _simulate_outcome(self, difficulty: float) -> bool:
        """Simulate episode outcome based on difficulty."""
        import random
        # Higher difficulty = lower success rate
        base_success = 0.7
        success_rate = base_success - (difficulty - 0.3) * 0.4
        return random.random() < success_rate
    
    def _log_progress(self, episode: int):
        """Log progress with world model info."""
        state = self.manager.state
        status = self.world.get_world_model_status()
        
        print(f"EP{episode:04d} | D={state.dissonance:.2f} | I={state.identity:.2f} | "
              f"Belief={status['belief']['boundary_estimate']:.2f} | "
              f"Conf={status['belief']['confidence']:.2f} | "
              f"Surprises={self.anomalies_new_method}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary comparing old vs new methods."""
        status = self.world.get_world_model_status()
        
        summary = {
            'total_episodes': self.config.get('total_episodes', 2000),
            'anomaly_detection': {
                'old_threshold_method': self.anomalies_old_method,
                'new_learned_method': self.anomalies_new_method,
                'difference': self.anomalies_new_method - self.anomalies_old_method,
                'learned_surprise_threshold': status['surprise_threshold'],
                'prediction_uncertainty': status['prediction_uncertainty'],
            },
            'false_world_test': {
                'patterns_injected': self.false_tester.false_patterns_injected,
                'patterns_resisted': self.false_tester.false_patterns_resisted,
                'resistance_score': self.false_tester.get_resistance_score(),
                'corruption_events': len(self.false_tester.belief_corruption_events),
            },
            'final_belief': status['belief'],
            'belief_updates': len(self.world.belief_history),
            'world_model': status,
        }
        
        # Print summary
        print("\n" + "="*70)
        print("📊 PHASE F RESULTS: Old vs Learned World Model")
        print("="*70)
        print(f"\n🔍 ANOMALY DETECTION:")
        print(f"  Old threshold method:  {self.anomalies_old_method} detections")
        print(f"  New learned method:    {self.anomalies_new_method} detections")
        print(f"  Difference:            {self.anomalies_new_method - self.anomalies_old_method:+d}")
        print(f"\n🧠 LEARNED MODEL:")
        print(f"  Surprise threshold:    {status['surprise_threshold']:.4f} (adaptive)")
        print(f"  Prediction uncertainty:{status['prediction_uncertainty']:.4f}")
        print(f"  Belief updates:        {len(self.world.belief_history)}")
        print(f"\n🎭 FALSE WORLD TEST:")
        print(f"  Patterns injected:     {self.false_tester.false_patterns_injected}")
        print(f"  Patterns resisted:     {self.false_tester.false_patterns_resisted}")
        print(f"  Resistance score:      {self.false_tester.get_resistance_score():.1%}")
        print(f"  Belief corruptions:    {len(self.false_tester.belief_corruption_events)}")
        print(f"\n📍 FINAL BELIEF:")
        print(f"  Boundary estimate:     {status['belief']['boundary_estimate']:.2f}")
        print(f"  Confidence:            {status['belief']['confidence']:.2f}")
        print("="*70)
        
        if self.false_tester.get_resistance_score() >= 0.66:
            print("🏆 VERDICT: RAVANA shows RESISTANCE to false patterns")
            print("   Suggests: Belief formation with inertia, not reactive")
        else:
            print("⚠️ VERDICT: RAVANA vulnerable to false patterns")
            print("   Suggests: Needs stronger confirmation mechanism")
        print("="*70)
        
        # Save
        with open(self.output_dir / "phase_f_world_model.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary


def main():
    """Run Phase F: Learned world model test."""
    # Create components
    governor = Governor(GovernorConfig())
    resolution = ResolutionEngine(partial_threshold=0.15)
    identity = IdentityEngine(initial_strength=0.5)
    
    manager = StateManager(governor, resolution, identity)
    
    # Strategy and intent
    strategy = StrategyLayer(StrategyConfig())
    learning = StrategyWithLearning(strategy)  # Remove LearningConfig, let it use default
    intent = IntentEngine(IntentConfig())
    planner = MicroPlanner(PlanningConfig())
    intent_strategy = IntentAwareStrategy(strategy, learning, intent)
    
    # Environment
    env = NonStationaryEnvironment(EnvironmentConfig())
    
    # NEW: Learned world model
    world_model = LearnedWorldModel(WorldModelConfig())
    
    # Pipeline
    config = {'total_episodes': 2000, 'debug_first_n': 20}
    pipeline = PhaseFTrainingPipeline(manager, intent_strategy, env, world_model, config)
    
    results = pipeline.train()
    
    return results


if __name__ == "__main__":
    main()
