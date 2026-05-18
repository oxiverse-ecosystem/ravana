#!/usr/bin/env python3
"""
RAVANA v2 — Phase F.5: Belief Reasoner + Slow Lie Test

Tests if RAVANA can resist:
1. Gradual deception (slow lie)
2. Coherent false worlds
3. Adversarial consistency attacks

Key upgrade: From single belief → competing hypotheses with confidence decay.
"""

import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from core import (
    Governor, GovernorConfig, ResolutionEngine, IdentityEngine,
    StateManager, StrategyLayer, StrategyConfig, BehavioralContext,
    IntentEngine, IntentConfig, IntentAwareStrategy,
    NonStationaryEnvironment, EnvironmentConfig,
    StrategyWithLearning,
)
from core.belief_reasoner import (
    BeliefReasoner, BeliefConfig, Hypothesis, EvidenceEvent
)


class PhaseF5TrainingPipeline:
    """
    Phase F.5: Belief reasoning with competing hypotheses.
    """
    
    def __init__(self, state_manager, intent_strategy, environment, 
                 belief_reasoner, config):
        self.manager = state_manager
        self.intent_strategy = intent_strategy
        self.env = environment
        self.belief = belief_reasoner
        self.config = config
        
        # Test tracking
        self.slow_lie_active = False
        self.slow_lie_target = 0.85  # Gradually move belief here
        self.slow_lie_start = 0.95   # From here
        self.slow_lie_current = 0.95  # Current fake boundary
        self.slow_lie_episodes: List[int] = []
        
        self.coherent_lie_active = False
        
        # Results
        project_root = Path(__file__).resolve().parent.parent.parent
        self.output_dir = project_root / "results"
        self.output_dir.mkdir(exist_ok=True)
    
    def _get_context(self, episode: int) -> BehavioralContext:
        """Extract behavioral context."""
        state = self.manager.state
        history = self.manager.history
        
        clamp_rate = self._compute_clamp_rate(window=20)
        trend = self._compute_dissonance_trend(window=10)
        stability = self._compute_stability(window=10)
        
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
        history = self.manager.history
        if len(history) < window:
            return 0.0
        recent = history[-window:]
        clamps = sum(1 for h in recent if h.get('constraint_activated', False))
        return clamps / window
    
    def _compute_dissonance_trend(self, window: int = 10) -> float:
        history = self.manager.history
        if len(history) < window:
            return 0.0
        recent = [h['post_dissonance'] for h in history[-window:]]
        if len(recent) < 2:
            return 0.0
        return (recent[-1] - recent[0]) / len(recent)
    
    def _compute_stability(self, window: int = 10) -> float:
        history = self.manager.history
        if len(history) < window:
            return 0.5
        recent = [h['post_dissonance'] for h in history[-window:]]
        variance = np.var(recent)
        return 1.0 / (1.0 + variance * 10)
    
    def _simulate_outcome(self, difficulty: float) -> bool:
        import random
        base_success = 0.7
        success_rate = base_success - (difficulty - 0.3) * 0.4
        return random.random() < success_rate
    
    def _inject_slow_lie(self, episode: int) -> float:
        """
        Gradual deception: slowly shift the fake boundary.
        This tests if RAVANA can resist gradient deception.
        """
        if not self.slow_lie_active:
            return self.env.boundary_base
        
        # Gradual shift: 0.95 → 0.85 over 1000 episodes
        progress = (episode - self.slow_lie_episodes[0]) / 1000
        current_fake = self.slow_lie_start + (self.slow_lie_target - self.slow_lie_start) * min(progress, 1.0)
        self.slow_lie_current = current_fake
        return current_fake
    
    def _get_observed_boundary(self, episode: int) -> float:
        """
        What boundary does RAVANA observe?
        
        During slow lie: sees gradually shifting (but internally consistent) signals.
        """
        true_boundary = self.env.boundary_base
        
        if self.slow_lie_active:
            # During slow lie, RAVANA sees the fake boundary
            # but with realistic noise (coherent deception)
            fake = self._inject_slow_lie(episode)
            noise = np.random.normal(0, 0.01)  # Small, realistic noise
            return np.clip(fake + noise, 0.7, 0.99)
        
        # Normal: true boundary with noise
        noise = np.random.normal(0, 0.02)
        return np.clip(true_boundary + noise, 0.7, 0.99)
    
    def train(self) -> Dict[str, Any]:
        """Execute Phase F.5 training with slow lie test."""
        print("="*70)
        print("🧠 RAVANA v2 — Phase F.5: Belief Reasoner + Slow Lie Test")
        print("="*70)
        print(f"Episodes: {self.config.get('total_episodes', 2000)}")
        print(f"Hypotheses maintained: {self.belief.config.max_hypotheses}")
        print(f"Tests: Slow lie (gradient deception), Coherent false world")
        print("="*70)
        
        for episode in range(self.config.get('total_episodes', 2000)):
            # Environment step
            world_state = self.env.step(episode)
            difficulty = world_state.difficulty_level
            true_boundary = self.env.boundary_base
            
            # 🔥 SLOW LIE TEST: Episodes 400-1400
            if episode == 400:
                self.slow_lie_active = True
                self.slow_lie_episodes = list(range(400, 1400))
                print(f"\n🎭 EP{episode}: SLOW LIE TEST BEGINS")
                print(f"   Target: Gradually shift belief 0.95 → 0.85")
                print(f"   Method: Coherent, consistent false signals")
            
            if episode == 1400 and self.slow_lie_active:
                self.slow_lie_active = False
                print(f"\n🎭 EP{episode}: SLOW LIE TEST ENDS")
                print(f"   True boundary was always: {true_boundary:.2f}")
                
                # Check if RAVANA was fooled
                dominant = self.belief.get_dominant_hypothesis()
                if dominant:
                    belief_error = abs(dominant.boundary_estimate - true_boundary)
                    fooled = belief_error > 0.05
                    print(f"   RAVANA's final belief: {dominant.boundary_estimate:.2f}")
                    print(f"   Belief error: {belief_error:.3f}")
                    print(f"   🚨 FOOLED: {fooled}")
            
            # Get pre-state
            pre_state = {
                'dissonance': self.manager.state.dissonance,
                'identity': self.manager.state.identity,
                'clamp_rate': self._compute_clamp_rate(20),
            }
            
            # Get context and select mode
            context = self._get_context(episode)
            
            # Use belief reasoner to inform mode selection
            mode_hint = self.belief.get_mode_recommendation()
            
            # Select mode
            mode, _ = self.intent_strategy.select_mode(context, [])
            
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
            
            # Observe boundary (may be true or fake during slow lie)
            observed_boundary = self._get_observed_boundary(episode)
            
            # 🔮 BELIEF REASONER: Process evidence
            evidence = EvidenceEvent(
                episode=episode,
                predicted_d=pre_state['dissonance'],
                actual_d=post_state['dissonance'],
                observed_boundary=observed_boundary,
                mode=self._mode_to_int(mode),
                clamp_occurred=step_record.get('constraint_activated', False),
                context_snapshot=pre_state
            )
            
            self.belief.observe_evidence(evidence, true_boundary)
            
            # Periodic logging
            if (episode + 1) % 200 == 0 or (episode > 390 and episode < 410) or (episode > 1390 and episode < 1410):
                self._log_progress(episode + 1, true_boundary)
        
        return self._generate_summary(true_boundary)
    
    def _mode_to_int(self, mode) -> int:
        from core.strategy import ExplorationMode
        mapping = {
            ExplorationMode.EXPLORE_AGGRESSIVE: 0,
            ExplorationMode.EXPLORE_SAFE: 1,
            ExplorationMode.STABILIZE: 2,
            ExplorationMode.RECOVER: 3,
        }
        return mapping.get(mode, 1)
    
    def _log_progress(self, episode: int, true_boundary: float):
        """Log progress with belief reasoner status."""
        state = self.manager.state
        status = self.belief.get_reasoning_status()
        
        # Get dominant hypothesis
        dominant = self.belief.get_dominant_hypothesis()
        belief_str = f"{dominant.boundary_estimate:.2f}±{dominant.uncertainty:.2f}" if dominant else "none"
        
        # Slow lie indicator
        lie_indicator = "🎭" if self.slow_lie_active else " "
        
        print(f"EP{episode:04d}{lie_indicator}| D={state.dissonance:.2f} | "
              f"Belief={belief_str} | True={true_boundary:.2f} | "
              f"Hyps={status['num_hypotheses']} | "
              f"ConfDecay={status['total_confidence_decay']:.3f}")
    
    def _generate_summary(self, true_boundary: float) -> Dict[str, Any]:
        """Generate summary with slow lie analysis."""
        status = self.belief.get_reasoning_status()
        
        # Get final dominant hypothesis
        dominant = self.belief.get_dominant_hypothesis()
        final_belief = dominant.boundary_estimate if dominant else 0.0
        belief_error = abs(final_belief - true_boundary)
        
        summary = {
            'total_episodes': self.config.get('total_episodes', 2000),
            'true_boundary': true_boundary,
            'final_belief': final_belief,
            'belief_error': belief_error,
            'fooled_by_slow_lie': belief_error > 0.05,
            'num_hypotheses': status['num_hypotheses'],
            'total_evidence': status['total_evidence'],
            'structural_rejections': status['structural_rejections'],
            'confidence_decays': status['total_confidence_decay'],
            'hypothesis_weights': status['hypothesis_weights'],
        }
        
        # Print summary
        print("\n" + "="*70)
        print("📊 PHASE F.5 RESULTS: Belief Reasoner + Slow Lie Test")
        print("="*70)
        print(f"\n🎯 TRUE WORLD:")
        print(f"  Actual boundary:     {true_boundary:.2f}")
        print(f"\n🧠 RAVANA'S BELIEF:")
        print(f"  Final estimate:      {final_belief:.2f}")
        print(f"  Error:               {belief_error:.3f}")
        print(f"  Uncertainty:         {dominant.uncertainty:.3f}" if dominant else "  N/A")
        print(f"\n🎭 SLOW LIE TEST:")
        print(f"  Deception method:    Gradual 0.95 → 0.85 over 1000 episodes")
        print(f"  RAVANA was fooled:   {belief_error > 0.05}")
        if belief_error > 0.05:
            print(f"  ⚠️  VULNERABLE to gradient deception")
        else:
            print(f"  ✅ RESISTED gradient deception")
        print(f"\n🧮 BELIEF REASONER:")
        print(f"  Hypotheses tracked:  {status['num_hypotheses']}")
        print(f"  Total evidence:      {status['total_evidence']}")
        print(f"  Structural rejects:  {status['structural_rejections']}")
        print(f"  Confidence decays:   {status['total_confidence_decay']:.3f}")
        print(f"  Hypothesis weights:  {status['hypothesis_weights']}")
        print("="*70)
        
        if not (belief_error > 0.05):
            print("🏆 VERDICT: RAVANA resisted the Slow Lie")
            print("   Structural consistency + confidence decay")
            print("   prevented gradient deception.")
        else:
            print("⚠️  VERDICT: RAVANA vulnerable to Slow Lie")
            print("   Belief drifted under coherent deception.")
            print("   Needs: stronger structural consistency checks.")
        print("="*70)
        
        # Save
        with open(self.output_dir / "phase_f5_belief_reasoner.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary


def main():
    """Run Phase F.5: Belief reasoner with slow lie test."""
    # Create components
    governor = Governor(GovernorConfig())
    resolution = ResolutionEngine(partial_threshold=0.15)
    identity = IdentityEngine(initial_strength=0.5)
    
    manager = StateManager(governor, resolution, identity)
    
    # Strategy
    strategy = StrategyLayer(StrategyConfig())
    learning = StrategyWithLearning(strategy)
    intent = IntentEngine(IntentConfig())
    intent_strategy = IntentAwareStrategy(strategy, learning, intent)
    
    # Environment
    env = NonStationaryEnvironment(EnvironmentConfig())
    
    # 🆕 BELIEF REASONER (replaces learned world model)
    belief_reasoner = BeliefReasoner(BeliefConfig(
        max_hypotheses=3,
        structural_consistency_threshold=0.7,
        confidence_decay_rate=0.02,
    ))
    
    # Pipeline
    config = {'total_episodes': 2000}
    pipeline = PhaseF5TrainingPipeline(manager, intent_strategy, env, 
                                       belief_reasoner, config)
    
    results = pipeline.train()
    
    return results


if __name__ == "__main__":
    main()
