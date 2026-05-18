#!/usr/bin/env python3
"""
RAVANA v2 — Phase G.5: Surgical Probing Test
From "uncertainty triggered" → "KL-maximizing experiment design"
"""

import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core import (
    Governor, GovernorConfig,
    ResolutionEngine, IdentityEngine, StateManager,
    StrategyWithLearning, StrategyLayer, StrategyConfig,
    StrategyLearningLayer, LearningConfig, BehavioralContext,
    IntentEngine, IntentConfig, IntentAwareStrategy,
    BeliefReasoner, BeliefConfig, Hypothesis,
    ActiveEpistemology, VoIConfig,
    SurgicalProbeSelector, SurgicalProbeConfig, ProbeType
)
from core.environment import NonStationaryEnvironment, EnvironmentConfig

import numpy as np
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class G5Config:
    """Config for surgical probing test."""
    total_episodes: int = 500
    debug_first_n: int = 20
    log_interval: int = 50
    
    # Test scenario: Two hypotheses with different boundary predictions
    true_boundary: float = 0.75
    alternative_boundary: float = 0.90
    
    # Initial confidences (alternative should be competitive)
    dominant_confidence: float = 0.50
    alternative_confidence: float = 0.45

class SurgicalProbeTest:
    """Test surgical probing with KL-maximizing probe selection."""
    
    def __init__(
        self,
        manager: StateManager,
        intent_strategy: IntentAwareStrategy,
        belief: BeliefReasoner,
        epistemology: ActiveEpistemology,
        surgical: SurgicalProbeSelector,
        env: NonStationaryEnvironment,
        config: G5Config = None
    ):
        self.manager = manager
        self.intent_strategy = intent_strategy
        self.belief = belief
        self.epistemology = epistemology
        self.surgical = surgical
        self.env = env
        self.config = config or G5Config()
        
        # Tracking
        self.probe_history: List[Dict] = []
        self.kl_history: List[float] = []
        self.surgical_probe_count: int = 0
        self.generic_probe_count: int = 0
        
    def _get_context(self) -> BehavioralContext:
        """Extract behavioral context (simplified)."""
        history = self.manager.history[-10:] if len(self.manager.history) >= 10 else self.manager.history
        
        if len(history) >= 3:
            recent_d = [h['post_dissonance'] for h in history[-3:]]
            d_trend = (recent_d[-1] - recent_d[0]) / len(recent_d)
        else:
            d_trend = 0.0
        
        recent_clamps = sum(1 for h in history if h.get('constraint_activated', False))
        clamp_rate = recent_clamps / len(history) if history else 0.0
        
        if len(history) >= 5:
            recent_d_full = [h['post_dissonance'] for h in history[-5:]]
            d_variance = np.var(recent_d_full)
        else:
            d_variance = 0.0
        
        # Calculate identity drift (simplified)
        if len(history) >= 2:
            recent_i = [h['post_identity'] for h in history[-2:]]
            i_drift = recent_i[-1] - recent_i[0]
        else:
            i_drift = 0.0
        
        return BehavioralContext(
            dissonance=self.manager.state.dissonance,
            identity=self.manager.state.identity,
            clamp_rate=clamp_rate,
            dissonance_trend=d_trend,
            identity_drift=i_drift,
            stability=1.0 - min(1.0, d_variance * 10),
            dissonance_variance=d_variance,
            recent_resolution_success=0.6
        )

    def train(self) -> Dict[str, Any]:
        """Run surgical probing test."""
        print("=" * 70)
        print("🧠 RAVANA v2 — Phase G.5: Surgical Probing")
        print("=" * 70)
        print("Test: Does RAVANA select probes that maximally separate H1 vs H2?")
        print("=" * 70)
        
        # Spawn two competing hypotheses
        self.belief.hypotheses = []
        
        h1 = Hypothesis(
            id=1,
            boundary_estimate=self.env.true_boundary,
            confidence=self.config.dominant_confidence,
            uncertainty=0.10,
            created_episode=0
        )
        self.belief.hypotheses.append(h1)
        
        h2 = Hypothesis(
            id=2,
            boundary_estimate=self.config.alternative_boundary,
            confidence=self.config.alternative_confidence,
            uncertainty=0.12,
            created_episode=0
        )
        self.belief.hypotheses.append(h2)
        
        print(f"\n🎭 SURGICAL PROBE TEST")
        print(f"   True boundary: {self.env.true_boundary}")
        print(f"   H1 (dominant): {h1.boundary_estimate} @ {h1.confidence:.2f} conf")
        print(f"   H2 (alternative): {h2.boundary_estimate} @ {h2.confidence:.2f} conf")
        print(f"   → KL-maximizing probe should separate these two")
        
        for episode in range(self.config.total_episodes):
            # Environment step
            world_state = self.env.step(episode)
            
            # Get context
            context = self._get_context()
            
            # SURGICAL PROBE SELECTION
            current_state = {
                'dissonance': self.manager.state.dissonance,
                'identity': self.manager.state.identity
            }
            
            probe_type, probe_meta = self.surgical.select_surgical_probe(
                self.belief.hypotheses,
                current_state,
                episode
            )
            
            # Determine action
            if probe_type:
                # Execute surgical probe
                action = f"surgical_{probe_type.value}"
                self.surgical_probe_count += 1
                
                # Calculate actual KL achieved (for learning)
                pre_belief = self.belief.current_belief
                pre_uncertainty = self.belief.current_uncertainty
                
                # Simulate probe effect (simplified)
                difficulty = 0.5 + np.random.normal(0, 0.1)
                difficulty = np.clip(difficulty, 0.2, 0.9)
                
                # Execute step with probe characteristics
                if "aggressive" in action:
                    correctness = np.random.random() < (0.6 - difficulty * 0.3)
                elif "high" in action:
                    correctness = np.random.random() < (0.5 - difficulty * 0.4)
                else:
                    correctness = np.random.random() < (0.7 - difficulty * 0.2)
                
                step_record = self.manager.step(
                    correctness=correctness,
                    difficulty=difficulty,
                    debug=episode < self.config.debug_first_n
                )
                
                # Calculate actual separation achieved
                post_belief = self.belief.current_belief
                belief_shift = abs(post_belief - pre_belief)
                
                # Record probe result
                self.surgical.record_probe_result(
                    probe_type=probe_type,
                    episode=episode,
                    actual_info_gain=belief_shift,
                    hypothesis_separation_achieved=belief_shift
                )
                
                self.kl_history.append(belief_shift)
                
                marker = "🔬"
                reason = f"KL={probe_meta.get('expected_kl', 0):.3f}"
                
            else:
                # Normal exploration
                action = "explore_normal"
                self.generic_probe_count += 1
                
                difficulty = 0.5 + np.random.normal(0, 0.1)
                difficulty = np.clip(difficulty, 0.2, 0.9)
                correctness = np.random.random() < (0.7 - difficulty * 0.2)
                
                step_record = self.manager.step(
                    correctness=correctness,
                    difficulty=difficulty,
                    debug=episode < self.config.debug_first_n
                )
                
                marker = "  "
                reason = probe_meta.get('reason', 'exploitation')
            
            # Log
            if (episode + 1) % self.config.log_interval == 0 or episode < self.config.debug_first_n:
                belief = self.belief.current_belief
                uncertainty = self.belief.current_uncertainty
                print(f"{marker}EP{episode+1:04d} | D={self.manager.state.dissonance:.2f} | "
                      f"Belief={belief:.2f}±{uncertainty:.2f} | "
                      f"Action={action:20s} | Reason={reason}")
        
        # Summary
        print("\n" + "=" * 70)
        print("📊 SURGICAL PROBE RESULTS")
        print("=" * 70)
        
        # Surgical analytics
        analytics = self.surgical.get_surgical_analytics()
        
        print(f"\n🔬 SURGICAL PROBING:")
        print(f"   Total surgical probes: {self.surgical_probe_count}")
        print(f"   Generic probes: {self.generic_probe_count}")
        print(f"   Surgical ratio: {self.surgical_probe_count / max(1, self.surgical_probe_count + self.generic_probe_count):.1%}")
        print(f"   Avg expected KL: {analytics.get('avg_expected_kl', 0):.3f}")
        print(f"   Avg actual separation: {analytics.get('avg_actual_separation', 0):.3f}")
        print(f"   KL calibration error: {analytics.get('kl_calibration_error', 0):.3f}")
        
        # Probe type effectiveness
        print(f"\n🎯 PROBE TYPE EFFECTIVENESS:")
        for probe_type, data in analytics.get('probe_type_effectiveness', {}).items():
            print(f"   {probe_type:20s}: {data['success_rate']:.1%} success ({data['n_probes']} probes)")
        
        # Final belief state
        final_belief = self.belief.current_belief
        final_uncertainty = self.belief.current_uncertainty
        
        print(f"\n🧠 BELIEF EVOLUTION:")
        print(f"   Final belief: {final_belief:.3f} (true: {self.env.true_boundary})")
        print(f"   Final uncertainty: {final_uncertainty:.3f}")
        print(f"   Remaining hypotheses: {len(self.belief.hypotheses)}")
        
        # Verdict
        surgical_ratio = self.surgical_probe_count / max(1, self.surgical_probe_count + self.generic_probe_count)
        
        if surgical_ratio > 0.2 and analytics.get('avg_actual_separation', 0) > 0.05:
            verdict = "🟢 SURGICAL THINKER"
            verdict_desc = "RAVANA selects KL-maximizing probes"
        elif self.surgical_probe_count > 10:
            verdict = "🟡 EMERGING SURGEON"
            verdict_desc = "Surgical probing present but not dominant"
        else:
            verdict = "🔵 STANDARD THINKER"
            verdict_desc = "Mostly generic exploration"
        
        print(f"\n{'='*70}")
        print(f"🏁 VERDICT: {verdict}")
        print(f"   {verdict_desc}")
        print(f"{'='*70}")
        
        # Save results
        results = {
            "phase": "G.5",
            "surgical_probes": self.surgical_probe_count,
            "generic_probes": self.generic_probe_count,
            "surgical_ratio": surgical_ratio,
            "analytics": analytics,
            "final_belief": final_belief,
            "final_uncertainty": final_uncertainty,
            "verdict": verdict,
            "verdict_description": verdict_desc
        }
        
        with open("results/g5_surgical_probes.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📊 Results saved to: results/g5_surgical_probes.json")
        print("="*70)
        
        return results


def main():
    """Entry point for Phase G.5 test."""
    # Components
    governor = Governor(GovernorConfig(
        max_dissonance=0.95, min_dissonance=0.15,
        max_identity=0.95, min_identity=0.10
    ))
    
    resolution = ResolutionEngine(partial_threshold=0.15)
    identity = IdentityEngine(initial_strength=0.5)
    manager = StateManager(governor, resolution, identity)
    
    # Strategy
    strategy = StrategyWithLearning(
        StrategyLayer(StrategyConfig()),
        StrategyLearningLayer(LearningConfig())
    )
    
    # Intent
    intent = IntentEngine(IntentConfig())
    intent_strategy = IntentAwareStrategy(strategy, None, intent)
    
    # Belief
    belief = BeliefReasoner(BeliefConfig())
    
    # Epistemology
    epistemology = ActiveEpistemology(belief, VoIConfig())
    
    # Surgical probe selector
    surgical = SurgicalProbeSelector(SurgicalProbeConfig())
    
    # Environment
    env = NonStationaryEnvironment(EnvironmentConfig(
        difficulty_cycle_period=200,
        cycle_amplitude=0.15
    ))
    env.true_boundary = 0.75
    env.alternative_boundary = 0.90
    
    # Config
    config = G5Config()
    
    # Run test
    pipeline = SurgicalProbeTest(
        manager, intent_strategy, belief,
        epistemology, surgical, env, config
    )
    
    results = pipeline.train()
    return results


if __name__ == "__main__":
    main()
