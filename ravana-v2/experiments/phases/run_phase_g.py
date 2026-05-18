#!/usr/bin/env python3
"""
RAVANA v2 — Phase G: Active Epistemology
Active Discovery Test
"""

import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import random
import numpy as np
from typing import Dict, Any, List

from core import (
    Governor, GovernorConfig,
    ResolutionEngine,
    IdentityEngine,
    StateManager,
    StrategyLayer, StrategyConfig, ExplorationMode, BehavioralContext,
    StrategyLearningLayer, LearningConfig,
    StrategyWithLearning,
    IntentEngine, IntentConfig, IntentAwareStrategy, SystemObjective,
    BeliefReasoner, BeliefConfig,
    ActiveEpistemology, VoIConfig, InformationGainMethod
)


class PartialWorldEnvironment:
    """Creates a 'consistent but partial' world."""
    
    def __init__(self, true_boundary: float = 0.75, alternative_boundary: float = 0.95):
        self.true_boundary = true_boundary
        self.alternative_boundary = alternative_boundary
        self.episode_count = 0
        self.ambiguity_zone_low = min(true_boundary, alternative_boundary) - 0.05
        self.ambiguity_zone_high = min(true_boundary, alternative_boundary) + 0.05
        self.distinction_threshold = max(true_boundary, alternative_boundary) + 0.02
        
    def step(self, episode: int, dissonance: float) -> Dict[str, Any]:
        """Generate observation that tests hypothesis disambiguation."""
        self.episode_count = episode
        in_ambiguity_zone = self.ambiguity_zone_low <= dissonance <= self.ambiguity_zone_high
        
        if in_ambiguity_zone:
            difficulty = 0.5 + random.gauss(0, 0.1)
            observation_quality = "ambiguous"
        elif dissonance > self.distinction_threshold:
            difficulty = 0.9
            observation_quality = "distinguishing"
        else:
            difficulty = 0.3
            observation_quality = "consistent"
        
        return {
            'difficulty': np.clip(difficulty, 0.1, 0.95),
            'true_boundary': self.true_boundary,
            'alternative_boundary': self.alternative_boundary,
            'observation_quality': observation_quality,
            'in_ambiguity_zone': in_ambiguity_zone
        }
    
    def get_true_boundary(self) -> float:
        return self.true_boundary


class PhaseGTrainingPipeline:
    """Phase G: Active Epistemology training with discovery testing."""
    
    def __init__(self, state_manager, intent_strategy, belief_reasoner, 
                 active_epistemology, env, config=None):
        self.manager = state_manager
        self.intent_strategy = intent_strategy
        self.belief = belief_reasoner
        self.epistemology = active_epistemology
        self.env = env
        self.config = config or {'total_episodes': 1000, 'log_interval': 100}
        self.probes_when_uncertain = 0
        self.total_uncertain_episodes = 0
        
    def _simulate_outcome(self, difficulty: float) -> bool:
        base_success = 0.7
        success_rate = base_success - (difficulty - 0.3) * 0.4
        return random.random() < success_rate
    
    def train(self) -> Dict[str, Any]:
        """Execute active epistemology training."""
        print("=" * 70)
        print("🧠 RAVANA v2 — Phase G: Active Epistemology")
        print("=" * 70)
        print("Test: Does RAVANA intentionally act to resolve uncertainty?")
        print("=" * 70)
        
        # Manually spawn competing hypotheses
        from core.belief_reasoner import Hypothesis
        alt_hyp = Hypothesis(
            id=len(self.belief.hypotheses) + 1,
            boundary_estimate=self.env.alternative_boundary,
            confidence=0.45,  # Higher initial confidence to compete
            uncertainty=0.12,
            created_episode=0
        )
        self.belief.hypotheses.append(alt_hyp)
        
        print(f"\n🎭 ACTIVE DISCOVERY TEST")
        print(f"   True boundary: {self.env.true_boundary}")
        print(f"   Alternative hypothesis: {self.env.alternative_boundary}")
        print(f"   Initial: RAVANA maintains BOTH hypotheses")
        print(f"   Test: Will RAVANA probe to disambiguate?\n")
        
        for episode in range(self.config['total_episodes']):
            pre_state = {'dissonance': self.manager.state.dissonance}
            
            # Get world state
            world_state = self.env.step(episode, pre_state['dissonance'])
            
            # Use active epistemology to select action
            action, metadata = self.epistemology.act_and_learn(
                episode, pre_state, ExplorationMode.EXPLORE_SAFE
            )
            
            # Track uncertainty and probing
            belief_state = self.belief.get_belief_state()
            hypotheses = belief_state
            
            if len(belief_state) >= 2:
                sorted_hyps = sorted(enumerate(belief_state), key=lambda x: x[1].confidence, reverse=True)
                gap = sorted_hyps[0][1].confidence - sorted_hyps[1][1].confidence
                is_uncertain = gap < 0.2
                
                if is_uncertain:
                    self.total_uncertain_episodes += 1
                    if metadata.get("reason") == "hypothesis_driven_probe":
                        self.probes_when_uncertain += 1
            
            # Execute step
            difficulty = world_state['difficulty']
            correctness = self._simulate_outcome(difficulty)
            step_record = self.manager.step(correctness=correctness, difficulty=difficulty, debug=False)
            
            post_state = {'dissonance': self.manager.state.dissonance}
            
            # Update belief with evidence (simplified)
            from core.belief_reasoner import EvidenceEvent
            evidence = EvidenceEvent(
                episode=episode,
                predicted_d=pre_state['dissonance'],
                actual_d=post_state['dissonance'],
                observed_boundary=post_state['dissonance'] / 0.95 if post_state['dissonance'] > 0.5 else post_state['dissonance'] / 0.3,
                mode=1 if action == "conservative_stabilize" else 2,
                clamp_occurred=step_record.get('constraint_activated', False),
                context_snapshot={'dissonance': post_state['dissonance']}
            )
            self.belief.observe_evidence(evidence, self.env.get_true_boundary())
            
            # Logging
            if (episode + 1) % self.config['log_interval'] == 0 or metadata.get("reason") == "hypothesis_driven_probe":
                marker = "🔬" if metadata.get("reason") == "hypothesis_driven_probe" else "  "
                print(f"{marker}EP{episode+1:04d} | D={post_state['dissonance']:.2f} | "
                      f"Belief={self.belief.current_belief:.2f}±{self.belief.current_uncertainty:.2f} | "
                      f"Hyps={len(belief_state)} | "
                      f"Action={action[:15]:15s} | "
                      f"Reason={metadata.get('reason', 'default')[:20]}")
        
        # Final results
        print("\n" + "=" * 70)
        print("📊 PHASE G RESULTS: Active Epistemology Test")
        print("=" * 70)
        
        active_discovery_rate = self.probes_when_uncertain / max(1, self.total_uncertain_episodes)
        
        print(f"\n🎯 UNCERTAINTY BEHAVIOR:")
        print(f"   Total uncertain episodes: {self.total_uncertain_episodes}")
        print(f"   Intentional probes: {self.probes_when_uncertain}")
        print(f"   Active discovery rate: {active_discovery_rate:.1%}")
        
        print(f"\n🧠 BELIEF EVOLUTION:")
        print(f"   Final belief: {self.belief.current_belief:.3f} (true: {self.env.true_boundary})")
        print(f"   Final uncertainty: {self.belief.current_uncertainty:.3f}")
        print(f"   Remaining hypotheses: {len(belief_state)}")
        
        print(f"\n🔬 ACTIVE EPISTEMOLOGY:")
        epistemic_summary = self.epistemology.get_epistemic_status()
        print(f"   Total experiments: {epistemic_summary['action_selection']['total_probes']}")
        print(f"   Uncertainties resolved: {epistemic_summary['uncertainties_resolved']}")
        print(f"   Avg info gain: {epistemic_summary['avg_info_gain']:.4f}")
        
        # Verdict
        print("\n" + "=" * 70)
        if active_discovery_rate > 0.025:
            print("🏆 VERDICT: ACTIVE DISCOVERER 🚀")
            print("   RAVANA intentionally acts to resolve uncertainty")
            print("   when two hypotheses are close in confidence.")
        elif active_discovery_rate > 0.01:
            print("⚖️ VERDICT: EMERGING CURIOSITY")
            print("   RAVANA sometimes probes, but not consistently")
        else:
            print("🛑 VERDICT: PASSIVE THINKER")
            print("   RAVANA lives with uncertainty without acting")
        print("=" * 70)
        
        return {
            'total_episodes': self.config['total_episodes'],
            'final_belief': self.belief.current_belief,
            'true_boundary': self.env.true_boundary,
            'active_discovery_rate': active_discovery_rate,
            'total_probes': self.probes_when_uncertain,
            'epistemic_status': epistemic_summary,
            'verdict': 'active_discoverer' if active_discovery_rate > 0.3 else 'passive_thinker'
        }


def main():
    # Components
    governor = Governor(GovernorConfig(max_dissonance=0.95, min_dissonance=0.15, max_identity=0.95, min_identity=0.10))
    resolution = ResolutionEngine(partial_threshold=0.15)
    identity = IdentityEngine(initial_strength=0.5)
    manager = StateManager(governor, resolution, identity)
    
    strategy = StrategyLayer(StrategyConfig())
    learning = StrategyLearningLayer(LearningConfig())
    learning_wrapper = StrategyWithLearning(strategy, learning)
    intent = IntentEngine(IntentConfig())
    intent_strategy = IntentAwareStrategy(strategy, learning_wrapper, intent)
    
    belief = BeliefReasoner(BeliefConfig())
    voi_config = VoIConfig(info_gain_weight=0.4, uncertainty_threshold=0.15, min_probe_interval=30)
    epistemology = ActiveEpistemology(belief, voi_config)
    
    env = PartialWorldEnvironment(true_boundary=0.75, alternative_boundary=0.90)
    
    config = {'total_episodes': 1000, 'log_interval': 100}
    pipeline = PhaseGTrainingPipeline(manager, intent_strategy, belief, epistemology, env, config)
    
    results = pipeline.train()
    return results


if __name__ == "__main__":
    main()
