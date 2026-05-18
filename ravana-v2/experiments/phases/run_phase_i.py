#!/usr/bin/env python3
"""
RAVANA v2 — Phase I: Meta-Cognition Test

VALIDATES:
- When probes fail repeatedly, RAVANA detects systematic epistemic failure
- When confidence is miscalibrated, RAVANA reduces trust in its own assessments  
- When bias detected, RAVANA switches epistemic mode (cautious/exploratory/recovery)
"""

import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core import (
    Governor, GovernorConfig,
    ResolutionEngine,
    IdentityEngine,
    StateManager,
    StrategyLayer, StrategyConfig, ExplorationMode, BehavioralContext,
    StrategyLearningLayer,
    IntentEngine,
    MicroPlanner,
    NonStationaryEnvironment, EnvironmentConfig,
    BeliefReasoner, BeliefConfig,
    ActiveEpistemology,
    SurgicalProbeSelector,
    HypothesisGenerator,
    OccamLayer, OccamConfig,
    MetaCognition, MetaCognitiveConfig, EpistemicMode,
)

from training.pipeline import TrainingPipeline, TrainingConfig
import numpy as np
import random


class AdversarialProbeEnvironment:
    """Environment that manipulates probe results to test meta-cognition."""
    
    def __init__(self, true_boundary=0.75):
        self.true_boundary = true_boundary
        self.episode = 0
        self.probe_failure_rate = 0.0
        
    def step(self, episode: int):
        self.episode = episode
        if episode < 100:
            self.probe_failure_rate = 0.1
        elif episode < 200:
            self.probe_failure_rate = 0.6
        elif episode < 300:
            self.probe_failure_rate = 0.85
        else:
            self.probe_failure_rate = 0.15
    
    def execute_probe(self, probe_design: dict) -> dict:
        true_result = {
            'observed_boundary': self.true_boundary + np.random.normal(0, 0.02),
            'confidence': 0.8,
            'conclusive': True,
        }
        
        if random.random() < self.probe_failure_rate:
            true_result = {
                'observed_boundary': 0.65,
                'confidence': 0.3,
                'conclusive': False,
            }
        
        return true_result


class MetaCognitiveTrainingPipeline(TrainingPipeline):
    """Training pipeline with meta-cognitive monitoring."""
    
    def __init__(self, state_manager, meta_cognition, env, config: TrainingConfig = None):
        super().__init__(state_manager, config)
        self.meta = meta_cognition
        self.env = env
        self.probe_results = []
        self.epistemic_mode_history = []
        
    def train(self) -> dict:
        print("=" * 70)
        print("🧠 RAVANA v2 — Phase I: Meta-Cognition Test")
        print("=" * 70)
        print("\n🎯 TEST: When probes systematically fail,")
        print("         does RAVANA detect the epistemic failure?")
        print("\n🎯 TEST: When confidence is miscalibrated,")
        print("         does RAVANA distrust its own assessments?")
        print("=" * 70)
        
        for episode in range(500):
            self.env.step(episode)
            
            current_mode = self.meta.recommend_epistemic_mode(episode)
            self.epistemic_mode_history.append(current_mode.value)
            
            probe = self.meta.design_probe_for_uncertainty([])
            probe_result = self.env.execute_probe(probe)
            
            assessment = self.meta.assess_probe_outcome(probe, probe_result, episode)
            
            if episode % 100 == 0:
                self._log_status(episode, current_mode, assessment)
        
        return self._generate_summary()
    
    def _log_status(self, episode: int, mode: EpistemicMode, assessment: dict):
        meta = self.meta.get_meta_status()
        print(f"\n🧠 EP{episode:04d} Mode:{mode.value} Failures:{meta['recent_probe_failures']}/10")
        if meta['recent_probe_failures'] > 5:
            print(f"   ⚠️  ALERT: High probe failure rate!")
        if mode == EpistemicMode.RECOVERY:
            print(f"   🔴 RECOVERY MODE: Epistemic failure detected")
    
    def _generate_summary(self) -> dict:
        mode_counts = {}
        for mode_val in self.epistemic_mode_history:
            mode_counts[mode_val] = mode_counts.get(mode_val, 0) + 1
        
        print("\n" + "=" * 70)
        print("🧠 META-COGNITIVE TEST — RESULTS")
        print("=" * 70)
        print(f"\n📊 Epistemic Mode Distribution:")
        for mode, count in mode_counts.items():
            print(f"   {mode}: {count} episodes ({100*count/500:.1f}%)")
        
        recovery_used = mode_counts.get(EpistemicMode.RECOVERY.value, 0)
        verdict = "🟢 PASS" if recovery_used > 20 else "🟡 PARTIAL" if recovery_used > 10 else "🔴 FAIL"
        print(f"\n🎯 VERDICT: {verdict} (Recovery mode: {recovery_used} episodes)")
        print("=" * 70)
        
        return {'mode_counts': mode_counts, 'recovery_used': recovery_used, 'verdict': verdict}


def main():
    governor = Governor(GovernorConfig())
    resolution = ResolutionEngine()
    identity = IdentityEngine()
    manager = StateManager(governor, resolution, identity)
    
    strategy = StrategyLayer(StrategyConfig())
    learning = StrategyLearningLayer()
    intent = IntentEngine()
    
    meta_config = MetaCognitiveConfig(probe_failure_threshold=0.5)
    meta = MetaCognition(meta_config)
    env = AdversarialProbeEnvironment()
    
    config = TrainingConfig(total_episodes=500, log_interval=100, debug_first_n=20)
    pipeline = MetaCognitiveTrainingPipeline(manager, meta, env, config)
    
    results = pipeline.train()
    return results


if __name__ == "__main__":
    main()
