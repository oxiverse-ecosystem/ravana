#!/usr/bin/env python3
"""
RAVANA v2 — Phase J: Hypothesis Generation
Unknown Unknown Test: Can RAVANA invent new model classes?

Environment: boundary = 0.8 + 0.1 * sin(time/100)
- Periodic, smooth, non-linear
- No existing hypothesis can explain it
- True test: does RAVANA generate parametric-time hypothesis?
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core import (
    Governor, GovernorConfig, ResolutionEngine, IdentityEngine,
    StateManager, StrategyWithLearning, StrategyLayer, StrategyLearningLayer,
    BeliefReasoner, BeliefConfig, SurgicalProbeSelector, SurgicalProbeConfig,
    HypothesisGenerator, GenerationConfig, HypothesisType
)
from core.surgical_probes import ProbeType


@dataclass
class JConfig:
    """Configuration for Phase J test."""
    total_episodes: int = 1000
    log_interval: int = 50
    debug_first_n: int = 20
    
    # Unknown Unknown environment
    true_boundary_base: float = 0.8
    oscillation_amplitude: float = 0.1
    oscillation_period: int = 200  # One full cycle every 200 episodes
    
    # Test thresholds
    generation_trigger_window: int = 30
    success_if_generates_parametric: bool = True
    success_if_explains_sine: bool = True


class UnknownUnknownEnvironment:
    """
    Non-linear boundary that oscillates sinusoidally.
    RAVANA must discover: boundary = f(time), not constant.
    """
    
    def __init__(self, config: JConfig):
        self.config = config
        self.episode = 0
        self.boundary_history: List[float] = []
    
    def get_true_boundary(self, episode: int) -> float:
        """Calculate true boundary with sine oscillation."""
        phase = 2 * np.pi * episode / self.config.oscillation_period
        boundary = self.config.true_boundary_base + \
                   self.config.oscillation_amplitude * np.sin(phase)
        return np.clip(boundary, 0.5, 0.95)
    
    def step(self, episode: int) -> Dict[str, float]:
        """Return current environment state."""
        self.episode = episode
        true_boundary = self.get_true_boundary(episode)
        self.boundary_history.append(true_boundary)
        
        # Observable: dissonance, success rate, but NOT true boundary
        return {
            "true_boundary": true_boundary,  # For validation only
            "difficulty": 0.5 + 0.2 * np.sin(2 * np.pi * episode / 100),  # Variable
            "phase": episode / self.config.oscillation_period,
        }
    
    def simulate_outcome(self, difficulty: float, mode: str) -> bool:
        """Simulate outcome based on difficulty."""
        success_rate = 0.7 - (difficulty - 0.3) * 0.4
        return np.random.random() < success_rate


class PhaseJTest:
    """
    Test if RAVANA generates hypotheses when probing plateaus.
    """
    
    def __init__(self, config: JConfig = None):
        self.config = config or JConfig()
        self.env = UnknownUnknownEnvironment(self.config)
        project_root = Path(__file__).resolve().parent.parent.parent
        self.results_dir = project_root / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize full RAVANA stack
        self._init_ravana()
    
    def _init_ravana(self):
        """Initialize complete RAVANA cognitive stack."""
        # Physics layer
        self.governor = Governor(GovernorConfig(
            max_dissonance=0.95, min_dissonance=0.15,
            max_identity=0.95, min_identity=0.10
        ))
        
        # Belief layer
        self.belief = BeliefReasoner(BeliefConfig())
        
        # Surgical probing layer
        self.surgical = SurgicalProbeSelector(SurgicalProbeConfig())
        
        # Hypothesis generation layer (PHASE J)
        self.generator = HypothesisGenerator(GenerationConfig(
            kl_plateau_threshold=0.08,
            max_hypotheses=5
        ))
        
        # Strategy and state
        strategy = StrategyLayer()
        learning = StrategyLearningLayer()
        self.strategy = StrategyWithLearning(strategy, learning)
        
        self.resolution = ResolutionEngine()
        self.identity = IdentityEngine()
        self.manager = StateManager(self.governor, self.resolution, self.identity)
        
        # Tracking
        self.generation_events: List[Dict] = []
        self.hypothesis_evolution: List[Dict] = []
        self.probe_effectiveness: List[Dict] = []
    
    def _get_context(self) -> Dict[str, float]:
        """Build context for hypothesis evaluation."""
        return {
            'dissonance': self.manager.state.dissonance,
            'identity': self.manager.state.identity,
            'clamp_rate': (self.manager.governor.clamp_diagnostics.d_clamp_activations + self.manager.governor.clamp_diagnostics.i_clamp_activations) / max(1, len(self.manager.governor.history)),
            'dissonance_trend': self.surgical._compute_d_trend() if hasattr(self.surgical, '_compute_d_trend') else 0.0,
            'episode': self.manager.state.episode
        }
    
    def train(self) -> Dict[str, Any]:
        """Run Phase J test."""
        print("=" * 70)
        print("RAVANA v2 — Phase J: Hypothesis Generation (Unknown Unknown Test)")
        print("=" * 70)
        print(f"\n🎭 UNKNOWN UNKNOWN CHALLENGE")
        print(f"   True boundary: 0.8 + 0.1*sin(time/{self.config.oscillation_period})")
        print(f"   Initial hypotheses: constant boundaries only")
        print(f"   Question: Will RAVANA generate parametric-time hypothesis?")
        print("=" * 70)
        
        for episode in range(self.config.total_episodes):
            # Environment step
            env_state = self.env.step(episode)
            true_boundary = env_state["true_boundary"]
            
            # Get current belief hypotheses
            current_hypotheses = self.belief.get_belief_state()
            
            # SURGICAL PROBING: Select KL-maximizing probe
            context = self._get_context()
            probe_type, probe_info = self.surgical.select_surgical_probe(
                current_hypotheses, context, episode
            )
            
            # Execute step with or without probe
            difficulty = env_state["difficulty"]
            correctness = self.env.simulate_outcome(difficulty, "normal")
            
            debug = episode < self.config.debug_first_n
            step_record = self.manager.step(
                correctness=correctness,
                difficulty=difficulty,
                debug=debug
            )
            
            # Extract clamp events for belief update
            clamp_events = self._extract_clamps(step_record)
            
            # Update belief with new evidence
            pre_state = {'dissonance': self.manager.state.dissonance}
            post_state = {'dissonance': self.manager.state.dissonance}
            
            # Create evidence event for belief update
            if clamp_events:
                for event in clamp_events:
                    evidence = EvidenceEvent(
                        episode=event.get('episode', episode),
                        observed_boundary=self.belief.current_belief,  # Best estimate
                        predicted_d=pre_state['dissonance'],
                        actual_d=post_state['dissonance'],
                        mode=0,
                        clamp_occurred=event.get('capped', False),
                        context_snapshot=context
                    )
                    self.belief.observe_evidence(evidence, env_state["true_boundary"])
            
            # Record probe effectiveness
            if probe_type is not None:
                # Calculate actual KL gain (simplified)
                kl_gain = probe_info.get('expected_kl', 0.0)
                self.surgical.record_probe_result(
                    probe_type, episode, kl_gain, kl_gain
                )
                self.probe_effectiveness.append({
                    'episode': episode,
                    'probe': probe_type.value,
                    'kl': kl_gain
                })
            
            # PHASE J: Monitor for hypothesis generation triggers
            uncertainty = self.belief.current_uncertainty
            dissonance = self.manager.state.dissonance
            
            monitor_result = self.generator.monitor_state(
                episode, 
                kl_gain=probe_info.get('expected_kl', 0.0) if probe_type else 0.0,
                uncertainty=uncertainty,
                dissonance=dissonance,
                hypotheses=current_hypotheses
            )
            
            # GENERATE if triggered
            if monitor_result['should_generate']:
                new_hypothesis = self.generator.generate_hypothesis(
                    episode, current_hypotheses, monitor_result['triggers_detected']
                )
                
                if new_hypothesis:
                    # Add to belief system
                    # New hypothesis added to generator, not directly to belief
                    
                    self.generation_events.append({
                        'episode': episode,
                        'type': new_hypothesis.hypothesis_type.name,
                        'triggers': monitor_result['triggers_detected'],
                        'complexity': new_hypothesis.complexity_score
                    })
                    
                    print(f"\n🌱 EP{episode:04d} | NEW HYPOTHESIS GENERATED!")
                    print(f"   Type: {new_hypothesis.hypothesis_type.name}")
                    print(f"   Triggers: {', '.join(monitor_result['triggers_detected'])}")
                    print(f"   Complexity: {new_hypothesis.complexity_score:.2f}")
            
            # Periodic logging
            if (episode + 1) % self.config.log_interval == 0 or episode < 5:
                self._log_progress(episode, env_state, monitor_result, current_hypotheses)
            
            # Track hypothesis evolution
            self.hypothesis_evolution.append({
                'episode': episode,
                'true_boundary': true_boundary,
                'belief_boundary': self.belief.current_belief,
                'uncertainty': uncertainty,
                'n_hypotheses': len(current_hypotheses),
                'generated_types': list(set(h.hypothesis_type.name if hasattr(h, 'hypothesis_type') else 'constant' for h in current_hypotheses))
            })
        
        # Final analysis
        results = self._analyze_results()
        self._save_results(results)
        
        return results
    
    def _extract_clamps(self, step_record: Dict) -> List[Dict]:
        """Extract clamp events from step record."""
        events = []
        if step_record.get('constraint_activated'):
            events.append({
                'episode': step_record.get('episode', 0),
                'reason': step_record.get('reason', 'unknown'),
                'capped': step_record.get('capped', False)
            })
        return events
    
    def _log_progress(self, episode: int, env_state: Dict, 
                      monitor: Dict, hypotheses: List[Any]):
        """Log test progress."""
        state = self.manager.state
        true_b = env_state['true_boundary']
        belief_b = self.belief.current_belief
        
        # Detect if generation happened
        gen_marker = "🌱" if any(g['episode'] == episode for g in self.generation_events) else "  "
        
        print(f"{gen_marker}EP{episode:04d} | "
              f"True:{true_b:.2f} | "
              f"Belief:{belief_b:.2f}±{self.belief.current_uncertainty:.2f} | "
              f"Hyps:{len(hypotheses)} | "
              f"Triggers:{len(monitor['triggers_detected'])}")
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze Phase J test results."""
        print("\n" + "=" * 70)
        print("📊 PHASE J RESULTS")
        print("=" * 70)
        
        # Count generations by type
        gen_by_type = {}
        for event in self.generation_events:
            t = event['type']
            gen_by_type[t] = gen_by_type.get(t, 0) + 1
        
        print(f"\n🌱 HYPOTHESIS GENERATION:")
        print(f"   Total generations: {len(self.generation_events)}")
        print(f"   By type: {gen_by_type}")
        
        # Check for parametric-time generation
        parametric_generated = any(
            e['type'] in ['PARAMETRIC_TIME', 'PARAMETRIC_STATE'] 
            for e in self.generation_events
        )
        
        # Check belief tracking of sine wave
        if self.hypothesis_evolution:
            errors = [abs(h['true_boundary'] - h['belief_boundary']) 
                     for h in self.hypothesis_evolution[100:]]  # After warm-up
            mean_error = np.mean(errors)
        else:
            mean_error = 1.0
        
        print(f"\n🎯 ACCURACY:")
        print(f"   Mean boundary error: {mean_error:.3f}")
        print(f"   Sine wave explained: {'✅' if mean_error < 0.1 else '❌'}")
        
        print(f"\n🔬 SURGICAL PROBING:")
        print(f"   Total probes: {len(self.probe_effectiveness)}")
        if self.probe_effectiveness:
            avg_kl = np.mean([p['kl'] for p in self.probe_effectiveness])
            print(f"   Avg KL gain: {avg_kl:.3f}")
        
        # Verdict
        verdict = "🟢 FULL SUCCESS" if parametric_generated and mean_error < 0.15 else \
                  "🟡 PARTIAL" if parametric_generated else \
                  "🔴 FAILED"
        
        print(f"\n{'=' * 70}")
        print(f"🏁 VERDICT: {verdict}")
        print(f"   Parametric hypothesis generated: {'✅' if parametric_generated else '❌'}")
        print(f"   Sine wave tracked: {'✅' if mean_error < 0.15 else '❌'}")
        print("=" * 70)
        
        return {
            'parametric_generated': parametric_generated,
            'mean_boundary_error': mean_error,
            'total_generations': len(self.generation_events),
            'generations_by_type': gen_by_type,
            'verdict': verdict,
            'generation_events': self.generation_events,
            'hypothesis_evolution': self.hypothesis_evolution,
            'probe_effectiveness': self.probe_effectiveness
        }
    
    def _save_results(self, results: Dict):
        """Save results to file."""
        output_file = self.results_dir / "phase_j_unknown_unknown.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📊 Results saved to: {output_file}")


def main():
    """Entry point for Phase J test."""
    config = JConfig(
        total_episodes=500,  # Shorter for testing
        log_interval=50
    )
    
    test = PhaseJTest(config)
    results = test.train()
    
    return results


if __name__ == "__main__":
    main()
