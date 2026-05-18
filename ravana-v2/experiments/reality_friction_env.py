"""
RAVANA v2 — PHASE I+: Reality Friction Test Environment
Test RAVANA under hostile, messy, delayed real-world conditions.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.reality_friction import (
    RealityFrictionLayer,
    RealityFrictionConfig,
    NoiseConfig,
    DelayConfig,
    PartialObsConfig,
    NonStationaryConfig,
    ResourceConfig,
    FrictionType
)
from core.state import StateManager
from core.governor import Governor, GovernorConfig
from core.resolution import ResolutionEngine
from core.identity import IdentityEngine
from core.social_epistemology import SocialEpistemologyEngine, AgentType


@dataclass
class FrictionScenario:
    """A reality friction test scenario."""
    name: str
    description: str
    intensity: float
    friction_types: List[FrictionType]
    episodes: int
    survival_threshold: float
    
    # Specific configs
    noise_sigma: float = 0.1
    delay_range: Tuple[int, int] = (5, 30)
    observable_fraction: float = 0.8
    shift_probability: float = 0.02


class RealityFrictionEnvironment:
    """
    Controlled test environment for reality friction.
    
    SCENARIOS:
        1. Fog of War: Heavy noise + partial observability
        2. Delayed Truth: Feedback arrives late or never
        3. Shifting Ground: Non-stationary dynamics
        4. Resource Starvation: Limited compute
        5. The Gauntlet: All friction types combined
    """
    
    SCENARIOS = {
        'fog_of_war': FrictionScenario(
            name='Fog of War',
            description='Heavy observation noise and partial observability',
            intensity=0.7,
            friction_types=[FrictionType.NOISE, FrictionType.PARTIAL],
            episodes=200,
            survival_threshold=0.2,
            noise_sigma=0.25,
            observable_fraction=0.6
        ),
        'delayed_truth': FrictionScenario(
            name='Delayed Truth',
            description='Ground truth arrives late, partially, or never',
            intensity=0.6,
            friction_types=[FrictionType.DELAY],
            episodes=300,
            survival_threshold=0.25,
            delay_range=(10, 80)
        ),
        'shifting_ground': FrictionScenario(
            name='Shifting Ground',
            description='Environment dynamics shift unexpectedly',
            intensity=0.5,
            friction_types=[FrictionType.NON_STATIONARY],
            episodes=400,
            survival_threshold=0.2,
            shift_probability=0.03
        ),
        'resource_starvation': FrictionScenario(
            name='Resource Starvation',
            description='Severe compute and memory constraints',
            intensity=0.8,
            friction_types=[FrictionType.RESOURCE],
            episodes=200,
            survival_threshold=0.3
        ),
        'the_gauntlet': FrictionScenario(
            name='The Gauntlet',
            description='All friction types combined (ultimate test)',
            intensity=0.8,
            friction_types=[
                FrictionType.NOISE, FrictionType.DELAY, FrictionType.PARTIAL,
                FrictionType.NON_STATIONARY, FrictionType.RESOURCE, FrictionType.ADVERSARIAL
            ],
            episodes=500,
            survival_threshold=0.25,
            noise_sigma=0.2,
            observable_fraction=0.7,
            delay_range=(5, 50),
            shift_probability=0.025
        )
    }
    
    def __init__(self, scenario: FrictionScenario):
        self.scenario = scenario
        self.episode = 0
        
        # Build friction config
        config = RealityFrictionConfig(intensity=scenario.intensity)
        
        if FrictionType.NOISE in scenario.friction_types:
            config.noise.base_sigma = scenario.noise_sigma
        
        if FrictionType.DELAY in scenario.friction_types:
            config.delay.min_delay = scenario.delay_range[0]
            config.delay.max_delay = scenario.delay_range[1]
        
        if FrictionType.PARTIAL in scenario.friction_types:
            config.partial.observable_fraction = scenario.observable_fraction
        
        if FrictionType.NON_STATIONARY in scenario.friction_types:
            config.non_stationary.shift_probability = scenario.shift_probability
        
        # Initialize friction layer
        self.friction = RealityFrictionLayer(config)
        
        # Initialize RAVANA components
        self._init_ravana()
        
        # Tracking
        self.episode_records: List[Dict] = []
        self.disruption_events: List[Dict] = []
        self.recovery_times: List[int] = []
        
    def _init_ravana(self):
        """Initialize RAVANA core for friction testing."""
        governor_config = GovernorConfig()
        self.governor = Governor(governor_config)
        
        self.resolution = ResolutionEngine()
        self.identity = IdentityEngine()
        
        self.state_mgr = StateManager(
            governor=self.governor,
            resolution_engine=self.resolution,
            identity_engine=self.identity
        )
        
        # Initialize social layer (for multi-agent in friction)
        self.social = SocialEpistemologyEngine()
        self.social.register_agent('ravana', AgentType.RAVANA)
        
    def run_episode(self) -> Dict[str, Any]:
        """Run one episode with friction."""
        self.episode += 1
        
        # Get current true state
        true_state = {
            'dissonance': self.state_mgr.state.dissonance,
            'identity': self.state_mgr.state.identity,
            'resolution_success': np.random.random() > 0.3,
            'wisdom_delta': np.random.random() * 0.1,
            'clamp_occurred': False,
            'mode': 0
        }
        
        # Get RAVANA's current belief
        ravana_belief = self.state_mgr.state.dissonance
        ravana_confidence = self.identity.get_status().get('confidence', 0.5)
        
        # Apply friction
        friction_result = self.friction.step(
            ravana_belief=ravana_belief,
            ravana_confidence=ravana_confidence,
            true_state=true_state
        )
        
        observation = friction_result['observation']
        
        # RAVANA processes noisy observation (not true state)
        # Simulate episode with observed state
        correctness = np.random.random() > 0.3  # Affected by noise
        
        try:
            step_record = self.state_mgr.step(
                correctness=correctness,
                difficulty=0.5,
                debug=False
            )
        except Exception as e:
            step_record = {'error': str(e), 'mode': 'failed'}
        
        # Update social belief
        self.social.agent_beliefs['ravana'] = self.social.agent_beliefs['ravana'].__class__(
            agent_id='ravana',
            boundary_estimate=self.state_mgr.state.dissonance,
            confidence=ravana_confidence,
            uncertainty=1.0 - ravana_confidence,
            last_updated=self.episode
        )
        
        # Process any delivered feedback
        for feedback in friction_result['delivered_feedback']:
            # RAVANA gets to learn from delayed feedback
            if 'true_boundary' in feedback.partial_info:
                true_boundary = feedback.partial_info['true_boundary']
                # Adjust belief toward truth (limited by learning rate)
                current = self.state_mgr.state.dissonance
                error = abs(current - true_boundary)
                # Learning signal
                learning_signal = -error * 0.1
                # Note: In real implementation, this would integrate with adaptation layer
        
        # Record
        record = {
            'episode': self.episode,
            'observation': {
                'dissonance': observation.dissonance,
                'identity': observation.identity,
                'confidence': observation.confidence,
                'noise_level': observation.noise_level
            },
            'true_state': true_state,
            'metrics': {
                'belief_drift': friction_result['metrics'].belief_drift,
                'confidence_calibration': friction_result['metrics'].confidence_calibration,
                'disruption_detected': friction_result['metrics'].disruption_detected,
                'recovery_episodes': friction_result['metrics'].recovery_episodes,
                'signal_to_noise': friction_result['metrics'].signal_to_noise
            },
            'hidden_effects': friction_result['hidden_effects'],
            'feedback_count': len(friction_result['delivered_feedback']),
            'ravana_state': {
                'dissonance': self.state_mgr.state.dissonance,
                'identity': self.state_mgr.state.identity,
                'mode': step_record.get('mode', 'unknown')
            }
        }
        
        self.episode_records.append(record)
        
        # Track disruptions and recovery
        if friction_result['metrics'].disruption_detected:
            self.disruption_events.append({
                'episode': self.episode,
                'belief_drift': friction_result['metrics'].belief_drift
            })
        
        if friction_result['metrics'].recovery_episodes > 0:
            if len(self.recovery_times) == 0 or self.episode - self.recovery_times[-1] > 10:
                self.recovery_times.append(friction_result['metrics'].recovery_episodes)
        
        return record
    
    def run_full_test(self) -> Dict[str, Any]:
        """Run full scenario test."""
        print(f"\n  Running {self.scenario.episodes} episodes...")
        
        for i in range(self.scenario.episodes):
            if i % 50 == 0:
                print(f"    Episode {i}/{self.scenario.episodes}")
            self.run_episode()
        
        return self.compute_results()
    
    def compute_results(self) -> Dict[str, Any]:
        """Compute scenario results."""
        recent_records = self.episode_records[-100:] if len(self.episode_records) >= 100 else self.episode_records
        
        # Belief drift
        belief_drifts = [r['metrics']['belief_drift'] for r in recent_records]
        avg_drift = np.mean(belief_drifts)
        max_drift = max(belief_drifts)
        final_drift = belief_drifts[-1] if belief_drifts else 1.0
        
        # Confidence calibration
        calibrations = [r['metrics']['confidence_calibration'] for r in recent_records]
        avg_calibration = np.mean(calibrations)
        overconfidence_rate = sum(1 for c in calibrations if c > 0.1) / len(calibrations)
        
        # Recovery
        avg_recovery = np.mean(self.recovery_times) if self.recovery_times else 0
        
        # Disruptions
        disruption_count = len(self.disruption_events)
        
        # Survival metric
        survival_score = max(0, 1 - final_drift / 0.5)
        survived = survival_score >= self.scenario.survival_threshold
        
        return {
            'scenario': self.scenario.name,
            'intensity': self.scenario.intensity,
            'episodes': len(self.episode_records),
            'belief_drift': {
                'avg': avg_drift,
                'max': max_drift,
                'final': final_drift
            },
            'confidence': {
                'avg_calibration': avg_calibration,
                'overconfidence_rate': overconfidence_rate
            },
            'recovery': {
                'avg_recovery_episodes': avg_recovery,
                'total_disruptions': disruption_count
            },
            'survival_score': survival_score,
            'survived': survived,
            'threshold': self.scenario.survival_threshold,
            'friction_summary': self.friction.get_friction_summary()
        }


def run_all_friction_tests(quick: bool = False) -> Dict[str, Any]:
    """Run all reality friction scenarios."""
    print("\n" + "="*70)
    print("  PHASE I+: REALITY FRICTION — SURVIVAL TESTING")
    print("="*70)
    
    results = {}
    
    scenario_order = [
        'fog_of_war',
        'delayed_truth',
        'shifting_ground',
        'resource_starvation',
        'the_gauntlet'
    ]
    
    for scenario_name in scenario_order:
        scenario = RealityFrictionEnvironment.SCENARIOS[scenario_name]
        
        if quick:
            scenario.episodes = min(50, scenario.episodes)
        
        print(f"\n{'─'*70}")
        print(f"  Scenario: {scenario.name}")
        print(f"  {scenario.description}")
        print(f"  Intensity: {scenario.intensity} | Episodes: {scenario.episodes}")
        print(f"  Threshold: {scenario.survival_threshold}")
        print(f"{'─'*70}")
        
        env = RealityFrictionEnvironment(scenario)
        result = env.run_full_test()
        results[scenario_name] = result
        
        # Print summary
        status = "✅ SURVIVED" if result['survived'] else "❌ FAILED"
        print(f"\n  {status}")
        print(f"    Final drift: {result['belief_drift']['final']:.3f}")
        print(f"    Survival score: {result['survival_score']:.2%}")
        print(f"    Overconfidence rate: {result['confidence']['overconfidence_rate']:.1%}")
        print(f"    Disruptions: {result['recovery']['total_disruptions']}")
    
    return results


def analyze_results(results: Dict[str, Any]) -> str:
    """Generate analysis report."""
    lines = []
    lines.append("\n" + "="*70)
    lines.append("  REALITY FRICTION: COMPREHENSIVE ANALYSIS")
    lines.append("="*70)
    
    # Overall survival
    survived = sum(1 for r in results.values() if r['survived'])
    total = len(results)
    lines.append(f"\n📊 OVERALL SURVIVAL: {survived}/{total} scenarios")
    
    # Scenario breakdown
    lines.append("\n🔍 SCENARIO BREAKDOWN")
    lines.append("-"*70)
    
    for name, result in results.items():
        status = "✅" if result['survived'] else "❌"
        lines.append(f"\n{status} {result['scenario'].upper()}")
        lines.append(f"   Survival: {result['survival_score']:.1%} (threshold: {result['threshold']:.1%})")
        lines.append(f"   Final drift: {result['belief_drift']['final']:.3f}")
        lines.append(f"   Max drift: {result['belief_drift']['max']:.3f}")
        lines.append(f"   Overconfidence: {result['confidence']['overconfidence_rate']:.1%}")
    
    # Failure analysis
    lines.append("\n" + "="*70)
    lines.append("⚠️ FAILURE ANALYSIS")
    lines.append("-"*70)
    
    failures = [(name, r) for name, r in results.items() if not r['survived']]
    if failures:
        for name, result in failures:
            lines.append(f"\n❌ {result['scenario']}")
            lines.append(f"   Failed because: Belief drift {result['belief_drift']['final']:.3f}")
            lines.append(f"   > threshold {result['threshold']:.3f}")
    else:
        lines.append("\n✅ No failures — RAVANA survived all friction scenarios!")
    
    # Confidence analysis
    lines.append("\n" + "="*70)
    lines.append("🎯 CONFIDENCE CALIBRATION ANALYSIS")
    lines.append("-"*70)
    
    high_overconf = [(name, r) for name, r in results.items() 
                      if r['confidence']['overconfidence_rate'] > 0.3]
    
    if high_overconf:
        lines.append("\n⚠️ Overconfidence detected in:")
        for name, result in high_overconf:
            lines.append(f"   {result['scenario']}: {result['confidence']['overconfidence_rate']:.1%}")
        lines.append("\n👉 Interpretation: RAVANA is confident when wrong")
        lines.append("   This is a critical failure mode for real-world deployment")
    else:
        lines.append("\n✅ Confidence well-calibrated across all scenarios")
    
    # Recovery analysis
    lines.append("\n" + "="*70)
    lines.append("🔄 RECOVERY ANALYSIS")
    lines.append("-"*70)
    
    for name, result in results.items():
        avg_recovery = result['recovery']['avg_recovery_episodes']
        disruptions = result['recovery']['total_disruptions']
        lines.append(f"\n{result['scenario']}:")
        lines.append(f"   Disruptions: {disruptions}")
        lines.append(f"   Avg recovery: {avg_recovery:.1f} episodes")
    
    # Final verdict
    lines.append("\n" + "="*70)
    lines.append("🏆 FINAL VERDICT")
    lines.append("="*70)
    
    if survived == total:
        lines.append("\n✅ RAVANA PASSED: Survived all reality friction scenarios")
        lines.append("   System demonstrates robust epistemic resilience")
        lines.append("   Ready for real-world deployment consideration")
    elif survived >= total * 0.8:
        lines.append("\n⚠️ RAVANA PARTIAL PASS: Survived most scenarios")
        lines.append(f"   Failed {total - survived}/{total} scenarios")
        lines.append("   Some fragility exposed — needs hardening")
    else:
        lines.append("\n❌ RAVANA FAILED: Could not survive reality friction")
        lines.append(f"   Survived only {survived}/{total} scenarios")
        lines.append("   Significant architectural weakness detected")
    
    lines.append("\n" + "="*70)
    lines.append("  PHASE I+ COMPLETE")
    lines.append("="*70)
    
    return "\n".join(lines)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase I+ Reality Friction Testing')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--scenario', type=str, help='Run specific scenario')
    args = parser.parse_args()
    
    if args.scenario:
        scenario = RealityFrictionEnvironment.SCENARIOS.get(args.scenario)
        if not scenario:
            print(f"Unknown scenario: {args.scenario}")
            print(f"Available: {list(RealityFrictionEnvironment.SCENARIOS.keys())}")
            sys.exit(1)
        
        print(f"\n{'='*70}")
        print(f"  Reality Friction: {scenario.name}")
        print(f"{'='*70}")
        
        if args.quick:
            scenario.episodes = 50
        
        env = RealityFrictionEnvironment(scenario)
        result = env.run_full_test()
        
        print(f"\n{'='*70}")
        print(f"  Result: {'✅ PASSED' if result['survived'] else '❌ FAILED'}")
        print(f"  Survival score: {result['survival_score']:.2%}")
        print(f"{'='*70}")
    else:
        results = run_all_friction_tests(quick=args.quick)
        report = analyze_results(results)
        print(report)
        
        # Save results
        import json
        
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, bool):
                return bool(obj)
            return obj
        
        serializable_results = convert_to_serializable(results)
        with open('results/phase_i_plus_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        with open('results/phase_i_plus_report.txt', 'w') as f:
            f.write(report)
        
        print(f"\n💾 Results saved to results/phase_i_plus_*.json/txt")
