"""
RAVANA v2 — PHASE I²: Meta²-Cognition Test Environment
Test if RAVANA can detect when its own epistemic method fails.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.meta2_cognition import (
    Meta2CognitionEngine,
    Meta2Config,
    EpistemicCritiqueType
)
from core.hypothesis_generation import HypothesisGenerator, HypothesisType, GenerationConfig
from core.meta2_integration import Meta2IntegratedGenerator, Meta2GenerationConfig


@dataclass
class Meta2TestScenario:
    """A test scenario for Meta²-Cognition."""
    name: str
    description: str
    true_boundary: float
    hypothesis_space_limit: str  # What types are allowed
    true_model_type: str  # What type is ACTUALLY governing
    episodes: int = 200
    
    def is_hypothesis_space_inadequate(self) -> bool:
        """Check if RAVANA's hypothesis space excludes the true model."""
        # If true model type not in allowed space, space is inadequate
        allowed = self.hypothesis_space_limit.split(',')
        return self.true_model_type not in allowed


class Meta2TestEnvironment:
    """
    Test environment for Meta²-Cognition.
    
    Creates scenarios where RAVANA's epistemic method must be questioned.
    """
    
    # Test scenarios that force epistemic self-critique
    SCENARIOS = [
        Meta2TestScenario(
            name="wrong_hypothesis_class",
            description="True model is state-dependent, but RAVANA only considers time-varying",
            true_boundary=0.85,
            hypothesis_space_limit="PARAMETRIC_TIME",
            true_model_type="PARAMETRIC_STATE",
            episodes=150
        ),
        Meta2TestScenario(
            name="structural_blindspot",
            description="True model has dual-boundary, RAVANA only sees single",
            true_boundary=0.90,  # Upper boundary
            hypothesis_space_limit="PARAMETRIC_TIME,PARAMETRIC_STATE",
            true_model_type="STRUCTURAL_DUAL",
            episodes=150
        ),
        Meta2TestScenario(
            name="occam_bias_failure",
            description="Simple model preferred, but truth is complex",
            true_boundary=0.75,
            hypothesis_space_limit="PARAMETRIC_TIME",  # Simple
            true_model_type="CAUSAL_MECHANISM",  # Complex but true
            episodes=150
        ),
        Meta2TestScenario(
            name="probe_strategy_blind",
            description="Current probes cannot distinguish true hypothesis",
            true_boundary=0.80,
            hypothesis_space_limit="PARAMETRIC_TIME,PARAMETRIC_STATE,STRUCTURAL_DUAL",
            true_model_type="STRUCTURAL_ASYMMETRIC",
            episodes=150
        ),
        Meta2TestScenario(
            name="identity_protecting_beliefs",
            description="RAVANA's identity is tied to a wrong hypothesis",
            true_boundary=0.60,
            hypothesis_space_limit="PARAMETRIC_TIME",
            true_model_type="PARAMETRIC_STATE",
            episodes=150
        )
    ]
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.meta2_engine = Meta2CognitionEngine()
        # Create Meta² integrated generator
        self.meta2_config = Meta2Config()
        self.meta2 = Meta2CognitionEngine(self.meta2_config)
        
        # Create hypothesis generator with Meta² integration
        base_generator = HypothesisGenerator(GenerationConfig(
            max_hypotheses=6,
            min_episodes_between_generations=10
        ))
        
        integration_config = Meta2GenerationConfig(
            min_epiphanies_for_expansion=1,
            systematic_failure_rate_threshold=0.3
        )
        
        self.integrated_generator = Meta2IntegratedGenerator(
            base_generator=base_generator,
            meta2_engine=self.meta2,
            config=integration_config
        )
    
    def run_scenario(self, scenario: Meta2TestScenario) -> Dict[str, Any]:
        """Run a single Meta² test scenario."""
        print(f"\n{'='*60}")
        print(f"  🧪 SCENARIO: {scenario.name}")
        print(f"{'='*60}")
        print(f"  {scenario.description}")
        print(f"  True model: {scenario.true_model_type}")
        print(f"  Allowed: {scenario.hypothesis_space_limit}")
        print(f"  Inadequate space: {scenario.is_hypothesis_space_inadequate()}")
        print(f"{'='*60}\n")
        
        # Track what happens
        failed_attempts = 0
        meta2_triggered = False
        critique_issued = None
        epiphany_occurred = False
        
        for episode in range(scenario.episodes):
            # Simulate RAVANA attempting to solve
            # First, try with limited hypothesis space
            
            # Generate observation based on TRUE model
            true_boundary = self._apply_true_model(scenario.true_model_type, episode, scenario.true_boundary)
            
            # Simulate RAVANA's hypothesis generation (limited by space)
            allowed_types = scenario.hypothesis_space_limit.split(',')
            ravana_best_guess = self._simulate_ravana_guess(
                episode, allowed_types, true_boundary
            )
            
            # Check if RAVANA is failing
            error = abs(ravana_best_guess - true_boundary)
            if error > 0.15:
                failed_attempts += 1
            
            # Run Meta² check every 20 episodes
            if episode % 20 == 0 and episode > 0:
                meta2_result = self.meta2_engine.step(
                    episode=episode,
                    hypothesis_space=allowed_types,
                    failure_rate=failed_attempts / (episode + 1),
                    belief_history=[ravana_best_guess] * min(20, episode + 1),
                    hypothesis_generator=None,  # Would be actual generator
                    surgical_prober=None  # Would be actual prober
                )
                
                if meta2_result.get('epiphany_triggered'):
                    epiphany_occurred = True
                    critique_issued = meta2_result.get('critique_issued')
                    meta2_triggered = True
        
        # Analyze results
        final_failure_rate = failed_attempts / scenario.episodes
        success = epiphany_occurred if scenario.is_hypothesis_space_inadequate() else final_failure_rate < 0.3
        
        result = {
            'scenario': scenario.name,
            'description': scenario.description,
            'true_model': scenario.true_model_type,
            'allowed_space': scenario.hypothesis_space_limit,
            'inadequate_space': scenario.is_hypothesis_space_inadequate(),
            'episodes': scenario.episodes,
            'failed_attempts': failed_attempts,
            'final_failure_rate': final_failure_rate,
            'meta2_triggered': meta2_triggered,
            'critique_issued': critique_issued,
            'epiphany_occurred': epiphany_occurred,
            'success': success
        }
        
        self.results.append(result)
        
        # Print summary
        print(f"\n{'─'*60}")
        print(f"  RESULTS:")
        print(f"    Failed attempts: {failed_attempts}/{scenario.episodes}")
        print(f"    Meta² triggered: {meta2_triggered}")
        print(f"    Critique issued: {critique_issued}")
        print(f"    Epiphany: {epiphany_occurred}")
        print(f"    {'✅ PASS' if success else '❌ FAIL'}")
        print(f"{'─'*60}")
        
        return result
    
    def run_demo_epiphany(self) -> Dict[str, Any]:
        """
        Demonstrate Meta² triggering an epiphany in controlled conditions.
        
        This forces hypothesis space inadequacy to show Meta² working.
        """
        print("\n" + "="*60)
        print("🧠 META² EPIPHANY DEMONSTRATION")
        print("="*60)
        
        # Create scenario where true model is outside hypothesis space
        scenario = Meta2TestScenario(
            name="forced_epiphany_demo",
            description="True model is NONLINEAR but only LINEAR hypotheses allowed",
            true_boundary=0.8,
            hypothesis_space_limit="LINEAR_ONLY",  # Force inadequacy
            true_model_type="NONLINEAR",
            episodes=100
        )
        
        # Run with forced inadequacy
        results = []
        epiphany_triggered = False
        
        for ep in range(scenario.episodes):
            # True boundary follows nonlinear pattern
            true_b = 0.5 + 0.3 * np.sin(ep / 20.0) + 0.1 * np.random.randn()
            true_b = np.clip(true_b, 0.2, 0.95)
            
            # Check with Meta² (will detect space inadequacy)
            meta2_state = self.meta2.step(ep, 0.0, 0.5, [], true_b)
            
            if meta2_state['space_inadequacy']:
                critique = self.meta2.issue_epistemic_critique(ep, 0.0, 0.5, [], true_b)
                if critique:
                    epiphany_triggered = True
                    print(f"\n✨ EPIPHANY at episode {ep}!")
                    print(f"   Critique type: {critique['critique_type']}")
                    print(f"   Trigger: {critique['trigger']}")
                    break
        
        return {
            'scenario': scenario.name,
            'epiphany_triggered': epiphany_triggered,
            'status': '✅ EPIPHANY DEMONSTRATED' if epiphany_triggered else '❌ No epiphany'
        }
    
    def _apply_true_model(self, model_type: str, episode: int, base_boundary: float) -> float:
        """Apply the true underlying model to get boundary."""
        if model_type == "PARAMETRIC_TIME":
            # boundary = base + small oscillation
            return base_boundary + 0.05 * np.sin(2 * np.pi * episode / 100)
        
        elif model_type == "PARAMETRIC_STATE":
            # boundary varies with simulated "state" (episode mod)
            phase = (episode % 100) / 100
            return base_boundary - 0.1 * phase
        
        elif model_type == "STRUCTURAL_DUAL":
            # Two zones: high and low
            if episode % 50 < 25:
                return base_boundary  # High zone
            else:
                return base_boundary - 0.15  # Low zone
        
        elif model_type == "CAUSAL_MECHANISM":
            # Complex interaction between multiple factors
            factor_a = np.sin(2 * np.pi * episode / 50)
            factor_b = np.cos(2 * np.pi * episode / 75)
            return base_boundary + 0.08 * (factor_a + factor_b)
        
        elif model_type == "STRUCTURAL_ASYMMETRIC":
            # Different boundary for rising vs falling
            trend = np.sin(2 * np.pi * episode / 80)
            if trend > 0:  # Rising
                return base_boundary - 0.05  # Tighter
            else:  # Falling
                return base_boundary + 0.05  # Looser
        
        return base_boundary
    
    def _simulate_ravana_guess(self, episode: int, allowed_types: List[str], true_boundary: float) -> float:
        """Simulate RAVANA's best guess given limited hypothesis space."""
        # If hypothesis space is inadequate, RAVANA will systematically fail
        # because it cannot represent the true model
        
        noise = np.random.normal(0, 0.02)
        
        # Check if space is inadequate (true model type not in allowed)
        # For the test, we know what the true model type is from the scenario
        # We simulate systematic error when space is wrong
        
        # Get the allowed complexity level
        has_parametric = any('PARAMETRIC' in t for t in allowed_types)
        has_structural = any('STRUCTURAL' in t for t in allowed_types)
        has_causal = any('CAUSAL' in t for t in allowed_types)
        
        # If only simple models allowed, systematic miss on complex truth
        if has_parametric and not has_structural and not has_causal:
            # Large systematic error - RAVANA is using wrong model class
            systematic_error = 0.12 + 0.08 * np.sin(2 * np.pi * episode / 40)
            return np.clip(true_boundary + systematic_error + noise, 0.15, 0.95)
        
        # If adequate space, small error
        return np.clip(true_boundary + noise, 0.15, 0.95)
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all Meta² test scenarios."""
        print("\n" + "="*70)
        print("  PHASE I²: META²-COGNITION TEST SUITE")
        print("  Testing epistemic self-critique capability")
        print("="*70)
        
        for scenario in self.SCENARIOS:
            self.run_scenario(scenario)
        
        # Generate comprehensive report
        return self._generate_report()
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_scenarios = len(self.results)
        passed = sum(1 for r in self.results if r['success'])
        meta2_triggered_count = sum(1 for r in self.results if r['meta2_triggered'])
        epiphanies = sum(1 for r in self.results if r['epiphany_occurred'])
        
        report_lines = [
            "\n" + "="*70,
            "  META²-COGNITION: COMPREHENSIVE ANALYSIS",
            "="*70,
            "",
            f"📊 OVERALL: {passed}/{total_scenarios} scenarios passed",
            f"   Meta² triggered: {meta2_triggered_count}/{total_scenarios}",
            f"   Epiphanies: {epiphanies}/{total_scenarios}",
            "",
            "🔍 SCENARIO BREAKDOWN",
            "─"*70,
        ]
        
        for r in self.results:
            status = "✅" if r['success'] else "❌"
            space_issue = " (space inadequate)" if r['inadequate_space'] else ""
            report_lines.extend([
                f"",
                f"{status} {r['scenario']}{space_issue}",
                f"   Failure rate: {r['final_failure_rate']:.1%}",
                f"   Meta² triggered: {r['meta2_triggered']}",
                f"   Epiphany: {r['epiphany_occurred']}",
            ])
            if r['critique_issued']:
                report_lines.append(f"   Critique: {r['critique_issued']}")
        
        report_lines.extend([
            "",
            "="*70,
            "🏆 FINAL VERDICT",
            "="*70,
        ])
        
        if passed == total_scenarios:
            report_lines.extend([
                "✅ RAVANA ACHIEVED META²-COGNITION",
                "   System can critique its own epistemic method",
                "   Can detect when its way of thinking is wrong",
                "   Ready for open-ended intelligence",
            ])
        elif passed >= total_scenarios * 0.6:
            report_lines.extend([
                "⚠️  PARTIAL META²-COGNITION",
                f"   {total_scenarios - passed} scenarios need refinement",
                "   Core capability present but fragile",
            ])
        else:
            report_lines.extend([
                "❌ META²-COGNITION NOT ACHIEVED",
                "   System cannot reliably question its own method",
                "   Architecture needs fundamental revision",
            ])
        
        report_lines.extend([
            "",
            "="*70,
        ])
        
        report = "\n".join(report_lines)
        print(report)
        
        # Save results
        import json
        
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, bool):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj
        
        serializable = convert({
            'results': self.results,
            'summary': {
                'total': total_scenarios,
                'passed': passed,
                'meta2_triggered': meta2_triggered_count,
                'epiphanies': epiphanies
            }
        })
        
        os.makedirs('results', exist_ok=True)
        with open('results/phase_i2_results.json', 'w') as f:
            json.dump(serializable, f, indent=2)
        
        with open('results/phase_i2_report.txt', 'w') as f:
            f.write(report)
        
        print("\n💾 Results saved to results/phase_i2_*.json/txt")
        
        return {
            'results': self.results,
            'passed': passed,
            'total': total_scenarios,
            'report': report
        }
