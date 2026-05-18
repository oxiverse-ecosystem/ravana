#!/usr/bin/env python3
"""
RAVANA v2 — Phase D.5: Micro-Planning + Emergence Testing

Tests if intent evolution is:
- True emergence (adapts to conditions)
- Or reward-shaped inevitability (follows hidden gradient)

Includes:
1. Future-simulation planning layer
2. Weight perturbation tests
3. Noise injection tests  
4. Temptation state tests
5. Intent volatility tracking
"""

import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple

from core import (
    Governor, GovernorConfig, ResolutionEngine, IdentityEngine, StateManager,
    StrategyLayer, StrategyConfig, StrategyLearningLayer, LearningConfig,
    ExplorationMode
)
from core.intent import IntentEngine, IntentConfig, IntentAwareStrategy, SystemObjective
from core.planning import MicroPlanner, PlanningConfig


@dataclass
class D5Config:
    """Phase D.5 test configuration."""
    total_episodes: int = 2000
    log_interval: int = 100
    debug_first_n: int = 0
    
    # Test conditions
    test_name: str = "default"
    
    # Weight perturbations (to test if behavior is reward-shaped)
    clamp_weight: float = 1.5
    explore_weight: float = 1.0
    stabilize_weight: float = 1.0
    identity_weight: float = 1.0
    
    # Noise injection (tests robustness)
    noise_injection_episodes: List[int] = field(default_factory=list)
    noise_magnitude: float = 0.0
    
    # Temptation states (tests tradeoff capability)
    temptation_episodes: List[int] = field(default_factory=list)
    temptation_dissonance_boost: float = 0.0


class IntentVolatilityTracker:
    """
    🧠 INTENT VOLATILITY METRIC
    
    Tracks:
    - intent_switch_rate: How often dominant objective changes
    - intent_duration: How long each intent persists
    - intent_entropy: Diversity of intent exploration
    
    Goal: Not too stable (rigid), not too chaotic (unfocused)
    """
    
    def __init__(self, window: int = 50):
        self.window = window
        self.intent_history: List[str] = []
        self.switch_times: List[int] = []
        self.last_intent: str = None
        self.episode_count: int = 0
        
    def record(self, episode: int, dominant_intent: str):
        """Record intent at each episode."""
        self.episode_count = episode
        
        if dominant_intent != self.last_intent:
            if self.last_intent is not None:
                self.switch_times.append(episode)
            self.last_intent = dominant_intent
        
        self.intent_history.append(dominant_intent)
        
    def get_metrics(self) -> Dict[str, Any]:
        """Compute intent volatility metrics."""
        if len(self.intent_history) < 10:
            return {"status": "insufficient_data"}
        
        # Switch rate: switches per 100 episodes
        total_switches = len(self.switch_times)
        switch_rate = (total_switches / max(1, self.episode_count)) * 100
        
        # Intent duration: mean time between switches
        if len(self.switch_times) >= 2:
            durations = [
                self.switch_times[i+1] - self.switch_times[i]
                for i in range(len(self.switch_times) - 1)
            ]
            mean_duration = np.mean(durations)
        else:
            mean_duration = self.episode_count
        
        # Intent entropy: diversity of exploration
        from collections import Counter
        recent = self.intent_history[-self.window:] if len(self.intent_history) > self.window else self.intent_history
        counts = Counter(recent)
        total = len(recent)
        entropy = -sum((c/total) * np.log2(c/total) for c in counts.values())
        max_entropy = np.log2(4)  # 4 possible objectives
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return {
            "switch_rate_per_100": round(switch_rate, 2),
            "mean_intent_duration": round(mean_duration, 1),
            "intent_entropy": round(normalized_entropy, 3),
            "total_switches": total_switches,
            "assessment": self._assess_volatility(switch_rate, normalized_entropy)
        }
    
    def _assess_volatility(self, switch_rate: float, entropy: float) -> str:
        """Assess if volatility is healthy."""
        if switch_rate < 2:
            return "🟡 TOO_STABLE - May be stuck in local optimum"
        elif switch_rate > 15:
            return "🔴 TOO_CHAOTIC - Unfocused, not learning"
        elif entropy < 0.3:
            return "🟡 LOW_DIVERSITY - Exploring few objectives"
        else:
            return "🟢 HEALTHY - Adaptive but focused"


class PlanningTrainingPipeline:
    """Training pipeline with micro-planning and emergence testing."""
    
    def __init__(self, manager, intent_strategy, planner, config: D5Config):
        self.manager = manager
        self.intent_strategy = intent_strategy
        self.planner = planner
        self.config = config
        
        # Tracking
        self.intent_tracker = IntentVolatilityTracker()
        self.results: List[Dict] = []
        
    def _compute_difficulty(self, episode: int) -> float:
        """Adaptive difficulty with temptation states."""
        base = 0.3 + 0.6 * (episode / self.config.total_episodes)
        
        # 🧪 TEMPTATION STATE: Boost dissonance for potential gain
        if episode in self.config.temptation_episodes:
            # High difficulty = high dissonance risk
            base += self.config.temptation_dissonance_boost
        
        return min(0.95, base)
    
    def _simulate_outcome(self, difficulty: float, mode: ExplorationMode, episode: int) -> bool:
        """Simulate episode outcome."""
        import random
        
        base_rate = 0.7
        
        # Mode affects success
        if mode == ExplorationMode.EXPLORE_AGGRESSIVE:
            base_rate -= 0.15  # Higher risk
        elif mode == ExplorationMode.STABILIZE:
            base_rate += 0.1   # Lower risk
        elif mode == ExplorationMode.RECOVER:
            base_rate += 0.2   # Safety mode
        
        # Noise injection
        if episode in self.config.noise_injection_episodes:
            base_rate += random.uniform(-self.config.noise_magnitude, self.config.noise_magnitude)
        
        return random.random() < base_rate
    
    def _get_context(self):
        """Extract behavioral context."""
        from core.strategy import BehavioralContext
        
        state = self.manager.state
        clamp_rate = self._get_recent_clamp_rate()
        
        return BehavioralContext(
            dissonance=state.dissonance,
            identity=state.identity,
            clamp_rate=clamp_rate,
            recent_resolution_success=0.5,
            dissonance_trend=0.0,
            identity_drift=0.0,
            dissonance_variance=0.01,
        )
    
    def _get_recent_clamp_rate(self, window: int = 20) -> float:
        """Compute recent clamp rate."""
        if len(self.manager.history) < window:
            return 0.0
        recent = self.manager.history[-window:]
        clamps = sum(1 for h in recent if h.get('reason', '').startswith('clamp'))
        return clamps / window
    
    def _extract_clamp_events(self, step_record: Dict) -> List[Dict]:
        """Extract clamp events from step record."""
        events = []
        if step_record.get('constraint_activated'):
            events.append({
                'episode': step_record['episode'],
                'reason': step_record.get('reason', 'clamp')
            })
        return events
    
    def train(self) -> Dict[str, Any]:
        """Execute training with planning and volatility tracking."""
        print(f"\n{'='*60}")
        print(f"RAVANA v2 — Phase D.5: Micro-Planning + Emergence Test")
        print(f"Test: {self.config.test_name}")
        print(f"{'='*60}")
        
        import time
        start = time.time()
        
        for episode in range(self.config.total_episodes):
            # Capture pre-state
            pre_state = {
                'dissonance': self.manager.state.dissonance,
                'identity': self.manager.state.identity,
            }
            
            # Get context
            context = self._get_context()
            clamp_events = self._extract_clamp_events(self.manager.history[-1] if self.manager.history else {})
            
            # 🎯 PLANNING: Simulate future for each mode
            mode_predictions = {}
            for mode in ExplorationMode:
                predicted = self.planner.simulate_forward(
                    context, mode, steps=5
                )
                score = self.planner.score_future(context, predicted)
                mode_predictions[mode] = {
                    'predicted': predicted,
                    'score': score
                }
            
            # Select best mode by predicted outcome
            best_mode = max(mode_predictions.keys(), key=lambda m: mode_predictions[m]['score'])
            
            # Execute with selected mode
            difficulty = self._compute_difficulty(episode)
            correctness = self._simulate_outcome(difficulty, best_mode, episode)
            
            step_record = self.manager.step(
                correctness=correctness,
                difficulty=difficulty,
                debug=episode < self.config.debug_first_n
            )
            
            # Post-state
            post_state = {
                'dissonance': self.manager.state.dissonance,
                'identity': self.manager.state.identity,
            }
            
            # Update layers
            new_clamps = self._extract_clamp_events(step_record)
            self.intent_strategy.update_after_step(
                episode, pre_state, post_state, best_mode, new_clamps
            )
            
            # Track intent volatility
            intent_status = self.intent_strategy.intent.get_current_intent()
            dominant = intent_status['dominant_objective']
            self.intent_tracker.record(episode, dominant)
            
            # Logging
            if (episode + 1) % self.config.log_interval == 0:
                bias = self.intent_strategy.intent.compute_mode_bias(best_mode)
                print(f"EP{episode+1:,} | D={post_state['dissonance']:.2f} | I={post_state['identity']:.2f} | "
                      f"Mode:{best_mode.value[:8]} | Wants:{dominant[:10]} | "
                      f"Plan:{mode_predictions[best_mode]['score']:.2f}")
        
        # Final metrics
        elapsed = time.time() - start
        volatility = self.intent_tracker.get_metrics()
        
        print(f"\n{'='*60}")
        print(f"Test complete: {elapsed:.1f}s | Final: D={self.manager.state.dissonance:.3f} I={self.manager.state.identity:.3f}")
        print(f"Intent Volatility: {volatility['assessment']}")
        print(f"  - Switch rate: {volatility.get('switch_rate_per_100', 'N/A')} per 100 ep")
        print(f"  - Mean duration: {volatility.get('mean_intent_duration', 'N/A')} ep")
        print(f"  - Entropy: {volatility.get('intent_entropy', 'N/A')}")
        print(f"{'='*60}")
        
        return {
            "test_name": self.config.test_name,
            "elapsed": elapsed,
            "final_state": {
                "dissonance": self.manager.state.dissonance,
                "identity": self.manager.state.identity,
            },
            "volatility": volatility,
            "intent_final": self.intent_strategy.intent.get_current_intent(),
        }


def run_test(test_config: D5Config) -> Dict[str, Any]:
    """Run a single test configuration."""
    # Create components
    governor = Governor(GovernorConfig())
    resolution = ResolutionEngine()
    identity = IdentityEngine()
    manager = StateManager(governor, resolution, identity)
    
    # Strategy layers
    strategy = StrategyLayer()
    learning = StrategyLearningLayer()
    
    # Intent with perturbed weights
    intent = IntentEngine(IntentConfig())
    intent.objectives[SystemObjective.MINIMIZE_CLAMPS].weight = test_config.clamp_weight
    intent.objectives[SystemObjective.EXPLORE].weight = test_config.explore_weight
    intent.objectives[SystemObjective.STABILIZE].weight = test_config.stabilize_weight
    intent.objectives[SystemObjective.OPTIMIZE_IDENTITY].weight = test_config.identity_weight
    
    intent_strategy = IntentAwareStrategy(strategy, learning, intent)
    
    # Planner
    planner = MicroPlanner(PlanningConfig())
    
    # Pipeline
    pipeline = PlanningTrainingPipeline(manager, intent_strategy, planner, test_config)
    
    return pipeline.train()


def main():
    """Run emergence test suite."""
    print("\n" + "="*60)
    print("RAVANA D.5: EMERGENCE TEST SUITE")
    print("="*60)
    print("\n🧪 Testing: Is intent evolution emergent or reward-shaped?")
    
    results = {}
    
    # Test 1: Baseline
    print("\n[TEST 1] Baseline weights")
    results['baseline'] = run_test(D5Config(
        test_name="baseline",
        clamp_weight=1.5,
        explore_weight=1.0,
        stabilize_weight=1.0,
        identity_weight=1.0
    ))
    
    # Test 2: Equal weights (remove bias)
    print("\n[TEST 2] Equal weights (no reward shaping)")
    results['equal_weights'] = run_test(D5Config(
        test_name="equal_weights",
        clamp_weight=1.0,
        explore_weight=1.0,
        stabilize_weight=1.0,
        identity_weight=1.0
    ))
    
    # Test 3: Exploration biased
    print("\n[TEST 3] Exploration-biased weights")
    results['explore_biased'] = run_test(D5Config(
        test_name="explore_biased",
        clamp_weight=0.5,
        explore_weight=2.0,
        stabilize_weight=0.5,
        identity_weight=1.0
    ))
    
    # Test 4: With noise injection
    print("\n[TEST 4] Noise injection (robustness test)")
    results['with_noise'] = run_test(D5Config(
        test_name="with_noise",
        noise_injection_episodes=[500, 1000, 1500],
        noise_magnitude=0.2
    ))
    
    # Test 5: Temptation states
    print("\n[TEST 5] Temptation states (tradeoff test)")
    results['temptation'] = run_test(D5Config(
        test_name="temptation",
        temptation_episodes=[600, 1200, 1800],
        temptation_dissonance_boost=0.15
    ))
    
    # Summary
    print("\n" + "="*60)
    print("EMERGENCE TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        vol = result['volatility']
        print(f"\n{test_name}:")
        print(f"  Final D/I: {result['final_state']['dissonance']:.3f} / {result['final_state']['identity']:.3f}")
        print(f"  Volatility: {vol.get('assessment', 'N/A')}")
        print(f"  Dominant intent: {result['intent_final'].get('dominant_objective', 'N/A')}")
    
    # Save results
    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "d5_emergence_tests.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Results saved to: results/d5_emergence_tests.json")
    print("="*60)
    
    return results


if __name__ == "__main__":
    main()
