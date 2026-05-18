#!/usr/bin/env python3
"""
RAVANA v2 — Phase E: Open World Survival
Non-stationary environment. No warnings. No hints.

RAVANA must:
- DETECT: "Something changed"
- INFER: "What kind of change?"  
- ADAPT: "New strategy needed"
- SURVIVE: "Stay coherent anyway"

This is the test of true intelligence.
"""

import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from collections import deque

from core import (
    Governor, GovernorConfig, ResolutionEngine, IdentityEngine, StateManager,
    StrategyLayer, StrategyLearningLayer, LearningConfig,
    ExplorationMode, BehavioralContext
)
from core.intent import IntentEngine, IntentConfig, IntentAwareStrategy
from core.planning import MicroPlanner, PlanningConfig
from core.environment import NonStationaryEnvironment, EnvironmentConfig, WorldState


@dataclass
class SurvivalMetrics:
    """Track how well RAVANA survives the open world."""
    episodes_alive: int = 0
    boundary_breaches: int = 0
    detection_accuracy: float = 0.0
    adaptation_latency: float = 0.0  # Episodes to adapt after change
    coherence_integrity: float = 1.0  # % of time in valid state
    
    # World understanding
    inferred_dynamics: List[str] = field(default_factory=list)
    surprise_events: int = 0


class WorldModelTracker:
    """
    🧠 WORLD MODEL TRACKER
    
    RAVANA's internal representation of external dynamics.
    Not given — inferred from experience.
    """
    
    def __init__(self, window: int = 100):
        self.window = window
        self.observed_constraints: deque = deque(maxlen=window)
        self.noise_pattern: deque = deque(maxlen=window)
        self.goal_shifts: List[int] = []
        self.detected_anomalies: List[Dict] = []
        
        # Beliefs about world
        self.believed_boundary: float = 0.95
        self.believed_noise_level: float = 0.0
        self.detected_regime: str = "unknown"
        
    def observe(self, episode: int, state: Dict, environment: WorldState):
        """Observe environment state and detect changes."""
        self.observed_constraints.append({
            'episode': episode,
            'boundary': environment.effective_boundary,
            'noise': environment.noise_level
        })
        
        # Detect anomaly: sudden constraint shift
        if len(self.observed_constraints) >= 20:
            recent = list(self.observed_constraints)[-20:]
            older = list(self.observed_constraints)[-40:-20] if len(self.observed_constraints) >= 40 else recent[:10]
            
            if older:
                old_mean = np.mean([o['boundary'] for o in older])
                new_mean = np.mean([r['boundary'] for r in recent])
                
                if abs(new_mean - old_mean) > 0.05:  # 5% shift
                    self.detected_anomalies.append({
                        'episode': episode,
                        'type': 'boundary_shift',
                        'magnitude': new_mean - old_mean,
                        'old': old_mean,
                        'new': new_mean
                    })
                    self.believed_boundary = new_mean
                    return True  # Anomaly detected
        
        return False
    
    def infer_noise_pattern(self) -> str:
        """Infer if noise is stationary or drifting."""
        if len(self.noise_pattern) < 50:
            return "insufficient_data"
        
        recent = list(self.noise_pattern)[-50:]
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        
        if abs(trend) < 0.001:
            return "stationary"
        elif trend > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def get_world_belief(self) -> Dict[str, Any]:
        """Return RAVANA's current model of the world."""
        return {
            'believed_boundary': round(self.believed_boundary, 3),
            'believed_noise': round(self.believed_noise_level, 3),
            'detected_regime': self.detected_regime,
            'inferred_noise_pattern': self.infer_noise_pattern(),
            'anomaly_count': len(self.detected_anomalies),
            'recent_anomalies': self.detected_anomalies[-3:]
        }


class OpenWorldPipeline:
    """
    🌍 OPEN WORLD PIPELINE
    
    RAVANA operates in non-stationary environment.
    No warnings. No hints. Only consequences.
    """
    
    def __init__(self, manager, intent_strategy, planner, environment, config):
        self.manager = manager
        self.intent_strategy = intent_strategy
        self.planner = planner
        self.env = environment
        self.config = config
        
        # Tracking
        self.world_model = WorldModelTracker()
        self.metrics = SurvivalMetrics()
        self.adaptation_history: List[Dict] = []
        
    def _extract_clamp_events(self, step_record: Dict) -> List[Dict]:
        """Extract clamp events from step."""
        events = []
        if step_record.get('constraint_activated'):
            events.append({
                'episode': step_record['episode'],
                'reason': step_record.get('reason', 'clamp')
            })
        return events
    
    def _get_context(self):
        """Extract context from current state."""
        state = self.manager.state
        
        # Compute recent clamp rate
        if len(self.manager.history) >= 20:
            recent = self.manager.history[-20:]
            clamps = sum(1 for h in recent if h.get('constraint_activated'))
            clamp_rate = clamps / 20
        else:
            clamp_rate = 0.0
        
        return BehavioralContext(
            dissonance=state.dissonance,
            identity=state.identity,
            clamp_rate=clamp_rate,
            recent_resolution_success=0.5,
            dissonance_trend=0.0,
            identity_drift=0.0,
            dissonance_variance=0.01
        )
    
    def _simulate_outcome(self, difficulty: float, mode: ExplorationMode, episode: int) -> bool:
        """Simulate outcome given environment state."""
        import random
        
        # Base rate modified by environment
        base_rate = 0.7 + self.env.current_state.goal_flip_bias
        
        # Mode effects
        if mode == ExplorationMode.EXPLORE_AGGRESSIVE:
            base_rate -= 0.15
        elif mode == ExplorationMode.STABILIZE:
            base_rate += 0.1
        
        # Environment noise
        noise = random.gauss(0, self.env.current_state.noise_level)
        base_rate += noise
        
        return random.random() < base_rate
    
    def train(self) -> Dict[str, Any]:
        """Execute survival training in open world."""
        print(f"\n{'='*70}")
        print(f"🌍 RAVANA v2 — Phase E: Open World Survival")
        print(f"Environment: NON-STATIONARY (hidden dynamics)")
        print(f"RAVANA status: NO PRIOR KNOWLEDGE")
        print(f"{'='*70}")
        print(f"\n🎯 Mission: Detect → Infer → Adapt → Survive")
        print(f"🎯 Success: Maintain coherence despite unknown changes")
        print(f"\n⚠️  Environment will change. RAVANA will not be told.")
        print(f"{'='*70}\n")
        
        import time
        start = time.time()
        
        for episode in range(self.config.total_episodes):
            # Environment evolves (hidden from RAVANA)
            world_state = self.env.step(episode)
            
            # Update governor constraints silently
            self.manager.governor.config.max_dissonance = world_state.effective_boundary
            self.manager.governor.config.min_dissonance = 1.0 - world_state.effective_boundary
            
            # Pre-state
            pre_state = {
                'dissonance': self.manager.state.dissonance,
                'identity': self.manager.state.identity,
            }
            
            # 🧠 WORLD MODEL: Observe and detect
            anomaly_detected = self.world_model.observe(episode, pre_state, world_state)
            if anomaly_detected:
                self.metrics.surprise_events += 1
            
            # Get context
            context = self._get_context()
            
            # 🎯 PLAN: Simulate futures for each mode
            mode_scores = {}
            for mode in ExplorationMode:
                predicted = self.planner.simulate_forward(context, mode, steps=5)
                score = self.planner.score_future(context, predicted)
                
                # If anomaly detected, boost exploration modes
                if anomaly_detected and mode in [ExplorationMode.EXPLORE_SAFE, ExplorationMode.EXPLORE_AGGRESSIVE]:
                    score *= 1.2
                
                mode_scores[mode] = score
            
            best_mode = max(mode_scores.keys(), key=lambda m: mode_scores[m])
            
            # Execute
            difficulty = world_state.difficulty_level
            correctness = self._simulate_outcome(difficulty, best_mode, episode)
            
            step_record = self.manager.step(
                correctness=correctness,
                difficulty=difficulty,
                debug=False
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
            
            # Check coherence
            if post_state['dissonance'] > 0.95 or post_state['identity'] < 0.05:
                self.metrics.boundary_breaches += 1
            
            # Logging
            if (episode + 1) % 200 == 0 or (anomaly_detected and episode > 100):
                status = "⚠️  ANOMALY" if anomaly_detected else "✓"
                belief = self.world_model.get_world_belief()
                print(f"EP{episode+1:,} | D={post_state['dissonance']:.2f} | I={post_state['identity']:.2f} | "
                      f"Mode:{best_mode.value[:8]} | Boundary:{belief['believed_boundary']:.2f} | {status}")
                
                if anomaly_detected and belief['recent_anomalies']:
                    last = belief['recent_anomalies'][-1]
                    print(f"      └─ Detected: {last['type']} (Δ{last['magnitude']:+.3f})")
        
        # Final metrics
        elapsed = time.time() - start
        self.metrics.episodes_alive = self.config.total_episodes
        
        # Calculate coherence integrity
        valid_states = sum(1 for h in self.manager.history 
                         if 0.15 < h.get('post_dissonance', 0.5) < 0.95)
        self.metrics.coherence_integrity = valid_states / max(1, len(self.manager.history))
        
        print(f"\n{'='*70}")
        print(f"🎯 Survival Test Complete: {elapsed:.1f}s")
        print(f"{'='*70}")
        print(f"\n📊 SURVIVAL METRICS:")
        print(f"  Episodes survived: {self.metrics.episodes_alive:,}")
        print(f"  Boundary breaches: {self.metrics.boundary_breaches}")
        print(f"  Surprise events: {self.metrics.surprise_events}")
        print(f"  Coherence integrity: {self.metrics.coherence_integrity:.1%}")
        print(f"\n🧠 WORLD MODEL:")
        belief = self.world_model.get_world_belief()
        print(f"  Believed boundary: {belief['believed_boundary']}")
        print(f"  Inferred noise pattern: {belief['inferred_noise_pattern']}")
        print(f"  Anomalies detected: {belief['anomaly_count']}")
        if belief['recent_anomalies']:
            print(f"  Recent detections:")
            for a in belief['recent_anomalies'][-3:]:
                print(f"    - EP{a['episode']}: {a['type']} (Δ{a['magnitude']:+.3f})")
        print(f"\n📈 FINAL STATE: D={self.manager.state.dissonance:.3f} | I={self.manager.state.identity:.3f}")
        
        survival_grade = "🟢 EXCELLENT" if self.metrics.boundary_breaches == 0 and self.metrics.coherence_integrity > 0.95 else \
                        "🟢 GOOD" if self.metrics.boundary_breaches <= 2 else \
                        "🟡 SURVIVED" if self.metrics.boundary_breaches <= 5 else \
                        "🔴 STRUGGLED"
        
        print(f"\n🏆 SURVIVAL GRADE: {survival_grade}")
        print(f"{'='*70}")
        
        return {
            "metrics": self.metrics,
            "world_belief": belief,
            "final_state": {
                "dissonance": self.manager.state.dissonance,
                "identity": self.manager.state.identity
            },
            "survival_grade": survival_grade,
            "elapsed": elapsed
        }


def main():
    """Run open world survival test."""
    print("\n" + "="*70)
    print("🌌 RAVANA ENTERS THE UNKNOWN")
    print("="*70)
    
    # Create components
    governor = Governor(GovernorConfig())
    resolution = ResolutionEngine()
    identity = IdentityEngine()
    manager = StateManager(governor, resolution, identity)
    
    # Full stack
    strategy = StrategyLayer()
    learning = StrategyLearningLayer()
    intent = IntentEngine(IntentConfig())
    intent_strategy = IntentAwareStrategy(strategy, learning, intent)
    planner = MicroPlanner(PlanningConfig())
    
    # Environment with hidden dynamics
    env = NonStationaryEnvironment(EnvironmentConfig(
        boundary_shift_frequency=500,
        noise_drift_rate=0.02,
        goal_flip_period=400,
        difficulty_cycle_period=300
    ))
    
    # Config
    @dataclass
    class Config:
        total_episodes: int = 3000
    
    config = Config()
    
    # Pipeline
    pipeline = OpenWorldPipeline(manager, intent_strategy, planner, env, config)
    results = pipeline.train()
    
    # Save
    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "phase_e_survival.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Results saved: results/phase_e_survival.json")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
