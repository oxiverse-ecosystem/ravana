#!/usr/bin/env python3
"""
RAVANA v2 — Phase B Training Entry Point
Adaptive learning from clamp diagnostics.
"""

import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core import Governor, GovernorConfig, ResolutionEngine, IdentityEngine, StateManager
from core.adaptation import PolicyTweakLayer, AdaptiveGovernorBridge, AdaptationConfig
from training.pipeline import TrainingPipeline, TrainingConfig


def main():
    """Execute Phase B training with adaptive layer."""
    print("=" * 70)
    print("RAVANA v2 — Phase B: Adaptive Intelligence")
    print("Learning from constitutional corrections")
    print("=" * 70)
    
    # Create components
    governor = Governor(GovernorConfig(
        max_dissonance=0.95,
        min_dissonance=0.15,
        max_identity=0.95,
        min_identity=0.10,
        target_dissonance=0.30, 
        dissonance_target=0.30, 
        identity_target=0.85,
        use_smoothed_dissonance=True, # Enable smoothing
        smoothing_alpha=0.15          # Smooth alpha
    ))
    
    # 🧠 Phase B: Add adaptation layer
    adaptation = PolicyTweakLayer(AdaptationConfig(
        learning_rate=0.05,
        clamp_penalty=5.0,
        exploration_bonus=0.20,
        max_tweak=0.05,
    ))
    
    # Bridge adaptation into governor flow
    adaptive_bridge = AdaptiveGovernorBridge(governor, adaptation)
    
    # Create resolution and identity engines
    resolution = ResolutionEngine(partial_threshold=0.15)
    identity = IdentityEngine(
        initial_strength=0.5,
        recovery_bias=0.1,
        stability_threshold=0.85
    )
    
    # Create state manager (Phase B: uses adaptive bridge)
    manager = PhaseBStateManager(adaptive_bridge, resolution, identity, smoothing_alpha=0.15)
    
    # Training config
    config = TrainingConfig(
        total_episodes=10000,  # 10K episodes for paper-grade convergence
        log_interval=500,      # Less frequent logs
        debug_first_n=0,
    )
    
    # Run training
    pipeline = PhaseBPipeline(manager, config)
    results = pipeline.train()
    
    # Final reports
    print("\n" + "=" * 70)
    print("PHASE B COMPLETE")
    print("=" * 70)
    print(governor.get_clamp_report())
    print()
    print(adaptation.get_learning_report())
    
    return results


class PhaseBStateManager:
    """
    State manager for Phase B with adaptive governor bridge.
    """
    
    def __init__(self, adaptive_bridge, resolution_engine, identity_engine, smoothing_alpha=0.2):
        from core.state import CognitiveState
        self.state = CognitiveState()
        self.bridge = adaptive_bridge
        self.governor = adaptive_bridge.governor  # Expose for pipeline access
        self.resolution = resolution_engine
        self.identity = identity_engine
        self.smoothing_alpha = smoothing_alpha
        self.history = []
        
    def step(self, correctness, difficulty=0.5, debug=False):
        """Execute one adaptive cognitive step."""
        pre_d = self.state.dissonance
        pre_i = self.state.identity
        pre_ema = self.state.dissonance_ema
        
        # 1. PREDICT: Estimate what WOULD happen to dissonance
        expected_change = -0.1 if correctness else 0.15
        estimated_post_d = np.clip(pre_d + expected_change, 0.15, 0.95)
        
        # 2. RESOLUTION: Compute what the resolution engine THINKS
        resolution_result = self.resolution.compute(
            episode=self.state.episode,
            prev_dissonance=pre_d,
            current_dissonance=estimated_post_d, 
            correctness=correctness,
            difficulty=difficulty,
            source="phase_b_step"
        )
        
        # 3. IDENTITY: Compute desired update from resolution delta
        desired_identity = self.identity.compute_update(
            resolution_delta=resolution_result["delta"],
            resolution_success=resolution_result["full_resolution"],
            regulated_identity_delta=0.0,
            current_dissonance=pre_d,
            resolution_streak=self.state.resolution_streak
        )
        identity_delta = desired_identity - pre_i
        
        # Adaptive bridge regulation
        from core.governor import CognitiveSignals
        signals = CognitiveSignals(
            dissonance_delta=resolution_result["delta"],
            identity_delta=identity_delta,
            exploration_drive=0.0,
            resolution_potential=resolution_result["partial_credit"],
            source="phase_b"
        )
        
        # Use smoothed dissonance for governor if enabled
        reg_d = pre_ema if getattr(self.governor.config, 'use_smoothed_dissonance', False) else pre_d
        
        regulated = self.bridge.step(reg_d, pre_i, signals, self.state.episode)
        
        # Apply regulated changes
        new_dissonance = np.clip(
            pre_d + regulated.dissonance_delta,
            self.governor.config.min_dissonance,
            self.governor.config.max_dissonance
        )
        
        # Update EMA
        new_ema = (1 - self.smoothing_alpha) * pre_ema + self.smoothing_alpha * new_dissonance
        
        regulated_identity = self.identity.compute_update(
            resolution_delta=resolution_result["delta"],
            resolution_success=resolution_result["full_resolution"],
            regulated_identity_delta=regulated.identity_delta,
            current_dissonance=pre_d,
            resolution_streak=self.state.resolution_streak
        )
        new_identity = np.clip(
            regulated_identity,
            self.governor.config.min_identity,
            self.governor.config.max_identity
        )
        
        # Update state
        from core.state import CognitiveState
        wisdom_generated = resolution_result["wisdom_generated"]
        
        self.state = CognitiveState(
            dissonance=new_dissonance,
            identity=new_identity,
            dissonance_ema=new_ema,
            episode=self.state.episode + 1,
            cycle=self.state.cycle + 1,
            accumulated_wisdom=self.state.accumulated_wisdom + wisdom_generated,
            resolution_streak=resolution_result["streak"],
            last_update_reason=regulated.reason,
            constraint_activated=regulated.capped or regulated.dampened
        )
        
        step_record = {
            "episode": self.state.episode,
            "pre_dissonance": pre_d,
            "post_dissonance": new_dissonance,
            "pre_identity": pre_i,
            "post_identity": new_identity,
            "resolution": resolution_result,
            "mode": regulated.mode.value,
            "wisdom": wisdom_generated,
            "reason": regulated.reason,
        }
        self.history.append(step_record)
        
        if debug:
            print(f"  [EP{step_record['episode']:04d}] "
                  f"D:{step_record['pre_dissonance']:.3f}→{step_record['post_dissonance']:.3f} "
                  f"I:{step_record['pre_identity']:.3f}→{step_record['post_identity']:.3f} "
                  f"Mode:{step_record['mode'][:4]}")
        
        return step_record
    
    def get_status(self):
        """Full system status including adaptation."""
        return {
            "state": self.state.snapshot(),
            "governor": self.governor.get_status(),
            "identity": self.identity.get_status(),
            "resolution": self.resolution.get_memory_status(),
            "adaptation": self.bridge.adaptation.get_status(),
            "total_steps": len(self.history),
        }


class PhaseBPipeline(TrainingPipeline):
    """Extended pipeline for Phase B with adaptation reporting."""
    
    def _generate_summary(self, elapsed):
        """Generate Phase B summary with adaptation metrics."""
        status = self.manager.get_status()
        
        clamp_report = self.manager.governor.get_clamp_report()
        clamp_metrics = self.manager.governor.get_clamp_metrics()
        adaptation_report = self.manager.bridge.adaptation.get_learning_report()
        
        summary = {
            "phase": "B",
            "total_episodes": self.config.total_episodes,
            "elapsed_seconds": elapsed,
            "final_state": status["state"],
            "governor_stats": status["governor"],
            "adaptation_stats": status["adaptation"],
            "clamp_metrics": clamp_metrics,
        }
        
        # Save main summary
        import json
        from pathlib import Path
        with open(self.output_dir / "phase_b_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save clamp events
        if hasattr(self.manager.governor.clamp_diagnostics, 'events'):
            events_data = [
                {
                    "episode": e.episode,
                    "variable": e.variable,
                    "before": e.before,
                    "after": e.after,
                    "correction": e.correction,
                    "layer": e.layer,
                    "reason": e.reason
                }
                for e in self.manager.governor.clamp_diagnostics.events
            ]
            with open(self.output_dir / "phase_b_clamp_events.json", "w") as f:
                json.dump(events_data, f, indent=2)
        
        # Print reports
        print(f"\n{clamp_report}")
        print(f"\n{adaptation_report}")
        
        return summary


if __name__ == "__main__":
    import numpy as np
    from core.resolution import ResolutionEngine
    from core.identity import IdentityEngine
    main()
