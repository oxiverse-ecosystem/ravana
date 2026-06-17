"""
RAVANA v2 — TRAINING PIPELINE
Clean training loop with governor-gated state evolution.

Phase A: Core loop + governor + resolution + identity
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training configuration."""
    total_episodes: int = 100000
    log_interval: int = 100
    checkpoint_interval: int = 1000
    debug_first_n: int = 50
    
    # Difficulty schedule
    initial_difficulty: float = 0.3
    max_difficulty: float = 0.9
    difficulty_ramp_episodes: int = 50000


class TrainingPipeline:
    """
    Clean training pipeline with debug hooks.
    """
    
    def __init__(self, state_manager, config: TrainingConfig = None):
        self.manager = state_manager
        self.config = config or TrainingConfig()
        
        # Metrics tracking
        self.metrics: List[Dict] = []
        
        # Output
        project_root = Path(__file__).resolve().parent.parent
        self.output_dir = project_root / "results"
        self.output_dir.mkdir(exist_ok=True)
        
    def _compute_difficulty(self, episode: int) -> float:
        """Adaptive difficulty: starts easy, ramps to max."""
        if episode >= self.config.difficulty_ramp_episodes:
            return self.config.max_difficulty
        
        progress = episode / self.config.difficulty_ramp_episodes
        return self.config.initial_difficulty + (self.config.max_difficulty - self.config.initial_difficulty) * progress
    
    def _simulate_outcome(self, difficulty: float) -> bool:
        """Simulate episode outcome based on difficulty."""
        import random
        # Higher difficulty = lower success rate
        base_success = 0.7
        success_rate = base_success - (difficulty - 0.3) * 0.4
        return random.random() < success_rate
    
    def train(self) -> Dict[str, Any]:
        """Execute full training run."""
        print(f"=" * 60)
        print(f"RAVANA v2 — Phase A Training")
        print(f"Total episodes: {self.config.total_episodes:,}")
        print(f"Governor: CENTRAL (non-optional)")
        print(f"=" * 60)
        
        start_time = time.time()
        
        for episode in range(self.config.total_episodes):
            # Compute difficulty
            difficulty = self._compute_difficulty(episode)
            
            # Simulate outcome
            correctness = self._simulate_outcome(difficulty)
            
            # Execute cognitive step (GOVERNOR-GATED)
            debug = episode < self.config.debug_first_n
            step_record = self.manager.step(
                correctness=correctness,
                difficulty=difficulty,
                debug=debug
            )
            
            # Periodic logging
            if (episode + 1) % self.config.log_interval == 0:
                self._log_progress(episode + 1, step_record, difficulty)
            
            # Hard assertions (debug only)
            if debug:
                self._assert_state_valid()
        
        # Final summary
        elapsed = time.time() - start_time
        summary = self._generate_summary(elapsed)
        
        print(f"\n{'=' * 60}")
        print(f"Training complete: {elapsed:.1f}s")
        print(f"Final: D={self.manager.state.dissonance:.3f} I={self.manager.state.identity:.3f}")
        print(f"{'=' * 60}")
        
        return summary
    
    def _log_progress(self, episode: int, record: Dict, difficulty: float):
        """Log training progress."""
        state = self.manager.state
        
        print(f"EP{episode:,}/{self.config.total_episodes:,} | "
              f"D={state.dissonance:.3f} | "
              f"I={state.identity:.3f} | "
              f"W={state.accumulated_wisdom:.2f} | "
              f"Mode:{record['mode'][:3]} | "
              f"Diff:{difficulty:.2f}")
    
    def _assert_state_valid(self):
        """Hard assertions for debugging."""
        state = self.manager.state
        config = self.manager.governor.config
        
        # These should NEVER trigger if governor is working
        assert state.dissonance <= config.max_dissonance + 0.01, \
            f"DISSONANCE CEILING BREACH: {state.dissonance} > {config.max_dissonance}"
        
        assert state.dissonance >= config.min_dissonance - 0.01, \
            f"DISSONANCE FLOOR BREACH: {state.dissonance} < {config.min_dissonance}"
        
        assert state.identity >= config.min_identity - 0.01, \
            f"IDENTITY FLOOR BREACH: {state.identity} < {config.min_identity}"
    
    def _generate_summary(self, elapsed: float) -> Dict[str, Any]:
        """Generate training summary."""
        status = self.manager.get_status()
        
        # 🔴 Get clamp diagnostics report
        clamp_report = self.manager.governor.get_clamp_report()
        clamp_metrics = self.manager.governor.get_clamp_metrics()
        
        summary = {
            "total_episodes": self.config.total_episodes,
            "elapsed_seconds": elapsed,
            "final_state": status["state"],
            "governor_stats": status["governor"],
            "resolution_stats": status["resolution"],
            "identity_stats": status["identity"],
            # 🔴 Clamp diagnostics
            "clamp_metrics": clamp_metrics,
        }
        
        # Save main summary
        with open(self.output_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # 🔴 Save detailed clamp event log if available
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
            with open(self.output_dir / "clamp_events.json", "w") as f:
                json.dump(events_data, f, indent=2)
        
        # 🔴 Print clamp report
        print(f"\n{clamp_report}")
        
        return summary


def main():
    """Entry point for training."""
    from ..core.governor import Governor, GovernorConfig
    from ..core.resolution import ResolutionEngine
    from ..core.identity import IdentityEngine
    from ..core.state import StateManager
    
    # Create components
    governor = Governor(GovernorConfig(
        max_dissonance=0.95,
        min_dissonance=0.15,
        max_identity=0.95,
        min_identity=0.10,
        dissonance_target=0.45,
        identity_target=0.65,
    ))
    
    resolution = ResolutionEngine(partial_threshold=0.15)
    identity = IdentityEngine(initial_strength=0.5)
    
    # Create state manager (wires everything together)
    manager = StateManager(governor, resolution, identity)
    
    # Create and run pipeline
    config = TrainingConfig(
        total_episodes=1000,  # Start small for testing
        log_interval=100,
        debug_first_n=20,
    )
    
    pipeline = TrainingPipeline(manager, config)
    results = pipeline.train()
    
    return results


if __name__ == "__main__":
    main()
