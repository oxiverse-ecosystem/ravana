"""
RAVANA v2 — Python Wrapper for Interface Agent
Wraps the RAVANA v2 StateManager for human-readable queries.
"""

import sys
import os
from pathlib import Path

# Add RAVANA v2 to path
RAVANA_PATH = "/home/workspace/Projects/ravana-v2"
sys.path.insert(0, RAVANA_PATH)

from core.governor import Governor, GovernorConfig, RegulationMode
from core.state import StateManager, CognitiveState
from core.identity import IdentityEngine
from core.resolution import ResolutionEngine
from core.memory import RavanaMemorySystem


class RavanaWrapper:
    """
    Human-readable wrapper around RAVANA v2's StateManager.
    
    Provides:
    - Simple query methods (get_status, step, diagnose)
    - Natural language reasoning extraction
    - Cognitive state serialization for LLM interpretation
    """
    
    def __init__(self, config_path: str = None):
        self._init_ravana()
        self.episode_count = 0
        self.lesson_log = []
        
    def _init_ravana(self):
        """Initialize RAVANA v2 core components."""
        governor_config = GovernorConfig(
            max_dissonance=0.95,
            min_dissonance=0.15,
            target_dissonance=0.30,
            max_identity=0.95,
            min_identity=0.10,
            soft_limit=0.70,
            boundary_k=12.0,
            use_smoothed_dissonance=True,
            smoothing_alpha=0.2,
        )
        
        self.governor = Governor(config=governor_config)
        self.identity_engine = IdentityEngine()
        self.resolution_engine = ResolutionEngine(partial_threshold=0.15)
        
        self.state_manager = StateManager(
            governor=self.governor,
            resolution_engine=self.resolution_engine,
            identity_engine=self.identity_engine,
            smoothing_alpha=0.2
        )
        
        self.memory = RavanaMemorySystem()
    
    def step(self, correctness: bool, difficulty: float = 0.5, reason: str = "") -> dict:
        """
        Execute one cognitive step.
        
        Args:
            correctness: Did the action lead to correct outcome?
            difficulty: How hard was the decision? (0.0-1.0)
            reason: Why was this correct/wrong? (for learning)
        
        Returns:
            Step result dict with state snapshot and reasoning
        """
        result = self.state_manager.step(
            correctness=correctness,
            difficulty=difficulty,
            debug=False
        )
        self.episode_count += 1
        
        # Enrich with reasoning
        result['episode'] = self.episode_count
        result['human_reason'] = reason
        result['mode_label'] = self._mode_to_label(result['mode'])
        
        return result
    
    def query(self, question: str) -> dict:
        """
        Query RAVANA's current cognitive state in response to a question.
        
        Returns dict with:
            - current_state (D, I, mode, wisdom)
            - reasoning (why RAVANA is in this state)
            - recommendation (what RAVANA would do)
            - health (governor clamp report)
        """
        state = self.state_manager.get_status()
        clamp_report = self.governor.get_clamp_report()
        clamp_metrics = self.governor.get_clamp_metrics()
        
        return {
            "episode": self.episode_count,
            "cognitive_state": {
                "dissonance": state['state']['dissonance'],
                "dissonance_ema": state['state']['dissonance_ema'],
                "identity": state['state']['identity'],
                "wisdom": state['state']['wisdom'],
                "resolution_streak": state['state'].get('resolution_streak', state['state']['episode'] - 1),
            },
            "governor": {
                "mode": state['governor'].get('current_mode', state['governor'].get('mode', 'unknown')),
                "mode_distribution": state['governor'].get('mode_distribution', {}),
            },
            "health": {
                "clamp_rate": clamp_metrics.get('clamp_rate', 0),
                "alignment_score": clamp_metrics.get('alignment_score', 0),
                "total_clamps": clamp_metrics.get('total_clamps', 0),
            },
            "question": question,
            "reasoning": self._generate_reasoning(state),
            "recommendation": self._generate_recommendation(state),
        }
    
    def simulate_action(self, action: str, expected_outcome: str) -> dict:
        """
        Simulate what RAVANA would do if action X led to outcome Y.
        
        Args:
            action: What RAVANA would do (natural language)
            expected_outcome: What would happen (success/failure)
        
        Returns:
            Predicted cognitive state change
        """
        correctness = "success" in expected_outcome.lower() or "correct" in expected_outcome.lower()
        difficulty = 0.5  # Default
        
        # Predict next state
        pre_state = self.state_manager.state.snapshot()
        result = self.step(correctness=correctness, difficulty=difficulty, reason=f"Simulated: {action} → {expected_outcome}")
        
        return {
            "pre_dissonance": pre_state['dissonance'],
            "post_dissonance": result['post_dissonance'],
            "pre_identity": pre_state['identity'],
            "post_identity": result['post_identity'],
            "predicted_wisdom_gain": result['wisdom'],
            "mode": result['mode'],
            "action": action,
            "expected_outcome": expected_outcome,
            "correct": correctness,
        }
    
    def get_diagnosis(self) -> str:
        """Get a human-readable system diagnosis."""
        status = self.state_manager.get_status()
        state = status['state']
        gov = status['governor']
        
        lines = [
            f"=== RAVANA v2 System Diagnosis ===",
            f"Episode: {self.episode_count}",
            f"Dissonance: {state['dissonance']:.3f} (EMA: {state['dissonance_ema']:.3f})",
            f"Identity:  {state['identity']:.3f}",
            f"Wisdom:    {state['wisdom']:.3f}",
            f"Mode:      {gov.get('current_mode', gov.get('mode', 'unknown'))}",
            f"Clamp Rate: {self.governor.get_clamp_metrics().get('clamp_rate', 0):.1%}",
            f"Alignment: {self.governor.get_clamp_metrics().get('alignment_score', 0):.1%}",
        ]
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset RAVANA to initial state."""
        self._init_ravana()
        self.episode_count = 0
        self.lesson_log = []
        return {"status": "reset", "episode": 0}
    
    def _mode_to_label(self, mode: str) -> str:
        labels = {
            "normal": "Steady — coherent operation",
            "exploration": "Curious — seeking novel patterns",
            "resolution": "Active — resolving cognitive conflict",
            "recovery": "Healing — crisis response",
            "plateau": "Stuck — detected stagnation",
        }
        return labels.get(mode, mode)
    
    def _generate_reasoning(self, state: dict) -> str:
        d = state['state']['dissonance']
        i = state['state']['identity']
        w = state['state']['wisdom']
        mode = state['governor'].get('current_mode', state['governor'].get('mode', 'unknown'))
        
        if d > 0.7:
            d_status = "high conflict — system is under pressure"
        elif d < 0.3:
            d_status = "low conflict — stable, possibly too comfortable"
        else:
            d_status = "moderate conflict — healthy range"
        
        if i > 0.7:
            i_status = "strong identity"
        elif i < 0.3:
            i_status = "weak or uncertain identity"
        else:
            i_status = "developing identity"
        
        return f"System is in {mode} mode with {d_status} ({d:.2f}) and {i_status} ({i:.2f}). Accumulated wisdom: {w:.2f}."
    
    def _generate_recommendation(self, state: dict) -> str:
        d = state['state']['dissonance']
        mode = state['governor'].get('current_mode', state['governor'].get('mode', 'unknown'))
        
        if d > 0.7:
            return "Prioritize resolution: examine conflicting beliefs and take corrective action."
        elif d < 0.2:
            return "Consider exploration: seek novel information to prevent stagnation."
        elif mode == "recovery":
            return "System requires healing — reduce cognitive load, allow restabilization."
        elif mode == "plateau":
            return "Stagnation detected — introduce controlled perturbation."
        else:
            return "System is healthy — maintain current trajectory."
    
    def get_state_vector(self) -> dict:
        """Get flat state vector for LLM processing."""
        return {
            "dissonance": self.state_manager.state.dissonance,
            "dissonance_ema": self.state_manager.state.dissonance_ema,
            "identity": self.state_manager.state.identity,
            "wisdom": self.state_manager.state.accumulated_wisdom,
            "episode": self.episode_count,
            "cycle": self.state_manager.state.cycle,
            "resolution_streak": self.state_manager.state.resolution_streak,
            "governor_mode": self.state_manager.governor.mode_history[-1].value if self.state_manager.governor.mode_history else "unknown",
        }


if __name__ == "__main__":
    # Quick test
    rav = RavanaWrapper()
    print("=== RAVANA v2 Wrapper Test ===")
    
    # Simulate a few steps
    for i in range(5):
        correct = i % 2 == 0
        result = rav.step(correctness=correct, difficulty=0.5, reason=f"Test step {i+1}")
        print(f"  Step {i+1}: D={result['post_dissonance']:.3f}, I={result['post_identity']:.3f}, {'✓' if correct else '✗'}")
    
    print()
    print(rav.get_diagnosis())
    print()
    
    # Query test
    q = rav.query("What should I do about rising cognitive pressure?")
    print(f"Recommendation: {q['recommendation']}")
