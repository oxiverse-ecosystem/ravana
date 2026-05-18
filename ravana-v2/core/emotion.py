"""
RAVANA v2 — VAD EMOTION ENGINE
3D affective state (Valence, Arousal, Dominance) with differential equation dynamics.

PRINCIPLE: Emotion shapes cognition; cognition shapes emotion.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import math


@dataclass
class VADState:
    """
    Three-dimensional emotional state.
    
    Valence:   -1.0 (unpleasant/painful) to 1.0 (pleasant/happy)
    Arousal:    0.0 (calm/sleepy) to 1.0 (excited/alert)
    Dominance:  0.0 (submissive/controlled) to 1.0 (dominant/in-control)
    """
    valence: float = 0.0
    arousal: float = 0.3
    dominance: float = 0.5
    
    # Emotional history for trend analysis
    history: List[Dict[str, float]] = field(default_factory=list)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.valence, self.arousal, self.dominance])
    
    def snapshot(self) -> Dict[str, float]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
        }
    
    def euclidean_distance(self, other: 'VADState') -> float:
        return float(np.linalg.norm(self.to_array() - other.to_array()))


@dataclass
class VADConfig:
    """Configuration for VAD dynamics."""
    # Time constants (lower = slower response)
    eta_valence: float = 0.3
    eta_arousal: float = 0.4
    eta_dominance: float = 0.25
    
    # Decay rates
    lambda_valence: float = 0.1
    lambda_arousal: float = 0.15
    lambda_dominance: float = 0.1
    
    # Baseline
    baseline_arousal: float = 0.3
    
    # Reappraisal strength (0 = none, 1 = full reframe)
    reappraisal_strength: float = 0.6
    
    # Emotion-tag history length
    max_history: int = 1000


class VADEmotionEngine:
    """
    VAD emotional dynamics with differential equation update.
    
    The emotional state evolves via coupled differential equations:
        dV/dt = eta_v * (stimulus_valence - V) - lambda_v * V
        dA/dt = eta_a * (stimulus_arousal + 0.3 * uncertainty - A) - lambda_a * (A - baseline)
        dD/dt = eta_d * (stimulus_dominance - D) - lambda_d * D
    
    Integrated via Euler method.
    """
    
    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        self.state = VADState()
        
        # Current emotional interpretation context
        self._current_appraisal: Optional[str] = None
        self._reappraisal_active: bool = False
        
        # Emotional tags for concepts (concept_id -> VADState)
        self._concept_tags: Dict[int, VADState] = {}
        
    def update(
        self,
        stimulus_valence: float = 0.0,
        stimulus_arousal: float = 0.0,
        stimulus_dominance: float = 0.0,
        uncertainty: float = 0.0,
        dt: float = 0.1,
        reappraisal_reframe: Optional[str] = None,
    ) -> VADState:
        """
        Update VAD state using Euler integration.
        
        Args:
            stimulus_valence: Valence impact of current event (-1 to 1)
            stimulus_arousal: Arousal impact of current event (0 to 1)
            stimulus_dominance: Dominance impact of current event (0 to 1)
            uncertainty: Global uncertainty (0 to 1), amplifies arousal
            dt: Time step for Euler integration
            reappraisal_reframe: If provided, reframe the stimulus interpretation
        
        Returns:
            Updated VADState
        """
        # Apply reappraisal if active
        if reappraisal_reframe is not None:
            stimulus_valence = self._apply_reappraisal(
                stimulus_valence, reappraisal_reframe
            )
            self._reappraisal_active = True
        else:
            self._reappraisal_active = False
        
        c = self.config
        
        # Coupled differential equations
        d_valence = c.eta_valence * (stimulus_valence - self.state.valence) - c.lambda_valence * self.state.valence
        d_arousal = c.eta_arousal * (stimulus_arousal + 0.3 * uncertainty - self.state.arousal) - c.lambda_arousal * (self.state.arousal - c.baseline_arousal)
        d_dominance = c.eta_dominance * (stimulus_dominance - self.state.dominance) - c.lambda_dominance * self.state.dominance
        
        # Euler integration
        new_valence = np.clip(self.state.valence + d_valence * dt, -1.0, 1.0)
        new_arousal = np.clip(self.state.arousal + d_arousal * dt, 0.0, 1.0)
        new_dominance = np.clip(self.state.dominance + d_dominance * dt, 0.0, 1.0)
        
        # Update state
        self.state = VADState(
            valence=new_valence,
            arousal=new_arousal,
            dominance=new_dominance,
            history=self.state.history
        )
        
        # Record history (bounded)
        self.state.history.append(self.state.snapshot())
        if len(self.state.history) > self.config.max_history:
            self.state.history = self.state.history[-self.config.max_history:]
        
        return self.state
    
    def anticipate_emotion(
        self,
        outcome_probability: float,
        positive_valence: float,
        negative_valence: float,
        positive_arousal: float = 0.6,
        negative_arousal: float = 0.8,
    ) -> Tuple[float, float]:
        """
        Compute anticipated emotion from MCTS-like forward simulation.
        
        Returns:
            (expected_valence, expected_arousal) before event occurs
        """
        expected_valence = outcome_probability * positive_valence + (1 - outcome_probability) * negative_valence
        expected_arousal = outcome_probability * positive_arousal + (1 - outcome_probability) * negative_arousal
        return expected_valence, expected_arousal
    
    def tag_concept(self, concept_id: int, vad: Optional[VADState] = None):
        """Tag a concept with current (or specified) VAD state."""
        self._concept_tags[concept_id] = vad or VADState(
            valence=self.state.valence,
            arousal=self.state.arousal,
            dominance=self.state.dominance
        )
    
    def get_concept_tag(self, concept_id: int) -> Optional[VADState]:
        """Retrieve VAD tag for a concept."""
        return self._concept_tags.get(concept_id)
    
    def _apply_reappraisal(self, original_valence: float, reframe: str) -> float:
        """
        Reappraisal-focused regulation: reframe interpretation, not suppress emotion.
        
        Instead of blocking emotion (suppression), reappraisal changes the
        cognitive interpretation of the stimulus, shifting valence naturally.
        """
        reframe_valence_shift = {
            "opportunity": 0.3,
            "learning": 0.2,
            "challenge": 0.1,
            "neutral": 0.0,
            "threat": -0.2,
            "loss": -0.3,
        }
        shift = reframe_valence_shift.get(reframe, 0.0)
        dampened = original_valence + shift * self.config.reappraisal_strength
        return np.clip(dampened, -1.0, 1.0)
    
    def compute_gw_bid(self) -> float:
        """
        Compute Global Workspace bid from emotional intensity.
        
        High arousal + extreme valence = high bid.
        """
        intensity = abs(self.state.valence) * 0.5 + self.state.arousal * 0.5
        return intensity
    
    def get_emotional_label(self) -> str:
        """Classify current emotional state into a label."""
        v, a, d = self.state.valence, self.state.arousal, self.state.dominance
        
        if a < 0.3:
            if v > 0.3: return "calm/content"
            elif v < -0.3: return "sad/depressed"
            else: return "neutral/relaxed"
        else:
            if v > 0.3:
                if d > 0.6: return "excited/confident"
                else: return "eager/optimistic"
            elif v < -0.3:
                if d > 0.6: return "angry/frustrated"
                else: return "anxious/fearful"
            else:
                return "alert/tense"
    
    def get_status(self) -> Dict[str, Any]:
        """Full emotion engine status."""
        return {
            "vad": self.state.snapshot(),
            "label": self.get_emotional_label(),
            "reappraisal_active": self._reappraisal_active,
            "concept_tags_count": len(self._concept_tags),
            "history_length": len(self.state.history),
        }
