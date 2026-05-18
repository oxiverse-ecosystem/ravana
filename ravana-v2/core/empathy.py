"""
RAVANA v2 — EMPATHY ENGINE
Theory of Mind via Gaussian Process regression.

PRINCIPLE: Understanding others' emotional states requires
inference under uncertainty, not direct observation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable
import math


@dataclass
class EmpathyConfig:
    """Configuration for empathy engine."""
    gp_length_scale: float = 1.0
    gp_noise_level: float = 0.1
    max_agents_tracked: int = 20
    empathy_influence_weight: float = 0.3
    trust_decay_rate: float = 0.01
    min_observations_for_prediction: int = 3


@dataclass
class OtherMind:
    """
    Model of another agent's internal state.
    
    Maintains a simple GP-like belief distribution over
    the other agent's VAD state given observed cues.
    """
    agent_id: int
    observations: List[Dict[str, float]] = field(default_factory=list)
    cue_vectors: List[np.ndarray] = field(default_factory=list)
    
    # Inferred VAD (mean, std)
    inferred_valence: Tuple[float, float] = (0.0, 0.5)
    inferred_arousal: Tuple[float, float] = (0.3, 0.3)
    inferred_dominance: Tuple[float, float] = (0.5, 0.3)
    
    # Trust
    trust_score: float = 0.5
    honesty_estimate: float = 0.5
    
    # Prediction history for calibration
    prediction_errors: List[float] = field(default_factory=list)


class EmpathyEngine:
    """
    Theory of Mind using Gaussian Process-inspired inference.
    
    Infers other agents' VAD states from observed behavioral cues
    using a simplified GP: RBF kernel similarity + noise model.
    """
    
    def __init__(self, config: Optional[EmpathyConfig] = None):
        self.config = config or EmpathyConfig()
        self.other_minds: Dict[int, OtherMind] = {}
        self._global_empathy_bias: float = 0.0
    
    def observe(
        self,
        agent_id: int,
        cues: np.ndarray,
        observed_vad: Optional[Tuple[float, float, float]] = None,
    ) -> OtherMind:
        """
        Observe another agent and update model of their internal state.
        
        Args:
            agent_id: ID of the observed agent
            cues: Behavioral feature vector (text sentiment, speech rate, etc.)
            observed_vad: If provided, (V, A, D) ground truth for learning
        
        Returns:
            Updated OtherMind model
        """
        if agent_id not in self.other_minds:
            self.other_minds[agent_id] = OtherMind(agent_id=agent_id)
        
        mind = self.other_minds[agent_id]
        mind.cue_vectors.append(cues)
        
        if observed_vad is not None:
            mind.observations.append({
                "valence": observed_vad[0],
                "arousal": observed_vad[1],
                "dominance": observed_vad[2],
                "cues": cues.copy(),
            })
            self._update_inference(mind)
        
        # Decay trust for agents without recent observations
        if agent_id in self.other_minds:
            mind.trust_score *= (1 - self.config.trust_decay_rate)
        
        return mind
    
    def _update_inference(self, mind: OtherMind):
        """Update inferred VAD using similarity-weighted observations (GP-lite)."""
        if len(mind.observations) < self.config.min_observations_for_prediction:
            return
        
        # Use RBF similarity between cue vectors to weight observations
        weights = []
        valences, arousals, dominances = [], [], []
        
        for obs in mind.observations:
            valences.append(obs["valence"])
            arousals.append(obs["arousal"])
            dominances.append(obs["dominance"])
        
        # Weighted average with recency bias
        n = len(valences)
        recency_weights = np.array([math.exp(-0.1 * (n - 1 - i)) for i in range(n)])
        recency_weights /= recency_weights.sum()
        
        v_mean = float(np.average(valences, weights=recency_weights))
        a_mean = float(np.average(arousals, weights=recency_weights))
        d_mean = float(np.average(dominances, weights=recency_weights))
        
        v_std = float(np.std(valences)) if len(valences) > 1 else 0.3
        a_std = float(np.std(arousals)) if len(arousals) > 1 else 0.3
        d_std = float(np.std(dominances)) if len(dominances) > 1 else 0.3
        
        mind.inferred_valence = (v_mean, v_mean + v_std)
        mind.inferred_arousal = (a_mean, a_mean + a_std)
        mind.inferred_dominance = (d_mean, d_mean + d_std)
    
    def infer_emotion(self, agent_id: int, cues: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict another agent's VAD state from cues.
        
        Returns:
            (mean_VAD, std_VAD) each as 3-element arrays
        """
        if agent_id not in self.other_minds:
            return np.array([0.0, 0.3, 0.5]), np.array([0.5, 0.3, 0.3])
        
        mind = self.other_minds[agent_id]
        
        if len(mind.observations) < self.config.min_observations_for_prediction:
            return np.array([0.0, 0.3, 0.5]), np.array([0.5, 0.3, 0.3])
        
        # Simplified GP prediction: similarity-weighted mean of training points
        similarities = []
        for obs in mind.observations:
            cue_diff = cues - obs["cues"]
            sim = math.exp(-0.5 * np.dot(cue_diff, cue_diff) / self.config.gp_length_scale)
            similarities.append(sim)
        
        similarities = np.array(similarities)
        sim_sum = similarities.sum()
        if sim_sum < 1e-8:
            return np.array([0.0, 0.3, 0.5]), np.array([0.5, 0.3, 0.3])
        
        weights = similarities / sim_sum
        
        v_pred = float(np.average([o["valence"] for o in mind.observations], weights=weights))
        a_pred = float(np.average([o["arousal"] for o in mind.observations], weights=weights))
        d_pred = float(np.average([o["dominance"] for o in mind.observations], weights=weights))
        
        # Prediction uncertainty: higher when cues are dissimilar to training
        max_sim = float(np.max(similarities))
        uncertainty_scale = 1.0 - max_sim
        v_std = 0.1 + 0.4 * uncertainty_scale
        a_std = 0.1 + 0.3 * uncertainty_scale
        d_std = 0.1 + 0.3 * uncertainty_scale
        
        return np.array([v_pred, a_pred, d_pred]), np.array([v_std, a_std, d_std])
    
    def compute_empathy_distance(self, own_vad: np.ndarray, other_vad: np.ndarray) -> float:
        """
        Compute empathy score: how aligned own emotion is with inferred other emotion.
        
        Returns:
            0.0 (no empathy) to 1.0 (full empathetic alignment)
        """
        distance = float(np.linalg.norm(own_vad - other_vad))
        # Normalize: max possible distance in VAD space ≈ sqrt(2^2 + 1^2 + 1^2) ≈ 2.45
        normalized_distance = min(distance / 2.45, 1.0)
        empathy = 1.0 - normalized_distance
        return empathy * self.config.empathy_influence_weight
    
    def get_trust_score(self, agent_id: int) -> float:
        """Get trust score for an agent."""
        if agent_id not in self.other_minds:
            return 0.5
        return self.other_minds[agent_id].trust_score
    
    def update_trust(self, agent_id: int, honesty_delta: float):
        """Update trust based on honesty/deception evidence."""
        if agent_id in self.other_minds:
            mind = self.other_minds[agent_id]
            mind.honesty_estimate = np.clip(
                mind.honesty_estimate + honesty_delta, 0.0, 1.0
            )
            mind.trust_score = 0.3 * mind.trust_score + 0.7 * mind.honesty_estimate
    
    def get_status(self) -> Dict[str, Any]:
        """Full empathy engine status."""
        return {
            "agents_tracked": len(self.other_minds),
            "global_empathy_bias": self._global_empathy_bias,
            "agents": {
                aid: {
                    "trust": m.trust_score,
                    "inferred_v": m.inferred_valence,
                    "observations": len(m.observations),
                }
                for aid, m in self.other_minds.items()
            }
        }
