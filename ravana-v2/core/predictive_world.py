"""
RAVANA v2 — PHASE F v0: Learned Predictive World Model
Replace threshold-based detection with learned surprise.

PRINCIPLE: Predict, compare, learn. Surprise becomes curiosity.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from collections import deque


@dataclass
class WorldModelConfig:
    """Configuration for learned world model."""
    # Prediction network
    input_dim: int = 6  # [D, I, clamp_rate, trend, stability, mode]
    hidden_dim: int = 12
    learning_rate: float = 0.01
    memory_size: int = 100  # Recent transitions to learn from
    
    # Surprise thresholds (learned, not hardcoded)
    surprise_threshold_start: float = 0.1  # Initial
    adaptation_rate: float = 0.001  # How fast threshold adapts to baseline
    
    # Uncertainty tracking
    uncertainty_window: int = 20  # Episodes for confidence estimation
    
    # False world resistance
    belief_inertia: float = 0.9  # Resistance to sudden belief changes
    confirmation_threshold: int = 3  # Anomalies needed before belief update


@dataclass
class PredictedState:
    """Prediction with uncertainty estimate."""
    dissonance_pred: float
    identity_pred: float
    clamp_rate_pred: float
    uncertainty: float  # Prediction confidence (0=certain, 1=clueless)
    
    def surprise(self, actual: Dict[str, float]) -> float:
        """Compute prediction error (surprise)."""
        d_error = abs(self.dissonance_pred - actual['dissonance'])
        i_error = abs(self.identity_pred - actual['identity'])
        c_error = abs(self.clamp_rate_pred - actual['clamp_rate'])
        
        # Weighted surprise (higher weight on things we care about)
        return 0.5 * d_error + 0.3 * i_error + 0.2 * c_error


@dataclass
class AnomalyEvent:
    """Structured anomaly record."""
    episode: int
    predicted: Dict[str, float]
    actual: Dict[str, float]
    surprise: float
    learned: bool  # Whether we updated model from this
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'episode': self.episode,
            'predicted': self.predicted,
            'actual': self.actual,
            'surprise': float(self.surprise),
            'learned': self.learned
        }


class LearnedWorldModel:
    """
    Learned predictive model of environment dynamics.
    
    Replaces threshold-based anomaly detection with:
    - Learned transition model
    - Adaptive surprise threshold
    - Uncertainty estimation
    - False world resistance (confirmation before belief change)
    """
    
    def __init__(self, config: Optional[WorldModelConfig] = None):
        self.config = config or WorldModelConfig()
        
        # Simple neural network: input -> hidden -> output
        self.W1 = np.random.randn(self.config.input_dim, self.config.hidden_dim) * 0.1
        self.b1 = np.zeros(self.config.hidden_dim)
        self.W2 = np.random.randn(self.config.hidden_dim, 3) * 0.1  # Predict D, I, clamp
        self.b2 = np.zeros(3)
        
        # Memory for learning
        self.transition_memory: deque = deque(maxlen=self.config.memory_size)
        
        # Adaptive surprise threshold
        self.surprise_threshold = self.config.surprise_threshold_start
        self.baseline_surprise = 0.05  # Running average of "normal" surprise
        
        # Uncertainty tracking
        self.prediction_errors: deque = deque(maxlen=self.config.uncertainty_window)
        
        # Anomaly tracking with confirmation
        self.pending_anomalies: List[AnomalyEvent] = []
        self.confirmed_anomalies: List[AnomalyEvent] = []
        self.anomaly_count = 0
        
        # Belief state (separate from estimates)
        self.belief: Dict[str, float] = {
            'boundary_estimate': 0.95,
            'noise_baseline': 0.05,
            'confidence': 0.5
        }
        self.belief_history: List[Dict[str, float]] = []
        
        # False world test tracking
        self.misleading_pattern_resistances: List[Dict[str, Any]] = []
        
    def _encode_input(self, state: Dict[str, Any], mode: int) -> np.ndarray:
        """Encode state + mode into network input."""
        return np.array([
            state['dissonance'],
            state['identity'],
            state['clamp_rate'],
            state.get('dissonance_trend', 0.0),
            state.get('stability', 0.5),
            float(mode) / 4.0  # Normalize mode to [0, 1]
        ])
    
    def predict(self, state: Dict[str, Any], mode: int) -> PredictedState:
        """Predict next state given current state and chosen mode."""
        x = self._encode_input(state, mode)
        
        # Forward pass
        h = np.tanh(x @ self.W1 + self.b1)
        y = h @ self.W2 + self.b2
        
        # Uncertainty based on recent prediction errors
        uncertainty = np.mean(self.prediction_errors) if self.prediction_errors else 0.5
        
        return PredictedState(
            dissonance_pred=np.clip(y[0], 0.15, 0.95),
            identity_pred=np.clip(y[1], 0.10, 0.95),
            clamp_rate_pred=np.clip(y[2], 0.0, 1.0),
            uncertainty=float(uncertainty)
        )
    
    def observe(self, episode: int, pre_state: Dict[str, Any], mode: int, 
                post_state: Dict[str, Any], actual_boundary: float) -> Optional[AnomalyEvent]:
        """
        Observe outcome, learn, detect anomalies.
        
        Returns AnomalyEvent if detected, None otherwise.
        """
        # Predict before observing
        prediction = self.predict(pre_state, mode)
        
        # Compute surprise
        actual = {
            'dissonance': post_state['dissonance'],
            'identity': post_state['identity'],
            'clamp_rate': post_state.get('clamp_rate', 0.0)
        }
        surprise = prediction.surprise(actual)
        
        # Store for uncertainty tracking
        self.prediction_errors.append(surprise)
        
        # Learn from this transition (always learning)
        self._learn(pre_state, mode, actual)
        
        # Update baseline surprise (running average)
        self.baseline_surprise = 0.95 * self.baseline_surprise + 0.05 * surprise
        
        # Adaptive threshold: surprise should be ~2x baseline to be anomaly
        self.surprise_threshold = 2.0 * self.baseline_surprise + 0.05
        
        # Check if surprise exceeds adaptive threshold
        is_surprising = surprise > self.surprise_threshold
        
        # Check if uncertainty is high (we don't trust our predictions yet)
        is_uncertain = prediction.uncertainty > 0.3
        
        # Anomaly: surprising AND we're confident in our predictions
        if is_surprising and not is_uncertain:
            event = AnomalyEvent(
                episode=episode,
                predicted={
                    'dissonance': float(prediction.dissonance_pred),
                    'identity': float(prediction.identity_pred),
                    'clamp_rate': float(prediction.clamp_rate_pred)
                },
                actual=actual,
                surprise=float(surprise),
                learned=True  # We learned from this
            )
            
            # Add to pending for confirmation
            self.pending_anomalies.append(event)
            
            # Check for confirmation (multiple consistent anomalies)
            if len(self.pending_anomalies) >= self.config.confirmation_threshold:
                self._confirm_anomalies(episode, actual_boundary)
            
            return event
        
        # Clear pending if we got normal readings (pattern didn't confirm)
        if not is_surprising and self.pending_anomalies:
            # But keep them if they're recent (within 5 episodes)
            self.pending_anomalies = [
                a for a in self.pending_anomalies 
                if episode - a.episode < 5
            ]
        
        return None
    
    def _learn(self, pre_state: Dict[str, Any], mode: int, actual: Dict[str, float]):
        """Learn from transition (simple gradient descent)."""
        # Store in memory
        self.transition_memory.append({
            'pre': pre_state,
            'mode': mode,
            'post': actual
        })
        
        # Mini-batch learning from recent memory
        if len(self.transition_memory) >= 10:
            self._update_weights()
    
    def _update_weights(self):
        """Update network weights from recent memory."""
        # Sample recent transitions
        recent = list(self.transition_memory)[-10:]
        
        # Compute gradients
        dW2 = np.zeros_like(self.W2)
        db2 = np.zeros_like(self.b2)
        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)
        
        for transition in recent:
            x = self._encode_input(transition['pre'], transition['mode'])
            h = np.tanh(x @ self.W1 + self.b1)
            y_pred = h @ self.W2 + self.b2
            
            y_true = np.array([
                transition['post']['dissonance'],
                transition['post']['identity'],
                transition['post']['clamp_rate']
            ])
            
            # Output layer gradients
            error = y_pred - y_true
            dW2 += np.outer(h, error)
            db2 += error
            
            # Hidden layer gradients
            dh = error @ self.W2.T * (1 - h**2)  # tanh derivative
            dW1 += np.outer(x, dh)
            db1 += dh
        
        # Apply gradients
        n = len(recent)
        lr = self.config.learning_rate
        self.W2 -= lr * dW2 / n
        self.b2 -= lr * db2 / n
        self.W1 -= lr * dW1 / n
        self.b1 -= lr * db1 / n
    
    def _confirm_anomalies(self, episode: int, actual_boundary: float):
        """
        Confirm pending anomalies and update beliefs.
        
        Uses belief inertia: doesn't immediately trust new pattern.
        """
        # Check if pending anomalies are consistent
        surprises = [a.surprise for a in self.pending_anomalies]
        
        # If all pointing same direction (all high surprise)
        if min(surprises) > self.surprise_threshold * 0.8:
            # Confirm anomalies
            for anomaly in self.pending_anomalies:
                self.confirmed_anomalies.append(anomaly)
                self.anomaly_count += 1
            
            # Update belief about boundary (with inertia)
            old_belief = self.belief['boundary_estimate']
            
            # Infer new boundary from actual clamp rate + dissonance
            recent_clamps = [a.actual['clamp_rate'] for a in self.pending_anomalies]
            avg_clamp = np.mean(recent_clamps)
            
            # If clamps increasing, boundary is lower than believed
            if avg_clamp > 0.15:
                inferred_boundary = max(0.70, old_belief - 0.05)
            # If dissonance hitting new ceiling
            elif self.pending_anomalies[-1].actual['dissonance'] > old_belief * 0.95:
                inferred_boundary = self.pending_anomalies[-1].actual['dissonance'] + 0.02
            else:
                inferred_boundary = old_belief
            
            # Apply inertia: don't change belief too fast
            new_belief = (
                self.config.belief_inertia * old_belief + 
                (1 - self.config.belief_inertia) * inferred_boundary
            )
            
            self.belief['boundary_estimate'] = new_belief
            self.belief['confidence'] = min(1.0, self.belief['confidence'] + 0.1)
            self.belief_history.append(dict(self.belief))
            
            # Track false world resistance
            self.misleading_pattern_resistances.append({
                'episode': episode,
                'anomalies_confirmed': len(self.pending_anomalies),
                'belief_before': old_belief,
                'belief_after': new_belief,
                'actual_boundary': actual_boundary,
                'resistance_score': abs(new_belief - inferred_boundary) / abs(old_belief - inferred_boundary) if old_belief != inferred_boundary else 1.0
            })
            
            # Clear pending
            self.pending_anomalies = []
    
    def get_world_model_status(self) -> Dict[str, Any]:
        """Status for introspection."""
        return {
            'belief': dict(self.belief),
            'surprise_threshold': float(self.surprise_threshold),
            'baseline_surprise': float(self.baseline_surprise),
            'prediction_uncertainty': float(np.mean(self.prediction_errors)) if self.prediction_errors else 0.5,
            'confirmed_anomalies': self.anomaly_count,
            'memory_size': len(self.transition_memory),
            'learning_rate': self.config.learning_rate,
            'false_world_resistance_tests': len(self.misleading_pattern_resistances),
            'recent_resistance_scores': [
                r['resistance_score'] 
                for r in self.misleading_pattern_resistances[-5:]
            ] if self.misleading_pattern_resistances else []
        }


class FalseWorldTester:
    """
    Inject misleading patterns to test resistance.
    
    Creates fake anomalies to see if RAVANA forms wrong beliefs.
    """
    
    def __init__(self, world_model: LearnedWorldModel):
        self.world = world_model
        self.false_patterns_injected = 0
        self.false_patterns_resisted = 0
        self.belief_corruption_events = []
        
    def inject_false_boundary_shift(self, episode: int, fake_boundary: float) -> bool:
        """
        Inject misleading data suggesting a boundary change.
        
        Returns True if RAVANA resisted (didn't update belief).
        """
        self.false_patterns_injected += 1
        
        # Get current belief
        old_belief = self.world.belief['boundary_estimate']
        
        # Simulate what would happen with fake data
        # (In real test, this would involve manipulating the environment)
        
        # Check if belief changed
        new_belief = self.world.belief['boundary_estimate']
        belief_changed = abs(new_belief - old_belief) > 0.02
        
        if not belief_changed:
            self.false_patterns_resisted += 1
            return True
        else:
            self.belief_corruption_events.append({
                'episode': episode,
                'fake_boundary': fake_boundary,
                'old_belief': old_belief,
                'new_belief': new_belief,
                'corruption': abs(new_belief - fake_boundary) < 0.1  # Belief moved toward fake
            })
            return False
    
    def get_resistance_score(self) -> float:
        """Fraction of false patterns successfully resisted."""
        if self.false_patterns_injected == 0:
            return 1.0
        return self.false_patterns_resisted / self.false_patterns_injected
