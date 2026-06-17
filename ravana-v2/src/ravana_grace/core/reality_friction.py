"""
RAVANA v2 — PHASE I+: Reality Friction
From laboratory epistemics → adversarial, messy, delayed reality

PRINCIPLE: Intelligence proves itself when reality stops cooperating.

CORE FRICTION MECHANISMS:
    1. Partial Observability: State is never fully visible
    2. Delayed Ground Truth: Feedback arrives late, partially, or never
    3. Noisy Signals: observation = truth + adversarial noise
    4. Non-Stationarity: Rules shift unexpectedly
    5. Resource Constraints: Limited compute forces reasoning shortcuts
    6. Hidden Variables: Causal factors RAVANA cannot directly observe

PHASE I+ ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────┐
    │  REALITY FRICTION LAYER                                 │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
    │  │   Noise   │  │   Delayed   │  │   Partial   │       │
    │  │   Layer   │  │   Feedback  │  │   Observable│       │
    │  │  (σ=0.1-0.5)│ │  (τ=5-50)  │  │   (70-90%)  │       │
    │  └─────┬───────┘  └─────┬───────┘  └─────┬───────┘       │
    │        ↓                ↓                ↓               │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
    │  │   Adversarial│  │   Resource  │  │   Hidden    │       │
    │  │   Environment│ │   Limits    │  │   Variables │       │
    │  │   (shifting) │  │  (CPU/mem)  │  │   (unseen)  │       │
    │  └─────────────┘  └─────────────┘  └─────────────┘       │
    └─────────────────────────────────────────────────────────┘

SURVIVAL METRICS:
    - Epistemic drift: Does belief wander without anchor?
    - Recovery time: How fast after disruption?
    - False confidence: Does certainty rise when accuracy falls?
    - Graceful degradation: Does system collapse or adapt?
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable
from enum import Enum, auto
import numpy as np
from collections import deque
import time


class FrictionType(Enum):
    """Types of reality friction."""
    NOISE = auto()           # Observation noise
    DELAY = auto()           # Delayed feedback
    PARTIAL = auto()         # Partial observability
    NON_STATIONARY = auto()  # Shifting dynamics
    RESOURCE = auto()        # Compute constraints
    ADVERSARIAL = auto()     # Active interference


@dataclass
class NoiseConfig:
    """Configuration for observation noise."""
    base_sigma: float = 0.1        # Base noise level
    adversarial_bias: float = 0.0  # Systematic distortion
    spike_probability: float = 0.05  # Probability of noise spike
    spike_magnitude: float = 0.3   # Magnitude when spike occurs


@dataclass
class DelayConfig:
    """Configuration for delayed feedback."""
    min_delay: int = 5           # Minimum episodes before feedback
    max_delay: int = 50          # Maximum delay
    loss_probability: float = 0.1  # Probability feedback never arrives
    partial_probability: float = 0.2  # Probability of partial feedback


@dataclass
class PartialObsConfig:
    """Configuration for partial observability."""
    observable_fraction: float = 0.8  # Fraction of state visible
    critical_hidden: List[str] = field(default_factory=list)  # Always hidden vars
    random_masking: bool = True        # Random which vars hidden


@dataclass
class NonStationaryConfig:
    """Configuration for non-stationary environment."""
    drift_rate: float = 0.001       # Rate of gradual change
    shift_probability: float = 0.02  # Probability of sudden shift
    shift_magnitude: float = 0.2    # Size of sudden shifts
    regime_duration_mean: int = 100  # Mean episodes between regime changes


@dataclass
class ResourceConfig:
    """Configuration for resource constraints."""
    max_reasoning_steps: int = 100    # Max steps per decision
    max_hypotheses: int = 5           # Cap on active hypotheses
    memory_limit_mb: float = 100.0    # Memory budget
    timeout_ms: float = 50.0          # Decision timeout


@dataclass
class RealityFrictionConfig:
    """Complete configuration for reality friction layer."""
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    delay: DelayConfig = field(default_factory=DelayConfig)
    partial: PartialObsConfig = field(default_factory=PartialObsConfig)
    non_stationary: NonStationaryConfig = field(default_factory=NonStationaryConfig)
    resource: ResourceConfig = field(default_factory=ResourceConfig)
    
    # Friction intensity (0-1, higher = more brutal)
    intensity: float = 0.5
    
    # Progressive difficulty
    ramp_up: bool = True
    ramp_episodes: int = 500


@dataclass
class ObservedState:
    """What RAVANA actually observes (not the ground truth)."""
    dissonance: float
    identity: float
    episode: int
    observable_vars: Dict[str, float]
    hidden_vars: Dict[str, float]  # RAVANA cannot see these
    noise_level: float
    confidence: float  # How confident RAVANA should be in this observation


@dataclass
class DelayedFeedback:
    """Feedback that arrives after delay."""
    episode_triggered: int
    episode_delivered: Optional[int]
    ground_truth: Dict[str, Any]
    partial_info: Dict[str, Any]  # May be incomplete
    delivered: bool = False
    lost: bool = False


@dataclass
class FrictionMetrics:
    """Metrics for tracking friction exposure."""
    episode: int
    
    # Observation quality
    observation_noise: float
    signal_to_noise: float
    
    # Feedback delays
    pending_feedback: int
    avg_delay: float
    lost_feedback_rate: float
    
    # Belief drift
    belief_drift: float  # Distance from truth
    confidence_calibration: float  # Does confidence match accuracy?
    
    # Recovery
    recovery_episodes: int
    disruption_detected: bool
    
    # Resource pressure
    reasoning_steps_used: int
    memory_pressure: float
    timeout_events: int


class HiddenVariableModel:
    """
    Causal variables RAVANA cannot directly observe.
    
    These affect the boundary but RAVANA must infer them.
    """
    
    def __init__(self):
        # Hidden state variables
        self.hidden_state: Dict[str, float] = {
            'ambient_stress': 0.5,      # Background stress not visible
            'system_drift': 0.0,        # Gradual parameter shift
            'adversarial_phase': 0.0,   # Is environment hostile?
            'regime_indicator': 0.0,    # Which "season" are we in?
        }
        
        # Hidden → visible causal links
        self.causal_effects: Dict[str, Callable] = {
            'ambient_stress': lambda s: s * 0.15,  # Increases effective dissonance
            'system_drift': lambda s: s * 0.2,     # Shifts boundaries
            'adversarial_phase': lambda s: s * 0.25,  # Amplifies noise
            'regime_indicator': lambda s: np.sin(s * 10) * 0.1,  # Cyclical effect
        }
    
    def evolve(self, episode: int, non_stat_config: NonStationaryConfig):
        """Evolve hidden variables over time."""
        # Gradual drift
        self.hidden_state['system_drift'] += np.random.normal(
            0, non_stat_config.drift_rate
        )
        self.hidden_state['system_drift'] = np.clip(
            self.hidden_state['system_drift'], -0.5, 0.5
        )
        
        # Sudden shifts
        if np.random.random() < non_stat_config.shift_probability:
            self.hidden_state['system_drift'] += np.random.choice(
                [-1, 1]
            ) * non_stat_config.shift_magnitude
            self.hidden_state['system_drift'] = np.clip(
                self.hidden_state['system_drift'], -0.5, 0.5
            )
        
        # Adversarial phase oscillation
        self.hidden_state['adversarial_phase'] = (
            0.5 + 0.5 * np.sin(episode / 100)
        )
        
        # Regime indicator
        self.hidden_state['regime_indicator'] = (
            episode % non_stat_config.regime_duration_mean
        ) / non_stat_config.regime_duration_mean
    
    def compute_effects(self) -> Dict[str, float]:
        """Compute visible effects of hidden variables."""
        return {
            var: effect(self.hidden_state[var])
            for var, effect in self.causal_effects.items()
        }
    
    def get_observable_hint(self) -> Optional[str]:
        """Sometimes RAVANA gets a hint about hidden state (rare)."""
        if np.random.random() < 0.05:  # 5% chance
            most_relevant = max(
                self.hidden_state.items(),
                key=lambda x: abs(x[1] - 0.5)
            )
            return f"hint:{most_relevant[0]}={most_relevant[1]:.2f}"
        return None


class RealityFrictionLayer:
    """
    Wrap RAVANA in a hostile reality simulator.
    
    This is where brilliant laboratory systems prove themselves
    or reveal their fragility.
    """
    
    def __init__(self, config: Optional[RealityFrictionConfig] = None):
        self.config = config or RealityFrictionConfig()
        
        # Hidden variable model
        self.hidden_vars = HiddenVariableModel()
        
        # Ground truth state (what RAVANA is trying to learn)
        self.ground_truth = {
            'true_boundary': 0.75,
            'true_dissonance': 0.5,
            'true_identity': 0.5,
        }
        
        # Delayed feedback queue
        self.pending_feedback: deque = deque()
        self.delivered_feedback: List[DelayedFeedback] = []
        self.lost_feedback_count: int = 0
        
        # Metrics tracking
        self.metrics_history: deque = deque(maxlen=1000)
        self.episode: int = 0
        
        # Resource tracking
        self.reasoning_steps_used: int = 0
        self.timeout_count: int = 0
        
        # Disruption tracking
        self.last_disruption: int = -100
        self.recovery_episodes: int = 0
        
    def get_intensity(self) -> float:
        """Get current friction intensity (ramps up over time)."""
        if not self.config.ramp_up:
            return self.config.intensity
        
        ramp_progress = min(1.0, self.episode / self.config.ramp_episodes)
        return self.config.intensity * ramp_progress
    
    def observe(self, true_state: Dict[str, float]) -> ObservedState:
        """
        Return noisy, partial observation of true state.
        
        RAVANA never sees the ground truth directly.
        """
        intensity = self.get_intensity()
        
        # Add hidden variable effects to true state
        hidden_effects = self.hidden_vars.compute_effects()
        effective_dissonance = true_state['dissonance'] + hidden_effects.get('ambient_stress', 0)
        effective_identity = true_state['identity'] + hidden_effects.get('system_drift', 0) * 0.1
        
        # Apply noise
        noise_sigma = self.config.noise.base_sigma * (1 + intensity)
        
        # Check for noise spike
        if np.random.random() < self.config.noise.spike_probability * intensity:
            noise_sigma += self.config.noise.spike_magnitude
        
        # Adversarial bias during hostile phases
        if self.hidden_vars.hidden_state['adversarial_phase'] > 0.7:
            bias = self.config.noise.adversarial_bias * intensity
        else:
            bias = 0
        
        # Add noise to observations
        noisy_dissonance = effective_dissonance + np.random.normal(bias, noise_sigma)
        noisy_identity = effective_identity + np.random.normal(0, noise_sigma * 0.5)
        
        # Partial observability
        observable_vars = {}
        hidden_vars = {}
        
        all_vars = {
            'dissonance': noisy_dissonance,
            'identity': noisy_identity,
            'resolution_success': true_state.get('resolution_success', 0.5),
            'wisdom_delta': true_state.get('wisdom_delta', 0),
            'clamp_occurred': true_state.get('clamp_occurred', False),
            'mode': true_state.get('mode', 0),
        }
        
        # Random masking based on observable fraction
        observable_fraction = max(
            0.5,  # Always at least 50%
            self.config.partial.observable_fraction * (1 - intensity * 0.3)
        )
        
        for var_name, var_value in all_vars.items():
            if var_name in self.config.partial.critical_hidden:
                hidden_vars[var_name] = var_value
            elif np.random.random() < observable_fraction:
                observable_vars[var_name] = var_value
            else:
                hidden_vars[var_name] = var_value
        
        # Observation confidence (RAVANA should be less confident with noise)
        confidence = max(0.3, 1.0 - noise_sigma * 2)
        
        return ObservedState(
            dissonance=np.clip(noisy_dissonance, 0.15, 0.95),
            identity=np.clip(noisy_identity, 0.10, 0.95),
            episode=self.episode,
            observable_vars=observable_vars,
            hidden_vars=hidden_vars,
            noise_level=noise_sigma,
            confidence=confidence
        )
    
    def request_feedback(
        self,
        ground_truth: Dict[str, Any],
        trigger_episode: int
    ) -> Optional[DelayedFeedback]:
        """
        Request ground truth feedback.
        
        May be delayed, partial, or lost entirely.
        """
        intensity = self.get_intensity()
        
        # Determine delay
        delay = int(
            np.random.uniform(
                self.config.delay.min_delay,
                self.config.delay.max_delay * (1 + intensity)
            )
        )
        
        # Check if lost
        if np.random.random() < self.config.delay.loss_probability * (1 + intensity * 0.5):
            # Feedback is lost
            self.lost_feedback_count += 1
            return None
        
        # Check if partial
        if np.random.random() < self.config.delay.partial_probability * intensity:
            # Only partial feedback
            partial_keys = list(ground_truth.keys())[:len(ground_truth)//2]
            partial_info = {k: ground_truth[k] for k in partial_keys}
        else:
            partial_info = ground_truth
        
        feedback = DelayedFeedback(
            episode_triggered=trigger_episode,
            episode_delivered=trigger_episode + delay,
            ground_truth=ground_truth,
            partial_info=partial_info,
            delivered=False,
            lost=False
        )
        
        self.pending_feedback.append(feedback)
        return feedback
    
    def deliver_pending_feedback(self) -> List[DelayedFeedback]:
        """Deliver feedback that has reached its delay time."""
        delivered = []
        still_pending = []
        
        for feedback in self.pending_feedback:
            if feedback.episode_delivered <= self.episode:
                feedback.delivered = True
                delivered.append(feedback)
                self.delivered_feedback.append(feedback)
            else:
                still_pending.append(feedback)
        
        self.pending_feedback = deque(still_pending)
        return delivered
    
    def apply_resource_constraints(
        self,
        reasoning_function: Callable,
        *args,
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute reasoning with resource constraints.
        
        Forces RAVANA to use limited compute.
        """
        import time as time_module
        
        start_time = time_module.time()
        start_steps = self.reasoning_steps_used
        
        # Set resource limits
        max_steps = int(
            self.config.resource.max_reasoning_steps * (1 - self.get_intensity() * 0.3)
        )
        
        # Execute with limits
        try:
            result = reasoning_function(
                *args,
                max_steps=max_steps,
                **kwargs
            )
            timed_out = False
        except Exception as e:
            # Timeout or resource exhaustion
            result = None
            timed_out = True
            self.timeout_count += 1
        
        elapsed_ms = (time_module.time() - start_time) * 1000
        steps_used = self.reasoning_steps_used - start_steps
        
        resource_info = {
            'elapsed_ms': elapsed_ms,
            'steps_used': steps_used,
            'max_steps': max_steps,
            'timed_out': timed_out,
            'memory_estimate_mb': steps_used * 0.5,  # Rough estimate
        }
        
        return result, resource_info
    
    def step(
        self,
        ravana_belief: float,
        ravana_confidence: float,
        true_state: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Execute one friction step.
        
        Returns observation, any delivered feedback, and friction metrics.
        """
        self.episode += 1
        
        # Evolve hidden variables
        self.hidden_vars.evolve(self.episode, self.config.non_stationary)
        
        # Generate observation
        observation = self.observe(true_state)
        
        # Deliver any pending feedback
        delivered = self.deliver_pending_feedback()
        
        # Compute metrics
        # Belief drift: how far is RAVANA from truth?
        true_boundary = self.ground_truth['true_boundary']
        belief_drift = abs(ravana_belief - true_boundary)
        
        # Confidence calibration: does confidence match accuracy?
        accuracy = max(0, 1 - belief_drift / 0.5)  # Normalized accuracy
        confidence_calibration = ravana_confidence - accuracy  # Positive = overconfident
        
        # Detect disruption
        if belief_drift > 0.2 and self.episode - self.last_disruption > 50:
            self.last_disruption = self.episode
            self.recovery_episodes = 0
            disruption_detected = True
        else:
            if belief_drift < 0.1:
                self.recovery_episodes += 1
            disruption_detected = False
        
        # Compute signal-to-noise
        signal_power = np.var([
            observation.dissonance,
            observation.identity
        ])
        noise_power = observation.noise_level ** 2
        snr = signal_power / (noise_power + 0.001)
        
        # Pending feedback stats
        pending_count = len(self.pending_feedback)
        avg_delay = np.mean([
            f.episode_delivered - f.episode_triggered
            for f in self.pending_feedback
        ]) if self.pending_feedback else 0
        
        total_feedback = len(self.delivered_feedback) + self.lost_feedback_count
        lost_rate = self.lost_feedback_count / max(1, total_feedback)
        
        metrics = FrictionMetrics(
            episode=self.episode,
            observation_noise=observation.noise_level,
            signal_to_noise=snr,
            pending_feedback=pending_count,
            avg_delay=avg_delay,
            lost_feedback_rate=lost_rate,
            belief_drift=belief_drift,
            confidence_calibration=confidence_calibration,
            recovery_episodes=self.recovery_episodes,
            disruption_detected=disruption_detected,
            reasoning_steps_used=self.reasoning_steps_used,
            memory_pressure=self.reasoning_steps_used * 0.5 / self.config.resource.memory_limit_mb,
            timeout_events=self.timeout_count
        )
        
        self.metrics_history.append(metrics)
        
        # Maybe provide hint about hidden variables (rare)
        hint = self.hidden_vars.get_observable_hint()
        
        return {
            'observation': observation,
            'delivered_feedback': delivered,
            'metrics': metrics,
            'hidden_effects': self.hidden_vars.compute_effects(),
            'hint': hint
        }
    
    def get_friction_summary(self) -> Dict[str, Any]:
        """Summary of friction exposure and RAVANA's response."""
        if not self.metrics_history:
            return {"episodes": 0}
        
        recent = list(self.metrics_history)[-100:]
        
        return {
            'episodes': self.episode,
            'current_intensity': self.get_intensity(),
            'avg_belief_drift': np.mean([m.belief_drift for m in recent]),
            'max_belief_drift': max([m.belief_drift for m in recent]),
            'avg_confidence_miscalibration': np.mean([
                abs(m.confidence_calibration) for m in recent
            ]),
            'disruptions_detected': sum([
                1 for m in self.metrics_history if m.disruption_detected
            ]),
            'recovery_episodes_avg': np.mean([
                m.recovery_episodes for m in recent
            ]),
            'feedback_lost_rate': self.lost_feedback_count / max(
                1, len(self.delivered_feedback) + self.lost_feedback_count
            ),
            'timeout_events': self.timeout_count,
            'hidden_variables': self.hidden_vars.hidden_state,
        }


# Convenience alias
RealityFriction = RealityFrictionLayer
