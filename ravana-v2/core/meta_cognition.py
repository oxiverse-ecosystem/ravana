"""
RAVANA v2 — PHASE I: Meta-Cognition Layer
Self-awareness of the epistemic process.

This layer prevents RAVANA from becoming:
- A confident fool (high confidence, wrong beliefs)
- A systematic misinterpreter (repeated probe failures)
- A frozen thinker (no mode switching when stuck)

KEY INSIGHT: Most systems optimize beliefs. 
RAVANA Phase I evaluates the *process* of forming beliefs.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum


class EpistemicMode(Enum):
    """Meta-cognitive modes for how to approach uncertainty."""
    CAUTIOUS = "cautious"        # High uncertainty, trust nothing
    EXPLORATORY = "exploratory"   # Moderate uncertainty, systematic probes
    RECOVERY = "recovery"         # Detected systematic failure, reset priors
    CONFIDENT = "confident"       # Well-calibrated, proceed normally


@dataclass
class ProbeResult:
    """Result of a designed probe."""
    probe_id: int
    design: Dict[str, Any]
    outcome: Dict[str, Any]
    conclusive: bool
    information_gain: float
    episode: int


@dataclass
class ReasoningQualityMetrics:
    """Metrics for reasoning quality over time."""
    prediction_accuracy: float = 0.0
    confidence_calibration: float = 0.0  # 1.0 = perfect
    hypothesis_discrimination: float = 0.0  # Can probes distinguish H?
    probe_effectiveness: float = 0.0
    belief_stability: float = 0.0
    

@dataclass
class MetaCognitiveConfig:
    """Configuration for meta-cognitive monitoring."""
    # Probe failure detection
    probe_failure_threshold: float = 0.5  # >50% inconclusive = alert
    probe_failure_window: int = 20
    
    # Confidence calibration
    confidence_calibration_window: int = 30
    calibration_drift_threshold: float = 0.2
    
    # Reasoning quality
    reasoning_quality_threshold: float = 0.6
    quality_degradation_limit: int = 50  # episodes of poor quality
    
    # Mode switching
    recovery_mode_min_episodes: int = 10
    cautious_mode_uncertainty_threshold: float = 0.3
    confident_mode_calibration_threshold: float = 0.15


class ReasoningQualityTracker:
    """Tracks quality of reasoning over time."""
    
    def __init__(self, config: MetaCognitiveConfig):
        self.config = config
        self.metrics_history: List[ReasoningQualityMetrics] = []
        self.probe_results: List[ProbeResult] = []
        self.prediction_errors: List[float] = []
        
    def record_prediction(self, predicted: Dict[str, Any], 
                          actual: Dict[str, Any], episode: int):
        """Record prediction accuracy."""
        # Calculate prediction error
        pred_boundary = predicted.get('boundary_estimate', 0.5)
        actual_boundary = actual.get('observed_boundary', 0.5)
        error = abs(pred_boundary - actual_boundary)
        
        self.prediction_errors.append(error)
        
        # Keep only recent history
        if len(self.prediction_errors) > self.config.confidence_calibration_window:
            self.prediction_errors.pop(0)
    
    def record_probe_result(self, result: ProbeResult):
        """Record result of a designed probe."""
        self.probe_results.append(result)
        
        # Keep only recent results
        if len(self.probe_results) > self.config.probe_failure_window:
            self.probe_results.pop(0)
    
    def compute_current_metrics(self) -> ReasoningQualityMetrics:
        """Compute current reasoning quality metrics."""
        metrics = ReasoningQualityMetrics()
        
        # Prediction accuracy
        if self.prediction_errors:
            metrics.prediction_accuracy = 1.0 - np.mean(self.prediction_errors)
        
        # Probe effectiveness
        if self.probe_results:
            conclusive_count = sum(1 for p in self.probe_results if p.conclusive)
            metrics.probe_effectiveness = conclusive_count / len(self.probe_results)
        
        # Confidence calibration
        if self.probe_results:
            # Check if reported confidence matches actual informativeness
            reported_confidences = [p.outcome.get('confidence', 0.5) for p in self.probe_results]
            actual_informativeness = [p.information_gain for p in self.probe_results]
            
            if reported_confidences and actual_informativeness:
                # Correlation between confidence and informativeness
                try:
                    from scipy.stats import pearsonr
                    corr, _ = pearsonr(reported_confidences, actual_informativeness)
                    metrics.confidence_calibration = max(0, corr)
                except:
                    metrics.confidence_calibration = 0.5
        
        return metrics


class ConfidenceCalibrator:
    """Tracks and corrects confidence calibration."""
    
    def __init__(self, config: MetaCognitiveConfig):
        self.config = config
        self.confidence_predictions: List[Tuple[float, str]] = []  # (confidence, outcome)
        self.calibration_curve: List[float] = []
        self.bias_estimate: float = 0.0  # Positive = overconfident
        
    def record_outcome(self, predicted_confidence: float, 
                       actual_outcome: str, episode: int):
        """
        Record: I said X confidence, but actual outcome was Y.
        
        actual_outcome: 'conclusive', 'inconclusive', 'ambiguous'
        """
        self.confidence_predictions.append((predicted_confidence, actual_outcome))
        
        # Keep only recent
        if len(self.confidence_predictions) > self.config.confidence_calibration_window:
            self.confidence_predictions.pop(0)
        
        # Update calibration curve
        self._update_calibration()
    
    def _update_calibration(self):
        """Update calibration statistics."""
        if len(self.confidence_predictions) < 10:
            return
        
        # Analyze: when I said high confidence, was I right?
        high_confidence_outcomes = [
            outcome for conf, outcome in self.confidence_predictions 
            if conf > 0.7
        ]
        
        if high_confidence_outcomes:
            success_rate = high_confidence_outcomes.count('conclusive') / len(high_confidence_outcomes)
            # If I say 70%+ confidence but only 40% success, I'm overconfident
            self.bias_estimate = (0.7 - success_rate)
    
    def get_calibration_status(self) -> Dict[str, Any]:
        """Return current calibration status."""
        return {
            'bias_estimate': self.bias_estimate,
            'is_well_calibrated': abs(self.bias_estimate) < self.config.calibration_drift_threshold,
            'confidence_predictions_count': len(self.confidence_predictions),
            'correction_factor': 1.0 - self.bias_estimate  # Apply to future confidences
        }
    
    def adjust_confidence(self, raw_confidence: float) -> float:
        """Apply calibration correction to raw confidence."""
        correction = 1.0 - self.bias_estimate
        return np.clip(raw_confidence * correction, 0.1, 0.95)


class BiasDetector:
    """Detects systematic biases in reasoning."""
    
    def __init__(self, config: MetaCognitiveConfig):
        self.config = config
        self.hypothesis_frequencies: Dict[str, int] = {}
        self.probe_selection_history: List[str] = []
        self.confirmation_pattern_count: int = 0
        
    def track_hypothesis_preference(self, selected_hypothesis: str, 
                                    available_hypotheses: List[str]):
        """Track if system is biased toward certain hypotheses."""
        self.hypothesis_frequencies[selected_hypothesis] = \
            self.hypothesis_frequencies.get(selected_hypothesis, 0) + 1
    
    def track_probe_selection(self, probe_type: str, 
                             hypothesis_being_tested: str):
        """Track probe selection patterns."""
        self.probe_selection_history.append((probe_type, hypothesis_being_tested))
        
        # Detect confirmation bias: always testing favorite hypothesis
        if len(self.probe_selection_history) >= 10:
            recent = self.probe_selection_history[-10:]
            _, tested = zip(*recent)
            most_common = max(set(tested), key=tested.count)
            if tested.count(most_common) > 7:  # 70% same target
                self.confirmation_pattern_count += 1
    
    def detect_reasoning_bias(self, episode: int) -> Dict[str, Any]:
        """Detect systematic biases in reasoning."""
        flags = []
        
        # Confirmation bias
        if self.confirmation_pattern_count > 3:
            flags.append('confirmation_bias')
        
        # Hypothesis fixation (always picks same one)
        if self.hypothesis_frequencies:
            total = sum(self.hypothesis_frequencies.values())
            max_freq = max(self.hypothesis_frequencies.values())
            if max_freq / total > 0.8:
                flags.append('hypothesis_fixation')
        
        # Overconfidence (will be detected by ConfidenceCalibrator)
        # Underconfidence (will be detected by ConfidenceCalibrator)
        
        return {
            'flags': flags,
            'confirmation_pattern_count': self.confirmation_pattern_count,
            'hypothesis_distribution': self.hypothesis_frequencies,
            'has_systematic_bias': len(flags) > 0
        }


class MetaCognition:
    """
    Central meta-cognitive layer for RAVANA.
    
    This layer asks: "How is my thinking going?"
    Not: "What do I believe?"
    
    It monitors:
    - Are my probes working?
    - Is my confidence calibrated?
    - Am I systematically biased?
    - Should I switch epistemic mode?
    """
    
    def __init__(self, config: Optional[MetaCognitiveConfig] = None):
        self.config = config or MetaCognitiveConfig()
        
        # Sub-components
        self.quality_tracker = ReasoningQualityTracker(self.config)
        self.calibrator = ConfidenceCalibrator(self.config)
        self.bias_detector = BiasDetector(self.config)
        
        # State
        self.current_mode: EpistemicMode = EpistemicMode.EXPLORATORY
        self.mode_start_episode: int = 0
        self.mode_history: List[Tuple[int, EpistemicMode]] = []
        
        # Alert tracking
        self.probe_failure_streak: int = 0
        self.recovery_mode_activations: int = 0
        
    def assess_probe_outcome(self, probe_design: Dict[str, Any],
                            probe_result: Dict[str, Any],
                            episode: int) -> Dict[str, Any]:
        """
        Assess: Was this probe successful?
        
        Returns assessment dict with quality score.
        """
        conclusive = probe_result.get('conclusive', True)
        confidence = probe_result.get('confidence', 0.5)
        
        # Record for tracking
        result = ProbeResult(
            probe_id=episode,
            design=probe_design,
            outcome=probe_result,
            conclusive=conclusive,
            information_gain=0.5 if conclusive else 0.1,
            episode=episode
        )
        self.quality_tracker.record_probe_result(result)
        
        # Track failure streak
        if not conclusive:
            self.probe_failure_streak += 1
        else:
            self.probe_failure_streak = 0
        
        # Update calibration
        self.calibrator.record_outcome(
            confidence, 
            'conclusive' if conclusive else 'inconclusive',
            episode
        )
        
        # Assess quality
        quality = 'high' if conclusive and confidence > 0.6 else \
                  'medium' if conclusive else 'low'
        
        return {
            'quality': quality,
            'conclusive': conclusive,
            'confidence': confidence,
            'failure_streak': self.probe_failure_streak,
            'alert': self.probe_failure_streak > 5
        }
    
    def update_calibration_from_outcome(self,
                                        predicted_confidence: float,
                                        actual_outcome: str,
                                        episode: int):
        """Update confidence calibration based on actual outcome."""
        self.calibrator.record_outcome(predicted_confidence, actual_outcome, episode)
    
    def detect_reasoning_bias(self, episode: int) -> Dict[str, Any]:
        """Detect systematic biases in current reasoning."""
        return self.bias_detector.detect_reasoning_bias(episode)
    
    def recommend_epistemic_mode(self, episode: int) -> EpistemicMode:
        """
        Recommend which epistemic mode to use based on meta-cognitive assessment.
        
        This is the KEY OUTPUT of the meta-cognitive layer.
        """
        # Get current status
        metrics = self.quality_tracker.compute_current_metrics()
        calibration = self.calibrator.get_calibration_status()
        
        # RECOVERY: Systematic probe failure detected
        if self.probe_failure_streak > 10 or metrics.probe_effectiveness < 0.3:
            if self.current_mode != EpistemicMode.RECOVERY:
                self._switch_mode(EpistemicMode.RECOVERY, episode)
                self.recovery_mode_activations += 1
            return EpistemicMode.RECOVERY
        
        # CAUTIOUS: High uncertainty or poor calibration
        if (not calibration['is_well_calibrated'] or 
            metrics.confidence_calibration < 0.5 or
            self.probe_failure_streak > 5):
            if self.current_mode != EpistemicMode.CAUTIOUS:
                self._switch_mode(EpistemicMode.CAUTIOUS, episode)
            return EpistemicMode.CAUTIOUS
        
        # CONFIDENT: Well-calibrated, effective probes
        if (calibration['is_well_calibrated'] and 
            metrics.probe_effectiveness > 0.7 and
            self.probe_failure_streak == 0):
            if self.current_mode != EpistemicMode.CONFIDENT:
                self._switch_mode(EpistemicMode.CONFIDENT, episode)
            return EpistemicMode.CONFIDENT
        
        # Default: EXPLORATORY
        if self.current_mode != EpistemicMode.EXPLORATORY:
            self._switch_mode(EpistemicMode.EXPLORATORY, episode)
        return EpistemicMode.EXPLORATORY
    
    def _switch_mode(self, new_mode: EpistemicMode, episode: int):
        """Record mode switch."""
        self.mode_history.append((episode, new_mode))
        self.current_mode = new_mode
        self.mode_start_episode = episode
    
    def design_probe_for_uncertainty(self, belief_state: List[Any]) -> Dict[str, Any]:
        """
        Design a probe specifically targeting current uncertainty.
        
        This is an INTENTIONAL experiment design, not passive observation.
        """
        if not belief_state:
            return {'type': 'exploratory', 'target': 'unknown'}
        
        # Based on current mode, design different probe strategies
        if self.current_mode == EpistemicMode.RECOVERY:
            # Try radically different approach
            return {
                'type': 'radical_probe',
                'target': 'fundamental_assumptions',
                'reason': 'systematic_failure_detected'
            }
        
        elif self.current_mode == EpistemicMode.CAUTIOUS:
            # Conservative, small perturbations
            return {
                'type': 'conservative_probe',
                'target': 'stable_region',
                'reason': 'high_uncertainty'
            }
        
        elif self.current_mode == EpistemicMode.CONFIDENT:
            # Can afford aggressive probes
            return {
                'type': 'information_maximizing',
                'target': 'uncertainty_boundary',
                'reason': 'well_calibrated'
            }
        
        else:  # EXPLORATORY
            return {
                'type': 'standard_probe',
                'target': 'current_hypotheses',
                'reason': 'systematic_exploration'
            }
    
    def get_meta_status(self) -> Dict[str, Any]:
        """Full meta-cognitive status."""
        metrics = self.quality_tracker.compute_current_metrics()
        calibration = self.calibrator.get_calibration_status()
        
        return {
            'current_mode': self.current_mode.value,
            'mode_duration': len(self.mode_history),
            'recent_probe_failures': self.probe_failure_streak,
            'calibration_error': abs(calibration['bias_estimate']),
            'is_well_calibrated': calibration['is_well_calibrated'],
            'reasoning_quality': {
                'prediction_accuracy': metrics.prediction_accuracy,
                'probe_effectiveness': metrics.probe_effectiveness,
                'confidence_calibration': metrics.confidence_calibration
            },
            'recovery_mode_activations': self.recovery_mode_activations,
            'mode_history': [(ep, m.value) for ep, m in self.mode_history[-5:]]
        }
