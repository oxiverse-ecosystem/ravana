"""
Meta-Cognition for RAVANA.

Monitors own cognitive processes: detects reasoning biases, calibrates confidence,
recommends epistemic modes. Based on: Fleming & Dolan (2012) metacognition review,
Friston's Active Inference (epistemic value).
"""
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any
from collections import deque


class EpistemicMode(Enum):
    """Epistemic stance for knowledge acquisition."""
    EXPLORATORY = "exploratory"      # seek novelty, tolerate error
    CONSERVATIVE = "conservative"    # trust existing, verify new
    CRITICAL = "critical"            # actively challenge beliefs
    INTEGRATIVE = "integrative"     # connect across domains


@dataclass
class MetaCognitiveConfig:
    """Configuration for meta-cognition."""
    probe_failure_threshold: float = 0.4
    confidence_calibration_window: int = 15


class MetaCognition:
    """Meta-cognitive monitor: tracks reasoning quality and recommends epistemic stance."""

    def __init__(self, config: Optional[MetaCognitiveConfig] = None):
        self.config = config or MetaCognitiveConfig()
        self.current_mode = EpistemicMode.EXPLORATORY
        self.probe_history: deque = deque(maxlen=20)
        self.calibration_errors: deque = deque(maxlen=30)
        self.bias_flags: List[str] = []

    def detect_reasoning_bias(self, turn: int) -> Dict[str, Any]:
        """Detect systematic reasoning biases from recent history."""
        biases = []

        # Confirmation bias: only accepting confirmatory evidence
        if len(self.probe_history) >= 5:
            recent = list(self.probe_history)[-5:]
            pos = sum(1 for p in recent if p.get('outcome') == 'confirm')
            if pos >= 4:
                biases.append("confirmation_bias")

        # Anchoring bias: stuck on first interpretation
        if len(self.probe_history) >= 3:
            recent = list(self.probe_history)[-3:]
            first_subj = recent[0].get('subject', '')
            if all(p.get('subject', '') == first_subj for p in recent):
                biases.append("anchoring_bias")

        # Overconfidence: calibration error
        if self.calibration_errors:
            mean_error = np.mean(list(self.calibration_errors))
            if mean_error > 0.2:  # systematic over/under-confidence
                biases.append("overconfidence" if mean_error > 0 else "underconfidence")

        self.bias_flags = biases
        return {'biases': biases, 'turn': turn}

    def recommend_epistemic_mode(self, turn: int) -> EpistemicMode:
        """Recommend epistemic mode based on cognitive state."""
        biases = self.detect_reasoning_bias(turn).get('biases', [])

        if 'confirmation_bias' in biases:
            self.current_mode = EpistemicMode.CRITICAL
        elif 'anchoring_bias' in biases:
            self.current_mode = EpistemicMode.EXPLORATORY
        elif 'overconfidence' in biases:
            self.current_mode = EpistemicMode.CONSERVATIVE
        elif len(biases) == 0:
            self.current_mode = EpistemicMode.INTEGRATIVE

        return self.current_mode

    def record_probe(self, subject: str, prediction: float, actual: float, outcome: str):
        """Record a cognitive probe for bias detection."""
        self.probe_history.append({
            'subject': subject,
            'prediction': prediction,
            'actual': actual,
            'outcome': outcome})  # 'confirm' | 'disconfirm' | 'surprise'

    def record_calibration(self, predicted_conf: float, actual_correct: bool):
        """Record confidence calibration error."""
        actual = 1.0 if actual_correct else 0.0
        error = abs(predicted_conf - actual)
        self.calibration_errors.append(error)

    def get_calibration_error(self) -> float:
        """Mean calibration error over window."""
        if not self.calibration_errors:
            return 0.0
        return float(np.mean(list(self.calibration_errors)))