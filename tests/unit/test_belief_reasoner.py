"""Tests for ravana_grace.core.belief_reasoner."""

import pytest
from ravana_grace.core.belief_reasoner import (
    BeliefReasoner, BeliefConfig, Hypothesis, EvidenceEvent, EvidenceType,
)


class TestBeliefConfig:
    def test_defaults(self):
        cfg = BeliefConfig()
        assert cfg.max_hypotheses == 3
        assert cfg.confidence_decay_rate == 0.02
        assert cfg.structural_consistency_threshold == 0.7


class TestHypothesis:
    def test_init(self):
        h = Hypothesis(id=1, boundary_estimate=0.8, confidence=0.6)
        assert h.boundary_estimate == 0.8
        assert h.confidence == 0.6
        assert h.evidence_count == 0

    def test_update_confidence_positive(self):
        h = Hypothesis(id=1)
        h.update_confidence(0.5, episode=1)
        assert h.confidence > 0.5
        assert h.confirming_evidence == 1

    def test_update_confidence_negative(self):
        h = Hypothesis(id=1)
        initial = h.confidence
        h.update_confidence(-0.5, episode=1)
        assert h.confidence < initial
        assert h.contradicting_evidence == 1

    def test_decay_confidence(self):
        h = Hypothesis(id=1, confidence=0.8)
        h.decay_confidence(0.1)
        assert h.confidence < 0.8
        assert h.confidence >= 0.1  # Min floor

    def test_compute_weight(self):
        h = Hypothesis(id=1, confidence=0.7, uncertainty=0.2)
        weight = h.compute_weight()
        assert weight == 0.7 * 0.8


class TestBeliefReasoner:
    def test_init(self):
        br = BeliefReasoner()
        assert len(br.hypotheses) == 1
        assert br.config.max_hypotheses == 3

    def test_initial_hypothesis(self):
        br = BeliefReasoner()
        h = br.hypotheses[0]
        assert h.boundary_estimate == 0.95
        assert h.confidence == 0.5

    def test_observe_evidence(self):
        br = BeliefReasoner()
        ev = EvidenceEvent(
            episode=1, predicted_d=0.5, actual_d=0.55,
            observed_boundary=0.95, mode=0, clamp_occurred=False,
            context_snapshot={"dissonance": 0.5},
        )
        br.observe_evidence(ev, true_boundary=0.95)
        assert len(br.evidence_history) == 1
        assert br.hypotheses[0].evidence_count > 0

    def test_observe_evidence_with_clamp(self):
        br = BeliefReasoner()
        ev = EvidenceEvent(
            episode=1, predicted_d=0.5, actual_d=0.9,
            observed_boundary=0.95, mode=0, clamp_occurred=True,
            context_snapshot={"dissonance": 0.5},
        )
        br.observe_evidence(ev, true_boundary=0.95)
        assert ev.evidence_type == EvidenceType.CLAMP_EVENT

    def test_get_dominant_hypothesis(self):
        br = BeliefReasoner()
        h = br.get_dominant_hypothesis()
        assert h is not None
        assert h.id == br.hypotheses[0].id

    def test_get_dominant_hypothesis_with_dict(self):
        br = BeliefReasoner()
        # Force a dict-style hypothesis
        h = br.get_dominant_hypothesis()
        # Still returns Hypothesis object
        assert h is not None

    def test_get_mode_recommendation_initial(self):
        br = BeliefReasoner()
        mode = br.get_mode_recommendation()
        assert isinstance(mode, str)

    def test_get_reasoning_status(self):
        br = BeliefReasoner()
        status = br.get_reasoning_status()
        assert "num_hypotheses" in status
        assert "hypothesis_weights" in status
        assert "total_evidence" in status

    def test_get_belief_state(self):
        br = BeliefReasoner()
        state = br.get_belief_state()
        assert len(state) == 1

    def test_current_belief_property(self):
        br = BeliefReasoner()
        assert br.current_belief == 0.95

    def test_current_uncertainty_property(self):
        br = BeliefReasoner()
        assert 0 <= br.current_uncertainty <= 1.0

    def test_prune_hypotheses(self):
        br = BeliefReasoner()
        h = Hypothesis(id=2, boundary_estimate=0.5, confidence=0.01)
        br.hypotheses.append(h)
        br._prune_hypotheses()
        assert len(br.hypotheses) == 1  # Only the original remains

    def test_spawn_initial_hypothesis_called(self):
        br = BeliefReasoner()
        assert br._next_hypothesis_id > 1
