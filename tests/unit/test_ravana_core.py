"""Tests for ravana/src/ravana/core modules: BeliefStore, DualProcessController, IdentityEngine, MeaningEngine, SleepConsolidation, MetaCognition, GlobalWorkspace, VADEmotionEngine."""

import pytest
import numpy as np
from ravana.core.belief_store import BeliefStore, BeliefConfig, UserBeliefProfile
from ravana.core.dual_process import DualProcessController, DualProcessConfig, Route, RouteDecision
from ravana.core.identity import IdentityEngine, IdentityConfig, IdentityState
from ravana.core.meaning import MeaningEngine, MeaningConfig, MeaningState
from ravana.core.sleep import SleepConsolidation, SleepConfig
from ravana.core.meta_cognition import MetaCognition, MetaCognitiveConfig, EpistemicMode
from ravana.core.global_workspace import GlobalWorkspace, GWConfig, WorkspaceItem
from ravana.core.emotion import VADEmotionEngine, VADConfig, VADState
from ravana.core.mirror import EmotionalMirrorEngine, MirrorConfig, MirrorState, UserEmotionDetector


# ── BeliefStore Tests ──

class TestBeliefConfig:
    def test_default_config(self):
        cfg = BeliefConfig()
        assert cfg.recency_decay == 0.1
        assert cfg.min_confidence_threshold == 0.1


class TestUserBeliefProfile:
    def test_default_profile(self):
        p = UserBeliefProfile(user_id="test")
        assert p.user_id == "test"
        assert p.beliefs == {}
        assert p.contradictions == []
        assert p.turn_num == 0


class TestBeliefStore:
    def test_default_init(self):
        bs = BeliefStore()
        assert bs.config.recency_decay == 0.1
        assert bs.current_user is None

    def test_set_user_creates_profile(self):
        bs = BeliefStore()
        bs.set_user("alice")
        assert bs.current_user == "alice"
        assert "alice" in bs.users

    def test_get_user_profile(self):
        bs = BeliefStore()
        profile = bs.get_user_profile("bob")
        assert profile.user_id == "bob"
        # Same user returns same profile
        assert bs.get_user_profile("bob") is profile

    def test_assert_belief(self):
        bs = BeliefStore()
        bs.set_user("alice")
        bs.assert_belief("trust", "is", "good", confidence=0.9)
        belief = bs.query_belief("trust", "is")
        assert belief is not None
        assert belief[0] == "good"
        assert belief[1] == 0.9

    def test_query_belief_nonexistent(self):
        bs = BeliefStore()
        bs.set_user("alice")
        assert bs.query_belief("unknown", "is") is None

    def test_detect_contradiction(self):
        bs = BeliefStore()
        bs.set_user("alice")
        bs.assert_belief("trust", "is", "good")
        contradiction = bs.detect_contradiction("trust", "is", "bad")
        assert contradiction is not None
        assert contradiction[0][0] == "good"
        assert contradiction[1] == "bad"

    def test_detect_contradiction_no_match(self):
        bs = BeliefStore()
        bs.set_user("alice")
        assert bs.detect_contradiction("trust", "is", "good") is None  # First assertion

    def test_resolve_contradiction(self):
        bs = BeliefStore()
        bs.set_user("alice")
        bs.resolve_contradiction(
            ("trust", "is", "good"), ("trust", "is", "bad"),
            choice="accept_new"
        )
        assert len(bs.users["alice"].contradictions) == 1

    def test_cross_reference_users(self):
        bs = BeliefStore()
        bs.set_user("alice")
        bs.assert_belief("trust", "is", "good")
        bs.set_user("bob")
        bs.assert_belief("trust", "is", "good")
        results = bs.cross_reference_users("trust", "is", ["alice", "bob"])
        assert len(results) == 2
        assert all(v[0] == "good" for v in results.values())

    def test_find_agreement(self):
        bs = BeliefStore()
        bs.set_user("alice")
        bs.assert_belief("trust", "is", "good")
        bs.set_user("bob")
        bs.assert_belief("trust", "is", "good")
        assert bs.find_agreement("trust", "is", ["alice", "bob"]) == "good"

    def test_find_disagreement(self):
        bs = BeliefStore()
        bs.set_user("alice")
        bs.assert_belief("trust", "is", "good")
        bs.set_user("bob")
        bs.assert_belief("trust", "is", "bad")
        disagreements = bs.find_disagreement("trust", "is", ["alice", "bob"])
        assert len(disagreements) == 1
        assert ("alice", "bob") in disagreements or ("bob", "alice") in disagreements

    def test_get_state_and_set_state(self):
        bs = BeliefStore()
        bs.set_user("alice")
        bs.assert_belief("trust", "is", "good")
        state = bs.get_state()
        assert state['global_turn'] == 0  # Not incremented by assert alone; call advance_turn()
        assert 'alice' in state['users']

        bs2 = BeliefStore()
        bs2.set_state(state)
        assert bs2.users['alice'].beliefs[("trust", "is")][0] == "good"

    def test_get_all_beliefs_for_subject(self):
        bs = BeliefStore()
        bs.set_user("alice")
        bs.assert_belief("trust", "is", "good")
        bs.assert_belief("trust", "creates", "friendship")
        beliefs = bs.get_all_beliefs_for_subject("trust")
        assert "alice" in beliefs
        assert len(beliefs["alice"]) == 2

    def test_advance_turn(self):
        bs = BeliefStore()
        bs.set_user("alice")
        initial = bs.users["alice"].turn_num
        bs.advance_turn()
        assert bs.global_turn == 1
        assert bs.users["alice"].turn_num == initial + 1

    def test_reconcile_no_contradictions(self):
        bs = BeliefStore()
        bs.set_user("alice")
        bs.assert_belief("trust", "is", "good")
        resolved = bs.reconcile()
        assert resolved == {}


# ── MeaningEngine Tests (ravana version) ──

class TestRavanaMeaningEngine:
    def test_default_init(self):
        me = MeaningEngine()
        assert me.state.accumulated_meaning == 0.0

    def test_compute_meaning_positive(self):
        me = MeaningEngine()
        me.compute_meaning(episode=1, pre_dissonance=0.8, post_dissonance=0.3,
                          pre_identity=0.5, post_identity=0.7, predictive_gain=0.5, effort=0.2)
        assert me.state.accumulated_meaning > 0.0

    def test_compute_meaning_zero(self):
        me = MeaningEngine()
        me.compute_meaning(1, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5)
        assert me.state.accumulated_meaning == pytest.approx(0.0, abs=1e-6)


# ── SleepConsolidation Tests (ravana version) ──

class TestRavanaSleepConsolidation:
    def test_default_init(self):
        sc = SleepConsolidation()
        assert sc.metrics["total_sleep_cycles"] == 0

    def test_run_cycle_basic(self):
        """Test basic sleep cycle with minimal graph."""
        from ravana_ml.graph import ConceptGraph
        graph = ConceptGraph(dim=8, max_nodes=100)

        # Add test nodes
        n1 = graph.add_node(vector=np.ones(8, dtype=np.float32), label="concept1")
        n2 = graph.add_node(vector=np.ones(8, dtype=np.float32), label="concept2")
        n3 = graph.add_node(vector=np.ones(8, dtype=np.float32), label="concept3")
        graph.add_edge(n1.id, n2.id, weight=0.5, relation_type="semantic")
        graph.add_edge(n2.id, n3.id, weight=0.3, relation_type="causal")
        graph.add_edge(n1.id, n3.id, weight=0.05, relation_type="semantic")  # weak edge

        sc = SleepConsolidation(SleepConfig(prune_threshold=0.1))

        metrics = sc.run_cycle(
            graph=graph,
            episodic_buffer=[],
            episodic_triples=[],
            belief_store=None,
            topic_list=[],
            user_model=None,
            impossible_queries=[],
            contradiction_map={},
        )

        assert isinstance(metrics, dict)
        assert "edges_strengthened" in metrics
        assert "edges_pruned" in metrics
        assert sc.metrics["total_sleep_cycles"] == 1

    def test_prune_weak_edges(self):
        from ravana_ml.graph import ConceptGraph
        graph = ConceptGraph(dim=4, max_nodes=100)
        n1 = graph.add_node(vector=np.ones(4, dtype=np.float32), label="a")
        n2 = graph.add_node(vector=np.ones(4, dtype=np.float32), label="b")
        graph.add_edge(n1.id, n2.id, weight=0.05, relation_type="semantic")

        sc = SleepConsolidation(SleepConfig(prune_threshold=0.1))
        pruned = sc._prune_weak_edges(graph, threshold=0.1)
        assert pruned == 1
        assert len(graph.edges) == 0

    def test_normalize_outgoing_weights(self):
        from ravana_ml.graph import ConceptGraph
        graph = ConceptGraph(dim=4, max_nodes=100)
        n1 = graph.add_node(vector=np.ones(4, dtype=np.float32), label="hub")
        n2 = graph.add_node(vector=np.ones(4, dtype=np.float32), label="a")
        n3 = graph.add_node(vector=np.ones(4, dtype=np.float32), label="b")
        e1 = graph.add_edge(n1.id, n2.id, weight=0.9, relation_type="semantic")
        e2 = graph.add_edge(n1.id, n3.id, weight=0.8, relation_type="semantic")

        sc = SleepConsolidation(SleepConfig(downscaling_budget=1.0))
        sc._normalize_outgoing_weights(graph, budget=1.0)
        # Total outgoing should be <= budget
        total = sum(e.weight for _, e in graph.get_outgoing(n1.id))
        assert total <= 1.0 + 1e-6


# ── MetaCognition Tests (ravana version) ──

class TestRavanaMetaCognition:
    def test_record_probe_and_bias(self):
        mc = MetaCognition()
        for _ in range(5):
            mc.record_probe(subject="trust", prediction=0.8, actual=0.8, outcome="confirm")
        biases = mc.detect_reasoning_bias(turn=1)['biases']
        assert 'confirmation_bias' in biases

    def test_record_calibration(self):
        mc = MetaCognition()
        mc.record_calibration(0.9, True)
        assert mc.get_calibration_error() == pytest.approx(0.1)
        mc.record_calibration(0.9, False)
        assert mc.get_calibration_error() == pytest.approx(0.5)

    def test_epistemic_mode_recommendation(self):
        mc = MetaCognition()
        for _ in range(5):
            mc.record_probe("x", 0.5, 0.5, "confirm")
        mode = mc.recommend_epistemic_mode(1)
        assert mode in list(EpistemicMode)


# ── GlobalWorkspace Tests (ravana version) ──

class TestRavanaGlobalWorkspace:
    def test_submit_and_compete(self):
        gw = GlobalWorkspace(config=GWConfig(capacity=3, broadcast_threshold=0.3))
        gw.submit_bid("emotion", {"v": 0.5}, urgency=0.8, valence=0.5, episode=1)
        gw.submit_bid("meaning", {"m": 0.3}, urgency=0.2, valence=0.1, episode=1)
        winners = gw.compete()
        assert len(winners) == 1
        assert winners[0].source == "emotion"


# ── VADEmotionEngine Tests (ravana version) ──

class TestRavanaVADEmotion:
    def test_update_and_label(self):
        vad = VADEmotionEngine()
        vad.update(stimulus_valence=0.5, stimulus_arousal=0.6)
        assert vad.state.valence > 0.0
        assert vad.state.arousal > 0.3
        label = vad.get_emotional_label()
        assert isinstance(label, str) and len(label) > 0

    def test_decay_over_time(self):
        vad = VADEmotionEngine()
        vad.update(stimulus_arousal=0.8)
        initial = vad.state.arousal
        for _ in range(50):
            vad.update()
        assert vad.state.arousal < initial


class TestUserEmotionDetector:
    """Tests for UserEmotionDetector — P2 Emotional Mirroring.

    Note: VAD lexicon is now learned, not hardcoded. Tests must learn
    the target words before detection, or rely on the universal seed set.
    """

    def _seed_word(self, detector, word, v, a, d):
        detector.learn_association(word, (v, a, d), confidence=1.0)

    def test_detect_positive_excitement(self):
        detector = UserEmotionDetector()
        self._seed_word(detector, "excited", 0.8, 0.85, 0.6)
        v, a, d = detector.detect("I am feeling very excited about this!")
        assert v > 0.3
        assert a > 0.5

    def test_detect_negative_frustration(self):
        detector = UserEmotionDetector()
        self._seed_word(detector, "frustrating", -0.65, 0.75, -0.2)
        self._seed_word(detector, "confusing", -0.3, 0.55, -0.35)
        v, a, d = detector.detect("This is really frustrating and confusing")
        assert v < -0.2
        assert a > 0.4

    def test_detect_fear_high_arousal(self):
        detector = UserEmotionDetector()
        self._seed_word(detector, "terrified", -0.85, 0.95, -0.6)
        v, a, d = detector.detect("I am absolutely terrified right now")
        assert v < -0.5
        assert a > 0.7

    def test_detect_neutral_low_arousal(self):
        detector = UserEmotionDetector()
        v, a, d = detector.detect("The sky is blue and grass is green.")
        assert abs(v) < 0.2
        assert a < 0.5

    def test_detect_empty_text(self):
        detector = UserEmotionDetector()
        v, a, d = detector.detect("")
        assert v == 0.0
        assert a == 0.3

    def test_detect_negation_flips_valence(self):
        detector = UserEmotionDetector()
        self._seed_word(detector, "happy", 0.75, 0.6, 0.5)
        v, a, d = detector.detect("I am not happy about this")
        assert v < 0

    def test_detect_intensifier_boost(self):
        detector = UserEmotionDetector()
        self._seed_word(detector, "exciting", 0.8, 0.85, 0.6)
        v1, a1, d1 = detector.detect("This is exciting")
        v2, a2, d2 = detector.detect("This is extremely exciting")
        assert abs(v2) >= abs(v1) or a2 >= a1

    def test_detect_stem_fallback(self):
        detector = UserEmotionDetector()
        # Morphological normalization maps "frustrating" -> "frustrated"
        self._seed_word(detector, "frustrated", -0.65, 0.75, -0.2)
        v, a, d = detector.detect("This is frustrating me")
        assert v < 0

    def test_detect_fallback_keywords(self):
        detector = UserEmotionDetector()
        self._seed_word(detector, "stupid", -0.6, 0.3, -0.1)
        self._seed_word(detector, "boring", -0.4, 0.1, -0.2)
        v, a, d = detector.detect("This is really stupid and boring")
        assert v < 0


class TestEmotionalMirrorEngine:
    """Tests for EmotionalMirrorEngine — P2 Mirror Neuron System.

    Note: VAD lexicon is now learned; tests seed non-seed words they need.
    """

    def _seed_word(self, detector, word, v, a, d):
        detector.learn_association(word, (v, a, d), confidence=1.0)

    def test_mirror_increases_arousal_for_excitement(self):
        vad = VADEmotionEngine()
        mirror = EmotionalMirrorEngine(MirrorConfig(mirror_strength=0.5, contagion_rate=0.5))
        self._seed_word(mirror.detector, "excited", 0.8, 0.85, 0.6)
        init_arousal = vad.state.arousal
        mirror.mirror(vad, "I am so excited about this!")
        assert vad.state.arousal > init_arousal

    def test_mirror_updates_valence_for_positive(self):
        vad = VADEmotionEngine()
        mirror = EmotionalMirrorEngine()
        self._seed_word(mirror.detector, "amazing", 0.85, 0.75, 0.6)
        self._seed_word(mirror.detector, "wonderful", 0.80, 0.60, 0.55)
        init_valence = vad.state.valence
        mirror.mirror(vad, "This is absolutely amazing and wonderful!")
        assert vad.state.valence > init_valence

    def test_mirror_updates_valence_for_negative(self):
        vad = VADEmotionEngine()
        mirror = EmotionalMirrorEngine()
        self._seed_word(mirror.detector, "frustrated", -0.65, 0.75, -0.2)
        mirror.mirror(vad, "I am so angry and frustrated right now")
        assert vad.state.valence < 0

    def test_neutral_text_does_not_engage_mirror(self):
        vad = VADEmotionEngine()
        mirror = EmotionalMirrorEngine()
        mirror.mirror(vad, "The sky is blue and grass is green.")
        assert mirror.state.mirror_engagement < 0.1

    def test_modulation_scales_with_arousal(self):
        vad = VADEmotionEngine()
        mirror = EmotionalMirrorEngine()
        self._seed_word(mirror.detector, "terrified", -0.85, 0.95, -0.6)
        mirror.mirror(vad, "I am terrified!")
        mod = mirror.get_modulation(vad.state)
        assert mod['temperature_mult'] >= 0.5
        assert mod['breadth_mult'] >= 0.5
        assert mod['verbosity_mult'] >= 0.5

    def test_modulation_defaults_when_not_engaged(self):
        mirror = EmotionalMirrorEngine()
        mod = mirror.get_modulation(VADEmotionEngine().state)
        assert mod['temperature_mult'] == 1.0
        assert mod['breadth_mult'] == 1.0
        assert mod['verbosity_mult'] == 1.0

    def test_detect_user_emotion_updates_state(self):
        mirror = EmotionalMirrorEngine()
        self._seed_word(mirror.detector, "excited", 0.8, 0.85, 0.6)
        uv, ua, ud = mirror.detect_user_emotion("I am really excited!")
        assert mirror.state.user_valence > 0.3
        assert mirror.state.user_arousal > 0.5

    def test_get_emotional_label(self):
        mirror = EmotionalMirrorEngine()
        self._seed_word(mirror.detector, "excited", 0.8, 0.85, 0.6)
        mirror.detect_user_emotion("I am so excited!")
        label = mirror.get_emotional_label()
        assert isinstance(label, str) and len(label) > 0

    def test_serialization_round_trip(self):
        mirror = EmotionalMirrorEngine()
        mirror.mirror(VADEmotionEngine(), "I am very happy")
        state_dict = mirror.state.to_dict()
        restored = MirrorState()
        restored.set_state(state_dict)
        assert abs(restored.user_valence - mirror.state.user_valence) < 0.001

    def test_rapport_builds_over_time(self):
        vad = VADEmotionEngine()
        mirror = EmotionalMirrorEngine()
        for _ in range(10):
            mirror.mirror(vad, "This is good!")
        assert mirror.state.rapport_level > 0.01
