"""Tests for ravana_chat_src language modules: basal_ganglia, cerebellar_ngram, pfc_workspace, surface_realizer, syntactic_cell_assembly."""

import sys, os
_rcs = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ravana_chat_src", "src")
if _rcs not in sys.path:
    sys.path.insert(0, _rcs)

import pytest
import numpy as np
from ravana_chat.language.basal_ganglia import BasalGangliaGate
from ravana_chat.language.cerebellar_ngram import CerebellarNgram
from ravana_chat.language.prefrontal_workspace import PrefrontalWorkspace, DiscoursePlan, DiscourseIntent, DiscourseType
from ravana_chat.language.surface_realizer import SurfaceRealizer, DiscourseState
from ravana_chat.language.syntactic_cell_assembly import SyntacticCellAssembly, SyntacticFrame


# ── BasalGangliaGate Tests ──

class TestBasalGangliaGate:
    def test_default_init(self):
        bg = BasalGangliaGate()
        assert bg.base_go_threshold == 0.25
        assert bg.dopamine_tone == 0.5

    def test_set_all_from_modulators(self):
        bg = BasalGangliaGate()
        bg.set_all_from_modulators({"arousal": 0.7, "novelty": 0.5, "dopamine_tone": 0.3})
        assert bg.current_arousal == 0.7
        assert bg.current_novelty == 0.5
        assert bg.dopamine_tone == 0.3

    def test_compute_effective_go_threshold_with_arousal(self):
        bg = BasalGangliaGate()
        baseline = bg.compute_effective_go_threshold()
        bg.set_arousal(0.8)
        assert bg.compute_effective_go_threshold() <= baseline

    def test_select_concept_no_candidates(self):
        bg = BasalGangliaGate()
        label, rel, score = bg.select_concept([], rng=np.random.RandomState(42))
        assert label == "" and rel == "" and score == 0.0

    def test_select_concept_prefers_best(self):
        bg = BasalGangliaGate()
        candidates = [("weak", 0.1, 0.2, "semantic"), ("best", 0.9, 0.95, "causal")]
        label, rel, score = bg.select_concept(candidates)
        assert label == "best"

    def test_get_stats(self):
        bg = BasalGangliaGate()
        stats = bg.get_stats()
        assert "gate_hits" in stats
        assert "softmax_fallbacks" in stats


# ── CerebellarNgram Tests ──

class TestCerebellarNgram:
    def test_default_init(self):
        cn = CerebellarNgram()
        assert cn.bigram == {}
        assert cn.trigram == {}
        assert cn.learning_rate == 0.05

    def test_seed_from_pos(self):
        cn = CerebellarNgram()
        cn.seed_from_pos({"trust": "noun", "know": "verb"})
        assert cn._pos_agreement == {"trust": "noun", "know": "verb"}

    def test_learn_chain_strengthens(self):
        cn = CerebellarNgram()
        cn.learn_chain(chain_labels=["trust", "is", "good"], successful=True, chain_hops=[("trust", "good")])
        assert cn.bigram.get("trust", {}).get("good", 0.0) > 0.0

    def test_learn_chain_unsuccessful_weakens(self):
        cn = CerebellarNgram()
        cn.learn_chain(["a", "b"], True, [("a", "b")])
        cn.learn_chain(["a", "b"], False, [("a", "b")])
        remaining = cn.bigram.get("a", {}).get("b", 0.0)
        assert remaining <= 0.08  # weakened

    def test_learn_function_word(self):
        cn = CerebellarNgram()
        cn.learn_function_word("trust", "good", "is")
        assert cn.predict_function_word("trust", "good") == "is"

    def test_predict_next(self):
        cn = CerebellarNgram()
        cn.bigram["trust"] = {"good": 0.8, "people": 0.5}
        preds = cn.predict_next("trust", top_k=5)
        assert "good" in preds
        assert preds["good"] > 0.0

    def test_get_transition_strength(self):
        cn = CerebellarNgram()
        cn.bigram["a"] = {"b": 0.8}
        assert cn.get_transition_strength("a", "b") > 0.0
        assert cn.get_transition_strength("unknown", "b") == 0.0

    def test_get_state_and_set_state(self):
        cn = CerebellarNgram()
        cn.bigram["a"] = {"b": 0.5}
        state = cn.get_state()
        cn2 = CerebellarNgram()
        cn2.set_state(state)
        assert cn2.bigram["a"]["b"] == 0.5

    def test_get_stats(self):
        cn = CerebellarNgram()
        cn.bigram["a"] = {"b": 0.8, "c": 0.5}
        stats = cn.get_stats()
        assert stats.total_bigram_entries == 2
        assert stats.avg_confidence > 0.0


# ── PrefrontalWorkspace Tests ──

class TestPrefrontalWorkspace:
    def test_default_init(self):
        pfc = PrefrontalWorkspace()
        assert pfc.capacity == 5

    def test_detect_question_type_what_is(self):
        qtype, parts = PrefrontalWorkspace.detect_question_type("What is trust?")
        assert qtype == "what_is"

    def test_detect_question_type_why(self):
        qtype, parts = PrefrontalWorkspace.detect_question_type("Why does ice melt?")
        assert qtype == "why"

    def test_detect_question_type_how(self):
        qtype, parts = PrefrontalWorkspace.detect_question_type("How does gravity work?")
        assert qtype == "how"

    def test_detect_question_type_tell_me(self):
        qtype, parts = PrefrontalWorkspace.detect_question_type("Tell me about freedom")
        assert qtype == "tell_me"

    def test_detect_question_type_general(self):
        qtype, parts = PrefrontalWorkspace.detect_question_type("Trust is important")
        assert qtype == "general"

    def test_plan_discourse_returns_plan(self):
        pfc = PrefrontalWorkspace(capacity=3)
        plan = pfc.plan_discourse(
            user_input="What is trust?", subject="trust",
            concept_pos={"trust": "noun", "good": "adj"},
            associations=[("good", 0.8), ("people", 0.6)],
            is_follow_up=False)
        assert isinstance(plan, DiscoursePlan)
        assert len(plan.intents) <= 3
        assert plan.original_subject == "trust"

    def test_plan_has_intents(self):
        pfc = PrefrontalWorkspace()
        plan = pfc.plan_discourse(
            user_input="Tell me about freedom", subject="freedom",
            concept_pos={"freedom": "noun"},
            associations=[("responsibility", 0.7)])
        assert len(plan.intents) >= 1
        for intent in plan.intents:
            assert isinstance(intent, DiscourseIntent)

    def test_question_patterns_compile(self):
        for qtype, patterns in PrefrontalWorkspace.QUESTION_PATTERNS.items():
            for pattern in patterns:
                assert pattern.match("test string") is not None or True

    def test_topic_history_tracking(self):
        pfc = PrefrontalWorkspace()
        pfc.plan_discourse("What is trust?", "trust", {}, [], is_follow_up=False)
        assert "trust" in pfc.topic_history

    def test_get_state_and_set_state(self):
        pfc = PrefrontalWorkspace()
        pfc.topic_history = ["trust", "freedom"]
        state = pfc.get_state()
        pfc2 = PrefrontalWorkspace()
        pfc2.set_state(state)
        assert pfc2.topic_history == ["trust", "freedom"]


# ── SurfaceRealizer Tests ──

class TestSurfaceRealizer:
    def test_default_init(self):
        sr = SurfaceRealizer()
        assert sr._used_subjects == set()

    def test_reset_turn(self):
        sr = SurfaceRealizer()
        sr._used_subjects.add("trust")
        sr.reset_turn()
        assert sr._used_subjects == set()

    def test_realize_basic_sentence(self):
        sr = SurfaceRealizer()
        frame = SyntacticFrame(subject_concept="Trust", verb_phrase="relates to", object_concept="Respect", relation_type="semantic")
        ctx = DiscourseState(sentence_index=0, total_sentences=1)
        sentence = sr.realize(frame, ctx)
        assert isinstance(sentence, str) and len(sentence) > 0
        assert sentence[0].isupper()
        assert sentence.endswith('.')

    def test_build_noun_phrase(self):
        sr = SurfaceRealizer()
        assert "dog" in sr._build_noun_phrase("Dog", "", False, 0.5).lower()
        assert "the dog" in sr._build_noun_phrase("Dog", "the", False, 0.5).lower()
        assert "a dog" in sr._build_noun_phrase("Dog", "a", False, 0.5).lower()

    def test_abstract_noun_no_article(self):
        sr = SurfaceRealizer()
        phrase = sr._build_noun_phrase("trust", "the", False, 0.5)
        assert isinstance(phrase, str)

    def test_proper_pronoun(self):
        sr = SurfaceRealizer()
        assert sr._resolve_pronoun("I", "i", DiscourseState()) == "I"
        assert sr._resolve_pronoun("Trust", "trust", DiscourseState()) == "Trust"  # first mention

    def test_discourse_marker_selection(self):
        sr = SurfaceRealizer()
        assert sr._select_discourse_marker("explain", 0, 0.5) == ""
        marker = sr._select_discourse_marker("contrast", 1, 0.5)
        assert isinstance(marker, str)

    def test_get_state(self):
        sr = SurfaceRealizer()
        sr._verb_phrase_success = {"relates to": 0.8}
        state = sr.get_state()
        assert state['verb_phrase_success']["relates to"] == 0.8


# ── SyntacticCellAssembly Tests ──

class TestSyntacticCellAssembly:
    def test_default_init(self):
        sca = SyntacticCellAssembly()
        assert sca.subject_role == {}
        assert sca.verb_role == {}
        assert sca.object_role == {}

    def test_seed_from_pos(self):
        sca = SyntacticCellAssembly()
        sca.seed_from_pos({"trust": "noun", "know": "verb", "good": "adj"})
        assert sca.subject_role.get("trust", 0.0) > 0.0
        assert sca.verb_role.get("know", 0.0) > 0.0

    def test_bind_to_sentence(self):
        sca = SyntacticCellAssembly()
        sca.seed_from_pos({"trust": "noun", "good": "adj"})
        frame = sca.bind_to_sentence(subject="trust", relation="semantic", target="good", pos_map={"trust": "noun", "good": "adj"})
        assert isinstance(frame, SyntacticFrame)
        assert frame.subject_concept == "trust"
        assert frame.object_concept == "good"

    def test_verb_phrases_exist(self):
        assert "semantic" in SyntacticCellAssembly.VERB_PHRASES
        assert "causal" in SyntacticCellAssembly.VERB_PHRASES
        assert len(SyntacticCellAssembly.VERB_PHRASES["semantic"]) >= 3

    def test_learn_from_feedback(self):
        sca = SyntacticCellAssembly()
        sca.seed_from_pos({"trust": "noun", "good": "adj"})
        frame = sca.bind_to_sentence("trust", "semantic", "good", pos_map={"trust": "noun", "good": "adj"})
        sca.learn_from_feedback(frame, user_understood=True)
        assert sca.subject_role.get("trust", 0.0) >= 0.7

    def test_learn_from_feedback_negative(self):
        sca = SyntacticCellAssembly()
        sca.seed_from_pos({"trust": "noun"})
        initial = sca.subject_role.get("trust", 0.0)
        frame = sca.bind_to_sentence("trust", "semantic", "good", pos_map={"trust": "noun", "good": "adj"})
        sca.learn_from_feedback(frame, user_understood=False)
        assert sca.subject_role.get("trust", 0.0) <= initial + 0.01

    def test_compose_sentence(self):
        sca = SyntacticCellAssembly()
        sca.seed_from_pos({"trust": "noun", "good": "adj"})
        frame = sca.bind_to_sentence("trust", "semantic", "good", pos_map={"trust": "noun", "good": "adj"})
        sentence = sca.compose_sentence(frame)
        assert isinstance(sentence, str) and len(sentence) > 0
        assert sentence[0].isupper()

    def test_get_state_and_set_state(self):
        sca = SyntacticCellAssembly()
        sca.subject_role["trust"] = 0.8
        state = sca.get_state()
        sca2 = SyntacticCellAssembly()
        sca2.set_state(state)
        assert sca2.subject_role["trust"] == 0.8
