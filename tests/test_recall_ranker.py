"""Unit tests for the Multi-Signal Recall Relevance Scorer (MRS).

Tests the RecallRanker class: candidate ranking, query-type detection,
entity match bonus, confidence weighting, threshold adaptation wiring,
and candidate collection from the PersonalFactStore.
"""
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
_src = os.path.join(_REPO_ROOT, "ravana", "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
_ml_src = os.path.join(_REPO_ROOT, "ravana_ml", "src")
if _ml_src not in sys.path:
    sys.path.insert(0, _ml_src)

from ravana.chat.recall_ranker import (
    RecallCandidate,
    RecallRanker,
    _QUERY_TYPE_PATTERNS,
    _QUERY_TYPE_WEIGHTS,
    _DEFAULT_TYPE_WEIGHTS,
)
from ravana.chat.relevance_store import (
    RelevanceThreshold,
    RelevanceScorer,
    record_false_positive,
    record_false_negative,
)


class TestQueryTypeDetection:
    def test_biographical_query(self):
        ranker = RecallRanker()
        assert ranker.detect_query_type("what is my name") == "biographical"
        assert ranker.detect_query_type("where do i live") == "biographical"

    def test_entity_query(self):
        ranker = RecallRanker()
        assert ranker.detect_query_type("what is my cat name") == "entity"

    def test_topical_query(self):
        ranker = RecallRanker()
        assert ranker.detect_query_type("what did i tell you about the sea") == "topical"

    def test_stance_query(self):
        ranker = RecallRanker()
        assert ranker.detect_query_type("what is my opinion on privacy") == "stance"

    def test_activity_query(self):
        ranker = RecallRanker()
        assert ranker.detect_query_type("what do i keep on my rooftop") == "activity"

    def test_material_query(self):
        ranker = RecallRanker()
        assert ranker.detect_query_type("what is my cabin made of") == "material"

    def test_count_query(self):
        ranker = RecallRanker()
        assert ranker.detect_query_type("how many pets do i have") == "count"

    def test_whereabouts_query(self):
        ranker = RecallRanker()
        assert ranker.detect_query_type("where do i keep my tools") == "whereabouts"

    def test_unknown_query(self):
        ranker = RecallRanker()
        assert ranker.detect_query_type("hello there") == "unknown"
        assert ranker.detect_query_type("") == "unknown"

    def test_all_patterns_are_valid_regex(self):
        import re
        for qtype, pattern in _QUERY_TYPE_PATTERNS:
            try:
                re.compile(pattern)
            except re.error:
                pytest.fail(f"Invalid regex for query type: {qtype}")


class TestCandidateRanking:
    def _make_candidate(self, answer, fact_key=("", "", ""), confidence=0.6,
                        turn_number=0, entity=None, candidate_type=None):
        return RecallCandidate(
            answer=answer, fact_key=fact_key, fact_value=answer,
            turn_number=turn_number, confidence=confidence,
            entity=entity, candidate_type=candidate_type,
        )

    def test_rank_empty_list(self):
        ranker = RecallRanker()
        assert ranker.rank([], "what is my name") == []

    def test_rank_single_candidate(self):
        ranker = RecallRanker()
        cand = self._make_candidate("your name is alice.")
        result = ranker.rank([cand], "what is my name")
        assert len(result) == 1

    def test_rank_orders_by_relevance(self):
        ranker = RecallRanker()
        cand1 = self._make_candidate("your cat name is whiskers.")
        cand2 = self._make_candidate("the weather is nice today.")
        result = ranker.rank([cand1, cand2], "what is my cat name")
        assert result[0].answer == "your cat name is whiskers."

    def test_rank_and_filter_below_threshold(self):
        ranker = RecallRanker()
        ranker._threshold.base_threshold = 0.99
        cand1 = self._make_candidate("your name is alice.")
        result = ranker.rank_and_filter([cand1], "what is my name")
        assert isinstance(result, list)

    def test_rank_and_filter_passes_relevant(self):
        ranker = RecallRanker()
        ranker._threshold.base_threshold = 0.01
        cand1 = self._make_candidate("your name is alice.")
        result = ranker.rank_and_filter([cand1], "what is my name")
        assert len(result) >= 1


class TestEntityMatchBonus:
    def test_entity_match_gives_bonus(self):
        ranker = RecallRanker()
        cand = RecallCandidate(
            answer="your cat name is whiskers.",
            fact_key=("cat", "name", "whiskers"),
            fact_value="whiskers", entity="cat",
        )
        bonus = ranker._entity_match_bonus("what is my cat name", cand)
        assert bonus > 0.0

    def test_no_entity_match_no_bonus(self):
        ranker = RecallRanker()
        cand = RecallCandidate(
            answer="the weather is nice.",
            fact_key=("weather", "is", "nice"),
            fact_value="nice",
        )
        bonus = ranker._entity_match_bonus("what is my cat name", cand)
        assert bonus == 0.0

    def test_entity_bonus_capped(self):
        ranker = RecallRanker()
        cand = RecallCandidate(
            answer="cat cat cat.",
            fact_key=("cat", "cat", "cat"),
            fact_value="cat", entity="cat",
        )
        bonus = ranker._entity_match_bonus("cat cat cat", cand)
        assert bonus <= 0.15


class TestConfidenceWeighting:
    def test_high_confidence_gives_bonus(self):
        ranker = RecallRanker()
        cand = RecallCandidate(
            answer="your name is alice.",
            fact_key=("", "", ""), fact_value="alice", confidence=1.0,
        )
        assert ranker._confidence_weight(cand) > 0.0

    def test_zero_confidence_no_bonus(self):
        ranker = RecallRanker()
        cand = RecallCandidate(
            answer="your name is alice.",
            fact_key=("", "", ""), fact_value="alice", confidence=0.0,
        )
        assert ranker._confidence_weight(cand) == 0.0

    def test_confidence_bonus_capped(self):
        ranker = RecallRanker()
        cand = RecallCandidate(
            answer="your name is alice.",
            fact_key=("", "", ""), fact_value="alice", confidence=1.0,
        )
        assert ranker._confidence_weight(cand) <= 0.10


class TestQueryTypeWeights:
    def test_all_types_have_weights(self):
        for qtype, weights in _QUERY_TYPE_WEIGHTS.items():
            for signal in ("content", "graph", "recency", "type"):
                assert signal in weights

    def test_default_weights_for_unknown(self):
        ranker = RecallRanker()
        weights = ranker._query_type_weights("unknown")
        assert weights == _DEFAULT_TYPE_WEIGHTS

    def test_weights_are_positive(self):
        for qtype, weights in _QUERY_TYPE_WEIGHTS.items():
            for signal, weight in weights.items():
                assert weight > 0


class TestScoreText:
    def test_score_text_empty(self):
        ranker = RecallRanker()
        assert ranker.score_text("query", "") == 0.0

    def test_score_text_relevant(self):
        ranker = RecallRanker()
        score = ranker.score_text("what is my name", "your name is alice.")
        assert score > 0.0

    def test_score_text_unrelevant(self):
        ranker = RecallRanker()
        score = ranker.score_text("what is my name", "the weather is nice.")
        assert score >= 0.0

    def test_score_text_with_query_type(self):
        ranker = RecallRanker()
        score = ranker.score_text("what is my name", "your name is alice.",
                                  query_type="biographical")
        assert score > 0.0


class TestFeedbackWiring:
    def test_wire_feedback_increases_fp_count(self):
        ranker = RecallRanker()
        initial = ranker._threshold.false_positive_count
        ranker.wire_feedback(correction_type="direct", severity=0.8, current_turn=5)
        assert ranker._threshold.false_positive_count > initial

    def test_wire_feedback_increases_total(self):
        ranker = RecallRanker()
        initial = ranker._threshold.total_scored
        ranker.wire_feedback(correction_type="direct", severity=0.5, current_turn=5)
        assert ranker._threshold.total_scored > initial

    def test_wire_feedback_adapts_threshold(self):
        ranker = RecallRanker()
        initial = ranker._threshold.base_threshold
        for i in range(15):
            ranker.wire_feedback(correction_type="direct", severity=0.5, current_turn=i)
        assert ranker._threshold.base_threshold != initial

    def test_record_true_positive(self):
        ranker = RecallRanker()
        initial = ranker._threshold.total_scored
        ranker.record_true_positive(current_turn=5)
        assert ranker._threshold.total_scored > initial


class TestCandidateCollection:
    def test_collect_empty_store(self):
        ranker = RecallRanker()
        assert ranker.collect_candidates("what is my name", None) == []

    def test_collect_from_store(self):
        from ravana.chat.personal_fact_store import PersonalFactStore
        ranker = RecallRanker()
        store = PersonalFactStore()
        store.assert_fact("i", "name", "alice", confidence=0.9)
        store.advance_turn()
        store.assert_fact("cat", "name", "whiskers", confidence=0.7)
        candidates = ranker.collect_candidates("what is my name", store)
        assert len(candidates) > 0

    def test_collect_skips_superseded(self):
        from ravana.chat.personal_fact_store import PersonalFactStore
        ranker = RecallRanker()
        store = PersonalFactStore()
        store.assert_fact("i", "name", "alice", confidence=0.9)
        store.advance_turn()
        store.assert_fact("i", "name", "bob", confidence=0.9)
        for key, fact in store.facts.items():
            if fact.value == "alice":
                fact.superseded = True
        candidates = ranker.collect_candidates("what is my name", store)
        for cand in candidates:
            assert "alice" not in cand.answer or "bob" in cand.answer

    def test_collect_respects_max(self):
        from ravana.chat.personal_fact_store import PersonalFactStore
        ranker = RecallRanker()
        store = PersonalFactStore()
        for i in range(20):
            store.assert_fact("i", f"attr{i}", f"val{i}", confidence=0.5)
            store.advance_turn()
        candidates = ranker.collect_candidates("attr", store, max_candidates=5)
        assert len(candidates) <= 5


class TestRecallCandidate:
    def test_to_dict(self):
        cand = RecallCandidate(
            answer="your name is alice.",
            fact_key=("i", "name", "alice"), fact_value="alice",
            turn_number=5, confidence=0.9, entity=None, candidate_type="name",
        )
        d = cand.to_dict()
        assert d["answer"] == "your name is alice."
        assert d["fact_key"] == ["i", "name", "alice"]
        assert d["turn_number"] == 5
        assert d["confidence"] == 0.9

    def test_default_values(self):
        cand = RecallCandidate(answer="test", fact_key=("", "", ""), fact_value="test")
        assert cand.turn_number == 0
        assert cand.confidence == 0.6
        assert cand.entity is None
        assert cand.source == "structured_recall"


class TestAttrToType:
    def test_name(self):
        assert RecallRanker()._attr_to_type("name") == "name"
    def test_location(self):
        assert RecallRanker()._attr_to_type("location") == "location"
    def test_work(self):
        assert RecallRanker()._attr_to_type("work") == "work"
    def test_unknown(self):
        assert RecallRanker()._attr_to_type("xyz") is None


class TestThresholdPersistence:
    def test_roundtrip(self):
        t = RelevanceThreshold()
        t.base_threshold = 0.25
        t.false_positive_count = 5
        d = t.to_dict()
        t2 = RelevanceThreshold.from_dict(d)
        assert t2.base_threshold == 0.25
        assert t2.false_positive_count == 5

    def test_shared_threshold(self):
        t = RelevanceThreshold()
        t.base_threshold = 0.3
        ranker = RecallRanker(threshold=t)
        assert ranker._threshold.base_threshold == 0.3
        ranker._threshold.base_threshold = 0.4
        assert t.base_threshold == 0.4
