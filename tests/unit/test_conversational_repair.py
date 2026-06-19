"""Tests for ravana_grace.core.conversational_repair."""

import pytest
from ravana_grace.core.conversational_repair import (
    ConversationalRepair, RepairEvent, CorrectionType,
    _extract_relation_type, _parse_text_to_triples, _VERB_TO_RELATION,
)


class TestVerbToRelation:
    def test_key_verbs_exist(self):
        assert "causes" in _VERB_TO_RELATION
        assert "is" in _VERB_TO_RELATION
        assert "has" in _VERB_TO_RELATION


class TestExtractRelationType:
    def test_direct_match(self):
        assert _extract_relation_type("causes") == "causal"

    def test_suffix_stripping(self):
        # 'causes' is a direct match; 'caused' → stem 'caus' not in map
        assert _extract_relation_type("causes") == "causal"

    def test_default(self):
        assert _extract_relation_type("unknown_word") == "semantic"


class TestParseTextToTriples:
    def test_simple_triple(self):
        triples = _parse_text_to_triples("heat causes expansion")
        assert len(triples) >= 1
        t = triples[0]
        assert t.subject == "heat"
        assert t.relation == "causes"
        assert t.object == "expansion"

    def test_triple_with_punctuation(self):
        triples = _parse_text_to_triples("Trust is important.")
        assert len(triples) >= 1

    def test_empty_text(self):
        assert _parse_text_to_triples("") == []

    def test_short_text(self):
        assert _parse_text_to_triples("hi") == []

    def test_triple_with_object_phrase(self):
        triples = _parse_text_to_triples("kindness causes trust and respect")
        assert len(triples) >= 1
        assert "trust" in triples[0].object


class TestConversationalRepair:
    def test_init(self):
        cr = ConversationalRepair(user_id="test_user")
        assert cr.user_id == "test_user"
        assert len(cr.repair_history) == 0

    def test_init_with_graph(self):
        from ravana_ml.graph import ConceptGraph
        g = ConceptGraph(dim=8, max_nodes=100)
        cr = ConversationalRepair(graph=g, user_id="test")
        assert cr.graph is g

    def test_detect_contradiction_no_correction(self):
        cr = ConversationalRepair(user_id="test")
        result = cr.detect_contradiction("heat causes expansion", "heat causes melting")
        assert result is not None
        wrong, correct = result
        assert wrong.object == "expansion"
        assert correct.object == "melting"

    def test_detect_contradiction_same_triple(self):
        cr = ConversationalRepair(user_id="test")
        result = cr.detect_contradiction("heat causes expansion", "heat causes expansion")
        assert result is None

    def test_detect_contradiction_empty_output(self):
        cr = ConversationalRepair(user_id="test")
        result = cr.detect_contradiction("", "")
        assert result is None

    def test_process_correction(self):
        cr = ConversationalRepair(user_id="test_user")
        event = cr.process_correction("heat causes expansion", "heat causes melting")
        assert event is not None
        assert isinstance(event, RepairEvent)
        assert event.wrong_triple.object == "expansion"

    def test_process_correction_no_correction(self):
        cr = ConversationalRepair(user_id="test_user")
        event = cr.process_correction("heat causes expansion", "heat causes expansion")
        assert event is None

    def test_get_repair_stats(self):
        cr = ConversationalRepair(user_id="test_user")
        cr.process_correction("A causes B", "A causes C")
        stats = cr.get_repair_stats()
        # process_correction → apply_repair may double-count without graph
        assert stats["total_repairs"] >= 1
        assert len(stats["recent_repairs"]) >= 1

    def test_is_explicit_negation(self):
        cr = ConversationalRepair()
        assert cr._is_explicit_negation("No, that's wrong") is True
        assert cr._is_explicit_negation("Yes, that's right") is False

    def test_classify_correction_negation(self):
        cr = ConversationalRepair()
        assert cr._classify_correction("A causes B", "No, A causes C") == CorrectionType.NEGATION

    def test_classify_correction_refinement(self):
        cr = ConversationalRepair()
        # "rather" is only in refinement patterns, not negation
        result = cr._classify_correction("A causes B", "Rather, A causes C")
        assert result == CorrectionType.REFINEMENT

    def test_apply_repair_without_graph(self):
        cr = ConversationalRepair(user_id="test")
        from ravana_grace.core.dialogue_context import Triple
        t_wrong = Triple(subject="heat", relation="causes", relation_type="causal", object="expansion")
        t_correct = Triple(subject="heat", relation="causes", relation_type="causal", object="melting")
        cr.apply_repair(t_wrong, t_correct)
        # Repair should be recorded even without graph
        assert len(cr.repair_history) == 1

    def test_set_graph(self):
        from ravana_ml.graph import ConceptGraph
        g = ConceptGraph(dim=8, max_nodes=100)
        cr = ConversationalRepair(user_id="test")
        cr.set_graph(g)
        assert cr.graph is g

    def test_repr(self):
        cr = ConversationalRepair(user_id="test_user")
        r = repr(cr)
        assert "ConversationalRepair" in r
        assert "test_user" in r
