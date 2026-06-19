"""Tests for ravana_ml/episode_injector: EpisodeInjector, Fact, PHARMACOLOGY_KB, ECOLOGY_KB, load_pharmacology_kb, load_ecology_kb."""

import pytest
import numpy as np
from ravana_ml.episode_injector import (
    EpisodeInjector, Fact, PHARMACOLOGY_KB, ECOLOGY_KB,
    load_pharmacology_kb, load_ecology_kb
)


# ── Fact Tests ──

class TestFact:
    def test_minimal_fact(self):
        fact = Fact(subject="coffee", relation="causes", object="alertness")
        assert fact.subject == "coffee"
        assert fact.relation == "causes"
        assert fact.object == "alertness"
        assert fact.confidence == 1.0
        assert fact.source == "manual"

    def test_custom_fact(self):
        fact = Fact(subject="coffee", relation="causes", object="alertness",
                    confidence=0.8, source="test")
        assert fact.confidence == 0.8
        assert fact.source == "test"

    def test_to_triple(self):
        fact = Fact("coffee", "causes", "alertness")
        assert fact.to_triple() == ("coffee", "causes", "alertness")

    def test_repr(self):
        fact = Fact("coffee", "causes", "alertness", confidence=0.9)
        r = repr(fact)
        assert "coffee" in r
        assert "causes" in r
        assert "0.90" in r


# ── EpisodeInjector Tests ──

class TestEpisodeInjector:
    @pytest.fixture
    def injector(self):
        """Create an EpisodeInjector with a mock model and tokenizer."""
        class MockTokenizer:
            def encode(self, text):
                # Return a deterministic list of fake token IDs
                return [hash(c) % 100 for c in text.split()]

        class MockModel:
            def __init__(self):
                from ravana_ml.graph import ConceptGraph
                self.graph = ConceptGraph(dim=8, max_nodes=100)

            def learn(self, ctx, tgt):
                pass  # No-op for testing

        model = MockModel()
        tok = MockTokenizer()
        return EpisodeInjector(model, tok)

    def test_default_init(self, injector):
        assert injector._injection_stats["total"] == 0
        assert injector._injected_facts == []

    def test_inject_facts_with_tuples(self, injector):
        facts = [("coffee", "causes", "alertness")]
        stats = injector.inject_facts(facts, epochs=3, confidence_weighted=False)
        assert stats["successful"] == 1
        assert stats["total"] == 1
        assert stats["epochs_trained"] == 3

    def test_inject_facts_with_fact_objects(self, injector):
        facts = [Fact("coffee", "causes", "alertness", confidence=0.9)]
        stats = injector.inject_facts(facts, epochs=5, confidence_weighted=False)
        assert stats["successful"] == 1

    def test_inject_facts_confidence_weighted(self, injector):
        # High confidence = more epochs
        facts = [Fact("coffee", "causes", "alertness", confidence=1.0)]
        stats = injector.inject_facts(facts, epochs=5, confidence_weighted=True)
        assert stats["epochs_trained"] == 5  # 1.0 * 5 = 5

    def test_inject_facts_low_confidence_fewer_epochs(self, injector):
        facts = [Fact("coffee", "causes", "alertness", confidence=0.3)]
        stats = injector.inject_facts(facts, epochs=5, confidence_weighted=True)
        assert stats["epochs_trained"] == 1  # max(1, int(5 * 0.3)) = max(1, 1) = 1

    def test_inject_facts_skips_short_triples(self, injector):
        facts = [("a",)]  # 1-tuple does not match 3 or 4-tuple normalization
        stats = injector.inject_facts(facts)
        # 1-tuples are not recognized by the normalizer (not length 3 or 4)
        # so they are silently ignored — not counted as skipped or total
        assert stats["successful"] == 0

    def test_inject_facts_4_tuple(self, injector):
        facts = [("coffee", "causes", "alertness", 0.95)]
        stats = injector.inject_facts(facts)
        assert stats["successful"] == 1

    def test_inject_from_dict(self, injector):
        knowledge = {"coffee": {"causes": ["alertness", "energy"]}}
        stats = injector.inject_from_dict(knowledge, confidence=0.8)
        assert stats["successful"] == 2
        assert len(injector._injected_facts) == 2

    def test_inject_from_dict_single_string(self, injector):
        knowledge = {"coffee": {"causes": "alertness"}}
        stats = injector.inject_from_dict(knowledge)
        assert stats["successful"] == 1

    def test_get_stats(self, injector):
        injector.inject_facts([("coffee", "causes", "alertness")])
        stats = injector.get_stats()
        assert stats["unique_facts"] == 1
        assert "graph_nodes" in stats
        assert "graph_edges" in stats

    def test_get_facts_for_subject(self, injector):
        injector.inject_facts([("coffee", "causes", "alertness"),
                               ("coffee", "contains", "caffeine"),
                               ("tea", "contains", "theine")])
        coffee_facts = injector.get_facts_for_subject("coffee")
        assert len(coffee_facts) == 2

    def test_summary(self, injector):
        injector.inject_facts([("coffee", "causes", "alertness")])
        summary = injector.summary()
        assert "EpisodeInjector" in summary
        assert "1/1" in summary  # successful/total

    def test_inject_multiple_epochs(self, injector):
        facts = [("coffee", "causes", "alertness"),
                 ("caffeine", "increases", "heart_rate")]
        stats = injector.inject_facts(facts, epochs=5, confidence_weighted=False)
        assert stats["total"] == 2
        assert stats["successful"] == 2
        assert stats["epochs_trained"] == 10  # 2 facts * 5 epochs


# ── Knowledge Base Tests ──

class TestPHARMACOLOGY_KB:
    def test_has_expected_drugs(self):
        assert "aspirin" in PHARMACOLOGY_KB
        assert "ibuprofen" in PHARMACOLOGY_KB
        assert "caffeine" in PHARMACOLOGY_KB
        assert "nicotine" in PHARMACOLOGY_KB

    def test_has_relations(self):
        for drug, relations in PHARMACOLOGY_KB.items():
            assert len(relations) >= 1
            for rel, objects in relations.items():
                assert len(objects) >= 1


class TestECOLOGY_KB:
    def test_has_expected_entities(self):
        assert "wolf" in ECOLOGY_KB
        assert "deer" in ECOLOGY_KB
        assert "bear" in ECOLOGY_KB
        assert "grass" in ECOLOGY_KB

    def test_has_relations(self):
        for entity, relations in ECOLOGY_KB.items():
            assert len(relations) >= 1


class TestLoadFunctions:
    def test_load_pharmacology_kb(self):
        facts = load_pharmacology_kb()
        assert len(facts) > 0
        assert all(isinstance(f, Fact) for f in facts)
        assert all(f.source == "pharmacology_kb" for f in facts)
        assert all(f.confidence == 0.9 for f in facts)

    def test_load_ecology_kb(self):
        facts = load_ecology_kb()
        assert len(facts) > 0
        assert all(isinstance(f, Fact) for f in facts)
        assert all(f.source == "ecology_kb" for f in facts)
