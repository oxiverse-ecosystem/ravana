"""Tests for the answer-path / learning-path bridge.

`WebToGraph` (optionally via the schema-consistency gate) writes what it
learns as TYPED edges in the concept graph. `consult_internal` is the read
side -- it runs before the web and answers from what the system already holds.

Before this bridge those two paths were disjoint: a fact the system had just
learned was invisible to `consult_internal`, which returned None and let the
caller fall through to analogical reasoning -- confabulating ("i don't have a
clean line on X yet, but Y keeps coming up nearby") about a fact it was
actively holding.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana_ml", "src"))


@pytest.fixture(scope="module")
def engine():
    import tempfile
    from ravana.chat.engine import CognitiveChatEngine
    _dir = tempfile.mkdtemp(prefix="ravana_bridge_")
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               data_dir=_dir, user_suffix="_bridge")


def _add_concept(eng, label):
    node = eng.graph.add_node(vector=np.zeros(64, dtype=np.float32), label=label)
    eng._concept_keywords.setdefault(label.lower(), []).append(node.id)
    return node


def _add_typed_edge(eng, subj, obj, rel="is_a", conf=0.8):
    """Add a typed edge the way the real writers do -- WITH provenance.

    The bootstrap's artifact edges (`edge_kind: 'unknown'`, no source) are
    deliberately not representable through this helper; see
    test_unprovenanced_artifacts_are_not_answered.
    """
    s = _add_concept(eng, subj)
    o = _add_concept(eng, obj)
    edge = eng.graph.add_edge(source=s.id, target=o.id, weight=0.5,
                              relation_type=rel, confidence=conf)
    if edge is not None and hasattr(edge, "source_metadata"):
        edge.source_metadata.update({
            "source": "test:fixture", "edge_kind": "curated",
        })
    return s, o


# ── the bridge itself ────────────────────────────────────────────────────────

def test_consult_internal_sees_learned_typed_edges(engine):
    from ravana.chat.brain_regions import consult_internal
    _add_typed_edge(engine, "zzzbridgealpha", "a kind of test concept",
                    rel="is_a", conf=0.8)
    ans = consult_internal("zzzbridgealpha", engine)
    assert ans is not None
    assert ans.source == "graph"
    assert "zzzbridgealpha" in ans.text
    assert "a kind of test concept" in ans.text


def test_consult_internal_returns_none_when_nothing_learned(engine):
    """Fail-closed: an unknown subject must not produce an answer."""
    from ravana.chat.brain_regions import consult_internal
    assert consult_internal("zzznotathing", engine) is None


def test_vacuous_relations_are_not_answered(engine):
    """A distributional-proximity edge asserts no fact. Surfacing one as a
    'fact' is the semantic-dementia failure: fluent but empty."""
    from ravana.chat.brain_regions import consult_internal
    _add_typed_edge(engine, "zzzbridgevac", "something nearby",
                    rel="semantic", conf=0.9)
    assert consult_internal("zzzbridgevac", engine) is None


def test_unprovenanced_artifacts_are_not_answered(engine):
    """The bootstrap's PMI/co-occurrence wiring produces a few thousand
    low-confidence edges (`edge_kind: 'unknown'`, no source). They are
    spreading-activation structure, not assertions -- answering from them is
    how a generic word like 'things' ends up with 'things comes before time'.
    """
    from ravana.chat.brain_regions import consult_internal
    s = _add_concept(engine, "zzzbridgeartifact")
    o = _add_concept(engine, "something else entirely")
    engine.graph.add_edge(source=s.id, target=o.id, weight=0.5,
                          relation_type="temporal", confidence=0.4)
    assert consult_internal("zzzbridgeartifact", engine) is None


def test_low_confidence_trace_is_hedged(engine):
    """A first-sighting novel fact is a weak trace. It must be reported as
    something picked up, not asserted as settled knowledge."""
    from ravana.chat.brain_regions import consult_internal
    _add_typed_edge(engine, "zzzbridgeweak", "a half-remembered thing",
                    rel="is_a", conf=0.3)
    ans = consult_internal("zzzbridgeweak", engine)
    assert ans is not None
    assert ans.text.lower().startswith("i picked up that")


def test_high_confidence_is_asserted_directly(engine):
    from ravana.chat.brain_regions import consult_internal
    _add_typed_edge(engine, "zzzbridgehigh", "a well established thing",
                    rel="is_a", conf=0.85)
    ans = consult_internal("zzzbridgehigh", engine)
    assert ans is not None
    assert not ans.text.lower().startswith("i picked up that")


def test_bridge_uses_production_predicate_phrasing(engine):
    """The read side must not invent phrasing the write side never asserted:
    it reuses the production relation-predicate table."""
    from ravana.chat.brain_regions import _relation_predicate
    assert _relation_predicate("is_a") == "is a kind of"
    assert _relation_predicate("part_of") == "is part of"
    # unknown relations fall back to a neutral phrase, never the raw tag
    assert _relation_predicate("some_new_relation") == "relates to"
    assert _relation_predicate("") == "relates to"


def test_lookup_is_case_insensitive(engine):
    """Some writers key by surface form ("H2O" is seeded that way) while the
    subject arrives lowercased. A real node must not be missed over a case
    difference."""
    from ravana.chat.brain_regions import consult_internal
    s = _add_concept(engine, "ZzBridgeMixed")
    o = _add_concept(engine, "a mixed case thing")
    edge = engine.graph.add_edge(source=s.id, target=o.id, weight=0.5,
                                 relation_type="is_a", confidence=0.8)
    edge.source_metadata.update({"source": "test:fixture",
                                 "edge_kind": "curated"})
    engine._concept_keywords.pop("zzbridgemixed", None)
    engine._concept_keywords["ZzBridgeMixed"] = [s.id]

    ans = consult_internal("zzbridgemixed", engine)
    assert ans is not None
    assert "a mixed case thing" in ans.text


def test_best_edges_are_surfaced_first(engine):
    from ravana.chat.brain_regions import consult_internal
    _add_typed_edge(engine, "zzzbridgeorder", "the strong one",
                    rel="is_a", conf=0.9)
    _add_typed_edge(engine, "zzzbridgeorder", "the weak one",
                    rel="has_a", conf=0.2)
    _add_typed_edge(engine, "zzzbridgeorder", "the middle one",
                    rel="part_of", conf=0.5)
    ans = consult_internal("zzzbridgeorder", engine)
    assert ans is not None
    # top two by confidence, strongest first
    assert "the strong one" in ans.text
    assert "the middle one" in ans.text
    assert "the weak one" not in ans.text


# ── end to end: learn, then answer ───────────────────────────────────────────

def test_full_turn_answers_from_a_learned_fact(engine):
    """The regression this bridge exists for: learn a fact, ask about it, and
    the reply must come from the graph rather than from analogical
    confabulation about a fact the system is holding."""
    from ravana.chat.brain_regions import consult_internal

    w2g = engine._get_web_to_graph()
    written = w2g.learn_text(
        "Zzzbridgefungus is a type of organism that decomposes wood.",
        source_url="web:zzzbridgefungus")
    assert written > 0, "the learning path must have written the fact"

    ans = consult_internal("zzzbridgefungus", engine)
    assert ans is not None, "the answer path must see what the learning path wrote"
    assert "zzzbridgefungus" in ans.text.lower()


def test_belief_store_is_consulted(engine):
    """A fact the system already answered should not be re-searched. This is
    the second half of the bridge: `_web_direct_answer` records an accepted
    snippet in BeliefStore, and until now nothing on the read side looked."""
    from ravana.chat.brain_regions import consult_internal
    engine.belief_store.assert_belief(
        "zzzbridgebelief", "def",
        "Zzbridgebelief is a thing we looked up.", 0.8)
    ans = consult_internal("zzzbridgebelief", engine)
    assert ans is not None
    assert ans.source == "belief"
    assert "a thing we looked up" in ans.text


def test_low_confidence_belief_is_hedged(engine):
    """A single unverified web snippet is not settled knowledge."""
    from ravana.chat.brain_regions import consult_internal
    engine.belief_store.assert_belief(
        "zzzbridgebeliefweak", "def",
        "Zzbridgebeliefweak is a maybe thing.", 0.5)
    ans = consult_internal("zzzbridgebeliefweak", engine)
    assert ans is not None
    assert ans.text.lower().startswith("i picked up that")


def test_belief_store_does_not_shadow_consolidated_knowledge(engine):
    """A curated graph edge outranks a low-confidence stored snippet."""
    from ravana.chat.brain_regions import consult_internal
    _add_typed_edge(engine, "zzzbridgeprio", "a consolidated fact",
                    rel="is_a", conf=0.9)
    engine.belief_store.assert_belief(
        "zzzbridgeprio", "def", "Zzbridgeprio is a shaky snippet.", 0.5)
    ans = consult_internal("zzzbridgeprio", engine)
    assert ans is not None
    assert ans.source == "graph"


def test_web_direct_answer_learns_from_the_snippet(engine, monkeypatch):
    """The loop closes: a snippet we accepted and spoke is fed to the SAME
    WebToGraph + schema-consistency gate as every other ingestion, so the
    concept graph acquires it -- not just BeliefStore."""
    from ravana.chat.models import CognitiveResponseContext

    calls = []
    _real = engine._get_web_to_graph()

    class _Spy:
        def learn_text(self, text, source_url=""):
            calls.append((text, source_url))
            return _real.learn_text(text, source_url=source_url)

    monkeypatch.setattr(engine, "_get_web_to_graph", lambda: _Spy())

    class _FakeSearch:
        def search(self, term, max_results=6, local_only=False):
            return [{"url": "https://en.wikipedia.org/wiki/Speed_of_light",
                     "content": "the speed of light in vacuum is about "
                                "299,792,458 metres per second",
                     "title": "x"}]

    engine.search_engine = _FakeSearch()
    ctx = CognitiveResponseContext(
        subject="speed of light",
        raw_input="what is the speed of light", valence=0.5)
    ctx.strategy = ""
    out = engine._web_direct_answer(ctx)
    assert out is not None, "web_direct_answer withheld a valid sourced snippet"
    assert calls, "an accepted snippet must be fed to WebToGraph"
    assert "speed of light" in calls[0][0].lower()


def test_full_turn_no_confabulation_about_a_held_fact(engine, monkeypatch):
    """Before the bridge the engine answered a learned fact with an analogical
    hedge ('i don't have a clean line on X yet, but Y keeps coming up
    nearby'). That is a confabulated citation of a fact it holds.

    Uses a REAL word: the downstream topical-coherence gate compares the
    subject against the answer in GloVe space, so a nonsense subject is
    withheld for having no vector -- a gate artefact, not a bridge failure.
    """
    monkeypatch.setenv("RAVANA_OFFLINE", "1")
    w2g = engine._get_web_to_graph()
    w2g.learn_text(
        "Quasar is a very luminous active galactic nucleus.",
        source_url="web:quasar_bridge")

    reply = engine.process_turn("what is quasar")
    assert isinstance(reply, str) and reply
    low = reply.lower()
    # it must not claim ignorance about a fact it is holding
    assert "don't have a clean line" not in low
    assert "don't really know that yet" not in low
    assert "quasar" in low
