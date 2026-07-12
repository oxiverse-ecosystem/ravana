"""Tests for the three brain-inspired fixes:
1. Sense disambiguation — context-augmented coherence helpers.
2. Category-error feasibility gate (frontopolar BA10 analog).
3. Counterfactual causal forward simulator (DMN + hippocampus analog).

The pure-logic methods are exercised on a lightweight engine instance created
via __new__ (bypassing the heavy GloVe/decoder __init__) so these stay fast.
The causal simulator gets a hand-built ConceptGraph.
"""

import hashlib
import re

import numpy as np

from ravana.chat.engine import CognitiveChatEngine
from ravana_ml.graph import ConceptGraph, ConceptEdge


def _bare_engine():
    """Build an engine without running __init__ (no GloVe/decoder load)."""
    eng = CognitiveChatEngine.__new__(CognitiveChatEngine)
    return eng


def test_category_error_detected_for_time_subject():
    eng = _bare_engine()
    # "tuesday" is a time concept; "color" is a physical/perceptual property.
    assert eng._is_category_error("what color is tuesday") == "color"
    assert eng._is_category_error("what colour is a day") == "colour"


def test_category_error_detected_for_mental_subject():
    eng = _bare_engine()
    # "thought" is a mental state; "mass"/"weigh" are physical properties.
    assert eng._is_category_error("how many kilograms does a thought weigh") == "weigh"
    assert eng._is_category_error("what is the mass of an idea") == "mass"


def test_category_error_not_flagged_for_physical_subject():
    eng = _bare_engine()
    # "sun" is a physical object -> it CAN have a color. No category error.
    assert eng._is_category_error("what color is the sun") is None
    # "tree" is living -> it can have a color too.
    assert eng._is_category_error("what color is a tree") is None


def test_category_error_response_is_honest():
    eng = _bare_engine()
    resp = eng._category_error_response("what color is tuesday", "tuesday", "color")
    assert "tuesday" in resp.lower()
    assert "time" in resp.lower()
    assert "color" in resp.lower()
    assert "?" in resp  # invites rephrase


# ── Track B Phase 6 (M6): ConceptNet as primary feasibility gate ─────────────

def _engine_with_conceptnet(primary: bool):
    """Bare engine with ConceptNet ontology loaded (if available)."""
    import os as _os
    eng = _bare_engine()
    eng.use_conceptnet_primary = primary
    # The literal category tables must exist for the non-primary fallback.
    eng._CATEGORY_OF_SUBJECT = getattr(CognitiveChatEngine, "_CATEGORY_OF_SUBJECT", {})
    eng._CATEGORY_AFFORDANCES = getattr(CognitiveChatEngine, "_CATEGORY_AFFORDANCES", {})
    eng._PROPERTY_CATEGORIES = getattr(CognitiveChatEngine, "_PROPERTY_CATEGORIES", {})
    # Locate the prebuilt ontology pickle.
    _cur = _os.path.abspath(__file__)
    ont_path = None
    for _ in range(10):
        _cand = _os.path.join(_cur, "data", "conceptnet", "ont.pkl")
        if _os.path.exists(_cand):
            ont_path = _cand
            break
        _cur = _os.path.dirname(_cur)
    if ont_path is None:
        eng._cn_ontology = None
        return eng, False
    from ravana.ontology.conceptnet import ConceptNetOntology
    eng._cn_ontology = ConceptNetOntology.load(ont_path)
    return eng, True


def test_conceptnet_primary_flag_off_uses_literal_dicts():
    """With ConceptNet-primary OFF (default), when ConceptNet is silent/absent
    the gate falls back to the literal _CATEGORY_OF_SUBJECT tables (no
    regression). With ConceptNet present it is already primary, so this test
    forces the silent path by clearing the ontology."""
    eng, _ = _engine_with_conceptnet(primary=False)
    eng._cn_ontology = None  # force the literal-dict fallback path
    assert eng._is_category_error("what color is tuesday") == "color"
    assert eng._is_category_error("what color is the sun") is None
    # The literal dict flags an abstract concept as unable to have a color.
    assert eng._is_category_error("what color is freedom") == "color"


def test_conceptnet_primary_flag_on_bypasses_literal_dicts():
    """With ConceptNet-primary ON, the literal _CATEGORY_OF_SUBJECT table is
    bypassed and ConceptNet's has_property() is the sole authority. The
    flag is only meaningful when ConceptNet is loaded; if it is absent the
    gate must not produce false positives (silent -> allow)."""
    eng, loaded = _engine_with_conceptnet(primary=True)
    if not loaded:
        pytest.skip("ConceptNet ontology pickle not available")
    # ConceptNet still flags the genuine category error (time can't have color).
    assert eng._is_category_error("what color is tuesday") == "color"
    # ConceptNet affords physical properties on concrete subjects -> no flag,
    # even though the literal dict would have flagged book/tree as untasteable.
    assert eng._is_category_error("what color is the sun") is None
    assert eng._is_category_error("what does a book taste like") is None
    assert eng._is_category_error("what color is a tree") is None


def test_conceptnet_primary_absent_is_safe():
    """If ConceptNet is unavailable AND primary is ON, the gate must not fall
    back to the literal dict (that is the point of M6) — it returns None
    (allow) rather than risk a false category error."""
    eng = _bare_engine()
    eng.use_conceptnet_primary = True
    eng._cn_ontology = None  # simulate missing ontology
    eng._CATEGORY_OF_SUBJECT = {"tuesday": "time"}
    assert eng._is_category_error("what color is tuesday") is None


def test_category_label_of_spells_out_category():
    """_category_label_of maps a known subject to a human-readable category
    label for the honest reply, and falls back gracefully on the primary path.
    """
    eng = _bare_engine()
    eng.use_conceptnet_primary = False
    eng._CATEGORY_OF_SUBJECT = getattr(CognitiveChatEngine, "_CATEGORY_OF_SUBJECT", {})
    assert "time" in eng._category_label_of("tuesday")
    # On the primary path for an unseen subject, a generic label is returned.
    eng.use_conceptnet_primary = True
    assert eng._category_label_of("flibbertigibbet") == "that kind of thing"


def test_sense_biasing_framing_adds_domain_hint():
    eng = _bare_engine()
    # Interpersonal framing -> psychology hint.
    assert "psychology" in eng._sense_biasing_framing(
        "why do people trust each other", "trust")
    # Legal/financial framing -> finance hint.
    assert "finance" in eng._sense_biasing_framing(
        "how does a trust estate work", "trust")
    # Plain definitional query -> bare subject, no spurious hint.
    assert eng._sense_biasing_framing("what is trust", "trust") == "trust"


def test_context_query_vector_blends_context():
    """Context vector should differ from the bare subject when context words
    carry sense (verifies it isn't just the bare noun).

    FIX (flaky-order bug): the original used ``abs(hash(word))`` for its
    "deterministic" pseudo-embedding. Python salts ``str.__hash__`` per process
    (PYTHONHASHSEED), so the same test produced different vectors on different
    launches — passing in isolation but failing when the suite's process seed
    happened to land on a degenerate case. We replace it with a STABLE
    hashlib-based embedding (seed-independent) and snapshot STOP_WORDS so no
    shared module state can leak in either. The assertion is unchanged in
    intent: context words must nudge the subject vector.
    """
    eng = _bare_engine()
    dim = 16

    def _glove(word):
        # Seed-independent deterministic pseudo-embedding.
        h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % 1000
        v = np.array([(h + i * 7) % dim for i in range(dim)], dtype=float)
        return v / (np.linalg.norm(v) + 1e-9)
    eng._glove_vector = _glove

    # Snapshot STOP_WORDS so a preceding test's mutation cannot change which
    # context words survive filtering in _context_query_vector.
    import ravana.chat.web_learning as _wl
    _orig_stop = set(_wl.STOP_WORDS)
    _wl.STOP_WORDS = set(_orig_stop)
    try:
        bare = eng._context_query_vector("trust", "what is trust")
        contextual = eng._context_query_vector("trust", "why do people trust each other")
        assert bare is not None and contextual is not None
        # Vectors should be numpy arrays of the right dim.
        assert len(bare) == dim and len(contextual) == dim
        # Contextual vector differs from bare (context words nudged it).
        assert not np.allclose(bare, contextual)
    finally:
        _wl.STOP_WORDS = _orig_stop


def _build_causal_graph():
    g = ConceptGraph(dim=16)
    sun = g.add_node(label="sun")
    light = g.add_node(label="light")
    orbit = g.add_node(label="orbit")
    life = g.add_node(label="life")
    g.add_edge(sun.id, light.id, relation_type="causal", weight=0.8)
    g.add_edge(sun.id, orbit.id, relation_type="causal", weight=0.8)
    g.add_edge(light.id, life.id, relation_type="causal", weight=0.7)
    return g, {"sun": [sun.id], "light": [light.id], "orbit": [orbit.id], "life": [life.id]}


def test_causal_forward_simulate_discovers_consequences():
    eng = _bare_engine()
    g, kw = _build_causal_graph()
    eng.graph = g
    eng._concept_keywords = kw
    chains = eng._causal_forward_simulate("sun", max_steps=4, top_k=3)
    assert chains, "simulator should find causal consequences from 'sun'"
    # The chain should mention 'sun' at the start and a downstream concept.
    joined = " ".join(chains).lower()
    assert "sun" in joined
    assert ("light" in joined or "orbit" in joined or "life" in joined)


def test_causal_forward_simulate_open_ended_no_end_required():
    eng = _bare_engine()
    g, kw = _build_causal_graph()
    eng.graph = g
    eng._concept_keywords = kw
    # Unlike the old _causal_chain_search (which bailed without an end node),
    # this must return consequences without any target being specified.
    out = eng._causal_forward_simulate("sun")
    assert isinstance(out, list) and len(out) > 0


def test_causal_forward_simulate_empty_for_unknown():
    eng = _bare_engine()
    g, kw = _build_causal_graph()
    eng.graph = g
    eng._concept_keywords = kw
    assert eng._causal_forward_simulate("nonexistent_concept") == []


def _engine_with_sense_framing():
    """Bare engine with the sense-biasing helper + a no-op rewrite stub."""
    eng = _bare_engine()
    from ravana.chat.web_learning import WebLearningMixin
    eng._sense_biasing_framing = WebLearningMixin._sense_biasing_framing.__get__(eng)
    eng._rewrite_query_for_web = lambda q, s: q
    return eng


def test_web_query_variants_injects_sense_framing_for_context():
    eng = _engine_with_sense_framing()
    variants = eng._web_query_variants(
        "why do people trust each other", "trust", is_conditional=False)
    assert any("psychology" in v for v in variants), variants


def test_web_query_variants_no_bias_for_bare_subject():
    eng = _engine_with_sense_framing()
    variants = eng._web_query_variants(
        "what is trust", "trust", is_conditional=False)
    assert not any("psychology" in v for v in variants), variants
