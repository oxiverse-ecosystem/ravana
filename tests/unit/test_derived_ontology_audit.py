"""
Pin the derived (non-hand-listed) behaviors introduced in the de-hardcoding
audit:
  * _derive_definition_purge: closed-class seed + graph-derived abstract
    attractors (not a frozen 50-word list).
  * AbstractionEngine._walk_hierarchy: graph IS_A/level ascent is primary,
    ABSTRACTION_HIERARCHY is only a fallback seed.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ravana", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ravana_ml", "src"))

from ravana.chat.engine import CognitiveChatEngine
from ravana_ml.graph import ConceptGraph, ConceptNode
from ravana.core.abstraction_engine import AbstractionEngine, AbstractionConfig


def _bare_engine():
    eng = CognitiveChatEngine.__new__(CognitiveChatEngine)
    # Minimal graph so _derive_definition_purge has something to walk.
    g = ConceptGraph(dim=16)
    # abstract hub: high degree, high abstraction_degree
    hub = g.add_node(label="meaning", vector=np.zeros(16, dtype=np.float32))
    hub.abstraction_degree = 0.9
    hub.level = 3
    leaf = g.add_node(label="purpose", vector=np.zeros(16, dtype=np.float32))
    for _ in range(5):
        n = g.add_node(vector=np.zeros(16, dtype=np.float32))
        g.add_edge(hub.id, n.id, weight=0.5)
    g.add_edge(leaf.id, hub.id, weight=0.5)
    eng.graph = g
    return eng


def test_derive_definition_purge_includes_closed_class_and_derived_attractors():
    eng = _bare_engine()
    purge = eng._derive_definition_purge()
    # Closed-class universal seed present.
    assert "you" in purge and "i" in purge and "they" in purge
    # Derived abstract attractor ("meaning" is a high-degree, high-abstractness
    # hub) is included WITHOUT being hand-listed.
    assert "meaning" in purge
    # A low-degree concrete leaf is NOT purged.
    assert "purpose" not in purge


def test_definition_purge_falls_back_to_seed_without_graph():
    eng = CognitiveChatEngine.__new__(CognitiveChatEngine)
    eng.graph = None
    purge = eng._derive_definition_purge()
    assert "you" in purge
    assert "meaning" not in purge  # nothing derived without a graph


def test_walk_hierarchy_prefers_graph_over_seed():
    g = ConceptGraph(dim=16)
    child = g.add_node(label="cat", vector=np.zeros(16, dtype=np.float32))
    parent = g.add_node(label="animal", vector=np.zeros(16, dtype=np.float32))
    parent.level = 2
    g.add_edge(child.id, parent.id, relation_type="isa", weight=0.8)
    eng = AbstractionEngine(g, AbstractionConfig(max_hierarchy_depth=3))
    path = eng._walk_hierarchy("cat")
    # Graph edge is the primary path; "animal" is reached from the graph.
    assert "cat" in path and "animal" in path


def test_definition_attraction_purges_chronic_low_coherence():
    """Phase 1 (Track B): a concept that has collected 3+ landed definitions
    that are mostly NON-ASSERTED (no copula / defining verb) is purged by the
    LEARNED attraction score, not by a hardcoded word list. The signal is
    GloVe-independent (vmPFC/mPFC reality monitor: a memory that chronically
    fails to assert anything is tagged unreliable)."""
    eng = _bare_engine()
    eng._definitions = {
        # 3 non-asserted fragments => frac_asserted = 0 < 0.34 => purged.
        "life": ["to achieve the goals you set in life",
                 "the player who scores most wins",
                 "a vague abstraction nobody pins down"],
        # well-defined concept: only 2 defs => below the volume threshold,
        # must NOT be purged (also they assert, so they'd pass anyway).
        "quokka": ["a small macropod marsupial found in Western Australia",
                   "the quokka is a herbivorous mammal"],
    }
    purge = eng._derive_definition_purge()
    assert "life" in purge          # learned attractor (non-asserted junk)
    assert "quokka" not in purge    # too few defs to call it an attractor


def test_definition_attraction_keeps_asserted_concept():
    """A concept whose landed definitions DO assert something is NOT purged
    even at volume — only chronically non-asserted junk triggers the gate.
    This is the key difference from the old hardcoded blocklist, which would
    have blocked 'life' unconditionally regardless of definition quality."""
    eng = _bare_engine()
    # 3 asserted definitions => frac_asserted = 1.0 => NOT an attractor.
    eng._definitions = {
        "life": ["life is the condition that distinguishes organisms",
                 "life is a characteristic of living systems",
                 "life is studied by biology"],
    }
    purge = eng._derive_definition_purge()
    assert "life" not in purge      # asserted definitions => kept


def test_hardcoded_abstract_blocklist_removed():
    """Phase 1 (Track B): the frozen abstract-word list ('life/love/time/...')
    is gone from _DEFINITION_CONCEPT_BLOCKLIST; those concepts are now handled
    by the learned attraction score. Only closed-class pronouns remain."""
    from ravana.chat.web_learning import WebLearningMixin
    blocked = WebLearningMixin._DEFINITION_CONCEPT_BLOCKLIST
    for _w in ("life", "love", "time", "death", "god", "world", "meaning",
               "happiness", "science", "freedom"):
        assert _w not in blocked, f"hardcoded abstract word {_w!r} still blocked"
    # Closed-class pronouns (mirror of _UNIVERSAL_PURGE) must remain.
    assert "you" in blocked and "i" in blocked and "they" in blocked


# ── Track B Phase 2 (M4): learned snippet-quality model ─────────────────────

def test_snippet_model_rejects_garbage_admits_good():
    """The learned structural-PE model (predictive coding) rejects boilerplate
    / navigation / code snippets and admits well-formed definitions."""
    from ravana.chat.snippet_quality import default_model
    m = default_model()
    good = [
        "trust is a belief in the reliability of another person",
        "gravity is the force that attracts objects with mass",
    ]
    bad = [
        "get the latest news from our newsletter sign up now",
        "fan art by @user123 | deviantart",
        "click here to download the app and subscribe",
        "def my_func(x): return x * 2  # boilerplate code",
    ]
    for g in good:
        assert not m.is_junk(g), f"good snippet wrongly rejected: {g!r}"
    for b in bad:
        assert m.is_junk(b), f"garbage snippet not caught: {b!r}"


def test_snippet_gate_backstop_with_flag_off():
    """With the learned model OFF (default), the engine's snippet gate still
    rejects known junk via the old hardcoded tables (no regression)."""
    eng = CognitiveChatEngine.__new__(CognitiveChatEngine)
    eng.use_cerebellar_snippet = False
    # Old-pattern junk (matches _SNIPPET_REJECT_SHAPES / _SNIPPET_NOISE).
    assert eng._snippet_is_structural_junk("get the latest news about this")
    assert eng._snippet_is_structural_junk("sign in to your account now")
    # A clean definition is NOT rejected.
    assert not eng._snippet_is_structural_junk(
        "trust is a belief in the reliability of another person")


def test_snippet_gate_learned_with_flag_on():
    """With the learned model ON, the engine's snippet gate additionally
    rejects structural garbage (boilerplate/code) that the old tables might
    miss, while still admitting good definitions."""
    eng = CognitiveChatEngine.__new__(CognitiveChatEngine)
    eng.use_cerebellar_snippet = True
    eng._snippet_model = None
    # Old table still works as backstop.
    assert eng._snippet_is_structural_junk("get the latest news about this")
    # Learned model catches boilerplate/code the old tables may not enumerate.
    assert eng._snippet_is_structural_junk(
        "click here to download the app and subscribe now")
    # Good definition admitted.
    assert not eng._snippet_is_structural_junk(
        "gravity is the force that attracts objects with mass")


# ── Track B Phase 3 (M5): learned per-domain source-trust ───────────────────

def test_source_trust_fallback_with_flag_off():
    """With source-trust OFF (default), the hardcoded allowlist is the
    backstop: a known encyclopedic domain is preferred, an unknown blog is
    neutral (not preferred). No regression vs the old constant."""
    eng = CognitiveChatEngine.__new__(CognitiveChatEngine)
    eng.use_source_trust = False
    eng._source_trust = {}
    assert eng._is_preferred_source("https://en.wikipedia.org/wiki/Trust")
    assert not eng._is_preferred_source("https://some-random-blog.net/post/9")


def test_source_trust_learned_promotes_and_demotes():
    """With source-trust ON, accepting snippets from a domain raises its trust
    above the 0.5 threshold (preferred), while rejecting snippets from a junk
    domain drives it to 0.0 (not preferred). Learned, not hardcoded."""
    eng = CognitiveChatEngine.__new__(CognitiveChatEngine)
    eng.use_source_trust = True
    eng._source_trust = {}
    # Wikipedia: accept 3 snippets -> +0.1 each -> 0.5 + 0.3 = 0.8 > 0.5.
    for _ in range(3):
        eng._record_source_outcome("https://en.wikipedia.org/wiki/Trust", True)
    assert eng._domain_trust("https://en.wikipedia.org/wiki/Trust") == pytest.approx(0.8)
    assert eng._is_preferred_source("https://en.wikipedia.org/wiki/Trust")
    # Junk domain: reject 3 snippets -> -0.2 each -> 0.5 - 0.6 = 0.0 (clamped).
    for _ in range(3):
        eng._record_source_outcome("https://spam-seo.net/x", False)
    assert eng._domain_trust("https://spam-seo.net/x") == 0.0
    assert not eng._is_preferred_source("https://spam-seo.net/x")
    # Untried domain stays neutral (0.5), not preferred.
    assert eng._domain_trust("https://brand-new-site.org/y") == 0.5
    assert not eng._is_preferred_source("https://brand-new-site.org/y")


def test_source_trust_clamps_and_survives_sleep():
    """Trust is clamped to [0, 1]; a sleep-surviving accept earns +0.2 (extra
    reward) vs +0.1 for a plain accept."""
    eng = CognitiveChatEngine.__new__(CognitiveChatEngine)
    eng.use_source_trust = True
    eng._source_trust = {}
    # Max out at 1.0 even with many accepts.
    for _ in range(10):
        eng._record_source_outcome("https://great.gov/fact", True,
                                    survived_sleep=True)
    assert eng._domain_trust("https://great.gov/fact") == 1.0
    # Floor at 0.0 even with many rejects.
    for _ in range(10):
        eng._record_source_outcome("https://bad.net/x", False)
    assert eng._domain_trust("https://bad.net/x") == 0.0
    # A single sleep-surviving accept from an untried domain = 0.5 + 0.2 = 0.7.
    eng2 = CognitiveChatEngine.__new__(CognitiveChatEngine)
    eng2.use_source_trust = True
    eng2._source_trust = {}
    eng2._record_source_outcome("https://learned.edu/p", True,
                                 survived_sleep=True)
    assert eng2._domain_trust("https://learned.edu/p") == pytest.approx(0.7)


def test_walk_hierarchy_seed_fallback_when_no_graph_edges():
    g = ConceptGraph(dim=16)
    g.add_node(label="justice", vector=np.zeros(16, dtype=np.float32))
    eng = AbstractionEngine(g, AbstractionConfig(max_hierarchy_depth=3))
    path = eng._walk_hierarchy("justice")
    # No graph hierarchy -> seeded fallback supplies superordinates.
    assert "justice" in path
    assert any(p != "justice" for p in path)
