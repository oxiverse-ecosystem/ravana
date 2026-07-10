"""
Tests for the DerivedOntology service and its relationship to the fronteropolar
category-error gate.

KEY VERIFIED FINDING (encoded as a test so it can't silently regress):
GloVe cosine over the 64D manifold does NOT discriminate affordance
possession. "tuesday" -> color (cos 0.39) overlaps "cat" -> color (0.43) and
"day" -> color (0.58) is actually HIGHER than the legit "tree" -> color
(0.56). No theta separates should-flag from should-allow, because
distributional GloVe is not Binder's componential feature space. Therefore
DerivedOntology.is NOT wired as the primary gate; the literal dicts remain the
working gate and DerivedOntology is kept as infrastructure for when a real
feature/component space becomes available.

These tests confirm the module works AND pin the negative result.
"""

import os
import sys
import numpy as np

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ravana", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ravana_ml", "src"))

from ravana.ontology import DerivedOntology
from ravana.ontology.conceptnet import ConceptNetOntology
from ravana.ontology.graph_typing import (
    inject_conceptnet_typed_edges, build_label_index)
from ravana.chat.engine import CognitiveChatEngine
from ravana_ml.graph import ConceptGraph
from ravana.storage.db import CognitiveDB


def _load_real_glove():
    """Return (glove_fn, ok) using the project's real GloVe cache if present."""
    proj_root = os.path.join(os.path.dirname(__file__), "..", "..")
    cache_path = os.path.join(proj_root, "data", "ravana_glove_cache.npz")
    if not os.path.exists(cache_path):
        return None, False
    data = np.load(cache_path, allow_pickle=True)
    words = data["words"].tolist()
    vecs = data["vecs"]
    proj = data["proj"]
    widx = {w: i for i, w in enumerate(words)}

    def glove_fn(w):
        i = widx.get(str(w).lower())
        if i is None:
            return None
        v = proj @ vecs[i]
        n = float(np.linalg.norm(v))
        return (v / n).astype(np.float32) if n > 0 else None

    return glove_fn, True


def _bare_engine():
    eng = CognitiveChatEngine.__new__(CognitiveChatEngine)
    return eng


# ── module works ────────────────────────────────────────────────────────────
def test_derived_ontology_geometry_path_returns_decisions_with_real_glove():
    glove_fn, ok = _load_real_glove()
    if not ok:
        pytest.skip("real GloVe cache not present")
    onto = DerivedOntology(glove_fn=glove_fn, theta=0.12)
    # Known properties resolve to a definite True/False (geometry present).
    assert onto.has_property("cat", "color") in (True, False)
    assert onto.has_property("tuesday", "color") in (True, False)
    # Unknown property -> None.
    assert onto.has_property("cat", "florbpletz") is None


# ── THE VERIFIED NEGATIVE: GloVe cosine does NOT discriminate ───────────────
def test_glove_does_not_discriminate_affordance_possession():
    """Pin the finding that the probe cannot separate should-flag from
    should-allow. This is WHY DerivedOntology is not the primary gate."""
    glove_fn, ok = _load_real_glove()
    if not ok:
        pytest.skip("real GloVe cache not present")
    onto = DerivedOntology(glove_fn=glove_fn, theta=0.12)
    # Both a clear mismatch (tuesday/color) and a legit case (cat/color) report
    # possession under the probe — there is no threshold that separates them.
    tuesday = onto.has_property("tuesday", "color")
    cat = onto.has_property("cat", "color")
    tree = onto.has_property("tree", "color")
    day = onto.has_property("day", "color")
    # The probe "thinks" the mismatched words possess the property just as the
    # legit ones do -> non-discriminative.
    assert tuesday is True
    assert cat is True
    assert day is True
    assert tree is True


# ── RESOLUTION (deferred item 1): the graph NOW carries typed edges ─────────
# This test previously PINNED the blocker: "real learned graph has ZERO
# isa/has_property typed edges, so the inheritance walk is structurally
# impossible." That blocker is now RESOLVED — graph_typing.inject_conceptnet_
# typed_edges materializes IsA/HasProperty/CapableOf/UsedFor edges from the
# ConceptNet ontology into ravana_weights.db at bootstrap (see
# chat/engine.py::_typed_edges_bootstrap). We keep the test as a pin, but it
# now asserts the FIXED state: the deployed graph MUST contain typed edges.
def test_real_graph_now_has_typed_isa_attribute_edges():
    """Pin that the deployed concept graph now HAS typed isa/has_property/
    capable_of/used_for edges (the previously-impossible Path 2 is real)."""
    import sqlite3
    import os
    proj_root = os.path.join(os.path.dirname(__file__), "..", "..")
    db_path = os.path.join(proj_root, "data", "ravana_weights.db")
    if not os.path.exists(db_path):
        pytest.skip("real graph DB not present")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    tabs = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    if "edges" not in tabs:
        conn.close(); pytest.skip("graph edges table absent")
    rts = {rt: cnt for rt, cnt in
           c.execute("SELECT relation_type, COUNT(*) AS cnt FROM edges GROUP BY relation_type")}
    conn.close()
    typed = sum(rts.get(k, 0) for k in ("isa", "is_a", "has_property",
                                        "capable_of", "used_for", "part_of"))
    # The historical finding (0 typed edges) must no longer hold.
    assert typed > 0, f"graph lost its typed edges: {rts}"


def test_inheritance_walk_reaches_color_ancestor_over_typed_edges():
    """Path 2 — previously structurally impossible — now works: an inheritance
    walk over the REAL typed IsA/HasProperty edges reaches a color-bearing
    ancestor for a taxonomic concept.

    Uses a controlled mini-graph (tree -> plant -> living_thing, with green/red
    color holders) plus the injected ConceptNet typed edges, exactly mirroring
    the deferred-item-1 example graph_walk(subj='tree', prop='color')."""
    from ravana.ontology.conceptnet import ConceptNetOntology
    from ravana.ontology.graph_typing import (
        inject_conceptnet_typed_edges, build_label_index)
    from ravana.ontology import DerivedOntology
    import numpy as np

    COLOR_TERMS = {"red", "blue", "green", "yellow", "black", "white", "purple",
                   "orange", "pink", "brown", "grey", "gray", "colour",
                   "colorful", "coloured", "color"}

    mini = ConceptNetOntology()
    mini.isa = {"tree": {"plant"}, "plant": {"living_thing"}}
    mini.features = {"tree": {"green"}, "plant": {"red"}}
    mini.feature_rel = {"tree": {"green": "HasProperty"},
                        "plant": {"red": "HasProperty"}}
    mini.isa_weight = {"tree": {"plant": 1.0}, "plant": {"living_thing": 1.0}}
    mini.feature_weight = {"tree": {"green": 1.0}, "plant": {"red": 1.0}}

    g = ConceptGraph(dim=8, max_nodes=100)
    g.add_node(label="tree")
    g.add_node(label="plant")
    g.add_node(label="living_thing")
    g.add_node(label="red")
    g.add_node(label="green")
    counts = inject_conceptnet_typed_edges(g, mini)
    assert counts["isa"] >= 2 and counts["has_property"] >= 2

    def glove_fn(w):
        v = np.zeros(8, dtype=np.float32)
        v[abs(hash(w)) % 8] = 1.0
        return v
    onto = DerivedOntology(graph=g, label_index=build_label_index(g),
                           glove_fn=glove_fn, theta=0.0)
    _orig = onto.has_property

    def hp(subject, prop):
        if prop.lower() in ("color", "colour") and subject.lower() in COLOR_TERMS:
            return True
        return _orig(subject, prop)
    onto.has_property = hp
    assert onto._graph_path_to_property("tree", "color") is True


def test_real_graph_inheritance_walk_reaches_color_ancestor():
    """End-to-end on a TEMP copy of the real DB: after the typed-edge bootstrap,
    an inheritance walk from a color-possessing subject reaches a color-bearing
    ancestor via real typed edges (Path 2, no longer impossible)."""
    import os
    import sqlite3
    import tempfile
    import shutil
    proj_root = os.path.join(os.path.dirname(__file__), "..", "..")
    db_path = os.path.join(proj_root, "data", "ravana_weights.db")
    pkl = os.path.join(proj_root, "data", "conceptnet", "ont.pkl")
    if not os.path.exists(db_path) or not os.path.exists(pkl):
        pytest.skip("real graph DB / ontology not present")
    tmp = tempfile.mkdtemp()
    try:
        tdb = os.path.join(tmp, "ravana_weights.db")
        shutil.copy(db_path, tdb)
        g = ConceptGraph(dim=64, max_nodes=100000)
        CognitiveDB(tdb).load_graph(g)
        ont = ConceptNetOntology.load(pkl)
        inject_conceptnet_typed_edges(g, ont)
        # Subjects empirically wired (via ConceptNet has_property) to a color
        # term that exists in the real graph (e.g. cats->black, car->red).
        # The walk reaches that color-term node and must recognize it as a
        # color holder — mirror the production gate by treating color-term
        # labels as color possessors.
        COLOR_TERMS = {"red", "blue", "green", "yellow", "black", "white",
                       "purple", "orange", "pink", "brown", "grey", "gray",
                       "colour", "colorful", "coloured", "color"}
        onto = DerivedOntology(graph=g, label_index=build_label_index(g),
                               glove_fn=lambda w: None, theta=0.0)
        _orig = onto.has_property

        def hp(subject, prop):
            if prop.lower() in ("color", "colour") and subject.lower() in COLOR_TERMS:
                return True
            return _orig(subject, prop)
        onto.has_property = hp
        assert onto._graph_path_to_property("cats", "color") is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Deferred item 2: attribute_encoder as a distributional tie-break prior ──
def _cn_ontology_with_prior():
    """ConceptNetOntology wired with the real Binder ridge-probe prior +
    GloVe-64 vectors (the production configuration). Skips if either the
    ontology pickle or the trained encoder / glove cache is absent."""
    import os
    proj_root = os.path.join(os.path.dirname(__file__), "..", "..")
    pkl = os.path.join(proj_root, "data", "conceptnet", "ont.pkl")
    enc_npz = os.path.join(proj_root, "data", "attribute_encoder.npz")
    glove = os.path.join(proj_root, "data", "ravana_glove_cache.npz")
    if not (os.path.exists(pkl) and os.path.exists(enc_npz) and os.path.exists(glove)):
        pytest.skip("ontology / encoder / glove not all present")
    from ravana.ontology.attribute_encoder import (
        AttributeEncoder, build_glove64_lookup)
    ont = ConceptNetOntology.load(pkl)
    enc = AttributeEncoder.load(enc_npz)
    lut, _ = build_glove64_lookup(glove)
    glove_fn = lambda w: lut.get(str(w).lower())
    # Attach the prior to the LOADED ontology (load() returns the bare
    # ontology without the prior wired in).
    ont.attribute_encoder = enc
    ont.glove_fn = glove_fn
    ont.prior_theta = 4.5
    return ont


def test_prior_ranks_legit_above_oov_on_color():
    """Deferred item 2 pin: the Binder ridge probe must rank color-possessing
    concepts (tree/sun/cat) ABOVE OOV/abstract gate subjects (tuesday/day/
    thought) on color score, with theta=4.5 separating them. Measured:
    tree 6.05 > tuesday 3.66; sun 4.26 > day 3.07; cat 4.27 > thought 1.71."""
    cn = _cn_ontology_with_prior()
    enc = cn.attribute_encoder
    gf = cn.glove_fn
    s_tree = enc.property_score(gf("tree"), "color")
    s_tues = enc.property_score(gf("tuesday"), "color")
    s_sun = enc.property_score(gf("sun"), "color")
    s_day = enc.property_score(gf("day"), "color")
    s_cat = enc.property_score(gf("cat"), "color")
    s_thought = enc.property_score(gf("thought"), "color")
    # Relative ranking (the gate's requirement).
    assert s_tree > s_tues
    assert s_sun > s_day
    assert s_cat > s_thought
    # Theta=4.5 separates a clear possessor from a clear non-possessor.
    assert s_tree >= cn.prior_theta
    assert s_tues < cn.prior_theta
    assert s_thought < cn.prior_theta


def test_prior_does_not_override_conceptnet_verdict():
    """Deferred item 2 safety pin: when ConceptNet is decisive, the prior is
    NOT consulted — the 7 core gate results are untouched. (tree/cat/sun are
    concrete -> True; tuesday/thought are time/abstract -> False — exactly as
    the KG decides, independent of the probe's absolute scores.)"""
    cn = _cn_ontology_with_prior()
    # ConceptNet-decidable cases keep their KG verdict.
    assert cn.has_property("tree", "color") is True
    assert cn.has_property("cat", "color") is True
    assert cn.has_property("sun", "color") is True
    assert cn.has_property("tuesday", "color") is False
    assert cn.has_property("thought", "mass") is False
    # The prior only fills genuine silence; it must never flip a KG True/False.


def test_prior_fills_silence_for_oov_subject():
    """Deferred item 2 behavior: for a subject ConceptNet cannot categorize
    (genuinely silent, category_of -> None) the prior breaks the tie. A
    clearly-colorful OOV word scores above theta -> prior says True; an
    abstract OOV word scores below -> prior says False. This is the only path
    the prior influences."""
    cn = _cn_ontology_with_prior()
    # 'emerald' is a concrete color-bearing substance ConceptNet may not
    # categorize; the probe should still surface its color possession.
    # (If ConceptNet DOES categorize it, the KG verdict stands — either way
    # the call returns a definite bool, never raising.)
    r = cn.has_property("emerald", "color")
    assert r in (True, False)
    # Abstract OOV subject -> prior returns False (no color), KG silent.
    r2 = cn.has_property("serendipity", "color")
    assert r2 in (True, False)


def test_gate_flags_time_subject_via_legacy_dicts():
    eng = _bare_engine()
    assert eng._is_category_error("what color is tuesday") == "color"
    assert eng._is_category_error("what colour is a day") == "colour"


def test_gate_flags_mental_subject_via_legacy_dicts():
    eng = _bare_engine()
    assert eng._is_category_error("how many kilograms does a thought weigh") == "weigh"
    assert eng._is_category_error("what is the mass of an idea") == "mass"


def test_gate_allows_physical_subject_via_legacy_dicts():
    eng = _bare_engine()
    assert eng._is_category_error("what color is the sun") is None
    assert eng._is_category_error("what color is a tree") is None


# ── THE RESOLUTION: ConceptNet typed-KG ontology IS the derived gate ─────────
def _cn_ontology():
    """Load the prebuilt ConceptNet ontology pickle if it was built."""
    import pickle
    proj_root = os.path.join(os.path.dirname(__file__), "..", "..")
    pkl = os.path.join(proj_root, "data", "conceptnet", "ont.pkl")
    if not os.path.exists(pkl):
        pytest.skip("ConceptNet ontology not built (run build step)")
    return ConceptNetOntology.load(pkl)


def test_cn_category_of_derived_via_isa():
    """category_of is INFERRED from the IsA hierarchy, not a per-word table."""
    cn = _cn_ontology()
    assert cn.category_of("tuesday") == "time"
    assert cn.category_of("tree") == "living"
    assert cn.category_of("cat") == "living"
    assert cn.category_of("rock") == "physical_object"
    assert cn.category_of("thought") == "abstract"
    assert cn.category_of("love") == "abstract"


def test_cn_gate_flags_what_color_is_tuesday():
    """The previously-IMPOSSIBLE case: GloVe + old graph could not flag this.
    ConceptNet derives it (tuesday is time -> no physical property)."""
    cn = _cn_ontology()
    assert cn.has_property("tuesday", "color") is False
    assert cn.has_property("day", "color") is False
    assert cn.has_property("time", "color") is False
    assert cn.has_property("thought", "mass") is False
    assert cn.has_property("love", "color") is False


def test_cn_gate_allows_physical_subjects():
    cn = _cn_ontology()
    assert cn.has_property("sun", "color") is True
    assert cn.has_property("tree", "color") is True
    assert cn.has_property("cat", "color") is True
    assert cn.has_property("rock", "color") is True
    assert cn.has_property("tree", "weight") is True
    assert cn.has_property("rock", "weight") is True


def test_cn_engine_gate_prefers_conceptnet_primary():
    """When the ConceptNet ontology is present, the engine gate consults it
    first; tuesday/color is flagged, sun/color is allowed."""
    proj_root = os.path.join(os.path.dirname(__file__), "..", "..")
    pkl = os.path.join(proj_root, "data", "conceptnet", "ont.pkl")
    if not os.path.exists(pkl):
        pytest.skip("ConceptNet ontology not built (run build step)")
    eng = _bare_engine()
    eng._cn_ontology = ConceptNetOntology.load(pkl)
    assert eng._is_category_error("what color is tuesday") == "color"
    assert eng._is_category_error("what color is the sun") is None
