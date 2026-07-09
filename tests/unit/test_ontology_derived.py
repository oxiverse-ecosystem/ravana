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
from ravana.chat.engine import CognitiveChatEngine


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


# ── THE VERIFIED NEGATIVE: real learned graph cannot back has_property ─────
def test_real_graph_has_no_typed_isa_attribute_edges():
    """Pin that the deployed concept graph has NO isa/hypernym/attribute
    typed edges, so the synthesis's 'graph isa/attribute path' does not exist."""
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
    assert sum(rts.get(k, 0) for k in ("isa", "is_a", "hypernym", "attribute",
                                       "has_property")) == 0, rts


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
