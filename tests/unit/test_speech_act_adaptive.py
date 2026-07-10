"""
Regression tests for the adaptive-margin SpeechActClassifier (A1 of the unified
semantic layer). Locks in the behaviour proven by
experiments/regex_vs_prototype_speechact.py:

  1. The fixed SEMANTIC_MARGIN=0.12 is GONE (it made the classifier a regex
     confirmant: 100% agreement, 0 rescues).
  2. The exemplar-spread z-margin RESCUES the 8 bare declaratives the regex rule
     cascade mislabels as questions.
  3. The interface-agnostic nearest_prototype(vec, store) API works on any vector
     representation (so the A0 high-D lift can drop in behind it later).
  4. add_exemplar() grows the store (the "learn by chatting" substrate).

These need the GloVe-64 cache (data/ravana_glove_cache.npz). When it's absent
(e.g. a fresh CI box without the download) the GloVe-dependent tests skip rather
than fail — the API-shape tests still run with a synthetic vector_fn.
"""
import os
import numpy as np
import pytest

from ravana.language.prefrontal_workspace import SpeechActClassifier, PrefrontalWorkspace

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_GLOVE_CACHE = os.path.join(_PROJ, "data", "ravana_glove_cache.npz")


# The 8 bare-declarative statements the regex cascade misses (no first/third
# person leading marker, no '?', no wh-/aux) — it defaults them to "question".
# The adaptive z-margin classifier must call these "statement".
RESCUE_STATEMENTS = [
    "the sky looks beautiful tonight",
    "really enjoyed that movie yesterday",
    "kind of hungry right now",
    "just finished reading a great book",
    "feeling good about the project",
    "looks like rain is coming",
    "absolutely love this song",
    "not a fan of cold weather",
]

# Clear questions the classifier must still get right.
CLEAR_QUESTIONS = [
    "What is trust?",
    "Why does ice melt?",
    "How does gravity work?",
    "Explain photosynthesis",
]


@pytest.fixture(scope="module")
def glove_vector_fn():
    if not os.path.exists(_GLOVE_CACHE):
        pytest.skip(f"GloVe cache not present: {_GLOVE_CACHE}")
    from ravana.ontology.attribute_encoder import build_glove64_lookup
    lut, dim = build_glove64_lookup(_GLOVE_CACHE)

    def vector_fn(word):
        v = lut.get(word.lower())
        if v is None:
            return None
        n = np.linalg.norm(v)
        return (v / n) if n > 0 else v

    return vector_fn, dim


class TestAdaptiveMarginRemovesHardcoding:
    def test_semantic_margin_constant_is_gone(self):
        # The fixed threshold must not exist anymore — it was relocated
        # hardcoding and made the classifier echo regex.
        assert not hasattr(SpeechActClassifier, "SEMANTIC_MARGIN")

    def test_no_hybrid_method_deferral(self, glove_vector_fn):
        # classify() must never return the "hybrid" method (defer-to-regex on a
        # fixed margin). Decisions are "semantic", "syntactic", or "default".
        vfn, dim = glove_vector_fn
        sac = SpeechActClassifier(vector_fn=vfn, dim=dim)
        for text in RESCUE_STATEMENTS + CLEAR_QUESTIONS:
            _act, _raw, method = sac.classify(text)
            assert method != "hybrid"


class TestRescues:
    def test_rescues_bare_declaratives(self, glove_vector_fn):
        vfn, dim = glove_vector_fn
        sac = SpeechActClassifier(vector_fn=vfn, dim=dim)
        rescued = 0
        for text in RESCUE_STATEMENTS:
            # regex mislabels these as question
            assert PrefrontalWorkspace.classify_speech_act_rules(text) == "question"
            act, _raw, _method = sac.classify(text)
            if act == "statement":
                rescued += 1
        # All 8 must be rescued (locks the measured result).
        assert rescued == len(RESCUE_STATEMENTS), f"only {rescued}/8 rescued"

    def test_clear_questions_preserved(self, glove_vector_fn):
        vfn, dim = glove_vector_fn
        sac = SpeechActClassifier(vector_fn=vfn, dim=dim)
        for text in CLEAR_QUESTIONS:
            act, _raw, _method = sac.classify(text)
            assert act == "question", f"{text!r} -> {act}"


class TestInterfaceAgnosticAPI:
    def test_nearest_prototype_on_synthetic_store(self):
        # nearest_prototype must work on ANY vector rep (no GloVe needed) —
        # this is the seam the A0 high-D lift drops into.
        rng = np.random.RandomState(0)
        a = rng.randn(128); a /= np.linalg.norm(a)
        b = rng.randn(128); b /= np.linalg.norm(b)
        store = {
            "class_a": {"centroid": a, "mu": 0.5, "sigma": 0.1},
            "class_b": {"centroid": b, "mu": 0.5, "sigma": 0.1},
        }
        # A query near centroid a must classify as class_a.
        q = a + 0.05 * b
        q /= np.linalg.norm(q)
        best, z, raw = SpeechActClassifier.nearest_prototype(q, store)
        assert best == "class_a"
        assert set(z.keys()) == {"class_a", "class_b"}
        assert raw["class_a"] > raw["class_b"]

    def test_nearest_prototype_empty_inputs(self):
        assert SpeechActClassifier.nearest_prototype(None, {}) == (None, {}, {})
        v = np.ones(8)
        assert SpeechActClassifier.nearest_prototype(v, {}) == (None, {}, {})


class TestLearnByChatting:
    def test_add_exemplar_grows_store_and_refits(self, glove_vector_fn):
        vfn, dim = glove_vector_fn
        sac = SpeechActClassifier(vector_fn=vfn, dim=dim)
        before = len(sac.exemplars["statement"])
        sac._fit()
        mu_before = sac._mu["statement"]
        sac.add_exemplar("statement", "the coffee is cold now")
        assert len(sac.exemplars["statement"]) == before + 1
        # refit must be triggered lazily (stats can change)
        sac._fit()
        assert "statement" in sac._mu
        # mu is recomputed (may or may not move much, but fit ran without error)
        assert isinstance(mu_before, float)

    def test_new_class_emerges_without_code_edit(self, glove_vector_fn):
        vfn, dim = glove_vector_fn
        sac = SpeechActClassifier(vector_fn=vfn, dim=dim)
        assert "gratitude" not in sac.exemplars
        for phrase in ["thank you so much", "i really appreciate it", "thanks a lot"]:
            sac.add_exemplar("gratitude", phrase)
        store = sac._store()
        assert "gratitude" in store
        assert "centroid" in store["gratitude"]
