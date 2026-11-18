"""Regression: no associative filler when semantic retrieval fails.

The dominant measured defect (15/203 replies, scratch/front_probes_n30.jsonl)
was the fixed template::

    the closest i get to earthquakes is rains and maybe environments -- and
    from what i've seen, water that falls from the sky as drops of liquid.

It echoes the subject, chains unrelated GloVe neighbours, and asserts a
definition of a *neighbour* as if it answered the question about the subject.

Brain basis (docs/REFLECTIVE_GAP_CALIBRATION.md): FOK tracks accessibility, not
correctness (Koriat 1993, Psychological Review 100(4):609-639); automatic
associative spillover is not controlled, cue-bound retrieval (Badre & Wagner
2002, Behav Cogn Neurosci Rev 1(3):206-218; Buckner et al. 2008, Ann NY Acad Sci
1124:1-38); a recognised, unfillable information gap is answered with calibrated
uncertainty and curiosity (Loewenstein 1994, Psychological Bulletin 116(1):75-98).
"""
import os
import sys
import random
import re

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(_REPO, "ravana", "src"),
           os.path.join(_REPO, "ravana_ml", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from ravana.chat.response_gen import ResponseGenMixin  # noqa: E402
from ravana.chat.models import CognitiveResponseContext  # noqa: E402


# Same detector the front-measurement harness uses to reject this failure mode.
DEGENERATE = re.compile(r"the closest i get to|and maybe|from what i've seen", re.I)


class _StubHost(ResponseGenMixin):
    """Minimal host: deterministic embeddings, canned KB, no GloVe load/save."""

    def __init__(self, kb=None):
        self._definitions = {}
        self._metrics = {}
        self._style_context = None
        self._reg_verbosity = 1.0
        self._concept_pos = {}
        self._kb = kb or {}
        # Deliberately no self._sentence_grounded -- mirrors the real engine,
        # where that name is a nested local function, not a mixin method.

    def _capitalize_subject(self, subject, raw_input):
        return subject  # real engine preserves lower case for common nouns

    @staticmethod
    def _vec(w):
        import hashlib
        v = np.array([b / 255.0 for b in hashlib.sha256(w.encode()).digest()[:16]])
        return v / (np.linalg.norm(v) + 1e-9)

    def _glove_vector(self, w):
        base = self._vec("earthquakes")
        if w == "earthquakes":
            return base
        if w in ("rains", "cat", "reason"):
            return 0.6 * base + 0.8 * self._vec(w)
        if w in ("environments", "dog", "mind"):
            return 0.5 * base + 0.87 * self._vec(w)
        return self._vec(w)

    def _compute_phrase_embedding(self, w):
        return self._glove_vector(w)

    def _is_function_word(self, w):
        return False

    def _topic_set_gate(self, candidates, subject, min_coherence=0.20):
        filtered = []
        sv = self._glove_vector(subject)
        for label, score in candidates:
            if label.lower() in set(subject.lower().split()):
                continue
            if float(np.dot(sv, self._glove_vector(label))) >= min_coherence:
                filtered.append((label, score))
        return filtered or [(subject, 1.0)]

    def _assoc_cos_ok(self, label, subj_vec, floor=0.25):
        if subj_vec is None:
            return False
        return float(np.dot(subj_vec, self._glove_vector(label))) >= floor

    def kb_describe(self, concept):
        return self._kb.get(concept.lower())

    def describe_from_cn(self, concept):
        return None

    def _coherence_ok(self, text, ctx):
        return (True, "stub")

    def condition_modality(self, band, ctx):
        return band


def _ctx(raw, subj, assocs):
    return CognitiveResponseContext(
        subject=subj, raw_input=raw, associated_concepts=assocs)


ASSOCS = [("rains", 0.6), ("environments", 0.5)]


def test_negative_control_the_old_template_is_rejected_by_the_detector():
    """The detector must reject the exact measured degenerate output."""
    old = ("the closest i get to earthquakes is rains and maybe environments "
           "-- and from what i've seen, water that falls from the sky as drops "
           "of liquid.")
    assert DEGENERATE.search(old), "detector no longer catches the known failure"


def test_no_grounded_material_yields_calibrated_gap_not_associative_filler():
    """REGRESSION: weak neighbours must not be promoted into an assertion."""
    random.seed(0)
    host = _StubHost(kb={"rains": "Water that falls from the sky as drops of liquid."})
    text, strategy = host._reflective_response(
        _ctx("what causes earthquakes", "earthquakes", ASSOCS))

    assert not DEGENERATE.search(text), "degenerate template emitted: %r" % text
    assert "rain" not in text.lower(), "neighbour definition leaked: %r" % text
    assert "from the sky" not in text.lower(), "confabulated fact: %r" % text
    assert strategy == "metacognitive_uncertainty", strategy
    low = text.lower()
    assert any(k in low for k in ("not sure", "don't really have", "outside what",
                                  "new to me", "not totally sure",
                                  "not completely sure")), text
    # Curiosity turn-back: an information gap opens a question (Loewenstein 1994).
    # `_join_clauses` normalises terminal punctuation to '.', so check the
    # question lexically rather than by a '?'.
    assert any(k in low for k in ("what made you", "your take", "your sense",
                                  "hear your", "wonder about")), text


def test_partial_grounded_material_about_subject_is_used():
    """Positive control: if the snippet is genuinely about the subject, use it."""
    random.seed(0)
    fact = "Earthquakes are sudden releases of energy in the earth's crust."
    host = _StubHost(kb={"rains": fact})
    text, strategy = host._reflective_response(
        _ctx("what causes earthquakes", "earthquakes", ASSOCS))

    assert not DEGENERATE.search(text), "degenerate surface: %r" % text
    assert strategy == "reflective_grounded", strategy
    assert "earthquakes" in text.lower() and "energy" in text.lower(), text
    assert host._metrics.get("reflective_grounded") == 1


def test_subject_grounding_binding_is_structural_not_a_topic_list():
    """A neighbour's definition is not evidence; a subject mention is."""
    helper = ResponseGenMixin._subject_grounded_evidence
    unrelated = "Water that falls from the sky as drops of liquid."
    assert helper(unrelated, "earthquakes") == ""
    assert helper("Rains are drops of liquid water.", "rain") != ""  # plural variant
    about = "Earthquakes are sudden releases of energy."
    assert helper(about, "earthquakes") == about
    assert helper(about, "global warming") == ""
    assert helper("", "earthquakes") == ""
