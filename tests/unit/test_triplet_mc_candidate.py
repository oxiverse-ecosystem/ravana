"""Section 6.4: additive triplet-inference MC candidate.

Contract under test (plan reports/opencode_64_plan.md 3.3):
- flag OFF (default): _try_fact_reasoning path byte-identical (candidate
  never consulted)
- flag ON + cold profiles: candidate abstains (None) — Wilson gates closed
- flag ON + warmed 'is' profile: candidate answers the syllogism MC and
  abstains on ambiguity
- candidate never displaces an evidence-based handler (wired after them,
  before plausibility_choice only)
"""
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "..", "ravana", "src"))

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402
from ravana.core.triplet_inference import Triple  # noqa: E402

Q = ("All men are mortal. Socrates is a man. "
     "Options: A) socrates is mortal B) socrates is immortal")


@pytest.fixture(scope="module")
def eng():
    d = tempfile.mkdtemp()
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d)
    return e


def _warm_is(e, n=20):
    for i in range(n):
        e.triplet_op.ingest_triple(Triple(f"wa{i}", "is", f"wb{i}"))
        e.triplet_op.ingest_triple(Triple(f"wb{i}", "is", f"wc{i}"))
        e.triplet_op.ingest_triple(Triple(f"wa{i}", "is", f"wc{i}"))


def test_flag_default_off(eng):
    assert eng.use_triplet_candidate is False


def test_cold_profiles_abstain(eng):
    """Before warmup: every Wilson gate closed -> None."""
    eng.use_triplet_candidate = True
    try:
        assert eng._triplet_mc_answer(Q, []) is None
    finally:
        eng.use_triplet_candidate = False


def test_warm_profile_answers_syllogism(eng):
    _warm_is(eng)
    prof = eng.triplet_op.memory.profiles["is"]
    assert prof.transitivity_lower() > 0.5
    eng.use_triplet_candidate = True
    try:
        assert eng._triplet_mc_answer(Q, []) == "socrates is mortal"
    finally:
        eng.use_triplet_candidate = False


def test_ambiguous_options_abstain(eng):
    """Both options contain the conclusion words -> abstain."""
    q = ("All men are mortal. Socrates is a man. "
         "Options: A) socrates is mortal B) socrates the mortal man")
    eng.use_triplet_candidate = True
    try:
        assert eng._triplet_mc_answer(q, []) is None
    finally:
        eng.use_triplet_candidate = False


def test_cold_predicate_abstains_even_when_warm_elsewhere(eng):
    """'causes' has no learned evidence: gate closed regardless of 'is'."""
    q = ("Rain causes floods. Floods cause damage. "
         "Options: A) rain causes damage B) damage causes rain")
    eng.use_triplet_candidate = True
    try:
        assert eng._triplet_mc_answer(q, []) is None
    finally:
        eng.use_triplet_candidate = False


def test_non_mc_input_abstains(eng):
    eng.use_triplet_candidate = True
    try:
        assert eng._triplet_mc_answer(
            "All men are mortal. Socrates is a man.", []) is None
    finally:
        eng.use_triplet_candidate = False


def test_flag_off_never_consults_candidate(eng, monkeypatch):
    """With the flag OFF the wiring must not even call the method."""
    called = {"n": 0}

    def _spy(*a, **kw):
        called["n"] += 1
        return None

    monkeypatch.setattr(eng, "_triplet_mc_answer", _spy)
    eng.use_triplet_candidate = False
    eng._try_fact_reasoning(Q)
    assert called["n"] == 0


def test_flag_on_consults_candidate_after_evidence_handlers(
        eng, monkeypatch):
    """With the flag ON the wiring consults the candidate for MC input.

    NOTE: _try_fact_reasoning early-returns when the hippocampal buffer
    holds no fact texts (engine.py ~1994), so seed one first — matching
    real benchmark state, where the buffer is always populated.
    """
    called = {"n": 0}

    def _spy(*a, **kw):
        called["n"] += 1
        return None  # abstain -> plausibility fallback still runs

    eng.hippocampal_buffer.store(
        "socrates", "mentioned", "Socrates lived in Athens.")
    monkeypatch.setattr(eng, "_triplet_mc_answer", _spy)
    eng.use_triplet_candidate = True
    try:
        eng._try_fact_reasoning(Q)
    finally:
        eng.use_triplet_candidate = False
    assert called["n"] == 1
