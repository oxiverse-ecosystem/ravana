"""RED->GREEN: the D4 word-salad limitation (round 2026-08-19T1026Z).

The Situation-Model free-decoder emitted word salad for a subject RAVANA
has NO durable knowledge of ("tired" — no definition, no web source, not in
the concept graph). The grounding gate (`_sm_response_grounded`) Step-1
accepted it because the QUERY-DERIVED associated words ('ever','lot','really')
are all GloVe-similar (>=0.30) to the subject, and the decoder then restates
those same associates. That conflates "words near the subject in the query"
with "RAVANA knows something about the subject".

This is the exact false-negative the round logged as D4: a subject with no
durable knowledge must NEVER ground a free-decode reply. Mirror the brain's
source-monitoring (Johnson 1993) + the decomposition path's own D2 guard:
an UNKNOWN subject is withheld, and loose association-similarity may only
LEAN on spreading activation for concepts that are ALREADY in the concept
graph (a known concept RAVANA has actually learned).

Run:
    python -m pytest tests/unit/test_sm_unknown_subject_grounding.py -v
"""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana_ml", "src"))

from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.models import CognitiveResponseContext


def _build_engine():
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               data_dir=r"/tmp/ravana_d4_unknown_subj")


def _ctx(subject, assoc):
    return CognitiveResponseContext(
        subject=subject, raw_input=f"tell me about {subject}",
        associated_concepts=[(a, 0.5) for a in assoc],
    )


# ── The literal D4 failure class: an UNKNOWN subject, whose only "associations"
#    are the query's own near-neighbours, must NOT ground free-decode output.
#    The decoder text restates those same associates via glue verbs. ─────────
def test_unknown_subject_free_decode_salad_is_ungrounded():
    eng = _build_engine()
    # 'tired' is not in the concept graph, has no definition, no web source.
    # Its associated_concepts are query-derived neighbours ('ever','lot',...),
    # ALL of which are GloVe-similar to 'tired' (sim >= 0.30).
    subj = "tired"
    assoc = ["ever", "never", "lot", "really", "everyone", "hard"]
    # Sanity: confirm it really is unknown durable knowledge.
    assert subj not in getattr(eng, "_definitions", {})
    assert subj not in getattr(eng, "_concept_sources", {})
    assert subj not in getattr(eng, "_concept_keywords", {})
    text = ("It appears that tired is akin to ever. It ties to lot. "
            "And it ultimately has a relationship with really.")
    ctx = _ctx(subj, assoc)
    assert eng._sm_response_grounded(ctx, text) is False


# ── The OLD (buggy) Step-1 would have passed the above because query assoc
#    words are GloVe-similar. Guard against that exact regression: even with a
#    subject that HAS a glove vector and sim>=0.30 associates, an unknown
#    subject must be withheld (no durable anchor). ────────────────────────────
def test_unknown_subject_with_similar_associates_still_ungrounded():
    eng = _build_engine()
    subj = "perplexed"
    assoc = ["confused", "baffled", "unsure", "lost", "wondering"]
    assert subj not in getattr(eng, "_concept_keywords", {})
    assert subj not in getattr(eng, "_definitions", {})
    text = ("perplexed relates to confused. it connects to baffled and "
            "it ultimately ties to unsure.")
    ctx = _ctx(subj, assoc)
    assert eng._sm_response_grounded(ctx, text) is False


# ── Control: a KNOWN concept (in the concept graph) may still lean on
#    association spreading — the fix must NOT over-suppress real knowledge.
#    'gravity' has a seeded definition; 'oxiverse' is bootstrapped into the
#    concept graph. Both must still ground a genuine, topical answer. ────────
def test_known_subject_with_definition_still_grounds():
    eng = _build_engine()
    assert "gravity" in getattr(eng, "_definitions", {})
    text = "gravity is a force that pulls objects toward each other on earth"
    ctx = _ctx("gravity", ["force", "earth", "mass", "pull"])
    assert eng._sm_response_grounded(ctx, text) is True


def test_known_graph_concept_leans_on_association():
    eng = _build_engine()
    # 'oxiverse' is bootstrapped into the concept graph (DOMAIN_CONCEPTS).
    assert "oxiverse" in getattr(eng, "_concept_keywords", {})
    text = "oxiverse builds a privacy-first ecosystem as an alternative to big tech"
    ctx = _ctx("oxiverse", ["privacy", "ecosystem", "big tech"])
    assert eng._sm_response_grounded(ctx, text) is True


# ── Integration: the SM free-decode path must NEVER emit word salad for an
#    UNKNOWN subject (the D4 limitation). The internal grounding gate inside
#    `_generate_with_situation_model` must withhold any free-decode output
#    whose subject is not in RAVANA's durable knowledge, so the path falls
#    back to honest uncertainty instead of emitting the salad. We exercise the
#    REAL gate (not a mock that bypasses it) and assert the invariant: for an
#    unknown subject, whatever the SM path produces must itself pass the
#    grounding monitor (i.e. it cannot be free-decode salad about a subject
#    RAVANA has never learned). ──────────────────────────────────────────────
def test_d4_unknown_subject_sm_path_does_not_emit_salad():
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="d4e2e",
                              data_dir=r"/tmp/ravana_d4_e2e")
    subj = "tired"
    # Confirm the subject really is unknown durable knowledge (the D4 class).
    assert subj not in getattr(eng, "_concept_keywords", {})
    assert subj not in getattr(eng, "_definitions", {})
    assert subj not in getattr(eng, "_concept_sources", {})
    ctx = CognitiveResponseContext(
        subject=subj, raw_input="do you ever get tired of answering questions?",
        associated_concepts=[("ever", 0.5), ("lot", 0.5), ("really", 0.5)],
    )
    # Build the situation vector the SM path expects so it actually attempts
    # decoding (otherwise it short-circuits to None and the test is vacuous).
    import numpy as np
    ctx.situation_vector = np.zeros(64, dtype=np.float32)
    ctx.situation_narrative = {"active_concepts": [("ever", 0.5)], "coherence": 0.5, "diversity": 0.5}

    sm_res = eng._generate_with_situation_model(ctx)
    if sm_res is None:
        # Correctly withheld (no grounded output for an unknown subject).
        return
    text, strat = sm_res
    # If it DID produce text, the internal gate must certify it as grounded
    # (which for an unknown subject only happens if it references durable
    # knowledge — impossible for 'tired'). free-decode salad must not slip.
    assert eng._sm_response_grounded(ctx, text) is True, (
        f"SM path emitted ungrounded text for unknown subject '{subj}': {text!r}")
    assert "it ties to" not in text.lower()
    assert "it ultimately has a relationship with" not in text.lower()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
