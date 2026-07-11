"""Direct tests for the Situation-Model grounding gate (Levelt monitor).

These exercise ResponseGenerator._sm_response_grounded and the SM dispatch
in _generate_response. They prove the fix resolves H1 (decoder) + H2 (syntax):
free-decoded fluent-but-false text is withheld and the query is correctly
routed into the honest-uncertainty / learning loop instead of being emitted
with a 0.55 "learn" score.

Run from repo root:
    python -m pytest tests/unit/test_sm_grounding_gate.py -v
"""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana_ml", "src"))

import numpy as np
from ravana.chat.models import CognitiveResponseContext
from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.constants import _is_word_salad


def _build_engine():
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              data_dir="/tmp/ravana_gate_test")
    return eng


def _ctx(subject, assoc, raw):
    return CognitiveResponseContext(
        subject=subject, raw_input=raw,
        associated_concepts=[(a, 0.5) for a in assoc],
    )


# ── 1. The literal word-salad example from the report ────────────────────────
# Fluent but false: passes _is_word_salad (>=3 novel content words => safety
# valve returns False), yet must be withheld by the grounding gate.
def test_fluent_false_word_salad_is_ungrounded():
    eng = _build_engine()
    text = ("black holes are the light and the space where the matter and "
            "the time bend")
    # Sanity: the old permissive guard would have let this through.
    assert _is_word_salad(text, subject="black holes") is False
    ctx = _ctx("black holes", ["space", "gravity", "time"], "what are black holes?")
    assert eng._sm_response_grounded(ctx, text) is False


# ── 2. Hub-noun confabulation (H2: syntax path) ──────────────────────────────
def test_hub_noun_confabulation_is_ungrounded():
    eng = _build_engine()
    # 'life'/'time'/'trust' hubs are within 0.45 of half the vocab, so they
    # sail through lexical grounding, but the emitted text does not reference
    # any verified fact about the subject, nor the subject itself.
    text = "trust is the light and the space where the matter and the time bend"
    ctx = _ctx("trust", ["relationship", "belief", "faith"], "what is trust?")
    assert eng._sm_response_grounded(ctx, text) is False


# ── 3. Grounded fluent text that DOES reference the subject is allowed ─────────
def test_grounded_text_referencing_subject_passes():
    eng = _build_engine()
    # 'ravana' is seeded with a definition + web source in the bootstrapped
    # domain concepts, so it has a verified fact; the text references it.
    text = "ravana is a cognitive architecture that learns from the web"
    ctx = _ctx("ravana", ["cognitive architecture", "hebbian learning"],
               "what is ravana?")
    assert eng._sm_response_grounded(ctx, text) is True


# ── 4. Grounded fluent text referencing a top associated concept passes ───────
def test_grounded_text_referencing_assoc_passes():
    eng = _build_engine()
    text = "oxiverse builds a privacy-first ecosystem as an alternative to big tech"
    ctx = _ctx("oxiverse", ["privacy", "ecosystem", "big tech"],
               "what is oxiverse?")
    # 'privacy'/'ecosystem'/'big tech' are seeded associations for oxiverse.
    assert eng._sm_response_grounded(ctx, text) is True


# ── 5. Subject with NO verified fact (unknown word) is ungrounded ────────────
def test_unknown_subject_is_ungrounded():
    eng = _build_engine()
    text = "zzxqwpl is a concept that relates to time and space and matter"
    ctx = _ctx("zzxqwpl", ["time", "space"], "what is zzxqwpl?")
    assert eng._sm_response_grounded(ctx, text) is False


# ── 6. Integration: SM dispatch withholds ungrounded fluent text and routes to
#    honest uncertainty instead of emitting confident garbage. ─────────────────
def test_sm_path_withholds_ungrounded_and_falls_to_uncertainty():
    eng = _build_engine()
    # Force the SM path: skill the decomposition (so priority decomp fails) and
    # make the situation path reachable. We patch the internal generators so the
    # only candidate is ungrounded fluent confabulation, and assert the final
    # response is honest uncertainty (metacognitive_uncertainty), not the garbage.
    from unittest import mock

    orig = eng._generate_with_situation_model
    garbage = ("black holes are the light and the space where the matter and "
               "the time bend")

    def fake_sm(ctx):
        # Simulates the OLD behaviour: decoder returns fluent-but-false text
        # that the old code would have emitted (it only checked _is_word_salad,
        # which this text passes).
        return (garbage, "situation_model_decoder")

    with mock.patch.object(eng, "_generate_with_situation_model", fake_sm):
        resp, strat = eng._generate_response(
            _ctx("black holes", ["space", "gravity", "time"], "what are black holes?"))
    # The gate must intercept it: the reply may not contain the garbage string,
    # and the strategy should NOT be the decoder path.
    assert "light and the space" not in resp, resp
    assert strat != "situation_model_decoder", (strat, resp)
    # Should fall to a graceful metacognitive turn, not word salad.
    assert strat in ("metacognitive_uncertainty", "graph_fallback") or \
        _is_word_salad(resp, subject="black holes") is False


# ── 7. M4-grade leak: off-topic web junk that name-drops the subject ─────────
# "is pluto a planet?" returned this (web junk for the COMPANY Pluto). It
# names the subject + has associations, so the old anchoring gate passed it.
# The hardened topical-coherence gate must withhold it.
def test_offtopic_web_junk_namedropping_subject_is_ungrounded():
    eng = _build_engine()
    garbage = ("pluto planet is about currently seeking energetic and motivated "
               "interns to join our dynamic team")
    # Pre-fill "pluto" associations so (1) verified-fact holds, mimicking the
    # live run where pluto had graph associations.
    ctx = _ctx("pluto", ["planet", "moon", "earth"], "is pluto a planet?")
    assert eng._sm_response_grounded(ctx, garbage) is False


# ── 8. A coherent factual answer about the subject PASSES the gate ───────────
def test_coherent_factual_answer_passes():
    eng = _build_engine()
    text = ("pluto is a dwarf planet in the kuiper belt, and it orbits the "
            "sun far beyond neptune")
    ctx = _ctx("pluto", ["planet", "moon", "dwarf"], "is pluto a planet?")
    assert eng._sm_response_grounded(ctx, text) is True


# ── 9. Junk-token laden reply that is STILL topically about the subject
# passes the coherence gate (UI-residue sanitization is the learner's job via
# _sanitize_definition_text, not this monitor). We assert the gate does NOT
# wrongly withhold a reply that is genuinely on-topic with pluto. ────────────
def test_junk_token_reply_still_topically_coherent_passes():
    eng = _build_engine()
    text = ("pluto add to word list collocation powered by britannica.com is a "
            "planet near neptune")
    ctx = _ctx("pluto", ["planet", "neptune"], "what is pluto?")
    # "planet near neptune" are real anchors → the reply IS about pluto; the
    # coherence gate must not over-reject it. (UI-chrome stripping is handled
    # upstream by the web learner, not this monitor.)
    assert eng._sm_response_grounded(ctx, text) is True


# ── 10. Off-topic junk with REPEATED subject mentions must still be withheld
# (regression guard for the earlier fragile "subject mentioned once" rule — a
# leaky gate would have passed this once the subject appeared twice). ─────────
def test_offtopic_junk_repeated_subject_still_ungrounded():
    eng = _build_engine()
    garbage = ("pluto pluto is currently seeking energetic and motivated interns "
               "to join our dynamic team building the pluto platform")
    ctx = _ctx("pluto", ["planet", "moon", "earth"], "is pluto a planet?")
    assert eng._sm_response_grounded(ctx, garbage) is False


# ── 11. Sentence-level self-referential tautology (H2 narrative drift) ────────
# A paragraph may contain one good sentence + one associative-drift sentence
# that merely restates the subject ("whales mammals possibly bring about whales
# mammals"). Steps (1)-(3) judge the reply as a whole, so the aggregate anchor
# passes; step (4) catches the single sentence repeating the multi-word subject
# >=2x. This is GloVe-independent, so it fires even without embeddings.
def test_self_referential_tautology_sentence_withheld():
    eng = _build_engine()
    text = ("Whales mammals is about large and charismatic marine species. "
            "whales mammals possibly bring about whales mammals. "
            "basically, it colors species.")
    ctx = _ctx("whales mammals", ["marine", "species", "ocean"], "are whales mammals?")
    assert eng._sm_response_grounded(ctx, text) is False


def test_tautology_guard_is_glove_independent():
    # Directly exercise step (4): even with no verified fact / no GloVe, a
    # multi-word subject repeated >=2x in one sentence is withheld.
    eng = _build_engine()
    text = "whales mammals possibly bring about whales mammals"
    ctx = _ctx("whales mammals", [], "are whales mammals?")
    assert eng._sm_response_grounded(ctx, text) is False


# ═══════════════════════════════════════════════════════════════════════════
# PER-SENTENCE (Levelt / Wernicke clause-grained) REGRESSION — the residual
# class found in the live chat run: a GOOD sentence + a DEGENERATE tail. The
# old whole-text gate let the tail ride along because the good sentence
# donated novelty/anchoring. The clause-grained monitor must withhold ANY
# reply containing one bad clause.
# ═══════════════════════════════════════════════════════════════════════════

# ── Q5 residual: real web sentence + truncated-subject repetition tail ──────
# "black holes bend spacetime is black holes bend" repeats the truncated head
# "black holes" + verb "bend" (Nozari/Botvinick conflict signal), even though
# the FULL subject "black holes bend spacetime" appears only once.
def test_degenerate_tail_hidden_behind_good_sentence_withheld_q5():
    eng = _build_engine()
    text = ("Black holes with masses of millions to billions of solar masses are "
            "found in the universe. black holes bend spacetime is black holes bend. "
            "This is important because, black holes bend spacetime directly is "
            "black holes bend.")
    ctx = _ctx("black holes bend spacetime",
               ["gravity", "spacetime", "universe", "mass"],
               "why do black holes bend spacetime?")
    assert eng._sm_response_grounded(ctx, text) is False


# ── Q3 residual: subject+glue clauses only, hidden behind a sentence that
#    names the subject ("Gravity semantic pet... gravity semantic going"). ────
def test_subject_glue_clause_hidden_behind_good_sentence_withheld_q3():
    eng = _build_engine()
    text = ("Gravity is one of the most fundamental forces of the universe. "
            "Gravity semantic pet. Perspectives vary, meaning of gravity. "
            "This is linked to, gravity semantic going.")
    ctx = _ctx("gravity", ["force", "universe", "mass"], "what is gravity?")
    assert eng._sm_response_grounded(ctx, text) is False


# ── Q8 residual: "life" hub with subject+glue clauses only ──────────────────
def test_subject_glue_clause_hub_noun_withheld_q8():
    eng = _build_engine()
    text = ("Life is a deeply meaningful topic. Life semantic people, which "
            "semantic cannot. Interestingly, life contrastive even. the "
            "significance of this is life causal great.")
    ctx = _ctx("life", ["meaning", "death", "consciousness"], "what is the meaning of life?")
    assert eng._sm_response_grounded(ctx, text) is False


# ── Control arm: the OLD whole-text guard (_is_word_salad) PASSED these
#    degenerate strings (>=3 novel words => safety valve returns False), so
#    they would have been emitted pre-fix. The per-sentence monitor is what
#    now withholds them — proving the fix, not a GloVe/luck artifact. ────────
def test_per_sentence_residual_control_old_guard_passed():
    from ravana.chat.constants import _is_word_salad
    q5 = ("Black holes with masses of millions to billions of solar masses are "
          "found in the universe. black holes bend spacetime is black holes bend.")
    q3 = ("Gravity is one of the most fundamental forces of the universe. "
          "Gravity semantic pet. gravity semantic going.")
    # Old guard: fluent-but-false slips through (the safety valve fires).
    assert _is_word_salad(q5, subject="black holes bend spacetime") is False
    assert _is_word_salad(q3, subject="gravity") is False


# ── Regression: a fully good multi-sentence answer still PASSES (no
#    over-suppression) — mirrors Q9 oxiverse / a real web sentence. ─────────
def test_good_multi_sentence_answer_passes():
    eng = _build_engine()
    text = ("Oxiverse is a next-generation intent-first search engine designed "
            "for effective discovery. It builds a privacy-first ecosystem as an "
            "alternative to big tech, and it learns from the web.")
    ctx = _ctx("oxiverse", ["privacy", "ecosystem", "big tech"],
               "tell me about oxiverse")
    assert eng._sm_response_grounded(ctx, text) is True


# ── Decomposition path: per-sub-answer Levelt/Wernicke monitor ──────────────
# The decomposition synthesis is built from independent sub-answers. A
# degenerate sub-answer (truncated-subject repetition, subject+glue filler)
# must be DROPPED, not emitted — while a genuinely grounded sub-answer is
# preserved. This is the clause-level analog of the SM monitor and closes the
# same whole-text blind spot in the decomposition path (the old
# _decomp_grounded passed the entire synthesis the moment ANY sub-question
# answered).
def test_decomposition_drops_degenerate_subanswer_keeps_good():
    from ravana.core.question_decomposition import SubQuestion, DecompositionResult
    from ravana.core.question_decomposition import QuestionCategory

    eng = _build_engine()
    good = ("Black holes are regions of spacetime where gravity is so strong "
            "that nothing can escape.")
    bad = "black holes bend spacetime is black holes bend."
    sqs = [
        SubQuestion(id=0, text="what causes black holes bend spacetime",
                    category=QuestionCategory.WHY, target_concept="black holes",
                    is_answered=True, answer=good),
        SubQuestion(id=1, text="how does black holes bend spacetime happen",
                    category=QuestionCategory.WHY, target_concept="black holes",
                    is_answered=True, answer=bad),
    ]
    decomp = DecompositionResult(
        original_query="why do black holes bend spacetime",
        main_subject="black holes bend spacetime",
        category=QuestionCategory.WHY,
        complexity=0.7, sub_questions=sqs, goal_stack=[],
    )
    # Minimal ctx carrying associations so the monitor can judge clauses.
    ctx = _ctx("black holes", ["gravity", "spacetime", "universe", "mass"],
               "why do black holes bend spacetime?")
    ctx.decomposition = decomp
    # Synthesizer joins answered sub-answers.
    class _Syn:
        def synthesize(self, result=None, answered=None, **kw):
            return " ".join(sq.answer for sq in answered if sq.answer)
    eng.answer_synthesizer = _Syn()

    res = eng._decomposition_generation_path(ctx)
    assert res is not None, "decomposition should still emit the good clause"
    text, strat = res
    assert "regions of spacetime" in text, text
    assert "is black holes bend" not in text, text
    assert strat == "decomposed_why"


def test_decomposition_all_degenerate_withheld():
    from ravana.core.question_decomposition import SubQuestion, DecompositionResult
    from ravana.core.question_decomposition import QuestionCategory

    eng = _build_engine()
    bad1 = "black holes bend spacetime is black holes bend."
    bad2 = "black holes bend spacetime directly is black holes bend."
    sqs = [
        SubQuestion(id=0, text="what causes black holes bend spacetime",
                    category=QuestionCategory.WHY, target_concept="black holes",
                    is_answered=True, answer=bad1),
        SubQuestion(id=1, text="how does black holes bend spacetime happen",
                    category=QuestionCategory.WHY, target_concept="black holes",
                    is_answered=True, answer=bad2),
    ]
    decomp = DecompositionResult(
        original_query="why do black holes bend spacetime",
        main_subject="black holes bend spacetime",
        category=QuestionCategory.WHY,
        complexity=0.7, sub_questions=sqs, goal_stack=[],
    )
    ctx = _ctx("black holes", ["gravity", "spacetime", "universe", "mass"],
               "why do black holes bend spacetime?")
    ctx.decomposition = decomp

    class _Syn:
        def synthesize(self, result=None, answered=None, **kw):
            return " ".join(sq.answer for sq in answered if sq.answer)
    eng.answer_synthesizer = _Syn()

    res = eng._decomposition_generation_path(ctx)
    assert res is None, f"all-degenerate synthesis must be withheld, got {res}"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
