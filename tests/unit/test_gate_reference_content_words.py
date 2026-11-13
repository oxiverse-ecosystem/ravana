"""The grounding gate must not accept word salad that happens to contain a
high-frequency associate.

MEASURED BUG (2026-11-14): for the subject 'blue', `ctx.associated_concepts`
contained the word 'often'. The gate's per-sentence reference check accepts ANY
associate appearing in the emitted text, so the pure word salad

    "Such there like many used often other."

PASSED `_sm_response_grounded`. The step-1 GloVe check already excludes function
words from the evidence set ("'there' (0.355) and 'i' (0.302) were passing the
check as evidence, which is noise, not grounding") -- but that exclusion was never
applied to the step-2 reference check.

A function word is not evidence that a sentence is about the subject. These tests
pin the content-word filter and the resulting verdicts.
"""
import pytest

from ravana.chat.constants import STOP_WORDS
from ravana.chat.engine import CognitiveChatEngine


@pytest.fixture(scope="module")
def eng(tmp_path_factory):
    return CognitiveChatEngine(
        data_dir=str(tmp_path_factory.mktemp("gate")), user_suffix="")


@pytest.fixture(scope="module")
def blue_ctx(eng):
    """Build a context for 'blue' the way process_turn does.

    Mirrors eval_decoder_groundedness.build_ctx: reset the situation model,
    activate from the input, spread from the subject's node ids, and assemble
    the CognitiveResponseContext.
    """
    import numpy as _np
    from ravana.chat.models import CognitiveResponseContext

    eng.situation_model.reset()
    raw_input = "what is blue"
    activated = eng._activate_from_input(raw_input)
    subject_ids = set(eng._concept_keywords.get("blue", []))
    associations = eng._spread_and_collect(activated, primary_ids=subject_ids)

    sentence_vector = eng._build_sentence_vector(raw_input)
    context_vector = eng._build_context_vector_from_input(raw_input, "blue")

    concept_embs = {}
    activations = {}
    for label, score in associations[:12]:
        ll = label.lower()
        nids = eng._concept_keywords.get(ll, [])
        if nids:
            node = eng.graph.get_node(nids[0])
            if node is not None and node.vector is not None:
                concept_embs[ll] = node.vector
                activations[ll] = max(activations.get(ll, 0.0), float(score))

    eng.situation_model.update(
        concept_embeddings=concept_embs,
        activations=activations,
        graph_get_vector_fn=eng._glove_vector,
        sentence_vector=sentence_vector,
        context_vector_input=context_vector,
    )
    return CognitiveResponseContext(
        subject="blue",
        raw_input=raw_input,
        associated_concepts=associations,
        bridge_concept=eng._find_bridge(associations, "blue"),
        situation_vector=eng.situation_model.get_blended_vector(),
        situation_narrative=eng.situation_model.get_narrative_suggestions(),
        sentence_vector=sentence_vector,
        context_vector=context_vector,
    )


class TestContentWordFilter:
    @pytest.mark.parametrize("word", [
        "often", "such", "there", "many", "used", "other", "the", "a",
        "this", "that", "it", "they", "very", "really", "just", "also",
    ])
    def test_non_referring_words_are_not_content_words(self, eng, word):
        assert not eng._is_content_word(word), \
            "%r cannot serve as grounding evidence" % word

    @pytest.mark.parametrize("word", [
        "colour", "scattering", "wavelength", "spectrum", "primate",
        "black holes", "sky blue", "quantum computing",
    ])
    def test_referring_words_are_content_words(self, eng, word):
        assert eng._is_content_word(word), \
            "%r should be usable as grounding evidence" % word

    def test_multi_word_phrase_with_a_function_word_is_rejected(self, eng):
        assert not eng._is_content_word("the same")
        assert not eng._is_content_word("such as colour")


class TestGateRejectsSalad:
    def test_word_salad_with_a_high_frequency_associate_is_rejected(
            self, eng, blue_ctx):
        """The exact measured failure."""
        salad = "Such there like many used often other."
        assert not eng._sm_response_grounded(blue_ctx, salad)

    def test_real_definition_still_passes(self, eng, blue_ctx):
        good = ("Blue is the colour of the short-wavelength end of the "
                "visible spectrum, between violet and green.")
        assert eng._sm_response_grounded(blue_ctx, good)

    def test_graph_neighbour_sentence_still_passes(self, eng, blue_ctx):
        good = ("The sky appears blue because molecules scatter shorter "
                "blue wavelengths.")
        assert eng._sm_response_grounded(blue_ctx, good)

    def test_subject_with_content_word_still_passes(self, eng, blue_ctx):
        assert eng._sm_response_grounded(blue_ctx, "Blue is a colour.")
