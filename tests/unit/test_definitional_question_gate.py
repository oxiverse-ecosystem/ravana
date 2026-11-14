"""Tests for the definitional-question gate in the episodic recall paths.

The gate exists because retrieval-by-cue fires on lexical overlap: after any
turn mentioning `water`, the query "what is water" replayed that turn instead
of answering from the definition store. The tests pin both halves -- a
definitional question must NOT reach recall, and a first-person recollection
question MUST still reach it.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ravana.chat.engine_memory import _is_definitional_question  # noqa: E402


class TestDefinitionalShape:
    @pytest.mark.parametrize("q", [
        "what is water",
        "what is gravity",
        "what causes rain",
        "who is meera",
        "what are clouds",
        "tell me about the ocean",
        "explain gravity",
    ])
    def test_world_questions_are_definitional(self, q):
        assert _is_definitional_question(q), q

    @pytest.mark.parametrize("q", [
        # wh-words that take a complement clause are NOT definitional: they are
        # disclosure follow-ups, and test_self_cued_episodic_recall pins that
        # "where does meera restore clocks" must recall.
        "why does it rain",
        "how does a battery work",
        "when does the sun rise",
        "where does meera restore clocks",
    ])
    def test_complement_wh_questions_are_not_definitional(self, q):
        assert not _is_definitional_question(q), q

    @pytest.mark.parametrize("q", [
        # not a question about the world
        "water",
        "tell me a sentence about water",
        "i like water",
    ])
    def test_non_world_shapes_are_not_definitional(self, q):
        assert not _is_definitional_question(q), q

    def test_opinion_shape_is_caught_earlier_not_by_this_gate(self):
        """`what do you think about water` matches the `what do` auxiliary.

        That is a false positive of THIS pattern, and it is safe because the
        opinion router's `_SELF_OPINION_SHAPE` gate runs first and returns
        before recall is consulted. Pinned so the ordering dependency is
        explicit rather than accidental.
        """
        import re
        from ravana.chat.engine_memory import _SELF_OPINION_SHAPE
        q = "what do you think about water"
        assert _is_definitional_question(q)
        assert re.search(_SELF_OPINION_SHAPE, q), \
            "opinion gate must run before the definitional gate"


class TestRecollectionExempt:
    """First-person recollection must still reach the recall path.

    The callers exempt queries containing i/me/my/mine/we/our BEFORE consulting
    the gate, so these are definitional by SHAPE but exempt by framing. Both
    halves are pinned here so neither can be dropped silently.
    """

    @pytest.mark.parametrize("q", [
        "what did i tell you about water",
        "what did i mention earlier",
        "have i told you about my brother",
    ])
    def test_recollection_is_first_person(self, q):
        import re
        assert re.search(r"\b(i|me|my|mine|we|our)\b", q), q

    @pytest.mark.parametrize("q", [
        "what did i tell you about water",
        "what did i mention earlier",
    ])
    def test_recollection_shape_is_definitional_but_exempt(self, q):
        assert _is_definitional_question(q), q
