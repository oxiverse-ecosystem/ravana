"""A definitional question is not a social speech act.

MEASURED BUG (2026-11-14): the social-intent prototype classifier over-triggers
on short inputs:

    "what is like"  -> invite    (should ABSTAIN)
    "what is being" -> greeting  (should ABSTAIN)

Both were routed to the chitchat path, which emitted decoder salad, because the
factual path -- which correctly answers "what is gravity" -- never ran.

The brain does not mentalize a definitional question: there is no addressee to
model. `_handle_chitchat` now pre-filters the exact definitional frame before the
classifier runs, so those queries reach the factual path (where a MISS degrades
to honest uncertainty).
"""
import pytest

from ravana.chat.engine import CognitiveChatEngine


@pytest.fixture(scope="module")
def eng(tmp_path_factory):
    return CognitiveChatEngine(
        data_dir=str(tmp_path_factory.mktemp("chitchat")), user_suffix="")


class TestDefinitionalQueryPreFilter:
    @pytest.mark.parametrize("q", [
        "what is like",
        "what is being",
        "what is gravity",
        "what is love",
        "what are black holes",
        "does water boil",
        "who wrote hamlet",
    ])
    def test_definitional_queries_are_not_chitchat(self, eng, q):
        assert eng._handle_chitchat(q, q) is None, \
            "%r is a request for a definition, not a social act" % q

    @pytest.mark.parametrize("q", [
        "hello",
        "hi there",
        "how are you",
        "what is up",
        "good morning",
        "bye",
        "thank you",
    ])
    def test_genuine_social_acts_still_route(self, eng, q):
        """The filter must not over-reach: real social intent still fires."""
        out = eng._handle_chitchat(q, q)
        assert out is not None, "%r should be handled as a social act" % q
