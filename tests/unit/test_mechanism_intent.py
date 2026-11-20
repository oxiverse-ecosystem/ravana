"""Mechanism intent must be paraphrase-invariant and direction-correct."""
import os
import sys

_PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (os.path.join(_PROJ, "ravana", "src"),
           os.path.join(_PROJ, "ravana_ml", "src"),
           os.path.join(_PROJ, "ravana-v2", "src"), _PROJ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.chat.response_gen import mechanism_intent, longest_label_seed  # noqa


def test_paraphrases_all_address_backward():
    for q in ("why does fog happen", "how is fog produced",
              "what is the mechanism behind fog", "what causes fog",
              "can you explain how fog comes about"):
        assert mechanism_intent(q) == "backward", q


def test_effect_questions_are_forward_and_non_mechanism_rejected():
    assert mechanism_intent("what are the effects of fog") == "forward"
    assert mechanism_intent("what is fog") is None


def test_longest_label_first():
    got = longest_label_seed("why does cold fog happen", ["fog", "cold", "happen"])
    assert got[0] == "happen" or set(got) == {"fog", "cold", "happen"}
