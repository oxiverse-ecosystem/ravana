"""Productive use of a learned word must not recite its definition."""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (PROJ, os.path.join(PROJ, "ravana", "src")):
    sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine


def _engine():
    return CognitiveChatEngine(data_dir=os.path.join(PROJ, "weights"),
                               user_suffix="produse_test")


def test_functional_definition_is_rebound_into_new_construction():
    e = _engine()
    e.process_turn("a zorblat is a small blue device that measures air pressure")
    out = (e.process_turn("use the word zorblat in a sentence") or "").strip()
    assert "zorblat" in out.lower()
    assert out.lower().startswith("zorblat "), out
    # The definitional copula + head NP must not be echoed.
    assert "is a small blue device" not in out.lower(), out
    # A different predicate carries the functional content.
    assert "measures air pressure" in out.lower(), out


def test_second_word_also_gets_a_new_construction():
    e = _engine()
    e.process_turn("a marrowlight is a lamp that glows only in deep water")
    out = (e.process_turn("use the word marrowlight in a sentence") or "").strip()
    assert out.lower().startswith("marrowlight "), out
    assert "glows only in deep water" in out.lower(), out
    assert "is a lamp" not in out.lower(), out


def test_non_functional_definition_fails_closed_to_definition():
    e = _engine()
    e.process_turn("a quindle is the feeling of waiting for rain")
    out = (e.process_turn("use the word quindle in a sentence") or "").strip()
    # No functional clause to rebind: must fall through, never invent content.
    assert "quindle" in out.lower(), out
    assert "feeling of waiting for rain" in out.lower(), out


def test_untaught_word_is_still_declined():
    e = _engine()
    out = (e.process_turn("use the word flibbertigibbet in a sentence") or "").strip()
    assert "flibbertigibbet" in out.lower(), out
    assert not out.lower().startswith("flibbertigibbet "), out
    assert ("not totally sure" in out.lower()
            or "outside what i know" in out.lower()
            or "solid grasp" in out.lower()), out


def test_two_identical_requests_produce_two_distinct_constructions():
    e = _engine()
    e.process_turn("a zorblat is a small blue device that measures air pressure")
    q = "use the word zorblat in a sentence"
    a1 = (e.process_turn(q) or "").strip()
    a2 = (e.process_turn(q) or "").strip()
    assert a1 != a2, (a1, a2)
    assert a1.lower().startswith("zorblat "), a1
    assert "called zorblat" in a2.lower(), a2
    # Neither echoes the definition.
    for out in (a1, a2):
        assert "is a small blue device" not in out.lower(), out


def test_paraphrases_all_deploy_the_learned_word():
    e = _engine()
    e.process_turn("a zorblat is a small blue device that measures air pressure")
    queries = [
        "make a sentence with zorblat",
        "give me an example using the word zorblat",
        "write a sentence using zorblat",
    ]
    for q in queries:
        out = (e.process_turn(q) or "").strip().lower()
        assert "zorblat" in out, (q, out)
        assert "measures air pressure" in out, (q, out)
        assert "can't physically" not in out, (q, out)
        assert "noted" not in out, (q, out)
