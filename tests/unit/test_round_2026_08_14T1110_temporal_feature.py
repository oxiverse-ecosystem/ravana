"""Regression tests — round 2026-08-14T1110Z FEATURE.

Date-grounded temporal recall for a LEARNED activity. Two gaps closed:

  Fix A — STEM-LINKED activity matching (NOT semantic/GloVe). The prior resolver
  linked a `does`/`event` fact to the dated `since` activity ONLY by identical
  leading verb ("study" == "study"). A rotated, paraphrased query ("all this
  volcano stuff") that shares NO token with the stored activity ("study basaltic
  eruptions") failed closed and fell through to a generic episodic echo — because
  the word "volcano" lived in a SEPARATE `does` fact ("start studying volcanoes
  back") that the verb-identity link never connected. The resolver now links every
  `does`/`event` fact to each `since` activity it shares a MORPHOLOGICAL STEM with
  ("volcanoes" == "volcano", "studying" == "study"), so the distinctive words
  still reach the match context and the paraphrase recalls the right dated fact.
  Data-driven, generalizes to any phrasing, fails closed (no shared stem -> no
  link -> zero overlap -> None).

  Fix B — NATURAL gerund display. The reply previously emitted the bare stored
  verb ("you started study basaltic eruptions in 2015") — broken English. It now
  realizes the activity as a grammatical gerund ("studying basaltic eruptions"),
  and drops a redundant inceptive verb ("started studying volcanoes" -> "studying
  volcanoes") to avoid "started starting studying".

No hardcoded reply; every answer slot is read from the PersonalFactStore and
morphologically generated. Fail-closed: an unrelated "when did i...?" (no
stored dated activity, semantic OR lexical) still returns None.
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(PROJ, "ravana_ml", "src"))

from ravana.chat.engine import (
    CognitiveChatEngine,
    _gerund_of,
    _verb_phrase_to_gerund,
)

_SUFFIX = "temporal_feature_14T1110"


def _cleanup():
    try:
        import os as _os
        _p = _os.path.join("weights", f"ravana_weights{_SUFFIX}.pkl")
        if _os.path.exists(_p):
            _os.remove(_p)
    except Exception:
        pass


def _teach(eng):
    # A volcanologist who started studying volcanoes in 2015 and also keeps
    # tarantulas — multiple dated activities so paraphrase routing is meaningful.
    eng.process_turn("i'm nadia")
    eng.process_turn("i started studying volcanoes back in 2015")
    eng.process_turn("i study basaltic eruptions")
    eng.process_turn("i also keep three tarantulas")


def test_semantic_date_recall_paraphrase():
    """'volcano stuff' shares no token with 'study basaltic eruptions' but must
    still recall the 2015 since-fact via semantic proximity."""
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix=_SUFFIX)
    _teach(eng)
    ans = eng._structured_recall("what year did i start all this volcano stuff again")
    _cleanup()
    assert ans is not None, "semantic date recall failed (fell through to echo)"
    assert "2015" in ans, f"wrong year recalled: {ans!r}"
    # grammatical gerund, not bare verb
    assert "studying" in ans, f"broken gerund in reply: {ans!r}"


def test_explicit_date_recall_still_grammatical():
    """A literal query must still recall correctly AND read grammatically."""
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix=_SUFFIX)
    _teach(eng)
    ans = eng._structured_recall("when did i start studying volcanoes")
    _cleanup()
    assert ans is not None and "2015" in ans, f"date recall failed: {ans!r}"
    assert "studying" in ans, f"broken gerund in reply: {ans!r}"


def test_how_long_gerund():
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix=_SUFFIX)
    _teach(eng)
    ans = eng._structured_recall("how long have i been studying volcanoes")
    _cleanup()
    assert ans is not None and "2015" in ans, f"duration recall failed: {ans!r}"
    assert "studying" in ans, f"broken gerund in reply: {ans!r}"


def test_unrelated_when_query_fails_closed():
    """An unrelated 'when did i...?' with no matching dated activity must return
    None (honest fallback), not latch onto an unrelated stored fact."""
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix=_SUFFIX)
    _teach(eng)
    # "the moon landing" shares no token with either stored activity
    # ("study basaltic eruptions" / "keep three tarantulas"), so the resolver
    # must fail closed.
    ans = eng._structured_recall("when did i start the moon landing")
    _cleanup()
    assert ans is None, f"expected fail-closed None, got: {ans!r}"


def test_gerund_morphology():
    """Unit-level: the morphological gerund generator handles regular,
    silent-e, doubling, and irregular verbs without an LLM."""
    assert _gerund_of("study") == "studying"
    assert _gerund_of("make") == "making"
    assert _gerund_of("run") == "running"
    assert _gerund_of("go") == "going"
    assert _gerund_of("keep") == "keeping"
    assert _verb_phrase_to_gerund("study basaltic eruptions") == "studying basaltic eruptions"
    assert _verb_phrase_to_gerund("keep three tarantulas") == "keeping three tarantulas"
