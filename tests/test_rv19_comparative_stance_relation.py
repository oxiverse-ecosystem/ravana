"""FIX-RV-19: a comparative stance must be ONE signed relation, not N keys.

Defect (cold-verified on `auto/round-2026-09-25T1155Z`, reproduced on
`github/main` @ 4287916a in-process, RAVANA_OFFLINE=1, clean suffix):

    "i think remote work is better than office work"

  -> stances: 'think remote work' +0.70, 'office work' +0.70, 'remote work' +0.70

Three failures in one utterance:
  1. the reporting verb `think` is FUSED into the topic key;
  2. ONE proposition is shredded into THREE keys;
  3. the comparative DIRECTION is destroyed — all three carry +0.70, so the
     loser (office work) is stored as ENDORSED and a later contrastive query
     cannot tell which side won.

These tests are RED on the unfixed tree and GREEN after the fix. They are
deliberately behavioural: they assert on the STORE and on a real contrastive
round-trip, never on the miner's internals.
"""
from __future__ import annotations

import os
import re
import sys

import pytest

_TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_TREE,
           os.path.join(_TREE, "ravana_ml", "src"),
           os.path.join(_TREE, "ravana", "src"),
           os.path.join(_TREE, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# A worktree must never import the MAIN checkout: the venv's editable install
# puts a finder on sys.meta_path, which outranks sys.path.
sys.meta_path = [f for f in sys.meta_path
                 if "editable" not in (type(f).__module__ or "").lower()
                 and "editable" not in (type(f).__name__ or "").lower()]

os.environ.setdefault("RAVANA_OFFLINE", "1")

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402
from ravana.chat.slot_naming import (  # noqa: E402
    is_subject_pronoun, learn_subject_pronoun, strip_reporting_frame)

DISCLOSURE = "i think remote work is better than office work"
WINNER, LOSER = "remote work", "office work"

# A reporting frame is `<subject-pronoun> <finite-verb> ...`. The verb itself is
# open-class, so these tests check for the FRAME (a leading pronoun) rather than
# for any particular verb -- asserting "no verb from a list" would bake the list
# into the test.
_LEADING_PRONOUN = re.compile(
    r"^\s*(i|we|you|they|she|he|it)\b\s+\w+(s|ed|ing)?\b", re.IGNORECASE)


@pytest.fixture(scope="module")
def engine():
    """A clean, isolated engine. No prior pickle, no cross-test state."""
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="rv19test")
    eng.process_turn("i like hiking a lot")
    eng.process_turn(DISCLOSURE)
    yield eng
    try:
        eng.stop_background_learning()
    except Exception:
        pass


def _stances(eng):
    return eng.user_model.opinions.stances


# ── Defect 1: the reporting frame must not survive into the key ──────────────
def test_no_stance_key_starts_with_a_reporting_frame(engine):
    """A topic key names the SUBJECT, not the user's way of asserting it."""
    for key in _stances(engine):
        assert not _LEADING_PRONOUN.match(key), (
            f"stance key {key!r} carries a reporting frame "
            f"(pronoun + verb); the key must name the subject")


def test_no_three_way_shredding(engine):
    """ONE comparative proposition must not become three stored keys."""
    keys = set(_stances(engine))
    # 'hiking' is the earlier unrelated disclosure and is expected.
    assert keys <= {"hiking", WINNER, LOSER}, (
        f"one proposition was shredded into extra keys: {sorted(keys)}")


# ── Defect 2: the comparative DIRECTION must survive ─────────────────────────
def test_winner_is_positive_and_loser_is_negative(engine):
    """The direction is the whole point of a comparative claim."""
    st = _stances(engine)
    assert WINNER in st, f"endorsed side missing from store: {sorted(st)}"
    assert LOSER in st, f"rejected side missing from store: {sorted(st)}"
    assert st[WINNER].polarity > 0, (
        f"winner {WINNER!r} has polarity {st[WINNER].polarity:+.2f}")
    assert st[LOSER].polarity < 0, (
        f"loser {LOSER!r} was stored as {st[LOSER].polarity:+.2f} -- the "
        f"comparative direction is destroyed")


def test_sides_are_not_equally_valenced(engine):
    """Two sides of one comparison must never carry the same polarity."""
    st = _stances(engine)
    assert st[WINNER].polarity != st[LOSER].polarity


# ── Defect 3: a later contrastive query must recover the WINNER ──────────────
@pytest.mark.parametrize("query", [
    "do you prefer remote work or office work?",
    "which do you prefer, remote work or office work?",
])
def test_contrastive_query_recovers_the_winner(engine, query):
    """The point of storing a signed relation is that a later contrast can
    read it. This is the capability the contrastive-routing fix relies on."""
    reply = engine.process_turn(query)
    text = (reply or "").lower()
    # The reply must resolve BOTH sides from the store, naming the winner as
    # the one RAVANA leans toward -- not as "still forming a view" (no stored
    # stance was found) and not as an echo of the disclosure turn.
    assert WINNER in text, f"winner not recovered from contrast query: {reply!r}"
    assert LOSER in text, f"loser not named in contrast reply: {reply!r}"
    assert "still forming" not in text, (
        f"contrast reply found no stance -- the sides are not recoverable: "
        f"{reply!r}")
    assert "you mentioned" not in text, (
        f"contrast query was answered by episodic echo, not by the stance "
        f"relation: {reply!r}")


def test_contrastive_lean_names_the_winner_not_the_loser(engine):
    """The lean must be attributable: 'for <winner>', never 'for <loser>'."""
    reply = (engine.process_turn("do you prefer remote work or office work?")
             or "").lower()
    for lean in ("strongly for", "for ", "lean toward"):
        if lean in reply:
            tail = reply.split(lean, 1)[1]
            assert tail.startswith(WINNER), (
                f"lean {lean!r} was attributed to {tail[:40]!r}, not the "
                f"winner {WINNER!r}: {reply!r}")
            return
    for lean in ("against", "wary of", "strongly against"):
        if lean in reply:
            tail = reply.split(lean, 1)[1]
            assert tail.startswith(LOSER), (
                f"negative lean {lean!r} attached to the wrong side: {reply!r}")
            return
    pytest.fail(f"no attributable lean in contrast reply: {reply!r}")


# ── Generality: not a fix tuned to this one sentence ─────────────────────────
# NOTE ON PROBE CHOICE. Two things are MEASURED here, not guessed, because
# both have bitten a test that assumed otherwise:
#   * These probes must be long enough to clear the engine's gibberish guard,
#     which rejects very short declaratives ("tea beats coffee" -> strategy
#     `gibberish_guard`, zero stances). That rejection is PRE-EXISTING and was
#     verified identical on the untouched baseline sha, so it is not this fix's
#     business and must not be "fixed" here by weakening the guard --
#     `test_short_comparative_is_gated_by_the_gibberish_guard` below pins it.
#   * The expected keys are the store keys MEASURED from a real run. A stance key
#     is the miner's content head, which keeps its modifiers ("small towns",
#     not "towns"), so guessing the head noun here would fail for a reason
#     unrelated to the relation being tested.
@pytest.mark.parametrize("disclosure,winner,loser", [
    ("i believe trains are better than planes for daily work", "trains", "planes"),
    ("honestly i think the mountains are finer than the coast",
     "mountains", "coast"),
    ("small towns make better humans than big cities",
     "small towns", "big cities"),
])
def test_comparatives_are_signed_relations_in_general(disclosure, winner, loser):
    """The fix must be grammatical, not keyed to 'remote work'."""
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix=f"rv19_{abs(hash(disclosure)) % 9999}")
    try:
        eng.process_turn(disclosure)
        st = eng.user_model.opinions.stances
        assert winner in st, f"{disclosure!r} did not key the winner: {sorted(st)}"
        assert loser in st, f"{disclosure!r} did not key the loser: {sorted(st)}"
        assert st[winner].polarity > 0 and st[loser].polarity < 0, (
            f"{disclosure!r} -> winner {st[winner].polarity:+.2f}, "
            f"loser {st[loser].polarity:+.2f}")
    finally:
        try:
            eng.stop_background_learning()
        except Exception:
            pass


def test_short_comparative_is_gated_by_the_gibberish_guard():
    """A very short comparative never reaches the miner at all.

    Documents a PRE-EXISTING, unrelated behaviour so a future reader does not
    mistake it for a regression of this fix, and so the probes above are not
    quietly rewritten into unreachable no-ops. Verified identical on the
    untouched baseline commit; this fix deliberately does not touch the guard.
    """
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="rv19gib")
    try:
        eng.process_turn("tea beats coffee")
        assert eng._last_strategy == "gibberish_guard", (
            f"expected the gibberish guard to claim this short comparative, "
            f"got strategy={eng._last_strategy!r}")
        assert not eng.user_model.opinions.stances, (
            "the miner should not run when the gibberish guard claims the turn")
    finally:
        try:
            eng.stop_background_learning()
        except Exception:
            pass


# ── The frame stripper itself ────────────────────────────────────────────────
# Pure-function tests. These pin BOTH what the stripper must do (remove a fused
# reporting verb with no verb vocabulary) and what it must NOT do, the second
# half being where a real regression was actually caught.
@pytest.mark.parametrize("phrase,expected", [
    ("i think remote work", "remote work"),
    ("honestly i think the mountains", "the mountains"),
    ("she prefers filter coffee", "filter coffee"),
    ("they believe the sea", "the sea"),
    ("we love hiking", "hiking"),
    ("i suspect the quiet", "the quiet"),
])
def test_frame_is_stripped_for_any_reporting_verb(phrase, expected):
    """The verb is whatever sits in the verb slot -- no verb is enumerated.

    'suspect' and 'prefers' are here precisely because they are NOT part of any
    reporting-verb list in the codebase: if a future change starts recognising
    specific verbs, these two still have to work by shape alone.
    """
    assert strip_reporting_frame(phrase) == expected


@pytest.mark.parametrize("phrase", [
    "remote work",
    "hiking",
    "most modern music",     # regression: 'most' is a stop word, not a pronoun
    "only the sea",
    "just noise",
    "like small talk",
    "better weather",
    "a lonely mountain",     # '-ly' topic with no pronoun after it
    "lonely",
])
def test_bare_noun_phrases_are_left_alone(phrase):
    """A span not opening with a PRONOUN is not a reporting frame.

    "most modern music" is the measured regression: an earlier revision let any
    token from the miner's closed-class stop set count as a subject, "most" was
    in that set, so "modern" was read as a verb and the key became "music". A
    function-word class is not a pronoun class.
    """
    assert strip_reporting_frame(phrase) == phrase


def test_pronoun_class_is_runtime_extensible():
    """The seed grammar must be widenable without a code change (no retraining)."""
    # An unseen token is not yet a subject, so the span is left intact.
    assert not is_subject_pronoun("thee")
    assert strip_reporting_frame("thee prefer tea") == "thee prefer tea"
    # ...and once learned, the same shape works with no further change: exactly
    # the subject and the finite verb are dropped, leaving the complement.
    assert learn_subject_pronoun("thee") is True
    assert is_subject_pronoun("thee")
    assert strip_reporting_frame("thee prefer tea") == "tea"
    # Learning is idempotent: re-registering a seed pronoun changes nothing.
    assert learn_subject_pronoun("i") is False


def test_frame_strip_is_a_noop_when_there_is_no_complement():
    """A bare "<pronoun> <verb>" span has no subject to recover -- leave it."""
    assert strip_reporting_frame("i think") == "i think"