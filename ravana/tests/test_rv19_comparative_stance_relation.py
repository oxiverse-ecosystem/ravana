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
sys.meta_path = [f for f in sys.meta_path
                 if "editable" not in (type(f).__module__ or "").lower()
                 and "editable" not in (type(f).__name__ or "").lower()]

os.environ.setdefault("RAVANA_OFFLINE", "1")

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402

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
@pytest.mark.parametrize("disclosure,winner,loser", [
    ("tea beats coffee", "tea", "coffee"),
    ("i believe trains are better than planes",
     "trains", "planes"),
    ("honestly i think the mountains are finer than the coast",
     "mountains", "coast"),
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