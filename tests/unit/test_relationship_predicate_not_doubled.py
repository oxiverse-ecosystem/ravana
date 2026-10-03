"""Regression: a relationship disclosure must not DOUBLE its predicate.

Defect (round 2026-09-30T0444Z, found in tmp/chat_log.txt TURN 4/19 of the
previous round): the utterance

    "my sister devika handles the glazing end"

rendered at recall as

    "your sister devika handles glazing end is devika handles the glazing end"

ROOT CAUSE: "handles" was in NEITHER the activity-verb lexicon NOR the
relation-verb lexicon, so the relationship miner's verb scan
(user_model.py, the is_activity_verb / is_relation_verb / is_aux_verb
cascade that sets _vidx) found no verb head. Control fell to the
name-only path, which sets

    _name = " ".join(_toks[:_vidx])          # user_model.py ~3417
    _attr = f"{_kin} {_name}"                 # user_model.py ~3616

with _vidx None and the clause packed into the name slot, producing the
malformed key ('i', 'sister devika handles glazing end', 'devika handles
the glazing end'). The recall renderer then emitted "<attr> is <value>"
and the predicate appeared twice.

FIX: the verb now scans correctly, so _name is just the NAME and the
predicate lives only in the value.

The point of these tests is the SHAPE OF THE KEY, not the prose: a name
slot must never contain the predicate. That invariant holds for any
relation word and any responsibility verb, so these cases deliberately
span four different relations/facts rather than pinning one sentence.
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for _p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest

from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.user_model import is_relation_verb


# (disclosure, name, verb) — four DIFFERENT relations, not one sentence.
CASES = [
    ("my sister devika handles the glazing end", "devika", "handles"),
    ("my brother caleb oversees the roof repairs", "caleb", "oversees"),
    ("my manager priya coordinates the vendor rollout", "priya", "coordinates"),
    ("my aunt bea administers the estate accounts", "bea", "administers"),
]


def _engine(i):
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               user_suffix=f"reg_dbl_{i}")


@pytest.mark.parametrize("utt,name,verb", CASES)
def test_name_slot_never_contains_the_predicate(utt, name, verb):
    """The fact's attr slot holds a NAME, never the whole clause."""
    eng = _engine(CASES.index((utt, name, verb)))
    eng.process_turn(utt)

    offending = []
    for k in eng.user_model.personal_facts.facts:
        attr = str(k[1]) if len(k) > 1 else ""
        if name in attr and verb in attr:
            offending.append(attr)

    assert not offending, (
        f"clause packed into the name slot for {utt!r}: {offending}")


@pytest.mark.parametrize("utt,name,verb", CASES)
def test_recall_reply_does_not_repeat_the_predicate(utt, name, verb):
    """The rendered reply states the predicate exactly once."""
    eng = _engine(CASES.index((utt, name, verb)))
    eng.process_turn(utt)
    reply = eng.process_turn(f"who is {name}")
    reply = reply if isinstance(reply, str) else str(reply)

    assert reply.lower().count(verb) <= 1, (
        f"predicate {verb!r} doubled in recall for {utt!r}: {reply!r}")


def test_responsibility_verbs_are_recognized_relation_verbs():
    """The class of verb that caused the defect is in the seed lexicon.

    Seed vocabulary (a lexicon, not an answer table): every entry is a
    verb RAVANA can also learn at runtime, and removing one only means
    one fewer responsibility shape is recognized.
    """
    for _utt, _name, verb in CASES:
        assert is_relation_verb(verb), f"{verb!r} not a relation verb"
