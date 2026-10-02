"""Shared naming/normalization helpers for stance + proposition keys.

Why this module exists
----------------------
A stance key must name the SUBJECT the user holds an attitude about, not the
way they asserted it. "i think remote work is better than office work" is an
attitude about *remote work*; keying it on "think remote work" fuses the
reporting verb into the topic and makes the key unrecallable by any later
question that names the subject.

The trap is that the obvious remedy -- a list of reporting verbs -- is a
keyword table wearing a seed's clothing: it can only strip verbs somebody
enumerated. This module ships NO verb vocabulary at all.

The structural rule
-------------------
An English reporting frame is `<subject-pronoun> <finite-verb> <complement>`.
The subject is always a PRONOUN, so the frame is detectable from its shape
alone: if a span opens with a pronoun, the token after it is by construction the
finite verb and the complement follows. That strips "i think", "she prefers",
"they believe", "we love", "i reckon", "i suspect" and any reporting verb
nobody has seen yet.

MEASURED LIMIT OF THE RULE -- do not widen it naively (found 2026-10-01)
--------------------------------------------------------------------------
An earlier revision accepted any token in the CALLER's closed-class stop set as
a frame subject. That was wrong, and measurably so: the miner's `_OPINION_STOP`
contains "most", "only", "just", "like", "than", ... so for "most modern music
is just wallpaper" the span opened with "most", the helper read "modern" as the
verb, and the key became "music" instead of "modern music" (caught by
tests/test_round_2026_08f_regression.py). A closed-class FUNCTION-WORD set and
a PRONOUN class are different things: only a pronoun can occupy the subject slot
of an English finite clause. The pronoun class is therefore owned HERE as
grammar and is NOT borrowed from the caller's stop list.

Growth path (seed-vs-hardcoding): the pronoun class is seed GRAMMAR, and RAVANA
can widen it at runtime via `learn_subject_pronoun` when a caller determines a
new token can fill the subject slot. Removing one entry degrades that one
pronoun's reporting frame and nothing else.

A leading ADVERB may precede the frame ("honestly i think ..."). Adverbial
`-ly` is morphology, not vocabulary, so it is handled as morphology AND only
when a pronoun actually follows it -- which keeps a genuine topic that merely
ends in "-ly" ("a lonely mountain") from being stripped.
"""
from __future__ import annotations

import re

__all__ = [
    "strip_reporting_frame",
    "is_subject_pronoun",
    "learn_subject_pronoun",
    "is_vacuous_subject",
    "learn_vacuous_subject",
    "SUBJECT_PRONOUNS",
    "VACUOUS_SUBJECTS",
]

# The seed PRONOUN class -- closed-class GRAMMAR: the words that can occupy the
# subject slot of an English finite clause. Not topic knowledge, not a reporting
# verb list, and not a synonym table. No entry here names an opinion, so nothing
# in it can be tuned to a probe.
SUBJECT_PRONOUNS = frozenset({
    "i", "we", "you", "they", "he", "she", "it", "who", "one", "ones", "them",
})

# Grown at runtime by `learn_subject_pronoun`; kept separate from the seed so a
# learned entry stays distinguishable from grammar.
_LEARNED_SUBJECTS: set[str] = set()

# ── The QUANTIFIED subject class (FIX-RV-24) ─────────────────────────────────
# A quantified subject ("nothing", "nobody", "everyone", "anything") is
# grammatical in the subject slot but denotes NO referent the user can hold an
# attitude about. A comparative whose winner side is such a quantifier is not a
# two-sided comparison at all: in "nothing beats cold water swimming jumping"
# the speaker is NOT endorsing "nothing" -- the assertion is entirely about the
# loser.
#
# WHY THIS IS GRAMMAR, NOT A KEYWORD TABLE. The pattern is structural: the
# subject slot is filled by a determiner-quantifier rather than by a referring
# expression. That is the same closed-class slot the PRONOUN set above owns, and
# it generalizes to every quantifier ("no one", "everything", "something"),
# present and past, singular and plural. It is not a reporting-verb list and
# nothing in it can be tuned to a probe.
#
# Growth path: `learn_vacuous_subject` lets RAVANA register another quantifier
# from experience, so this stays seed grammar rather than a frozen table.
VACUOUS_SUBJECTS = frozenset({
    "nothing", "nobody", "none", "everyone", "everybody", "anything",
    "something", "everything", "all", "any", "whatever", "whoever",
    "no one", "no-one", "none of",
})

_LEARNED_VACUOUS: set[str] = set()


def learn_vacuous_subject(token: str) -> bool:
    """Register `token` as a subject that denotes no referent.

    The runtime growth path for the quantified-subject class, so RAVANA
    extends its own grammar from experience instead of needing a code change
    or a retrain. Returns True when the class actually grew.
    """
    t = (token or "").strip().lower().strip("'")
    if not t or not t.isalpha():
        return False
    if is_vacuous_subject(t):
        return False
    _LEARNED_VACUOUS.add(t)
    return True


def is_vacuous_subject(token: str) -> bool:
    """True when `token` fills the subject slot without denoting a referent."""
    t = (token or "").strip().lower()
    if not t:
        return False
    if t in VACUOUS_SUBJECTS or t in _LEARNED_VACUOUS:
        return True
    # "none of" / "no one" are multiword; match on the head quantifier so
    # "none of my habits" is recognized without enumerating the complement.
    head = t.split()[0] if t.split() else t
    return head in VACUOUS_SUBJECTS or head in _LEARNED_VACUOUS

_TOKEN_RE = re.compile(r"[a-z'][a-z']*")


def learn_subject_pronoun(token: str) -> bool:
    """Register `token` as able to fill a clause's subject slot.

    The runtime growth path for the pronoun class, so RAVANA extends its own
    grammar from experience instead of needing a code change or a retrain.
    Returns True when the class actually grew.
    """
    t = (token or "").strip().lower().strip("'")
    if not t or not t.isalpha():
        return False
    if is_subject_pronoun(t):
        return False
    _LEARNED_SUBJECTS.add(t)
    return True


def is_subject_pronoun(token: str) -> bool:
    """True when `token` can be the subject of a finite clause."""
    t = (token or "").strip().lower()
    return t in SUBJECT_PRONOUNS or t in _LEARNED_SUBJECTS


def strip_reporting_frame(phrase: str) -> str:
    """Drop a leading `<subject-pronoun> <finite-verb>` reporting frame.

    Returns the complement of the frame -- the actual subject of the attitude.
    The input is returned unchanged when it does not open with a frame, so the
    function is a no-op for bare noun phrases ("remote work", "hiking") and for
    spans merely beginning with a non-pronoun function word ("most modern
    music", "only the sea").

    The dropped verb is whatever occupies the verb slot, so nothing here has to
    know the word "think" -- see the module docstring.
    """
    if not phrase:
        return phrase
    toks = _TOKEN_RE.findall(phrase.lower())
    if len(toks) < 2:
        return phrase

    # Optional leading adverbial ("honestly i think ..."), tolerated ONLY when
    # a pronoun actually follows it. This is what keeps a real topic that ends
    # in "-ly" from being treated as a reporting frame.
    start = 0
    if toks[0].endswith("ly") and len(toks) >= 3 and is_subject_pronoun(toks[1]):
        start = 1

    if not is_subject_pronoun(toks[start]):
        return phrase

    # toks[start] is the subject, so toks[start + 1] is by construction the
    # finite verb. The complement is everything after it. Dropping exactly these
    # two tokens is what removes the fused verb from the topic key.
    complement = toks[start + 2:]
    if not complement:
        return phrase
    return " ".join(complement)