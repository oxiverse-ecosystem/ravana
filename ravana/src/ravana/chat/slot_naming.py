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
The subject is always a CLOSED-class pronoun, so the FRAME is detectable from
its shape alone: if a span opens with a subject pronoun, the token after it is
by construction the finite verb and the complement follows. That strips "i
think", "she prefers", "they believe", "we love", "i reckon", "i suspect" and
any reporting verb nobody has seen yet.

The closed-class set is PASSED IN by the caller from the vocabulary it already
owns (the miner's `_OPINION_STOP`), so this module holds no lexicon of its own
and widens automatically when the caller's set grows.

A leading ADVERB may precede the frame ("honestly i think ..."). Adverbial
`-ly` is morphology, not vocabulary, so it is handled as morphology AND only
when the token after it is a subject pronoun -- which keeps a genuine topic
that merely ends in "-ly" ("a lonely mountain") from being stripped.
"""
from __future__ import annotations

import re

__all__ = ["strip_reporting_frame", "SUBJECT_PRONOUNS"]

# Closed-class subject pronouns. This is GRAMMAR, not topic knowledge: these
# are the words that can occupy the subject slot of an English finite clause.
# It is passed to `strip_reporting_frame` alongside the caller's stop set so
# the effective closed-class grows with the caller's vocabulary.
SUBJECT_PRONOUNS = frozenset({
    "i", "we", "you", "they", "he", "she", "it", "who",
})

_TOKEN_RE = re.compile(r"[a-z'][a-z']*")


def _is_subject(token: str, closed_class) -> bool:
    """True when `token` can be the subject of a finite clause."""
    return token in SUBJECT_PRONOUNS or token in closed_class


def strip_reporting_frame(phrase: str, closed_class=()) -> str:
    """Drop a leading `<subject-pronoun> <finite-verb>` reporting frame.

    Returns the complement of the frame -- the actual subject of the attitude.
    The input is returned unchanged when it does not open with a frame, so the
    function is a no-op for bare noun phrases ("remote work", "hiking").

    `closed_class` is the CALLER's own closed-class vocabulary. It is consulted
    so a pronoun the caller's stop set knows but this module's grammar set does
    not still counts as a frame subject -- one owner of slot naming, N callers
    that agree by construction.
    """
    if not phrase:
        return phrase
    toks = _TOKEN_RE.findall(phrase.lower())
    if len(toks) < 2:
        return phrase

    closed = closed_class if isinstance(closed_class, (set, frozenset)) else set(closed_class or ())

    # Optional leading adverbial ("honestly i think ..."), tolerated ONLY when
    # a subject pronoun actually follows it. This is what keeps a real topic
    # that ends in "-ly" from being treated as a reporting frame.
    start = 0
    if toks[0].endswith("ly") and len(toks) >= 3 and _is_subject(toks[1], closed):
        start = 1

    if not _is_subject(toks[start], closed):
        return phrase

    # tok[start] is the subject, so tok[start + 1] is by construction the finite
    # verb. The complement is everything after it. Dropping exactly these two
    # tokens is what removes the fused verb from the topic key.
    complement = toks[start + 2:]
    if not complement:
        return phrase
    return " ".join(complement)