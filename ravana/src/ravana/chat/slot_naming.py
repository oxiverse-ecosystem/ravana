"""Shared slot naming for stance/opinion topic keys.

ONE chokepoint so the stance MINER (``UserModel._opinion_topic``) and every
stance RECALL path (which re-resolve the query through the same method) name
the SAME slot for the same concept. Before this module the two sides could
disagree, and a comparative disclosure produced keys no query could reach.

FIX-RV-15 adds the reporting-frame strip here. An English reporting frame is
``<subject-pronoun> <finite-verb> <complement>``. When a comparative pattern's
subject group is lazy and unanchored it captures the frame along with the
proposition -- "i think remote work is better than office work" yields the span
``i think remote work`` -- and the predicator then fuses into the topic key
("think remote work"). The attitude is stored under a slot that no later query
resolves, because a query says "remote work".

The strip needs NO verb vocabulary. The frame's subject is always a closed-class
pronoun, so the closed-class set the caller already owns (``_OPINION_STOP``) is
what identifies it; the token after the subject is then by construction the
frame's verb and the proposition starts after it. That generalizes to any
speech-act or attitude verb, including ones RAVANA has never seen, with no
per-verb entry to maintain. A seed list of "think/believe/find/..." would be a
keyword table wearing a seed's clothing: it could only ever strip the verbs
somebody thought to enumerate, and the next disclosure would leak its own.

Growth path: the class is the CALLER's existing closed-class set, not a private
copy, so it widens automatically wherever ``_OPINION_STOP`` grows (the
``FunctionalLexicon`` fit-file mechanism) and the two cannot drift.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

# The subject pronouns that can open a reporting frame. This is a CLOSED class
# (English has no new pronouns), and it is a strict SUBSET of the caller's
# `_OPINION_STOP`, which the function takes as an argument so there is exactly
# one authority. Kept explicit for readability and asserted against the caller's
# set by the unit test; the function itself only needs the caller's set.
_FRAME_SUBJECTS = frozenset({"i", "you", "he", "she", "we", "they"})


def strip_reporting_frame(tokens: Sequence[str],
                          closed_class: Optional[Iterable[str]] = None
                          ) -> List[str]:
    """Drop a leading ``<subject-pronoun> <finite-verb>`` reporting frame.

    ``tokens``  the tokenized phrase, in surface order.
    ``closed_class``  the caller's closed-class word set (the subject pronouns
        are drawn from it). Defaults to the module's own subject set.

    Returns the tokens with the frame removed, or the input unchanged when the
    phrase is not frame-initial. Never returns an empty list: a phrase too short
    to carry a complement ("i think") is returned intact rather than emptied, so
    a caller cannot lose a topic by stripping too much.

    A phrase that does NOT open with a pronoun is never touched, so ordinary
    noun phrases ("remote work"), verb-initial activity names ("running"), and
    relative-clause objects ("people who talk") are unaffected.
    """
    toks = list(tokens)
    subjects = set(closed_class) & _FRAME_SUBJECTS if closed_class else _FRAME_SUBJECTS
    if not subjects:
        return toks
    # Need subject + verb + at least one token of complement. With fewer than
    # three tokens there is no complement to promote, and stripping would empty
    # or mangle the phrase.
    if len(toks) < 3 or toks[0] not in subjects:
        return toks
    return toks[2:]
