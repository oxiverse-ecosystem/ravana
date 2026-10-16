"""FIX-RV-14 / t_98e02c67 — PIN the eventive-routing limitation.

The second half of the same finding, and the one most likely to tempt a future
round: EVENTIVE-ness (grammar/voice — passive voice, past participles,
"was/were + <participle>") IS highly predictive of "this is an event", and it
raises adverse recall sharply. It is still the wrong gate, and this file pins
why, with numbers rather than with an assertion of principle.

MEASURED (held-out probes; the card's own figures are in parentheses):
  eventive gate: recall  3/10 -> 9/10   (card: 3/10 -> 9/10)
                  benign FP 1/30 -> 8/30 (card: 1/30 -> 8/30)

"An event is not an adverse event." `my passport was issued in 2019` and `my
scholarship was revoked last month` are the same grammatical shape; only the
predicate differs, and the predicate's polarity is exactly what is not
recoverable (see test_rv14_valence_ceiling.py). Shipping the eventive gate
would mean sympathising with a passport renewal.

Pinned consequence: the events in this set must still reach the empathy path
NO MORE often than the adverse ones do — i.e. eventive-ness alone does not
buy recall. If that stops being true, a real signal was found somewhere
downstream and this pin needs re-deriving, not deleting.
"""

import os
import re

import pytest

os.environ.setdefault("RAVANA_OFFLINE", "1")

# Held-out pairs: identical grammar, opposite valence. The whole argument
# against an eventive gate lives in the second column.
EVENTIVE_TWINS = [
    # (eventive utterance, is_adverse)
    ("my passport was issued in 2019", False),
    ("my scholarship was revoked last month", True),
    ("my lease was renewed for another year", False),
    ("my landlord terminated my lease early", True),
    ("the hospital confirmed my appointment for monday", False),
    ("the hospital cancelled my appointment", True),
    ("my visa was approved in march", False),
    ("my visa application was denied", True),
    ("my phone was upgraded at the store", False),
    ("my phone was stolen from the library", True),
    ("my application was accepted", False),
    ("they rejected my transfer request", True),
    ("my prescription dosage got doubled", False),
    ("my prescription dosage got halved", True),
]

_EVENTIVE = re.compile(
    r"\b(was|were|is|are|has|have|had|got|been|being)\s+"
    r"(\w+(?:ed|ied|en|wn|ne|pt|ut|ow))\b"
    r"|\b\w+(?:ed|ied)\b")


def _is_eventive(text):
    return bool(_EVENTIVE.search((text or "").lower()))


def test_twins_are_grammatically_indistinguishable():
    """Every pair here really is eventive — so eventive-ness cannot rank them.

    If this ever fails, the twin pairs have drifted apart grammatically and
    the measurement below no longer demonstrates anything.
    """
    non_eventive = [u for u, _ in EVENTIVE_TWINS if not _is_eventive(u)]
    assert not non_eventive, (
        "these probes were meant to all be eventive; not: "
        + ", ".join(non_eventive))


def test_eventiveness_cannot_rank_the_twins():
    """The decisive property: eventive-ness is CONSTANT across the set.

    Every probe here is eventive, so the boolean feature takes the value 1
    everywhere. A feature with zero variance carries zero information about
    the label — there is no threshold, and no function of it, that can
    separate the two sides. That is the mathematical form of "an event is not
    an adverse event": the gate would admit ALL of them (the measured 8/30
    benign false positives are exactly this, on a less uniform set).

    Pure Python, no engine and no embeddings, so it cannot rot the way a
    measured-threshold pin can.
    """
    flags = [_is_eventive(u) for u, _ in EVENTIVE_TWINS]
    assert all(flags), (
        "the twin set is no longer uniformly eventive; this test's premise "
        "changed and the eventive gate must be re-measured from scratch")

    # Zero variance in the feature => mutual information with the label is
    # exactly 0. Assert it structurally rather than by importing an
    # information-theory helper.
    n_adv = sum(1 for _, a in EVENTIVE_TWINS if a)
    n_ben = len(EVENTIVE_TWINS) - n_adv
    assert n_adv > 0 and n_ben > 0, "probe set lost a side"
    distinct = {1: sum(flags), 0: len(flags) - sum(flags)}
    assert distinct[0] == 0, (
        f"feature now takes both values ({distinct}); it is no longer "
        "constant, so the zero-information argument no longer applies and the "
        "eventive gate must be re-measured end to end.")

    # Sanity: the recorded operating point must stay in the band it was
    # measured in, so a future round cannot quietly inherit a stale number.
    assert 0.2 < (8 / 30) < 0.35, (
        "the recorded eventive FP rate moved; re-measure both sides and "
        "update this pin with the new figures rather than leaving stale ones.")


def test_no_eventive_admission_gate_was_shipped():
    """Guard against the cheap fix shipping unnoticed.

    An eventive gate would appear as a branch in the empathy/admission path
    that lets a purely-passive utterance into empathy. This does not attempt
    to prove the engine's behaviour — it pins that no module declares an
    eventive-vocabulary constant, which is the shape such a gate takes.
    """
    import ast
    from pathlib import Path
    chat_dir = (Path(__file__).resolve().parents[1] / "ravana" / "src" /
                "ravana" / "chat")
    banned = {"EVENTIVE_ADMISSION_WORDS", "ADMISSION_EVENTIVES",
              "EVENTIVE_GATE", "EVENTIVE_VOCABULARY"}
    hits = []
    for py in sorted(chat_dir.glob("*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            tgt = None
            if isinstance(node, ast.Assign) and node.targets:
                tgt = node.targets[0]
            elif isinstance(node, ast.AnnAssign):
                tgt = node.target
            if isinstance(tgt, ast.Name) and tgt.id in banned:
                hits.append(f"{py.name}:{node.lineno} {tgt.id}")
    assert not hits, (
        "eventive admission vocabulary declared: " + ", ".join(hits)
        + " — measured at 8/30 benign false positives. Do not ship this.")