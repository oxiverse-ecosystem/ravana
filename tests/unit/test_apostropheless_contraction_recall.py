"""Apostrophe-less contraction reaches the same recall answer (round 2026-10-03T2040Z).

RESIDUAL from the previous round, made concrete here: a transcription drops the
apostrophe ("whats my dog called"), and every grammatical gate in the recall
path tests the SPELLED-OUT shape. The fact was in the store and the resolver
returned None, so the turn fell through to the empty fallback. The fix is one
morphological expander (``topic_head.expand_contractions``) applied once at the
recall resolver's entry, extending the elision class by the same RULE the n't
class already used — not a query->answer table.

Three things are proved here:
  (a) the apostrophe-less spelling answers identically to the spelled-out one;
  (b) the spelled-out behaviour is UNCHANGED (no regression at the gate);
  (c) the normaliser must NOT widen a recall gate — a clause-leading
      disclosure ("what i love is running") is not an inverted question and
      must still be MINED, not answered.
"""
import os

os.environ.setdefault("RAVANA_OFFLINE", "1")

import pytest

from ravana.chat.topic_head import expand_contractions


# ── (c-1) the morphology itself ────────────────────────────────────────────
@pytest.mark.parametrize("raw,expected", [
    ("whats my dog called", "what is my dog called"),
    ("shes my dog", "she is my dog"),
    ("hes my dog", "he is my dog"),
    ("theres my dog", "there is my dog"),
    ("whos my dog", "who is my dog"),
    ("wheres my dog", "where is my dog"),
    ("hows my dog", "how is my dog"),
])
def test_elision_class_expands_by_rule(raw, expected):
    """A form nobody enumerated still reduces — the rule, not a table."""
    assert expand_contractions(raw) == expected


@pytest.mark.parametrize("keep", [
    "its the wedging table",     # possessive, must NOT become "it is"
    "my dog was sleeping",       # already a copula
    "my dog has a bed",          # closed-class word ending in s
    "i love dogs",               # plural noun, stem is not a copula-taker
    "what i love is running",    # the clause-leading disclosure, untouched
])
def test_elision_leaves_real_words_alone(keep):
    assert expand_contractions(keep) == keep


def test_nt_class_is_unchanged():
    """The pre-existing morphology must not regress."""
    assert expand_contractions("i don't know") == "i do not know"
    assert expand_contractions("it ain't real") == "it is not real"


# ── (a)/(b) end-to-end through the live engine ────────────────────────────
def _engine():
    from ravana.chat.engine import CognitiveChatEngine
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               user_suffix="contraction_norm")


def _reply(eng, q):
    r = eng.process_turn(q)
    return (r[1] if isinstance(r, tuple) else str(r))


def test_apostrophe_less_recall_matches_spelled_out():
    eng = _engine()
    eng.process_turn("my dog biscuit sleeps on the wedging table")
    spelled = _reply(eng, "what is my dog called")
    bare = _reply(eng, "whats my dog called")
    # (a) identical answer, and it is the NAME from the store
    assert bare == spelled
    assert "biscuit" in bare


def test_spelled_out_behaviour_unchanged():
    eng = _engine()
    eng.process_turn("my dog biscuit sleeps on the wedging table")
    out = _reply(eng, "what is my dog called")
    assert "biscuit" in out


def test_clause_leading_disclosure_still_reaches_the_mining_path():
    """(c) the normaliser must not widen a recall gate.

    "what i love is running" opens with a question word but is a DECLARATION.
    If the normaliser or the gate treated it as a question, the recall
    resolver would swallow the turn and answer it instead of leaving it to the
    mining path. So the invariant is that the RECALL RESOLVER stands down for
    it — before and after the fix alike — which is what keeps the disclosure
    available to mine.

    NOTE on what this does and does not claim: the stance miner does not yet
    mine this clause-leading shape (measured: no fact is stored either way).
    That is a SEPARATE pre-existing gap, reproduced identically on clean HEAD,
    and asserting it here would pin a capability this fix does not deliver.
    """
    from ravana.chat.engine import CognitiveChatEngine
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="clause_lead_disclosure")
    assert eng._structured_recall("what i love is running") is None
    # ...and the normaliser must not have rewritten it into an inversion.
    assert expand_contractions("what i love is running") == "what i love is running"


def test_clause_leading_disclosure_is_not_answered_as_a_question():
    """The turn must not come back as a recall ANSWER for a declaration."""
    eng = _engine()
    out = _reply(eng, "what i love is running")
    assert not out.lower().startswith("you told me")
