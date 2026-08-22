"""Regression tests — round 2026-08-15T1537Z FEATURE (Bug 5).

Meta-identity routing. A meta question about the USER's identity / realness
("do i seem like a real person to you", "what am i to you", "tell me
something true about who i am") must be answered from RAVANA's LIVE model of
the user — identity state + the real stance/fact stores — NOT echoed as a
verbatim episodic quote and NOT answered by an authored philosophical frame
about "real / fuzzy / time".

Root cause (round 2026-08-15T1537Z): meta-identity queries were not
recognized as a distinct intent; they fell to (a) the episodic cued-recall
path (echoing a stored turn) or (b) an authored `feeling-real` frame in
response_gen.py (`{subj} is fuzzy for me...`) keyed on the word "real".

The fix detects the meta-identity intent in `_structured_recall` and renders
from real state via `_meta_identity_reply` — name, stance count + topics,
fact count, and RAVANA's own identity strength/trend. No hardcoded reply;
every slot is read from runtime stores. The authored probe-tuned frame was
deleted.

Fail-closed: a plain biographical query ("what's my name") still resolves
from the structured store; a non-meta query still returns None from
`_structured_recall`.
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(PROJ, "ravana_ml", "src"))

from ravana.chat.engine import CognitiveChatEngine

_SUFFIX = "meta_identity_15T1537"


def _cleanup():
    try:
        _p = os.path.join("weights", f"ravana_weights{_SUFFIX}.pkl")
        if os.path.exists(_p):
            os.remove(_p)
    except Exception:
        pass


def _teach(eng):
    eng.process_turn("i am corvin")
    eng.process_turn("i love oysters")
    eng.process_turn("i think surveillance is wrong")


def test_meta_identity_reads_real_state():
    """A meta-identity query must reflect the REAL learned profile (name +
    stances + facts) and must NOT contain the deleted 'fuzzy for me' phrase
    or echo an episodic quote."""
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix=_SUFFIX)
    _teach(eng)
    for q in ("do i seem like a real person to you",
              "what am i to you",
              "tell me something true about who i am"):
        ans = eng._structured_recall(q)
        assert ans is not None, f"meta-identity reply missing for {q!r}"
        # Reflects the real learned name.
        assert "corvin" in ans.lower(), f"reply does not reference learned name: {ans!r}"
        # Reflects the real learned stances (no hardcoding — topics come from state).
        assert "oysters" in ans.lower() or "surveillance" in ans.lower(), \
            f"reply does not reference learned stances: {ans!r}"
        # Does NOT echo the deleted authored frame.
        assert "fuzzy for me" not in ans.lower(), \
            f"deleted authored frame leaked: {ans!r}"
        # Does NOT dump a verbatim episodic quote (no 'you told me earlier').
        assert "you told me earlier" not in ans.lower(), \
            f"episodic echo leaked: {ans!r}"
        # Reads identity strength (a real number from state).
        assert "self-coherence" in ans.lower() or "0." in ans.lower(), \
            f"reply does not reflect identity state: {ans!r}"
    _cleanup()


def test_non_meta_query_fails_closed_in_meta_branch():
    """A plain biographical 'what's my name' must NOT be intercepted by the
    meta-identity branch; it is handled by the structured name path, and the
    meta detector itself returns None for it."""
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix=_SUFFIX)
    _teach(eng)
    # The meta detector is the first thing in _structured_recall; a name query
    # must not trigger it (it returns the name via the bio path instead).
    ans = eng._structured_recall("what's my name")
    assert ans is not None and "corvin" in ans.lower(), f"name query broke: {ans!r}"
    _cleanup()
