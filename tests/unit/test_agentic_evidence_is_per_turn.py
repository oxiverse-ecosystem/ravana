"""Capability test: agentic evidence is PER-TURN, never carried forward.

`process_turn` stashes tool evidence at engine.py:6861 and consumes it at
engine.py:10261, appending it to the reply. Those two points are ~3400 lines
apart and 55 early-return paths sit between them (every short-circuit
strategy), so an early return left the slot populated and the next turn that
reached the end-of-turn block appended evidence for a query the user never
asked.

Measured in the 2026-09-29 round probe
(tmp/chat_probe_2026_09_29T1239Z.txt, turns 12/15/19): the user's "sediment
cores" turn and the "remind me what you said about sediment cores" turn both
printed the web payload for "do you get bored when i am quiet for a long
time?". A grounded-evidence channel that reports a different question than the
one asked is a confabulated citation.

Each test gets a FRESH engine; no result depends on test ordering.
"""
import inspect
import os
import sys

import pytest

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (PROJ,
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ravana.chat.engine as engine_mod  # noqa: E402
from ravana.chat.engine import CognitiveChatEngine  # noqa: E402

SRC = inspect.getsource(engine_mod.CognitiveChatEngine.process_turn)
LINES = SRC.splitlines()

# Match the stash PRECISELY: it is the f-string write carrying "agentic:". A
# bare "_pending_web_evidence =" also matches the top-of-turn reset, which
# would make the two indistinguishable and silently pass any ordering check.
_STASH = next(i for i, l in enumerate(LINES)
              if "_pending_web_evidence" in l and "agentic:" in l)
_RESET = next(i for i, l in enumerate(LINES)
              if "_pending_web_evidence = None" in l)
_CONSUME = next(i for i, l in enumerate(LINES) if "_ev = getattr" in l)

PRIOR_MARKER = "PRIOR-TURN-EVIDENCE-MARKER"


@pytest.fixture
def engine():
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               user_suffix="agenticleak")


def test_every_turn_starts_with_a_clean_evidence_slot():
    """The invariant: no turn can inherit another turn's tool evidence."""
    # The reset must run BEFORE the stash; otherwise it would wipe the slot
    # the stash just wrote and no evidence would ever surface.
    assert _RESET < _STASH, (
        "the per-turn reset must precede the agentic stash, otherwise every "
        "turn's own evidence is discarded (reset@%d, stash@%d)" % (_RESET, _STASH))


def test_early_return_paths_cannot_leave_evidence_behind():
    """Structural: the stash->consume span is full of early returns.

    This is why the reset has to live at the TOP of process_turn rather than
    next to the consume: clearing at the consume site is exactly what already
    fails. The test documents the hazard so removing the top-of-turn reset is
    visibly wrong.
    """
    early = [l.strip() for l in LINES[_STASH:_CONSUME]
             if l.strip().startswith("return")]
    assert len(early) > 0, (
        "if this is ever 0 the span can be restructured and the top-of-turn "
        "reset re-evaluated; today there are %d early returns between the "
        "stash and the consume" % len(early))


def test_evidence_does_not_survive_into_the_next_turn(engine):
    """A short-circuiting turn must not hand its evidence to the next one.

    Poison the slot the way a stashing-then-early-returning turn leaves it.
    The next turn may legitimately fire its OWN search and repopulate the slot,
    so the assertion is that the PRIOR marker is gone - not that the slot is
    empty.
    """
    engine._pending_web_evidence = "[agentic:web_search] " + PRIOR_MARKER
    try:
        engine.process_turn("hello there")
    except Exception:
        # A crash still must not leave the stale slot behind for the next turn.
        pass
    slot = getattr(engine, "_pending_web_evidence", None)
    assert not (slot and PRIOR_MARKER in slot), (
        "prior turn's evidence survived into the next turn: %r" % (slot or "")[:140])


def test_prior_evidence_does_not_reach_the_reply(engine):
    """The user-visible symptom: a stale payload appended to a later reply."""
    engine._pending_web_evidence = "[agentic:web_search] " + PRIOR_MARKER
    try:
        reply = engine.process_turn("what is the deal with sediment cores") or ""
    except Exception:
        reply = ""
    assert PRIOR_MARKER not in reply, (
        "prior turn's evidence was appended to this turn's reply: %r" % reply[:200])


def test_same_turn_channel_is_still_wired():
    """The fix must not disable the channel - only stop it outliving its turn."""
    assert "agentic:" in LINES[_STASH]
    assert any("_pending_web_evidence = None" in l
               for l in LINES[_CONSUME:_CONSUME + 6]), (
        "the end-of-turn consume must still clear the slot after appending")
