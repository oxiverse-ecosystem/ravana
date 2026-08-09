"""Regression tests for round 2026-08-09h fixes.

Covers four defect classes surfaced by a fresh-probe chat (persona
"Indira", a coastal forager / slipware ceramicist) that were NOT fixed by
round-g:

  H1  possessive-entity mining/recall with the BARE form ("my cat mire is
      ...", "my partner cole keeps ...") — the 's form was already handled
      by round-g (test_recall_confabulation_2026g), but the bare form was
      not, so recall returned the USER's own name or a raw echo.
  H2/H5 self/other boundary for RAVANA-about-RAVANA questions phrased
      without the exact round-g keyword set ("your own take", "thread about
      yourself", "recognize yourself") — these leaked a stored USER fact or a
      raw user utterance (identity confusion).
  H3  count/quantity CORRECTION whose prior activity fact was never stored
      ("it's nine slipware jugs i fired, not a first kiln") — the detector
      needs num+entity even when no prior 'does' fact exists, and the recall
      query can carry an extra noun ("how many slipware jugs did i make").
  H4  degenerate single/multi-token mirror ack ("got it — tide.", "yeah,
      wild food honest.") — now only reflects a single clean noun.

All replies are built from REAL state (valence, identity strength, stored
facts); none are authored multi-sentence pools. See tmp/reports/ROUND_2026-
08-09h_REPORT.md.
"""

import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
sys.path[:0] = [
    r"C:\Users\Likhith\Documents\Projects\ravana\ravana\src",
]

from ravana.chat.engine import CognitiveChatEngine


def _new_engine(suffix):
    return CognitiveChatEngine(
        dim=64, seed=42, baby_mode=True, user_suffix=f"t_h_{suffix}")


def test_h1_bare_possessive_entity_mining_and_recall():
    """Bare 'my <ent> <name> is/verb <desc>' stores under the ENTITY key
    and recalls correctly — must NOT return the user's own name."""
    eng = _new_engine("h1")
    eng.process_turn("i'm indira, for the record")
    eng.process_turn(
        "my cat mire is a half-feral tabby who hunts the saltmarsh voles")
    eng.process_turn(
        "my partner cole keeps the tide tables and reads them like scripture")
    # recall partner name
    r = eng.process_turn("what's my partner's name again? i lose track")
    assert "cole" in r.lower(), f"expected partner cole, got: {r!r}"
    assert "indira" not in r.lower(), f"leaked user name, got: {r!r}"
    # recall cat name
    r2 = eng.process_turn("do you still remember my cat's name?")
    assert "mire" in r2.lower(), f"expected cat mire, got: {r2!r}"
    assert "indira" not in r2.lower(), f"leaked user name, got: {r2!r}"
    # entity facts stored under entity key, not 'i'
    ent_keys = [k for k in eng.user_model.personal_facts.facts
                if isinstance(k, tuple) and len(k) == 3 and k[0] == "partner"]
    assert any(k[1] == "name" and getattr(v, "value", v) == "cole"
               for k, v in eng.user_model.personal_facts.facts.items()
               if k == ("partner", "name", "cole"))
    eng.stop_background_learning()


def test_h2_self_introspection_routes_to_state_not_user_fact():
    """RAVANA-about-RAVANA questions must answer from identity state, never
    replay a stored USER fact or raw user utterance."""
    eng = _new_engine("h2")
    eng.process_turn("i'm indira")
    eng.process_turn("my cat mire is a half-feral tabby")
    for q in (
        "what was your own take on slipware versus the wheel, the thing you told me?",
        "one last thing — what's the one thread about yourself you'd hold onto?",
        "do you recognize yourself yet, or are you still piecing it together?",
    ):
        r = eng.process_turn(q)
        # identity-state driven: references 'me'/'i' (ravana) not the user's
        # name/life
        assert "indira" not in r.lower(), f"leaked user name on {q!r}: {r!r}"
        assert "mire" not in r.lower(), f"leaked user fact on {q!r}: {r!r}"
        assert ("i'm" in r.lower() or "me" in r.lower() or "who i am" in r.lower()
                or "myself" in r.lower()), f"not self-state-driven on {q!r}: {r!r}"
    eng.stop_background_learning()


def test_h3_count_correction_without_prior_fact_persists_and_recalls():
    """A quantity correction with no prior stored activity fact still learns
    online and is recallable via a 'how many X did i make' query."""
    eng = _new_engine("h3")
    eng.process_turn("i fired my first full slipware kiln and the shop took it")
    r = eng.process_turn(
        "oh, i misspoke earlier — it's nine slipware jugs i fired, not a "
        "first kiln, i lost count and under-said it")
    # the correction ack proves detection (the flag is reset after use)
    assert "correct" in r.lower(), f"correction not detected/acked: {r!r}"
    # the corrected count was learned online (no retrain)
    assert "nine" in r.lower(), f"corrected count not learned: {r!r}"
    # recall the corrected count
    r2 = eng.process_turn("so how many slipware jugs did i make, after that?")
    assert "nine" in r2.lower(), f"expected nine recalled, got: {r2!r}"
    eng.stop_background_learning()


def test_h4_mirror_only_reflects_single_clean_noun():
    """The assertion mirror must not reflect a multi-word clause fragment
    (e.g. 'wild food honest', 'lost urchin') — those fall back to a
    topic-less lead."""
    eng = _new_engine("h4")
    r = eng.process_turn("wild food is honest, i was wrong to doubt it")
    # must not produce a garbled multi-word mirror like 'yeah, wild food honest.'
    assert "wild food honest" not in r.lower(), f"garbled mirror: {r!r}"
    assert "lost urchin" not in r.lower(), f"garbled mirror: {r!r}"
    eng.stop_background_learning()
