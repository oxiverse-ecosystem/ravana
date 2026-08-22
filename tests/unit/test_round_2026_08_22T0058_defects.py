"""RED->GREEN regression tests for RAVANA round 2026-08-22T0058Z.

DEFECT A — stale stance echo on flipped contradiction.
  "i love knitting ..." then "i've come to hate knitting ..." establishes a
  NEGATIVE stance (polarity < 0) in user_model.opinions.stances. A follow-up
  "do you still think i love knitting?" (the adverb "still" between subject and
  the think-verb previously blocked the user-stance matcher) must read the LIVE
  flipped stance ("against knitting"), NOT replay the original "you love
  knitting" belief echo. Content comes from the live UserStanceStore.

DEFECT B — leading-modifier relationship disclosures were dropped.
  "my OLD mentor Dr. Osei taught me ...", "my DEAR friend Priya runs ...",
  "my LATE uncle Bram brews ..." carry a modifier BEFORE the relationship word.
  The miner took only the first word after "my" as the relation head, so
  "old"/"dear"/"late" became the (non-)relationship head and the WHOLE
  disclosure was dropped — a later "who is my mentor?" had nothing to recall.
  Fix strips leading non-relationship modifiers until relation_of() recognizes
  a relationship word, so the disclosure mines + recalls like any other.

Both fixes are structural (one shared lexicon, no per-role branch, no
retraining, no authored prose). Reply content is driven by REAL mined/live
state.
"""
import os, sys, io, contextlib
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine


def _new_engine(suffix):
    import glob, os as _os
    for f in (glob.glob(f"weights/ravana_weights{suffix}.pkl")
              + glob.glob(f"weights/ravana_usermodel{suffix}.pkl")):
        _os.remove(f)
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)


def test_defect_a_stance_flip_reflected():
    """Flipped stance must be read back, not the stale original belief."""
    eng = _new_engine("test_dA")
    with contextlib.redirect_stdout(io.StringIO()):
        eng.process_turn("i love knitting, it's the most calming thing in the world.")
        eng.process_turn("actually, you know what, i've come to hate knitting "
                         "— it just makes my hands ache.")
    # live stance should now be NEGATIVE
    _key = eng.user_model.opinions.resolve_topic("knitting")
    _s = eng.user_model.opinions.stances.get(_key)
    assert _s is not None and _s.polarity < 0, f"expected negative knitting stance, got {_s}"
    # the revisit query must read the flipped stance, not echo "you love knitting"
    ans = eng._user_stance_reply("do you still think i love knitting?")
    assert ans is not None, "user-stance reply returned None (fell through)"
    assert "against" in ans, f"expected flipped stance 'against', got: {ans!r}"
    assert "love knitting" not in ans, f"stale original leaked: {ans!r}"
    # full turn path (not just the direct method)
    with contextlib.redirect_stdout(io.StringIO()):
        full = eng.process_turn("do you still think i love knitting?")
    assert "against" in full.lower(), f"full turn leaked stale stance: {full!r}"
    print("[OK] DEFECT A: flipped stance reflected on still-question")


def test_defect_b_leading_modifier_relationship():
    """Relationship disclosure with a leading modifier must mine + recall."""
    eng = _new_engine("test_dB")
    with contextlib.redirect_stdout(io.StringIO()):
        eng.process_turn("my old mentor, Dr. Osei, taught me field biology "
                         "and how to read animal tracks.")
        eng.process_turn("my dear friend Priya runs a little bookshop by the river.")
        eng.process_turn("my late uncle Bram brews cider from apples he grows himself.")
    rel = [(a, f.value) for (s, a, _), f in
           eng.user_model.personal_facts.facts.items()
           if s == "i" and not f.superseded]
    assert ("mentor dr. osei",) in [(a,) for a, _ in rel], (
        f"mentor disclosure (with modifier 'old') not mined: {rel}")
    assert ("friend priya",) in [(a,) for a, _ in rel], (
        f"friend disclosure (with modifier 'dear') not mined: {rel}")
    assert ("uncle bram",) in [(a,) for a, _ in rel], (
        f"uncle disclosure (with modifier 'late') not mined: {rel}")
    # recall via reverse-name resolver
    with contextlib.redirect_stdout(io.StringIO()):
        ans = eng.process_turn("who is dr. osei to me?")
    assert "mentor" in ans.lower(), f"mentor not recalled: {ans!r}"
    with contextlib.redirect_stdout(io.StringIO()):
        ans2 = eng.process_turn("what does my mentor do for me?")
    assert "taught" in ans2.lower(), f"mentor activity not recalled: {ans2!r}"
    print("[OK] DEFECT B: leading-modifier relationships mined + recalled")


if __name__ == "__main__":
    test_defect_a_stance_flip_reflected()
    test_defect_b_leading_modifier_relationship()
    print("\nALL ROUND-2026-08-22T0058Z REGRESSION TESTS PASSED")
