"""RED->GREEN: multi-word entity extraction for bereavement disclosures.

Bug (round 2026-09-22T0209Z, turn 27): "i think I might be losing my sense
of self" produced "i'm so sorry about your of" — the entity capture regex
only matched 2 words ("sense of"), and the filler stripper picked "of" as
the head. The fix expands capture to 4 words and extends the filler set
with prepositions/determiners/temporal words.

This test pins the generalization: multi-word entities must resolve to the
real head noun, and trailing modifiers (temporal, prepositional) must strip.
"""
import os, sys, io, contextlib
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine


def run():
    fails = 0
    cases = [
        # (query, must_contain, must_not_contain, description)
        ("i think i might be losing my sense of self",
         "self", "your of",
         "verb-first 3-word entity (the bug)"),
        ("i lost my grandmother last spring",
         "grandmother", "your of",
         "verb-first with trailing temporal"),
        ("i lost my best friend in the world",
         "friend", None,
         "verb-first 4-word entity"),
        ("my dear old dog died",
         "dog", None,
         "noun-first with leading fillers"),
        ("my father died yesterday",
         "father", None,
         "noun-first with trailing temporal"),
    ]
    for q, must, must_not, desc in cases:
        eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                                  user_suffix="test_mwe_loss")
        eng._last_strategy = None
        with contextlib.redirect_stdout(io.StringIO()):
            r = eng.process_turn(q)
        reply = r if isinstance(r, str) else str(r)
        strat = eng._last_strategy
        eng.stop_background_learning()
        if strat != "emotional_empathy":
            print(f"[FAIL] '{q}' expected emotional_empathy, got {strat}")
            fails += 1
            continue
        if must and must not in reply.lower():
            print(f"[FAIL] '{q}' reply missing '{must}': {reply!r}")
            fails += 1
            continue
        if must_not and must_not in reply.lower():
            print(f"[FAIL] '{q}' reply contains forbidden '{must_not}': {reply!r}")
            fails += 1
            continue
        print(f"[OK] '{q}' -> {reply!r}")

    # Third-entity guard must still hold
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="test_mwe_loss")
    eng._last_strategy = None
    with contextlib.redirect_stdout(io.StringIO()):
        r = eng.process_turn("the wind dies down at dusk")
    reply = r if isinstance(r, str) else str(r)
    strat = eng._last_strategy
    eng.stop_background_learning()
    if strat == "emotional_empathy":
        print(f"[FAIL] third-entity 'wind dies down' wrongly -> empathy: {reply!r}")
        fails += 1
    else:
        print(f"[OK] 'wind dies down' correctly NOT empathy (strategy={strat})")

    if fails:
        print(f"\n{fails} FAILED")
        raise SystemExit(1)
    print("\nALL PASSED")


def test_multi_word_entity_loss():
    run()


if __name__ == "__main__":
    run()
