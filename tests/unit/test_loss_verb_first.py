"""RED->GREEN: verb-first possession-loss disclosures route to grief empathy
and name the lost entity (not "feeling lost is hard").

Round 2026-08-19T1026Z: the self-possessive loss detector only matched the
NOUN-FIRST shape "my <noun> <loss-term>" (my dog died). The VERB-FIRST shape
"<loss-term> my <noun>" (i lost my grandmother) did NOT match, so a genuine
bereavement routed to the generic "feeling lost is hard" empathy frame
(measured T15: "i lost my grandmother" -> "feeling lost is hard"). The
loss-term "lost" was treated as a felt-state word, not a possession-loss event.

This test pins the generalization: verb-first disclosures must route to
emotional_empathy with a reply that NAMES the real entity (grandmother /
colony / wallet / friend), and must NOT say "feeling lost". Third-entity /
narrative uses ("the wind dies down") must still NOT trigger bereavement.

The reply content is driven by REAL extracted state (the disclosed entity),
never authored prose.
"""
import os, sys, io, contextlib
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine


def run():
    fails = 0
    verb_first = [
        ("i lost my grandmother last spring", "grandmother"),
        ("i lost my colony", "colony"),
        ("i lost my wallet", "wallet"),
        ("my friend passed away", "friend"),
    ]
    for q, entity in verb_first:
        eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                                  user_suffix="test_vfloss")
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
        if "feeling lost" in reply.lower():
            print(f"[FAIL] '{q}' still says 'feeling lost' -> {reply!r}")
            fails += 1
            continue
        if entity not in reply.lower():
            print(f"[FAIL] '{q}' reply does not name entity '{entity}': {reply!r}")
            fails += 1
            continue
        print(f"[OK] '{q}' -> grief empathy, names {entity}")
    # Third-entity / narrative must NOT trigger bereavement.
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="test_vfloss")
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


def test_loss_verb_first():
    run()


if __name__ == "__main__":
    run()
