"""RED->GREEN: relationship/pet disclosures with common verbs are mined AND
reverse-lookupable by entity name.

Round 2026-08-19T1026Z: relationship/pet disclosures were only mined when an
ACTIVITY VERB was recognised by the narrow seed lexicon. Verbs the lexicon
lacked (sends, knocks, left/gave, named, passes) caused the disclosure to be
DROPPED entirely, so later reverse-lookups had nothing to retrieve:
  - "what did my grandmother give me?" -> "i don't really have a solid grasp"
  - "who is Mochi to me?" -> self-intro (wrong)
  - "who is Pora and where is she?" -> "your partner's name is marisol"
After widening the verb seed vocabulary + always mining the relationship+name
(independent of verb recognition), the SAME disclosures now mine into the
PersonalFactStore and resolve correctly via structured recall.

This is a GENERALIZATION (6f), not a re-specialization: the verb lexicon is
shared seed data (RAVANA-expandable) used by the single mining path for
relatives, pets, and possessions alike — no per-entity branch was added.

The reply content is driven by REAL mined state (the disclosed entity + name +
activity), never authored prose.
"""
import os, sys, io, contextlib
from pathlib import Path
os.environ["RAVANA_OFFLINE"] = "1"
# Derive repo root from this test file's location
PROJ = Path(__file__).resolve().parent.parent.parent
for p in (str(PROJ), str(PROJ / "ravana_ml" / "src"), str(PROJ / "ravana" / "src")):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine


def run():
    fails = 0
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="test_revlookup")
    disclosures = [
        "my sister Pora sends me postcards from the steppe",
        "my cat Mochi knocks my pottery off the shelf on purpose",
        "my cat mochi sleeps on the router",  # lowercase name (6f generalization)
        "my grandmother Indira left me her brass compass",
        "my grandmother left me her brass compass",
    ]
    for d in disclosures:
        with contextlib.redirect_stdout(io.StringIO()):
            eng.process_turn(d)

    checks = [
        ("what did my grandmother give me?",
         ("grandmother", "brass compass")),
        ("who is Mochi to me?", ("cat", "mochi")),
        ("who is mochi to me?", ("cat", "mochi")),  # lowercase query too
        ("who is Pora and where is she?", ("sister", "pora")),
    ]
    for q, (needle1, needle2) in checks:
        eng._last_strategy = None
        with contextlib.redirect_stdout(io.StringIO()):
            r = eng.process_turn(q)
        reply = r if isinstance(r, str) else str(r)
        strat = eng._last_strategy
        eng.stop_background_learning()
        if strat != "structured_recall":
            print(f"[FAIL] '{q}' expected structured_recall, got {strat}: {reply!r}")
            fails += 1
            continue
        if needle1 not in reply.lower() or needle2 not in reply.lower():
            print(f"[FAIL] '{q}' reply missing '{needle1}'/'{needle2}': {reply!r}")
            fails += 1
            continue
        print(f"[OK] '{q}' -> {reply!r}")

    # Negative: a common-noun object must NOT be mined as a pet name.
    eng.user_model.personal_facts.facts.clear()
    eng.user_model.mine_personal_facts("my pet rock collection sits on the shelf")
    _bad = [(s, a, v) for (s, a, v), f in eng.user_model.personal_facts.facts.items()
            if not f.superseded and a.startswith("cat") or "rock" in (s + a + v)]
    if _bad:
        print(f"[FAIL] common-noun 'rock collection' wrongly mined as pet: {_bad!r}")
        fails += 1
    else:
        print("[OK] 'my pet rock collection' not mined as a pet name")

    if fails:
        print(f"\n{fails} FAILED")
        raise SystemExit(1)
    print("\nALL PASSED")


def test_relation_reverse_lookup():
    run()


if __name__ == "__main__":
    run()
