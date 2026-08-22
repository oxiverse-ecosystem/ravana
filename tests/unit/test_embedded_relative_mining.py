"""RED->GREEN: hyphenated-kin and embedded-lowercase-name relationship
disclosures are mined AND reverse-lookupable by entity name.

Round 2026-08-21T2156Z defect D2. Two real disclosures were DROPPED by the
relationship miner, so a later reverse-name query ("who is X to me") had
nothing to retrieve and fell through to an identity self-intro:

  - "my great-aunt Hortense folds a thousand paper cranes every winter"
    -> the "my <kin>" regex captured only "great" (or left "great-aunt"
       unmapped by relation_of), so the disclosure was skipped; "who is
       hortense to me?" returned RAVANA's own intro instead.
  - "my friend wren, she's a ceramicist"
    -> embedded relative clause with a LOWERCASE name and a pronoun-copula
       ("she's a ...") matched no verb class, so the name-less fallback
       fired and nothing was stored; "who is wren to me?" returned the
       identity blurb.

Fix: (a) normalize multi-word kin modifiers ("great-aunt" / "great aunt") to
a single relation head via the shared relation_attrs lexicon; (b) capture an
embedded relative clause ("<name>, she's a <descriptor>") as the relationship
+ name + value when no activity verb is recognized. Both are structural
(generalize to ANY relationship word RAVANA has learned) and online — no
per-entity branch, no retraining, no authored reply. Content comes from the
user's own words.

Reply content is driven by REAL mined state, never authored prose.
"""
import os, sys, io, contextlib
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine


def run():
    fails = 0
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="test_embedrel")
    disclosures = [
        "my great-aunt Hortense folds a thousand paper cranes every winter",
        "my friend wren, she's a ceramicist",
    ]
    for dd in disclosures:
        with contextlib.redirect_stdout(io.StringIO()):
            eng.process_turn(dd)

    rel_facts = [(a, f.value) for (s, a, _), f in
                 eng.user_model.personal_facts.facts.items()
                 if s == "i" and not f.superseded]
    # great-aunt disclosure is NORMALIZED to its head ("great-aunt" -> "aunt",
    # per defect D2's modifier-collapse design) and mined under the combined
    # attr "aunt hortense" with the activity clause as the value.
    assert ("aunt hortense",) in [(a,) for a, _ in rel_facts], (
        f"great-aunt disclosure not mined: {rel_facts}")
    # friend+embedded disclosure mined under combined attr "friend wren"
    assert ("friend wren",) in [(a,) for a, _ in rel_facts], (
        f"embedded 'friend wren' disclosure not mined: {rel_facts}")
    print("[OK] great-aunt + embedded friend disclosures mined")

    checks = [
        # bare "who is X" -> recaller renders "your {rel}." (relationship head
        # only; the name is the query target, not echoed back).
        ("who is hortense to me?", ("aunt", "")),
        ("who is wren to me?", ("friend", "")),
        # "...and what does she do?" -> appends the stored value.
        ("who is wren to me and what does she do?", ("friend", "ceramicist")),
    ]
    for q, (needle1, needle2) in checks:
        eng._last_strategy = None
        with contextlib.redirect_stdout(io.StringIO()):
            r = eng.process_turn(q)
        reply = r if isinstance(r, str) else str(r)
        eng.stop_background_learning()
        if eng._last_strategy != "structured_recall":
            print(f"[FAIL] '{q}' expected structured_recall, got "
                  f"{eng._last_strategy}: {reply!r}")
            fails += 1
            continue
        if needle1 not in reply.lower():
            print(f"[FAIL] '{q}' reply missing '{needle1}': {reply!r}")
            fails += 1
            continue
        if needle2 and needle2 not in reply.lower():
            print(f"[FAIL] '{q}' reply missing '{needle2}': {reply!r}")
            fails += 1
            continue
        print(f"[OK] '{q}' -> {reply!r}")

    if fails:
        print(f"\n{fails} FAILED")
        raise SystemExit(1)
    print("\nALL PASSED")


def test_embedded_relative_mining():
    run()


if __name__ == "__main__":
    run()
