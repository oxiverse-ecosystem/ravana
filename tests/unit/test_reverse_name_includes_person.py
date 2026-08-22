"""RED->GREEN: reverse-name recall must include the person's name (round
2026-08-21T2156Z defect D5).

Prior bug: for a combined-attr relationship fact like
('i','neighbor mr. sato','keeps bees'), the reverse-name resolver (engine.py
_structured_recall) set _rel = " ".join(_attr_toks[:-1]), stripping the final
token (the name), then rendered only "your {_rel}." — so "who is Mr. Sato to me?"
answered "your neighbor mr.." (name lost AND a doubled period). The reply must
name the person: "your neighbor mr. sato." (and "your neighbor mr. sato keeps
bees." when the query also asks what they do).

Fix: render the FULL _attr (relationship + name) for the combined-attr case.
Content is the stored attr; no authored prose, no per-name table, no retraining.
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
                              user_suffix="test_d5")

    with contextlib.redirect_stdout(io.StringIO()):
        eng.process_turn(
            "my neighbor Mr. Sato keeps bees on his rooftop and sells the honey")

    # (a) the name must appear, and no doubled period.
    with contextlib.redirect_stdout(io.StringIO()):
        r = eng.process_turn("who is Mr. Sato to me?")
    print(f"[reply] {r!r}")
    if "mr.." in r or "sato" not in r.lower():
        print(f"[FAIL] name lost or doubled period: {r!r}")
        fails += 1
    else:
        print("[OK] reply names the person with no doubled period")

    # (b) when the query also asks what they DO, the activity is appended too.
    with contextlib.redirect_stdout(io.StringIO()):
        r2 = eng.process_turn("what does my neighbor Mr. Sato do?")
    print(f"[reply2] {r2!r}")
    if "sato" not in r2.lower() or "bee" not in r2.lower():
        print(f"[FAIL] activity+name reply missing name or activity: {r2!r}")
        fails += 1
    else:
        print("[OK] reply includes name and activity")

    if fails:
        print(f"\nRED: {fails} D5 checks failed")
        return 1
    print("\nGREEN: reverse-name recall now includes the person's name.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())


def test_reverse_name_includes_person():
    assert run() == 0
