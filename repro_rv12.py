"""Repro for FIX-RV-12: modifier-heavy stance topics.

Run with the PYTHONPATH set to THIS worktree so the patched code is imported
(the committed tests hardcode the main repo path and would import main instead).
"""
import os, sys
os.environ["RAVANA_OFFLINE"] = "1"
WS = os.path.dirname(os.path.abspath(__file__))
for p in (WS, f"{WS}\\ravana_ml\\src", f"{WS}\\ravana\\src"):
    sys.path.insert(0, p)
import ravana.chat.user_model as um_mod
print("user_model loaded from:", um_mod.__file__)

from ravana.chat.user_model import UserModel

CASES = [
    # (raw topic phrase, expected, why)
    ("cooking earlier in the morning", "cooking", "trailing temporal + PP"),
    ("jazz music is relaxing", "jazz music", "trailing copula residue"),
    ("open source software", "open source software", "regression: must be unchanged"),
    ("dear old jazz clubs", "jazz clubs", "leading modifiers"),
    ("letterpress printing", "letterpress printing", "regression: must be unchanged"),
    ("petrichor after a storm", "petrichor", "trailing prepositional tail"),
    ("cold water swimming jumping", "cold water swimming", "second activity cut"),
    ("the solitude of the lighthouse", "solitude", "PP cut"),
    ("small talk at the village market", "small talk", "PP cut"),
    ("people who talk in theatres", "people who talk", "relative bridge"),
]

def main():
    um = UserModel.__new__(UserModel)
    fails = []
    for phrase, expected, why in CASES:
        got = um._opinion_topic(phrase)
        ok = (got == expected)
        print(f"{'ok ' if ok else 'FAIL'} {phrase!r} -> {got!r} (expected {expected!r}) [{why}]")
        if not ok:
            fails.append((phrase, got, expected))
    print()
    if fails:
        print(f"RED: {len(fails)}/{len(CASES)} cases fail")
        return 1
    print(f"GREEN: all {len(CASES)} topic-extraction cases pass")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
