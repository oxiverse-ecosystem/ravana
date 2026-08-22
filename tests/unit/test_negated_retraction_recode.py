"""RED->GREEN: a NEGATED reassessment recodes a HELD stance (online reversal).

Round 2026-08-21T2156Z defect D1. The free-form contradiction recoder
(mine_stance_reversal -> _assess_reversal_polarity) used DUMB substring
matching: it saw the word "hate" inside "i don't actually hate crowds" and
read it as a NEGATIVE reassessment term — the SAME sign as the held
"crowds" stance (-0.95) — so the contradiction was (wrongly) treated as
same-sign and the stale attitude persisted. A later "do you think i like
crowds or not" then reported "strongly against crowds", contradicting what
the user had just retracted.

The fix: detect a NEGATION token (not / n't / no longer / never / ...) within
a small window of the reassessment term and FLIP the estimated polarity, so
"i don't actually hate X" is read as a POSITIVE reversal of a held-negative
stance and recodes it. Fully online; the negation lexicon is seed structure
(RAVANA-expandable), no per-topic rule, no retraining.

No reply string is asserted; every recoded value comes from the user's real
words (reassessment affect) and the live stance store. The same-sign control
(affirmed "i still hate crowds") is left untouched (honest — no guessed
reversal from a negation alone).
"""
import os, sys, tempfile, shutil
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine


def _build():
    d = tempfile.mkdtemp(prefix="ravana_negretract_")
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d), d


def _pol(eng, key):
    s = eng.user_model.opinions.stances.get(key)
    return None if s is None else s.polarity


def run():
    fails = 0

    # ---- 1) Negated reassessment recodes a HELD negative stance (the bug) ----
    eng, d = _build()
    with open(os.devnull, "w") as dn, __import__("contextlib").redirect_stdout(dn):
        eng.process_turn("i hate crowds, they drain me completely, i avoid them")
    held = _pol(eng, "crowds")
    if held is None or held >= 0:
        print(f"[FAIL] negative crowds stance not established: {held}")
        fails += 1
    else:
        with open(os.devnull, "w") as dn, __import__("contextlib").redirect_stdout(dn):
            eng.process_turn(
                "wait, i think i misspoke earlier — i don't actually hate crowds, "
                "i just get tired at the end of a long day")
        after = _pol(eng, "crowds")
        if after is None or after <= 0.0 or after == held:
            print(f"[FAIL] negated retraction did NOT recode held stance: "
                  f"before={held:+.3f} after={after}")
            fails += 1
        else:
            print(f"[OK] negated retraction recoded crowds: "
                  f"{held:+.3f} -> {after:+.3f}")
    shutil.rmtree(d, ignore_errors=True)

    # ---- 2) Without the "misspoke" cue, plain negated phrase still recodes ----
    eng, d = _build()
    with open(os.devnull, "w") as dn, __import__("contextlib").redirect_stdout(dn):
        eng.process_turn("i hate crowds, they drain me completely")
    held = _pol(eng, "crowds")
    with open(os.devnull, "w") as dn, __import__("contextlib").redirect_stdout(dn):
        eng.process_turn("actually i don't really hate crowds, i just need quiet "
                         "at the end of a long day")
    after = _pol(eng, "crowds")
    if held is None or after is None or after <= 0.0:
        print(f"[FAIL] plain negated retraction not recoded: before={held} "
              f"after={after}")
        fails += 1
    else:
        print(f"[OK] plain negated retraction recoded crowds: {held:+.3f} -> {after:+.3f}")
    shutil.rmtree(d, ignore_errors=True)

    # ---- 3) Affirmed negation control: "i still hate crowds" stays negative ----
    eng, d = _build()
    with open(os.devnull, "w") as dn, __import__("contextlib").redirect_stdout(dn):
        eng.process_turn("i hate crowds, they drain me completely")
    held = _pol(eng, "crowds")
    with open(os.devnull, "w") as dn, __import__("contextlib").redirect_stdout(dn):
        eng.process_turn("i still hate crowds, they're the worst")
    after = _pol(eng, "crowds")
    if after is None or after != held:
        print(f"[FAIL] affirmed 'i still hate crowds' wrongly recoded: "
              f"before={held} after={after}")
        fails += 1
    else:
        print(f"[OK] affirmed hate left stance untouched: {after:+.3f}")
    shutil.rmtree(d, ignore_errors=True)

    if fails:
        print(f"\nRED: {fails} negated-retraction checks failed")
        return 1
    print("\nGREEN: negated reassessments ('i don't actually hate X') recode the "
          "held stance; affirmed negatives are left untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())


def test_negated_retraction_recode():
    assert run() == 0
