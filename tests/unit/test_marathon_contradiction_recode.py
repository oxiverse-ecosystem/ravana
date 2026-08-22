"""RED->GREEN: free-form contradiction recodes a HELD stance when the user
negates a BASE sentiment verb, even when the same utterance states a fresh
positive preference (round 2026-08-21T2156Z marathon-gap fix).

Prior behavior: _assess_reversal_polarity only knew reassessment IDIOMS
("hate" / "gone off" / "come around to" ...). A base positive verb ("love" /
"like" / "enjoy") used in a contradiction ("i don't actually LOVE marathon
running, i prefer short sprints") was invisible, so the free-form recode never
fired and a held +0.95 stance persisted. Worse, the co-mentioned fresh
preference "prefer" (a longer non-negated base-positive term) OUTRANKED the
negated "love" under the longest-term-wins rule, so even adding base verbs
left the recode suppressed.

Fix: (a) add base sentiment-verb sets (_REASSESS_SENT_POS/_NEG) so a negated
base verb is recognized; (b) a NEGATED term is a contradiction signal and must
outrank any non-negated term regardless of length, so "don't actually love X,
i prefer Y" reads as a NEGATIVE reassessment of the held-positive stance, not
as the +0.8 of the fresh "prefer".

Verified behavior (real engine, no authored prose): "i love running marathons"
-> running-marathons +0.95; retraction "i don't actually love marathon running,
i prefer short sprints" -> recoded to -0.275 with ack "you've changed your
mind about running marathons". The D1 control ("i don't actually hate crowds")
still recodes a held-negative stance to positive.

No reply string is asserted; every recoded value comes from the user's real
words (reassessment affect) and the live stance store.
"""
import os, sys, tempfile, shutil
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine


def _build():
    d = tempfile.mkdtemp(prefix="ravana_marathon_")
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d), d


def _pol(eng, key):
    s = eng.user_model.opinions.stances.get(key)
    return None if s is None else s.polarity


def run():
    fails = 0

    # 1) Negated BASE-positive verb recodes a HELD positive stance, even with a
    #    co-mentioned fresh preference in the same turn.
    eng, d = _build()
    with open(os.devnull, "w") as dn, __import__("contextlib").redirect_stdout(dn):
        eng.process_turn("i love running marathons, the longer the better, it clears my head")
    held = _pol(eng, "running marathons")
    if held is None or held <= 0:
        print(f"[FAIL] base-positive stance not established: {held}")
        fails += 1
    else:
        with open(os.devnull, "w") as dn, __import__("contextlib").redirect_stdout(dn):
            eng.process_turn(
                "wait, i think i misspoke earlier — i don't actually love marathon "
                "running, it wrecks my knees, i prefer short sprints")
        after = _pol(eng, "running marathons")
        if after is None or after >= 0.0 or after == held:
            print(f"[FAIL] negated-base-positive retraction did NOT recode held "
                  f"stance: before={held:+.3f} after={after}")
            fails += 1
        else:
            print(f"[OK] negated 'love' recoded running-marathons: "
                  f"{held:+.3f} -> {after:+.3f}")
    shutil.rmtree(d, ignore_errors=True)

    # 2) D1 control still holds: negated base-NEGATIVE verb recodes a held
    #    negative stance to positive.
    eng, d = _build()
    with open(os.devnull, "w") as dn, __import__("contextlib").redirect_stdout(dn):
        eng.process_turn("i hate crowds, they drain me completely")
    held = _pol(eng, "crowds")
    with open(os.devnull, "w") as dn, __import__("contextlib").redirect_stdout(dn):
        eng.process_turn("i don't actually hate crowds, i just need quiet at the end of a long day")
    after = _pol(eng, "crowds")
    if held is None or after is None or after <= 0.0:
        print(f"[FAIL] D1 negated-base-negative control broken: before={held} after={after}")
        fails += 1
    else:
        print(f"[OK] D1 negated 'hate' still recodes crowds: {held:+.3f} -> {after:+.3f}")
    shutil.rmtree(d, ignore_errors=True)

    # 3) Non-negated base verb with no contradiction leaves the stance alone.
    eng, d = _build()
    with open(os.devnull, "w") as dn, __import__("contextlib").redirect_stdout(dn):
        eng.process_turn("i love running marathons")
    held = _pol(eng, "running marathons")
    with open(os.devnull, "w") as dn, __import__("contextlib").redirect_stdout(dn):
        eng.process_turn("i actually love winter, it's the best")
    after = _pol(eng, "running marathons")
    if after != held:
        print(f"[FAIL] non-negated 'love winter' wrongly moved running-marathons: "
              f"{held} -> {after}")
        fails += 1
    else:
        print(f"[OK] non-negated base verb left held stance untouched: {after:+.3f}")
    shutil.rmtree(d, ignore_errors=True)

    if fails:
        print(f"\nRED: {fails} marathon-gap checks failed")
        return 1
    print("\nGREEN: negated base sentiment verbs recode the held stance; a "
          "co-mentioned fresh preference does not suppress the reversal; D1 "
          "negated-idiom control still holds; non-negated verbs are left alone")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())


def test_marathon_contradiction_recode():
    assert run() == 0
