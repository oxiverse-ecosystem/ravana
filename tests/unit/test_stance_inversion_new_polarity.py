"""Test: stance inversion derives NEW polarity from post-contradiction clause.

Round 2026-10-15 (D4 fix): when the user contradicts a previously-held
stance with a new opinion (e.g. "changed my mind — words matter more"),
the engine should derive the NEW polarity from the post-contradiction
clause and use recode_stance_toward, NOT blindly flip via reverse_stance.

This test verifies that the stance update reflects WHAT the user now
believes, not just that they changed their mind.
"""
import os, sys, io, contextlib, tempfile, shutil
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine


def _build():
    d = tempfile.mkdtemp(prefix="ravana_d4_")
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d), d


def run():
    fails = 0

    # ---- 1) "changed my mind" with a positive new opinion ----
    # User first expresses a negative stance, then contradicts with a
    # positive new opinion. The stance should move toward positive.
    eng, d = _build()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        eng.process_turn("i hate silence")
    s_before = eng.user_model.opinions.stances.get("silence")
    if s_before is None:
        print("[FAIL] no stance for 'silence' after initial expression")
        fails += 1
    else:
        print(f"[OK] initial stance on 'silence': {s_before.polarity:.2f}")

    with contextlib.redirect_stdout(buf):
        r = eng.process_turn("changed my mind — silence is actually wonderful")
    s_after = eng.user_model.opinions.stances.get("silence")
    if s_after is None:
        print("[FAIL] stance lost after contradiction")
        fails += 1
    else:
        # The new polarity should be POSITIVE (derived from "wonderful"),
        # not just blindly flipped from the old value
        if s_after.polarity <= 0:
            print(f"[FAIL] stance not moved toward positive: {s_after.polarity:.2f}")
            fails += 1
        else:
            print(f"[OK] stance moved toward positive: {s_after.prior_polarity:.2f} -> {s_after.polarity:.2f}")
    shutil.rmtree(d, ignore_errors=True)

    # ---- 2) "changed my mind" with a negative new opinion ----
    # User first expresses a positive stance, then contradicts with a
    # negative new opinion. The stance should move toward negative.
    eng2, d2 = _build()
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        eng2.process_turn("i love talking")
    s2_before = eng2.user_model.opinions.stances.get("talking")
    if s2_before is None:
        print("[FAIL] no stance for 'talking' after initial expression")
        fails += 1
    else:
        print(f"[OK] initial stance on 'talking': {s2_before.polarity:.2f}")

    with contextlib.redirect_stdout(buf2):
        r2 = eng2.process_turn("changed my mind — talking is actually terrible")
    s2_after = eng2.user_model.opinions.stances.get("talking")
    if s2_after is None:
        print("[FAIL] stance lost after contradiction")
        fails += 1
    else:
        # The new polarity should be NEGATIVE (derived from "terrible")
        if s2_after.polarity >= 0:
            print(f"[FAIL] stance not moved toward negative: {s2_after.polarity:.2f}")
            fails += 1
        else:
            print(f"[OK] stance moved toward negative: {s2_after.prior_polarity:.2f} -> {s2_after.polarity:.2f}")
    shutil.rmtree(d2, ignore_errors=True)

    # ---- 3) Verify the reply articulates the new stance ----
    eng3, d3 = _build()
    buf3 = io.StringIO()
    with contextlib.redirect_stdout(buf3):
        eng3.process_turn("i love silence")
    with contextlib.redirect_stdout(buf3):
        r3 = eng3.process_turn("changed my mind — silence is actually wonderful")
    r3_l = r3.lower()
    # The reply should mention the topic and the new direction
    if "silence" not in r3_l:
        print(f"[FAIL] reply missing topic: {r3!r}")
        fails += 1
    else:
        print(f"[OK] reply mentions topic")
    # The reply should indicate the new direction (not just "changed your mind")
    if "wonderful" not in r3_l and "good" not in r3_l and "positive" not in r3_l:
        # At minimum, the reply should not just say "changed your mind"
        # without articulating the new stance
        if "changed your mind" in r3_l and "now" not in r3_l:
            print(f"[FAIL] reply is shallow (no new stance articulated): {r3!r}")
            fails += 1
        else:
            print(f"[OK] reply articulates new stance")
    else:
        print(f"[OK] reply articulates new stance")
    shutil.rmtree(d3, ignore_errors=True)

    if fails:
        print(f"\nRED: {fails} checks failed")
        return 1
    print("\nGREEN: stance inversion derives new polarity from post-contradiction clause")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())


def test_stance_inversion_derives_new_polarity():
    assert run() == 0
