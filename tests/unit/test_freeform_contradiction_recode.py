"""RED->GREEN: free-form contradiction recodes a HELD stance (no retraction keyword).

Round 2026-08-20T1229Z-followup, residual limitation #1 (carried from
2026-08-20T0701Z residual #1). Provenance keying was bridged earlier, but a
contradiction the user states WITHOUT a retraction keyword / "but" concession /
"can't" limitation cue ("actually i've gone off winter", "not all street art is
good", "they wear me out these days") was never detected, so the stale positive
stance the user already held persisted un-reversed.

This test fails BEFORE the capability: after such an utterance the held stance's
polarity stays positive (>= +0.9), because mine_stance_reversal found no cue to
fire on and the opinion miner wrote nothing. It passes AFTER: the held stance is
recalibrated toward the new (opposed) attitude, and the same-sign / no-signal
controls are left untouched (honest — no guessed reversal).

No reply string is asserted; every recoded value comes from the user's real words
(reassessment-affect lexicon) and the live stance store (resolve_topic bridge).
"""
import os, sys, tempfile, shutil
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine


def _build():
    d = tempfile.mkdtemp(prefix="ravana_contradict_")
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d), d


def _pol(eng, key):
    s = eng.user_model.opinions.stances.get(key)
    return None if s is None else s.polarity


def run():
    fails = 0

    # ---- 1) Free-form contradiction recodes a HELD stance (the bug) ----
    eng, d = _build()
    eng.process_turn("i really love street art, especially big murals on warehouse walls")
    held = _pol(eng, "street art")
    if held is None or held <= 0:
        print(f"[FAIL] positive street-art stance not established: {held}")
        fails += 1
    else:
        eng.process_turn(
            "i saw a mural downtown that was just tagged over a local business's "
            "sign — changed my mind, not all street art is good")
        after = _pol(eng, "street art")
        if after is None or after > 0.0:
            print(f"[FAIL] free-form contradiction did NOT recode held stance: "
                  f"before={held:+.3f} after={after}")
            fails += 1
        else:
            print(f"[OK] street-art recoded by contradiction: {held:+.3f} -> {after:+.3f}")
    shutil.rmtree(d, ignore_errors=True)

    # ---- 2) Broad co-mention bridging (winter -> silence stance) recodes ----
    eng, d = _build()
    eng.process_turn("i love the silence of deep winter, it's the only quiet i get")
    held_w = _pol(eng, "silence")
    eng.process_turn("actually i've gone off winter, the cold gets to me now")
    after_w = _pol(eng, "silence")
    if held_w is None or after_w is None or after_w > 0.0:
        print(f"[FAIL] winter free-form contradiction did not recode 'silence' "
              f"stance: before={held_w} after={after_w}")
        fails += 1
    else:
        print(f"[OK] winter->silence recoded: {held_w:+.3f} -> {after_w:+.3f}")
    shutil.rmtree(d, ignore_errors=True)

    # ---- 3) Same-sign reassessment must NOT recode (no false reversal) ----
    eng, d = _build()
    eng.process_turn("i love the silence of deep winter")
    held = _pol(eng, "silence")
    eng.process_turn("i still love winter, it's the best part of the year")
    after = _pol(eng, "silence")
    if after is None or after != held:
        print(f"[FAIL] same-sign reassessment wrongly recoded stance: "
              f"before={held} after={after}")
        fails += 1
    else:
        print(f"[OK] same-sign reassessment left stance untouched: {after:+.3f}")
    shutil.rmtree(d, ignore_errors=True)

    # ---- 4) No reassessment term -> honest no-op (no guessed reversal) ----
    eng, d = _build()
    eng.process_turn("i really love street art")
    held = _pol(eng, "street art")
    eng.process_turn("street art is interesting to think about")
    after = _pol(eng, "street art")
    if after is None or after != held:
        print(f"[FAIL] neutral utterance wrongly recoded stance: "
              f"before={held} after={after}")
        fails += 1
    else:
        print(f"[OK] neutral utterance left stance untouched: {after:+.3f}")
    shutil.rmtree(d, ignore_errors=True)

    # ---- 5) Idempotent: repeating the same contradiction does not double-flip ----
    eng, d = _build()
    eng.process_turn("i love the silence of deep winter")
    eng.process_turn("actually i've gone off winter, the cold gets to me now")
    p1 = _pol(eng, "silence")
    eng.process_turn("actually i've gone off winter, the cold gets to me now")
    p2 = _pol(eng, "silence")
    if p1 is None or p1 != p2:
        print(f"[FAIL] repeated contradiction not idempotent: {p1} vs {p2}")
        fails += 1
    else:
        print(f"[OK] repeated contradiction idempotent: {p1:+.3f}")
    shutil.rmtree(d, ignore_errors=True)

    if fails:
        print(f"\nRED: {fails} free-form contradiction recode checks failed")
        return 1
    print("\nGREEN: free-form contradictions recode held stances; same-sign/no-signal "
          "utterances are left untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())


def test_freeform_contradiction_recode():
    assert run() == 0
