"""RED->GREEN: Natural language contradiction-revision for stance inversion.

Round 2026-09-21T1954Z (FIX-RV-01): the user says "i love pineapple on pizza"
then "actually i think pineapple on pizza is disgusting". Previously RAVANA
just said "ok, noted" — it did NOT invert the prior stance.

The fix widens the contradiction-revision regex to cover natural contradiction
signals: "actually", "on second thought", "i changed my mind", "i was wrong",
"i take it back", "never mind", "i don't think so anymore".
"""
import os, sys, io, contextlib, tempfile, shutil
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine


def _build():
    d = tempfile.mkdtemp(prefix="ravana_nc_")
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d), d


def run():
    fails = 0

    # ---- 1) "actually" inverts a held positive stance ----
    eng, d = _build()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        eng.process_turn("i love pineapple on pizza")
    with contextlib.redirect_stdout(buf):
        r2 = eng.process_turn("actually i think pineapple on pizza is disgusting")
    s = eng.user_model.opinions.stances.get("pineapple")
    if s is None or s.polarity >= 0:
        print(f"[FAIL] stance NOT inverted: {s.polarity if s else 'None'}")
        fails += 1
    else:
        print(f"[OK] stance inverted: +{s.prior_polarity:.2f} -> {s.polarity:.2f}")
    r2_l = r2.lower()
    if "pineapple" not in r2_l:
        print(f"[FAIL] reply missing topic: {r2!r}")
        fails += 1
    else:
        print(f"[OK] reply mentions topic")
    shutil.rmtree(d, ignore_errors=True)

    # ---- 2) "on second thought" inverts a held negative stance ----
    eng2, d2 = _build()
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        eng2.process_turn("i hate cold weather")
    with contextlib.redirect_stdout(buf2):
        r3 = eng2.process_turn("on second thought, cold weather is fine")
    s2 = eng2.user_model.opinions.stances.get("cold weather")
    if s2 is None or s2.polarity <= 0:
        print(f"[FAIL] 'on second thought' NOT inverted: {s2.polarity if s2 else 'None'}")
        fails += 1
    else:
        print(f"[OK] 'on second thought' inverted: {s2.prior_polarity:.2f} -> {s2.polarity:.2f}")
    shutil.rmtree(d2, ignore_errors=True)

    if fails:
        print(f"\nRED: {fails} checks failed")
        return 1
    print("\nGREEN: natural contradiction-revision works")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())


def test_natural_contradiction_stance_inversion():
    assert run() == 0
