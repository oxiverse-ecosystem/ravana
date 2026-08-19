"""RED->GREEN: RAVANA records its own stances durably and answers a
revisit query ("do you still feel that way about X?") from that record.

Round 2026-08-19T0625Z limitation #2: opinion questions were answered but
never persisted, so "do you still feel that way about X?" could not be
answered from a recorded stance. This test fails before the capability exists
(_agent_own_stances store absent, _route_own_stance_revisit absent) and passes
after: the stance is written to a persisted store and survives a save/load, and
a later revisit question reports the recorded orientation.

The reply is driven by REAL recorded state (the topic target + the stored
polarity word / confidence), never authored prose per topic.
"""
import os, sys, io, contextlib, tempfile, shutil
from pathlib import Path
os.environ["RAVANA_OFFLINE"] = "1"
# Derive repo root from this test file's location
PROJ = Path(__file__).resolve().parent.parent.parent
for p in (str(PROJ), str(PROJ / "ravana_ml" / "src"), str(PROJ / "ravana" / "src")):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine


def _build():
    d = tempfile.mkdtemp(prefix="ravana_own_stance_")
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d), d


def run():
    fails = 0

    # ---- 1) Real stance question records durably into _agent_own_stances ----
    eng, d = _build()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = eng.process_turn("what do you think about open source")
    reply = r if isinstance(r, str) else r.get("reply", "")
    # The capability writes the stance it computed into the persisted store.
    rec = eng._agent_own_stances.get("open source")
    if not rec:
        print("[FAIL] open-source stance was NOT recorded into _agent_own_stances")
        fails += 1
    else:
        # The recorded polarity word must be the REAL seeded value's word.
        if "value" not in rec[0] and "strongly value" not in rec[0]:
            print(f"[FAIL] recorded word not a real value word: {rec!r}")
            fails += 1
        else:
            print(f"[OK] stance recorded: {rec}")

    # ---- 2) The record survives a save/load roundtrip (durable) ----
    try:
        eng.save()
        eng2 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d)
        eng2.load()
        rec2 = eng2._agent_own_stances.get("open source")
        if not rec2:
            print("[FAIL] recorded stance was lost on save/load")
            fails += 1
        else:
            print(f"[OK] stance survived save/load: {rec2}")
    except Exception as e:
        print(f"[FAIL] save/load roundtrip raised: {e}")
        fails += 1
    finally:
        shutil.rmtree(d, ignore_errors=True)

    # ---- 3) Revisit query answers from the RECORDED stance ----
    eng3, d3 = _build()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        eng3.process_turn("what do you think about open source")
        rev = eng3.process_turn("do you still feel that way about open source?")
    rev_reply = rev if isinstance(rev, str) else rev.get("reply", "")
    rev_l = rev_reply.lower()
    # Must cite the recorded topic and affirm continuity from the record, not
    # fall through to a recomputed provisional "still forming a view" line.
    if "open source" not in rev_l:
        print(f"[FAIL] revisit reply does not name the recorded topic: {rev_reply!r}")
        fails += 1
    elif "still forming a view" in rev_l:
        print(f"[FAIL] revisit recomputed a provisional stance instead of using "
              f"the record: {rev_reply!r}")
        fails += 1
    else:
        print(f"[OK] revisit answered from record: {rev_reply[:90]!r}")
    shutil.rmtree(d3, ignore_errors=True)

    # ---- 4) Revisit on a topic with NO record is honest (no fabrication) ----
    eng4, d4 = _build()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rev2 = eng4.process_turn("do you still feel that way about quokkas?")
    rev2_reply = rev2 if isinstance(rev2, str) else rev2.get("reply", "")
    if "recorded view" not in rev2_reply.lower() and "don't actually have" not in rev2_reply.lower():
        print(f"[FAIL] revisit with no record did not answer honestly: {rev2_reply!r}")
        fails += 1
    else:
        print(f"[OK] no-record revisit honest: {rev2_reply[:80]!r}")
    shutil.rmtree(d4, ignore_errors=True)

    if fails:
        print(f"\nRED: {fails} own-stance persistence checks failed")
        return 1
    print("\nGREEN: RAVANA records own stances durably and answers revisit from the record")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())


def test_own_stance_persistence():
    assert run() == 0
