import os, sys, io, contextlib, tempfile, shutil
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

def _build():
    d = tempfile.mkdtemp(prefix="ravana_own_stance_")
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d), d

# ---- 1) Real stance question records durably into _agent_stances ----
eng, d = _build()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    r = eng.process_turn("what do you think about open source")
reply = r if isinstance(r, str) else r.get("reply", "")
print("PART 1 reply:", repr(reply))
rec = eng._agent_stances.get("open source")
print("PART 1 rec:", rec)
if not rec:
    print("[FAIL] open-source stance was NOT recorded into _agent_stances")
else:
    if "value" not in rec[0] and "strongly value" not in rec[0]:
        print(f"[FAIL] recorded word not a real value word: {rec!r}")
    else:
        print(f"[OK] stance recorded: {rec}")

# ---- 2) The record survives a save/load roundtrip (durable) ----
try:
    eng.save()
    eng2 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d)
    eng2.load()
    rec2 = eng2._agent_stances.get("open source")
    print("PART 2 rec2:", rec2)
    if not rec2:
        print("[FAIL] recorded stance was lost on save/load")
    else:
        print(f"[OK] stance survived save/load: {rec2}")
except Exception as e:
    print(f"[FAIL] save/load roundtrip raised: {e}")
finally:
    shutil.rmtree(d, ignore_errors=True)

# ---- 3) Revisit query answers from the RECORDED stance ----
eng3, d3 = _build()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    eng3.process_turn("what do you think about open source")
    rev = eng3.process_turn("do you still feel that way about open source?")
rev_reply = rev if isinstance(rev, str) else rev.get("reply", "")
print("PART 3 reply:", repr(rev_reply))
rev_l = rev_reply.lower()
if "open source" not in rev_l:
    print(f"[FAIL] revisit reply does not name the recorded topic: {rev_reply!r}")
elif "still forming a view" in rev_l:
    print(f"[FAIL] revisit recomputed a provisional stance instead of using the record: {rev_reply!r}")
else:
    print(f"[OK] revisit answered from record: {rev_reply[:90]!r}")
shutil.rmtree(d3, ignore_errors=True)

# ---- 4) Revisit on a topic with NO record is honest (no fabrication) ----
eng4, d4 = _build()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rev2 = eng4.process_turn("do you still feel that way about quokkas?")
rev2_reply = rev2 if isinstance(rev2, str) else rev2.get("reply", "")
print("PART 4 reply:", repr(rev2_reply))
if "recorded view" not in rev2_reply.lower() and "don't actually have" not in rev2_reply.lower():
    print(f"[FAIL] revisit with no record did not answer honestly: {rev2_reply!r}")
else:
    print(f"[OK] no-record revisit honest: {rev2_reply[:80]!r}")
shutil.rmtree(d4, ignore_errors=True)
