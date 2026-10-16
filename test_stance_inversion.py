import os, sys, io, contextlib, tempfile, shutil
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

def _build():
    d = tempfile.mkdtemp(prefix="ravana_d4_")
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d), d

# Test 1: "changed my mind" with positive new opinion
eng, d = _build()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    eng.process_turn("i hate silence")
s_before = eng.user_model.opinions.stances.get("silence")
print(f"[OK] initial stance on 'silence': {s_before.polarity:.2f}" if s_before else "[FAIL] no stance")

with contextlib.redirect_stdout(buf):
    r = eng.process_turn("changed my mind — silence is actually wonderful")
s_after = eng.user_model.opinions.stances.get("silence")
if s_after:
    print(f"stance after: polarity={s_after.polarity}, prior={s_after.prior_polarity}")
    print(f"  type(polarity)={type(s_after.polarity)}")
    if s_after.polarity is not None and s_after.polarity > 0:
        print("[OK] stance moved toward positive")
    else:
        print(f"[FAIL] stance not moved toward positive: {s_after.polarity}")
else:
    print("[FAIL] stance lost after contradiction")

# Check reply
print(f"reply: {r!r}")
