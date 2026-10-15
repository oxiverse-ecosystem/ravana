import os, sys, io, contextlib, tempfile, shutil
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

def _new_engine(suffix):
    import glob, os as _os
    for f in (glob.glob(f"weights/ravana_weights{suffix}.pkl")
              + glob.glob(f"weights/ravana_usermodel{suffix}.pkl")):
        _os.remove(f)
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)

eng = _new_engine("test_dA_debug")
with contextlib.redirect_stdout(io.StringIO()):
    eng.process_turn("i love knitting, it's the most calming thing in the world.")
    eng.process_turn("actually, you know what, i've come to hate knitting — it just makes my hands ache.")
_key = eng.user_model.opinions.resolve_topic("knitting")
_s = eng.user_model.opinions.stances.get(_key)
print("STANCE:", _s.polarity if _s else None)
print("STANCE key:", _key)
print("All stances:", {k: v.polarity for k, v in eng.user_model.opinions.stances.items()})

ans = eng._user_stance_reply("do you still think i love knitting?")
print("USER_STANCE_REPLY:", repr(ans))

with contextlib.redirect_stdout(io.StringIO()):
    full = eng.process_turn("do you still think i love knitting?")
print("FULL TURN:", repr(full))
