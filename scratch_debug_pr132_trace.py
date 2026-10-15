import os, sys, io, contextlib, tempfile, shutil
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

def _build():
    d = tempfile.mkdtemp(prefix="ravana_contradict_")
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d), d

eng, d = _build()
eng.process_turn("i really love street art, especially big murals on warehouse walls")
print("Stances after turn 1:", {k: v.polarity for k, v in eng.user_model.opinions.stances.items() if 'street' in k or 'mural' in k or 'art' in k})

# Check what _turn_start_stance_keys contains
eng._turn_start_stance_keys = set(
    getattr(getattr(eng, "user_model", None), "opinions", None).stances.keys()
    if getattr(getattr(eng, "user_model", None), "opinions", None) is not None
    else set()
)
print("turn_start_keys:", eng._turn_start_stance_keys)

# Now process the contradiction
r = eng.process_turn("i saw a mural downtown that was just tagged over a local business's sign — changed my mind, not all street art is good")
print("Reply:", repr(r))
print("Strategy:", eng._last_strategy)
print("Stances after turn 2:", {k: v.polarity for k, v in eng.user_model.opinions.stances.items() if 'street' in k or 'mural' in k or 'art' in k})

# Check if _route_own_stance_inversion is being called
import types
orig = eng._route_own_stance_inversion
def traced(inp):
    result = orig(inp)
    print(f"  [TRACE] _route_own_stance_inversion({inp!r}) -> {result!r}")
    return result
eng._route_own_stance_inversion = traced

orig2 = eng._route_own_stance_revisit
def traced2(inp):
    result = orig2(inp)
    print(f"  [TRACE] _route_own_stance_revisit({inp!r}) -> {result!r}")
    return result
eng._route_own_stance_revisit = traced2

r2 = eng.process_turn("i saw a mural downtown that was just tagged over a local business's sign — changed my mind, not all street art is good")
print("Reply2:", repr(r2))
print("Strategy2:", eng._last_strategy)
