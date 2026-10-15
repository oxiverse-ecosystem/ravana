import os, sys, io, contextlib, tempfile, shutil
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

# ---- 1) Free-form contradiction recodes a HELD stance (the bug) ----
eng, d = _build()
eng.process_turn("i really love street art, especially big murals on warehouse walls")
held = _pol(eng, "street art")
print("PART 1 held:", held)
eng.process_turn("i saw a mural downtown that was just tagged over a local business's sign — changed my mind, not all street art is good")
after = _pol(eng, "street art")
print("PART 1 after:", after)
if after is None or after > 0.0:
    print(f"[FAIL] free-form contradiction did NOT recode held stance: before={held:+.3f} after={after}")
else:
    print(f"[OK] street-art recoded by contradiction: {held:+.3f} -> {after:+.3f}")
shutil.rmtree(d, ignore_errors=True)

# ---- 2) Broad co-mention bridging (winter -> silence stance) recodes ----
eng, d = _build()
eng.process_turn("i love the silence of deep winter, it's the only quiet i get")
held_w = _pol(eng, "silence")
eng.process_turn("actually i've gone off winter, the cold gets to me now")
after_w = _pol(eng, "silence")
print("PART 2 held_w:", held_w, "after_w:", after_w)
if held_w is None or after_w is None or after_w > 0.0:
    print(f"[FAIL] winter free-form contradiction did not recode 'silence' stance: before={held_w} after={after_w}")
else:
    print(f"[OK] winter->silence recoded: {held_w:+.3f} -> {after_w:+.3f}")
shutil.rmtree(d, ignore_errors=True)

# ---- 5) Idempotent: repeating the same contradiction does not double-flip ----
eng, d = _build()
eng.process_turn("i love the silence of deep winter")
eng.process_turn("actually i've gone off winter, the cold gets to me now")
p1 = _pol(eng, "silence")
eng.process_turn("actually i've gone off winter, the cold gets to me now")
p2 = _pol(eng, "silence")
print("PART 5 p1:", p1, "p2:", p2)
if p1 is None or p1 != p2:
    print(f"[FAIL] repeated contradiction not idempotent: {p1} vs {p2}")
else:
    print(f"[OK] repeated contradiction idempotent: {p1:+.3f}")
shutil.rmtree(d, ignore_errors=True)
