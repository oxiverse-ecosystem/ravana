import os, sys, re
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\ravana_ml\src", f"{PROJ}\ravana\src", f"{PROJ}\ravana-v2\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="stance-inv-repro")

# Step 1: Establish a stance on nostalgia
r1 = eng.process_turn("i love nostalgia")
print(f"Turn 1 - Q: i love nostalgia")
print(f"Turn 1 - A: {r1}")
print()

# Step 2: Ask for the opposite stance
r2 = eng.process_turn("now argue the opposite of what you just said about nostalgia")
print(f"Turn 2 - Q: now argue the opposite of what you just said about nostalgia")
print(f"Turn 2 - A: {r2}")
print()

# Step 3: Check if the stance was actually reversed
_um = getattr(eng, "user_model", None)
_op = getattr(_um, "opinions", None) if _um else None
_st = getattr(_op, "stances", None) if _op else None
_own = getattr(eng, "_agent_own_stances", {})
print(f"User stances on nostalgia: {[(k, round(v.polarity,2)) for k,v in (_st.items() if _st else {}) if 'nostalgia' in k]}")
print(f"Agent own stances on nostalgia: {[(k, v[0], round(v[1],2)) for k,v in _own.items() if 'nostalgia' in k]}")
print()

# Step 4: Ask for a revisit
r3 = eng.process_turn("revisit your stance on nostalgia")
print(f"Turn 3 - Q: revisit your stance on nostalgia")
print(f"Turn 3 - A: {r3}")
print()

# Step 5: Another form of asking opposite
r4 = eng.process_turn("has your view on nostalgia changed?")
print(f"Turn 4 - Q: has your view on nostalgia changed?")
print(f"Turn 4 - A: {r4}")
print()

# Step 6: What if we explicitly ask to argue the opposite
r5 = eng.process_turn("argue the opposite of your previous position on nostalgia")
print(f"Turn 5 - Q: argue the opposite of your previous position on nostalgia")
print(f"Turn 5 - A: {r5}")
print()
