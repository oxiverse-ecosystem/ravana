import os, sys
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, PROJ + r"\ravana_ml\src", PROJ + r"\ravana\src", PROJ + r"\ravana-v2\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="testbelief99")
r = eng.process_turn("i have always believed that technology makes people more isolated")
print("REPLY:", r)
print("BELIEFS:", eng.belief_store.get_state()["beliefs"])
eng.save() 