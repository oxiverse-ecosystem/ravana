import os, sys
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src", f"{PROJ}\\ravana-v2\\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine
print("MRO:", [c.__name__ for c in CognitiveChatEngine.__mro__])
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="test_mro")
print("_relevance_gate defined in:", eng._relevance_gate.__qualname__)
