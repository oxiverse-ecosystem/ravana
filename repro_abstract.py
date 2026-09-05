import os, sys
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\ravana_ml\src", f"{PROJ}\ravana\src", f"{PROJ}\ravana-v2\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="repro_abstract")
queries = [
    "tell me about love",
    "what is the meaning of life",
    "explain consciousness",
    "tell me about gravity",
    "what is photosynthesis",
]
for q in queries:
    r = eng.process_turn(q)
    print(f"Q: {q}")
    print(f"A: {r.get('reply', r) if isinstance(r, dict) else r}")
    print("---")
