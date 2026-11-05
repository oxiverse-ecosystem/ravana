import os, sys
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, PROJ + r"\ravana_ml\src", PROJ + r"\ravana\src", PROJ + r"\ravana-v2\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine
phrases = [
 "i have always believed that technology makes people more isolated",
 "i've always thought open source is the future",
 "i have long felt that privacy matters",
 "i have always believed in free software",
]
for i, ph in enumerate(phrases):
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="tb%d" % i)
    r = eng.process_turn(ph)
    bs = eng.belief_store.get_state()["beliefs"]
    print(ph, "->", r, "| beliefs:", list(bs.values())) 