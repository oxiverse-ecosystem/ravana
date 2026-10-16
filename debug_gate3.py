import os, sys
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src", f"{PROJ}\\ravana-v2\\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine
import numpy as np
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="test_gate_debug3")

# Check what _ground_query returns
q1 = "do you ever feel curious about open source"
g1 = eng._ground_query(q1)
print(f"_ground_query({q1!r}) = {g1}")

q2 = "what is sleep"
g2 = eng._ground_query(q2)
print(f"_ground_query({q2!r}) = {g2}")

q3 = "what do you dream about"
g3 = eng._ground_query(q3)
print(f"_ground_query({q3!r}) = {g3}")

# Check "open" vs "curious"
v1 = eng._glove_vector("open")
v2 = eng._glove_vector("curious")
if v1 is not None and v2 is not None:
    sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    print(f"open vs curious: {sim:.4f}")

# Check "source" vs "curious"
v1 = eng._glove_vector("source")
v2 = eng._glove_vector("curious")
if v1 is not None and v2 is not None:
    sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    print(f"source vs curious: {sim:.4f}")

# Check "open" vs words in the answer
answer = "i don't really have a solid grasp on curious so far"
for aw in ["curious", "solid", "grasp", "really"]:
    v1 = eng._glove_vector("open")
    v2 = eng._glove_vector(aw)
    if v1 is not None and v2 is not None:
        sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
        print(f"open vs {aw}: {sim:.4f}")
