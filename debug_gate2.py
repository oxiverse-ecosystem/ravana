import os, sys
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src", f"{PROJ}\\ravana-v2\\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine
import numpy as np
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="test_gate_debug2")

# Test case: "open source" query with "curious" answer
subj = "open source"
answer = "i don't really have a solid grasp on curious so far"
subj_vec = eng._glove_vector(subj)
print(f"subj: {subj}")
if subj_vec is None:
    print("  subj_vec is None!")
else:
    for aw in ["curious", "solid", "grasp", "really"]:
        av = eng._glove_vector(aw)
        if av is not None:
            sim = float(np.dot(subj_vec, av) / (np.linalg.norm(subj_vec) * np.linalg.norm(av) + 1e-9))
            print(f"  {subj} vs {aw}: {sim:.4f}")
        else:
            print(f"  {subj} vs {aw}: None")

# Also check "open source" vs "curious" directly
print(f"\nopen source vs curious: ", end="")
v1 = eng._glove_vector("open")
v2 = eng._glove_vector("source")
v3 = eng._glove_vector("curious")
if v1 is not None and v3 is not None:
    print(f"open vs curious: {float(np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3) + 1e-9)):.4f}")
if v2 is not None and v3 is not None:
    print(f"source vs curious: {float(np.dot(v2, v3) / (np.linalg.norm(v2) * np.linalg.norm(v3) + 1e-9)):.4f}")
