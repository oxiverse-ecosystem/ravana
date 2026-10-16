import os, sys
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src", f"{PROJ}\\ravana-v2\\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="test_gate_debug")
subj = "sleep"
answer = "car engine repair requires mechanical tools and oil changes"
subj_vec = eng._glove_vector(subj)
print(f"subj: {subj}")
for aw in ["car", "engine", "repair", "requires", "mechanical", "tools", "oil", "changes"]:
    av = eng._glove_vector(aw)
    if av is not None and subj_vec is not None:
        import numpy as np
        sim = float(np.dot(subj_vec, av) / (np.linalg.norm(subj_vec) * np.linalg.norm(av) + 1e-9))
        print(f"  {subj} vs {aw}: {sim:.4f}")
    else:
        print(f"  {subj} vs {aw}: None")
# Also check the analogical confabulation case
print("\nAnalogical confabulation case:")
subj2 = "dream"
answer2 = "when i let my mind wander, i find myself drawn to people who hoard every book ever read"
subj_vec2 = eng._glove_vector(subj2)
print(f"subj: {subj2}")
for aw in ["mind", "wander", "find", "drawn", "people", "hoard", "book", "read"]:
    av = eng._glove_vector(aw)
    if av is not None and subj_vec2 is not None:
        import numpy as np
        sim = float(np.dot(subj_vec2, av) / (np.linalg.norm(subj_vec2) * np.linalg.norm(av) + 1e-9))
        print(f"  {subj2} vs {aw}: {sim:.4f}")
    else:
        print(f"  {subj2} vs {aw}: None")
