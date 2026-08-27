import os, sys
os.environ["RAVANA_OFFLINE"] = "1"
_PROJ = os.getcwd()
for p in (_PROJ, _PROJ + "/ravana/src", _PROJ + "/ravana_ml/src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine


def facts(eng):
    pf = getattr(eng.user_model, "personal_facts", None)
    out = {}
    for k, v in pf.facts.items():
        if k[0] == "i" and not getattr(v, "superseded", False):
            out.setdefault(k[1], []).append(v.value)
    return out


eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="probe_x")
for t in ["my sister is sarah", "my car is the blue one",
          "i have an axolotl named nyx",
          "my brother is a tall guy called bob", "my dog is wren"]:
    eng.process_turn(t)
print("FACTS:", facts(eng))
