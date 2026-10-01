"""What keys does the ENGINE actually store for candidate generality probes?

Prints real store state per probe so the test's expected keys are MEASURED,
not guessed.
"""
from __future__ import annotations
import os, sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (TREE, os.path.join(TREE, "ravana_ml", "src"),
          os.path.join(TREE, "ravana", "src"), os.path.join(TREE, "ravana-v2", "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.meta_path = [f for f in sys.meta_path
                 if "editable" not in (type(f).__module__ or "").lower()]

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402

PROBES = [
    "i believe trains are better than planes for daily work",
    "honestly i think the mountains are finer than the coast",
    "honestly i think the mountains are finer than the coastline",
    "small towns make better humans than big cities",
    "small towns make better humans than big cities, i think",
    "i think cycling is better than driving around the city",
]

for n, q in enumerate(PROBES):
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix=f"rv19k{n}")
    try:
        eng.process_turn(q)
        st = eng.user_model.opinions.stances
        print(f"\n{q!r}  strategy={getattr(eng, '_last_strategy', None)}")
        for k, v in sorted(st.items()):
            print(f"    {k!r:26s} pol={v.polarity:+.2f}")
    finally:
        try:
            eng.stop_background_learning()
        except Exception:
            pass