"""Cold reproduction of FIX-RV-19 (comparative stance shredded into 3 keys).

Run with the repo venv, offline, on a CLEAN suffix:
    RAVANA_OFFLINE=1 <repo>/.venv-real/Scripts/python.exe tools/rv19_repro.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (TREE,
           os.path.join(TREE, "ravana_ml", "src"),
           os.path.join(TREE, "ravana", "src"),
           os.path.join(TREE, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# A worktree must never import the MAIN checkout (editable-install finder).
sys.meta_path = [f for f in sys.meta_path
                 if "editable" not in (type(f).__module__ or "").lower()
                 and "editable" not in (type(f).__name__ or "").lower()]

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402
import ravana.chat.engine as _eng  # noqa: E402

print("engine tree:", _eng.__file__)

SUFFIX = "rv19repro"

eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=SUFFIX)
turns = [
    "i like hiking a lot",
    "i think remote work is better than office work",
]
for t in turns:
    r = eng.process_turn(t)
    print(f"TURN {t!r} -> {r!r}")

print("\n--- STANCES ---")
for k, v in sorted(eng.user_model.opinions.stances.items()):
    print(f"  {k!r:40s} pol={getattr(v, 'polarity', None):+.2f} "
          f"conf={getattr(v, 'confidence', None)}")

print("\n--- CONTRASTIVE RECOVERY ---")
q = "do you prefer remote work or office work?"
print("query:", q)
print("reply:", eng.process_turn(q))
eng.stop_background_learning()