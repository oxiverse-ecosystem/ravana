"""Where does a contrastive self-opinion query land, and what do the sides read?

Isolates: (a) which strategy handles it, (b) what each side's stored stance is,
(c) whether the two sides are distinguishable (i.e. a winner is recoverable).
"""
from __future__ import annotations
import os, sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (TREE, os.path.join(TREE, "ravana_ml", "src"),
           os.path.join(TREE, "ravana", "src"), os.path.join(TREE, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.meta_path = [f for f in sys.meta_path
                 if "editable" not in (type(f).__module__ or "").lower()]

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402

SUFFIX = sys.argv[1] if len(sys.argv) > 1 else "rv19probe"
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=SUFFIX)

DISCLOSE = "i think remote work is better than office work"
eng.process_turn(DISCLOSE)

ops = eng.user_model.opinions
print("--- stored stances")
for k, v in sorted(ops.stances.items()):
    print(f"  {k!r:30s} pol={v.polarity:+.2f} conf={v.confidence:.2f}")

for side in ("remote work", "office work"):
    s = ops.query_stance(side)
    print(f"query_stance({side!r}) -> "
          f"{None if s is None else (round(s.polarity, 2), round(s.confidence, 2))}")

print("\n--- query routing")
for q in ("do you prefer remote work or office work?",
          "which do you prefer, remote work or office work?",
          "do you think you're more of a remote worker or an office worker?"):
    r = eng.process_turn(q)
    strat = getattr(eng, "last_strategy", None)
    print(f"  {q}\n    -> strategy={strat} reply={r!r}")

print("\n--- structured_recall / contrast gates")
for name in ("_structured_recall", "_route_self_query", "_is_self_opinion_query"):
    print(f"  has {name}: {hasattr(eng, name)}")
sr = getattr(eng, "_structured_recall", None)
if callable(sr):
    print("  _structured_recall(contrast q) ->",
          sr("do you prefer remote work or office work?"))

eng.stop_background_learning()