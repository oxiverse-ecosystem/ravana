"""Verify item C: self-model no longer swallows answerable questions.

On the LIVE engine (process_turn):
  - "why do people feel lonely in a crowd" (answerable, contains 'feel')
    must NOT be answered by the canned self-model branch (strategy !=
    'self_model'); the question reaches an answerable path.
  - "do you feel lonely sometimes" (genuinely self-addressed) MUST still
    route to the self-model branch (strategy == 'self_model').
Mirrors scripts/ravana_chat.py import order.
"""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_PROJ,
         os.path.join(_PROJ, "ravana_ml", "src"),
         os.path.join(_PROJ, "ravana", "src"),
         os.path.join(_PROJ, "ravana-v2", "src")):
    sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine

eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                          data_dir="/tmp/ravana_c_test")

# answerable question that previously got swallowed by the self-model branch
ans_lonely = eng.process_turn("why do people feel lonely in a crowd")
strat_lonely = eng._last_strategy
# genuinely self-addressed question
ans_self = eng.process_turn("do you feel lonely sometimes")
strat_self = eng._last_strategy
# control: factual question should not be self_model either
ans_fact = eng.process_turn("what is gravity")
strat_fact = eng._last_strategy

print(f"[C] 'why do people feel lonely in a crowd' -> [{strat_lonely}] {ans_lonely!r}")
print(f"    not_self_model = {strat_lonely != 'self_model'}  (expect True: answered, not canned turn-back)")
print(f"[C] 'do you feel lonely sometimes'         -> [{strat_self}] {ans_self!r}")
print(f"    is_self_model = {strat_self == 'self_model'}  (expect True: genuine self-address)")
print(f"[C] 'what is gravity'                        -> [{strat_fact}] (control, expect != self_model)")

eng.stop_background_learning()

ok = (strat_lonely != "self_model") and (strat_self == "self_model") and (strat_fact != "self_model")
print("\nVERDICT:", "CONFIRMED — answerable questions no longer swallowed; genuine self-queries still reflected."
      if ok else "CHECK — a C assertion failed.")
raise SystemExit(0 if ok else 1)
