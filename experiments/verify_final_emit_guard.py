"""Verify the fail-closed final-emit salad guard (research item B).

Mirrors the import order of scripts/ravana_chat.py so the ravana_ml / ravana
package init resolves. Forces the Q21 word-salad escape string through
_final_emit_guard on the Situation-Model decoder path and asserts it is
withheld — and that _disable_grounding_gate does NOT bypass the guard.
"""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.join(_PROJ, "ravana_ml", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana-v2", "src"))

from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.models import CognitiveResponseContext

e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                        data_dir="/tmp/ravana_guard_test")
ctx = CognitiveResponseContext(subject="memory",
                                raw_input="how do you microwave a memory?")

Q21 = ("Deserves humans continents fool interstitials harmful eat people "
       "claude key habitats care rain rapidly honesty great high planting "
       "matthew fine time effect new online.")
COHERENT = ("Photosynthesis is a fundamental biological process that plants, "
            "algae, and certain bacteria use for converting light energy into "
            "chemical energy.")

verdicts = []
out = e._final_emit_guard(Q21, ctx, strategy="situation_model_decoder")
ok = out != Q21
verdicts.append(ok)
print(f"[guard] Q21 garbage blocked={ok}  -> {'CONFIRMED' if ok else 'FAIL'}")

outc = e._final_emit_guard(COHERENT, ctx, strategy="situation_model_decoder")
okc = outc == COHERENT
verdicts.append(okc)
print(f"[guard] coherent def passes={okc}  -> {'CONFIRMED' if okc else 'FAIL'}")

# The kill-switch must NOT bypass the fail-closed guard.
e._disable_grounding_gate = True
out2 = e._final_emit_guard(Q21, ctx, strategy="situation_model_decoder")
ok2 = out2 != Q21
verdicts.append(ok2)
print(f"[guard] Q21 blocked even with _disable_grounding_gate=True={ok2}  "
      f"-> {'CONFIRMED' if ok2 else 'FAIL'}")

all_ok = all(verdicts)
print("\nVERDICT:", "CONFIRMED — fail-closed guard intercepts Q21 on the "
      "decoder path and is not bypassed by the A/B kill-switch."
      if all_ok else "CHECK — a guard assertion failed.")
raise SystemExit(0 if all_ok else 1)
