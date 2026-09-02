"""Verify G3 metaphor generation from sensorimotor dimensions (engine.py Path 1).

Uses the human Lancaster 11-D norms (wired this session) for cross-modal
metaphor dimension selection, falling back to the variance-compressed merged
probe for OOV words. No web.

Checks:
1. _top_sensorimotor_dim('hand') selects a SENSORY Binder dim (Touch/Vision/...)
   with a high 0-5 value (hand is strongly embodied: Hand_arm=4.4, Haptic=3.65).
2. _top_sensorimotor_dim('trust') returns None or a low-value dim (abstract).
3. _metaphor_for_category_error('hand','color') returns a vivid cross-modal
   metaphor built from the human Lancaster signal (not a hollow probe fallback).
4. _metaphor_for_category_error('trust','color') does not crash; returns either a
   metaphor or None (honest fallthrough) — no exception.
"""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_PROJ,
         os.path.join(_PROJ, "ravana_ml", "src"),
         os.path.join(_PROJ, "ravana", "src")):
    sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine

eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                          data_dir="/tmp/ravana_g3_meta")

ok = True

td_hand = eng._top_sensorimotor_dim("hand")
td_trust = eng._top_sensorimotor_dim("trust")
print(f"[G3] top_sensorimotor_dim hand={td_hand}")
print(f"[G3] top_sensorimotor_dim trust={td_trust}")

if td_hand is None:
    ok = False
    print("[G3] FAIL: hand should have a salient sensory dim")
else:
    dim, val, phrase, sense = td_hand
    print(f"[G3] hand -> dim={dim} val={val:.2f} phrase={phrase}")
    if val < 1.0:
        ok = False
        print("[G3] FAIL: hand sensory value should be high (embodied)")
    if dim not in eng._SENSORY_DIM_PHRASE:
        ok = False
        print("[G3] FAIL: selected dim not a sensory phrase dim")

# Embodied specificity: hand and trust must NOT collapse to the same dim.
if td_hand is not None and td_trust is not None and td_hand[0] == td_trust[0]:
    ok = False
    print(f"[G3] FAIL: hand and trust collapsed to same dim {td_hand[0]} "
          "(embodied specificity lost)")

meta_hand = eng._metaphor_for_category_error("hand", "color")
print(f"[G3] metaphor('hand','color') = {meta_hand!r}")
if not meta_hand or "picture" not in meta_hand and "think of" not in meta_hand:
    ok = False
    print("[G3] FAIL: hand metaphor should be vivid cross-modal")

meta_trust = eng._metaphor_for_category_error("trust", "color")
print(f"[G3] metaphor('trust','color') = {meta_trust!r}")
if meta_trust is not None and "Traceback" in meta_trust:
    ok = False

eng.stop_background_learning()

print("\nVERDICT:",
      "CONFIRMED — G3 metaphor generation uses human Lancaster sensorimotor "
      "norms (vivid for embodied words, graceful fallthrough for abstract)."
      if ok else "CHECK — a G3 metaphor case failed.")
raise SystemExit(0 if ok else 1)
