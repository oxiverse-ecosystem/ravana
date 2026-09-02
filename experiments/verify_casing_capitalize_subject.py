"""Verify item casing/P4 fix directly (no web, fast).

The user-visible bug was: subject "france" (web/unknown, not in the manual
_proper_nouns set) rendered lowercase in answers. The fix is _capitalize_subject
now consulting the FIT capitalization prior (SUBTLEX) instead of the manual set
only. This test checks that method directly (no process_turn / no web).
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
                          data_dir="/tmp/ravana_p4_cap")

# Strong proper noun typed lowercase -> canonical capitalized form.
c1 = eng._capitalize_subject("france", "tell me about france")
# Strong proper noun typed with caps -> preserved.
c2 = eng._capitalize_subject("france", "Tell me about France")
# Ambiguous word typed lowercase -> stays lowercase (canonical uncertain).
c3 = eng._capitalize_subject("apple", "what is an apple")
# Generic noun -> lowercase.
c4 = eng._capitalize_subject("life", "what is life")
# Proper noun from manual set -> capitalized (simulate runtime population:
# the set fills during real process_turn from user input / domain concepts).
eng._proper_nouns.add("ravana")
c5 = eng._capitalize_subject("ravana", "tell me about ravana")

print(f"[P4] france(lower input)  -> {c1!r}  (expect 'France')")
print(f"[P4] france(cap input)    -> {c2!r}  (expect 'France')")
print(f"[P4] apple(lower input)   -> {c3!r}  (expect 'apple')")
print(f"[P4] life                 -> {c4!r}  (expect 'life')")
print(f"[P4] ravana(manual set)   -> {c5!r}  (expect 'Ravana')")

eng.stop_background_learning()
ok = (c1 == "France" and c2 == "France" and c3 == "apple"
      and c4 == "life" and c5 == "Ravana")
print("\nVERDICT:", "CONFIRMED — subject casing now data-derived (prior + manual set), not manual-set-only."
      if ok else "CHECK — a casing case failed.")
raise SystemExit(0 if ok else 1)
