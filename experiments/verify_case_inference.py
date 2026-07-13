"""Verify item casing/P4: context-sensitive case_infer replaces hardcoded .upper().

Tests the data-derived casing on known inputs (no web, no engine):
1. Sentence-start common noun -> capitalized (positional route):
   "the apple fell." -> "The apple fell."
2. Sentence-start proper noun -> capitalized (positional):
   "france is nice." -> "France is nice."
3. Mid-sentence proper noun (lexical prior) -> capitalized:
   "i visited paris last monday." -> "I visited Paris last Monday."
4. Mid-sentence common noun (low cap_prob) -> stays lowercase:
   "the bank was closed." -> "The bank was closed."  (bank cap_prob 0.12)
5. Ambiguous "apple" mid-sentence stays lowercase (cap_prob 0.27 < 0.5):
   "he ate an apple." -> "He ate an apple."
6. Multi-sentence: second sentence start capitalized too.
7. entity_words hook (Phase 5): a word forced capitalized via graph signal.

Mirrors scripts/ravana_chat.py import order for the module path.
"""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_PROJ,
         os.path.join(_PROJ, "ravana_ml", "src"),
         os.path.join(_PROJ, "ravana", "src")):
    sys.path.insert(0, p)

from ravana.chat.case_distribution import case_infer, get_store

store = get_store()
cases = [
    ("the apple fell.", "The apple fell."),
    ("france is nice.", "France is nice."),
    ("i visited paris last monday.", "I visited Paris last Monday."),
    ("the bank was closed.", "The bank was closed."),
    ("he ate an apple.", "He ate an apple."),
    ("hi. how are you.", "Hi. How are you."),
]
print("[P4] store size:", len(store))
ok = True
for inp, exp in cases:
    got = case_infer(inp, store=store)
    good = got == exp
    ok = ok and good
    print(f"[P4] {inp!r} -> {got!r}  expect {exp!r}  {'OK' if good else 'FAIL'}")

# entity_words hook (Phase 5): force-capitalize a word the prior would lowercase
ent = case_infer("the ravana system learns.", store=store, entity_words={"ravana"})
exp_ent = "The Ravana system learns."
good_ent = ent == exp_ent
ok = ok and good_ent
print(f"[P4] entity_words: 'the ravana system learns.' -> {ent!r}  expect {exp_ent!r}  {'OK' if good_ent else 'FAIL'}")

print("\nVERDICT:", "CONFIRMED — context-sensitive casing works (position + lexical prior + entity hook)."
      if ok else "CHECK — a P4 case failed.")
raise SystemExit(0 if ok else 1)
