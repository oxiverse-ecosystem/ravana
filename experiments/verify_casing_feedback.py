"""Verify item casing/P3: online user-feedback (N400 prediction-error) channel.

P3 replaces the old in-memory, binary _proper_nouns set-add with a WEIGHTED,
PERSISTED case-distribution update:
- record_feedback(word, capitalized) accumulates [cap_count, total] per word
- cap_prob consults override -> feedback (if >= _FB_MIN_OBS) -> SUBTLEX -> default
- the store (incl. feedback) is saved to data/case_dist.json and reloaded

Tests the N400 analog directly on the store + persistence round-trip. No web.
"""
import os
import sys
import json
import tempfile

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_PROJ,
         os.path.join(_PROJ, "ravana_ml", "src"),
         os.path.join(_PROJ, "ravana", "src")):
    sys.path.insert(0, p)

# Use an isolated json so we don't pollute the real data/case_dist.json
_TMP = tempfile.mkdtemp()
import ravana.chat.case_distribution as cd
cd._CASE_JSON = os.path.join(_TMP, "case_dist.json")
cd._STORE_CACHE = None

store = cd.get_store(rebuild=True)

# 1) base SUBTLEX prior for a low-cap word (e.g. 'bank' ~0.12, 'france' ~0.99)
print(f"[P3] base cap_prob(bank)={store.cap_prob('bank'):.4f} (SUBTLEX, low)")
print(f"[P3] base cap_prob(france)={store.cap_prob('france'):.4f} (SUBTLEX, high)")

# 2) single observation must NOT flip a well-established prior (robust to noise)
store.record_feedback("france", False)   # user typed 'france' lowercase once
print(f"[P3] cap_prob(france) after 1 lowercase obs = {store.cap_prob('france'):.4f} "
      f"(expect ~0.99, NOT flipped)")

# 3) enough observations of an OOV/ambiguous word DOES update the model
#    Use a word with low SUBTLEX prior that the user consistently capitalizes.
for _ in range(5):
    store.record_feedback("quxcorp", True)   # made-up entity, user always 'QuxCorp'
print(f"[P3] cap_prob(quxcorp) after 5 capitalized obs = {store.cap_prob('quxcorp'):.4f} "
      f"(expect 1.0, feedback overrides default 0.0)")

# 4) persistence: save, reload from a fresh store object, confirm feedback survives
store.save()
reloaded = cd.CaseDistributionStore.load(cd._CASE_JSON)
print(f"[P3] reloaded cap_prob(quxcorp) = {reloaded.cap_prob('quxcorp'):.4f} "
      f"(expect 1.0, persisted)")
print(f"[P3] reloaded cap_prob(france) = {reloaded.cap_prob('france'):.4f} "
      f"(expect ~0.99, SUBTLEX untouched)")

# 5) override path (high-confidence typo / stable proper noun)
store.override("iphone", 1.0)
print(f"[P3] cap_prob(iphone) after override=1.0 -> {store.cap_prob('iphone'):.4f} (expect 1.0)")

ok = (abs(store.cap_prob('france') - 0.99) < 0.05
      and store.cap_prob('quxcorp') == 1.0
      and reloaded.cap_prob('quxcorp') == 1.0
      and reloaded.cap_prob('france') > 0.9
      and store.cap_prob('iphone') == 1.0)
print("\nVERDICT:", "CONFIRMED — online casing feedback is weighted, robust to a "
      "single noisy observation, and persists across restart (N400 analog)."
      if ok else "CHECK — a P3 case failed.")
raise SystemExit(0 if ok else 1)
