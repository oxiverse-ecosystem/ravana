"""End-to-end casing proof (research casing plan) — production path.

Calls the REAL production method _human_like_uncertainty (the path that emits
subject casing for an ungrounded topic) with a constructed CognitiveResponseContext,
so we prove the generated answer casing through the actual pipeline (not a
reimplementation). Web is irrelevant here (the method doesn't search).

Confirms:
- "france"  -> "France"  (SUBTLEX prior, Phase 4)
- "microsoft" -> "Microsoft" (graph IsA entity, OOV from SUBTLEX, Phase 5)
- "apple" -> "apple" (ambiguous, stays lowercase)
- "bank"  -> "bank"  (known-common noun, lexical wins over graph 'company')
"""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_PROJ,
         os.path.join(_PROJ, "ravana_ml", "src"),
         os.path.join(_PROJ, "ravana", "src")):
    sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.models import CognitiveResponseContext

eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                          data_dir="/tmp/ravana_e2e_case2")

def unc(subject: str) -> str:
    ctx = CognitiveResponseContext(subject=subject, raw_input=subject)
    res = eng._human_like_uncertainty(ctx)
    if isinstance(res, tuple):
        return res[0] or ""
    return res or ""

cases = {
    "france": "France",
    "microsoft": "Microsoft",
    "apple": "apple",
    "bank": "bank",
}
results = {}
for subj, expect in cases.items():
    out = unc(subj)
    ok = expect in out
    results[subj] = (out, expect, ok)
    print(f"[E2E] subject={subj!r:12} expect={expect!r:10} -> {out[:70]!r}")
    print(f"      contains {expect!r}: {ok}")

eng.stop_background_learning()

all_ok = all(ok for _, _, ok in results.values())
print("\nVERDICT:", "CONFIRMED — production _human_like_uncertainty emits correct casing "
      "(France / Microsoft / apple / bank) via the real pipeline."
      if all_ok else "CHECK — a case failed; see above.")
raise SystemExit(0 if all_ok else 1)
