"""A/B attribution probe: counts actual _triplet_mc_answer firings.

Wraps the engine method with a counter, runs the harness on a subset of
benchmarks, reports how many times the additive candidate RETURNED an
answer (non-None) vs abstained. Distinguishes real candidate effect
from run-to-run variance.

Usage:
  python scripts/tmp_ab_attribution.py --benchmarks memory_consistency,adversarial,reasoning [--triplet-candidate ...]
"""
import json
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, os.path.join(_root, "ravana", "src"))
sys.path.insert(0, _here)

COUNTS = {"consulted": 0, "answered": 0, "answers": []}

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402

_orig = CognitiveChatEngine._triplet_mc_answer


def _probed(self, user_input, fact_texts):
    COUNTS["consulted"] += 1
    out = _orig(self, user_input, fact_texts)
    if out is not None:
        COUNTS["answered"] += 1
        COUNTS["answers"].append(
            {"q": (user_input or "")[:200], "a": str(out)[:120]})
    return out


CognitiveChatEngine._triplet_mc_answer = _probed

import evaluate_ravana as _ev  # noqa: E402

try:
    _ev.main()
finally:
    out_path = os.path.join(_root, "reports", "ab_attribution.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(COUNTS, f, indent=2)
    print(f"[attribution] consulted={COUNTS['consulted']} "
          f"answered={COUNTS['answered']} -> {out_path}")
