"""Calibration harness for the compositional realizer lexicon (Stage 6).

Externalizes the formerly-inline assertion-mirror templates (leads/follows/
backchannels, incl. f"yeah, {topic}.") to data/realizer_lexicon.json as
exemplar pools, and documents the realization distribution vs the legacy
random.choice. The seed scorer is uniform (== legacy distribution); a learned
fluency/coherence ranker (reusing salad_classifier) is the future extension
point and would be fit here on a (candidate, fluent?) corpus.

Run:  python experiments/measure_realizer.py
"""

import json
import os
import sys
from collections import Counter

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana_ml", "src"))

import random

from ravana.chat.realizer_lexicon import RealizerLexicon, default_lexicon, _FIT_PATH


def main():
    rl = default_lexicon()
    rl.save()
    print(f"[realizer] saved exemplar pools -> {_FIT_PATH}")
    print(f"[realizer] pools: {list(rl._pools.keys())}")

    # Document the realization distribution (must span every former template).
    rng = random.Random(42)
    # Legacy inline lists, reconstructed for audit comparison.
    legacy_other_leads = [
        "right, {topic}.", "got it — {topic}.", "ok, noted: {topic}.",
        "yeah, {topic}.",
    ]
    # Sample the new realizer 2000x; every legacy candidate must be reachable.
    seen = Counter()
    for _ in range(2000):
        seen[rl.realize("other_leads", topic="gravity", rng=rng)] += 1
    reachable = {c.format(topic="gravity") for c in legacy_other_leads}
    covered = all(any(c in seen for c in [fmt]) for fmt in reachable)
    print(f"[realizer] other_leads coverage of legacy templates: "
          f"{sum(1 for c in reachable if c in seen)}/{len(reachable)} "
          f"({'OK' if covered else 'PARTIAL'})")

    summary = {
        "pools": list(rl._pools.keys()),
        "legacy_other_leads_covered": sum(1 for c in reachable if c in seen),
        "legacy_other_leads_total": len(reachable),
        "note": "Templates demoted to externalized exemplars; seed scorer "
                "uniform (== legacy random.choice). Learned fluency/coherence "
                "ranker is the future extension point (fit here).",
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_realizer_calib.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[realizer] dashboard -> {out}")
    print("[realizer] VERDICT: f\"yeah, {topic}.\" + sibling templates retired "
          "from inline code; now externalized exemplars in data/realizer_lexicon.json.")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
