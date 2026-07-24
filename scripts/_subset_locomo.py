"""Fast LoCoMo subset harness — reuses the REAL grader + runner from
evaluate_ravana.py so any score reported is the genuine artifact, not a
reimplementation.

Usage:
    python scripts/_subset_locomo.py [max_cases] [dialogue_cap]

Defaults: max_cases=120 (≈ dlg0 + start of dlg1), all dialogues.
"""
import os
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_PROJ, os.path.join(_PROJ, "ravana", "src"),
           os.path.join(_PROJ, "ravana_ml", "src"),
           os.path.join(_PROJ, "ravana-v2", "src"),
           os.path.join(_PROJ, "scripts")):
    sys.path.insert(0, _p)

import numpy as np
import evaluate_ravana as EV


def main():
    max_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    EV.MAX_CASES = {"_default": max_cases}
    print(f"[subset] max_cases={max_cases}")

    # Boot a single engine (per the eval's own config).
    t0 = time.time()
    engine = EV.CognitiveChatEngine(dim=64, seed=42, baby_mode=True)
    print(f"[subset] engine boot {time.time()-t0:.1f}s")

    EV._init_benchmarks()
    locomo = EV.BENCHMARKS["locomo"]
    EV._ensure_cases_loaded("locomo")
    print(f"[subset] loaded {len(locomo['cases'])} cases")

    result = EV.run_benchmark_category(engine, "locomo", locomo)

    # Per-category breakdown.
    cats = {1: "single-hop", 2: "temporal", 3: "multi-hop",
            4: "open-domain", 5: "adversarial"}
    by_cat = {}
    for c, d in zip(locomo["cases"], result["case_scores"]):
        by_cat.setdefault(c["category"], []).append(d)
    print("\n=== SUBSET PER-CATEGORY ===")
    for cat in sorted(by_cat):
        arr = by_cat[cat]
        print(f"  cat{cat} {cats.get(cat,'?'):12s}: "
              f"{np.mean(arr):.3f}  (n={len(arr)})")
    print(f"\n=== SUBSET OVERALL: {result['average_score']:.3f} "
          f"(n={len(result['case_scores'])}) ===")
    # Emit machine-readable for diffing.
    out = {
        "overall": round(float(result["average_score"]), 4),
        "by_cat": {str(k): round(float(np.mean(v)), 4)
                   for k, v in sorted(by_cat.items())},
        "n": len(result["case_scores"]),
    }
    print("JSON:" + str(out).replace("'", '"'))


if __name__ == "__main__":
    main()
