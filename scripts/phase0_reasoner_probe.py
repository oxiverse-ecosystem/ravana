"""Phase 0 probe: run the evaluation harness with logging shims around the
two old reasoners targeted by section 6.4:

  - answer_universal_syllogism (patched at its engine_memory import binding)
  - fact_reasoning._closure    (patched intra-module)

Every call where the old reasoner produced a usable result is recorded to
reports/phase0_protected_manifest.json — the "protected set" that the
flag-gated triplet routing must never regress.

Usage (same flags as evaluate_ravana.py, forwarded verbatim):
  python scripts/phase0_reasoner_probe.py --skip-train --no-curiosity \
      --semantic-grade --output reports/phase0_baseline.json
"""
import json
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, os.path.join(_root, "ravana", "src"))
sys.path.insert(0, _here)

PROBE_LOG = {"syllogism": [], "closure": []}

# ---- patch answer_universal_syllogism at the CALLER's binding ----
import ravana.chat.engine_memory as _em  # noqa: E402

_orig_syll = _em.answer_universal_syllogism


def _probed_syll(text):
    out = _orig_syll(text)
    if out is not None:
        PROBE_LOG["syllogism"].append(
            {"text": (text or "")[:300], "answer": str(out)[:200]})
    return out


_em.answer_universal_syllogism = _probed_syll

# ---- patch _closure inside fact_reasoning (intra-module calls) ----
import ravana.core.fact_reasoning as _fr  # noqa: E402

_orig_closure = _fr._closure


def _probed_closure(seed, fact_sets, *a, **kw):
    frontier, last_new, used = _orig_closure(seed, fact_sets, *a, **kw)
    if used:  # closure actually chained through at least one fact
        PROBE_LOG["closure"].append({
            "seed": sorted(list(seed))[:12],
            "n_facts": len(fact_sets),
            "n_used": len(used),
            "last_new": sorted(list(last_new))[:12] if last_new else [],
        })
    return frontier, last_new, used


_fr._closure = _probed_closure

# ---- run the harness with forwarded argv ----
import evaluate_ravana as _ev  # noqa: E402

try:
    _ev.main()
finally:
    os.makedirs(os.path.join(_root, "reports"), exist_ok=True)
    manifest = os.path.join(_root, "reports", "phase0_protected_manifest.json")
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({
            "syllogism_fired": len(PROBE_LOG["syllogism"]),
            "closure_fired": len(PROBE_LOG["closure"]),
            "syllogism_cases": PROBE_LOG["syllogism"],
            "closure_calls": PROBE_LOG["closure"][:500],
        }, f, indent=2)
    print(f"[probe] manifest -> {manifest} "
          f"(syllogism={len(PROBE_LOG['syllogism'])}, "
          f"closure={len(PROBE_LOG['closure'])})")
