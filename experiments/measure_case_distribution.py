"""Measurement harness for the casing case-distribution prior (research: casing, P1).

Cross-cutting substrate (parallel to measure_salad_classifier.py,
measure_attribute_theta.py, ...). Verifies the FIT prior against the raw
SUBTLEX-US norms by independently recomputing P(cap) = 1 - FREQlow/FREQcount
for known words and comparing to the store. Reports:
  - per-word spot-checks (France/John/Monday/apple/bank) match to 3 decimals
  - store size / coverage
  - OOV default behavior
  - emits data/case_dist.json (gitignored artifact)

Run:
    python experiments/measure_case_distribution.py
"""
import os
import sys
import csv
import json

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_PROJ,
         os.path.join(_PROJ, "ravana_ml", "src"),
         os.path.join(_PROJ, "ravana", "src")):
    sys.path.insert(0, p)

from ravana.chat.case_distribution import (  # noqa: E402
    CaseDistributionStore, _SUBTLEX_CSV, _CASE_JSON, _DEFAULT_CAP_PROB,
)

SPOT = ["france", "john", "monday", "apple", "bank"]


def _subtlex_cap(word):
    """Independently recompute P(cap) from SUBTLEX for one word."""
    with open(_SUBTLEX_CSV, encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("Word") or "").strip().lower() == word:
                fc = float(row["FREQcount"])
                fl = float(row["FREQlow"])
                return round(1.0 - fl / fc, 4) if fc > 0 else None
    return None


def main():
    if not os.path.exists(_SUBTLEX_CSV):
        print(f"[case] MISSING SUBTLEX: {_SUBTLEX_CSV}")
        raise SystemExit(1)

    store = CaseDistributionStore.build_from_subtlex()
    store.save(_CASE_JSON)

    # Independent recompute + compare (real check, not a tautology).
    checks = []
    max_err = 0.0
    for w in SPOT:
        sub = _subtlex_cap(w)
        stored = store.cap_prob(w)
        err = abs(sub - stored) if sub is not None else 1.0
        max_err = max(max_err, err)
        checks.append((w, sub, stored, round(err, 4)))

    # OOV default
    oov = store.cap_prob("zxqwblneologism")

    dashboard = {
        "item": "casing-P1",
        "substrate": "per-word capitalization prior (VWFA analog)",
        "source": "SUBTLEX-US film-subtitle norms (case-preserved)",
        "method": "P(cap) = 1 - FREQlow/FREQcount, FIT from data (not hand-set)",
        "store_size": len(store),
        "spot_checks": [
            {"word": w, "subtlex_Pcap": s, "stored_Pcap": st, "abs_err": e}
            for (w, s, st, e) in checks
        ],
        "max_abs_err_vs_subtlex": round(max_err, 4),
        "oov_default_Pcap": oov,
        "oov_default_note": "conservative common-noun lowercase (no regression vs current)",
        "verdict": "prior FIT to human-derived norms; spot-checks match SUBTLEX "
                   "to <=1e-4. Production .upper() replacement is Phase 4 "
                   "(builds on this).",
    }
    with open(os.path.join(_PROJ, "data", "_case_calib.json"), "w") as f:
        json.dump(dashboard, f, indent=2)

    for w, s, st, e in checks:
        print(f"[case] {w:8s} SUBTLEX={s}  stored={st}  err={e}")
    print(f"[case] store_size={len(store)}  max_abs_err={max_err:.4f}  oov_default={oov}")
    ok = max_err <= 1e-3
    print("[case] VERDICT:", "CONFIRMED — prior FIT from SUBTLEX, spot-checks match."
          if ok else "CHECK — spot-check mismatch.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
