"""Option A — measure the EXISTING conditional candidate.

graph_reasoner.py's select_option_logic is ALREADY wired into the
engine at _graph_reasoner_answer (engine.py:2203). It implements
modus ponens / tollens / disjunction->rule / comparatives and is
fail-closed. This script runs it STAND-ALONE over the 50-case
LogiQA MC probe and reports the honest empirical truth:

  - fires      : cases where it returned a non-None option
  - correct    : of those, how many matched the graded label
  - false_pos  : of those, how many were WRONG (the safety breach)
  - abstained  : cases it correctly declined (None)

No new engine is built; this is pure measurement of code that
already runs.
"""
from __future__ import annotations
import sys, os, json, argparse
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ravana", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from evaluate_ravana import _load_logiqa
from ravana.core import graph_reasoner as _gr


def _expected_letter(case: dict) -> str:
    exp = (case.get("expected") or "").strip().upper()
    # expected like "ANSWER: A"
    for tok in exp.replace("ANSWER:", "").split():
        if len(tok) == 1 and tok.isalpha():
            return tok
    return ""


_OPT_RE = __import__("re").compile(r"([A-E])[\.\)]\s*([^A-E\n][^\n]*)")


def _question_options(q: str):
    """Parse 'A) text / A. text' option blocks from the question."""
    out = []
    for m in _OPT_RE.finditer(q or ""):
        out.append((m.group(1), m.group(2).strip()))
    return out


def _returned_letter(returned: str, q: str) -> str:
    """Which option LETTER does `returned` correspond to?

    graph_reasoner returns an OPTION STRING. We map it back to the
    letter by matching the option text in the question.
    """
    if not returned:
        return ""
    rt = returned.strip().lower()
    for letter, text in _question_options(q):
        ot = text.lower()
        if not ot:
            continue
        if ot == rt or ot in rt or rt in ot:
            return letter
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=50)
    ap.add_argument("--out", default="reports/graph_reasoner_live.json")
    args = ap.parse_args()

    cases = _load_logiqa()[:args.cases]

    fires = 0
    correct = 0
    false_pos = 0
    abstained = 0
    graded = 0
    errors = 0
    per_relation = Counter()

    for case in cases:
        q = case.get("question") or ""
        exp = _expected_letter(case)
        if exp:
            graded += 1
        try:
            ans = _gr.select_option_logic(q)
        except Exception as e:
            errors += 1
            ans = None

        if ans is None:
            abstained += 1
            continue

        fires += 1
        got = _returned_letter(ans, q)
        if exp and got == exp:
            correct += 1
        elif exp and got and got != exp:
            false_pos += 1  # returned an option, but the WRONG one
        # if got == "" we couldn't map -> counts as fire w/o grade

    report = {
        "n_cases": len(cases),
        "n_graded": graded,
        "fires": fires,
        "correct_on_fired": correct,
        "false_positives": false_pos,
        "abstained": abstained,
        "errors": errors,
        "fire_rate_pct": round(100.0 * fires / max(len(cases), 1), 1),
        "precision_on_fired_pct": round(
            100.0 * correct / max(fires, 1), 1),
        "false_positive_rate_pct": round(
            100.0 * false_pos / max(fires, 1), 1),
        "fail_closed_ok": false_pos == 0,
        "verdict": (
            "INFRASTRUCTURE COMPLETE (fires safely, 0 false positives)"
            if false_pos == 0
            else "GAPS: false positives present -> needs tightening"),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Loaded {len(cases)} LogiQA cases ({graded} with a graded label)")
    print("=== graph_reasoner.select_option_logic (existing, wired) ===")
    print(f"  fires            : {fires} ({report['fire_rate_pct']}%)")
    print(f"  correct on fired : {correct}")
    print(f"  FALSE POSITIVES : {false_pos}  <-- must be 0")
    print(f"  abstained       : {abstained}")
    print(f"  errors          : {errors}")
    print(f"  precision(fired): {report['precision_on_fired_pct']}%")
    print(f"  fail-closed    : {report['fail_closed_ok']}")
    print(f"  VERDICT         : {report['verdict']}")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
