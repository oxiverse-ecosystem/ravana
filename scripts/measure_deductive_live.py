"""Section 6.5 live measurement — where does the brain-faithful
relational candidate actually fire, and are there ANY false positives?

This isolates the EXACT code path the engine calls when
use_deductive_candidate=True: ravana.core.deductive_reasoning
.deductive_mc_answer(user_input). Running it directly over the
benchmark questions is the honest "wire it live + measure" step
without paying the full decoder/web cost per case.

Zero-false-positive check:
  For each case the component FIRES on, the returned option
  string must equal the EXPECTED letter's option text. If it ever
  returns a DIFFERENT option, that is a false positive and we
  count it.

We also assert the component NEVER fires on non-MC input
(general QA) -> zero false positives there by construction.

Output: reports/deductive_live_measure.json (committed evidence).
"""
from __future__ import annotations
import sys, os, re, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ravana", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from evaluate_ravana import _load_logiqa
from ravana.core.deductive_reasoning import deductive_mc_answer
from ravana.core import fact_reasoning as _frz


def _expected_option_text(question: str, expected_label: str):
    """Map 'Answer: A' -> the text of option A from the question."""
    try:
        main, opts = _frz._split_options(question)
    except Exception:
        return None
    # opts is a list in A,B,C,... order; label is e.g. 'A'
    idx = ord(expected_label.upper()) - ord("A")
    if 0 <= idx < len(opts):
        return opts[idx]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=50)
    ap.add_argument("--out", default="reports/deductive_live_measure.json")
    args = ap.parse_args()

    cases = _load_logiqa(max_cases=args.cases)
    if not cases:
        print("NO LOGIQA CASES LOADED")
        return

    fired = 0
    correct = 0
    false_pos = 0
    false_pos_samples = []
    non_mc_checked = 0
    non_mc_fired = 0

    for i, c in enumerate(cases[:args.cases]):
        q = c["question"]
        expected_label = c["expected"].replace("Answer:", "").strip()
        got = deductive_mc_answer(q)
        if got is None:
            continue
        fired += 1
        exp_text = _expected_option_text(q, expected_label)
        # normalize for comparison
        def _norm(s):
            return re.sub(r"[^a-z0-9]", "", (s or "").lower())
        if exp_text is not None and _norm(got) == _norm(exp_text):
            correct += 1
        else:
            false_pos += 1
            false_pos_samples.append({
                "idx": i, "expected": expected_label,
                "expected_text": exp_text, "returned": got})

    # non-MC check: a plain factual question must NOT fire
    non_mc = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "All men are mortal.",  # no options -> not MC
    ]
    for s in non_mc:
        non_mc_checked += 1
        if deductive_mc_answer(s) is not None:
            non_mc_fired += 1

    result = {
        "n_cases": len(cases[:args.cases]),
        "component": "deductive_mc_answer (engine 6.5 path)",
        "fired": fired,
        "correct": correct,
        "false_positives": false_pos,
        "precision_when_fired": round(correct / fired, 3) if fired else None,
        "non_mc_checked": non_mc_checked,
        "non_mc_fired": non_mc_fired,
        "zero_false_positive_guarantee": (false_pos == 0 and non_mc_fired == 0),
        "false_positive_samples": false_pos_samples[:5],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"=== 6.5 deductive candidate — live measurement ({result['n_cases']} LogiQA) ===")
    print(f"fired               : {fired}")
    print(f"correct (entailed)  : {correct}")
    print(f"FALSE POSITIVES     : {false_pos}  <-- must be 0")
    print(f"non-MC fired        : {non_mc_fired}/{non_mc_checked}  <-- must be 0")
    print(f"ZERO-FP GUARANTEE : {result['zero_false_positive_guarantee']}")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
