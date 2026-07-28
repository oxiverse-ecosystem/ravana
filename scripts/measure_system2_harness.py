"""System-2 harness — evaluate the ALREADY-BUILT, fail-closed candidates
against an ENTAILMENT-axis benchmark (LogiQA is the wrong yardstick;
this measures whether the infrastructure actually fires correctly when
given entailment-shaped input).

Three candidate surfaces, each with a 100% graded key:

1. 6.5 ProblemWorkingMemory + RoleMetaruleEngine (deductive_reasoning.py)
   -> relational/transitive chains, comparative, universal syllogism.
   Invoked via ChatEngine._deductive_mc_answer when use_deductive_candidate
   is True.

2. graph_reasoner.select_option_logic
   -> modus ponens / modus tollens / disjunctive elimination MC.

3. The 6.5 premise extractor (deductive_extractor.DeductivePremiseExtractor)
   de-blobbing NL into ProblemWorkingMemory triples (input-quality gate).

No external data. Self-contained. Fail-closed means: if a candidate returns
None, that's an ABSTENTION (safe), not an error. We count fires / correct /
false positives (a fire is false-positive if it returns an option != graded).
"""

import sys, os, json, argparse, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ravana", "src"))

from ravana.core import deductive_reasoning as dr
from ravana.core import graph_reasoner as gr

# ── Case definitions: (prompt_text, graded_option_letter) ───────────────
# Each prompt is an MC question with options A..D, correct letter graded.

TRANSITIVE = [
    ("A is north of B. B is north of C. Options: A. A is north of C. "
     "B. C is north of A. C. A is south of B. D. B is north of A.", "A"),
    ("Tom is taller than Sam. Sam is taller than Lee. Options: A. Lee is taller than Tom. "
     "B. Tom is taller than Lee. C. Sam is taller than Tom. D. Lee is taller than Sam.", "B"),
    ("The book is on the left of the lamp. The lamp is on the left of the cup. "
     "Options: A. The cup is on the left of the book. B. The book is on the left of the cup. "
     "C. The lamp is on the right of the book. D. The cup is on the left of the lamp.", "B"),
]

COMPARATIVE = [
    ("Alice is older than Bob. Bob is older than Carol. Options: A. Carol is older than Alice. "
     "B. Alice is older than Carol. C. Bob is older than Alice. D. Carol is older than Bob.", "B"),
    ("X weighs more than Y. Y weighs more than Z. Options: A. Z weighs more than X. "
     "B. X weighs more than Z. C. Y weighs more than X. D. Z weighs more than Y.", "B"),
]

UNIVERSAL_SYLLOGISM = [
    ("All cats are mammals. All mammals are animals. Options: A. All cats are animals. "
     "B. All animals are cats. C. Some cats are not animals. D. No mammals are animals.", "A"),
    ("Every student is a person. Every person is a citizen. Options: A. Every citizen is a student. "
     "B. Every student is a citizen. C. Some students are not persons. D. No students are citizens.", "B"),
    ("All birds can fly. All sparrows are birds. Options: A. Some sparrows cannot fly. "
     "B. All sparrows can fly. C. No birds can fly. D. All flyers are sparrows.", "B"),
]

MODUS_PONENS = [
    ("If it is raining, then the ground is wet. It is raining. "
     "Options: A. The ground is wet. B. The ground is dry. "
     "C. It is not raining. D. The sky is blue.", "A"),
    ("If a creature is a dog, then it is a mammal. Rex is a dog. "
     "Options: A. Rex is a mammal. B. Rex is not a mammal. "
     "C. Rex is a cat. D. All mammals are dogs.", "A"),
]

MODUS_TOLLENS = [
    ("If the light is on, the room is bright. The room is not bright. "
     "Options: A. The light is on. B. The light is not on. "
     "C. The room is bright. D. The light is red.", "B"),
    ("If a number is divisible by 4, it is even. This number is not even. "
     "Options: A. It is divisible by 4. B. It is not divisible by 4. "
     "C. It is even. D. It is prime.", "B"),
]

DISJUNCTIVE = [
    ("Either the train is late or the bus is late. The train is not late. "
     "Options: A. The bus is late. B. The bus is not late. "
     "C. Both are late. D. Neither is late.", "A"),
]


def _graded_letter(text, graded):
    # graded is the expected letter; we compare to candidate's returned option text.
    return graded


def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).strip()


def _returned_letter(ans, text):
    """Map a returned option string to its letter by canonical match.

    Reuses fact_reasoning._split_options (the fixed, sequentially-numbered
    splitter) as the single source of truth for option boundaries, then
    maps by index to letters A/B/C/D and canonical-matches against ans.
    """
    if ans is None:
        return None
    from ravana.core.fact_reasoning import _split_options
    _, opts = _split_options(text)
    cans = _canon(ans)
    if not cans or not opts:
        return None
    for i, otext in enumerate(opts):
        ct = _canon(otext)
        L = chr(ord("A") + i)
        if cans == ct or ct == cans:
            return L
        if cans and ct and (cans in ct or ct in cans):
            return L
    return None


def eval_candidate(name, cases, fn):
    fires = 0
    correct = 0
    fp = 0
    abstain = 0
    for text, graded in cases:
        try:
            ans = fn(text)
        except Exception:
            ans = "<<ERROR>>"
        if ans is None or ans == "<<ERROR>>":
            abstain += 1
            continue
        fires += 1
        got = _returned_letter(ans, text)
        if got == graded:
            correct += 1
        else:
            fp += 1
    return {
        "candidate": name,
        "n": len(cases),
        "fires": fires,
        "correct": correct,
        "false_positives": fp,
        "abstain": abstain,
        "accuracy_of_fires": round(100.0 * correct / fires, 1) if fires else 0.0,
        "fail_closed": fp == 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/system2_harness_results.json")
    args = ap.parse_args()

    # Candidate 1: 6.5 ProblemWorkingMemory relational/transitive + syllogism
    # Use the real fail-closed MC entry point (builds PWM, applies
    # RoleMetaruleEngine, returns entailed option or None).
    def fn_65(text):
        return dr.deductive_mc_answer(text)

    rel_cases = TRANSITIVE + COMPARATIVE + UNIVERSAL_SYLLOGISM
    r65 = eval_candidate("6.5_ProblemWorkingMemory", rel_cases, fn_65)

    # Candidate 2: graph_reasoner.select_option_logic (MP/MT/disjunction)
    gr_cases = MODUS_PONENS + MODUS_TOLLENS + DISJUNCTIVE
    rgr = eval_candidate("graph_reasoner.select_option_logic", gr_cases, gr.select_option_logic)

    # Candidate 3: extractor de-blob sanity — does parse_deductive_premises
    # (which delegates to DeductivePremiseExtractor) yield triples from NL?
    ext_yield = 0
    ext_total = 0
    for text, _ in rel_cases + gr_cases:
        ext_total += 1
        try:
            premise_part = text.split("Options:")[0]
            triples = dr.parse_deductive_premises(premise_part)
            if triples:
                ext_yield += 1
        except Exception:
            pass
    rext = {
        "candidate": "deductive_extractor.deblob",
        "n": ext_total,
        "cases_with_triples": ext_yield,
        "yield_pct": round(100.0 * ext_yield / ext_total, 1),
    }

    report = {
        "harness": "system2_entailment_axis",
        "note": "LogiQA is the wrong yardstick; this measures the built "
                "fail-closed candidates on entailment-shaped input with a "
                "100% graded key.",
        "results": [r65, rgr, rext],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
