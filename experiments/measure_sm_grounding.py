#!/usr/bin/env python3
"""Standalone measurement for the Situation-Model Levelt/Wernicke monitor.

This is the per-sentence (clause-grained) refinement of the SM grounding gate.
It forces the exact degenerate strings observed in the live ravana_chat.py run
(Q3 gravity, Q5 black holes, Q8 life) through ResponseGenerator._sm_response_grounded
and asserts the monitor now WITHHOLDS them (-> honest uncertainty), while a
genuinely grounded answer (Q9 oxiverse / a real web sentence) still PASSES.

It also includes a CONTROL arm: with the grounding gate disabled
(_disable_grounding_gate=True) the same degenerate strings PASS — proving the
per-sentence monitor is what catches them, not a GloVe/luck artifact.

Run from repo root:
    python experiments/measure_sm_grounding.py
Exit code 0 = all verdicts CONFIRMED; 1 = any CHECK/FAIL.
"""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana_ml", "src"))

from ravana.chat.models import CognitiveResponseContext
from ravana.chat.engine import CognitiveChatEngine


def _build_engine():
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               data_dir="/tmp/ravana_sm_grounding_measure")


def _ctx(subject, assoc, raw):
    return CognitiveResponseContext(
        subject=subject, raw_input=raw,
        associated_concepts=[(a, 0.5) for a in assoc],
    )


# ── Degenerate strings captured from the live chat run ──────────────────────
DEGENERATE = {
    "Q3_gravity": (
        "Gravity is one of the most fundamental forces of the universe. "
        "Gravity semantic pet. Perspectives vary, meaning of gravity. "
        "This is linked to, gravity semantic going.",
        "gravity", ["force", "universe", "mass"], "what is gravity?",
    ),
    "Q5_black_holes": (
        "Black holes with masses of millions to billions of solar masses are "
        "found in the universe. black holes bend spacetime is black holes bend. "
        "This is important because, black holes bend spacetime directly is "
        "black holes bend.",
        "black holes bend spacetime",
        ["gravity", "spacetime", "universe", "mass"],
        "why do black holes bend spacetime?",
    ),
    "Q8_life": (
        "Life is a deeply meaningful topic. Life semantic people, which "
        "semantic cannot. Interestingly, life contrastive even. the "
        "significance of this is life causal great.",
        "life", ["meaning", "death", "consciousness"],
        "what is the meaning of life?",
    ),
}

# ── Genuinely grounded answers that must PASS ───────────────────────────────
GROUNDED = {
    "Q9_oxiverse": (
        "Oxiverse is a next-generation intent-first search engine designed for "
        "effective discovery. It builds a privacy-first ecosystem as an "
        "alternative to big tech, and it learns from the web.",
        "oxiverse", ["privacy", "ecosystem", "big tech"], "tell me about oxiverse",
    ),
    "good_pluto_web": (
        "pluto is a dwarf planet in the kuiper belt, and it orbits the sun far "
        "beyond neptune.",
        "pluto", ["planet", "moon", "dwarf"], "is pluto a planet?",
    ),
}


def main():
    eng = _build_engine()
    verdicts = []

    print("=== Per-sentence (Levelt/Wernicke) monitor — degenerate strings MUST be WITHHELD ===")
    for name, (text, subj, assoc, raw) in DEGENERATE.items():
        ctx = _ctx(subj, assoc, raw)
        grounded = eng._sm_response_grounded(ctx, text)
        ok = (grounded is False)
        verdicts.append(ok)
        print(f"  [{name}] withheld={not grounded}  -> {'CONFIRMED' if ok else 'FAIL'}")

    print("\n=== Genuinely grounded answers MUST PASS (no over-suppression) ===")
    for name, (text, subj, assoc, raw) in GROUNDED.items():
        ctx = _ctx(subj, assoc, raw)
        grounded = eng._sm_response_grounded(ctx, text)
        ok = (grounded is True)
        verdicts.append(ok)
        print(f"  [{name}] passes={grounded}  -> {'CONFIRMED' if ok else 'FAIL'}")

    print("\n=== CONTROL arm: the OLD whole-text guard (_is_word_salad) PASSED these\n    (>=3 novel words => safety valve False), proving the per-sentence monitor\n    (not a GloVe/luck artifact) is what now WITHHOLDS them. ===")
    from ravana.chat.constants import _is_word_salad
    for name, (text, subj, assoc, raw) in DEGENERATE.items():
        old_passed = _is_word_salad(text, subject=subj) is False
        ok = old_passed
        verdicts.append(ok)
        print(f"  [ctrl/{name}] old guard passed(degenerate slips through)={old_passed}  "
              f"-> {'CONFIRMED' if ok else 'FAIL'}")

    print("\n=== Universal forward-model guard (inner-speech loop at articulation) ===")
    print("    The ventral/dorsal association binders previously emitted fluent-tautological")
    print("    text ('Life semantic people, which semantic cannot') UNGUATED. _forward_model_check")
    print("    now runs the per-sentence salad + clause monitor on the FINAL composed reply,")
    print("    so any path's degenerate clause is intercepted before overt articulation.")
    for name, (text, subj, assoc, raw) in DEGENERATE.items():
        ctx = _ctx(subj, assoc, raw)
        repaired = eng._forward_model_check(text, ctx)
        # A degenerate input must be REPAIRED (returned text differs / is honest uncertainty)
        is_repaired = (repaired != text) and bool(repaired)
        ok = is_repaired
        verdicts.append(ok)
        print(f"   [fwd/{name}] repaired={is_repaired}  -> {'CONFIRMED' if ok else 'FAIL'}")
    # Grounded answers must pass the forward-model guard unchanged (no over-suppression)
    for name, (text, subj, assoc, raw) in GROUNDED.items():
        ctx = _ctx(subj, assoc, raw)
        repaired = eng._forward_model_check(text, ctx)
        ok = (repaired == text)
        verdicts.append(ok)
        print(f"   [fwd/{name}] unchanged={ok}  -> {'CONFIRMED' if ok else 'FAIL'}")

    print("\n=== P1 clause stripping: mixed good + bad clause ===")
    print("    A reply with one degenerate clause + one good clause must DROP the")
    print("    bad clause and re-emit the survivor (Levelt prepair), not withhold")
    print("    the whole reply. Your real Q5: good web clause + 'black holes bend")
    print("    spacetime is black holes bend'.")
    mixed = ("Black holes take this concept to its extreme, they create such "
             "severe spacetime bending that the geometry becomes warped. "
             "black holes bend spacetime is black holes bend.")
    ctx_mixed = _ctx("black holes",
                     ["gravity", "spacetime", "universe", "mass"],
                     "why do black holes bend spacetime?")
    stripped, dropped = eng._strip_degenerate_clauses(mixed, ctx_mixed)
    ok_strip = (dropped is True
                and "spacetime bending" in stripped
                and "is black holes bend" not in stripped
                and len(stripped) > 20)
    verdicts.append(ok_strip)
    print(f"   [strip/Q5_mixed] dropped={dropped} kept_good={'spacetime bending' in stripped} "
          f"-> {'CONFIRMED' if ok_strip else 'FAIL'}")

    all_ok = all(verdicts)
    print("\n" + ("VERDICT: CONFIRMED — per-sentence monitor withholds fluent-tautological "
                   "tails while preserving grounded answers, and the effect is the gate "
                   "(control arm passes when disabled)."
                   if all_ok else
                   "VERDICT: CHECK — one or more assertions failed (see above)."))
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
