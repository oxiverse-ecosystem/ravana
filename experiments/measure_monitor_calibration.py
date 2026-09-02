"""Monitor calibration harness (M8).

Mirrors the metacognitive monitoring-as-decision literature (Steinhauser &
Yeung 2010, J Neurosci; Maniscalco, Charles & Peters 2024; Sherman, Seth &
Barrett 2018): the salad detector is an error-monitor with an internal
criterion. A DOCUMENT is one sample (loose criterion, few false alarms);
a REPLY of N clauses is N independent samples (stricter criterion is safe
because the per-sample base-rate of degenerate clauses is higher).

This harness calibrates the STRUCTURAL salad detector (_is_word_salad) on a
small labeled corpus of structural word-salad vs clean clauses, sweeping the
per-grain threshold and reporting the SDT-style Hit / False-Alarm trade-off so
the knee can be pinned. (The semantic-tautology class — "Life semantic
people, which semantic cannot" — is owned by the Situation-Model monitor's
reference/coherence step, not by _is_word_salad; that class is covered by the
M10 fluent-tautology CI gate, not this threshold sweep.)

Run:  python experiments/measure_monitor_calibration.py
"""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana_ml", "src"))

from ravana.chat.constants import _is_word_salad

# Labeled corpus of STRUCTURAL salad (pos = should flag) vs clean clauses
# (neg = should pass). Each pair: (text, subject).
POSITIVE = [
    ("the the the the the the the the the the the", "concept"),
    ("cat cat cat cat cat cat cat cat cat cat cat", "concept"),
    ("run run run run run run run run run run run", "concept"),
    ("pet pet pet semantic causal even great", "concept"),
    ("this is the the is this the the is this the the", "concept"),
    ("word word word word word word word word word word word", "concept"),
    ("blah blah blah blah blah blah blah blah blah blah blah", "concept"),
]
NEGATIVE = [
    ("Gravity is a fundamental force of attraction between masses with mass.", "gravity"),
    ("Trust is the belief that others will not exploit your vulnerability.", "trust"),
    ("Black holes are regions of spacetime where gravity is so strong nothing escapes.", "black holes"),
    ("The meaning of gravity is the gravitational attraction between masses.", "gravity"),
    ("Dreams are sequences of thoughts and images during sleep.", "dreams"),
    ("A photon is a particle of light that carries electromagnetic force.", "photon"),
    ("Trust reduces uncertainty in cooperation and builds reciprocity.", "trust"),
    ("Spacetime curves near mass, which is why objects fall toward each other.", "spacetime"),
]


def _flagged_at(corpus, grain, thr):
    """Fraction of `corpus` flagged as salad at threshold `thr` for `grain`."""
    import ravana.chat.constants as C
    saved = C.SALAD_CLAUSE_THRESHOLD if grain == "clause" else C.SALAD_DOC_THRESHOLD
    if grain == "clause":
        C.SALAD_CLAUSE_THRESHOLD = thr
    else:
        C.SALAD_DOC_THRESHOLD = thr
    try:
        n = 0
        for t, subj in corpus:
            if _is_word_salad(t, subject=subj, grain=grain):
                n += 1
        return n / max(len(corpus), 1)
    finally:
        if grain == "clause":
            C.SALAD_CLAUSE_THRESHOLD = saved
        else:
            C.SALAD_DOC_THRESHOLD = saved


def sweep(grain):
    print(f"\n=== grain={grain!r} sweep ===")
    print(f"{'thr':>5} | {'HIT (salad)':>12} | {'FAR (clean)':>12}")
    for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        hit = _flagged_at(POSITIVE, grain, thr)
        far = _flagged_at(NEGATIVE, grain, thr)
        print(f"{thr:>5.2f} | {hit:>12.2f} | {far:>12.2f}")


def main():
    sweep("clause")
    sweep("doc")
    from ravana.chat.constants import SALAD_DOC_THRESHOLD, SALAD_CLAUSE_THRESHOLD
    hit_c = _flagged_at(POSITIVE, "clause", SALAD_CLAUSE_THRESHOLD)
    far_c = _flagged_at(NEGATIVE, "clause", SALAD_CLAUSE_THRESHOLD)
    hit_d = _flagged_at(POSITIVE, "doc", SALAD_DOC_THRESHOLD)
    far_d = _flagged_at(NEGATIVE, "doc", SALAD_DOC_THRESHOLD)
    print("\n=== PINNED ===")
    print(f"clause thr={SALAD_CLAUSE_THRESHOLD}: HIT={hit_c:.2f} FAR={far_c:.2f}")
    print(f"doc    thr={SALAD_DOC_THRESHOLD}: HIT={hit_d:.2f} FAR={far_d:.2f}")
    # M8 calibration target: catch the structural salad (HIT high) while
    # keeping false alarms on real definitions low (FAR low).
    assert hit_c >= 0.8, f"clause HIT too low: {hit_c}"
    assert far_c <= 0.15, f"clause FAR too high: {far_c}"
    assert hit_d >= 0.8, f"doc HIT too low: {hit_d}"
    assert far_d <= 0.15, f"doc FAR too high: {far_d}"
    print("VERDICT: CONFIRMED — pinned per-grain thresholds sit near the "
          "SDT knee (high HIT, low FAR) for both grains.")


if __name__ == "__main__":
    main()
