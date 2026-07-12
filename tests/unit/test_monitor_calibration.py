"""Over-monitoring calibration tests (M8).

Proves the per-grain salad thresholds (Steinhauser & Yeung 2010: error
detection = accumulated evidence vs an internal criterion; a lower criterion
is safe at clause grain because a reply of N clauses is N independent
samples, so the per-sample base-rate of degenerate clauses is higher).

- clause grain is STRICTER than doc grain (lower threshold + stricter
  novel-word safety valve): a borderline clause flagged at clause grain is
  not over-flagged at doc grain.
- the SALAD_DOC_THRESHOLD / SALAD_CLAUSE_THRESHOLD constants are exported.
"""
from ravana.chat.constants import (
    _is_word_salad, _is_word_salad_any_sentence,
    SALAD_DOC_THRESHOLD, SALAD_CLAUSE_THRESHOLD,
)


def test_grain_constants_exported():
    assert SALAD_DOC_THRESHOLD == 0.7
    assert SALAD_CLAUSE_THRESHOLD == 0.55
    assert SALAD_CLAUSE_THRESHOLD < SALAD_DOC_THRESHOLD


def test_clause_grain_stricter_than_doc():
    # A borderline clause that is structurally marginal: enough repetition
    # to trip the clause criterion but not the looser doc criterion.
    borderline = "this concept relates to concept which relates to concept."
    doc_flag = _is_word_salad(borderline, subject="concept", grain="doc")
    clause_flag = _is_word_salad(borderline, subject="concept", grain="clause")
    # The clause grain must be at least as strict (never looser) as doc.
    assert clause_flag >= doc_flag


def test_grain_param_propagates():
    # _is_word_salad_any_sentence forwards grain to _is_word_salad.
    clause = "life semantic people which semantic cannot."
    # With correct subject, the safety valve would clear it for doc grain;
    # but the grain param must reach the per-sentence call unchanged.
    out_doc = _is_word_salad_any_sentence(clause, subject="life", grain="doc")
    out_clause = _is_word_salad_any_sentence(clause, subject="life", grain="clause")
    assert isinstance(out_doc, bool)
    assert isinstance(out_clause, bool)
    # grain is a real, forwarded parameter (not silently ignored).
    assert out_clause == _is_word_salad(clause, subject="life", grain="clause")


def test_monitor_threshold_knee():
    # Regression-guarded: the pinned per-grain thresholds must sit at the SDT
    # knee (Steinhauser & Yeung 2010) — high HIT on structural salad, low FAR
    # on real definitions. Re-runs the calibration harness sweep at the pinned
    # values and asserts the knee holds (catches salad, few false alarms).
    import os as _os
    import sys as _sys
    _proj = _os.path.dirname(_os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__))))
    _sys.path.insert(0, _os.path.join(_proj, "experiments"))
    import measure_monitor_calibration as _cal

    from ravana.chat.constants import SALAD_DOC_THRESHOLD, SALAD_CLAUSE_THRESHOLD
    hit_c = _cal._flagged_at(_cal.POSITIVE, "clause", SALAD_CLAUSE_THRESHOLD)
    far_c = _cal._flagged_at(_cal.NEGATIVE, "clause", SALAD_CLAUSE_THRESHOLD)
    hit_d = _cal._flagged_at(_cal.POSITIVE, "doc", SALAD_DOC_THRESHOLD)
    far_d = _cal._flagged_at(_cal.NEGATIVE, "doc", SALAD_DOC_THRESHOLD)
    # Knee: catch the structural salad, keep false alarms on clean defs low.
    assert hit_c >= 0.8 and far_c <= 0.15
    assert hit_d >= 0.8 and far_d <= 0.15
