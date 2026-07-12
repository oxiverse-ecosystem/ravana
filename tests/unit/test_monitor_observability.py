"""M10 observability tests: structured monitor log + fluent-tautology CI gate.

Brain analog: Pe explicit-error-signaling component (Steinhauser & Yeung 2010)
makes the monitor's decision observable (not just the Ne/ERN evidence). These
tests assert (1) guard fires append a structured record naming the monitor and
the dropped clause, and (2) the fluent-tautological signature is dead on a
fixed corpus (regression-proof CI gate).
"""
import os
import sys

import pytest

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)
_SRC = os.path.join(_PROJ, "ravana", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
_ML = os.path.join(_PROJ, "ravana_ml", "src")
if _ML not in sys.path:
    sys.path.insert(0, _ML)

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402
from ravana.chat.monitor_gate import detects_fluent_tautology  # noqa: E402


def _bare_engine():
    """Build a CognitiveChatEngine without loading saved weights / GloVe"""
    eng = CognitiveChatEngine.__new__(CognitiveChatEngine)
    eng._monitor_log = []
    eng._trace_enabled = False
    eng._disable_grounding_gate = True  # keep the monitor isolated from graph
    return eng


def test_monitor_report_empty():
    eng = _bare_engine()
    rep = eng.monitor_report()
    assert rep["total_fires"] == 0
    assert rep["by_monitor"] == {}
    assert rep["recent"] == []


def test_log_monitor_fire_record_shape():
    eng = _bare_engine()
    eng._log_monitor_fire("clause-strip", "Life semantic people, which semantic cannot.", "salad")
    rep = eng.monitor_report()
    assert rep["total_fires"] == 1
    assert rep["by_monitor"]["clause-strip"] == 1
    assert rep["by_reason"]["salad"] == 1
    entry = rep["recent"][0]
    assert set(entry.keys()) == {"ts", "path", "monitor", "dropped_clause", "reason"}
    assert entry["monitor"] == "clause-strip"
    assert entry["dropped_clause"] == "Life semantic people, which semantic cannot."
    assert isinstance(entry["ts"], float)


def test_strip_degenerate_clause_logs_fire():
    """A degenerate clause run through _strip_degenerate_clauses appends a
    structured monitor-log entry naming the monitor + dropped clause (CI gate).

    We monkeypatch the SM monitor to return False (simulating it catching the
    clause) so the logging wiring is exercised without needing a full graph;
    the salad-detection itself is owned by M4/M8 tests, not this observability
    test.
    """
    eng = _bare_engine()
    eng._disable_grounding_gate = False
    eng._sm_response_grounded = lambda ctx, c, skip_step1=False: False

    class _Ctx:
        subject = "life"
        raw_input = "what is life"

    repaired, dropped = eng._strip_degenerate_clauses(
        "Life semantic people, which semantic cannot.", _Ctx())
    assert dropped is True
    rep = eng.monitor_report()
    assert rep["total_fires"] >= 1
    assert any(e["monitor"] == "clause-strip" for e in rep["recent"])
    assert any("semantic cannot" in e["dropped_clause"] for e in rep["recent"])


# ── Fluent-tautology signature CI gate (fixed corpus, Q3/Q5/Q8 class) ──────
_POSITIVE = [
    ("life", "Life semantic people, which semantic cannot."),
    ("life", "From another angle, life contrastive way, which contrastive even."),
    ("black holes", "black holes bend is black holes bend."),
    ("trust", "Trust semantic related connected linked means does."),
    ("gravity", "Gravity causal great even cannot which."),
]
_NEGATIVE = [
    ("gravity", "Gravity is the force that pulls masses together."),
    ("trust", "Trust means relying on someone because you believe they care."),
    ("black holes", "Black holes bend spacetime so strongly that light cannot escape."),
    ("life", "Life is the process by which organisms grow, reproduce, and adapt."),
]


@pytest.mark.parametrize("subject,text", _POSITIVE)
def test_fluent_tautology_ci_gate_positive(subject, text):
    # The fluent-tautological signature must be detected (it is the lesion we
    # killed in M4). If this returns False, the lesion silently reopened.
    assert detects_fluent_tautology(text, subject=subject) is True


@pytest.mark.parametrize("subject,text", _NEGATIVE)
def test_fluent_tautology_ci_gate_negative(subject, text):
    # Real definitions must NOT be flagged as fluent-tautological.
    assert detects_fluent_tautology(text, subject=subject) is False
