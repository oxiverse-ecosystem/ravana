"""Unit tests for the cross-turn self-consistency monitor.

Key cases from the spec:
  - "is AI beneficial" followed by "is AI harmful" -> conflict_detected=True
  - "what is gravity" followed by "tell me about gravity" -> conflict_detected=False
"""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana-v2", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana_ml", "src"))

# A tiny fake glove so embedding extraction works without loading 1.5GB.
import numpy as _np


def _fake_glove(word):
    # Deterministic pseudo-embedding: hash the word -> unit vector offset.
    h = hash(word) % 50
    v = _np.zeros(50, dtype=float)
    v[h % 50] = 1.0
    return tuple(v.tolist())


from ravana.chat.consistency_monitor import ConsistencyMonitor


def _mon():
    return ConsistencyMonitor(glove_fn=_fake_glove, mode="annotate")


def test_contradiction_flagged():
    m = _mon()
    r1 = m.check("AI is beneficial to society.", turn=1)
    assert not r1.conflict_detected  # first claim, no prior
    r2 = m.check("AI is harmful to society.", turn=5)
    assert r2.conflict_detected, "beneficial->harmful should flag"


def test_consistent_not_flagged():
    m = _mon()
    m.check("Gravity is a force that pulls things down.", turn=1)
    r2 = m.check("Tell me about gravity and how it works.", turn=4)
    assert not r2.conflict_detected, "same-topic consistent should NOT flag"


def test_no_monitor_no_crash():
    m = ConsistencyMonitor(glove_fn=None, mode="annotate")
    r = m.check("", turn=1)
    assert not r.conflict_detected


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]:
        fn()
        print("PASS", fn.__name__)
    print("ALL CONSISTENCY-MONITOR TESTS PASSED")
