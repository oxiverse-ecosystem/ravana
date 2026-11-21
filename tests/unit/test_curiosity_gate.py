"""Curiosity-gated web encoding (default OFF) — contract tests.

The five prior web-loop tweaks all tuned the READ schedule using a self-rated
gap (low FOK) and all failed. D7 adds an untried axis: an EXTERNAL information
gap (re-ask / correction) gating the synchronous web read, behind a flag that
defaults OFF. These tests exercise the gate predicates without a network call
or a full engine boot (CognitiveChatEngine.__new__ avoids GloVe loading).
"""
from __future__ import annotations

import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (PROJ, os.path.join(PROJ, "ravana", "src"), os.path.join(PROJ, "ravana_ml", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.chat.engine import CognitiveChatEngine


def _bare_engine(enabled: bool):
    e = CognitiveChatEngine.__new__(CognitiveChatEngine)
    e._curiosity_gate_enabled = enabled
    e._curiosity_signals = {}
    e._prev_query_words = []
    e._curiosity_reads_this_session = 0
    e.turn_count = 0
    return e


def test_default_off_preserves_prior_behaviour():
    e = _bare_engine(False)
    e._track_curiosity_signal("what is photosynthesis")
    assert e._curiosity_signals == {}          # no tracking when disabled
    assert e._curiosity_gate_allows("photosynthesis") is True
    assert e._curiosity_gate_allows("anything") is True
    assert e._curiosity_reads_this_session == 0  # no budget consumed when off


def test_flag_off_by_default_in_init_source():
    src = open(os.path.join(PROJ, "ravana", "src", "ravana", "chat", "engine.py"),
               encoding="utf-8").read()
    assert 'os.environ.get("RAVANA_CURIOSITY_GATE", "0") == "1"' in src


def test_reatask_is_an_external_gap_signal():
    e = _bare_engine(True)
    e._track_curiosity_signal("what is photosynthesis")
    assert e._curiosity_gate_allows("photosynthesis") is False  # no signal yet
    e.turn_count = 1
    e._track_curiosity_signal("tell me about photosynthesis again")  # re-ask
    assert e._curiosity_need("photosynthesis") is True
    assert e._curiosity_gate_allows("photosynthesis") is True
    assert e._curiosity_reads_this_session == 1


def test_correction_marks_previous_topic():
    e = _bare_engine(True)
    e._track_curiosity_signal("what drives the ocean current")
    e.turn_count = 1
    e._track_curiosity_signal("no, actually that's wrong")
    assert e._curiosity_need("ocean") is True
    assert e._curiosity_need("current") is True


def test_read_budget_is_bounded():
    e = _bare_engine(True)
    # A re-ask creates the external signal; each subsequent signaled turn would
    # want a web read, but the per-session cap of 3 blocks further reads.
    e._track_curiosity_signal("why does amber form")
    for i in range(1, 6):
        e.turn_count = i
        e._track_curiosity_signal("why does amber form again")
        assert e._curiosity_gate_allows("amber") is (i <= 3)
    assert e._curiosity_reads_this_session == 3


def test_signals_decay():
    e = _bare_engine(True)
    e._track_curiosity_signal("what is a comet")
    e.turn_count = 1
    e._track_curiosity_signal("what is a comet again")
    assert e._curiosity_need("comet") is True
    e.turn_count = 20
    e._track_curiosity_signal("unrelated topic entirely")
    assert e._curiosity_need("comet") is False
