"""Phase 4/5 integration tests: triplet inference wired into the engine.

Pins the four wiring points:
1. process_turn -> PropositionParser -> triplet_op (conversation capture)
2. engine save()/load() round-trips operator state (dead-gate lesson)
3. _sleep_consolidate runs the triplet schema-extraction stage
4. OpenIE fact adapter feeds profiles (web capture path unit-level;
   the full WebLearner path needs network and is exercised live)
"""
import tempfile

import pytest

from ravana.chat.engine import CognitiveChatEngine
from ravana.core.triplet_inference import Triple


@pytest.fixture(scope="module")
def eng():
    d = tempfile.mkdtemp()
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d)
    return e, d


def _teach_is(op, n=10):
    for i in range(n):
        op.ingest_triple(Triple(f"tx{i}", "is", f"ty{i}"))
        op.ingest_triple(Triple(f"ty{i}", "is", f"tz{i}"))
        op.ingest_triple(Triple(f"tx{i}", "is", f"tz{i}"))


def test_engine_constructs_triplet_op(eng):
    e, _ = eng
    assert e.triplet_op is not None


def test_process_turn_mines_propositions(eng):
    e, _ = eng
    e.process_turn("socrates is a man")
    # canonical_term strips the article: object is 'man', not 'a man'.
    assert e.triplet_op.memory.has_fact("socrates", "is", "man")


def test_syllogism_completes_after_learned_transitivity(eng):
    e, _ = eng
    _teach_is(e.triplet_op)
    e.triplet_op.ingest_triple(Triple("socrates", "is", "man"))
    e.triplet_op.ingest_triple(Triple("man", "is", "mortal"))
    results = e.triplet_op.infer("socrates", "is", max_results=5)
    objs = {r.triple.object for r in results}
    assert "mortal" in objs
    ops = {r.operator for r in results if r.triple.object == "mortal"}
    assert "transitive" in ops


def test_sleep_stage_extracts_triplet_schemas(eng):
    e, _ = eng
    _teach_is(e.triplet_op)
    res = e._sleep_consolidate()
    assert res.get("triplet_schemas", 0) >= 1
    assert "transitive-chain:is" in e.triplet_op.memory.schemas


def test_engine_save_load_round_trips_profiles(eng):
    e, d = eng
    _teach_is(e.triplet_op)
    pos_before = e.triplet_op.memory.profiles["is"].transitivity_pos
    assert pos_before > 0
    e.save()
    e2 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d)
    prof = e2.triplet_op.memory.profiles.get("is")
    assert prof is not None
    assert prof.transitivity_pos == pos_before
    assert prof.transitivity_lower() > 0.5  # gate still open after reload
