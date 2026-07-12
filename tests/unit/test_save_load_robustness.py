"""Save/load robustness tests (M5).

Proves: a stale/corrupt ravana_weights.pkl that USED to make load()
throw and silently discard ALL learned state now (a) stamps a schema
version + checksum, (b) recovers from a corrupt graph (rebuilds a
fresh ConceptGraph while restoring everything else), and (c) never
returns a silent blank — a partial restore is attempted and logged.

Each engine gets a UNIQUE temp dir (tempfile.mkdtemp) so a prior
pytest run's save cannot pollute the current one (the load() auto-loads
on __init__), and `e2` reuses `e`'s dir so it loads the very
file `e` corrupted.

Run from repo root:
    python -m pytest tests/unit/test_save_load_robustness.py -v
"""
import os
import sys
import pickle
import tempfile

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana_ml", "src"))

from ravana.chat.engine import CognitiveChatEngine
from ravana_ml.graph import ConceptGraph


def _fresh_engine(data_dir):
    # Caller passes a SHARED unique dir so e2 loads e's corrupted file.
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=data_dir)
    return e


def _new_dir(tag):
    return tempfile.mkdtemp(prefix=f"ravana_m5_{tag}_")


def test_schema_stamp_and_checksum_file():
    d = _new_dir("stamp")
    e = _fresh_engine(d)
    e.process_turn("hi")
    msg = e.save()
    assert "saved" in msg, msg
    # schema_version + state_checksum are in the pickled state
    with open(e._save_path, "rb") as f:
        state = pickle.load(f)
    assert state.get("schema_version") == e.SAVE_SCHEMA_VERSION
    assert isinstance(state.get("state_checksum"), str)
    # checksum sidecar written
    sha_path = e._save_path + ".sha"
    assert os.path.exists(sha_path), "expected .pkl.sha sidecar"
    with open(sha_path) as fh:
        assert fh.read().strip() == state["state_checksum"]


def test_partial_restore_on_corrupt_graph():
    """A str graph (the old sanitizer artifact) must NOT wipe all state."""
    d = _new_dir("corrupt")
    e = _fresh_engine(d)
    e.process_turn("what is gravity?")
    e._definitions["gravity"] = "Gravity is a fundamental force."
    e.turn_count = 7
    e.save()
    # Corrupt the graph in the saved pkl: replace with a str.
    with open(e._save_path, "rb") as f:
        state = pickle.load(f)
    state["graph"] = "this is not a graph"
    with open(e._save_path, "wb") as f:
        pickle.dump(state, f)
    # Reload into a NEW engine in the SAME dir — should partially
    # restore, not blank.
    e2 = _fresh_engine(d)
    ok = e2.load()
    assert ok is True, "load() must succeed via partial restore"
    assert isinstance(e2.graph, ConceptGraph), "graph rebuilt as ConceptGraph"
    # Other state preserved (not silently discarded).
    assert e2.turn_count == 7, e2.turn_count
    assert e2._definitions.get("gravity") == "Gravity is a fundamental force."


def test_partial_restore_on_corrupt_field():
    """A corrupt identity_state must skip only that field, keep the rest."""
    d = _new_dir("field")
    e = _fresh_engine(d)
    e.process_turn("hi")
    e.turn_count = 12
    e.save()
    with open(e._save_path, "rb") as f:
        state = pickle.load(f)
    # Make identity_state un-restorable (wrong shape).
    state["identity_state"] = {"bogus": object()}
    with open(e._save_path, "wb") as f:
        pickle.dump(state, f)
    e2 = _fresh_engine(d)
    ok = e2.load()
    assert ok is True
    # Turn count + graph (other fields) still restored.
    assert e2.turn_count == 12
    assert isinstance(e2.graph, ConceptGraph)
    # Identity field was skipped (kept fresh) — process_turn must not crash
    # on the missing .strength attribute.
    e2.process_turn("how are you?")
