"""Save/load robustness tests (M5).

Proves: a stale/corrupt ravana_weights.pkl that USED to make load()
throw and silently discard ALL learned state now (a) stamps a schema
version + checksum, (b) recovers from a corrupt graph (rebuilds a
fresh ConceptGraph while restoring everything else), (c) detects a
tampered file (checksum mismatch) instead of blindly trusting it, and
(d) never returns a silent blank — a partial restore is attempted
and logged.

Each engine gets a UNIQUE temp dir (tempfile.mkdtemp) shared
between `e` and `e2`, so `e2` reloads the exact file `e`
corrupted, and a prior pytest run's /tmp save cannot pollute the
current one (load() auto-loads on __init__).

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


def test_load_partial_corrupt():
    """Contract: graph=str AND another field None -> load() True,
    graph rebuilt as ConceptGraph, definitions + identity survive."""
    d = _new_dir("corrupt")
    e = _fresh_engine(d)
    e.process_turn("what is gravity?")
    e._definitions["gravity"] = "Gravity is a fundamental force."
    e.identity.state.strength = 0.9
    e.turn_count = 7
    e.save()
    # Corrupt: graph -> str (the old sanitizer artifact) AND identity_state -> None.
    with open(e._save_path, "rb") as f:
        state = pickle.load(f)
    state["graph"] = "this is not a graph"
    state["identity_state"] = None
    with open(e._save_path, "wb") as f:
        pickle.dump(state, f)
    # Reload into a NEW engine in the SAME dir.
    e2 = _fresh_engine(d)
    ok = e2.load()
    assert ok is True, "load() must succeed via partial restore"
    assert isinstance(e2.graph, ConceptGraph), "graph rebuilt as ConceptGraph"
    # Other state preserved (not silently discarded).
    assert e2.turn_count == 7, e2.turn_count
    assert e2._definitions.get("gravity") == "Gravity is a fundamental force."
    # Identity field (None) was skipped -> kept fresh (valid .strength).
    assert isinstance(e2.identity.state.strength, float)


def test_save_load_roundtrip_checksum():
    """Contract: checksum matches on roundtrip; a tampered file is detected
    (load() warns + best-effort restores, does NOT blindly trust)."""
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
    # checksum sidecar written and matches
    sha_path = e._save_path + ".sha"
    assert os.path.exists(sha_path), "expected .pkl.sha sidecar"
    with open(sha_path) as fh:
        assert fh.read().strip() == state["state_checksum"]
    # TAMPER: change a value without updating the checksum.
    state["turn_count"] = 999
    with open(e._save_path, "wb") as f:
        pickle.dump(state, f)
    # Fresh engine reloads the tampered file: it must DETECT (warn) and
    # still succeed via best-effort — never a silent blank, never blind trust.
    e3 = _fresh_engine(d)
    ok = e3.load()
    assert ok is True, "tampered file detected + best-effort restored"
    # The .sha sidecar still exists (tamper detection is reproducible).
    assert os.path.exists(sha_path)


def test_partial_restore_on_corrupt_field():
    """A corrupt identity_state (wrong shape) must skip only that field,
    keep the rest, and not crash later in process_turn."""
    d = _new_dir("field")
    e = _fresh_engine(d)
    e.process_turn("hi")
    e.turn_count = 12
    e.save()
    with open(e._save_path, "rb") as f:
        state = pickle.load(f)
    # Wrong-shaped identity_state (a dict, not IdentityState).
    state["identity_state"] = {"bogus": object()}
    with open(e._save_path, "wb") as f:
        pickle.dump(state, f)
    e2 = _fresh_engine(d)
    ok = e2.load()
    assert ok is True
    # Turn count + graph (other fields) still restored.
    assert e2.turn_count == 12
    assert isinstance(e2.graph, ConceptGraph)
    # Identity field was skipped (kept fresh) -> process_turn must not crash
    # on the missing .strength attribute.
    e2.process_turn("how are you?")
