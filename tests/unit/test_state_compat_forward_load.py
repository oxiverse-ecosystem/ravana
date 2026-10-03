"""Round t_eee413e7 — a missing class in the snapshot must not blank the mind.

THE DEFECT THIS LOCKS DOWN
--------------------------
``weights/ravana_weights*.pkl`` is the canonical self and is shared across
rounds, but each round runs its own commit. Round t_e45928d1 saved the snapshot
from a branch carrying ``ravana.chat.evaluative_polarity``; that branch never
reached main. Every later ``_load()`` raised ``ModuleNotFoundError``, returned
False, and the engine booted BLANK — measured by the t_eee413e7 probe as
``turn_count=0`` with a 116-node seed graph. The whole accumulated self, gone,
because of one class reference, reported only as a ``[Load error]`` print the
loop never reads.

These tests assert the CONTRACT, not the prose:
  * a snapshot referencing an unknown ravana class loads, and everything else
    in it survives;
  * the dropped class is reported, not swallowed;
  * a missing THIRD-PARTY class still raises (a real env fault must be loud);
  * the legacy ravana_chat alias still resolves;
  * the placeholder survives a re-save, so a degraded load can be written back.
"""
import os
import pickle
import sys

import pytest

# Derive this tree's source roots from THIS FILE. The previous revision
# hardcoded the main checkout's absolute path, which prepends the main repo
# to sys.path and makes a worktree run import the main tree's ravana/ --
# exactly the wrong-tree failure this file exists to help avoid.
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana", "src"), os.path.join(PROJ, "ravana-v2", "src")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from ravana.chat import state_compat
from ravana.chat.state_compat import ravana_unpickler, report_dropped

# A module that genuinely does not exist, standing in for a class that only
# existed on the branch that wrote the snapshot. The byte-splice needs a
# same-length partner, so the fake leaf is padded to the real leaf's width --
# derived, never hand-counted.
from ravana.chat.personal_fact_store import PersonalFact  # noqa: E402

_REAL_LEAF = PersonalFact.__module__.split(".")[-1]
REAL_MODULE = PersonalFact.__module__
FAKE_MODULE = "ravana.chat." + ("x" * len(_REAL_LEAF))
assert len(REAL_MODULE) == len(FAKE_MODULE)
assert FAKE_MODULE not in sys.modules


def _build_pickle_with_missing_class(extra, real_mod=REAL_MODULE,
                                     fake_mod=FAKE_MODULE):
    """Return pickle bytes in which `real_mod` is named as the absent `fake_mod`.

    ``pickle.dump`` correctly refuses to write a class whose module does not
    exist, so the bytes are produced by pickling something real and then
    splicing the module name. The two names are the SAME LENGTH so every
    length-prefix in the stream stays valid. This yields exactly the on-disk
    shape of a snapshot written on the branch that had the module.
    """
    assert len(real_mod) == len(fake_mod), "byte-splice needs equal lengths"
    blob = pickle.dumps(extra)
    assert real_mod.encode() in blob, "anchor module not present in pickle"
    return blob.replace(real_mod.encode(), fake_mod.encode())


def _roundtrip_bytes(blob, path):
    with open(path, "wb") as f:
        f.write(blob)
    with open(path, "rb") as f:
        return ravana_unpickler(f).load()


def _roundtrip(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    with open(path, "rb") as f:
        return ravana_unpickler(f).load()


def test_missing_ravana_class_does_not_abort_the_load(tmp_path):
    from ravana.chat.personal_fact_store import PersonalFact  # real class to rename

    blob = _build_pickle_with_missing_class(
        {"turn_count": 412, "identity": {"strength": 0.62}, "ghost": PersonalFact("i", "job", "printer")})
    state = _roundtrip_bytes(blob, tmp_path / "s.pkl")
    # The whole dict came back, not an exception and not a partial.
    assert state["turn_count"] == 412, "sibling state was lost with the missing class"
    assert state["identity"]["strength"] == 0.62
    # The missing class degraded to an inert placeholder, not a live object.
    assert repr(state["ghost"]) == f"<dropped {FAKE_MODULE}.PersonalFact>"
    assert not hasattr(state["ghost"], "value"), \
        "the placeholder must not answer as if it were the real component"


def test_dropped_class_is_reported_not_swallowed(tmp_path):
    from ravana.chat.personal_fact_store import PersonalFact

    blob = _build_pickle_with_missing_class({"g": PersonalFact("i", "job", "printer")})
    _roundtrip_bytes(blob, tmp_path / "s.pkl")
    msg = report_dropped()
    assert msg, "a degraded load must say so -- silence is the original bug"
    assert f"{FAKE_MODULE}.PersonalFact" in msg
    assert "rest of the self was restored" in msg


def test_report_is_empty_on_a_clean_load(tmp_path):
    _roundtrip({"turn_count": 7}, tmp_path / "s.pkl")
    assert report_dropped() == "", "a clean load must not cry degradation"


def test_missing_third_party_class_still_raises(tmp_path):
    """A missing numpy/torch is a real environment fault and must be loud."""
    with open(tmp_path / "x.pkl", "wb") as f:
        f.write(b"")  # any readable handle; find_class is what's under test
    up = ravana_unpickler(open(tmp_path / "x.pkl", "rb"))
    with pytest.raises((ModuleNotFoundError, ImportError)):
        up.find_class("some_thirdparty_pkg.sub", "Thing")


def test_legacy_entry_module_alias_still_resolves(tmp_path):
    """A snapshot saved from `python scripts/ravana_chat.py` must still load.

    `scripts/ravana_chat` and `ravana_chat` are not the same length, so the
    module is renamed in a way that keeps the stream intact: the class is
    looked up under the alias directly through find_class.
    """
    from ravana.chat.engine import CognitiveChatEngine

    with open(tmp_path / "x.pkl", "wb") as f:
        f.write(b"")
    up = ravana_unpickler(open(tmp_path / "x.pkl", "rb"))
    assert up.find_class("ravana_chat", "CognitiveChatEngine") is CognitiveChatEngine
    assert up.find_class("__main__", "CognitiveChatEngine") is CognitiveChatEngine
    assert report_dropped() == "", "an alias hit is a resolution, not a degradation"


def test_placeholder_class_is_stable_per_name():
    a = state_compat._placeholder_class("ravana.chat.x", "Y")
    b = state_compat._placeholder_class("ravana.chat.x", "Y")
    c = state_compat._placeholder_class("ravana.chat.x", "Z")
    assert a is b, "same missing name must resolve to one class"
    assert a is not c, "different missing names must not collide"


def test_placeholder_absorbs_state_and_resaves():
    """Re-saving a degraded snapshot must not crash on the placeholder."""
    o = state_compat._placeholder_class("ravana.chat.q", "Z")()
    o.__setstate__({"anything": [1, 2, 3]})
    again = pickle.loads(pickle.dumps(o))
    assert repr(again) == "<dropped ravana.chat.q.Z>"
