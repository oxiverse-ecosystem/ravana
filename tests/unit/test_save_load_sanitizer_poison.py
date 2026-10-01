"""Regression test for the sanitizer snapshot-poisoning defect (2026-10-02).

RED-CAPABLE: on the pre-fix code this file FAILS. Two independent assertions,
one per defect:

  1. WRITER (`_safe_pickle_dump`): attaching ONE unpicklable attribute anywhere
     inside `user_model` used to cause the sanitizer to replace the ENTIRE
     user_model with the string "<unpicklable:UserModel>". Pre-fix, the
     assertion `isinstance(um, UserModel)` fails.

  2. LOADER (`load()`): a genuinely poisoned snapshot (user_model == a plain
     string) used to raise AttributeError deep inside load(), which aborted the
     whole restore and returned False. Pre-fix, the assertion `ok is True`
     fails.

Defect 2 is proven against a HAND-BUILT poisoned snapshot rather than by
re-introducing defect 1, so this half stays meaningful even after the writer is
fixed.
"""
import os
import pickle
import sys
import tempfile

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana_ml", "src"))

from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.user_model import UserModel


class _Unpicklable:
    """A leaf that cannot be pickled, so it triggers the sanitizer path."""

    def __reduce__(self):
        raise TypeError("intentionally unpicklable")


def _new_dir(tag):
    return tempfile.mkdtemp(prefix=f"ravana_poison_{tag}_")


def test_one_bad_leaf_does_not_poison_user_model():
    """WRITER: one unpicklable attribute must NOT cost us the whole UserModel."""
    d = _new_dir("writer")
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d)
    e.process_turn("i am a vegetarian who loves hiking")

    e.user_model._evil = _Unpicklable()
    e.save()

    with open(e._save_path, "rb") as f:
        state = pickle.load(f)

    um = state["user_model"]
    assert not isinstance(um, str), (
        "user_model was replaced by a placeholder string "
        f"({um!r}) - one unpicklable attribute must not discard the whole model"
    )
    assert isinstance(um, UserModel), type(um).__name__
    # The rest of the model must have survived, not just its type.
    assert hasattr(um, "edge_reactivations")
    assert hasattr(um, "query_concepts")
    # The offending leaf is what gets dropped, and only that.
    assert not isinstance(getattr(um, "_evil", None), _Unpicklable)


def test_load_degrades_instead_of_aborting_on_poisoned_user_model():
    """LOADER: a placeholder user_model must degrade, not abort the restore."""
    d = _new_dir("loader")
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d)
    e.process_turn("i am a vegetarian who loves hiking")
    e.turn_count = 12
    e.save()

    # Hand-build the poisoned snapshot the old writer used to produce.
    with open(e._save_path, "rb") as f:
        state = pickle.load(f)
    state["user_model"] = "<unpicklable:UserModel>"
    with open(e._save_path, "wb") as f:
        pickle.dump(state, f)

    e2 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d)
    ok = e2.load()

    assert ok is True, "load() must not abort on a placeholder user_model"
    # Unrelated state must still be restored (this is what the abort destroyed).
    assert e2.turn_count == 12, e2.turn_count
    # And the engine must be usable afterwards.
    e2.process_turn("what do you know about me?")