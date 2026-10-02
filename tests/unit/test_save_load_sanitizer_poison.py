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

def test_sanitizer_preserves_a_dict_subclass_type():
    """WRITER, defect 3 (FIX-RV-23): the sanitizer must not flatten a mapping
    SUBCLASS to a plain dict.

    RED-CAPABLE: on the pre-fix code this FAILS. The sanitizer rebuilt every
    dict as a literal `{...}`, so a `defaultdict(list)` came back a `dict` --
    a silent TYPE change on the load path.

    Why it matters here, concretely: `pickle.dumps(graph)` fails (the graph
    reaches a sqlite3.Connection through the engine's hooks), so every save
    takes the sanitizer path. `ConceptGraph` is not a container, so it took
    the shallow-clone branch and its `_outgoing` attribute was rebuilt as a
    plain dict. `add_edge` then raised KeyError on
    `self._outgoing[source].append(...)`, so every graph write on a RESUMED
    engine failed and was swallowed -- the engine looked like it had stopped
    learning across a restart
    (test_sleep_episodic_replay.py::test_sleep_consolidation_survives_repeated_runs).

    Pinned directly on the sanitizer, without booting an engine, so the
    invariant is stated once and does not depend on the graph's hooks still
    being unpicklable.
    """
    from collections import Counter, OrderedDict, defaultdict

    d = _new_dir("subclass")
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d)
    state = {
        "dd": defaultdict(list, {1: ["a"]}),
        "od": OrderedDict([("k", "v")]),
        "cnt": Counter("aab"),
        "plain": {"x": 1},
    }
    # Force the sanitizer path with one unpicklable leaf.
    state["bad"] = _Unpicklable()
    assert e._safe_pickle_dump(state, os.path.join(d, "sub.pkl")) is True

    with open(os.path.join(d, "sub.pkl"), "rb") as f:
        back = pickle.load(f)

    assert type(back["dd"]) is defaultdict, type(back["dd"])
    assert back["dd"].default_factory is list, back["dd"].default_factory
    assert back["dd"][1] == ["a"], back["dd"][1]
    # THE load-path-critical behaviour: a missing key must still auto-vivify.
    assert back["dd"][99] == [], "defaultdict semantics lost across save/load"
    assert type(back["od"]) is OrderedDict, type(back["od"])
    assert type(back["cnt"]) is Counter, type(back["cnt"])
    assert back["cnt"]["a"] == 2, back["cnt"]
    assert type(back["plain"]) is dict
    e.stop_background_learning()


def test_a_resumed_engine_can_still_write_to_its_graph():
    """END-TO-END consequence of defect 3, at the seam a user experiences.

    A graph edge added AFTER a save/load must actually exist afterwards. This
    is the invariant the KeyError silently broke; asserting it at the graph
    level (not the sleep metric) keeps the signal even if the consolidation
    reporting changes.
    """
    d = _new_dir("resumed_write")
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d)
    a = e.graph.add_node(label="alpha_probe")
    b = e.graph.add_node(label="beta_probe")
    e.save()
    e.stop_background_learning()

    e2 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d)
    a2 = e2.graph.add_node(label="gamma_probe")
    b2 = e2.graph.add_node(label="delta_probe")
    e2.graph.add_edge(a2.id, b2.id, weight=0.4,
                      relation_type="episodic", confidence=0.5)
    edge = e2.graph.get_edge(a2.id, b2.id)
    assert edge is not None, (
        "a resumed engine could not write a new edge: the adjacency index lost "
        "its defaultdict behaviour across save/load")
    assert edge.relation_type == "episodic", edge.relation_type
    e2.stop_background_learning()
