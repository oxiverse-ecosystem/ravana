"""FIX-RV-16 regression: the online learning path must be ALIVE.

Before this card, `learning_count` was structurally always 0 in an
interactive session and novel vocabulary never entered the ConceptGraph,
because `_learn_from_text` (the only node-minting routine) is reachable only
from the web fetchers and `process_turn` never called it.

These tests assert the acceptance criterion directly: RAVANA learns from ONE
conversation, online, with no rebuild.

Every check uses a real `assert`, never `return bool` — pytest reports a test
that returns a non-None value as PASSED regardless of its value.
"""

import os
import glob

import pytest

os.environ.setdefault("RAVANA_OFFLINE", "1")

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402


# Words the engine's seed graph is NOT expected to know. Chosen to be ordinary
# English content words, not to match any implementation detail: the point is
# that the route admits whatever is novel, not these in particular.
_PROBE_NOVEL = [
    "kalamazoo", "trombone", "espadrille", "quokka", "obelisk",
    "tessellate", "winslow", "kumquat", "zeppelin", "marzipan",
]


def _fresh(suffix):
    """A clean isolated suffix: no pickle contamination from a crashed worker."""
    for f in glob.glob(f"weights/*{suffix}*.pkl"):
        try:
            os.remove(f)
        except OSError:
            pass
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)


def _labels(eng):
    return {n.label.lower() for n in eng.graph.nodes.values() if n and n.label}


def test_learning_count_starts_at_zero():
    """Baseline: a fresh engine has learned nothing yet."""
    eng = _fresh("rv16t_zero")
    try:
        assert int(getattr(eng, "_learning_count", 0) or 0) == 0, \
            "a fresh engine should start with learning_count 0"
    finally:
        eng.stop_background_learning()


def test_novel_vocabulary_enters_graph_and_learning_count_rises():
    """THE acceptance test: one conversation, no rebuild, real graph growth."""
    eng = _fresh("rv16t_grow")
    try:
        before_labels = _labels(eng)
        before_nodes = len(eng.graph.nodes)
        before_edges = len(eng.graph.edges)
        before_lc = int(getattr(eng, "_learning_count", 0) or 0)

        # A disclosure full of words the engine cannot already know.
        words = " ".join(_PROBE_NOVEL)
        eng.process_turn(f"i spent the afternoon near the {words} collection")

        after_labels = _labels(eng)
        lc = int(getattr(eng, "_learning_count", 0) or 0)

        admitted = sorted((after_labels - before_labels) - {"user"})
        assert lc > before_lc, (
            f"learning_count did not rise: {before_lc} -> {lc}. The online "
            "learning path is dead again."
        )
        assert len(eng.graph.nodes) > before_nodes, (
            f"no node growth: {before_nodes} -> {len(eng.graph.nodes)}"
        )
        assert admitted, (
            "no novel vocabulary reached the graph. Admitted: "
            f"{sorted(after_labels - before_labels)}"
        )
        # The novel words must be ACTUALLY the ones admitted, not some
        # unrelated token that happened to pass the gate.
        overlap = [w for w in _PROBE_NOVEL if w in after_labels]
        assert len(overlap) >= 3, (
            f"expected several probe words in the graph, got {overlap}"
        )
        assert len(eng.graph.edges) > before_edges, (
            f"admitted nodes were not wired: edges {before_edges} -> "
            f"{len(eng.graph.edges)}"
        )
    finally:
        eng.stop_background_learning()


def test_admitted_words_are_connected_by_typed_edges():
    """Admission without typed edges is not admission to a ConceptGraph."""
    eng = _fresh("rv16t_typed")
    try:
        for w in ("peregrine", "tessellate", "obelisk", "quokka"):
            eng.process_turn(f"i saw a {w} on the walk")
        ck = eng._concept_keywords
        for w in ("peregrine", "tessellate", "obelisk", "quokka"):
            nids = ck.get(w, [])
            assert nids, f"{w} was not admitted to the graph"
            has_edge = any(
                any(tid != nid for tid, _e in eng.graph.get_outgoing(nid))
                for nid in nids
            )
            assert has_edge, f"{w} was admitted as an isolated node (no edge)"
    finally:
        eng.stop_background_learning()


def test_growth_survives_save_load():
    """Growth must be durable, not an in-memory illusion.

    Guards the pickle-contamination pitfall: a crashed worker's stale pkl
    defeats suffix isolation and mimics learning.
    """
    suffix = "rv16t_persist"
    eng = _fresh(suffix)
    try:
        eng.process_turn("the peregrine and the quokka were both near the obelisk")
        saved_lc = int(getattr(eng, "_learning_count", 0) or 0)
        saved_nodes = len(eng.graph.nodes)
        assert saved_lc > 0, "nothing was learned, so persistence is untestable"
    finally:
        eng.stop_background_learning()
        eng.save()

    eng2 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)
    try:
        eng2.load()
        lc = int(getattr(eng2, "_learning_count", 0) or 0)
        assert lc >= saved_lc, (
            f"learning_count regressed across save/load: {saved_lc} -> {lc}"
        )
        labels = _labels(eng2)
        for w in ("peregrine", "quokka", "obelisk"):
            assert w in labels, (
                f"{w} did not survive save/load — the graph is rebuilt from "
                "seeds on load, so first-party growth must be re-persisted"
            )
        assert len(eng2.graph.nodes) >= saved_nodes - 3, (
            f"graph shrank badly across save/load: {saved_nodes} -> "
            f"{len(eng2.graph.nodes)}"
        )
    finally:
        eng2.stop_background_learning()


def test_structural_junk_is_refused():
    """The first-party route must not become a junk hole.

    Keyboard mashes, cyclic repeats and vowel-less strings are rejected by the
    SHARED structural floor, so the route cannot drift from the web gate's
    notion of junk.
    """
    eng = _fresh("rv16t_junk")
    try:
        before = _labels(eng)
        eng.process_turn("xkcdxk qwertyui asdfgh olol bcdfg my keyboard mash jjjj")
        after = _labels(eng)
        leaked = (after - before) - {"user"}
        for bad in ("xkcdxk", "qwertyui", "asdfgh", "olol", "bcdfg", "jjjj"):
            assert bad not in leaked, (
                f"structural junk {bad!r} was admitted to the concept graph"
            )
    finally:
        eng.stop_background_learning()


def test_no_vocabulary_hardcoded_in_the_module():
    """The route must be generic: naming a content word would fake learning.

    This asserts the NO-HARDCODING rule mechanically. The probe words are
    ordinary nouns/verbs chosen because the implementation has no reason to
    contain any of them.
    """
    import ravana.chat.first_party_admission as fpa
    src = open(fpa.__file__, encoding="utf-8").read()
    # Strip comments and docstrings: prose may legitimately mention a word,
    # but CODE may not contain a vocabulary of them.
    code_lines = []
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        code_lines.append(line.split("  #")[0])
    code = "\n".join(code_lines)
    for w in _PROBE_NOVEL:
        assert f'"{w}"' not in code and f"'{w}'" not in code, (
            f"{w!r} is hardcoded in the admission module — the route must "
            "extract vocabulary generically, not from a frozen list"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
