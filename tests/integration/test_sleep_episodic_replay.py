"""Test sleep-phase episodic pair replay — Phase 3b4 consolidation."""
import sys, os, uuid
os.environ["RAVANA_OFFLINE"] = "1"

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
          os.path.join(PROJ, "ravana", "src"),
          os.path.join(PROJ, "ravana-v2", "src")):
    sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine
from ravana.core.episodic_binder import EpisodicBinder, EpisodicBinderConfig


# Each test needs its own persisted self. EpisodicBinder marks a pair
# `consolidated` once it drains into the graph, and that flag survives in the
# engine pickle. A fixed user_suffix therefore makes these tests pass once and
# then fail on every later run against the same checkout -- they do NOT fail on
# CI, where weights/ is gitignored and each run starts clean, which is why this
# looked like an unreproducible "pre-existing" failure. A per-run suffix keeps
# the tests hermetic. See backlog task 9.
def _fresh_suffix(label):
    return f"{label}_{uuid.uuid4().hex[:12]}"


def test_sleep_consolidates_episodic_pairs_to_graph():
    """High-confidence rehearsed pairs graduate to graph edges during sleep."""
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix=_fresh_suffix("test_sleep_epi"))

    # Bind and rehearse pairs
    eng.episodic_binder.bind("cat", "pixel", context="my cat is pixel",
                             confidence=0.8, user_fact=True)
    eng.episodic_binder.bind("cat", "pixel", confidence=0.8, user_fact=True)
    eng.episodic_binder.bind("cat", "pixel", confidence=0.8, user_fact=True)

    # Non-user pair
    eng.episodic_binder.bind("sun", "hot", context="the sun is hot", confidence=0.7)
    eng.episodic_binder.bind("sun", "hot", confidence=0.7)
    eng.episodic_binder.bind("sun", "hot", confidence=0.7)

    # Register concept keywords so the binder can map concepts to graph nodes
    eng._concept_keywords["sun"] = [eng.graph.add_node(label="sun") or list(eng.graph.nodes.keys())[-1]]
    eng._concept_keywords["hot"] = [eng.graph.add_node(label="hot") or list(eng.graph.nodes.keys())[-1]]

    # Run sleep consolidation
    result = eng._sleep_consolidate()

    # The sun-hot pair should have graduated (user_fact pairs should NOT)
    assert result.get('episodic_pairs_graduated', 0) >= 1, \
        f"Expected >=1 graduated, got {result.get('episodic_pairs_graduated', 0)}"
    assert result.get('episodic_user_facts_withheld', 0) >= 1, \
        f"Expected >=1 user_fact withheld, got {result.get('episodic_user_facts_withheld', 0)}"

    # Verify an episodic edge exists
    sun_ids = eng._concept_keywords.get("sun", [])
    hot_ids = eng._concept_keywords.get("hot", [])
    if sun_ids and hot_ids:
        edge = eng.graph.get_edge(sun_ids[0], hot_ids[0])
        assert edge is not None, "Expected episodic edge sun->hot"
        assert edge.relation_type == "episodic", f"Expected relation_type='episodic', got '{edge.relation_type}'"

    eng.save()
    print("PASS: test_sleep_consolidates_episodic_pairs_to_graph")


def test_sleep_consolidation_survives_repeated_runs():
    """A resumed engine must still consolidate freshly-rehearsed pairs.

    Regression guard for the non-hermetic suffix bug: once a pair is marked
    `consolidated` that flag persists in the pickle, so a later run binding
    NEW pairs must still see them graduate. Fails if a stale consolidated flag
    is ever allowed to suppress the whole episodic drain.
    """
    suffix = _fresh_suffix("test_sleep_repeat")
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)
    eng.episodic_binder.bind("sun", "hot", context="the sun is hot", confidence=0.7)
    eng.episodic_binder.bind("sun", "hot", confidence=0.7)
    eng.episodic_binder.bind("sun", "hot", confidence=0.7)
    # The drain resolves each concept through _concept_keywords to find its
    # graph node; without this the pair is silently skipped, not graduated.
    eng._concept_keywords["sun"] = [eng.graph.add_node(label="sun")]
    eng._concept_keywords["hot"] = [eng.graph.add_node(label="hot")]
    first = eng._sleep_consolidate()
    assert first.get('episodic_pairs_graduated', 0) >= 1, \
        f"first cycle did not graduate: {first.get('episodic_pairs_graduated', 0)}"
    eng.save()
    eng.stop_background_learning()

    # Resume the SAME persisted self and rehearse a different pair.
    eng2 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)
    eng2.episodic_binder.bind("kiln", "cracked", context="the kiln cracked", confidence=0.7)
    eng2.episodic_binder.bind("kiln", "cracked", confidence=0.7)
    eng2.episodic_binder.bind("kiln", "cracked", confidence=0.7)
    eng2._concept_keywords["kiln"] = [eng2.graph.add_node(label="kiln")]
    eng2._concept_keywords["cracked"] = [eng2.graph.add_node(label="cracked")]
    second = eng2._sleep_consolidate()
    assert second.get('episodic_pairs_graduated', 0) >= 1, \
        ("resumed engine failed to consolidate a newly-rehearsed pair: "
         f"{second.get('episodic_pairs_graduated', 0)}")
    eng2.stop_background_learning()
    print("PASS: test_sleep_consolidation_survives_repeated_runs")


def test_sleep_no_crash_on_empty_binder():
    """Sleep consolidation with empty episodic binder doesn't crash."""
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix=_fresh_suffix("test_sleep_empty"))
    result = eng._sleep_consolidate()
    assert isinstance(result, dict)
    assert result.get('episodic_pairs_graduated', 0) == 0
    eng.save()
    print("PASS: test_sleep_no_crash_on_empty_binder")


if __name__ == "__main__":
    test_sleep_consolidates_episodic_pairs_to_graph()
    test_sleep_consolidation_survives_repeated_runs()
    test_sleep_no_crash_on_empty_binder()
    print("\nAll tests PASSED")
