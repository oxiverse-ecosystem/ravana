"""Test sleep-phase episodic pair replay — Phase 3b4 consolidation."""
import sys, os
os.environ["RAVANA_OFFLINE"] = "1"

PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src", f"{PROJ}\\ravana-v2\\src"):
    sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine
from ravana.core.episodic_binder import EpisodicBinder, EpisodicBinderConfig


def test_sleep_consolidates_episodic_pairs_to_graph():
    """High-confidence rehearsed pairs graduate to graph edges during sleep."""
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="test_sleep_epi")

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


def test_sleep_no_crash_on_empty_binder():
    """Sleep consolidation with empty episodic binder doesn't crash."""
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="test_sleep_empty")
    result = eng._sleep_consolidate()
    assert isinstance(result, dict)
    assert result.get('episodic_pairs_graduated', 0) == 0
    eng.save()
    print("PASS: test_sleep_no_crash_on_empty_binder")


if __name__ == "__main__":
    test_sleep_consolidates_episodic_pairs_to_graph()
    test_sleep_no_crash_on_empty_binder()
    print("\nAll tests PASSED")
