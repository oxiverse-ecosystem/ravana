"""Tests for EpisodicBinder — CA3-style fast concept-pair binding.

Validates the core claim: can RAVANA recall the whole from a fragment
after ONE showing (single-exposure binding + pattern completion)?
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ravana', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ravana_ml', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ravana-v2', 'src'))

from ravana.core.episodic_binder import EpisodicBinder, EpisodicBinderConfig, EpisodicPair


def test_one_shot_binding():
    """Bind two concepts in one exposure, retrieve from a fragment."""
    binder = EpisodicBinder(EpisodicBinderConfig(max_pairs=50))

    # One exposure: "my cat's name is Pixel"
    pair = binder.bind("cat", "Pixel", context="my cat's name is Pixel",
                       confidence=0.9, user_fact=True)
    assert pair is not None
    assert pair.concept_a in ("cat", "pixel")
    assert pair.concept_b in ("cat", "pixel")

    # Pattern completion: cue "cat" → retrieve "Pixel"
    results = binder.retrieve("cat")
    assert len(results) >= 1
    partners = [r[0] for r in results]
    assert "pixel" in partners
    print("PASS: test_one_shot_binding — one exposure, fragment recall works")


def test_bidirectional_retrieval():
    """Retrieval works from either side of the pair."""
    binder = EpisodicBinder(EpisodicBinderConfig(max_pairs=50))
    binder.bind("luna", "moon", context="Luna is the moon", confidence=0.85)

    # Forward: luna → moon
    r1 = binder.retrieve("luna")
    assert any(p[0] == "moon" for p in r1)

    # Backward: moon → luna
    r2 = binder.retrieve("moon")
    assert any(p[0] == "luna" for p in r2)
    print("PASS: test_bidirectional_retrieval")


def test_rehearsal_strengthens():
    """Re-binding the same pair increases confidence."""
    binder = EpisodicBinder(EpisodicBinderConfig(max_pairs=50))
    p1 = binder.bind("dog", "Rex")
    assert p1 is not None
    conf_before = p1.confidence

    p2 = binder.bind("dog", "Rex")
    assert p2 is not None
    assert p2.confidence > conf_before
    assert p2.rehearsal_count >= 2
    print("PASS: test_rehearsal_strengthens")


def test_retrieval_practice():
    """Retrieving a pair strengthens it (testing effect)."""
    binder = EpisodicBinder(EpisodicBinderConfig(max_pairs=50))
    binder.bind("river", "Nile", confidence=0.7)

    results = binder.retrieve("river")
    pair = results[0][1]
    conf_after = pair.confidence
    # Retrieval practice adds 0.05
    assert conf_after > 0.7
    print("PASS: test_retrieval_practice")


def test_decay():
    """Unhearsed pairs decay over many turns."""
    binder = EpisodicBinder(EpisodicBinderConfig(max_pairs=50, decay_turns=5))
    binder.bind("temp", "fact", confidence=0.3, user_fact=False)

    # Advance many turns without retrieval
    for _ in range(20):
        binder.advance_turn()

    results = binder.retrieve("temp")
    # Either decayed below threshold or removed entirely
    assert len(results) == 0
    print("PASS: test_decay")


def test_consolidation_candidates():
    """High-rehearsal pairs become consolidation candidates."""
    binder = EpisodicBinder(EpisodicBinderConfig(max_pairs=50, min_rehearsal_for_consolidate=2))
    binder.bind("Paris", "France", confidence=0.8, user_fact=True)

    # Rehearse twice
    binder.bind("Paris", "France", confidence=0.8)
    binder.bind("Paris", "France", confidence=0.8)

    candidates = binder.get_consolidation_candidates()
    assert len(candidates) >= 1
    assert candidates[0].concept_a in ("paris", "france")
    print("PASS: test_consolidation_candidates")


def test_user_fact_boundary():
    """User facts are flagged and should NOT consolidate to world graph."""
    binder = EpisodicBinder(EpisodicBinderConfig(max_pairs=50))
    binder.bind("my", "secret", context="my secret is hidden",
                confidence=0.9, user_fact=True)

    # User facts should still be retrievable
    results = binder.retrieve("my")
    assert len(results) >= 1

    # But flagged as user_fact for source monitoring
    assert results[0][1].user_fact is True
    print("PASS: test_user_fact_boundary")


def test_many_pairs_no_cross_contamination():
    """Multiple distinct pairs don't cross-contaminate on retrieval."""
    binder = EpisodicBinder(EpisodicBinderConfig(max_pairs=100))
    binder.bind("cat", "Pixel")
    binder.bind("dog", "Rex")
    binder.bind("bird", "Tweety")

    r = binder.retrieve("cat")
    partners = [p[0] for p in r]
    assert "pixel" in partners
    assert "rex" not in partners
    assert "tweety" not in partners
    print("PASS: test_many_pairs_no_cross_contamination")


if __name__ == "__main__":
    test_one_shot_binding()
    test_bidirectional_retrieval()
    test_rehearsal_strengthens()
    test_retrieval_practice()
    test_decay()
    test_consolidation_candidates()
    test_user_fact_boundary()
    test_many_pairs_no_cross_contamination()
    print("\nAll EpisodicBinder tests PASSED.")
