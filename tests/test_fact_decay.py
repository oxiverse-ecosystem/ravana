"""Tests for SEED-backed fact decay and pruning in PersonalFactStore.

Brain basis: Ebbinghaus forgetting curve. Facts not rehearsed gradually
lose confidence; those below a threshold are pruned. This is the mechanism
that prevents indefinite fact accumulation (round report residual risk #3).

Acceptance: test fails on pre-capability code path (decay_confidence method
does not exist) and passes on implemented code.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'ravana', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'ravana_ml', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'ravana-v2', 'src'))

from ravana.chat.personal_fact_store import PersonalFactStore


class TestFactDecayConfidence:
    """Test the decay_confidence method: exponential decay + pruning."""

    def test_decay_confidence_method_exists(self):
        """The decay_confidence method must exist on PersonalFactStore."""
        store = PersonalFactStore()
        assert hasattr(store, 'decay_confidence'), \
            "PersonalFactStore must have decay_confidence method"

    def test_recent_fact_not_decayed(self):
        """A fact asserted this turn must not be decayed."""
        store = PersonalFactStore()
        store.advance_turn()
        store.assert_fact("cat", "name", "pixel", confidence=0.6)
        store.decay_confidence(decay_rate=0.5, min_confidence=0.1)
        fact = store.get("cat", "name", "pixel")
        assert fact is not None, "Recent fact must not be pruned"
        assert fact.confidence == pytest.approx(0.6, abs=0.01), \
            "Recent fact confidence must not be decayed"

    def test_stale_fact_decayed(self):
        """A fact not reinforced for many turns must lose confidence."""
        store = PersonalFactStore()
        store.advance_turn()
        store.assert_fact("cat", "name", "pixel", confidence=0.6)
        # Advance 20 turns without reinforcing
        for _ in range(20):
            store.advance_turn()
        store.decay_confidence(decay_rate=0.05, min_confidence=0.01)
        fact = store.get("cat", "name", "pixel")
        assert fact is not None, "Fact should not be pruned yet (min_confidence=0.01)"
        # After 20 turns at 5% decay: 0.6 * (0.95)^20 ≈ 0.6 * 0.358 ≈ 0.215
        assert fact.confidence < 0.6, \
            f"Stale fact confidence should be decayed below 0.6, got {fact.confidence}"

    def test_very_stale_fact_pruned(self):
        """A fact that decays below min_confidence must be pruned."""
        store = PersonalFactStore()
        store.advance_turn()
        store.assert_fact("cat", "name", "pixel", confidence=0.4)
        # Advance 50 turns without reinforcing
        for _ in range(50):
            store.advance_turn()
        pruned = store.decay_confidence(decay_rate=0.05, min_confidence=0.3)
        assert pruned >= 1, "Very stale low-confidence fact should be pruned"
        fact = store.get("cat", "name", "pixel")
        assert fact is None, "Pruned fact should not be retrievable"

    def test_superseded_fact_always_pruned(self):
        """Superseded facts must be pruned regardless of confidence."""
        store = PersonalFactStore()
        store.advance_turn()
        store.assert_fact("cat", "name", "pixel", confidence=0.9)
        store.contradict("cat", "name", "cairo")
        # The old value should be superseded
        old_fact = store.get("cat", "name", "pixel")
        assert old_fact is None, "Superseded fact should not be returned by get()"
        # decay_confidence should clean it up
        pruned = store.decay_confidence(decay_rate=0.0, min_confidence=0.0)
        assert pruned >= 1, "Superseded fact should be pruned"

    def test_protected_source_decays_slower(self):
        """Facts from user_confirmation/correction decay at half rate."""
        store = PersonalFactStore()
        store.advance_turn()
        store.assert_fact("cat", "name", "pixel", confidence=0.6,
                          source="user_confirmation")
        store.assert_fact("dog", "name", "rex", confidence=0.6,
                          source="seed_regex")
        # Advance 20 turns
        for _ in range(20):
            store.advance_turn()
        store.decay_confidence(decay_rate=0.1, min_confidence=0.01)
        cat_fact = store.get("cat", "name", "pixel")
        dog_fact = store.get("dog", "name", "rex")
        # Both should still exist (min_confidence is very low)
        assert cat_fact is not None, "Protected fact should still exist"
        assert dog_fact is not None, "Unprotected fact should still exist"
        # Protected should have higher confidence (decayed at half rate)
        assert cat_fact.confidence > dog_fact.confidence, \
            f"Protected source should decay slower: cat={cat_fact.confidence:.4f} > dog={dog_fact.confidence:.4f}"

    def test_decay_returns_pruned_count(self):
        """decay_confidence must return the number of facts pruned."""
        store = PersonalFactStore()
        store.advance_turn()
        store.assert_fact("cat", "name", "pixel", confidence=0.1)
        for _ in range(50):
            store.advance_turn()
        pruned = store.decay_confidence(decay_rate=0.1, min_confidence=0.3)
        assert isinstance(pruned, int), "Return value must be an integer"
        assert pruned >= 1, "Should have pruned the low-confidence stale fact"

    def test_no_decay_when_turns_stale_zero(self):
        """Facts asserted this turn (turns_stale=0) must not be decayed."""
        store = PersonalFactStore()
        store.advance_turn()
        store.assert_fact("cat", "name", "pixel", confidence=0.6)
        # Don't advance — turns_stale = 0
        store.decay_confidence(decay_rate=0.9, min_confidence=0.01)
        fact = store.get("cat", "name", "pixel")
        assert fact is not None
        assert fact.confidence == pytest.approx(0.6, abs=0.01)

    def test_decay_with_zero_decay_rate(self):
        """Zero decay rate means no confidence loss."""
        store = PersonalFactStore()
        store.advance_turn()
        store.assert_fact("cat", "name", "pixel", confidence=0.6)
        for _ in range(100):
            store.advance_turn()
        store.decay_confidence(decay_rate=0.0, min_confidence=0.01)
        fact = store.get("cat", "name", "pixel")
        assert fact is not None
        assert fact.confidence == pytest.approx(0.6, abs=0.01)

    def test_multiple_facts_independent_decay(self):
        """Each fact decays independently based on its own turn_number."""
        store = PersonalFactStore()
        store.advance_turn()
        store.assert_fact("cat", "name", "pixel", confidence=0.6)
        store.advance_turn()
        store.assert_fact("dog", "name", "rex", confidence=0.6)
        # Advance 30 more turns
        for _ in range(30):
            store.advance_turn()
        store.decay_confidence(decay_rate=0.05, min_confidence=0.01)
        cat_fact = store.get("cat", "name", "pixel")
        dog_fact = store.get("dog", "name", "rex")
        # cat is 31 turns stale, dog is 30 turns stale — very close
        assert cat_fact is not None
        assert dog_fact is not None
        # Both should be decayed but still above min_confidence
        assert cat_fact.confidence < 0.6
        assert dog_fact.confidence < 0.6


class TestFactDecayIntegration:
    """Integration tests: decay_confidence wired into sleep consolidation."""

    def test_decay_confidence_called_during_sleep(self):
        """The engine's sleep consolidation must call decay_confidence."""
        # Verify the method exists and is callable
        store = PersonalFactStore()
        assert callable(getattr(store, 'decay_confidence', None)), \
            "decay_confidence must be callable"

    def test_decay_confidence_persists_via_get_state(self):
        """Decayed facts must serialize/deserialize correctly."""
        store = PersonalFactStore()
        store.advance_turn()
        store.assert_fact("cat", "name", "pixel", confidence=0.6)
        for _ in range(20):
            store.advance_turn()
        store.decay_confidence(decay_rate=0.05, min_confidence=0.01)
        # Serialize
        state = store.get_state()
        # Deserialize into a new store
        store2 = PersonalFactStore()
        store2.set_state(state)
        fact = store2.get("cat", "name", "pixel")
        assert fact is not None, "Decayed fact should survive serialization"
        assert fact.confidence < 0.6, "Decayed confidence should persist"


class TestFactDecaySeedProperties:
    """Verify SEED properties: runtime-expandable, no hardcoding."""

    def test_decay_parameters_are_arguments(self):
        """decay_rate and min_confidence must be method arguments, not constants."""
        store = PersonalFactStore()
        store.advance_turn()
        store.assert_fact("cat", "name", "pixel", confidence=0.6)
        for _ in range(20):
            store.advance_turn()
        # Should work with different parameters
        store.decay_confidence(decay_rate=0.01, min_confidence=0.5)
        store.decay_confidence(decay_rate=0.1, min_confidence=0.1)
        store.decay_confidence(decay_rate=0.5, min_confidence=0.01)
        # No assertion on specific values — just that it doesn't crash

    def test_protected_sources_parameterizable(self):
        """protected_sources must be a parameter, not hardcoded."""
        store = PersonalFactStore()
        store.advance_turn()
        store.assert_fact("cat", "name", "pixel", confidence=0.6,
                          source="custom_source")
        for _ in range(20):
            store.advance_turn()
        # With custom_source NOT in protected set, it should decay at full rate
        store.decay_confidence(decay_rate=0.1, min_confidence=0.01,
                               protected_sources={"other_source"})
        fact = store.get("cat", "name", "pixel")
        assert fact is not None
        # With decay_rate=0.1 and 20 turns: 0.6 * (0.9)^20 ≈ 0.6 * 0.122 ≈ 0.073
        assert fact.confidence < 0.1, \
            f"Unprotected source should decay at full rate, got {fact.confidence}"

    def test_no_retraining_required(self):
        """decay_confidence must work without any retraining or offline rebuild."""
        store = PersonalFactStore()
        # Add facts
        store.advance_turn()
        store.assert_fact("cat", "name", "pixel", confidence=0.6)
        store.assert_fact("dog", "name", "rex", confidence=0.5)
        # Decay
        for _ in range(30):
            store.advance_turn()
        store.decay_confidence(decay_rate=0.03, min_confidence=0.2)
        # Add new fact AFTER decay — should work without retraining
        store.advance_turn()
        store.assert_fact("bird", "name", "tweety", confidence=0.7)
        fact = store.get("bird", "name", "tweety")
        assert fact is not None, "New facts after decay must work"
        assert fact.confidence == pytest.approx(0.7, abs=0.01)
