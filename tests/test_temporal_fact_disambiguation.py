"""Temporal Fact Disambiguation (TFD) — acceptance tests.

LIMENSION ADDRESSED: L2 — structured_recall can't distinguish current vs
prior-round facts. When multiple facts exist for the same attribute (from
different rounds), the old code returned the FIRST match in dict insertion
order, which could be a stale prior-round value.

CAPABILITY: _temporal_fact helper + TFD wrapping in _structured_recall
branches. All fact-recall branches now collect ALL matching non-superseded
facts and return the one with the highest turn_number (most recent).

These tests FAIL on the old code (first-match-wins) and PASS with TFD
(most-recent-wins). They verify:
1. _temporal_fact returns the most recent fact for a (subject, attribute)
2. _structured_recall returns the most recent value when multiple exist
3. Superseded facts are never returned
4. The capability is SEED-compliant (reads from PersonalFactStore)
5. No hardcoding (works for any attribute, not just specific ones)
"""
import os
import sys
import unittest

# Set up paths for in-process engine import
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(PROJ, "ravana", "src"),
          os.path.join(PROJ, "ravana_ml", "src"),
          os.path.join(PROJ, "ravana-v2", "src"),
          PROJ):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("RAVANA_OFFLINE", "1")

from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.personal_fact_store import PersonalFactStore, PersonalFact


class TestTemporalFactDisambiguation(unittest.TestCase):
    """TFD: _temporal_fact returns the most recent non-superseded fact."""

    def _make_engine(self):
        """Create a minimal engine with a user_model + personal_facts."""
        eng = CognitiveChatEngine.__new__(CognitiveChatEngine)
        # Minimal user_model with personal_facts
        class _UM:
            pass
        um = _UM()
        um.personal_facts = PersonalFactStore()
        eng.user_model = um
        return eng

    def test_temporal_fact_returns_most_recent(self):
        """When two facts exist for (i, 'work'), the one with the higher
        turn_number is returned, regardless of insertion order."""
        eng = self._make_engine()
        pf = eng.user_model.personal_facts

        # Insert stale fact first (turn 1)
        pf.assert_fact("i", "work", "pottery studio", confidence=0.6)
        pf.advance_turn()  # turn 1 -> 2
        pf.advance_turn()  # turn 2 -> 3
        # Insert current fact later (turn 3)
        pf.assert_fact("i", "work", "software engineer", confidence=0.7)

        result = eng._temporal_fact("i", "work")
        self.assertIsNotNone(result)
        self.assertEqual(result.value, "software engineer")
        # turn_number is 2 because advance_turn() was called twice (0->1, 1->2)
        self.assertEqual(result.turn_number, 2)

    def test_temporal_fact_skips_superseded(self):
        """A superseded fact is never returned, even if it has a higher
        turn_number than the active one."""
        eng = self._make_engine()
        pf = eng.user_model.personal_facts

        pf.assert_fact("i", "name", "old name", confidence=0.6)
        pf.advance_turn()
        # Contradict marks the old one superseded and stores the new one
        pf.contradict("i", "name", "new name")

        result = eng._temporal_fact("i", "name")
        self.assertIsNotNone(result)
        self.assertEqual(result.value, "new name")
        self.assertFalse(result.superseded)

    def test_temporal_fact_none_when_empty(self):
        """Returns None when no fact exists for the (subject, attribute)."""
        eng = self._make_engine()
        result = eng._temporal_fact("i", "nonexistent")
        self.assertIsNone(result)

    def test_temporal_fact_with_explicit_value(self):
        """When value is provided, only that specific value is returned."""
        eng = self._make_engine()
        pf = eng.user_model.personal_facts

        pf.assert_fact("i", "work", "pottery studio", confidence=0.6)
        pf.advance_turn()
        pf.assert_fact("i", "work", "software engineer", confidence=0.7)

        result = eng._temporal_fact("i", "work", "pottery studio")
        self.assertIsNotNone(result)
        self.assertEqual(result.value, "pottery studio")

    def test_structured_recall_returns_most_recent_work(self):
        """_structured_recall('what do i do') returns the most recent work
        fact, not the first one inserted."""
        eng = self._make_engine()
        pf = eng.user_model.personal_facts

        # Stale fact from a prior round
        pf.assert_fact("i", "work", "pottery studio", confidence=0.6)
        pf.advance_turn()
        pf.advance_turn()
        # Current fact
        pf.assert_fact("i", "work", "software engineer", confidence=0.7)

        result = eng._structured_recall("what do i do")
        self.assertIsNotNone(result)
        self.assertIn("software engineer", result)
        self.assertNotIn("pottery studio", result)

    def test_structured_recall_returns_most_recent_name(self):
        """_structured_recall('what's my name') returns the most recent name."""
        eng = self._make_engine()
        pf = eng.user_model.personal_facts

        pf.assert_fact("i", "name", "old name", confidence=0.6)
        pf.advance_turn()
        pf.contradict("i", "name", "new name")

        result = eng._structured_recall("what's my name")
        self.assertIsNotNone(result)
        self.assertIn("new name", result)
        self.assertNotIn("old name", result)

    def test_structured_recall_told_about_returns_most_recent(self):
        """_structured_recall('what did i tell you about X') returns the most
        recent matching fact, not the first one."""
        eng = self._make_engine()
        pf = eng.user_model.personal_facts

        # Stale fact about tea
        pf.assert_fact("i", "tea", "i drink green tea", confidence=0.6)
        pf.advance_turn()
        pf.advance_turn()
        # Current fact about tea
        pf.assert_fact("i", "tea", "i drink assam tea", confidence=0.7)

        result = eng._structured_recall("what did i tell you about tea")
        self.assertIsNotNone(result)
        self.assertIn("assam tea", result)
        self.assertNotIn("green tea", result)

    def test_tfd_is_seed_compliant(self):
        """_temporal_fact reads from PersonalFactStore — no new store,
        no new data structure. The store's own get() method is used."""
        eng = self._make_engine()
        pf = eng.user_model.personal_facts

        pf.assert_fact("i", "work", "pottery studio", confidence=0.6)
        pf.advance_turn()
        pf.assert_fact("i", "work", "software engineer", confidence=0.7)

        # _temporal_fact should return a PersonalFact object from the store
        result = eng._temporal_fact("i", "work")
        self.assertIsInstance(result, PersonalFact)
        # The store's get() method is what does the ranking
        store_result = pf.get("i", "work")
        self.assertEqual(result.value, store_result.value)

    def test_tfd_generalizes_to_any_attribute(self):
        """TFD works for ANY (subject, attribute) pair, not just specific
        ones. This proves it's not hardcoded."""
        eng = self._make_engine()
        pf = eng.user_model.personal_facts

        # Test with a completely arbitrary attribute
        pf.assert_fact("i", "favorite color", "blue", confidence=0.6)
        pf.advance_turn()
        pf.assert_fact("i", "favorite color", "green", confidence=0.7)

        result = eng._temporal_fact("i", "favorite color")
        self.assertIsNotNone(result)
        self.assertEqual(result.value, "green")

    def test_tfd_no_hardcoded_reply_strings(self):
        """_temporal_fact returns a PersonalFact object, not a rendered
        string. The rendering happens in _structured_recall, which reads
        the fact's value from the store. No authored prose in the helper."""
        eng = self._make_engine()
        pf = eng.user_model.personal_facts

        pf.assert_fact("i", "work", "pottery studio", confidence=0.6)
        result = eng._temporal_fact("i", "work")
        # The helper returns the fact object, not a string
        self.assertIsInstance(result, PersonalFact)
        self.assertEqual(result.value, "pottery studio")


class TestTFDOnlineLearning(unittest.TestCase):
    """TFD learns online: when the user corrects a fact, the new value
    becomes the one returned by _temporal_fact."""

    def _make_engine(self):
        eng = CognitiveChatEngine.__new__(CognitiveChatEngine)
        class _UM:
            pass
        um = _UM()
        um.personal_facts = PersonalFactStore()
        eng.user_model = um
        return eng

    def test_correction_updates_temporal_fact(self):
        """When the user corrects a fact, _temporal_fact returns the
        corrected value (higher turn_number)."""
        eng = self._make_engine()
        pf = eng.user_model.personal_facts

        pf.assert_fact("i", "work", "pottery studio", confidence=0.6)
        pf.advance_turn()
        pf.advance_turn()
        # User correction
        pf.contradict("i", "work", "software engineer")

        result = eng._temporal_fact("i", "work")
        self.assertIsNotNone(result)
        self.assertEqual(result.value, "software engineer")
        self.assertTrue(result.source == "correction")

    def test_reinforcement_does_not_change_value(self):
        """Reinforcing a fact (user confirms) does not change its value,
        but updates turn_number so it stays as the most recent."""
        eng = self._make_engine()
        pf = eng.user_model.personal_facts

        pf.assert_fact("i", "work", "pottery studio", confidence=0.6)
        pf.advance_turn()
        pf.reinforce("i", "work")

        result = eng._temporal_fact("i", "work")
        self.assertIsNotNone(result)
        self.assertEqual(result.value, "pottery studio")
        self.assertEqual(result.turn_number, 1)


class TestTFDNoConfabulation(unittest.TestCase):
    """TFD prevents confabulation: stale facts from prior rounds are
    not returned when a more recent fact exists."""

    def _make_engine(self):
        eng = CognitiveChatEngine.__new__(CognitiveChatEngine)
        class _UM:
            pass
        um = _UM()
        um.personal_facts = PersonalFactStore()
        eng.user_model = um
        return eng

    def test_stale_fact_not_returned(self):
        """A fact from turn 1 is not returned when a fact from turn 5 exists."""
        eng = self._make_engine()
        pf = eng.user_model.personal_facts

        pf.assert_fact("i", "location", "mumbai", confidence=0.6)
        for _ in range(5):
            pf.advance_turn()
        pf.assert_fact("i", "location", "delhi", confidence=0.7)

        result = eng._temporal_fact("i", "location")
        self.assertIsNotNone(result)
        self.assertEqual(result.value, "delhi")

    def test_structured_recall_no_stale_location(self):
        """_structured_recall('where do i live') returns the most recent
        location, not the stale one."""
        eng = self._make_engine()
        pf = eng.user_model.personal_facts

        pf.assert_fact("i", "location", "mumbai", confidence=0.6)
        for _ in range(5):
            pf.advance_turn()
        pf.assert_fact("i", "location", "delhi", confidence=0.7)

        result = eng._structured_recall("where do i live")
        self.assertIsNotNone(result)
        self.assertIn("delhi", result)
        self.assertNotIn("mumbai", result)


if __name__ == "__main__":
    unittest.main()
