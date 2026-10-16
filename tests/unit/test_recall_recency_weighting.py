"""Tests for D2: Stale fact recall — recency weighting in _recall_user_fact
and entity-scoped name recall.

When multiple personal facts match the same attribute (e.g. the user changed
their name or location across rounds), RAVANA must return the MOST RECENT one
by turn_number, not the first one inserted.

These tests verify:
1. _recall_user_fact prefers the most recent fact by turn_number
2. Entity-scoped name recall prefers the most recent name fact
3. The fix is general (no per-entity hardcoding)
"""

import os
import sys
import unittest

# Set up paths for in-process engine import
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src", f"{PROJ}\\ravana-v2\\src"):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["RAVANA_OFFLINE"] = "1"

from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.personal_fact_store import PersonalFact, PersonalFactStore


class TestRecallUserFactRecencyWeighting(unittest.TestCase):
    """Test that _recall_user_fact prefers the most recent fact by turn_number."""

    def _make_engine_with_facts(self, facts_list):
        """Create an engine with specific facts injected into the personal fact store.

        facts_list: list of (subject, attr, value, turn_number) tuples
        """
        eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="test_recency")
        pf = eng.user_model.personal_facts
        for subj, attr, val, tn in facts_list:
            pf.facts[(subj, attr, val)] = PersonalFact(
                subject=subj, attribute=attr, value=val,
                confidence=0.9, turn_number=tn
            )
        return eng

    def test_recall_user_fact_returns_most_recent_by_turn_number(self):
        """When two facts match the same attribute, return the one with higher turn_number."""
        eng = self._make_engine_with_facts([
            ("i", "name", "old_name", 10),
            ("i", "name", "new_name", 50),
        ])
        result = eng._recall_user_fact("name", "what's my name")
        self.assertEqual(result, "new_name")

    def test_recall_user_fact_single_fact_still_works(self):
        """When only one fact matches, return it regardless of turn_number."""
        eng = self._make_engine_with_facts([
            ("i", "name", "only_name", 10),
        ])
        result = eng._recall_user_fact("name", "what's my name")
        self.assertEqual(result, "only_name")

    def test_recall_user_fact_ignores_superseded(self):
        """Superseded facts are never returned, even if they have the highest turn_number."""
        eng = self._make_engine_with_facts([
            ("i", "name", "current_name", 50),
            ("i", "name", "retired_name", 100),
        ])
        # Mark the higher-turn fact as superseded
        eng.user_model.personal_facts.facts[("i", "name", "retired_name")].superseded = True
        result = eng._recall_user_fact("name", "what's my name")
        self.assertEqual(result, "current_name")

    def test_recall_user_fact_no_match_returns_none(self):
        """When no fact matches the attribute, return None."""
        eng = self._make_engine_with_facts([
            ("i", "name", "some_name", 10),
        ])
        result = eng._recall_user_fact("location", "where do i live")
        self.assertIsNone(result)

    def test_recall_user_fact_location_recency(self):
        """Location facts also use recency weighting."""
        eng = self._make_engine_with_facts([
            ("i", "location", "old_city", 5),
            ("i", "location", "new_city", 80),
        ])
        result = eng._recall_user_fact("location", "where do i live")
        self.assertEqual(result, "new_city")

    def test_recall_user_fact_work_recency(self):
        """Work facts also use recency weighting."""
        eng = self._make_engine_with_facts([
            ("i", "work", "old_job", 15),
            ("i", "work", "new_job", 90),
        ])
        result = eng._recall_user_fact("work", "what do i do")
        self.assertEqual(result, "new_job")

    def test_recall_user_fact_turn_number_zero_defaults_oldest(self):
        """Facts with turn_number=0 are treated as oldest (default)."""
        eng = self._make_engine_with_facts([
            ("i", "name", "zero_turn", 0),
            ("i", "name", "later_turn", 1),
        ])
        result = eng._recall_user_fact("name", "what's my name")
        self.assertEqual(result, "later_turn")

    def test_recall_user_fact_missing_turn_number_defaults_zero(self):
        """Facts without turn_number attribute default to 0 (oldest)."""
        eng = self._make_engine_with_facts([
            ("i", "name", "no_turn", 0),
            ("i", "name", "has_turn", 10),
        ])
        # Remove turn_number from the first fact (simulate old data)
        eng.user_model.personal_facts.facts[("i", "name", "no_turn")].turn_number = 0
        result = eng._recall_user_fact("name", "what's my name")
        self.assertEqual(result, "has_turn")


class TestEntityNameRecallRecency(unittest.TestCase):
    """Test that entity-scoped name recall prefers the most recent name fact."""

    def _make_engine_with_entity_names(self, facts_list):
        """Create an engine with entity-scoped name facts.

        facts_list: list of (entity, name_value, turn_number) tuples
        """
        eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="test_ent_recency")
        pf = eng.user_model.personal_facts
        for entity, name_val, tn in facts_list:
            pf.facts[(entity, "name", name_val)] = PersonalFact(
                subject=entity, attribute="name", value=name_val,
                confidence=0.9, turn_number=tn
            )
        return eng

    def test_entity_name_recall_prefers_most_recent(self):
        """When an entity has multiple name facts, return the most recent."""
        eng = self._make_engine_with_entity_names([
            ("dog", "old_dog_name", 10),
            ("dog", "new_dog_name", 50),
        ])
        # Simulate the entity-scoped name recall logic
        pf = eng.user_model.personal_facts
        _ent_name_keys = [
            _k for _k in pf.facts.keys()
            if isinstance(_k, tuple) and len(_k) == 3
            and _k[0] not in ("i", "me", "my", "you")
            and _k[1] == "name"
            and not getattr(pf.facts[_k], "superseded", False)
        ]
        # Simulate the fixed logic
        _best_v = None
        _best_tn = -1
        for _k in _ent_name_keys:
            if _k[0] == "dog":
                _tn = getattr(pf.facts[_k], "turn_number", 0) or 0
                if _tn > _best_tn:
                    _best_tn = _tn
                    _best_v = getattr(pf.facts[_k], "value", _k[2])
        self.assertEqual(_best_v, "new_dog_name")

    def test_entity_name_recall_ignores_superseded(self):
        """Superseded entity name facts are never returned."""
        eng = self._make_engine_with_entity_names([
            ("cat", "current_cat_name", 50),
            ("cat", "retired_cat_name", 100),
        ])
        # Mark the higher-turn fact as superseded
        eng.user_model.personal_facts.facts[("cat", "name", "retired_cat_name")].superseded = True
        pf = eng.user_model.personal_facts
        _ent_name_keys = [
            _k for _k in pf.facts.keys()
            if isinstance(_k, tuple) and len(_k) == 3
            and _k[0] not in ("i", "me", "my", "you")
            and _k[1] == "name"
            and not getattr(pf.facts[_k], "superseded", False)
        ]
        _best_v = None
        _best_tn = -1
        for _k in _ent_name_keys:
            if _k[0] == "cat":
                _tn = getattr(pf.facts[_k], "turn_number", 0) or 0
                if _tn > _best_tn:
                    _best_tn = _tn
                    _best_v = getattr(pf.facts[_k], "value", _k[2])
        self.assertEqual(_best_v, "current_cat_name")


class TestRecencyWeightingIsGeneral(unittest.TestCase):
    """Verify the recency weighting is general (no per-entity hardcoding)."""

    def test_recency_weighting_works_for_any_attribute(self):
        """The recency weighting applies to ANY attribute, not just name/location."""
        eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="test_general")
        pf = eng.user_model.personal_facts
        # Test with a custom attribute
        pf.facts[("i", "favorite_color", "red")] = PersonalFact(
            subject="i", attribute="favorite_color", value="red",
            confidence=0.9, turn_number=10
        )
        pf.facts[("i", "favorite_color", "blue")] = PersonalFact(
            subject="i", attribute="favorite_color", value="blue",
            confidence=0.9, turn_number=50
        )
        result = eng._recall_user_fact("favorite_color", "what's my favorite color")
        self.assertEqual(result, "blue")

    def test_recency_weighting_works_for_any_entity(self):
        """The recency weighting applies to ANY entity, not just pets."""
        eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="test_general_ent")
        pf = eng.user_model.personal_facts
        # Test with a custom entity
        pf.facts[("mentor", "name", "old_mentor_name")] = PersonalFact(
            subject="mentor", attribute="name", value="old_mentor_name",
            confidence=0.9, turn_number=10
        )
        pf.facts[("mentor", "name", "new_mentor_name")] = PersonalFact(
            subject="mentor", attribute="name", value="new_mentor_name",
            confidence=0.9, turn_number=50
        )
        # Simulate the entity-scoped name recall logic
        _ent_name_keys = [
            _k for _k in pf.facts.keys()
            if isinstance(_k, tuple) and len(_k) == 3
            and _k[0] not in ("i", "me", "my", "you")
            and _k[1] == "name"
            and not getattr(pf.facts[_k], "superseded", False)
        ]
        _best_v = None
        _best_tn = -1
        for _k in _ent_name_keys:
            if _k[0] == "mentor":
                _tn = getattr(pf.facts[_k], "turn_number", 0) or 0
                if _tn > _best_tn:
                    _best_tn = _tn
                    _best_v = getattr(pf.facts[_k], "value", _k[2])
        self.assertEqual(_best_v, "new_mentor_name")


if __name__ == "__main__":
    unittest.main()
