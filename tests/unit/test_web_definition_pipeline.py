"""Tests for definition extraction pipeline in WebLearningMixin."""

import pytest
import re
from unittest.mock import MagicMock
from ravana.chat.web_learning import WebLearningMixin


class DummyEngine(WebLearningMixin):
    """Minimal mock engine inheriting from WebLearningMixin."""
    def __init__(self):
        self._definitions = {}
        self._concept_keywords = {}
        self._trace_enabled = True
        # Mock other required class attributes if needed
        self.turn_count = 1
        self._network_available = True


def test_heuristic_multi_word_subject():
    """Verify heuristic extracts definition for multi-word subjects (dark energy)."""
    engine = DummyEngine()
    text = "Dark energy is a hypothetical form of energy that exerts a negative, repulsive gravity."
    engine._extract_definitions(text, "dark energy")
    
    assert "dark energy" in engine._definitions
    assert engine._definitions["dark energy"] == "a hypothetical form of energy that exerts a negative, repulsive gravity"


def test_heuristic_alternative_copula():
    """Verify heuristic extracts definition using alternative defining verbs (refers to)."""
    engine = DummyEngine()
    text = "Quantum entanglement refers to a physical phenomenon that occurs when a pair or group of particles is generated."
    engine._extract_definitions(text, "quantum entanglement")
    
    assert "quantum entanglement" in engine._definitions
    assert engine._definitions["quantum entanglement"] == "a physical phenomenon that occurs when a pair or group of particles is generated"


def test_regex_also_known_as():
    """Verify regex patterns match 'also known as' definitions."""
    engine = DummyEngine()
    # Concept 'blockchain' is in _concept_keywords
    engine._concept_keywords["blockchain"] = [1]
    text = "Blockchain, also known as distributed ledger technology, is the technology behind bitcoin."
    engine._extract_definitions(text, "blockchain")
    
    assert "blockchain" in engine._definitions
    assert engine._definitions["blockchain"] == "distributed ledger technology"


def test_heuristic_pronoun_fallback():
    """Verify heuristic fallback for pronoun references ('It is a...')."""
    engine = DummyEngine()
    text = "It is a decentralized database."
    engine._extract_definitions(text, "blockchain")
    
    assert "blockchain" in engine._definitions
    assert engine._definitions["blockchain"] == "a decentralized database"


def test_regex_called_pattern():
    """Verify regex pattern matches 'Y, called X' definitions."""
    engine = DummyEngine()
    engine._concept_keywords["dark energy"] = [1]
    text = "A mysterious force, called dark energy, is driving the accelerated expansion of the universe."
    engine._extract_definitions(text, "dark energy")
    
    assert "dark energy" in engine._definitions
    assert engine._definitions["dark energy"] == "a mysterious force"
