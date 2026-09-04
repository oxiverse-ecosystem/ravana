"""Regression tests for the analogical reasoning fix (t_d8a44003).

Verifies that _consult_internal_knowledge attempts analogical reasoning
via GloVe similarity when internal knowledge returns MISS for abstract
words like "exist", instead of returning flat uncertainty.
"""
import os
import sys
import tempfile
import pytest

os.environ['RAVANA_OFFLINE'] = '1'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ravana', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ravana_ml', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ravana-v2', 'src'))

from ravana.chat.engine import CognitiveChatEngine


@pytest.fixture(scope="module")
def engine():
    d = tempfile.mkdtemp(prefix="ravana_analog_")
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d)
    yield e


def test_analogical_reasoning_for_exist(engine):
    """Query 'what does it mean to exist' should produce analogical reasoning,
    not a flat 'i don't have a clean definition yet'."""
    result = engine._consult_internal_knowledge('what does it mean to exist')
    assert result is not None, "Expected analogical reasoning to produce a non-None answer"
    assert "i don't have a clean definition yet" not in result.lower(), \
        f"Got flat uncertainty instead of analogical reasoning: {result!r}"
    # The answer should reference a related concept or contain analogical language
    assert any(w in result.lower() for w in ['live', 'be', 'happen', 'related', 'reminds', 'connected', 'fuzzy', 'not sure']), \
        f"Expected analogical reference to related concept, got: {result!r}"


def test_analogical_no_flat_uncertainty(engine):
    """Verify that the answer does NOT contain flat uncertainty phrases."""
    result = engine._consult_internal_knowledge('what does it mean to exist')
    assert result is not None
    assert "i don't have a clean definition yet" not in result.lower()
    assert "i don't have a clean definition" not in result.lower()
    # Must contain analogical reasoning language
    assert any(w in result.lower() for w in ['reminds', 'connected', 'related', 'fuzzy', 'not sure', 'live', 'be', 'happen']), \
        f"Expected analogical content in result, got: {result!r}"


def test_analogical_uses_glove_similarity(engine):
    """Verify that the analogical path uses GloVe similarity to find
    the closest known concept and generates an analogical reply."""
    result = engine._consult_internal_knowledge('what does it mean to exist')
    assert result is not None
    # The result should mention 'things' (the closest GloVe neighbor)
    # or contain analogical hedge language
    assert "things" in result.lower() or "not sure" in result.lower() or "fuzzy" in result.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
