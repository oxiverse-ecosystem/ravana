"""Regression tests for the analogical reasoning fix (t_d8a44003 / RV-3).

Verifies that _consult_internal_knowledge attempts analogical reasoning
via GloVe similarity + concept-graph structure when internal knowledge
returns MISS for abstract words like "exist", instead of returning the
flat hedge "i'm not sure about X, but it reminds me of Y — they feel
related somehow" (the degenerate RV-3 failure mode).

The analogical reply must be STRUCTURALLY DERIVED:
  - GloVe cosine picks the closest known concept in the graph
  - The graph's real edge structure (direct edge between subj & known,
    or the known concept's strongest outgoing edge type) supplies the
    relational frame
  - The frame + the two real concept labels are the only inputs to the
    reply — no authored per-query prose.
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


def _result_lower(eng):
    return eng._consult_internal_knowledge('what does it mean to exist').lower()


def test_analogical_reasoning_for_exist(engine):
    """Query 'what does it mean to exist' should produce a graph-derived
    analogical reply, not the flat uncertainty hedge."""
    result = engine._consult_internal_knowledge('what does it mean to exist')
    assert result is not None, "Expected analogical reasoning to produce a non-None answer"
    rl = result.lower()
    # OLD degenerate RV-3 hedge must NOT appear.
    assert "they feel related somehow" not in rl, \
        f"Got the flat RV-3 hedge instead of analogical reasoning: {result!r}"
    # Must be a real analogical frame (structured "X is like Y — ..."),
    # not the flat "i don't have a clean definition yet" line.
    assert "is like" in rl or "sits close to" in rl or "keeps coming up nearby" in rl, \
        f"Expected a structural analogical frame, got: {result!r}"
    # Must reference a real graph concept (the GloVe-nearest known neighbor).
    assert any(w in rl for w in ['nature', 'things', 'life', 'everything']), \
        f"Expected reference to a known graph concept, got: {result!r}"


def test_analogical_no_flat_uncertainty(engine):
    """Verify the reply does NOT contain the flat uncertainty phrases
    that the RV-3 fix removed."""
    rl = _result_lower(engine)
    assert "i don't have a clean definition yet" not in rl
    assert "i don't have a clean definition" not in rl
    # Must contain structural analogical language (frame + known concept),
    # NOT the old hedge vocabulary the fix replaced.
    assert ("is like" in rl or "sits close to" in rl
            or "keeps coming up nearby" in rl), \
        f"Expected structural analogical content, got: {engine._consult_internal_knowledge('what does it mean to exist')!r}"


def test_analogical_uses_graph_structure(engine):
    """Verify the analogical reply is derived from the concept graph's
    real edge structure, not a hardcoded hedge."""
    result = engine._consult_internal_knowledge('what does it mean to exist')
    assert result is not None
    rl = result.lower()
    # The reply must name the known concept that grounded the analogy
    # (nature is the best GloVe neighbor in the graph that also carries
    # outgoing edge structure — things has no outgoing edges).
    known_concepts = ['nature', 'things', 'life', 'everything', 'we', 'know']
    assert any(w in rl for w in known_concepts), \
        f"Expected the reply to reference a known graph concept, got: {result!r}"
    # Must be a framed analogical sentence, not a bare hedge.
    assert "is like" in rl or "sits close to" in rl or "keeps coming up nearby" in rl, \
        f"Expected a framed analogical reply, got: {result!r}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
