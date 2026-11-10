"""Regression tests for the analogical reasoning fix (t_d8a44003 / RV-3).

Verifies that analogical reasoning is attempted via GloVe similarity +
concept-graph structure when internal knowledge returns MISS for abstract
words like "exist", instead of returning the flat hedge "i'm not sure
about X, but it reminds me of Y — they feel related somehow" (the
degenerate RV-3 failure mode).

ROUND 2026-11-10 — the contract these tests exercise MOVED, twice.

1. REACHABILITY. The analogy fallback no longer answers from inside
   `_consult_internal_knowledge`; that method now returns None on a MISS so
   the caller falls through to generation. Measured over a 46-query probe
   set, answering there made this single path the largest generation bypass
   (13 turns, 12 of them analogy templates). The analogy itself is unchanged
   and still available as a last resort via `_try_analogical_reasoning`'s
   other call sites.

2. NEIGHBOUR QUALITY. The gate now declines to analogise when no good
   neighbour exists. Measured GloVe neighbours of "exist":

       0.639 there   0.635 although   0.618 many
       0.611 rather  0.610 must       0.609 example

   The top five are function words; the best content word ("example") is not
   an analogy for existence. So for this query the correct reply is honest
   uncertainty, and "exist is like example" would be a regression, not a
   success. These tests previously asserted the opposite.

So the contract now is:
  - `_consult_internal_knowledge` does NOT preempt generation on a MISS.
  - The analogy path, WHEN IT FIRES, produces a structurally derived frame
    and never builds one on a function word.
  - The flat RV-3 hedge never comes back.


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
    """The full turn, not the internal consult.

    The consult no longer answers on a MISS (by design -- see module
    docstring), so asserting on it would test the absence of an answer
    rather than the quality of one.
    """
    return (eng.process_turn('what does it mean to exist') or '').strip().lower()


def test_analogical_reasoning_for_exist(engine):
    """Query 'what does it mean to exist' must NOT produce the flat RV-3 hedge.

    It may produce a structurally derived analogy OR honest uncertainty --
    which one depends on whether a qualifying neighbour exists, and for
    "exist" none does (see module docstring). What must never come back is
    the degenerate hedge.
    """
    result = (engine.process_turn('what does it mean to exist') or '').strip()
    assert result, "Expected a non-empty reply"
    rl = result.lower()
    assert "they feel related somehow" not in rl, \
        f"Got the flat RV-3 hedge: {result!r}"
    assert "i don't have a clean definition yet" not in rl, \
        f"Got the pre-RV-3 flat hedge: {result!r}"


def test_analogical_no_flat_uncertainty(engine):
    """The reply must not carry the flat uncertainty phrases RV-3 removed."""
    rl = _result_lower(engine)
    assert "i don't have a clean definition yet" not in rl
    assert "i don't have a clean definition" not in rl
    assert "they feel related somehow" not in rl


def test_analogical_uses_graph_structure(engine):
    """When an analogy IS produced it must be graph-derived, not a hedge.

    Asserted on the analogy path directly rather than on a specific query,
    because whether an analogy fires now depends on neighbour quality --
    which is the point of the gate.
    """
    result = engine._try_analogical_reasoning('exist', 'what does it mean to exist')
    if result is None:
        # No qualifying neighbour: the honest and correct outcome.
        return
    rl = result.lower()
    assert "they feel related somehow" not in rl, \
        f"Regressed to the RV-3 hedge: {result!r}"
    assert ("is like" in rl or "sits close to" in rl
            or "keeps coming up nearby" in rl), \
        f"Expected a framed analogical reply, got: {result!r}"


def test_consult_does_not_preempt_generation(engine):
    """A MISS must NOT answer from the consult.

    Answering there made this path the largest generation bypass measured
    (13 of 46 probe turns, 12 of them analogy templates). The consult now
    returns None on a MISS so the caller falls through to generation; the
    analogy remains available as a last resort elsewhere.
    """
    assert engine._consult_internal_knowledge('what does it mean to exist') is None


def test_analogy_refuses_function_word_neighbour(engine):
    """A function word is not a concept and must not ground an analogy.

    Measured failure: "what does it mean to exist" -> "exist is like there
    -- one tends to follow the other in time". The nearest GloVe neighbour
    of "exist" is the function word "there" (cosine 0.639). The
    neighbour-quality gate (cosine >= 0.45, content words only) must reject
    it, so the reply is either a real analogy on a content word or no
    analogy at all.
    """
    result = (engine.process_turn('what does it mean to exist') or '').strip().lower()
    if 'is like' in result or 'sits close to' in result:
        for _fw in (' there ', ' it ', ' they ', ' them ', ' this ',
                    ' that ', ' rather ', ' although ', ' must '):
            assert _fw not in result, \
                f"Analogy built on a function word: {result!r}"
    # else: no analogy, which is the honest and acceptable outcome.


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
