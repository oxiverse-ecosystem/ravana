"""Test: agent forms a real stance on a semantically related topic via user stance grounding.

Demonstrates the capability: when the user discloses a stance on a topic
(e.g. "privacy"), and then asks the agent what it thinks about a semantically
related but different topic that RESOLVES to the user's disclosed stance
via the opinion store (e.g. "data privacy" resolves to "privacy"),
the agent derives a grounded stance from the user's learned stance
instead of defaulting to "still forming a view".

This is the core capability the round's "still forming overuse" limitation
calls for: fuzzy/semantic stance recall so that topics near known concepts
get real stances, not the empty fallback.
"""
import os, sys
os.environ['RAVANA_OFFLINE'] = '1'
PROJ = 'C:/Users/Likhith/Documents/Projects/ravana'
for p in (PROJ, f'{PROJ}/ravana_ml/src', f'{PROJ}/ravana/src', f'{PROJ}/ravana-v2/src'):
    if p not in sys.path: sys.path.insert(0, p)
import pytest

@pytest.fixture(scope="module")
def engine():
    from ravana.chat.engine import CognitiveChatEngine
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="stance_cap")
    # Disclose a strong user stance on "privacy".
    eng.process_turn("i really care about privacy and digital rights")
    eng.process_turn("privacy is the most important thing to me")
    yield eng
    try:
        eng.stop_background_learning()
    except Exception:
        pass


def test_agent_forms_stance_on_semantic_variant(engine):
    """RAVANA must derive a real stance on 'data privacy' from its 'privacy' stance.

    Before the fix: _agent_stance_on falls through to 'still forming' because
    GloVe cosine between 'data privacy' and the nearest graph node ('privacy')
    is below the 0.45 strong-gate threshold or no node is close enough.
    After the fix: a stance grounded in the user's privacy stance is derived
    and recorded in _agent_own_stances.
    """
    r = engine.process_turn("what do you think about data privacy")
    assert "still forming" not in r, (
        f"Capability gap: agent fell through to 'still forming' on a "
        f"semantically grounded topic: {r!r}")
    # The agent must have recorded its own stance.
    own = getattr(engine, "_agent_own_stances", {})
    assert "data privacy" in own, (
        f"Agent did not record its own stance on 'data privacy': {own!r}")
    # The stance must be grounded — not a vague "still forming" fallback.
    _word, _conf, _reason, _turn = own["data privacy"]
    assert _conf >= 0.35, f"stance confidence too low: {_conf}"
    assert "still forming" not in _word, f"stance word is still forming: {_word!r}"


def test_agent_leans_toward_related_topic(engine):
    """A positive user stance on 'privacy' must produce a positive agent lean
    on a topic that resolves to the user's 'privacy' stance
    via the opinion store (e.g. 'data privacy' resolves to 'privacy')."""
    # Verify the user stance was mined first.
    us = engine.user_model.opinions.resolve_topic("privacy")
    assert us is not None, "precondition: user stance on 'privacy' not mined"
    st = engine.user_model.opinions.query_stance(us)
    assert st is not None and st.confidence >= 0.35, \
        f"precondition: user stance confidence too low: {st}"
    assert st.polarity > 0.0, \
        f"precondition: user stance on 'privacy' not positive: {st.polarity}"

    # Ask about a topic that resolves to the user's 'privacy' stance
    # via the opinion store's substring matching.
    r = engine.process_turn("what's your take on data privacy")
    # Must NOT be the hollow fallback.
    assert "still forming" not in r, (
        f"Related topic 'data privacy' fell through to 'still forming': {r!r}")
    # Agent must have recorded its own stance.
    own = getattr(engine, "_agent_own_stances", {})
    assert "data privacy" in own, (
        f"Agent did not record stance on 'data privacy': {own!r}")
    _word, _conf, _reason, _turn = own["data privacy"]
    assert _conf >= 0.35
    # Grounded in user's positive privacy stance -> polarity should be positive.
    assert _word not in ("am still forming a view on",), \
        f"stance word is still-forming fallback: {_word!r}"