"""Tests for the pivot Message type (docs/LANGUAGE_PRODUCTION_PIVOT.md M1/M5)."""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_PROJ,
           os.path.join(_PROJ, "ravana_ml", "src"),
           os.path.join(_PROJ, "ravana", "src"),
           os.path.join(_PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.chat.message import Message, FeatureBundle


def test_message_is_grounded_requires_graph_node():
    m = Message()
    assert not m.is_grounded()
    m2 = Message(agent_node="concept:dog")
    assert m2.is_grounded()


def test_active_set_capped_at_four():
    m = Message(agent_node="a", patient_node="b",
                modifiers=[("time", "c"), ("place", "d"), ("mood", "e")])
    assert len(m.active_nodes()) == 4


def test_feature_bundle_defaults():
    fb = FeatureBundle()
    assert fb.number == "sg"
    assert fb.tense == "present"


def test_message_importable_from_engine_package():
    import ravana.chat.message as mod
    assert hasattr(mod, "Message")
    assert hasattr(mod, "FeatureBundle")
