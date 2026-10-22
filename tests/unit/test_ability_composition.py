import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (REPO, os.path.join(REPO, "ravana_ml", "src"),
           os.path.join(REPO, "ravana", "src"),
           os.path.join(REPO, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("RAVANA_OFFLINE", "1")

from ravana.agent.ability_composer import (  # noqa: E402
    register_primitive, compose_ability, run_ability, explain_ability,
    learn_from_instruction, learn_from_demonstration, has_ability,
)


def test_basic_compose():
    register_primitive("add", lambda a, b: a + b)
    register_primitive("double", lambda x: x * 2)
    compose_ability("add_then_double", ["add", "double"])
    assert run_ability("add_then_double", 3, 4) == 14
    assert explain_ability("add_then_double") == ["add", "double"]
    assert has_ability("add_then_double")


def test_engine_acquire():
    from ravana.chat.engine import CognitiveChatEngine
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True)
    eng.register_primitive("inc", lambda x: x + 1)
    eng.compose_ability("inc_twice", ["inc", "inc"])
    assert eng.run_ability("inc_twice", 5) == 7
    assert eng.explain_ability("inc_twice") == ["inc", "inc"]


def test_learn_instruction():
    register_primitive("neg", lambda x: -x)
    learn_from_instruction("compose neg_double from neg, double")
    # double already registered
    assert has_ability("neg_double")
    assert "neg" in explain_ability("neg_double")


def test_learn_demo():
    register_primitive("sq", lambda x: x * x)
    learn_from_demonstration("sq_then_inc", ["sq", "increment"])
    assert has_ability("sq_then_inc")
    assert run_ability("sq_then_inc", 4) == 17


def test_explain_unknown():
    assert explain_ability("nope_xyz") in ([], None) or explain_ability("nope_xyz") == []
