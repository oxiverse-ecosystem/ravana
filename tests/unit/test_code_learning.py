"""Tests for the code-learning capability (pivot task #104).

Code understanding + generation via the SAME mechanism as natural-language
cognition: a typed concept graph learned online from snippets, generation as
graph traversal. No templates, no LLM, no retraining.
"""
import pytest

from ravana.chat.code_learning import CodeConceptGraph, CodeLearner


# ── online learning ──────────────────────────────────────────────────────────

def test_learn_snippet_grows_graph():
    g = CodeConceptGraph()
    assert g.stats()["nodes"] == 0
    g.learn_snippet("def add(a, b):\n    return a + b\n")
    s = g.stats()
    assert s["nodes"] > 0
    assert s["edges"] > 0


def test_learn_is_incremental_no_retrain():
    g = CodeConceptGraph()
    g.learn_snippet("def add(a, b):\n    return a + b\n")
    n1 = g.stats()["nodes"]
    g.learn_snippet("def mul(x, y):\n    z = x * y\n    return z\n")
    n2 = g.stats()["nodes"]
    assert n2 > n1  # second snippet extended the SAME graph, no rebuild


def test_learn_unparseable_snippet_does_not_crash():
    g = CodeConceptGraph()
    # not valid python — falls back to token co-occurrence, must not raise
    g.learn_snippet("def broken(:\n  ???\n")
    assert g.stats()["nodes"] > 0


def test_learn_extracts_code_from_prose_preamble():
    """A user sharing code with a preamble ('here's my code:') must still build
    structural edges — the prose must not break ast.parse."""
    g = CodeConceptGraph()
    g.learn_snippet("here's my code:\n\ndef add(a, b):\n    return a + b\n")
    assert "add" in g.nodes
    assert "a" in g.neighbors("add", "has_param")


def test_learn_extracts_fenced_code():
    g = CodeConceptGraph()
    g.learn_snippet("check this out:\n```python\ndef run(x):\n    return x\n```\ncool right?")
    assert g.nodes.get("run", {}).get("kind") == "function"


# ── structural edges ─────────────────────────────────────────────────────────

def test_function_has_params():
    g = CodeConceptGraph()
    g.learn_snippet("def greet(name, greeting):\n    pass\n")
    assert set(g.neighbors("greet", "has_param")) == {"name", "greeting"}


def test_class_has_method_and_inherits():
    g = CodeConceptGraph()
    g.learn_snippet("class Dog(Animal):\n    def bark(self):\n        pass\n")
    assert "Dog" in g.nodes and g.nodes["Dog"]["kind"] == "class"
    assert "bark" in g.neighbors("Dog", "has_method")
    assert "Animal" in g.neighbors("Dog", "inherits")


def test_function_scoped_calls_and_assigns():
    g = CodeConceptGraph()
    g.learn_snippet(
        "def process(data):\n"
        "    cleaned = normalize(data)\n"
        "    score = compute(cleaned)\n"
        "    return score\n"
    )
    calls = g.neighbors("process", "calls")
    assigns = g.neighbors("process", "assigns")
    assert "normalize" in calls and "compute" in calls
    assert "cleaned" in assigns and "score" in assigns


def test_calls_do_not_leak_across_functions():
    g = CodeConceptGraph()
    # helper is called only inside process, not inside other
    g.learn_snippet(
        "def helper():\n    pass\n"
        "def process():\n    helper()\n"
        "def other():\n    pass\n"
    )
    assert "helper" in g.neighbors("process", "calls")
    assert "helper" not in g.neighbors("other", "calls")


def test_import_nodes_recorded():
    g = CodeConceptGraph()
    g.learn_snippet("import os\nfrom pathlib import Path\n")
    assert "os" in g.nodes
    assert "Path" in g.nodes
    assert "pathlib" in g.nodes
    assert g.nodes["pathlib"]["kind"] == "module"


# ── generation from learned structure (no templates beyond syntax) ───────────

def test_generate_function_uses_learned_params():
    g = CodeConceptGraph()
    g.learn_snippet("def add(a, b):\n    return a + b\n")
    out = g.generate_function("add")
    assert out.startswith("def add(a, b):")


def test_generate_function_body_from_learned_calls():
    g = CodeConceptGraph()
    g.learn_snippet("def run():\n    setup()\n    main()\n    teardown()\n")
    out = g.generate_function("run")
    assert "def run():" in out
    # body is composed from the learned calls, not authored
    assert "setup()" in out and "main()" in out and "teardown()" in out


def test_generate_function_body_from_learned_assigns():
    g = CodeConceptGraph()
    g.learn_snippet("def calc(x):\n    y = x * 2\n    return y\n")
    out = g.generate_function("calc")
    assert "y = None  # learned assignment" in out


def test_generate_empty_when_nothing_learned():
    g = CodeConceptGraph()
    out = g.generate_function("brand_new")
    # honest skeleton — no invented body
    assert out == "def brand_new():\n    pass"


def test_generation_differs_with_state():
    """The HARD RULE: same name, different learned state -> different output."""
    g1 = CodeConceptGraph()
    g1.learn_snippet("def f(a):\n    return a\n")
    g2 = CodeConceptGraph()
    g2.learn_snippet("def f(a):\n    helper()\n    return a\n")
    assert g1.generate_function("f") != g2.generate_function("f")


def test_generate_class_emits_methods():
    g = CodeConceptGraph()
    g.learn_snippet("class Dog(Animal):\n    def bark(self):\n        pass\n    def sit(self):\n        pass\n")
    out = g.generate_class("Dog")
    assert out.startswith("class Dog(Animal):")
    assert "def bark(self):" in out
    assert "def sit(self):" in out


def test_generate_class_empty_body():
    g = CodeConceptGraph()
    out = g.generate_class("Empty")
    assert out == "class Empty:\n    pass"


# ── CodeLearner driver ───────────────────────────────────────────────────────

def test_learner_learn_returns_delta():
    l = CodeLearner(None)
    r = l.learn("def f(a):\n    return a\n")
    assert r["delta_nodes"] > 0
    assert r["after"]["nodes"] > r["before"]["nodes"]


def test_learner_known_functions_and_classes():
    l = CodeLearner(None)
    l.learn("def add(a, b):\n    return a + b\n")
    l.learn("class Dog(Animal):\n    def bark(self):\n        pass\n")
    assert "add" in l.known_functions()
    assert "Dog" in l.known_classes()


def test_learner_generate_dispatch():
    l = CodeLearner(None)
    l.learn("class Dog(Animal):\n    def bark(self):\n        pass\n")
    assert l.generate("Dog", kind="class").startswith("class Dog(Animal):")
    assert l.generate("Dog", kind="function").startswith("def Dog(")


def test_generation_is_deterministic():
    """Reproducibility rule: same seed/state -> same output."""
    code = "def f(a, b):\n    c = a + b\n    helper(c)\n    return c\n"
    a = CodeConceptGraph(); a.learn_snippet(code)
    b = CodeConceptGraph(); b.learn_snippet(code)
    assert a.generate_function("f") == b.generate_function("f")
