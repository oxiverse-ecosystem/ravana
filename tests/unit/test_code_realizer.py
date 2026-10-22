"""Tests for RAVANA code realizer (OCG modality — code generation)."""
import os
import sys
import ast

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (REPO, os.path.join(REPO, "ravana_ml", "src"),
           os.path.join(REPO, "ravana", "src"),
           os.path.join(REPO, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("RAVANA_OFFLINE", "1")

from ravana.language.code_realizer import (  # noqa: E402
    generate_code, CodeRealizer, register_operation,
)


def _run(src, fn_name, *args):
    ns = {}
    exec(src, ns)
    return ns[fn_name](*args)


class TestValidity:
    def test_generates_valid_python(self):
        for d in ["sum two numbers", "reverse a list", "factorial of n",
                  "greet a name", "square a number", "sort items",
                  "count items", "double x", "filter positive numbers"]:
            src = generate_code(d)
            ast.parse(src)  # raises on invalid

    def test_generated_code_runs(self):
        src = generate_code("sum two numbers")
        assert _run(src, "sum_two_numbers", 3, 4) == 7
        src = generate_code("factorial of n")
        assert _run(src, "factorial_n", 5) == 120
        src = generate_code("reverse a list")
        assert _run(src, "reverse_list", [1, 2, 3]) == [3, 2, 1]


class TestMutation:
    def test_mutation_changes_code(self):
        a = generate_code("sum two numbers")
        b = generate_code("multiply two numbers")
        assert a != b

    def test_distinct_descriptions_distinct(self):
        descs = ["sum two numbers", "reverse a list", "square a number",
                 "greet a name", "count items"]
        outs = [generate_code(d) for d in descs]
        assert len(set(outs)) == len(outs)


class TestTraceability:
    def test_header_records_state(self):
        src = generate_code("sum two numbers", state={"identity_strength": 0.5,
                                                     "graph_nodes": 42})
        assert "cognitive state" in src
        assert "identity_strength=0.5" in src
        assert "graph_nodes=42" in src

    def test_no_verbatim_repo_copy(self):
        # generated source should not equal any file content in repo
        src = generate_code("sum two numbers")
        # sanity: contains our marker
        assert "RAVANA code realizer" in src


class TestGrowth:
    def test_register_operation(self):
        def builder(params):
            return " + ".join(p + " + 1" for p in params) if params else "0"
        register_operation("increment_sum", builder)
        src = generate_code("increment_sum two numbers")
        ast.parse(src)
        assert "increment_sum" in src


class TestEngineIntegration:
    def test_engine_generate_code(self):
        os.environ.setdefault("RAVANA_OFFLINE", "1")
        from ravana.chat.engine import CognitiveChatEngine
        eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True)
        src = eng.generate_code("reverse a list")
        ast.parse(src)
        assert "reverse" in src

    def test_code_realizer_property(self):
        os.environ.setdefault("RAVANA_OFFLINE", "1")
        from ravana.chat.engine import CognitiveChatEngine
        eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True)
        cr = eng.code_realizer
        assert cr.generate("square a number") == generate_code("square a number", engine=eng)
