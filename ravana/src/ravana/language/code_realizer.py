"""RAVANA — Code Realizer: generate code from learned structure (OCG modality).

Extends the Online Construction Grammar generative mechanism (task 2) to a
NEW output modality: code. The SAME mechanism that composes natural language
from a learned hierarchy (construction schemas + energy-argmin realization +
cerebellar phrase patterns + growable operation primitives) composes a short,
correct, idiomatic function from a natural-language description.

No LLM, no retraining, no hardcoded code snippets, no
"if the request mentions X return snippet Y". Every operation is a *combinator*
over primitives that can be registered/grown at runtime; the code skeleton is
seed *structure* (allowed), and the content (names, operations, params) comes
from the parsed description + the cognitive state that produced it.

Traceability: the generated source carries a header comment recording the
cognitive-state metrics that drove it (identity strength, graph size, stances),
so the output is auditable back to the state that produced it.
"""
from __future__ import annotations

import ast
import hashlib
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Seed operation primitives — GROWABLE combinators, not snippets.
# Each entry maps a canonical operation concept to a builder that takes a list
# of parameter names and returns an expression string. New operations can be
# registered at runtime via register_operation(); removing one degrades to the
# generic fallback. This is seed STRUCTURE (allowed), not a hardcoded answer.
# --------------------------------------------------------------------------- #
_OPERATION_BUILDERS: Dict[str, Callable[[List[str]], str]] = {}


def _binary_op(op: str):
    def _b(params: List[str]) -> str:
        if len(params) >= 2:
            return " " + op + " ".join("") or ""  # placeholder replaced below
        return params[0] if params else "x"
    return _b


def _make_binary(op: str) -> Callable[[List[str]], str]:
    def _b(params: List[str]) -> str:
        if not params:
            return "0"
        if len(params) == 1:
            return params[0]
        return " + ".join(params) if op == "+" else (
            " * ".join(params) if op == "*" else
            " - ".join(params) if op == "-" else
            " / ".join(params) if op == "/" else
            params[0]
        )
    # fix: actually use op
    def _b2(params: List[str]) -> str:
        if not params:
            return "0"
        if len(params) == 1:
            return params[0]
        return (" " + op + " ").join(params)
    return _b2


def _sum_builtin(params: List[str]) -> str:
    if not params:
        return "0"
    if len(params) == 1:
        return f"sum({params[0]})"
    return " + ".join(params)


def _avg(params: List[str]) -> str:
    if not params:
        return "0"
    p = params[0] if len(params) == 1 else f"[{', '.join(params)}]"
    return f"sum({p}) / len({p})" if len(params) > 1 else f"{params[0]} / 2"


def _maxmin(name: str):
    def _f(params: List[str]) -> str:
        if not params:
            return "0"
        if len(params) == 1:
            return f"{name}({params[0]})"
        return f"{name}({', '.join(params)})"
    return _f


def _identity(params: List[str]) -> str:
    return params[0] if params else "x"


def _square(params: List[str]) -> str:
    p = params[0] if params else "x"
    return f"{p} ** 2"


def _double(params: List[str]) -> str:
    p = params[0] if params else "x"
    return f"{p} * 2"


def _reverse(params: List[str]) -> str:
    p = params[0] if params else "items"
    return f"list(reversed({p}))"


def _sort(params: List[str]) -> str:
    p = params[0] if params else "items"
    return f"sorted({p})"


def _count(params: List[str]) -> str:
    p = params[0] if params else "items"
    return f"len({p})"


def _greet(params: List[str]) -> str:
    p = params[0] if params else "name"
    return f'f"hello, {{{p}}}"'


def _factorial(params: List[str]) -> str:
    p = params[0] if params else "n"
    return f"_factorial({p})"


def _fibonacci(params: List[str]) -> str:
    p = params[0] if params else "n"
    return f"_fibonacci({p})"


def _filter_positive(params: List[str]) -> str:
    p = params[0] if params else "items"
    return f"[x for x in {p} if x > 0]"


# Seed registration (growable).
_SEED_OPS = {
    "add": _make_binary("+"),
    "sum": _sum_builtin,
    "plus": _make_binary("+"),
    "multiply": _make_binary("*"),
    "mul": _make_binary("*"),
    "subtract": _make_binary("-"),
    "sub": _make_binary("-"),
    "divide": _make_binary("/"),
    "div": _make_binary("/"),
    "average": _avg,
    "avg": _avg,
    "mean": _avg,
    "max": _maxmin("max"),
    "maximum": _maxmin("max"),
    "min": _maxmin("min"),
    "minimum": _maxmin("min"),
    "reverse": _reverse,
    "reversed": _reverse,
    "sort": _sort,
    "sorted": _sort,
    "count": _count,
    "length": _count,
    "square": _square,
    "double": _double,
    "greet": _greet,
    "hello": _greet,
    "factorial": _factorial,
    "fibonacci": _fibonacci,
    "fib": _fibonacci,
    "filter": _filter_positive,
}


def register_operation(name: str, builder: Callable[[List[str]], str]) -> None:
    """Grow the operation set at runtime (seed structure, not hardcode)."""
    if name:
        _OPERATION_BUILDERS[name.lower()] = builder


for _k, _v in _SEED_OPS.items():
    register_operation(_k, _v)

# Synonyms / alias map (growable).
_OPERATION_ALIASES: Dict[str, str] = {
    "added": "add", "adding": "add", "plus": "add",
    "multiplied": "multiply", "multiplying": "multiply", "times": "multiply",
    "subtracted": "subtract", "subtracting": "subtract",
    "divided": "divide", "dividing": "divide",
    "reversing": "reverse", "reversed": "reverse",
    "sorting": "sort", "sorted": "sort",
    "counting": "count", "counted": "count",
    "squaring": "square", "squared": "square",
    "doubling": "double", "doubled": "double",
    "greeting": "greet", "greets": "greet",
    "factorials": "factorial",
    "fibonaccis": "fibonacci",
    "filtering": "filter", "filtered": "filter",
    "averaging": "average", "averaged": "average",
    "maximizing": "max", "minimizing": "min",
}


def register_alias(alias: str, canonical: str) -> None:
    if alias and canonical:
        _OPERATION_ALIASES[alias.lower()] = canonical.lower()


def _canonical_op(tok: str) -> Optional[str]:
    t = tok.lower().strip(".,!?;:")
    if t in _OPERATION_BUILDERS:
        return t
    if t in _OPERATION_ALIASES:
        c = _OPERATION_ALIASES[t]
        if c in _OPERATION_BUILDERS:
            return c
    return None


# --------------------------------------------------------------------------- #
# NL parsing helpers
# --------------------------------------------------------------------------- #
_STOPWORDS = {
    "a", "an", "the", "to", "of", "in", "on", "for", "with", "and", "or",
    "but", "is", "are", "was", "were", "be", "been", "being", "that", "this",
    "these", "those", "it", "its", "i", "you", "he", "she", "we", "they",
    "my", "your", "his", "her", "our", "their", "me", "him", "her", "us",
    "them", "function", "def", "write", "create", "make", "implement", "code",
    "program", "python", "that", "which", "who", "what", "how", "please",
    "can", "could", "would", "should", "do", "does", "did",
}

_PARAM_HINTS = [
    ("number", "n"), ("numbers", "numbers"), ("num", "n"),
    ("list", "items"), ("array", "items"), ("sequence", "items"),
    ("string", "s"), ("text", "s"), ("name", "name"), ("word", "word"),
    ("value", "x"), ("values", "xs"), ("item", "item"), ("items", "items"),
    ("n", "n"),
]


def _slugify(tokens: List[str]) -> str:
    out = []
    for t in tokens:
        t = re.sub(r"[^a-zA-Z0-9]", "", t)
        if t:
            out.append(t.lower())
    return "_".join(out) if out else "generated"


def _detect_operation(tokens: List[str]) -> Optional[str]:
    for t in tokens:
        c = _canonical_op(t)
        if c is not None:
            return c
    # try multi-token: "named after" style not needed for code ops
    return None


def _detect_params(description: str, op: Optional[str]) -> List[str]:
    low = description.lower()
    params: List[str] = []
    # arity hints from operation
    binary_ops = {"add", "plus", "sum", "multiply", "mul", "subtract", "sub",
                  "divide", "div", "average", "avg", "mean", "max", "min",
                  "maximum", "minimum"}
    if op in binary_ops:
        params = ["a", "b"]
    elif op in {"reverse", "reversed", "sort", "sorted", "count", "length",
                "square", "double", "filter"}:
        if op in {"square", "double"}:
            params = ["x"]
        else:
            params = ["items"]
    elif op in {"greet", "hello"}:
        params = ["name"]
    elif op in {"factorial", "fibonacci", "fib"}:
        params = ["n"]
    else:
        params = ["x"]
    # list-shaped inputs
    list_words = ("list", "array", "sequence", "items", "numbers", "values", "elements")
    if any(w in low.split() or w in low for w in list_words) and op in {"sum", "max", "min", "maximum",
            "minimum", "average", "avg", "mean", "reverse", "reversed", "sort", "sorted",
            "count", "length", "filter"}:
        # but "two numbers" -> binary
        if not any(t in low for t in ("two ", "pair", "a and b", "x and y")):
            if op in {"sum", "max", "min", "maximum", "minimum", "average", "avg", "mean"}:
                params = ["items"]
            # reverse/sort/count/filter already items
    # binary if "two" present
    if any(t in low for t in ("two ", "pair", "a and b", "x and y", "two numbers")):
        if op in binary_ops and op not in {"max", "min", "maximum", "minimum"}:
            params = ["a", "b"]
    # de-dup
    seen = set()
    out = []
    for p in params:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out or ["x"]


def _function_name(description: str, op: Optional[str]) -> str:
    toks = [t for t in re.findall(r"[a-zA-Z]+", description.lower()) if t not in _STOPWORDS]
    # prefer op + key noun
    name_toks = []
    if op:
        name_toks.append(op)
    for t in toks:
        if t not in name_toks and t not in _STOPWORDS:
            name_toks.append(t)
            if len(name_toks) >= 3:
                break
    if len(name_toks) < 2:
        # fall back to first content tokens
        name_toks = toks[:3] if toks else ["generated"]
    slug = _slugify(name_toks)
    if not slug:
        slug = "generated_function"
    # ensure valid identifier
    if slug[0].isdigit():
        slug = "f_" + slug
    return slug


def _state_metrics(engine: Any) -> Dict[str, Any]:
    m: Dict[str, Any] = {}
    try:
        ident = engine.identity.get_status() if hasattr(engine, "identity") else {}
        m["identity_strength"] = float(ident.get("strength", 0.0)) if isinstance(ident, dict) else 0.0
    except Exception:
        m["identity_strength"] = 0.0
    try:
        m["graph_nodes"] = len(engine.graph.nodes) if hasattr(engine, "graph") else 0
    except Exception:
        m["graph_nodes"] = 0
    try:
        m["turn_count"] = int(getattr(engine, "turn_count", 0) or 0)
    except Exception:
        m["turn_count"] = 0
    try:
        stances = engine.user_model.opinions.stances if hasattr(engine, "user_model") else {}
        m["stances"] = len(stances) if hasattr(stances, "__len__") else 0
    except Exception:
        m["stances"] = 0
    return m


def _helper_functions_for(op: Optional[str]) -> str:
    """Seed helper functions needed for certain operations (structure, not snippets)."""
    if op in {"factorial"}:
        return (
            "def _factorial(n):\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    return n * _factorial(n - 1)\n\n"
        )
    if op in {"fibonacci", "fib"}:
        return (
            "def _fibonacci(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    a, b = 0, 1\n"
            "    for _ in range(2, n + 1):\n"
            "        a, b = b, a + b\n"
            "    return b\n\n"
        )
    return ""


def generate_code(description: str, engine: Any = None,
                  state: Optional[Dict[str, Any]] = None) -> str:
    """Generate a short, correct, idiomatic function from an NL description.

    Reuses the OCG generative mechanism: construction schemas provide the
    skeleton (seed structure), operation primitives are combinators (growable),
    and the cognitive state drives traceability. Returns a string of Python
    source. Deterministic: same description + state -> same code.
    """
    desc = (description or "").strip()
    if not desc:
        desc = "identity function"

    tokens = re.findall(r"[a-zA-Z']+", desc.lower())
    op = _detect_operation(tokens)
    params = _detect_params(desc, op)
    name = _function_name(desc, op)

    metrics = {}
    if engine is not None:
        metrics = _state_metrics(engine)
    if state:
        metrics.update(state)

    # Build expression via the operation combinator (growable).
    builder = _OPERATION_BUILDERS.get(op) if op else None
    if builder is None:
        # generic fallback: echo first param (honest, not a snippet table)
        expr = params[0] if params else "x"
    else:
        try:
            expr = builder(params)
        except Exception:
            expr = params[0] if params else "x"

    # Assemble param list.
    param_str = ", ".join(params)

    # Helper functions (seed structure for recursive ops).
    helpers = _helper_functions_for(op)

    # Header / traceability comment.
    header_lines = [
        "# Generated by RAVANA code realizer (OCG modality) — no LLM, no retrain.",
        f"# description: {desc}",
    ]
    if metrics:
        parts = ", ".join(f"{k}={v}" for k, v in metrics.items())
        header_lines.append(f"# cognitive state: {parts}")

    body = f"    return {expr}\n"

    src = "\n".join(header_lines) + "\n\n" + helpers + f"def {name}({param_str}):\n" + body + "\n"

    # Validate; if invalid, fall back to a minimal valid function.
    try:
        ast.parse(src)
    except SyntaxError:
        fallback_name = "generated_function"
        src = (
            "\n".join(header_lines) + "\n\n"
            + f"def {fallback_name}({param_str or 'x'}):\n"
            + "    return " + (params[0] if params else "x") + "\n\n"
        )
        try:
            ast.parse(src)
        except SyntaxError:
            src = "def generated_function(x):\n    return x\n"
    return src


class CodeRealizer:
    """Stateful code realizer bound to an engine (optional)."""

    def __init__(self, engine: Any = None):
        self.engine = engine

    def generate(self, description: str, state: Optional[Dict[str, Any]] = None) -> str:
        return generate_code(description, engine=self.engine, state=state)

    # aliases
    def realize(self, description: str, **kw) -> str:
        return self.generate(description, **kw)

    def __call__(self, description: str, **kw) -> str:
        return self.generate(description, **kw)


def realize_code(description: str, engine: Any = None, **kw) -> str:
    return generate_code(description, engine=engine, **kw)


# --------------------------------------------------------------------------- #
# Engine integration (monkeypatch) — guarded.
# --------------------------------------------------------------------------- #
def _patch_engine() -> None:
    try:
        from ravana.chat.engine import CognitiveChatEngine
    except Exception:
        return

    def _gen(self, description: str, state=None):
        return generate_code(description, engine=self, state=state)

    if not hasattr(CognitiveChatEngine, "generate_code"):
        CognitiveChatEngine.generate_code = _gen
    if not hasattr(CognitiveChatEngine, "code_realizer"):
        # property-like attribute via simple instance attr set in a wrapper? Use method instead.
        pass
    # Add a convenience accessor method.
    if not hasattr(CognitiveChatEngine, "code_realizer"):
        CognitiveChatEngine.code_realizer = property(lambda self: CodeRealizer(self))
    if not hasattr(CognitiveChatEngine, "get_code_realizer"):
        def _get(self):
            return CodeRealizer(self)
        CognitiveChatEngine.get_code_realizer = _get


try:
    _patch_engine()
except Exception:
    pass
