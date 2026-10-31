"""Code learning: apply RAVANA's learning architecture to programming languages.

Same mechanism as natural-language cognition: tokens/concepts become nodes in a
typed concept graph; co-occurrence and syntactic relations become typed edges;
generation is graph traversal over the LEARNED structure, NOT templates and NOT
an LLM.

Online + incremental: new code snippets update the graph during normal operation.
No retraining, no authored reply pools. A function generated later is composed
from the edges that were actually learned from the snippets seen so far — change
the learned code and the generated output changes; with nothing learned, the
generator emits only the minimal syntactic skeleton (an honest empty state).
"""
from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional


_KEYWORDS = {
    "def", "class", "return", "if", "else", "elif", "for", "while", "try",
    "except", "import", "from", "as", "with", "lambda", "yield", "raise",
    "pass", "break", "continue", "True", "False", "None", "and", "or", "not",
    "in", "is",
}


def _extract_code(text: str) -> str:
    """Pull the code portion out of a natural-language message.

    A user who shares code usually wraps it in prose (\"here's my code:\"), a
    markdown fence (```python ... ```), or an indented block. Parsing the whole
    message as Python fails on the prose and silently loses every structural
    edge. This isolates the code so the AST can be built; if nothing code-like is
    found the original text is returned (token co-occurrence still runs)."""
    if not text:
        return text
    # 1. markdown code fences win outright
    fences = re.findall(r"```(?:[A-Za-z0-9_+-]*)\n?(.*?)```", text, re.DOTALL)
    if fences:
        return "\n".join(fences).strip()
    # 2. otherwise take from the first line that starts a real code construct
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if re.match(r"^(def |class |import |from |@|return |if |for |while |try:|else:)", s) \
           or (ln[:1] in (" ", "\t") and s and not s.startswith("#") and len(ln) - len(s) >= 1):
            start = i
            break
    if start is not None:
        return "\n".join(lines[start:]).strip()
    return text


class CodeConceptGraph:
    """Typed concept graph over code tokens/symbols.

    Edges are typed relations learned from the AST:
      has_param   function -> parameter name
      has_method  class    -> method (function) name
      contains    function -> statement-kind (e.g. "Return", "Call")
      calls       function -> callee name (a call inside the function body)
      assigns     function -> target variable name (an assignment in the body)
      inherits    class    -> base-class name
      exports     module   -> imported name
      follows     token    -> next token (co-occurrence, parse-independent)
    """
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self._adj: Dict[str, List[str]] = {}

    def _add_node(self, label: str, kind: str = "token") -> str:
        if label not in self.nodes:
            self.nodes[label] = {"label": label, "kind": kind, "count": 0}
        self.nodes[label]["count"] += 1
        return label

    def _add_edge(self, src: str, dst: str, rel: str) -> None:
        self.edges.append({"source": src, "target": dst, "rel": rel})
        self._adj.setdefault(src, []).append(dst)

    def learn_snippet(self, code: str) -> None:
        """Ingest a code snippet into the graph (online, incremental)."""
        code = _extract_code(code)
        try:
            tree = ast.parse(code)
            self._learn_ast(tree)
        except SyntaxError:
            pass
        # token-level co-occurrence edges (fallback / supplement when unparseable)
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code)
        prev = None
        for t in tokens:
            kind = "keyword" if t in _KEYWORDS else "identifier"
            self._add_node(t, kind)
            if prev is not None and prev != t:
                self._add_edge(prev, t, "follows")
            prev = t

    def _learn_ast(self, tree: ast.AST) -> None:
        """Recursive descent so we can attach call/assign edges to the ENCLOSING
        function (ast.walk is flat and loses scope)."""
        # Pass 1: functions + params + class methods + bases + imports.
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self._add_node(node.name, "function")
                for arg in node.args.args:
                    self._add_node(arg.arg, "param")
                    self._add_edge(node.name, arg.arg, "has_param")
                for stmt in node.body:
                    kind = type(stmt).__name__
                    self._add_node(kind, "stmt_kind")
                    self._add_edge(node.name, kind, "contains")
            elif isinstance(node, ast.ClassDef):
                self._add_node(node.name, "class")
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        self._add_node(base.id, "base")
                        self._add_edge(node.name, base.id, "inherits")
                # attach methods to the class
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        self._add_edge(node.name, item.name, "has_method")
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = getattr(node, "module", None)
                if mod:
                    self._add_node(mod, "module")
                for alias in node.names:
                    self._add_node(alias.name, "imported")
                    if mod:
                        self._add_edge(mod, alias.name, "exports")
        # Pass 2: function-scoped calls + assignments (needs enclosing scope).
        self._learn_body_edges(tree)

    def _enclosing_functions(self, tree: ast.AST):
        """Yield (func_name, stmt) for every statement under each function,
        tracking the enclosing function name."""
        stack: List[Optional[str]] = [None]
        func_nodes = {id(n): n.name for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef)}
        # Walk with an explicit stack to know the current function.
        for parent in ast.walk(tree):
            if isinstance(parent, ast.FunctionDef):
                for stmt in parent.body:
                    yield from self._walk_scope(stmt, parent.name)

    def _walk_scope(self, node: ast.AST, func_name: str):
        yield func_name, node
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef):
                # nested function: don't attribute its statements to the outer
                yield from self._walk_scope(child, child.name)
            else:
                yield from self._walk_scope(child, func_name)

    def _learn_body_edges(self, tree: ast.AST) -> None:
        for func_name, stmt in self._enclosing_functions(tree):
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                    self._add_node(sub.func.id, "called")
                    self._add_edge(func_name, sub.func.id, "calls")
                elif isinstance(sub, ast.Assign):
                    for tgt in sub.targets:
                        if isinstance(tgt, ast.Name):
                            self._add_node(tgt.id, "assigned")
                            self._add_edge(func_name, tgt.id, "assigns")

    def neighbors(self, label: str, rel: Optional[str] = None) -> List[str]:
        out: List[str] = []
        seen = set()
        for e in self.edges:
            if e["source"] == label and (rel is None or e["rel"] == rel):
                t = e["target"]
                if t not in seen:
                    seen.add(t)
                    out.append(t)
        return out

    def generate_function(self, name: str, params: Optional[List[str]] = None) -> str:
        """Generate a function skeleton composed from the LEARNED graph.

        No authored body: params come from has_param edges, the body from the
        calls/assignments actually learned for this function. With nothing learned
        it degrades to the minimal honest skeleton (params + `pass`).
        """
        params = list(params or [])
        learned_params = self.neighbors(name, "has_param")
        if learned_params and not params:
            params = learned_params
        lines = [f"def {name}({', '.join(params)}):"]
        body: List[str] = []
        for callee in self.neighbors(name, "calls"):
            body.append(f"    {callee}()  # learned call")
        for var in self.neighbors(name, "assigns"):
            body.append(f"    {var} = None  # learned assignment")
        if not body:
            # fall back to learned statement-kind hints as comments (still graph)
            for s in self.neighbors(name, "contains"):
                body.append(f"    # {s}")
        if not body:
            body.append("    pass")
        lines.extend(body)
        return "\n".join(lines)

    def generate_class(self, name: str) -> str:
        """Generate a class skeleton from learned methods + bases."""
        bases = self.neighbors(name, "inherits")
        header = f"class {name}" + (f"({', '.join(bases)})" if bases else "") + ":"
        lines = [header]
        methods = self.neighbors(name, "has_method")
        if methods:
            for m in methods:
                mparams = self.neighbors(m, "has_param")
                lines.append(f"    def {m}({', '.join(mparams)}):")
                lines.append("        pass")
        else:
            lines.append("    pass")
        return "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        rels: Dict[str, int] = {}
        for e in self.edges:
            rels[e["rel"]] = rels.get(e["rel"], 0) + 1
        return {"nodes": len(self.nodes), "edges": len(self.edges), "relations": rels}


class CodeLearner:
    """High-level driver tying CodeConceptGraph to RAVANA's engine (optional).

    Engine-agnostic: the graph learns online from any snippet passed to learn().
    When constructed with an engine it can also surface learned functions/classes
    back to the chat path, but it never blocks or alters the existing reply flow.
    """
    def __init__(self, engine=None):
        self.engine = engine
        self.graph = CodeConceptGraph()

    def learn(self, code: str) -> Dict[str, Any]:
        before = self.graph.stats()
        self.graph.learn_snippet(code)
        after = self.graph.stats()
        return {
            "before": before,
            "after": after,
            "delta_nodes": after["nodes"] - before["nodes"],
            "delta_edges": after["edges"] - before["edges"],
        }

    def generate(self, target: str, kind: str = "function") -> str:
        if kind == "class":
            return self.graph.generate_class(target)
        return self.graph.generate_function(target)

    def known_functions(self) -> List[str]:
        return sorted(n for n, d in self.graph.nodes.items() if d.get("kind") == "function")

    def known_classes(self) -> List[str]:
        return sorted(n for n, d in self.graph.nodes.items() if d.get("kind") == "class")
