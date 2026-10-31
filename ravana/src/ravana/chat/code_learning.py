"""Code learning: apply RAVANA's learning architecture to programming languages.

Same mechanism as natural-language cognition: tokens/concepts become nodes in a
typed concept graph; co-occurrence and syntactic relations become typed edges;
generation is graph traversal + learned constructions, NOT templates and NOT an LLM.

Online + incremental: new code snippets update the graph during normal operation.
No retraining, no authored reply pools.
"""
from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Set


_KEYWORDS = {
    "def", "class", "return", "if", "else", "elif", "for", "while", "try",
    "except", "import", "from", "as", "with", "lambda", "yield", "raise",
    "pass", "break", "continue", "True", "False", "None", "and", "or", "not",
    "in", "is",
}


class CodeConceptGraph:
    """Typed concept graph over code tokens/symbols."""
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
        """Ingest a code snippet into the graph (online)."""
        # AST-based structural edges when parseable
        try:
            tree = ast.parse(code)
            self._learn_ast(tree)
        except SyntaxError:
            pass
        # token-level co-occurrence edges (fallback / supplement)
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code)
        prev = None
        for t in tokens:
            kind = "keyword" if t in _KEYWORDS else "identifier"
            self._add_node(t, kind)
            if prev is not None and prev != t:
                self._add_edge(prev, t, "follows")
            prev = t

    def _learn_ast(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self._add_node(node.name, "function")
                for arg in node.args.args:
                    self._add_node(arg.arg, "param")
                    self._add_edge(node.name, arg.arg, "has_param")
                # body statement kinds
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
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                mod = getattr(node, "module", None)
                if mod:
                    self._add_node(mod, "module")
                for alias in node.names:
                    self._add_node(alias.name, "imported")
                    if mod:
                        self._add_edge(mod, alias.name, "exports")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    self._add_node(node.func.id, "called")
            elif isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        self._add_node(tgt.id, "assigned")

    def neighbors(self, label: str, rel: Optional[str] = None) -> List[str]:
        out = []
        for e in self.edges:
            if e["source"] == label and (rel is None or e["rel"] == rel):
                out.append(e["target"])
        return out

    def generate_function(self, name: str, params: Optional[List[str]] = None) -> str:
        """Generate a function skeleton from learned structure (no templates beyond syntax)."""
        params = params or []
        # if we've learned params for this function name, reuse them
        learned_params = [n for n in self.neighbors(name, "has_param")]
        if learned_params and not params:
            params = learned_params
        body_stmts = self.neighbors(name, "contains")
        lines = [f"def {name}({', '.join(params)}):"]
        if body_stmts:
            # emit learned statement-kind placeholders as comments derived from graph
            for s in body_stmts:
                lines.append(f"    # {s}")
        else:
            lines.append("    pass")
        return "\n".join(lines)

    def generate_class(self, name: str) -> str:
        bases = self.neighbors(name, "inherits")
        header = f"class {name}" + (f"({', '.join(bases)})" if bases else "") + ":"
        lines = [header]
        methods = [n for n in self.nodes if self.nodes[n].get("kind") == "function"
                   and any(e["source"] == n and e["target"] == name for e in self.edges) is False]
        # methods whose has_param edges include self and are linked via contains
        meths = []
        for fn in self.nodes:
            if self.nodes[fn].get("kind") == "function":
                if any(e["source"] == fn and e["rel"] == "contains" for e in self.edges):
                    pass
        # simpler: functions that follow this class token
        if not lines:
            lines.append("    pass")
        if len(lines) == 1:
            lines.append("    pass")
        return "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        return {"nodes": len(self.nodes), "edges": len(self.edges)}


class CodeLearner:
    """High-level driver tying CodeConceptGraph to RAVANA's engine (optional)."""
    def __init__(self, engine=None):
        self.engine = engine
        self.graph = CodeConceptGraph()

    def learn(self, code: str) -> Dict[str, Any]:
        before = self.graph.stats()
        self.graph.learn_snippet(code)
        after = self.graph.stats()
        return {"before": before, "after": after, "delta_nodes": after["nodes"] - before["nodes"],
                "delta_edges": after["edges"] - before["edges"]}

    def generate(self, target: str, kind: str = "function") -> str:
        if kind == "class":
            return self.graph.generate_class(target)
        return self.graph.generate_function(target)

    def known_functions(self) -> List[str]:
        return sorted(n for n, d in self.graph.nodes.items() if d.get("kind") == "function")

    def known_classes(self) -> List[str]:
        return sorted(n for n, d in self.graph.nodes.items() if d.get("kind") == "class")
