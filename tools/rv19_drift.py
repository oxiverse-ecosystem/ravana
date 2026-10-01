"""Prove the routing gate in engine.py has DRIFTED from the shared constant.

engine.py defines an inline `_selfopinion` regex at the process_turn gate.
engine_memory.py defines `_SELF_OPINION_SHAPE`, used by engine_self_query.py
(`_agent_opinion`) and by engine_memory.py itself to suppress episodic recall.
Both are read here with `ast` so this probe is a pure-text check and can never
drift from the source it audits.
"""
from __future__ import annotations
import ast, os, re, sys

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAT = os.path.join(TREE, "ravana", "src", "ravana", "chat")


def literal_string(path: str, name: str) -> str:
    """Return the concatenated string literal assigned to `name`."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    return ast.literal_eval(node.value)
    raise SystemExit(f"{name} not found in {path}")


def inline_gate(path: str) -> str:
    """Return the regex literal passed to the inline `_selfopinion = re.search`."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "attr", None) == "search"
                and any(isinstance(t, ast.Name) and t.id == "_selfopinion"
                        for t in node.targets)):
            return ast.literal_eval(node.value.args[0])
    raise SystemExit("inline _selfopinion gate not found")


INLINE = inline_gate(os.path.join(CHAT, "engine.py"))
SHARED = literal_string(os.path.join(CHAT, "engine_memory.py"), "_SELF_OPINION_SHAPE")

QUERIES = [
    "do you prefer remote work or office work?",
    "which do you prefer, remote work or office work?",
    "do you think you're more of a remote worker or an office worker?",
    "do you think jazz is great?",
    "what's your take on remote work?",
]

print(f"{'query':62s} {'inline':>7s} {'shared':>7s}")
print("-" * 80)
diverge = 0
for q in QUERIES:
    a = bool(re.search(INLINE, q, re.IGNORECASE))
    b = bool(re.search(SHARED, q, re.IGNORECASE))
    if a != b:
        diverge += 1
    print(f"{q:62s} {str(a):>7s} {str(b):>7s}"
          f"{'  <-- DIVERGES' if a != b else ''}")

print(f"\n'prefer' in inline gate : {'prefer' in INLINE}")
print(f"'prefer' in shared shape: {'prefer' in SHARED}")
print(f"queries that diverge    : {diverge}")
sys.exit(1 if diverge else 0)