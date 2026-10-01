"""Which miner patterns fire on the FIX-RV-19 utterance, and which keys win.

Extracts the REAL pattern tuple from user_model.py's AST (not a re-typed copy)
so this cannot drift from what the miner actually runs.
"""
from __future__ import annotations
import ast, os, re, sys

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(TREE, "ravana", "src", "ravana", "chat", "user_model.py")
src = open(SRC, encoding="utf-8").read()
tree = ast.parse(src)

pats = None
for node in ast.walk(tree):
    if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple):
        names = [e.id for e in node.target.elts if isinstance(e, ast.Name)]
        if names[:3] == ["_pat", "_pol", "_conf"]:
            pats = ast.literal_eval(node.iter)
            break
if pats is None:
    raise SystemExit("opinion pattern tuple not found")

q = sys.argv[1] if len(sys.argv) > 1 else "i think remote work is better than office work"
print(f"utterance: {q!r}\n{len(pats)} patterns\n")
for i, (pat, pol, conf) in enumerate(pats):
    for m in re.finditer(pat, q, re.IGNORECASE):
        hit = [m.group(g) for g in range(1, m.re.groups + 1)]
        print(f"[{i:2d}] groups={m.re.groups} pol={pol:+.2f} lastindex={m.lastindex} hit={hit}")