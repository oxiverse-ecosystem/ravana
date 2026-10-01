"""List the miner patterns with 2+ capture groups (i.e. DYADIC comparatives).

The fix branches on the PATTERN's arity -- a grammatical property -- so this
enumerates exactly which patterns that would affect.
"""
from __future__ import annotations
import ast, os, re, sys

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(TREE, "ravana", "src", "ravana", "chat", "user_model.py")
tree = ast.parse(open(SRC, encoding="utf-8").read())

pats = None
for node in ast.walk(tree):
    if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple):
        names = [e.id for e in node.target.elts if isinstance(e, ast.Name)]
        if names[:3] == ["_pat", "_pol", "_conf"]:
            pats = ast.literal_eval(node.iter)
            break

PROBES = [
    "i think remote work is better than office work",
    "tea beats coffee",
    "i believe trains are better than planes",
    "honestly i think the mountains are finer than the coast",
    "small towns make better humans than cities",
    "the sea is a better teacher than any classroom",
    "i like hiking a lot",
    "i am really into pottery",
    "her constant humming gets to me",
]

dyadic = [(i, p, pol, conf) for i, (p, pol, conf) in enumerate(pats)
         if re.compile(p, re.IGNORECASE).groups >= 2]
print(f"patterns total      : {len(pats)}")
print(f"patterns with >=2 grp: {len(dyadic)}")
for i, p, pol, conf in dyadic:
    print(f"  [{i:2d}] pol={pol:+.2f} {p[:100]!r}")

print("\n--- what each probe hits (showing only the LAST-indexed group, as today)")
for q in PROBES:
    print(f"\n{q!r}")
    for i, p, pol, conf in dyadic:
        for m in re.finditer(p, q, re.IGNORECASE):
            g1 = m.group(1)
            gl = m.group(m.lastindex)
            print(f"  [{i:2d}] g1={g1!r} lastindex={m.lastindex} -> {gl!r}")
sys.exit(0)