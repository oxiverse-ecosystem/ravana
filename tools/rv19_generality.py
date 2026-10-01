"""Why do the generality probes still miss the loser side?

For each probe: which patterns fire, what groups they capture, and what the
store actually holds. The arity fix can only store a loser when the PATTERN
captures one -- a one-group comparative pattern has no loser to read.
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

sys.path.insert(0, os.path.join(TREE, "ravana_ml", "src"))
sys.path.insert(0, os.path.join(TREE, "ravana", "src"))
from ravana.chat.slot_naming import strip_reporting_frame  # noqa: E402

STOP = {"the", "a", "an", "my", "your", "our", "their", "i", "you", "he", "she",
        "we", "they", "me", "and", "but", "or", "of", "to", "in", "on", "at",
        "for", "with", "from", "by", "as", "than", "is", "are", "was", "were",
        "be", "not", "do", "does", "did", "really", "very", "just", "only"}

PROBES = [
    "tea beats coffee",
    "i believe trains are better than planes",
    "honestly i think the mountains are finer than the coast",
]

for q in PROBES:
    print(f"\n=== {q!r}")
    any_hit = False
    for i, (p, pol, conf) in enumerate(pats):
        for m in re.finditer(p, q, re.IGNORECASE):
            any_hit = True
            n = m.re.groups
            if n >= 2:
                sides = [(m.group(1), pol), (m.group(n), -pol)]
            else:
                sides = [(m.group(m.lastindex), pol)]
            print(f"  [{i:2d}] groups={n} pol={pol:+.2f}")
            for raw, sp in sides:
                raw = (raw or "").strip().lower()
                stripped = strip_reporting_frame(raw, STOP)
                flag = "  FRAME-STRIPPED" if stripped != raw else ""
                print(f"        raw={raw!r} -> {stripped!r}  sign={sp:+.2f}{flag}")
    if not any_hit:
        print("  (no pattern fired at all)")