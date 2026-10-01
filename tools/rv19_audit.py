"""AST-level hardcoding audit, scoped to ADDED lines of the FIX-RV-19 diff.

The conventional `git diff | grep -oE '"[a-z][^"]{45,}"'` also matches
docstrings and comments, so a fix that CITES the failing probe in its own
documentation looks like a violation. This walks the parsed AST of each added
line, skips docstrings, and reports only EXECUTABLE string literals.

Shared with any RAVANA branch that needs the same check:
    python tools/rv19_audit.py <base-ref>
"""
from __future__ import annotations
import ast, subprocess, sys

BASE = sys.argv[1] if len(sys.argv) > 1 else "github/main"
SRC_FILES = [
    "ravana/src/ravana/chat/user_model.py",
    "ravana/src/ravana/chat/engine.py",
    "ravana/src/ravana/chat/slot_naming.py",
]

diff = subprocess.run(
    ["git", "diff", BASE, "--"] + SRC_FILES,
    capture_output=True, text=True, encoding="utf-8", errors="replace").stdout

# Collect the set of NEW-line numbers per file from the unified diff.
added: dict[str, set[int]] = {}
cur = None
for line in diff.split("\n"):
    if line.startswith("+++ b/"):
        cur = line[6:]
        added.setdefault(cur, set())
    elif line.startswith("@@"):
        # @@ -a,b +c,d @@
        seg = line.split("+")[1].split("@@")[0].strip()
        start = int(seg.split(",")[0])
    elif cur and line.startswith("+") and not line.startswith("+++"):
        added[cur].add(start)
        start += 1
    elif cur and line.startswith("-") and not line.startswith("---"):
        pass
    elif cur and line.startswith(" "):
        start += 1

PROBE_TERMS = ("remote work", "office work", "hiking", "trains", "planes",
               "mountains", "coast", "tea", "coffee")

long_lits: list[tuple[str, int, str]] = []
probe_derived: list[tuple[str, int, str]] = []
n_exec = 0

for path, lines in added.items():
    if not lines:
        continue
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except (OSError, SyntaxError) as exc:
        print(f"SKIP {path}: {exc}")
        continue

    # Docstring positions per module/class/function, so a fix that cites the
    # failing probe in its own documentation is not counted as authored prose.
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                docstrings.add(body[0].lineno)

    # A string that is an operand of a regex-ish call (re.compile/match/search/
    # finditer/findall/sub/split) or of BinOp is PATTERN VOCABULARY, not prose.
    regex_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "attr", None) or getattr(fn, "id", None)
            if name in {"compile", "match", "search", "finditer", "findall",
                        "sub", "split", "fullmatch"}:
                for a in list(node.args) + [k.value for k in node.keywords]:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str):
                        regex_nodes.add(id(a))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            if isinstance(node.right, ast.Constant) and \
                    isinstance(node.right.value, str):
                regex_nodes.add(id(node.right))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if node.end_lineno is None or node.lineno not in lines:
            continue
        if node.lineno in docstrings:
            continue
        s = node.value
        is_regex = id(node) in regex_nodes
        n_exec += 1
        if len(s) >= 45 and "\n" not in s:
            kind = "REGEX" if is_regex else "PROSE"
            long_lits.append((path, node.lineno, f"{kind} len={len(s)}"))
            low = s.lower()
            if any(t in low for t in PROBE_TERMS):
                probe_derived.append((path, node.lineno, s[:90]))

print(f"added lines scanned : {sum(len(v) for v in added.values())}")
print(f"executable literals : {n_exec}")
print(f"literals >= 45 chars: {len(long_lits)}")
for p, ln, desc in long_lits:
    print(f"  {p}:{ln}  {desc}")
print(f"\nprobe-derived among them: {len(probe_derived)}")
for p, ln, s in probe_derived:
    print(f"  !! {p}:{ln}  {s!r}")

# Prose (non-regex) literals are the ones that would constitute authored
# personality. Regex/regex-alike vocabulary is the allowed seed kind.
prose = [x for x in long_lits if x[2].startswith("PROSE")]
print(f"\nlong NON-regex literals: {len(prose)}")
for p, ln, _ in prose:
    print(f"  {p}:{ln}")
sys.exit(1 if probe_derived else 0)