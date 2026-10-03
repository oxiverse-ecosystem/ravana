"""Bind sys.path/sys.meta_path to THIS worktree.

`.venv-real` carries an EDITABLE install whose meta_path finder hard-maps
`ravana` to the MAIN checkout (C:\\Users\\Likhith\\Documents\\Projects\\ravana).
That finder outranks sys.path, so a worktree probe silently measures MAIN.
Strip it and pin this tree, then assert it took.
"""
import sys


def bind(root):
    sys.meta_path[:] = [
        f for f in sys.meta_path
        if "editable" not in getattr(f, "__module__", "").lower()
        and "editable" not in type(f).__name__.lower()
        and "editable" not in getattr(type(f), "__name__", "").lower()
    ]
    for p in (f"{root}\\ravana_ml\\src", f"{root}\\ravana\\src",
              f"{root}\\ravana-v2\\src", root):
        if p not in sys.path:
            sys.path.insert(0, p)
    for m in [m for m in sys.modules
              if m == "ravana" or m.startswith(("ravana.", "ravana_ml", "ravana_grace"))]:
        del sys.modules[m]


def verify(root):
    import ravana.chat.engine as E
    here = root.lower().replace("/", "\\")
    p = E.__file__.replace("/", "\\").lower()
    if here not in p:
        raise SystemExit(
            f"WRONG TREE BOUND: ravana.chat.engine resolved to {E.__file__}\n"
            f"expected inside {root}")
    print(f"# tree bound: {E.__file__}")
    return True