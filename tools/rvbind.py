"""Pytest plugin that pins an RAVANA source tree so a WORKTREE run cannot
silently import the MAIN checkout.

Why this exists
---------------
The repo venv (`.venv-real`) carries an EDITABLE install of `ravana` whose
`__editable__...finder` module sits on `sys.meta_path`. A meta-path finder
outranks `sys.path`, so pytest running inside a git worktree can execute the
worktree's `tests/` against the MAIN repo's `ravana/` package. `conftest.py`
only reorders `sys.path` and cannot win against `sys.meta_path`. The symptom is a
wall of `ModuleNotFoundError` for modules the worktree's own source never
imports, or traceback paths pointing outside the worktree.

What it does
------------
1. Drops every `sys.meta_path` finder whose module name contains "editable".
2. Pins the tree's four source roots at the front of `sys.path`.
3. Purges already-imported `ravana*` modules that resolved OUTSIDE this tree.
4. Asserts, in `pytest_report_header`, that `ravana.chat.engine` and a branch-
   specific module resolve INSIDE this tree, so a wrong-tree run is reported
   instead of silently producing garbage counts.

Usage
-----
    pytest tests/unit/ -p rvbind

Committed so every branch can copy it; it is a test harness utility, not engine
code, and is safe to duplicate across worktrees.
"""
from __future__ import annotations

import os
import sys

_TREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ROOTS = [
    _TREE,
    os.path.join(_TREE, "ravana_ml", "src"),
    os.path.join(_TREE, "ravana", "src"),
    os.path.join(_TREE, "ravana-v2", "src"),
]


def _strip_editable_finders() -> int:
    """Remove editable-install meta-path finders that outrank sys.path."""
    kept = []
    removed = 0
    for finder in sys.meta_path:
        name = type(finder).__module__ or ""
        cls = type(finder).__name__ or ""
        if "editable" in name.lower() or "editable" in cls.lower():
            removed += 1
            continue
        kept.append(finder)
    sys.meta_path = kept
    return removed


def _purge_foreign_ravana() -> int:
    """Drop ravana* modules that were imported from outside this tree."""
    purged = 0
    for name, mod in list(sys.modules.items()):
        if name != "ravana" and not name.startswith("ravana."):
            continue
        path = getattr(mod, "__file__", None)
        if not path:
            continue
        try:
            outside = not os.path.abspath(path).startswith(_TREE)
        except (TypeError, ValueError):
            outside = True
        if outside:
            del sys.modules[name]
            purged += 1
    return purged


def _inside(path: str | None) -> bool:
    if not path:
        return False
    try:
        return os.path.abspath(path).startswith(_TREE)
    except (TypeError, ValueError):
        return False


def pytest_configure(config):
    _strip_editable_finders()
    for root in reversed(_ROOTS):
        if os.path.isdir(root) and root not in sys.path:
            sys.path.insert(0, root)
    _purge_foreign_ravana()


def pytest_report_header(config):
    import importlib

    for name in ("ravana", "ravana.chat.engine"):
        try:
            mod = importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - reported, not raised
            return f"rvbind: FAILED to import {name}: {exc}"
        if not _inside(getattr(mod, "__file__", None)):
            return (f"rvbind: WRONG TREE -- {name} resolved to "
                    f"{getattr(mod, '__file__', None)}, not under {_TREE}. "
                    f"Run pytest with `-p rvbind` from this worktree.")
    return f"rvbind: bound to {_TREE}"