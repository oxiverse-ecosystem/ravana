"""
Pytest configuration — dynamic import helpers.

Controls how test files resolve imports:

  Mode       Env var                        Import path
  ──────────────────────────────────────────────────────────────────
  dev        RAVANA_PACKAGE_TEST unset       core.governor / agent.* / scripts.*
  package    RAVANA_PACKAGE_TEST=1           ravana_grace.core.governor / ……

Dev mode adds the source tree to sys.path so tests work without the
package installed.  Package mode uses the installed wheel (PyPI / local
build) so we can validate a release before publishing.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / "src"
_PKG_ROOT = _SRC_ROOT / "ravana_grace"  # contains core/, agent/, …

# ── mode detection ──────────────────────────────────────────────────
_PACKAGE_TEST = os.environ.get("RAVANA_PACKAGE_TEST", "").lower() in (
    "1",
    "true",
    "yes",
)

if not _PACKAGE_TEST:
    # Dev mode — wire up source-tree directories so bare imports work.
    _DEV_PATHS = [
        str(_SRC_ROOT),                                # ravana_grace.* (full package)
        str(_PKG_ROOT),                                # core.*, agent.*
        str(_PKG_ROOT / "interface_agent" / "scripts"),  # scripts.*
    ]
    for _p in _DEV_PATHS:
        if _p not in sys.path:
            sys.path.insert(0, _p)


# ── generic resolver ────────────────────────────────────────────────
def _resolve(prefix: str, dev_modpath: str, names: tuple):
    """Return one or more names from the module resolved at *prefix*.

    *prefix*      – dotted path relative to ``ravana_grace`` (package mode)
    *dev_modpath* – dotted path to ``importlib.import_module`` (dev mode)
    *names*       – attribute(s) to extract; empty → return the module.
    """
    if _PACKAGE_TEST:
        mod = importlib.import_module(f"ravana_grace.{prefix}")
    else:
        mod = importlib.import_module(dev_modpath)
    if not names:
        return mod
    if len(names) == 1:
        return getattr(mod, names[0])
    return tuple(getattr(mod, n) for n in names)


# ── public helpers ──────────────────────────────────────────────────
def _strip_dot(s: str) -> str:
    """Strip a trailing dot so ``importlib.import_module`` works."""
    return s.rstrip(".")


def import_core(submodule: str, *names: str):
    """Import from ``core.{submodule}``.

    Dev:  ``from core.{submodule} import …``
    Pkg:  ``from ravana_grace.core.{submodule} import …``

    When *submodule* is ``""``, imports directly from ``core`` / ``ravana_grace.core``.
    """
    pkg = _strip_dot(f"core.{submodule}")
    dev = _strip_dot(f"core.{submodule}")
    return _resolve(pkg, dev, names)


def import_agent(submodule: str, *names: str):
    """Import from ``agent.{submodule}``.

    Dev:  ``from agent.{submodule} import …``
    Pkg:  ``from ravana_grace.agent.{submodule} import …``
    """
    pkg = _strip_dot(f"agent.{submodule}")
    dev = _strip_dot(f"agent.{submodule}")
    return _resolve(pkg, dev, names)


def import_scripts(submodule: str, *names: str):
    """Import from ``interface_agent/scripts/{submodule}``.

    Dev:  ``from {submodule} import …``  (scripts dir on sys.path)
    Pkg:  ``from ravana_grace.interface_agent.scripts.{submodule} import …``
    """
    pkg = _strip_dot(f"interface_agent.scripts.{submodule}")
    dev = submodule
    return _resolve(pkg, dev, names)
