"""wtbind — force a RAVANA test run to bind to THIS tree, not another checkout.

Why this exists: `.venv-real` carries an EDITABLE install whose meta_path
finder (`__editable___ravana_0_3_0_finder._EditableFinder`) hard-maps the
`ravana` package to the MAIN checkout. That finder sits on sys.meta_path, so
it outranks sys.path — a pytest run inside a git worktree can silently execute
the worktree's tests/ against the MAIN repo's ravana/ and report a wall of
ModuleNotFoundError / behavioural failures that are pure fiction.

Also present is a PATH HOOK (`__editable__.ravana-0.3.0.finder.__path_hook__`)
on sys.path_hooks, which needs removing too.

This plugin (as a pytest plugin, and importable as a plain function for
in-process drivers):

  1. removes every meta_path finder + path hook belonging to an editable install
  2. pins this tree's four source roots at the FRONT of sys.path
  3. purges already-imported ravana* modules that resolved outside this tree
  4. ASSERTS in pytest_report_header that ravana.chat.engine and a
     branch-specific module resolve inside this tree

Step 4 is what makes a wrong-tree run impossible rather than merely unlikely.
"""
from __future__ import annotations

import os
import sys

#: Populated by bind_to_tree(); the tree every later assertion compares against.
BOUND_ROOT: str | None = None


def _is_editable(obj) -> bool:
    """True for a meta_path finder or path hook belonging to an editable install.

    Matches on the CLASS name, the module name, AND the repr, because the
    install ships as a bare instance whose ``__name__`` is empty (checking
    ``__name__`` alone silently misses it — verified 2026-10-01).
    """
    haystack = " ".join(
        str(x) for x in (
            type(obj).__name__,
            getattr(obj, "__name__", "") or "",
            getattr(obj, "__module__", "") or "",
            repr(obj),
        )
    ).lower()
    return "editable" in haystack


def _source_roots(root: str) -> list[str]:
    """The four source roots that make up a RAVANA tree, in priority order."""
    return [os.path.join(root, "ravana", "src"),
            os.path.join(root, "ravana_ml", "src"),
            os.path.join(root, "ravana-v2", "src"),
            root]


class _TreeBoundFinder:
    r"""Force `ravana*` to resolve inside the bound tree, whatever sys.path says.

    Removing the editable-install finder is NECESSARY but not SUFFICIENT. 18
    files under tests/unit/ hardcode the main checkout and do
    ``sys.path.insert(0, r"C:\Users\Likhith\Documents\Projects\ravana\ravana\src")``
    at MODULE IMPORT time — i.e. during pytest collection, long after any
    plugin hook has run. Whichever of those imports first wins, and a run in a
    worktree silently executes every later test against the MAIN checkout.

    This finder sits at sys.meta_path[0], so it is consulted BEFORE the normal
    path finder and re-asserts the binding at import time. It only claims the
    top-level RAVANA packages, and only when a same-named package actually
    exists in the bound tree — everything else falls through untouched, so it
    cannot shadow an unrelated third-party `ravana`.
    """

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self._roots = [os.path.abspath(p) for p in _source_roots(self.root)]

    # A finder must not claim everything (that would break every import).
    def find_module(self, fullname, path=None):  # pragma: no cover - legacy API
        return None

    def find_spec(self, fullname, path=None, target=None):
        import importlib.util
        top = fullname.split(".")[0]
        if top not in _BOUND_PACKAGES:
            return None
        # Claim ONLY the top-level package. Submodules must resolve through the
        # parent's __path__ (set below); claiming them too makes this finder
        # re-enter itself and blow the import stack with a RecursionError.
        if "." in fullname:
            return None
        for src_root in self._roots:
            pkg_dir = os.path.join(src_root, top)
            init = os.path.join(pkg_dir, "__init__.py")
            if os.path.isfile(init):
                return importlib.util.spec_from_file_location(
                    fullname, init,
                    submodule_search_locations=[pkg_dir])
        # Not a package in this tree (e.g. a namespace dir) — let it fall
        # through to the normal resolution rather than claiming it wrongly.
        return None

    def invalidate_caches(self):  # pragma: no cover - import protocol
        return None


#: Top-level packages the binder claims. Deliberately narrow: these are
#: RAVANA's own, and claiming any more would shadow unrelated site-packages.
_BOUND_PACKAGES = frozenset({"ravana", "ravana_ml", "ravana_grace", "ravana_v2"})


def bind_to_tree(root: str) -> str:
    """Bind this interpreter to `root`. Returns the absolute root."""
    global BOUND_ROOT
    root = os.path.abspath(root)
    BOUND_ROOT = root

    before = len(sys.meta_path)
    sys.meta_path = [f for f in sys.meta_path if not _is_editable(f)]
    before_hooks = len(sys.path_hooks)
    sys.path_hooks = [h for h in sys.path_hooks if not _is_editable(h)]
    sys.path_importer_cache.clear()

    for source_root in _source_roots(root):
        source_root = os.path.abspath(source_root)
        if os.path.isdir(source_root) and source_root in sys.path:
            sys.path.remove(source_root)
        if os.path.isdir(source_root):
            sys.path.insert(0, source_root)

    # Purge ravana* modules that came from another checkout.
    for name in [n for n in list(sys.modules)
                 if n == "ravana" or n.startswith("ravana.")]:
        mod_file = getattr(sys.modules[name], "__file__", None) or ""
        if os.path.abspath(mod_file).lower().startswith(root.lower()):
            continue
        del sys.modules[name]

    # Re-assert the binding at import time (see _TreeBoundFinder).
    finder = _TreeBoundFinder(root)
    sys.meta_path = [f for f in sys.meta_path
                     if not isinstance(f, _TreeBoundFinder)]
    sys.meta_path.insert(0, finder)
    return root


def assert_bound(root: str | None = None) -> str:
    """Raise unless ravana.chat.engine resolves inside the bound tree."""
    root = os.path.abspath(root or BOUND_ROOT or os.getcwd())
    import ravana.chat.engine as engine_mod  # noqa: WPS433
    resolved = os.path.abspath(engine_mod.__file__)
    if not resolved.lower().startswith(root.lower()):
        raise AssertionError(
            "WRONG TREE: ravana.chat.engine resolved to\n"
            f"  {resolved}\n"
            f"but this run must bind to\n  {root}\n"
            "An editable-install finder is still on sys.meta_path / "
            "sys.path_hooks. This run's results are NOT evidence."
        )
    return resolved


def pytest_report_header(config):  # pragma: no cover - pytest hook
    root = os.path.abspath(str(config.rootpath))
    resolved = assert_bound(root)
    return f"wtbind: BOUND ravana.chat.engine -> {resolved}"


def pytest_configure(config):  # pragma: no cover - pytest hook
    """Bind BEFORE collection so conftest/test imports resolve correctly."""
    root = bind_to_tree(str(config.rootpath))
    _link_shared_glove_cache(root)


def _link_shared_glove_cache(root: str) -> str | None:
    """Point this tree's GloVe cache at the one the main checkout already built.

    `CognitiveChatEngine` resolves its cache to `<repo>/data/ravana_glove_cache.npz`,
    and `data/` is gitignored — so a fresh worktree has no cache and the engine
    silently runs with NO GloVe vectors ("[GloVe] Offline mode ... running
    without GloVe"). `auto_expand_concepts` needs those vectors, so without this
    any evidence-ingestion assertion is vacuous.

    This copies the main checkout's cache into the worktree (a hard link, so no
    144MB duplicate). It only ever ADDS a file; it never deletes, and never
    touches `ravana/data/` (the `git clean` landmine).
    """
    cache_name = "ravana_glove_cache.npz"
    target = os.path.join(root, "data", cache_name)
    if os.path.exists(target):
        return target
    main_root = r"C:\Users\Likhith\Documents\Projects\ravana"
    if os.path.abspath(main_root).lower() == root.lower():
        return None
    source = os.path.join(main_root, "data", cache_name)
    if not os.path.exists(source):
        return None
    os.makedirs(os.path.dirname(target), exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        # Different volume, or links unavailable: a copy is still correct, just
        # slower. Never let this break the run.
        import shutil
        shutil.copy2(source, target)
    return target
