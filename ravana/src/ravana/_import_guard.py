"""Non-silent import-guard logging.

RAVANA historically wrapped optional imports in try/except that set a `_HAS_*`
flag and then *swallowed the error* — so when a REQUIRED capability (e.g.
threadpoolctl, python-dateutil) was missing, the engine booted "fine" and a
whole pipeline silently died. This module makes that impossible: every guard
that trips now logs at WARNING (genuinely-optional) or ERROR (internal/required)
with the module name and what capability is now unavailable, exactly once per
(module, kind) pair (re-imports are not repeated).

Usage (in the except clause of a guarded import):

    from ravana._import_guard import report_missing
    try:
        import some_module
        HAS_SOME = True
    except Exception:
        HAS_SOME = False
        report_missing("some_module", "feature name", kind="optional")

`kind` is one of:
    - "optional"  -> logs WARNING (heavy/extra feature; degradation intended)
    - "required"  -> logs ERROR   (core capability; was mislabelled optional)
    - "internal"  -> logs ERROR   (RAVANA's own module; a failed import is a bug)
"""
import logging

_log = logging.getLogger("ravana.imports")
_reported: set[tuple[str, str]] = set()


def report_missing(module: str, capability: str, kind: str = "optional") -> None:
    """Log (once) that `module` failed to import, disabling `capability`.

    kind: "optional" | "required" | "internal"
      optional/required -> WARNING (caller may also raise in CI via the gate test)
      internal          -> ERROR   (a real import failure that must never be hidden)
    """
    key = (module, kind)
    if key in _reported:
        return
    _reported.add(key)
    level = logging.ERROR if kind == "internal" else logging.WARNING
    if kind == "internal":
        text = ("Import of internal module %r failed — RAVANA's own capability "
                "%r is now DEAD. This is a bug (syntax/circular/bad import), not "
                "an optional feature." % (module, capability))
    elif kind == "required":
        text = ("REQUIRED dependency %r is missing — capability %r is disabled. "
                "This must be declared in pyproject.toml [project].dependencies; "
                "a missing install silently weakens the engine." % (module, capability))
    else:
        text = ("Optional dependency %r not installed — capability %r is "
                "disabled (degradation is intended; install it for full "
                "functionality)." % (module, capability))
    _log.log(level, text)
