"""Pickle forward/backward compatibility for the cognitive state snapshot.

WHY THIS EXISTS
---------------
``weights/ravana_weights*.pkl`` is RAVANA's canonical self: identity, stances,
personal facts, beliefs, the concept graph. One snapshot is shared across every
round, and each round may be running a DIFFERENT commit of the tree (the QA loop
puts each round on its own ``auto/round-<STAMP>`` branch).

Pickle stores a class by ``(module, name)``. If a round saves the snapshot from a
commit that has a module a later commit does not, then EVERY later load aborts
with ``ModuleNotFoundError`` and ``_load()`` returns ``False`` -- the engine
silently boots as a blank baby. The whole 600 MB self is discarded because of
one unresolvable class reference, and nothing but a one-line ``[Load error]``
print says so.

Measured 2026-09-30 (round t_eee413e7): the canonical snapshot had been written
by round t_e45928d1, which added ``ravana.chat.evaluative_polarity``. That
branch was never merged to main, so ``main`` had no such module and
``_load()`` failed on every single boot. The round probe measured
``turn_count=0`` and a 116-node seed graph -- a blank mind, every round, with no
error surfaced anywhere the loop looks.

THE RULE
--------
A missing class must cost you THAT class's state, never the entire mind. So an
unresolvable class resolves to an inert placeholder that accepts construction
and ``__setstate__`` and does nothing, and the load reports exactly which
classes were dropped. A degraded component is recoverable; a silently blank
brain is not.

This is deliberately not a per-module allowlist. Any ``ravana*`` module that
cannot be imported degrades the same way, so renaming or splitting a module
never costs the whole self again.
"""
from __future__ import annotations

import pickle
from typing import List, Tuple

# Modules whose absence we tolerate. A missing third-party module (numpy,
# torch) is a real environment fault and must still raise loudly.
_TOLERATED_PREFIXES = ("ravana",)


def _is_tolerated(module: str) -> bool:
    head = module.split(".", 1)[0]
    return head in _TOLERATED_PREFIXES or module in (
        "ravana_chat", "scripts.ravana_chat", "__main__",
    )


# One placeholder CLASS per (module, name) so identity is stable within a load
# and an isinstance check against the same missing name cannot collide with a
# different one.
_PLACEHOLDERS: dict[Tuple[str, str], type] = {}

# Filled in by the most recent load; the engine prints it so a degraded load is
# visible instead of silent.
LAST_DROPPED_CLASSES: List[Tuple[str, str]] = []


def _placeholder_class(module: str, name: str) -> type:
    key = (module, name)
    existing = _PLACEHOLDERS.get(key)
    if existing is not None:
        return existing

    def __init__(self, *args, **kwargs):  # noqa: N807 - pickle protocol
        pass

    def __setstate__(self, state):
        # Absorb whatever the snapshot held. Keeping the raw dict costs a few
        # bytes and makes the loss inspectable if a module ever comes back.
        self._orphaned_state = state

    def __reduce__(self):
        return (_placeholder_instance, (module, name))

    cls = type(
        f"_Missing{module.replace('.', '_').title()}_{name}",
        (object,),
        {
            "__init__": __init__,
            "__setstate__": __setstate__,
            "__reduce__": __reduce__,
            "__repr__": lambda self: f"<dropped {module}.{name}>",
        },
    )
    _PLACEHOLDERS[key] = cls
    return cls


def _placeholder_instance(module: str, name: str):
    """Module-level factory so a re-saved placeholder rebuilds as an INSTANCE.

    Returning the class itself (a common slip) would make every re-save promote
    the field to a class object, and the second save would then fail outright.
    """
    return _placeholder_class(module, name)()


def ravana_unpickler(f) -> pickle.Unpickler:
    """Build an Unpickler that degrades unknown ravana classes instead of dying.

    Handles the three historical module-name aliases (``ravana_chat``,
    ``scripts.ravana_chat``, ``__main__``) that snapshots saved from a direct
    script run still carry.
    """
    dropped: List[Tuple[str, str]] = []

    class _RavanaUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, AttributeError, ImportError):
                # Legacy alias for the CLI entry module.
                if module in ("ravana_chat", "scripts.ravana_chat", "__main__"):
                    for alias in ("scripts.ravana_chat", "ravana_chat"):
                        try:
                            return super().find_class(alias, name)
                        except (ModuleNotFoundError, AttributeError, ImportError):
                            continue
                if _is_tolerated(module):
                    dropped.append((module, name))
                    return _placeholder_class(module, name)
                raise

        def load(self):
            global LAST_DROPPED_CLASSES
            try:
                return super().load()
            finally:
                LAST_DROPPED_CLASSES = list(dropped)

    return _RavanaUnpickler(f)


def report_dropped(prefix: str = "  ") -> str:
    """One-line, human-readable report of classes this load could not resolve."""
    if not LAST_DROPPED_CLASSES:
        return ""
    names = ", ".join(f"{m}.{n}" for m, n in LAST_DROPPED_CLASSES)
    return (f"{prefix}[Load degraded] {len(LAST_DROPPED_CLASSES)} saved class(es) "
            f"absent from this build: {names}. The rest of the self was restored; "
            f"those components start from defaults and are rebuilt as they are used.")
