"""CI gate: no RAVANA capability may fail to import *silently*.

Background
---------
RAVANA wraps optional imports in try/except and sets a ``_HAS_*`` / ``HAS_*``
flag. When the import fails the code did not crash and did not log — the
capability simply stopped existing. Two live instances shipped undetected
(threadpoolctl, python-dateutil). The fix in this branch makes every guard LOG
on failure (see ``ravana._import_guard``); this test makes a *silent* failure
impossible to ship by asserting, on every CI run:

  1. Every capability flag ``ravana.chat.engine`` exposes is True, EXCEPT the
     small, documented allowlist of genuinely-optional modules (bs4,
     trafilatura). The flag set is discovered from the engine at runtime, so a
     NEW guard that silently trips is caught automatically — nothing to
     hardcode or keep in sync by hand.

  2. Each REQUIRED dependency (third-party hard deps + RAVANA's own internal
     modules) imports directly with NO try/except. A missing required module
     raises ImportError here and the test goes RED.

This is a red-capable gate: see the report for the red->green demonstration
(run with dateutil blocked).

The allowlist of intentionally-optional modules is legitimate config — keep it
small, explicit and commented. Do NOT add to it to make a genuinely-required
capability "pass"; classify honestly instead.
"""
import os
import re
import sys

import pytest

pytestmark = pytest.mark.ci

import ravana.chat.engine as _engine  # noqa: E402

# Genuinely-optional capabilities whose absence is intended (heavy/extra
# feature). Everything else on the engine MUST be present.
OPTIONAL_FLAG_ALLOWLIST = {
    "HAS_BS4",          # BeautifulSoup HTML parsing for web scraping (bs4)
    "HAS_TRAFILATURA",  # structured web extraction (trafilatura)
}

# Required third-party hard dependencies (declared in pyproject
# [project].dependencies). A missing one must fail CI loudly.
REQUIRED_THIRD_PARTY = [
    "threadpoolctl",  # OpenBLAS/MKL thread-pool pin (numpy #27989 AV guard)
    "dateutil",       # relative/ordinal date grounding
    "numpy",          # vector math everywhere
    "scipy",          # sparse graph storage
]

# RAVANA's own internal modules. A failed import here is ALWAYS a bug, never an
# optional feature — the engine must not boot "fine" with a dumber pipeline.
REQUIRED_INTERNAL_MODULES = [
    "ravana.chat.harm_intent_gate",
    "ravana.chat.support_router",
    "ravana.chat.consistency_monitor",
    "ravana.chat.snippet_quality",
    "ravana.chat.salad_classifier",
    "ravana.chat.snippet_pe_config",
    "ravana.chat.pos_model",
    "ravana.chat.intent_router",
    "ravana.chat.functional_lexicon",
]


def _engine_capability_flags():
    return [a for a in dir(_engine)
            if re.match(r"^(_HAS_|HAS_)[A-Z_]+$", a)]


def test_every_capability_flag_true_except_allowlist():
    """Source-derived: assert every engine capability flag is True but the
    documented optional allowlist. New guards are included automatically."""
    flags = _engine_capability_flags()
    assert flags, "engine exposes no capability flags — scan broken"
    failed = []
    for f in flags:
        if f in OPTIONAL_FLAG_ALLOWLIST:
            continue
        if getattr(_engine, f) is not True:
            failed.append(f)
    assert not failed, (
        "Silent capability loss detected — these engine flags are False: %s. "
        "A guarded import failed without failing CI; fix the missing module "
        "or classify it as genuinely optional in OPTIONAL_FLAG_ALLOWLIST."
        % ", ".join(sorted(failed))
    )


@pytest.mark.parametrize("mod", REQUIRED_THIRD_PARTY)
def test_required_third_party_importable(mod):
    """Each required third-party dep imports directly (no try/except)."""
    __import__(mod)


@pytest.mark.parametrize("mod", REQUIRED_INTERNAL_MODULES)
def test_required_internal_modules_importable(mod):
    """Each RAVANA internal module imports directly (no try/except)."""
    __import__(mod)


def test_dateutil_grounding_enabled():
    """The dateutil-gated relative-date grounding must be live."""
    from ravana.core import temporal_grounding
    assert temporal_grounding._HAVE_DATEUTIL is True, (
        "python-dateutil not importable — all relative-date grounding is dead"
    )


def test_analogy_relation_predictor_resolves():
    """Regression for a silently-broken internal import: RelationPredictor was
    imported from a module that never defined it (failed silently for months).
    It must now resolve to the real RAVANA class."""
    from ravana.core import analogy_engine
    assert analogy_engine.RelationPredictor is not None, (
        "RelationPredictor failed to import — analogy engine is dead"
    )
    assert analogy_engine.RelationPredictor.__name__ == "RelationPredictor"
