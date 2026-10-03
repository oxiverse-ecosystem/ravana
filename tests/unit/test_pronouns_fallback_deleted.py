"""Pin the deletion of SurfaceRealizer.PRONOUNS_FALLBACK (task 13).

Why this guard exists
---------------------
`SurfaceRealizer.PRONOUNS_FALLBACK` was a 7-entry identity map
{"i":"i", "you":"you", "we":"we", "they":"they", "he":"he", "she":"she", "it":"it"}.
Two independent properties made it unreachable dead code:

1. `_resolve_pronoun` already returned `subject` VERBATIM three lines earlier
   for exactly that 7-tuple, so the map lookup could never fire for any key
   the map held.
2. Every value equalled its own key, so even a reachable lookup would have
   returned the input unchanged.

This module asserts the symbol is GONE (the de-hardcoding remedy: a diff that
is net-negative on prose) and that the guard that replaced it still returns the
subject verbatim. If the identity map is ever reintroduced, these tests go red.

No authored reply strings are introduced here -- this is a structural guard.
"""

import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_ROOT,
           os.path.join(_ROOT, "ravana", "src"),
           os.path.join(_ROOT, "ravana_ml", "src"),
           os.path.join(_ROOT, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.language.surface_realizer import SurfaceRealizer  # noqa: E402


def test_pronouns_fallback_symbol_is_deleted():
    """The identity map must not come back as a class attribute."""
    assert not hasattr(SurfaceRealizer, "PRONOUNS_FALLBACK"), (
        "SurfaceRealizer.PRONOUNS_FALLBACK was reintroduced. It was a 7-entry "
        "identity map whose every value equalled its own key, and whose keys "
        "were already returned verbatim by the guard earlier in "
        "_resolve_pronoun, so it was unreachable dead code."
    )


def test_source_file_has_no_pronouns_fallback_reference():
    """No live reference survives anywhere in the realizer source."""
    import ravana.language.surface_realizer as mod
    src_path = mod.__file__
    with open(src_path, "r", encoding="utf-8") as fh:
        src = fh.read()
    assert "PRONOUNS_FALLBACK" not in src, (
        "a PRONOUNS_FALLBACK reference survived in %s" % src_path
    )


@pytest.mark.parametrize("pronoun", ["i", "you", "we", "they", "he", "she", "it"])
def test_first_person_guard_still_returns_subject_verbatim(pronoun):
    """Deleting the map must not disturb the real guard that replaced it.

    This is the behavioural contract the dead map used to shadow: a subject
    that IS a base pronoun is echoed unchanged, with no GloVe classifier call.
    """
    import inspect

    sr = SurfaceRealizer.__new__(SurfaceRealizer)  # no boot cost
    discourse = _FakeDiscourse()
    out = sr._resolve_pronoun(pronoun, pronoun, discourse)
    assert out == pronoun, (
        "guard at _resolve_pronoun must echo the base pronoun %r verbatim, got %r"
        % (pronoun, out)
    )


class _FakeDiscourse:
    """Minimal DiscourseState stand-in: _resolve_pronoun reads only repetitions."""

    subject_repetitions = 0


def test_resolve_pronoun_still_classifies_non_pronoun_subjects():
    """Deleting the map must not break the real (GloVe) path for real nouns.

    A non-pronoun subject must NOT be echoed verbatim when it is a repeated
    referent -- that is the branch the dead map sat above.
    """
    sr = SurfaceRealizer.__new__(SurfaceRealizer)
    # __init__ is bypassed for speed; _resolve_pronoun reads this set.
    sr._used_subjects = set()

    class _Repeated:
        subject_repetitions = 3

    out = sr._resolve_pronoun("gravity", "gravity", _Repeated())
    assert isinstance(out, str) and out
    assert out != "gravity", (
        "a repeated non-pronoun subject must resolve to a pronoun via the "
        "classifier, not be echoed verbatim; got %r" % out
    )