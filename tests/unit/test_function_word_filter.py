"""The function-word filter must actually filter.

`_is_function_word` is the documented safety net under the learned POS tagger:
a word in the closed-class set is a function word regardless of what the
distributional tagger says. It had been completely inert.

Two causes, both silent:
  * `_closed_class("grammatical_concepts")` resolved to an EMPTY set. The
    canonical set lives on the engine as an INSTANCE attribute
    (`self._GRAMMATICAL_CONCEPTS`), while the lookup only checked class
    attributes -- so it fell through to a module-level load.
  * `FunctionalLexicon.grammatical_concepts()` raised `KeyError` (the seed
    table has no such key), which `_closed_class` swallowed.

With `use_learned_pos=True` (the default) the net was therefore `_w in set()` --
always False. Filler survived the coherence gate and was spliced into hedge
frames as though it were content:

    "the closest i get to time is only and world what's your take on it?"

These tests pin the filter and the output it protects.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana_ml", "src"))


@pytest.fixture(scope="module")
def engine():
    import tempfile
    from ravana.chat.engine import CognitiveChatEngine
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               data_dir=tempfile.mkdtemp(prefix="ravana_fw_"),
                               user_suffix="_funcword")


# ── the filter itself ────────────────────────────────────────────────────────

def test_closed_class_resolves_to_the_engine_set(engine):
    """The lookup must reach the canonical set, not an empty fallback."""
    resolved = engine._closed_class("grammatical_concepts")
    assert resolved, "_closed_class('grammatical_concepts') resolved empty"
    assert "only" in resolved
    assert "really" in resolved or "the" in resolved


def test_function_words_are_detected_under_learned_pos(engine):
    """`only` is in the closed-class set; the tagger calls it a noun. The
    safety net must win, or the filter is decorative."""
    if not getattr(engine, "use_learned_pos", False):
        pytest.skip("learned-POS path disabled")
    assert engine._concept_pos.get("only") == "noun", (
        "precondition: the tagger mislabels 'only' as a noun")
    assert engine._is_function_word("only") is True


def test_content_words_are_not_filtered(engine):
    """The fix must not over-block: real concepts stay available."""
    for w in ("ocean", "gravity", "time", "photosynthesis", "quasar"):
        assert engine._is_function_word(w) is False, w


def test_grammatical_concepts_accessor_does_not_raise():
    """A missing seed key must yield an empty set, not a KeyError that the
    caller silently swallows."""
    from ravana.chat.functional_lexicon import FunctionalLexicon
    fl = FunctionalLexicon()
    assert isinstance(fl.grammatical_concepts, set)


def test_closed_class_reaches_the_lexicon(engine):
    """`_closed_class` must handle BOTH accessor shapes.

    Every FunctionalLexicon accessor is a @property, but the old code did
    `getattr(_fl, name)()` -- calling the returned set -- which raises
    `TypeError: 'set' object is not callable`. The bare except swallowed it, so
    this branch NEVER returned anything and every lexicon-backed filter in the
    engine was silently empty.
    """
    for name in ("subject_context", "conditional_frame", "attr_words",
                 "grammatical_concepts", "leading_modifiers",
                 "trailing_modifiers"):
        resolved = engine._closed_class(name)
        assert resolved, "%s resolved empty" % name


def test_closed_class_handles_method_shaped_accessors():
    """A method-shaped accessor must still work -- the old code assumed
    methods, which is why the property-shaped majority broke."""
    from ravana.chat.engine import CognitiveChatEngine

    class _Fl:
        def grammatical_concepts(self):
            return {"the", "a"}

        @property
        def attr_words(self):
            return {"height", "weight"}

    eng = CognitiveChatEngine.__new__(CognitiveChatEngine)
    eng._func_lex = _Fl()
    assert eng._closed_class("grammatical_concepts") == {"the", "a"}
    assert eng._closed_class("attr_words") == {"height", "weight"}


def test_closed_class_falls_back_to_the_engine_set():
    """An engine holding the set as an INSTANCE attribute must still resolve
    it -- that is where _GRAMMATICAL_CONCEPTS lives."""
    from ravana.chat.engine import CognitiveChatEngine
    eng = CognitiveChatEngine.__new__(CognitiveChatEngine)
    eng._func_lex = None
    eng._GRAMMATICAL_CONCEPTS = {"the", "only"}
    assert eng._closed_class("grammatical_concepts") == {"the", "only"}


# ── the output it protects ───────────────────────────────────────────────────

def test_hedge_does_not_splice_filler(engine, monkeypatch):
    """Regression: the egregious case. Offline so the hedged-ignorance path
    (not a web answer) produces the reply."""
    monkeypatch.setenv("RAVANA_OFFLINE", "1")
    reply = engine.process_turn("explain time") or ""
    low = reply.lower()
    # the filler that used to be spliced in as if it were the gist
    assert "only and world" not in low
    assert "is only" not in low
