"""REGRESSION: the four engine->user_model bindings must survive a reload.

DOCUMENTED DEFECT (see docs/CAPABILITY_EVALUATIVE_PREDICATE_POLARITY.md section 5)
-----------------------------------------------------------------------------------
engine.py:1087 creates ``self.user_model = UserModel()`` and then injects four
live engine bindings into THAT instance:

    engine.py:1113   _episodic_index
    engine.py:1114   _episodic_transcript
    engine.py:1120   _concept_vocab
    engine.py:1129   _glove_vector_fn

``__init__`` later auto-loads whenever a save file exists
(``engine.py:1971-1972``), and ``_load()`` REPLACES the whole object
(``engine.py:11382``, and again ``engine.py:11410`` when the dedicated
user_models/ file wins). Nothing re-applies the four bindings. A grep over the
whole ``_load()`` body (engine.py:10958-11687) finds zero re-assignments.

Measured impact: the evaluative-predicate capability is geometry-ALIVE in a
first session (5/6 unseen words judged) and geometry-DEAD in every later one
(0/6, and end-to-end mining mints ZERO stances).

WHY A NAIVE TEST WOULD PASS (the trap this file is built around)
----------------------------------------------------------------
Two distinct shapes were observed, and BOTH fail closed:

  1. the attribute is absent  -> vector_fn=None -> every lookup abstains;
  2. the attribute survives the pickle as a **STRING** (a bound method cannot
     be pickled) -> ``model._vector_fn`` is TRUTHY, so ``assert fn is not None``
     PASSES, but calling it raises ``TypeError: 'str' object is not callable``,
     which ``score()`` swallows at evaluative_polarity.py:155.

So these tests deliberately do NOT assert "the attribute exists". They assert
that an UNSEEN word -- one the seed lexicon has never judged, so the memory path
cannot answer it -- produces a real geometric read, and that the injected
callable is actually callable.

Every test ASSERTS. A test that returns a bool is always green under pytest.

RAVANA_OFFLINE=1 .venv-real/Scripts/python.exe -m pytest \
    tests/unit/test_reload_bindings_survive.py -q

CURRENT STATUS: xfail(strict=False), measured 2026-10-03 -- 5 failed, 4 passed.
The defect is REAL and UNFIXED; fixing it is a code change to engine.py, which
is out of scope for a docs card. These are landed as xfail rather than deleted
so the day someone re-applies the four bindings after _load(), pytest reports
XPASS and the marker is visibly wrong. The doc page is the authoritative
record until then.
"""

import os
import sys

import pytest

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Words in NO seed lexicon, so `score()` cannot answer them from `judged`
# memory -- only from geometry.
UNSEEN = ["dreadful", "sturdy", "wretched", "clumsy", "mediocre"]

BINDINGS = ("_glove_vector_fn", "_concept_vocab", "_episodic_index",
            "_episodic_transcript")

# The defect is documented and real, but fixing it means editing engine.py's
# _load() -- a code change, not a docs change, so this card does not make it.
# non-strict so the suite stays green in CI while remaining a live tripwire:
# the day the four bindings are re-applied after load(), these report XPASS and
# the marker is visibly wrong instead of silently becoming a dead test.
_KNOWN = pytest.mark.xfail(
    reason="engine._load() reassigns self.user_model without re-applying the "
           "four engine bindings (engine.py:11382/11410); documented in "
           "docs/CAPABILITY_EVALUATIVE_PREDICATE_POLARITY.md section 5",
    strict=False,
)


def _engine(suffix):
    from ravana.chat.engine import CognitiveChatEngine
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)


@pytest.fixture(scope="module")
def after_reload():
    """Boot, save, then boot again so __init__ takes the auto-load branch.

    The auto-load branch is the ordinary path for every session after the
    first, so this is the shape that matters.
    """
    suffix = "test_reload_bindings"
    first = _engine(suffix)
    try:
        first.save()
    finally:
        try:
            first.stop_background_learning()
        except Exception:
            pass
        del first
    second = _engine(suffix)
    yield second
    try:
        second.stop_background_learning()
    except Exception:
        pass


class TestBindingsSurviveReload:
    @_KNOWN
    def test_glove_vector_fn_is_actually_callable(self, after_reload):
        """Catches shape 2 -- a STRINGIFIED binding is truthy but not callable.

        This is the assertion that a naive `is not None` check would miss,
        and it is the one that actually failed in the field.
        """
        fn = getattr(after_reload.user_model, "_glove_vector_fn", None)
        assert fn is not None, \
            "_glove_vector_fn was dropped by the reload (engine.py:11382/11410)"
        assert callable(fn), (
            f"_glove_vector_fn survived the reload as {type(fn).__name__}, "
            f"not a callable: a bound method cannot be pickled, so the "
            f"binding must be re-applied after user_model is reassigned"
        )

    @_KNOWN
    def test_glove_vector_fn_returns_a_real_vector(self, after_reload):
        """A callable that raises is still broken. Demand an actual vector."""
        fn = after_reload.user_model._glove_vector_fn
        vec = fn("overpriced")
        assert vec is not None, \
            "_glove_vector_fn returned None for an in-vocabulary word"
        assert len(vec) > 0, \
            f"_glove_vector_fn returned an empty vector: {vec!r}"

    @_KNOWN
    def test_binding_points_at_the_live_engine(self, after_reload):
        """A stale owner means reads go to a dead projection."""
        fn = after_reload.user_model._glove_vector_fn
        owner = getattr(fn, "__self__", None)
        assert owner is after_reload, (
            "_glove_vector_fn is bound to a DIFFERENT engine instance "
            f"(id {id(owner) if owner is not None else None} vs the live "
            f"{id(after_reload)}); the binding must be re-applied after load()"
        )

    @pytest.mark.parametrize("attr", BINDINGS)
    def test_each_binding_is_present_and_live(self, after_reload, attr):
        val = getattr(after_reload.user_model, attr, None)
        assert val is not None, (
            f"user_model.{attr} was dropped by the reload; engine.py:1113-1129 "
            f"injects it but _load() reassigns self.user_model without "
            f"re-applying it"
        )

    @_KNOWN
    def test_evaluative_geometry_survives_the_reload(self, after_reload):
        """THE end-to-end assertion: an UNSEEN word must still be judged.

        Probing an unseen word is what makes this honest -- `overpriced`
        would be answered from `judged` memory, which needs no vector_fn at
        all, and would pass even with the capability completely dead.
        """
        model = after_reload.user_model._ensure_evaluative_polarity()
        judged = model.stats()["judged"]
        for w in UNSEEN:
            pre = model.remembered(w)
            assert pre is None, (
                f"probe word {w!r} is already in memory (judged={judged}); it "
                f"cannot test the geometry path -- pick a genuinely unseen word"
            )
        answered = [w for w in UNSEEN if model.score(w) is not None]
        assert answered, (
            "after a reload the evaluative-predicate model abstained on ALL "
            f"{len(UNSEEN)} unseen words -- the geometry path is dead because "
            "the engine->user_model bindings are not re-applied after load()"
        )

    @_KNOWN
    def test_mining_mints_a_stance_after_the_reload(self, after_reload):
        """THE user-visible symptom: zero stances mined post-reload."""
        import contextlib
        import io

        um = after_reload.user_model
        before = set(um.opinions.stances)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            after_reload.process_turn("i think the harbour ferry is dreadful")
            after_reload.process_turn("i think my old bike is sturdy")
        new = set(um.opinions.stances) - before
        assert new, (
            "no stance was mined after the reload: the evaluative-predicate "
            "miner abstains because _glove_vector_fn was dropped, so an "
            "unlisted judgment like 'dreadful' never reaches the store"
        )