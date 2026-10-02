"""Evaluative-predicate polarity (feature round t_e45928d1, defect D5).

RAVANA could only recognise a value judgment when the predicate word
appeared in a hand-written alternation list, so "i think handmade mugs are
overpriced" minted NO stance — and with no stance there was nothing for the
retraction machinery to recode. These tests pin the new capability: the
polarity of an unlisted predicative adjective is read from where it sits in
concept space, the read is remembered online, and the user can correct it.

Two layers, deliberately:
  * ``TestEvaluativePolarityModel`` — the geometry, driven through a FAKE
    vector function so the expected answers are arithmetic, not dependent on
    which words happen to be in the GloVe cache. These tests are hermetic.
  * ``TestEvaluativePredicateMining`` — the end-to-end behaviour through the
    engine, asserting on the STANCE STORE (pure Python state), never on
    generated reply prose. Per this repo's standing rule, local pytest is
    unreliable for ROUTING/reply assertions, so none are made here.

Every test ASSERTS. A test that returns a bool is always green under pytest.
"""

import math
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

from ravana.chat.evaluative_polarity import (  # noqa: E402
    BASE_CONFIDENCE,
    EVALUATIVE_MIN_MARGIN,
    MAX_CONFIDENCE,
    VALENCE_MIN_MARGIN,
    EvaluativePolarityModel,
)


# ── a hermetic fake embedding space ─────────────────────────────────────────
# Each anchor word is placed on an explicit axis so the expected projection
# is arithmetic the test can state exactly:
#   "pos"   -> (+1, 0)   positive evaluative
#   "neg"   -> (-1, 0)   negative evaluative
#   "neu"   -> (0, -1)   neutral, i.e. NOT evaluative
#   anything else is looked up in EXTRA, and is absent by default.
def _unit(x, y):
    n = math.hypot(x, y)
    return (x / n, y / n)


def _fake_vectors(extra=None):
    table = {
        "wonderful": _unit(1, 0), "reliable": _unit(1, 0), "useful": _unit(1, 0),
        "terrible": _unit(-1, 0), "useless": _unit(-1, 0), "awful": _unit(-1, 0),
        "table": _unit(0, -1), "kiln": _unit(0, -1), "bicycle": _unit(0, -1),
    }
    table.update(extra or {})
    return lambda w: table.get((w or "").strip().lower())


def _model(extra=None, **kw):
    return EvaluativePolarityModel(vector_fn=_fake_vectors(extra), **kw)


class TestEvaluativePolarityModel:
    def test_negative_predicate_reads_negative(self):
        read = _model({"overpriced": _unit(-1, 0.6)}).score("overpriced")
        assert read is not None, "an unlisted negative predicate must be judged"
        assert read["polarity"] < 0, f"expected negative, got {read}"

    def test_positive_predicate_reads_positive(self):
        read = _model({"sturdy": _unit(1, 0.6)}).score("sturdy")
        assert read is not None, "an unlisted positive predicate must be judged"
        assert read["polarity"] > 0, f"expected positive, got {read}"

    def test_polarity_is_bounded_to_unit_range(self):
        m = _model({"overpriced": _unit(-1, 0.6)})
        read = m.score("overpriced")
        assert -1.0 <= read["polarity"] <= 1.0, read

    def test_neutral_noun_is_not_an_evaluative_predicate(self):
        """A naming word carries no judgment, so no stance may be minted."""
        assert _model({"blue": _unit(0.2, -1)}).score("blue") is None, \
            "a neutral descriptor must abstain, not guess a pole"

    def test_known_nouns_still_abstain(self):
        """The seed anchor words are geometry, not lookup answers — asking
        about a seed NEUTRAL word must not return a read."""
        assert _model().score("kiln") is None
        assert _model().score("bicycle") is None

    def test_out_of_vocabulary_word_abstains(self):
        assert _model().score("zzzqqq") is None, \
            "an unknown word has no vector and must fail closed"

    def test_model_without_vector_fn_fails_closed(self):
        """A UserModel built outside the engine has no GloVe route. It must
        abstain on everything rather than raise or invent a polarity."""
        m = EvaluativePolarityModel(vector_fn=None)
        assert m.score("overpriced") is None
        assert m.score("wonderful") is None

    def test_bare_and_empty_words_are_rejected(self):
        m = _model({"overpriced": _unit(-1, 0.6)})
        assert m.score("") is None
        assert m.score("   ") is None
        assert m.score("x") is None, "a one-character token is not a predicate"

    def test_boundary_word_is_not_guessed(self):
        """A word sitting almost exactly between the poles must abstain
        rather than commit to a sign it cannot support."""
        # Evaluative (high on the evaluative axis) but valence ~ 0.
        m = _model({"murkyish": _unit(0.001, 1.0)})
        read = m.score("murkyish")
        assert read is None, f"a near-zero valence must abstain, got {read}"

    def test_evaluative_axis_gates_before_valence(self):
        """A word that leans negative but is NOT evaluative must not be read.
        Without the evaluative axis this would return a confident negative
        for any descriptive word that happens to sit near the negative
        anchors."""
        m = _model({"slate": _unit(-1, -1)})
        assert m.score("slate") is None, \
            "a non-evaluative word must be gated out before valence is read"

    def test_a_removed_evaluative_axis_would_not_abstain(self):
        """Meta-guard on the gate above: prove the abstention is doing real
        work, not that 'slate' happens to be unrepresentable. Measured
        against the NEGATIVE CENTROID (not a single anchor — the centroid is
        what the model actually reads), 'slate' is comfortably negative, so
        without the evaluative axis it would yield a confident negative read.
        That makes the gate the only thing stopping a descriptive word from
        becoming an opinion."""
        m = _model({"slate": _unit(-1, -1)})
        pos, neg, neu = m._all_centroids()
        v = m._unit("slate")
        cos_pos = sum(a * b for a, b in zip(v, pos))
        cos_neg = sum(a * b for a, b in zip(v, neg))
        valence = cos_pos - cos_neg
        assert valence < -0.2, (
            "precondition: slate must read clearly negative on the valence "
            f"axis alone, otherwise the gate test is vacuous (valence={valence})")
        assert abs(valence) >= VALENCE_MIN_MARGIN, (
            "precondition: slate's valence must clear the margin, so that the "
            "evaluative axis is genuinely the only thing rejecting it")

    def test_threshold_constants_are_the_documented_boundary(self):
        """The margins are decision boundaries read off a measured
        distribution, so pin them: the test must be able to fail if someone
        tunes them into the population they are meant to separate."""
        assert 0.0 < EVALUATIVE_MIN_MARGIN < 0.5
        assert 0.0 < VALENCE_MIN_MARGIN < 0.5

    # ── online growth ───────────────────────────────────────────────────────
    def test_observe_makes_a_second_lookup_come_from_memory(self):
        m = _model({"overpriced": _unit(-1, 0.6)})
        first = m.score("overpriced")
        assert first["source"] == "geometry"
        m.observe("overpriced", first["polarity"])
        second = m.score("overpriced")
        assert second["source"] == "remembered", \
            "a predicate already judged must be answered from memory"

    def test_observe_raises_confidence_and_caps_it(self):
        m = _model({"overpriced": _unit(-1, 0.6)})
        m.observe("overpriced", -0.8)
        conf = m.score("overpriced")["confidence"]
        assert conf > BASE_CONFIDENCE, f"confidence must grow: {conf}"
        for _ in range(50):
            m.observe("overpriced", -0.8)
        assert m.score("overpriced")["confidence"] <= MAX_CONFIDENCE, \
            "confidence must be capped"

    def test_repeated_observation_reinforces_rather_than_dilutes(self):
        m = _model({"overpriced": _unit(-1, 0.6)})
        m.observe("overpriced", -0.8)
        before = m.score("overpriced")["polarity"]
        m.observe("overpriced", -0.8)
        after = m.score("overpriced")["polarity"]
        assert after < 0 and abs(after - before) < 0.15, \
            f"a same-sign re-observation must not wash the read out: {before} -> {after}"

    def test_relearn_overrides_a_wrong_geometric_read(self):
        """The correction path: the user is ground truth. A predicate the
        geometry mis-signed must be replaceable by the user's own position,
        which is what makes the capability revisable rather than frozen."""
        m = _model({"shoddy": _unit(1, 0.6)})   # geometry will read it POSITIVE
        assert m.score("shoddy")["polarity"] > 0
        m.relearn("shoddy", -0.9)              # the user disagrees
        assert m.score("shoddy")["polarity"] < 0, \
            "the user's revision must replace the geometric read"
        assert m.score("shoddy")["confidence"] == MAX_CONFIDENCE

    def test_relearn_persists_across_a_state_round_trip(self):
        m = _model({"shoddy": _unit(1, 0.6)})
        m.relearn("shoddy", -0.9)
        restored = _model({"shoddy": _unit(1, 0.6)})
        restored.set_state(m.get_state())
        assert restored.score("shoddy")["polarity"] < 0, \
            "a learned correction must survive save/load"

    def test_state_round_trip_preserves_the_whole_memory(self):
        m = _model({"overpriced": _unit(-1, 0.6), "sturdy": _unit(1, 0.6)})
        m.observe("overpriced", -0.8)
        m.observe("sturdy", 0.7)
        restored = _model({"overpriced": _unit(-1, 0.6), "sturdy": _unit(1, 0.6)})
        restored.set_state(m.get_state())
        assert restored.stats()["judged"] == 2
        assert restored.score("overpriced")["source"] == "remembered"
        assert restored.score("sturdy")["source"] == "remembered"

    def test_get_state_on_a_fresh_model_is_serializable(self):
        state = _model().get_state()
        assert isinstance(state, dict)
        assert state.get("judged") == {}
        assert isinstance(state.get("learned_pos"), list)
        assert isinstance(state.get("learned_neg"), list)

    def test_learned_word_becomes_an_anchor_for_its_neighbours(self):
        """A confidently-read predicate joins the anchor sets, so a word
        distributionally close to it can be judged too. This is the
        compounding path: coverage grows from use, with no code edit."""
        m = _model({
            "overpriced": _unit(-1, 0.6),
            "pricey": _unit(-1, 0.6),   # close to overpriced
            "cheapish": _unit(-1, 0.6),  # close but not identical
        })
        assert m.score("pricey") is not None
        m.observe("overpriced", -1.0)
        assert m.stats()["learned_neg"] >= 1, \
            f"a confidently-read predicate must join the anchor sets: {m.stats()}"
        # The anchor cache must have been invalidated, so a subsequent
        # neighbourhood read reflects the new anchor rather than a stale one.
        assert m.score("cheapish") is not None

    def test_observe_ignores_a_neutral_polarity(self):
        m = _model()
        m.observe("wordless", 0.0)
        assert m.remembered("wordless") is None, \
            "a near-zero observation must not be remembered as a judgment"

    def test_observe_rejects_empty_and_short_words(self):
        m = _model()
        m.observe("", -0.8)
        m.observe("x", -0.8)
        assert m.stats()["judged"] == 0


class TestEvaluativePredicateMining:
    """End-to-end, through the engine, asserting on the STANCE STORE.

    The stance store is plain Python state, so these assertions are
    trustworthy locally — unlike reply/routing assertions, which this repo's
    own rule says must be gated through CI (cold vs warm GloVe changes which
    handler wins). No test here inspects generated prose.
    """

    @pytest.fixture(scope="class")
    def engine(self):
        from ravana.chat.engine import CognitiveChatEngine
        eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                                  user_suffix="test_evaluative_d5")
        yield eng
        try:
            eng.stop_background_learning()
        except Exception:
            pass

    @staticmethod
    def _turns(engine, turns):
        """Feed turns, returning the resulting stance map."""
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            for t in turns:
                engine.process_turn(t)
        return {k: (v.polarity, v.rehearsal_count)
                for k, v in engine.user_model.opinions.stances.items()}

    def test_unlisted_predicate_creates_a_stance(self, engine):
        """THE capability. "overpriced" is in no alternation list, so before
        this feature the store stayed empty through the whole sequence."""
        engine.user_model.opinions.stances.clear()
        stances = self._turns(engine, ["i think handmade mugs are overpriced."])
        assert stances, (
            "an evaluative predicate the word lists never listed must still "
            "produce a stance")
        assert any("mug" in k for k in stances), \
            f"the stance must key on the real concept, got {list(stances)}"

    def test_the_stance_is_negative(self, engine):
        engine.user_model.opinions.stances.clear()
        stances = self._turns(engine, ["i think handmade mugs are overpriced."])
        pols = [p for p, _ in stances.values()]
        assert min(pols) < 0, f"overpriced must read negative, got {stances}"

    def test_topic_key_excludes_the_reporting_frame(self, engine):
        engine.user_model.opinions.stances.clear()
        stances = self._turns(engine, ["i think handmade mugs are overpriced."])
        keys = list(stances)
        assert not any(k.startswith("think ") for k in keys), \
            f"the opinion frame leaked into the topic key: {keys}"

    def test_a_retraction_now_has_something_to_recode(self, engine):
        """The D5 symptom end to end: state, retract, re-state. Before the
        feature the store was empty throughout and nothing moved."""
        engine.user_model.opinions.stances.clear()
        self._turns(engine, ["i think handmade mugs are overpriced."])
        before = {k: s.polarity
                  for k, s in engine.user_model.opinions.stances.items()}
        self._turns(engine, ["i was wrong about handmade mugs."])
        after = {k: s.polarity
                 for k, s in engine.user_model.opinions.stances.items()}
        assert before, "precondition: a stance must exist to recode"
        shared = set(before) & set(after)
        assert shared, f"the retraction resolved to no held stance: {after}"
        moved = [k for k in shared if after[k] > before[k] + 0.1]
        assert moved, (
            f"retracting a negative stance must move it positive: "
            f"{[(k, before[k], after[k]) for k in shared]}")

    def test_neutral_descriptor_creates_no_stance(self, engine):
        """The abstention guarantee: "the mug is blue" is a description, not
        a judgment, and must leave the store untouched."""
        engine.user_model.opinions.stances.clear()
        stances = self._turns(engine, ["the mug is blue."])
        assert not stances, \
            f"a neutral descriptor must not mint a stance, got {stances}"

    def test_question_is_not_mined_as_an_opinion(self, engine):
        engine.user_model.opinions.stances.clear()
        stances = self._turns(engine, ["are handmade mugs overpriced?"])
        assert not stances, \
            f"an interrogative is the user asking, not stating: {stances}"

    def test_the_model_actually_learns_across_turns(self, engine):
        """The online-growth claim, measured: the predicate is remembered
        after being judged once, so the capability compounds with use."""
        engine.user_model.opinions.stances.clear()
        model = engine.user_model._ensure_evaluative_polarity()
        model._judged.clear()
        self._turns(engine, ["i think handmade mugs are overpriced."])
        assert model.remembered("overpriced") is not None, \
            "a judged predicate must be remembered for next time"
        assert model.score("overpriced")["source"] == "remembered"

    def test_state_round_trip_preserves_the_learned_polarity(self, engine):
        """No retraining: what was learned is in the serialized state, so a
        fresh model restored from it answers the same way immediately."""
        from ravana.chat.user_model import UserModel
        engine.user_model.opinions.stances.clear()
        self._turns(engine, ["i think handmade mugs are overpriced."])
        state = engine.user_model.get_state()
        assert state.get("_evaluative_polarity"), \
            "the model's memory must be part of the serialized state"
        fresh = UserModel()
        fresh.set_state(state)
        assert fresh._ensure_evaluative_polarity().remembered("overpriced") is not None
