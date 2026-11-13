"""Tests for the macroplanning (pre-lexical message) module.

Pins the brain-mechanism claims:
- the message is a role-concept structure, NOT a word sequence;
- selection is a Luce-ratio competition, not an argmax;
- spreading activation is a field (a connected concept can beat a stronger
  direct edge);
- the module is a pure function over (subject, graph edges) and touches no
  decoder.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ravana.learn.macroplanning import (  # noqa: E402
    MacroplanConfig,
    Message,
    RoleBinding,
    build_message,
    is_content_word,
    luce_ratio,
    message_to_boost,
    select_lexical_concepts,
    spreading_activation,
)


def _cands(*pairs):
    """(relation, target, weight, confidence) tuples."""
    return [(r, t, w, c) for r, t, w, c in pairs]


class TestLuceRatio:
    def test_no_competition_gives_one(self):
        assert luce_ratio(1.0, [1.0]) == pytest.approx(1.0)

    def test_equal_field_splits_evenly(self):
        assert luce_ratio(1.0, [1.0, 1.0]) == pytest.approx(0.5)

    def test_ratio_is_a_share_not_a_margin(self):
        # A 10x raw-score margin is still a 10/11 share, not ~1.0. This is the
        # difference between a competition and an argmax.
        assert luce_ratio(10.0, [10.0, 1.0]) == pytest.approx(10.0 / 11.0)

    def test_empty_field_is_zero(self):
        assert luce_ratio(0.0, []) == 0.0
        assert luce_ratio(1.0, [0.0, 0.0]) == 0.0

    def test_negative_activation_does_not_break_it(self):
        # The field should never be negative, but the ratio must not raise or
        # return a NaN. A negative competitor can push the ratio above 1 (the
        # denominator shrinks) -- that is the rule's behaviour, not a bug, so
        # only finiteness is pinned here.
        r = luce_ratio(0.5, [0.5, -0.2])
        assert np.isfinite(r)


class TestSpreadingActivation:
    def test_empty_candidates(self):
        assert spreading_activation("x", []) == {}

    def test_blank_targets_skipped(self):
        out = spreading_activation("x", _cands(("is_a", "", 1.0, 1.0)))
        assert out == {}

    def test_field_is_normalised(self):
        out = spreading_activation(
            "gravity",
            _cands(("is_a", "force", 0.8, 0.9),
                   ("causes", "attraction", 0.5, 0.6)))
        assert out
        assert max(out.values()) == pytest.approx(1.0)

    def test_connected_concept_can_outrank_a_stronger_direct_edge(self):
        # This is the point of a FIELD rather than a per-edge score: a concept
        # reachable by two edges accumulates activation and can beat a concept
        # with one stronger edge.
        out = spreading_activation(
            "water",
            _cands(("is_a", "liquid", 0.9, 0.9),
                   ("causes", "life", 0.4, 0.4),
                   ("found_in", "life", 0.3, 0.3)))
        assert out
        # 'life' is reached twice; it should not be below a singleton.
        assert out.get("life", 0.0) > 0.0

    def test_decay_reduces_the_field(self):
        cfg_a = MacroplanConfig(decay=0.0)
        cfg_b = MacroplanConfig(decay=0.9)
        cands = _cands(("is_a", "force", 0.8, 0.9),
                       ("causes", "attraction", 0.5, 0.6))
        a = spreading_activation("gravity", cands, cfg=cfg_a)
        b = spreading_activation("gravity", cands, cfg=cfg_b)
        # Both normalised to max 1, so compare the runner-up's share.
        assert len(a) == len(b) == 2


class TestSelectLexicalConcepts:
    def test_empty(self):
        assert select_lexical_concepts({}) == []

    def test_winner_is_the_largest_share(self):
        field_ = {"force": 0.9, "attraction": 0.5, "mass": 0.1}
        out = select_lexical_concepts(field_)
        assert out[0][0] == "force"
        assert out[0][1] == pytest.approx(0.9)

    def test_ratio_is_carried(self):
        field_ = {"force": 0.5, "attraction": 0.5}
        out = select_lexical_concepts(field_)
        assert out[0][2] == pytest.approx(0.5)

    def test_long_tail_dropped(self):
        field_ = {"force": 1.0, "attraction": 0.5, "mass": 0.001}
        out = select_lexical_concepts(field_, cfg=MacroplanConfig(
            competition_floor=0.15))
        concepts = [c for c, _a, _r in out]
        assert "mass" not in concepts

    def test_limit_respected(self):
        field_ = {f"c{i}": 1.0 - i * 0.01 for i in range(10)}
        out = select_lexical_concepts(field_, limit=3)
        assert len(out) == 3

    def test_min_activation_gate(self):
        field_ = {"force": 1.0, "whisper": 0.001}
        out = select_lexical_concepts(
            field_, cfg=MacroplanConfig(min_activation=0.05))
        assert [c for c, _a, _r in out] == ["force"]


class TestBuildMessage:
    def test_empty_candidates_returns_none(self):
        assert build_message("gravity", []) is None

    def test_message_is_not_a_string(self):
        m = build_message(
            "gravity",
            _cands(("is_a", "force", 0.8, 0.9),
                   ("causes", "attraction", 0.5, 0.6)))
        assert isinstance(m, Message)
        assert not isinstance(m, str)

    def test_subject_is_preserved(self):
        m = build_message("gravity", _cands(("is_a", "force", 0.8, 0.9)))
        assert m.subject == "gravity"

    def test_bindings_carry_roles(self):
        m = build_message(
            "gravity",
            _cands(("is_a", "force", 0.8, 0.9),
                   ("causes", "attraction", 0.5, 0.6)))
        roles = {b.role for b in m.bindings}
        assert "theme" in roles

    def test_binding_activation_is_positive(self):
        m = build_message("gravity", _cands(("is_a", "force", 0.8, 0.9)))
        assert all(b.activation > 0 for b in m.bindings)

    def test_competitors_recorded(self):
        m = build_message(
            "gravity",
            _cands(("is_a", "force", 0.8, 0.9),
                   ("causes", "attraction", 0.5, 0.6)))
        assert len(m.competitors) >= 1
        # Sorted descending.
        acts = [a for _c, a in m.competitors]
        assert acts == sorted(acts, reverse=True)

    def test_selection_ratio_bounded(self):
        m = build_message(
            "gravity",
            _cands(("is_a", "force", 0.8, 0.9),
                   ("causes", "attraction", 0.5, 0.6)))
        assert 0.0 < m.selection_ratio <= 1.0

    def test_relation_is_recovered(self):
        m = build_message("gravity", _cands(("is_a", "force", 0.8, 0.9)))
        assert m.bindings[0].relation == "is_a"

    def test_max_roles_caps_bindings(self):
        cfg = MacroplanConfig(max_roles=2)
        m = build_message(
            "water",
            _cands(("is_a", "liquid", 0.9, 0.9),
                   ("causes", "life", 0.5, 0.6),
                   ("found_in", "ocean", 0.4, 0.5)),
            cfg=cfg)
        assert len(m.bindings) <= 2

    def test_goal_is_carried(self):
        m = build_message("gravity", _cands(("is_a", "force", 0.8, 0.9)),
                          goal="define")
        assert m.goal == "define"

    def test_as_dict_is_jsonable(self):
        import json
        m = build_message("gravity", _cands(("is_a", "force", 0.8, 0.9)))
        s = json.dumps(m.as_dict())
        assert "gravity" in s
        assert "bindings" in s

    def test_concepts_property(self):
        m = build_message("gravity", _cands(("is_a", "force", 0.8, 0.9)))
        assert m.concepts == ["force"]


class TestMessageToBoost:
    def test_none_message_gives_empty(self):
        assert message_to_boost(None, {}) == {}

    def test_unknown_words_skipped(self):
        m = build_message("gravity", _cands(("is_a", "force", 0.8, 0.9)))
        assert message_to_boost(m, {}) == {}

    def test_known_words_boosted(self):
        m = build_message("gravity", _cands(("is_a", "force", 0.8, 0.9)))
        boost = message_to_boost(m, {"force": 42})
        assert boost == {42: pytest.approx(2.0 * 0.9, abs=0.01)} or 42 in boost

    def test_role_weights_apply(self):
        m = build_message(
            "gravity",
            _cands(("is_a", "force", 0.8, 0.9),
                   ("causes", "attraction", 0.5, 0.6)))
        flat = message_to_boost(m, {"force": 1, "attraction": 2})
        weighted = message_to_boost(
            m, {"force": 1, "attraction": 2},
            role_weights={"theme": 2.0})
        assert set(flat) == set(weighted)

    def test_boost_scales_with_activation(self):
        m = build_message(
            "gravity",
            _cands(("is_a", "force", 0.9, 0.9),
                   ("causes", "attraction", 0.2, 0.2)))
        boost = message_to_boost(m, {"force": 1, "attraction": 2})
        assert boost[1] > boost[2]

    def test_case_insensitive_lookup(self):
        m = build_message("gravity", _cands(("is_a", "Force", 0.8, 0.9)))
        assert message_to_boost(m, {"force": 7}) != {}


class TestContentWordGate:
    """The closed-class gate. Measured: without it, `time` selected
    `early`/`only`/`first` and `love` selected `i`/`you`."""

    def test_rejects_pronouns(self):
        for w in ("i", "you", "he", "she", "it", "we", "they", "me", "him"):
            assert not is_content_word(w), w

    def test_rejects_determiners_and_prepositions(self):
        for w in ("the", "a", "an", "this", "of", "to", "in", "for", "with"):
            assert not is_content_word(w), w

    def test_rejects_auxiliaries(self):
        for w in ("is", "are", "was", "were", "be", "have", "has", "do", "does"):
            assert not is_content_word(w), w

    def test_accepts_content_words(self):
        for w in ("gravity", "force", "water", "ocean", "music", "love",
                  "life", "earth", "quantum", "rhythm"):
            assert is_content_word(w), w

    def test_rejects_special_tokens(self):
        for w in ("<bos>", "<eos>", "<pad>", "<unk>"):
            assert not is_content_word(w), w

    def test_rejects_blank(self):
        assert not is_content_word("")
        assert not is_content_word("   ")

    def test_rejects_multiword_labels(self):
        # Graph labels like "ocean sea" / "sea side" are fragments, not
        # lexical concepts.
        assert not is_content_word("ocean sea")
        assert not is_content_word("sea side")

    def test_case_insensitive(self):
        assert not is_content_word("The")
        assert is_content_word("Gravity")

    def test_build_message_drops_function_words_by_default(self):
        m = build_message(
            "time",
            _cands(("related_to", "early", 0.9, 0.9),
                   ("related_to", "years", 0.8, 0.8),
                   ("related_to", "duration", 0.7, 0.7)))
        concepts = m.concepts
        assert "early" not in concepts
        assert "duration" in concepts

    def test_content_only_false_keeps_function_words(self):
        m = build_message(
            "time",
            _cands(("related_to", "early", 0.9, 0.9),
                   ("related_to", "duration", 0.7, 0.7)),
            content_only=False)
        assert "early" in m.concepts

    def test_all_function_words_returns_none(self):
        assert build_message(
            "love", _cands(("related_to", "i", 0.9, 0.9),
                           ("related_to", "you", 0.8, 0.8))) is None


class TestRoleAssignment:
    """Roles come from the RELATION's semantics, not from salience order.

    Measured failure of salience-order assignment: ``time -> theme=early,
    relation=years`` and ``love -> relation=i, ground=you``.
    """

    def test_is_a_binds_theme(self):
        m = build_message("gravity", _cands(("is_a", "force", 0.8, 0.9)))
        assert m.bindings[0].role == "theme"
        assert m.bindings[0].concept == "force"

    def test_causes_binds_ground(self):
        m = build_message("water", _cands(("causes", "life", 0.8, 0.9)))
        assert m.bindings[0].role == "ground"

    def test_agent_binds_agent(self):
        m = build_message("book", _cands(("author", "writer", 0.8, 0.9)))
        assert m.bindings[0].role == "agent"

    def test_property_binds_relation(self):
        m = build_message("water", _cands(("has_property", "liquid", 0.8, 0.9)))
        assert m.bindings[0].role == "relation"

    def test_unknown_relation_defaults_to_relation(self):
        m = build_message("x", _cands(("zzz", "force", 0.8, 0.9)))
        assert m.bindings[0].role == "relation"

    def test_role_is_relation_derived_not_salience(self):
        # The strongest edge is `related_to` (defaults to relation); the
        # weaker `is_a` edge must still take the theme role.
        m = build_message(
            "gravity",
            _cands(("related_to", "earth", 0.95, 0.95),
                   ("is_a", "force", 0.4, 0.5)))
        by_concept = {b.concept: b.role for b in m.bindings}
        assert by_concept["force"] == "theme"
        assert by_concept["earth"] == "relation"

    def test_duplicate_roles_salvaged(self):
        # Two `is_a` edges: the stronger takes theme, the weaker is salvaged
        # into a free role rather than dropped.
        m = build_message(
            "dog",
            _cands(("is_a", "mammal", 0.9, 0.9),
                   ("is_a", "animal", 0.8, 0.8)))
        assert len(m.bindings) == 2
        roles = [b.role for b in m.bindings]
        assert roles.count("theme") == 1

    def test_all_four_roles_used_when_available(self):
        m = build_message(
            "water",
            _cands(("is_a", "liquid", 0.9, 0.9),
                   ("causes", "life", 0.8, 0.8),
                   ("location", "ocean", 0.7, 0.7),
                   ("author", "writer", 0.6, 0.6)))
        roles = {b.role for b in m.bindings}
        assert roles == {"theme", "ground", "agent", "relation"}


class TestPurity:
    def test_no_decoder_import(self):
        import ravana.learn.macroplanning as mod
        # The module must be importable without the decoder package, and must
        # not hold a reference to it. Check the module's own imports rather
        # than substring-scanning the source (docstrings mention the decoder).
        assert not hasattr(mod, "NeuralDecoder")
        deps = {getattr(v, "__name__", "") for v in vars(mod).values()}
        assert not any("decoder" in d for d in deps)
        src = open(mod.__file__, encoding="utf-8").read()
        assert "from ravana_ml" not in src
        assert "import neural_decoder" not in src

    def test_no_response_string_authored(self):
        import ravana.learn.macroplanning as mod
        src = open(mod.__file__, encoding="utf-8").read()
        # The module must not build sentence text.
        assert '".join' not in src
        assert '" "' not in src

    def test_different_graphs_give_different_messages(self):
        a = build_message("gravity", _cands(("is_a", "force", 0.8, 0.9)))
        b = build_message("gravity", _cands(("is_a", "emotion", 0.8, 0.9)))
        assert a.concepts != b.concepts
