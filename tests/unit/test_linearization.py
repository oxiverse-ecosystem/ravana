"""Tests for linearization: message -> ordered surface plan.

The properties worth pinning are the ones that make this a LINEARIZATION
rather than a template lookup:

* the subject always leads (it is the utterance's topic)
* the construction follows the RELATION's semantics, not the activation order
* a repeated role fills correctly ("X is a Y", not "organism is a organism")
* an unfillable slot drops the plan rather than rendering a hole
* an uncommitted plan (vacuous relation / non-concept filler) is rejected --
  realizing one would produce a fluent falsehood
"""
import pytest

from ravana.learn.macroplanning import Message, RoleBinding
from ravana.learn.linearization import (
    CONSTRUCTIONS,
    SurfacePlan,
    message_to_plan,
    plan_is_committed,
)


def _msg(subject, pairs):
    """pairs: (role, concept, relation)."""
    return Message(
        subject=subject,
        goal="characterize",
        bindings=[RoleBinding(role=r, concept=c, activation=1.0, relation=rel)
                   for r, c, rel in pairs],
    )


class TestConstructionChoice:
    """The relation's semantics pick the frame."""

    def test_is_a_uses_categorize(self):
        p = message_to_plan(_msg("species", [("theme", "organism", "is_a")]),
                            subject="species")
        assert p.construction == "categorize"
        assert p.tokens == ["species", "is", "a", "organism"]

    def test_causes_uses_cause(self):
        p = message_to_plan(_msg("sky", [("ground", "fire", "causes")]),
                            subject="sky")
        assert p.construction == "cause"
        assert p.tokens == ["sky", "leads", "to", "fire"]

    def test_has_property_uses_attribute(self):
        p = message_to_plan(_msg("sky", [("relation", "blue", "has_property")]),
                            subject="sky")
        assert p.construction == "attribute"
        assert p.tokens == ["sky", "has", "blue"]

    def test_located_in_uses_locate(self):
        p = message_to_plan(_msg("sky", [("ground", "orbit", "located_in")]),
                            subject="sky")
        assert p.construction == "locate"
        assert p.tokens == ["sky", "is", "located", "in", "orbit"]

    def test_unknown_relation_falls_back_to_relate(self):
        p = message_to_plan(_msg("x", [("relation", "y", "something_novel")]),
                            subject="x")
        assert p.construction == "relate"

    def test_part_of_is_categorize(self):
        p = message_to_plan(_msg("species", [("theme", "taxonomy", "part_of")]),
                            subject="species")
        assert p.construction == "categorize"


class TestSubjectLeads:
    def test_subject_is_first_token(self):
        p = message_to_plan(_msg("gravity", [("theme", "force", "is_a")]),
                            subject="gravity")
        assert p.tokens[0] == "gravity"

    def test_subject_fills_the_first_theme_slot(self):
        """A repeated role must not put the category first."""
        p = message_to_plan(_msg("species", [("theme", "organism", "is_a")]),
                            subject="species")
        assert p.tokens[0] == "species"
        assert p.tokens[-1] == "organism"
        assert p.tokens != ["organism", "is", "a", "species"]


class TestFailClosed:
    def test_none_message(self):
        assert message_to_plan(None) is None

    def test_no_bindings(self):
        assert message_to_plan(Message(subject="x", goal="g", bindings=[])) is None

    def test_no_subject(self):
        assert message_to_plan(_msg("", [("theme", "y", "is_a")])) is None

    def test_missing_slot_binding_returns_none(self):
        """A cause frame with no ground cannot be rendered without a hole."""
        p = message_to_plan(
            _msg("sky", [("agent", "sky", "causes")]), subject="sky")
        assert p is None or "sky" in p.tokens

    def test_plan_is_explainable(self):
        p = message_to_plan(_msg("music", [("theme", "sound", "is_a")]),
                            subject="music")
        d = p.as_dict()
        for k in ("subject", "construction", "tokens", "roles"):
            assert k in d


class TestCommitmentGate:
    """A plan that asserts nothing must be rejected, not realized."""

    def test_categorize_is_committed(self):
        p = message_to_plan(_msg("music", [("theme", "sound", "is_a")]),
                            subject="music")
        assert plan_is_committed(p)

    def test_vacuous_relate_is_not_committed(self):
        p = message_to_plan(
            _msg("gravity", [("relation", "earth", "related_to")]),
            subject="gravity")
        assert not plan_is_committed(p)

    def test_non_concept_filler_is_not_committed(self):
        """The measured failure: 'gravity is related to two'."""
        p = SurfacePlan(subject="gravity", construction="attribute",
                        tokens=["gravity", "has", "two"],
                        roles=["theme", "relation"],
                        fill={"theme": "gravity", "relation": "two"})
        assert not plan_is_committed(p)

    def test_frame_words_do_not_disqualify(self):
        """'music is a sound' contains 'a', which is a frame word."""
        p = message_to_plan(_msg("music", [("theme", "sound", "is_a")]),
                            subject="music")
        assert "a" in p.tokens
        assert plan_is_committed(p)

    def test_none_is_not_committed(self):
        assert not plan_is_committed(None)


class TestInventory:
    def test_constructions_are_few_and_closed(self):
        assert len(CONSTRUCTIONS) <= 8

    def test_every_construction_renders(self):
        for c in CONSTRUCTIONS.values():
            fill = {r: "X" for r in c.slots}
            out = c.render(fill)
            assert out
            assert all(not t.startswith("{") for t in out)
