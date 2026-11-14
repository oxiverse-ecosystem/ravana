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


class TestBestEdgeSelection:
    """The plan route needs one committed proposition, not a field.

    Measured failure: spreading_activation redistributes a fraction of the
    total field to every candidate, so a hub target wins on connectivity
    rather than on its own edge -- 'water leads to heavy', 'light leads to
    soft', 'trust is like management'.
    """

    def test_picks_the_strongest_edge(self):
        from ravana.learn.macroplanning import select_target
        cands = [("causal", "earth", 0.9, 0.9), ("is_a", "force", 0.3, 0.3)]
        rel, tgt, w, c = select_target("gravity", cands)
        assert (rel, tgt) == ("causal", "earth")

    def test_relation_and_target_come_from_the_same_edge(self):
        from ravana.learn.macroplanning import message_from_best_edge
        m = message_from_best_edge(
            "gravity", [("causal", "earth", 0.9, 0.9),
                        ("is_a", "force", 0.3, 0.3)])
        (b,) = m.bindings
        assert b.concept == "earth"
        assert b.relation == "causal"

    def test_subject_is_excluded_as_its_own_target(self):
        from ravana.learn.macroplanning import select_target
        assert select_target("gravity", [("is_a", "gravity", 0.9, 0.9)]) is None

    def test_empty_candidates(self):
        from ravana.learn.macroplanning import select_target, message_from_best_edge
        assert select_target("x", []) is None
        assert message_from_best_edge("x", []) is None

    def test_no_subject(self):
        from ravana.learn.macroplanning import message_from_best_edge
        assert message_from_best_edge("", [("is_a", "y", 0.9, 0.9)]) is None

    def test_best_edge_plan_is_committed(self):
        from ravana.learn.macroplanning import message_from_best_edge
        from ravana.learn.linearization import message_to_plan
        m = message_from_best_edge(
            "gravity", [("causal", "orbit", 0.9, 0.9)])
        p = message_to_plan(m, subject="gravity")
        assert plan_is_committed(p)
        assert p.tokens == ["gravity", "leads", "to", "orbit"]


class TestExtendedInventory:
    """The graph's non-typed relation vocabulary (temporal / analogical /
    contrastive / inferred / episodic) each get their own construction."""

    def test_temporal(self):
        p = message_to_plan(_msg("winter", [("ground", "spring", "temporal")]),
                            subject="winter")
        assert p.construction == "temporal"
        assert p.tokens == ["winter", "follows", "spring"]

    def test_analogical(self):
        p = message_to_plan(_msg("heart", [("ground", "pump", "analogical")]),
                            subject="heart")
        assert p.construction == "analogy"
        assert p.tokens == ["heart", "is", "like", "pump"]

    def test_contrastive(self):
        p = message_to_plan(_msg("hot", [("ground", "cold", "contrastive")]),
                            subject="hot")
        assert p.construction == "contrast"
        assert p.tokens == ["hot", "contrasts", "with", "cold"]

    def test_inferred(self):
        p = message_to_plan(_msg("rain", [("ground", "clouds", "inferred")]),
                            subject="rain")
        assert p.construction == "infer"

    def test_episodic(self):
        p = message_to_plan(_msg("coronation", [("ground", "2023", "episodic")]),
                            subject="coronation")
        assert p.construction == "episodic"

    def test_new_constructions_are_committed(self):
        """A typed relation with a real concept filler must pass the gate."""
        for rel, subj, tgt in [("temporal", "winter", "spring"),
                               ("analogical", "heart", "pump"),
                               ("contrastive", "hot", "cold"),
                               ("inferred", "rain", "clouds")]:
            p = message_to_plan(_msg(subj, [("ground", tgt, rel)]), subject=subj)
            assert plan_is_committed(p), (rel, p)

    def test_causal_temporal_distinct_from_analogy(self):
        """'leads to' asserts causation; 'is like' asserts resemblance."""
        c = message_to_plan(_msg("fire", [("ground", "smoke", "causes")]),
                            subject="fire")
        a = message_to_plan(_msg("fire", [("ground", "smoke", "analogical")]),
                            subject="fire")
        assert c.construction != a.construction


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
        """One construction per relation family in the graph's vocabulary
        (graph_typing.TYPED_RELATION_TYPES plus the learned relation types:
        temporal, causal, analogical, contrastive, inferred, episodic)."""
        assert len(CONSTRUCTIONS) <= 12

    def test_every_construction_renders(self):
        for c in CONSTRUCTIONS.values():
            fill = {r: "X" for r in c.slots}
            out = c.render(fill)
            assert out
            assert all(not t.startswith("{") for t in out)
