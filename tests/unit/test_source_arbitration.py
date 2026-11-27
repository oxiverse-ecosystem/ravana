"""Unit tests for the pure, default-off source-arbitration module.

No engine boot, no network, no store write. Every test calls the real module.
"""
from __future__ import annotations

import dataclasses

import pytest

from ravana.learn.source_arbitration import (
    Action, MIN_CREDIBILITY, SourceAssertion, arbitrate, enabled,
    integration_hook, self_loop_decision, source_credibility,
)


def assertion(label="association", *, subject="guild", sense="primary",
              kind="item_description", license="CC0-1.0", negated=False,
              corroboration=1, origin="s1", relation="is_a"):
    return SourceAssertion(
        subject=subject, relation=relation, label=label, license=license,
        sense=sense, evidence_kind=kind, origin_id=origin,
        negated=negated, corroboration=corroboration)


class TestDefaultOff:
    def test_env_flag_defaults_off(self, monkeypatch):
        monkeypatch.delenv("RAVANA_SOURCE_ARBITRATION", raising=False)
        assert enabled() is False

    def test_integration_hook_inert_until_opt_in(self, monkeypatch):
        monkeypatch.delenv("RAVANA_SOURCE_ARBITRATION", raising=False)
        assert integration_hook(None, assertion()) is None
        monkeypatch.setenv("RAVANA_SOURCE_ARBITRATION", "1")
        assert integration_hook(None, assertion()).action is Action.ADMIT


class TestSourceCredibility:
    def test_primary_item_description_is_strongest_admitted_source(self):
        assert source_credibility(assertion()) == pytest.approx(0.7)

    def test_alternate_type_description_is_weaker(self):
        low = assertion("student association", sense="alternate",
                        kind="type_description")
        assert source_credibility(low) == pytest.approx(0.2625)

    def test_unlicensed_or_unknown_tier_is_zero(self):
        assert source_credibility(assertion(license="unknown")) == 0.0
        assert source_credibility(assertion(kind="unknown")) == 0.0
        assert source_credibility(assertion(sense="unknown")) == 0.0


class TestArbitration:
    def test_low_credibility_challenger_cannot_demote_incumbent(self):
        high = assertion("association")
        low = assertion("student association", sense="alternate",
                        kind="type_description")
        decision = arbitrate(high, low)
        assert decision.action is Action.SEPARATE
        assert decision.reason == "incumbent_source_not_inferior"
        assert decision.incumbent_label == "association"
        assert decision.alternative_confidence < 0.20

    def test_below_floor_challenger_is_separated_not_rejected_when_incumbent_exists(self):
        decision = arbitrate(assertion("association"),
                             assertion("student association", sense="alternate"))
        assert decision.action is Action.SEPARATE
        assert decision.alternative_confidence > 0.0

    def test_tie_favours_incumbent(self):
        first = assertion("hydrometeor")
        second = assertion("phenomenon", origin="other")
        assert arbitrate(first, second).action is Action.SEPARATE

    def test_better_corroborated_challenger_may_supersede(self):
        low = assertion("student association", sense="alternate",
                        kind="type_description")
        high = assertion("association", corroboration=2)
        decision = arbitrate(low, high)
        assert decision.action is Action.SUPERSEDE
        assert decision.reason == "higher_credibility_corroborated"
        assert decision.slot_confidence > 0.7

    def test_better_source_without_corroboration_is_only_separated(self):
        low = assertion("student association", sense="alternate",
                        kind="type_description")
        high = assertion("association", corroboration=0)
        assert arbitrate(low, high).action is Action.SEPARATE

    def test_absolute_floor_rejects_weak_challenger(self):
        assert source_credibility(assertion(sense="alternate")) < MIN_CREDIBILITY
        decision = arbitrate(None, assertion(sense="alternate"))
        assert decision.action is Action.REJECT

    def test_negation_vetoes_before_any_source_advantage(self):
        decision = arbitrate(assertion("class"),
                             assertion("card game", negated=True,
                                       corroboration=3))
        assert decision.action is Action.REJECT
        assert decision.reason == "explicit_negation"
        assert decision.slot_confidence == 0.0

    def test_same_label_reinforces(self):
        decision = arbitrate(assertion("association"),
                             assertion("association", origin="s2"))
        assert decision.action is Action.REINFORCE

    def test_first_admitted_assertion_fills_empty_slot(self):
        decision = arbitrate(None, assertion("economy"))
        assert decision.action is Action.ADMIT
        assert decision.slot_confidence == pytest.approx(0.85)

    def test_decision_is_pure_and_repeatable(self):
        a = assertion("association")
        b = assertion("student association", sense="alternate",
                      kind="type_description")
        before = dataclasses.asdict(a), dataclasses.asdict(b)
        first = arbitrate(a, b)
        second = arbitrate(a, b)
        assert first == second
        assert (dataclasses.asdict(a), dataclasses.asdict(b)) == before

    def test_credibility_never_comes_from_label_text_or_stored_weight(self):
        # Same source credentials, different labels => identical credibility.
        a = assertion("association")
        b = assertion("totally unrelated phrase")
        assert source_credibility(a) == source_credibility(b)


class TestSelfLoop:
    def test_admitted_identity_is_written(self):
        cred = source_credibility(assertion("market", subject="market"))
        assert self_loop_decision("market", "market", cred) is Action.WRITE_IDENTITY_SLOT

    def test_unadmitted_identity_is_skipped(self):
        assert self_loop_decision("market", "market", 0.0) is Action.SKIP

    def test_non_identity_is_unchanged(self):
        assert self_loop_decision("guild", "association", 0.7) is Action.NORMAL

    def test_case_and_whitespace_do_not_hide_identity(self):
        cred = source_credibility(assertion("Market", subject=" market "))
        assert self_loop_decision(" market ", "Market", cred) is Action.WRITE_IDENTITY_SLOT
