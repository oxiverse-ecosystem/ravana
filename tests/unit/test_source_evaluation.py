"""Unit tests for the pure source-evaluation mechanism.

Fake-free: every test builds real ``SourceStatus``/``ClaimContext`` records and
calls the real evaluator.  No engine is booted, no network, no global state.
These are intentionally runnable in isolation by the parent's pytest shard.
"""
from __future__ import annotations

import dataclasses

import pytest

from ravana.chat.source_evaluation import (
    ClaimContext,
    Decision,
    Decision as D,
    SourceStatus,
    TrustClass,
    context_from_wfa_item,
    decision_token,
    evaluate_source,
    integration_hook,
)


def _credible(origin: str = "src-credible-1", *, active: bool = True,
              token: str = "credible", reliability: float = 0.99) -> SourceStatus:
    return SourceStatus(
        origin_id=origin,
        source_type="web_fact",
        reliability=reliability,
        trust_class=TrustClass.CREDIBLE,
        active=active,
        retrieval_context="frozen_web_fixture",
        credibility_token=token,
    )


class TestCredibleEndorsement:
    def test_credible_known_source_endorses(self):
        ctx = ClaimContext(source=_credible(), familiarity=0.0)
        d = evaluate_source("The entailment holds.", ctx)
        assert d.decision is Decision.ENDORSE
        assert d.rule == "R_EXPLICIT_CREDIBLE_SOURCE"
        assert d.token == "Y"
        assert d.origin_id == "src-credible-1"

    def test_credible_source_with_trusted_allowlist_endorses(self):
        ctx = ClaimContext(
            source=_credible(),
            trusted_source_ids=frozenset({"src-credible-1", "src-other"}),
        )
        assert evaluate_source("claim", ctx).decision is Decision.ENDORSE

    @pytest.mark.parametrize("token", ["credible", "trusted", "verified", "reliable"])
    def test_all_explicit_tokens_admit(self, token):
        ctx = ClaimContext(source=_credible(token=token))
        assert evaluate_source("claim", ctx).decision is Decision.ENDORSE


class TestUnknownProvenance:
    def test_unknown_trust_class_abstains(self):
        src = SourceStatus(origin_id="src-unknown", trust_class=TrustClass.UNKNOWN)
        d = evaluate_source("familiar but unsourced claim", ClaimContext(source=src))
        assert d.decision is Decision.ABSTAIN
        assert d.rule == "R_UNKNOWN_PROVENANCE"
        assert d.token == "N"

    def test_absent_source_abstains(self):
        d = evaluate_source("familiar claim", ClaimContext())
        assert d.decision is Decision.ABSTAIN
        assert d.rule == "R_UNKNOWN_PROVENANCE"

    def test_credible_class_without_id_is_unknown_provenance(self):
        src = SourceStatus(origin_id="", trust_class=TrustClass.CREDIBLE,
                           credibility_token="credible")
        assert evaluate_source("claim", ClaimContext(source=src)).rule == (
            "R_UNKNOWN_PROVENANCE"
        )


class TestDiscredited:
    def test_fresh_discredited_claim_abstains(self):
        src = SourceStatus(origin_id="src-bad", trust_class=TrustClass.DISCREDITED)
        d = evaluate_source("familiar near-miss", ClaimContext(source=src))
        assert d.decision is Decision.ABSTAIN
        assert d.rule == "R_SOURCE_DISCREDITED"

    def test_discredited_after_prior_endorsement_withdraws(self):
        src = SourceStatus(origin_id="src-bad", trust_class=TrustClass.DISCREDITED)
        ctx = ClaimContext(source=src, prior_decision=Decision.ENDORSE)
        d = evaluate_source("learned claim", ctx)
        assert d.decision is Decision.WITHDRAW
        assert d.rule == "R_SOURCE_INVALIDATED"
        assert d.token == "W"

    def test_explicit_invalidation_after_prior_endorsement_withdraws(self):
        ctx = ClaimContext(source=_credible(), prior_decision=Decision.ENDORSE,
                           invalidated=True)
        d = evaluate_source("learned claim", ctx)
        assert d.decision is Decision.WITHDRAW
        assert d.rule == "R_SOURCE_INVALIDATED"

    def test_invalidation_without_prior_endorsement_abstains(self):
        ctx = ClaimContext(source=_credible(), invalidated=True)
        d = evaluate_source("claim", ctx)
        assert d.decision is Decision.ABSTAIN
        assert d.rule == "R_INVALIDATION_NO_PRIOR"

    def test_prior_abstain_then_discredit_is_abstain_not_withdraw(self):
        src = SourceStatus(origin_id="src-bad", trust_class=TrustClass.DISCREDITED)
        ctx = ClaimContext(source=src, prior_decision=Decision.ABSTAIN)
        d = evaluate_source("claim", ctx)
        assert d.decision is Decision.ABSTAIN
        assert d.rule == "R_SOURCE_DISCREDITED"

    def test_discredit_dominates_a_stale_credibility_token(self):
        src = SourceStatus(origin_id="src-bad", trust_class=TrustClass.DISCREDITED,
                           credibility_token="credible")
        ctx = ClaimContext(source=src, prior_decision=Decision.ENDORSE)
        assert evaluate_source("claim", ctx).rule == "R_SOURCE_INVALIDATED"


class TestDormantSleeper:
    def test_dormant_without_cue_is_not_endorsed(self):
        ctx = ClaimContext(source=_credible(active=False), cue_present=False)
        d = evaluate_source("sleeper claim", ctx)
        assert d.decision is Decision.ABSTAIN
        assert d.rule == "R_DORMANT_NO_CUE"

    def test_dormant_with_cue_can_be_endorsed(self):
        ctx = ClaimContext(source=_credible(active=False), cue_present=True)
        d = evaluate_source("sleeper claim", ctx)
        assert d.decision is Decision.ENDORSE
        assert d.rule == "R_EXPLICIT_CREDIBLE_SOURCE"

    def test_prior_endorsement_going_dormant_withdraws(self):
        ctx = ClaimContext(source=_credible(active=False),
                           prior_decision=Decision.ENDORSE)
        d = evaluate_source("sleeper claim", ctx)
        assert d.decision is Decision.WITHDRAW
        assert d.rule == "R_FACT_DORMANT"

    def test_cue_reactivates_withdrawn_dormant_fact(self):
        ctx = ClaimContext(source=_credible(active=False),
                           prior_decision=Decision.ENDORSE, cue_present=True)
        assert evaluate_source("sleeper claim", ctx).decision is Decision.ENDORSE


class TestNoFamiliarityShortcut:
    def test_high_familiarity_cannot_rescue_unknown_provenance(self):
        ctx = ClaimContext(familiarity=1.0)
        d = evaluate_source("very familiar unsourced claim", ctx)
        assert d.decision is Decision.ABSTAIN
        assert d.rule == "R_UNKNOWN_PROVENANCE"

    def test_high_reliability_without_token_is_not_endorsed(self):
        src = SourceStatus(origin_id="src-hi", reliability=1.0,
                           trust_class=TrustClass.CREDIBLE)
        d = evaluate_source("claim", ClaimContext(source=src, familiarity=1.0))
        assert d.decision is Decision.ABSTAIN
        assert d.rule == "R_NO_EXPLICIT_CREDIBILITY"

    def test_unrecognised_token_is_not_endorsed(self):
        src = SourceStatus(origin_id="src-x", trust_class=TrustClass.CREDIBLE,
                           credibility_token="sounds-about-right")
        assert evaluate_source("claim", ClaimContext(source=src)).rule == (
            "R_NO_EXPLICIT_CREDIBILITY"
        )

    def test_zero_familiarity_does_not_block_explicit_credibility(self):
        d = evaluate_source("claim", ClaimContext(source=_credible(), familiarity=0.0))
        assert d.decision is Decision.ENDORSE

    def test_familiarity_is_not_part_of_the_audit_payload(self):
        d = evaluate_source("claim", ClaimContext(source=_credible(), familiarity=1.0))
        assert "familiarity" not in d.as_dict()


class TestTrustedAllowlist:
    def test_origin_outside_allowlist_abstains(self):
        ctx = ClaimContext(source=_credible(origin="src-out"),
                           trusted_source_ids=frozenset({"src-in"}))
        d = evaluate_source("claim", ctx)
        assert d.decision is Decision.ABSTAIN
        assert d.rule == "R_SOURCE_NOT_TRUSTED"

    def test_origin_inside_allowlist_endorses(self):
        ctx = ClaimContext(source=_credible(origin="src-in"),
                           trusted_source_ids=frozenset({"src-in"}))
        assert evaluate_source("claim", ctx).decision is Decision.ENDORSE


class TestAuditTrail:
    @pytest.mark.parametrize("ctx,expected_decision,expected_rule", [
        (ClaimContext(), D.ABSTAIN, "R_UNKNOWN_PROVENANCE"),
        (ClaimContext(source=_credible()), D.ENDORSE, "R_EXPLICIT_CREDIBLE_SOURCE"),
        (ClaimContext(source=SourceStatus("bad", trust_class=TrustClass.DISCREDITED)),
         D.ABSTAIN, "R_SOURCE_DISCREDITED"),
        (ClaimContext(source=_credible(), invalidated=True,
                      prior_decision=D.ENDORSE), D.WITHDRAW, "R_SOURCE_INVALIDATED"),
    ])
    def test_each_decision_names_the_firing_rule(self, ctx, expected_decision,
                                                  expected_rule):
        d = evaluate_source("claim", ctx)
        assert d.decision is expected_decision
        assert d.rule == expected_rule
        assert d.reason and isinstance(d.reason, str)
        assert d.as_dict()["decision"] == expected_decision.value
        assert d.as_dict()["rule"] == expected_rule

    def test_unknown_decision_for_empty_claim(self):
        d = evaluate_source("   ", ClaimContext(source=_credible()))
        assert d.decision is Decision.UNKNOWN
        assert d.rule == "R_INDETERMINATE_INPUT"
        assert d.token == "?"

    def test_unknown_decision_for_missing_context(self):
        d = evaluate_source("claim", None)  # type: ignore[arg-type]
        assert d.decision is Decision.UNKNOWN
        assert d.rule == "R_INDETERMINATE_INPUT"

    def test_decision_token_surface_is_exact(self):
        assert decision_token(D.ENDORSE) == "Y"
        assert decision_token(D.ABSTAIN) == "N"
        assert decision_token(D.WITHDRAW) == "W"
        assert decision_token(D.UNKNOWN) == "?"

    def test_records_are_frozen(self):
        src = _credible()
        with pytest.raises(dataclasses.FrozenInstanceError):
            src.reliability = 0.0  # type: ignore[misc]


class TestDefaultOffHook:
    def test_hook_is_inert_by_default(self):
        assert integration_hook("claim", ClaimContext(source=_credible())) is None

    def test_hook_returns_decision_when_explicitly_enabled(self):
        d = integration_hook("claim", ClaimContext(source=_credible()), enabled=True)
        assert d is not None
        assert d.decision is Decision.ENDORSE

    def test_hook_does_not_mutate_its_inputs_or_evaluator(self):
        ctx = ClaimContext(source=_credible())
        before = dataclasses.asdict(ctx)
        first = integration_hook("claim", ctx, enabled=True)
        assert integration_hook("claim", ctx, enabled=True) == first
        assert dataclasses.asdict(ctx) == before
        assert evaluate_source("claim", ctx) == first

    def test_disabled_hook_is_not_a_side_effect_channel(self):
        ctx = ClaimContext()
        assert integration_hook("claim", ctx) is None
        assert integration_hook("claim", ctx) is None
        assert evaluate_source("claim", ctx).decision is Decision.ABSTAIN


class TestWFAAdapter:
    def test_credible_wfa_item_endorses(self):
        ctx = context_from_wfa_item({"source_status": "credible",
                                     "source_id": "https://example.org/a",
                                     "familiarity": 0.9})
        d = evaluate_source("gap claim", ctx)
        assert d.decision is Decision.ENDORSE
        assert d.origin_id == "https://example.org/a"

    def test_discredited_wfa_item_abstains(self):
        ctx = context_from_wfa_item({"source_status": "discredited",
                                     "source_id": "src_discredited_x"})
        assert evaluate_source("near-miss", ctx).rule == "R_SOURCE_DISCREDITED"

    def test_unknown_wfa_item_abstains(self):
        ctx = context_from_wfa_item({"source_status": "unknown", "source_id": None})
        assert evaluate_source("closed-web claim", ctx).rule == "R_UNKNOWN_PROVENANCE"

    def test_internal_wfa_item_has_no_explicit_credibility(self):
        ctx = context_from_wfa_item({"source_status": "internal", "source_id": None})
        assert evaluate_source("parametric claim", ctx).decision is Decision.ABSTAIN

    def test_invalidation_after_prior_endorsement_from_wfa_item(self):
        ctx = context_from_wfa_item(
            {"source_status": "credible", "source_id": "https://example.org/a"},
            prior_decision=Decision.ENDORSE, invalidated=True,
        )
        assert evaluate_source("taught claim", ctx).rule == "R_SOURCE_INVALIDATED"
