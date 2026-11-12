"""Tests for Committed-Proposition Gating.

The properties worth pinning are the ones that make CPG brain-faithful rather
than a generic filter:

* high prediction error RAISES the criterion (it does not invite speech)
* a token is admissible only if it realizes a committed target
* failure RETRIES with a relaxed criterion instead of falling through
* rejection raises prediction error, so the graph becomes more conservative
"""
import pytest

from ravana.learn.committed_proposition import (
    CPGConfig,
    acc_criterion,
    commit,
    commitment_strength,
    decide,
    update_commitment,
)


def _props(specs, pe=0.0):
    """specs: (relation, target, edge_weight, edge_confidence)."""
    return commit("x", specs, prediction_error=pe)


class TestCommitmentStrength:
    def test_all_fields_present_scores_higher(self):
        cfg = CPGConfig()
        weak = commitment_strength(edge_weight=0.5, cfg=cfg)
        strong = commitment_strength(
            edge_weight=0.9, edge_confidence=0.9,
            node_confidence=0.9, node_stability=0.9, cfg=cfg)
        assert strong > weak

    def test_no_evidence_scores_zero(self):
        assert commitment_strength() == 0.0

    def test_out_of_range_clamped(self):
        s = commitment_strength(edge_weight=7.0, edge_confidence=-2.0)
        assert 0.0 <= s <= 1.0

    def test_weights_sum_to_one(self):
        assert CPGConfig().strength_weights_sum() == pytest.approx(1.0)

    def test_propositions_are_sorted_strongest_first(self):
        ps = _props([("causes", "a", 0.9, 0.9), ("related_to", "b", 0.1, 0.1)])
        assert ps[0].target == "a"
        assert ps[0].strength > ps[1].strength


class TestACCCriterion:
    """The counter-intuitive property: uncertainty demands MORE evidence."""

    def test_prediction_error_raises_the_criterion(self):
        cfg = CPGConfig()
        assert acc_criterion(1.0, 0.0, cfg) > acc_criterion(0.0, 0.0, cfg)

    def test_free_energy_raises_the_criterion(self):
        cfg = CPGConfig()
        assert acc_criterion(0.0, 1.0, cfg) > acc_criterion(0.0, 0.0, cfg)

    def test_criterion_is_never_negative(self):
        cfg = CPGConfig(theta_base=0.0)
        assert acc_criterion(0.0, 0.0, cfg) == 0.0

    def test_zero_gains_disable_modulation(self):
        cfg = CPGConfig(alpha_pe=0.0, beta_fe=0.0)
        assert acc_criterion(1.0, 1.0, cfg) == pytest.approx(cfg.theta_base)


class TestCommit:
    def test_empty_candidates_yield_no_propositions(self):
        assert commit("x", []) == []

    def test_blank_targets_are_skipped(self):
        assert commit("x", [("causes", "", 0.9, 0.9)]) == []

    def test_proposition_carries_its_fields(self):
        (p,) = _props([("causes", "fire", 0.8, 0.7)])
        assert p.subject == "x"
        assert p.relation == "causes"
        assert p.target == "fire"
        assert p.edge_weight == pytest.approx(0.8)


class TestDecide:
    def test_strong_proposition_commits(self):
        d = decide("x", _props([("causes", "fire", 0.9, 0.9)]))
        assert d.committed
        assert d.admissible_targets[0].target == "fire"

    def test_weak_proposition_does_not_commit(self):
        d = decide("x", _props([("related_to", "zzz", 0.01, 0.01)]))
        assert not d.committed

    def test_no_propositions_is_not_committed(self):
        d = decide("x", [])
        assert not d.committed
        assert "no candidate" in d.reason

    def test_high_prediction_error_can_withhold_a_marginal_commitment(self):
        """The brain demands more evidence under uncertainty."""
        specs = [("causes", "fire", 0.4, 0.4)]
        calm = decide("x", _props(specs, pe=0.0))
        uncertain = decide("x", _props(specs, pe=1.0))
        assert uncertain.theta > calm.theta
        # Whatever the outcome, it follows the stated criterion.
        assert uncertain.committed == bool(uncertain.admissible_targets)

    def test_retry_relaxes_the_criterion(self):
        """Conduit d'approche: retry with a relaxed bar, do not fall through."""
        cfg = CPGConfig(theta_base=0.9, relax_step=0.2, max_retries=3)
        d = decide("x", _props([("causes", "fire", 0.5, 0.5)]), cfg=cfg)
        assert d.retries_used > 0
        assert d.final_theta < d.theta

    def test_retry_gives_up_after_max_retries(self):
        cfg = CPGConfig(theta_base=1.0, relax_step=0.01, max_retries=2)
        d = decide("x", _props([("causes", "fire", 0.05, 0.05)]), cfg=cfg)
        assert not d.committed
        assert d.retries_used == 2
        assert "escalate" in d.reason

    def test_token_boost_only_for_committed_targets(self):
        ps = _props([("causes", "fire", 0.9, 0.9),
                     ("related_to", "zzz", 0.01, 0.01)])
        d = decide("x", ps, token_for_target={"fire": 11, "zzz": 22})
        assert 11 in d.token_boost
        assert 22 not in d.token_boost

    def test_no_token_map_means_no_boost(self):
        d = decide("x", _props([("causes", "fire", 0.9, 0.9)]))
        assert d.token_boost == {}

    def test_stronger_commitment_gets_a_stronger_boost(self):
        ps = _props([("causes", "fire", 0.95, 0.95),
                     ("causes", "smoke", 0.55, 0.55)])
        d = decide("x", ps, token_for_target={"fire": 11, "smoke": 22})
        assert d.token_boost[11] > d.token_boost[22]

    def test_no_suppression_by_default(self):
        """suppress=0.0 is the pre-NoGo behaviour: boost only."""
        ps = _props([("causes", "fire", 0.9, 0.9),
                     ("related_to", "zzz", 0.05, 0.05)])
        d = decide("x", ps, token_for_target={"fire": 11, "zzz": 22})
        assert d.token_boost and not d.token_suppress

    def test_nogo_suppresses_inadmissible_targets(self):
        cfg = CPGConfig(suppress=2.0)
        ps = _props([("causes", "fire", 0.9, 0.9),
                     ("related_to", "zzz", 0.02, 0.02)])
        d = decide("x", ps, token_for_target={"fire": 11, "zzz": 22}, cfg=cfg)
        assert 11 in d.token_boost
        assert 22 not in d.token_boost
        assert 22 in d.token_suppress

    def test_nogo_spares_near_misses(self):
        """A target close to the weakest admissible one is not suppressed --
        suppressing it would trade real signal for marginal purity.

        fire=0.585 and smoke=0.422 are 0.163 apart, inside the 0.3 margin, so
        smoke must be spared.
        """
        cfg = CPGConfig(suppress=2.0, suppress_margin=0.3)
        ps = _props([("causes", "fire", 0.9, 0.9),
                     ("causes", "smoke", 0.55, 0.55)])
        d = decide("x", ps, token_for_target={"fire": 11, "smoke": 22}, cfg=cfg)
        assert 11 in d.token_boost
        assert 22 not in d.token_suppress

    def test_nogo_does_not_suppress_when_nothing_is_admissible(self):
        """If no proposition clears theta there is nothing to prefer, so
        suppression is meaningless and must not fire."""
        cfg = CPGConfig(suppress=2.0)
        ps = _props([("related_to", "zzz", 0.01, 0.01)])
        d = decide("x", ps, token_for_target={"zzz": 22}, cfg=cfg)
        assert not d.committed
        assert not d.token_suppress

    def test_weaker_inadmissible_target_is_suppressed_harder(self):
        cfg = CPGConfig(suppress=2.0)
        ps = _props([("causes", "fire", 0.9, 0.9),
                     ("related_to", "a", 0.2, 0.2),
                     ("related_to", "b", 0.02, 0.02)])
        d = decide("x", ps, token_for_target={"fire": 11, "a": 22, "b": 33},
                   cfg=cfg)
        assert d.token_suppress[33] > d.token_suppress[22]

    def test_relation_gating_boosts_the_committed_relation_word(self):
        """A committed proposition is (subject, relation, target). Boosting
        only the target leaves the relation free -- and an unsupported
        assertion is exactly a relation the graph does not support."""
        cfg = CPGConfig(relation_words={"causes": ("leads", "causes")})
        ps = _props([("causes", "fire", 0.9, 0.9)])
        d = decide("x", ps, token_for_target={"fire": 11, "leads": 33},
                   cfg=cfg)
        assert 11 in d.token_boost      # target
        assert 33 in d.token_boost      # relation word

    def test_relation_gating_does_not_boost_uncommitted_relations(self):
        cfg = CPGConfig(relation_words={"causes": ("leads",),
                                        "temporal": ("follows",)})
        ps = _props([("causes", "fire", 0.9, 0.9),
                     ("temporal", "zzz", 0.02, 0.02)])
        d = decide("x", ps, token_for_target={"fire": 11, "leads": 33,
                                              "follows": 44}, cfg=cfg)
        assert 33 in d.token_boost
        assert 44 not in d.token_boost

    def test_relation_gating_off_by_default(self):
        ps = _props([("causes", "fire", 0.9, 0.9)])
        d = decide("x", ps, token_for_target={"fire": 11, "leads": 33})
        assert 11 in d.token_boost
        assert 33 not in d.token_boost

    def test_relation_word_absent_from_vocab_is_skipped(self):
        cfg = CPGConfig(relation_words={"causes": ("versus",)})
        ps = _props([("causes", "fire", 0.9, 0.9)])
        d = decide("x", ps, token_for_target={"fire": 11}, cfg=cfg)
        assert 11 in d.token_boost
        assert len(d.token_boost) == 1

    def test_decision_is_explainable(self):
        d = decide("x", _props([("causes", "fire", 0.9, 0.9)]))
        out = d.as_dict()
        for k in ("subject", "committed", "theta", "retries_used", "reason"):
            assert k in out


class TestUpdateCommitment:
    def test_acceptance_reinforces_and_cools(self):
        (p,) = _props([("causes", "fire", 0.5, 0.5)], pe=0.4)
        conf, pe = update_commitment(p, accepted=True)
        assert conf > 0.5
        assert pe < 0.4

    def test_rejection_weakens_and_raises_error(self):
        (p,) = _props([("causes", "fire", 0.5, 0.5)], pe=0.4)
        conf, pe = update_commitment(p, accepted=False)
        assert conf < 0.5
        assert pe > 0.4

    def test_updates_are_bounded(self):
        (p,) = _props([("causes", "fire", 1.0, 1.0)], pe=0.0)
        conf, pe = update_commitment(p, accepted=True)
        assert 0.0 <= conf <= 1.0 and 0.0 <= pe <= 1.0
        (q,) = _props([("causes", "fire", 0.0, 0.0)], pe=1.0)
        conf, pe = update_commitment(q, accepted=False)
        assert 0.0 <= conf <= 1.0 and 0.0 <= pe <= 1.0

    def test_rejection_raises_the_next_criterion(self):
        """The loop closes: a proposition that produced ungrounded speech
        should be harder to commit to next time."""
        (p,) = _props([("causes", "fire", 0.5, 0.5)], pe=0.3)
        _, pe_after = update_commitment(p, accepted=False)
        before = acc_criterion(p.prediction_error, 0.0)
        after = acc_criterion(pe_after, 0.0)
        assert after > before
