"""Tests for the noun-admissibility veto filter.

The filter is veto-only: it rejects on positive evidence of non-noun-hood and
admits everything else. The tests pin the MEASURED behaviour, including the
two designs that were tried and rejected.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana", "src"))

from ravana.learn.noun_admissibility import (  # noqa: E402
    filter_noun_candidates, looks_like_noun,
)


class TestClosedClassVeto:
    """The measured failures: these must be rejected."""

    @pytest.mark.parametrize("w", [
        "only", "however", "rather", "now", "there", "further", "often",
        "even", "just", "also", "two", "three", "good",
    ])
    def test_adverbs_and_function_words_rejected(self, w):
        assert not looks_like_noun(w)

    def test_the_measured_time_case(self):
        """`time` picked `analogical -> only` over `semantic -> years`.

        With the filter applied to the pool before ranking, `only` is removed
        and `years` wins -- measured on the live store as the difference
        between "time is different from an only" and a noun target.
        """
        cands = [
            ("semantic", "years", 0.9, 0.8),
            ("analogical", "only", 0.95, 0.9),
        ]
        kept = filter_noun_candidates(cands)
        assert [c[1] for c in kept] == ["years"]


class TestRealNounsAdmitted:
    """Recall 1.000 on gold -- these must NOT be vetoed."""

    @pytest.mark.parametrize("w", [
        "force", "earth", "liquid", "life", "water", "body", "dog", "sun",
        "sound", "ocean", "gravity", "years", "climate", "learning", "music",
        "animal", "natural", "journal", "mammal", "oxygen", "environment",
        "development", "history", "theory", "plant",
    ])
    def test_real_nouns_admitted(self, w):
        assert looks_like_noun(w), w


class TestAdjectiveVeto:
    @pytest.mark.parametrize("w", [
        "environmental", "mathematical", "psychological", "physical",
        "practical", "individual", "ethical",
        "different", "ancient", "primary",
    ])
    def test_adjectives_rejected(self, w):
        assert not looks_like_noun(w), w

    @pytest.mark.parametrize("w", ["original", "social", "global", "moral",
                                   "natural", "animal", "mammal", "journal"])
    def test_ambiguous_al_words_are_allowlisted_as_nouns(self, w):
        """`-al` is genuinely ambiguous, so these are decided by allowlist.

        `original`/`social` are adjectival in most uses but are graph labels
        used as nouns here; the allowlist is the measured set of -al nouns on
        THIS graph. Flagged as the least certain part of the filter.
        """
        assert looks_like_noun(w), w


class TestPosHintIsUnused:
    """The map is the defective artifact; trusting it inherits its errors.

    Measured: a POS-hint veto rejected 8 real nouns (`climate`->verb via
    '-ate', `learning`->verb via '-ing', `music`->adj via '-ic',
    `animal`->adj via '-al'). So the hint must NOT be consulted.
    """

    @pytest.mark.parametrize("w,hint", [
        ("climate", "verb"),
        ("learning", "verb"),
        ("music", "adj"),
        ("animal", "adj"),
        ("natural", "adj"),
        ("oxygen", "verb"),
    ])
    def test_mislabelled_nouns_still_admitted(self, w, hint):
        assert looks_like_noun(w, pos_hint=hint), (w, hint)
        assert looks_like_noun(w, pos_map={w: hint}), (w, hint)


class TestFilterCandidates:
    def test_vetoed_candidate_removed_before_ranking(self):
        """The whole point: a vetoed candidate must not be rankable."""
        cands = [
            ("semantic", "years", 0.9, 0.8),      # noun, weaker edge
            ("analogical", "only", 0.95, 0.9),    # adverb, stronger edge
        ]
        kept = filter_noun_candidates(cands)
        assert [c[1] for c in kept] == ["years"]

    def test_empty_pool_stays_empty(self):
        assert filter_noun_candidates([("rel", "only", 1.0, 1.0)]) == []

    def test_real_pool_passes_through(self):
        cands = [("is_a", "force", 0.5, 0.5), ("causes", "earth", 0.4, 0.4)]
        assert len(filter_noun_candidates(cands)) == 2

    def test_malformed_entries_skipped(self):
        assert filter_noun_candidates([(), ("rel",), None]) == []


class TestMultiwordRejected:
    def test_phrases_rejected(self):
        """Multi-word labels are handled by `_head_noun`, not this filter."""
        assert not looks_like_noun("living organism")
        assert not looks_like_noun("electromagnetic radiation")
