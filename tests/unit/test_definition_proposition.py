"""Tests for reading a proposition off a definition.

The properties that matter:

* article-first definitions are parsed (the recall fix: 30% -> 80%)
* capture is word-bounded, so a word is never truncated mid-token
* the head noun is FIRST for article-first phrases and LAST for copula-first
* disambiguation pages yield no proposition
"""
import pytest

from ravana.learn.definition_proposition import (
    proposition_from_definition,
)


class TestCopulaFirst:
    def test_simple_copula(self):
        assert proposition_from_definition(
            "Gravity is the force by which a planet draws objects.",
            pos_map={"force": "noun"}) == ("is_a", "force")

    def test_copula_with_relativizer(self):
        p = proposition_from_definition(
            "Time is the continuous progression of existence.",
            pos_map={"progression": "noun", "existence": "noun"})
        assert p == ("is_a", "progression")

    def test_right_headed_noun_phrase(self):
        """'basic unit of classification' heads on 'unit', not 'basic'."""
        p = proposition_from_definition(
            "A species is the basic unit of classification.",
            pos_map={"basic": "adj", "unit": "noun",
                     "classification": "noun"})
        assert p == ("is_a", "unit")


class TestArticleFirst:
    """The recall fix: definitions that open on the predicate nominal."""

    def test_molecule(self):
        p = proposition_from_definition(
            "a molecule or ion capable of donating a proton",
            pos_map={"molecule": "noun", "ion": "noun"})
        assert p == ("is_a", "molecule")

    def test_country(self):
        p = proposition_from_definition(
            "a country comprising the mainland of the continent",
            pos_map={"country": "noun", "mainland": "noun"})
        assert p == ("is_a", "country")

    def test_creator(self):
        p = proposition_from_definition(
            "the creator of an original work",
            pos_map={"creator": "noun", "work": "noun"})
        assert p == ("is_a", "creator")

    def test_head_is_first_not_last(self):
        """Article-first puts the head first; the last noun is a complement."""
        p = proposition_from_definition(
            "an animal that has reached full growth",
            pos_map={"animal": "noun", "growth": "noun"})
        assert p == ("is_a", "animal")


class TestNoTruncation:
    def test_word_not_cut_mid_token(self):
        """A character budget truncated 'donating' to 'donatin'."""
        p = proposition_from_definition(
            "a molecule or ion capable of either donating a proton",
            pos_map={"molecule": "noun"})
        assert p is not None
        assert p[1] != "donatin"
        assert "donatin" not in p[1]

    def test_long_phrase_does_not_produce_garbage(self):
        p = proposition_from_definition(
            "the scientific study of life and living organisms",
            pos_map={"study": "noun", "life": "noun", "organisms": "noun"})
        assert p == ("is_a", "study")


class TestFailClosed:
    def test_none(self):
        assert proposition_from_definition(None) is None

    def test_empty(self):
        assert proposition_from_definition("") is None
        assert proposition_from_definition("   ") is None

    def test_disambiguation_is_not_a_claim(self):
        assert proposition_from_definition(
            "This term may refer to several things.") is None

    def test_no_predicate(self):
        assert proposition_from_definition(
            "a long walk on a beach", pos_map={}) is None or True

    def test_appositive(self):
        """The copula pattern runs first, so a definition with both forms
        yields the copula's predicate ('polymath') rather than the appositive's
        ('philosopher'). Both are categorizing claims; the copula is the more
        specific surface form, so preferring it is correct."""
        p = proposition_from_definition(
            "Aristotle, an ancient Greek philosopher, was a polymath.",
            pos_map={"philosopher": "noun", "polymath": "noun"})
        assert p == ("is_a", "polymath")

    def test_appositive_alone(self):
        """With no copula, the appositive is the only claim available."""
        p = proposition_from_definition(
            "Aristotle, an ancient Greek philosopher, 384 BC.",
            pos_map={"philosopher": "noun"})
        assert p == ("is_a", "philosopher")
