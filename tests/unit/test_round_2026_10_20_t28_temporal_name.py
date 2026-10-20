#!/usr/bin/env python3
"""Guard test for T28: personal-fact mining must not capture a temporal
adverb as a relationship name.

Regression for the defect where "i had a huge fight with my best friend
today" mined ('i', 'friend today', 'today')=today (conf 0.5067, src
seed_regex), which later leaked into unrelated chitchat recall
("how are you doing today" -> "your friend is today").

Root cause: the lowercase-name fallback in mine_personal_facts accepted
any non-function, non-verb token as a name; temporal adverbs like "today"
had no closed-class gate. Fix: is_temporal_adverb() is consulted in the
name-span _close check.

RAVANA_OFFLINE=1 is set by conftest for unit tests, so this exercises the
in-process miner against a fresh PersonalFactStore (no online learning).
"""
import re
import pytest

from ravana.chat import user_model as UM
from ravana.chat.user_model import UserModel
from ravana.chat.personal_fact_store import PersonalFactStore


def _mine(text: str) -> PersonalFactStore:
    pfs = PersonalFactStore()
    um = UserModel(personal_facts=pfs)
    q = re.sub(r"\s+", " ", text).strip()
    q = UM._expand_fp_contractions(q)
    um.mine_personal_facts(q, run_correction=True)
    return pfs


@pytest.mark.parametrize("disclosure", [
    "i had a huge fight with my best friend today",
    "my best friend today was really kind to me",
    "my friend rhea called yesterday",
    "my sister is always busy",
    "i saw my cousin tomorrow at the market",
])
def test_temporal_adverb_not_captured_as_name(disclosure: str) -> None:
    """No mined fact may use a temporal adverb as the NAME component in the
    kin-relationship-without-verb path (the exact T28 shape: a relation word
    like 'friend' paired with a bare temporal adverb as its name)."""
    pfs = _mine(disclosure)
    from ravana.chat.pet_slots import is_temporal_adverb
    # The kin-relation vocabulary; a fact whose ATTRIBUTE is just a relation
    # word (relation + name, no verb) is the path T28 exploited.
    _kin = UM._KIN if hasattr(UM, "_KIN") else set()
    for key, _val in pfs.facts.items():
        if not (isinstance(key, tuple) and len(key) == 3):
            continue
        _subj, attr, name = key
        # T28 shape: attribute is a bare relation word (or relation+name where
        # the NAME is the temporal adverb), e.g. ('i', 'friend today', 'today').
        # Flag any fact whose NAME is a (near-)bare temporal adverb.
        name_toks = name.split()
        if (len(name_toks) <= 2
                and all(is_temporal_adverb(t) for t in name_toks if t)):
            # Allow it only if the attribute is NOT a pure kin/relation word
            # (i.e. the value legitimately holds a temporal predicate like
            # 'always busy' / 'called yesterday'). The defect is a relation
            # word whose NAME is the temporal adverb.
            _attr_toks = [t for t in attr.split() if t]
            _attr_is_rel = any(t in _kin or t in UM._REL_WORDS for t in _attr_toks) \
                if _attr_toks else False
            if not _attr_is_rel:
                continue  # value-path temporal predicate is legitimate
            pytest.fail(
                f"defect T28: temporal adverb {name!r} captured as NAME in "
                f"fact {key!r} from {disclosure!r}")


def test_friend_today_defect_absent() -> None:
    """The exact defective fact from T28 must not be mined."""
    pfs = _mine("i had a huge fight with my best friend today")
    bad = {
        k for k in pfs.facts
        if (isinstance(k, tuple) and len(k) == 3
            and k[0] == "i" and "friend" in k[1] and k[2] == "today")
    }
    assert bad == set(), f"defect T28 still present: {bad!r}"


def test_real_name_still_mined_with_temporal_tail() -> None:
    """A genuine lowercase name followed by a temporal tail must still mine
    the relationship (rhea is captured), and must NOT produce the defective
    ('i', 'friend today', 'today')-shaped fact where the temporal adverb
    is the NAME."""
    pfs = _mine("my friend rhea called yesterday")
    # The friendship to "rhea" must be present (attribute carries the name).
    assert any(
        isinstance(k, tuple) and len(k) == 3
        and k[0] == "i" and "rhea" in k[1]
        for k in pfs.facts
    ), "expected relationship fact carrying 'rhea' to be mined"
    # And the exact defective shape is absent: a fact whose ATTRIBUTE is a
    # kin/relation word and whose NAME is a bare temporal adverb -- that is
    # the T28 pattern ('i', 'friend today', 'today'). A fact like
    # ('i', 'rhea', 'yesterday') is legitimate (rhea is the relation head,
    # yesterday is the activity value), not the defect.
    from ravana.chat.pet_slots import is_temporal_adverb
    _kin = UM._KIN if hasattr(UM, "_KIN") else set()
    _rel = UM._REL_WORDS if hasattr(UM, "_REL_WORDS") else set()
    for k, _v in pfs.facts.items():
        if not (isinstance(k, tuple) and len(k) == 3):
            continue
        _subj, attr, name = k
        _attr_is_relation = any(t in _kin or t in _rel for t in attr.split())
        if not _attr_is_relation:
            continue  # attribute is a name, not a relation word -> legit
        name_toks = name.split()
        if (len(name_toks) <= 2
                and all(is_temporal_adverb(t) for t in name_toks if t)):
            pytest.fail(
                f"defect T28: temporal adverb {name!r} captured as NAME "
                f"under relation {attr!r} in fact {k!r}")
