"""process_turn-level regression tests for category-aware enumeration recall.

Round 2026-08-16T1745Z noted a residual limitation (measured as probe turn T59:
"name everyone in my family you've heard about" -> "noted."): RAVANA had NO path
to ENUMERATE the entities it had learned in a category. Biographical facts about
relatives and pets are mined into the PersonalFactStore, but a query that asks for
the SET of them ("name everyone in my family", "name all my pets", "who have i
told you about") fell through to a generic acknowledgement or a nonsense echo.

This capability scans the LIVE PersonalFactStore for relationship/pet facts and
lists them. It is fully store-driven (every name + relationship + detail comes
from a runtime fact RAVANA mined), uses a SHARED relation-attribute lexicon module
(relation_attrs.py) so the miner and the recaller agree by construction (mirrors
the pet_slots.py lesson), and is fail-closed (returns None when nothing is stored,
so a brand-new user gets honest uncertainty instead of an empty list).

No LLM, no retraining, no hardcoded reply. General across every relationship and
animal species the user discloses.

Verifying: RAVANA_OFFLINE=1 pytest tests/test_round_2026_08_16_1745_enum.py -v

HARDCODING NOTE: assertions check REAL stored state and the membership of the
returned entities (each disclosed name must appear), not a verbatim reply string.
The reply shape ("you've told me about ...: a, b, c.") is a thin connective around
the entity list read from the store — content comes from cognition.
"""
import os

os.environ.setdefault("RAVANA_OFFLINE", "1")

import pytest

from ravana.chat.engine import CognitiveChatEngine


def _make(tmpdir, suffix):
    return CognitiveChatEngine(
        dim=64, seed=42, baby_mode=True,
        data_dir=tmpdir, user_suffix=suffix,
    )


def _feed(eng, lines):
    for q in lines:
        eng.process_turn(q)


# ── 1. Family enumeration lists every disclosed relative ─────────────────────
def test_enumerate_family_via_process_turn(tmpdir):
    """After disclosing two relatives, 'name everyone in my family' must list
    BOTH by name (content from the store), not fall through to 'noted.'."""
    e = _make(tmpdir, "_enumfam")
    _feed(e, [
        "my grandmother Indira weaves baskets from river reeds",
        "my brother Arjun climbs mountains, mostly in Nepal",
    ])
    reply = e.process_turn("name everyone in my family you've heard about")
    r = reply.lower()
    assert "noted" not in r, f"enumeration fell through to ack: {reply!r}"
    assert "indira" in r, f"grandmother missing from family list: {reply!r}"
    assert "arjun" in r, f"brother missing from family list: {reply!r}"
    assert "grandmother" in r and "brother" in r, f"relationships not labelled: {reply!r}"


def test_enumerate_family_other_relationship_word(tmpdir):
    """A DIFFERENT relationship word proves generalization (not a hardcoded
    'grandmother'/ 'brother' branch)."""
    e = _make(tmpdir, "_enumfam2")
    _feed(e, [
        "my uncle Vivek repairs old radios for fun",
        "my niece Priya is an astronomer",
    ])
    reply = e.process_turn("list the people in my family")
    r = reply.lower()
    assert "noted" not in r, f"enumeration fell through to ack: {reply!r}"
    assert "vivek" in r and "priya" in r, f"relatives missing: {reply!r}"
    assert "uncle" in r and "niece" in r, f"relationships not labelled: {reply!r}"


# ── 2. Pet enumeration lists every disclosed animal ──────────────────────────
def test_enumerate_pets_via_process_turn(tmpdir):
    """'name all my pets' must list each disclosed animal by species + name, not
    fall through to 'noted.'."""
    e = _make(tmpdir, "_enumpet")
    _feed(e, [
        "i keep a cat named Mochi who knocks over all my plants",
        "i have a dog named Biscuit who guards the house",
    ])
    reply = e.process_turn("name all my pets")
    r = reply.lower()
    assert "noted" not in r, f"pet enumeration fell through to ack: {reply!r}"
    assert "mochi" in r and "biscuit" in r, f"pets missing: {reply!r}"
    assert "cat" in r and "dog" in r, f"species not labelled: {reply!r}"


# ── 3. General "who have i told you about" enumerates BOTH relatives and pets ─
def test_enumerate_everyone_mentioned(tmpdir):
    """'who have i told you about' must enumerate the people/animals disclosed,
    drawn from the relationship + pet facts in the store."""
    e = _make(tmpdir, "_enumall")
    _feed(e, [
        "my grandmother Indira weaves baskets",
        "my brother Arjun climbs mountains",
        "i keep a cat named Mochi",
    ])
    reply = e.process_turn("who have i told you about")
    r = reply.lower()
    assert "noted" not in r, f"enumeration fell through to ack: {reply!r}"
    assert "indira" in r and "arjun" in r and "mochi" in r, (
        f"not all disclosed entities listed: {reply!r}")


# ── 4. Fail-closed: nothing stored -> honest uncertainty (no fabricated list) ──
def test_enumerate_family_empty_is_honest(tmpdir):
    """A brand-new user with no relatives disclosed must NOT invent a family
    list; the capability returns None and the honest pipeline answers."""
    e = _make(tmpdir, "_enumempty")
    reply = e.process_turn("name everyone in my family")
    r = reply.lower()
    assert "noted" not in r, f"empty enumeration should not ack: {reply!r}"
    # Must not fabricate a relative name; "indira"/"arjun" were never disclosed.
    assert "indira" not in r and "arjun" not in r, (
        f"empty enumeration fabricated relatives: {reply!r}")
