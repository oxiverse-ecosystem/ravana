"""process_turn-level regression tests for the D7 fix (round 2026-08-16T1745Z).

D7 closes the recurring L2 residual limitation: relationship-ACTIVITY disclosures
("my grandmother Indira weaves baskets", "my brother Arjun climbs mountains")
were never mined, so every cued recall of that family member echoed an unrelated
fact. The fix (a) mines ANY "my <kin> <Name> <verb> <object>" disclosure as a
combined-attr personal fact, (b) renders verb-phrase values WITHOUT a spurious
copula in both the cued-recall path AND the disclosure-acknowledgement path, and
(c) adds a reverse-NAME lookup so "who is indira to me" resolves the relationship.

These tests drive the FULL CognitiveChatEngine.process_turn flow (fresh engine,
fixed seed, RAVANA_OFFLINE=1, isolated data_dir). They never call the miners
directly, so a routing regression (miner unreachable from the live path) is caught
here, not hidden. Mirrors the coverage lesson from round 2026-08f's independent
audit.

Verifying coverage: RAVANA_OFFLINE=1 pytest tests/test_round_2026_08_16_1745_d7.py -v

HARDCODING NOTE: every assertion below checks REAL stored state (the
PersonalFactStore) and the GRAMMAR of the rendered reply (absence of a spurious
"is <verb>" copula). No reply string is asserted verbatim; we only assert that the
relationship word + activity appear, which proves the content comes from the store.
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


def _fact_value(eng, attr):
    """Return the (non-superseded) stored value for subject 'i', attr, or None."""
    pf = eng.user_model.personal_facts.facts
    for (s, a, _v), f in pf.items():
        if s == "i" and a.lower() == attr.lower() and not getattr(f, "superseded", False):
            return f.value
    return None


# ── 1. Relationship-activity disclosure is MINED (real path) ──────────────────
def test_relationship_activity_mined_via_process_turn(tmpdir):
    """'my grandmother Indira weaves baskets from river reeds' must store a
    combined-attr fact ('grandmother indira' -> 'weaves baskets') through the
    real process_turn path, not just the direct miner."""
    e = _make(tmpdir, "_d7mine")
    e.process_turn("my grandmother Indira weaves baskets from river reeds")
    val = _fact_value(e, "grandmother indira")
    assert val is not None, "relationship-activity fact was not mined"
    assert val == "weaves baskets", f"unexpected stored value: {val!r}"


# ── 2. Disclosure ACK renders WITHOUT a spurious copula (the run-336 gap) ─────
def test_ack_renders_verb_phrase_without_copula(tmpdir):
    """The acknowledgement must read 'your grandmother indira weaves baskets',
    NOT 'your grandmother indira is weaves baskets' (the copula bug the
    acknowledgement path still had after run 336's recall fix)."""
    e = _make(tmpdir, "_d7ack")
    reply = e.process_turn("my grandmother Indira weaves baskets from river reeds")
    r = reply.lower()
    assert "is weaves" not in r, f"spurious copula in ack: {reply!r}"
    assert "grandmother indira weaves baskets" in r, f"ack missing activity: {reply!r}"


def test_ack_name_less_relationship_without_copula(tmpdir):
    """Name-less disclosure 'my sister climbs rocks' acks 'your sister climbs
    rocks' (attr is just the relation word, still reachable from 'my sister')."""
    e = _make(tmpdir, "_d7acknl")
    reply = e.process_turn("my sister climbs rocks")
    r = reply.lower()
    assert "is climbs" not in r, f"spurious copula in ack: {reply!r}"
    assert "your sister climbs rocks" in r, f"ack missing activity: {reply!r}"


# ── 3. Cued recall resolves the relationship-activity fact ───────────────────
def test_recall_grandmother_activity_via_process_turn(tmpdir):
    """After the disclosure, 'what's my grandmother's name and what does she
    make?' must recall the stored activity, grammatically ('your grandmother
    indira weaves baskets'), not echo an unrelated fact."""
    e = _make(tmpdir, "_d7recg")
    e.process_turn("my grandmother Indira weaves baskets from river reeds")
    reply = e.process_turn("what's my grandmother's name and what does she make?")
    r = reply.lower()
    assert "is weaves" not in r, f"spurious copula in recall: {reply!r}"
    assert "grandmother indira weaves baskets" in r, f"recall missing activity: {reply!r}"


def test_recall_brother_hobby_via_process_turn(tmpdir):
    """'does my brother have a hobby?' -> 'your brother arjun climbs mountains'."""
    e = _make(tmpdir, "_d7recb")
    e.process_turn("my brother Arjun climbs mountains, mostly in Nepal")
    reply = e.process_turn("does my brother have a hobby?")
    r = reply.lower()
    assert "is climbs" not in r, f"spurious copula in recall: {reply!r}"
    assert "brother arjun climbs mountains" in r, f"recall missing activity: {reply!r}"


# ── 4. Reverse-NAME lookup (general, not per-name table) ─────────────────────
def test_reverse_name_lookup_resolves_relationship(tmpdir):
    """'who is indira to me?' must resolve the relationship from the combined-attr
    fact ('grandmother indira') and answer 'your grandmother' — general across
    every relationship the user named, no per-name table."""
    e = _make(tmpdir, "_d7rev")
    e.process_turn("my grandmother Indira weaves baskets from river reeds")
    reply = e.process_turn("who is indira to me?")
    assert "grandmother" in reply.lower(), f"reverse-name lookup failed: {reply!r}"


def test_reverse_name_lookup_other_relationship(tmpdir):
    """Same mechanism for a DIFFERENT relationship word proves generalization
    (not a hardcoded 'indira' branch)."""
    e = _make(tmpdir, "_d7rev2")
    e.process_turn("my uncle Vivek repairs old radios for fun")
    reply = e.process_turn("who is vivek to me?")
    assert "uncle" in reply.lower(), f"reverse-name lookup failed for other kin: {reply!r}"
    val = _fact_value(e, "uncle vivek")
    assert val is not None and "repairs" in val, f"uncle activity not mined: {val!r}"
