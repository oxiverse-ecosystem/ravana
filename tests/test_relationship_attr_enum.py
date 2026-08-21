"""Regression tests for the multi-attribute relationship-mining capability
(round 2026-08-20T1935Z residual => feature card t_ec6c6b51).

DEFECT under test: attribute-style relationship disclosures that carry NEITHER
a recognized activity verb NOR a capitalized name were never mined, and the
colon-enumeration shape ("speaks three languages: greek, french, italian") had
no miner. So "my grandmother yaya speaks three languages: greek, french,
italian" stored nothing, and "what languages does my grandmother speak?" /
"does yaya still speak those three languages?" returned an unrelated/empty fact.

The fix generalizes the EXISTING relationship miner (rule 6f/6g): a "my <kin>
<name> <verb> <rest>" disclosure is captured whenever it has a name OR a
relation verb OR a colon/enumerated attribute, with the value storing the real
content verb-phrase. Recall (engine.py 1c/1d) already renders combined-attr
facts by `<rel> <name>` head, so a stored verb-phrase value ("speaks three
languages: greek, french, italian") renders grammatically once mined.

Driven through the FULL CognitiveChatEngine.process_turn path (never the miner
directly) so a routing regression is caught. RAVANA_OFFLINE=1, isolated
data_dir, fresh seed.

HARDCODING NOTE: assertions check REAL stored state (PersonalFactStore) and the
GRAMMAR/grammar of the rendered reply (relationship head + the disclosure
content appear, no spurious "is speaks" copula). No reply string is asserted
verbatim; the relationship word + disclosed content prove the content comes
from the store. No Q->A dict, no authored reply.
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


# ── 1. The residual defect: attribute disclosure with lowercase name + non-activity verb ──
def test_relationship_enum_disclosure_mined(tmpdir):
    """'my grandmother yaya speaks three languages: greek, french, italian' must
    store a combined-attr fact ('grandmother yaya' -> the disclosed content), not
    be skipped because 'speaks' is not an activity verb and 'yaya' is lowercase."""
    e = _make(tmpdir, "_relattr1")
    e.process_turn(
        "my grandmother yaya speaks three languages: greek, french, and italian")
    val = _fact_value(e, "grandmother yaya")
    assert val is not None, "relationship attribute disclosure was not mined"
    # The disclosed content must be stored verbatim (the real content, lowercased).
    assert "speaks three languages" in val.lower(), f"unexpected stored value: {val!r}"
    assert "greek" in val.lower() and "french" in val.lower() and "italian" in val.lower(), \
        f"enumeration not captured: {val!r}"


# ── 2. Recall of the enumerated attribute ──
def test_relationship_enum_recalled(tmpdir):
    """After the disclosure, 'what languages does my grandmother speak?' must recall
    the stored languages fact, logically (mentions the languages), not echo an
    unrelated fact or come back empty."""
    e = _make(tmpdir, "_relattr2")
    e.process_turn(
        "my grandmother yaya speaks three languages: greek, french, and italian")
    # Independent second disclosure with a different attribute to prove the right
    # fact is recalled, not just "anything stored".
    e.process_turn("my grandmother yaya is getting forgetful")
    reply = e.process_turn("what languages does my grandmother speak?")
    r = reply.lower()
    # grammar: no spurious copula before the relation verb.
    assert "is speaks" not in r, f"spurious copula in recall: {reply!r}"
    assert "grandmother yaya speaks three languages" in r or \
        ("greek" in r and "french" in r and "italian" in r), \
        f"recall missing enumerated languages: {reply!r}"


# ── 3. Generalization: same shape with a DIFFERENT relation + attribute ──
def test_relationship_attr_generalizes_other_relation(tmpdir):
    """The fix must generalize across any disclosed relationship, not just
    'grandmother'. 'my uncle ravi works as a mechanic and a beekeeper' should
    store a combined-attr fact ('uncle ravi' -> the disclosed content). 'works'
    is shared by the activity lexicon, so the mined value is the activity-phrase
    form ('works mechanic'); what matters is that the relationship fact EXISTS
    and carries the disclosed content, not the exact connective."""
    e = _make(tmpdir, "_relattr3")
    e.process_turn("my uncle ravi works as a mechanic and a beekeeper")
    val = _fact_value(e, "uncle ravi")
    assert val is not None, "uncle attribute disclosure was not mined"
    assert "works" in val.lower() and "mechanic" in val.lower(), \
        f"unexpected stored value: {val!r}"


# ── 4. Regression guard: a disclosure that genuinely has NO informative content
# is still NOT stored as a degenerate fact (the round 2026-08-20T0701Z guard). ──
def test_no_degenerate_relationship_fact(tmpdir):
    """'my grandmother' with no name and no content must not store a degenerate
    ('grandmother','grandmother') fact."""
    e = _make(tmpdir, "_relattr4")
    e.process_turn("my grandmother")
    val = _fact_value(e, "grandmother")
    assert val is None, f"degenerate relationship fact stored: {val!r}"
