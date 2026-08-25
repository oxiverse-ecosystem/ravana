"""
Regression tests for round 2026-08-13T1656Z — FEATURE:
relation-word -> entity recall (the forward half of the relationship index).

Residual limitation named in the round report (section 7.1): the store keys a
bare disclosure ("my brother dev works as a paramedic") UNDER THE PERSON'S
NAME (subject=name, attr="relationship", val="brother"). The reverse resolver
("who is dev to me") was added this round, but queries that name the RELATION
WORD ("what does my brother do for work", "what is my sister's job") still
echoed an unrelated episode, because the entity-attributed recall resolver
looked facts up under the word "brother"/"sister" as if it were the entity and
found nothing.

These tests assert the NEW behavior: the store exposes a forward
`resolve_relation(<relword>) -> entity-name` index, and the engine's
`_structured_recall` uses it to answer relation-word queries from the
structured store (not the episodic echo). They fail on the pre-feature code
(`resolve_relation` does not exist; the queries return None / an episode).

No authored reply prose; every answer slot is read from the durable store.
The capability is structural: it generalizes across friend/sister/brother/pet
and any spoken attribute (job/role/name/does/is), all via the shared
_RELATION_VOCAB seed set.
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(PROJ, "ravana_ml", "src"))

from ravana.chat.user_model import UserModel
from ravana.chat.personal_fact_store import PersonalFactStore
from ravana.chat.engine import CognitiveChatEngine


def _entity_facts(text):
    um = UserModel()
    um.personal_facts.facts.clear()
    um.mine_personal_facts(text, run_correction=True)
    return {
        (a, b, c): f.value
        for (a, b, c), f in um.personal_facts.facts.items()
        if not getattr(f, "superseded", False)
    }


# ── forward index on the store ──────────────────────────────────
def test_store_resolve_relation_brother():
    ef = _entity_facts("my brother dev works as a paramedic in leeds.")
    assert ("dev", "relationship", "brother") in ef, ef
    name = UserModel().personal_facts.resolve_relation("brother")
    # a fresh empty store would return None; the mined store must resolve
    um = UserModel()
    um.personal_facts.facts.clear()
    um.mine_personal_facts("my brother dev works as a paramedic in leeds.")
    got = um.personal_facts.resolve_relation("brother")
    assert got == "dev", f"expected 'dev', got {got!r}"


def test_store_resolve_relation_unknown_returns_none():
    um = UserModel()
    um.personal_facts.facts.clear()
    um.mine_personal_facts("i keep three pet raccoons.")
    # 'brother' was never disclosed -> fail closed
    assert um.personal_facts.resolve_relation("brother") is None
    # the index must NOT confuse an entity stored UNDER its own name
    # (e.g. pet 'raccoons') with a relation word
    assert um.personal_facts.resolve_relation("raccoons") is None


def test_store_resolve_relation_sister():
    um = UserModel()
    um.personal_facts.facts.clear()
    um.mine_personal_facts("my sister meera is a marine biologist.")
    assert um.personal_facts.resolve_relation("sister") == "meera"


# ── engine-level relation-word recall (the user-facing capability) ──
def _new_engine(suffix):
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               user_suffix=suffix)


def test_brother_job_from_store_not_episode():
    eng = _new_engine("feat_bro_job")
    eng.process_turn("my brother dev works as a paramedic in leeds.")
    # the high-value phrasing that regressed to an episode echo before:
    ans = eng._structured_recall("what does my brother do for work?")
    assert ans is not None, "relation-word query must resolve from store"
    assert "paramedic" in ans.lower(), ans
    assert "dev" not in ans.lower(), ans  # framed by relation word, not name
    # synonym phrasing must also resolve
    ans2 = eng._structured_recall("what is my brother's job?")
    assert ans2 is not None and "paramedic" in ans2.lower(), ans2


def test_sister_job_from_store_not_episode():
    eng = _new_engine("feat_sis_job")
    eng.process_turn("my sister meera is a marine biologist who studies squid.")
    ans = eng._structured_recall("what is my sister job?")
    assert ans is not None, "sister job must resolve from store"
    assert "marine biologist" in ans.lower(), ans
    # possessive synonym
    ans2 = eng._structured_recall("what's my sister's job?")
    assert ans2 is not None and "marine biologist" in ans2.lower(), ans2


def test_relation_word_query_returns_none_when_unstored():
    eng = _new_engine("feat_unstored")
    eng.process_turn("i keep three pet raccoons that steal my tomatoes.")
    # no brother/sister ever disclosed -> honest fail-closed, not a confab
    ans = eng._structured_recall("what does my brother do for work?")
    assert ans is None, f"unstored relation must return None, got {ans!r}"


def test_reverse_and_forward_agree():
    """The forward index is the structural complement of the reverse lookup:
    'who is dev to me' and 'what does my brother do' must both reference the
    same stored entity, with content from the store (not an episode)."""
    eng = _new_engine("feat_agree")
    eng.process_turn("my brother dev works as a paramedic in leeds.")
    rev = eng._structured_recall("who is dev to me?")
    fwd = eng._structured_recall("what does my brother do for work?")
    assert rev is not None and "brother" in rev.lower(), rev
    assert fwd is not None and "paramedic" in fwd.lower(), fwd
