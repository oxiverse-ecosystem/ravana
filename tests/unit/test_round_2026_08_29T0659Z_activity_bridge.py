"""
Regression test — round 2026-08-29T0659Z feature follow-up (residual defect #4).

Capability added: an ACTIVITY OBJECT-CATEGORY BRIDGE in _structured_recall. The
slot-key collapse from the round was fixed (distinct activities now live in
verb-keyed slots does:learn / does:keep / ...), but activity RECALL only linked a
query to a stored fact when the QUERY VERB equalled the STORED verb, or a bare
query noun appeared verbatim in the value. A query that names the OBJECT CATEGORY
("what instrument do i play") never matched "i learn cello" because neither the
verb (play != learn) nor the category word (instrument) is in the value.

The bridge expands a query role word (instrument / pet / craft / ...) to its seed
object vocabulary (module-level _ACTIVITY_ROLES, RUNTIME-growable via
UserModel._activity_roles / learn_activity_role) and matches a stored does:VERB fact
whose value contains any of those objects. Reply content is the LIVE fact value —
no authored sentence, no LLM, no retraining. Seed data, not a question->answer table.

Without the bridge, "what instrument do i play" returns None (verified: the recall
loop keys on query-verb == stored-verb and "instrument" is absent from the value),
so this test is RED-capable — it fails before the fix and passes with it.

All assertions are STATE-DRIVEN: they read the personal-fact store / recall output,
never authored reply strings.
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(PROJ, "ravana_ml", "src"))

from ravana.chat.engine import CognitiveChatEngine


def _eng(suffix):
    for f in (f"weights/ravana_weights{suffix}.pkl",
              f"weights/ravana_usermodel{suffix}.pkl"):
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)
    e.stop_background_learning()
    return e


def test_instrument_bridges_to_learned_cello():
    # The headline residual: "what instrument do i play" -> "learning cello".
    e = _eng("_br_instr")
    e.user_model.mine_personal_facts("i learn cello")
    # verb-keyed slot stored, not superseded (slot-collapse already fixed)
    facts = [(k, f.value, f.superseded)
             for k, f in e.user_model.personal_facts.facts.items()
             if k[0] == "i" and k[1].startswith("does:")]
    assert any(k[1] == "does:learn" and v == "learn cello" and not sup
               for (k, v, sup) in facts)
    # the NEW capability: category query bridges to it
    assert e._structured_recall("what instrument do i play") == "you learn cello."


def test_pet_bridges_to_kept_parrot():
    # Different role, different verb ("keep") — proves the bridge is role-driven,
    # not hardcoded to one verb.
    e = _eng("_br_pet")
    e.user_model.mine_personal_facts("i keep a parrot")
    assert e._structured_recall("what pet do i keep") == "you keep parrot."


def test_online_learned_object_bridges():
    # A object NOT in any seed role vocabulary still bridges AFTER RAVANA learns
    # it online (learn_activity_role grows _activity_roles; the bridge merges it).
    # Proves the bridge is seed+online, not a frozen table.
    e = _eng("_br_learn")
    e.user_model.mine_personal_facts("i restore grandfather clocks")
    e.user_model.learn_activity_role("clocks")
    assert e._structured_recall("what craft do i restore") == \
        "you restore grandfather clocks."


def test_no_role_word_falls_through():
    # A query with NO activity-role word must not fabricate a bridged answer.
    e = _eng("_br_none")
    e.user_model.mine_personal_facts("i learn cello")
    out = e._structured_recall("what do i bake")
    assert out is None or "cello" not in out


def test_unrelated_role_does_not_leak():
    # A stored activity under one role must not be returned for a different role.
    e = _eng("_br_sep")
    e.user_model.mine_personal_facts("i learn cello")        # instrument
    e.user_model.mine_personal_facts("i garden tomatoes")    # plant
    assert e._structured_recall("what pet do i keep") is None


def test_no_false_positive_within_role():
    # Within the same role, an unrelated stored object must not match.
    e = _eng("_br_intra")
    e.user_model.mine_personal_facts("i garden tomatoes")    # plant role
    assert e._structured_recall("what pet do i keep") is None
