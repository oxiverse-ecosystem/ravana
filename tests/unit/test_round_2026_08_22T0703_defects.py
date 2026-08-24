import os, sys
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
sys.path.insert(0, PROJ)
import tempfile, glob
from ravana.chat.engine import CognitiveChatEngine

def fresh_engine(suffix):
    for f in glob.glob(f"weights/ravana_weights{suffix}.pkl") + glob.glob(f"weights/ravana_usermodel{suffix}.pkl"):
        try: os.remove(f)
        except OSError: pass
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)

def facts_i(eng):
    pf = eng.user_model.personal_facts.facts
    return [(k, getattr(v,"value",v)) for k,v in pf.items()
            if isinstance(k,tuple) and len(k)==3 and k[0]=="i"]

def test_d1_pet_activity():
    """Pet activity disclosure must store + recall the activity (DEFECT D1)."""
    eng = fresh_engine("test_d1_pet")
    eng.process_turn("my ferret Pip hides my car keys under the couch cushions")
    fs = facts_i(eng)
    act = [v for k,v in fs if k[1]=="ferret_activity"]
    assert act, f"pet activity not mined: {fs}"
    assert "hides car keys" in act[0], f"activity value wrong: {act}"
    r1 = eng._structured_recall("which of my pets hides things in the couch?")
    assert r1 and "hides car keys" in r1, f"which-pet query failed: {r1}"
    r2 = eng._structured_recall("what does my ferret do?")
    assert r2 and "hides car keys" in r2, f"what-does-my-pet failed: {r2}"
    r3 = eng._structured_recall("what does pip do with the car keys?")
    assert r3 and "hides car keys" in r3, f"by-name query failed: {r3}"
    print("PASS test_d1_pet_activity:", r1, "|", r2, "|", r3)

def test_d2_headless_possessive_name():
    """Headless possessive 'my daughter name is ingrid' must store a clean
    entity-scoped name fact and NOT a malformed 'daughter name is' fact
    (DEFECT D2)."""
    eng = fresh_engine("test_d2_hp")
    eng.process_turn("my daughter name is ingrid, she's nine and already codes little games")
    fs = facts_i(eng)
    malformed = [k for k,v in fs if k[1]=="daughter name is"]
    assert not malformed, f"malformed 'daughter name is' fact stored: {fs}"
    # clean entity-scoped fact must exist
    ent = [ (k,v) for k,v in eng.user_model.personal_facts.facts.items()
            if isinstance(k,tuple) and len(k)==3 and k[0]=="daughter" and k[1]=="name" ]
    assert ent, f"no entity-scoped daughter name fact: {fs}"
    assert ent[0][1].value.lower().startswith("ingrid"), f"name value wrong: {ent}"
    # reverse-name recall must work
    r = eng._structured_recall("who is ingrid to me?")
    assert r and "daughter" in r.lower(), f"reverse-name 'who is ingrid' failed: {r}"
    print("PASS test_d2_headless_possessive_name:", ent[0][1].value, "|", r)

def test_compound_query_decomposition():
    """RESIDUAL (round 2026-08-22T0703Z): a compound interrogative that asks
    BOTH a pet's name AND its activity must answer BOTH conjuncts, not just the
    first. Before this fix the query returned only the name ('your ferret is
    pip.') and dropped 'what does he do with my keys'. The capability is
    general (multi-part query decomposition) — it reuses the existing
    store-driven resolver per clause, so it also covers 'who is X and what do
    they do' style compounds."""
    eng = fresh_engine("test_compound")
    eng.process_turn("my pet ferret Pip hides my car keys under the couch")
    # The single-clause queries must still work (D1 regression guard).
    r_name = eng._structured_recall("what's my ferret's name?")
    r_act = eng._structured_recall("what does he do with my keys?")
    assert r_name and "pip" in r_name.lower(), f"name clause failed: {r_name}"
    assert r_act and "hides car keys" in r_act, f"activity clause failed: {r_act}"
    # The COMPOUND must answer BOTH.
    r = eng._structured_recall(
        "what's my ferret's name and what does he do with my keys?")
    assert r is not None, "compound query returned None"
    assert "pip" in r.lower(), f"compound missing name: {r}"
    assert "hides car keys" in r.lower(), f"compound missing activity: {r}"
    # Exactly one coordinating 'and', no double period.
    assert r.count(" and ") == 1, f"compound join wrong: {r!r}"
    assert not r.endswith(".."), f"compound double period: {r!r}"
    print("PASS test_compound_query_decomposition:", r)

if __name__ == "__main__":
    test_d1_pet_activity()
    test_d2_headless_possessive_name()
    test_compound_query_decomposition()
    print("ALL PASS")
