"""Regression tests for round 2026-08-09g defects (D1/D2/D3).

- D1 confabulation: biographical / possessive-entity / count / self-introspection
  recall must resolve from the LIVE store / self-model, never echo an unrelated
  stored user utterance as "you told me earlier: ...".
- D2 degenerate ack: a confession with no extractable fact must get a
  state-driven reflective ack, not the hardcoded "got it - thanks for telling
  me." hollow template (verified by counting that the hollow string is NOT the
  reply for genuine disclosures).
- D3 dropped correction: "it's seven hives now, i split one" must supersede the
  prior count fact so a later "how many hives do i have" returns the NEW number.

Run: RAVANA_OFFLINE=1 python -m pytest tests/unit/test_recall_confabulation_2026g.py -q
"""
import os
os.environ.setdefault("RAVANA_OFFLINE", "1")

import sys, time
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (_PROJ, f"{_PROJ}/ravana_ml/src", f"{_PROJ}/ravana/src", f"{_PROJ}/ravana-v2/src"):
    sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine


def _new_engine(suffix="test2026g"):
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)
    return eng


def test_d1_possessive_entity_recall():
    """'what's my partner's name' must return the partner's name, not the user's."""
    eng = _new_engine("t_d1_ent")
    eng.process_turn("i'm bram, by the way")
    eng.process_turn("my partner's name is wren, she helps me with the hive inspections")
    # probe: own name should be bram; partner name should be wren (not bram)
    r = eng.process_turn("what's my partner's name again? i forget if i told you")
    assert "wren" in r.lower(), f"expected partner name wren, got: {r!r}"
    assert "bram" not in r.lower(), f"partner recall leaked user's own name: {r!r}"


def test_d1_where_do_i_keep_recall():
    """'where do i keep the bees' must resolve from the 'does' fact, not confabulate."""
    eng = _new_engine("t_d1_keep")
    eng.process_turn("i keep six hives of bees on a rooftop above the bakery on fifth")
    eng.process_turn("i restore old mechanical typewriters as a side trade")
    r = eng.process_turn("where do i keep the bees, exactly? remind me what i said")
    assert "six hives" in r.lower(), f"expected 'keep six hives', got: {r!r}"
    # must NOT be the typewriter fact
    assert "typewriter" not in r.lower(), f"confabulated typewriter fact: {r!r}"


def test_d1_count_recall():
    """'how many hives do i have' must read the count from the does fact."""
    eng = _new_engine("t_d1_cnt")
    eng.process_turn("i keep six hives of bees on a rooftop")
    r = eng.process_turn("how many hives do i have?")
    assert "six" in r.lower(), f"expected 'six' hives, got: {r!r}"


def test_d1_self_introspection_not_user_fact():
    """A RAVANA-about-RAVANA question must not echo a USER fact (honey opinion)."""
    eng = _new_engine("t_d1_self")
    eng.process_turn("i think raw honey straight from the comb beats any store syrup")
    r = eng.process_turn("what was your read on whether you're really thinking, the line you gave me?")
    assert "honey" not in r.lower(), f"self-introspection echoed a USER fact: {r!r}"
    # should be about RAVANA, not a user fact
    assert ("ravana" in r.lower() or "i'm" in r.lower() or "myself" in r.lower()
            or "thinking" in r.lower() or "process" in r.lower()), \
        f"self-introspection reply not about RAVANA: {r!r}"


def test_d3_count_correction_supersedes():
    """'it's seven hives now, i split one' must update the count fact."""
    eng = _new_engine("t_d3_cnt")
    eng.process_turn("i keep six hives of bees on a rooftop")
    eng.process_turn("oh wait, i was wrong earlier, it's seven hives now, i split one last week")
    # the correction should be acknowledged (not "ok, noted: wait")
    r_ack = eng._last_responses[-1].lower() if eng._last_responses else ""
    assert "wait" not in r_ack or "noted: wait" not in r_ack, f"correction dropped: {r_ack!r}"
    # later count query must reflect the NEW number
    r2 = eng.process_turn("so how many hives do i have now, after the split?")
    assert "seven" in r2.lower(), f"expected updated count 'seven', got: {r2!r}"


def test_d2_hollow_ack_not_on_confession():
    """A pure confession with no extractable fact must NOT return the hollow ack."""
    eng = _new_engine("t_d2_hollow")
    # force a low-valence state so the reflective ack picks the negative band
    try:
        eng.emotion.state.valence = -0.5
    except Exception:
        pass
    r = eng.process_turn("i felt hollow pulling out the dead brood, like a small funeral")
    # the hollow template must not be the reply
    assert r.strip() != "got it — thanks for telling me.", \
        f"degenerate hollow ack still firing on confession: {r!r}"
    # the reflective ack should reference the heaviness (state-derived)
    assert ("heavy" in r.lower() or "hard" in r.lower() or "trusting" in r.lower()
            or "listening" in r.lower() or "taking it in" in r.lower()), \
        f"reply not state-reflective: {r!r}"


if __name__ == "__main__":
    t0 = time.time()
    for fn in (test_d1_possessive_entity_recall, test_d1_where_do_i_keep_recall,
               test_d1_count_recall, test_d1_self_introspection_not_user_fact,
               test_d3_count_correction_supersedes, test_d2_hollow_ack_not_on_confession):
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            print(f"ERR  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"done in {time.time()-t0:.1f}s")
