"""
Regression test for round 2026-08-09T1953Z fact-mining fix.

Asserts that mine_personal_facts captures first-person activities stated as
gerunds / continuous tenses / irregular past forms / wrapper verbs, and
firsthand experience (event) disclosures — the classes that previously fell
through to the hollow "got it — thanks for telling me." ack and became
unrecallable. These are STATE-DRIVEN captures (the verb is a seed
vocabulary; the stored object is the resolved content head), NOT hardcoded
reply strings.
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(PROJ, "ravana_ml", "src"))

from ravana.chat.user_model import UserModel


def _capture(text):
    um = UserModel()
    um.personal_facts.facts.clear()
    um.mine_personal_facts(text, run_correction=True)
    return {
        (a, b): f.value
        for (a, b, c), f in um.personal_facts.facts.items()
        if not getattr(f, "superseded", False)
    }


def test_gerund_activity_captured():
    caps = _capture("i throw pots at a community studio on thursdays")
    assert ("i", "does") in caps, caps
    assert "throw pots" in caps[("i", "does")], caps


def test_continuous_tense_activity_captured():
    caps = _capture("i've been training a juniper bonsai for six years")
    assert ("i", "does") in caps, caps
    assert "juniper bonsai" in caps[("i", "does")], caps


def test_irregular_past_activity_captured():
    caps = _capture("i keep a saltwater reef tank in the living room")
    assert ("i", "does") in caps, caps
    assert "reef tank" in caps[("i", "does")], caps


def test_event_past_doubled_consonant_captured():
    # "repotted" = repot + ted (doubled-consonant irregular past)
    caps = _capture("i repotted the juniper this spring and found a root")
    assert ("i", "event") in caps, caps
    assert "repot juniper" in caps[("i", "event")], caps


def test_event_irregular_past_captured():
    caps = _capture("the power went out and i lost a favia coral to heat")
    assert ("i", "event") in caps, caps
    assert "lost favia coral" in caps[("i", "event")], caps


def test_wrapper_verb_captured():
    caps = _capture("i started growing air plants on the studio windowsill")
    assert ("i", "does") in caps, caps
    assert "growing air plants" in caps[("i", "does")], caps


def test_clause_boundary_stops_object():
    # The object must stop at a clause boundary, not swallow the trailing
    # "but ..." tail (regression: used to store the whole clause).
    caps = _capture("i love the crazed glaze - but i actually prefer a clean one")
    # love -> stance (opinion), not a personal_fact; the point is no malformed
    # "does"/"event" fact with a trailing "but" should appear.
    for (subj, attr), val in caps.items():
        assert "but" not in val, f"object leaked clause boundary: {val!r}"
