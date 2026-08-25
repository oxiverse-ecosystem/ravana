"""Regression tests for round 2026-08-14T0103Z — Defect B: malformed
'does' / 'event' personal facts from framer/temporal/negation words.

The open-class verb miner (and the seeded ACTIVITY/EVENT blocks) treated
ANY word after "i" as the verb, so framer / temporal / negation words
preceding the real activity verb were stored as garbage activities:
  "i won't buy fish"        -> does="won't buy fish"
  "i used to love ..."      -> does="used love"
  "i first lit a kiln"      -> does="first lit"
  "i mis-spoke earlier"     -> does="mis-spoke earlier"
  "i probably won't print"  -> does="probably won't print"
Real activities ("i gather morels", "i count meteor showers", "i keep a
sourdough starter") must still be captured. No hardcoded reply; tests the
miner's stored state, not a generated string.
"""

import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _um():
    sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
    from ravana.chat.user_model import UserModel
    return UserModel()


def _does(t):
    um = _um()
    um.mine_personal_facts(t)
    return [v.value for k, v in um.personal_facts.facts.items()
            if isinstance(k, tuple) and k[1] == "does"]


def test_framer_words_not_stored_as_activity():
    cases = [
        ("i still won't buy fish at the market, but i'll eat the mackerel i catch off the rocks.", "won't"),
        ("no, my cat fig is a russian blue, not a tabby -- i mis-spoke earlier.", "mis-spoke"),
        ("i used to love the smell of wet clay but the damp is now wrecking my hands.", "used"),
        ("seriously, if a photo isn't on film i probably won't print it.", "probably"),
        ("this november marks eight years since i first lit a kiln.", "first"),
        # modality / auxiliaries must never become a 'does' activity
        ("i should handle the firing schedule myself.", "should handle"),
        ("i can lift the kiln shelf when it's cool.", "can lift"),
        ("i will finish the dinner set by friday.", "will finish"),
    ]
    for t, bad in cases:
        got = _does(t)
        assert all(bad not in d for d in got), \
            f"{t!r} produced bad 'does': {got!r}"
    # questions must not leak any activity facts at all
    got = _does("how do you think i should handle that?")
    assert not got, \
        f"Question produced 'does' facts: {got!r}"


def test_real_activities_still_captured():
    cases = [
        ("i gather morels under the beeches but only after the last frost.", "gather morels"),
        ("i count meteor showers from the gallery every august.", "count meteor showers"),
        ("i keep a sourdough starter i named bishop and bake before the morning firing.", "keep sourdough starter"),
        ("i run about four trail races a year and i've done twelve marathons.", "run four trail races"),
    ]
    for t, exp in cases:
        got = _does(t)
        assert any(exp in d for d in got), \
            f"{t!r} lost real activity (expected {exp!r}), got {got!r}"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
