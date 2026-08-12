"""process_turn-level regression tests for round 2026-08f (t_f04e8f53).

Independent audit t_cd6396f6 found ZERO coverage of any of the round's four
fixes through the REAL ``CognitiveChatEngine.process_turn`` path. The round
verified each miner by direct function call and reported stance deltas that are
unreproducible from the live engine -- which is exactly why the concession fix
shipped broken and the possessive-ack gaps went unnoticed.

These tests drive the full process_turn flow (fresh engine, fixed seed,
RAVANA_OFFLINE=1). They do NOT call the miners directly for the reversal/ack
behaviors, so a routing regression (miner unreachable from the live path) is
caught here, not hidden.

Each test corresponds to one of the four audited fixes:

  1. Concession reversal        (depends on t_58d5f3ac / commit d363c6a)
  2. Possessive disclosure ack  (depends on t_79d3621d / commit bdaec4a)
  3. Location clause trim       (kept green; round-2026-08f, commit 3e74925)
  4. Opinion formation          (comparative/superlative/dismissive;
                                round-2026-08f, commit ffa80f1)

Verifying coverage: RAVANA_OFFLINE=1 pytest tests/test_round_2026_08f_regression.py
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


# ── Marker strings that indicate the turn was ROUTED into the emotional-support
# / empathy path instead of acknowledged as a fact. Used ONLY to assert their
# ABSENCE on benign disclosures (we never assert on a hardcoded reply sentence).
_EMPATHY_OPENERS = (
    "i hear you", "i'm here for", "i'm so sorry", "feeling rough",
    "feeling sad is hard", "what happened", "what set it off",
)


def _is_empathy_reply(reply: str) -> bool:
    r = (reply or "").lower()
    return any(op in r for op in _EMPATHY_OPENERS)


# ── 1. Concession reversal ──────────────────────────────────────────────────
def test_concession_reversal_moves_held_stance_to_neutral_via_process_turn(tmpdir):
    """A natural concession (\"i thought X but Y\") must route through the real
    process_turn path, reach the stance-reversal miner, and move the HELD topic
    X toward neutral (0.0) -- not stay stale, and not fabricate a reversal about
    the new preference Y (a topic with no prior stance)."""
    e = _make(tmpdir, "_conc")
    e.user_model.opinions.stances.clear()
    e.user_model.opinions.express_stance("sea", polarity=0.7, confidence=0.8)

    reply = e.process_turn(
        "i thought the sea was a good teacher but actually i prefer mountains now")

    sea = e.user_model.opinions.stances.get("sea")
    assert sea is not None, "held 'sea' stance must still exist after concession"
    assert sea.polarity == 0.0, f"sea should reverse to neutral, got {sea.polarity}"
    # ack references the conceded topic, NOT the fabricated 'mountains' reversal.
    assert "sea" in (reply or "").lower(), reply
    assert "mountains" not in (reply or "").lower(), reply


def test_concession_without_prior_stance_does_not_fabricate(tmpdir):
    """A concession about a topic RAVANA never held a stance on must NOT fabricate
    a reversal ack for the new-preference clause."""
    e = _make(tmpdir, "_concnone")
    e.user_model.opinions.stances.clear()
    reply = e.process_turn(
        "i thought climbing was my thing but actually i prefer the beach now")
    assert "beach" not in (reply or "").lower(), reply
    assert "changed your mind about beach" not in (reply or "").lower(), reply


def test_concession_ack_not_routed_to_support(tmpdir):
    """The concession turn is an attitude recode, not an emotional-support
    disclosure -- it must be acked, not met with comfort."""
    e = _make(tmpdir, "_concsup")
    e.user_model.opinions.stances.clear()
    e.user_model.opinions.express_stance("cities", polarity=-0.6, confidence=0.8)
    reply = e.process_turn(
        "i used to think cities were thrilling but now i find the countryside calmer")
    assert not _is_empathy_reply(reply), reply
    cities = e.user_model.opinions.stances.get("cities")
    assert cities is not None and cities.polarity == 0.0, cities


# ── 2. Possessive disclosure ack ─────────────────────────────────────────────
def test_possessive_disclosure_acked_not_routed_to_support(tmpdir):
    """A possessive disclosure (partner / dog / child) must be ACKED as a stored
    fact, not routed into the emotional-support / empathy path when no distress
    is present. The ack must reference the CORRECT ENTITY."""
    e = _make(tmpdir, "_poss")

    # 1) Partner disclosure -> acked, entity = partner (not "your").
    reply1 = e.process_turn("my partner's name is Pell")
    assert not _is_empathy_reply(reply1), reply1
    assert "your name is pell" not in (reply1 or "").lower(), reply1
    assert "partner" in (reply1 or "").lower(), reply1
    fact1 = e.user_model.personal_facts.get("partner", "name")
    assert fact1 is not None and fact1.value == "pell", fact1

    # 2) Dog disclosure -> acked as a fact, not met with comfort.
    reply2 = e.process_turn("my dog is a sheepdog named Cairn")
    assert not _is_empathy_reply(reply2), reply2
    assert "dog" in (reply2 or "").lower(), reply2
    fact2 = e.user_model.personal_facts.get("i", "dog")
    assert fact2 is not None and "cairn" in fact2.value.lower(), fact2

    # 3) Child disclosure -> acked as a fact, not met with comfort.
    reply3 = e.process_turn("my child is a curious kid named Sam")
    assert not _is_empathy_reply(reply3), reply3
    assert "child" in (reply3 or "").lower(), reply3
    fact3 = e.user_model.personal_facts.get("i", "child")
    assert fact3 is not None and "sam" in fact3.value.lower(), fact3


def test_genuine_distress_still_routes_to_empathy(tmpdir):
    """Guard against over-firing the misfire gate: a disclosure that DOES contain
    a real suffering/distress signal must still reach empathy."""
    e = _make(tmpdir, "_distress")
    assert _is_empathy_reply(e.process_turn("my dog died")), "bereavement -> empathy"
    assert _is_empathy_reply(e.process_turn("i am sad")), "present distress -> empathy"
    assert _is_empathy_reply(e.process_turn("my friend is hurting")), "other-suffering -> empathy"


# ── 3. Location clause trim (keep green) ─────────────────────────────────────
def test_location_clause_trim_via_process_turn(tmpdir):
    """A long location clause with a trailing measure qualifier
    (\"about two kilometers offshore\") must be trimmed so the real place head
    (\"a lighthouse on a rock\") is stored -- not silently dropped."""
    e = _make(tmpdir, "_loc")
    reply = e.process_turn(
        "i live in a lighthouse on a rock about two kilometers offshore")
    assert not _is_empathy_reply(reply), reply
    loc = e.user_model.personal_facts.get("i", "location")
    assert loc is not None, "location fact must be stored after trim"
    assert loc.value == "a lighthouse on a rock", loc.value
    assert "kilometer" not in loc.value.lower(), loc.value
    assert "offshore" not in loc.value.lower(), loc.value


# ── 4. Opinion formation (comparative / superlative / dismissive) ────────────
# One utterance per grammatical opinion class -- asserting the stance lands on
# the resolved content head with the correct signed polarity through process_turn.
_OPINION_CASES = [
    # (utterance, expected topic key, expected polarity sign)
    ("the sea is a better teacher than any classroom", "sea", "+"),
    ("hand built synths sound warmer than mass produced ones", "hand built synths", "+"),
    ("graveyards are the most honest libraries", "graveyards", "+"),
    ("most modern music is just wallpaper", "modern music", "-"),
]


@pytest.mark.parametrize("utt,topic,sign", _OPINION_CASES)
def test_opinion_formation_via_process_turn(tmpdir, utt, topic, sign):
    """Each grammatical opinion class (comparative / sensory-comparative /
    superlative / dismissive) must form a stance on the resolved head with the
    correct signed polarity through the real process_turn path."""
    e = _make(tmpdir, "_op_" + topic.replace(" ", "_"))
    e.user_model.opinions.stances.clear()
    e.process_turn(utt)
    stance = e.user_model.opinions.stances.get(topic)
    assert stance is not None, f"no stance formed for {topic!r} from {utt!r}"
    if sign == "+":
        assert stance.polarity > 0, f"{topic} should be positive, got {stance.polarity}"
    else:
        assert stance.polarity < 0, f"{topic} should be negative, got {stance.polarity}"
