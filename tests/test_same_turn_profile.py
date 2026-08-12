"""Same-turn memory + learned user profile + opinions (Plan A/B/C).

Boots the real engine and verifies:
  A5  - a personal fact stated THIS turn is recallable THIS turn
        ("my name is X. what is my name?" / "my cat is pixel. what's my cat's name?")
  B   - PersonalFactStore: gradeable + correctable; survives save/reload
  C   - UserStanceStore: opinions captured + recalled separately from facts

The boot is heavy (GloVe64 + graph), so this is a single combined test that
reuses a temp data dir. Keep it focused.
"""
import os
import sys
import tempfile

import pytest

sys.path[:0] = ['ravana/src', 'ravana_ml/src', 'ravana-v2/src', os.getcwd()]

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402


@pytest.fixture(scope="module")
def tmpdir():
    return tempfile.mkdtemp(prefix='ravana_stp_')


def _make(tmpdir, suffix):
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               data_dir=tmpdir, user_suffix=suffix)


def test_same_turn_personal_fact_and_opinion(tmpdir):
    # A5: name + pet stated this turn, recalled this turn.
    e = _make(tmpdir, '_a')
    e.process_turn("my cat is pixel")
    ans = e.process_turn("what's my cat's name?")
    assert 'pixel' in (ans or '').lower(), ans

    # B: persistence across save/reload.
    e.save()
    e4 = _make(tmpdir, '_a')
    e4.load()
    h = e4.user_model.personal_facts.get('i', 'cat')
    assert h is not None and h.value == 'pixel'


def test_same_turn_name_recall(tmpdir):
    e2 = _make(tmpdir, '_b')
    e2.process_turn("my name is alex")
    ans2 = e2.process_turn("what is my name?")
    assert 'alex' in (ans2 or '').lower(), ans2


def test_correction_supersedes_prior_value(tmpdir):
    # B: correction supersedes the old value.
    e3 = _make(tmpdir, '_c')
    e3.process_turn("my dog is rex")
    e3.user_model.personal_facts.contradict('i', 'dog', 'max')
    hit = e3.user_model.personal_facts.get('i', 'dog')
    assert hit is not None and hit.value == 'max'


def test_opinions_captured_and_recalled(tmpdir):
    # C: opinions captured + recalled, separate from facts.
    e5 = _make(tmpdir, '_d')
    e5.process_turn("i really like cats")
    e5.process_turn("i hate dogs")
    cats = e5.process_turn("do you know what i think about cats?")
    assert 'like' in (cats or '').lower(), cats
    dogs = e5.process_turn("what do you know about what i think of dogs?")
    assert 'dislike' in (dogs or '').lower(), dogs


def test_distinct_species_coexist(tmpdir):
    """A pet slot keeps its SPECIES, so a cat and a dog do not collide and a
    cued recall resolves to the animal actually asked about."""
    e = _make(tmpdir, '_sp')
    e.process_turn("my cat is pixel")
    e.process_turn("my dog is rex")
    assert e.user_model.personal_facts.get('i', 'cat').value == 'pixel'
    assert e.user_model.personal_facts.get('i', 'dog').value == 'rex'
    ans = e.process_turn("what is my dog's name?")
    assert 'rex' in (ans or '').lower(), ans


def test_correction_loop_and_world_graph_isolation(tmpdir):
    """Investigation fixes (reports/user_model_investigation.md, since removed):

    Gap 1 - a LIVE corrective turn ("no, my cat is milo") supersedes the old
            value through the wired contradict() path (no manual store call).
    Gap 2 - user self-disclosures are withheld from the world-graph drain
            (user_facts_withheld) while genuine world facts still graduate,
            and the personal_facts channel is unaffected.
    """
    e = _make(tmpdir, '_fx')
    e.process_turn("my cat is pixel")
    e.process_turn("i live in berlin")

    # Gap 2: sleep must NOT graduate user disclosures into the world graph.
    # Hard invariant: no user_fact may ever reach _ensure_relation, so the
    # world-graph graduate count is deterministically 0 regardless of which
    # candidates the (turn/age-threshold based) consolidation selector picks.
    res = e._sleep_consolidate()
    assert res.get('buffer_facts_graduated', 0) == 0, res
    assert res.get('user_facts_withheld', 0) >= 1, res
    assert res.get('personal_facts_graduated', 0) >= 1, res

    # ...but a genuine world fact still graduates.
    e.hippocampal_buffer.store("paris", "is_capital_of", "france",
                               confidence=0.95)
    e.hippocampal_buffer.store("paris", "is_capital_of", "france",
                               confidence=0.95)
    res2 = e._sleep_consolidate()
    assert res2.get('buffer_facts_graduated', 0) >= 1, res2

    # Gap 1: live correction supersedes via the wired loop.
    e.process_turn("no, my cat is milo")
    ans = e.process_turn("what is my cat's name?")
    assert 'milo' in (ans or '').lower(), ans
    old = [v for (s, a, v), f in e.user_model.personal_facts.facts.items()
           if a == 'cat' and f.superseded]
    assert 'pixel' in old, old


def test_concession_reversal_reaches_stance_via_process_turn(tmpdir):
    """Regression for t_58d5f3ac: a natural concession ("i thought X was good
    but now Y") must reach the stance-reversal miner through the REAL
    process_turn path (it routes via the self_disclosure strategy, which
    early-returns). The conceded topic (X) must be the one reversed - NOT the
    new preference (Y) - and the held stance must move toward neutral (0.0 for
    a soft concession), never stay stale, and the ack must not fabricate a
    reversal about a topic with no prior stance.

    Covers THREE distinct concession phrasings structurally (no fixed list):
    "i thought ... but ...", "i used to think ... but now ...",
    "i told you ... but actually ...".
    """
    e = _make(tmpdir, '_conc')
    e.user_model.opinions.stances.clear()
    e.user_model.opinions.express_stance("sea", polarity=0.7, confidence=0.8)
    e.user_model.opinions.express_stance("cities", polarity=-0.6, confidence=0.8)
    e.user_model.opinions.express_stance("meat", polarity=0.8, confidence=0.8)

    # 1) "i thought X but Y" - conceded topic X = sea.
    reply1 = e.process_turn(
        "i thought the sea was a good teacher but actually i prefer mountains now")
    sea = e.user_model.opinions.stances.get("sea")
    assert sea is not None and sea.polarity == 0.0, sea
    assert "sea" in (reply1 or "").lower(), reply1
    assert "mountains" not in (reply1 or "").lower(), reply1

    # 2) "i used to think X but now Y" - conceded X = cities (against -> neutral).
    reply2 = e.process_turn(
        "i used to think cities were thrilling but now i find the countryside calmer")
    cities = e.user_model.opinions.stances.get("cities")
    assert cities is not None and cities.polarity == 0.0, cities
    assert "cities" in (reply2 or "").lower(), reply2
    assert "countryside" not in (reply2 or "").lower(), reply2

    # 3) "i told you X but actually Y" - conceded X = meat (for -> neutral).
    reply3 = e.process_turn(
        "i told you meat was the best protein but actually i'm going plant-based")
    meat = e.user_model.opinions.stances.get("meat")
    assert meat is not None and meat.polarity == 0.0, meat
    assert "meat" in (reply3 or "").lower(), reply3
    assert "plant" not in (reply3 or "").lower(), reply3

    # Idempotency: replay of the SAME utterance in a fresh turn must NOT
    # double-flip (process_turn mines it twice internally). Polarity stays 0.0.
    e2 = _make(tmpdir, '_conc2')
    e2.user_model.opinions.stances.clear()
    e2.user_model.opinions.express_stance("sea", polarity=0.7, confidence=0.8)
    e2.process_turn("i thought the sea was a good teacher but actually i prefer mountains now")
    sea2 = e2.user_model.opinions.stances.get("sea")
    assert sea2 is not None and sea2.polarity == 0.0, sea2
    assert sea2.rehearsal_count == 2, sea2  # seeded once + reversed once


def test_concession_without_prior_stance_does_not_fabricate(tmpdir):
    """A concession about a topic RAVANA never held a stance on must NOT
    fabricate a reversal ack for that topic (no prior stance => nothing to
    reverse). The new-preference clause must not leak as a reversed stance."""
    e = _make(tmpdir, '_concnone')
    e.user_model.opinions.stances.clear()
    reply = e.process_turn(
        "i thought climbing was my thing but actually i prefer the beach now")
    assert "beach" not in (reply or "").lower(), reply
    assert "changed your mind about beach" not in (reply or "").lower(), reply


# Marker strings that indicate the turn was ROUTED into the emotional-support /
# empathy path instead of acknowledged as a fact. These are the existing empathy
# reply openings, used only to ASSERT THEIR ABSENCE on benign disclosures (we do
# NOT assert on any specific ack wording — the ack content is RAVANA-generated,
# not a hardcoded string). This is the documented RAVANA "Support-router misfire"
# defect class: a benign self-disclosure must be acked as a fact, not met with
# comfort, when no genuine distress is present.
_EMPATHY_OPENERS = (
    "i hear you", "i'm here for", "i'm so sorry", "feeling rough",
    "feeling sad is hard", "what happened", "what set it off",
)


def _is_empathy_reply(reply: str) -> bool:
    r = (reply or "").lower()
    return any(op in r for op in _EMPATHY_OPENERS)


def test_possessive_disclosure_acked_not_routed_to_support(tmpdir):
    """Regression for t_79d3621d: a possessive disclosure ("my dog is X",
    "my partner's name is Y", "my child is Z") must be ACKED as a stored fact,
    NOT routed into the emotional-support / empathy path when no distress is
    present. The ack must reference the CORRECT ENTITY (partner, not "your").

    Asserts on routing + entity attribution derived from the live store — never
    on a hardcoded reply sentence. Structural: covers three distinct entities
    (partner / dog / child) without a per-entity ack table.
    """
    e = _make(tmpdir, '_poss')

    # 1) Partner disclosure -> acked, entity = partner (not "your").
    reply1 = e.process_turn("my partner's name is Pell")
    assert not _is_empathy_reply(reply1), reply1
    # Correct entity attribution: must NOT mis-attribute to the user ("your name").
    assert "your name is pell" not in (reply1 or "").lower(), reply1
    assert "partner" in (reply1 or "").lower(), reply1
    # Fact really stored under the partner entity.
    fact1 = e.user_model.personal_facts.get("partner", "name")
    assert fact1 is not None and fact1.value == "pell", fact1

    # 2) Dog disclosure -> acked as a fact, not met with comfort.
    reply2 = e.process_turn("my dog is a sheepdog named Cairn")
    assert not _is_empathy_reply(reply2), reply2
    # The entity word appears in the ack (dog), confirming correct attribution.
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
    """Guard against over-firing the misfire gate: a disclosure that DOES
    contain a real suffering/distress signal must still reach the empathy path,
    not be swallowed as a bare fact ack. Covers bereavement + present-state
    distress to prove the gate keys off genuine distress, not the utterance
    shape alone.
    """
    e = _make(tmpdir, '_distress')

    # Bereavement: "my dog died" -> grief empathy.
    reply1 = e.process_turn("my dog died")
    assert _is_empathy_reply(reply1), reply1

    # Present-state distress: "i am sad" -> empathy (not a fact ack).
    reply2 = e.process_turn("i am sad")
    assert _is_empathy_reply(reply2), reply2

    # Other-suffering: "my friend is hurting" -> empathy.
    reply3 = e.process_turn("my friend is hurting")
    assert _is_empathy_reply(reply3), reply3



