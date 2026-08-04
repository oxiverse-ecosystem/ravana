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
