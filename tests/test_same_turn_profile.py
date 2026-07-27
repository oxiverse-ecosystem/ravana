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

    e2 = _make(tmpdir, '_b')
    e2.process_turn("my name is alex")
    ans2 = e2.process_turn("what is my name?")
    assert 'alex' in (ans2 or '').lower(), ans2

    # B: correction supersedes the old value.
    e3 = _make(tmpdir, '_c')
    e3.process_turn("my dog is rex")
    e3.user_model.personal_facts.contradict('i', 'dog', 'max')
    hit = e3.user_model.personal_facts.get('i', 'dog')
    assert hit is not None and hit.value == 'max'

    # C: opinions captured + recalled, separate from facts.
    e5 = _make(tmpdir, '_d')
    e5.process_turn("i really like cats")
    e5.process_turn("i hate dogs")
    cats = e5.process_turn("do you know what i think about cats?")
    assert 'like' in (cats or '').lower(), cats
    dogs = e5.process_turn("what do you know about what i think of dogs?")
    assert 'dislike' in (dogs or '').lower(), dogs

    # B: persistence across save/reload.
    e.save()
    e4 = _make(tmpdir, '_a')
    e4.load()
    h = e4.user_model.personal_facts.get('i', 'cat')
    assert h is not None and h.value == 'pixel'
