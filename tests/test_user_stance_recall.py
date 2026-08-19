"""Limitation H — USER-stance recall (feature t_d6e10e53, round 2026-08-17T0622Z).

A question that asks RAVANA whether the USER likes/loves/hates something
("do you think i like spicy food or not?") is about the USER's own stated
preference, which RAVANA stores in user_model.opinions.stances. Previously
these queries matched the broad self-opinion gate and routed to
_route_self_query, which computed RAVANA's OWN (empty) stance and returned the
generic "still figuring that out" hedge — a self/other boundary error. This
capability consults the user's held stance and answers from it.

This test fails WITHOUT the capability (the reply is the generic hedge, not a
stance-grounded answer) and passes WITH it.
"""
import os
import re
import sys
import tempfile

import pytest

sys.path[:0] = ['ravana/src', 'ravana_ml/src', 'ravana-v2/src', os.getcwd()]

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402


@pytest.fixture(scope="module")
def tmpdir():
    return tempfile.mkdtemp(prefix='ravana_ustance_')


def _make(tmpdir, suffix):
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               data_dir=tmpdir, user_suffix=suffix)


def _held_stance(e, topic):
    return e.user_model.opinions.stances.get(topic)


def test_user_stance_recall_positive(tmpdir):
    # Disclose a preference, then ask RAVANA (3rd person) whether the USER likes it.
    e = _make(tmpdir, '_pos')
    e.process_turn("i love spicy food")
    assert _held_stance(e, "spicy food") is not None, "stance must be mined from disclosure"

    ans = e.process_turn("do you think i like spicy food or not?")
    assert "spicy food" in (ans or "").lower(), ans
    assert "for" in (ans or "").lower(), ans  # polarity must read positive
    assert e._last_strategy == "user_stance_recall", e._last_strategy


def test_user_stance_recall_negative(tmpdir):
    e = _make(tmpdir, '_neg')
    e.process_turn("i hate cold coffee")
    assert _held_stance(e, "cold coffee") is not None

    ans = e.process_turn("do you think i hate cold coffee?")
    assert "cold coffee" in (ans or "").lower(), ans
    assert "against" in (ans or "").lower(), ans  # polarity must read negative
    assert e._last_strategy == "user_stance_recall", e._last_strategy


def test_user_stance_recall_paraphrase_links(tmpdir):
    # "love jazz" is mined as a stance on "jazz"; the query "do you think i love jazz"
    # must resolve to the same held key via the live store resolver.
    e = _make(tmpdir, '_para')
    e.process_turn("i adore jazz")
    assert _held_stance(e, "jazz") is not None

    ans = e.process_turn("do you think i love jazz?")
    assert "jazz" in (ans or "").lower(), ans
    assert "for" in (ans or "").lower(), ans
    assert e._last_strategy == "user_stance_recall", e._last_strategy


def test_user_stance_recall_fail_closed_when_no_stance(tmpdir):
    # Topic the user never stated a preference on -> honest downstream, NOT a
    # fabricated stance read.
    e = _make(tmpdir, '_none')
    ans = e.process_turn("do you think i like quantum physics?")
    assert e._last_strategy != "user_stance_recall", \
        "must not claim a stance the user never expressed"
    # Word-boundary check for the polarity word "for" — a plain substring test
    # false-positives on words like "forming" (as in "still forming a view"),
    # which is the honest fail-closed hedge, not a fabricated stance claim.
    ans_l = (ans or "").lower()
    assert "quantum physics" not in ans_l or not re.search(r"\bfor\b", ans_l), ans


def test_user_stance_recall_does_not_capture_agent_self_opinion(tmpdir):
    # A genuine AGENT self-opinion question ("do you think we should protect
    # mangroves", RAVANA is the attitude holder) must NOT be absorbed by the
    # user-stance guard — it has no held user stance and is a question about
    # RAVANA, so it routes to the self-model resolver as before.
    e = _make(tmpdir, '_agent')
    ans = e.process_turn("do you think we should protect mangroves?")
    assert e._last_strategy != "user_stance_recall", e._last_strategy
