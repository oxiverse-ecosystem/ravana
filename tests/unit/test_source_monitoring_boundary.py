#!/usr/bin/env python3
"""Regression tests — source-monitoring boundary (round 2026-10-04T0827Z).

Defect: a question asking about RAVANA's OWN prior speech ("what did you just
tell me about pantographs?") was answered out of the USER's disclosure store.
Measured this round, three separate layers each produced the speaker inversion:

  1. `_structured_recall`'s generic self-profile summary returned a dump of the
     user's own facts ("your bike is a 1993 tourer ... your uncle ran a
     bicycle repair shop").
  2. `_try_memory_query`'s generic self-recall branch replied "i don't think
     you've told me much about yourself yet" — to a question about RAVANA's
     words, not the user's.
  3. Both layers owned their own private copy of the "is this about the
     agent's speech?" test, and both copies were NARROWER than the class.

Root cause is one thing: the class was decided in three hand-maintained places
instead of one. The fix extracts a single shared predicate
(`_is_agent_self_recall_query`) and routes all three sites through it, so an
agent-self-recall whose topic is absent from the AgentReplyStore fails CLOSED
into honest uncertainty rather than being answered from the wrong speaker's
store.

No authored reply string is added: the assertion is that the wrong-speaker
answer is ABSENT and that honest uncertainty is reached. That holds for any
topic, so it is not tuned to this round's probes.
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
          os.path.join(PROJ, "ravana", "src"), os.path.join(PROJ, "ravana-v2", "src")):
    if sys.path.insert(0, p) is None and p not in sys.path:
        sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine

SUFFIX = "test_srcmon_0827"

# Fragments that can ONLY come from the user's disclosure store. If any of
# these appear in the answer to a question about the agent's own speech, the
# user/agent boundary has been breached.
_USER_CHANNEL = ("you told me", "your bike", "your uncle", "your sister",
                 "your dog", "your cat", "your bike is")


def _eng():
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=SUFFIX)


def test_shared_predicate_is_the_single_source_of_truth():
    """The predicate classifies the class structurally, for ANY phrasing."""
    e = _eng()
    # agent-self recall
    for q in ("what did you just tell me about pantographs",
              "what did you say about zylophones",
              "remind me what you said about sediment cores",
              "earlier you mentioned telescopes",
              "do you remember what you told me about kilns"):
        assert e._is_agent_self_recall_query(q), f"should classify as agent-self: {q!r}"
    # user-disclosure recall must NOT be classified as agent-self
    for q in ("what did i tell you about my sister",
              "what do i think about tide mills",
              "what have i said about pottery",
              "how do i feel about it"):
        assert not e._is_agent_self_recall_query(q), \
            f"user-disclosure recall misclassified as agent-self: {q!r}"
    # plain world questions must NOT be classified as agent-self
    for q in ("what is a kiln", "how does a tide mill work",
              "who is meera", "i run a pottery studio in kochi"):
        assert not e._is_agent_self_recall_query(q), \
            f"world query misclassified as agent-self: {q!r}"


def test_speaker_rule_generalises_past_topic_nouns():
    """The user-channel exemption turns on the SPEAKER, not on topic nouns.

    Post-fix A/B measured two misclassifications of the first version of the
    exemption: a topic-noun list plus a bare `do you remember (what|when) i`
    clause. Both were speaker/verb mismatches, and both are covered here with
    phrasings that do not reuse this round's probe topics.
    """
    e = _eng()
    # Agent-channel even though the TOPIC belongs to the user channel.
    assert e._is_agent_self_recall_query("what did you say about my bicycle"), \
        "agent speech about a user-channel topic must stay agent-channel"
    assert e._is_agent_self_recall_query("do you remember what i asked you earlier"), \
        "the user ASKING the agent is agent-channel, not a user disclosure"
    # ...while genuine user disclosures stay in the user channel.
    for q in ("what did i tell you about my bicycle",
              "do you remember what i told you last week",
              "what did i say about the shop",
              "how do i feel about it",
              "what was i talking about"):
        assert not e._is_agent_self_recall_query(q), \
            f"user disclosure misclassified as agent-self: {q!r}"


def test_structured_recall_refuses_agent_self_recall():
    """Layer 1: the generic self-profile summary must not answer it."""
    e = _eng()
    e.process_turn("my bike is a 1993 tourer painted the wrong shade")
    out = e._structured_recall("what did you just tell me about pantographs")
    assert out is None, f"expected fail-closed None, got user-channel answer: {out!r}"


def test_memory_query_refuses_agent_self_recall():
    """Layer 2: the generic self-recall branch must not answer it."""
    e = _eng()
    e.process_turn("my uncle ran a bicycle repair shop in trivandrum")
    out = e._try_memory_query("what did you say about zylophones")
    assert out is None, f"expected fail-closed None, got: {out!r}"


def test_end_to_end_no_user_facts_as_agent_words():
    """The user/agent inversion must not reach the reply end-to-end."""
    e = _eng()
    e.process_turn("my bike is a 1993 tourer painted the wrong shade")
    e.process_turn("my uncle ran a bicycle repair shop in trivandrum")
    out = e.process_turn("what did you just tell me about pantographs")
    low = (out or "").lower()
    for frag in _USER_CHANNEL:
        assert frag not in low, \
            f"user-channel content {frag!r} leaked into an agent-self-recall answer: {out!r}"


def test_agent_self_recall_still_answers_from_own_store():
    """The capability is NOT disabled: a genuinely stored own reply still answers.

    Guards against a 'fix' that simply kills agent-self recall.
    """
    e = _eng()
    e.process_turn("tell me about thalassocratic trade routes")
    out = e.process_turn("what did you just tell me about thalassocratic trade routes")
    assert out is not None, "stored own-reply recall regressed to None"
    assert "thalassocratic" in (out or "").lower() or "route" in (out or "").lower(), \
        f"expected RAVANA's own prior reply, got: {out!r}"


def test_user_disclosure_recall_still_uses_user_store():
    """Guards the other direction: the user channel must NOT be broken."""
    e = _eng()
    e.process_turn("my sister devika handles the glazing end")
    out = e.process_turn("what did i tell you about devika")
    low = (out or "").lower()
    assert "glazing" in low or "devika" in low, \
        f"user-disclosure recall regressed: {out!r}"


def teardown_module(module):
    try:
        e = _eng()
        e.stop_background_learning()
    except Exception:
        pass