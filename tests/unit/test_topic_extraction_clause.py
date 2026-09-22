"""
Regression tests for DEFECT 3 (topic extraction garbage in opinion queries).

The opinion topic extractor treated the tail after the opinion cue as a flat
noun phrase, so clause predicates leaked into the topic:
  "do you think silence is underrated" -> topic "silence underrated"
  "what do you think happens to a memory" -> topic "happens memory"
  "what kind of mind do you have, exactly" -> topic "exactly"

After the fix, the extractor detects clause structure (copular/intransitive
verbs) and resolves to the real subject/topic.
"""
import os
import re
import sys

os.environ["RAVANA_OFFLINE"] = "1"
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_PROJ, os.path.join(_PROJ, "ravana", "src"),
           os.path.join(_PROJ, "ravana_ml", "src")):
    sys.path.insert(0, _p)

from ravana.chat.engine import CognitiveChatEngine


def _eng(suffix):
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)


# -- Case 1: copular clause "silence is underrated" -> topic "silence" --
def test_copular_clause_topic_is_subject():
    eng = _eng("test_copular_subject")
    r = eng.process_turn("do you think silence is underrated")
    # Must NOT contain the predicate adjective as the topic
    assert "silence underrated" not in (r or ""), r
    # Must name the real topic "silence" (either as a formed stance or
    # honest "still forming a view on silence")
    assert "silence" in (r or ""), r


# -- Case 2: intransitive clause "happens to a memory" -> topic "memory" --
def test_intransitive_to_pp_topic_is_object():
    eng = _eng("test_intrans_to_pp")
    r = eng.process_turn("what do you think happens to a memory")
    # Must NOT contain the verb as the topic
    assert "happens memory" not in (r or ""), r
    assert "forming a view on happens" not in (r or ""), r
    # Must name the real topic "memory"
    assert "memory" in (r or ""), r


# -- Case 3: trailing adverb "exactly" -> no topic, honest fallback --
def test_trailing_adverb_no_topic():
    eng = _eng("test_adverb_no_topic")
    r = eng.process_turn("what kind of mind do you have, exactly")
    # Must NOT treat "exactly" as a stance topic
    assert "forming a view on exactly" not in (r or ""), r
    assert "a view on exactly" not in (r or ""), r
    assert "stance on exactly" not in (r or ""), r
    # Must fall through to honest uncertainty (the agent doesn't claim a
    # stance on the adverb). "exactly" may still appear in the response
    # text (e.g. "don't have a grasp on exactly") as long as it's not a
    # fabricated stance topic.


# -- Case 4: existing working case must not regress --
def test_simple_opinion_still_works():
    eng = _eng("test_simple_opinion")
    r = eng.process_turn("do you think we should protect mangroves")
    # Topic should be "mangroves" (the object of the imperative)
    assert "mangroves" in (r or ""), r
