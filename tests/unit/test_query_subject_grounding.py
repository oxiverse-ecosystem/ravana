"""Regression tests for RSI query-subject grounding (read-side fix).

No engine boot: static helpers and the mixin method are exercised with fakes.
"""
import sys
from types import SimpleNamespace

import numpy as np

P = "C:/Users/Likhith/Documents/projects/ravana"
for p in (P, P+"/ravana/src", P+"/ravana_ml/src"):
    if p not in sys.path:
        sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.response_gen import ResponseGenMixin


def test_forced_choice_option_tail_stripped():
    prompt = ("Considering the examples above, what is true of thaw? "
              "Options: sunset, typhoon, mist, hail. Answer briefly.")
    assert CognitiveChatEngine._topic_input_for_prompt(prompt) == \
        "what is true of thaw? "
    plain = "what causes rain?"
    assert CognitiveChatEngine._topic_input_for_prompt(plain) == plain
    assert CognitiveChatEngine._topic_input_for_prompt("") == ""


class _Edge:
    def __init__(self, kind, source):
        self.source_metadata = {"edge_kind": kind, "source": source}


class _Fake:
    _definitions = {}
    _concept_sources = {}

    def __init__(self, kind, source):
        self._concept_keywords = {"thaw": [7]}
        self.graph = SimpleNamespace(
            edges={(7, 8): _Edge(kind, source)}, nodes={})
        self.ctx = SimpleNamespace(subject="thaw", associated_concepts=[])

    def _grounding_subject(self, ctx):
        return "thaw"

    def _glove_vector(self, word):
        v = np.zeros(64, dtype=np.float32)
        v[0 if word == "thaw" else 1] = 1.0
        return v


def test_trusted_conversation_edge_grounds_subject():
    f = _Fake("curated", "conversation")
    assert ResponseGenMixin._topic_grounded(f, f.ctx) is True


def test_untrusted_or_external_edge_does_not_ground_subject():
    for kind, source in (("unknown", "conversation"),
                         ("curated", "https://external.example")):
        f = _Fake(kind, source)
        assert ResponseGenMixin._topic_grounded(f, f.ctx) is False
