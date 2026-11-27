"""Wiring tests for the default-OFF lexical-superordinate emission hook.

No engine boot, no store write, no benchmark import and no WordNet corpus:
these call the REAL shipped methods as unbound functions on tiny fakes and run
the REAL ``process_turn`` wrapper on a stand-in that stubs only the expensive
internals. The pure scorer is already covered by test_lexical_superordinate.py;
this file covers the boundary wiring and its fail-closed/deferral contract.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Optional

import pytest

from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.response_gen import (
    ResponseGenMixin,
    lexical_teaching_pairs_from_graph,
)
from ravana.learn import lexical_superordinate as ls

FLAG = ls.ENV_FLAG
PROMPT = "what is true of tree? Options: shrub, petal"


def _flag(monkeypatch, on):
    if on:
        monkeypatch.setenv(FLAG, "1")
    else:
        monkeypatch.delenv(FLAG, raising=False)


def _install_fake(monkeypatch, table):
    """Deterministic stand-in for the lexical resource; symmetric lookup."""
    monkeypatch.setattr(
        ls, "path_similarity",
        lambda a, b: table.get((a, b), table.get((b, a))))


class _FakeGraph:
    """Minimal read-only ConceptGraph stand-in with provenance metadata."""

    def __init__(self, pairs=(), trusted=True, relation="is_a"):
        self.nodes = {}
        self.edges = {}
        self._ids = {}
        kind = "curated" if trusted else "unknown"
        source = "conversation" if trusted else "web"
        for s, t in pairs:
            sid, tid = self._id(s), self._id(t)
            self.edges[(sid, tid)] = SimpleNamespace(
                source_metadata={"edge_kind": kind, "source": source},
                relation_type=relation,
            )

    def _id(self, label):
        key = str(label).strip().lower()
        if key not in self._ids:
            idx = len(self._ids) + 1
            self._ids[key] = idx
            self.nodes[idx] = SimpleNamespace(label=label)
        return self._ids[key]

    def get_node(self, idx):
        return self.nodes.get(idx)


def _hook(fake, raw, subject=None, current=None):
    return ResponseGenMixin._lexical_superordinate_candidate(
        fake, raw, subject, current)


def _fake_engine(graph=None, subject: Optional[str] = "tree"):
    return SimpleNamespace(graph=graph, _concept_keywords={},
                           _last_subject=subject)


class _BoundaryEngine:
    """Supports the real process_turn wrapper with stubbed internals."""

    process_turn = CognitiveChatEngine.process_turn

    def __init__(self, response, graph=None, subject="tree", include_hook=True):
        self._response = response
        self.graph = graph
        self._concept_keywords = {}
        self._last_subject = subject
        self.turn_count = 0
        self._pending_source_context = None
        self._source_gate_ctx = None
        self._calls = 0
        if include_hook:
            def _counted(raw, subj=None, current=None):
                self._calls += 1
                return ResponseGenMixin._lexical_superordinate_candidate(
                    self, raw, subj, current)
            self._lexical_superordinate_candidate = _counted

    def _style_conditioning_enabled(self):
        return False

    def _process_turn_impl(self, _user_input):
        return self._response

    def _record_dialogue_turn(self, *a, **k):
        pass

    def _measure_turn_alignment(self, *a, **k):
        pass


# ── Adapter: stored demonstrations, provenance-filtered ─────────────────────

def test_adapter_reads_only_trusted_demonstrations_and_dedupes():
    graph = _FakeGraph([("a", "b"), ("a", "b"), ("x", "y")])
    assert lexical_teaching_pairs_from_graph(graph) == [("a", "b"), ("x", "y")]


def test_adapter_excludes_untrusted_edges():
    graph = _FakeGraph([("a", "b")], trusted=False)
    assert lexical_teaching_pairs_from_graph(graph) == []


def test_adapter_is_read_only_and_has_no_key_parameter():
    sig = inspect.signature(lexical_teaching_pairs_from_graph)
    assert set(sig.parameters) == {"graph"}


# ── Hook method contract ────────────────────────────────────────────────────

def test_hook_inert_when_flag_off(monkeypatch):
    _flag(monkeypatch, False)
    graph = _FakeGraph([("a", "b")])
    _install_fake(monkeypatch, {("a", "b"): 0.40, ("tree", "shrub"): 0.90})
    assert _hook(_fake_engine(graph), PROMPT) is None


def test_hook_emits_from_stored_demonstrations_when_on(monkeypatch):
    _flag(monkeypatch, True)
    graph = _FakeGraph([("a", "b")])
    _install_fake(monkeypatch, {("a", "b"): 0.40,
                                ("tree", "shrub"): 0.70,
                                ("tree", "petal"): 0.20})
    assert _hook(_fake_engine(graph), PROMPT) == "shrub"


def test_hook_abstains_with_no_teaching_pairs(monkeypatch):
    _flag(monkeypatch, True)
    _install_fake(monkeypatch, {("tree", "shrub"): 0.99})
    assert _hook(_fake_engine(_FakeGraph()), PROMPT) is None


def test_hook_abstains_when_lexical_resource_missing(monkeypatch):
    _flag(monkeypatch, True)
    _install_fake(monkeypatch, {})
    graph = _FakeGraph([("a", "b")])
    assert _hook(_fake_engine(graph), PROMPT) is None


def test_hook_defers_to_a_bare_candidate_from_an_earlier_hook(monkeypatch):
    _flag(monkeypatch, True)
    graph = _FakeGraph([("a", "b")])
    _install_fake(monkeypatch, {("a", "b"): 0.40,
                                ("tree", "shrub"): 0.70,
                                ("tree", "petal"): 0.20})
    assert _hook(_fake_engine(graph), PROMPT, current="shrub") is None


def test_hook_defers_to_a_directly_taught_trusted_edge(monkeypatch):
    _flag(monkeypatch, True)
    # A trusted conversation causal edge shrub -> tree must win over the
    # lexical hypothesis, exactly as H3/H4 deferral requires.
    graph = _FakeGraph([("shrub", "tree"), ("a", "b")],
                       relation="causal")
    _install_fake(monkeypatch, {("a", "b"): 0.40,
                                ("tree", "shrub"): 0.70,
                                ("tree", "petal"): 0.20})
    assert _hook(_fake_engine(graph), PROMPT) is None


def test_hook_ignores_non_forced_choice_input(monkeypatch):
    _flag(monkeypatch, True)
    graph = _FakeGraph([("a", "b")])
    _install_fake(monkeypatch, {("a", "b"): 0.40,
                                ("tree", "shrub"): 0.70,
                                ("tree", "petal"): 0.20})
    assert _hook(_fake_engine(graph), "tell me about trees") is None
    assert _hook(_fake_engine(graph), "what is true of tree? Options: shrub") is None


def test_hook_fails_closed_on_malformed_graph(monkeypatch):
    _flag(monkeypatch, True)
    _install_fake(monkeypatch, {("a", "b"): 0.40,
                                ("tree", "shrub"): 0.90})

    class _Broken:
        @property
        def edges(self):
            raise RuntimeError("corrupt store")

    assert _hook(_fake_engine(_Broken()), PROMPT) is None


def test_hook_uses_question_subject_when_none_staged(monkeypatch):
    """No staged subject on an early-return path: derive it from the question."""
    _flag(monkeypatch, True)
    graph = _FakeGraph([("a", "b")])
    _install_fake(monkeypatch, {("a", "b"): 0.40,
                                ("tree", "shrub"): 0.70,
                                ("tree", "petal"): 0.20})
    assert _hook(_fake_engine(graph, subject=None), PROMPT) == "shrub"


def test_hook_uses_head_noun_of_a_phrase_subject(monkeypatch):
    _flag(monkeypatch, True)
    graph = _FakeGraph([("a", "b")])
    _install_fake(monkeypatch, {("a", "b"): 0.40,
                                ("tree", "shrub"): 0.70,
                                ("tree", "petal"): 0.20})
    fake = _fake_engine(graph)
    assert _hook(fake, PROMPT, subject="true tree") == "shrub"


def test_hook_public_signature_has_no_answer_key_parameter():
    params = inspect.signature(
        ResponseGenMixin._lexical_superordinate_candidate).parameters
    assert set(params) == {"self", "raw_input", "subject", "current_response"}


# ── process_turn boundary: reachability + flag equivalence ──────────────────

def test_boundary_fires_only_when_flag_on(monkeypatch):
    graph = _FakeGraph([("a", "b")])
    sims = {("a", "b"): 0.40, ("tree", "shrub"): 0.70,
            ("tree", "petal"): 0.20}
    _install_fake(monkeypatch, sims)

    _flag(monkeypatch, False)
    off = _BoundaryEngine("baseline hedge", graph)
    assert off.process_turn(PROMPT) == "baseline hedge"
    assert off._calls == 0            # not merely inert: never called

    _flag(monkeypatch, True)
    on = _BoundaryEngine("baseline hedge", graph)
    assert on.process_turn(PROMPT) == "shrub"
    assert on._calls == 1             # reachable from the outer boundary


def test_boundary_flag_off_on_are_identical_without_lexical_evidence(monkeypatch):
    graph = _FakeGraph([("a", "b")])
    _install_fake(monkeypatch, {})     # no WordNet evidence -> abstain
    _flag(monkeypatch, False)
    off = _BoundaryEngine("baseline hedge", graph).process_turn(PROMPT)
    _flag(monkeypatch, True)
    on = _BoundaryEngine("baseline hedge", graph).process_turn(PROMPT)
    assert on == off == "baseline hedge"


def test_boundary_duck_typed_engine_without_hook_is_unchanged(monkeypatch):
    _flag(monkeypatch, True)
    graph = _FakeGraph([("a", "b")])
    _install_fake(monkeypatch, {("a", "b"): 0.40,
                                ("tree", "shrub"): 0.90,
                                ("tree", "petal"): 0.10})
    fake = _BoundaryEngine("baseline hedge", graph, include_hook=False)
    assert fake.process_turn(PROMPT) == "baseline hedge"


def test_boundary_chatter_path_is_untouched(monkeypatch):
    _flag(monkeypatch, True)
    graph = _FakeGraph([("a", "b")])
    _install_fake(monkeypatch, {("a", "b"): 0.40,
                                ("tree", "shrub"): 0.90})
    fake = _BoundaryEngine("chatter reply", graph)
    assert fake.process_turn("hello there") == "chatter reply"
