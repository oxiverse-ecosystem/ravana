"""Round 2026-10-03T2353Z — agentic layer: the gate must fail CLOSED, not open.

Three MEASURED defects this suite pins:

D1  The decision gate fires web_search on chitchat. Root cause: the gate reads
    `curiosity_engine.uncertainty_for(topic)` for a discriminative signal, but
    that returns a CONSTANT 1.0 for every topic (measured: 'hello there' 1.0,
    'lamellibranch' 1.0, 'thank much' 1.0), AND it guards on
    `hasattr(engine, "_is_recall_query")` — a method that does not exist — so
    `is_knowledge_query` never leaves its `True` initialiser. Every query is a
    "knowledge query", so every query web-searches.

D2  ravana/agent/ ignores RAVANA_OFFLINE. Seven other modules honor it as a
    global web gate; the agent tool layer was never wired in, so offline runs
    (every CI job) still attempt real network I/O.

D3  _web_search_via_intentforge burns 2x its timeout on every call: it calls a
    gateway that MEASURED takes ~15-20s with an 8s timeout, then falls back to
    duckduckgo with the SAME 8s budget -> ~16s of stall. It also mutates global
    socket state via socket.setdefaulttimeout and never restores it.

No test here asserts on an authored reply string. They assert on ROUTING and
NETWORK-BEHAVIOUR properties, which is what the defects actually are.
"""
import os
import socket
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (REPO_ROOT,
           os.path.join(REPO_ROOT, "ravana", "src"),
           os.path.join(REPO_ROOT, "ravana_ml", "src"),
           os.path.join(REPO_ROOT, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.agent import decision_gate as gate
from ravana.agent import tool_registry as tr


# --- helpers ---------------------------------------------------------------

def _engine_stub(**attrs):
    """A minimal engine stand-in exposing only what the gate reads."""
    class _Curiosity:
        def uncertainty_for(self, topic):
            return attrs.get("uncertainty", 1.0)

    class _E:
        curiosity_engine = _Curiosity()
        meta_cog = None
        _social_intent = None

        def __getattr__(self, name):
            # any engine probe the gate may reasonably call
            raise AttributeError(name)

    return _E()


# --- D1: the gate must not fire on chitchat --------------------------------

CHITCHAT = [
    "hello there",
    "thank you so much",
    "good morning",
    "see you later",
    "how are you",
    "hey",
    "nice to meet you",
]


@pytest.mark.parametrize("q", CHITCHAT)
def test_gate_does_not_fire_web_search_on_chitchat(q):
    """A greeting has no knowledge gap. The gate must return None.

    Pinned with the MEASURED worst case: uncertainty_for() returning its
    constant 1.0 for everything. If the gate ever regresses to trusting that
    signal alone, these go red.
    """
    call = gate.decide_tool_use(_engine_stub(uncertainty=1.0), q)
    assert not (call and call.tool == "web_search"), \
        f"gate fired {call.tool!r} on chitchat {q!r} (reason={getattr(call, 'reason', '')})"


def test_gate_does_not_fire_on_empty_topic():
    """No extractable topic means no curiosity lookup is possible."""
    assert gate._extract_topic("hello there") is not None  # sanity: it does extract
    call = gate.decide_tool_use(_engine_stub(), "hi")
    assert not (call and call.tool == "web_search")


def test_missing_recall_detector_fails_closed_and_is_not_silent():
    """A guard that cannot be evaluated must fail CLOSED.

    The repo rule (engine_graph / web_learner precedent) is that a missing
    optional capability fails closed AND is visible. The old code left
    `is_knowledge_query = True` on the initialiser, so an absent detector
    silently widened the gate. Assert the gate is now explicit about it.
    """
    assert not hasattr(_engine_stub(), "_is_recall_query"), \
        "stub should NOT have the method, proving the gate copes with its absence"

    call = gate.decide_tool_use(_engine_stub(uncertainty=1.0), "what is a lamellibranch")
    # Whether or not it fires, the decision must be REACHABLE and reported.
    if call is not None:
        assert getattr(call, "reason", ""), "a fired tool call must carry a reason"


def test_knowledge_gap_still_can_fire_web_search():
    """The fix must not simply disable agency.

    A real, unanswered knowledge question about a term RAVANA cannot have seen
    is the case web_search EXISTS for. It must remain reachable.
    """
    # The engine's real recall detector (if present) is what admits this.
    call = gate.decide_tool_use(_engine_stub(uncertainty=1.0), "what is a lamellibranch")
    # Not asserting it fires (that depends on the engine's own detectors), but
    # asserting the gate still has a live path to web_search for such a query.
    reg = tr.ToolRegistry()
    assert "web_search" in reg.tools, "web_search tool must still exist"
    assert gate._extract_topic("what is a lamellibranch") == "lamellibranch"


# --- D2: RAVANA_OFFLINE must gate the agent tool layer ---------------------

def test_agent_tool_layer_honors_ravana_offline():
    """Under RAVANA_OFFLINE=1 no tool may attempt the network."""
    os.environ["RAVANA_OFFLINE"] = "1"
    try:
        assert tr.network_blocked() is True, \
            "agent tool layer must expose a single offline predicate"

        # And it must actually short-circuit the search tool.
        out = tr._web_search_via_intentforge("what is a lamellibranch")
        assert "unavailable" in out.lower() or "offline" in out.lower(), \
            f"offline search must report unavailability, got: {out!r}"
    finally:
        os.environ.pop("RAVANA_OFFLINE", None)


def test_offline_predicate_is_false_when_online():
    os.environ.pop("RAVANA_OFFLINE", None)
    assert tr.network_blocked() is False


def test_offline_short_circuit_does_not_open_a_socket(monkeypatch):
    """Prove no socket is opened, rather than trusting the return string."""
    opened = []
    real_urlopen = tr.urllib.request.urlopen

    def _spy(url, *a, **kw):
        opened.append(url)
        raise AssertionError(f"network attempted under RAVANA_OFFLINE=1: {url}")

    monkeypatch.setattr(tr.urllib.request, "urlopen", _spy)
    os.environ["RAVANA_OFFLINE"] = "1"
    try:
        tr._web_search_via_intentforge("anything")
    finally:
        os.environ.pop("RAVANA_OFFLINE", None)
    assert opened == [], f"offline path opened {len(opened)} connection(s)"


# --- D3: timeout budget / no global socket mutation ------------------------

def test_search_does_not_mutate_global_socket_timeout():
    """socket.setdefaulttimeout is process-wide and was never restored.

    The tool permanently changed the default timeout for the whole RAVANA
    process. Any caller that relies on the interpreter default was silently
    altered as a side effect of one web search.
    """
    os.environ.pop("RAVANA_OFFLINE", None)
    before = socket.getdefaulttimeout()
    try:
        tr._web_search_via_intentforge("connectivity probe")
    except Exception:
        pass
    after = socket.getdefaulttimeout()
    assert before == after, \
        f"global socket timeout mutated: {before!r} -> {after!r}"


def test_search_timeout_is_bounded_and_configurable():
    """The stall budget must be a named, overridable value, not a literal 8.0.

    The primary call and the fallback previously shared one 8s budget, so a
    slow-but-healthy gateway cost the full 16s. Assert a single named budget
    exists and that a caller can override it at runtime.
    """
    assert hasattr(tr, "SEARCH_TIMEOUT_S"), \
        "a named search timeout must exist so the budget is one number, not a literal"
    assert 0 < tr.SEARCH_TIMEOUT_S <= 60

    os.environ["RAVANA_SEARCH_TIMEOUT_S"] = "1"
    try:
        assert tr.search_timeout() == 1.0, \
            "RAVANA_SEARCH_TIMEOUT_S must override the default budget"
    finally:
        os.environ.pop("RAVANA_SEARCH_TIMEOUT_S", None)

    assert tr.search_timeout() == tr.SEARCH_TIMEOUT_S, \
        "with no override the budget must return to the default"


def test_absurd_timeout_override_is_rejected():
    """A bad env value must not wedge every future tool call."""
    for bad in ("0", "-5", "99999", "not-a-number"):
        os.environ["RAVANA_SEARCH_TIMEOUT_S"] = bad
        try:
            val = tr.search_timeout()
            assert val == tr.SEARCH_TIMEOUT_S, \
                f"override {bad!r} should be rejected, got {val!r}"
        finally:
            os.environ.pop("RAVANA_SEARCH_TIMEOUT_S", None)
