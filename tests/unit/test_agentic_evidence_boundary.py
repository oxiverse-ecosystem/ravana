"""FIX-RV-17: tool output is INTERNAL evidence, never part of the answer.

The defect: on a first-person disclosure ("when i was a teenager i lived in
mumbai") the agentic pre-check fired web_search, stashed the raw IntentForge
response as a string, and the end-of-turn block did

    response = f"{response}\\n{_pending_web_evidence}"

so the user received their reply followed by a 1200-character PREFIX of a JSON
envelope, cut off mid-string. Four faults in one reply: raw payload in the
answer, a world-search for a disclosure, an off-topic result fed back as if it
answered the question, and a payload no consumer could parse.

These tests assert the RULE at the boundary rather than the exact wording, and
they are written to go RED if the fix is reverted:

  - the reply must never carry a tool-output marker
  - a first-person declaration must not reach web_search
  - tool evidence must be parsed into records and written into the graph
  - an off-topic result set must contribute nothing
  - a truncated envelope must yield zero records, never half a document

No authored prose is asserted anywhere: the tests check structure, markers, and
counts, so they cannot be satisfied by writing a nicer sentence.
"""
import json
import os
import sys

import pytest

os.environ.setdefault("RAVANA_OFFLINE", "1")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(_REPO, "ravana", "src"),
           os.path.join(_REPO, "ravana_ml", "src"),
           os.path.join(_REPO, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.agent import decision_gate as _gate  # noqa: E402
from ravana.agent import evidence as _ev  # noqa: E402
from ravana.agent.evidence import parse_tool_output, relevant_records  # noqa: E402
from ravana.agent.tool_registry import (_MAX_PAYLOAD_CHARS,  # noqa: E402
                                        _TRUNCATED_JSON)

# The exact turn from the defect report.
DEFECT_TURN = "when i was a teenager i lived in mumbai"

# Markers that must never appear in a user-facing reply. These are the strings
# the old code concatenated on; asserting their absence is the boundary rule.
TOOL_MARKERS = ("[agentic:", "[web:", "[site]", "[script]", "[git]",
                "[tool error]", "[tool BLOCKED]")


def _envelope(results):
    return json.dumps({"category": "informational", "intent": "local",
                       "results": results})


def _raw(results, marker="web:intentforge"):
    return f"[{marker}] " + _envelope(results)


# --- 1. THE BOUNDARY: no reply may carry tool output ------------------------

def test_reply_never_carries_a_tool_output_marker():
    """The red-capable core assertion, at the process_turn boundary.

    A raw tool payload used to be appended to the reply. Revert the fix and
    this fails: the turn returns a reply containing "[agentic:web_search]".
    """
    from ravana.chat.engine import CognitiveChatEngine

    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="rv17marker")
    try:
        for turn in (DEFECT_TURN,
                     "what is the capital of france",
                     "i moved to berlin two years ago",
                     "explain quantum entanglement"):
            reply = eng.process_turn(turn) or ""
            for marker in TOOL_MARKERS:
                assert marker not in reply, (
                    f"tool output leaked into the user-facing reply "
                    f"(marker={marker!r}) on turn {turn!r}: {reply!r}")
    finally:
        try:
            eng.stop_background_learning()
        except Exception:
            pass


def test_parsed_envelope_never_reaches_a_reply_even_when_search_fires():
    """Force the tool path to fire, and assert the payload is still not in the reply.

    The disclosure gate means the defect turn no longer searches, so this test
    drives the boundary directly: it stashes a raw payload the way the old code
    did and asserts the end-of-turn block does not append it.
    """
    from ravana.chat.engine import CognitiveChatEngine

    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="rv17stash")
    try:
        reply = eng.process_turn(DEFECT_TURN) or ""
        # Simulate the pre-fix state: a raw payload pending at end of turn.
        eng._pending_web_evidence = _raw([
            {"title": "Unrelated", "url": "http://example.invalid/a",
             "content": "a completely different teenager entirely"}])
        after = eng.process_turn("and what is the capital of france") or ""
        assert "[web:intentforge]" not in after, (
            f"a pending raw payload was appended to the reply: {after!r}")
        assert eng._pending_web_evidence is None, (
            "the pending payload was not cleared, so it can leak on a later turn")
        assert reply  # the first turn produced a real reply
    finally:
        try:
            eng.stop_background_learning()
        except Exception:
            pass


# --- 2. THE ROUTING: a disclosure is not a world query ----------------------

@pytest.mark.parametrize("disclosure", [
    DEFECT_TURN,
    "i moved to berlin two years ago",
    "i was a teacher before that",
    "my favourite colour is green",
    "i grew up near the coast",
    "we decided to go with postgres",
])
def test_first_person_declaration_is_not_a_search(disclosure):
    """A first-person declaration carries no information gap to look up."""
    assert _gate._is_self_attributive_disclosure(disclosure) is True, (
        f"first-person disclosure not detected structurally: {disclosure!r}")


@pytest.mark.parametrize("not_disclosure", [
    # world knowledge
    "what is the capital of france",
    "who wrote dune",
    "explain quantum entanglement",
    # a question the user asks ABOUT themselves — must still be answerable
    "what did i do in mumbai",
    "where did i live as a teenager",
    "why do i hate mondays",
    # imperatives are tasks, handled by the tool-noun / social-intent paths
    "search for python tutorials",
    "list all branches",
    "show me the log",
    # third / second person and bare nominals are about the world
    "anant ambani went viral",
    "you should try python",
    "mumbai is crowded",
    "the roman empire fell",
])
def test_non_disclosure_is_still_searchable(not_disclosure):
    """The gate must not become a blocklist that swallows real questions."""
    assert _gate._is_self_attributive_disclosure(not_disclosure) is False, (
        f"over-gated: a non-disclosure was treated as a disclosure: "
        f"{not_disclosure!r}")


def test_defect_turn_does_not_route_to_web_search():
    """End-to-end at the gate: the defect turn produces no ToolCall."""
    from unittest.mock import MagicMock
    from ravana.agent.tool_registry import ToolRegistry

    engine = MagicMock(spec=[])
    engine._is_recall_query = MagicMock(return_value=True)
    engine.curiosity_engine = MagicMock()
    # Maximum curiosity pressure: without the frame gate this WOULD search.
    engine.curiosity_engine.uncertainty_for = MagicMock(return_value=1.0)
    engine.meta_cog = MagicMock()
    engine.meta_cog.current_mode = MagicMock()
    engine.meta_cog.current_mode.value = "UNCERTAIN"
    engine._social_intent = None

    call = _gate.decide_tool_use(engine, DEFECT_TURN, ToolRegistry())
    assert call is None, (
        f"a first-person disclosure was routed to a tool: {call}")


def test_world_query_still_reaches_web_search_under_pressure():
    """The gate removes the wrong search, not the capability."""
    from unittest.mock import MagicMock
    from ravana.agent.tool_registry import ToolRegistry

    engine = MagicMock(spec=[])
    engine._is_recall_query = MagicMock(return_value=True)
    engine.curiosity_engine = MagicMock()
    engine.curiosity_engine.uncertainty_for = MagicMock(return_value=1.0)
    engine.meta_cog = MagicMock()
    engine.meta_cog.current_mode = MagicMock()
    engine.meta_cog.current_mode.value = "UNCERTAIN"
    engine._social_intent = None

    call = _gate.decide_tool_use(engine, "who wrote dune", ToolRegistry())
    assert call is not None and call.tool == "web_search", (
        "web_search became unreachable for a genuine world query")


# --- 3. THE PARSE: raw payload becomes structured records -------------------

def test_intentforge_envelope_parses_into_records():
    raw = _raw([
        {"title": "Mumbai", "url": "http://example.invalid/mumbai",
         "content": "Mumbai is the capital of Maharashtra and a large city",
         "authority": 0.9, "is_local": True},
    ])
    records = parse_tool_output(raw, "what is the capital of maharashtra")
    assert len(records) == 1, f"envelope did not parse: {records}"
    rec = records[0]
    assert rec.title == "Mumbai"
    assert rec.authority == 0.9
    assert rec.is_local is True
    assert "Mumbai" in rec.content


def test_truncated_envelope_yields_no_records():
    """A mid-string cut cannot be parsed, and must not become fake evidence.

    This is the shape of the old `data[:1200]` payload.
    """
    good = _envelope([
        {"title": "Mumbai guide", "url": "http://example.invalid/m",
         "content": "Mumbai is a city in India with a long history"},
    ])
    cut = good[:40]  # a prefix, exactly what the old code produced
    with pytest.raises(ValueError):
        json.loads(cut)  # not parseable, by construction
    records = parse_tool_output(f"[web:intentforge] {cut}", "mumbai")
    assert records == [], (
        f"a truncated envelope produced records instead of nothing: {records}")


def test_registry_truncation_marker_declares_itself():
    """An over-long document is reported as truncated, not silently sliced."""
    assert json.loads(_TRUNCATED_JSON)["truncated"] is True
    assert json.loads(_TRUNCATED_JSON)["results"] == []
    assert _MAX_PAYLOAD_CHARS > 1200, (
        "the bound must be a whole-document bound, larger than the old slice")


def test_unreachable_tool_yields_no_records():
    """An honest failure is zero evidence, never a fabricated record."""
    assert parse_tool_output("", "anything") == []
    records = parse_tool_output(
        "[web] search unavailable: timed out / <urlopen error timed out>",
        "capital of france")
    assert records == [], f"an error payload became evidence: {records}"


# --- 4. THE RELEVANCE: an off-topic result set contributes nothing ----------

def test_off_topic_results_are_dropped():
    """The defect's actual result: news about a DIFFERENT teenager.

    A single shared word ("teenager") is not an answer about the user's own
    life; coverage arithmetic over the query's own terms must reject it.
    """
    raw = _raw([
        {"title": "Anant Ambani admitted to hospital",
         "url": "http://example.invalid/news",
         "content": ("After videos of him in the pond went viral, Anant Ambani "
                     "stepped in to help. The teenager has now been admitted "
                     "to Sir HN Reliance Foundation Hospital for medical "
                     "evaluation and treatment.")},
    ])
    assert parse_tool_output(raw, DEFECT_TURN) == [], (
        "an unrelated result about a different teenager was kept as evidence")


def test_on_topic_result_is_kept():
    """A result that actually covers the query survives the same filter."""
    raw = _raw([
        {"title": "Mumbai childhood",
         "url": "http://example.invalid/bio",
         "content": ("As a teenager in Mumbai I lived near the sea, in a "
                     "flat in Bandra, and went to school there.")},
    ])
    records = parse_tool_output(raw, DEFECT_TURN)
    assert len(records) == 1, f"a genuinely relevant result was dropped: {records}"
    assert records[0].coverage > 0.0


def test_relevance_uses_coverage_not_a_topic_list():
    """Same shape, different topic: the filter follows the QUERY, not a list.

    If relevance were keyword-based, swapping the subject would break it. The
    decision here is driven by the query's own terms, so an unrelated document
    is dropped for a topic nobody wrote down anywhere.
    """
    query = "the capital of madagascar is antananarivo"
    raw = _raw([
        {"title": "Unrelated cycling news",
         "url": "http://example.invalid/cycling",
         "content": "A rider won the race in a narrow sprint on Sunday."},
    ])
    assert parse_tool_output(raw, query) == [], (
        "an off-topic result was kept for a query it does not cover")


def test_empty_query_keeps_records_rather_than_claiming_no_answer():
    """Discarding on an empty basis would be an unearned 'no answer'."""
    raw = _raw([{"title": "T", "url": "http://example.invalid/x",
                 "content": "some content here"}])
    assert len(parse_tool_output(raw, "")) == 1


# --- 5. THE INGESTION: evidence becomes cognitive state ---------------------

def test_evidence_ingestion_writes_into_the_concept_graph():
    """Evidence is parsed into the graph — that is what replaces the reply text."""
    from ravana.chat.engine import CognitiveChatEngine

    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="rv17ingest")
    try:
        before_nodes = len(eng.graph.nodes)
        raw = _raw([
            {"title": "Mumbai", "url": "http://example.invalid/mumbai",
             "content": ("Mumbai is the capital of Maharashtra, a coastal city "
                         "in western India known for its harbour and skyline")},
        ])
        records = parse_tool_output(raw, "what is the capital of maharashtra")
        assert records, "precondition: the fixture must be relevant"
        summary = _ev.ingest_into_graph(eng._auto_expand_concepts, records)
        assert summary["records"] == 1
        assert len(eng.graph.nodes) > before_nodes, (
            f"evidence ingestion added no concepts to the graph "
            f"(before={before_nodes}, after={len(eng.graph.nodes)})")
    finally:
        try:
            eng.stop_background_learning()
        except Exception:
            pass


def test_ingestion_of_nothing_is_a_no_op():
    """No relevant evidence means no graph mutation and no fabrication."""
    from ravana.chat.engine import CognitiveChatEngine

    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="rv17noop")
    try:
        before = len(eng.graph.nodes)
        summary = _ev.ingest_into_graph(eng._auto_expand_concepts, [])
        assert summary["records"] == 0
        assert len(eng.graph.nodes) == before
    finally:
        try:
            eng.stop_background_learning()
        except Exception:
            pass


def test_action_log_does_not_keep_the_raw_payload():
    """A truncated JSON envelope in the action log is the same defect, one
    step further from the user."""
    from ravana.chat.engine import CognitiveChatEngine

    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="rv17log")
    try:
        raw = _raw([{"title": "Mumbai", "url": "http://example.invalid/m",
                     "content": "Mumbai is the capital city of Maharashtra"}])
        records = parse_tool_output(raw, "capital of maharashtra")
        eng._record_agent_action("web_search", "capital of maharashtra", raw,
                                 records=records)
        entry = eng._agent_action_log[-1]
        assert "outcome_head" not in entry, (
            "the raw payload prefix is still being stored in the action log")
        assert entry["evidence"], "structured evidence summary missing"
        assert '"results"' not in json.dumps(entry), (
            f"raw JSON envelope retained in the action log: {entry}")
    finally:
        try:
            eng.stop_background_learning()
        except Exception:
            pass


# --- 6. NO RETRAINING: the whole path is runtime-only -----------------------

def test_parse_and_filter_touch_no_persisted_state():
    """The fix is a boundary/parsing fix, so it must work in one call, cold."""
    raw = _raw([
        {"title": "Mumbai", "url": "http://example.invalid/m",
         "content": "Mumbai is the capital of Maharashtra in India"},
    ])
    first = parse_tool_output(raw, "capital of maharashtra")
    second = parse_tool_output(raw, "capital of maharashtra")
    assert len(first) == len(second) == 1, (
        "relevance is not deterministic — it must be a pure function of the "
        "query and the payload, with nothing learned or rebuilt between calls")
