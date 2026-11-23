"""Fake-based tests for the RSI query-time evidence helpers.

No engine boot, no graph, no network, no pytest fixtures that construct an
engine. Every test uses plain dataclass fakes and the module's pure functions.
The env flag is restored after each test.
"""
import os
import sys

import pytest

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import the module flat so no chat/engine package __init__ runs (no engine boot).
_CHAT_DIR = os.path.join(PROJ, "ravana", "src", "ravana", "chat")
for _p in (_CHAT_DIR, PROJ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import schema_induction as si  # noqa: E402


QUERY = ("Considering the examples above, what is true of thaw? "
         "Options: sunset, typhoon, mist, hail. Answer briefly.")


@pytest.fixture(autouse=True)
def _flag_off():
    old = os.environ.get(si.ENV_FLAG)
    os.environ.pop(si.ENV_FLAG, None)
    yield
    if old is None:
        os.environ.pop(si.ENV_FLAG, None)
    else:
        os.environ[si.ENV_FLAG] = old


def test_mask_options_pins_real_subject_not_last_option():
    main, options, masked = si.mask_options(QUERY)
    assert "thaw" in main
    assert options == ["sunset", "typhoon", "mist", "hail"]
    assert "hail" not in masked and "typhoon" not in masked
    assert "Options:" not in masked


def test_no_options_leaves_query_untouched():
    main, options, masked = si.mask_options("what is thaw")
    assert main == masked == "what is thaw"
    assert options == []


def test_flag_off_disables_trusted_edge_lookup():
    edge = si.EdgeView("mist", "causal", "thaw", "curated", "conversation", 0.9)
    assert si.trusted_incident_edges("thaw", [edge]) == []
    assert si.grounding_from_trusted_edge("thaw", [edge], ["mist"]) is None


def test_trusted_edge_admitted_untrusted_rejected():
    taught = si.EdgeView("mist", "causal", "thaw", "curated", "conversation", 0.9)
    cooccur = si.EdgeView("hail", "semantic", "thaw", "co_occurrence", "", 0.99)
    os.environ[si.ENV_FLAG] = "1"
    admitted = si.trusted_incident_edges("thaw", [taught, cooccur])
    assert admitted == [taught]
    assert cooccur not in admitted


def test_taught_edge_selects_taught_candidate_not_option():
    taught = si.EdgeView("mist", "causal", "thaw", "curated", "conversation", 0.9)
    os.environ[si.ENV_FLAG] = "1"
    choice = si.grounding_from_trusted_edge(
        "thaw", [taught], candidates=["sunset", "typhoon", "mist", "hail"])
    assert choice == "mist"


def test_grounding_fails_closed_when_candidate_not_supported():
    taught = si.EdgeView("mist", "causal", "thaw", "curated", "conversation", 0.9)
    os.environ[si.ENV_FLAG] = "1"
    assert si.grounding_from_trusted_edge(
        "thaw", [taught], candidates=["sunset", "hail"]) is None


def test_single_trace_never_induces_schema():
    dec = si.Declaration("mist", "CAUSE", "thaw")
    tr = si.TrustedTrace(dec, {"edge_kind": "curated"}, "t1")
    assert si.induce_schema_hypotheses([tr]) == []


def test_converged_schema_is_unassertable():
    traces = [
        si.TrustedTrace(si.Declaration("gale", "CAUSE", "rainbow"), {}, "t1"),
        si.TrustedTrace(si.Declaration("mist", "CAUSE", "thaw"), {}, "t2"),
    ]
    hyps = si.induce_schema_hypotheses(traces)
    assert len(hyps) == 1
    assert hyps[0].support == 2
    assert hyps[0].assertable is False
    assert si.SchemaHypothesis("CAUSE", 2, (), True).assertable is False


def test_schema_alone_never_admits_for_query():
    hyp = si.SchemaHypothesis("CAUSE", 2, (), False)
    assert si.admit_for_query(hyp, []) is False
    observed = [si.TrustedTrace(si.Declaration("mist", "CAUSE", "thaw"), {}, "t2")]
    assert si.admit_for_query(hyp, observed) is True


def test_parse_declaration_inverse_and_negation():
    phrases = {"causes": ("CAUSE", 1), "is caused by": ("CAUSE", -1)}
    forward = si.parse_declaration("gale causes rainbow", phrases)
    inverse = si.parse_declaration("rainbow is caused by gale", phrases)
    assert forward is not None and forward.key == ("gale", "CAUSE", "rainbow")
    assert inverse is not None and inverse.key == ("gale", "CAUSE", "rainbow")
    assert si.parse_declaration("gale does not cause rainbow", phrases) is None
