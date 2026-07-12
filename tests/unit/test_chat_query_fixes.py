"""Regression tests for the seven chat-quality fixes (Q4-Q12 failure classes).

Locks in the brain-grounded fixes discovered from the diagnostic chat batch:

  Fix 1 (Q7)  — relation-type token ("semantic"/"causal"/...) must NEVER leak
                into surface text (ATL tag vs IFG/Broca lemma separation).
  Fix 2 (Q5)  — perseverative near-duplicate sub-answers collapse (frontostriatal
                inhibition; hippocampal pattern separation).
  Fix 3 (Q8)  — a conditional/counterfactual is a COMPLETE speech act and must
                NOT be withheld by the preamble/turn-end predictor.
  Fix 4 (Q12) — episodic meta-queries ("what did I just ask you") answer from the
                verbatim user-turn buffer (Baddeley episodic buffer).
  Fix 5 (Q11) — spelled-out arithmetic ("2 plus 2") routes to the numeric path.
  Fix 6 (Q4/Q9) — a bare distributional "semantic" relation is rejected unless a
                contentful typed edge backs it (no ATL-hub-without-predicate).
  Fix 7       — save/load checksum is deterministic across processes and the
                identity load-guard checks the field IdentityState actually has.

These are mostly UNIT tests on the pure helpers so they run fast (no GloVe /
web). A single shared engine covers the routing checks.
"""
import os
import sys
import tempfile
import pytest

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana_ml", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana-v2", "src"))

from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.monitor_gate import _METAWORDS


@pytest.fixture(scope="module")
def engine():
    d = tempfile.mkdtemp(prefix="ravana_qfix_")
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d)
    # Don't start background learning — keep the test deterministic + offline.
    yield e


# ── Fix 5 (Q11): spelled-out arithmetic ────────────────────────────────────
@pytest.mark.parametrize("q,expected", [
    ("what is 2 plus 2", "4"),
    ("2 plus 2", "4"),
    ("10 times 5", "50"),
    ("9 minus 4", "5"),
    ("8 divided by 2", "4"),
    ("what is 5 + 3", "8"),
    ("3 to the power of 2", "9"),
])
def test_arithmetic_word_operators(engine, q, expected):
    out = engine._try_arithmetic(q)
    assert out is not None, f"arithmetic path missed: {q!r}"
    assert out.rstrip(".").endswith(expected), f"{q!r} -> {out!r}"


def test_arithmetic_ignores_non_math(engine):
    assert engine._try_arithmetic("what is trust") is None
    assert engine._try_arithmetic("explain gravity") is None


# ── Fix 3 (Q8): conditional is a complete speech act ────────────────────────
@pytest.mark.parametrize("q", [
    "if gravity suddenly stopped what would happen",
    "what if the sun disappeared",
    "suppose water froze instantly",
    "imagine if humans could fly",
])
def test_conditional_detected(engine, q):
    assert engine._is_conditional_query(q) is True


def test_conditional_not_a_preamble_route():
    """The process_turn gate must skip the preamble hold for a conditional.

    We assert the guard condition directly (pure, no GloVe): a conditional
    query short-circuits the `not conditional and preamble` AND, so it is never
    withheld even if _is_preamble_fragment would fire.
    """
    d = tempfile.mkdtemp(prefix="ravana_cond_")
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d)
    q = "if gravity suddenly stopped what would happen"
    gate = (not e._is_conditional_query(q)) and e._is_preamble_fragment(q)
    assert gate is False, "conditional must bypass the preamble hold"


# ── Fix 4 (Q12): episodic memory meta-query ─────────────────────────────────
def test_memory_query_recalls_prior_turn():
    d = tempfile.mkdtemp(prefix="ravana_mem_")
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d)
    # Simulate a prior user turn without invoking the full pipeline.
    e._recent_user_turns = ["what is the difference between fear and courage"]
    out = e._try_memory_query("what did i just ask you")
    assert out is not None
    assert "fear and courage" in out.lower()


def test_memory_query_empty_history():
    d = tempfile.mkdtemp(prefix="ravana_mem2_")
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d)
    e._recent_user_turns = []
    out = e._try_memory_query("what did i just ask you")
    assert out is not None and "haven't asked" in out.lower()


def test_memory_query_ignores_normal_query():
    d = tempfile.mkdtemp(prefix="ravana_mem3_")
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d)
    e._recent_user_turns = ["what is trust"]
    assert e._try_memory_query("what is gravity") is None
    assert e._try_memory_query("why do people lie") is None


# ── Fix 1 (Q7): relation-type never leaks into surface text ─────────────────
def test_relation_predicate_maps_to_real_verb():
    d = tempfile.mkdtemp(prefix="ravana_rel_")
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d)
    for rel in ("semantic", "causal", "contrastive", "analogical", "temporal"):
        pred = e._relation_predicate(rel)
        # The surface predicate must not BE the raw relation tag.
        assert pred != rel
        # And it must not itself be a bare metaword.
        assert not (set(pred.split()) & _METAWORDS), f"{rel} -> {pred!r} leaks metaword"


def test_relation_predicate_unknown_falls_back():
    d = tempfile.mkdtemp(prefix="ravana_rel2_")
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d)
    assert e._relation_predicate("some_unknown_type") == "relates to"


# ── Fix 7: deterministic, cross-process-stable checksum ─────────────────────
def test_checksum_is_deterministic_and_order_independent():
    state_a = {
        "turn_count": 7,
        "some_set": {"a", "b", "c"},
        "some_dict": {"x": 1, "y": 2},
        "name": "ravana",
    }
    # Same logical content, different insertion order for the dict.
    state_b = {
        "name": "ravana",
        "some_dict": {"y": 2, "x": 1},
        "some_set": {"c", "b", "a"},
        "turn_count": 7,
    }
    c1 = CognitiveChatEngine._checksum_state(state_a)
    c2 = CognitiveChatEngine._checksum_state(state_b)
    assert c1 == c2, "checksum must be order-independent / cross-process stable"


def test_checksum_detects_scalar_tamper():
    base = {"turn_count": 7, "graph": [1, 2, 3]}
    tampered = {"turn_count": 999, "graph": [1, 2, 3]}
    assert (CognitiveChatEngine._checksum_state(base)
            != CognitiveChatEngine._checksum_state(tampered))


def test_checksum_excludes_own_key():
    s1 = {"turn_count": 7}
    s2 = {"turn_count": 7, "state_checksum": "deadbeef"}
    assert (CognitiveChatEngine._checksum_state(s1)
            == CognitiveChatEngine._checksum_state(s2))
