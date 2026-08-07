"""Regression tests for experiential self-model routing (personality).

Root cause (investigated): self-referential probes ("do you ever feel lonely",
"what are you afraid of", "would you rather...") were routed to the SEMANTIC
pipeline — either the episodic echo ("you told me earlier: ...", a
source-monitoring error) or the internal-knowledge / web consult (a
dictionary/definition of the grounded subject, e.g. "something may refer to
..."). Brain-faithful fix (Northoff et al. 2006): self-referential processing
is functionally dissociable from and PRECEDES semantic retrieval, so these are
answered from the self-model + affect via `_route_self_experience`, which
process_turn now admits BEFORE the fact-reasoning echo and the semantic
consult.

Run from repo root:
    python -m pytest tests/unit/test_self_experience_routing.py -v
"""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana_ml", "src"))

from ravana.chat.engine import CognitiveChatEngine


def _build_engine():
    # data_dir with already-seeded weights keeps the boot fast and offline.
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               data_dir="/tmp/ravana_m5_test")


# ── 1. self-experience probes are intercepted as self_experience, not dict ──
def test_self_experience_routes_to_persona():
    eng = _build_engine()
    probes = [
        "do you ever feel lonely?",
        "what are you afraid of?",
        "do you have any regrets?",
        "tell me how you feel when you learn something new",
        "would you rather be human or stay an ai?",
        "if you had a body, what would you look like?",
        "do you have free will?",
        "what makes you happy?",
        "are you sad right now?",
    ]
    for q in probes:
        eng.process_turn(q)
        strat = getattr(eng, "_last_strategy", "")
        assert strat == "self_experience", f"{q} -> {strat}"
        out = eng._last_responses[-1]
        assert not out.startswith("you told me earlier"), f"{q} echoed user: {out}"
        assert "refer to" not in out, f"{q} leaked definition: {out}"


# ── 2. world / third-person / opinion queries are NOT hijacked ───────────────
def test_world_queries_not_hijacked():
    eng = _build_engine()
    # Third-person experiencer: about people, not the agent.
    eng.process_turn("do people feel lonely in a crowd?")
    assert getattr(eng, "_last_strategy", "") != "self_experience", \
        "third-person query hijacked by self gate"

    # Opinion-about-topic: leave to the stance resolver, not the persona gate.
    eng.process_turn("what do you think about music?")
    assert getattr(eng, "_last_strategy", "") != "self_experience", \
        "opinion query hijacked by self gate"

    # World fact-seek: must NOT become a self-probe.
    eng.process_turn("what is trust?")
    assert getattr(eng, "_last_strategy", "") != "self_experience", \
        "definitional query hijacked by self gate"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))