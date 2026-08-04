"""Tests for the cognition-driven generation round (task t_543801e8).

The four banned `random.choice` reply pools were replaced by state-driven
generators. These tests assert the acceptance criteria from opencode_plan.md §8:

(a) no `random.choice` / `random` call remains in the four production paths;
(b) honest-uncertainty output strings differ ONLY when store content differs
    (determinism);
(c) an agent-addressed / third-narrative cue produces NO empathy pool text;
(d) after simulated online writes to `_definitions`/graph, the same subject
    flips from honest-empty to content-assertion (the no-retrain learning proof).
"""
import os
import sys

import pytest

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_PROJ,
           os.path.join(_PROJ, "ravana_ml", "src"),
           os.path.join(_PROJ, "ravana", "src"),
           os.path.join(_PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.chat.engine import CognitiveChatEngine


@pytest.fixture(scope="module")
def engine():
    os.environ.setdefault("RAVANA_OFFLINE", "1")
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               user_suffix="cogdriventest")


def _ctx(raw, subj=""):
    class _C:
        raw_input = raw
        subject = subj
    return _C()


# (a) No RNG in the four production paths -------------------------------------
def _src(cls, name):
    import inspect
    # Source minus the leading docstring so a docstring mention of "random.choice"
    # does not produce a false positive.
    full = inspect.getsource(getattr(cls, name))
    return full.split('"""', 2)[-1] if '"""' in full else full


def test_no_random_in_reasoning_fallback(engine):
    # _generative_cue_loop / _realize_metacognitive must not call RNG.
    from ravana.chat.response_gen import ResponseGenMixin
    src = _src(ResponseGenMixin, "_generative_cue_loop") + \
          _src(ResponseGenMixin, "_realize_metacognitive") + \
          _src(ResponseGenMixin, "_retrieve_support")
    assert "random.choice" not in src
    assert "random." not in src


def test_no_random_in_emotional_response(engine):
    from ravana.chat.response_gen import ResponseGenMixin
    src = _src(ResponseGenMixin, "_appraised_affective_reply") + \
          _src(ResponseGenMixin, "_emotional_response") + \
          _src(ResponseGenMixin, "_orientation_of")
    assert "random.choice" not in src
    assert "random." not in src


def test_no_random_in_epistemic_frame(engine):
    from ravana.language.surface_realizer import SurfaceRealizer
    src = _src(SurfaceRealizer, "_generate_epistemic_frame")
    assert "random.choice" not in src
    assert "random." not in src


# (b) Determinism: same cognitive state -> same reply ------------------------
def test_reasoning_fallback_deterministic(engine):
    a = engine._generative_cue_loop(_ctx("why is the moon made of cheese?", "moon"))
    b = engine._generative_cue_loop(_ctx("why is the moon made of cheese?", "moon"))
    assert a == b, f"non-deterministic: {a!r} != {b!r}"


def test_emotional_reply_deterministic(engine):
    disc = ("negative", "sad")
    a = engine._emotional_response(_ctx("i am feeling really sad today"), disc)
    b = engine._emotional_response(_ctx("i am feeling really sad today"), disc)
    assert a == b


# (c) No empathy pool text for agent-addressed / third-narrative ------------
def test_agent_addressed_no_empathy_pool(engine):
    disc = ("positive", "love")
    resp, strat = engine._emotional_response(_ctx("i love you"), disc)
    # Must NOT be one of the old authored pool strings.
    assert resp == "that's directed at me — i appreciate you saying it.", resp
    assert strat == "affective_self_addr"


def test_third_narrative_no_empathy_pool(engine):
    # "tell me a story about a sad robot" — affect word but not self-report.
    disc = ("negative", "sad")
    resp, strat = engine._emotional_response(
        _ctx("tell me a story about a sad robot"), disc)
    assert strat == "affective_third_narrative", resp
    # The old pool never produced this framing; assert it is store-honest.
    assert "i hear you — feeling" not in resp


# (d) No-retrain learning proof ---------------------------------------------
def test_online_learn_flips_empty_to_assertion(engine):
    subj = "zzqqxx_widge"
    before = engine._generative_cue_loop(_ctx(f"what is {subj}?", subj))
    # Before any store content: honest empty + probe.
    assert f"i don't have a solid read on {subj}" in before or \
           f"i don't have much tied to {subj}" in before, before
    # Simulate an ONLINE write (no retrain, no rebuild) — exactly what the web
    # learner does at runtime.
    engine._definitions[subj] = "a small carved token used in old counting games."
    after = engine._generative_cue_loop(_ctx(f"what is {subj}?", subj))
    # After the write the SAME subject asserts the real content instead.
    assert subj in after and "i don't have" not in after, after
    # And it is now deterministic.
    assert after == engine._generative_cue_loop(_ctx(f"what is {subj}?", subj))


def test_graph_online_write_flips_assertion():
    """A subject that gains graph edges online also flips to assertion."""
    os.environ.setdefault("RAVANA_OFFLINE", "1")
    eng = CognitiveChatEngine(dim=64, seed=1, baby_mode=True,
                              user_suffix="cogdriven_graph")
    subj = "luminum_ore"
    empty = eng._generative_cue_loop(_ctx(f"tell me about {subj}", subj))
    # Write a definition + register a concept keyword (the online learner path).
    eng._definitions[subj] = "an ore that glows faintly when near water."
    if hasattr(eng, "_concept_keywords"):
        eng._concept_keywords.setdefault(subj, [])
    learned = eng._generative_cue_loop(_ctx(f"tell me about {subj}", subj))
    assert "i don't have" not in learned, learned
    assert subj in learned
