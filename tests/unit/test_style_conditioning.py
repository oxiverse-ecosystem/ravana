"""Tests for Phase 2 style conditioning.

These exercise the real module (no mocks for the style derivation or the
alignment metric) and the real engine wiring where it matters. The
neuroscience grounding is in the module docstring: interactive alignment
(Pickering & Garrod 2004), lexical entrainment / conceptual pacts (Brennan
& Clark 1996), language-style matching (Ireland et al. 2011), audience
design (Clark & Murphy 1982; Bell 1984).
"""

import ast
import inspect
import os
import sys
import textwrap

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana_ml", "src"))

from ravana.storage.db import CognitiveDB  # noqa: E402
from ravana.learn.dialogue_memory import (  # noqa: E402
    DialogueMemory, DialogueMemoryConfig, DialogueTurn,
)
from ravana.learn.style_conditioning import (  # noqa: E402
    StyleContext, alignment_metric, apply_style_to_realizer,
    build_style_context_from_memory, derive_style_context,
    distribution_distance, function_word_distribution, select_modality,
    style_family, variant_weight_multiplier,
)


DIM = 8

# Same deterministic topic-axis trick as test_dialogue_memory.py, so the
# retrieval path is exercised for real (the style context is derived from
# retrieved turns, so a real retrieval makes the test end-to-end).
_TOPIC_AXES = {
    "music": 0, "song": 0, "melody": 0, "tune": 0,
    "ocean": 1, "sea": 1, "wave": 1, "tide": 1,
    "mountain": 2, "peak": 2, "cliff": 2, "summit": 2,
    "desert": 3, "dune": 3, "sand": 3, "arid": 3,
    "forest": 4, "tree": 4, "leaf": 4, "grove": 4,
    "river": 5, "stream": 5, "brook": 5, "water": 5,
    "cloud": 6, "sky": 6, "rain": 6, "storm": 6,
    "stone": 7, "rock": 7, "boulder": 7, "pebble": 7,
}


def _stable_axis(word: str) -> int:
    import zlib
    return zlib.crc32(word.encode("utf-8")) % DIM


def _vec_fn(word):
    w = (word or "").lower()
    if not w:
        return None
    axis = _TOPIC_AXES.get(w)
    if axis is None:
        axis = _stable_axis(w)
    v = np.zeros(DIM, dtype=np.float32)
    v[axis] = 1.0
    return v


def _db(tmpdir, name="test.db"):
    return CognitiveDB(os.path.join(tmpdir, name))


class _Turn:
    """Minimal stand-in for DialogueTurn (only the fields the derivation reads)."""

    def __init__(self, user_utterance, ravana_reply="", turn_index=0):
        self.user_utterance = user_utterance
        self.ravana_reply = ravana_reply
        self.turn_index = turn_index


# ── StyleContext is derived from the USER half, not RAVANA's replies ─────────

def test_style_context_uses_user_half_not_ravana_replies():
    """The user's utterances drive the profile; RAVANA's replies are ignored.

    Two exemplars with IDENTICAL user utterances but wildly different
    RAVANA replies must produce the same style context. If the derivation
    read the reply half, the two contexts would differ.
    """
    user_texts = [
        "what is the ocean like?",
        "how does the tide work?",
        "why is the sea salty?",
    ]
    a = derive_style_context([_Turn(u, "completely different reply one")
                              for u in user_texts])
    b = derive_style_context([_Turn(u, "an entirely unrelated answer two")
                              for u in user_texts])
    assert a.n_exemplars == 3
    assert b.n_exemplars == 3
    assert a.mean_length == pytest.approx(b.mean_length)
    assert a.question_rate == pytest.approx(b.question_rate)
    assert a.function_word_profile == pytest.approx(b.function_word_profile)
    assert a.top_function_words == b.top_function_words


def test_style_context_detects_question_heavy_user():
    """A user who asks everything has a high question rate and low directness."""
    q = derive_style_context([_Turn("what is the ocean?"),
                              _Turn("how does the tide work?"),
                              _Turn("why is the sea salty?"),
                              _Turn("where do waves come from?")])
    assert q.question_rate == pytest.approx(1.0)
    assert q.directness < 0.62   # not "direct"


def test_style_context_detects_direct_user():
    """A terse, declarative user has a high directness score."""
    d = derive_style_context([_Turn("ocean"),
                              _Turn("tide"),
                              _Turn("sea"),
                              _Turn("wave")])
    assert d.question_rate == pytest.approx(0.0)
    assert d.directness > 0.62


def test_style_context_function_word_profile_normalised():
    """The profile sums to ~1.0 and lists the most frequent function words."""
    ctx = derive_style_context([_Turn("what is the ocean?"),
                                _Turn("how does the tide work?")])
    assert ctx.function_word_profile
    total = sum(ctx.function_word_profile.values())
    assert total == pytest.approx(1.0, abs=1e-6)
    # Most frequent first.
    freqs = list(ctx.function_word_profile.values())
    assert freqs == sorted(freqs, reverse=True)
    assert ctx.top_function_words


def test_style_context_empty_when_no_exemplars():
    """No exemplars -> an unavailable context (the fail-open signal)."""
    ctx = derive_style_context([])
    assert ctx.n_exemplars == 0
    assert not ctx.available
    assert ctx.function_word_profile == {}


def test_style_context_ignores_ravana_reply_text_entirely():
    """A reply full of function words must not enter the profile."""
    user = ["ocean", "tide", "sea"]
    reply_heavy = ["the the the and and or but if when where which"]
    ctx = derive_style_context([_Turn(u, r) for u, r in zip(user, reply_heavy)])
    # The user's utterances are single content words: no function words.
    assert ctx.function_word_profile == {}
    assert ctx.top_function_words == []


# ── fail-open: empty memory -> generation unchanged ──────────────────────────

def test_empty_memory_yields_no_style_context(tmpdir):
    """An empty store produces no style context (fail-open)."""
    db = _db(tmpdir)
    mem = DialogueMemory(db, vector_fn=_vec_fn, dim=DIM)
    assert mem.available
    assert build_style_context_from_memory(mem, "music", k=3) is None
    db.close()


def test_unavailable_memory_yields_no_style_context():
    """A memory with no DB connection yields no style context."""

    class _Broken:
        conn = None
        available = False

    assert build_style_context_from_memory(_Broken(), "music", k=3) is None


def test_no_glove_yields_no_style_context(tmpdir):
    """With no vector function, retrieval returns nothing -> no context."""
    db = _db(tmpdir)
    mem = DialogueMemory(db, vector_fn=None, dim=DIM)
    mem.record_turn(1, "i love music", "music is a pattern of sound",
                    subject="music")
    assert build_style_context_from_memory(mem, "music", k=3) is None
    db.close()


def test_conditioning_noop_without_style():
    """Every consumer no-ops when the style context is None."""

    class _Realizer:
        _variant_weights = {"svo": 1.0, "cleft": 1.0}

    r = _Realizer()
    assert apply_style_to_realizer(r, None) is False
    assert r._variant_weights == {"svo": 1.0, "cleft": 1.0}
    assert select_modality(None, "related_weak") == "related_weak"
    assert variant_weight_multiplier("svo", None) == 1.0


def test_conditioning_noop_with_empty_style():
    """An unavailable StyleContext (n_exemplars == 0) is also a no-op."""

    class _Realizer:
        _variant_weights = {"svo": 1.0}

    r = _Realizer()
    assert apply_style_to_realizer(r, StyleContext()) is False
    assert r._variant_weights == {"svo": 1.0}


# ── conditioning applies to the realizer weights ────────────────────────────

def test_direct_user_biases_plain_variants():
    """A direct user boosts plain SVO and suppresses elaborate variants."""
    style = derive_style_context([_Turn("ocean"), _Turn("tide"),
                                  _Turn("sea"), _Turn("wave")])
    assert style_family(style) == "direct"
    assert variant_weight_multiplier("svo", style) > 1.0
    assert variant_weight_multiplier("cleft", style) < 1.0


def test_elaborate_user_biases_elaborate_variants():
    """An indirect, hedged user boosts elaborate variants."""
    style = derive_style_context([_Turn("perhaps the ocean is somewhat vast, maybe?"),
                                  _Turn("i think the tide possibly works that way?")])
    assert style_family(style) == "elaborate"
    assert variant_weight_multiplier("cleft", style) > 1.0
    assert variant_weight_multiplier("svo", style) < 1.0


def test_apply_style_multiplies_existing_weights():
    """The realizer's own weights are multiplied, never replaced."""

    class _Realizer:
        _variant_weights = {"svo": 2.0, "cleft": 0.5}

    style = derive_style_context([_Turn("ocean"), _Turn("tide"),
                                  _Turn("sea"), _Turn("wave")])
    r = _Realizer()
    assert apply_style_to_realizer(r, style) is True
    # svo was boosted (2.0 * 1.5 = 3.0), cleft suppressed (0.5 * 0.6 = 0.3).
    assert r._variant_weights["svo"] == pytest.approx(3.0)
    assert r._variant_weights["cleft"] == pytest.approx(0.3)


def test_modality_conditioning_by_directness():
    """A direct user gets a more certain marker; an indirect user a softer one."""
    direct = derive_style_context([_Turn("ocean"), _Turn("tide"),
                                   _Turn("sea"), _Turn("wave")])
    indirect = derive_style_context([
        _Turn("perhaps the ocean is somewhat vast, maybe?"),
        _Turn("i think the tide possibly works that way?")])
    assert select_modality(direct, "unknown") == "likely"
    assert select_modality(indirect, "certain") == "possible"
    # The trace-strength band is the one the ignorance hedge frames actually
    # use, so it must shift too -- otherwise conditioning silently no-ops for
    # exactly the replies it was meant to affect.
    assert select_modality(direct, "related_weak") == "related_strong"
    assert select_modality(indirect, "related_strong") == "related_weak"
    # Neutral profile must not move the marker, on either scale. Measured on
    # candidate fixtures: plain statements score 0.770 (direct) and fully
    # hedged ones 0.350 (indirect) -- the metric is driven by hedge rate and
    # utterance length, so a genuinely mid-scale profile needs one hedged and
    # one plain utterance. Fixture C below measures 0.420.
    neutral = derive_style_context([
        _Turn("the ocean is perhaps quite vast"),
        _Turn("what do you think the tide does?")])
    assert 0.38 < neutral.directness < 0.62, neutral.directness
    assert select_modality(neutral, "related_weak") == "related_weak"
    assert select_modality(neutral, "unknown") == "unknown"


# ── alignment metric ─────────────────────────────────────────────────────────

def test_alignment_metric_is_computable_and_bounded():
    """The metric returns all four channels in range."""
    m = alignment_metric(["what is the ocean like?"],
                         ["the ocean is vast and deep"])
    for key in ("function_word_distance", "length_ratio",
                "question_rate_delta", "composite"):
        assert key in m
    assert 0.0 <= m["function_word_distance"] <= 1.0
    assert 0.0 <= m["composite"] <= 1.0


def test_alignment_metric_moves_toward_alignment():
    """A reply that mirrors the user's function words scores higher.

    This is the measurable test of the hypothesis: as RAVANA's phrasing
    converges toward the user's, the composite alignment must rise.
    """
    user = ["what is the ocean like?", "how does the tide work?",
            "why is the sea salty?"]
    # Deliberately mismatched reply: long, declarative, different function words.
    mismatched = alignment_metric(user, [
        "the ocean is a large body of salt water that covers most of the "
        "planet surface and is very deep",
        "tides are caused by the gravitational pull of the moon on the "
        "earth oceans and they rise and fall twice a day",
        "sea salt comes from rocks on the land and rivers carry it to the "
        "ocean where it accumulates over millions of years",
    ])
    # Aligned reply: short, question-returning, similar function-word mix.
    aligned = alignment_metric(user, [
        "what is the ocean? it is vast.",
        "how does the tide work? the moon pulls it.",
        "why is the sea salty? the rocks bring salt.",
    ])
    assert aligned["composite"] > mismatched["composite"]
    assert aligned["function_word_distance"] < mismatched["function_word_distance"]


def test_alignment_metric_empty_windows():
    """Empty windows return the neutral/empty metric, not an exception."""
    m = alignment_metric([], [])
    assert m["composite"] == 0.0
    assert m["function_word_distance"] == 1.0


def test_distribution_distance_bounds():
    """Jensen-Shannon distance is 0 for identical, 1 for disjoint."""
    p = {"the": 0.5, "is": 0.5}
    assert distribution_distance(p, p) == pytest.approx(0.0, abs=1e-9)
    assert distribution_distance(p, {"and": 1.0}) == pytest.approx(1.0, abs=1e-9)
    assert distribution_distance({}, {}) == pytest.approx(0.0)
    assert distribution_distance(p, {}) == pytest.approx(1.0)


def test_function_word_distribution_ignores_content_words():
    """Only closed-class words enter the distribution."""
    d = function_word_distribution("the ocean is vast")
    assert "the" in d
    assert "is" in d
    assert "ocean" not in d
    assert "vast" not in d


# ── engine wiring ────────────────────────────────────────────────────────────

def test_engine_style_conditioning_off_by_default(tmpdir):
    """The flag is off on a fresh engine, so generation is unchanged."""
    from ravana.chat.engine import CognitiveChatEngine
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                            data_dir=str(tmpdir), user_suffix="_style_off")
    assert e.use_style_conditioning is False
    # With the flag off, no style context is ever built.
    assert e.build_style_context("hello") is None
    e.dialogue_memory.close()


def test_engine_conditioning_does_not_fire_when_disabled(tmpdir):
    """Even with a populated memory, a disabled flag means no conditioning."""
    from ravana.chat.engine import CognitiveChatEngine
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                            data_dir=str(tmpdir), user_suffix="_style_disabled")
    e.process_turn("what is the ocean like?")
    e.process_turn("how does the tide work?")
    assert e.dialogue_memory.count() == 2
    # Flag still off -> no context, no realizer change.
    assert e.use_style_conditioning is False
    assert e.build_style_context("ocean") is None
    e.dialogue_memory.close()


def test_engine_conditioning_fires_when_enabled(tmpdir):
    """With the flag on and a populated memory, a style context is built."""
    from ravana.chat.engine import CognitiveChatEngine
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                            data_dir=str(tmpdir), user_suffix="_style_on")
    e.use_style_conditioning = True
    e.process_turn("what is the ocean like?")
    e.process_turn("how does the tide work?")
    e.process_turn("why is the sea salty?")
    assert e.dialogue_memory.count() == 3
    ctx = e.build_style_context("ocean tide")
    assert ctx is not None
    assert ctx.n_exemplars > 0
    assert ctx.question_rate > 0.5
    e.dialogue_memory.close()


def test_engine_alignment_log_populates(tmpdir):
    """The alignment metric is recorded automatically across turns."""
    from ravana.chat.engine import CognitiveChatEngine
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                            data_dir=str(tmpdir), user_suffix="_style_align")
    for q in ("what is the ocean like?", "how does the tide work?",
              "why is the sea salty?", "where do waves come from?"):
        e.process_turn(q)
    assert len(e._alignment_log) > 0
    last = e._alignment_log[-1]["metric"]
    assert 0.0 <= last["composite"] <= 1.0
    e.dialogue_memory.close()


def test_engine_alignment_trend_is_computable(tmpdir):
    """alignment_trend returns a float once there are two samples.

    The trend compares the mean of the first ``window`` samples against the
    mean of the last ``window`` samples. With a window of 1 the two samples
    are compared directly.
    """
    from ravana.chat.engine import CognitiveChatEngine
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                            data_dir=str(tmpdir), user_suffix="_style_trend")
    assert e.alignment_trend() is None
    e.record_alignment(1, {"composite": 0.3})
    e.record_alignment(2, {"composite": 0.6})
    assert e.alignment_trend(window=1) == pytest.approx(0.3, abs=1e-6)
    e.dialogue_memory.close()


def test_engine_style_context_survives_reload(tmpdir):
    """The flag and context persist through save/load."""
    from ravana.chat.engine import CognitiveChatEngine
    suffix = "_style_reload"
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                            data_dir=str(tmpdir), user_suffix=suffix)
    e.use_style_conditioning = True
    e.process_turn("what is the ocean like?")
    e.process_turn("how does the tide work?")
    ctx = e.build_style_context("ocean tide")
    assert ctx is not None
    e.save()
    e.dialogue_memory.close()

    e2 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                             data_dir=str(tmpdir), user_suffix=suffix)
    assert e2.use_style_conditioning is True
    assert e2._style_context is not None
    assert e2._style_context.n_exemplars == ctx.n_exemplars
    e2.dialogue_memory.close()


# ── de-hardcoding: no long string literals ──────────────────────────────────

def test_module_has_no_long_string_literals():
    """The module authors no reply strings.

    Every NON-DOCSTRING string literal must be short (<= 45 chars, the same
    bar ``chat/hedges.py`` uses for its marker lexicon). A long literal would
    be authored prose smuggled into the conditioning layer. Docstrings are
    excluded: they are documentation, not runtime strings.
    """
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "ravana", "src", "ravana", "learn",
                        "style_conditioning.py")
    with open(path, "r", encoding="utf-8") as fh:
        source = fh.read()
    tree = ast.parse(source)
    # Collect the spans of every docstring so they can be excluded.
    docstring_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) \
                    and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                doc = body[0].value
                for ln in range(doc.lineno, (doc.end_lineno or doc.lineno) + 1):
                    docstring_lines.add(ln)
    long_literals = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if len(node.value) > 45 and node.lineno not in docstring_lines:
                long_literals.append((node.lineno, node.value[:60]))
    assert long_literals == [], (
        "long string literals found (authored prose?): %s" % long_literals)


def test_module_is_python310_compatible():
    """No match statements / PEP 604 unions in runtime positions."""
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "ravana", "src", "ravana", "learn",
                        "style_conditioning.py")
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src, feature_version=(3, 10))
    # No ast.Match node (the structural test for a match statement).
    assert not any(isinstance(n, ast.Match) for n in ast.walk(tree))
    # No PEP 604 union (``X | Y``) in a runtime annotation position. A set
    # union (``set(a) | set(b)``) is ordinary Python and is allowed; only
    # annotation contexts are checked.
    for parent in ast.walk(tree):
        ann = None
        if isinstance(parent, ast.AnnAssign):
            ann = parent.annotation
        elif isinstance(parent, ast.arg):
            ann = parent.annotation
        elif isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            ann = parent.returns
        if ann is None:
            continue
        for sub in ast.walk(ann):
            if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.BitOr):
                raise AssertionError(
                    "PEP 604 union in annotation at line %s" % sub.lineno)


# ── integration: conditioning reaches the syntax path ────────────────────────

def test_conditioning_reaches_realizer_on_real_engine(tmpdir):
    """With the flag on, the realizer's variant weights are actually nudged."""
    from ravana.chat.engine import CognitiveChatEngine
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                            data_dir=str(tmpdir), user_suffix="_style_realizer")
    e.use_style_conditioning = True
    # A terse, declarative user -> direct family -> plain variants boosted.
    for q in ("ocean", "tide", "sea", "wave"):
        e.process_turn(q)
    before = dict(e.surface_realizer._variant_weights)
    ctx = e.build_style_context("ocean tide")
    assert ctx is not None
    changed = e.apply_style_conditioning(ctx)
    after = dict(e.surface_realizer._variant_weights)
    assert changed is True
    assert before != after
    e.dialogue_memory.close()
