"""De-hardcoding guard for chat/hedges.py (backlog task 4).

The module used to hold ``_HEDGE_FRAMES`` — twenty hand-authored English
templates keyed by (mechanism, modality), with {subj}/{rel}/{snip} spliced
in. The content of the reply came from the lookup table, not from RAVANA's
state. These tests pin the de-hardcoded contract so the table cannot quietly
come back:

  (a) NO AUTHORED PROSE — every reply-producing literal in the module is a
      short modal tag or a thin connective. This is the mechanical form of
      the audit grep, run on the SOURCE so a future edit is caught here
      rather than by a human auditor.
  (b) CONTENT IS STATE — the subject/relation/snippet the caller retrieved is
      what shows up in the frame; two different topics get two different
      frames, and neither one gets a canned clause.
  (c) MODALITY IS MEASURED — the marker tracks modality_from_support, a real
      support number, not a keyword.
  (d) THE LEXICON GROWS AT RUNTIME — the deciding seed-vs-hardcoding test.
      RAVANA can add its own marker online, with no code change and no
      retrain.
"""
import ast
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(_REPO, "ravana", "src"),
           os.path.join(_REPO, "ravana_ml", "src"),
           os.path.join(_REPO, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.chat import hedges  # noqa: E402
from ravana.chat.hedges import (  # noqa: E402
    MODALITY, TRACE, hedge_frame, modality_from_support, register_marker,
)

_HEDGES_PY = os.path.join(_REPO, "ravana", "src", "ravana", "chat", "hedges.py")


# --- (a) NO AUTHORED PROSE ----------------------------------------------------
# The audit grep's threshold. A reply literal at or above this length is the
# signature of authored personality behind a lookup, so it must not exist.
MAX_REPLY_LITERAL = 45


def _docstring_nodes(tree):
    """The set of Constant nodes that are docstrings (never reply strings).

    A docstring is a constant in the AST, so the naive walk flags the module
    docstring as 'authored prose'. Excluding docstrings is correct, not a
    loosening: prose in a docstring is documentation, prose in a frame is a
    scripted reply, and only the second one RAVANA can never grow out of.
    """
    ids = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            ids.add(id(first.value))
    return ids


def _string_literals():
    """Every NON-docstring string literal in hedges.py, as (lineno, value)."""
    with open(_HEDGES_PY, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), _HEDGES_PY)
    docstrings = _docstring_nodes(tree)
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docstrings):
            out.append((node.lineno, node.value))
    return out


def test_no_long_reply_literals_in_hedges_source():
    """No reply-producing string in hedges.py is authored prose."""
    long_literals = [
        (lineno, val)
        for lineno, val in _string_literals()
        if len(val) >= MAX_REPLY_LITERAL and not val.startswith((" ", "-", "="))
    ]
    assert not long_literals, (
        "hedges.py contains authored reply prose: %r — the hedge frame must be "
        "composed from state, not looked up" % (long_literals,)
    )


def test_hedge_frames_table_is_gone():
    """The authored template table must not reappear."""
    assert not hasattr(hedges, "_HEDGE_FRAMES"), (
        "_HEDGE_FRAMES is an authored-prose table; it must stay deleted"
    )


def test_every_marker_is_a_short_modal_tag():
    """Markers are vocabulary, not sentences."""
    for state in list(MODALITY) + list(TRACE):
        assert state in hedges._MARKERS, "no marker registered for %r" % state
        assert len(hedges._MARKERS[state]) < MAX_REPLY_LITERAL, (
            "marker for %r is authored prose, not a modal tag: %r"
            % (state, hedges._MARKERS[state])
        )


# --- (b) CONTENT IS STATE ----------------------------------------------------
def test_ignorance_frame_reports_the_actual_relation():
    """The retrieved relation is the content; nothing is canned around it."""
    frame = hedge_frame("ignorance", "related_weak", subj="myelination",
                        rel="saltatory conduction")
    assert "saltatory conduction" in frame
    assert "myelination" in frame


def test_different_topics_render_differently():
    """Variety comes from what RAVANA retrieved, not from picking a template."""
    a = hedge_frame("ignorance", "related_weak", subj="myelination", rel="saltatory conduction")
    b = hedge_frame("ignorance", "related_weak", subj="photosynthesis", rel="chloroplast")
    assert a != b, "two different topics rendered the same frame"


def test_frame_is_deterministic_no_rng():
    """Same state -> same frame. The old modulo picker is gone."""
    args = ("ignorance", "related_strong", "entropy", "disorder")
    assert hedge_frame(*args) == hedge_frame(*args)


def test_web_frame_carries_retrieved_snippet_and_source():
    frame = hedge_frame("web", "possible", snip="tidal locking slows a moon's spin",
                        src="arxiv.org")
    assert "tidal locking" in frame
    assert "arxiv.org" in frame


def test_unregistered_mechanism_falls_back_without_crashing():
    """Callers must never crash on an unregistered (mechanism, modality)."""
    frame = hedge_frame("some_new_mechanism", "likely", subj="entropy", rel="disorder")
    assert isinstance(frame, str)
    assert "entropy" in frame


# --- (c) MODALITY IS MEASURED ------------------------------------------------
def test_modality_tracks_the_support_number():
    """The marker follows modality_from_support — a real measurement."""
    assert modality_from_support(0.90) == "likely"
    assert modality_from_support(0.40) == "possible"
    assert modality_from_support(0.05) == "unknown"


def test_stronger_support_hedges_less():
    """Higher measured support => weaker hedge, across the whole support range."""
    strong = hedge_frame("counterfactual", modality_from_support(0.90), subj="gravity")
    weak = hedge_frame("counterfactual", modality_from_support(0.05), subj="gravity")
    assert "probably" in strong
    assert "i'm not certain" in weak
    assert len(weak) > len(strong), "lower support produced the weaker hedge"


def test_trace_bands_hedge_in_strength_order():
    """A STRONG graph trace must hedge LESS than a weak one.

    The metacognitive-ignorance 3-state is ordered knowledge: a strong
    related trace is closer to verified than a weak one. Inverting that
    ordering makes RAVANA report LESS confidence the better its evidence
    gets, which is worse than the prose this module replaced. Pinned because
    the ordering is invisible in the rendered strings — every band still
    produces a fluent sentence.
    """
    strong = hedge_frame("ignorance", "related_strong", subj="entropy", rel="disorder")
    weak = hedge_frame("ignorance", "related_weak", subj="entropy", rel="disorder")
    none = hedge_frame("ignorance", "no_trace", subj="entropy", rel="disorder")
    # The weak band must hedge more than the strong band...
    assert "reaching" in weak
    assert "reaching" not in strong
    # ...and only no_trace may outright deny having a handle on the subject.
    assert "don't have a handle" in none
    assert "don't have a handle" not in strong
    assert "don't have a handle" not in weak


def test_no_trace_is_the_only_band_that_denies_a_handle():
    """Guards the band semantics: denial is reserved for the empty-trace band."""
    for band in ("related_strong", "related_weak"):
        frame = hedge_frame("ignorance", band, subj="entropy", rel="disorder")
        assert "don't have a handle" not in frame, (
            "%s wrongly denies holding a trace" % band
        )


# --- (d) THE LEXICON GROWS AT RUNTIME ----------------------------------------
def test_marker_lexicon_is_extensible_at_runtime():
    """THE deciding test: RAVANA can change this by itself, through experience.

    Without a runtime growth path this would be a frozen table wearing a
    seed's clothing, and the whole de-hardcoding would be cosmetic.
    """
    before = hedge_frame("ignorance", "related_weak", subj="entropy", rel="disorder")
    try:
        register_marker("related_weak", "i could be off, but")
        after = hedge_frame("ignorance", "related_weak", subj="entropy", rel="disorder")
        assert before != after, "register_marker did not change the frame"
        assert "i could be off, but" in after
    finally:
        register_marker("related_weak", "i might be reaching, but")
    restored = hedge_frame("ignorance", "related_weak", subj="entropy", rel="disorder")
    assert restored == before, "restore failed — the lexicon is not reversible"


def test_register_marker_ignores_junk_without_raising():
    """A bad runtime call degrades gracefully instead of breaking callers."""
    before = hedge_frame("ignorance", "related_weak", subj="entropy", rel="disorder")
    register_marker("", "x")
    register_marker("related_weak", None)
    register_marker(None, "x")
    assert hedge_frame("ignorance", "related_weak", subj="entropy", rel="disorder") == before
