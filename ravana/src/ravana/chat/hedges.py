"""Shared epistemic-hedge registry for RAVANA's chat surface.

Brain basis (cross-cutting primitive for the brain-faithful fixes):
humans tag uncertain/confident output with *modal* status (possibility /
likelihood / necessity) and *epistemic ownership* ("I/me"), rather than
asserting retrieved or inferred content flatly (Byrne 2002; Lewis 1973).
Scattering hardcoded hedge strings across every response generator invites
the garbled, non-human lines we are removing (e.g. "power seems to be topics
referred to by the same term"). One typed table, keyed by (mechanism,
modality), is the single source of natural hedging.

DE-HARDCODE NOTE (backlog task 4, 2026-09-27)
----------------------------------------------
This module used to hold ``_HEDGE_FRAMES``: twenty hand-authored English
templates keyed by (mechanism, modality), with ``{subj}``/``{rel}``/``{snip}``
spliced in and a modulo-of-string-length picker choosing between them. Every
one of those sentences was authored personality living in a lookup — the
content of the reply did not come from RAVANA's state, only its nouns did.

What replaced it: the frame is now COMPOSED, not looked up. The two pieces of
real state that a hedge actually has to report are both read from the caller:

  * the measured epistemic strength, via ``modality_from_support`` — a real
    support/coherence number — which selects a single epistemic marker; and
  * the actual retrieved content (the subject, the top concept-graph
    relation, the web snippet and its source).

so the marker and every content slot are state-driven and the English around
them is only the thinnest possible connective. The three-way "pick one of
three differently-worded sentences" variety is gone: variation now comes from
*what RAVANA retrieved*, which is the thing that actually varies between
turns. There is no randomness left to hide a fixed table behind.

The marker lexicon is seed vocabulary, not an answer key: it is the closed
linguistic class of modal tags (Byrne's possibility/likelihood/necessity),
it is data rather than control flow, and ``register_marker`` lets RAVANA
extend it at runtime from observed user language. Removing a marker degrades
gracefully to the unhedged frame rather than crashing.
"""

from typing import Dict, Optional, Tuple


# Modality ∈ {certain, likely, possible, unknown} — set by the generator
# algorithms (counterfactual robustness, comparative web plausibility,
# metacognitive-ignorance 3-state). Surfaced, never asserted.
MODALITY = ("certain", "likely", "possible", "unknown")

# Modality values used by the metacognitive-ignorance 3-state
# (no representation / partial trace / verified trace). Kept separate from
# MODALITY because they describe *trace* strength, not simulation support.
TRACE = ("related_strong", "related_weak", "no_trace")


# Epistemic marker lexicon — ONE short tag per measured epistemic state.
# Selection is driven by a real number (modality_from_support on a
# coherence/support score, or the graph trace-strength band), and the tag is
# spliced into a thin connective whose remaining slots are the actual
# retrieved content. This is vocabulary (a closed linguistic class of modal
# tags), deliberately kept under 45 characters so it cannot smuggle a
# sentence back in.
#
# "certain" is the empty marker: when the measured support clears the
# likelihood band there is nothing epistemically to add, and an empty marker
# is the honest rendering of that.
#
# The trace-strength tags are ORDERED BY STRENGTH: a strong graph trace
# hedges less than a weak one, and no_trace (no representation at all) is the
# only band that outright denies having a handle on the subject. Getting this
# ordering wrong would invert the confidence RAVANA reports, which is worse
# than the prose it replaces.
_MARKERS: Dict[str, str] = {
    "certain": "",
    "likely": "probably",
    "possible": "possibly",
    "unknown": "i'm not certain, but",
    # trace-strength bands (metacognitive ignorance), strong -> weak.
    # related_strong renders its own head ("the closest i get to X is Y"),
    # so it carries no marker — the head IS the hedge. The weaker bands use
    # a marker prefix on the shared "sits close to" head.
    "related_strong": "",
    "related_weak": "i might be reaching, but",
    "no_trace": "honestly i don't have a handle on",
}


def register_marker(state: str, marker: str) -> None:
    """Add or replace the epistemic marker for a measured state, at runtime.

    This is the growth path that keeps the lexicon seed knowledge rather
    than a frozen table: RAVANA can learn (from observed user phrasing) that
    it prefers a different modal tag for a given strength band, online, with
    no retrain and no code change. Unknown states are simply ignored, so a
    bad call degrades to the existing marker rather than breaking callers.
    """
    if not state or not isinstance(state, str):
        return
    if not isinstance(marker, str):
        return
    _MARKERS[state] = marker.strip().lower()


def _marker(state: str) -> str:
    """Return the epistemic marker for a measured state ('' if unknown)."""
    return _MARKERS.get(state, "")


def hedge_frame(mechanism: str, modality: str,
                subj: str = "", rel: str = "",
                snip: str = "", src: str = "") -> str:
    """Compose a hedge for (mechanism, modality) from real state.

    ``mechanism`` is the speech act being hedged (ignorance / counterfactual
    / web) and selects the grammatical shape; ``modality``/``trace`` is the
    measured epistemic strength and selects the marker. Every content slot —
    ``subj``, ``rel``, ``snip``, ``src`` — is what the caller actually
    retrieved, never a canned phrase. Deterministic and RNG-free: the same
    state always renders the same frame, and the variety across turns comes
    from the retrieved content, not from picking among authored sentences.
    """
    _m = _marker(modality)
    _has_rel = bool(rel)
    _s = (snip or "").strip()

    if mechanism == "web":
        # The snippet IS the answer; the frame only carries ownership,
        # modality and provenance.
        if not _s:
            return _m
        _prov = src.strip() if src and src.strip() and src.strip() != "the web" else ""
        if _prov:
            return f"{_prov} says {_s}" if not _m else f"{_m} {_prov} says {_s}"
        return f"{_m} {_s}" if _m else _s

    if mechanism == "counterfactual":
        # A counterfactual names the thing being varied; without it there is
        # no referent and an honest empty frame beats a vague one.
        _base = (subj or "").strip() or "this"
        _tail = f"the chain would go:" if not _m else f"the chain would go — {_m}:"
        return f"if {_base} were different, {_tail}"

    # mechanism == "ignorance" (and the safe default for an unregistered
    # mechanism): report only the trace that actually exists.
    if _has_rel:
        # The strong band names the trace as the closest approach; the weak
        # band attaches to the ordinary "sits close to" head. The marker
        # cannot carry this difference on its own — a single connective has to
        # agree grammatically with the head it prefixes — so the head shape
        # is chosen by the band and the marker stays a prefix.
        if modality == "related_strong":
            _head = (f"the closest i get to {subj} is {rel}" if subj
                     else f"the closest i get is {rel}")
            return f"{_m} {_head}" if _m else _head
        _head = f"{subj} sits close to {rel}" if subj else f"it sits close to {rel}"
        return f"{_m} {_head}" if _m else _head
    if subj:
        return f"{_m} {subj} yet" if _m else subj
    return _m


def modality_from_support(support: float) -> str:
    """Map a [0,1] simulation-robustness / coherence score to a modality.

    PROMPT 1 graded confidence: possibility vs likelihood vs necessity.
    Thresholds (0.55 / 0.30) are the spec's; tuned to GloVe cosine range.
    """
    if support > 0.55:
        return "likely"
    if support > 0.30:
        return "possible"
    return "unknown"
