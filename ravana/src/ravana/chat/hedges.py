"""Shared epistemic-hedge registry for RAVANA's chat surface.

Brain basis (cross-cutting primitive for the brain-faithful fixes):
humans tag uncertain/confident output with *modal* status (possibility /
likelihood / necessity) and *epistemic ownership* ("I/me"), rather than
asserting retrieved or inferred content flatly (Byrne 2002; Lewis 1973).
Scattering hardcoded hedge strings across every response generator invites
the garbled, non-human lines we are removing (e.g. "power seems to be topics
referred to by the same term"). One typed table, keyed by (mechanism,
modality, speech_act), is the single source of natural hedging.

No LLM, no network. Pure strings + light selection logic that draws on the
engine's existing ctx (valence, modality) so lead-ins/hedges vary by *intent*,
not at random.
"""

from typing import Dict, List, Optional, Tuple


# Modality ∈ {certain, likely, possible, unknown} — set by the generator
# algorithms (counterfactual robustness, comparative web plausibility,
# metacognitive-ignorance 3-state). Surfaced, never asserted.
MODALITY = ("certain", "likely", "possible", "unknown")


# Hedge frames keyed by (mechanism, modality). Each entry is a list of
# candidate templates; {subj}/{rel}/{note} are filled by the caller.
# Designed to read as a person thinking out loud, with explicit epistemic
# ownership ("I/me"), not as a knowledge base.
_HEDGE_FRAMES: Dict[Tuple[str, str], List[str]] = {
    # --- PROMPT 2: metacognitive ignorance (no definition, partial trace) ---
    ("ignorance", "related_strong"): [
        "that reminds me of {rel} — not sure i'd call it a definition, though",
        "it's tied to {rel} in my head, but i wouldn't swear to that",
        "the thread i keep pulling is {rel}; i don't have a clean definition yet",
    ],
    ("ignorance", "related_weak"): [
        "i might be reaching, but it could be related to {rel}",
        "not certain at all, but {rel} comes to mind",
        "i'm fuzzy on this — maybe it connects to {rel}?",
    ],
    ("ignorance", "no_trace"): [
        "honestly i don't have a handle on {subj} yet",
        "i'm still figuring out {subj} — no real grasp so far",
        "i don't really know {subj} yet, to be straight with you",
    ],
    # --- PROMPT 1: counterfactual modality markers ---
    ("counterfactual", "likely"): [
        "if {subj} were different, the chain would likely go:",
        "one way to think about it — {subj} changing would probably mean:",
        "here's the most likely ripple if {subj} were other than it is:",
    ],
    ("counterfactual", "possible"): [
        "if {subj} were different, it could lead to:",
        "hard to say exactly, but one plausible path if {subj} changed is:",
        "it's possible that {subj} being different would bring:",
    ],
    ("counterfactual", "unknown"): [
        "i'm not certain, but if {subj} were removed, a plausible path is:",
        "this is a guess, but {subj} changing might set off:",
        "not sure i can simulate that cleanly, yet if {subj} were different:",
    ],
    # --- PROMPT 3: web-answer source tagging (comparative modesty) ---
    ("web", "certain"): [
        "according to {src}, {snip}",
        "from what i found ({src}): {snip}",
    ],
    ("web", "likely"): [
        "some sources describe it as {snip} — i'd treat that as fairly solid",
        "the gist from {src} is {snip}",
    ],
    ("web", "possible"): [
        "i found a forum saying {snip}, but take that with a grain of salt",
        "a source describes it as {snip}, though i'm not fully certain",
    ],
    ("web", "unknown"): [
        "i couldn't verify that cleanly — {snip} is the best i found, but it's shaky",
    ],
}


def hedge_frame(mechanism: str, modality: str,
                subj: str = "", rel: str = "",
                snip: str = "", src: str = "") -> str:
    """Pick a natural hedge template for (mechanism, modality).

    Falls back to a generic honest frame if the key is missing, so callers
    never crash on an unregistered pair. Selection is deterministic given the
    inputs (no RNG) — varied by *intent* (mechanism+modality), satisfying the
    "not random" requirement.
    """
    choices = _HEDGE_FRAMES.get((mechanism, modality))
    if not choices:
        # Safe fallback: a neutral, owned hedge.
        return f"i'm not fully certain, but {subj or rel} comes to mind"
    # Deterministic pick: hash of the filled content keeps it stable per topic
    # while still varying across topics (avoids monotone repetition).
    idx = (len(subj) + len(rel) + len(snip)) % len(choices)
    tpl = choices[idx]
    try:
        return tpl.format(subj=subj, rel=rel, snip=snip, src=src, note="")
    except (KeyError, IndexError):
        return tpl


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
