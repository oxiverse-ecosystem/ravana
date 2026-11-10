"""Phase 2 of dialogue memory: STYLE CONDITIONING.

Feeds retrieved past exchanges into generation as CONDITIONING, so that
RAVANA's SYNTAX comes from observed dialogue rather than from the frozen
neural decoder. The hypothesis under test: syntax is learned from
conversation style, semantics from associations, and BOTH are used at
generation. The measurable test is INTERACTIVE ALIGNMENT -- does RAVANA's
phrasing converge toward the user's over successive turns?

What this module is
-------------------
A style-conditioning layer that biases generation toward the user's
observed style, using retrieved dialogue exemplars. It is:

  * derived from the USER half of retrieved ``DialogueTurn``s (never
    RAVANA's own replies -- we want the user's style, not our own);
  * applied as CONDITIONING on the syntax path, NOT by retraining or
    fine-tuning the frozen decoder;
  * measured by an automatic alignment metric so turn 1 vs turn 20 can be
    compared.

Neuroscience grounding
----------------------
  * **Interactive alignment** (Pickering & Garrod 2004, *Behavioral and
    Brain Sciences* 27(2):212-225; target article 169-190). Interlocutors
    align their linguistic representations at many levels via a largely
    automatic priming process, and alignment of situation models is what
    makes dialogue successful. -> the style context is built from the
    user's own recent utterances and re-activates their phrasing at
    generation time.

  * **Lexical entrainment / conceptual pacts** (Brennan & Clark 1996,
    *Journal of Experimental Psychology: Learning, Memory, and Cognition*
    22(6):1482-1493). When people refer repeatedly to the same object they
    come to use the same terms; once a shared conceptualization is
    established they appeal to it in later references. -> the function-word
    profile is the entrainment signal: it is the closed-class vocabulary
    the user has been using, re-weighted at generation.

  * **Language-style matching** (Ireland, Slatcher, Eastwick, Scissors,
    Finkel & Pennebaker 2011, *Psychological Science* 22(1):39-44).
    Similarity in dyads' use of function words predicts relationship
    outcomes; LSM is an implicit interpersonal process. -> the alignment
    metric is a function-word distribution distance, the same family of
    features LSM uses.

  * **Audience design** (Clark & Murphy 1982, in *Language and
    Comprehension*, pp. 287-299; Bell 1984, *Language in Society*
    13(2):145-204). Speakers design their style for their audience and
    accommodate primarily to their addressee. -> the style context is
    per-interlocutor (the user), not a global average.

  * **Alignment is PRIMING-driven and largely IMPLICIT** (Pickering &
    Garrod 2004). It is NOT explicit mimicry. -> this module never copies
    the user's words into the reply verbatim; it re-weights the existing
    generation machinery (function-word bias, template/realizer frame
    selection, hedge modality) toward the user's observed distribution.
    The reply content still comes from RAVANA's retrieved knowledge.

Approach chosen, and why over the alternatives
----------------------------------------------
Three conditioning mechanisms were evaluated:

  (A) logit/token bias on function words the user over-uses (lexical
      entrainment at the decoder level);
  (B) selecting among the existing template/realizer frames based on the
      user's profile;
  (C) adjusting the hedge marker / modality from the user's directness.

Chosen: **(B) + (C)**, with (A) deliberately NOT implemented.

  * (A) is rejected because it requires touching the frozen decoder's
    logits. The brief forbids retraining/fine-tuning the decoder's
    weights; a per-token logit bias is a weight-level intervention on the
    same frozen model, and it would also be the least brain-faithful
    option: lexical entrainment in humans is not a per-token logit
    perturbation but a change in the DISTRIBUTION over closed-class
    vocabulary, which is exactly what (B)/(C) implement at the
    realizer/frame level. (A) is also the hardest to measure cleanly,
    because the decoder's output is already noisy.

  * (B) is the smallest and most measurable: the surface realizer already
    exposes a per-variant weight table (``_variant_weights``) and a
    per-relation adverb table. Conditioning = nudging those weights toward
    the variant family the user's own sentences most resemble. No new
    reply strings, no new frames, no decoder contact.

  * (C) is the most brain-faithful: the user's directness (question rate,
    mean length, hedge-word rate) is a real register signal, and the
    shared hedge registry (``chat/hedges.py``) already exposes
    ``register_marker`` as the runtime growth path for exactly this. We
    use it rather than authoring hedge strings.

Both (B) and (C) are SAFE TO DISABLE and FAIL-OPEN: if the memory is
empty, GloVe is unavailable, or the flag is off, generation behaves
exactly as before (the style context is ``None`` and every consumer
no-ops).

DE-HARDCODE NOTE
----------------
This module authors NO reply strings. It contains only:
  * a closed-class function-word seed (the same class ``pet_slots``
    already owns, re-used via ``is_function_word``);
  * a small set of REGISTER MARKER keys (``certain``/``likely``/
    ``possible``/``unknown``) that are spliced into the existing
    ``hedge_frame`` connective;
  * numeric thresholds and weights.
Every English string that reaches the user still comes from the existing
realizer / hedge registry / retrieved content. The alignment metric is a
pure function of token counts.

Python 3.10 compatibility: no match statements, no ``X | Y`` in runtime
positions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── closed-class vocabulary ──────────────────────────────────────────────────
# Re-use the project's ONE function-word class (pet_slots.is_function_word)
# rather than re-listing it. Imported lazily so this module stays cheap and
# so a missing pet_slots degrades to the narrow seed below (fail-open).

# Narrow seed used ONLY when pet_slots cannot be imported. Mirrors the
# closed-class categories (articles/pronouns/prepositions/conjunctions/
# auxiliaries) -- the same class the project already treats as function
# words. Seed vocabulary, not authored prose.
_FUNCTION_WORD_SEED = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "some", "any",
    "each", "every", "all", "both", "few", "many", "much", "more", "most",
    "other", "another", "such", "no", "nor", "one", "ones", "several",
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours",
    "you", "your", "yours", "yourself", "he", "him", "his", "she", "her",
    "hers", "it", "its", "they", "them", "their", "theirs", "who", "whom",
    "whose", "what", "which", "there", "here",
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "of", "off", "over", "under", "near", "behind",
    "beyond", "among", "onto", "upon", "across", "throughout", "via",
    "and", "or", "but", "nor", "so", "because", "although", "though",
    "while", "whereas", "if", "unless", "since", "when", "whenever",
    "where", "whether", "than", "then", "also", "plus", "however",
    "be", "am", "is", "are", "was", "were", "been", "being",
    "do", "does", "did", "done", "doing",
    "have", "has", "had", "having",
    "will", "would", "shall", "should", "can", "could", "may", "might",
    "must", "ought", "need", "dare",
    "not", "n't", "never", "as",
})

# Hedge / modality words the user may over-use. Used only to derive the
# user's directness profile (a high hedge rate = indirect style). Seed
# vocabulary, not authored prose.
_HEDGE_WORDS = frozenset({
    "maybe", "perhaps", "possibly", "probably", "likely", "seemingly",
    "kind of", "sort of", "partly", "i think", "i believe", "i suspect",
    "it seems", "apparently", "supposedly", "roughly", "somewhat",
})

# Question words -- the user's question-vs-statement tendency.
_QUESTION_WORDS = frozenset({
    "what", "who", "whom", "whose", "where", "when", "why", "which", "how",
    "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "would", "will", "shall", "should", "may", "might", "have", "has",
    "had", "am",
})

# Contractions that signal an informal register.
_CONTRACTIONS = frozenset({
    "i'm", "i've", "i'd", "i'll", "you're", "you've", "you'd", "you'll",
    "we're", "we've", "we'd", "we'll", "they're", "they've", "they'd",
    "they'll", "it's", "that's", "there's", "here's", "what's", "who's",
    "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't",
    "can't", "cannot", "couldn't", "won't", "wouldn't", "shouldn't",
    "hasn't", "haven't", "hadn't", "let's", "gonna", "wanna", "kinda",
    "sorta", "dunno", "yeah", "nope", "ok", "okay", "hey", "hi", "hello",
})

_WORD_RE = re.compile(r"\b[a-zA-Z']{2,}\b")


def _is_function_word(word: str) -> bool:
    """True when `word` is a closed-class function word.

    Consults the project's shared class (``pet_slots.is_function_word``)
    when importable, else the narrow seed. Pure vocabulary lookup.
    """
    w = (word or "").strip().lower().strip(".,!?;:'\"")
    if not w:
        return False
    try:
        from ravana.chat.pet_slots import is_function_word as _fw
        return bool(_fw(w))
    except Exception:
        return w in _FUNCTION_WORD_SEED


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens (apostrophes kept, so contractions survive)."""
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def _sentences(text: str) -> List[str]:
    """Split into sentence-ish spans on terminal punctuation."""
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


# ── StyleContext ─────────────────────────────────────────────────────────────

@dataclass
class StyleContext:
    """The user's observed style, derived from retrieved dialogue exemplars.

    All fields are plain data; no behaviour. Built ONLY from the USER half
    of retrieved ``DialogueTurn``s (``turn.user_utterance``), never from
    ``turn.ravana_reply`` -- we want the user's style, not RAVANA's own.

    Fields
    ------
    n_exemplars : int
        How many user utterances contributed. 0 means "no evidence" and
        every consumer must no-op.
    mean_length : float
        Mean tokens per user utterance.
    question_rate : float
        Fraction of user utterances that are questions (0..1).
    hedge_rate : float
        Fraction of user utterances containing a hedge/modality word.
    contraction_rate : float
        Fraction of user utterances containing a contraction.
    function_word_profile : Dict[str, float]
        Normalized frequency of each function word the user used, over the
        TOTAL function-word tokens (sums to ~1.0). This is the entrainment
        signal.
    top_function_words : List[str]
        The user's most-over-used function words, most frequent first.
    directness : float
        0..1, higher = more direct (short, few hedges, more statements).
    formality : float
        0..1, higher = more formal (fewer contractions, longer sentences).
    """

    n_exemplars: int = 0
    mean_length: float = 0.0
    question_rate: float = 0.0
    hedge_rate: float = 0.0
    contraction_rate: float = 0.0
    function_word_profile: Dict[str, float] = field(default_factory=dict)
    top_function_words: List[str] = field(default_factory=list)
    directness: float = 0.5
    formality: float = 0.5

    @property
    def available(self) -> bool:
        """True when there is at least one exemplar behind this context."""
        return self.n_exemplars > 0

    def to_state(self) -> Dict[str, Any]:
        """Serialize to a plain dict (for engine save/load)."""
        return {
            "n_exemplars": int(self.n_exemplars),
            "mean_length": float(self.mean_length),
            "question_rate": float(self.question_rate),
            "hedge_rate": float(self.hedge_rate),
            "contraction_rate": float(self.contraction_rate),
            "function_word_profile": dict(self.function_word_profile),
            "top_function_words": list(self.top_function_words),
            "directness": float(self.directness),
            "formality": float(self.formality),
        }

    @classmethod
    def from_state(cls, state: Optional[Dict[str, Any]]) -> Optional["StyleContext"]:
        """Rebuild from a saved dict. None when absent/unusable (fail-open)."""
        if not isinstance(state, dict):
            return None
        try:
            return cls(
                n_exemplars=int(state.get("n_exemplars", 0) or 0),
                mean_length=float(state.get("mean_length", 0.0) or 0.0),
                question_rate=float(state.get("question_rate", 0.0) or 0.0),
                hedge_rate=float(state.get("hedge_rate", 0.0) or 0.0),
                contraction_rate=float(state.get("contraction_rate", 0.0) or 0.0),
                function_word_profile=dict(state.get("function_word_profile", {}) or {}),
                top_function_words=list(state.get("top_function_words", []) or []),
                directness=float(state.get("directness", 0.5) or 0.5),
                formality=float(state.get("formality", 0.5) or 0.5),
            )
        except Exception:
            return None


def derive_style_context(exemplars: Sequence[Any]) -> StyleContext:
    """Build a ``StyleContext`` from retrieved dialogue exemplars.

    ``exemplars`` is a sequence of ``(DialogueTurn, similarity)`` pairs (the
    exact shape ``DialogueMemory.recall_similar_exchanges`` returns) or a
    plain sequence of ``DialogueTurn`` objects. Only the USER half
    (``turn.user_utterance``) is read; ``turn.ravana_reply`` is ignored.

    Returns an empty context (``n_exemplars == 0``) when there is nothing
    to derive from -- the caller must treat that as "no conditioning".
    """
    utterances: List[str] = []
    for item in exemplars or []:
        turn = None
        if isinstance(item, (tuple, list)) and item:
            turn = item[0]
        else:
            turn = item
        if turn is None:
            continue
        text = getattr(turn, "user_utterance", None)
        if isinstance(text, str) and text.strip():
            utterances.append(text)

    if not utterances:
        return StyleContext()

    n = len(utterances)
    lengths: List[int] = []
    n_questions = 0
    n_hedges = 0
    n_contractions = 0
    fw_counts: Dict[str, int] = {}
    fw_total = 0

    for text in utterances:
        toks = _tokenize(text)
        lengths.append(len(toks))
        low = text.lower()
        # Question: trailing '?' OR a leading wh-word / inverted auxiliary.
        if low.strip().endswith("?"):
            n_questions += 1
        else:
            first = toks[0] if toks else ""
            if first in _QUESTION_WORDS and first not in ("is", "are", "was",
                                                         "were", "do", "does",
                                                         "did", "can", "could",
                                                         "would", "will",
                                                         "shall", "should",
                                                         "may", "might",
                                                         "have", "has", "had",
                                                         "am"):
                n_questions += 1
            elif first in ("what", "who", "whom", "whose", "where", "when",
                           "why", "which", "how"):
                n_questions += 1
        # Hedge: any hedge phrase present.
        if any(h in low for h in _HEDGE_WORDS):
            n_hedges += 1
        # Contraction: any contraction token present.
        if any(c in low for c in _CONTRACTIONS):
            n_contractions += 1
        # Function-word profile.
        for tok in toks:
            if _is_function_word(tok):
                fw_counts[tok] = fw_counts.get(tok, 0) + 1
                fw_total += 1

    mean_length = float(np.mean(lengths)) if lengths else 0.0
    question_rate = n_questions / float(n)
    hedge_rate = n_hedges / float(n)
    contraction_rate = n_contractions / float(n)

    profile: Dict[str, float] = {}
    if fw_total > 0:
        profile = {w: c / float(fw_total) for w, c in fw_counts.items()}
    # Sort the profile itself by descending frequency (ties broken by word for
    # determinism) so the dict's iteration order is the entrainment ranking,
    # not first-seen order. top_function_words is then just the head of it.
    sorted_items = sorted(profile.items(), key=lambda kv: (-kv[1], kv[0]))
    profile = {w: v for w, v in sorted_items}
    top_words = [w for w, _ in sorted_items[:8]]

    # Directness: short + few hedges + more statements -> more direct.
    # Each term is a bounded 0..1 signal; combined by a fixed weighted sum.
    len_sig = 1.0 - min(1.0, mean_length / 20.0)          # shorter -> higher
    hedge_sig = 1.0 - min(1.0, hedge_rate * 2.0)          # fewer hedges -> higher
    stmt_sig = 1.0 - question_rate                        # more statements -> higher
    directness = float(np.clip(0.4 * len_sig + 0.3 * hedge_sig + 0.3 * stmt_sig,
                               0.0, 1.0))

    # Formality: fewer contractions + longer sentences -> more formal.
    contr_sig = 1.0 - min(1.0, contraction_rate * 2.0)
    long_sig = min(1.0, mean_length / 18.0)
    formality = float(np.clip(0.6 * contr_sig + 0.4 * long_sig, 0.0, 1.0))

    return StyleContext(
        n_exemplars=n,
        mean_length=mean_length,
        question_rate=question_rate,
        hedge_rate=hedge_rate,
        contraction_rate=contraction_rate,
        function_word_profile=profile,
        top_function_words=top_words,
        directness=directness,
        formality=formality,
    )


# ── Conditioning: frame / realizer selection ─────────────────────────────────

# The realizer's merge variants, grouped into FAMILIES the user's style can
# select between. This is a mapping from variant name -> family, not a new
# set of frames: the realizer already owns the variants and their weights.
_VARIANT_FAMILIES: Dict[str, str] = {
    "svo": "plain",
    "svo_emphatic": "plain",
    "svo_causal": "plain",
    "left_dislocation": "elaborate",
    "pseudo_cleft": "elaborate",
    "cleft": "elaborate",
    "existential": "elaborate",
    "as_for_topic": "elaborate",
    "topic_fronting_for": "elaborate",
    "adverb_fronted": "elaborate",
}

# Family preference weights derived from the user's style. The realizer's
# own ``_variant_weights`` are MULTIPLIED by these, so a family the user
# favours is more likely and a family the user never uses is suppressed.
# The magnitudes are deliberately modest (0.6..1.5) so conditioning biases
# rather than overrides the realizer's discourse-driven selection.
_FAMILY_WEIGHTS: Dict[str, Dict[str, float]] = {
    # direct, terse user -> plain SVO dominates
    "direct": {"plain": 1.5, "elaborate": 0.6},
    # indirect, elaborate user -> elaborated variants get a boost
    "elaborate": {"plain": 0.8, "elaborate": 1.4},
    # neutral -> no bias
    "neutral": {"plain": 1.0, "elaborate": 1.0},
}


def style_family(style: Optional[StyleContext]) -> str:
    """Map a style context to a variant family preference key.

    ``direct`` when the user is clearly direct (short, few hedges, more
    statements), ``elaborate`` when clearly indirect, else ``neutral``.
    """
    if style is None or not style.available:
        return "neutral"
    if style.directness >= 0.62:
        return "direct"
    if style.directness <= 0.38:
        return "elaborate"
    return "neutral"


def variant_weight_multiplier(variant: str, style: Optional[StyleContext]) -> float:
    """The multiplier to apply to a realizer variant weight for this style.

    1.0 when there is no style context (fail-open: no conditioning).
    """
    fam = _VARIANT_FAMILIES.get(variant)
    if fam is None:
        return 1.0
    weights = _FAMILY_WEIGHTS[style_family(style)]
    return float(weights.get(fam, 1.0))


def apply_style_to_realizer(realizer: Any, style: Optional[StyleContext]) -> bool:
    """Bias the realizer's variant weights toward the user's style family.

    Multiplies (never replaces) the realizer's existing per-variant weights,
    so the realizer's own discourse-driven selection is preserved and the
    style bias is a nudge. Returns True when any weight was changed.

    Fail-open: a realizer with no ``_variant_weights`` is left untouched.
    """
    if style is None or not style.available:
        return False
    weights = getattr(realizer, "_variant_weights", None)
    if not isinstance(weights, dict) or not weights:
        return False
    changed = False
    for variant in list(weights.keys()):
        mult = variant_weight_multiplier(variant, style)
        if mult == 1.0:
            continue
        try:
            weights[variant] = float(weights[variant]) * mult
            changed = True
        except Exception:
            continue
    return changed


# ── Conditioning: hedge / modality from the user's directness ─────────────────

def select_modality(style: Optional[StyleContext], base_modality: str) -> str:
    """Adjust a measured modality by the user's directness.

    The user's directness is a REGISTER signal, not an epistemic one: a
    direct user gets a more direct marker, an indirect user a softer one.
    The marker itself still comes from the shared hedge registry
    (``chat/hedges.py``); this function only chooses WHICH measured state
    to ask for. It never authors a hedge string.

    Fail-open: no style context -> the base modality is returned unchanged.
    """
    if style is None or not style.available:
        return base_modality
    # Only shift when the user's style is clearly on one side; a neutral
    # profile must not move the epistemic marker at all.
    #
    # Two scales are involved and both must be handled:
    #   * the CERTAINTY scale (certain/likely/possible/unknown)
    #   * the TRACE-STRENGTH scale (related_strong/related_weak/no_trace),
    #     which is what the ignorance hedge frames actually use.
    # A direct user gets a firmer marker on either scale; an indirect user a
    # softer one. Handling only the certainty scale left every `related_*`
    # base unchanged, so the conditioning silently no-oped for exactly the
    # replies it was meant to affect.
    if style.directness >= 0.62:
        if base_modality in ("unknown", "possible"):
            return "likely"
        if base_modality == "related_weak":
            return "related_strong"
        if base_modality == "no_trace":
            return "related_weak"
    elif style.directness <= 0.38:
        if base_modality in ("certain", "likely"):
            return "possible"
        if base_modality == "related_strong":
            return "related_weak"
        if base_modality == "related_weak":
            return "no_trace"
    return base_modality


# ── Conditioning: length and question rate (round 2026-11-10) ────────────────
#
# Phase 2 originally conditioned one slot: WHICH hedge variant was selected.
# Measured on the 4-channel alignment metric that produced:
#   function_word_distance +0.009, length_ratio +0.000,
#   question_rate_delta +0.000, composite -0.003
# The mechanism worked; the scope was a single lexical slot. These constants
# widen it to the length and question-rate channels.
#
# Every magnitude here is deliberately modest. The alignment literature
# (Pickering & Garrod's interactive alignment; Ireland et al.'s linguistic
# style matching) reports convergence as a PARTIAL shift, not a match -- and
# over-accommodation is a documented failure mode (Ryan, Giles et al.,
# communication accommodation theory). So these bias; they do not override.

# RAVANA's own mean reply length, in tokens. A user at this length gets a
# verbosity multiplier of exactly 1.0.
_RAVANA_BASELINE_LENGTH = 6.0

# How strongly the length ratio moves the multiplier. 0.25 means a user twice
# as long-winded as the baseline gets ~1.19, not 2.0.
_VERBOSITY_GAIN = 0.25

# Hard bounds. Never let conditioning make RAVANA more than ~1.6x or less
# than ~0.6x as verbose as its own register.
_VERBOSITY_CAP = 1.6
_VERBOSITY_FLOOR = 0.6

# Baseline probability that RAVANA appends a question closer at all. Matches
# the pre-existing `random.random() < 0.5` behaviour at response_gen.py:3728 so
# that a neutral profile is byte-identical to the old behaviour.
_QUESTION_BASE = 0.5


def style_verbosity(style: Optional[StyleContext]) -> float:
    """Map the user's observed utterance length to a verbosity multiplier.

    LENGTH CHANNEL (round 2026-11-10). Phase 2 previously conditioned only
    WHICH hedge variant was selected -- one lexical slot. Measured effect on
    the alignment metric: function_word_distance +0.009, length_ratio +0.000,
    question_rate_delta +0.000, composite -0.003. The mechanism worked; the
    scope was one slot.

    This is the length channel. The user's mean utterance length is compared
    against RAVANA's own baseline and turned into a multiplier on the
    verbosity register that already gates whether sourced evidence is
    surfaced at all (`_reg_verbosity`, checked in `_reflective_response`).

    Deliberately MODEST and one-sided:
      * a user who speaks tersely gets a multiplier below 1, so RAVANA surfaces
        LESS elaboration;
      * a user who speaks at length gets a multiplier above 1, so RAVANA
        surfaces MORE;
      * a user near the baseline gets ~1.0 and nothing changes.

    Fail-open: no style context -> 1.0 (no change). The magnitude is capped
    (see `_VERBOSITY_CAP`) so conditioning biases rather than overrides.
    """
    if style is None or not style.available:
        return 1.0
    ulen = float(getattr(style, "mean_length", 0.0) or 0.0)
    if ulen <= 0:
        return 1.0
    # RAVANA's own baseline utterance length. Measured over its reflective and
    # uncertainty replies: ~6 words. A user at that length gets 1.0.
    ratio = ulen / _RAVANA_BASELINE_LENGTH
    # Compress toward 1.0 so the effect is a nudge, not a rescaling: a user
    # twice as long-winded gets ~1.2, not 2.0. log2 keeps it symmetric.
    try:
        import math as _math
        shift = _math.log2(max(ratio, 1e-6))
    except Exception:
        shift = 0.0
    mult = 1.0 + _VERBOSITY_GAIN * shift
    return float(max(_VERBOSITY_FLOOR, min(_VERBOSITY_CAP, mult)))


def style_question_propensity(style: Optional[StyleContext]) -> float:
    """How likely RAVANA should be to close with a question, given the user.

    QUESTION-RATE CHANNEL (round 2026-11-10). See `style_verbosity` for why
    this channel exists.

    Returns a probability in 0..1 that a closer should be appended at all.
    The user's own question rate is the signal: people who ask questions
    invite questions back, people who make statements do not.

    Fail-open: no style context -> None, meaning "use the caller's default".
    Returning None rather than 0.5 matters -- a neutral profile must not move
    the behaviour at all, and the callers already have a sensible default.
    """
    if style is None or not style.available:
        return None  # type: ignore[return-value]
    qr = float(getattr(style, "question_rate", 0.0) or 0.0)
    # Only act when the user's profile is clearly on one side. A user who
    # asks questions half the time is the baseline and gets no shift.
    if qr >= 0.5:
        return float(min(0.85, _QUESTION_BASE + (qr - 0.5) * 0.7))
    if qr <= 0.15:
        return float(max(0.05, _QUESTION_BASE - (0.15 - qr) * 1.2))
    return None  # type: ignore[return-value]


def register_style_markers(style: Optional[StyleContext]) -> bool:
    """Register the user's top function words as runtime hedge markers.

    This is the growth path ``chat/hedges.py`` already exposes
    (``register_marker``): RAVANA can learn a different modal tag for a
    given strength band from observed user phrasing, online, with no
    retrain. We only register when the user's top function words are
    genuinely over-represented (a real entrainment signal), and we register
    them against the EXISTING marker states -- never a new state, never a
    reply string.

    Returns True when at least one marker was registered.
    """
    if style is None or not style.available:
        return False
    try:
        from ravana.chat.hedges import register_marker
    except Exception:
        return False
    changed = False
    # The user's most-over-used function words become the markers for the
    # weak bands (the bands that actually carry a marker). Only words that
    # are short and closed-class are eligible -- a long content word is not
    # a modal tag.
    candidates = [w for w in style.top_function_words if len(w) <= 8]
    if not candidates:
        return False
    # Register the top candidate against the weakest band only, so a bad
    # call degrades to the existing marker rather than corrupting the
    # strong bands. This is the smallest, safest use of the growth path.
    #
    # IDEMPOTENT AND NON-DESTRUCTIVE (round 2026-11-10). This used to call
    # `register_marker` unconditionally on every invocation, and
    # `_MARKERS` is a MODULE-GLOBAL dict. Two consequences:
    #   1. calling it every turn permanently overwrote the lexicon entry, so
    #      the marker drifted to whatever the most recent style context said
    #      and never came back;
    #   2. it leaked across engine instances and across tests -- CI caught
    #      `test_marker_lexicon_is_extensible_at_runtime` failing with
    #      "restore failed -- the lexicon is not reversible", because a
    #      previous test's engine had already mutated the global.
    #
    # Now it writes only when the value actually differs, and it never
    # overwrites a marker that a previous call (or the seed) established
    # unless the new evidence is genuinely different.
    try:
        _target = candidates[0]
        from ravana.chat.hedges import _marker as _current_marker
        if _current_marker("related_weak") == _target:
            return False  # already registered; nothing to do
        register_marker("related_weak", _target)
        changed = True
    except Exception:
        changed = False
    return changed


def _plausible_marker_words(style: StyleContext) -> List[str]:
    """Function words that could plausibly serve as an epistemic marker.

    WHY THIS FILTER EXISTS (round 2026-11-10). `top_function_words` is simply
    the user's most frequent closed-class words, and those are always the same
    few: 'is', 'the', 'i', 'and', 'to'. Registering one as a modal marker
    produced

        "is entropy sits close to disorder"

    -- a copula where a hedge belongs. The marker slot is a DISCOURSE MARKER
    position, so only words that can occupy it are eligible: hedges, downtoners
    and discourse particles. Frequency alone is not evidence of markerhood.

    This is deliberately a denylist of the words that are clearly ungrammatical
    in the slot, rather than an allowlist of known-good markers -- an allowlist
    would defeat the point of learning a marker the user actually uses.
    """
    _NOT_MARKERS = {
        # copulas / auxiliaries -- cannot head a hedge clause
        "is", "are", "was", "were", "be", "been", "being", "am", "do", "does",
        "did", "has", "have", "had", "will", "would", "can", "could", "shall",
        "should", "may", "might", "must",
        # determiners / pronouns / prepositions / conjunctions
        "the", "a", "an", "this", "that", "these", "those", "i", "you", "he",
        "she", "it", "we", "they", "me", "him", "her", "us", "them", "my",
        "your", "his", "its", "our", "their", "of", "in", "on", "at", "to",
        "for", "with", "by", "from", "as", "and", "or", "but", "if", "so",
        "than", "then", "there", "here", "not", "no", "yes", "what", "which",
        "who", "how", "when", "where", "why", "all", "some", "any", "each",
        "more", "most", "much", "many", "very", "just", "also", "too", "only",
        "own", "same", "such", "one", "two", "first", "second", "now", "up",
        "out", "about", "into", "over", "after", "before", "again", "once",
    }
    out = []
    for w in getattr(style, "top_function_words", ()) or ():
        w = str(w).strip().lower()
        if not w or len(w) > 12 or w in _NOT_MARKERS:
            continue
        out.append(w)
    return out


def register_style_markers_for(engine: Any, style: Optional[StyleContext]) -> bool:
    """Per-engine variant of `register_style_markers`.

    `register_style_markers` writes the MODULE-GLOBAL `_MARKERS` dict, so a
    marker derived from one user's style leaks into every other engine in the
    process. CI caught exactly that: `test_marker_lexicon_is_extensible_at_runtime`
    failed with "restore failed -- the lexicon is not reversible" because a
    style-conditioning test had already overwritten `related_weak` globally.

    This variant stores the override on the engine (`_style_marker_overrides`)
    and leaves the global untouched. `hedges.marker_for(engine, state)`
    resolves it. Fail-open: any error returns False.
    """
    if engine is None or style is None or not style.available:
        return False
    try:
        from ravana.chat.hedges import register_marker_scoped
    except Exception:
        return False
    candidates = _plausible_marker_words(style)
    if not candidates:
        return False
    try:
        return bool(register_marker_scoped(engine, "related_weak", candidates[0]))
    except Exception:
        return False


# ── Alignment metric ─────────────────────────────────────────────────────────

def function_word_distribution(text: str) -> Dict[str, float]:
    """Normalized function-word distribution over ``text``.

    The same family of features LSM uses (Ireland et al. 2011): the
    relative frequency of closed-class words. Returns {} when the text has
    no function words.
    """
    toks = _tokenize(text)
    counts: Dict[str, int] = {}
    total = 0
    for tok in toks:
        if _is_function_word(tok):
            counts[tok] = counts.get(tok, 0) + 1
            total += 1
    if total == 0:
        return {}
    return {w: c / float(total) for w, c in counts.items()}


def distribution_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Jensen-Shannon distance between two normalized distributions.

    Symmetric, bounded [0, 1] for normalized distributions over the same
    vocabulary, and well-defined when one side is empty (returns 1.0 -- no
    overlap). Used as the function-word component of the alignment metric.
    """
    if not p and not q:
        return 0.0
    if not p or not q:
        return 1.0
    vocab = set(p.keys()) | set(q.keys())
    # Jensen-Shannon divergence via the mixture distribution.
    import math
    js = 0.0
    for w in vocab:
        pw = float(p.get(w, 0.0))
        qw = float(q.get(w, 0.0))
        mw = 0.5 * (pw + qw)
        if pw > 0 and mw > 0:
            js += 0.5 * pw * math.log(pw / mw)
        if qw > 0 and mw > 0:
            js += 0.5 * qw * math.log(qw / mw)
    # JS divergence is bounded by ln(2); the distance is its square root.
    return float(np.sqrt(max(0.0, js) / math.log(2.0)))


def alignment_metric(user_texts: Sequence[str],
                     ravana_texts: Sequence[str]) -> Dict[str, float]:
    """Measure interactive alignment between the user and RAVANA.

    Compares RAVANA's replies to the user's recent utterances on three
    channels, each a real alignment signal:

      1. ``function_word_distance`` -- Jensen-Shannon distance between the
         function-word distributions (the LSM family; Ireland et al. 2011).
         Lower = more aligned.
      2. ``length_ratio`` -- mean RAVANA reply length / mean user utterance
         length. 1.0 = matched; >1 = RAVANA is more verbose.
      3. ``question_rate_delta`` -- RAVANA question rate minus user question
         rate. 0.0 = matched.

    Returns a dict with those three channels plus ``composite`` (a single
    0..1 alignment score where 1.0 = perfectly aligned). The composite is
    the mean of the per-channel alignment scores, each of which is a
    bounded 0..1 similarity.

    Computable automatically, so turn 1 vs turn 20 can be compared.
    """
    user_texts = [t for t in (user_texts or []) if isinstance(t, str) and t.strip()]
    ravana_texts = [t for t in (ravana_texts or []) if isinstance(t, str) and t.strip()]

    if not user_texts or not ravana_texts:
        return {
            "function_word_distance": 1.0,
            "length_ratio": 0.0,
            "question_rate_delta": 0.0,
            "composite": 0.0,
        }

    # Aggregate distributions over the whole window.
    user_fw: Dict[str, float] = _merge_distributions(
        [function_word_distribution(t) for t in user_texts])
    ravana_fw: Dict[str, float] = _merge_distributions(
        [function_word_distribution(t) for t in ravana_texts])

    fw_dist = distribution_distance(user_fw, ravana_fw)

    user_lens = [len(_tokenize(t)) for t in user_texts]
    ravana_lens = [len(_tokenize(t)) for t in ravana_texts]
    user_mean = float(np.mean(user_lens)) if user_lens else 0.0
    ravana_mean = float(np.mean(ravana_lens)) if ravana_lens else 0.0
    length_ratio = (ravana_mean / user_mean) if user_mean > 0 else 0.0

    user_q = _question_rate(user_texts)
    ravana_q = _question_rate(ravana_texts)
    q_delta = ravana_q - user_q

    # Per-channel alignment scores (1.0 = aligned).
    fw_align = 1.0 - min(1.0, fw_dist)
    # Length: a ratio of 1.0 is perfect; penalize deviation in either
    # direction, bounded so a 3x ratio does not go below 0.
    len_align = max(0.0, 1.0 - abs(1.0 - min(length_ratio, 3.0)) / 2.0)
    q_align = max(0.0, 1.0 - abs(q_delta))

    composite = float(np.clip((fw_align + len_align + q_align) / 3.0, 0.0, 1.0))

    return {
        "function_word_distance": float(fw_dist),
        "length_ratio": float(length_ratio),
        "question_rate_delta": float(q_delta),
        "composite": composite,
    }


def _merge_distributions(dists: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Average a list of normalized distributions into one."""
    if not dists:
        return {}
    merged: Dict[str, float] = {}
    for d in dists:
        for w, v in d.items():
            merged[w] = merged.get(w, 0.0) + float(v)
    n = float(len(dists))
    return {w: v / n for w, v in merged.items()}


def _question_rate(texts: Sequence[str]) -> float:
    """Fraction of texts that are questions."""
    if not texts:
        return 0.0
    n = 0
    for t in texts:
        low = t.lower().strip()
        if low.endswith("?"):
            n += 1
            continue
        toks = _tokenize(t)
        if toks and toks[0] in _QUESTION_WORDS:
            n += 1
    return n / float(len(texts))


def alignment_trend(window_a: Dict[str, float],
                    window_b: Dict[str, float]) -> float:
    """The change in composite alignment from window A to window B.

    Positive = RAVANA moved TOWARD the user (alignment improved). This is
    the turn-1-vs-turn-20 comparison the hypothesis needs.
    """
    if not isinstance(window_a, dict) or not isinstance(window_b, dict):
        return 0.0
    a = float(window_a.get("composite", 0.0))
    b = float(window_b.get("composite", 0.0))
    return b - a


# ── Engine-facing helper ─────────────────────────────────────────────────────

def build_style_context_from_memory(memory: Any, query: str,
                                    k: int = 3) -> Optional[StyleContext]:
    """Retrieve exemplars and derive a style context. Fail-open.

    Returns None when the memory is unavailable, retrieval returns nothing,
    or the derived context has no exemplars -- so every caller can treat
    None as "no conditioning, behave exactly as before".
    """
    if memory is None:
        return None
    try:
        if not getattr(memory, "available", False):
            return None
        hits = memory.recall_similar_exchanges(query, k=k)
    except Exception:
        return None
    if not hits:
        return None
    ctx = derive_style_context(hits)
    if not ctx.available:
        return None
    return ctx
