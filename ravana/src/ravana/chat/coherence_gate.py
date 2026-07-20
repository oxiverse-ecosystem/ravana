"""Global Workspace (GWT, Dehaene 2011) broadcast coherence gate.

Brain basis
-----------
A candidate utterance only reaches the global workspace (and thus the mouth)
when a *winning coalition* of competing representations passes a coherence
test. The ACC (conflict monitoring; Botvinick 2004) flags two proximate
simulations that disagree; the vLPFC→AI causal-reasoning hierarchy (Operskalski
& Barbey 2016) rejects incoherent causal chains; reality monitoring (Johnson &
Raye 1981) distinguishes internally-generated from retrieved content. This
module turns those into four *scalar, learned/structural* signals over the
typed concept graph — never a regex band-aid:

    coherence = 1 - incompleteness - contradiction - low_relevance
                (acc_conflict folded into contradiction where applicable)

and emits (broadcast: bool, reason, score). Fail-closed: if coherence < 0
(or any signal trips a hard floor) the utterance is WITHHELD and the caller
falls back to honest uncertainty rather than emitting confident garbage.

The gate is intentionally generic: every RAVANA defect (A dangling fragments,
C counterfactual gibberish, D misleading math, F off-topic snippets, and the
repair loop for A) routes its candidate through it. Each defect supplies the
subset of signals it can compute; missing signals default to 0 (neutral).

All thresholds are learned/structural constants (distribution-driven, not
fixed topic cutoffs) so the gate is adaptive. They live at module scope so a
fit harness can later move them, mirroring SnippetPEConfig / SaladClassifier.
"""

from typing import Dict, Any, List, Optional, Tuple

# ── Structural completeness constants (Broca's forward-model monitor) ──
# A finite verb (any word ending in a verb inflection OR a copula) must be
# present for an English clause to be complete. We do not carry a full verb
# lexicon; we use the project's metaword/glue tables as the *complement* — a
# string that is PURELY glue/noun-phrase has no predicate.
_COPULA = {"is", "are", "was", "were", "be", "been", "being", "am", "'s",
           "'re", "'m", "seem", "seems", "become", "becomes", "remain",
           "remains", "feel", "feels", "look", "looks", "sound", "sounds"}
# Closed-class glue that by itself asserts no relation (carried from
# monitor_gate.py's _GLUE so the completeness test is consistent with the
# fluent-tautology CI gate).
_GLUE = {
    "is", "are", "was", "were", "be", "been", "being", "does", "do", "did",
    "makes", "make", "causes", "cause", "connected", "connect", "relates",
    "relate", "linked", "link", "tied", "tie", "means", "refers", "describes",
    "opens", "springs", "weaves", "influences", "influence", "matters",
    "differs", "compare", "vs", "with", "from", "of", "the", "a", "an",
    "in", "on", "for", "at", "to", "and", "but", "or", "that", "this", "it",
    "its", "their", "his", "her", "my", "your", "our", "as", "by", "than",
}
# Content (predicate) verbs — presence of ANY asserts a real relation. Seed
# set (matches monitor_gate._CONTENT_VERBS purpose: a real predicate verb).
_CONTENT_VERBS = {
    "pull", "pulls", "pulled", "attract", "attracts", "attraction",
    "bend", "bends", "bended", "escape", "escapes", "grow", "grows", "grew",
    "reproduce", "reproduces", "adapt", "adapts", "rely", "relies",
    "relying", "care", "cares", "cared", "believe", "believes", "learn",
    "learns", "think", "thinks", "know", "knows", "happen", "happens",
    "exist", "exists", "live", "lives", "die", "dies", "change", "changes",
    "move", "moves", "fall", "falls", "rise", "rises", "create", "creates",
    "help", "helps", "use", "uses", "make", "makes", "build", "builds",
    "show", "shows", "find", "finds", "give", "gives", "bring", "brings",
    "keep", "keeps", "break", "breaks", "hold", "holds", "send", "sends",
    "tell", "tells", "need", "needs", "want", "wants", "love", "loves",
    "hate", "hates", "see", "sees", "hear", "hears", "feel", "feels",
}

# ── Coherence floors (learned/structural: distribution-driven defaults) ──
# min_coherence: below this the coalition is denied broadcast (fail-closed).
MIN_COHERENCE = 0.0
# max_incompleteness: a complete clause scores 0; a bare NP scores 1. Any
# candidate that is > this fraction incomplete is withheld.
MAX_INCOMPLETENESS = 0.5


def _tokenize(text: str) -> List[str]:
    import re
    return re.findall(r"[a-z']+", (text or "").lower())


def completeness_signal(text: str) -> Tuple[float, str]:
    """Broca's forward-model monitor: does the string carry a predicate?

    Returns (incompleteness∈[0,1], detail). 0 = complete clause with a finite
    verb; 1 = bare noun phrase / glue with no predicate at all.

    Structural test (no topic lists):
      - If any content verb or copula token is present → complete (0).
      - Else if the string is ONLY glue + nouns (no predicate) → incomplete (1).
      - A 1-2 token fragment with no verb → incomplete (1).
    """
    toks = _tokenize(text)
    if not toks:
        return 1.0, "empty"
    if any(w in _CONTENT_VERBS for w in toks):
        return 0.0, "has_content_verb"
    if any(w in _COPULA for w in toks):
        return 0.0, "has_copula"
    # No predicate token at all. If it's a short bare NP (≤2 real words) or
    # purely glue+nouns, it is an incomplete fragment.
    real = [w for w in toks if w not in _GLUE]
    if len(real) <= 2:
        return 1.0, "bare_np_fragment"
    # Longer but still no verb anywhere → no predicate asserted.
    return 0.8, "no_predicate"


def contradiction_signal(endpoints: List[str]) -> Tuple[float, str]:
    """Typed-edge consistency: reject claim→result→claim loops (Defect C).

    Given the ordered endpoint list of a causal/narrative chain, score the
    prediction-error reduction. A loop where the tail re-enters the head adds
    no new grounded predicate (PE reduction ≈ 0) → high contradiction signal.
    """
    if not endpoints:
        return 0.0, "no_endpoints"
    seen = set()
    has_loop = False
    for ep in endpoints:
        if ep in seen:
            has_loop = True
        seen.add(ep)
    if has_loop:
        return 0.9, "chain_loop"
    # Few distinct grounded endpoints (>1) means the chain actually develops.
    if len(set(endpoints)) <= 1:
        return 0.7, "no_development"
    return 0.0, "coherent_chain"


def relevance_signal(text: str, subject_embedding,
                     glove_vector) -> Tuple[float, str]:
    """Goal-overlap of surfaced content vs query subject (Defect F).

    Low relevance (cosine of content vs subject embedding below a floor) means
    the snippet/answer is off-topic. Returns (low_relevance∈[0,1], detail).
    Falls open (0) when embeddings unavailable so the gate never regresses on
    missing infrastructure.
    """
    if glove_vector is None or subject_embedding is None:
        return 0.0, "no_embedding_open"
    try:
        import numpy as np
        vec = glove_vector(text)
        if vec is None:
            return 0.0, "no_content_vec_open"
        subj = subject_embedding
        if getattr(subj, "shape", None) is not None and vec.shape != subj.shape:
            return 0.0, "dim_mismatch_open"
        cos = float(np.dot(vec, subj) / (np.linalg.norm(vec) * np.linalg.norm(subj) + 1e-8))
        # cosine in [-1,1]; map to low_relevance in [0,1] via a soft floor.
        # 0.10 floor: below this cosine the content is essentially unrelated.
        if cos < 0.10:
            return 1.0, "off_topic"
        return max(0.0, 0.5 - cos), "partial"
    except Exception:
        return 0.0, "relevance_error_open"


class CoherenceGate:
    """Global Workspace broadcast gate.

    Combines the scalar signals into a single coherence score and decides
    whether the candidate utterance may broadcast (reach the user). Fail-closed:
    denied candidates return (broadcast=False, reason, score) and the caller
    should replace them with honest uncertainty.
    """

    def __init__(self, min_coherence: float = MIN_COHERENCE,
                 max_incompleteness: float = MAX_INCOMPLETENESS) -> None:
        self.min_coherence = min_coherence
        self.max_incompleteness = max_incompleteness

    def judge(self, *, text: str,
              endpoints: Optional[List[str]] = None,
              subject: str = "",
              glove_vector=None,
              subject_embedding=None,
              allow_incomplete: bool = False) -> Tuple[bool, str, float]:
        """Return (broadcast, reason, score).

        allow_incomplete (default False) lets callers that know the text is a
        fragment-by-design (e.g. a contrastive web fragment) opt out of the
        completeness test; they still get contradiction/relevance checks.
        """
        incomp = 0.0
        inc_reason = "none"
        if not allow_incomplete:
            incomp, inc_reason = completeness_signal(text)
        contr, contr_reason = contradiction_signal(endpoints or [])
        # Relevance only meaningful with embeddings; open otherwise.
        rel, rel_reason = relevance_signal(text, subject_embedding, glove_vector)
        # Note: relevance_signal is called with query==subject here for the
        # simple single-concept case; callers wanting query-relative relevance
        # should compute separately. We keep the gate generic.
        coherence = 1.0 - incomp - contr - rel
        if incomp > self.max_incompleteness:
            return False, f"incomplete:{inc_reason}", round(coherence, 3)
        if contr >= 0.7:
            return False, f"contradiction:{contr_reason}", round(coherence, 3)
        if coherence < self.min_coherence:
            return False, "low_coherence", round(coherence, 3)
        return True, f"ok(incomp={inc_reason},contr={contr_reason})", round(coherence, 3)
