"""
RAVANA Verb Lexicon — Thematic-Role-Driven Verb Selection
==========================================================
Replaces VERB_PHRASES (57 canned phrases) with a semantically-indexed
verb lexicon. Grounded in Levelt's lemma retrieval model (1989) — verbs
are lemmas selected via competitive semantic activation, not random choice.

Theoretical grounding:
- Levelt, W. J. M. (1989). Speaking: From intention to articulation.
  Lemma selection is driven by conceptual activation + grammatical constraints.
- Bornkessel-Schlesewsky & Schlesewsky (2008). The P600 reflects
  syntactic unification cost, modulated by verb-argument fit.
- Ferretti et al. (2001). Verbs automatically activate thematic roles
  (agent, patient, instrument) during comprehension.
- Frank, M. J., et al. (2004). By carrot or by stick: cognitive reinforcement
  learning in Parkinsonism. Dopamine modulates exploration/exploitation
  trade-off in decision making — applied here to lexical selection.
- Levelt, W. J. M., Roelofs, A., & Meyer, A. S. (1999). A theory of
  lexical access in speech production (WEAVER++). Lemma selection is
  competitive with noise — recently selected lemmas are transiently
  inhibited (refractory period) to prevent perseveration.

Design:
- Verb phrases are grouped by relation type (semantic, causal, contrastive, etc.)
- Selection uses softmax-with-temperature over all candidates, where
  temperature is modulated by dopamine_tone (continuous, no hard threshold)
  - Low dopamine → low temperature → near-deterministic (exploit best match)
  - High dopamine → high temperature → more exploration (lawful stochasticity)
- After selection, the chosen verb enters a refractory period (temporary
  inhibition) to prevent same-verb perseveration across consecutive sentences.
  This mirrors the lemma inhibition mechanism in WEAVER++.
- High similarity (close concepts) → simpler, more direct verbs
- Low similarity (distant concepts) → more complex, hedging verbs
- This mirrors P600 amplitude modulation: unexpected combinations require
  more complex syntactic processing
"""

import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Callable, Set


class VerbLexicon:
    """Thematically-organized verb lexicon with semantic selection.

    Each verb has:
    - phrase: the actual verb phrase string
    - relation: which relation type it expresses
    - complexity: 0-1 (simple "is" vs elaborate "has a lot to do with")
    - vector: approximate GloVe embedding (set at init time)

    Selection algorithm:
    1. Filter verbs matching the requested relation type
    2. Score each verb by semantic similarity to avg(subject, object)
    3. High similarity → prefer simple verbs (direct, high confidence)
    4. Low similarity → prefer complex verbs (hedging, exploratory)
    5. Softmax-with-temperature selection (continuous, no hard threshold)
       Low DA → low temperature → near-deterministic
       High DA → high temperature → lawful stochasticity
    6. Refractory period: recently selected verbs are temporarily inhibited
       to prevent perseveration (WEAVER++ lemma inhibition)
    """

    # Refractory period state: verbs selected in the current turn
    _refractory: Set[str] = set()
    _refractory_relation: str = ""  # relation of the last selection batch

    @classmethod
    def reset_refractory(cls):
        """Clear the refractory period (call at start of each turn)."""
        cls._refractory.clear()
        cls._refractory_relation = ""

    @classmethod
    def _softmax(cls, scores: List[float], temperature: float) -> List[float]:
        """Softmax with temperature.

        Args:
            scores: Raw scores (higher = better)
            temperature: 0+ (lower = more deterministic, higher = more uniform)

        Returns:
            Probability distribution
        """
        if temperature < 0.01:
            temperature = 0.01
        shifted = [s - max(scores) for s in scores]
        exp_s = [np.exp(s / temperature) for s in shifted]
        total = sum(exp_s)
        return [e / total for e in exp_s]

    VERB_PATTERNS = {
        'semantic': [
            "ties into",
            "is part of",
            "plays a role in",
            "feeds into",
            "goes hand in hand with",
            "is bound up with",
            "is deeply connected with",
            "is tied to",
            "has a relationship with",
            "has a lot to do with",
        ],
        'causal': [
            "leads to",
            "creates",
            "causes",
            "brings about",
            "influences",
            "gives rise to",
            "results in",
            "sparks",
            "triggers",
            "fuels",
            "contributes to",
            "drives",
            "prompts",
        ],
        'contrastive': [
            "contrasts with",
            "differs from",
            "stands against",
            "challenges",
            "is the opposite of",
            "clashes with",
            "pulls against",
            "runs counter to",
            "is at odds with",
            "diverges from",
            "pushes back against",
        ],
        'analogical': [
            "is like",
            "resembles",
            "mirrors",
            "echoes",
            "is similar to",
            "can be compared to",
            "is akin to",
            "parallels",
            "reflects",
            "brings to mind",
            "reminds us of",
        ],
        'temporal': [
            "comes before",
            "follows",
            "leads into",
            "precedes",
            "happens before",
            "occurs after",
            "ushers in",
            "paves the way for",
            "sets the stage for",
            "traces back to",
        ],
        'episodic': [
            "brings up",
            "recalls",
            "reminds us of",
            "is linked with",
            "ties into",
            "feeds into",
        ],
    }

    COMPLEXITY: Dict[str, float] = {}

    _glove_vector_fn: Optional[Callable] = None

    @classmethod
    def _init_complexity(cls):
        if cls.COMPLEXITY:
            return
        cls.COMPLEXITY = {
            "ties into": 0.55,
            "is part of": 0.30,
            "plays a role in": 0.70,
            "feeds into": 0.50,
            "goes hand in hand with": 0.85,
            "is bound up with": 0.80,
            "is deeply connected with": 0.80,
            "is tied to": 0.40,
            "has a relationship with": 0.60,
            "has a lot to do with": 0.75,
            "leads to": 0.35,
            "creates": 0.25,
            "causes": 0.20,
            "brings about": 0.60,
            "influences": 0.40,
            "gives rise to": 0.70,
            "results in": 0.45,
            "sparks": 0.40,
            "triggers": 0.35,
            "fuels": 0.35,
            "contributes to": 0.55,
            "drives": 0.30,
            "prompts": 0.40,
            "contrasts with": 0.50,
            "differs from": 0.45,
            "stands against": 0.55,
            "challenges": 0.35,
            "is the opposite of": 0.65,
            "clashes with": 0.50,
            "pulls against": 0.60,
            "runs counter to": 0.65,
            "is at odds with": 0.65,
            "diverges from": 0.55,
            "pushes back against": 0.65,
            "is like": 0.20,
            "resembles": 0.30,
            "mirrors": 0.35,
            "echoes": 0.30,
            "is similar to": 0.35,
            "can be compared to": 0.55,
            "is akin to": 0.50,
            "parallels": 0.40,
            "reflects": 0.35,
            "brings to mind": 0.60,
            "reminds us of": 0.55,
            "comes before": 0.30,
            "follows": 0.20,
            "leads into": 0.40,
            "precedes": 0.35,
            "happens before": 0.35,
            "occurs after": 0.35,
            "ushers in": 0.50,
            "paves the way for": 0.60,
            "sets the stage for": 0.60,
            "traces back to": 0.55,
            "brings up": 0.40,
            "recalls": 0.35,
            "reminds us of": 0.50,
            "is linked with": 0.45,
        }

    @classmethod
    def set_glove_fn(cls, fn: Callable):
        cls._glove_vector_fn = fn

    @classmethod
    def get_phrases(cls, relation: str) -> List[str]:
        return cls.VERB_PATTERNS.get(relation, cls.VERB_PATTERNS['semantic'])

    @classmethod
    def select_verb(cls, relation: str, subject: str = "",
                    object: str = "", dopamine_tone: float = 0.5,
                    vector_fn: Optional[Callable] = None) -> str:
        """Select verb phrase by semantic similarity with softmax sampling.

        P600-grounded algorithm (Phase 6a):
        1. Compute semantic similarity between subject and object
        2. High similarity (close concepts) → simple, direct verbs (low P600 cost)
        3. Low similarity (distant concepts) → complex, hedging verbs (high P600 cost)
        4. Softmax-with-temperature selection:
           - Low DA → low temperature → near-deterministic (exploit)
           - High DA → high temperature → lawful stochasticity (explore)
        5. Refractory period: verbs selected recently are penalized to prevent
           perseveration (WEAVER++ lemma inhibition, Levelt et al. 1999)

        Args:
            relation: Relation type (semantic, causal, contrastive, etc.)
            subject: Subject concept
            object: Object concept
            dopamine_tone: 0-1, exploration vs exploitation
            vector_fn: Optional function to get concept vector by label

        Returns:
            Selected verb phrase string
        """
        cls._init_complexity()
        fn = vector_fn or cls._glove_vector_fn
        phrases = cls.get_phrases(relation)
        if not phrases:
            return "relates to"

        # Reset refractory if relation changed (new turn or new intent)
        if relation != cls._refractory_relation:
            cls.reset_refractory()

        # Compute semantic similarity between subject and object
        similarity = 0.5
        if fn is not None and subject and object:
            try:
                sv = fn(subject)
                ov = fn(object)
                if sv is not None and ov is not None:
                    sim = float(np.dot(sv, ov))
                    similarity = max(0.0, min(1.0, (sim + 1.0) * 0.5))
            except Exception:
                pass

        # Score each phrase by complexity-similarity alignment
        # High similarity → prefer low complexity (simple, direct)
        # Low similarity → prefer high complexity (hedging, exploratory)
        raw_scores = []
        for phrase in phrases:
            comp = cls.COMPLEXITY.get(phrase, 0.5)
            # Target complexity: similarity maps to [0.15, 0.55] range
            target = 0.15 + (1.0 - similarity) * 0.55
            score = 1.0 - abs(comp - target)
            # Apply refractory penalty: recently selected verbs get -0.5
            if phrase in cls._refractory:
                score -= 0.5
            raw_scores.append((phrase, score))

        if not raw_scores:
            return "relates to"

        phrases_list, scores_list = zip(*raw_scores)

        # Temperature: maps dopamine_tone [0.0, 1.0] → temperature [0.05, 0.5]
        # Low DA = low temperature = sharp distribution (deterministic)
        # High DA = high temperature = flatter distribution (exploratory)
        temperature = 0.05 + dopamine_tone * 0.45

        # Softmax sampling
        probs = cls._softmax(list(scores_list), temperature)
        idx = random.choices(range(len(phrases_list)), weights=probs, k=1)[0]
        selected = phrases_list[idx]

        # Add to refractory period
        cls._refractory.add(selected)
        cls._refractory_relation = relation

        return selected

    @classmethod
    def select_verb_cerebellar(cls, relation: str, cerebellar_ngram,
                                subject: str = "", object: str = "",
                                dopamine_tone: float = 0.5,
                                vector_fn: Optional[Callable] = None) -> str:
        """Select verb with cerebellar n-gram override.

        If the cerebellar n-gram has a preferred pattern for this
        (relation, subject, object) triple, boost the relevant phrase.
        Otherwise, use the standard semantic similarity algorithm.
        """
        phrase = cls.select_verb(relation, subject, object,
                                 dopamine_tone, vector_fn)

        if cerebellar_ngram is not None:
            ngram_key = f"phrase:{relation}"
            ngram_result = cerebellar_ngram.predict_next(ngram_key, top_k=3)
            if ngram_result and dopamine_tone < 0.4:
                return list(ngram_result.keys())[0]

        return phrase
