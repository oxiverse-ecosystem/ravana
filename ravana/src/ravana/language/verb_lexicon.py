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

Design:
- Verb phrases are grouped by relation type (semantic, causal, contrastive, etc.)
- Selection is driven by semantic vector similarity between verb and (subject, object)
- High similarity → simpler, more direct verbs (low unification cost)
- Low similarity → more complex, hedging verbs (high unification cost)
- This mirrors P600 amplitude modulation: unexpected combinations require
  more complex syntactic processing
- Dopamine tone adds exploration noise on top of the similarity signal
"""

import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Callable


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
    5. Dopamine tone adds random exploration noise
    """

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
        """Select verb phrase by semantic similarity.

        P600-grounded algorithm:
        1. Compute semantic similarity between subject and object
        2. High similarity (close concepts) → simple, direct verbs
        3. Low similarity (distant concepts) → complex, hedging verbs
        4. Dopamine tone adds exploration noise

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
        scores = []
        for phrase in phrases:
            comp = cls.COMPLEXITY.get(phrase, 0.5)
            # Target complexity: similarity maps to [0.15, 0.55] range
            target = 0.15 + (1.0 - similarity) * 0.55
            score = 1.0 - abs(comp - target)
            scores.append((phrase, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Dopamine modulation: high DA → pick from top-k with noise
        if dopamine_tone > 0.6 and len(scores) > 2:
            k = max(2, int(len(scores) * (1.0 - dopamine_tone * 0.5)))
            k = min(k, len(scores) - 1)
            candidates = scores[:max(3, k)]
            weights = [s + 0.1 for _, s in candidates]
            return random.choices(candidates, weights=weights, k=1)[0][0]

        # Deterministic: pick highest scoring
        return scores[0][0]

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
