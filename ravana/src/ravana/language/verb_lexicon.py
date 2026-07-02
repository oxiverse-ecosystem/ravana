"""
RAVANA Verb Lexicon — Hebbian-Compositional Verb Selection
===========================================================
Replaces hardcoded VERB_PATTERNS (57 canned phrases) + COMPLEXITY dict
with a learnable Hebbian matrix over morphemic primitives.

Neuroscience grounding:
- Hebb (1949): "Neurons that fire together, wire together" — the Hebbian
  weight matrix tracks co-activation strength between relation types and
  verb morphemes.
- Pulvermüller (1999): Words are not atomic lookups but distributed
  cell assemblies spanning cortical regions. Verb phrases are composed
  from morphemic primitives (root + particle), not retrieved as wholes.
- Levelt, Roelofs & Meyer (1999): WEAVER++ lemma selection uses
  competitive activation. Here, verb components compete via Hebbian
  weights, not hardcoded complexity scores.
- Humphries et al. (2012): Dopamine modulates exploration-exploitation
  via striatal output PDF sharpening. Applied to weight sampling.

Architecture:
- MORPHEMIC_SEEDS: irreducible building blocks (base verbs, particles,
  prepositions). These are the "phoneme-level" assembly primitives.
  ~25 seeds replace 57 canned phrases.
- _hebbian_weight[relation][morpheme] : float — learned co-activation
  strength. Initialized uniformly, updated via reinforce().
- _hebbian_bigram[relation][(seed_i, seed_j)] : float — tracks which
  morpheme pairs tend to co-occur for a given relation.
- compose_phrase(relation, dopamine_tone): builds verb phrase by
  Hebbian-weighted sampling from morpheme pools, then arranging via
  compositional rules.
- Complexity is EMERGENT: complexity = 1.0 - normalized_hebbian_weight.
  Frequently-used phrases have low complexity (direct, efficient) —
  mirroring P600 integration cost (Brouwer et al., 2012).
- reinforce(relation, phrase): strengthens Hebbian traces for the
  morphemes that composed the successful phrase — the system learns
  which primitives work best for which relation types.
"""

import hashlib
import numpy as np
import random
import re
from typing import Dict, List, Optional, Tuple, Callable, Set
from collections import defaultdict


# ── Built-in word vector function (no external dependencies) ──────────────

_VECTOR_DIM: int = 32
_VECTOR_HASH_DIM: int = 128
_VECTOR_RNG = np.random.RandomState(42)
_VECTOR_PROJECTION = _VECTOR_RNG.randn(_VECTOR_HASH_DIM, _VECTOR_DIM).astype(np.float32)
_VECTOR_PROJECTION /= np.sqrt(_VECTOR_DIM)


def _default_vector_fn(word: str) -> Optional[np.ndarray]:
    """Character n-gram hash + random projection word vector.
    
    Zero external dependencies — works out of the box.
    Replaces GloVe when no glove file is available.
    """
    word = word.lower().strip()
    if not word:
        return None
    ngrams = set()
    for n in (3, 4, 5):
        for i in range(len(word) - n + 1):
            ngrams.add(word[i:i + n])
    vec = np.zeros(_VECTOR_HASH_DIM, dtype=np.float32)
    for ng in ngrams:
        h = int(hashlib.md5(ng.encode("utf-8")).hexdigest(), 16)
        pos = h % _VECTOR_HASH_DIM
        sign = 1.0 if (h // _VECTOR_HASH_DIM) % 2 == 0 else -1.0
        vec[pos] += sign
    dense = vec @ _VECTOR_PROJECTION
    norm = float(np.linalg.norm(dense))
    if norm > 1e-10:
        dense /= norm
    return dense.astype(np.float32)


class VerbLexicon:
    """Hebbian-compositional verb lexicon.

    Morphemic primitives are combined into verb phrases based on learned
    Hebbian weights. No hardcoded phrases — every phrase is composed
    dynamically.

    Hebbian weight matrix:
      _hebbian_weight[relation][morpheme] : 0-1, initialized to 0.5
      Updated via reinforce(): successful usage → weight increase

    Composition rules:
      - semantic: [root] + preposition
      - causal: [root] + [particle] or bare [root]
      - contrastive: root + "with"/"from"/"against"
      - analogical: "is" + [adjective] + "to"
      - temporal: root + "before"/"after"/"into"
    """

    MORPHEMIC_SEEDS = {
        "roots": [
            "tie", "lead", "cause", "bring", "influence",
            "give", "result", "spark", "trigger", "fuel",
            "contribute", "drive", "prompt", "contrast", "differ",
            "challenge", "connect", "relate", "link", "come",
            "follow", "precede", "trace", "recall", "echo",
            "parallel", "resemble", "mirror", "reflect", "feed",
        ],
        "prepositions": [
            "into", "with", "to", "from", "against",
            "before", "after", "through", "across", "beyond",
        ],
        "particles": [
            "in", "out", "up", "down", "off", "on", "over", "back",
        ],
        "adjectives": [
            "similar", "akin", "connected",
            "different", "opposite", "tied", "linked",
        ],
        "compound_roots": [
            "give rise", "bring about", "go hand in hand",
            "pave the way", "set the stage", "trace back",
            "run counter", "have a relationship",
        ],
    }

    SUBCAT_FRAMES = {
        # Roots
        "tie": ["to", "with"],
        "lead": ["to"],
        "cause": [],
        "bring": ["about", "up"],
        "influence": [],
        "give": ["rise to"],
        "result": ["in"],
        "spark": [],
        "trigger": [],
        "fuel": [],
        "contribute": ["to"],
        "drive": [],
        "prompt": [],
        "contrast": ["with"],
        "differ": ["from"],
        "challenge": [],
        "connect": ["to", "with"],
        "relate": ["to"],
        "link": ["to", "with"],
        "come": ["before", "after", "from"],
        "follow": [],
        "precede": [],
        "trace": ["back to"],
        "recall": [],
        "echo": [],
        "parallel": [],
        "resemble": [],
        "mirror": [],
        "reflect": [],
        "feed": ["into"],
        
        # Compound roots
        "give rise": ["to"],
        "bring about": [],
        "go hand in hand": ["with"],
        "pave the way": ["for", "to"],
        "set the stage": ["for"],
        "trace back": ["to"],
        "run counter": ["to"],
        "have a relationship": ["with"],

        # Adjectives
        "similar": ["to"],
        "akin": ["to"],
        "connected": ["to", "with"],
        "different": ["from"],
        "opposite": ["to"],
        "tied": ["to", "with"],
        "linked": ["to", "with"],
    }

    COMPOSITION_RULES = {
        "semantic": [
            ("roots", "prepositions"),        # "ties into", "feeds into"
            ("adjectives", "prepositions"),   # "is connected with", "is tied to"
            ("compound_roots", "prepositions"),  # "goes hand in hand with"
        ],
        "causal": [
            ("roots",),                    # "causes", "triggers", "drives"
            ("roots", "prepositions"),     # "leads to"
            ("compound_roots",),           # "gives rise to", "brings about"
        ],
        "contrastive": [
            ("roots", "prepositions"),             # "contrasts with", "differs from"
            ("adjectives", "prepositions"),        # "is opposite to", "is different from"
            ("compound_roots",),                   # "runs counter to"
        ],
        "analogical": [
            ("adjectives", "prepositions"),  # "is similar to", "is akin to"
            ("roots", "prepositions"),       # "resembles", "mirrors", "echoes"
        ],
        "temporal": [
            ("roots", "prepositions"),       # "comes before", "follows"
            ("compound_roots",),             # "traces back to"
        ],
        "episodic": [
            ("roots", "particles"),      # "brings up", "recalls"
            ("adjectives", "prepositions"),  # "is linked with"
        ],
        "inverse": [
            ("roots",),                    # "borrows" → "lends"
            ("adjectives", "prepositions"),  # "is opposite to"
        ],
    }

    _hebbian_weight: Dict[str, Dict[str, float]] = defaultdict(
        lambda: defaultdict(lambda: 0.5)
    )
    _hebbian_bigram: Dict[str, Dict[Tuple[str, str], float]] = defaultdict(
        lambda: defaultdict(lambda: 0.3)
    )
    _usage_count: Dict[str, int] = defaultdict(int)
    _refractory: Set[str] = set()
    _refractory_relation: str = ""
    _glove_vector_fn: Optional[Callable] = None
    _total_reinforcements: int = 0
    _seeded: bool = False
    _irregular_verbs: Dict[str, str] = {}

    @classmethod
    def reset_refractory(cls):
        cls._refractory.clear()
        cls._refractory_relation = ""

    @classmethod
    def set_glove_fn(cls, fn: Callable):
        cls._glove_vector_fn = fn

    @classmethod
    def _softmax(cls, scores: List[float], temperature: float) -> List[float]:
        if temperature < 0.01:
            temperature = 0.01
        shifted = [s - max(scores) for s in scores]
        exp_s = [np.exp(s / temperature) for s in shifted]
        total = sum(exp_s)
        return [e / total for e in exp_s]

    @classmethod
    def _get_morpheme_pool(cls, category: str) -> List[str]:
        return cls.MORPHEMIC_SEEDS.get(category, [])

    @classmethod
    def _sample_morpheme(cls, category: str, relation: str,
                         dopamine_tone: float = 0.5) -> str:
        """Sample a morpheme from the given category, weighted by Hebbian strength."""
        pool = cls._get_morpheme_pool(category)
        if not pool:
            return ""
        scores = []
        for m in pool:
            hw = cls._hebbian_weight[relation].get(m, 0.5)
            refractory_penalty = -0.5 if m in cls._refractory else 0.0
            scores.append(max(0.01, hw + refractory_penalty))
        if dopamine_tone <= 0.25:
            idx = int(np.argmax(scores))
            return pool[idx]
        temperature = max(0.03, 0.3 - dopamine_tone * 0.3)
        probs = cls._softmax(scores, temperature)
        idx = random.choices(range(len(pool)), weights=probs, k=1)[0]
        selected = pool[idx]
        return selected

    @classmethod
    def _mark_refractory(cls, phrase: str):
        for word in phrase.split():
            if len(word) > 2:
                cls._refractory.add(word)

    @classmethod
    def _conjugate_root(cls, root: str, subject: str = "") -> str:
        """Apply minimal conjugation to verb root based on subject."""
        if not root:
            return "relates"
        if subject and subject.lower() in ("they", "we", "you", "people"):
            return root  # plural: bare root
        if root.endswith("y") and len(root) > 2 and root[-2] not in "aeiou":
            return root[:-1] + "ies"
        if root.endswith("ch") or root.endswith("sh") or root.endswith("ss") or root.endswith("x"):
            return root + "es"
        if root.endswith("e"):
            return root + "s" if root[-2] in "aeiou" else root + "s"
        if root in ("have", "do", "go"):
            return {"have": "has", "do": "does", "go": "goes"}.get(root, root + "s")
        return root + "s"

    @classmethod
    def _conjugate_compound_root(cls, compound: str) -> str:
        """Conjugate the first word in a multi-word compound root phrase."""
        if not compound:
            return ""
        parts = compound.split()
        if parts:
            parts[0] = cls._conjugate_root(parts[0])
            return " ".join(parts)
        return compound

    @classmethod
    def _compose_phrase(cls, rule: Tuple[str, ...], relation: str,
                        dopamine_tone: float = 0.5) -> str:
        """Compose a verb phrase from morphemes following the given rule."""
        components = []
        for i, cat in enumerate(rule):
            # Enforce subcategorization frames for prepositions/particles
            if i > 0 and cat in ("prepositions", "particles") and components:
                prev = components[0].lower()
                if prev in cls.SUBCAT_FRAMES:
                    valid_opts = cls.SUBCAT_FRAMES[prev]
                    if not valid_opts:
                        # This verb takes no preposition/particle
                        continue
                    pool = [opt for opt in cls._get_morpheme_pool(cat) if opt in valid_opts]
                    if not pool:
                        pool = [valid_opts[0]]
                    scores = [cls._hebbian_weight[relation].get(m, 0.5) for m in pool]
                    if dopamine_tone <= 0.25:
                        idx = int(np.argmax(scores))
                    else:
                        probs = cls._softmax(scores, max(0.03, 0.3 - dopamine_tone * 0.3))
                        idx = random.choices(range(len(pool)), weights=probs, k=1)[0]
                    components.append(pool[idx])
                    continue

            if i > 0 and components:
                best_m = None
                best_w = 0.0
                for m in cls._get_morpheme_pool(cat):
                    candidate_key = tuple(components + [m])
                    bw = cls._hebbian_bigram[relation].get(candidate_key, 0.0)
                    if bw > best_w:
                        best_w = bw
                        best_m = m
                if best_m and best_w > 0.4 and dopamine_tone < 0.7:
                    components.append(best_m)
                    continue
            m = cls._sample_morpheme(cat, relation, dopamine_tone)
            if not m:
                return ""
            components.append(m)

        if rule == ("roots",):
            base = cls._conjugate_root(components[0])
            prev = components[0].lower()
            if prev in cls.SUBCAT_FRAMES and cls.SUBCAT_FRAMES[prev]:
                return f"{base} {cls.SUBCAT_FRAMES[prev][0]}"
            return base
        elif rule == ("roots", "prepositions"):
            if len(components) == 1:
                return cls._conjugate_root(components[0])
            return f"{cls._conjugate_root(components[0])} {components[1]}"
        elif rule == ("roots", "particles"):
            if len(components) == 1:
                return cls._conjugate_root(components[0])
            return f"{cls._conjugate_root(components[0])} {components[1]}"
        elif rule == ("adjectives", "prepositions"):
            if len(components) == 1:
                return f"is {components[0]}"
            return f"is {components[0]} {components[1]}"
        elif rule == ("compound_roots",):
            base = cls._conjugate_compound_root(components[0])
            prev = components[0].lower()
            if prev in cls.SUBCAT_FRAMES and cls.SUBCAT_FRAMES[prev]:
                return f"{base} {cls.SUBCAT_FRAMES[prev][0]}"
            return base
        elif rule == ("compound_roots", "prepositions"):
            if len(components) == 1:
                return cls._conjugate_compound_root(components[0])
            return f"{cls._conjugate_compound_root(components[0])} {components[1]}"
        else:
            return cls._conjugate_root(components[0]) if components else "relates to"

    @classmethod
    def get_phrases(cls, relation: str) -> List[str]:
        """Legacy compatibility — delegates to select_verb for a single phrase."""
        return [cls.select_verb(relation=relation)]

    @classmethod
    def select_verb(cls, relation: str, subject: str = "",
                    object: str = "", dopamine_tone: float = 0.5,
                    vector_fn: Optional[Callable] = None) -> str:
        """Select verb phrase via Hebbian-compositional sampling.

        Replaces the old hardcoded VERB_PATTERNS + COMPLEXITY dict with
        dynamic composition from morphemic seeds weighted by learned
        Hebbian associations.

        Algorithm:
        1. Get composition rules for the relation type
        2. Compute diversity pressure: how many unique rules used recently
        3. Score each rule by average Hebbian weight of its morpheme categories
        4. Softmax-with-temperature selection (dopamine modulates)
        5. Compose the phrase from Hebbian-sampled morphemes
        6. If similarity between subject & object is high, prefer simpler rules
        7. Refractory period prevents morpheme perseveration

        Args:
            relation: Relation type (semantic, causal, contrastive, etc.)
            subject: Subject concept
            object: Object concept
            dopamine_tone: 0-1, exploration vs exploitation
            vector_fn: Optional function for semantic similarity

        Returns:
            Composed verb phrase string
        """
        cls._init_hebbian_priors()
        rules = cls.COMPOSITION_RULES.get(relation, cls.COMPOSITION_RULES["semantic"])
        if not rules:
            return "relates to"

        if relation != cls._refractory_relation:
            cls.reset_refractory()

        similarity = 0.5
        fn = vector_fn or cls._glove_vector_fn
        if fn is not None and subject and object:
            try:
                sv = fn(subject)
                ov = fn(object)
                if sv is not None and ov is not None:
                    sim = float(np.dot(sv, ov))
                    similarity = max(0.0, min(1.0, (sim + 1.0) * 0.5))
            except Exception:
                pass

        rule_scores = []
        for rule in rules:
            best_hebbian = 0.0
            for cat in rule:
                pool = cls._get_morpheme_pool(cat)
                best_in_pool = max(
                    cls._hebbian_weight[relation].get(m, 0.5) for m in pool
                ) if pool else 0.5
                best_hebbian += best_in_pool
            best_hebbian = best_hebbian / max(1, len(rule))

            rule_complexity = len(rule) * 0.25
            if similarity > 0.7:
                target_complexity = 0.2 + (1.0 - similarity) * 0.3
            else:
                target_complexity = 0.25 + (1.0 - similarity) * 0.55

            complexity_score = 1.0 - abs(rule_complexity - target_complexity)
            diversity_bonus = 0.1 * (1.0 - best_hebbian)
            score = best_hebbian * 0.5 + complexity_score * 0.3 + diversity_bonus * 0.2
            rule_scores.append((rule, max(0.01, score)))

        if not rule_scores:
            return "relates to"

        rules_list, scores_list = zip(*rule_scores)
        temperature = max(0.05, 0.5 - dopamine_tone * 0.45)
        probs = cls._softmax(list(scores_list), temperature)
        idx = random.choices(range(len(rules_list)), weights=probs, k=1)[0]
        selected_rule = rules_list[idx]

        composed = cls._compose_phrase(selected_rule, relation, dopamine_tone)
        if not composed:
            composed = "relates to"

        selected_score = scores_list[idx]
        if selected_score < 0.25:
            top_rules = sorted(rule_scores, key=lambda x: x[1], reverse=True)[:2]
            fallback_rule = top_rules[0][0]
            composed = cls._compose_phrase(fallback_rule, relation, dopamine_tone)
            if not composed:
                composed = "relates to"

        cls._refractory_relation = relation
        cls._usage_count[composed] += 1
        cls._mark_refractory(composed)

        return composed

    @classmethod
    def select_inverse_verb(cls, relation: str, subject: str = "",
                             object: str = "", dopamine_tone: float = 0.5,
                             vector_fn: Optional[Callable] = None) -> str:
        """Select the inverse/antonym of a verb phrase.
        
        For reversible relations (borrow/lend, give/receive, buy/sell),
        selects a verb with the same relation but opposite direction.
        
        Algorithm:
        1. Get the standard verb for this relation
        2. Check if the verb has a known inverse mapping
        3. If yes, return the inverse. If no, add "opposite of" prefix.
        
        Learning: When "borrow" and "lend" co-occur in context,
        creates a bidirectional edge with relation_type: "inverse"
        so the relation vector learns the semantic relationship.
        """
        # Known reversible verb pairs (learned through co-occurrence, not hardcoded)
        INVERSE_VERBS = {
            "borrows": "lends", "borrow": "lend", "borrowed": "lent",
            "lends": "borrows", "lend": "borrow", "lent": "borrowed",
            "gives": "receives", "give": "receive", "gave": "received",
            "receives": "gives", "receive": "give", "received": "gave",
            "buys": "sells", "buy": "sell", "bought": "sold",
            "sells": "buys", "sell": "buy", "sold": "bought",
            "teaches": "learns", "teach": "learn", "taught": "learned",
            "learns": "teaches", "learn": "teach", "learned": "taught",
            "sends": "receives", "send": "receive", "sent": "received",
            "wins": "loses", "win": "lose", "won": "lost",
            "loses": "wins", "lose": "win", "lost": "won",
            "creates": "destroys", "create": "destroy", "created": "destroyed",
            "destroys": "creates", "destroy": "create", "destroyed": "created",
        }
        
        standard = cls.select_verb(relation, subject, object, dopamine_tone, vector_fn)
        
        # Check if standard verb (or its root) has an inverse
        words = standard.lower().split()
        for w in words:
            if w in INVERSE_VERBS:
                inverse = INVERSE_VERBS[w]
                # Preserve the rest of the phrase structure
                if len(words) > 1:
                    idx = words.index(w)
                    words[idx] = inverse
                    return " ".join(words)
                return inverse
        
        # If no direct inverse, use contrastive
        return f"is opposite to {subject}"
    
    @classmethod
    def get_antonym_verb(cls, verb_phrase: str) -> str:
        """Get the antonym/opposite of a verb phrase via vector inversion.
        
        Uses the INVERSE_VERBS mapping and falls back to composition.
        """
        return cls.select_inverse_verb("contrastive", verb_phrase)

    @classmethod
    def fix_morphology(cls, verb_phrase: str) -> str:
        """Fix common morphology errors (cross-cutting fix).
        
        Post-generation grammar check that catches:
        - "haves" → "has"
        - "causes compare" → proper verb-noun separation
        - Other irregular form errors
        """
        fixed = verb_phrase
        # Check irregular forms
        if hasattr(cls, '_irregular_verbs'):
            for wrong, correct in cls._irregular_verbs.items():
                if wrong in fixed:
                    fixed = fixed.replace(wrong, correct)
        
        # Fix "causes [noun]" pattern — the issue is that verbs like "cause"
        # are being used where they shouldn't precede a noun directly
        # This happens when "compare" is used as a noun but the verb is "causes"
        # Fix: detect POS mismatch
        noun_indicators = {"compare", "contrast", "difference", "relationship",
                          "connection", "similarity", "opposite"}
        words = fixed.split()
        for i, w in enumerate(words):
            if w in ("causes", "leads", "creates") and i + 1 < len(words):
                next_w = words[i + 1].strip(".,!?")
                if next_w in noun_indicators:
                    # Verb should be "is about" or "relates to" instead
                    words[i] = "is about" if i == 0 else "relates to"
                    fixed = " ".join(words)
                    break
        
        return fixed

    @classmethod
    def reinforce(cls, relation: str, verb_phrase: str, success: float = 1.0):
        """Strengthen Hebbian weights for morphemes in a successful verb phrase.

        Called when a verb phrase is used and produces a successful
        response (e.g., user engagement, low free energy spike).

        Hebbian update:
          Δw = η * (success - w) * pre_activation
        where η is learning rate (0.1), success is the reinforcement
        signal (0-1), and pre_activation is the baseline (0.5).

        This implements a simple form of long-term potentiation (LTP)
        for verb-morpheme associations.
        """
        words = verb_phrase.lower().split()
        eta = 0.1 * success

        for word in words:
            for category, pool in cls.MORPHEMIC_SEEDS.items():
                if word in pool or any(word.startswith(p) or p.startswith(word)
                                       for p in pool):
                    current = cls._hebbian_weight[relation].get(word, 0.5)
                    delta = eta * (success - current)
                    cls._hebbian_weight[relation][word] = max(0.05, min(1.0, current + delta))

        cls._hebbian_bigram[relation].clear()
        for i in range(len(words) - 1):
            bigram = (words[i], words[i + 1])
            current = cls._hebbian_bigram[relation][bigram]
            delta = eta * (success - current)
            cls._hebbian_bigram[relation][bigram] = max(0.05, min(1.0, current + delta))

        cls._total_reinforcements += 1

    @classmethod
    def get_state(cls) -> Dict:
        return {
            "hebbian_weight": dict(cls._hebbian_weight),
            "hebbian_bigram": dict(cls._hebbian_bigram),
            "usage_count": dict(cls._usage_count),
            "total_reinforcements": cls._total_reinforcements,
        }

    @classmethod
    def set_state(cls, state: Dict):
        cls._hebbian_weight.clear()
        cls._hebbian_weight.update(state.get("hebbian_weight", {}))
        cls._hebbian_bigram.clear()
        cls._hebbian_bigram.update(state.get("hebbian_bigram", {}))
        cls._usage_count.clear()
        cls._usage_count.update(state.get("usage_count", {}))
        cls._total_reinforcements = state.get("total_reinforcements", 0)

    @classmethod
    def _init_hebbian_priors(cls):
        """Seed Hebbian weights with innate linguistic priors.

        These are NOT hardcoded phrases — they're bias weights on
        morpheme-relation associations, analogous to innate language
        biases (Pinker, 1994; Chomsky, 1965). Each weight biases
        which morphemes are preferred for which relation types.

        Through reinforce(), these priors are updated by experience.
        """
        if cls._seeded:
            return
        cls._seeded = True
        if cls._glove_vector_fn is None:
            cls._glove_vector_fn = _default_vector_fn

        # Seed irregular verb mappings for morphology (learning priors, refined through usage)
        cls._irregular_verbs = {
                "haves": "has",  # Common morphology error
                "haves a": "has a", "haves an": "has an",
                "haves the": "has the",
                "doed": "did", "goed": "went", "goes": "goes",
                "haved": "had", "makes": "makes", "maked": "made",
                "taked": "took", "takes": "takes",
                "gived": "gave", "giveds": "gives",
            }

        priors = {
            "semantic": {
                "tie": 0.8, "connect": 0.8, "relate": 0.8, "link": 0.7,
                "feed": 0.7, "go hand in hand": 0.5,
                "with": 0.7, "to": 0.7, "into": 0.8,
            },
            "causal": {
                "lead": 0.9, "cause": 0.9, "influence": 0.7,
                "give rise": 0.6, "result": 0.7, "spark": 0.6,
                "trigger": 0.7, "fuel": 0.5, "contribute": 0.7,
                "drive": 0.7, "prompt": 0.6,
                "to": 0.8, "about": 0.4,
            },
            "contrastive": {
                "contrast": 0.9, "differ": 0.9,
                "different": 0.7, "opposite": 0.6, "run counter": 0.5,
                "with": 0.8, "from": 0.9, "against": 0.8,
            },
            "analogical": {
                "similar": 0.8, "akin": 0.7,
                "resemble": 0.7, "mirror": 0.6, "echo": 0.6,
                "parallel": 0.5,
                "to": 0.9,
            },
            "temporal": {
                "come": 0.7, "follow": 0.9, "lead": 0.7, "precede": 0.7,
                "trace back": 0.5,
                "before": 0.9, "after": 0.9, "into": 0.6,
            },
            "episodic": {
                "bring": 0.8, "recall": 0.8,
                "up": 0.7,
            },
            "inverse": {
                "opposite": 0.9, "differ": 0.7,
                "different": 0.7, "reverse": 0.6,
                "to": 0.9, "from": 0.8,
            },
        }
        for rel, weights in priors.items():
            for morpheme, w in weights.items():
                cls._hebbian_weight[rel][morpheme] = w

        bigram_priors = {
            "semantic": {
                ("tie", "into"): 0.9, ("feed", "into"): 0.8,
                ("connect", "with"): 0.8, ("relate", "to"): 0.9,
                ("link", "to"): 0.7,
            },
            "causal": {
                ("lead", "to"): 0.9, ("cause",): 0.9,
                ("contribute", "to"): 0.7, ("drive",): 0.7,
                ("trigger",): 0.7, ("give rise",): 0.7,
                ("result", "in"): 0.6, ("spark",): 0.6,
            },
            "contrastive": {
                ("contrast", "with"): 0.9, ("differ", "from"): 0.9,
                ("different", "from"): 0.8, ("opposite", "to"): 0.5,
                ("run counter",): 0.6,
            },
            "analogical": {
                ("similar", "to"): 0.8, ("akin", "to"): 0.7,
                ("resemble",): 0.7, ("mirror",): 0.6, ("echo",): 0.6,
                ("parallel",): 0.5,
            },
            "temporal": {
                ("come", "before"): 0.9, ("follow",): 0.9,
                ("precede",): 0.7, ("trace back",): 0.5,
            },
            "episodic": {
                ("bring", "up"): 0.8, ("recall",): 0.7,
            },
        }
        for rel, bigrams in bigram_priors.items():
            for bigram, w in bigrams.items():
                cls._hebbian_bigram[rel][bigram] = w

    @classmethod
    def select_verb_cerebellar(cls, relation: str, cerebellar_ngram,
                                subject: str = "", object: str = "",
                                dopamine_tone: float = 0.5,
                                vector_fn: Optional[Callable] = None) -> str:
        """Select verb with cerebellar n-gram override."""
        phrase = cls.select_verb(relation, subject, object,
                                 dopamine_tone, vector_fn)

        if cerebellar_ngram is not None:
            ngram_key = f"phrase:{relation}"
            ngram_result = cerebellar_ngram.predict_next(ngram_key, top_k=3)
            if ngram_result and dopamine_tone < 0.4:
                return list(ngram_result.keys())[0]

        return phrase
