"""
RAVANA Prefrontal Workspace — Discourse Planner
=================================================
Plans multi-sentence discourse BEFORE generating. Inspired by Hagoort's
MUC (Memory, Unification, Control) framework (2005).

KEY DESIGN DECISIONS:
- Plans before generating: discourse intents are determined before any chain walk
- Cross-sentence coherence: each sentence's intent knows what came before
- Replaces independent "per-sentence seen sets" with structured discourse state
- Capacity: 7±2 slots (standard working memory), teens = 5
- Strategy selection is deterministic based on question type, not random
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import re
import numpy as np


class DiscourseType:
    """Discourse intent types for sentence planning."""
    EXPLAIN = "explain"              # subject → semantic relation → object
    CAUSAL_EXPLAIN = "causal_explain"  # subject → causal relation → effect
    ELABORATE = "elaborate"          # go deeper on the previous concept
    CONTRAST = "contrast"            # subject vs opposite perspective
    CONNECT = "connect"             # link to broader context
    ASK_BACK = "ask_back"           # end with a question to the user
    CONTINUE = "continue"           # continue previous topic
    SELF_REFERENCE = "self_reference"  # "I think", "I feel" — epistemic stance
    STATEMENT = "statement"         # user is TELLING RAVANA something (assertion)


@dataclass
class DiscourseIntent:
    """A single sentence's discourse intent."""
    type: str                        # DiscourseType value
    subject: str                     # Main subject of this sentence
    primary_relation: str = "semantic"  # Graph edge type to traverse
    target_concept: str = ""         # Main object of discussion
    secondary_concept: str = ""      # For compare/contrast
    use_epistemic_hedge: bool = False  # "I think", "maybe", etc.
    end_with_question: bool = False  # "what do you think?"
    discourse_marker: str = ""       # "furthermore", "however", "also"
    seen_so_far: set = field(default_factory=set)  # concepts already used


@dataclass
class DiscoursePlan:
    """Full discourse plan for one response turn."""
    intents: List[DiscourseIntent] = field(default_factory=list)
    original_subject: str = ""
    question_type: str = "unknown"


class SpeechActClassifier:
    """Hybrid neuro-symbolic speech-act (illocutionary-force) classifier.

    Why hybrid & future-proof (cognitive-science grounding)
    -------------------------------------------------------
    The literature is unanimous that robust speech-act recognition needs
    BOTH signals, not either alone:
      * Distributional / embedding semantics abstract to higher-level
        meaning and handle paraphrases the brittle rule cascade misses
        (Conrad et al. 2026; TweetAct logistic regression reaches F1 .70 by
        fusing SEMANTIC + SYNTACTIC features, and dropping either group
        hurts F1).
      * But embeddings alone are weak on short declaratives — "classification
        of declaratives remained difficult" (Conrad et al.).
      * Best results come from PROTOTYPE vectors in semantic space
        (Snell et al. prototypical networks; centroid-in-semantic-space
        analyses in psychiatry). A class is the centroid of seed exemplars;
        an utterance is classified by nearest prototype (cosine). This is
        few-shot / extensible: add exemplar phrases and the prototype moves
        — no retraining.
      * Logical-symbolic priors (first word, aux-inversion) stabilise the
        "core" of each class (Lyutikova hybrid: contextual embeddings +
        minimal logical implicants).

    So we classify by semantic PROTOTYPE proximity, refined by a symbolic
    syntactic prior — robust to wording, transparent (we can report the
    nearest prototype + score), and future-proof (extend PROTOTYPES).
    """

    # Seed exemplars per illocutionary force. Adding phrases == learning a
    # new instance; no retraining (prototype networks are few-shot).
    # Kept small and balanced so cosine distances are comparable across
    # classes. Extend freely for directives / expressives / commissives.
    PROTOTYPES = {
        "question": [
            "what is", "what are", "what do", "what does", "what can", "what would",
            "why is", "why are", "why do", "why does", "how is", "how are",
            "how do", "how does", "how can", "who is", "who are", "who was",
            "where is", "where are", "when is", "when did", "which one",
            "are you", "do you", "did you", "can you", "will you", "would you",
            "is it", "was he", "have they", "should i", "could we", "may i",
            "tell me about", "explain", "define", "describe",
        ],
        "statement": [
            "i am", "i'm", "i was", "i have", "we are", "we were", "my name is",
            "he is", "she is", "they are", "it is", "this is", "that was",
            "nothing much", "not much", "i think", "i feel", "i like", "i love",
            "i hate", "he said", "she went", "they made",
            "thanks", "thank you", "nice to meet you", "good to see you",
        ],
    }

    # No hardcoded margin. The decision boundary is a LEARNED property of each
    # class: an utterance's membership is the z-score of its cosine to the class
    # centroid, normalised by how tightly that class's own exemplars cluster
    # (mu_c, sigma_c). Empirically (experiments/regex_vs_prototype_speechact.py)
    # this beats the regex cascade +5/30 with zero constants, and — unlike a
    # fixed 0.12 gate — it re-shapes itself as chat adds exemplars, which is the
    # "learn by chatting" requirement. A fixed threshold is just relocated
    # hardcoding; the exemplar spread is not.

    def __init__(self, vector_fn=None, dim: int = 64):
        """
        Args:
            vector_fn: callable word -> Optional[np.ndarray] (unit vector).
                       Supplies the semantic space (e.g. RAVANA's GloVe).
            dim: embedding dimensionality.
        """
        self.vector_fn = vector_fn
        self.dim = dim
        # Mutable exemplar store, seeded from PROTOTYPES and grown at runtime via
        # add_exemplar() (confirmed chat turns). This is the "learn by chatting"
        # substrate — new classes appear simply by adding exemplars under a new
        # key; no enum edit, no redeploy.
        self.exemplars: Dict[str, List[str]] = {
            cls: list(phrases) for cls, phrases in self.PROTOTYPES.items()
        }
        # Fitted per-class stats (centroid + exemplar-spread mean/std).
        self._centroid: Dict[str, np.ndarray] = {}
        self._mu: Dict[str, float] = {}
        self._sigma: Dict[str, float] = {}
        self._fitted = False
        # Back-compat alias: older callers/tests may read _proto_vecs (centroids).
        self._proto_vecs: Optional[Dict[str, np.ndarray]] = None

    def add_exemplar(self, cls: str, text: str) -> None:
        """Append a learned exemplar (e.g. a confirmed chat turn) under class
        `cls`, marking stats dirty for lazy refit. Unseen classes are created
        automatically — this is how new intents emerge without code edits."""
        self.exemplars.setdefault(cls, []).append(text)
        self._fitted = False

    def _fit(self) -> None:
        """Compute each class's centroid and its exemplar-to-centroid similarity
        spread (mu, sigma). The spread IS the boundary — no hardcoded margin."""
        if self._fitted:
            return
        cen: Dict[str, np.ndarray] = {}
        mu: Dict[str, float] = {}
        sigma: Dict[str, float] = {}
        for cls, phrases in self.exemplars.items():
            vecs = [self._sentence_vector(p) for p in phrases]
            vecs = [v for v in vecs if v is not None]
            if not vecs:
                continue
            M = np.stack(vecs)
            c = M.mean(axis=0)
            n = np.linalg.norm(c)
            if n > 0:
                c = c / n
            sims = M @ c  # each exemplar's cosine to its own class centroid
            cen[cls] = c
            mu[cls] = float(sims.mean())
            # Floor sigma so a single/degenerate exemplar set can't div-by-zero.
            sigma[cls] = float(sims.std()) or 1e-3
        self._centroid = cen
        self._mu = mu
        self._sigma = sigma
        self._proto_vecs = cen  # back-compat
        self._fitted = True

    def _build_prototypes(self) -> None:
        """Back-compat shim for external callers of the old name."""
        self._fit()

    def _store(self) -> Dict[str, Dict]:
        """Serialise fitted stats into the interface-agnostic store shape
        consumed by nearest_prototype()."""
        self._fit()
        return {
            cls: {"centroid": self._centroid[cls],
                  "mu": self._mu[cls],
                  "sigma": self._sigma[cls]}
            for cls in self._centroid
        }

    @staticmethod
    def nearest_prototype(vec, store):
        """Interface-agnostic nearest-prototype lookup (the unified mechanism).

        `vec` is ANY utterance representation — a GloVe centroid today, a
        Word2HyperVec high-D hypervector after A0 — and `store` is a dict
        {class: {"centroid": ndarray, "mu": float, "sigma": float}}. Returns
        (best_class, z_scores, raw_cosines). The decision is argmax of the
        exemplar-spread z-score  (cos(vec, centroid) - mu) / sigma. Because the
        boundary is a learned per-class property, the A0 high-D lift drops in
        behind this exact signature with no rewrite.
        """
        if vec is None or not store:
            return None, {}, {}
        vnorm = float(np.linalg.norm(vec))
        z: Dict[str, float] = {}
        raw: Dict[str, float] = {}
        for cls, stat in store.items():
            cen = stat["centroid"]
            denom = vnorm * float(np.linalg.norm(cen))
            sim = float(np.dot(vec, cen) / denom) if denom > 0 else 0.0
            raw[cls] = sim
            sig = stat.get("sigma") or 1e-3
            z[cls] = (sim - stat.get("mu", 0.0)) / sig
        best = max(z, key=z.get)
        return best, z, raw

    def _sentence_vector(self, text: str) -> Optional[np.ndarray]:
        """Mean-pooled, unit-normalised embedding of the utterance's known
        words (the 'centroid in semantic space' representation)."""
        if self.vector_fn is None:
            return None
        words = re.findall(r"[a-z']+", text.lower())
        vecs = []
        for w in words:
            v = self.vector_fn(w)
            if v is not None:
                vecs.append(v)
        if not vecs:
            return None
        arr = np.stack(vecs).mean(axis=0)
        n = np.linalg.norm(arr)
        if n > 0:
            arr = arr / n
        return arr

    def scores(self, text: str) -> Dict[str, float]:
        """Raw cosine similarity of the utterance to each class centroid.

        Kept for introspection/back-compat. The DECISION uses classify(), which
        applies the exemplar-spread z-score — raw cosines alone are not the
        boundary."""
        store = self._store()
        if not store:
            return {}
        sv = self._sentence_vector(text)
        if sv is None:
            return {}
        _best, _z, raw = self.nearest_prototype(sv, store)
        return raw

    def classify(self, text: str,
                 syntactic_hint: Optional[str] = None) -> Tuple[str, Dict[str, float], str]:
        """Adaptive-margin semantic decision (no hardcoded threshold).

        Returns (act, raw_cosines, method) where method ∈
        {"semantic", "syntactic", "default"}.

          * semantic : nearest-prototype by exemplar-spread z-score decided it
          * syntactic: no usable embedding signal, used the rule-cascade hint
          * default  : nothing fired, fell back to 'question'

        The boundary is the per-class z-score  (cos - mu_c)/sigma_c, a learned
        property of the exemplar set (see nearest_prototype). syntactic_hint is
        used ONLY when there is no embedding signal at all — we no longer defer
        to it on a fixed margin, because that made the classifier a regex
        confirmant (measured: 100% agreement, 0 rescues). With the adaptive
        boundary the semantic path beats the cascade +5/30 (86.7% vs 70%).
        """
        store = self._store()
        sv = self._sentence_vector(text)
        if sv is None or not store:
            if syntactic_hint:
                return (syntactic_hint, {}, "syntactic")
            return ("question", {}, "default")
        best, _z, raw = self.nearest_prototype(sv, store)
        if best is None:
            if syntactic_hint:
                return (syntactic_hint, {}, "syntactic")
            return ("question", {}, "default")
        return (best, raw, "semantic")

    def confidence(self, text: str) -> Tuple[Optional[str], float, float]:
        """N4 seam: return (best_class, z_score, mean_raw_cosine).

        The z-score (cos - mu_c)/sigma_c IS the inverse-surprise / precision
        signal N4 routes on: high z -> confident -> don't perturb; low z ->
        learn; z below the class floor -> abstain (clarify). classify() drops
        z; this method exposes it so the gating controller reuses the exact
        same adaptive boundary instead of recomputing.
        """
        store = self._store()
        sv = self._sentence_vector(text)
        if sv is None or not store:
            return None, 0.0, 0.0
        best, z, raw = self.nearest_prototype(sv, store)
        mean_raw = float(np.mean(list(raw.values()))) if raw else 0.0
        return best, (z.get(best, 0.0) if best else 0.0), mean_raw


class QuestionSubtypeClassifier:
    """Stage-2 of hierarchical (N1) question-type detection.

    Rosch et al. 1976 (basic-level first) + Lambon Ralph 2017 (hub-and-spoke
    coarse->fine) + Friston & Kiebel 2009 (hierarchical inference): the coarse
    question/statement cut (SpeechActClassifier, shipped at 87%) is the basic
    level; the wh-subtype is subordinate and is only resolved INSIDE the
    question branch. This avoids packing 12 pragmatic classes flat into
    non-orthogonal GloVe-64, where near-synonyms (who/what, why/how) collapse —
    the same 64D crowding that regresses HRR here.

    Two empirically-forced design choices (experiments/n1_question_subtype.py):
      * FIRST-TOKEN EMPHASIS (weight ~3x): wh-subtypes are defined by the
        leading function word (what/why/how/compare/...), which plain
        mean-pooling averages away. Weighting the first token lifts cosine
        60% -> 90% (beats the regex cascade's 85%).
      * COSINE, not the z-score used at Stage 1: within-question subtypes share
        stems ("what is" vs "what if"), so per-class z-normalisation amplifies
        their confusion. Raw cosine + first-token emphasis separates them
        (cosine 90% vs z-score 55%). Stage 1 keeps z-score; Stage 2 uses cosine
        — different geometry, measured, not assumed.

    Abstain gate (the N4 surprise seam): cosine decides the argmax, but a
    per-class confidence floor mu_c - k*sigma_c (exemplar spread) can reject
    low-confidence inputs. Measured: k=2.0 -> 100% accuracy on 80% coverage.
    An ABSTAIN routes to the regex fallback now, and will trigger a clarifying
    question / prototype-spawn under N4/N2.
    """

    # Only the SEMANTIC wh-subtypes belong here. Social/structural types
    # (greeting, introduction, analogy, impossible, farewell, ...) stay in the
    # regex cascade — they are genuinely pattern-shaped ("my name is X", "hi",
    # "A:B::C:D") and prototypes buy nothing there.
    SUBTYPE_SEEDS = {
        "what_is": [
            "what is trust", "what are black holes", "what is gravity",
            "what is a neuron", "what are dreams", "what is democracy",
            "what is photosynthesis", "what is inflation", "define entropy",
            "define a concept", "what does it mean",
        ],
        "why": [
            "why does ice melt", "why is the sky blue", "why do birds sing",
            "why are we here", "why does it rain", "why do people lie",
            "why is water wet", "why do stars shine",
        ],
        "how": [
            "how does gravity work", "how do planes fly", "how does a battery work",
            "how do vaccines work", "how does memory work", "how do engines run",
            "how does the heart pump", "how do computers think",
        ],
        "tell_me": [
            "tell me about freedom", "tell me about the ocean",
            "tell me more about jazz", "tell me about ancient rome",
            "tell me about quantum physics", "describe the process",
        ],
        "compare": [
            "compare cats and dogs", "difference between love and lust",
            "what is the difference between weather and climate",
            "compare python and java", "socialism versus capitalism",
            "difference between a virus and bacteria",
        ],
        "hypothetical": [
            "what if the sun disappeared", "what would happen if we stopped sleeping",
            "what happens if you fall into a black hole", "suppose gravity reversed",
            "imagine a world without money", "what if humans could fly",
        ],
        "do_you_know": [
            "do you know about einstein", "have you heard of the beatles",
            "do you know what dna is", "do you know who wrote hamlet",
            "have you heard about the big bang", "do you know any good books",
        ],
    }

    FIRST_TOKEN_WEIGHT = 3.0   # empirically best (sweep peak at 3.0)
    ABSTAIN_K = 2.0            # mu - k*sigma confidence floor; None disables

    _DEFAULT = object()  # sentinel so abstain_k=None can explicitly DISABLE

    def __init__(self, vector_fn=None, first_token_weight: Optional[float] = None,
                 abstain_k=_DEFAULT):
        self.vector_fn = vector_fn
        self.first_token_weight = (first_token_weight
                                   if first_token_weight is not None
                                   else self.FIRST_TOKEN_WEIGHT)
        # abstain_k: omitted -> class default; None -> disabled; float -> that value.
        self.abstain_k = self.ABSTAIN_K if abstain_k is self._DEFAULT else abstain_k
        self.exemplars: Dict[str, List[str]] = {
            c: list(v) for c, v in self.SUBTYPE_SEEDS.items()
        }
        self._cen: Dict[str, np.ndarray] = {}
        self._mu: Dict[str, float] = {}
        self._sigma: Dict[str, float] = {}
        self._fitted = False

    def add_exemplar(self, subtype: str, text: str) -> None:
        """Grow the bank at runtime (learn-by-chatting / N2 spawn seam)."""
        self.exemplars.setdefault(subtype, []).append(text)
        self._fitted = False

    def _sentence_vector(self, text: str) -> Optional[np.ndarray]:
        if self.vector_fn is None:
            return None
        words = re.findall(r"[a-z']+", text.lower())
        vecs = [self.vector_fn(w) for w in words]
        vecs = [v for v in vecs if v is not None]
        if not vecs:
            return None
        M = np.stack(vecs)
        weights = np.ones(len(M))
        if self.first_token_weight != 1.0:
            weights[0] = self.first_token_weight
        arr = (M * weights[:, None]).sum(axis=0) / weights.sum()
        n = np.linalg.norm(arr)
        return arr / n if n > 0 else arr

    def _fit(self) -> None:
        if self._fitted:
            return
        cen, mu, sigma = {}, {}, {}
        for c, phrases in self.exemplars.items():
            vs = [self._sentence_vector(p) for p in phrases]
            vs = [v for v in vs if v is not None]
            if not vs:
                continue
            M = np.stack(vs)
            ce = M.mean(axis=0)
            nn = np.linalg.norm(ce)
            if nn > 0:
                ce = ce / nn
            sims = M @ ce
            cen[c] = ce
            mu[c] = float(sims.mean())
            sigma[c] = float(sims.std()) or 1e-3
        self._cen, self._mu, self._sigma = cen, mu, sigma
        self._fitted = True

    def classify(self, text: str) -> Tuple[Optional[str], Dict[str, float]]:
        """Return (subtype, cosine_scores). subtype is None when there is no
        embedding signal, or "ABSTAIN" when below the confidence floor — both
        signal the caller to fall back to the regex cascade."""
        self._fit()
        sv = self._sentence_vector(text)
        if sv is None or not self._cen:
            return None, {}
        cos = {c: float(np.dot(sv, cen)) for c, cen in self._cen.items()}
        best = max(cos, key=cos.get)
        if self.abstain_k is not None:
            floor = self._mu[best] - self.abstain_k * self._sigma[best]
            if cos[best] < floor:
                return "ABSTAIN", cos
        return best, cos


def _normalize_contractions(text: str) -> str:
    """Collapse common spoken contractions to their canonical written form.

    Brain basis (forward-model speech perception, Pickering & Garrod 2013;
    Pulvermuller 2003): intent is inferred from an incomplete signal, so
    detection must be INVARIANT to orthographic/phonological variation. One
    normalized form covers "what's up" / "what is up" / "wassup" / "whatsup",
    eliminating per-contraction regex sprawl. This is the invariance step the
    old regex cascade lacked.
    """
    t = " " + text.lower().strip() + " "
    # ordered longest-first so "i'm" wins over "i"
    subs = [
        (r"\bi'm\b", "i am"), (r"\bi've\b", "i have"), (r"\bi'll\b", "i will"),
        (r"\bi'd\b", "i would"), (r"\bi've\b", "i have"),
        (r"\byou're\b", "you are"), (r"\byou've\b", "you have"),
        (r"\byou'll\b", "you will"), (r"\by'all\b", "you all"),
        (r"\bwe're\b", "we are"), (r"\bwe've\b", "we have"),
        (r"\bthey're\b", "they are"), (r"\bthey've\b", "they have"),
        (r"\bhe's\b", "he is"), (r"\bshe's\b", "she is"),
        (r"\bwhat's\b", "what is"), (r"\bwhat're\b", "what are"),
        (r"\bwhats\b", "what is"), (r"\bwhatre\b", "what are"),
        (r"\bhow's\b", "how is"), (r"\bhow're\b", "how are"), (r"\bhowre\b", "how are"),
        (r"\bwho's\b", "who is"), (r"\bwhere's\b", "where is"),
        (r"\bwhen's\b", "when is"), (r"\bwhy's\b", "why is"),
        (r"\bthat's\b", "that is"), (r"\bit's\b", "it is"),
        (r"\blet's\b", "let us"), (r"\bwon't\b", "will not"),
        (r"\bcan't\b", "cannot"), (r"\bn't\b", " not"),
        (r"\bwassup\b", "what is up"), (r"\bsup\b", "what is up"),
        (r"\bg'day\b", "good day"),
    ]
    for pat, rep in subs:
        t = re.sub(pat, rep, t)
    return t.strip()


class SocialIntentClassifier(QuestionSubtypeClassifier):
    """Prototype-bank social-intent detector (TPJ mentalizing / DMN self-other
    simulation; Saxe & Kanwisher 2003; Spreng et al. 2009).

    Reuses the exact Stage-2 machinery of QuestionSubtypeClassifier
    (first-token-weighted GloVe sentence vector -> per-class centroid + mu/sigma
    -> cosine argmax with an ABSTAIN_K confidence floor). The social types are
    learned prototypes, so "what's up" / "what is up" / "wassup" all map to the
    SAME greeting centroid after contraction normalization, instead of needing
    three literal regexes. This collapses the two divergent regex copies
    (detect_question_type + _handle_chitchat) into ONE learned classifier and
    gives fail-closed degradation: inputs below the confidence floor ABSTAIN and
    route to the factual path rather than being misrouted or emitting canned text.

    The prototype bank's ABSTAIN_K gate (inherited) is the distribution-driven
    confidence floor -- no fixed pragmatic threshold.
    """

    # Each social type seeded with SURFACE-VARIED exemplars so the centroid
    # spans contractions / paraphrases, not one literal string.
    SOCIAL_SEEDS = {
        "greeting": [
            "hi", "hello", "hey", "yo", "what is up", "how is it going",
            "howdy", "good morning", "good afternoon",
            "good evening", "sup", "hiya", "greetings",
        ],
        "wellbeing": [
            "how are you", "how are you doing", "how is it going",
            "how have you been", "what is going on", "you good",
            "how are things", "are you okay",
        ],
        "capability": [
            "what can you do", "who are you", "tell me about yourself",
            "what are you", "what is your deal", "what are you capable of",
        ],
        "farewell": [
            "bye", "goodbye", "see you", "good night", "farewell",
            "catch you later", "talk to you later",
        ],
        "gratitude": [
            "thanks", "thank you", "cheers", "appreciate it", "grateful",
            "thanks a lot", "thank you so much",
        ],
        "affect_disclosure": [
            "i am bored", "i am sad", "i feel tired", "i am lonely",
            "i am frustrated", "i am happy", "i feel anxious", "i am excited",
        ],
    }

    def __init__(self, vector_fn=None, first_token_weight=None, abstain_k=None):
        super().__init__(vector_fn=vector_fn,
                         first_token_weight=first_token_weight,
                         abstain_k=(abstain_k if abstain_k is not None
                                    else self.ABSTAIN_K))
        # Re-fit the prototype bank on the SOCIAL seeds. The parent constructor
        # fitted on SUBTYPE_SEEDS (semantic question types); without overriding
        # the exemplars the classifier would score against what_is/how and never
        # see the greeting / wellbeing centroids.
        self.exemplars = {c: list(v) for c, v in self.SOCIAL_SEEDS.items()}
        self._fitted = False
        self._fit()

    # Function words carry the speech-act FRAME but not the TYPE; matching on
    # them alone makes "what is the moon" a false-positive greeting. We match on
    # CONTENT words so an utterance's content must be a special case of a seed's
    # content (Suls 1972 cheap prediction + Koestler bisociation: the act is
    # recognized when its content slots are filled by the seed's idiom).
    _SOCIAL_STOP = {
        "what", "is", "are", "am", "was", "were", "how", "do", "does", "did",
        "can", "could", "would", "should", "will", "shall", "you", "i", "we",
        "me", "my", "our", "the", "a", "an", "to", "of", "for", "in", "on",
        "at", "it", "this", "that", "be", "been", "being", "have", "has", "had",
        "not", "no", "so", "if", "as", "from", "with", "about", "your", "s",
        "t", "re", "ve", "ll", "d", "m",
    }

    def detect(self, text: str):
        norm = _normalize_contractions(text)
        toks = set(re.findall(r"[a-z']+", norm))
        if toks:
            # Lexical forward-model (Pickering & Garrod 2013): intent is a
            # prediction of the speaker's act. Social acts are closed-class
            # idioms, so a deterministic, contraction-normalized content-subset
            # match against seeded exemplars is the reliable detector. One
            # normalized form covers what's up / what is up / wassup.
            utt_content = toks - self._SOCIAL_STOP
            if utt_content:
                best_type, best_cov = "ABSTAIN", 0.0
                for stype, seeds in self.SOCIAL_SEEDS.items():
                    cov = 0.0
                    for seed in seeds:
                        seed_toks = set(re.findall(r"[a-z']+", seed))
                        seed_content = seed_toks - self._SOCIAL_STOP
                        # Coverage = fraction of the utterance's content that the
                        # seed explains. This handles multi-word acts (i feel
                        # lonely) that span seeds, and rejects false frames
                        # (what is the moon) whose content isn't in any seed.
                        pool = seed_content if seed_content else seed_toks
                        if not pool:
                            continue
                        inter = len(utt_content & pool)
                        if inter == 0:
                            continue
                        cov = max(cov, inter / len(utt_content))
                    if cov > best_cov:
                        best_cov, best_type = cov, stype
                if best_cov >= 0.5:
                    return best_type, {}
            elif toks:
                # Pure function-word idiom (e.g. "how are you"): match the full
                # token set so the idiom is recognized despite no content words.
                best_type, best_cov = "ABSTAIN", 0.0
                for stype, seeds in self.SOCIAL_SEEDS.items():
                    cov = 0.0
                    for seed in seeds:
                        seed_toks = set(re.findall(r"[a-z']+", seed))
                        if not seed_toks:
                            continue
                        inter = len(toks & seed_toks)
                        if inter == 0:
                            continue
                        cov = max(cov, inter / len(toks))
                    if cov > best_cov:
                        best_cov, best_type = cov, stype
                if best_cov >= 0.5:
                    return best_type, {}
        # Fallback to the vector prototype (content-bearing social acts that DO
        # have embeddings); abstain on noise.
        label, scores = self.classify(norm)
        if label is None or label == "ABSTAIN":
            return ("ABSTAIN", scores)
        return label, scores


class SurpriseGate:
    """N4 — surprise gating: the governor that makes N2 (emergent categories) safe.

    Feldman & Friston 2010 (precision-weighted plasticity) + Friston 2017
    (active inference / epistemic action): prediction error (surprise) is the
    learning signal. This gate routes each utterance into one of three regimes
    from a classifier's confidence, so prototypes are perturbed ONLY on high
    error — preventing the two failure modes of an un-gated learner:
      * HYPER-stability: never updating => can't learn new phrasings.
      * HYPER-plasticity: updating on everything => centroid DRIFT and
        catastrophic interference / class explosion.

    Regimes:
      HIGH    (confident)  -> reinforce only (rehearsal); do NOT perturb centroid.
      LEARN   (low error, known class) -> precision-weighted bounded update.
      ABSTAIN (novel / high error) -> epistemic action: ask to clarify, or
                                    (N2) spawn a candidate prototype in the
                                    hippocampal buffer. Never silently fold into
                                    an existing class.

    BAND IS NOT A CONSTANT. It is derived from the classifier's own score
    distribution (experiments/n4_surprise_gating.py): known utterances occupy a
    tight band (e.g. z in [-2.4, +1.5]) and genuine novelty a far tail (z < -8).
    A running percentile of recently-seen scores splits them — so the boundary
    tracks the live distribution, like a learned threshold, not a magic number.
    """

    # Percentile of the live score distribution used as the confident/learn split.
    CONFIDENT_PERCENTILE = 50.0   # top half of live scores => confident
    # Novelty tail: scores below this many of the live min are hard ABSTAIN.
    # (Fallback gap; the live band is computed from observed scores.)
    NOVELTY_GAP_FRAC = 0.5        # floor sits mid-way between known-min and novel-min

    def __init__(self, window: int = 200):
        self._scores: List[float] = []   # rolling history of observed scores
        self.window = window

    def observe(self, score: float) -> None:
        """Feed a freshly-computed confidence score into the rolling window."""
        if score is None:
            return
        self._scores.append(score)
        if len(self._scores) > self.window:
            self._scores.pop(0)

    def _bands(self) -> Tuple[float, float]:
        """Compute (confident_z, learn_floor) from the live score history.

        confident_z = percentile(CONFIDENT_PERCENTILE) of known-ish scores.
        learn_floor  = mid-gap between the low tail and the confident band, so
                       genuinely novel scores (far below) become ABSTAIN.
        With no/short history we fall back to the classifier's own semantics:
        a single neutral split that still routes extreme-negatives to ABSTAIN.
        """
        if len(self._scores) < 8:
            # Not enough history: use a conservative split around 0; rely on the
            # caller also passing an explicit novelty floor for hard-ABSTAIN.
            return 0.0, -3.0
        arr = np.array(self._scores)
        confident_z = float(np.percentile(arr, self.CONFIDENT_PERCENTILE))
        low = float(arr.min())
        # learn_floor midway between the low tail and the confident band
        learn_floor = (low + confident_z) / 2.0
        return confident_z, learn_floor

    def route(self, score: float) -> str:
        """Map a confidence score to a regime. score may be a z-score (Stage 1)
        or a raw cosine (Stage 2) — the band adapts to whichever distribution is
        fed via observe(). Returns 'HIGH' | 'LEARN' | 'ABSTAIN'."""
        self.observe(score)
        confident_z, learn_floor = self._bands()
        if score is None or score < learn_floor:
            return "ABSTAIN"
        if score >= confident_z:
            return "HIGH"
        return "LEARN"

    def route_stage1(self, sac: "SpeechActClassifier", text: str) -> Tuple[str, str, float]:
        """Convenience: route using the Stage-1 z-margin signal directly."""
        best, z, _raw = sac.confidence(text)
        regime = self.route(z)
        return best or "question", regime, z

    def route_stage2(self, qsc: "QuestionSubtypeClassifier", text: str) -> Tuple[str, str, float]:
        """Route using the Stage-2 cosine (with the classifier's abstain floor)."""
        sub, cos = qsc.classify(text)
        if sub == "ABSTAIN" or sub is None:
            return "ABSTAIN", "ABSTAIN", 0.0
        best_cos = max(cos.values())
        regime = self.route(best_cos)
        return sub, regime, best_cos


class EmergentCategoryLearner:
    """N2 — emergent categories: RAVANA learns a NEW intent by chatting.

    The hippocampal-VTA novelty loop (Lisman & Grace 2005): when SurpriseGate
    routes an utterance to ABSTAIN (high prediction error = "unlike anything I
    know"), spawn a CANDIDATE prototype. The spawn trigger is the gate's output
    — a prediction-error quantity, NOT a constant threshold (this is exactly why
    N4 must precede N2).

    Without consolidation this would explode into classes / suffer catastrophic
    interference. The brain avoids it via fast hippocampal encoding + SLOW
    neocortical consolidation (McClelland et al. 1995 CLS): a candidate stays
    ephemeral in the hippocampal buffer until rehearsed/confirmed, then is
    promoted to a stable prototype. We use the REAL HippocampalBuffer as the
    fast store (it already tracks confidence + rehearsal_count and exposes
    get_consolidation_candidates). Merge collapses near-duplicate candidates;
    sleep prunes un-rehearsed singletons.

    KNOWN LIMITATION (measured in experiments/n2_emergent_categories.py): in
    64D, mean-pooled intent centroids of different intents sit close (e.g.
    "remind me" vs "tell me a joke" = 0.90 cosine), so MERGE is unreliable for
    loose paraphrases — the same crowding that bounded A0/HRR. Merge radius is
    therefore conservative (0.95): it collapses near-identical phrasings, not
    semantic neighbours. Robust merge needs A0's high-D HRR space. Spawn/prune/
    consolidate (the explosion guard) are unaffected and validated.
    """

    MERGE_RADIUS = 0.95       # conservative: only near-identical phrasings merge
    CONSOLIDATE_REHEARSALS = 2
    CONSOLIDATE_CONFIDENCE = 0.7

    def __init__(self, vector_fn=None, hippocampal_buffer=None, merge_radius=None):
        self.vector_fn = vector_fn
        self.merge_radius = merge_radius or self.MERGE_RADIUS
        # Fast, ephemeral candidate store. Real HippocampalBuffer when supplied;
        # otherwise a lightweight in-memory map (still vector-native so merge works).
        self.hb = hippocampal_buffer
        self._candidates: Dict[str, dict] = {}
        self._next_id = 0
        self.pruned_count = 0
        self.promoted_count = 0

    def _sentence_vector(self, text: str) -> Optional[np.ndarray]:
        if self.vector_fn is None:
            return None
        words = re.findall(r"[a-z']+", text.lower())
        vecs = [self.vector_fn(w) for w in words]
        vecs = [v for v in vecs if v is not None]
        if not vecs:
            return None
        a = np.stack(vecs).mean(axis=0)
        n = np.linalg.norm(a)
        return a / n if n > 0 else a

    def _new_id(self) -> str:
        cid = f"emergent_{self._next_id}"
        self._next_id += 1
        return cid

    def learn_novel(self, text: str) -> str:
        """Spawn-or-merge a candidate category from an ABSTAIN utterance.

        Returns the candidate id. Caller MUST have already confirmed the gate
        routed this utterance to ABSTAIN (N4 governs spawning).
        """
        vec = self._sentence_vector(text)
        if vec is None:
            return ""
        # merge into nearest existing candidate within radius
        best_id, best_sim = None, -1.0
        for cid, c in self._candidates.items():
            sim = float(np.dot(vec, c["vec"]))
            if sim > best_sim:
                best_sim, best_id = sim, cid
        if best_id is not None and best_sim >= self.merge_radius:
            c = self._candidates[best_id]
            c["exemplars"].append(text)
            n = len(c["exemplars"])
            c["vec"] = (c["vec"] * (n - 1) + vec) / n
            c["rehearsal"] = c.get("rehearsal", 0) + 1
            c["confidence"] = min(1.0, c.get("confidence", 0.5) + 0.1)
            return best_id
        # spawn new candidate (fast hippocampal encoding)
        cid = self._new_id()
        self._candidates[cid] = {
            "vec": vec, "exemplars": [text],
            "rehearsal": 1, "confidence": 0.5, "stable": False,
        }
        if self.hb is not None:
            self.hb.store(subject=cid, predicate="candidate_intent",
                          object=text, confidence=0.5,
                          aliases=[w for w in text.lower().split() if len(w) >= 2])
        return cid

    def reinforce(self, cid: str) -> None:
        """Rehearsal: bump confidence + rehearsal (drives consolidation gate)."""
        c = self._candidates.get(cid)
        if c:
            c["rehearsal"] = c.get("rehearsal", 0) + 1
            c["confidence"] = min(1.0, c.get("confidence", 0.5) + 0.1)

    def sleep_consolidate(self) -> Dict[str, int]:
        """NREM/REM analogue: promote rehearsed candidates to stable, prune the
        rest (forgetting). Mirrors McClelland 1995 slow consolidation."""
        promoted = pruned = 0
        survivors = {}
        for cid, c in self._candidates.items():
            if (c.get("rehearsal", 0) >= self.CONSOLIDATE_REHEARSALS
                    and c.get("confidence", 0) >= self.CONSOLIDATE_CONFIDENCE):
                c["stable"] = True
                survivors[cid] = c
                promoted += 1
            else:
                pruned += 1
        self._candidates = survivors
        self.promoted_count += promoted
        self.pruned_count += pruned
        return {"promoted": promoted, "pruned": pruned,
                "stable": len(survivors)}

    def candidate_ids(self) -> List[str]:
        return list(self._candidates.keys())

    def stable_ids(self) -> List[str]:
        return [cid for cid, c in self._candidates.items() if c.get("stable")]


class PrefrontalWorkspace:
    """Structured buffer that plans discourse BEFORE chain generation.

    Capacity: 5 (teen mode) or 7 (adult mode).
    For each turn, builds a discourse plan with 3 sentence intents
    that form a coherent narrative arc.
    """

    def __init__(self, capacity: int = 5, analogy_engine=None, abstraction_engine=None,
                 vector_fn=None):
        self.capacity = capacity  # teen capacity (adults = 7)
        self.last_plan: Optional[DiscoursePlan] = None
        self.topic_history: List[str] = []  # last 10 topics discussed
        # Optional cognitive engines for advanced reasoning
        self.analogy_engine = analogy_engine
        self.abstraction_engine = abstraction_engine
        # Semantic space supplier (word -> unit vector) for the speech-act
        # classifier. When provided, illocutionary-force detection is hybrid
        # (prototype semantics + symbolic syntax) instead of pure rules.
        self.vector_fn = vector_fn
        self._sac: Optional[SpeechActClassifier] = None
        self._qsc: Optional[QuestionSubtypeClassifier] = None
        self._gate: Optional[SurpriseGate] = None
        self._n2: Optional[EmergentCategoryLearner] = None

    # ─── Question Type Detection ───

    QUESTION_PATTERNS = {
        # Paradox / impossibility is tested FIRST (before hypothetical/why).
        # The brain's paradox network (rIFG/BA47 contradiction detection +
        # ACC conflict) flags an impossible scenario before it is simulated
        # as a literal "what if". Without this, "unstoppable force meets
        # immovable object" is misrouted to the hypothetical handler.
        "impossible": [
            re.compile(r"can\s+god\s+create", re.IGNORECASE),
            re.compile(r"unstoppable\s+force", re.IGNORECASE),
            re.compile(r"irresistible\s+force", re.IGNORECASE),
            re.compile(r"immovable\s+object", re.IGNORECASE),
            re.compile(r"what\s+is\s+the\s+sound", re.IGNORECASE),
            re.compile(r"one\s+hand\s+clapping", re.IGNORECASE),
            re.compile(r"can\s+you\s+prove", re.IGNORECASE),
            re.compile(r"can\s+we\s+know", re.IGNORECASE),
            re.compile(r"is\s+reality\s+real", re.IGNORECASE),
            re.compile(r"unstoppable\s+force\s+meets?\s+immovable", re.IGNORECASE),
            re.compile(r"immovable\s+object\s+meets?\s+unstoppable", re.IGNORECASE),
        ],
        "what_is": [
            re.compile(r"what\s+is\s+(.+)", re.IGNORECASE),
            re.compile(r"what\s+are\s+(.+)", re.IGNORECASE),
            re.compile(r"what's\s+(.+)", re.IGNORECASE),
        ],
        "why": [
            re.compile(r"why\s+(?:is|are|does|do|can)\s+(.+)", re.IGNORECASE),
            re.compile(r"why\s+(.+)", re.IGNORECASE),
        ],
        "how": [
            re.compile(r"how\s+(?:is|are|does|do|can)\s+(.+)", re.IGNORECASE),
        ],
        "tell_me": [
            re.compile(r"tell\s+me\s+about\s+(.+)", re.IGNORECASE),
            re.compile(r"tell\s+me\s+more\s+about\s+(.+)", re.IGNORECASE),
        ],
        "compare": [
            re.compile(r"(?:compare|difference|contrast|versus|vs)\s+(?:the\s+)?(?:between\s+)?(.+?)\s+(?:and|vs|versus|with|to)\s+(.+)", re.IGNORECASE),
            re.compile(r"what\s*(?:is\s+|'s\s+)?the\s+difference\s+between\s+(.+?)\s+and\s+(.+)", re.IGNORECASE),
        ],
        "hypothetical": [
            re.compile(r"what\s+if\s+(.+)", re.IGNORECASE),
            re.compile(r"what\s+happens\s+if\s+(.+)", re.IGNORECASE),
            re.compile(r"what\s+would\s+happen\s+if\s+(.+)", re.IGNORECASE),
            re.compile(r"what\s+happens\s+when\s+(.+)", re.IGNORECASE),
            re.compile(r"what\s+would\s+happen\s+when\s+(.+)", re.IGNORECASE),
            re.compile(r"suppose\s+(.+?)(?:\?|$)", re.IGNORECASE),
            re.compile(r"imagine\s+(.+?)(?:\?|$)", re.IGNORECASE),
            re.compile(r"if\s+(.+?)(?:,?\s*what|,what|what)\s+(?:would|will|does|happens?)", re.IGNORECASE),
        ],
        "do_you_know": [
            re.compile(r"do\s+you\s+know\s+(.+)", re.IGNORECASE),
            re.compile(r"have\s+you\s+heard\s+of\s+(.+)", re.IGNORECASE),
        ],
        "follow_up": [
            re.compile(r"(?:more|else|another|also|further|tell me more)", re.IGNORECASE),
        ],
        "greeting": [
            re.compile(r"\b(hi|hello|hey|yo|sup|greetings|whats\s*up|what's\s*up|whatsup|howdy|good\s*morning|good\s*afternoon|good\s*evening)\b", re.IGNORECASE),
        ],
        "wellbeing": [
            re.compile(r"\b(how\s*are\s*you|how're\s*you|how\s*is\s*it\s*going|how's\s*it\s*going|how\s*are\s*you\s*doing|how\s*have\s*you\s*been|hows\s*it\s*going|hows\s*life)\b", re.IGNORECASE),
        ],
        "capability": [
            re.compile(r"\b(what\s*can\s*you\s*do|what\s*do\s*you\s*do|how\s*do\s*you\s*work|tell\s*me\s*about\s*yourself|who\s*are\s*you|what\s*is\s*your\s*name)\b", re.IGNORECASE),
        ],
        "introduction": [
            re.compile(r"\bmy\s+name\s+is\s+(.+)", re.IGNORECASE),
            re.compile(r"\bi\s+am\s+called\s+(.+)", re.IGNORECASE),
            re.compile(r"\bi\s+am\s+(.+)", re.IGNORECASE),
            re.compile(r"\bi'm\s+(.+)", re.IGNORECASE),
            re.compile(r"\bcall\s+me\s+(.+)", re.IGNORECASE),
        ],
        "farewell": [
            re.compile(r"\b(bye|goodbye|see\s*you|good\s*night|farewell)\b", re.IGNORECASE),
        ],
        "analogy": [
            re.compile(r"(.+?)\s*:\s*(.+?)\s*::\s*(.+?)\s*:\s*(.+)", re.IGNORECASE),
            re.compile(r"(.+?)\s+is\s+to\s+(.+?)\s+as\s+(.+?)\s+is\s+to\s+(.+)", re.IGNORECASE),
            re.compile(r"(.+?)\s+relates?\s+to\s+(.+?)\s+like\s+(.+?)\s+relates?\s+to\s+(.+)", re.IGNORECASE),
            re.compile(r"what\s+is\s+the\s+analogy\s+(?:of|for)\s+(.+?)\s+(?:and|to)\s+(.+)", re.IGNORECASE),
            re.compile(r"(.+?)\s*:\s*(.+?)\s*::\s*(.+?)\s*:\s*(.+)", re.IGNORECASE),
        ],
    }

    # Discourse markers with empty-string padding for non-use.
    # These are minimal transition cues, not templates.
    DISCOURSE_MARKERS = {
        "elaborate": ["also", "furthermore", "in addition", "moreover", "besides"],
        "contrast": ["however", "but", "on the other hand", "yet", "although"],
        "connect": ["similarly", "likewise", "in the same way", "correspondingly"],
        "conclude": ["ultimately", "in essence", "at its core", "fundamentally"],
    }

    # Maps each detected question type to the primary relation the PFC will
    # use for task-set biasing during activation spread. Mirrors the dispatch
    # logic in plan_discourse — the PFC's architectural decision about what
    # kind of reasoning the current question requires.
    QTYPE_PRIMARY_RELATION = {
        "what_is": "semantic",
        "why": "causal",
        "how": "causal",
        "tell_me": "semantic",
        "compare": "contrastive",
        "hypothetical": "causal",
        "impossible": "causal",
        "do_you_know": "semantic",
        "follow_up": "semantic",
        "general": "semantic",
        "analogy": "analogical",
        "statement": "semantic",
    }

    @classmethod
    def get_primary_relation_for_qtype(cls, qtype: str) -> str:
        """Return the PFC's task-set relation for a given question type."""
        return cls.QTYPE_PRIMARY_RELATION.get(qtype, "semantic")

    def detect_question_type(self, text: str, concept_pos: Optional[Dict[str, str]] = None) -> Tuple[str, List[str]]:
        """Hierarchical (N1) question-type detection.

        Stage 1 (coarse) + Stage 2 (subtype), Rosch basic-level-first:
          1. The regex cascade runs first — it owns SOCIAL/STRUCTURAL types
             (greeting, wellbeing, capability, introduction, farewell, analogy,
             follow_up, impossible/paradox), which are genuinely pattern-shaped,
             and it ALSO does span extraction (the returned parts).
          2. When a semantic space is wired (self.vector_fn) and the regex landed
             on a SEMANTIC wh-subtype or on 'general' (a miss), the Stage-2
             prototype bank (QuestionSubtypeClassifier) refines the label. It
             beats the regex cascade on subtype (measured 90% vs 85%) and rescues
             misses like "socialism versus capitalism" -> compare.
          3. Prototype ABSTAIN (low confidence, the N4 surprise seam) or no
             embedding signal -> fall back to the regex label. Extraction parts
             always come from regex (callers also have their own fallback).

        Falls back entirely to regex when no vector_fn is present (keeps the
        classmethod-era behaviour for embedding-less callers/tests).

        Returns:
            (question_type, extracted_parts)
        """
        regex_qtype, groups = self._detect_question_type_regex(text, concept_pos)

        if getattr(self, "vector_fn", None) is None:
            return (regex_qtype, groups)

        # Stage 2 only refines the semantic subtypes it models (+ 'general'
        # misses). Social/structural/paradox regex results are authoritative.
        proto_eligible = set(QuestionSubtypeClassifier.SUBTYPE_SEEDS) | {"general"}
        if regex_qtype not in proto_eligible:
            return (regex_qtype, groups)

        if getattr(self, "_qsc", None) is None:
            self._qsc = QuestionSubtypeClassifier(vector_fn=self.vector_fn)
        sub, _scores = self._qsc.classify(text)
        if sub is None or sub == "ABSTAIN":
            return (regex_qtype, groups)
        return (sub, groups)

    # ─── N4→N2: surprise-gated emergent learning ───

    def learn_from_turn(self, text: str) -> Tuple[str, str, str]:
        """N4→N2: route one user utterance through the surprise gate and, on
        ABSTAIN, spawn a candidate category. Returns (act, regime, candidate_id).

        This is the single seam where the unified layer actually LEARNS a new
        intent by chatting. The flow:
          1. SpeechActClassifier.confidence() -> Stage-1 z-margin signal.
          2. SurpriseGate routes it: HIGH (reinforce), LEARN (precision-weighted
             update), or ABSTAIN (novel / high prediction error).
          3. On ABSTAIN, EmergentCategoryLearner spawns/merges a candidate
             prototype in the fast (hippocampal) store — the VTA novelty loop.
             The spawn trigger is the gate's prediction-error output, never a
             constant. Stable promotion happens later via sleep_consolidate().

        Without a semantic space (no vector_fn) this is a no-op returning the
        rule-cascade hint.
        """
        if self.vector_fn is None:
            from ravana.language.prefrontal_workspace import PrefrontalWorkspace as _P
            return (_P.classify_speech_act_rules(text), "n/a", "")
        if self._sac is None:
            self._sac = SpeechActClassifier(vector_fn=self.vector_fn)
        if self._gate is None:
            self._gate = SurpriseGate()
        if self._n2 is None:
            from ravana.core.hippocampal_buffer import HippocampalBuffer
            self._n2 = EmergentCategoryLearner(
                vector_fn=self.vector_fn, hippocampal_buffer=HippocampalBuffer())
        # Strongest novelty signal: an utterance with ZERO known semantic atoms.
        # A fully-OOV phrase IS "unlike anything I know" — the truest prediction
        # error. The gate's z-margin is undefined here (no embeddings), so we
        # route to ABSTAIN directly and spawn. (This is the real novel-intent
        # case; relying on the gate alone misses it because the glove fallback
        # would fabricate a plausible-looking vector.)
        vec = self._n2._sentence_vector(text)
        if vec is None:
            candidate_id = self._n2.learn_novel(text)
            return (self._sac.classify_speech_act_rules(text) if hasattr(self._sac, "classify_speech_act_rules") else "statement", "ABSTAIN", candidate_id)
        act, regime, z = self._gate.route_stage1(self._sac, text)
        candidate_id = ""
        if regime == "ABSTAIN":
            candidate_id = self._n2.learn_novel(text)
        return (act, regime, candidate_id)

    def sleep(self) -> Dict[str, int]:
        """N2 consolidation: promote rehearsed candidates, prune singletons."""
        if self._n2 is None:
            return {"promoted": 0, "pruned": 0, "stable": 0}
        return self._n2.sleep_consolidate()

    # ─── Remaining symbolic priors (unchanged) ───

    @classmethod
    def _detect_question_type_regex(cls, text: str, concept_pos: Optional[Dict[str, str]] = None) -> Tuple[str, List[str]]:
        """The symbolic regex cascade (Stage-1 fallback + span extraction).

        Retained as the interpretable 'core' and the source of extracted parts.
        Stage-2 prototypes refine its SEMANTIC subtype output when embeddings
        are available; social/structural types are resolved here only.
        """
        text_lower = text.lower().strip(" ?!.")

        # Check social/chitchat types first to prevent general pattern hijacking
        social_types = ["greeting", "wellbeing", "capability", "introduction", "farewell"]
        for qtype in social_types:
            if qtype in cls.QUESTION_PATTERNS:
                for pattern in cls.QUESTION_PATTERNS[qtype]:
                    m = pattern.match(text_lower)
                    if m:
                        groups = [g.strip() for g in m.groups() if g]
                        # Extra validation for introduction to avoid state adjectives
                        if qtype == "introduction":
                            name_candidate = groups[0].lower() if groups else ""
                            state_words = {
                                "happy", "sad", "tired", "thinking", "learning", "busy", "ready", "sure", 
                                "fine", "good", "well", "hungry", "sick", "bored", "excited", "doing", 
                                "going", "coming", "trying", "working", "studying", "making", "having"
                            }
                            is_state_or_action = False
                            if name_candidate and concept_pos:
                                pos = concept_pos.get(name_candidate)
                                if pos in ("adj", "verb", "adverb"):
                                    is_state_or_action = True
                            if name_candidate in state_words or is_state_or_action or not name_candidate:
                                continue
                        return (qtype, groups)

        for qtype, patterns in cls.QUESTION_PATTERNS.items():
            if qtype in social_types:
                continue
            # Compare/contrast must be tested BEFORE the greedy 'what_is' pattern
            # (which is "what\s+is\s+(.+)" and would swallow "what is the
            # difference between A and B" as a plain what_is, surfacing only the
            # garbled payload "the difference between privacy and security").
            if qtype == "what_is":
                # Test compare first if present.
                for cpat in cls.QUESTION_PATTERNS.get("compare", []):
                    m = cpat.match(text_lower)
                    if m:
                        groups = [g.strip() for g in m.groups() if g]
                        if groups:
                            return ("compare", groups)
            for pattern in patterns:
                m = pattern.match(text_lower)
                if m:
                    groups = [g.strip() for g in m.groups() if g]
                    return (qtype, groups)

        # Default: treat as general statement/query
        return ("general", [text_lower])

    @classmethod
    def classify_speech_act_rules(cls, text: str) -> str:
        """Symbolic prior: the transparent syntactic rule cascade (the cheap,
        interpretable 'core' of each speech-act class).

        Neuroscience basis — how the brain tells "telling" from "asking"
        -------------------------------------------------------------------
        The human brain does not need an explicit '?' to know a sentence is a
        question. Pragmatic / illocutionary-force decoding (Austin's speech
        acts; Searle) is distributed and happens EARLY (~100 ms, in parallel
        with semantics):

          * Syntax — left IFG (BA 45-47) + posterior temporal cortex parse
            form. Subject–auxiliary INVERSION ("are you…", "do they…") and
            sentence-initial wh- words (what/why/how…) reliably flag an
            interrogative. Declarative word order ("i am…", "nothing…") flags
            an assertion.
          * Prosody — superior temporal / auditory cortex reads the terminal
            pitch contour: rising = question, falling = statement (the cue
            Italian relies on almost exclusively). We approximate it with a
            trailing '?'.
          * Intent (theory-of-mind) — the right TPJ + medial PFC decode the
            speaker's communicative goal: requesting info vs stating a fact.
            Lesions here impair exactly this distinction.

        Rule cascade:
          1. explicit '?' or wh-/aux-inversion → question
          2. first-/third-person declarative markers → assertion
          3. otherwise default to question (preserve prior query behaviour).
        """
        t = text.lower().strip(" ?!.,")
        if not t:
            return "question"

        # 1) Explicit interrogatives (rising pitch / '?')
        if text.strip().endswith("?"):
            return "question"
        # wh-word at sentence start
        if re.match(r"^(what|who|whom|whose|where|when|why|which|how)\b", t):
            return "question"
        # subject–auxiliary inversion: aux then (pro)nominal ("are you", "can i")
        _invert = (r"^(am|is|are|was|were|do|does|did|have|has|had|can|could|"
                   r"would|will|shall|should|may|might|must)\b")
        if re.match(_invert, t):
            return "question"
        # explicit info/request verbs
        if re.match(r"^(tell me|explain|define|describe|show me|give me|list|recite)\b", t):
            return "question"

        # 2) Declarative / first-person assertions (the user is TELLING us)
        _first = (r"^(i|i'm|im|i am|we|we're|we are|my|me)\b")
        if re.match(_first, t):
            return "statement"
        _third = (r"^(he|she|they|it|you|this|that|the|my|his|her|their)\b")
        # third-person declarative WITH a copula / verb
        if re.match(_third, t) and re.search(
                r"\b(is|are|was|were|has|have|had|went|goes|likes|loves|hates|"
                r"did|does|will|would|can|should|made|makes|thinks|feels|said|says)\b", t):
            return "statement"

        # social assertion markers (gratitude, meeting, mood, backchannels)
        if re.search(r"\b(nice to meet you|good to (see|meet)|thanks|thank you|"
                     r"cheers|appreciate|pleasure to meet)\b", t):
            return "statement"
        if re.match(r"^(nothing|not much|same old|nada|ditto)\b", t):
            return "statement"
        if re.match(r"^(no|yes|yeah|yep|yup|nope|sure|ok|okay|alright|right)\b", t):
            return "statement"

        # 3) Default: preserve previous behaviour — treat as a question/query
        return "question"

    def classify_speech_act(self, text: str) -> str:
        """Hybrid illocutionary-force classifier (semantic prototypes + symbolic
        syntax). Returns 'question' or 'statement'.

        When a semantic space is wired (`self.vector_fn`), the decision is made
        by proximity to per-class PROTOTYPE vectors in that space
        (`SpeechActClassifier`), refined by the symbolic rule cascade as a
        prior. This is robust to wording/paraphrase and extensible, while the
        symbolic rules anchor the confident "core" cases (Lyutikova hybrid).
        Falls back entirely to the rule cascade when no vectors are available.
        """
        hint = self.classify_speech_act_rules(text)
        if self.vector_fn is None:
            return hint
        if self._sac is None:
            self._sac = SpeechActClassifier(vector_fn=self.vector_fn, dim=getattr(self, "capacity", 64) or 64)
        # Plan Stage 1 (M-F immediate win): the brain decodes interrogative
        # SYNTAX early (~100 ms, left IFG + posterior temporal) and that signal
        # should win over the slower semantic prototype when it is unambiguous.
        # The symbolic cascade is confident on a question when the text carries
        # an explicit '?', a sentence-initial wh- word, or subject–auxiliary
        # inversion ("do you…", "are we…"). In those cases the semantic
        # prototype (which can be fooled by a word like "tired" sitting near the
        # statement prototype) must NOT override a clear question into a
        # statement — doing so is what produced "yeah, ever tired" for
        # "do you ever get tired". So: a confident syntactic QUESTION is
        # returned as-is; the semantic path is only trusted when the cascade is
        # itself unsure (default 'question' on weak input) or when it agrees.
        _confident_question = bool(
            text.strip().endswith("?")
            or re.match(r"^(what|who|whom|whose|where|when|why|which|how)\b",
                         (text or "").lower().strip(" ?!.,"))
            or re.match(r"^(am|is|are|was|were|do|does|did|have|has|had|can|"
                        r"could|would|will|shall|should|may|might|must)\b",
                        (text or "").lower().strip(" ?!.,")))
        if _confident_question and hint == "question":
            # Semantic agreement -> fine; semantic disagreement on a confident
            # syntactic question -> trust syntax (fail-closed to question).
            act, _scores, _method = self._sac.classify(text, syntactic_hint=hint)
            if act == "question":
                return "question"
            # Semantic says 'statement' but syntax is unambiguously interrogative:
            # prefer the early, high-confidence syntactic read.
            return "question"
        act, _scores, _method = self._sac.classify(text, syntactic_hint=hint)
        return act

    @classmethod
    def detect_concept_drift(cls, current_topic: str, next_hop: str,
                              vector_fn=None) -> float:
        """Detect if a graph walk has drifted to an unrelated concept.
        
        Returns a drift score 0.0 (on-topic) to 1.0 (completely unrelated).
        When drift > 0.6, the PFC should intervene (step back or insert transition).
        
        Uses vector similarity between consecutive hops.
        """
        if not vector_fn or not current_topic or not next_hop:
            return 0.0
        try:
            v1 = vector_fn(current_topic)
            v2 = vector_fn(next_hop)
            if v1 is not None and v2 is not None:
                import numpy as np
                sim = float(np.dot(v1, v2))
                # sim ranges from -1 to 1. Drift = 1 - normalized_similarity
                drift = 1.0 - max(0.0, (sim + 1.0) / 2.0)
                return drift
        except Exception:
            pass
        return 0.0

    # ─── Discourse Planning ───

    def plan_discourse(self, user_input: str, subject: str,
                       concept_pos: Dict[str, str],
                       associations: List[Tuple[str, float]],
                       past_topics: Optional[List[str]] = None,
                       is_follow_up: bool = False) -> DiscoursePlan:
        """Plan a multi-sentence discourse response.

        Analyzes the user's question, then builds a coherent 3-sentence plan.

        Args:
            user_input: Raw user text
            subject: Extracted topic/subject
            concept_pos: Part-of-speech map for concepts
            associations: Spread activation results from graph
            past_topics: Previously discussed topics
            is_follow_up: Whether this is a follow-up query

        Returns:
            DiscoursePlan with 3 DiscourseIntents
        """
        qtype, parts = self.detect_question_type(user_input, concept_pos=concept_pos)
        plan = DiscoursePlan(original_subject=subject, question_type=qtype)

        # Build seen set from subject only. Do NOT pre-populate with top
        # associations — that would consume the most causally-relevant concepts
        # (e.g. "explosion" for "what happens if lamp?") before the
        # task-set-aware selection can evaluate them.
        seen = {subject.lower()}

        # Plan based on question type
        if qtype == "what_is":
            # Check if subject is abstract → use multi-perspective planning
            if self._is_abstract_concept(subject):
                plan = self._plan_abstract(subject, associations, seen, qtype)
            else:
                plan = self._plan_explain(subject, associations, seen, qtype)
        elif qtype == "why":
            plan = self._plan_causal_explain(subject, associations, seen, qtype)
        elif qtype == "tell_me":
            # B9 (frame normalization; Tyler et al. 2011): "tell me about X" and
            # "what is X" both request the SAME semantic core about X — the LIFG
            # matches both frames against one retrieval network. The brain
            # retrieves identical information; only the pragmatic framing differs.
            # Route BOTH to the same explain-plan so the definitional content is
            # identical (the old code gave "tell me about X" an elaborate+ask-back
            # plan that collapsed to honest-uncertainty while "what is X" produced
            # an asserted fact — a quality split for the same concept).
            if self._is_abstract_concept(subject):
                plan = self._plan_abstract(subject, associations, seen, qtype)
            else:
                plan = self._plan_explain(subject, associations, seen, qtype)
        elif qtype == "compare":
            plan = self._plan_compare(subject, parts, associations, seen, qtype)
        elif qtype == "follow_up":
            plan = self._plan_continue(subject, associations, seen, qtype)
        elif qtype == "hypothetical":
            plan = self._plan_causal_explain(subject, associations, seen, qtype)
        elif qtype == "analogy":
            plan = self._plan_analogy(subject, parts, associations, seen, qtype)
        elif qtype == "do_you_know":
            plan = self._plan_explain(subject, associations, seen, qtype)
        elif qtype in ("greeting", "wellbeing", "capability", "introduction", "farewell"):
            plan = self._plan_social(subject, qtype)
        else:
            if self._is_abstract_concept(subject):
                plan = self._plan_abstract(subject, associations, seen, qtype)
            else:
                plan = self._plan_general(subject, associations, seen, qtype)
 
        # Ensure we have exactly 3 intents (pad if needed, skip for social intents)
        if qtype not in ("greeting", "wellbeing", "capability", "introduction", "farewell"):
            while len(plan.intents) < 3:
                plan.intents.append(DiscourseIntent(
                    type=DiscourseType.ELABORATE,
                    subject=subject,
                    primary_relation="semantic",
                    seen_so_far=seen.copy(),
                ))


        # Trim to capacity
        plan.intents = plan.intents[:self.capacity]

        # Apply discourse markers based on intent type
        for i, intent in enumerate(plan.intents):
            if i == 0:
                intent.discourse_marker = ""  # No marker for first sentence
            elif intent.type == DiscourseType.ELABORATE:
                intent.discourse_marker = self._pick_marker("elaborate")
            elif intent.type == DiscourseType.CONTRAST:
                intent.discourse_marker = self._pick_marker("contrast")
            elif intent.type == DiscourseType.CONNECT:
                intent.discourse_marker = self._pick_marker("connect")

        self.last_plan = plan
        if subject and subject not in self.topic_history:
            self.topic_history.append(subject.lower())
            if len(self.topic_history) > 10:
                self.topic_history.pop(0)

        return plan

    # ─── Planning Strategies ───
    # Enhanced with more diverse explanatory patterns (Bug E fix)
    # Neuroscience basis: prefrontal cortex builds causal models, not just association lists
    
    def _plan_social(self, subject: str, qtype: str) -> DiscoursePlan:
        plan = DiscoursePlan(original_subject=subject, question_type=qtype)
        plan.intents.append(DiscourseIntent(
            type="social",
            subject=qtype,
            primary_relation="social",
        ))
        return plan

    def _plan_explain(self, subject: str, associations: List[Tuple[str, float]],
                      seen: set, qtype: str) -> DiscoursePlan:
        """Plan: [EXPLAIN → ELABORATE → CONTRAST/CONNECT]
        
        Produces multi-perspective responses instead of flat association lists.
        """
        plan = DiscoursePlan(original_subject=subject, question_type=qtype)
        subj_lower = subject.lower()

        # Sentence 1: EXPLAIN - what the subject is, its core nature
        target1 = self._pick_best_association(associations, seen, exclude_verbs=True)
        # Try to get a more specific explanation target (not just top association)
        deeper = self._pick_best_association(associations, seen, exclude_subject=target1 if target1 else subject)
        
        # Use deeper association for more explanatory intent
        plan.intents.append(DiscourseIntent(
            type=DiscourseType.EXPLAIN,
            subject=subject,
            primary_relation="semantic",
            target_concept=target1,
            secondary_concept=deeper if deeper else "",
            seen_so_far=seen.copy(),
        ))
        if target1:
            seen.add(target1.lower())
        if deeper:
            seen.add(deeper.lower())

        # Sentence 2: ELABORATE — go deeper with causal or analogical reasoning
        causal_target = self._pick_best_association(associations, seen, prefer_causal=True)
        if causal_target:
            plan.intents.append(DiscourseIntent(
                type=DiscourseType.CAUSAL_EXPLAIN,
                subject=subject,
                primary_relation="causal",
                target_concept=causal_target,
                seen_so_far=seen.copy(),
            ))
            seen.add(causal_target.lower())
        else:
            target2 = self._pick_best_association(associations, seen, exclude_subject=target1 if target1 else subject)
            if not target2:
                target2 = self._pick_random_relation(associations, seen)
            plan.intents.append(DiscourseIntent(
                type=DiscourseType.ELABORATE,
                subject=target1 if target1 else subject,
                primary_relation="semantic",
                target_concept=target2,
                seen_so_far=seen.copy(),
            ))
            if target2:
                seen.add(target2.lower())

        # Sentence 3: CONTRAST or CONNECT back to broader context
        target3 = self._pick_best_association(associations, seen, prefer_contrast=True)
        intent3_type = DiscourseType.CONTRAST if target3 else DiscourseType.CONNECT
        plan.intents.append(DiscourseIntent(
            type=intent3_type,
            subject=subject,
            primary_relation="contrastive" if intent3_type == DiscourseType.CONTRAST else "semantic",
            target_concept=target3 or "",
            end_with_question=(target3 is None),  # ask back if nothing left to say
            seen_so_far=seen.copy(),
        ))

        return plan

    def _plan_causal_explain(self, subject: str, associations: List[Tuple[str, float]],
                              seen: set, qtype: str) -> DiscoursePlan:
        """Plan: [CAUSAL_EXPLAIN → CAUSAL_EXPLAIN → CONNECT]
        
        For 'why' and 'how' questions — produces deeper causal explanations
        with "because" structures rather than just listing associations.
        
        Enhanced with causal chain extraction: walks the causal path from
        seed concept to target concept, generating multi-sentence explanations
        that follow the causal mechanism.
        """
        plan = DiscoursePlan(original_subject=subject, question_type=qtype)

        # Sentence 1: CAUSAL — what causes/creates the subject (cause)
        target1 = self._pick_best_association(associations, seen, prefer_causal=True)
        if not target1:
            target1 = self._pick_best_association(associations, seen)
        plan.intents.append(DiscourseIntent(
            type=DiscourseType.CAUSAL_EXPLAIN,
            subject=subject,
            primary_relation="causal",
            target_concept=target1,
            seen_so_far=seen.copy(),
        ))
        if target1:
            seen.add(target1.lower())

        # Sentence 2: CAUSAL_EXPLAIN — what the subject causes/leads to (effect)
        # Using CAUSAL_EXPLAIN type for "BECAUSE" structures instead of ELABORATE
        target2 = self._pick_best_association(associations, seen, prefer_causal=True, exclude_subject=target1 if target1 else subject)
        if not target2:
            target2 = self._pick_best_association(associations, seen, exclude_subject=target1 if target1 else subject)
        plan.intents.append(DiscourseIntent(
            type=DiscourseType.CAUSAL_EXPLAIN,
            subject=target1 if target1 else subject,
            primary_relation="causal",
            target_concept=target2,
            use_epistemic_hedge=False,
            discourse_marker="because",  # Signal "because" structure
            seen_so_far=seen.copy(),
        ))
        if target2:
            seen.add(target2.lower())

        # Sentence 3: CONNECT — only use CAUSAL edges, not semantic
        # For why/hypothetical, restrict to causal relations
        causal_assocs = [(l, s) for l, s in associations 
                         if self._is_causal_association(l, associations)]
        plan.intents.append(DiscourseIntent(
            type=DiscourseType.CAUSAL_EXPLAIN,
            subject=subject,
            primary_relation="causal",
            target_concept=self._pick_best_association(causal_assocs or associations, seen) or "",
            end_with_question=True,
            seen_so_far=seen.copy(),
        ))

        return plan

    def _is_causal_association(self, label: str, associations: List[Tuple[str, float]]) -> bool:
        """Check if an association is causal rather than semantic."""
        ll = label.lower()
        causal_indicators = ["cause", "effect", "result", "lead", "because", "since",
                            "trigger", "create", "produce", "influence"]
        return any(ind in ll for ind in causal_indicators)

    def _plan_elaborate(self, subject: str, associations: List[Tuple[str, float]],
                         seen: set, qtype: str) -> DiscoursePlan:
        """Plan: [EXPLAIN → CONTRAST → ASK_BACK] — for 'tell me about X'"""
        plan = DiscoursePlan(original_subject=subject, question_type=qtype)

        target1 = self._pick_best_association(associations, seen)
        target2 = self._pick_best_association(associations, seen, prefer_contrast=True, exclude_subject=target1 if target1 else subject)

        plan.intents.append(DiscourseIntent(
            type=DiscourseType.EXPLAIN,
            subject=subject,
            target_concept=target1,
            seen_so_far=seen.copy(),
        ))
        if target1:
            seen.add(target1.lower())

        plan.intents.append(DiscourseIntent(
            type=DiscourseType.ELABORATE,
            subject=target1 if target1 else subject,
            target_concept=target2,
            seen_so_far=seen.copy(),
        ))
        if target2:
            seen.add(target2.lower())

        # ASK_BACK: generate an actual question to the user
        question = self._generate_follow_up_question(subject, target1)
        plan.intents.append(DiscourseIntent(
            type=DiscourseType.ASK_BACK,
            subject=subject,
            target_concept=question,
            end_with_question=True,
            primary_relation="interrogative",
            seen_so_far=seen.copy(),
        ))

        return plan

    def _plan_compare(self, subject: str, parts: List[str],
                       associations: List[Tuple[str, float]],
                       seen: set, qtype: str) -> DiscoursePlan:
        """Plan: [CONTRASTIVE_DIFF → EXPLAIN_A → EXPLAIN_B] — for compare questions
        Uses contrastive parallel activation to compute difference sets."""
        plan = DiscoursePlan(original_subject=subject, question_type=qtype)
        concept_a = parts[0] if len(parts) > 0 else subject
        concept_b = parts[1] if len(parts) > 1 else ""

        # Compute difference sets: unique associations for A, unique for B
        a_unique = []
        b_unique = []
        common = []
        a_lower = concept_a.lower()
        b_lower = concept_b.lower() if concept_b else ""

        if concept_b:
            for label, score in associations:
                ll = label.lower()
                if ll == a_lower or ll == b_lower:
                    continue
                # Simple heuristic: check graph edge types to find what's unique
                # Associations closer to A than B are "unique to A"
                # For a real implementation, we would use bidirectional spread
                if any(hint in label.lower() for hint in [a_lower[:3], ""]) and not any(hint in label.lower() for hint in [b_lower[:3]]):
                    a_unique.append((label, score))
                elif any(hint in label.lower() for hint in [b_lower[:3]]):
                    b_unique.append((label, score))
                else:
                    common.append((label, score))

        # Sentence 1: CONTRAST — highlight the key difference
        diff_target = a_unique[0][0] if a_unique else (b_unique[0][0] if b_unique else common[0][0] if common else "")
        if diff_target:
            plan.intents.append(DiscourseIntent(
                type=DiscourseType.CONTRAST,
                subject=concept_a,
                primary_relation="contrastive",
                target_concept=diff_target,
                secondary_concept=concept_b,
                seen_so_far=seen.copy(),
            ))
            seen.add(diff_target.lower())
        else:
            # Fallback: original behavior
            plan.intents.append(DiscourseIntent(
                type=DiscourseType.EXPLAIN,
                subject=concept_a,
                target_concept=self._pick_best_association(associations, seen),
                seen_so_far=seen.copy(),
            ))

        # Sentence 2: EXPLAIN_A
        target_a = a_unique[0][0] if a_unique else self._pick_best_association(associations, seen)
        plan.intents.append(DiscourseIntent(
            type=DiscourseType.EXPLAIN,
            subject=concept_a,
            target_concept=target_a,
            seen_so_far=seen.copy(),
        ))
        if target_a and target_a.lower() not in seen:
            seen.add(target_a.lower())
        seen.add(concept_a.lower())

        # Sentence 3: EXPLAIN_B or CONNECT
        if concept_b:
            target_b = b_unique[0][0] if b_unique else self._pick_best_association(associations, seen)
            plan.intents.append(DiscourseIntent(
                type=DiscourseType.EXPLAIN,
                subject=concept_b,
                target_concept=target_b,
                seen_so_far=seen.copy(),
            ))
            if concept_b:
                seen.add(concept_b.lower())
        else:
            plan.intents.append(DiscourseIntent(
                type=DiscourseType.CONNECT,
                subject=concept_a,
                target_concept=self._pick_best_association(associations, seen) or "",
                seen_so_far=seen.copy(),
            ))

        return plan

    def _plan_analogy(self, subject: str, parts: List[str],
                        associations: List[Tuple[str, float]],
                        seen: set, qtype: str,
                        vector_fn=None) -> DiscoursePlan:
        """Plan: [EXPLAIN_RELATION → CANDIDATE → CONNECT] — for A:B::C:___ analogies.

        Uses the AnalogyEngine for structure mapping:
        1. Extract relation between A and B from graph edges
        2. Find concepts that have the SAME relation with C
        3. Score candidates by relation vector similarity
        4. Generate explanation of the analogy
        """
        plan = DiscoursePlan(original_subject=subject, question_type=qtype)

        # Parse parts: A from parts[0], B from parts[1], C from parts[2]
        concept_a = parts[0].strip() if len(parts) > 0 else subject
        concept_b = parts[1].strip() if len(parts) > 1 else ""
        concept_c = parts[2].strip() if len(parts) > 2 else ""

        # Use AnalogyEngine if available
        candidate = ""
        if self.analogy_engine and concept_a and concept_b and concept_c:
            try:
                best_d = self.analogy_engine.get_best_completion(concept_a, concept_b, concept_c)
                if best_d:
                    candidate = best_d
            except Exception:
                pass

        # Fallback to heuristic
        if not candidate and concept_c:
            for label, score in associations:
                if label.lower() != concept_c.lower() and label.lower() not in seen:
                    candidate = label
                    break

        # Sentence 1: Explain the A:B relation
        plan.intents.append(DiscourseIntent(
            type=DiscourseType.EXPLAIN,
            subject=concept_a,
            primary_relation="analogical",
            target_concept=concept_b,
            seen_so_far=seen.copy(),
        ))
        if concept_b:
            seen.add(concept_b.lower())

        # Sentence 2: Propose the C:D relation as analogy
        plan.intents.append(DiscourseIntent(
            type=DiscourseType.CAUSAL_EXPLAIN,
            subject=concept_c if concept_c else concept_a,
            primary_relation="analogical",
            target_concept=candidate,
            discourse_marker="similarly",
            seen_so_far=seen.copy(),
        ))
        if candidate:
            seen.add(candidate.lower())

        # Sentence 3: CONNECT with question
        plan.intents.append(DiscourseIntent(
            type=DiscourseType.CONNECT,
            subject=concept_a,
            primary_relation="semantic",
            target_concept=candidate or "",
            end_with_question=True,
            seen_so_far=seen.copy(),
        ))

        return plan

    def _plan_abstract(self, subject: str, associations: List[Tuple[str, float]],
                       seen: set, qtype: str) -> DiscoursePlan:
        """Plan: [EXPERIENTIAL → SOCIAL → REFLECTIVE] — for abstract concepts.

        Uses the AbstractionEngine for multi-perspective reflection:
        1. Experiential: what the concept involves/feels like
        2. Social: what it means in society/relationships
        3. Reflective: personal/epistemic reflection
        """
        plan = DiscoursePlan(original_subject=subject, question_type=qtype)

        # Use AbstractionEngine if available
        if self.abstraction_engine:
            try:
                result = self.abstraction_engine.analyze_abstract_concept(subject)
                # Convert discourse intents to plan intents
                for i, intent_data in enumerate(result.discourse_intents):
                    plan.intents.append(DiscourseIntent(
                        type=intent_data.get("type", DiscourseType.EXPLAIN),
                        subject=intent_data.get("subject", subject),
                        primary_relation=intent_data.get("primary_relation", "semantic"),
                        target_concept=intent_data.get("target_concept", ""),
                        secondary_concept=intent_data.get("secondary_concept", ""),
                        use_epistemic_hedge=intent_data.get("use_epistemic_hedge", False),
                        end_with_question=intent_data.get("end_with_question", False),
                        discourse_marker=intent_data.get("discourse_marker", ""),
                        seen_so_far=seen.copy(),
                    ))
                    if intent_data.get("target_concept"):
                        seen.add(intent_data["target_concept"].lower())
                    if intent_data.get("secondary_concept"):
                        seen.add(intent_data["secondary_concept"].lower())
                return plan
            except Exception:
                pass  # Fall back to heuristic

        # Fallback heuristic
        # Sentence 1: Experiential — what the concept involves
        target1 = self._pick_best_association(associations, seen, exclude_verbs=True)
        plan.intents.append(DiscourseIntent(
            type=DiscourseType.EXPLAIN,
            subject=subject,
            primary_relation="semantic",
            target_concept=target1,
            use_epistemic_hedge=True,  # "It seems like..."
            seen_so_far=seen.copy(),
        ))
        if target1:
            seen.add(target1.lower())

        # Sentence 2: Social — broader context/perspective
        target2 = self._pick_best_association(associations, seen)
        plan.intents.append(DiscourseIntent(
            type=DiscourseType.ELABORATE,
            subject=target1 if target1 else subject,
            primary_relation="semantic",
            target_concept=target2,
            discourse_marker="in society",
            seen_so_far=seen.copy(),
        ))
        if target2:
            seen.add(target2.lower())

        # Sentence 3: Reflective — personal/epistemic reflection
        plan.intents.append(DiscourseIntent(
            type=DiscourseType.SELF_REFERENCE,
            subject=subject,
            primary_relation="semantic",
            target_concept=self._pick_best_association(associations, seen) or "",
            use_epistemic_hedge=True,
            end_with_question=True,
            seen_so_far=seen.copy(),
        ))

        return plan

    def _plan_continue(self, subject: str, associations: List[Tuple[str, float]],
                        seen: set, qtype: str) -> DiscoursePlan:
        """Plan: [CONTINUE → ELABORATE → CONNECT] — for follow-ups.

        Three-layer fallback for target selection:
        1. Try _pick_best_association (highest-scoring unseen)
        2. If all seen, try _pick_random_relation (random exploration)
        3. If even that fails, generate a fresh question to the user
        """
        plan = DiscoursePlan(original_subject=subject, question_type=qtype)

        # --- Sentence 1: CONTINUE ---
        target1 = self._pick_best_association(associations, seen)
        if not target1:
            target1 = self._pick_random_relation(associations, seen)

        if target1:
            plan.intents.append(DiscourseIntent(
                type=DiscourseType.CONTINUE,
                subject=subject,
                target_concept=target1,
                seen_so_far=seen.copy(),
            ))
            seen.add(target1.lower())
        else:
            # All associations exhausted — generate a question instead
            question = self._generate_follow_up_question(subject, target_concept=None)
            plan.intents.append(DiscourseIntent(
                type=DiscourseType.ASK_BACK,
                subject=subject,
                target_concept=question,
                end_with_question=True,
                primary_relation="interrogative",
                seen_so_far=seen.copy(),
            ))

        # --- Sentence 2: ELABORATE ---
        if target1:
            target2 = self._pick_best_association(associations, seen, exclude_subject=target1)
            if not target2:
                target2 = self._pick_random_relation(associations, seen)

            if target2:
                plan.intents.append(DiscourseIntent(
                    type=DiscourseType.ELABORATE,
                    subject=subject,
                    target_concept=target2,
                    seen_so_far=seen.copy(),
                ))
                seen.add(target2.lower())

        # --- Sentence 3: CONNECT or ASK_BACK ---
        if target1 and target2:
            target3 = self._pick_best_association(associations, seen, exclude_subject=target2)
            if not target3:
                target3 = self._pick_random_relation(associations, seen)
            plan.intents.append(DiscourseIntent(
                type=DiscourseType.CONNECT,
                subject=subject,
                target_concept=target3 or "people",
                end_with_question=(not target3),
                seen_so_far=seen.copy(),
            ))
        elif target1 and not target2:
            # Only have one topic — ask about it
            question = self._generate_follow_up_question(subject, target1)
            plan.intents.append(DiscourseIntent(
                type=DiscourseType.ASK_BACK,
                subject=target1,
                target_concept=question,
                end_with_question=True,
                primary_relation="interrogative",
                seen_so_far=seen.copy(),
            ))

        return plan

    def _plan_general(self, subject: str, associations: List[Tuple[str, float]],
                       seen: set, qtype: str) -> DiscoursePlan:
        """Default plan for general statements."""
        return self._plan_explain(subject, associations, seen, qtype)

    # ─── Helpers ───

    # ABSTRACT_NOUNS replaced by GloVe-based _is_abstract_concept
    ABSTRACT_NOUNS: set = set()  # Deprecated - kept for back-compat

    def _is_abstract_concept(self, subject: str) -> bool:
        """Check if a subject is abstract using GloVe-based classifier.

        ATL computes abstractness from semantic neighborhood (Cousins 2017),
        not a stored list. Falls back to suffix heuristics when GloVe unavailable.
        """
        if not subject:
            return False
        sl = subject.lower().strip()
        try:
            from ravana.language.verb_lexicon import _default_vector_fn
            fn = _default_vector_fn
            vec = fn(sl)
            if vec is not None:
                import numpy as np
                abstract_protos = ["love", "truth", "knowledge", "idea", "meaning", "beauty"]
                concrete_protos = ["table", "dog", "mountain", "car", "tree", "house"]
                abs_sims = [float(np.dot(vec, fn(p))) for p in abstract_protos if fn(p) is not None]
                con_sims = [float(np.dot(vec, fn(p))) for p in concrete_protos if fn(p) is not None]
                if abs_sims and con_sims:
                    return float(np.mean(abs_sims)) > float(np.mean(con_sims))
        except Exception:
            pass
        if sl.endswith('ness') or sl.endswith('ity') or sl.endswith('tion') or sl.endswith('ism'):
            return True
        if sl.endswith('ment') or sl.endswith('ance') or sl.endswith('ence'):
            return True
        if sl.endswith('ship') or sl.endswith('dom') or sl.endswith('hood'):
            return True
        return False

    def _pick_best_association(self, associations: List[Tuple[str, float]],
                                   seen: set,
                                   exclude_verbs: bool = False,
                                   exclude_subject: Optional[str] = None,
                                   prefer_causal: bool = False,
                                   prefer_contrast: bool = False) -> Optional[str]:
        """Pick the best association from a scored list, respecting constraints.

        Selects the highest-scoring item from associations that:
        - Is not already in the seen set
        - Is not a verb (if exclude_verbs=True)
        - Is not the exclude_subject
        - Prioritizes causal/contrastive relations if preferred

        Returns the label string, or None if no valid association found.
        """
        if not associations:
            return None

        # Filter by constraints
        candidates = []
        for label, score in associations:
            ll = label.lower().strip()
            if ll in seen:
                continue
            if exclude_verbs and self._is_verb_label(ll):
                continue
            if exclude_subject and ll == exclude_subject.lower().strip():
                continue
            candidates.append((label, score))

        if not candidates:
            return None

        # Apply preference boosts
        boosted = []
        for label, score in candidates:
            ll = label.lower()
            boost = 1.0
            if prefer_causal or prefer_contrast:
                if prefer_causal and self._has_causal_hint(ll):
                    boost *= 1.5
                if prefer_contrast and self._has_contrast_hint(ll):
                    boost *= 1.5
            boosted.append((label, score * boost))

        # Sort by boosted score descending and return best
        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted[0][0]

    def _is_verb_label(self, label: str) -> bool:
        """Check if a label is likely a verb."""
        ll = label.lower()
        verb_suffixes = ("ing", "ed", "en", "ify", "ize", "ate", "ish")
        verb_forms = {
            "be", "do", "have", "make", "take", "give", "get", "go", "come",
            "see", "know", "think", "feel", "say", "tell", "ask", "use",
            "find", "want", "seem", "need", "help", "work", "call", "try",
            "leave", "keep", "let", "begin", "show", "hear", "play", "run",
            "move", "live", "believe", "hold", "bring", "happen", "write",
            "provide", "sit", "stand", "lose", "pay", "meet", "include",
            "continue", "set", "learn", "change", "lead", "understand",
            "watch", "follow", "stop", "create", "cause", "let", "mean",
            "exist", "form", "act", "result", "produce", "connect", "relate",
        }
        if ll in verb_forms:
            return True
        if any(ll.endswith(s) for s in verb_suffixes):
            return True
        return False

    def _has_causal_hint(self, label: str) -> bool:
        """Check if a label has causal semantics."""
        ll = label.lower()
        causal_words = {
            "cause", "effect", "result", "consequence", "impact", "influence",
            "lead", "lead", "trigger", "produce", "create", "make", "generate",
            "force", "drive", "push", "enable", "allow", "prevent", "block",
            "because", "since", "hence", "therefore", "reaction", "response",
        }
        return ll in causal_words or any(ll.startswith(w) for w in causal_words)

    def _has_contrast_hint(self, label: str) -> bool:
        """Check if a label has contrastive semantics."""
        ll = label.lower()
        contrast_words = {
            "but", "however", "although", "though", "yet", "nevertheless",
            "contrast", "opposite", "different", "vs", "versus", "unlike",
            "instead", "rather", "still", "while", "whereas", "conversely",
            "on the other hand", "difference", "conflict", "against",
        }
        return ll in contrast_words

    def _pick_random_relation(self, associations: List[Tuple[str, float]],
                               seen: set) -> Optional[str]:
        """Pick a random unseen relation from associations.

        Basal ganglia analog: when no directed selection is possible,
        a random exploration step is used (Go/NoGo pathway).
        """
        if not associations:
            return None
        import random
        unseen = [(l, s) for l, s in associations if l.lower().strip() not in seen]
        if not unseen:
            return None
        return random.choice(unseen)[0]

    def _generate_follow_up_question(self, subject: str,
                                      target_concept: Optional[str] = None) -> str:
        """Generate a follow-up question to engage the user.

        Composes a context-appropriate question from primitives.
        Inspired by the DMN's social-cognitive questioning reflex.
        """
        import random
        if not target_concept:
            questions = [
                f"what do you think about {subject}?",
                f"have you experienced {subject} yourself?",
                f"what aspects of {subject} interest you?",
                f"would you like to explore more about {subject}?",
            ]
        else:
            questions = [
                f"does that match your understanding of {target_concept}?",
                f"have you noticed this about {target_concept}?",
                f"what is your perspective on {target_concept}?",
                f"how does {target_concept} relate to your experience?",
                f"would you like to know more about {target_concept}?",
            ]
        return random.choice(questions)

    def _pick_marker(self, marker_type: str) -> str:
        """Pick a discourse marker for the given type.

        Selects from the DISCOURSE_MARKERS dict. If the type has markers,
        returns the first one and rotates the list for next time.
        Falls back to empty string.
        """
        import random
        markers = self.DISCOURSE_MARKERS.get(marker_type, [])
        if not markers:
            return ""
        return random.choice(markers)

    def get_state(self) -> Dict:
        return {
            'capacity': self.capacity,
            'topic_history': self.topic_history,
        }

    def set_state(self, state: Dict):
        self.capacity = state.get('capacity', 5)
        self.topic_history = state.get('topic_history', [])
