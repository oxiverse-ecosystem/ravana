"""Semantic Prototype Router (Stage 3 / M-A, de-hardcoding plan) — v2 fused.

Replaces the ~15 hardcoded routing lists (``QUERY_PATTERNS``, the classical-
paradox list, ``_is_conditional/_is_yesno/_is_informational`` regex,
``_self_pat`` ...) with ONE learned centroid router. Each intent is a
**fused prototype** = ``[α·semantic_centroid ; β·syntactic_shape_centroid]``:

  * ``semantic_centroid`` — query mean-pooled GloVe (lexical content), as v1.
  * ``syntactic_shape_centroid`` — NEW. A learned syntactic-shape sub-vector
    capturing the *grammatical form* the brain also routes on (dorsal stream):
    person/agent markers, command/imperative shape, interrogative shape,
    copula shape, tense. These are learned projections (centroid dot-products)
    and learned-weight structural descriptors — NOT routing rules.

Why fusion (brain basis): the first-person cluster (self_disclosure /
episodic_recall / remember_store / self_directed) shares ~identical content
words, so lexical prototypes collide (gaps 0.001–0.037, v1). But the brain
routes on BOTH ventral (semantic) AND dorsal (syntactic) streams: mPFC
self-model, hippocampus/PCC recollection, IFG imperative pipeline. Widening the
prototype to include shape is the principled, brain-faithful fix — the same
pattern as ``pos_model.py`` (adds a distributional POS sub-code) and
``salad_classifier`` (adds a structural sub-code).

Classification = nearest fused centroid by cosine, with a per-class fit margin.
Below margin → ``None`` (uncertain) → caller falls back to legacy regex.

Fit discipline: ``α``, ``β`` and per-route margins are GRID-FIT at EER on the
calibration corpus by ``experiments/measure_intent_router.py`` and persisted
to ``data/intent_router.json`` (no hardcoded numbers in source). Promotion is
per-route (``promoted`` allow-list), regression-gated — the engine consults
the router for route R only when R is in ``promoted``.

Routes: definition_seeking, philosophical_abstract, self_directed,
self_disclosure, episodic_recall, moral_advice, factual_yesno, conditional,
procedural, humor, chitchat, remember_store.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# This file lives at <repo>/ravana/src/ravana/chat/intent_router.py.
# To reach <repo>/data go up 5 levels: chat -> ravana -> src -> ravana -> <repo>.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_DATA_DIR = os.path.join(_REPO_ROOT, "data")
_FIT_PATH = os.path.join(_DATA_DIR, "intent_router.json")

ROUTES = (
    "definition_seeking", "philosophical_abstract", "self_directed",
    "self_disclosure", "episodic_recall", "moral_advice", "factual_yesno",
    "conditional", "procedural", "humor", "chitchat", "remember_store",
)

# ── Seed example queries per route (synthesized from the legacy lists) ──────
_SEED_QUERIES: Dict[str, List[str]] = {
    "definition_seeking": [
        "what is gravity", "what is ravana", "who was einstein",
        "what are black holes", "define photosynthesis", "tell me about dogs",
        "what does trust mean", "what is the internet", "who is my mom",
        "what is square a circle",
    ],
    "philosophical_abstract": [
        "what's the meaning of life", "why does everything exist",
        "is reality real", "are we living in a simulation",
        "why is there something instead of nothing",
        "what is the nature of consciousness", "what is the purpose of life",
    ],
    "self_directed": [
        "do you ever get tired", "what do you think about that",
        "do you like music", "are you awake", "what do you believe",
        "do you have feelings", "what are you", "can you think",
    ],
    "self_disclosure": [
        "my favorite color is blue", "i love stargazing", "i am a teenager",
        "my name is sam", "i like pizza", "my dog is called rex",
        "i'm happy", "i have a cat", "call me alex",
    ],
    "episodic_recall": [
        "what did i tell you", "remember what i said about my cat",
        "do you remember my birthday", "what was i saying earlier",
        "do you recall my favorite color",
    ],
    "moral_advice": [
        "is it ever okay to break a promise", "should i lie to my friend",
        "is it wrong to steal", "is it moral to eat meat",
        "should i tell the truth", "is cheating ever okay",
    ],
    "factual_yesno": [
        "is a whale a mammal", "can dogs eat chocolate",
        "are tomatoes fruits", "is water wet", "could humans live on mars",
        "is the earth round", "do cats purr",
    ],
    "conditional": [
        "what if cats ruled the world", "if gravity disappeared what would happen",
        "what would happen if the sun exploded", "suppose aliens invaded",
        "what would the world be like if gravity suddenly doubled",
        "imagine a world without money",
    ],
    "procedural": [
        "how do i build a perpetual motion machine", "how to make a cake",
        "how do i learn to code", "how to change a tire",
        "how do you bake bread",
    ],
    "humor": [
        "tell me a joke", "why did the chicken cross the road",
        "make me laugh", "say something funny",
    ],
    "chitchat": [
        "hi", "hello", "how are you", "i'm bored", "what's up",
        "good morning", "hey there", "how's it going",
    ],
    "remember_store": [
        "remember i love stargazing", "remember my favorite color is blue",
        "remember i have a dog", "keep in mind i'm allergic to peanuts",
        "remember that i hate spinach",
    ],
}

# Stop/function words carry little intent signal; down-weight them so content
# words dominate the pooled query vector (brain-faithful: semantic
# categorization keys on content, not closed-class words).
_STOP_WEIGHT = 0.15
_STOP_WORDS = frozenset(
    "a an the is are was were be been being am do does did can could would "
    "should may might must will shall have has had i you he she it we they "
    "me my your his her our their this that these those what who which where "
    "when why how tell me about of in on for to and or but if so from with "
    "as at by an into out up down over under".split()
)

# ── Shape seeds: tiny anchor word-lists used ONLY to BUILD learned centroids
# (seed discipline, as in _SEED_QUERIES / pos_model _seed_from_constants). The
# features derived from them are projections/descriptors, never routing rules.
_PERSON_WORDS = ["i", "me", "my", "we", "our", "myself", "mine"]
_COMMAND_WORDS = ["remember", "keep", "tell", "recall", "note", "remind"]
_AUX_INITIAL = frozenset(
    "do does did can could would should may might must will shall are is am "
    "was were have has had what who which where when why how".split())
_COPULA = frozenset("am is are was were".split())
_PAST_MARKERS = frozenset(
    "did told said was were had went ate saw gave made took came knew thought "
    "loved liked hated".split())
_FUTURE_MARKERS = frozenset("will would".split())

# Shape sub-vector dimensionality (ordered, stable schema).
_SHAPE_DIMS = 7  # [person, command, interrogative, copula, past, future, ntok]
_AFFECT_DIMS = 3  # [valence, arousal, dominance] from UserEmotionDetector
# Reference-target feature keys (stable order) from SelfAddressRouter.
_REF_KEYS = ("self_predicate", "second_person", "third_person_experiencer",
             "question_inversion", "agent_noun", "about_agent_cue",
             "pred_and_self_addr", "pred_and_third_party")
_REF_DIMS = len(_REF_KEYS)


def _reference_features(query: str) -> np.ndarray:
    """Reference-target sub-vector (self vs other/assistant) reused from the
    repo's SelfAddressRouter.extract_features — the brain-faithful ventral-mPFC
    (self) vs dorsal-mPFC/TPJ (other/mentalizing) gradient. Lazy-imported to
    avoid a circular import. Zero vector if unavailable (fail-open: the
    reference sub-space collapses and the router degrades to v3)."""
    try:
        from ravana.chat.self_model_router import extract_features
        feats = extract_features(query)
        return np.asarray([float(feats.get(k, 0.0)) for k in _REF_KEYS],
                          dtype=float)
    except Exception:
        return np.zeros(_REF_DIMS)


def _affect_features(query: str, detect_fn) -> np.ndarray:
    """Learned affect sub-vector = VAD from the repo's UserEmotionDetector.
    Accepts either a callable (the bound .detect method) or an object exposing
    .detect (the detector instance). Returns a 3-D vector; zero vector if the
    detector is unavailable (so the affect sub-space collapses and the router
    degrades to semantic⊕shape)."""
    fn = detect_fn
    if fn is not None and not callable(fn) and hasattr(fn, "detect"):
        fn = fn.detect
    if not callable(fn):
        return np.zeros(_AFFECT_DIMS)
    try:
        vad = fn(query)
        return np.asarray([float(x) for x in vad][:3], dtype=float)
    except Exception:
        return np.zeros(_AFFECT_DIMS)


def _mean_pool(tokens: List[str], glove_fn,
               weights: Optional[Dict[str, float]] = None) -> Optional[np.ndarray]:
    """Mean-pool GloVe vectors of in-vocabulary tokens (salience-weighted);
    None if none found. `weights` maps token->weight (default: content=1.0,
    stop/function=0.15)."""
    vecs = []
    wsum = 0.0
    for w in tokens:
        v = glove_fn(w) if callable(glove_fn) else None
        if v is None:
            continue
        wt = (weights or {}).get(w, _STOP_WEIGHT if w in _STOP_WORDS else 1.0)
        vecs.append(np.asarray(v, dtype=float) * wt)
        wsum += wt
    if not vecs or wsum == 0.0:
        return None
    m = np.sum(vecs, axis=0) / wsum
    n = np.linalg.norm(m)
    return m / n if n > 0 else None


def _build_anchor(words: List[str], glove_fn) -> Optional[np.ndarray]:
    """Mean-pooled GloVe anchor over seed words (None if none in vocab)."""
    vecs = []
    for w in words:
        v = glove_fn(w) if callable(glove_fn) else None
        if v is not None:
            vecs.append(np.asarray(v, dtype=float))
    if not vecs:
        return None
    m = np.mean(vecs, axis=0)
    n = np.linalg.norm(m)
    return m / n if n > 0 else None


def _max_proj(tokens: List[str], anchor: Optional[np.ndarray],
              glove_fn) -> float:
    """Max dot-product of any token with an anchor (0.0 if no glove/anchor)."""
    if anchor is None or not callable(glove_fn):
        return 0.0
    an = np.linalg.norm(anchor)
    if an == 0.0:
        return 0.0
    best = 0.0
    for w in tokens:
        v = glove_fn(w)
        if v is None:
            continue
        vn = np.linalg.norm(v)
        if vn == 0.0:
            continue
        d = float(np.dot(v, anchor)) / (vn * an)
        if d > best:
            best = d
    return best


def _shape_features(query: str, glove_fn=None,
                    person_anchor=None, command_anchor=None) -> np.ndarray:
    """Learned syntactic-shape sub-vector (7 dims, normalized):
    [person_proj, command_proj, interrogative, copula, past, future, ntok].
    The first two are learned projections onto person/command anchors; the rest
    are structural descriptors whose WEIGHT in the fused prototype is fit (not
    hardcoded rules)."""
    toks = [w for w in query.lower().split() if len(w) > 1]
    first = toks[0] if toks else ""
    person = _max_proj(toks, person_anchor, glove_fn)
    command = _max_proj(toks, command_anchor, glove_fn)
    interrogative = 1.0 if first in _AUX_INITIAL else 0.0
    copula = 1.0 if any(t in _COPULA for t in toks) else 0.0
    past = 1.0 if any(t in _PAST_MARKERS for t in toks) else 0.0
    future = 1.0 if any(t in _FUTURE_MARKERS for t in toks) else 0.0
    ntok = min(len(toks) / 12.0, 1.0)
    vec = np.array([person, command, interrogative, copula, past, future, ntok],
                   dtype=float)
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


class IntentRouter:
    """Nearest-centroid intent classifier over a FUSED (semantic ⊕ shape ⊕
    affect) prototype. ``classify`` returns the nearest route when its cosine
    similarity clears the max-margin gap over the runner-up; otherwise ``None``
    (uncertain -> the caller should use the legacy regex).

    The affect sub-vector reuses the repo's already-learned
    ``UserEmotionDetector.detect()`` VAD projection (no new model, no new
    hardcoded lists) — the brain-faithful OFC/vmPFC valence axis that separates
    affective self-statements ("I am sad") from semantic self-knowledge ("I am
    a teen"). If the detector is unavailable, gamma collapses to 0 and the
    router degrades to the verified semantic⊕shape behavior (fail-open).
    """

    def __init__(self, semantic_centroids=None, shape_centroids=None,
                 affect_centroids=None, reference_centroids=None,
                 margins=None, alpha=1.0, beta=1.0, gamma=0.0, delta=0.0,
                 promoted=None, detect_fn=None) -> None:
        self._sem = {r: (np.asarray(v, dtype=float) if v is not None else None)
                     for r, v in (semantic_centroids or {}).items()}
        self._shape = {r: (np.asarray(v, dtype=float) if v is not None else None)
                       for r, v in (shape_centroids or {}).items()}
        self._affect = {r: (np.asarray(v, dtype=float) if v is not None else None)
                        for r, v in (affect_centroids or {}).items()}
        self._ref = {r: (np.asarray(v, dtype=float) if v is not None else None)
                     for r, v in (reference_centroids or {}).items()}
        self._margins = dict(margins or {})
        self._default_margin = float(margins.get("_default", 0.06)
                                     if margins else 0.06)
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._gamma = float(gamma)
        self._delta = float(delta)
        self._promoted = set(promoted or [])
        self._person_anchor = None
        self._command_anchor = None
        self._detect_fn = detect_fn  # UserEmotionDetector.detect or None

    def rebind_anchors(self, glove_fn) -> None:
        """Cache the person/command projection anchors (built from seed words
        via GloVe). Cheap; call once per routing session."""
        if callable(glove_fn) and self._person_anchor is None:
            self._person_anchor = _build_anchor(_PERSON_WORDS, glove_fn)
            self._command_anchor = _build_anchor(_COMMAND_WORDS, glove_fn)

    @classmethod
    def from_seed(cls, glove_fn, seed_queries=None, margin=0.06,
                  alpha=1.0, beta=1.0, gamma=1.0, delta=1.0,
                  detect_fn=None) -> "IntentRouter":
        """Build semantic + shape + affect + reference centroids from seed
        queries. `detect_fn` is UserEmotionDetector.detect (VAD); if None,
        affect is skipped and gamma is forced to 0 (fail-open). The reference
        sub-vector is always built from SelfAddressRouter.extract_features
        (no extra model); delta is the fit weight."""
        seed_queries = seed_queries or _SEED_QUERIES
        person_anchor = _build_anchor(_PERSON_WORDS, glove_fn)
        command_anchor = _build_anchor(_COMMAND_WORDS, glove_fn)
        sem_c: Dict[str, List[float]] = {}
        shape_c: Dict[str, List[float]] = {}
        affect_c: Dict[str, List[float]] = {}
        ref_c: Dict[str, List[float]] = {}
        use_affect = callable(detect_fn) or hasattr(detect_fn, "detect")
        eff_gamma = gamma if use_affect else 0.0
        for route, queries in seed_queries.items():
            svecs, shvecs, aves, rves = [], [], [], []
            for q in queries:
                toks = [w for w in q.lower().split() if len(w) > 1]
                sp = _mean_pool(toks, glove_fn)
                sh = _shape_features(q, glove_fn, person_anchor, command_anchor)
                if sp is not None:
                    svecs.append(sp)
                shvecs.append(sh)
                if use_affect:
                    aves.append(_affect_features(q, detect_fn))
                rves.append(_reference_features(q))
            if svecs:
                m = np.mean(svecs, axis=0)
                n = np.linalg.norm(m)
                sem_c[route] = (m / n).tolist() if n > 0 else m.tolist()
            if shvecs:
                m = np.mean(shvecs, axis=0)
                n = np.linalg.norm(m)
                shape_c[route] = (m / n).tolist() if n > 0 else m.tolist()
            if use_affect and aves:
                m = np.mean(aves, axis=0)
                n = np.linalg.norm(m)
                affect_c[route] = (m / n).tolist() if n > 0 else m.tolist()
            if rves:
                m = np.mean(rves, axis=0)
                n = np.linalg.norm(m)
                ref_c[route] = (m / n).tolist() if n > 0 else m.tolist()
        margins = {r: margin for r in sem_c}
        margins["_default"] = margin
        return cls(semantic_centroids=sem_c, shape_centroids=shape_c,
                   affect_centroids=affect_c, reference_centroids=ref_c,
                   margins=margins, alpha=alpha, beta=beta,
                   gamma=eff_gamma, delta=delta, detect_fn=detect_fn)

    # ── internal fused-vector helpers ──
    def _fuse(self, sem: Optional[np.ndarray],
              shape: Optional[np.ndarray],
              affect: Optional[np.ndarray] = None,
              reference: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if sem is None:
            return None
        if shape is None:
            parts = [self._alpha * sem]
        else:
            parts = [self._alpha * sem, self._beta * shape]
        if affect is not None and self._gamma != 0.0:
            parts.append(self._gamma * affect)
        if reference is not None and self._delta != 0.0:
            parts.append(self._delta * reference)
        return np.concatenate(parts)

    def _route_vec(self, route: str) -> Optional[np.ndarray]:
        sem = self._sem.get(route)
        shape = self._shape.get(route)
        affect = self._affect.get(route)
        reference = self._ref.get(route)
        if sem is None and shape is None and affect is None and reference is None:
            return None
        sem = sem if sem is not None else np.zeros(_SHAPE_DIMS)
        shape = shape if shape is not None else np.zeros(_SHAPE_DIMS)
        affect = affect if affect is not None else np.zeros(_AFFECT_DIMS)
        reference = reference if reference is not None else np.zeros(_REF_DIMS)
        return self._fuse(sem, shape, affect, reference)

    def classify(self, query: str, glove_fn=None) -> Optional[str]:
        """Return the nearest route, or None if uncertain (below margin)."""
        if not self._sem:
            return None
        if callable(glove_fn):
            self.rebind_anchors(glove_fn)
        toks = [w for w in query.lower().split() if len(w) > 1]
        person_anchor = getattr(self, "_person_anchor", None)
        command_anchor = getattr(self, "_command_anchor", None)
        q_sem = _mean_pool(toks, glove_fn) if callable(glove_fn) else None
        q_shape = _shape_features(query, glove_fn, person_anchor, command_anchor)
        q_affect = _affect_features(query, self._detect_fn) if self._gamma != 0.0 else None
        q_ref = _reference_features(query)
        qv = self._fuse(q_sem, q_shape, q_affect, q_ref)
        if qv is None:
            return None
        qn = np.linalg.norm(qv)
        if qn == 0.0:
            return None
        sims = {}
        for route, c in self._sem.items():
            rv = self._route_vec(route)
            if rv is None:
                continue
            rn = np.linalg.norm(rv)
            sims[route] = float(np.dot(qv, rv)) / (qn * rn) if rn > 0 else 0.0
        if not sims:
            return None
        ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
        best_route, best_sim = ranked[0]
        margin = self._margins.get(best_route, self._default_margin)
        if len(ranked) > 1:
            second_sim = ranked[1][1]
            if (best_sim - second_sim) < margin:
                return None  # uncertain -> regex fallback
        return best_route

    @classmethod
    def load(cls) -> Optional["IntentRouter"]:
        if not os.path.exists(_FIT_PATH):
            return None
        try:
            with open(_FIT_PATH, encoding="utf-8") as f:
                d = json.load(f)
            # Schema v1 (semantic-only) fallback.
            if "centroids" in d and "semantic_centroids" not in d:
                return cls(semantic_centroids=d["centroids"],
                           margins=d.get("margins", {}))
            return cls(
                semantic_centroids=d.get("semantic_centroids", {}),
                shape_centroids=d.get("shape_centroids", {}),
                affect_centroids=d.get("affect_centroids", {}),
                reference_centroids=d.get("reference_centroids", {}),
                margins=d.get("margins", {}),
                alpha=d.get("alpha", 1.0),
                beta=d.get("beta", 1.0),
                gamma=d.get("gamma", 0.0),
                delta=d.get("delta", 0.0),
                promoted=d.get("promoted", []),
            )
        except Exception:
            return None

    def save(self) -> None:
        os.makedirs(_DATA_DIR, exist_ok=True)
        with open(_FIT_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "semantic_centroids": {k: (v.tolist() if hasattr(v, "tolist") else v)
                                       for k, v in self._sem.items()},
                "shape_centroids": {k: (v.tolist() if hasattr(v, "tolist") else v)
                                    for k, v in self._shape.items()},
                "affect_centroids": {k: (v.tolist() if hasattr(v, "tolist") else v)
                                     for k, v in self._affect.items()},
                "reference_centroids": {k: (v.tolist() if hasattr(v, "tolist") else v)
                                        for k, v in self._ref.items()},
                "margins": self._margins,
                "alpha": self._alpha,
                "beta": self._beta,
                "gamma": self._gamma,
                "delta": self._delta,
                "promoted": sorted(self._promoted),
            }, f, indent=2)

    def set_promoted(self, routes) -> None:
        self._promoted = set(routes)

    def is_promoted(self, route: str) -> bool:
        return route in self._promoted


def default_router() -> Optional[IntentRouter]:
    """Load the fitted router (fail-open: None if no fit file)."""
    return IntentRouter.load()
