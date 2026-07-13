"""Learned distributional word-salad classifier (research item B).

Replaces the UNIMPLEMENTED "neural perplexity" branch of the rule-based
``_is_word_salad`` (constants.py) with a real, fitted model.

Design (per the cross-cutting research plan, item B):
  - Gupta et al. "BERT & Family Eat Word Salad" (AAAI 2021): train a validity
    classifier on destructive transformations (shuffle / copy-sort / word-stuff)
    of coherent text, rather than thresholding hand-tuned structural bonuses.
  - Misra et al.: topic-entropy as a coherence signal (true docs stay on a few
    topics); WordSaladChopper (EMNLP 2025): detect repetitive loops on hidden
    states and calibrate the boundary via Equal Error Rate (EER).

The model is a logistic regression over LM-FREE distributional features:
  1. type_token_ratio           (lexical diversity)
  2. centroid_coherence         (mean cosine of sentence centroids to doc centroid)
  3. ngram_repeat               (bigram/trigram repetition rate)
  4. topic_entropy              (normalized entropy over sentence-topic clusters)
  5. novel_content_ratio        (tautology signal: response content not in subject)
  6. anchor_density             (copula/determiner presence — fluent glue)
  7. length_norm                (log word count, guards one-word fragments)

The decision boundary is FIT to labeled valid/invalid data via EER on a held-out
split (not hard-coded 0.7 / 0.55). The fit lives in data/salad_classifier.json
and is produced by experiments/measure_salad_classifier.py. If the fit file is
absent, the classifier degrades gracefully to the legacy rule-based
``_is_word_salad`` (so it is never a regression source) — provenance-favoring.

This classifier is intended to be wired as a FAIL-CLOSED final-emit guard
(engine._final_emit_guard) that the ``_disable_grounding_gate`` A/B kill-switch
CANNOT bypass. See that function for the wiring rationale.

All features are computed from the shared GloVe-64 projection cache so there is
no LLM dependency and no per-token inference cost beyond the centroids.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
# This file lives at <repo>/ravana/src/ravana/chat/salad_classifier.py. To reach
# <repo>/data we must go up 5 levels: chat -> ravana -> src -> ravana -> <repo>.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_DATA_DIR = os.path.join(_REPO_ROOT, "data")
_GLOVE_CACHE = os.path.join(_DATA_DIR, "ravana_glove_cache.npz")
_FIT_PATH = os.path.join(_DATA_DIR, "salad_classifier.json")

# Stopwords (mirrors constants.STOP_WORDS subset; self-contained to avoid import cycle)
_STOP = set("""
a an the and or but if then else when while of to in for on by at with from as is are was were
be been being do does did has have had will would should could may might must i you he she it we
they me my your his her our their this that these those not no yes what which who whom whose how why
about into over under between among through during before after above below off up down out near
""".split())


# ── GloVe index (shared projection cache) ──────────────────────────────────────
class _GloveIndex:
    """Lazy, process-wide GloVe-64 lookup using the project's projected cache."""

    _inst: Optional["_GloveIndex"] = None

    def __init__(self) -> None:
        self._vec: Optional[np.ndarray] = None
        self._proj: Optional[np.ndarray] = None
        self._lut: Dict[str, np.ndarray] = {}
        if os.path.exists(_GLOVE_CACHE):
            try:
                d = np.load(_GLOVE_CACHE, allow_pickle=True)
                words = [w.lower() for w in d["words"]]
                self._vec = d["vecs"].astype(np.float32)
                self._proj = d["proj"].astype(np.float32)
                self._lut = {w: self._vec[i] for i, w in enumerate(words)}
            except Exception:
                self._lut = {}

    @classmethod
    def inst(cls) -> "_GloveIndex":
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def vec(self, word: str) -> Optional[np.ndarray]:
        w = word.lower().strip()
        v = self._lut.get(w)
        if v is not None:
            return v
        # Project a 100-D GloVe vector on the fly if present in the raw cache.
        if self._vec is not None and self._proj is not None:
            raw = self._lut.get(w)
            if raw is not None and raw.shape[0] == self._proj.shape[1]:
                return self._proj @ raw
        return None

    def sentence_centroid(self, text: str) -> Optional[np.ndarray]:
        vecs = [self.vec(w) for w in re.findall(r"[a-z']+", text.lower())]
        vecs = [v for v in vecs if v is not None]
        if not vecs:
            return None
        c = np.mean(vecs, axis=0)
        n = np.linalg.norm(c)
        return c / n if n > 0 else c


# ── Feature extraction ────────────────────────────────────────────────────────
def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z']+", text.lower())


def extract_features(text: str, subject: Optional[str] = None) -> Dict[str, float]:
    """LM-free distributional features for the salad classifier.

    The dominant salad signal is TOPIC COHERENCE: random unrelated words have
    vectors that scatter in embedding space, whereas coherent text clusters
    around a topic. We compute this at the WORD level (not just sentence level)
    so single-sentence garbage — the canonical Q21 escape class — is caught
    (sentence-level coherence is undefined for one sentence and previously
    defaulted to 1.0, letting garbage through).
    """
    glove = _GloveIndex.inst()
    words = _words(text)
    n = len(words)
    feats: Dict[str, float] = {}

    # Word vectors + doc centroid (skip stopwords for the coherence signal).
    # Each word vector is NORMALIZED so the dot with the unit doc-centroid is a
    # true cosine in [-1, 1] (GloVe vectors have varying magnitude; an
    # un-normalized dot inflates coherence and pushes all values outside the
    # entropy histogram range, silently zeroing the entropy feature).
    content = [w for w in words if w not in _STOP]
    wvecs = []
    for w in content:
        v = glove.vec(w)
        if v is None:
            continue
        nv = np.linalg.norm(v)
        if nv > 0:
            wvecs.append(v / nv)
    if wvecs:
        doc_c = np.mean(wvecs, axis=0)
        doc_c /= (np.linalg.norm(doc_c) + 1e-9)
        # (1a) WORD-LEVEL coherence: mean cosine of each content word to the
        #      doc centroid. Random words -> scattered -> LOW. Coherent -> HIGH.
        #      This is the primary salad signal and works for single sentences.
        cos_w = [float(np.dot(v, doc_c)) for v in wvecs]
        feats["word_coherence"] = float(np.mean(cos_w))
        # (1b) coherence SPREAD: stdev of word-to-centroid cosines. Salad has
        #      high variance (some words align, most don't); coherent is tight.
        feats["word_coherence_spread"] = float(np.std(cos_w)) if len(cos_w) > 1 else 0.0
    else:
        feats["word_coherence"] = 0.0
        feats["word_coherence_spread"] = 0.0

    # 2. Type-token ratio (lexical diversity).
    feats["type_token_ratio"] = (len(set(words)) / n) if n else 0.0

    # 3. Sentence-level centroid coherence (multi-sentence only; for a single
    #    sentence we fall back to the word-level signal above, so it is NOT
    #    forced to 1.0).
    sents = _sentences(text)
    sent_vecs = [glove.sentence_centroid(s) for s in sents]
    sent_vecs = [v for v in sent_vecs if v is not None]
    if len(sent_vecs) >= 2:
        doc_cs = np.mean(sent_vecs, axis=0)
        doc_cs /= (np.linalg.norm(doc_cs) + 1e-9)
        cos_s = [float(np.dot(v, doc_cs)) for v in sent_vecs]
        feats["centroid_coherence"] = float(np.mean(cos_s))
    else:
        # single sentence: reuse word-level coherence (already computed)
        feats["centroid_coherence"] = feats["word_coherence"]

    # 4. N-gram repetition rate (bigram/trigram repeats).
    if n >= 4:
        bigrams = [tuple(words[i:i+2]) for i in range(n-1)]
        trigrams = [tuple(words[i:i+3]) for i in range(n-2)]
        bg_uniq = len(set(bigrams))
        tg_uniq = len(set(trigrams))
        feats["ngram_repeat"] = 1.0 - (bg_uniq / max(len(bigrams), 1)) * 0.5 \
            - (tg_uniq / max(len(trigrams), 1)) * 0.5
    else:
        feats["ngram_repeat"] = 0.0

    # 5. Topic entropy: entropy of content-word cosine-to-centroid distribution.
    #    Coherent text -> most words near the centroid (low entropy); salad ->
    #    words spread across angles (high entropy). Works for any length.
    if len(wvecs) >= 3:
        bins = np.histogram(cos_w, bins=5, range=(-1, 1))[0].astype(float)
        bins = bins[bins > 0]
        p = bins / bins.sum()
        ent = float(-(p * np.log(p)).sum())
        feats["topic_entropy"] = ent / np.log(5)
    else:
        feats["topic_entropy"] = 0.0

    # 6. Novel content ratio vs subject (tautology signal).
    if subject:
        subj_set = set(re.findall(r"[a-z']+", subject.lower()))
        cset = [w for w in content if w not in subj_set]
        feats["novel_content_ratio"] = (len(set(cset)) / max(len(cset), 1)) if cset else 0.0
    else:
        feats["novel_content_ratio"] = (len(set(content)) / max(len(content), 1)) if content else 0.0

    # 7. Anchor density (copula/determiner glue — fluent text has some).
    _anchors = set("is are was were has have had do does did a an the of to in for on by".split())
    feats["anchor_density"] = (sum(1 for w in words if w in _anchors) / n) if n else 0.0

    # 8. Length (log word count).
    feats["length_norm"] = float(np.log1p(n))

    return feats


_FEATURE_KEYS = [
    "word_coherence", "word_coherence_spread", "type_token_ratio",
    "centroid_coherence", "ngram_repeat", "topic_entropy",
    "novel_content_ratio", "anchor_density", "length_norm",
]


# ── Model (fitted logistic boundary) ──────────────────────────────────────────
class SaladClassifier:
    """Logistic boundary over distributional features; boundary fit via EER."""

    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 bias: float = 0.0, threshold: float = 0.5) -> None:
        self.weights = {k: float(weights.get(k, 0.0)) for k in _FEATURE_KEYS}
        self.bias = float(bias)
        self.threshold = float(threshold)

    @classmethod
    def load(cls) -> Optional["SaladClassifier"]:
        if not os.path.exists(_FIT_PATH):
            return None
        try:
            with open(_FIT_PATH, encoding="utf-8") as f:
                d = json.load(f)
            return cls(weights=d.get("weights", {}), bias=d.get("bias", 0.0),
                       threshold=d.get("threshold", 0.5))
        except Exception:
            return None

    def save(self) -> None:
        with open(_FIT_PATH, "w", encoding="utf-8") as f:
            json.dump({"weights": self.weights, "bias": self.bias,
                       "threshold": self.threshold}, f, indent=2)

    def score(self, text: str, subject: Optional[str] = None) -> float:
        """Raw logistic score (higher = more salad)."""
        feats = extract_features(text, subject)
        z = self.bias
        for k in _FEATURE_KEYS:
            z += self.weights.get(k, 0.0) * feats.get(k, 0.0)
        # logistic: P(salad)
        return float(1.0 / (1.0 + np.exp(-z)))

    def is_salad(self, text: str, subject: Optional[str] = None) -> bool:
        return self.score(text, subject) >= self.threshold


# ── Destructive transformations (Gupta et al. AAAI 2021) ───────────────────────
def destructive_transform(text: str, rng: Optional[np.random.Generator] = None) -> str:
    """Produce an INVALID variant of coherent `text` for negative training.

    Mirrors Gupta et al. (AAAI 2021): validity is learned from destructive
    edits. Three modes:
      0 shuffle            — randomize token order (preserves vocab, kills syntax)
      1 copy-sort          — sort a copy (removes order, keeps vocab)
      2 random-unrelated   — replace ~60% of tokens with UNRELATED vocab words,
                             so the sentence loses topic coherence (the dominant
                             signal the learned model uses: centroid_coherence /
                             topic_entropy). This is the harshest, most
                             salad-like transform and the one that best exercises
                             the escape-class detection.
    """
    words = _words(text)
    if len(words) < 4:
        return text
    if rng is None:
        rng = np.random.default_rng(abs(hash(" ".join(words))) & 0xFFFFFFFF)
    kind = int(rng.integers(0, 3))
    if kind == 0:
        sh = words[:]
        rng.shuffle(sh)
        return " ".join(sh)
    if kind == 1:
        return " ".join(sorted(words, key=lambda w: w))
    # random-unrelated: splice in off-topic words from a disjoint filler vocab
    _FILLER = ("purple bicycle quantum octopus silently waterfall empathy neon tomato "
               "calculate horizon velvet thunder mountain spoon whisper carbon algebra "
               "fog lighthouse mercy zebra decimal canyon locket tiger candle orbit frost "
               "library comet puzzle meadow cannon echo river piano volcano marble oxygen "
               "kettle gravity ember ladder theory canal velvet whisper").split()
    out = []
    for w in words:
        out.append(w)
        if rng.random() < 0.6:
            out.append(_FILLER[int(rng.integers(0, len(_FILLER)))])
    return " ".join(out)


# ── Singleton accessor (graceful degradation) ──────────────────────────────────
_model: Optional[SaladClassifier] = None
_model_loaded = False


def get_classifier() -> Optional[SaladClassifier]:
    """Return the fitted classifier, or None if no fit exists (caller falls
    back to the legacy rule-based detector)."""
    global _model, _model_loaded
    if not _model_loaded:
        _model = SaladClassifier.load()
        _model_loaded = True
    return _model


def is_salad_learned(text: str, subject: Optional[str] = None) -> Optional[bool]:
    """Learned salad verdict, or None if the model is not fitted."""
    clf = get_classifier()
    if clf is None:
        return None
    return clf.is_salad(text, subject)
