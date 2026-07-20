"""
RAVANA — Round 5 (D1): Self-supervised junk_score.
=====================================================================
The C1 fix (Round 4) gates graph-node admission on junk_score, but its
weights and threshold (theta=0.5) were hand-set constants, and the
SaladClassifier it wraps was fit on a synthetic corpus. This module replaces
that with a continually-fit classifier whose LABELS are free from the graph's
own consolidation outcomes — no manual labeling, no external corpus required.

Brain-faithful design (hippocampal self-labeling + error-driven refit):
  - The consolidation cycle IS the training signal. A token promoted across
    >= k distinct sources / surviving a sleep-degree prune = "consolidated"
    (positive, like a hippocampal trace crossing to neocortex). A token
    junk-gated or pruned / never-reactivated = "decayed" (negative).
  - Two noisy oracles (topology + external grounding + PMI-stability) are
    aggregated (Dawid-Skene-style, with a neutral bucket) into weak proxy
    labels, aged > T cycles to avoid circular self-labeling.
  - A logistic (warm-started SGD) refits every K consolidation cycles; theta
    is DERIVED from the observed prior-positive rate, not hand-set.
  - The hand-rule structural floor (keyboard-mash / POS-tag / website-shape /
    vowel-less) stays a NON-learnable backstop — unambiguous junk regardless
    of data. With ZERO labels, junk_score reproduces the Round-4 formula
    EXACTLY (cold-start == current), so there is no regression.

Only the SOFT features (glove OOV, salad, low-degree, low-source-count,
PMI-instability) are fit. The structural floor is hard.
"""

from __future__ import annotations

import os
import re
import json
import math
import time
import hashlib
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

# ── structural shapes (non-learnable backstop) ──
_WEBSITE_SHAPE = re.compile(
    r"(com|net|org|edu|gov|io|html|php|asp|jsp|www)$", re.I)
_POS_TAGS = {"adj", "adv", "noun", "verb", "nouns", "verbs", "adjs", "advs",
             "prep", "conj", "det", "pron", "aux", "adjp", "np", "vp"}

# Data path for persisted weak labels + fitted model (set by the engine).
_DATA_DIR = None
_LABELS_PATH = None
_MODEL_PATH = None


def configure(data_dir: Optional[str]) -> None:
    """Point the scorer at the engine's data dir for persistence."""
    global _DATA_DIR, _LABELS_PATH, _MODEL_PATH
    _DATA_DIR = data_dir
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)
        _LABELS_PATH = os.path.join(data_dir, "junk_labels.jsonl")
        _MODEL_PATH = os.path.join(data_dir, "junk_classifier.json")
    else:
        _LABELS_PATH = None
        _MODEL_PATH = None


# ── structural floor (identical to Round-4 constants, non-learnable) ──
def _structural_floor(word: str) -> float:
    w = word.lower().strip("'\"")
    if not w:
        return 1.0
    s = 0.0
    if _is_keyboard_mash(w):
        s += 0.45
    if w in _POS_TAGS:
        s += 0.5
    if _WEBSITE_SHAPE.search(w) or any(ch.isdigit() for ch in w):
        s += 0.35
    _vowels = set("aeiouy")
    _vc = sum(1 for ch in w if ch in _vowels)
    if len(w) >= 4 and _vc == 0:
        s += 0.4
    return s


def _is_keyboard_mash(w: str) -> bool:
    # Reuse the canonical checker from constants if importable, else inline.
    try:
        from ravana.chat.constants import _is_keyboard_mash as _km
        return _km(w)
    except Exception:
        pass
    if len(w) < 4:
        return False
    runs = 1
    best = 1
    for i in range(1, len(w)):
        if abs(ord(w[i]) - ord(w[i - 1])) <= 2:
            runs += 1
            best = max(best, runs)
        else:
            runs = 1
    return best >= 4


def _salad_signal(word: str) -> float:
    """Word-level salad signal in [0,1] (learned SaladClassifier if available)."""
    try:
        from ravana.chat.salad_classifier import is_salad_learned
        if is_salad_learned(word) is True:
            return 1.0
    except Exception:
        pass
    return 0.0


# ── external grounding oracle: WordNet sense-inventory (D1-2 weak labeler) ──
# A token present in WordNet's synsets is a real dictionary word (distant
# positive grounding). Absent = OOV / neologism / website-name / hash-noise
# (distant negative). Lazy import so the module degrades gracefully where
# nltk/WordNet is unavailable (falls back to GloVe-magnitude grounding).
_WORDNET_OK = None  # None=unknown, True/False once probed


def wordnet_grounding(word: str) -> float:
    """Return 1.0 if `word` has >=1 WordNet synset, else 0.0.

    Brain-faithful external-grounding oracle for the self-supervised junk
    classifier: a word anchored in a real sense inventory is almost never junk.
    """
    global _WORDNET_OK
    w = word.lower().strip("'\"")
    if not w:
        return 0.0
    if _WORDNET_OK is False:
        return 0.0
    try:
        from nltk.corpus import wordnet as wn
        _WORDNET_OK = True
        return 1.0 if len(wn.synsets(w)) > 0 else 0.0
    except Exception:
        _WORDNET_OK = False
        return 0.0


# ── feature names (soft, fit) ──
FEATURES = ["glove_oo", "salad", "low_degree", "low_sources",
            "pmi_unstable", "wordnet_present"]


def _extract_features(word: str, glove_mag: Optional[float] = None,
                      degree: Optional[int] = None,
                      source_count: Optional[int] = None,
                      pmi_stability: Optional[float] = None,
                      wordnet_present: Optional[float] = None) -> np.ndarray:
    glove_oo = max(0.0, (0.5 - (glove_mag if glove_mag is not None else 0.0))) / 0.5
    salad = _salad_signal(word)
    low_degree = max(0.0, (8 - (degree if degree is not None else 8))) / 8.0
    low_sources = max(0.0, (2 - (source_count if source_count is not None else 2))) / 2.0
    pmi_unstable = 0.0 if pmi_stability is None else max(0.0, 1.0 - pmi_stability)
    if wordnet_present is None:
        wordnet_present = float(wordnet_grounding(word))
    return np.array([glove_oo, salad, low_degree, low_sources, pmi_unstable,
                     wordnet_present], dtype=np.float64)


class OnlineJunkClassifier:
    """Continually-fit logistic over the soft junk features.

    Cold-start: w = 0 (sigmoid(0) = 0.5 baseline) and NOT ready, so callers
    fall back to the hand-weighted Round-4 formula. After MIN_LABELS accrued
    and a refit, `ready` flips True and theta is derived from the data.
    """

    MIN_LABELS = 32

    def __init__(self, theta0: float = 0.5, w: Optional[List[float]] = None):
        self.theta = theta0
        self.w = np.array(w if w else [0.0] * len(FEATURES), dtype=np.float64)
        self.ready = False
        self.n_seen = 0
        # Anti-collapse monitor history (kept bounded).
        self._theta_hist: List[float] = []
        self._kappa_hist: List[float] = []
        self._brier_hist: List[float] = []
        self._ingress_hist: List[float] = []
        self._hub_stability: Optional[float] = None
        self._last_w = self.w.copy()
        self._last_y: Optional[np.ndarray] = None

    # ── persistence ──
    def to_dict(self) -> dict:
        return {
            "theta": self.theta,
            "w": self.w.tolist(),
            "ready": self.ready,
            "n_seen": self.n_seen,
            "theta_hist": self._theta_hist[-50:],
            "kappa_hist": self._kappa_hist[-50:],
            "brier_hist": self._brier_hist[-50:],
            "ingress_hist": self._ingress_hist[-50:],
            "hub_stability": self._hub_stability,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OnlineJunkClassifier":
        w = d.get("w")
        # Feature-dimension guard: if a persisted model was fit with a
        # different feature set (e.g. before WordNet was added), discard it
        # rather than loading a mismatched weight vector — it refits cleanly.
        if isinstance(w, list) and len(w) != len(FEATURES):
            w = None
            d = dict(d, ready=False)
        obj = cls(theta0=d.get("theta", 0.5), w=w)
        obj.ready = d.get("ready", False)
        obj.n_seen = d.get("n_seen", 0)
        obj._theta_hist = d.get("theta_hist", [])
        obj._kappa_hist = d.get("kappa_hist", [])
        obj._brier_hist = d.get("brier_hist", [])
        obj._ingress_hist = d.get("ingress_hist", [])
        obj._hub_stability = d.get("hub_stability")
        return obj

    # ── math ──
    def predict(self, feat: np.ndarray) -> float:
        z = float(self.w.dot(feat))
        return 1.0 / (1.0 + math.exp(-z))

    def _sgd_step(self, feat: np.ndarray, y: float, lr: float = 0.1) -> float:
        p = self.predict(feat)
        grad = (p - y) * feat
        self.w -= lr * grad
        return p

    def refit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1,
              epochs: int = 8) -> Optional[dict]:
        """Batch SGD refit over weak labels. Returns monitor delta or None."""
        if X.shape[0] < self.MIN_LABELS:
            return None
        self._last_w = self.w.copy()
        self._last_y = y.copy()
        n = X.shape[0]
        # Shuffle for SGD stability.
        perm = np.random.RandomState(42).permutation(n)
        Xs, ys = X[perm], y[perm]
        # Brier (before).
        _brier_before = float(np.mean([(self.predict(Xs[i]) - ys[i]) ** 2 for i in range(n)]))
        for _ in range(epochs):
            for i in range(n):
                self._sgd_step(Xs[i], ys[i], lr=lr)
        _brier_after = float(np.mean([(self.predict(Xs[i]) - ys[i]) ** 2 for i in range(n)]))
        # Derive theta from the observed prior-positive rate (data-derived
        # decision boundary, not hand-set).
        pos_rate = float(np.mean(y))
        self.theta = float(min(0.9, max(0.1, pos_rate)))
        self.ready = True
        self.n_seen = n
        # Anti-collapse monitors.
        self._theta_hist.append(self.theta)
        self._brier_hist.append(_brier_after)
        # Cohen's kappa between previous-fit predictions and this-fit predictions
        # on the same batch (collapse detector: kappa -> 0 means the model is
        # no longer agreeing with itself across refits = runaway).
        _prev_pred = np.array([1.0 / (1.0 + math.exp(-float(self._last_w.dot(Xs[i])))) for i in range(n)])
        _cur_pred = np.array([self.predict(Xs[i]) for i in range(n)])
        _k = _cohen_kappa(_prev_pred >= 0.5, _cur_pred >= 0.5)
        self._kappa_hist.append(_k)
        return {
            "n": n, "brier_before": _brier_before, "brier_after": _brier_after,
            "theta": self.theta, "kappa": _k,
        }


def _cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa between two boolean label vectors."""
    a = a.astype(bool)
    b = b.astype(bool)
    n = a.size
    if n == 0:
        return 1.0
    pa = a.mean()
    pb = b.mean()
    po = (a == b).mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe == 1.0:
        return 1.0
    return float((po - pe) / (1 - pe))


# ── weak-label buffer (the self-labeling oracle) ──
class JunkLabelBuffer:
    """Collects weak proxy labels from consolidation outcomes, aggregates
    three noisy oracles (topology / external-grounding / PMI-stability) with a
    neutral bucket, ages them > T cycles, and persists to junk_labels.jsonl.

    Oracle rules (Dawid-Skene-style majority-with-confidence):
      topology : promoted/surviving-hub -> +1 ; pruned/never-reactivated -> -1
      grounding: in GloVe vocab (mag>=0.5) -> +0.5 ; OOV -> -0.5 ; unknown -> 0
      pmi      : stable across >=2 sources -> +0.5 ; unstable -> -0.5 ; <2 -> 0
    Aggregated score: clamp(sum, -1, 1). Proxy label = sign with |score|>=0.5,
    else neutral (not used for fitting — avoids forcing ambiguous cases).
    """

    def __init__(self, data_dir: Optional[str] = None, age_cycles: int = 2,
                 max_positive_frac: float = 0.4):
        self._age_cycles = age_cycles
        self._max_pos_frac = max_positive_frac
        self._cycle = 0
        self._first_seen: Dict[str, int] = {}
        self._pending: Dict[str, dict] = {}   # word -> accumulated oracle votes
        self._flushed: List[Tuple[str, int, dict]] = []  # (word, label, meta)
        self._pos = 0
        self._neg = 0
        self._neutral = 0
        self._bio = {"admitted": 0, "relabeled_negative": 0}
        if data_dir:
            configure(data_dir)

    def tick(self) -> None:
        """Advance the consolidation cycle counter (call from _sleep_consolidate)."""
        self._cycle += 1

    def record(self, word: str, oracle: str, vote: float,
               meta: Optional[dict] = None) -> None:
        """Record one oracle vote for a token. oracle in
        {topology, grounding, pmi}; vote in [-1, 1]."""
        w = word.lower().strip("'\"")
        if not w:
            return
        if w not in self._first_seen:
            self._first_seen[w] = self._cycle
        rec = self._pending.setdefault(w, {"topology": 0.0, "grounding": 0.0,
                                            "pmi": 0.0, "meta": meta or {}})
        rec[oracle] = vote
        rec["meta"] = meta or rec["meta"]

    def _aggregate(self, rec: dict) -> Optional[int]:
        score = rec["topology"] + rec["grounding"] + rec["pmi"]
        score = max(-1.0, min(1.0, score))
        if abs(score) < 0.5:
            return 0  # neutral bucket
        return 1 if score > 0 else -1

    def flush_aged(self) -> List[Tuple[str, int, dict]]:
        """Flush labels for tokens first-seen > age_cycles ago. Returns the
        newly-emitted (word, label, meta) list and persists to disk."""
        out = []
        ready = [w for w, rec in self._pending.items()
                 if (self._cycle - self._first_seen.get(w, self._cycle)) > self._age_cycles]
        for w in ready:
            rec = self._pending.pop(w)
            lab = self._aggregate(rec)
            if lab is None:
                self._neutral += 1
                continue
            # Cold-start guard: cap positive fraction at max_positive_frac of
            # all labels emitted so far (bootstrap can't run away positive).
            if lab == 1 and (self._pos + 1) > self._max_pos_frac * (self._pos + self._neg + 1):
                lab = -1  # demote to negative if cap exceeded
            if lab == 1:
                self._pos += 1
            else:
                self._neg += 1
            self._flushed.append((w, lab, rec.get("meta", {})))
            out.append((w, lab, rec.get("meta", {})))
        if out and _LABELS_PATH:
            try:
                with open(_LABELS_PATH, "a", encoding="utf-8") as f:
                    for w, lab, meta in out:
                        f.write(json.dumps({"word": w, "label": lab,
                                            "cycle": self._cycle, "meta": meta}) + "\n")
            except Exception:
                pass
        return out

    def load_persisted(self) -> List[Tuple[str, int, dict]]:
        """Load historical (word, label, meta) triples from disk for refit."""
        if not _LABELS_PATH or not os.path.exists(_LABELS_PATH):
            return []
        rows = []
        try:
            with open(_LABELS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        rows.append((d["word"], int(d["label"]), d.get("meta", {})))
                    except Exception:
                        continue
        except Exception:
            pass
        return rows

    def feature_row(self, word: str, glove_mag: Optional[float],
                    meta: dict) -> np.ndarray:
        degree = meta.get("degree")
        source_count = meta.get("source_count")
        pmi_stability = meta.get("pmi_stability")
        wordnet_present = meta.get("wordnet_present")
        if wordnet_present is None:
            wordnet_present = float(wordnet_grounding(word))
        return _extract_features(word, glove_mag=glove_mag, degree=degree,
                                 source_count=source_count, pmi_stability=pmi_stability,
                                 wordnet_present=wordnet_present)

    def report_counts(self) -> dict:
        return {"pos": self._pos, "neg": self._neg, "neutral": self._neutral,
                "flushed": len(self._flushed),
                "pos_frac": (self._pos / max(1, self._pos + self._neg))}


# ── module-global singleton (shared across the process) ──
_CLF: Optional[OnlineJunkClassifier] = None
_BUF: Optional[JunkLabelBuffer] = None


def _get_classifier(reload: bool = False) -> Optional[OnlineJunkClassifier]:
    global _CLF
    if _CLF is not None and not reload:
        return _CLF
    if _MODEL_PATH and os.path.exists(_MODEL_PATH):
        try:
            with open(_MODEL_PATH, "r", encoding="utf-8") as f:
                _CLF = OnlineJunkClassifier.from_dict(json.load(f))
            return _CLF
        except Exception:
            pass
    return _CLF


def _set_classifier(clf: OnlineJunkClassifier) -> None:
    global _CLF
    _CLF = clf


def get_buffer(data_dir: Optional[str] = None) -> JunkLabelBuffer:
    global _BUF
    # Always (re)point at the supplied dir so the singleton can't get stuck on
    # a None/early dir (record_label relies on the configured _DATA_DIR).
    if data_dir is not None:
        configure(data_dir)
    if _BUF is None:
        _BUF = JunkLabelBuffer(data_dir=_DATA_DIR)
    return _BUF


def record_label(word: str, kind: str, glove_mag: Optional[float] = None,
                 meta: Optional[dict] = None) -> None:
    """Convenience entry from admission / prune code.

    kind in:
      'promoted'   -> topology positive (consolidated trace)
      'junk'       -> topology negative (junk-gated at admission)
      'pruned'     -> topology negative (sleep-degree decay)
      'hub'        -> topology positive (surviving hub)
    Grounding/pmi oracles are filled from meta when available.
    """
    buf = get_buffer(_DATA_DIR)
    topo = {"promoted": 1.0, "hub": 1.0, "junk": -1.0, "pruned": -1.0}.get(kind, 0.0)
    if topo == 0.0:
        return
    # External-grounding oracle (D1-2): a token anchored in WordNet's sense
    # inventory is a real dictionary word (distant positive); otherwise, fall
    # back to GloVe-magnitude (a word with a real embedding is also grounded).
    # Absent from both => distant negative (OOV / neologism / website-name).
    grounding = 0.0
    _wn = wordnet_grounding(word)
    if _wn > 0.0:
        grounding = 0.5
    elif glove_mag is not None and glove_mag >= 0.5:
        grounding = 0.5
    else:
        grounding = -0.5
    pmi = 0.0
    if meta:
        sc = meta.get("source_count")
        if sc is not None:
            pmi = 0.5 if sc >= 2 else -0.5
    buf.record(word, "topology", topo, meta=meta)
    if grounding:
        buf.record(word, "grounding", grounding, meta=meta)
    if pmi:
        buf.record(word, "pmi", pmi, meta=meta)


# ── public scorer (delegates to the fitted classifier; cold-start == current) ──
def junk_score(word: str, glove_mag: Optional[float] = None,
               degree: Optional[int] = None,
               source_count: Optional[int] = None,
               pmi_stability: Optional[float] = None) -> float:
    """Return junk probability in [0,1] for a candidate graph-node label.

    Cold-start (no fitted model / not ready): reproduces the Round-4 hand-
    weighted formula EXACTLY (structural floor + glove_low*0.35 + salad*0.4),
    so θ=0.5 behavior is unchanged. Once the classifier is ready (enough
    self-supervised labels), the SOFT features are combined by the learned
    logistic instead; the structural floor remains a hard backstop.
    """
    w = (word or "").lower().strip("'\"")
    if not w:
        return 1.0
    s = _structural_floor(w)
    if s >= 1.0:
        return 1.0
    clf = _get_classifier()
    if clf is not None and clf.ready:
        feat = _extract_features(w, glove_mag=glove_mag, degree=degree,
                                 source_count=source_count, pmi_stability=pmi_stability)
        soft = clf.predict(feat)
    else:
        # Cold-start soft combination == Round-4 hand weights.
        soft = 0.0
        if glove_mag is not None and glove_mag < 0.5:
            soft += 0.35
        soft += _salad_signal(w) * 0.4
    return min(1.0, s + soft)


def refit_now(force: bool = False) -> Optional[dict]:
    """Refit the global classifier from persisted + pending weak labels.

    Called from the sleep/consolidation cycle (every K cycles). Returns the
    monitor delta dict, or None if insufficient labels. Persists the model.
    """
    global _CLF
    buf = get_buffer(_DATA_DIR)
    emitted = buf.flush_aged()
    rows = buf.load_persisted()
    if len(rows) < OnlineJunkClassifier.MIN_LABELS and not force:
        return None
    # Clip to a bounded recent window for the fit (avoid unbounded growth).
    rows = rows[-4000:]
    X = np.array([buf.feature_row(w, m.get("glove_mag"), m) for (w, _, m) in rows])
    y = np.array([1.0 if lab == 1 else 0.0 for (_, lab, _) in rows])
    clf = _get_classifier() or OnlineJunkClassifier()
    delta = clf.refit(X, y)
    _set_classifier(clf)
    if _MODEL_PATH:
        try:
            with open(_MODEL_PATH, "w", encoding="utf-8") as f:
                json.dump(clf.to_dict(), f)
        except Exception:
            pass
    return delta
