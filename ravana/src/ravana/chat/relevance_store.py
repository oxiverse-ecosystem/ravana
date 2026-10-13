"""SEED-backed relevance-scored recall layer.

Multi-signal relevance scoring for the recall pipeline. Replaces binary
pass/fail guards with a continuous score and an autonomously adapting
threshold. All parameters live in a persisted data store (SEED) — RAVANA
can expand/revise them at runtime from conversation feedback.

No LLM, no hardcoded reply logic, no per-topic tables, no retraining.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ── Stopwords for content-word extraction (shared philosophy with engine) ──
_CONTENT_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "once", "here", "there", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "can",
    "will", "just", "should", "now", "i", "me", "my", "we", "our",
    "you", "your", "he", "him", "his", "she", "her", "it", "its",
    "they", "them", "their", "this", "that", "these", "those", "am",
    "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doing", "would", "could",
    "shall", "may", "might", "must", "of", "as", "what", "which",
    "who", "whom", "how", "where", "when", "why", "tell", "remember",
    "recall", "remind", "mentioned", "said", "told", "know", "think",
    "feel", "like", "love", "hate", "want", "need", "get", "got",
    "make", "made", "go", "went", "come", "came", "see", "saw",
    "look", "looked", "find", "found", "give", "gave", "take", "took",
    "something", "anything", "everything", "nothing", "thing", "things",
    "stuff", "really", "very", "quite", "pretty", "kind", "sort",
    "way", "lot", "bit", "much", "many", "still", "even", "also",
    "back", "away", "around", "together", "already", "yet", "ever",
    "never", "always", "sometimes", "often", "usually", "maybe",
    "perhaps", "probably", "actually", "basically", "literally",
    "honestly", "seriously", "definitely", "certainly", "obviously",
    "clearly", "apparently", "supposedly", "presumably", "the",
}


def _content_words(text: str) -> Set[str]:
    """Extract content words from text (lowercase, stopword-removed)."""
    if not text:
        return set()
    words = re.findall(r"[a-z']+", text.lower())
    return {w.strip(".,!?;:") for w in words
            if len(w) >= 3 and w not in _CONTENT_STOPWORDS}


def _jaccard_overlap(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity between two word sets."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def _glove_cosine(vec_a: Optional[np.ndarray],
                  vec_b: Optional[np.ndarray]) -> float:
    """Cosine similarity between two vectors. Returns 0.0 if either is None."""
    if vec_a is None or vec_b is None:
        return 0.0
    a = np.asarray(vec_a, dtype=np.float64).flatten()
    b = np.asarray(vec_b, dtype=np.float64).flatten()
    if a.shape != b.shape or a.size == 0:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _mean_embedding(words: Set[str],
                    glove_fn) -> Optional[np.ndarray]:
    """Compute mean GloVe embedding for a set of words."""
    if not words or glove_fn is None:
        return None
    vecs = []
    for w in words:
        v = glove_fn(w)
        if v is not None:
            vecs.append(np.asarray(v, dtype=np.float64).flatten())
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def _type_alignment(query_type: Optional[str],
                    candidate_type: Optional[str]) -> float:
    """Type match score: 1.0 if same, 0.5 if related, 0.0 if unrelated."""
    if not query_type or not candidate_type:
        return 0.5  # neutral when type unknown
    if query_type == candidate_type:
        return 1.0
    # Related types (e.g., "pet" and "animal", "person" and "friend")
    _RELATED_TYPES = {
        ("pet", "animal"), ("animal", "pet"),
        ("person", "friend"), ("friend", "person"),
        ("person", "family"), ("family", "person"),
        ("possession", "object"), ("object", "possession"),
        ("place", "location"), ("location", "place"),
        ("activity", "hobby"), ("hobby", "activity"),
    }
    if (query_type, candidate_type) in _RELATED_TYPES:
        return 0.5
    return 0.0


@dataclass
class RelevanceThreshold:
    """Learned threshold for relevance gating. Updated incrementally.

    SEED knowledge: all values are stored in a persisted data store,
    not in if/elif branches. RAVANA can adjust base_threshold up/down
    based on conversation feedback (false positive/negative signals).
    """
    base_threshold: float = 0.15
    content_weight: float = 0.40
    graph_weight: float = 0.25
    recency_weight: float = 0.20
    type_weight: float = 0.15
    min_candidates: int = 1
    adaptation_rate: float = 0.01
    false_positive_count: int = 0
    false_negative_count: int = 0
    total_scored: int = 0
    last_adapted_turn: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_threshold": self.base_threshold,
            "content_weight": self.content_weight,
            "graph_weight": self.graph_weight,
            "recency_weight": self.recency_weight,
            "type_weight": self.type_weight,
            "min_candidates": self.min_candidates,
            "adaptation_rate": self.adaptation_rate,
            "false_positive_count": self.false_positive_count,
            "false_negative_count": self.false_negative_count,
            "total_scored": self.total_scored,
            "last_adapted_turn": self.last_adapted_turn,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RelevanceThreshold":
        return cls(
            base_threshold=float(d.get("base_threshold", 0.15)),
            content_weight=float(d.get("content_weight", 0.40)),
            graph_weight=float(d.get("graph_weight", 0.25)),
            recency_weight=float(d.get("recency_weight", 0.20)),
            type_weight=float(d.get("type_weight", 0.15)),
            min_candidates=int(d.get("min_candidates", 1)),
            adaptation_rate=float(d.get("adaptation_rate", 0.01)),
            false_positive_count=int(d.get("false_positive_count", 0)),
            false_negative_count=int(d.get("false_negative_count", 0)),
            total_scored=int(d.get("total_scored", 0)),
            last_adapted_turn=int(d.get("last_adapted_turn", 0)),
        )


class RelevanceScorer:
    """Multi-signal relevance scorer for recall candidates.

    Computes a continuous relevance score in [0, 1] based on:
    1. Content-word overlap (Jaccard)
    2. Graph/semantic distance (GloVe cosine)
    3. Recency (exponential decay by turn distance)
    4. Type match (entity/attribute alignment)

    The scorer is stateless; all learned parameters live in the
    RelevanceThreshold data store.
    """

    def __init__(self, glove_fn=None):
        self._glove_fn = glove_fn

    def score(self,
              query: str,
              candidate_text: str,
              candidate_turn: int = 0,
              query_turn: int = 0,
              query_type: Optional[str] = None,
              candidate_type: Optional[str] = None,
              threshold: Optional[RelevanceThreshold] = None) -> float:
        """Compute multi-signal relevance score in [0, 1]."""
        if threshold is None:
            threshold = RelevanceThreshold()

        q_words = _content_words(query)
        c_words = _content_words(candidate_text)

        # Signal 1: Content overlap (Jaccard)
        _content = _jaccard_overlap(q_words, c_words)

        # Signal 2: Graph distance (GloVe cosine of mean embeddings)
        q_vec = _mean_embedding(q_words, self._glove_fn)
        c_vec = _mean_embedding(c_words, self._glove_fn)
        _graph = _glove_cosine(q_vec, c_vec)

        # Signal 3: Recency (exponential decay)
        _recency = math.exp(-0.01 * abs(query_turn - candidate_turn))

        # Signal 4: Type match
        _type = _type_alignment(query_type, candidate_type)

        # Weighted combination
        score = (threshold.content_weight * _content +
                 threshold.graph_weight * _graph +
                 threshold.recency_weight * _recency +
                 threshold.type_weight * _type)

        return min(1.0, max(0.0, score))

    def score_with_signals(self,
                           query: str,
                           candidate_text: str,
                           candidate_turn: int = 0,
                           query_turn: int = 0,
                           query_type: Optional[str] = None,
                           candidate_type: Optional[str] = None,
                           threshold: Optional[RelevanceThreshold] = None
                           ) -> Tuple[float, Dict[str, float]]:
        """Compute score and return individual signal values for debugging."""
        if threshold is None:
            threshold = RelevanceThreshold()

        q_words = _content_words(query)
        c_words = _content_words(candidate_text)

        _content = _jaccard_overlap(q_words, c_words)
        q_vec = _mean_embedding(q_words, self._glove_fn)
        c_vec = _mean_embedding(c_words, self._glove_fn)
        _graph = _glove_cosine(q_vec, c_vec)
        _recency = math.exp(-0.01 * abs(query_turn - candidate_turn))
        _type = _type_alignment(query_type, candidate_type)

        score = (threshold.content_weight * _content +
                 threshold.graph_weight * _graph +
                 threshold.recency_weight * _recency +
                 threshold.type_weight * _type)

        signals = {
            "content": _content,
            "graph": _graph,
            "recency": _recency,
            "type": _type,
        }
        return min(1.0, max(0.0, score)), signals


def adapt_threshold(threshold: RelevanceThreshold,
                    current_turn: int = 0) -> None:
    """Adjust threshold based on accumulated error signals.

    If false positives dominate (returning irrelevant results), RAISE
    the threshold (be more selective). If false negatives dominate
    (missing relevant results), LOWER the threshold (be more permissive).

    Simple gradient descent on the error rate — no retraining, just
    incremental parameter adjustment.
    """
    if threshold.total_scored < 10:
        return  # not enough data yet

    fp_rate = threshold.false_positive_count / threshold.total_scored
    fn_rate = threshold.false_negative_count / threshold.total_scored

    if fp_rate > fn_rate + 0.05:
        # Too many false positives — be more selective
        threshold.base_threshold = min(0.5,
                                       threshold.base_threshold + threshold.adaptation_rate)
    elif fn_rate > fp_rate + 0.05:
        # Too many false negatives — be more permissive
        threshold.base_threshold = max(0.05,
                                      threshold.base_threshold - threshold.adaptation_rate)

    threshold.last_adapted_turn = current_turn


def record_false_positive(threshold: RelevanceThreshold,
                          current_turn: int = 0) -> None:
    """Record a false positive signal (returned irrelevant result)."""
    threshold.false_positive_count += 1
    threshold.total_scored += 1
    adapt_threshold(threshold, current_turn)


def record_false_negative(threshold: RelevanceThreshold,
                          current_turn: int = 0) -> None:
    """Record a false negative signal (missed relevant result)."""
    threshold.false_negative_count += 1
    threshold.total_scored += 1
    adapt_threshold(threshold, current_turn)
