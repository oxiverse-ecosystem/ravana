"""Multi-Signal Recall Relevance Scorer (MRS).

Ranks recall candidates from the PersonalFactStore using the existing
RelevanceScorer signals plus entity match bonus, query-type weighting,
and confidence weighting. All parameters live in persisted data stores
(SEED) — RAVANA can expand/revise them at runtime from conversation
feedback.

No LLM, no hardcoded reply logic, no per-topic tables, no retraining.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .relevance_store import (
    RelevanceScorer,
    RelevanceThreshold,
    _content_words,
    _jaccard_overlap,
    _glove_cosine,
    _mean_embedding,
    _type_alignment,
    record_false_positive,
    record_false_negative,
)


# ── Query-type patterns (reused from _structured_recall, no new patterns) ──
# Order matters: more specific patterns must come BEFORE more general ones.
_QUERY_TYPE_PATTERNS: List[Tuple[str, str]] = [
    ("material", r"\b(?:what'?s|what\s+is|what\s+material\s+is|what\s+is\s+the\s+material\s+of)\s+(?:my|the|our|your|a|an)?\s*\w+(?:'s)?\s+(?:made\s+of|made\s+from|material|built\s+of|built\s+from)\b"),
    ("count", r"\bhow\s+many\s+\w+"),
    ("whereabouts", r"\bwhere\s+do\s+i\s+(?:keep|have|store|keep\s+on|have\s+on|raise|grow|put|place)\b"),
    ("stance", r"\b(what\s+(?:did|do)\s+i\s+(?:think|feel)\s+(?:of|about)|"
               r"how\s+(?:do|did)\s+i\s+feel\s+about|"
               r"what(?:'s| is)\s+my\s+(?:opinion|stance)\s+(?:on|about|of))\b"),
    ("activity", r"\b(what\s+do\s+i\s+(?:keep|have|grind|forage|race|play|raise|grow|do|study|research|make|build|write|paint|carve|brew|bake|craft|teach|run|operate|manage|restore|tend|practice|code|design|volunteer|cook|fish|hike|garden|farm|lead|organize|collect|watch|read|learn|sail|knit|forge|fly))\b"),
    ("entity", r"\b(my\s+\w+(?:'s)?\s+(name|age|breed|job|work|is|does|location|"
               r"live|color|type|kind|favorite))\b"),
    ("possessive", r"\bmy\s+\w+(?:'s)?\s+(name|age|breed|job|work|is|does|location|live|color|type|kind|favorite)\b"),
    ("topical", r"\b(what\s+(?:did|do)\s+i\s+(?:tell|say)\s+(?:you|me)\s+about|"
                r"what\s+(?:do|did)\s+i\s+(?:think|feel)\s+(?:of|about)|"
                r"how\s+(?:do|did)\s+i\s+feel\s+about|"
                r"what(?:'s| is)\s+my\s+(?:opinion|stance)\s+(?:on|about|of)|"
                r"do\s+you\s+(?:remember|recall)\s+what\s+(?:i\s+)?(?:said|mentioned|told|shared)\s+(?:you\s+)?about)\b"),
    ("biographical", r"\b(my name|who am i|what am i called|where do i live|"
                      r"what city|what town|where am i from|where did i grow up|"
                      r"where do i work|what do i do|what's my job|what is my work)\b"),
]

_QUERY_TYPE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "biographical": {"content": 1.2, "graph": 0.8, "recency": 0.6, "type": 1.3},
    "entity": {"content": 1.3, "graph": 0.7, "recency": 0.5, "type": 1.4},
    "topical": {"content": 1.1, "graph": 1.2, "recency": 0.8, "type": 0.9},
    "stance": {"content": 0.9, "graph": 1.3, "recency": 0.7, "type": 1.2},
    "activity": {"content": 1.0, "graph": 1.0, "recency": 0.9, "type": 1.0},
    "possessive": {"content": 1.2, "graph": 0.8, "recency": 0.6, "type": 1.3},
    "material": {"content": 1.1, "graph": 1.0, "recency": 0.7, "type": 1.1},
    "count": {"content": 1.0, "graph": 0.9, "recency": 0.8, "type": 1.0},
    "whereabouts": {"content": 1.1, "graph": 0.9, "recency": 0.7, "type": 1.1},
}

_DEFAULT_TYPE_WEIGHTS = {"content": 1.0, "graph": 1.0, "recency": 1.0, "type": 1.0}


@dataclass
class RecallCandidate:
    """A single recall candidate with all signals needed for ranking."""
    answer: str
    fact_key: Tuple[str, str, str]
    fact_value: str
    turn_number: int = 0
    confidence: float = 0.6
    entity: Optional[str] = None
    candidate_type: Optional[str] = None
    source: str = "structured_recall"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "fact_key": list(self.fact_key),
            "fact_value": self.fact_value,
            "turn_number": self.turn_number,
            "confidence": self.confidence,
            "entity": self.entity,
            "candidate_type": self.candidate_type,
            "source": self.source,
        }


class RecallRanker:
    """Multi-signal recall relevance scorer.

    Ranks recall candidates using:
    1. Content-word overlap (Jaccard) — from RelevanceScorer
    2. Graph/semantic distance (GloVe cosine) — from RelevanceScorer
    3. Recency (exponential decay) — from RelevanceScorer
    4. Type match (entity/attribute alignment) — from RelevanceScorer
    5. Entity match bonus — NEW: query entity word in candidate fact key
    6. Confidence weighting — NEW: fact confidence from PersonalFactStore
    7. Query-type weighting — NEW: per-type signal weight adjustments

    The ranker is stateless; all learned parameters live in the
    RelevanceThreshold data store. Learning is online/incremental.
    """

    def __init__(self, glove_fn=None, threshold: Optional[RelevanceThreshold] = None):
        self._glove_fn = glove_fn
        self._threshold = threshold or RelevanceThreshold()
        self._scorer = RelevanceScorer(glove_fn=glove_fn)

    def detect_query_type(self, query: str) -> str:
        """Classify the query type using the SAME regex patterns that
        _structured_recall already uses. No new patterns."""
        q = (query or "").lower().strip()
        if not q:
            return "unknown"
        for qtype, pattern in _QUERY_TYPE_PATTERNS:
            if re.search(pattern, q):
                return qtype
        return "unknown"

    def _entity_match_bonus(self, query: str, candidate: RecallCandidate) -> float:
        """Entity match bonus: if the query contains an entity word that
        appears in the candidate's fact key, add a bonus."""
        q = (query or "").lower().strip()
        if not q:
            return 0.0
        q_words = _content_words(q)
        if not q_words:
            return 0.0
        _key_str = " ".join(str(p) for p in candidate.fact_key).lower()
        _key_words = set(re.findall(r"[a-z']+", _key_str))
        _matches = q_words & _key_words
        if not _matches:
            return 0.0
        _bonus = len(_matches) / max(1, len(q_words))
        return min(0.15, _bonus * 0.15)

    def _confidence_weight(self, candidate: RecallCandidate) -> float:
        """Confidence weighting: facts with higher confidence get a small bonus."""
        conf = max(0.0, min(1.0, candidate.confidence))
        return conf * 0.10

    def _query_type_weights(self, query_type: str) -> Dict[str, float]:
        """Get the signal weight adjustments for a query type."""
        return _QUERY_TYPE_WEIGHTS.get(query_type, _DEFAULT_TYPE_WEIGHTS)

    def rank(
        self,
        candidates: List[RecallCandidate],
        query: str,
        query_type: Optional[str] = None,
        query_turn: int = 0,
    ) -> List[RecallCandidate]:
        """Rank candidates by multi-signal relevance."""
        if not candidates:
            return []
        if query_type is None:
            query_type = self.detect_query_type(query)
        _type_weights = self._query_type_weights(query_type)
        scored: List[Tuple[float, RecallCandidate]] = []
        for cand in candidates:
            _content = _jaccard_overlap(
                _content_words(query), _content_words(cand.answer))
            q_vec = _mean_embedding(_content_words(query), self._glove_fn)
            c_vec = _mean_embedding(_content_words(cand.answer), self._glove_fn)
            _graph = _glove_cosine(q_vec, c_vec)
            _recency = math.exp(-0.01 * abs(query_turn - cand.turn_number))
            _type = _type_alignment(query_type, cand.candidate_type)
            _content_w = self._threshold.content_weight * _type_weights.get("content", 1.0)
            _graph_w = self._threshold.graph_weight * _type_weights.get("graph", 1.0)
            _recency_w = self._threshold.recency_weight * _type_weights.get("recency", 1.0)
            _type_w = self._threshold.type_weight * _type_weights.get("type", 1.0)
            score = (_content_w * _content +
                     _graph_w * _graph +
                     _recency_w * _recency +
                     _type_w * _type)
            score += self._entity_match_bonus(query, cand)
            score += self._confidence_weight(cand)
            score = min(1.0, max(0.0, score))
            scored.append((score, cand))
        scored.sort(key=lambda x: (x[0], x[1].confidence), reverse=True)
        return [c for _, c in scored]

    def rank_and_filter(
        self,
        candidates: List[RecallCandidate],
        query: str,
        query_type: Optional[str] = None,
        query_turn: int = 0,
    ) -> List[RecallCandidate]:
        """Rank candidates and filter out those below the threshold."""
        ranked = self.rank(candidates, query, query_type, query_turn)
        threshold = self._threshold.base_threshold
        return [c for c in ranked if self._score_candidate(c, query, query_type, query_turn) >= threshold]

    def score_text(
        self,
        query: str,
        answer: str,
        query_type: Optional[str] = None,
        query_turn: int = 0,
    ) -> float:
        """Score a plain text answer against the query (MRS scoring)."""
        if not answer:
            return 0.0
        if query_type is None:
            query_type = self.detect_query_type(query)
        _type_weights = self._query_type_weights(query_type)
        _content = _jaccard_overlap(
            _content_words(query), _content_words(answer))
        q_vec = _mean_embedding(_content_words(query), self._glove_fn)
        c_vec = _mean_embedding(_content_words(answer), self._glove_fn)
        _graph = _glove_cosine(q_vec, c_vec)
        _recency = math.exp(-0.01 * abs(query_turn - 0))
        _type = _type_alignment(query_type, None)
        _content_w = self._threshold.content_weight * _type_weights.get("content", 1.0)
        _graph_w = self._threshold.graph_weight * _type_weights.get("graph", 1.0)
        _recency_w = self._threshold.recency_weight * _type_weights.get("recency", 1.0)
        _type_w = self._threshold.type_weight * _type_weights.get("type", 1.0)
        score = (_content_w * _content +
                 _graph_w * _graph +
                 _recency_w * _recency +
                 _type_w * _type)
        q_words = _content_words(query)
        a_words = _content_words(answer)
        if q_words and a_words:
            _matches = q_words & a_words
            if _matches:
                score += min(0.15, (len(_matches) / max(1, len(q_words))) * 0.15)
        score += 0.6 * 0.10
        return min(1.0, max(0.0, score))

    def _score_candidate(
        self,
        candidate: RecallCandidate,
        query: str,
        query_type: Optional[str] = None,
        query_turn: int = 0,
    ) -> float:
        """Compute the score for a single candidate."""
        if query_type is None:
            query_type = self.detect_query_type(query)
        _type_weights = self._query_type_weights(query_type)
        _content = _jaccard_overlap(
            _content_words(query), _content_words(candidate.answer))
        q_vec = _mean_embedding(_content_words(query), self._glove_fn)
        c_vec = _mean_embedding(_content_words(candidate.answer), self._glove_fn)
        _graph = _glove_cosine(q_vec, c_vec)
        _recency = math.exp(-0.01 * abs(query_turn - candidate.turn_number))
        _type = _type_alignment(query_type, candidate.candidate_type)
        _content_w = self._threshold.content_weight * _type_weights.get("content", 1.0)
        _graph_w = self._threshold.graph_weight * _type_weights.get("graph", 1.0)
        _recency_w = self._threshold.recency_weight * _type_weights.get("recency", 1.0)
        _type_w = self._threshold.type_weight * _type_weights.get("type", 1.0)
        score = (_content_w * _content +
                 _graph_w * _graph +
                 _recency_w * _recency +
                 _type_w * _type)
        score += self._entity_match_bonus(query, candidate)
        score += self._confidence_weight(candidate)
        return min(1.0, max(0.0, score))

    def collect_candidates(
        self,
        query: str,
        personal_fact_store: Any,
        max_candidates: int = 10,
    ) -> List[RecallCandidate]:
        """Collect recall candidates from the PersonalFactStore."""
        if personal_fact_store is None:
            return []
        q = (query or "").lower().strip()
        if not q:
            return []
        q_words = _content_words(q)
        if not q_words:
            return []
        candidates: List[RecallCandidate] = []
        _seen_keys: Set[Tuple[str, str, str]] = set()
        try:
            facts = getattr(personal_fact_store, "facts", {})
            if not facts:
                return []
            for _key, _fact in facts.items():
                if not (isinstance(_key, tuple) and len(_key) == 3):
                    continue
                if getattr(_fact, "superseded", False):
                    continue
                _subj, _attr, _val = _key
                _val_str = (getattr(_fact, "value", _val) or "").strip()
                if not _val_str:
                    continue
                _fact_text = f"{_subj} {_attr} {_val_str}".lower()
                _fact_words = set(re.findall(r"[a-z']+", _fact_text))
                _overlap = q_words & _fact_words
                if not _overlap:
                    continue
                _entity = _subj if _subj not in ("i", "me", "my", "you") else None
                _cand_type = self._attr_to_type(_attr)
                _answer = self._render_candidate_answer(_subj, _attr, _val_str, q)
                cand = RecallCandidate(
                    answer=_answer,
                    fact_key=_key,
                    fact_value=_val_str,
                    turn_number=getattr(_fact, "turn_number", 0),
                    confidence=getattr(_fact, "confidence", 0.6),
                    entity=_entity,
                    candidate_type=_cand_type,
                    source="structured_recall",
                )
                if _key not in _seen_keys:
                    _seen_keys.add(_key)
                    candidates.append(cand)
                if len(candidates) >= max_candidates:
                    break
        except Exception:
            pass
        return candidates

    def _attr_to_type(self, attr: str) -> Optional[str]:
        """Map a fact attribute to a candidate type for type alignment."""
        _attr_l = attr.lower()
        if _attr_l in ("name", "named", "called"):
            return "name"
        if _attr_l in ("location", "live", "live in", "from"):
            return "location"
        if _attr_l in ("work", "job", "occupation"):
            return "work"
        if _attr_l in ("favorite", "likes", "like"):
            return "preference"
        if _attr_l in ("does", "event", "is", "has"):
            return "activity"
        if _attr_l in ("age", "breed", "color", "type", "kind"):
            return "attribute"
        return None

    def _render_candidate_answer(self, subj: str, attr: str, val: str, query: str) -> str:
        """Render a candidate answer from the fact store."""
        if subj == "i":
            if attr == "name":
                return f"your name is {val}."
            if attr == "location":
                return f"you live in {val}."
            if attr == "work":
                return f"you work as {val}."
            if attr == "favorite":
                return f"your favorite is {val}."
            if attr == "likes":
                return f"you like {val}."
            if attr == "does" or attr.startswith("does:"):
                return f"you {val}."
            if attr == "is":
                return f"you are {val}."
            if attr.startswith("event"):
                return f"you {val}."
            return f"your {attr} is {val}."
        else:
            if attr == "name":
                return f"your {subj}'s name is {val}."
            if attr == "location":
                return f"you live in {val}."
            if attr == "favorite":
                return f"your favorite {subj} is {val}."
            if attr == "likes":
                return f"you mentioned you like {val}."
            if attr == "does" or attr.startswith("does:"):
                return f"your {subj} does {val}."
            if attr == "is":
                return f"your {subj} is {val}."
            if attr.startswith("event"):
                return f"you {val}."
            return f"your {subj}'s {attr} is {val}."

    def wire_feedback(
        self,
        correction_type: str,
        severity: float = 0.5,
        current_turn: int = 0,
    ) -> None:
        """Wire a correction feedback signal to the threshold adaptation."""
        _adapted_rate = self._threshold.adaptation_rate * (1.0 + severity)
        old_rate = self._threshold.adaptation_rate
        self._threshold.adaptation_rate = min(0.05, _adapted_rate)
        record_false_positive(self._threshold, current_turn)
        self._threshold.adaptation_rate = old_rate

    def record_true_positive(self, current_turn: int = 0) -> None:
        """Record a true positive signal (user confirmed a recall)."""
        self._threshold.total_scored += 1
        if self._threshold.total_scored >= 10:
            from .relevance_store import adapt_threshold
            adapt_threshold(self._threshold, current_turn)
