from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum


class CorrectionType(Enum):
    """Type of correction detected from user feedback."""
    DIRECT = "direct"          # Explicit "no", "that's wrong", "that's not right"
    INDIRECT_REASK = "reask"   # User rephrases same question (implies dissatisfaction)
    SENTIMENT_DROP = "sentiment"  # Emotional valence drops after response
    CORRECTION_WITH_FACT = "fact"  # User explicitly provides corrected fact


@dataclass
class Correction:
    """Record of a user correction for consolidation and learning."""
    turn: int
    correction_type: CorrectionType
    subject: str
    incorrect_response: str
    user_correction_text: str
    corrected_fact: Optional[Tuple[str, str, str]] = None  # (subject, relation, correct_value)
    severity: float = 0.5
    resolved: bool = False
    weakened_edges: List[Tuple[int, int]] = field(default_factory=list)
    added_edges: List[Tuple[int, int]] = field(default_factory=list)
    web_verified: bool = False


@dataclass
class FailedQuery:
    query: str = ""
    subject: str = ""
    activated_concepts: List[str] = field(default_factory=list)
    strategies_tried: List[str] = field(default_factory=list)
    best_guess_response: str = ""
    turn: int = 0
    free_energy_at_time: float = 0.0
    resolved: bool = False
    response_quality: float = 0.0
    strategy: str = ""


@dataclass
class ChainHop:
    from_label: str
    to_label: str
    relation_type: str
    weight: float
    confidence: float
    temperature: float
    candidates: int
    rlm_confidence: float = 0.0
    contradiction: str = ""


@dataclass
class ChainTrace:
    hops: List[ChainHop] = field(default_factory=list)
    max_hops: int = 0
    completed: bool = False


@dataclass
class CognitiveResponseContext:
    subject: str = ""
    relation: str = ""
    object: str = ""
    raw_input: str = ""
    associated_concepts: List[Tuple[str, float]] = field(default_factory=list)
    bridge_concept: str = ""
    valence: float = 0.0
    arousal: float = 0.3
    dominance: float = 0.5
    emotional_label: str = "neutral"
    identity_strength: float = 0.5
    identity_trend: float = 0.0
    dissonance: float = 0.5
    processing_route: str = "system1_fast"
    route_reason: str = "default"
    past_topics: List[str] = field(default_factory=list)
    turn_count: int = 0
    meaning_generated: float = 0.0
    exploration_drive: float = 0.0
    learned_recently: bool = False
    recall_mode: bool = False
    sentence_vector: Any = None
    discourse_context: str = ""
    content_vector: Any = None
    context_vector: Any = None
    fok_unresolved: bool = False
    situation_vector: Any = None
    situation_narrative: Dict[str, Any] = field(default_factory=dict)
    decomposition: Any = None
    sub_questions: List[Dict[str, Any]] = field(default_factory=list)
