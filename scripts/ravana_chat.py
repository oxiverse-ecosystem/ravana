#!/usr/bin/env python3
"""
RAVANA Baby — cognitive chat that starts like a baby and learns from the web
============================================================================
No commands. No LLM. Pure RAVANA cognitive architecture.
Starts knowing ~180 teen-level concepts with GloVe semantic embeddings
and typed graph edges (causal, contrastive, analogical, temporal, semantic),
auto-learns from the internet when it doesn't know something.

Usage:
    python scripts/ravana_chat.py
"""

import sys, os, time, random, json, re, argparse, pickle
import urllib.request
from urllib.parse import quote
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _proj_root)
sys.path.insert(0, os.path.join(_proj_root, "ravana-v2"))

from ravana_ml.graph import ConceptGraph, ConceptEdge
from core.emotion import VADEmotionEngine, VADConfig
from core.identity import IdentityEngine
from core.meaning import MeaningEngine, MeaningConfig
from core.dual_process import DualProcessController, DualProcessConfig
from core.global_workspace import GlobalWorkspace, GWConfig
from core.meta_cognition import MetaCognition, MetaCognitiveConfig, EpistemicMode
from core.sleep import SleepConsolidation, SleepConfig


# Try importing beautifulsoup4 for HTML parsing (optional but recommended)
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

SEARCH_API = "https://api.oxiverse.com/search?q="


# ─── Teen Vocabulary: what a teenager knows (~180 words) ───
TEEN_CONCEPTS = [
    # ── Social & Identity ──
    ("hello", "hi hey greeting sup"),
    ("bye", "goodbye farewell later"),
    ("yes", "okay yeah agree absolutely"),
    ("no", "nope negative disagree"),
    ("please", "polite request kindly"),
    ("thanks", "thank appreciate gratitude"),
    ("sorry", "apologize forgive regret"),
    ("i", "me myself my own"),
    ("you", "your yourself thou"),
    ("we", "us our together group"),
    ("they", "them their other people"),
    ("friend", "buddy pal companion ally"),
    ("people", "human society community crowd"),
    ("person", "individual human being someone"),

    # ── Advanced Social Concepts ──
    ("trust", "rely faith confidence belief"),
    ("justice", "fairness equality right moral"),
    ("hypocrisy", "contradict inconsistent double standard fake"),
    ("empathy", "compassion understanding feeling care"),
    ("respect", "admire honor esteem regard"),
    ("identity", "self character personality who"),
    ("culture", "tradition society custom heritage"),
    ("power", "control influence authority strength"),
    ("freedom", "liberty choice independence right"),
    ("responsibility", "duty obligation accountability burden"),

    # ── Epistemic & Abstract ──
    ("truth", "real fact honest genuine accurate"),
    ("belief", "faith opinion conviction view"),
    ("knowledge", "wisdom understanding awareness learning"),
    ("meaning", "purpose significance essence point"),
    ("pattern", "structure repetition system cycle"),
    ("system", "network framework structure organization"),
    ("perspective", "viewpoint angle lens outlook"),
    ("context", "situation background setting circumstance"),
    ("paradox", "contradiction puzzle ironic dilemma"),
    ("principle", "rule value standard moral axiom"),
    ("theory", "hypothesis idea framework explanation"),
    ("evidence", "proof data fact support clue"),
    ("analysis", "examination study breakdown evaluation"),
    ("conclusion", "result inference deduction summation"),
    ("logic", "reason rational sense coherence"),
    ("intuition", "instinct gut feeling hunch"),
    ("wisdom", "insight knowledge judgment prudence"),

    # ── Abstract Qualities ──
    ("complex", "complicated intricate sophisticated layered"),
    ("significant", "important meaningful major notable"),
    ("fundamental", "basic essential core foundation"),
    ("inevitable", "unavoidable certain destined fated"),
    ("possible", "maybe potential feasible plausible"),
    ("obvious", "clear apparent evident obvious"),
    ("subtle", "nuanced delicate faint indirect"),
    ("profound", "deep meaningful significant thoughtful"),

    # ── Core Verbs (expanded) ──
    ("want", "wish desire need crave"),
    ("like", "enjoy love prefer appreciate"),
    ("go", "move leave walk proceed"),
    ("come", "arrive approach appear"),
    ("see", "look watch observe perceive"),
    ("hear", "listen sound overhear"),
    ("eat", "food meal consume devour"),
    ("drink", "water thirsty sip beverage"),
    ("sleep", "rest nap bed unconscious"),
    ("play", "fun game toy recreation"),
    ("help", "assist aid support serve"),
    ("make", "create build produce cause"),
    ("get", "receive obtain acquire understand"),
    ("know", "understand aware learn recognize"),
    ("think", "believe consider wonder reason"),
    ("say", "tell speak talk express"),
    ("feel", "sense emotion touch experience"),
    ("love", "care affection adore cherish"),
    ("give", "share present offer donate"),
    ("take", "grab seize accept choose"),

    # ── Advanced Verbs ──
    ("analyze", "examine study evaluate break down"),
    ("conclude", "decide infer deduce determine"),
    ("reflect", "ponder contemplate meditate consider"),
    ("question", "challenge doubt inquire interrogate"),
    ("explore", "discover investigate venture search"),
    ("understand", "comprehend grasp realize fathom"),
    ("compare", "contrast relate match evaluate"),
    ("criticize", "judge critique evaluate assess"),
    ("assume", "presume suppose guess speculate"),
    ("imagine", "envision dream visualize conceive"),
    ("connect", "relate associate link bridge"),
    ("influence", "affect shape impact sway"),
    ("struggle", "fight conflict strive contend"),
    ("challenge", "dare confront test oppose"),

    # ── Core Nouns (expanded) ──
    ("water", "drink wet rain liquid"),
    ("food", "eat meal snack nutrition"),
    ("home", "house room family shelter"),
    ("sun", "light warm day star"),
    ("moon", "night star dark lunar"),
    ("tree", "plant leaf flower forest"),
    ("bird", "fly animal feather wing"),
    ("dog", "pet puppy bark canine"),
    ("cat", "kitten meow pet feline"),
    ("book", "read story page novel"),
    ("song", "music sing melody rhythm"),
    ("world", "earth globe planet universe"),
    ("nature", "environment wild natural earth"),
    ("time", "clock moment age duration"),
    ("life", "living existence being survive"),
    ("death", "die end mortality passing"),
    ("mind", "brain thought consciousness psyche"),
    ("heart", "organ emotion core center"),
    ("science", "study research knowledge method"),
    ("history", "past story legacy record"),
    ("art", "creative expression beauty culture"),

    # ── Abstract Verbs ──
    ("cause", "produce create generate result"),
    ("change", "transform shift modify evolve"),
    ("grow", "develop expand mature increase"),
    ("learn", "study discover understand master"),
    ("teach", "educate instruct explain mentor"),
    ("create", "make invent produce generate"),
    ("destroy", "ruin break eliminate devastate"),
    ("protect", "defend guard shield secure"),
    ("accept", "embrace welcome acknowledge agree"),
    ("reject", "refuse deny dismiss decline"),

    # ── Adjectives (expanded) ──
    ("good", "nice great fine positive"),
    ("bad", "wrong negative evil harmful"),
    ("big", "large huge giant massive"),
    ("small", "tiny little mini slight"),
    ("hot", "warm burn fire heated"),
    ("cold", "cool freeze ice chilly"),
    ("happy", "joy glad smile content"),
    ("sad", "cry unhappy upset sorrow"),
    ("scared", "afraid fear frighten anxious"),
    ("angry", "furious mad frustrated rage"),
    ("tired", "sleepy exhausted fatigue drained"),
    ("excited", "eager enthusiastic thrilled pumped"),
    ("curious", "interested inquisitive nosy wonder"),
    ("confused", "lost puzzled baffled uncertain"),
    ("bored", "uninterested dull tired weary"),
    ("proud", "accomplished satisfied dignified confident"),
    ("lonely", "isolated alone abandoned disconnected"),
    ("grateful", "thankful appreciative indebted blessed"),

    # ── Emotion Words ──
    ("anxiety", "worry nervous tension stress"),
    ("excitement", "thrill enthusiasm anticipation energy"),
    ("frustration", "annoyance irritation aggravation anger"),
    ("hope", "optimism aspiration wish dream"),
    ("fear", "terror dread panic horror"),
    ("joy", "delight happiness bliss pleasure"),
    ("grief", "sorrow loss mourning lament"),
    ("surprise", "shock amazement astonishment wonder"),
    ("guilt", "remorse regret shame blame"),
    ("disappointment", "letdown regret dissatisfaction dismay"),

    # ── Imagination & Future ──
    ("future", "tomorrow ahead later coming"),
    ("past", "history ago previous yesterday"),
    ("machine", "device engine mechanism robot"),
    ("invention", "creation innovation discovery breakthrough"),
    ("invent", "create design devise pioneer"),
    ("possibility", "potential chance opportunity likelihood"),
    ("imagination", "creativity fantasy vision dream"),
    ("impossible", "unlikely hopeless absurd ridiculous"),
    ("journey", "travel adventure voyage quest"),
    ("secret", "hidden mystery private unknown"),
    ("experiment", "trial test attempt investigation"),

    # ── Relational / Meta ──
    ("and", "also plus together"),
    ("so", "therefore thus hence"),
    ("then", "next after afterwards"),
    ("link", "connect join bond tie"),
    ("why", "reason because explanation cause"),
    ("how", "method way process means"),
    ("what", "which thing object identity"),
    ("if", "suppose whether maybe perhaps"),
    ("but", "however yet although though"),
    ("because", "since due cause reason"),
    ("maybe", "perhaps possibly probably could"),
    ("always", "forever constant perpetual eternal"),
    ("never", "not once zero none"),

    # ── Basic concepts (retained) ──
    ("up", "above high sky rise"),
    ("down", "below low ground fall"),
    ("in", "inside within interior"),
    ("out", "outside exit exterior"),
    ("here", "this place near present"),
    ("there", "that place far distant"),
    ("now", "today present moment current"),
    ("later", "soon future after eventual"),
    ("more", "extra additional plus further"),
    ("all", "every everything whole total"),
    ("some", "few several part partial"),
    ("one", "single first unique individual"),
    ("two", "second pair both double"),
    ("many", "multiple numerous several abundant"),
]

# ─── Stop words to filter during article reading ───
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "were", "be", "been",
    "are", "am", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "this",
    "that", "these", "those", "it", "its", "they", "them", "their",
    "we", "our", "you", "your", "he", "she", "him", "her", "his",
    "not", "no", "nor", "so", "if", "then", "than", "too", "very",
    "just", "about", "also", "into", "over", "after", "before",
    "between", "through", "during", "because", "while", "which",
    "who", "whom", "what", "when", "where", "why", "how", "all",
    "each", "every", "both", "few", "more", "most", "some", "any",
}


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

@dataclass
class ChainHop:
    from_label: str
    to_label: str
    relation_type: str
    weight: float
    confidence: float
    temperature: float
    candidates: int  # how many edges were considered at this hop
    rlm_confidence: float = 0.0  # RLMv2 triple verification score
    contradiction: str = ""  # contradiction detail if detected

@dataclass
class ChainTrace:
    hops: List[ChainHop] = field(default_factory=list)
    max_hops: int = 0
    completed: bool = False

@dataclass
class UserModel:
    """Tracks user interaction patterns via edge reactivation frequency.

    No hardcoded patterns — preferences emerge from graph traversal stats.
    """
    edge_reactivations: Dict[Tuple[str, str], int] = field(default_factory=dict)
    query_concepts: Set[str] = field(default_factory=set)

    def observe_chain(self, hops: List[Tuple[str, str]], is_user_query: bool = False):
        """Record which edges were traversed during a chain walk."""
        for from_label, to_label in hops:
            key = (from_label.lower(), to_label.lower())
            self.edge_reactivations[key] = self.edge_reactivations.get(key, 0) + 1
        if is_user_query:
            for from_label, to_label in hops:
                self.query_concepts.add(from_label.lower())
                self.query_concepts.add(to_label.lower())

    def inferred_preferences(self, threshold: int = 2) -> Dict[Tuple[str, str], int]:
        """Return edges the user has reactivated >= threshold times.
        Used for display/debugging, NOT for boost calculation."""
        return {(f, t): c for (f, t), c in self.edge_reactivations.items()
                if c >= threshold}

    def activation_boost_for(self, concept: str) -> Dict[str, float]:
        """Continuous activation boost for edges reachable from concept.

        Uses sigmoid confidence: 0 visits → 1.0x, 1 visit → 1.15x,
        2 visits → 1.2x, saturating at 1.3x.
        No hard threshold — even a single visit provides a small boost."""
        boost: Dict[str, float] = {}
        cl = concept.lower()
        for (from_c, to_c), count in self.edge_reactivations.items():
            if from_c == cl:
                boost[to_c] = 1.0 + (count / (count + 1.0)) * 0.3
        return boost

@dataclass
class CognitiveResponseContext:
    """All cognitive signals available for response generation."""
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
    learned_recently: bool = False  # Did we just learn something new?
    recall_mode: bool = False  # Episodic re-traversal vs generic chain walk


class BeliefStore:
    """Tracks asserted belief triples and contradiction history.

    Each belief is (subject, predicate, value) with confidence and timestamp.
    Contradictions are detected and recorded for edge-weight modulation.
    """
    def __init__(self):
        self.beliefs: Dict[Tuple[str, str], Tuple[str, float, int]] = {}
        # (subject, predicate) -> (value, confidence, turn)
        self.contradictions: List[Tuple[Tuple, Tuple, int]] = []
        # [(old_triple, new_triple, turn)]
        self.resolution_history: Dict[str, str] = {}
        # triple_str -> "accept_new" | "reject_new" | "both"
        self.turn_num = 0

    def advance_turn(self):
        self.turn_num += 1

    def assert_belief(self, subject_id: str, predicate: str,
                       value: str, confidence: float = 0.8):
        key = (subject_id, predicate)
        self.beliefs[key] = (value, confidence, self.turn_num)

    def query_belief(self, subject_id: str,
                     predicate: str) -> Optional[Tuple[str, float, int]]:
        return self.beliefs.get((subject_id, predicate))

    def detect_contradiction(self, subject_id: str, predicate: str,
                              new_value: str) -> Optional[Tuple]:
        existing = self.query_belief(subject_id, predicate)
        if existing and existing[0] != new_value:
            old_triple = (subject_id, predicate, existing[0])
            new_triple = (subject_id, predicate, new_value)
            self.contradictions.append((old_triple, new_triple, self.turn_num))
            return (existing, new_value, existing[2])
        return None

    def resolve_contradiction(self, old_triple: Tuple, new_triple: Tuple,
                               choice: str):
        self.contradictions.append((old_triple, new_triple, self.turn_num))
        key = str(new_triple)
        self.resolution_history[key] = choice

    def reconcile(self) -> Dict[Tuple[str, str], Tuple[str, float, int]]:
        """Resolve contradictions: pick winner by confidence * recency decay.

        The belief store keyed by (subject, predicate) holds only the latest
        value per pair, but contradiction history lists every flip. We scan
        contradictions to find each contested pair and pick the winner based
        on confidence × recency (later turns = higher recency).
        """
        resolved: Dict[Tuple[str, str], Tuple[str, float, int]] = {}
        # Group contradiction candidates by (subject, predicate)
        groups: Dict[Tuple[str, str], List[Tuple[str, float, int]]] = {}
        for old_triple, new_triple, c_turn in self.contradictions:
            # old_triple = (subject, predicate, old_value)
            # new_triple = (subject, predicate, new_value)
            subj, pred = old_triple[0], old_triple[1]
            key = (subj, pred)
            # Add both sides of the contradiction
            old_val, old_conf = old_triple[2], 0.0
            new_val, new_conf = new_triple[2], 0.0
            # Look up actual confidence from belief store if available
            cur = self.beliefs.get(key)
            if cur:
                if cur[0] == old_val:
                    old_conf = cur[1]
                elif cur[0] == new_val:
                    new_conf = cur[1]
            groups.setdefault(key, []).append(
                (old_val, old_conf if old_conf > 0 else 0.5, c_turn))
            groups.setdefault(key, []).append(
                (new_val, new_conf if new_conf > 0 else 0.5, c_turn))
        # Deduplicate within each group
        for key, candidates in groups.items():
            seen = set()
            unique = []
            for c in candidates:
                if c[0] not in seen:
                    seen.add(c[0])
                    unique.append(c)
            if len(unique) < 2:
                continue
            def decay_score(vc: Tuple) -> float:
                _, conf, turn = vc
                recency = 1.0 / (1.0 + (self.turn_num - turn) * 0.1)
                return conf * recency
            winner = max(unique, key=decay_score)
            # Update the belief store with the winner
            self.beliefs[key] = (winner[0], winner[1], self.turn_num)
            resolved[key] = self.beliefs[key]
        return resolved

    def get_state(self) -> Dict:
        return {
            'beliefs': self.beliefs,
            'contradictions': self.contradictions,
            'resolution_history': self.resolution_history,
            'turn_num': self.turn_num,
        }

    def set_state(self, state: Dict):
        self.beliefs = state.get('beliefs', {})
        self.contradictions = state.get('contradictions', [])
        self.resolution_history = state.get('resolution_history', {})
        self.turn_num = state.get('turn_num', 0)


class CognitiveChatEngine:
    """RAVANA cognitive chat engine — starts as a baby, learns from the web."""

    def __init__(self, dim: int = 64, seed: int = 42, baby_mode: bool = True, data_dir: Optional[str] = None, user_suffix: str = ""):
        self.dim = dim
        self.rng = np.random.RandomState(seed)

        self.graph = ConceptGraph(dim=dim, max_nodes=10000)
        self.baby_mode = baby_mode
        self._concept_labels: Set[str] = set()  # set of primary concept labels

        # GloVe embeddings (loaded lazily during seeding)
        self._glove_vecs: Optional[Dict[str, np.ndarray]] = None
        self._glove_proj: Optional[np.ndarray] = None
        self._glove_dim: int = 100
        # Phase 2.1: GloVe vector cache (avoid recomputing projection)
        self._glove_vector_cache: Dict[str, np.ndarray] = {}
        # Phase 2.3: Warm-start cache file path

        # Cognitive engines (emotion, identity, meaning, dual-process, global workspace)
        self.emotion = VADEmotionEngine(VADConfig(eta_valence=0.3, eta_arousal=0.4, eta_dominance=0.25))
        self.identity = IdentityEngine(initial_strength=0.25, momentum_factor=0.3, recovery_bias=0.15)
        self.meaning = MeaningEngine(MeaningConfig(w_dissonance_reduction=0.3,
             w_identity_coherence=0.3, w_predictive_power=0.4, effort_kappa=0.5))
        self.dual_process = DualProcessController(DualProcessConfig(
            system2_confidence_threshold=0.25, system2_novelty_threshold=0.4, max_consecutive_system2=5))
        self.gw = GlobalWorkspace(GWConfig(capacity=7, broadcast_threshold=0.3, decay_rate=0.1))

        # State
        self.turn_count = 0
        # Phase 3.1: Topic-indexed conversation store (dict, last 50)
        self._topic_list: List[str] = []
        self._topic_store: Dict[str, Dict] = {}
        # Phase 3.4: Response-aware context
        self._response_context: List[Dict] = []
        self._last_responses: List[str] = []
        self._last_strategy: str = ""
        self._free_energy = 0.0
        self._learning_count = 0
        self._learned_this_turn = False
        # Phase 1.3: Deferred web learning queue
        self._pending_learning_queue: List[str] = []
        # Phase 5: Auto offline fallback (None = untested, False = down)
        self._network_available: Optional[bool] = None
        self._network_retry_turn: int = 0  # retry network every 20 turns if down
        # Phase 1.4: Per-session rate limit (max 1 search per 3 turns)
        self._turns_since_last_search: int = 0
        self._concept_keywords: Dict[str, List[int]] = {}
        # Phase 5: Use data_dir if provided
        if data_dir:
            self._save_path = os.path.join(data_dir, f"ravana_weights{user_suffix}.pkl")
            self._glove_cache_path = os.path.join(data_dir, "ravana_glove_cache.npz")
        else:
            self._save_path = os.path.join(_proj_root, f"ravana_weights{user_suffix}.pkl")
            self._glove_cache_path = os.path.join(_proj_root, "ravana_glove_cache.npz")
        self.sleep_cycles_completed = 0
        self._chain_traces: List[ChainTrace] = []
        # Phase 7: Impossible Query Registry
        self._impossible_queries: List[FailedQuery] = []
        self._last_strategy_used: str = ""
        self._trace_enabled = False
        self._contradiction_map: Dict[str, Set[str]] = {}
        self._belief_assertions: List[Tuple[str, str, str]] = []
        self._recall_mode: bool = False
        self.user_model = UserModel()
        self._last_hops: List[List[Tuple[str, str]]] = []  # concept -> strength (decays)
        self._last_chain_hops: List[List[Tuple[str, str]]] = []  # Phase 3.4: snapshot before clear
        # Phase 8: Prefrontal workspace — holds subject + top associations for on-topic focus
        self._prefrontal_buffer: List[str] = []
        # Phase 9: PFC gating — dynamic gating threshold modulated by arousal (teen = weaker gating)
        self._pfc_gating_enabled = True
        self._pfc_buffer_capacity = 7  # typical working memory capacity
        # Phase 9b: Prediction error tracking (surprise signal for Active Inference)
        self._mean_prediction_error = 0.0
        self._prediction_error_count = 0
        # Integration toggles (can be disabled via CLI)
        self.use_vad = True
        self.use_rlm = True
        self.use_beliefs = True
        self.belief_store = BeliefStore()

        # Fix 2: Dormant edge tracking — auto-wired GloVe edges are invisible
        # until the user model visits them at least once.
        self._dormant_edges: Set[Tuple[int, int]] = set()

        # Build reverse lookup from connector word → relation type
        self._CONNECTOR_TO_REL: Dict[str, str] = {}
        for rel_type, tiers in self._EDGE_CONNECTORS.items():
            for entry in tiers:
                options = entry[1] if isinstance(entry, tuple) and len(entry) == 2 else entry[2]
                for opt in options:
                    self._CONNECTOR_TO_REL[opt] = rel_type
        self._CONNECTOR_SET = set(self._CONNECTOR_TO_REL.keys())

        # Try loading saved weights first
        # Meta-cognition layer
        self.meta_cog = MetaCognition(MetaCognitiveConfig(
            probe_failure_threshold=0.4,
            confidence_calibration_window=15,
        ))

        # Sleep consolidation
        self.sleep_engine = SleepConsolidation(SleepConfig(
            pressure_threshold=0.3,
            counterfactual_rate=0.15,
            emotional_flip_rate=0.08,
        ))
        self._sleep_pressure = 0.0
        self._last_sleep_episode = 0

        # Concept-emotion tags
        self._concept_vad: Dict[int, Tuple[float, float, float]] = {}
        # Phase 10-17: Instance state initialization
        self._sentence_schema: Dict[str, float] = {}
        self._mean_sentence_pe: float = 0.0
        self._sentence_pe_count: int = 0
        self._current_context_vector: Optional[np.ndarray] = None
        self._modulated_vectors: Dict[int, np.ndarray] = {}
        self._state_dependent_boosts: Dict[str, Dict[str, float]] = {}
        self._cognitive_state: str = "default"
        self._state_duration: int = 0
        self._cognitive_state_hold: int = 0
        self._schema_mode: bool = False
        self._activation_fatigue: Dict[int, float] = {}
        self._recent_traversals: List[Tuple[int, int]] = []
        self._visited_concepts: Set[str] = set()
        self._dopamine_tone: float = 0.5
        self._td_error_history: List[float] = []
        self._expected_strength: float = 0.25
        self._episodic_edges: Dict[Tuple[int, int], Any] = {}
        self._semantic_edges: Dict[Tuple[int, int], Any] = {}
        self._cerebellar_ngram: Dict[str, Dict[str, float]] = {}
        self._cerebellar_depth: Dict[str, float] = {}
        self._concept_confidence: Dict[str, float] = {}
        self._calibration_error: float = 0.0
        self._metacognitive_review_turn: int = 0


        if os.path.exists(self._save_path):
            loaded = self._load()
            if loaded:
                print(f"  [Loaded] Remembered {len(self.graph.nodes)} words from before!")
                return

        # Seed teen concepts from scratch
        self._init_glove()
        self._seed_concepts()
        print(f"  [Teen] Knows {len(self.graph.nodes)} words, ready to learn!")

    # ─── Teen Graph Seeding ───

    def _seed_concepts(self):
        """Seed the graph with teenager-level vocabulary and typed relationships."""
        all_labels = []
        for label, keywords in TEEN_CONCEPTS:
            # GloVe vector if available, else hash-based random
            vec = self._glove_vector(label)
            if vec is None:
                h = hash(label) % 10000
                vr = np.random.RandomState(h + 42)
                vec = vr.randn(self.dim).astype(np.float32) * 0.15
                for kw in keywords.split():
                    kh = hash(kw) % 5000
                    kr = np.random.RandomState(kh)
                    vec += kr.randn(self.dim).astype(np.float32) * 0.05
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
            node = self.graph.add_node(vector=vec, label=label)
            all_labels.append(label)
            self._concept_labels.add(label.lower())

            # Map keywords
            for kw in keywords.split():
                kl = kw.lower()
                self._concept_keywords.setdefault(kl, []).append(node.id)
            self._concept_keywords.setdefault(label, []).append(node.id)
            if "_" in label:
                for part in label.split("_"):
                    if len(part) >= 3:
                        self._concept_keywords.setdefault(part, []).append(node.id)

        # Build concept_id map after seeding
        label_to_id = {}
        for nid, node in self.graph.nodes.items():
            if node.label:
                label_to_id[node.label] = nid

        # ── Typed Edges: (source, target, relation_type, weight) ──
        # Semantic (is-a, similar to): baseline connections
        semantic_edges = [
            ("hello", "bye"), ("hello", "thanks"), ("thanks", "sorry"),
            ("yes", "no"), ("good", "bad"), ("big", "small"), ("hot", "cold"),
            ("up", "down"), ("in", "out"), ("here", "there"),
            ("now", "later"), ("more", "some"), ("one", "many"),
            ("happy", "sad"), ("joy", "grief"), ("excited", "bored"),
            ("love", "hate"), ("hope", "fear"),
            ("trust", "respect"), ("freedom", "responsibility"),
            ("knowledge", "wisdom"), ("knowledge", "truth"),
            ("sun", "moon"), ("sun", "light"), ("tree", "bird"),
            ("dog", "cat"), ("dog", "friend"),
            ("water", "rain"), ("food", "eat"),
            ("book", "read"), ("book", "story"), ("song", "music"),
            ("science", "knowledge"), ("history", "story"),
            ("life", "death"), ("mind", "heart"),
            ("time", "change"), ("world", "nature"),
            ("time", "future"), ("time", "past"), ("future", "past"),
            ("machine", "invention"), ("invention", "create"),
            ("imagination", "dream"), ("imagination", "create"),
            ("impossible", "possible"), ("possibility", "maybe"),
            ("future", "change"), ("past", "history"),
            ("invent", "create"), ("invent", "explore"),
            ("journey", "explore"), ("experiment", "discovery"),
            ("i", "we"), ("i", "you"),
            ("all", "many"), ("work", "effort"),
            ("play", "fun"), ("play", "game"),
            ("why", "because"), ("if", "maybe"),
        ]

        # Causal: A causes/creates/produces B
        causal_edges = [
            ("sun", "hot"), ("sun", "light"),
            ("eat", "food"), ("drink", "water"),
            ("sleep", "tired"), ("play", "happy"),
            ("love", "joy"), ("fear", "anxiety"),
            ("cause", "change"), ("learn", "knowledge"),
            ("study", "understanding"), ("practice", "skill"),
            ("challenge", "struggle"), ("struggle", "growth"),
            ("power", "responsibility"), ("freedom", "choice"),
            ("question", "curiosity"), ("explore", "discovery"),
            ("trust", "friendship"), ("hypocrisy", "distrust"),
            ("grief", "sadness"), ("hope", "motivation"),
            ("rejection", "loneliness"), ("acceptance", "belonging"),
            ("anger", "conflict"), ("empathy", "understanding"),
            ("criticism", "growth"), ("failure", "learning"),
            ("change", "growth"), ("time", "change"),
            ("knowledge", "wisdom"), ("experience", "wisdom"),
            ("art", "expression"), ("science", "progress"),
            ("imagination", "invention"), ("experiment", "knowledge"),
            ("machine", "future"), ("journey", "discovery"),
        ]

        # Temporal: A then/before/after B
        temporal_edges = [
            ("birth", "life"), ("life", "death"),
            ("morning", "noon"), ("noon", "night"),
            ("question", "answer"), ("cause", "effect"),
            ("learn", "understand"), ("struggle", "succeed"),
        ]

        # Contrastive: A vs B (opposites)
        contrastive_edges = [
            ("good", "bad"), ("love", "hate"), ("life", "death"),
            ("truth", "lie"), ("freedom", "oppression"),
            ("courage", "fear"), ("hope", "despair"),
            ("knowledge", "ignorance"), ("justice", "injustice"),
            ("create", "destroy"), ("accept", "reject"),
            ("always", "never"),
        ]

        # Analogical: A is like B (different domains, similar pattern)
        analogical_edges = [
            ("mind", "garden"), ("life", "journey"),
            ("knowledge", "light"), ("time", "river"),
            ("society", "organism"), ("identity", "building"),
            ("memory", "library"), ("heart", "engine"),
        ]

        # Apply all edges with their types
        self._apply_edges(label_to_id, semantic_edges, "semantic", 0.35)
        self._apply_edges(label_to_id, causal_edges, "causal", 0.45)
        self._apply_edges(label_to_id, temporal_edges, "temporal", 0.40)
        self._apply_edges(label_to_id, contrastive_edges, "contrastive", 0.50)
        self._apply_edges(label_to_id, analogical_edges, "analogical", 0.30)

        # ── Hub edges: connect self-referential and high-frequency concepts
        # This densifies the graph so response generation can walk richer paths
        hub_edges = [
            # Self → cognitive actions
            ("i", "think", "causal"), ("i", "feel", "causal"),
            ("i", "know", "causal"), ("i", "want", "causal"),
            ("i", "like", "causal"), ("i", "say", "causal"),
            ("we", "people", "semantic"), ("we", "they", "semantic"),
            ("you", "person", "semantic"), ("you", "friend", "semantic"),
            # Cognitive hubs → content
            ("think", "knowledge", "causal"), ("think", "question", "causal"),
            ("think", "analyze", "causal"), ("think", "explore", "causal"),
            ("feel", "love", "causal"), ("feel", "happy", "causal"),
            ("feel", "sad", "causal"), ("feel", "fear", "causal"),
            ("feel", "excited", "causal"),
            ("know", "learn", "causal"), ("know", "truth", "semantic"),
            ("know", "knowledge", "semantic"), ("know", "understand", "causal"),
            ("want", "freedom", "causal"), ("want", "love", "causal"),
            ("want", "more", "causal"), ("want", "power", "causal"),
            ("like", "love", "semantic"), ("like", "play", "causal"),
            ("like", "friend", "semantic"), ("like", "music", "semantic"),
            ("like", "art", "semantic"),
            # Extra connective density
            ("people", "society", "semantic"), ("people", "culture", "semantic"),
            ("learn", "knowledge", "causal"), ("learn", "understand", "causal"),
            ("understand", "knowledge", "semantic"),
            ("question", "curious", "causal"), ("explore", "discover", "causal"),
        ]
        for src, tgt, rel in hub_edges:
            sid = label_to_id.get(src)
            tid = label_to_id.get(tgt)
            if sid is not None and tid is not None and self.graph.get_edge(sid, tid) is None:
                bw = 0.35 if rel == "semantic" else 0.45
                self.graph.add_edge(sid, tid, weight=bw + self.rng.uniform(0, 0.15),
                                   relation_type=rel)

        # ── Auto-wire: connect GloVe-similar concepts (>0.6) as semantic edges
        auto_count = 0
        nids = list(label_to_id.values())
        for i in range(len(nids)):
            ni = self.graph.get_node(nids[i])
            if ni is None or ni.vector is None:
                continue
            for j in range(i + 1, len(nids)):
                if self.graph.get_edge(nids[i], nids[j]) is not None:
                    continue
                nj = self.graph.get_node(nids[j])
                if nj is None or nj.vector is None:
                    continue
                sim = float(np.dot(ni.vector, nj.vector))
                if sim > 0.6:
                    weight = min(0.5, sim * 0.5)
                    edge = self.graph.add_edge(nids[i], nids[j], weight=weight, relation_type="semantic")
                    edge.confidence = 0.001  # dormant: invisible until visited
                    self._dormant_edges.add((nids[i], nids[j]))
                    auto_count += 1

        self._all_labels = label_to_id
        self._build_contradiction_map(contrastive_edges)
        print(f"  [Teen] Seeded {len(self.graph.nodes)} concepts, {len(self.graph.edges)} connections ({auto_count} auto-wired) across 5 relation types")

    # ─── GloVe Semantic Vectors ───

    def _init_glove(self):
        """Load GloVe 100D vectors and build projection to self.dim.

        Phase 2.3: Warm-start — tries to load pre-computed projected vectors
        from 'ravana_glove_cache.npz' first. Falls back to reading the raw
        GloVe file and caching the result for next time.
        """
        # Phase 2.3: Try warm-start cache first
        if os.path.exists(self._glove_cache_path):
            try:
                data = np.load(self._glove_cache_path, allow_pickle=True)
                words = data['words'].tolist()
                vecs = data['vecs']
                proj = data['proj']
                self._glove_dim = int(data['glove_dim'])
                self._glove_proj = proj
                self._glove_vecs = {}
                for i, w in enumerate(words):
                    self._glove_vecs[w] = vecs[i]
                # Pre-populate vector cache with PROJECTED (not raw) vectors
                for w, raw_vec in self._glove_vecs.items():
                    pv = self._glove_proj @ raw_vec
                    norm = np.linalg.norm(pv)
                    if norm > 0:
                        pv /= norm
                    self._glove_vector_cache[w] = pv.astype(np.float32)
                print(f"  [GloVe] Loaded {len(self._glove_vecs)} projected vectors from cache ({self._glove_dim}D -> {self.dim}D)")
                return
            except Exception:
                print(f"  [GloVe] Cache load failed, re-reading from file...")

        # Fall back to reading raw GloVe file
        for name in ['glove.6B.100d.txt', 'glove.6B.50d.txt']:
            path = os.path.join(_proj_root, 'data', 'glove', name)
            if os.path.exists(path):
                self._glove_dim = 100 if '100d' in name else 50
                break
        else:
            return
        glove_path = os.path.join(_proj_root, 'data', 'glove', f'glove.6B.{self._glove_dim}d.txt')
        self._glove_vecs = {}
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != self._glove_dim + 1:
                    continue
                self._glove_vecs[parts[0]] = np.array([float(x) for x in parts[1:]], dtype=np.float32)
        # Random orthogonal projection: glove_dim → dim
        rng = np.random.RandomState(42)
        max_d = max(self._glove_dim, self.dim)
        full_q, _ = np.linalg.qr(rng.randn(max_d, max_d).astype(np.float32))
        self._glove_proj = full_q[:self.dim, :self._glove_dim].copy()
        self._glove_proj *= np.sqrt(float(self._glove_dim) / float(self.dim))
        print(f"  [GloVe] {len(self._glove_vecs)} words, {self._glove_dim}D -> {self.dim}D")

        # Phase 2.3: Save projected vectors as warm-start cache
        try:
            words_list = list(self._glove_vecs.keys())
            vecs_array = np.array([self._glove_vecs[w] for w in words_list], dtype=np.float32)
            np.savez_compressed(
                self._glove_cache_path,
                words=words_list,
                vecs=vecs_array,
                proj=self._glove_proj,
                glove_dim=self._glove_dim,
            )
            print(f"  [GloVe] Saved projected cache ({len(words_list)} words)")
        except Exception as e:
            print(f"  [GloVe] Warning: could not save cache: {e}")

    def _glove_vector(self, label: str) -> Optional[np.ndarray]:
        """Look up a label in GloVe, project to self.dim, return unit vector.

        Phase 2.1: Results are cached so repeated lookups (e.g. auto-expansion
        for every input word) avoid recomputing the projection.
        """
        if self._glove_vecs is None:
            return None
        w = label.lower().strip()
        # Check cache first (Phase 2.1)
        cached = self._glove_vector_cache.get(w)
        if cached is not None:
            return cached
        vec = self._glove_vecs.get(w)
        if vec is None and len(w) > 1:
            vec = self._glove_vecs.get(w.rstrip('s'))
        if vec is None and len(w) > 2:
            vec = self._glove_vecs.get(w[:-1])
        if vec is not None:
            pv = self._glove_proj @ vec
            norm = np.linalg.norm(pv)
            if norm > 0:
                pv /= norm
            result = pv.astype(np.float32)
            self._glove_vector_cache[w] = result
            # Also cache variants for fast lookup
            if w.rstrip('s') != w:
                self._glove_vector_cache[w.rstrip('s')] = result
            if len(w) > 2 and w[:-1] != w:
                self._glove_vector_cache[w[:-1]] = result
            return result
        return None

    # ─── Phase 1: Auto-Expansion from Every Message ───

    def _auto_expand_concepts(self, text: str) -> int:
        """Phase 1.1+1.2: Auto-expand graph from user input.

        For each meaningful word in input that is not yet a graph concept:
        - If word has a GloVe vector: add as concept node, wire to top-5 GloVe
          nearest neighbors (Phase 1.1), then auto-wire to ALL existing concepts
          with cosine > 0.5 (Phase 1.2).
        - If word has no GloVe vector: skip (web search handles later).

        Returns number of new concepts added.
        """
        # Extract meaningful words from input
        words = re.findall(r"[a-zA-Z']{3,}", text.lower())
        meaningful = set()
        for w in words:
            wc = w.strip("'")
            if wc not in STOP_WORDS and len(wc) >= 3:
                meaningful.add(wc)
        if not meaningful:
            return 0

        # Build set of existing graph labels for fast lookup
        existing_labels = set()
        for nid, node in self.graph.nodes.items():
            if node.label:
                existing_labels.add(node.label.lower())

        new_count = 0
        new_nodes: Dict[str, int] = {}  # word -> node_id

        # Phase 1.1: Add each new word as a concept (GloVe only, skip non-GloVe)
        for word in meaningful:
            if word in existing_labels:
                continue
            vec = self._glove_vector(word)
            if vec is None:
                continue  # Not in GloVe — web search can handle later
            node = self.graph.add_node(vector=vec, label=word)
            self._concept_keywords[word] = self._concept_keywords.get(word, []) + [node.id]
            self._concept_labels.add(word.lower())
            existing_labels.add(word)
            new_nodes[word] = node.id
            new_count += 1

        if not new_nodes:
            return 0

        # Phase 1.1+1.2: Wire new concepts to existing graph concepts via vector similarity
        for word, nid in new_nodes.items():
            node = self.graph.get_node(nid)
            if node is None or node.vector is None:
                continue

            # Compute cosine similarity against ALL existing concepts
            similarities = []
            for other_nid, other_node in self.graph.nodes.items():
                if other_nid == nid:
                    continue
                if other_node.label is None or other_node.vector is None:
                    continue
                if other_node.label.lower() in new_nodes:
                    continue  # Skip other freshly-added concepts (they wire themselves)
                sim = float(np.dot(node.vector, other_node.vector))
                if sim > 0.3:
                    similarities.append((other_nid, sim))

            if not similarities:
                continue

            # Phase 2.4: Use np.argpartition for O(N) top-5 selection
            # instead of O(N log N) full sort
            sim_array = np.array([s[1] for s in similarities], dtype=np.float32)
            nid_array = np.array([s[0] for s in similarities], dtype=np.int32)

            # Phase 9b: Wire using prediction-error-based learning
            # Initial weight = raw GloVe similarity (no arbitrary cap)
            # Then one gradient step refines the weight toward accurate prediction
            k_top = min(5, len(sim_array))
            if k_top > 0:
                top_idx = np.argpartition(sim_array, -k_top)[-k_top:]
                top_order = np.argsort(-sim_array[top_idx])
                top_idx = top_idx[top_order]
                wired_11 = 0
                for idx in top_idx:
                    if wired_11 >= 5:
                        break
                    sim = float(sim_array[idx])
                    other_nid = int(nid_array[idx])
                    if sim > 0.4 and self.graph.get_edge(nid, other_nid) is None:
                        weight = min(0.6, sim * 0.6)
                        self.graph.add_edge(nid, other_nid, weight=weight, relation_type="semantic")
                        wired_11 += 1

            # Phase 1.2 + 9b: Auto-wire to ALL existing concepts where sim > 0.5
            for idx in range(len(similarities)):
                sim = float(sim_array[idx])
                if sim > 0.5 and self.graph.get_edge(nid, int(nid_array[idx])) is None:
                    weight = min(0.6, sim * 0.6)
                    self.graph.add_edge(nid, int(nid_array[idx]), weight=weight, relation_type="semantic")

        return new_count

    # ─── Phase 9b: Active Inference / Prediction Error ───
    # Based on: Friston's Free Energy Principle & Active Inference
    # (Friston, Parr, Pezzulo 2022; Bogacz 2017 tutorial)
    #
    # Each edge weight is a PREDICTION: "from concept A, you can reach concept B
    # with this expected similarity." When the actual GloVe vector similarity differs
    # from the edge weight, prediction error drives learning — not additive increments.

    def _prediction_error(self, src_vec: np.ndarray, tgt_vec: np.ndarray,
                           current_weight: float) -> Tuple[float, float]:
        """Compute prediction error and gradient for an edge weight.

        Prediction: edge weight predicts cosine similarity between vectors.
        Error: MSE between predicted similarity and actual cosine similarity.
        Gradient: d(error)/d(weight) = 2 * (weight - cosine) * (-1)

        Returns (error, gradient).
        """
        actual_sim = float(np.dot(src_vec, tgt_vec))
        error = (current_weight - actual_sim) ** 2
        gradient = 2.0 * (current_weight - actual_sim)
        return error, gradient

    def _update_edge_from_error(self, edge, src_vec: np.ndarray,
                                 tgt_vec: np.ndarray, learning_rate: float = 0.15):
        """Update edge weight using gradient descent on prediction error.

        Low error → weight stays (prediction is accurate)
        High error → weight moves toward actual similarity (prediction improves)
        Confidence adjusts inversely to error (low error → high confidence)
        """
        error, gradient = self._prediction_error(src_vec, tgt_vec, edge.weight)
        # Gradient descent: move weight toward reducing error
        edge.weight -= learning_rate * gradient * 0.5
        edge.weight = np.clip(edge.weight, 0.01, 0.99)
        # Confidence: inverse of error (squashed to 0-1)
        edge.confidence = max(0.05, 1.0 - np.tanh(error * 3.0))
        # Track running mean prediction error (surprise signal)
        alpha = 0.05
        self._mean_prediction_error = (1 - alpha) * self._mean_prediction_error + alpha * error
        self._prediction_error_count += 1
        return error

    def _build_contradiction_map(self, contrastive_edges: List[Tuple[str, str]]):
        """Build a map of concept → set of antonym concepts from contrastive edges."""
        for a, b in contrastive_edges:
            al, bl = a.lower(), b.lower()
            self._contradiction_map.setdefault(al, set()).add(bl)
            self._contradiction_map.setdefault(bl, set()).add(al)

    def _rebuild_contradiction_map(self):
        """Rebuild contradiction map from graph edges (used after load)."""
        for (src, tgt), edge in self.graph.edges.items():
            if edge.relation_type != "contrastive":
                continue
            sn = self.graph.nodes.get(src)
            tn = self.graph.nodes.get(tgt)
            if sn and sn.label and tn and tn.label:
                al, bl = sn.label.lower(), tn.label.lower()
                self._contradiction_map.setdefault(al, set()).add(bl)
                self._contradiction_map.setdefault(bl, set()).add(al)

    def _is_contradictory(self, source_label: str, target_label: str,
                          relation_type: str) -> bool:
        """Check if asserting (source, relation, target) contradicts a logged belief.

        A contradiction occurs when source already asserted about concept X,
        and target is a known antonym of X (from the contradiction map).
        """
        tl = target_label.lower()
        sl = source_label.lower()
        for bel_src, bel_rel, bel_tgt in self._belief_assertions:
            if bel_src.lower() != sl:
                continue
            # Check if target is an antonym of the previously asserted object
            bel_antonyms = self._contradiction_map.get(bel_tgt.lower(), set())
            if tl in bel_antonyms:
                return True
        return False

    def _log_assertions(self, chains: List[str], subject: str):
        """Extract and log (subject, relation, object) triples from generated chains."""
        starter_values = set(self._EDGE_TO_STARTER.values())
        for chain_str in chains:
            parts = chain_str.strip(".").split()
            # Skip discourse starter if present (e.g. "but", "then", "like")
            start = 1 if parts and parts[0] in starter_values else 0
            if len(parts) - start < 3:
                continue
            if start == 0:
                cur_subj = parts[0]
            else:
                cur_subj = parts[1]
            # Walk (connector, object) pairs
            i = start + 1 if start == 0 else start + 2
            actual_start = start + 1 if start == 0 else start + 1
            for j in range(actual_start, len(parts) - 1, 2):
                connector = parts[j]
                obj = parts[j + 1]
                rel_type = "semantic"
                for rtype, glabel in self._EDGE_TO_GRAPH_LABEL.items():
                    if glabel == connector:
                        rel_type = rtype
                        break
                self._belief_assertions.append((cur_subj, rel_type, obj))
                cur_subj = obj

    def _apply_edges(self, label_to_id, edge_list, rel_type, base_weight):
        """Apply a list of (src, tgt) edges to the graph with a relation type."""
        for src, tgt in edge_list:
            sid = label_to_id.get(src)
            tid = label_to_id.get(tgt)
            if sid is not None and tid is not None and self.graph.get_edge(sid, tid) is None:
                self.graph.add_edge(sid, tid, weight=base_weight + self.rng.uniform(0, 0.15),
                                   relation_type=rel_type)

    # ─── Web Learning ───

    def learn_from_web(self, query: str) -> str:
        """Search the web, fetch articles, extract concepts, and learn from them.

        Phase 5: Auto offline fallback. If the network API fails (timeout, DNS,
        HTTP error), sets _network_available = False and falls back silently.
        No error messages leak to the user — just returns a short summary.
        """
        self._learned_this_turn = True
        self._learning_count += 1

        # Phase 5: Skip API call if we already know the network is down
        # (unless it's time for a retry — check every 20 turns)
        if self._network_available is False:
            if self._network_retry_turn > 0 and self.turn_count >= self._network_retry_turn:
                self._network_available = None  # try again
                self._network_retry_turn = 0
            else:
                # Still try GloVe-only learning from the query text itself
                known_count = self._learn_from_text(query + " " + query, query)
                if known_count > 0:
                    return f"learned {known_count} things about {query}"
                return f"offline - already knew about {query}"

        query_clean = quote(query)

        try:
            # Step 1: Search (Phase 5: shorter 3s timeout for faster fallback)
            search_url = f"{SEARCH_API}{query_clean}"
            req = urllib.request.Request(search_url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) RAVANA-Baby/1.0'
            })
            resp = urllib.request.urlopen(req, timeout=3)
            data = json.loads(resp.read().decode('utf-8'))
            results = data.get('results', [])

            # Network worked — mark as available
            if self._network_available is None:
                self._network_available = True

            if not results:
                return self._learn_from_snippets(query, [])

            # Step 2: Get snippets and try to fetch article text
            snippets = []

            for i, r in enumerate(results[:3]):  # Top 3 results
                snippet = r.get('content', '') or ''
                title = r.get('title', '') or ''
                url = r.get('url', '') or ''
                text = f"{title}. {snippet}"

                # Try to fetch full article text
                if url:
                    try:
                        article_text = self._fetch_article_text(url)
                        if article_text and len(article_text) > len(text):
                            text = article_text[:3000]
                    except Exception:
                        pass

                snippets.append(text)

            # Step 3: Extract and learn concepts from all gathered text
            combined_text = " ".join(snippets)
            new_concepts_added = self._learn_from_text(combined_text, query)

            if new_concepts_added > 0:
                return f"learned {new_concepts_added} new things about {query}"
            else:
                return f"read about {query} but already knew the words"

        except (urllib.request.URLError, urllib.request.HTTPError,
                ConnectionError, TimeoutError, OSError, json.JSONDecodeError):
            # Phase 5: Network failure — mark as unavailable, fall back silently
            self._network_available = False
            # Still try GloVe-only learning from the query text
            known_count = self._learn_from_text(query + " " + query, query)
            if known_count > 0:
                return f"learned {known_count} things about {query}"
            return f"offline - already knew about {query}"
        except Exception:
            # Any other error — also fall back silently
            self._network_available = False
            return f"offline"

    def _fetch_article_text(self, url: str) -> Optional[str]:
        """Fetch a URL and extract readable article text."""
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            resp = urllib.request.urlopen(req, timeout=8)
            html = resp.read().decode('utf-8', errors='replace')

            if HAS_BS4:
                soup = BeautifulSoup(html, 'html.parser')
                # Remove script/style elements
                for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                    tag.decompose()
                # Try to find main content
                main = soup.find('article') or soup.find('main') or soup.find('body')
                if main:
                    text = main.get_text(separator=' ', strip=True)
                else:
                    text = soup.get_text(separator=' ', strip=True)
                return re.sub(r'\s+', ' ', text)[:5000]
            else:
                # Simple regex-based tag stripping
                text = re.sub(r'<[^>]+>', ' ', html)
                text = re.sub(r'\s+', ' ', text).strip()
                return text[:3000]
        except Exception:
            return None

    def _learn_from_text(self, text: str, topic: str) -> int:
        """Extract important keywords from text, add them as concepts, form connections.

        Returns number of new concepts added.
        """
        # Tokenize and count word frequencies
        words = re.findall(r"[a-zA-Z']{3,}", text.lower())
        word_counts = {}
        for w in words:
            wc = w.strip("'")
            if wc not in STOP_WORDS and len(wc) >= 3:
                word_counts[wc] = word_counts.get(wc, 0) + 1

        if not word_counts:
            return 0

        # Sort by frequency, take top 30
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        important_words = [w for w, c in sorted_words[:30]]

        # Always add the topic word
        topic_lower = topic.lower().strip()
        topic_words = [tw for tw in re.findall(r"[a-zA-Z']{3,}", topic_lower) if tw not in STOP_WORDS]
        important_words = topic_words + [w for w in important_words if w not in topic_words]

        # Check which words are new vs known
        existing_labels = set()
        for nid, node in self.graph.nodes.items():
            if node.label:
                existing_labels.add(node.label.lower())

        new_count = 0
        label_to_id = {}
        for nid, node in self.graph.nodes.items():
            if node.label:
                label_to_id[node.label] = nid

        # Add new concepts
        for word in important_words:
            if word in existing_labels:
                continue
            # GloVe vector if available, else hash-based random
            vec = self._glove_vector(word)
            if vec is None:
                h = hash(word) % 50000
                vr = np.random.RandomState(h + 100)
                vec = vr.randn(self.dim).astype(np.float32) * 0.1
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
            node = self.graph.add_node(vector=vec, label=word)
            label_to_id[word] = node.id
            self._concept_keywords[word] = self._concept_keywords.get(word, []) + [node.id]
            self._concept_labels.add(word.lower())
            existing_labels.add(word)
            new_count += 1

        # Form connections between co-occurring important words
        # Words that appear together in the article get edges
        article_words_lower = [w for w in re.findall(r"[a-zA-Z']{3,}", text.lower())
                              if w not in STOP_WORDS and len(w.strip("'")) >= 3]
        # Get unique important words that appear in the text
        present_important = list(set(w for w in article_words_lower if w in important_words))

        for i in range(len(present_important)):
            for j in range(i + 1, len(present_important)):
                w1, w2 = present_important[i], present_important[j]
                nid1 = label_to_id.get(w1)
                nid2 = label_to_id.get(w2)
                if nid1 is not None and nid2 is not None:
                    # Don't duplicate existing edges, just boost weight
                    existing = self.graph.get_edge(nid1, nid2)
                    if existing is None:
                        weight = max(0.3, min(0.7, 0.2 + word_counts.get(w1, 1) * word_counts.get(w2, 1) * 0.001))
                        self.graph.add_edge(nid1, nid2, weight=weight, relation_type="semantic")
                    else:
                        existing.weight = min(0.9, existing.weight + 0.05)

        # Connect new words to existing related concepts via vector similarity
        # Only do this for the topic word and first 2 important words (others
        # rely on co-occurrence edges, which are more topically coherent)
        for word in important_words[:3]:
            nid = label_to_id.get(word)
            if nid is None:
                continue
            node = self.graph.get_node(nid)
            if node is None:
                continue
            edge_count = 0
            for existing_nid, existing_node in self.graph.nodes.items():
                if existing_nid == nid or (existing_node.label and existing_node.label in important_words):
                    continue
                if existing_node.vector is not None and node.vector is not None:
                    sim = float(np.dot(node.vector, existing_node.vector))
                    if sim > 0.3:
                        if self.graph.get_edge(nid, existing_nid) is None:
                            weight = max(0.25, min(0.5, sim * 0.5))
                            self.graph.add_edge(nid, existing_nid, weight=weight,
                                                relation_type="semantic")
                            edge_count += 1
            # Guarantee at least one connection for the topic word
            if edge_count == 0 and word == important_words[0]:
                best_sim = 0.0
                best_nid = None
                for existing_nid, existing_node in self.graph.nodes.items():
                    if existing_nid == nid or existing_node.label in important_words:
                        continue
                    if existing_node.vector is not None and node.vector is not None:
                        sim = float(np.dot(node.vector, existing_node.vector))
                        if sim > best_sim:
                            best_sim = sim
                            best_nid = existing_nid
                if best_nid is not None:
                    weight = max(0.25, min(0.4, best_sim * 0.5))
                    self.graph.add_edge(nid, best_nid, weight=weight, relation_type="semantic")
        return new_count

    def _learn_from_snippets(self, query: str, snippets: List[str]) -> str:
        """Learn from search snippet text when article fetch fails."""
        combined = f"{query} " + " ".join(snippets[:3])
        count = self._learn_from_text(combined, query)
        if count > 0:
            return f"learned {count} new things about {query} from search snippets"
        return f"read about {query} but already knew those words"

    # ─── Core Response Pipeline ───

    def process_turn(self, user_input: str) -> str:
        """Process input and generate a response, auto-learning when needed."""
        self.turn_count += 1
        self._learned_this_turn = False

        # Step 1: Find matching concepts
        activated = self._activate_from_input(user_input)

        # Step 1b.5: Phase 1 — Auto-expand graph from every message
        # Every input word that has a GloVe vector becomes a new concept,
        # wired to top-5 neighbors and all similar existing concepts.
        # No web search needed for expansion — purely local GloVe.
        new_concepts = self._auto_expand_concepts(user_input)
        if new_concepts > 0 and self._trace_enabled:
            print(f"  [trace]   auto-expanded {new_concepts} new concepts from input")
        # Re-activate in case new concepts were added
        if new_concepts > 0:
            activated = self._activate_from_input(user_input)

        # Phase 7: Store activated IDs for strategy framework
        self._last_activated_ids = list(activated)

        # Step 1b.75: Phase 3.3 + 9c — Detect recall triggers with hippocampal reactivation
        recall_topic = self._detect_recall_trigger(user_input)
        self._recall_mode = recall_topic is not None
        if recall_topic:
            # Phase 9c: Use hippocampal indexing to reactivate the distributed pattern
            reactivated = self._recall_hippocampal(recall_topic)
            if reactivated:
                for nid in reactivated:
                    if nid not in activated:
                        activated.append(nid)
                if self._trace_enabled:
                    print(f"  [trace]   hippocampal recall -> '{recall_topic}' "
                          f"reactivated {len(reactivated)} concepts")
            else:
                # Fallback: simple node activation
                rt_nids = self._concept_keywords.get(recall_topic.lower(), [])
                for nid in rt_nids:
                    if nid not in activated:
                        activated.append(nid)
                        self.graph.activate(nid, 0.8)
                if self._trace_enabled:
                    print(f"  [trace]   recall trigger -> '{recall_topic}' activated at 0.8")

        # Step 1c: If this is a follow-up (more/else/also), reactivate the latest
        # past topic so the graph walks find it naturally
        self._activation_boost: Optional[Dict[str, float]] = None
        if self._is_follow_up(user_input) and self._topic_list:
            last_topic = self._topic_list[-1]
            lt_nids = self._concept_keywords.get(last_topic.lower(), [])
            for nid in lt_nids:
                if nid not in activated:
                    activated.append(nid)
                    self.graph.activate(nid, 0.6)
            # Compute activation boost from user model's inferred preferences
            self._activation_boost = self.user_model.activation_boost_for(last_topic)
            # Phase 3.4: Bias chain walking toward edges from original response
            if self._response_context:
                last_ctx = self._response_context[-1]
                if last_ctx['subject'].lower() == last_topic.lower():
                    for f, t in last_ctx['hops']:
                        key = (f.lower(), t.lower())
                        self.user_model.edge_reactivations[key] = \
                            self.user_model.edge_reactivations.get(key, 0) + 1

        # Step 2: Extract topic
        subject, obj = self._extract_topic(user_input, activated)
        relation = "is"

        # Step 2b: Primary IDs — only these concepts spread activation
        # (other input-matched concepts provide context but don't propagate)
        subject_ids = set()
        sl = subject.lower()
        if sl in self._concept_keywords:
            subject_ids.update(self._concept_keywords[sl])

        # Step 3: Spread activation through graph
        associations = self._spread_and_collect(activated, primary_ids=subject_ids)

        # Step 4: Collect unknown words for deferred web learning
        # Phase 1.4: No hard lifetime cap — per-session rate limit (max 1 search per 3 turns)
        input_words = [w.strip(".,!?") for w in user_input.lower().split()
                      if len(w.strip(".,!?")) >= 3]
        known_words = sum(1 for w in input_words if w in self._concept_keywords)
        unknown_meaningful = [w for w in input_words
                              if w not in self._concept_keywords and w not in STOP_WORDS]
        # Phase 1.3: Collect unknown words into queue instead of searching synchronously
        if unknown_meaningful and self.baby_mode:
            for w in unknown_meaningful:
                if w not in self._pending_learning_queue:
                    self._pending_learning_queue.append(w)

        # Phase 11.1: Build context vector for this turn
        if subject:
            self._current_context_vector = self._build_context_vector(subject)
        else:
            self._current_context_vector = None
        
        # Step 5: Emotional modulation (with concept-specific tagging)
        self._update_emotion(user_input)
        for nid in activated:
            self._concept_vad[nid] = (
                self.emotion.state.valence,
                self.emotion.state.arousal,
                self.emotion.state.dominance,
            )

        # Step 6: Dual-process route
        confidence = self.identity.state.strength * 0.5 + 0.2
        route = self.dual_process.decide_route(
            confidence=confidence,
            novelty=0.1 if associations else 0.6,
            stakes=0.15,
        )

        # Step 6b: Meta-cognitive assessment (if we have enough turns)
        if self.turn_count > 3 and self.turn_count % 3 == 0:
            bias_report = self.meta_cog.detect_reasoning_bias(self.turn_count)
            epistemic_mode = self.meta_cog.recommend_epistemic_mode(self.turn_count)
            # Phase 17.4: Metacognitive review every 5 turns
            if self.turn_count % 5 == 0:
                self._metacognitive_review()
        else:
            epistemic_mode = self.meta_cog.current_mode

        # Step 6c: Sleep pressure accumulation
        self._sleep_pressure += 0.02 + 0.01 * (1.0 - confidence)
        if self._sleep_pressure > 0.3 and (self.turn_count - self._last_sleep_episode) > 8:
            # Run a mini sleep cycle: consolidate knowledge
            self._sleep_consolidate()
            self._last_sleep_episode = self.turn_count
            self._sleep_pressure = 0.0

        # Step 7: Past topics
        past = self._recall_past(subject, obj)

        # Step 8: Build context and generate response
        # Phase 12: Detect brain state for schema modulation
        state = self._detect_brain_state()
        schema_ids = set()
        if state == 'heteromodal' or state == 'default':
            schema_ids = self._activate_schema(subject)
            if self._trace_enabled and schema_ids:
                print(f'  [trace]   {state} mode: schema activated {len(schema_ids)} concepts')

        ctx = CognitiveResponseContext(
            subject=subject, relation=relation, object=obj, raw_input=user_input,
            associated_concepts=associations,
            bridge_concept=self._find_bridge(associations, subject),
            valence=self.emotion.state.valence, arousal=self.emotion.state.arousal,
            dominance=self.emotion.state.dominance,
            emotional_label=self.emotion.get_emotional_label(),
            identity_strength=self.identity.state.strength,
            identity_trend=self.identity.get_trend(),
            dissonance=self._free_energy,
            processing_route=route.route.value, route_reason=route.reason,
            past_topics=past, turn_count=self.turn_count,
            meaning_generated=self.meaning.accumulated_meaning,
            exploration_drive=0.3 * (1 - self.identity.state.strength) + 0.2 * self.emotion.state.arousal,
            learned_recently=self._learned_this_turn,
            recall_mode=getattr(self, '_recall_mode', False),
        )

        response, strategy = self._generate_response(ctx)
        self._last_strategy = strategy

        # Step 9: Update cognitive state
        self._update_state(ctx)

        # Phase 3.1 + 9c: Track topics with hippocampal indexing
        if subject:
            sl = subject.lower()
            # Build hop labels for hippocampal index
            hop_labels = []
            for hops_list in self._last_chain_hops:
                for f, t in hops_list:
                    hop_labels.append((f, t))
            # Create hippocampal index (stores sparese pointers to graph pattern)
            self._hippocampal_index_topic(subject, list(activated) if activated else [],
                                          hop_labels)
            if not any(t.lower() == sl for t in self._topic_list):
                self._topic_list.append(subject)
        # Keep last 50 topics
        if len(self._topic_list) > 50:
            removed = self._topic_list[:-50]
            self._topic_list = self._topic_list[-50:]
            for r in removed:
                self._topic_store.pop(r.lower(), None)

        # Step 11: Store episodic memory — link subject to its associations
        self._store_episodic(subject, associations)

        self._last_responses.append(response)
        if len(self._last_responses) > 10:
            self._last_responses = self._last_responses[-10:]

        # Phase 16.5: Update cerebellar n-gram model
        for hops_list in self._last_chain_hops:
            self._update_cerebellar_ngram(hops_list)
        
        # Phase 3.4: Store response context for follow-up bias
        hop_labels = []
        for hops_list in self._last_chain_hops:
            for f, t in hops_list:
                hop_labels.append((f, t))
        self._response_context.append({
            'subject': subject,
            'response': response,
            'hops': hop_labels,
            'turn': self.turn_count,
        })
        if len(self._response_context) > 10:
            self._response_context = self._response_context[-10:]

        # Step 12: Phase 1.3 — Flush deferred learning queue (rate-limited)
        self._turns_since_last_search += 1
        if (self._pending_learning_queue and
                self._turns_since_last_search >= 3):  # Max 1 search per 3 turns
            learn_query = self._pending_learning_queue.pop(0)
            if self._trace_enabled:
                print(f"  [trace]   deferred web learn: {learn_query}")
            # Phase 5: learn_from_web handles offline fallback internally
            learn_summary = self.learn_from_web(learn_query)
            # If network is down, re-queue the item for later
            if self._network_available is False:
                self._pending_learning_queue.insert(0, learn_query)
                # Schedule network retry in 20 turns (only on first failure)
                if self._network_retry_turn == 0:
                    self._network_retry_turn = self.turn_count + 20
            self._turns_since_last_search = 0

        return response

    def _extract_learning_query(self, text: str, activated_ids: List[int]) -> Optional[str]:
        """Extract what topic RAVANA should search for.

        Uses the LEAST-known word (not matched to any concept) as the query.
        """
        words = re.findall(r"[a-zA-Z']{3,}", text.lower())
        # Find words NOT matched to any concept
        matched_labels = set()
        for nid in activated_ids:
            node = self.graph.get_node(nid)
            if node and node.label:
                matched_labels.add(node.label.lower())

        meaningful = [w.strip("'") for w in words if w.strip("'") not in STOP_WORDS]
        # Pick the last meaningful word that is NOT already known
        for w in reversed(meaningful):
            if w not in matched_labels:
                return w
        # If all words are known, use the last meaningful word anyway
        if meaningful:
            return meaningful[-1]
        return None

    def _activate_from_input(self, text: str) -> List[int]:
        """Find matching concepts using keywords and label matching."""
        words = re.findall(r"[a-zA-Z']{1,}", text.lower())
        scores: Dict[int, float] = {}

        for w in words:
            if w in self._concept_keywords:
                for nid in self._concept_keywords[w]:
                    scores[nid] = scores.get(nid, 0) + 5.0

        for nid, node in self.graph.nodes.items():
            if not node.label:
                continue
            label = node.label.lower()
            s = scores.get(nid, 0.0)
            for w in words:
                if label == w:
                    s += 5.0
                elif len(w) >= 3 and (label == w or label.startswith(w + " ") or (" " + w + " ") in label or label.endswith(" " + w)):
                    s += 3.0
                elif len(label) >= 3 and label in w:
                    s += 2.0
            if s > 0:
                scores[nid] = s

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        activated = []
        for nid, sc in sorted_scores[:5]:
            self.graph.activate(nid, min(1.0, sc * 0.15))
            activated.append(nid)
        return activated

    QUESTION_WORDS = {"what", "why", "how", "when", "where", "who", "which",
                        "does", "do", "is", "are", "can", "will", "would",
                        "could", "should", "did", "have", "has", "had"}

    # Words that signal the user is asking for more on a previous topic
    FOLLOW_UP_WORDS = {"more", "else", "another", "also", "further",
                       "other", "additionally", "favorite"}

    # Phase 3.3: Recall trigger patterns
    RECALL_TRIGGERS = [
        "remember when", "remember we", "earlier you", "you said",
        "you mentioned", "we talked about", "we discussed", "before you",
        "what did you say about", "what did we say about",
        "recall", "previously", "last time",
    ]

    def _is_follow_up(self, text: str) -> bool:
        words = set(w.lower().strip(".,!?") for w in text.split())
        return bool(words & self.FOLLOW_UP_WORDS)

    def _store_episodic(self, subject: str, associations: List[Tuple[str, float]]):
        """Create episodic edges linking current subject to top associations.
        Phase 3.2: On revisit, boost weight. 3+ visits => migrate to semantic."""
        if not subject or not associations:
            return
        subj_nids = self._concept_keywords.get(subject.lower(), [])
        if not subj_nids:
            return
        subj_nid = subj_nids[0]
        for assoc_label, _ in associations[:3]:
            assoc_nids = self._concept_keywords.get(assoc_label.lower(), [])
            if not assoc_nids:
                continue
            assoc_nid = assoc_nids[0]
            existing = self.graph.get_edge(subj_nid, assoc_nid)
            if existing is None:
                self.graph.add_edge(subj_nid, assoc_nid,
                                    weight=0.15, relation_type="episodic")
            elif existing.relation_type == "episodic":
                sl = subject.lower()
                entry = self._topic_store.get(sl, {})
                visits = entry.get('visit_count', 1) if isinstance(entry, dict) else 1
                if visits >= 3:
                    existing.relation_type = "semantic"
                    existing.weight = min(0.40, existing.weight + 0.15)
                elif visits >= 2:
                    existing.weight = min(0.30, existing.weight + 0.10)

    TOPIC_SKIP_WORDS = {"i", "you", "we", "they", "he", "she", "it", "me", "my",
                        "your", "our", "their", "him", "her", "its", "this", "that",
                        "these", "those", "there", "here", "some", "any", "all",
                        "each", "every", "both", "one", "more", "most", "few",
                        "very", "too", "just", "about", "also", "then", "than",
                        "now", "then", "well", "like", "such", "same", "still",
                        "even", "much", "really", "quite",
                        # Generic verbs picked up by keyword matching instead of the real subject
                        "think", "know", "feel", "want", "need", "go", "come",
                        "get", "say", "make", "take", "see", "hear", "tell",
                        "give", "let", "put", "keep", "look", "find", "ask",
                        # Generic adjectives & filler that make poor conversation topics
                        "good", "bad", "big", "small", "always", "never", "maybe",
                        "if", "but", "in", "out", "up", "down",
                        "point", "way", "thing", "stuff",
                        "and", "so"}

    def _extract_topic(self, text: str, activated: List[int]) -> Tuple[str, str]:
        """Extract the main topic from input. Uses graph-activated concepts
        first, then falls back to pattern detection.

        For 'what is trust' -> 'trust'
        For 'you know i was thinking about trust' -> 'trust' (skips 'you', 'i')
        For 'does learning change your brain' -> 'learning'
        """
        # Check for "what is X" / "what's X" patterns first
        m = re.match(r"what'?s?\s+(?:is\s+|are\s+)?(.+)", text.lower().strip(" ?!."))
        if m:
            phrase = m.group(1).strip()
            words = phrase.split()
            # Prefer concepts that exist in the graph, fall back to last meaningful word
            for w in reversed(words):
                wc = w.strip(".,!?")
                if (len(wc) > 2 and wc not in self.QUESTION_WORDS
                        and wc not in self.TOPIC_SKIP_WORDS
                        and wc not in STOP_WORDS):
                    if wc in self._concept_labels:
                        return (wc, text)
            # Try keyword-matched concepts
            for w in reversed(words):
                wc = w.strip(".,!?")
                if (len(wc) > 2 and wc not in self.QUESTION_WORDS
                        and wc not in self.TOPIC_SKIP_WORDS
                        and wc not in STOP_WORDS
                        and wc in self._concept_keywords):
                    return (wc, text)
            # No graph concept found — return last meaningful word
            for w in reversed(words):
                wc = w.strip(".,!?")
                if (len(wc) > 2 and wc not in self.QUESTION_WORDS
                        and wc not in self.TOPIC_SKIP_WORDS
                        and wc not in STOP_WORDS):
                    return (wc, text)
            if words:
                return (words[-1].strip(".,!?"), text)

        # Use best activated concept (skip question, topic-skip, and short words)
        if activated:
            best_real = None
            for nid in activated:
                node = self.graph.get_node(nid)
                if node and node.label:
                    lbl = node.label.lower()
                    if (len(lbl) > 2 and lbl not in self.QUESTION_WORDS
                            and lbl not in self.TOPIC_SKIP_WORDS):
                        if best_real is None:
                            best_real = (node.label, text)
            if best_real:
                return best_real
            nid = activated[0]
            node = self.graph.get_node(nid)
            if node and node.label:
                return (node.label, text)

        # Fallback: find meaningful words
        words = [w.strip(".,!?") for w in text.lower().split()
                 if len(w.strip(".,!?")) > 2
                 and w.strip(".,!?") not in self.QUESTION_WORDS
                 and w.strip(".,!?") not in self.TOPIC_SKIP_WORDS
                 and w.strip(".,!?") not in STOP_WORDS]
        if words:
            return (words[-1], text)

        first = text.split()[0] if text.split() else ""
        return (first, text)

    def _spread_and_collect(self, seed_ids: List[int],
                             primary_ids: Optional[Set[int]] = None) -> List[Tuple[str, float]]:
        """Propagate activation through graph edges (3 hops).

        Only concepts in primary_ids (or all seed_ids if not specified)
        serve as activation sources. Other seed_ids get context activation
        (0.3) but don't propagate — they only prevent their neighbors from
        being collected as novel associations.
        """
        if not seed_ids:
            return []
        seed_set = set(seed_ids)
        spread_set = primary_ids if primary_ids else seed_set

        # Base activation: context (0.3) for all seeds, full (1.0) for spread sources
        for nid in seed_ids:
            self.graph.activate(nid, 1.0 if nid in spread_set else 0.3)

        for hop in range(3):
            new_acts: Dict[int, float] = {}
            decay = 0.7 ** (hop + 1)
            for nid in list(self.graph._active_nodes):
                node = self.graph.nodes.get(nid)
                if node is None or node.activation <= 0.01:
                    continue
                # Only propagate from primary spread sources
                if nid not in spread_set:
                    continue
                # Follow outgoing edges
                for tid, edge in self.graph.get_outgoing(nid):
                    if tid in seed_set:
                        continue
                    signal = node.activation * edge.weight * edge.confidence * decay
                    if signal > 0.01:
                        new_acts[tid] = new_acts.get(tid, 0.0) + signal
                # Follow incoming edges (semantic network is effectively undirected)
                for src, edge in self.graph.get_incoming(nid):
                    if src in seed_set:
                        continue
                    signal = node.activation * edge.weight * edge.confidence * decay
                    if signal > 0.01:
                        new_acts[src] = new_acts.get(src, 0.0) + signal
            for nid, sig in new_acts.items():
                if nid in self.graph.nodes:
                    self.graph.activate(nid, sig)

        collected = []
        for nid, node in self.graph.nodes.items():
            if nid not in seed_set and node.activation > 0.05:
                collected.append((node.label or f"c{nid}", float(node.activation)))

        collected.sort(key=lambda x: x[1], reverse=True)
        for node in self.graph.nodes.values():
            node.activation = 0.0
        return collected[:12]

    def _find_bridge(self, assoc: List[Tuple[str, float]], subj: str) -> str:
        if not assoc:
            return subj if subj else ""
        best = assoc[0][0]
        if best.lower() == subj.lower() and len(assoc) > 1:
            return assoc[1][0]
        return best

    def _update_emotion(self, text: str):
        """More nuanced emotional processing — teenage range of emotions."""
        positive = {"good", "great", "happy", "love", "nice", "fun", "yay", "wow",
                     "cool", "amazing", "awesome", "wonderful", "beautiful", "excited",
                     "grateful", "proud", "hopeful", "joy", "interesting"}
        negative = {"bad", "sad", "scared", "angry", "hurt", "cry", "mean",
                     "terrible", "awful", "upset", "frustrated", "anxious",
                     "worried", "disappointed", "lonely", "guilty", "afraid"}
        curious = {"why", "how", "what", "wonder", "curious", "interesting",
                    "really", "tell me", "explain", "mean"}
        words = set(w.lower().strip(".,!?") for w in text.split())
        sv = 0.0
        sa = 0.2  # baseline engagement floor
        # Positive words boost valence
        if words & positive:
            sv += 0.4
            sa += 0.2
        # Negative words lower valence
        if words & negative:
            sv -= 0.4
            sa += 0.25
        # Curiosity words increase arousal (engagement)
        if words & curious:
            sa += 0.3
            if sv == 0.0:
                sv += 0.05  # slight positive bias for curiosity
        # Learning excitement
        if self._learned_this_turn:
            sa += 0.3
            sv += 0.2
        # Novelty-based arousal (unknown words = mild surprise)
        input_words = [w for w in words if len(w) >= 3]
        known = sum(1 for w in input_words if w in self._concept_keywords)
        if input_words and known / len(input_words) < 0.5:
            sa += 0.15  # novelty surprise
        # Phase 9b: Prediction error surprise (Active Inference)
        # High prediction error = world doesn't match expectations = arousal
        if self._prediction_error_count > 5:
            pe_surprise = min(0.4, self._mean_prediction_error * 2.0)
            sa += pe_surprise
        # Phase 10.4: N400-like arousal modulation from per-hop prediction error
        if hasattr(self, '_mean_sentence_pe') and self._sentence_pe_count > 0:
            n400_surprise = min(0.3, self._mean_sentence_pe * 2.0)
            sa += n400_surprise
        # Phase 14.4: Identity prediction error
        if hasattr(self, '_expected_strength'):
            identity_pe = abs(self.identity.state.strength - self._expected_strength)
            if identity_pe > 0.3:
                sa += min(0.3, identity_pe * 0.5)
        # Phase 7.5: Curiosity drive — boost arousal for impossible queries
        if getattr(self, '_last_strategy_used', '') in ('G_uncertainty', 'F_web_research'):
            sa += 0.6  # strong curiosity arousal for impossible query
            sv += 0.1  # slight positive valence for curiosity
        self.emotion.update(stimulus_valence=sv, stimulus_arousal=sa,
                           stimulus_dominance=self.identity.state.strength * 0.4 + 0.2,
                           uncertainty=self._free_energy * 0.5, dt=1.0)

    def _detect_recall_trigger(self, text: str) -> Optional[str]:
        """Phase 3.3: Detect if user is recalling a past topic."""
        text_lower = text.lower()
        for trigger in self.RECALL_TRIGGERS:
            if trigger in text_lower:
                trigger_idx = text_lower.index(trigger) + len(trigger)
                after = text_lower[trigger_idx:].strip().strip(".,!?").split()
                for word in after:
                    for t in reversed(self._topic_list):
                        if t.lower() == word or (len(word) >= 3 and t.lower().startswith(word)):
                            return t
                if self._topic_list:
                    return self._topic_list[-1]
        return None

    def _recall_past(self, subj: str, obj: str) -> List[str]:
        related = []
        for t in self._topic_list:
            pl = t.lower()
            sl = subj.lower()
            if pl != sl and (pl in sl or sl in pl or len(set(pl.split()) & set(sl.split())) > 0):
                related.append(t)
        return related[:3]

    # ─── Phase 9c: Hippocampal Indexing ───
    # Based on: Teyler & Rudy (2007) hippocampal indexing theory.
    # The hippocampus stores an INDEX to distributed neocortical patterns,
    # not the memory content itself. Reactivation of the index → reactivation
    # of the distributed pattern → memory experience.
    #
    # Instead of storing full topic content, we store a sparse index:
    # which concept IDs were activated, which edges were traversed.
    # During recall, the index reactivates the distributed graph pattern.

    def _hippocampal_index_topic(self, subject: str, activated_ids: List[int],
                                   hop_labels: List[Tuple[str, str]]):
        """Create a hippocampal index for the current topic and store it.

        The index is a lightweight pointer to the distributed graph pattern
        (concept IDs + edge references), not the content itself.
        """
        sl = subject.lower()
        # Build index: which concept nodes were activated
        indexed_concepts = list(set(activated_ids))

        # Build index: which edge pairs were traversed
        indexed_edges = [(f.lower(), t.lower()) for f, t in hop_labels]

        # Store as lightweight index, not full content
        index_entry = {
            'label': subject,
            'turn': self.turn_count,
            'indexed_concepts': indexed_concepts[:10],  # sparse index
            'indexed_edges': indexed_edges[:5],
            'vad': (self.emotion.state.valence, self.emotion.state.arousal,
                    self.emotion.state.dominance),
            'visit_count': 1,
            'response_summary': '',  # placeholder, not content
        }

        if sl not in self._topic_store:
            self._topic_store[sl] = index_entry
        else:
            entry = self._topic_store[sl]
            entry['visit_count'] += 1
            entry['turn'] = self.turn_count
            # Merge new indexed concepts
            existing_cons = set(entry.get('indexed_concepts', []))
            existing_cons.update(indexed_concepts[:10])
            entry['indexed_concepts'] = list(existing_cons)[:15]
            entry['vad'] = index_entry['vad']

    def _recall_hippocampal(self, topic: str) -> Optional[List[int]]:
        """Reactivate a hippocampal index, spreading activation through the
        indexed graph pattern to reconstruct the memory experience.

        Returns the list of reactivated concept IDs, or None if topic not found.
        """
        entry = self._topic_store.get(topic.lower())
        if not entry:
            return None

        reactivated = []
        # Phase 1: Reactivate indexed concepts (sparse pattern)
        for nid in entry.get('indexed_concepts', []):
            node = self.graph.get_node(nid)
            if node and node.label:
                self.graph.activate(nid, 0.5)
                reactivated.append(nid)

        # Phase 2: Spread activation through indexed edges (pattern completion)
        for f_label, t_label in entry.get('indexed_edges', []):
            f_nids = self._concept_keywords.get(f_label.lower(), [])
            t_nids = self._concept_keywords.get(t_label.lower(), [])
            for fn in f_nids:
                for tn in t_nids:
                    edge = self.graph.get_edge(fn, tn)
                    if edge:
                        # Strengthen episodic edges during recall (pattern strengthening)
                        if edge.relation_type == "episodic":
                            edge.weight = min(0.35, edge.weight + 0.05)
                        # Activate both endpoints
                        self.graph.activate(fn, 0.4)
                        self.graph.activate(tn, 0.4)
                        if fn not in reactivated:
                            reactivated.append(fn)
                        if tn not in reactivated:
                            reactivated.append(tn)

        # Phase 3: Activate the subject concept at higher strength
        subj_nids = self._concept_keywords.get(topic.lower(), [])
        for sn in subj_nids:
            self.graph.activate(sn, 0.7)
            if sn not in reactivated:
                reactivated.append(sn)

        return reactivated

    # ─── Graph-Driven Response Generation ───
    # NO hardcoded strings. ALL content words are concept labels from the graph.
    # Edge relation types (stored in graph edge data) are mapped to graph concept
    # labels that already exist in the seeded vocabulary — no external text.

    # Tiered connectors: (min_weight, [word, ...])
    # Auto-wired edges: min=0.30, ~0.30-0.40 common, max=0.50
    # Weight ≥ 0.33 (~top 25%): stronger connector (e.g. "link", "and")
    # Weight 0.30-0.33 (bulk): default "connect"
    # Lower temperature = conservative (pick first option), higher = random.
    _EDGE_CONNECTORS = {
        "semantic": [
            (0.35, ["link", "and", "connect"]),
            (0.0, ["connect"]),
        ],
        "causal": [
            (0.33, ["make", "create", "cause"]),
            (0.0, ["cause", "so", "because"]),
        ],
        "analogical": [
            (0.33, ["like", "love"]),
            (0.0, ["like"]),
        ],
        "contrastive": [
            (0.20, ["but", "but", "but"]),
            (0.0, ["but"]),
        ],
        "temporal": [
            (0.28, ["change", "then"]),
            (0.0, ["then"]),
        ],
        "episodic": [
            (0.20, ["connect", "and"]),
            (0.0, ["connect"]),
        ],
    }

    _EDGE_TO_GRAPH_LABEL = {
        "causal": "cause",
        "analogical": "like",
        "semantic": "connect",
        "contrastive": "but",
        "temporal": "change",
        "episodic": "connect",
    }

    # Edge type → discourse starter (all labels exist in seeded TEEN_CONCEPTS)
    _EDGE_TO_STARTER = {
        "causal": "because",
        "contrastive": "but",
        "semantic": "and",
        "analogical": "like",
        "temporal": "then",
    }

    def _get_edge_info(self, src_label: str, tgt_label: str) -> Optional[Dict]:
        """Get edge information between two concept labels from the graph."""
        src_ids = self._concept_keywords.get(src_label.lower(), [])
        tgt_ids = self._concept_keywords.get(tgt_label.lower(), [])
        for sid in src_ids:
            for tid in tgt_ids:
                edge = self.graph.get_edge(sid, tid)
                if edge:
                    return {'weight': edge.weight, 'confidence': edge.confidence,
                            'relation_type': edge.relation_type, 'reversed': False}
                edge = self.graph.get_edge(tid, sid)
                if edge:
                    return {'weight': edge.weight, 'confidence': edge.confidence,
                            'relation_type': edge.relation_type, 'reversed': True}
        return None

    def _group_associations(self, subject: str, assocs: List,
                             seen_labels: Set[str],
                             max_total: int = 5) -> Dict[str, List[str]]:
        """Group associations by their relation type to subject."""
        groups: Dict[str, List[str]] = {}
        for concept, _ in assocs:
            if len(seen_labels) > max_total:
                break
            key = concept.lower()
            if key in seen_labels:
                continue
            seen_labels.add(key)
            edge_info = self._get_edge_info(subject, concept)
            rel = edge_info['relation_type'] if edge_info else "semantic"
            groups.setdefault(rel, []).append(concept)
        return groups

    def _rel_clause(self, subject: str, rel: str, concepts: List[str]) -> str:
        """Build a clause like 'subject cause concept1 concept2'."""
        connector = self._pick_connector(rel, 0.5, 1.0, 0.25)
        concepts = [c for c in concepts if c.lower() != connector]
        if not concepts:
            return ""
        if connector:
            return f"{subject} {connector} {' '.join(concepts)}"
        return f"{subject} {' '.join(concepts)}"

    def _build_phrase(self, subject: str, groups: Dict[str, List[str]]) -> str:
        """Build a phrase from grouped associations: subject connector group1 group2..."""
        parts = [subject]
        for rel, concepts in groups.items():
            connector = self._EDGE_TO_GRAPH_LABEL.get(rel, "")
            if connector:
                if parts[-1] != connector:
                    parts.append(connector)
                concepts = [c for c in concepts if c.lower() != connector]
            parts.extend(concepts)
        return " ".join(parts)

    # ─── VAD Modulation ───

    def _get_temperature(self, sentence_idx: int,
                         base_temps: Optional[List[float]] = None) -> float:
        """Return temperature for sentence, scaled by current arousal.

        High arousal → more exploration (higher temp).
        Low arousal → more focused (lower temp).
        """
        if not getattr(self, 'use_vad', True):
            base = (base_temps or [0.08, 0.15, 0.25])[min(sentence_idx, 2)]
            return base
        base_temps = base_temps or [0.08, 0.15, 0.25]
        base = base_temps[min(sentence_idx, 2)]
        arousal = self.emotion.state.arousal  # 0..1
        arousal_factor = 0.7 + (arousal * 0.6)  # [0.7, 1.3]
        temp = base * arousal_factor
        # Phase 12.2: Brain state modulation of temperature
        state = getattr(self, '_cognitive_state', 'default')
        if state == "heteromodal":
            temp *= 1.3  # more exploration
        elif state == "unimodal":
            temp *= 0.7  # more precision
        # Phase 14.3: Dopamine tone modulation of temperature
        dt = getattr(self, '_dopamine_tone', 0.5)
        dt_factor = 0.7 + (dt - 0.5) * 1.0  # [0.7, 1.1] at [0.1, 0.9]
        temp *= dt_factor
        return temp

    def _vad_decay(self, rate: float = 0.95):
        """Return VAD toward neutral each turn."""
        if not getattr(self, 'use_vad', True):
            return
        s = self.emotion.state
        s.valence *= rate
        s.arousal = 0.3 + (s.arousal - 0.3) * rate
        s.dominance = 0.5 + (s.dominance - 0.5) * rate

    # ─── Relation Verifier (lightweight RLMv2 wrapper) ───

    def _classify_triple(self, source_label: str, target_label: str,
                         relation_type: str) -> Dict[str, Any]:
        """Verify a (source, relation, target) triple using graph + vector data.

        Returns:
            {'confidence': float (0..1),
             'top_relations': List[(rel_type, score)],
             'is_analogical': bool,
             'suggested_type': str}
        """
        src_nids = self._concept_keywords.get(source_label.lower(), [])
        tgt_nids = self._concept_keywords.get(target_label.lower(), [])
        if not src_nids or not tgt_nids:
            return {'confidence': 0.0, 'top_relations': [],
                    'is_analogical': False, 'suggested_type': relation_type}

        # Edge-based confidence
        edge_info = self._get_edge_info(source_label, target_label)
        if edge_info:
            edge_conf = edge_info['weight'] * edge_info['confidence']
            matches = edge_info['relation_type'] == relation_type
        else:
            edge_conf = 0.0
            matches = False

        # Vector similarity confidence (cosine)
        src_node = self.graph.get_node(src_nids[0])
        tgt_node = self.graph.get_node(tgt_nids[0])
        vec_conf = 0.0
        if src_node and tgt_node and src_node.vector is not None and tgt_node.vector is not None:
            vec_conf = float(np.dot(src_node.vector, tgt_node.vector))

        # Blend: edge (0.6) + vector (0.4), with boost if edge matches
        if matches:
            confidence = min(1.0, edge_conf * 0.7 + vec_conf * 0.3)
        elif edge_conf > 0:
            confidence = 0.3 + edge_conf * 0.3
        else:
            confidence = max(0.0, vec_conf * 0.4 - 0.1)

        # Rank all relation types by vector-based plausibility
        top_relations = self._rank_relations(source_label, target_label, relation_type)

        # Cross-domain detection: different concept categories
        is_analogical = self._is_cross_domain(source_label, target_label)

        return {
            'confidence': confidence,
            'top_relations': top_relations,
            'is_analogical': is_analogical,
            'suggested_type': top_relations[0][0] if top_relations else relation_type,
        }

    DOMAIN_MAP = {
        "emotion": {"love", "hate", "fear", "joy", "sad", "angry", "happy", "scared",
                     "excited", "curious", "confused", "bored", "proud", "lonely",
                     "grateful", "anxiety", "excitement", "frustration", "hope",
                     "grief", "surprise", "guilt", "disappointment",
                     "empathy", "lonely", "heart", "feel"},
        "abstract": {"truth", "justice", "freedom", "power", "knowledge", "wisdom",
                      "meaning", "pattern", "system", "perspective", "context",
                      "paradox", "principle", "theory", "evidence", "analysis",
                      "conclusion", "logic", "intuition", "identity", "culture",
                      "responsibility", "trust", "hypocrisy", "respect",
                      "belief", "significant", "fundamental", "inevitable",
                      "possible", "obvious", "subtle", "profound"},
        "social": {"friend", "people", "person", "trust", "respect",
                    "culture", "power", "freedom", "responsibility",
                    "we", "they", "you", "i", "friend"},
        "concrete": {"water", "food", "home", "sun", "moon", "tree", "bird",
                      "dog", "cat", "book", "song", "world", "nature",
                      "machine", "invention", "water", "food", "sun"},
        "action": {"make", "create", "destroy", "protect", "accept", "reject",
                    "analyze", "conclude", "reflect", "question", "explore",
                    "understand", "compare", "criticize", "assume", "imagine",
                    "cause", "change", "grow", "learn", "teach",
                    "go", "come", "see", "hear", "eat", "drink", "sleep",
                    "play", "help", "get", "know", "think", "say", "feel",
                    "give", "take"},
    }

    def _is_cross_domain(self, label_a: str, label_b: str) -> bool:
        """Check if two concepts belong to different domains."""
        la = label_a.lower()
        lb = label_b.lower()
        domain_a = None
        domain_b = None
        for domain, words in self.DOMAIN_MAP.items():
            if la in words:
                domain_a = domain
            if lb in words:
                domain_b = domain
        return domain_a is not None and domain_b is not None and domain_a != domain_b

    def _rank_relations(self, source_label: str, target_label: str,
                        default_type: str) -> List[Tuple[str, float]]:
        """Rank relation types by vector-based plausibility for a concept pair."""
        src_nids = self._concept_keywords.get(source_label.lower(), [])
        tgt_nids = self._concept_keywords.get(target_label.lower(), [])
        if not src_nids or not tgt_nids:
            return [(default_type, 0.5)]

        src_node = self.graph.get_node(src_nids[0])
        tgt_node = self.graph.get_node(tgt_nids[0])
        if src_node is None or tgt_node is None:
            return [(default_type, 0.5)]

        src_vec = src_node.vector
        tgt_vec = tgt_node.vector
        if src_vec is None or tgt_vec is None:
            return [(default_type, 0.5)]

        # Causal scoring: positive + strong magnitude = likely causal
        cosine = float(np.dot(src_vec, tgt_vec))
        mag_ratio = float(np.linalg.norm(tgt_vec) / max(np.linalg.norm(src_vec), 0.001))

        scores = {
            "semantic": max(0.0, cosine),
            "causal": max(0.0, 0.3 + 0.3 * mag_ratio - 0.2 * (1.0 - abs(cosine))),
            "contrastive": max(0.0, 0.5 - 0.5 * cosine),
            "analogical": max(0.0, 0.2 + 0.3 * cosine - 0.3 * abs(mag_ratio - 1.0)),
            "temporal": 0.1,
            "emotional": 0.1,
        }

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked

    def _find_vector_neighbor(self, label: str) -> Optional[str]:
        """Find the most semantically similar concept (GloVe cosine similarity)."""
        nids = self._concept_keywords.get(label.lower(), [])
        if not nids:
            return None
        nid = nids[0]
        node = self.graph.get_node(nid)
        if node is None or node.vector is None:
            return None
        best_sim = 0.45
        best_label = None
        for other_nid, other_node in self.graph.nodes.items():
            if other_nid == nid or other_node.label is None:
                continue
            if other_node.vector is None:
                continue
            sim = float(np.dot(node.vector, other_node.vector))
            if sim > best_sim:
                best_sim = sim
                best_label = other_node.label
        return best_label

    def _phrase_from_label(self, label: str, seen_labels: Set[str],
                           max_concepts: int = 4) -> Optional[str]:
        """Walk the graph from a single label and return a phrase."""
        nids = self._concept_keywords.get(label.lower(), [])
        if not nids:
            return None
        assocs = self._spread_and_collect(nids, primary_ids=set(nids))
        if not assocs:
            return None
        local_seen = set(seen_labels)
        local_seen.discard(label.lower())
        groups = self._group_associations(label, assocs, local_seen, max_total=max_concepts)
        if not groups:
            return None
        phrase = self._build_phrase(label, groups)
        if phrase == label:
            return None
        return phrase

    def _pick_connector(self, relation_type: str, weight: float,
                        confidence: float, temperature: float) -> str:
        """Pick a connector word based on edge weight, temp, and VAD state.

        Higher weight → stronger connector (e.g. "link" over "connect").
        Higher temperature → more random selection among options.
        High arousal → prefer stronger/assertive connectors.
        Low arousal → prefer weaker/softer connectors.
        All returned words are graph concept labels."""
        tiers = self._EDGE_CONNECTORS.get(relation_type, [])
        if not tiers:
            return ""
        for min_weight, options in tiers:
            if weight >= min_weight:
                # VAD modulation: arousal shifts preference within the tier
                if getattr(self, 'use_vad', True):
                    arousal = self.emotion.state.arousal
                    valence = self.emotion.state.valence
                    dominance = self.emotion.state.dominance
                    # High arousal + positive valence → prefer assertive (first) connector
                    if arousal > 0.50 and valence > 0.10 and len(options) > 1:
                        return options[0]
                    # Low arousal + negative valence → prefer weakest (last) connector
                    if arousal < 0.25 and valence < -0.10 and len(options) > 1:
                        return options[-1]
                # Standard temperature-driven choice
                if temperature < 0.2:
                    return options[0]
                return self.rng.choice(options)
        return tiers[-1][1][0]

    
    # ── Phase 10: Predictive Coding in Chain Walking ──
    # Neuroscience: N400 = lexico-semantic prediction error (Wang, Noureddine & Kuperberg 2025)
    # Each hop predicts the next concept; mismatch = prediction error

    def _compute_hop_prediction_error(self, cur_vec: np.ndarray,
                                       subj_vec: np.ndarray,
                                       pfc_centroid: np.ndarray,
                                       chosen_vec: np.ndarray) -> float:
        """Compute prediction error for a chain walk hop.
        
        Predicts next concept as weighted blend of current, subject, and PFC centroid.
        Error = 1.0 - cosine(predicted, actual)
        Returns 0.0 if vectors are invalid.
        """
        if cur_vec is None or chosen_vec is None:
            return 0.0
        if subj_vec is None:
            subj_vec = np.zeros_like(cur_vec)
        if pfc_centroid is None:
            pfc_centroid = np.zeros_like(cur_vec)
        
        predicted = cur_vec * 0.6 + subj_vec * 0.3 + pfc_centroid * 0.1
        norm = np.linalg.norm(predicted)
        if norm > 0:
            predicted /= norm
        actual_cos = float(np.dot(predicted, chosen_vec))
        error = 1.0 - actual_cos  # 0 = perfect, 1 = completely surprising
        return error

    def _update_sentence_schema(self, subject: str, new_concepts: List[str]):
        """Phase 10.3: Sparse-updating sentence schema.
        
        Schema only changes between sentences, not during chain walking.
        New concepts at weight 0.5, old concepts decay by 0.8.
        """
        # Decay existing
        for k in list(self._sentence_schema.keys()):
            self._sentence_schema[k] *= 0.8
            if self._sentence_schema[k] < 0.1:
                del self._sentence_schema[k]
        # Add subject always at high weight
        self._sentence_schema[subject.lower()] = 1.0
        # Add new concepts at moderate weight
        for c in new_concepts[:3]:
            cl = c.lower()
            if cl not in self._sentence_schema:
                self._sentence_schema[cl] = 0.5


    # ── Phase 11: Context-Dependent Vector Modulation ──
    # Neuroscience: PFC neurons represent SAME word differently per context (Nature 2024)
    # HPC broadcasts context state to OFC via theta synchronization (Nature Comms 2025)

    def _build_context_vector(self, subject: str) -> np.ndarray:
        """Build a context vector from topic, recent history, PFC, and emotion.
        
        Components:
        - Subject topic vector × 0.4
        - Mean of last 5 response concept vectors × 0.3
        - Current PFC buffer centroid × 0.2
        - Current VAD emotional vector × 0.1
        """
        components = []
        weights = []
        
        # Subject vector
        subj_nids = self._concept_keywords.get(subject.lower(), [])
        if subj_nids:
            subj_node = self.graph.get_node(subj_nids[0])
            if subj_node and subj_node.vector is not None:
                components.append(subj_node.vector)
                weights.append(0.4)
        
        # Recent response centroid
        recent_vecs = []
        for resp in self._last_responses[-3:]:
            for w in resp.split():
                wn = self._concept_keywords.get(w.lower(), [])
                if wn:
                    wn_node = self.graph.get_node(wn[0])
                    if wn_node and wn_node.vector is not None:
                        recent_vecs.append(wn_node.vector)
        if recent_vecs:
            components.append(np.mean(recent_vecs, axis=0))
            weights.append(0.3)
        
        # PFC buffer centroid
        pfc_vecs = []
        for bl in self._prefrontal_buffer:
            bn = self._concept_keywords.get(bl, [])
            if bn:
                bn_node = self.graph.get_node(bn[0])
                if bn_node and bn_node.vector is not None:
                    pfc_vecs.append(bn_node.vector)
        if pfc_vecs:
            components.append(np.mean(pfc_vecs, axis=0))
            weights.append(0.2)
        
        # Emotional vector
        e_vec = np.array([self.emotion.state.valence,
                          self.emotion.state.arousal,
                          self.emotion.state.dominance], dtype=np.float32)
        e_pad = np.zeros(self.dim, dtype=np.float32)
        e_pad[:3] = e_vec
        components.append(e_pad)
        weights.append(0.1)
        
        if not components:
            return np.zeros(self.dim, dtype=np.float32)
        
        ctx = np.average(np.array(components), axis=0, weights=np.array(weights))
        norm = np.linalg.norm(ctx)
        if norm > 0:
            ctx /= norm
        return ctx.astype(np.float32)

    def _modulate_vector(self, node_id: int) -> Optional[np.ndarray]:
        """Phase 11.2: Return context-modulated vector for a concept node.
        
        Modulated = original_vector + 0.15 * context_vector, then normalized.
        Returns None if node or vector is missing.
        """
        node = self.graph.get_node(node_id)
        if node is None or node.vector is None:
            return None
        ctx = getattr(self, '_current_context_vector', None)
        if ctx is None or np.all(ctx == 0):
            return node.vector
        modulated = node.vector + 0.15 * ctx
        norm = np.linalg.norm(modulated)
        if norm > 0:
            modulated /= norm
        return modulated.astype(np.float32)

    def _starter_from_chain(self, chain: str, subject: str,
                             connector_counts: Optional[Dict[str, int]] = None) -> str:
        """Extract the first edge relation type from a chain string and return
        a discourse starter (graph label). The matched starter is weighted 3x
        higher but other starters still have a chance — prevents monotony when
        most edges are semantic."""
        if not chain:
            return "and"
        matched = "and"
        parts = chain.split()
        if len(parts) >= 3:
            conn = parts[1].lower()
            rel_type = self._CONNECTOR_TO_REL.get(conn)
            if rel_type:
                starter = self._EDGE_TO_STARTER.get(rel_type)
                if starter:
                    matched = starter
        candidates = list(self._EDGE_TO_STARTER.values())
        # Phase 4.4: If "connect" has been used >3 times, force a different starter
        if connector_counts and connector_counts.get('connect', 0) >= 3:
            # Force non-semantic starter (but, because, like, then)
            non_semantic = [s for s in candidates if s != 'and']
            if non_semantic:
                candidates = non_semantic
                # Boost the matched starter weight within non-semantic
                weights = np.array([3.0 if s == matched else 1.0 for s in candidates], dtype=np.float64)
                weights /= weights.sum()
                return candidates[self.rng.choice(len(candidates), p=weights)]
        weights = np.array([3.0 if s == matched else 1.0 for s in candidates], dtype=np.float64)
        weights /= weights.sum()
        return candidates[self.rng.choice(len(candidates), p=weights)]

    # ─── Phase 9: Prefrontal Gating (Neuroscience-Inspired Working Memory) ───

    def _prefrontal_gate_candidates(self, candidates: List[Tuple],
                                      subject: str) -> List[Tuple]:
        """Gate candidates through prefrontal working memory buffer.

        Boosts concepts actively held in the prefrontal buffer (working memory).
        Mildly suppresses unrelated concepts to maintain topic coherence.
        Gating strength is modulated by arousal — teens have weaker gating
        under high arousal (limbic > PFC).

        Based on: Ott & Nieder (2019) — Dopamine and Cognitive Control in PFC.
        PFC D1/D2 receptors gate sensory input into working memory.
        """
        if not self._prefrontal_buffer or subject.lower() not in self._prefrontal_buffer:
            return candidates

        # Arousal modulates gating strength (high arousal = leaky gate)
        arousal = self.emotion.state.arousal if getattr(self, 'use_vad', True) else 0.3
        gate_strength = max(0.1, 0.5 - arousal * 0.4)  # 0.5 at calm, 0.26 at peak arousal
        boost_factor = 1.0 + 0.5 * gate_strength       # 1.5 at calm, 1.13 at peak
        penalty_factor = 0.6 + 0.3 * gate_strength     # 0.75 at calm, 0.68 at peak

        gated = []
        for sig, tgt_lbl, edge, d in candidates:
            if tgt_lbl.lower() in self._prefrontal_buffer:
                sig *= boost_factor
            else:
                # Check vector similarity to buffer contents
                tgt_nids = self._concept_keywords.get(tgt_lbl.lower(), [])
                if tgt_nids:
                    tgt_node = self.graph.get_node(tgt_nids[0])
                    if tgt_node and tgt_node.vector is not None:
                        best_cos = 0.0
                        for b_label in self._prefrontal_buffer:
                            b_nids = self._concept_keywords.get(b_label.lower(), [])
                            if b_nids:
                                b_node = self.graph.get_node(b_nids[0])
                                if b_node and b_node.vector is not None:
                                    cos = float(np.dot(tgt_node.vector, b_node.vector))
                                    if cos > best_cos:
                                        best_cos = cos
                        if best_cos > 0.25:
                            sig *= (1.0 + 0.35 * best_cos * gate_strength)
                        else:
                            sig *= penalty_factor
                    else:
                        sig *= penalty_factor
                else:
                    sig *= penalty_factor
            gated.append((sig, tgt_lbl, edge, d))
        return gated

    def _prefrontal_maintain_buffer(self, subject: str, chain_concepts: List[str]):
        """Update prefrontal working memory buffer with recent chain concepts.

        Maintains the subject + most vector-similar recently-used concepts.
        Caps at ~7 items (typical working memory capacity).
        Old buffer entries decay unless they are still semantically relevant.

        Based on: Chatham, Frank & Badre (2014) — Corticostriatal output gating
        during selection from working memory. PFC + basal ganglia control
        what stays in working memory and what gets expressed.
        """
        buffer = {subject.lower()}

        for label in chain_concepts:
            if label.lower() == subject.lower():
                continue
            if len(buffer) >= self._pfc_buffer_capacity:
                break
            subj_nids = self._concept_keywords.get(subject.lower(), [])
            lbl_nids = self._concept_keywords.get(label.lower(), [])
            if subj_nids and lbl_nids:
                subj_node = self.graph.get_node(subj_nids[0])
                lbl_node = self.graph.get_node(lbl_nids[0])
                if (subj_node and subj_node.vector is not None
                        and lbl_node and lbl_node.vector is not None):
                    cos = float(np.dot(lbl_node.vector, subj_node.vector))
                    if cos > 0.25:
                        buffer.add(label.lower())

        # Merge with existing buffer (keep old entries if still relevant)
        for label in self._prefrontal_buffer:
            if len(buffer) >= self._pfc_buffer_capacity:
                break
            if label not in buffer:
                lbl_nids = self._concept_keywords.get(label.lower(), [])
                subj_nids = self._concept_keywords.get(subject.lower(), [])
                if lbl_nids and subj_nids:
                    lbl_node = self.graph.get_node(lbl_nids[0])
                    subj_node = self.graph.get_node(subj_nids[0])
                    if (lbl_node and lbl_node.vector is not None
                            and subj_node and subj_node.vector is not None):
                        cos = float(np.dot(lbl_node.vector, subj_node.vector))
                        if cos > 0.15:
                            buffer.add(label.lower())

        self._prefrontal_buffer = list(buffer)

    
    # ── Phase 12: Schema-Level Activation ──
    # Neuroscience: vmPFC activates whole semantic schemas from minimal input (Entropy 2026)
    # Two brain states: heteromodal (integrative) and unimodal (focused) (Comms Bio 2024)

    def _activate_schema(self, subject: str) -> Set[int]:
        """Phase 12.1: Activate a schema cluster around the subject.
        
        Schema = top-K GloVe neighbors (cosine > 0.5) around the subject.
        All concepts in the schema get activation × 1.3 during spread-and-collect.
        Returns set of concept IDs in the schema.
        """
        subj_nids = self._concept_keywords.get(subject.lower(), [])
        if not subj_nids:
            return set()
        subj_nid = subj_nids[0]
        subj_node = self.graph.get_node(subj_nid)
        if subj_node is None or subj_node.vector is None:
            return set()
        
        # Dynamic threshold based on prediction error
        pe = getattr(self, '_mean_prediction_error', 0.3)
        threshold = 0.6 if pe < 0.2 else (0.4 if pe > 0.5 else 0.5)
        
        schema_ids = {subj_nid}
        for other_nid, other_node in self.graph.nodes.items():
            if other_nid == subj_nid or other_node.vector is None:
                continue
            cos = float(np.dot(subj_node.vector, other_node.vector))
            if cos > threshold:
                schema_ids.add(other_nid)
                self.graph.activate(other_nid, 0.6)
        return schema_ids

    def _detect_brain_state(self) -> str:
        """Phase 12.2: Detect whether RAVANA is in heteromodal (integrative) or unimodal (focused) state.
        
        Heteromodal: confidence < 0.3 OR mean_prediction_error > 0.4 OR novelty > 0.6
        Unimodal: confidence > 0.5 AND mean_prediction_error < 0.2 AND novelty < 0.3
        """
        confidence = self.identity.state.strength * 0.5 + 0.2
        pe = getattr(self, '_mean_prediction_error', 0.3)
        novelty = 0.1 if len(self._last_responses) > 0 else 0.6
        
        if confidence < 0.3 or pe > 0.4 or novelty > 0.6:
            new_state = "heteromodal"
        elif confidence > 0.5 and pe < 0.2 and novelty < 0.3:
            new_state = "unimodal"
        else:
            new_state = "default"
        
        # Damped state transitions: hold for 2 turns before switching
        if new_state != self._cognitive_state:
            if self._cognitive_state_hold > 0:
                self._cognitive_state_hold -= 1
            else:
                self._cognitive_state = new_state
                self._cognitive_state_hold = 2
                self._state_duration = 0
        else:
            self._cognitive_state_hold = 0
            self._state_duration += 1
        
        return self._cognitive_state


    # ── Phase 13: Adaptive Gain Control ──
    # Neuroscience: Semantic satiation = bottom-up gain reduction (Zhang, Comms Bio 2024)
    # Adolescent PFC has high Glu:GABA, more exploration, less stability (PMC 2024)

    def _apply_activation_fatigue(self, candidates: List[Tuple]) -> List[Tuple]:
        """Phase 13.1: Reduce signal for recently overused concepts.
        
        Fatigue accumulates by 0.15 per activation, decays 0.95 per turn.
        Signal *= max(0.3, 1.0 - fatigue). High-fatigue nodes fade after ~5 activations.
        """
        fatigued = []
        for item in candidates:
            sig, tgt_lbl, edge, d = item
            tgt_nids = self._concept_keywords.get(tgt_lbl.lower(), [])
            fatigue = 0.0
            for nid in tgt_nids:
                fatigue = max(fatigue, self._activation_fatigue.get(nid, 0.0))
            adjusted = sig * max(0.3, 1.0 - fatigue)
            fatigued.append((adjusted, tgt_lbl, edge, d))
        return fatigued

    def _apply_edge_repetition_penalty(self, candidates: List[Tuple]) -> List[Tuple]:
        """Phase 13.2: Penalize recently-traversed edges.
        
        Same edge within 1 turn: signal × 0.3
        Same edge within 2 turns: signal × 0.6
        Same edge within 3 turns: signal × 0.8
        """
        penalized = []
        for sig, tgt_lbl, edge, d in candidates:
            penalty = 1.0
            if edge is not None:
                key = (edge.source, edge.target)
                for i, (s, t) in enumerate(reversed(self._recent_traversals)):
                    if (s, t) == key:
                        turns_ago = i + 1
                        if turns_ago <= 1:
                            penalty = 0.3
                        elif turns_ago <= 2:
                            penalty = 0.6
                        elif turns_ago <= 3:
                            penalty = 0.8
                        break
            penalized.append((sig * penalty, tgt_lbl, edge, d))
        return penalized

    def _apply_novelty_bonus(self, candidates: List[Tuple]) -> List[Tuple]:
        """Phase 13.4: Bonus for unvisited concepts (teen exploration).
        
        First 50 turns: ×1.3 for novel concepts, after 50: ×1.05.
        """
        if self.turn_count < 50:
            novelty_mult = 1.3
        else:
            novelty_mult = 1.05
        
        boosted = []
        for sig, tgt_lbl, edge, d in candidates:
            if tgt_lbl.lower() not in self._visited_concepts:
                sig *= novelty_mult
            boosted.append((sig, tgt_lbl, edge, d))
        return boosted


    # ── Phase 14: Online Dopamine-Gated Learning ──
    # Neuroscience: Striatal DA signals prediction errors across ALL domains (Costa, Sci Adv 2025)
    # DA prediction errors drive internal model updates (Gershman, Nat Neurosci 2024)

    def _td_learn(self, cur_label: str, tgt_label: str, edge,
                   hop_prediction_error: float, td_lr: float = 0.1):
        """Phase 14.1: Update edge weight using temporal difference learning.
        
        TD_error = V_actual - V_expected
        V_expected = edge.weight * edge.confidence
        V_actual = 1.0 - hop_prediction_error
        Positive TD: edge better than expected → strengthen.
        Negative TD: edge worse than expected → weaken.
        """
        if edge is None:
            return
        V_expected = edge.weight * edge.confidence
        V_actual = 1.0 - hop_prediction_error
        td_error = V_actual - V_expected
        
        # Modulate learning rate by dopamine tone
        dt = getattr(self, '_dopamine_tone', 0.5)
        effective_lr = td_lr * (0.5 + dt)  # 0.06 at low, 0.14 at high
        
        delta = effective_lr * td_error
        edge.weight = np.clip(edge.weight + delta, 0.01, 0.99)
        
        # Update confidence based on absolute TD error
        abs_error = abs(td_error)
        if abs_error < 0.2:
            edge.confidence = min(1.0, edge.confidence + 0.05)
        elif abs_error > 0.5:
            edge.confidence = max(0.05, edge.confidence - 0.1)
        
        # Track TD error history
        self._td_error_history.append(td_error)
        if len(self._td_error_history) > 20:
            self._td_error_history = self._td_error_history[-20:]

    def _update_dopamine_tone(self):
        """Phase 14.2: Update dopamine tone based on mean TD error, novelty, and repetition.
        
        dopamine_tone = 0.5 + 0.3 * mean_TD + 0.2 * novelty - 0.1 * repetition_penalty
        Clamped to [0.1, 0.9].
        """
        if not self._td_error_history:
            return
        mean_TD = np.mean(np.abs(self._td_error_history[-5:]))
        
        # Novelty: fraction of unseen concepts in last response
        if self._last_responses:
            last_words = set(self._last_responses[-1].lower().split())
            novel = sum(1 for w in last_words if w not in self._visited_concepts)
            novelty = novel / max(1, len(last_words))
        else:
            novelty = 0.5
        
        # Repetition penalty
        if len(self._recent_traversals) > 5:
            unique = len(set(self._recent_traversals[-5:]))
            repetition = 1.0 - (unique / 5.0)
        else:
            repetition = 0.0
        
        self._dopamine_tone = np.clip(
            0.5 + 0.3 * mean_TD + 0.2 * novelty - 0.1 * repetition,
            0.1, 0.9
        )

    def _detect_latent_cause_switch(self) -> bool:
        """Phase 14.5: Detect when prediction error is consistently high across edges.
        
        If mean absolute TD > 0.4 for 3+ hops: signal latent cause switch.
        Returns True if switch detected.
        """
        if len(self._td_error_history) < 3:
            return False
        recent = np.abs(self._td_error_history[-3:])
        if float(np.mean(recent)) > 0.4:
            return True
        return False

    def _walk_chain(self, label: str, seen: Set[str], max_hops: int,
                    temperature: float = 0.25,
                    contradiction_penalty: float = 0.6,
                    activation_boost: Optional[Dict[str, float]] = None,
                    subject_proximity: Optional[str] = None,
                    episodic_first: bool = False) -> Optional[str]:
        """Walk a path through the graph from label, temperature-weighted.
        temperature=0 → greedy (always strongest), temperature=1 → near-uniform.
        Each hop adds `connector concept` to the chain. Returns None if no path.

        Applies contradiction_penalty to edges whose target contradicts a
        previously asserted belief about the source concept.
        activation_boost: {target_label: multiplier} from user model preferences.

        When episodic_first=True, only episodic edges are considered for the
        first hop, falling through to all edge types if none exist.

        Records detailed hop info in self._chain_traces when self._trace_enabled."""
        nids = self._concept_keywords.get(label.lower(), [])
        if not nids:
            return None
        chain = [label]
        chain_labels = {label.lower()}  # path-level cycle detection
        cur_nid = nids[0]
        cur_label = label
        hops = 0
        local_hops: List[Tuple[str, str]] = []
        trace = ChainTrace(max_hops=max_hops) if self._trace_enabled else None
        while hops < max_hops:
            candidates = []
            for tid, edge in self.graph.get_outgoing(cur_nid):
                tn = self.graph.nodes.get(tid)
                if tn is None or tn.label is None or tn.label.lower() in seen:
                    continue
                if tn.label.lower() in chain_labels:
                    continue  # cycle detected within this chain
                candidates.append((edge.weight * edge.confidence, tn.label, edge, "out"))
            for src, edge in self.graph.get_incoming(cur_nid):
                sn = self.graph.nodes.get(src)
                if sn is None or sn.label is None or sn.label.lower() in seen:
                    continue
                if sn.label.lower() in chain_labels:
                    continue  # cycle detected within this chain
                candidates.append((edge.weight * edge.confidence, sn.label, edge, "in"))
            if not candidates:
                break
            # Fix 2: Boost dormant edge confidence if endpoints co-occur in user model
            if self._dormant_edges:
                boosted = []
                for sig, tgt_lbl, edge, d in candidates:
                    edge_pair = (cur_nid, edge.target) if d == "out" else (edge.source, cur_nid)
                    if edge_pair in self._dormant_edges:
                        key = (cur_label.lower(), tgt_lbl.lower())
                        visit_count = self.user_model.edge_reactivations.get(key, 0)
                        if visit_count > 0 or edge.confidence > 0.15:
                            # Awaken: enough contextual justification
                            self._dormant_edges.discard(edge_pair)
                            edge.confidence = 0.3
                            sig = edge.weight * 0.3  # recompute signal with awakened confidence
                    boosted.append((sig, tgt_lbl, edge, d))
                candidates = boosted
            if not candidates:
                break
            # Fix 1: Episodic-first mode (recall) — only episodic edges for this hop
            if episodic_first:
                episodic_cands = [(s, l, e, d) for (s, l, e, d) in candidates
                                  if e.relation_type == "episodic"]
                if episodic_cands:
                    candidates = episodic_cands
            if not candidates:
                break
            # Apply contradiction penalty (from _belief_assertions)
            if contradiction_penalty > 0 and self._contradiction_map:
                penalized = []
                for sig, tgt_lbl, edge, d in candidates:
                    if self._is_contradictory(cur_label, tgt_lbl, edge.relation_type):
                        sig *= (1.0 - contradiction_penalty)
                    penalized.append((sig, tgt_lbl, edge, d))
                candidates = penalized
            # Preference boost: continuous sigmoid from edge reactivation frequency
            if self.user_model.edge_reactivations:
                boosted = []
                for sig, tgt_lbl, edge, d in candidates:
                    key = (cur_label.lower(), tgt_lbl.lower())
                    count = self.user_model.edge_reactivations.get(key, 0)
                    boost = 1.0 + (count / (count + 1.0)) * 0.3
                    boosted.append((sig * boost, tgt_lbl, edge, d))
                candidates = boosted
            # Activation boost from follow-up handler (targeted preference bias)
            if activation_boost:
                boosted = []
                for sig, tgt_lbl, edge, d in candidates:
                    boost = activation_boost.get(tgt_lbl.lower(), 1.0)
                    boosted.append((sig * boost, tgt_lbl, edge, d))
                candidates = boosted
            # ── VAD Edge Preference ──
            if getattr(self, 'use_vad', True):
                valence = self.emotion.state.valence
                dominance = self.emotion.state.dominance
                vad_boosted = []
                max_w = max((c[0] for c in candidates), default=0.001)
                for sig, tgt_lbl, edge, d in candidates:
                    adj = sig
                    # Positive mood: prefer stronger existing paths
                    if valence > 0.10:
                        adj *= (1.0 + 0.15 * (edge.weight / max_w))
                    # Negative mood: slightly flatten all weights (exploration)
                    elif valence < -0.10:
                        adj *= 0.9
                    # Low dominance: less assertive path selection
                    if dominance < 0.35:
                        adj *= (0.6 + 0.4 * edge.weight / max_w)
                    vad_boosted.append((adj, tgt_lbl, edge, d))
                candidates = vad_boosted
            # ── Subject Proximity Bonus (Phase 4.2) ──
            # Bias toward concepts that are vector-similar to the original subject
            if subject_proximity is not None:
                prox_nids = self._concept_keywords.get(subject_proximity.lower(), [])
                prox_node = self.graph.get_node(prox_nids[0]) if prox_nids else None
                if prox_node is not None and prox_node.vector is not None:
                    prox_boosted = []
                    for sig, tgt_lbl, edge, d in candidates:
                        tgt_nids_cand = self._concept_keywords.get(tgt_lbl.lower(), [])
                        if not tgt_nids_cand:
                            prox_boosted.append((sig, tgt_lbl, edge, d))
                            continue
                        tgt_node_cand = self.graph.get_node(tgt_nids_cand[0])
                        if tgt_node_cand is not None and tgt_node_cand.vector is not None:
                            cos = float(np.dot(tgt_node_cand.vector, prox_node.vector))
                            prox_boost = 1.0 + 0.15 * max(0.0, cos)
                            prox_boosted.append((sig * prox_boost, tgt_lbl, edge, d))
                        else:
                            prox_boosted.append((sig, tgt_lbl, edge, d))
                    candidates = prox_boosted
            # ── Prefrontal Gating (Phase 9) ──
            # Filter through working memory buffer — boosts on-topic, suppresses drift
            if getattr(self, '_pfc_gating_enabled', True) and self._prefrontal_buffer:
                gate_subject = subject_proximity if subject_proximity else label
                candidates = self._prefrontal_gate_candidates(candidates, gate_subject)
            # ── RLMv2 Confidence Modulation ──
            rlm_data = {}  # tgt_label -> confidence for trace logging
            if getattr(self, 'use_rlm', True):
                rlm_boosted = []
                for sig, tgt_lbl, edge, d in candidates:
                    triple = self._classify_triple(cur_label, tgt_lbl, edge.relation_type)
                    rlm_conf_local = triple['confidence']
                    rlm_data[tgt_lbl.lower()] = rlm_conf_local
                    adj = sig
                    if rlm_conf_local < 0.4:
                        adj *= 0.7
                    elif rlm_conf_local > 0.75:
                        adj *= 1.15
                    if triple['is_analogical']:
                        adj *= 0.85
                    rlm_boosted.append((adj, tgt_lbl, edge, d))
                candidates = rlm_boosted
            # ── BeliefStore Confidence Weighting ──
            if getattr(self, 'use_beliefs', True):
                belief_boosted = []
                for sig, tgt_lbl, edge, d in candidates:
                    belief = self.belief_store.query_belief(cur_label, edge.relation_type)
                    adj = sig
                    if belief:
                        value, bconf, turn = belief
                        if value == tgt_lbl:
                            adj *= (1.0 + 0.2 * bconf)
                        else:
                            adj *= (1.0 - 0.4 * bconf)
                    belief_boosted.append((adj, tgt_lbl, edge, d))
                candidates = belief_boosted
            # Temperature-weighted selection
            if temperature > 0 and len(candidates) > 1:
                sigs = np.array([c[0] for c in candidates])
                # Clamp to avoid numerical issues with near-zero sigs
                sigs = np.clip(sigs, 1e-10, None)
                weights = np.exp(sigs / temperature)
                weights /= weights.sum()
                idx = self.rng.choice(len(candidates), p=weights)
            else:
                idx = max(range(len(candidates)), key=lambda i: candidates[i][0])
            best_sig, best_label, best_edge, direction = candidates[idx]
            # Fix 2: Awaken dormant edge when first traversed
            if self._dormant_edges:
                be_pair = (cur_nid, best_edge.target) if direction == "out" else (best_edge.source, cur_nid)
                if be_pair in self._dormant_edges:
                    self._dormant_edges.discard(be_pair)
                    best_edge.confidence = 0.3  # awakened to normal level
            connector = self._pick_connector(best_edge.relation_type, best_edge.weight,
                                             best_edge.confidence, temperature)
            # Retry same hop if best neighbor IS the connector word
            if best_label.lower() == connector.lower():
                seen.add(best_label.lower())
                continue
            if connector and chain[-1] != connector:
                chain.append(connector)
            chain.append(best_label)
            chain_labels.add(best_label.lower())
            if connector:
                chain_labels.add(connector.lower())
            seen.add(best_label.lower())
            # Record belief assertion BEFORE ChainHop creation (contradiction_detail needed for trace)
            self._belief_assertions.append((cur_label, best_edge.relation_type, best_label))
            contradiction_detail: str = ""
            if getattr(self, 'use_beliefs', True):
                # Check for contradiction BEFORE asserting (detect overwrite)
                c = self.belief_store.detect_contradiction(
                    cur_label, best_edge.relation_type, best_label)
                if c is not None:
                    (old_val, old_conf, old_turn), new_val, _ = c
                    contradiction_detail = (
                        f"{cur_label} -> {old_val} (turn {old_turn}) vs "
                        f"{cur_label} -> {new_val} @ conf {best_edge.weight:.3f}")
                    if self._trace_enabled:
                        print(f"  [trace]     contradiction: {contradiction_detail}")
                self.belief_store.assert_belief(
                    cur_label, best_edge.relation_type, best_label,
                    confidence=best_edge.weight)
                self.belief_store.advance_turn()
            # RLM confidence for the selected edge
            rlm_hop_conf = rlm_data.get(best_label.lower(), 0.0) if rlm_data else 0.0
            if trace is not None:
                trace.hops.append(ChainHop(
                    from_label=cur_label, to_label=best_label,
                    relation_type=best_edge.relation_type,
                    weight=best_edge.weight, confidence=best_edge.confidence,
                    temperature=temperature, candidates=len(candidates),
                    rlm_confidence=rlm_hop_conf,
                    contradiction=contradiction_detail))
            local_hops.append((cur_label, best_label))
            cur_label = best_label
            cur_nid = self._concept_keywords.get(best_label.lower(), [None])[0]
            if cur_nid is None:
                break
            hops += 1
        if trace is not None:
            trace.completed = hops >= max_hops
            self._chain_traces.append(trace)
        if local_hops:
            self._last_hops.append(local_hops)
        if len(chain) <= 1:
            return None
        return " ".join(chain)


    # ─── Phase 7: Multi-Strategy Reasoning ───

    def _bridge_prospecting(self, activated_ids: List[int], subject: str,
                             seen: Set[str], temperature: float) -> Optional[str]:
        """Strategy B: Find a bridge concept connecting multiple activated inputs.
        Reuses _spread_and_collect to find the highest-activation non-seed concept."""
        if len(activated_ids) < 2:
            return None
        assocs = self._spread_and_collect(activated_ids, primary_ids=set(activated_ids))
        if not assocs:
            return None
        bridge = assocs[0][0]
        if bridge.lower() in seen or bridge.lower() == subject.lower():
            if len(assocs) > 1:
                bridge = assocs[1][0]
            else:
                return None
        # Walk from the bridge concept
        return self._walk_chain(bridge, seen, max_hops=2, temperature=temperature,
                                subject_proximity=subject)

    def _analogical_detour(self, subject: str, seen: Set[str],
                            temperature: float) -> Optional[str]:
        """Strategy C: Walk from analogically-linked concepts to the subject."""
        subj_nids = self._concept_keywords.get(subject.lower(), [])
        if not subj_nids:
            return None
        subj_nid = subj_nids[0]
        analogical_targets = []
        for tid, edge in self.graph.get_outgoing(subj_nid):
            if edge.relation_type == "analogical":
                tn = self.graph.get_node(tid)
                if tn and tn.label and tn.label.lower() not in seen:
                    analogical_targets.append(tn.label)
        for src, edge in self.graph.get_incoming(subj_nid):
            if edge.relation_type == "analogical":
                sn = self.graph.get_node(src)
                if sn and sn.label and sn.label.lower() not in seen:
                    analogical_targets.append(sn.label)
        if not analogical_targets:
            return None
        # Walk from the first analogical target
        start = analogical_targets[0]
        seen.add(start.lower())
        return self._walk_chain(start, seen, max_hops=2, temperature=temperature,
                                subject_proximity=subject)

    def _contrastive_flip(self, subject: str, seen: Set[str],
                           temperature: float) -> Optional[str]:
        """Strategy D: Walk from the opposite of the subject (contrastive edges)."""
        antonyms = self._contradiction_map.get(subject.lower(), set())
        if not antonyms:
            return None
        for antonym in antonyms:
            if antonym not in seen:
                seen.add(antonym)
                result = self._walk_chain(antonym, seen, max_hops=2, temperature=temperature)
                if result:
                    return f"{subject} but {result}"
        return None

    def _sub_question_decompose(self, subject: str, query: str,
                                 seen: Set[str], temperature: float) -> Optional[str]:
        """Strategy E: Split multi-word queries into existing graph concepts."""
        words = re.findall(r"[a-zA-Z']{3,}", query.lower())
        existing = [w for w in words
                    if w in self._concept_keywords and w != subject.lower() and w not in seen]
        if not existing:
            return None
        # Pick the most connected existing concept and walk from it
        best_word = existing[0]
        seen.add(best_word)
        return self._walk_chain(best_word, seen, max_hops=2, temperature=temperature,
                                subject_proximity=subject)

    def _compose_uncertainty_response(self, query: str, subject: str,
                                       activated_concepts: List[str],
                                       strategies_tried: List[str]) -> str:
        """Strategy G: Express uncertainty using graph labels only.
        Pattern: '[subject] connect [c1] [c2]. curious learn more.'
        All words must be existing graph labels."""
        c1 = ""
        c2 = ""
        if activated_concepts:
            c1 = activated_concepts[0]
            if len(activated_concepts) > 1:
                c2 = activated_concepts[1]
        if not c1 or c1.lower() == subject.lower():
            # Use the subject's vector neighbor
            neighbor = self._find_vector_neighbor(subject)
            if neighbor and neighbor.lower() != subject.lower():
                c1 = neighbor
        if not c2 or c2.lower() == subject.lower() or c2.lower() == c1.lower():
            other = self._find_vector_neighbor(c1 if c1 else subject)
            if other and other.lower() != subject.lower() and other.lower() != c1.lower():
                c2 = other
        parts = []
        if c1:
            parts.append(f"{subject} connect {c1}")
        if c2:
            parts.append(f"{c1} connect {c2}") if c1 else parts.append(f"{subject} connect {c2}")
        parts.append("curious learn more")
        return " ".join(parts) + "."

    # ─── Phase 8: Sentence Formatting & Prefrontal Coherence ───

    def _format_sentence(self, chain_str: str, subject: str,
                          connector_counts: Dict[str, int],
                          sentence_idx: int) -> str:
        """Format a chain into a sentence that stays on-topic.

        Uses three strategies for sentence variety while keeping subject prominent:
        0: raw chain (first sentence, establishes topic)
        1: starter + subject + chain_tail (re-anchor to subject)
        2: starter + i-perspective + subject + chain_tail (self-reference, 25% chance)
           or starter + subject + chain_tail (75% chance)

        All words are graph-native labels.
        """
        conn = self._starter_from_chain(chain_str, subject, connector_counts)
        parts = chain_str.split()
        concepts = [p for p in parts if p.lower() not in self._CONNECTOR_SET]
        chain_conns = [p for p in parts if p.lower() in self._CONNECTOR_SET]

        if not concepts:
            return chain_str + "."

        # Sentence 0: raw chain as-is
        if sentence_idx <= 0:
            return chain_str + "."

        # Check if subject already appears in chain
        subj_lower = subject.lower()
        has_subject = any(subj_lower == c.lower() for c in concepts)
        if has_subject:
            return f"{conn} {chain_str}."

        # Subject not in chain — likely from alt concept fallback, use chain as-is
        # Don't force the subject into unrelated context
        if not has_subject and sentence_idx > 0:
            return f"{conn} {chain_str}."

        # Build re-anchored sentence: [starter] [subject] [first_connector] [last_concept]
        first_conn = chain_conns[0] if chain_conns else \
            self._pick_connector("semantic", 0.40, 1.0, 0.3)
        last_c = concepts[-1]

        # Sentence 2+: 25% chance of "i think/feel/know" perspective (teen self-reference)
        if sentence_idx >= 2 and self.rng.random() < 0.25:
            perspective = self.rng.choice(["i think", "i feel", "i know"])
            return f"{conn} {perspective} {subject} {first_conn} {last_c}."

        return f"{conn} {subject} {first_conn} {last_c}."

    def _generate_response(self, ctx: CognitiveResponseContext) -> Tuple[str, str]:
        """Walk progressive chains, each anchored to the subject for coherence.

        Phase 8: Prefrontal-guided coherence — every sentence walks from the
        subject (not from the previous sentence's tail), keeping the response
        on-topic. Lower temperatures reduce random drift. Occasional "i think"
        or "i feel" self-reference mimics teenage speech patterns.

        Phase 4: Anchor-based coherence:
        - 4.1: Re-anchor to subject if final concept drifts
        - 4.2: Subject-proximity bonus biases each hop toward the original topic
        - 4.3: Track used concepts across sentences for diversity
        - 4.4: Minimize monotonic "connect" chains
        """
        subject = ctx.subject
        assocs = ctx.associated_concepts
        act_boost = getattr(self, '_activation_boost', None)
        self._last_hops = []

        if not assocs:
            if subject:
                neighbor = self._find_vector_neighbor(subject)
                if neighbor and neighbor.lower() != subject.lower():
                    return (f"{subject} connect {neighbor}.", "associative")
                return (subject + ".", "associative")
            return ("...", "associative")

        # Phase 9c: Per-sentence working memory (teen-like limited PFC)
        # Each sentence has its OWN seen set so later sentences can explore
        # the same neighborhood as earlier ones. This mimics teens' limited
        # working memory — they don't strictly track what they just said.
        sentences = []

        # Phase 9: Initialize prefrontal buffer with subject (seeds working memory)
        if getattr(self, '_pfc_gating_enabled', True):
            self._prefrontal_buffer = [subject.lower()]

        # Phase 8: Walk ALL chains from the subject (prefrontal anchoring)
        # Each sentence independently walks from subject so there's no topic drift.
        # Lower temperatures keep walks focused, subject_proximity boosts on-topic candidates.
        # Occasional "i think/feel/know" self-reference mimics teenage speech patterns.
        temps = [self._get_temperature(0), self._get_temperature(1), self._get_temperature(2)]
        connector_counts: Dict[str, int] = {}

        # Hot-cognition: high arousal → shorter sentences (teens speak less coherently when excited)
        # Sentence 1 uses 1 hop to establish topic, sentences 2-3 use 2 hops for exploration.
        # This prevents exhausting the graph in sentence 1.
        arousal = self.emotion.state.arousal if getattr(self, 'use_vad', True) else 0.3
        if arousal > 0.6:
            s_hops = [1, 1, 1]
        elif arousal > 0.4:
            s_hops = [1, 2, 2]
        else:
            s_hops = [2, 2, 2]

        # Fix 1: Recall mode — re-traverse episodic memory first, then extend
        if ctx.recall_mode:
            # Sentence 1: walk episodic edges from subject (re-trace conversation)
            s1_seen = {subject.lower()}
            chain1 = self._walk_chain(subject, s1_seen, max_hops=s_hops[0], temperature=temps[0],
                                      activation_boost=act_boost,
                                      subject_proximity=subject, episodic_first=True)
            if not chain1:
                chain1 = self._walk_chain(subject, s1_seen, max_hops=s_hops[0], temperature=temps[0],
                                           activation_boost=act_boost,
                                           subject_proximity=subject)
            if chain1:
                sentences.append(self._format_sentence(chain1, subject, connector_counts, 0))
                chain1_concepts = [p for p in chain1.split() if p.lower() not in self._CONNECTOR_SET]
                if getattr(self, '_pfc_gating_enabled', True):
                    self._prefrontal_maintain_buffer(subject, chain1_concepts)
                for word in chain1.split():
                    if word.lower() in self._CONNECTOR_SET:
                        connector_counts[word.lower()] = connector_counts.get(word.lower(), 0) + 1

            # Sentence 2: extend semantically from subject (explore beyond episodic)
            s2_seen = {subject.lower()}
            chain2 = self._walk_chain(subject, s2_seen, max_hops=s_hops[1], temperature=temps[1],
                                      activation_boost=act_boost,
                                      subject_proximity=subject)
            if chain2:
                sentences.append(self._format_sentence(chain2, subject, connector_counts, 1))
                chain2_concepts = [p for p in chain2.split() if p.lower() not in self._CONNECTOR_SET]
                if getattr(self, '_pfc_gating_enabled', True):
                    self._prefrontal_maintain_buffer(subject, chain2_concepts)
                for word in chain2.split():
                    if word.lower() in self._CONNECTOR_SET:
                        connector_counts[word.lower()] = connector_counts.get(word.lower(), 0) + 1

            # Sentence 3: summary perspective from subject
            s3_seen = {subject.lower()}
            chain3 = self._walk_chain(subject, s3_seen, max_hops=s_hops[2], temperature=temps[2],
                                      activation_boost=act_boost,
                                      subject_proximity=subject)
            if chain3:
                sentences.append(self._format_sentence(chain3, subject, connector_counts, 2))
                chain3_concepts = [p for p in chain3.split() if p.lower() not in self._CONNECTOR_SET]
                if getattr(self, '_pfc_gating_enabled', True):
                    self._prefrontal_maintain_buffer(subject, chain3_concepts)
                for word in chain3.split():
                    if word.lower() in self._CONNECTOR_SET:
                        connector_counts[word.lower()] = connector_counts.get(word.lower(), 0) + 1
            if not sentences:
                neighbor = self._find_vector_neighbor(subject)
                if neighbor and neighbor.lower() != subject.lower():
                    return (f"{subject} connect {neighbor}.", "associative")
                return (subject + ".", "associative")
        else:
            # Sentence 1: walk from subject (own seen set)
            s1_seen = {subject.lower()}
            chain1 = self._walk_chain(subject, s1_seen, max_hops=s_hops[0], temperature=temps[0],
                                      activation_boost=act_boost,
                                      subject_proximity=subject)
            if not chain1:
                neighbor = self._find_vector_neighbor(subject)
                if neighbor and neighbor.lower() != subject.lower():
                    return (f"{subject} connect {neighbor}.", "associative")
                return (subject + ".", "associative")
            sentences.append(self._format_sentence(chain1, subject, connector_counts, 0))
            chain1_concepts = [p for p in chain1.split() if p.lower() not in self._CONNECTOR_SET]
            if getattr(self, '_pfc_gating_enabled', True):
                self._prefrontal_maintain_buffer(subject, chain1_concepts)
            for word in chain1.split():
                if word.lower() in self._CONNECTOR_SET:
                    connector_counts[word.lower()] = connector_counts.get(word.lower(), 0) + 1

            # Sentence 2: walk from subject again with own seen set
            s2_seen = {subject.lower()}
            chain2 = self._walk_chain(subject, s2_seen, max_hops=s_hops[1], temperature=temps[1],
                                      activation_boost=act_boost,
                                      subject_proximity=subject)
            if not chain2:
                chain2 = self._walk_chain(subject, s2_seen, max_hops=1, temperature=temps[1],
                                           activation_boost=act_boost, subject_proximity=subject)
            if chain2:
                sentences.append(self._format_sentence(chain2, subject, connector_counts, 1))
                chain2_concepts = [p for p in chain2.split() if p.lower() not in self._CONNECTOR_SET]
                if getattr(self, '_pfc_gating_enabled', True):
                    self._prefrontal_maintain_buffer(subject, chain2_concepts)
                for word in chain2.split():
                    if word.lower() in self._CONNECTOR_SET:
                        connector_counts[word.lower()] = connector_counts.get(word.lower(), 0) + 1

            # Sentence 3: walk from subject again with own seen set
            s3_seen = {subject.lower()}
            chain3 = self._walk_chain(subject, s3_seen, max_hops=s_hops[2], temperature=temps[2],
                                      activation_boost=act_boost,
                                      subject_proximity=subject)
            if chain3:
                sentences.append(self._format_sentence(chain3, subject, connector_counts, 2))
                chain3_concepts = [p for p in chain3.split() if p.lower() not in self._CONNECTOR_SET]
                if getattr(self, '_pfc_gating_enabled', True):
                    self._prefrontal_maintain_buffer(subject, chain3_concepts)
                for word in chain3.split():
                    if word.lower() in self._CONNECTOR_SET:
                        connector_counts[word.lower()] = connector_counts.get(word.lower(), 0) + 1

        # Phase 7.2: Check if response is weak (no substantive path found)
        # Store which strategy was used for impossible query tracking
        if len(sentences) < 2 or all(len(s.strip(".").split()) < 3 for s in sentences):
            # Build fallback seen set from all sentences so far
            fallback_seen = {subject.lower()}
            for s in sentences:
                for w in s.strip(".").split():
                    wl = w.lower()
                    if wl not in self._CONNECTOR_SET:
                        fallback_seen.add(wl)
            seen = fallback_seen
            strategies_tried = ["A_direct_chain"]
            strategy_result = None

            # Strategy B: Bridge Prospecting
            if strategy_result is None:
                bridge_result = self._bridge_prospecting(
                    getattr(self, '_last_activated_ids', []), subject, seen, temps[1])
                if bridge_result:
                    conn = self._starter_from_chain(bridge_result, subject, connector_counts)
                    strategy_result = f"{conn} {bridge_result}."
                    strategies_tried.append("B_bridge")
                    for w in bridge_result.split():
                        if w.lower() in self._CONNECTOR_SET:
                            connector_counts[w.lower()] = connector_counts.get(w.lower(), 0) + 1

            # Strategy C: Analogical Detour
            if strategy_result is None:
                analog_result = self._analogical_detour(subject, seen, temps[1])
                if analog_result:
                    conn = self._starter_from_chain(analog_result, subject, connector_counts)
                    strategy_result = f"{conn} {analog_result}."
                    strategies_tried.append("C_analogical")
                    for w in analog_result.split():
                        if w.lower() in self._CONNECTOR_SET:
                            connector_counts[w.lower()] = connector_counts.get(w.lower(), 0) + 1

            # Strategy D: Contrastive Flip
            if strategy_result is None:
                contrast_result = self._contrastive_flip(subject, seen, temps[1])
                if contrast_result:
                    strategy_result = f"but {contrast_result}."
                    strategies_tried.append("D_contrastive")

            # Strategy E: Sub-Question Decomposition
            if strategy_result is None:
                decomp_result = self._sub_question_decompose(subject, ctx.raw_input, seen, temps[1])
                if decomp_result:
                    conn = self._starter_from_chain(decomp_result, subject, connector_counts)
                    strategy_result = f"{conn} {decomp_result}."
                    strategies_tried.append("E_decompose")

            # Strategy F: Web Research Mode (if network available)
            if strategy_result is None and getattr(self, '_network_available', True) is not False:
                search_result = getattr(self, 'learn_from_web', None)
                if search_result and subject:
                    try:
                        self.learn_from_web(subject)
                        strategies_tried.append("F_web_research")
                        # Retry A-E with expanded graph
                        retry = self._walk_chain(subject, seen, max_hops=2,
                                                  temperature=temps[1],
                                                  subject_proximity=subject)
                        if retry:
                            conn = self._starter_from_chain(retry, subject, connector_counts)
                            strategy_result = f"{conn} {retry}."
                    except Exception:
                        pass

            # Strategy G: Honest Uncertainty
            if strategy_result is None:
                activated_labels = [a[0] for a in assocs[:5]]
                uncertainty = self._compose_uncertainty_response(
                    ctx.raw_input, subject, activated_labels, strategies_tried)
                strategy_result = uncertainty
                strategies_tried.append("G_uncertainty")

            if strategy_result:
                sentences.append(strategy_result)
                self._last_strategy_used = strategies_tried[-1] if strategies_tried else "A"
                if self._trace_enabled:
                    print(f"  [trace]   strategy: {', '.join(strategies_tried)}")
            else:
                self._last_strategy_used = "G_uncertainty"

        # Phase 7: Record impossible query if confidence was low
        if len(sentences) < 2:
            activated_labels = [a[0] for a in assocs[:5]]
            failed = FailedQuery(
                query=ctx.raw_input, subject=subject,
                activated_concepts=activated_labels,
                strategies_tried=getattr(self, '_last_strategy_used', 'A'),
                best_guess_response=" ".join(sentences) if sentences else subject,
                turn=self.turn_count,
                free_energy_at_time=self._free_energy,
            )
            self._impossible_queries.append(failed)
            if len(self._impossible_queries) > 50:
                self._impossible_queries = self._impossible_queries[-50:]

        # Phase 4.1: Force chain re-anchoring for each sentence
        # Check if final concept of each sentence connects back to subject
        # If not, try to add a re-anchor hop or shorten the sentence
        re_anchored = []
        for sent in sentences:
            words = sent.strip(".").split()
            if len(words) >= 3:
                last_concept = words[-1]
                # Check if last concept has an edge connecting back to subject
                lc_lower = last_concept.lower()
                subj_lower = subject.lower()
                if lc_lower != subj_lower and lc_lower not in self._CONNECTOR_SET:
                    has_path = False
                    lc_nids = self._concept_keywords.get(lc_lower, [])
                    subj_nids = self._concept_keywords.get(subj_lower, [])
                    if lc_nids and subj_nids:
                        # Check direct edge
                        for ln in lc_nids:
                            for sn in subj_nids:
                                if self.graph.get_edge(ln, sn) or self.graph.get_edge(sn, ln):
                                    has_path = True
                                    break
                            if has_path:
                                break
                    if not has_path:
                        # Try vector similarity — if threshold met, add re-anchor hop
                        lc_node = self.graph.get_node(lc_nids[0]) if lc_nids else None
                        subj_node = self.graph.get_node(subj_nids[0]) if subj_nids else None
                        if lc_node and lc_node.vector is not None and subj_node and subj_node.vector is not None:
                            cos = float(np.dot(lc_node.vector, subj_node.vector))
                            if cos > 0.3:
                                # Add re-anchor: subject is reachable via proximity
                                re_anchored.append(sent)
                            else:
                                # No path back — end sentence early (drop last concept + connector)
                                trimmed = " ".join(words[:-2]) if len(words) >= 3 else " ".join(words)
                                re_anchored.append(trimmed + ".")
                        else:
                            # Can't check — keep as-is
                            re_anchored.append(sent)
                    else:
                        re_anchored.append(sent)
                else:
                    re_anchored.append(sent)
            else:
                re_anchored.append(sent)
        sentences = re_anchored

        # Phase 4.3: Ensure sentence diversity — check that sentence 3 explores
        # a different facet than sentence 1 (uses the temperature escalation)
        # Already handled by increasing temperature from 0.15 to 0.40

        self._log_assertions(sentences, subject)

        # Observe all chain hops in the user model
        for hops in self._last_hops:
            self.user_model.observe_chain(hops, is_user_query=False)
        # Phase 3.4: Save snapshot before clearing for response context
        self._last_chain_hops = list(self._last_hops)
        self._activation_boost = None
        self._last_hops = []
        return (" ".join(sentences), "associative")


    def _update_state(self, ctx: CognitiveResponseContext):
        self._free_energy = max(0.0, 0.3 + 0.2 * (1.0 - ctx.identity_strength) - 0.08 * len(ctx.associated_concepts))
        self.meaning.compute_meaning(
            episode=self.turn_count,
            pre_dissonance=self._free_energy + 0.05,
            post_dissonance=self._free_energy,
            pre_identity=ctx.identity_strength - 0.02,
            post_identity=ctx.identity_strength,
            predictive_gain=0.3 if ctx.associated_concepts else 0.1,
            effort=0.2,
        )
        correct = len(ctx.associated_concepts) > 0
        new_s = self.identity.compute_update(
            resolution_delta=abs(self._free_energy - 0.5) * 0.1,
            resolution_success=correct,
            regulated_identity_delta=0.03 if correct else -0.01,
            current_dissonance=self._free_energy,
            resolution_streak=sum(1 for r in self._last_responses if len(r) > 20),
            correctness=correct,
        )
        self.identity.apply_update(new_s)
        # Clamp identity
        if self.identity.state.strength > 1.0:
            self.identity.state.strength = 1.0
        if self.identity.state.strength < 0.0:
            self.identity.state.strength = 0.0

        self.gw.submit_bid(source="dialogue",
            payload={"subject": ctx.subject, "turn": self.turn_count},
            urgency=0.3 + 0.15 * min(len(ctx.associated_concepts), 4) / 4.0,
            valence=self.emotion.state.valence, episode=self.turn_count)
        self.gw.compete()

    # ── Sleep Consolidation ──

    
    # ── Phase 15: Complementary Learning Systems ──
    # Neuroscience: Hippocampus (fast, sparse) + Neocortex (slow, overlapping) (McClelland 1995)
    # Consolidation only when it aids generalization (Nat Neurosci 2023)

    def _add_episodic_edge(self, src: int, tgt: int, weight: float = 0.3, confidence: float = 0.5):
        """Phase 15.1: Add a fast-learning, high-decay episodic edge."""
        if not hasattr(self, '_episodic_edges'):
            self._episodic_edges = {}
        key = (src, tgt)
        if key not in self._episodic_edges:
            edge = ConceptEdge(src, tgt, weight=weight, confidence=confidence, relation_type="episodic")
            self._episodic_edges[key] = edge

    def _add_semantic_edge(self, src: int, tgt: int, weight: float = 0.4, confidence: float = 0.3):
        """Phase 15.1: Add a slow-learning, stable semantic edge."""
        if not hasattr(self, '_semantic_edges'):
            self._semantic_edges = {}
        key = (src, tgt)
        if key not in self._semantic_edges:
            edge = ConceptEdge(src, tgt, weight=weight, confidence=confidence, relation_type="semantic")
            self._semantic_edges[key] = edge

    def _decay_episodic_edges(self):
        """Phase 15.2: Decay episodic edges — 10% per turn, remove below 0.05."""
        if not hasattr(self, '_episodic_edges'):
            return
        to_remove = []
        for (src, tgt), edge in self._episodic_edges.items():
            edge.weight *= 0.90
            if edge.weight < 0.05:
                to_remove.append((src, tgt))
        for key in to_remove:
            del self._episodic_edges[key]

    def _consolidate_to_semantic(self):
        """Phase 15.5: Transfer frequently-used episodic edges to semantic store.
        
        Edges traversed 2+ times get consolidated. If semantic equivalent exists,
        weighted average merge. Episodic weight halved after consolidation.
        Only consolidates if it aids generalization (Phase 15.6).
        """
        if not hasattr(self, '_episodic_edges') or not hasattr(self, '_semantic_edges'):
            return 0
        
        # Count traversals from edge_reactivations
        consolidated = 0
        for (src, tgt), edge in list(self._episodic_edges.items()):
            # Check if traversed 2+ times
            src_node = self.graph.get_node(src)
            tgt_node = self.graph.get_node(tgt)
            if src_node is None or tgt_node is None:
                continue
            key = (src_node.label.lower(), tgt_node.label.lower())
            count = self.user_model.edge_reactivations.get(key, 0)
            
            if count >= 2 and self._should_consolidate(src, tgt, edge.weight):
                sem_key = (src, tgt)
                if sem_key in self._semantic_edges:
                    sem_edge = self._semantic_edges[sem_key]
                    sem_edge.weight = (sem_edge.weight + edge.weight * count) / (1 + count)
                else:
                    self._add_semantic_edge(src, tgt, weight=edge.weight * 0.8, confidence=0.3)
                edge.weight *= 0.5
                consolidated += 1
        return consolidated

    def _should_consolidate(self, src: int, tgt: int, epi_weight: float) -> bool:
        """Phase 15.6: Only consolidate if it aids generalization.
        
        If existing semantic edges from src predict a similar weight, consolidate.
        If very different, don't consolidate (exception, not pattern).
        """
        if not hasattr(self, '_semantic_edges') or not self._semantic_edges:
            return True
        weights = [e.weight for (s, t), e in self._semantic_edges.items() if s == src]
        if not weights:
            return True
        pred_weight = np.mean(weights)
        delta = abs(epi_weight - pred_weight)
        return delta < 0.3


    # ── Phase 16: Thalamocortical Gating & Forward Model ──
    # Neuroscience: High-order thalamic nuclei gate conscious perception (Science 2024)
    # Cerebellar forward model predicts upcoming sensory consequences (Neuron 2026)

    def _thalamic_gate(self, candidates: List[Tuple]) -> List[Tuple]:
        """Phase 16.1: Gate candidates — only the most salient pass through.
        
        Salience = signal_ratio * 0.4 + novelty * 0.3 + schema_relevance * 0.3
        Gate threshold: salience > 0.2
        """
        if not candidates:
            return []
        max_sig = max(c[0] for c in candidates) or 0.001
        gated = []
        for sig, tgt_lbl, edge, d in candidates:
            novelty = 1.0 if tgt_lbl.lower() not in self._visited_concepts else 0.3
            schema_rel = 1.2 if tgt_lbl.lower() in self._sentence_schema else 0.8
            salience = (sig / max_sig) * 0.4 + novelty * 0.3 + schema_rel * 0.3
            if salience > 0.2:
                gated.append((sig, tgt_lbl, edge, d))
        return gated

    def _cerebellar_predictor(self, cur_label: str) -> Dict[str, float]:
        """Phase 16.3: Predict next concepts from past traversal patterns.
        
        Uses n-gram model over _last_chain_hops.
        Returns dict of {predicted_label: probability}.
        """
        predictions = {}
        for hops_list in self._last_chain_hops[-10:]:
            for i, (f, t) in enumerate(hops_list):
                if f.lower() == cur_label.lower():
                    predictions[t.lower()] = predictions.get(t.lower(), 0) + 1
        if predictions:
            total = sum(predictions.values())
            return {k: v / total for k, v in predictions.items()}
        return {}

    def _get_cerebellar_depth(self, label: str) -> float:
        """Phase 16.5: Get expected number of hops from past patterns."""
        return self._cerebellar_depth.get(label.lower(), 0.0)

    def _update_cerebellar_ngram(self, hops: List[Tuple[str, str]]):
        """Update cerebellar n-gram model with a completed chain."""
        for i, (f, t) in enumerate(hops):
            fl, tl = f.lower(), t.lower()
            if fl not in self._cerebellar_ngram:
                self._cerebellar_ngram[fl] = {}
            self._cerebellar_ngram[fl][tl] = self._cerebellar_ngram[fl].get(tl, 0) + 1
            
            # Also learn depth
            remaining = len(hops) - i - 1
            current_depth = self._cerebellar_depth.get(fl, 0.0)
            self._cerebellar_depth[fl] = current_depth * 0.7 + remaining * 0.3


    # ── Phase 17: Meta-Learning ──
    # Neuroscience: Brain tracks own uncertainty, calibrates confidence (PMC 2024)
    # Epistemic value = expected information gain (Free Energy Principle)

    def _update_concept_confidence(self, label: str, prediction_error: float):
        """Phase 17.1: Update per-concept confidence.
        
        Initial: 0.3 for seed, 0.1 for learned.
        Increases by 0.05 on use, decreases by 0.1 on high PE.
        """
        cl = label.lower()
        current = self._concept_confidence.get(cl, 0.3 if cl in self._concept_labels else 0.1)
        if prediction_error < 0.2:
            current += 0.05
        elif prediction_error > 0.5:
            current -= 0.1
        self._concept_confidence[cl] = np.clip(current, 0.05, 0.99)

    def _compute_epistemic_value(self, label: str) -> float:
        """Phase 17.2: Compute epistemic value of exploring a concept.
        
        epistemic_value = uncertainty * expected_information_gain
        """
        conf = self._concept_confidence.get(label.lower(), 0.1)
        uncertainty = 1.0 - conf
        pe = self._mean_prediction_error
        info_gain = 1.0 - pe  # High PE = high potential information gain
        return uncertainty * info_gain

    def _calibrated_confidence(self, label: str) -> float:
        """Phase 17.3: Get calibrated confidence for a concept.
        
        Adjusts raw confidence by calibration error (subtract overconfidence).
        """
        raw = self._concept_confidence.get(label.lower(), 0.1)
        calibration_penalty = getattr(self, '_calibration_error', 0.0) * 0.2
        return max(0.05, raw - calibration_penalty)

    def _metacognitive_review(self):
        """Phase 17.4: Run metacognitive review every 5 turns.
        
        1. Check response diversity (same connector used 3+ times)
        2. Check concept coverage (same 5 concepts keep appearing)
        3. Check prediction error trend
        """
        if self.turn_count < 5:
            return
        
        issues = []
        
        # 1. Connector diversity
        if hasattr(self, '_response_context') and len(self._response_context) >= 3:
            last_3 = ''.join(c['response'] for c in self._response_context[-3:])
            for connector in ['connect', 'cause', 'but', 'like', 'then']:
                if last_3.count(connector) >= 3:
                    issues.append(f"overused connector '{connector}'")
        
        # 2. Concept coverage
        if hasattr(self, '_last_responses') and len(self._last_responses) >= 3:
            all_words = []
            for r in self._last_responses[-3:]:
                all_words.extend(r.lower().split())
            word_counts = {}
            for w in all_words:
                word_counts[w] = word_counts.get(w, 0) + 1
            # Check if same concept appears 5+ times
            for w, c in word_counts.items():
                if c >= 5 and w not in {'connect', 'but', 'like', 'cause', 'then', 'and', 'the'}:
                    issues.append(f"repetitive concept '{w}' ({c}x)")
        
        # 3. Prediction error trend
        if len(self._td_error_history) >= 5:
            recent_pe = np.mean(np.abs(self._td_error_history[-3:]))
            older_pe = np.mean(np.abs(self._td_error_history[:-3])) if len(self._td_error_history) > 3 else 0.3
            if recent_pe > older_pe * 1.2:
                issues.append(f"increasing PE trend (recent={recent_pe:.2f}, older={older_pe:.2f})")
        
        # Log findings
        if issues and hasattr(self, '_trace_enabled') and self._trace_enabled:
            print(f"  [meta]   review: {', '.join(issues)}")

    def _sleep_consolidate(self):
        """Run a mini sleep cycle to strengthen useful patterns and weaken noise.

        Teenagers' brains consolidate learning during sleep! This mimics that
        by replaying recent topics through the graph and applying Hebbian
        strengthening to co-activated concepts.
        """
        if len(self.graph.edges) < 3:
            return

        # Phase 9b: Prediction-error-driven sleep consolidation
        # Instead of blind +0.02 increments, use prediction error to guide updates.
        # Edges that accurately predict vector similarity get strengthened.
        # Edges with high prediction error get corrected (not just weakened).
        for topic in self._topic_list[-5:]:
            tids = self._concept_keywords.get(topic.lower(), [])
            if len(tids) < 2:
                continue
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    edge = self.graph.get_edge(tids[i], tids[j])
                    if edge:
                        src_node = self.graph.get_node(tids[i])
                        tgt_node = self.graph.get_node(tids[j])
                        if (src_node and src_node.vector is not None
                                and tgt_node and tgt_node.vector is not None):
                            error = self._update_edge_from_error(
                                edge, src_node.vector, tgt_node.vector, learning_rate=0.08)

        # Phase 2: Synaptic pruning (neuroscience-inspired: prune weak, unused connections)
        edges_to_prune = []
        for (src, tgt), edge in self.graph.edges.items():
            if edge.weight < 0.08 and edge.confidence < 0.10:
                edges_to_prune.append((src, tgt))
        for src, tgt in edges_to_prune:
            self.graph.remove_edge(src, tgt)

        # Phase 3: Identity consolidation
        self.identity.state.strength = min(1.0, self.identity.state.strength + 0.02)

        # Phase 4: Meaning consolidation
        self.meaning.compute_meaning(
            episode=self.turn_count,
            pre_dissonance=self._free_energy + 0.02,
            post_dissonance=self._free_energy,
            pre_identity=self.identity.state.strength - 0.01,
            post_identity=self.identity.state.strength,
            predictive_gain=0.2,
            effort=0.1,
        )

        # Phase 5: Emotion regulation (calm down after sleep)
        self.emotion.update(
            stimulus_valence=0.0,
            stimulus_arousal=-0.1,
            stimulus_dominance=0.05,
            uncertainty=0.0,
        )

        self.sleep_engine.accumulate_pressure(-0.15)  # Reduce sleep pressure
        self.sleep_cycles_completed += 1

        # Phase 7.6: Sleep-replay impossible queries
        replayed = 0
        for iq in self._impossible_queries:
            if iq.resolved:
                continue
            subj_nids = self._concept_keywords.get(iq.subject.lower(), [])
            if subj_nids:
                subj_node = self.graph.get_node(subj_nids[0])
                if subj_node and subj_node.vector is not None:
                    # Try to auto-wire to all existing concepts with high similarity
                    new_edges = 0
                    for other_nid, other_node in self.graph.nodes.items():
                        if other_nid == subj_nids[0]:
                            continue
                        if other_node.vector is not None and other_node.label:
                            sim = float(np.dot(subj_node.vector, other_node.vector))
                            if sim > 0.5 and self.graph.get_edge(subj_nids[0], other_nid) is None:
                                weight = min(0.6, sim * 0.6)
                                ne = self.graph.add_edge(subj_nids[0], other_nid, weight=weight,
                                                     relation_type="semantic")
                                ne.confidence = 0.001  # dormant
                                self._dormant_edges.add((subj_nids[0], other_nid))
                                # Run prediction error correction for accuracy
                                if other_node.vector is not None:
                                    new_edge = self.graph.get_edge(subj_nids[0], other_nid)
                                    if new_edge:
                                        self._update_edge_from_error(
                                            new_edge, subj_node.vector, other_node.vector,
                                            learning_rate=0.1)
                                new_edges += 1
                    if new_edges > 0:
                        iq.resolved = True
                        replayed += 1
        if replayed > 0 and getattr(self, '_trace_enabled', False):
            print(f"  [trace]   sleep replay: resolved {replayed} impossible queries")

        # Fix 3: Belief reconciliation — resolve contradictions by recency × confidence
        if getattr(self, 'use_beliefs', True) and self.belief_store.beliefs:
            before = len(self.belief_store.contradictions)
            resolved = self.belief_store.reconcile()
            after = len(self.belief_store.contradictions)
            if after > before and getattr(self, '_trace_enabled', False):
                print(f"  [trace]   sleep belief: reconciled {len(resolved)} contradictions")

    def print_traces(self, label: str = ""):
        """Print all chain walk traces from the last response."""
        if not self._chain_traces:
            return
        print(f"  [trace] {label}: {len(self._chain_traces)} chains")
        for ci, t in enumerate(self._chain_traces):
            print(f"  [trace]   chain {ci}: {t.max_hops} max, {'done' if t.completed else 'short'}")
            for i, h in enumerate(t.hops):
                dir_sym = " -> " if h.relation_type != "episodic" else " ~~ "
                extra = ""
                if h.rlm_confidence > 0:
                    extra += f" [RLM: {h.rlm_confidence:.2f}]"
                if h.contradiction:
                    extra += f" [CON: {h.contradiction}]"
                print(f"  [trace]     hop {i}: {h.from_label}{dir_sym}{h.to_label}  "
                      f"[{h.relation_type}] w={h.weight:.3f} c={h.confidence:.3f} "
                      f"t={h.temperature:.2f} ({h.candidates} cand){extra}")
        # Phase 7: Print impossible query count if any
        if self._impossible_queries:
            unresolved = sum(1 for iq in engine._impossible_queries if not iq.resolved)
            print(f"  [trace]   impossible queries: {len(engine._impossible_queries)} total, {unresolved} unresolved")
        # Print user model state
        if self.user_model.edge_reactivations:
            print(f"  [trace]   user_model: {len(self.user_model.edge_reactivations)} edge visits")
            prefs = self.user_model.inferred_preferences(threshold=1)
            if prefs:
                for (frm, to), cnt in sorted(prefs.items(), key=lambda x: -x[1])[:5]:
                    print(f"  [trace]     pref: {frm} -> {to} (visit={cnt})")
        # Print belief store state
        if getattr(self, 'use_beliefs', False) and hasattr(self, 'belief_store'):
            bs = self.belief_store
            if bs.beliefs:
                print(f"  [trace]   belief_store: {len(bs.beliefs)} beliefs, "
                      f"{len(bs.contradictions)} contradictions")
                for (subj, pred), (val, conf, turn) in list(bs.beliefs.items())[:3]:
                    print(f"  [trace]     belief: {subj} . {pred} = {val} @ {conf:.2f} (turn {turn})")
        # Print VAD state
        print(f"  [trace]   vad: v={self.emotion.state.valence:.2f} "
              f"a={self.emotion.state.arousal:.2f} d={self.emotion.state.dominance:.2f}")
        if self._prefrontal_buffer:
            print(f"  [trace]   pfc_buffer: {self._prefrontal_buffer[:5]}")
        self._chain_traces.clear()

    def save(self) -> str:
        """Save full cognitive state to disk. Returns path to save file."""
        state = {
            'graph': self.graph,
            'concept_keywords': self._concept_keywords,
            'turn_count': self.turn_count,
            'topic_list': self._topic_list,
            'topic_store': self._topic_store,
            'response_context': self._response_context,
            'last_responses': self._last_responses,
            'last_strategy': self._last_strategy,
            'free_energy': self._free_energy,
            'learning_count': self._learning_count,
            'identity_state': self.identity.state,
            'identity_momentum': self.identity.last_delta,
            'vad_valence': self.emotion.state.valence,
            'vad_arousal': self.emotion.state.arousal,
            'vad_dominance': self.emotion.state.dominance,
            'meaning_accumulated': self.meaning.accumulated_meaning,
            'dim': self.dim,
            'rng_state': self.rng.get_state(),
            # Teen additions
            'sleep_pressure': self._sleep_pressure,
            'last_sleep_episode': self._last_sleep_episode,
            'sleep_cycles_completed': self.sleep_cycles_completed,
            'concept_vad': self._concept_vad,
            'meta_mode': self.meta_cog.current_mode.value,
            'contradiction_map': self._contradiction_map,
            'user_model': self.user_model,
            'use_vad': getattr(self, 'use_vad', True),
            'use_rlm': getattr(self, 'use_rlm', True),
            'use_beliefs': getattr(self, 'use_beliefs', True),
            'belief_store_state': getattr(self, 'belief_store', BeliefStore()).get_state(),
            'prefrontal_buffer': self._prefrontal_buffer,
            'mean_prediction_error': self._mean_prediction_error,
            'prediction_error_count': self._prediction_error_count,
        }
        try:
            # Phase 6.1: Checkpoint rotation — save every 25 turns
            if self.turn_count > 0 and self.turn_count % 25 == 0:
                checkpoint_path = self._save_path.replace('.pkl', f'_{self.turn_count}.pkl')
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(state, f)
                # Keep last 3 checkpoints, remove older ones
                import glob
                checkpoints = sorted(glob.glob(self._save_path.replace('.pkl', '_[0-9]*.pkl')))
                for old_cp in checkpoints[:-3]:
                    try:
                        os.remove(old_cp)
                    except OSError:
                        pass
        except Exception:
            pass
        try:
            with open(self._save_path, 'wb') as f:
                pickle.dump(state, f)
            size_kb = os.path.getsize(self._save_path) / 1024
            return f"saved {size_kb:.0f}KB to {os.path.basename(self._save_path)}"
        except Exception as e:
            return f"save failed: {e}"
    def _load(self) -> bool:
        """Load cognitive state from disk. Returns True if successful."""
        try:
            with open(self._save_path, 'rb') as f:
                state = pickle.load(f)

            # Restore graph
            self.graph = state['graph']
            self._concept_keywords = state['concept_keywords']
            self.turn_count = state['turn_count']
            self._topic_list = state.get('topic_list', [])
            self._topic_store = state.get("topic_store", {})
            self._response_context = state.get("response_context", [])
            self._last_responses = state['last_responses']
            self._last_strategy = state['last_strategy']
            self._free_energy = state['free_energy']
            self._learning_count = state['learning_count']

            # Restore identity
            self.identity.state = state['identity_state']
            self.identity.last_delta = state['identity_momentum']

            # Restore emotion VAD
            self.emotion.state.valence = state['vad_valence']
            self.emotion.state.arousal = state['vad_arousal']
            self.emotion.state.dominance = state['vad_dominance']

            # Restore meaning
            self.meaning.accumulated_meaning = state['meaning_accumulated']

            # Restore RNG
            self.rng.set_state(state['rng_state'])

            # Restore teen state (optional — may not exist in old saves)
            self._sleep_pressure = state.get('sleep_pressure', 0.0)
            self._last_sleep_episode = state.get('last_sleep_episode', 0)
            self.sleep_cycles_completed = state.get('sleep_cycles_completed', 0)
            self._concept_vad = state.get('concept_vad', {})
            meta_mode_str = state.get('meta_mode', 'exploratory')
            try:
                self.meta_cog.current_mode = EpistemicMode(meta_mode_str)
            except ValueError:
                self.meta_cog.current_mode = EpistemicMode.EXPLORATORY

            # Ensure parent_graph is set on all edges (pickle might not preserve this)
            for edge in self.graph.edges.values():
                edge.parent_graph = self.graph

            # Restore contradiction map (may not exist in old saves)
            self._contradiction_map = state.get('contradiction_map', {})
            # Restore user model
            self.user_model = state.get('user_model', UserModel())
            # Restore belief store
            bs_state = state.get('belief_store_state', None)
            if bs_state:
                self.belief_store.set_state(bs_state)

            # Restore prefrontal buffer
            self._prefrontal_buffer = state.get('prefrontal_buffer', [])
            # Restore prediction error state
            self._mean_prediction_error = state.get('mean_prediction_error', 0.0)
            self._prediction_error_count = state.get('prediction_error_count', 0)

            return True
        except Exception as e:
            print(f"  [Load error] {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Pure natural language chat, no commands
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RAVANA - teenage mind on the web")
    parser.add_argument("--dim", type=int, default=64, help="Graph dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--reset", action="store_true", help="Delete saved weights and start fresh")
    parser.add_argument("--chat", type=str, default=None,
        help='Send queries in batch mode. Use | to separate multiple queries. '
             'Outputs Q: and A: lines for easy parsing. E.g.: --chat "hi|what is trust|bye"')
    parser.add_argument("--strategy", action="store_true", help="Include strategy name in --chat output")
    parser.add_argument("--trace", action="store_true", help="Print edge-level chain traces")
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD emotion modulation")
    parser.add_argument("--no-rlm", action="store_true", help="Disable RLMv2 triple verification")
    parser.add_argument("--no-beliefs", action="store_true", help="Disable belief store")

    parser.add_argument("--user", type=str, default=None,
                        help="User name for multi-user isolation (creates user-specific save files)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Custom data directory for weights and GloVe cache")
    parser.add_argument("--export-graph", type=str, default=None,
                        help="Export graph to JSON file (all concepts + edges)")
    parser.add_argument("--import-graph", type=str, default=None,
                        help="Import graph from JSON file (merge into existing)")
    parser.add_argument("--stats", action="store_true",
                        help="Print graph statistics")
    parser.add_argument("--concept", type=str, default=None,
                        help="Show what RAVANA knows about a concept")
    args = parser.parse_args()

    # Handle --reset
    if args.user:
        save_path = os.path.join(_proj_root, f"ravana_weights{args.user}.pkl")
    else:
        save_path = os.path.join(_proj_root, "ravana_weightsNone.pkl")
    if args.reset:
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f"  [Reset] Deleted {os.path.basename(save_path)}, starting fresh!")
        else:
            print(f"  [Reset] No saved weights found, starting fresh!")

    data_dir = args.data_dir
    user_suffix = args.user if args.user else None
    engine = CognitiveChatEngine(dim=args.dim, seed=args.seed, baby_mode=True, data_dir=data_dir, user_suffix=user_suffix)
    if args.trace:
        engine._trace_enabled = True
    if args.no_vad:
        engine.use_vad = False
        print("  [Config] VAD modulation disabled")
    if args.no_rlm:
        engine.use_rlm = False
        print("  [Config] RLMv2 triple verification disabled")
    if args.no_beliefs:
        engine.use_beliefs = False
        print("  [Config] Belief store disabled")

    # ── BATCH MODE (--chat) ──
    # ── Phase 6 CLI Actions ──
    if args.export_graph:
        try:
            import json
            g = engine.graph
            data = {
                "nodes": [{"id": n.id, "label": n.label} for n in g.nodes.values()],
                "edges": [{"source": src, "target": tgt,
                           "relation": e.relation_type, "weight": e.weight}
                          for (src, tgt), e in g.edges.items()],
            }
            with open(args.export_graph, "w") as f:
                json.dump(data, f, indent=2)
            ec = len(data["edges"])
            nc = len(data["nodes"])
            print(f"  [Export] Exported {nc} nodes + {ec} edges to {args.export_graph}")
        except Exception as e:
            print(f"  [Export] Failed: {e}")
        engine.save()
        return
    if args.import_graph:
        try:
            import json
            with open(args.import_graph, "r") as f:
                data = json.load(f)
            g = engine.graph
            count = 0
            for node_data in data.get("nodes", []):
                nid = node_data.get("id")
                label = node_data.get("label", "")
                if nid is not None and label:
                    g.add_node(nid, label=label)
                    count += 1
            for edge_data in data.get("edges", []):
                src = edge_data.get("source")
                tgt = edge_data.get("target")
                rel = edge_data.get("relation", "related")
                w = edge_data.get("weight", 0.5)
                if src is not None and tgt is not None:
                    g.add_edge(src, tgt, relation_type=rel, weight=w)
            print(f"  [Import] Imported {len(data.get('nodes', []))} nodes + {len(data.get('edges', []))} edges")
        except Exception as e:
            print(f"  [Import] Failed: {e}")
        engine.save()
        return
    if args.stats:
        print(f"  [Stats] Graph has {len(engine.graph.nodes)} nodes and {len(engine.graph.edges)} edges")
        print(f"  [Stats] Turn count: {engine.turn_count}")
        return
    if args.concept:
        nids = engine._concept_keywords.get(args.concept.lower(), [])
        if nids:
            node = engine.graph.get_node(nids[0])
            if node:
                outgoing = engine.graph.get_outgoing(nids[0])
                print(f"  [Concept] '{args.concept}': {len(outgoing)} edges, vector dim={len(node.vector) if node.vector is not None else 0}")
                for tgt_node, e in outgoing[:10]:
                    tgt_label = engine.graph.get_node(tgt_node).label if engine.graph.get_node(tgt_node) else "?"
                    print(f"    -> {tgt_label} [{e.relation_type}] w={e.weight:.3f}")
            else:
                print(f"  [Concept] '{args.concept}' found but no node data")
        else:
            print(f"  [Concept] '{args.concept}' not found in graph")
        return

    if args.chat is not None:
        queries = [q.strip() for q in args.chat.split("|") if q.strip()]
        if not queries:
            return
        results = []
        for i, q in enumerate(queries):
            t0 = time.time()
            try:
                resp = engine.process_turn(q)
            except Exception as e:
                resp = f"[error: {e}]"
            elapsed = time.time() - t0
            strategy = engine._last_strategy if args.strategy else ""
            strat_tag = f" [{strategy}]" if strategy else ""
            results.append((q, resp, elapsed))
            print(f"Q{i+1}: {q}")
            print(f"A{i+1}: {resp}{strat_tag}")
            if elapsed > 0.5:
                print(f"     [...{elapsed:.1f}s]")
            if args.trace:
                engine.print_traces(f"Q{i+1}")
            print()
        result = engine.save()
        print(f"  [{result}]")
        print(f"  [Stats] Turns: {engine.turn_count}, Words: {len(engine.graph.nodes)}, Sleeps: {engine.sleep_cycles_completed}")
        return

    # ── INTERACTIVE MODE ──
    print()
    print("  ============================================")
    print("   RAVANA - teenage mind, learning from the web...")
    print("  ============================================")
    print()

    if engine.turn_count == 0:
        print()
        print("  Hey! I'm a teenage mind — I know some things but")
        print("  I'm always curious to learn more. I can think about")
        print("  causes, patterns, and different perspectives.")
        print("  Talk to me about anything!")
    else:
        print(f"  Welcome back! I now know {len(engine.graph.nodes)} words across {len(engine.graph.edges)} connections.")
        print(f"  I've slept {engine.sleep_cycles_completed} times to consolidate my learning.")
    print()

    try:
        while True:
            try:
                user_input = input("  You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue

            # Detect quit naturally
            if user_input.lower() in ("bye", "goodbye", "see you", "good night"):
                print(f"\n  RAVANA: Bye bye! I'll remember what you taught me!")
                return

            try:
                t0 = time.time()
                response = engine.process_turn(user_input)
                elapsed = time.time() - t0
                print(f"\n  RAVANA: {response}")

                # Show learning stats every 5 turns
                if engine._learning_count > 0 and engine.turn_count % 5 == 0:
                    print(f"  [I've learned {engine._learning_count} times from the web and know "
                          f"{len(engine.graph.nodes)} words now!]")

                if elapsed > 0.5:
                    print(f"  [...took a moment to think...]")

            except Exception as e:
                print(f"\n  RAVANA: Hmm, I got confused. Let me try again!")
                if "--debug" in sys.argv:
                    import traceback
                    traceback.print_exc()
    finally:
        # Auto-save on any exit
        result = engine.save()
        print(f"  [{result}]")


if __name__ == "__main__":
    main()
