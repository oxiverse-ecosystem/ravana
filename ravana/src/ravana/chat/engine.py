"""
RAVANA Cognitive Chat Engine — main orchestrator.
Contains CognitiveChatEngine with __init__, process_turn, save/_load.
Helper classes in models.py, user_model.py, belief_store.py.
"""
import sys, os, time, random, json, re, threading, hashlib
import urllib.request
import socket
socket.setdefaulttimeout(4.0)
from urllib.error import URLError
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import deque, Counter

# Import constants from shared module
_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, _proj_root)
sys.path.insert(0, os.path.join(_proj_root, "ravana_ml", "src"))
sys.path.insert(0, os.path.join(_proj_root, "ravana", "src"))
sys.path.insert(0, os.path.join(_proj_root, "ravana-v2", "src"))

from ravana_ml.graph import ConceptGraph, ConceptEdge
from ravana_grace.core.emotion import VADEmotionEngine, VADConfig
from ravana_grace.core.identity import IdentityEngine
from ravana_grace.core.meaning import MeaningEngine, MeaningConfig
from ravana_grace.core.dual_process import DualProcessController, DualProcessConfig
from ravana_grace.core.global_workspace import GlobalWorkspace, GWConfig
from ravana_grace.core.meta_cognition import MetaCognition, MetaCognitiveConfig, EpistemicMode
from ravana_grace.core.sleep import SleepConsolidation, SleepConfig
from ravana.language.basal_ganglia import BasalGangliaGate
from ravana.language.cerebellar_ngram import CerebellarNgram, CerebellarState
from ravana.language.prefrontal_workspace import PrefrontalWorkspace, DiscourseIntent, DiscoursePlan, DiscourseType
from ravana.language.syntactic_cell_assembly import SyntacticCellAssembly, SyntacticFrame
from ravana.language.surface_realizer import SurfaceRealizer, DiscourseState
from ravana_ml.nn.neural_decoder import NeuralDecoder
from ravana.core import UserEmotionDetector, EmotionalMirrorEngine, MirrorConfig
from ravana.core.hippocampal_buffer import HippocampalBuffer, HippocampalConfig
from ravana.core.proposition_parser import PropositionParser
from ravana.core.causal_schema import CausalSchemaLearner, CausalSchemaConfig
from ravana.core.implicature_detector import ImplicatureDetector
from ravana.core.relation_memory import RelationMemory, RelationMemoryConfig
from ravana.core.quantity_modifier import QuantityModifierSystem
from ravana.core.situation_model import SituationModel
from ravana.core.event_schema import EventSchemaLibrary

# Optional bs4
try:
    import bs4  # noqa: F401
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

# Import constants
from .constants import TEEN_CONCEPTS, WEB_GARBAGE, STOP_WORDS, ConceptPosDict
from .web_learning import WebLearningMixin
import pickle
from ravana.web.learner import SearchEngine

# Re-export _is_word_salad for response validation
from .constants import _is_word_salad
from .constants import _is_keyboard_mash
from ravana.language.verb_lexicon import VerbLexicon
from .models import FailedQuery, ChainHop, ChainTrace, CognitiveResponseContext, Correction, CorrectionType
from .user_model import UserModel
from .belief_store import BeliefStore
from ravana.nn.rlm import Plasticity

# Phase 1 & 2 Imports
from ravana.core.predictive_coding import PredictiveCodingLearner
from ravana.core.coherence import CoherenceNetwork
from ravana.core.working_memory import WorkingMemory
from ravana.storage.db import CognitiveDB, migrate_pickle_to_sqlite
from ravana.core.vsa import VSAManager
from ravana.language.schemas import SchemaLibrary
from ravana.core.system1 import System1Attractor
from ravana.core.system2 import System2Simulator

# Phase 3 Imports
from ravana.learn.curiosity import CuriosityEngine
from ravana.learn.consolidation import HippocampalReplay
from ravana.language.register import RegisterController


class CognitiveChatEngine(WebLearningMixin):  # Methods inherited from mixins
    """RAVANA cognitive chat engine — starts as a baby, learns from the web.
    
    Methods are organized across 4 mixin modules:
    - engine.py (this file): __init__, process_turn, save, _load, user model, memory
    - chain_walker.py: ChainWalkerMixin — graph traversal, relation inference
    - response_gen.py: ResponseGenMixin — neural decoder, chitchat, templates
    - web_learning.py: WebLearningMixin — web search, background learning, curiosity
    """
    # Connector words for each relation type
    _EDGE_CONNECTORS = {
        "causal": [("cause", ["because", "since", "as"]), ("result", ["leads to", "causes"]), ("effect", ["so", "therefore"])],
        "contrastive": [("contrast", ["but", "however", "yet"]), ("unexpected", ["nevertheless", "still"])],
        "semantic": [("identity", ["is like", "refers to", "means"]), ("relation", ["relates to", "connects with"])],
        "temporal": [("after", ["after", "then", "next"]), ("during", ["while", "during"])],
        "analogical": [("simile", ["like", "similar to"]), ("meta", ["acts as", "functions like"])],
    }



    def __init__(self, dim: int = 64, seed: int = 42, baby_mode: bool = True, data_dir: Optional[str] = None, user_suffix: str = ""):
        self.dim = dim
        self.rng = np.random.RandomState(seed)

        # Update global STOP_WORDS to filter out conversational filler/debris
        STOP_WORDS.update({"please", "sorry", "thanks", "thank", "hello", "hi", "hey", "bye", "goodbye"})

        self.graph = ConceptGraph(dim=dim, max_nodes=10000)
        self.baby_mode = baby_mode
        self._concept_labels: Set[str] = set()  # set of primary concept labels

        # Definitional knowledge store: concept -> definition string
        # Inspired by ATL convergence zones (Binder & Desai 2011): the brain
        # stores category membership ("X is a Y") as stable neocortical
        # representations, separate from associative episodic edges.
        self._definitions: Dict[str, str] = {}

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
        # Emotional mirror engine -- modulates verbosity and temperature based on user emotion
        self.mirror_engine = EmotionalMirrorEngine(MirrorConfig(mirror_strength=0.55, contagion_rate=0.45))

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
        # Phase A: Concept POS tags for syntactic assembly
        self._concept_pos = ConceptPosDict()
        # Phase SM: Situation Model - DMN-like continuous cognitive workspace
        self.situation_model = SituationModel(dim=self.dim, dmn_decay=0.6)
        # Phase ES: Event Schema Library - procedural/process knowledge
        self.event_schema_lib = EventSchemaLibrary()

        # Phase 5: Use data_dir if provided
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)
            self._save_path = os.path.join(data_dir, f"ravana_weights{user_suffix}.pkl")
            self._glove_cache_path = os.path.join(data_dir, "ravana_glove_cache.npz")
        else:
            os.makedirs(os.path.join(_proj_root, "data"), exist_ok=True)
            self._save_path = os.path.join(_proj_root, "data", f"ravana_weights{user_suffix}.pkl")
            self._glove_cache_path = os.path.join(_proj_root, "data", "ravana_glove_cache.npz")
        self.sleep_cycles_completed = 0
        self._chain_traces: List[ChainTrace] = []
        # Phase 7: Impossible Query Registry
        self._impossible_queries: List[FailedQuery] = []
        # Template invariance tracker: subject -> list of frame signatures
        # Each frame signature is a frozenset of structural verbs/relators
        # used in the response. When the same signature appears across many
        # different subjects, it indicates generic template reuse (confabulation).
        self._response_frame_history: Dict[str, List[frozenset]] = {}
        # FOK pre-check counter: number of times we pre-queued learning this turn
        self._fok_pre_queued: bool = False
        self._fok_pause_done: bool = False  # Prevents infinite LPFC loop per turn
        # Recency boost tracking (dopamine novelty signal analog):
        # Labels of concepts recently learned from web search. During spread,
        # these get a 1.5x activation boost, mimicking VTA dopamine signaling
        # that prioritizes new memories (STC hypothesis, Redondo & Morris 2011).
        self._recently_learned_labels: Set[str] = set()
        self._recent_learn_turn: int = 0
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
        # Phase 11.3: Discourse context (cross-turn accumulation, N400/P600 integration)
        self._sentence_vector: Optional[np.ndarray] = None
        self._discourse_context: Optional[np.ndarray] = None
        # Phase 11.4: Orthogonal context/content subspaces (PMC 2025)
        # Content = what we're talking about (semantics)
        # Context = how we're talking about it (pragmatics, discourse frame)
        self._content_vector: Optional[np.ndarray] = None
        self._context_vector: Optional[np.ndarray] = None
        # Phase 9b: Prediction error tracking (surprise signal for Active Inference)
        self._mean_prediction_error = 0.0
        self._prediction_error_count = 0
        # Integration toggles (can be disabled via CLI)
        self.use_vad = True
        self.use_rlm = True
        self.use_beliefs = True
        self.belief_store = BeliefStore()

        # Plasticity engine for Hebbian learning and episodic triples
        self.plasticity = Plasticity(self.graph, base_lr=0.005)

        # New cognitive modules (Phase 2-5)
        self.hippocampal_buffer = HippocampalBuffer(HippocampalConfig(max_facts=50, decay_turns=50))
        self.proposition_parser = PropositionParser()
        self.causal_schema = CausalSchemaLearner(CausalSchemaConfig())
        self.implicature_detector = ImplicatureDetector()
        self.relation_memory = RelationMemory(RelationMemoryConfig())
        self.quantity_modifier = QuantityModifierSystem()
        self._pending_quantity_result = None

        # Fix 2: Dormant edge tracking — auto-wired GloVe edges are invisible
        # until the user model visits them at least once.
        self._dormant_edges: Set[Tuple[int, int]] = set()

        # Phase 1 & 2 Integrations
        self._db_path = self._save_path.replace(".pkl", ".db")
        if os.path.exists(self._save_path) and not os.path.exists(self._db_path):
            try:
                migrate_pickle_to_sqlite(self._save_path, self._db_path)
            except Exception as e:
                print(f"  [Migration] Legacy migration failed: {e}")
                
        self.db = CognitiveDB(self._db_path)
        self.working_memory = WorkingMemory(capacity=self._pfc_buffer_capacity)
        self.predictive_coding_learner = PredictiveCodingLearner(self.graph)
        self.coherence_net = CoherenceNetwork()
        self.vsa_manager = VSAManager(dim=self.dim)
        self.schema_library = SchemaLibrary(self.vsa_manager)
        self.system1_attractor = System1Attractor(self.graph, threshold=0.4)
        self.system2_simulator = System2Simulator(self.graph, self.causal_schema)

        # Phase 3 Integrations
        self.curiosity_engine = CuriosityEngine(rng=self.rng)
        self.hippocampal_replay = HippocampalReplay(capacity=200)
        self.register_controller = RegisterController(default_register="casual")

        # Build reverse lookup from connector word → relation type
        self._CONNECTOR_TO_REL: Dict[str, str] = {}
        for rel_type, tiers in self._EDGE_CONNECTORS.items():
            for entry in tiers:
                options = entry[1] if isinstance(entry, tuple) and len(entry) == 2 else entry[2]
                for opt in options:
                    self._CONNECTOR_TO_REL[opt] = rel_type
        self._CONNECTOR_SET = set(self._CONNECTOR_TO_REL.keys())

        # Concepts that are grammatical/function words - should never be frame targets
        # Actual grammatical/function words that should never be frame targets
        # (prepositions, pronouns, conjunctions, determiners, particles)
        # NOT content words like "love", "time", "life", "take", "make", "certain", "impossible" 
        # those are valid targets!
        self._GRAMMATICAL_CONCEPTS = {
            # Prepositions/particles
            "out", "in", "on", "off", "up", "down", "over", "under", "above",
            "below", "through", "across", "between", "among", "around", "about",
            "after", "before", "since", "until", "during", "while", "when",
            "where", "why", "how", "what", "who", "which", "whom", "whose",
            "here", "there", "now", "then", "later",
            "soon", "ago", "back", "away", "forward", "backward", "inside",
            "outside", "near", "far",
            # Pronouns
            "we", "they", "them", "their", "us", "our", "he", "she", "him", "her",
            "i", "you", "me", "my", "mine", "your", "yours", "his", "hers",
            "its", "ours", "theirs", "myself", "yourself", "himself", "herself",
            "itself", "ourselves", "yourselves", "themselves",
            # Determiners/quantifiers
            "a", "an", "the", "this", "that", "these", "those",
            "some", "any", "every", "each", "all", "both", "either", "neither",
            "much", "many", "few", "little", "more", "most", "less", "least",
            "enough", "several", "one", "two", "three", "first", "second", "last",
            "other", "another",
            # Determiner-like adjectives that make poor discourse targets
            "such", "same", "different", "certain", "whole", "own", "particular",
            # Conjunctions
            "and", "or", "but", "nor", "yet", "so", "for", "because", "since",
            "although", "though", "if", "unless", "until", "while", "when",
            "where", "whether", "than", "as", "like",
            # Auxiliary/modal verbs (function words)
            "be", "am", "is", "are", "was", "were", "been", "being",
            "have", "has", "had", "do", "does", "did", "doing",
            "can", "could", "will", "would", "shall", "should",
            "may", "might", "must", "ought", "need", "dare",
            # Particles/adverbs that are purely grammatical
            "not", "no", "yes", "very", "too", "also", "just", "only",
            "even", "still", "already", "yet", "again", "once", "twice",
            "maybe", "perhaps", "probably", "possibly",
            "here", "there", "where", "why", "how", "what", "when",
            # Discourse markers / connectives
            "instead", "therefore", "however", "moreover", "furthermore",
            "besides", "nevertheless", "nonetheless", "accordingly", "consequently",
            "thus", "hence", "accordingly", "subsequently", "meanwhile",
            # Conversational filler/debris
            "please", "sorry", "thanks", "thank", "hello", "hi", "hey", "bye", "goodbye",
        }

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
        self._sleep_metrics: Dict[str, Any] = {
            "edges_strengthened": 0,
            "edges_pruned": 0,
            "episodic_consolidated": 0,
            "impossible_queries_resolved": 0,
            "total_sleep_cycles": 0,
            "last_sleep_turn": 0,
            "last_sleep_metrics": {},
        }
        self._sleep_schedule_turns: int = 20  # Run sleep every N turns regardless of pressure
        self._sleep_schedule_time: int = 300  # Run sleep every N seconds (5 min) if no turns

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
        # Opt 6: O(1) hashmap for repetition penalty
        self._recent_traversal_map: Dict[Tuple[int, int], int] = {}
        self._visited_concepts: Set[str] = set()
        self._dopamine_tone: float = 0.5
        self._td_error_history: List[float] = []
        self._expected_strength: float = 0.25
        self._episodic_edges: Dict[Tuple[int, int], Any] = {}
        self._semantic_edges: Dict[Tuple[int, int], Any] = {}
        # Phase 15.4: Pre-built index for O(1) dual-store lookup

        # Phase 15.4: Pre-built src-indexed lookups for O(1) dual-store access
        self._semantic_by_src: Dict[int, list] = {}
        self._episodic_by_src: Dict[int, list] = {}
        # Phase B: Basal Ganglia Gate — Go/NoGo gating replaces temperature softmax
        self.basal_ganglia = BasalGangliaGate()
        # Phase C: Cerebellar n-gram — sparse sequence learning for grammatical transitions
        self.cerebellar_ngram = CerebellarNgram()
        # Phase D: Prefrontal workspace — discourse planning before generation
        self.pfc_workspace = PrefrontalWorkspace(capacity=5)
        self._proper_nouns = set()
        # Phase E: Syntactic cell assemblies — Hebbian role learning with seeded priors
        self.syntactic_assembly = SyntacticCellAssembly(learning_rate=0.05)
        self.syntactic_assembly.proper_nouns = self._proper_nouns
        # Phase F: Surface realizer — rule-governed English morphology with dopamine modulation
        self.surface_realizer = SurfaceRealizer()
        self.surface_realizer.proper_nouns = self._proper_nouns
        # Phase 6: Wire vector function for semantic verb selection (VerbLexicon)
        self.surface_realizer.set_vector_fn(self._get_modulated_vector)
        VerbLexicon.set_glove_fn(self._get_modulated_vector)
        self._cerebellar_ngram: Dict[str, Dict[str, float]] = {}
        self._cerebellar_depth: Dict[str, float] = {}
        self._concept_confidence: Dict[str, float] = {}
        self._calibration_error: float = 0.0
        self._metacognitive_review_turn: int = 0
        # Background web learning
        self._bg_learning_thread: Optional[threading.Thread] = None
        self._bg_learning_active: bool = False
        self._bg_learning_queue: List[str] = []  # queries to research in background
        self._bg_lock = threading.Lock()
        self._vocab_lock = threading.RLock()
        self._graph_lock = threading.RLock()
        self._bg_idle_event = threading.Event()  # set when user sends a message
        self._bg_search_count: int = 0  # total background searches performed
        self._bg_multi_search_max: int = 1  # related searches to expand per query (reduced from 3)
        self._bg_idle_search_count: int = 0  # searches done in current idle period

        # Search engine (instance-specific so circuit breaker settings apply)
        self.search_engine = SearchEngine()

        # Phase 18: Curiosity Drive - autonomous topic selection for background learning
        self._curiosity_drive_enabled: bool = True  # can be disabled via --no-curiosity
        self._curiosity_cycles_this_session: int = 0  # bg auto-select count per idle session
        self._concept_visit_count: Dict[str, int] = {}  # how many times each concept was visited
        self._concept_learning_progress: Dict[str, float] = {}  # rate of prediction error decrease per concept
        self._concept_pe_delta: Dict[str, float] = {}  # per-step PE delta (positive = learning progress)
        self._curiosity_topics_queue: List[str] = []  # topics autonomously selected for research
        self._last_auto_learn_turn: int = 0  # turn when we last autonomously selected topics
        self._curiosity_urgency: float = 0.0  # overall curiosity drive (0-1)
        # Phase 18b: User priming - track recent user topics for curiosity boosting
        self._user_query_topics: List[str] = []  # last 10 topics user asked about
        self._user_last_topic: str = ""  # most recent user topic
        self._activation_boost: Optional[Dict[str, float]] = None
        # Solution #2: Reasoning mode (stochastic / deterministic / exploratory)
        self.reasoning_mode: str = "stochastic"

        # Consistency tracking for deterministic mode: (subject_hash, tuple(seen)) -> path
        self._consistency_paths: Dict[int, List[str]] = {}
        self._consistency_trace: List[str] = []
        # Phase 18c: Multi-source consensus tracking for hallucination guard
        self._concept_sources: Dict[str, Set[str]] = {}  # concept -> set of source URLs
        # Phase 18d: Explored contradiction pairs (prevent re-queuing "good vs bad debate")
        self._explored_contradictions: Set[Tuple[str, str]] = set()

        # Phase: Correction Log (ACC/ERN error correction circuit)
        self._correction_log: List[Correction] = []

        # Neural decoder — initialized lazily after graph is ready
        self.neural_decoder: Optional[NeuralDecoder] = None
        self._decoder_word_to_idx: Dict[str, int] = {}
        self._decoder_idx_to_word: Dict[int, str] = {}
        self._decoder_word_to_embed: Dict[str, np.ndarray] = {}
        self._decoder_vocab_built: bool = False
        self._decoder_training_count: int = 0
        self._decoder_web_training_count: int = 0
        self._decoder_seed_training_count: int = 0
        self._saved_decoder_state: dict = {}

        # Always init GloVe first (cheap if cache exists) so graph vectors
        # are real semantic vectors, not hash-random fallbacks.
        self._init_glove()
        if os.path.exists(self._save_path):
            loaded = self._load()
            if loaded:
                self._revector_existing_nodes()
                self._bootstrap_domain_concepts()
                # Load decoder FIRST (before building vocab) to detect saved vocab size
                if hasattr(self, '_saved_decoder_state') and self._saved_decoder_state:
                    # Determine saved vocab size from output_proj weight
                    # State dict format: {'param_name': {'data': tensor, ...}}
                    out_proj = self._saved_decoder_state.get('output_proj.weight', {})
                    if 'data' in out_proj and hasattr(out_proj['data'], 'shape'):
                        saved_vocab = out_proj['data'].shape[0]
                        # Temporarily override vocab size for _build_decoder_vocab
                        self._forced_vocab_size = saved_vocab
                self._revector_existing_nodes()
                self._bootstrap_domain_concepts()
                self._build_decoder_vocab()
                # Now load the decoder state (vocab sizes match)
                if self._saved_decoder_state and self.neural_decoder is not None:
                    try:
                        self.neural_decoder.load_state_dict(self._saved_decoder_state)
                        self._saved_decoder_state = {}
                    except Exception:
                        self._saved_decoder_state = {}
                # Decoder loaded successfully — skip seed training if already done
                needs_train = self._decoder_training_count < 1000
                self._needs_seed_training = needs_train
                self._needs_synthetic_training = False
                self._freeze_decoder_vocab = True  # Freeze decoder vocab during inference
                print(f"  [Loaded] Remembered {len(self.graph.nodes)} words from before!")
                return

        # Cold start (no saved weights): seed everything from scratch.
        self._seed_concepts()
        self._bootstrap_domain_concepts()
        self._build_decoder_vocab()
        # Skip initial corpus training during cold start — the training
        # script (train.py) handles it separately with more control.
        # This shaves ~hours off first-time initialization.
        self._needs_seed_training = False
        self._needs_synthetic_training = False
        print(f"  [Teen] Knows {len(self.graph.nodes)} words, ready to learn!")

    # ─── Neural Decoder Vocabulary ───

    # Vocab grows dynamically with learning — no artificial cap.
    # Only limit is GPU-free performance: monitors generation latency
    # and auto-adjusts if needed. Default to large for fluent English.
    MAX_DECODER_VOCAB_SIZE = 15000



    def _get_curiosity_scores(self, max_topics: int = 10) -> List[Tuple[str, float]]:
        """
        Compute curiosity scores for graph concepts using prediction free energy.
        
        Combines:
        1. Node-level prediction_free_energy (from active inference)
        2. Edge-level prediction_free_energy (from edge prediction errors)
        3. Contradiction involvement (cognitive dissonance)
        4. Dormant edges (unexplored connections)
        5. Low visit count (novelty)
        
        Returns list of (concept_label, score) sorted by curiosity descending.
        """
        scores = {}
        seen = set()
        
        # Source 1: Node-level prediction free energy (Active Inference: surprise drives learning)
        for nid, node in list(self.graph.nodes.items()):
            if node.label:
                pe = getattr(node, 'prediction_free_energy', 0.0)
                if pe > 0.1:
                    label = node.label.lower()
                    if label not in seen and len(label) >= 3:
                        scores[label] = scores.get(label, 0.0) + pe * 2.0  # weight node PE higher
                        seen.add(label)
        
        # Source 2: Edge-level prediction free energy (edges with high prediction error)
        for (src, tgt), edge in list(self.graph.edges.items()):
            edge_pe = getattr(edge, 'prediction_free_energy', 0.0)
            if edge_pe > 0.05:
                sn = self.graph.nodes.get(src)
                tn = self.graph.nodes.get(tgt)
                for node in (sn, tn):
                    if node and node.label:
                        label = node.label.lower()
                        if len(label) >= 3:
                            scores[label] = scores.get(label, 0.0) + edge_pe * 1.5
        
        # Source 3: Contradiction-involved concepts (cognitive dissonance)
        for label in self._contradiction_map:
            l = label.lower()
            if len(l) >= 3:
                scores[l] = scores.get(l, 0.0) + 1.0
        
        # Source 4: Concepts with dormant (unexplored) edges
        if hasattr(self, '_dormant_edges') and self._dormant_edges:
            dormant_counts = {}
            for src, tgt in self._dormant_edges:
                sn = self.graph.nodes.get(src)
                tn = self.graph.nodes.get(tgt)
                if sn and sn.label:
                    dormant_counts[sn.label.lower()] = dormant_counts.get(sn.label.lower(), 0) + 1
                if tn and tn.label:
                    dormant_counts[tn.label.lower()] = dormant_counts.get(tn.label.lower(), 0) + 1
            for label, count in dormant_counts.items():
                if len(label) >= 3:
                    scores[label] = scores.get(label, 0.0) + min(count * 0.5, 2.0)
        
        # Source 5: Novelty - least visited concepts
        if hasattr(self, '_concept_visit_count'):
            visit_counts = [(lbl, cnt) for lbl, cnt in self._concept_visit_count.items() if len(lbl) >= 3]
            if visit_counts:
                max_visits = max(cnt for _, cnt in visit_counts)
                for label, count in visit_counts:
                    novelty = 1.0 - (count / max(max_visits, 1))
                    scores[label] = scores.get(label, 0.0) + novelty * 0.5
        
        # Sort by score descending
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:max_topics]

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
                vecs = data['vecs']  # shape (n_words, glove_dim) - RAW vectors
                proj = data['proj']
                self._glove_dim = int(data['glove_dim'])
                self._glove_proj = proj
                
                # Vectorized batch projection: (dim, glove_dim) @ (glove_dim, n_words) -> (dim, n_words)
                projected = self._glove_proj @ vecs.T  # shape (dim, n_words)
                # Normalize all projected vectors in batch
                norms = np.linalg.norm(projected, axis=0)
                norms[norms == 0] = 1.0  # avoid division by zero
                projected = (projected / norms).astype(np.float32)  # shape (dim, n_words)
                
                # Populate dicts
                self._glove_vecs = {words[i]: vecs[i] for i in range(len(words))}
                self._glove_vector_cache = {words[i]: projected[:, i] for i in range(len(words))}
                
                print(f"  [GloVe] Loaded {len(self._glove_vecs)} projected vectors from cache ({self._glove_dim}D -> {self.dim}D)")
                return
            except Exception as e:
                print(f"  [GloVe] Cache load failed: {e}, re-reading from file...")

        # Fall back to reading raw GloVe file
        glove_dir = os.path.join(_proj_root, 'data', 'glove')
        for name in ['glove.6B.100d.txt', 'glove.6B.50d.txt']:
            path = os.path.join(glove_dir, name)
            if os.path.exists(path):
                self._glove_dim = 100 if '100d' in name else 50
                break
        else:
            # No local GloVe file — attempt auto-download
            print("  [GloVe] No local GloVe file found. Attempting auto-download...")
            if self._download_glove(glove_dir):
                # Retry finding the file after download
                for name in ['glove.6B.100d.txt', 'glove.6B.50d.txt']:
                    path = os.path.join(glove_dir, name)
                    if os.path.exists(path):
                        self._glove_dim = 100 if '100d' in name else 50
                        break
            else:
                print("  [GloVe] Auto-download failed. Running without GloVe vectors.")
                return
        glove_path = os.path.join(glove_dir, f'glove.6B.{self._glove_dim}d.txt')
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



    def _download_glove(self, glove_dir: str) -> bool:
        """Download GloVe 6B vectors from Stanford NLP.
        
        Downloads glove.6B.zip (~822 MB), extracts glove.6B.100d.txt and glove.6B.50d.txt.
        Uses streaming download with progress indicator.
        
        Returns True on success, False on failure.
        """
        import zipfile
        import io
        
        glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"
        zip_path = os.path.join(glove_dir, "glove.6B.zip")
        
        try:
            os.makedirs(glove_dir, exist_ok=True)
            
            print(f"  [GloVe] Downloading from {glove_url}...")
            req = urllib.request.Request(glove_url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) RAVANA/1.0'
            })
            
            # Stream download with progress
            with urllib.request.urlopen(req, timeout=300) as resp:
                total_size = int(resp.headers.get('Content-Length', 0))
                downloaded = 0
                chunk_size = 8192
                
                with open(zip_path, 'wb') as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0 and downloaded % (50 * 1024 * 1024) < chunk_size:
                            pct = downloaded / total_size * 100
                            print(f"  [GloVe] Download progress: {pct:.1f}% ({downloaded / 1024 / 1024:.0f} MB)")
                
                if total_size > 0:
                    pct = downloaded / total_size * 100
                    print(f"  [GloVe] Download complete: {pct:.1f}% ({downloaded / 1024 / 1024:.0f} MB)")
            
            # Extract the needed files
            print("  [GloVe] Extracting glove.6B.100d.txt and glove.6B.50d.txt...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for target in ['glove.6B.100d.txt', 'glove.6B.50d.txt']:
                    if target in zf.namelist():
                        zf.extract(target, glove_dir)
                        print(f"  [GloVe] Extracted {target}")
            
            # Clean up zip file to save space
            try:
                os.remove(zip_path)
                print("  [GloVe] Cleaned up zip archive")
            except Exception:
                pass
            
            return True
            
        except Exception as e:
            print(f"  [GloVe] Download failed: {e}")
            # Clean up partial download
            try:
                if os.path.exists(zip_path):
                    os.remove(zip_path)
            except Exception:
                pass
            return False



    def _revector_existing_nodes(self) -> int:
        # Re-project any graph node whose vector was hash-random by
        # replacing it with a GloVe projection when a matching word is
        # available. Safe to call repeatedly: it is a no-op when a node
        # already matches its GloVe vector.
        if self._glove_vecs is None:
            return 0
        updated = 0
        for nid, node in list(self.graph.nodes.items()):
            if not node.label:
                continue
            if node.vector is None or node.vector.shape[0] != self.dim:
                continue
            gv = self._glove_vector(node.label)
            if gv is None:
                continue
            diff = float(((node.vector - gv) ** 2).sum())
            if diff < 1e-4:
                continue
            node.vector = gv.astype(np.float32)
            updated += 1
        if updated:
            self.graph._vectors_dirty = True
            try:
                self.graph._rebuild_vector_matrix()
            except Exception:
                pass
        return updated



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

    def _get_modulated_vector(self, word: str) -> Optional[np.ndarray]:
        """Retrieve a context-modulated vector for a word (LIFG-ATL modulation).
        
        Warps the static GloVe/node vector towards the current turn's discourse context.
        """
        # Get baseline GloVe vector
        base_vec = self._glove_vector(word)
        if base_vec is None:
            return None
        
        # Get current situation model context vector (LIFG signal)
        ctx_vec = getattr(self, '_context_vector', None)
        if ctx_vec is None or not np.any(ctx_vec != 0):
            return base_vec
        
        # Determine modulation strength based on arousal
        arousal = self.emotion.state.arousal if hasattr(self, 'emotion') else 0.5
        beta = 0.2 + 0.3 * arousal  # warp between 20% and 50%
        
        # Warp the vector
        modulated = (1.0 - beta) * base_vec + beta * ctx_vec
        norm = np.linalg.norm(modulated)
        if norm > 0:
            modulated /= norm
        return modulated.astype(np.float32)

    # ─── Phase 1: Auto-Expansion from Every Message ───



    def _user_input_is_gibberism(self, text: str) -> bool:
        """Detect user input that contains no real words at all (random
        letter-salad like 'asdf qwer zxcv'). Such input should not be treated
        as a learnable concept and confabulated about.

        We refuse only when (a) there is no question/learning intent and
        (b) not a single meaningful token is found in GloVe / the known
        concept graph. This keeps genuine (if obscure) learning queries like
        'what is quokka' flowing through, while blocking pure nonsense."""
        toks = re.findall(r"[a-zA-Z']+", text.lower())
        meaningful = [w for w in toks if len(w) >= 2 and w not in STOP_WORDS
                      and not w.isdigit()]
        if len(meaningful) < 2:
            return False
        question_words = {
            "what", "why", "who", "how", "where", "when", "which", "is",
            "are", "was", "were", "do", "does", "did", "can", "could",
            "would", "should", "will", "tell", "explain", "describe",
            "define", "name", "give", "show", "make", "help",
        }
        if any(w in question_words for w in meaningful):
            return False
        if self._glove_vecs is not None:
            for w in meaningful:
                # Keyboard mashing (e.g. 'asdf', 'qwer', 'zxcv') is random
                # letter salad even if a token happens to be in GloVe.
                if _is_keyboard_mash(w):
                    continue
                if self._glove_vector(w) is not None:
                    return False
                if w in self._concept_keywords:
                    return False
                if w in getattr(self, "_proper_nouns", set()):
                    return False
            return True
        # GloVe unavailable: fall back to the vowel heuristic (real English
        # words almost always contain a vowel).
        for w in meaningful:
            if _is_keyboard_mash(w):
                continue
            if any(ch in w for ch in "aeiouy"):
                return False
        return True

    def process_turn(self, user_input: str) -> str:
        """Process input and generate a response, auto-learning when needed."""
        # Guard: reject pure letter-salad so it is not treated as a concept and
        # confabulated about.
        if self._user_input_is_gibberism(user_input):
            self._last_strategy = "gibberish_guard"
            resp = "hmm, that doesn't really make sense to me — could you say it another way?"
            self._last_responses.append(resp)
            if len(self._last_responses) > 10:
                self._last_responses = self._last_responses[-10:]
            return resp

        # Scan user query for proper nouns dynamically
        try:
            words = user_input.strip().split()
            if len(words) > 1:
                for w in words[1:]:  # Skip first word (sentence start capitalized)
                    clean_w = w.strip(".,!?\"'()[]{}*:;")
                    if clean_w and clean_w[0].isupper() and clean_w.lower() not in STOP_WORDS:
                        self._proper_nouns.add(clean_w.lower())
        except Exception:
            pass

        self.turn_count += 1
        self._learned_this_turn = False
        self._cascade_for_quality = False
        self._fok_pause_done = False
        self.user_model.reset_correction_flags()  # Reset LPFC pause flag each turn
        # Decay recency boost: clear after 10 turns (synaptic tag window)
        if hasattr(self, '_recent_learn_turn') and self.turn_count - self._recent_learn_turn > 10:
            self._recently_learned_labels.clear()

        # Intercept direct identity/preference questions about the user: "what is my name", "who am i", etc.
        clean_input = user_input.lower().strip(" ?!.")
        identity_questions = [
            "what is my name", "what's my name", "do you know my name", "who am i", "tell me my name", "who i am"
        ]
        likes_questions = [
            "what do i like", "what do i love", "do you know what i like", "do you know what i love", 
            "tell me what i like", "tell me what i love", "what i like", "what i love"
        ]
        interests_questions = [
            "what am i interested in", "what do i want to learn", "what do i want to learn about",
            "do you know what i want to learn", "tell me what i want to learn", "what i'm interested in",
            "what i am interested in"
        ]
        
        is_identity_query = clean_input in identity_questions or clean_input.endswith("who am i") or clean_input.endswith("what is my name")
        is_likes_query = clean_input in likes_questions or clean_input.endswith("what do i like") or clean_input.endswith("what do i love")
        is_interests_query = clean_input in interests_questions or clean_input.endswith("what am i interested in") or clean_input.endswith("what do i want to learn")
        
        m_fav_q = re.search(r"\bwhat(?:'s|\s+is)\s+my\s+favorite\s+(.+)", clean_input, re.IGNORECASE)
        
        if is_identity_query or is_likes_query or is_interests_query or m_fav_q:
            response = ""
            if is_identity_query:
                name = getattr(self.user_model, 'user_name', "")
                if name:
                    nl = name.lower()
                    details = ""
                    if nl in self._definitions:
                        details = self._definitions[nl]
                    elif nl in self._concept_keywords:
                        activated_ids = self._concept_keywords[nl]
                        associations = self._spread_and_collect(activated_ids, primary_ids=set(activated_ids))
                        if associations:
                            connected = []
                            for label, _ in associations[:3]:
                                if label.lower() != nl and label.lower() not in STOP_WORDS:
                                    connected.append(label.lower())
                            if connected:
                                details = "connected to " + " and ".join(connected)
                    
                    if details:
                        response = f"your name is {name}. from what i know, you are {details}."
                    else:
                        response = f"your name is {name}! we've been chatting for a bit."
                else:
                    response = "i don't know your name yet! what is your name?"
            
            elif is_likes_query:
                prefs = getattr(self.user_model, 'preferences', {})
                likes = prefs.get("likes", [])
                if likes:
                    response = f"you mentioned that you like {', '.join(likes)}!"
                else:
                    response = "i don't know what you like yet! what are some things you like?"
            
            elif is_interests_query:
                prefs = getattr(self.user_model, 'preferences', {})
                interests = prefs.get("interests", [])
                if interests:
                    response = f"you want to learn about or are interested in {', '.join(interests)}!"
                else:
                    response = "i'm not sure what you're interested in yet. what would you like to learn about?"
            
            elif m_fav_q:
                category = m_fav_q.group(1).strip(" .!?")
                prefs = getattr(self.user_model, 'preferences', {})
                favs = prefs.get("favorites", {})
                if category in favs:
                    response = f"your favorite {category} is {favs[category]}!"
                else:
                    response = f"i don't know your favorite {category} yet! what is it?"
            
            self._last_strategy = "user_identity"
            self._last_responses.append(response)
            if len(self._last_responses) > 10:
                self._last_responses = self._last_responses[-10:]
            self.notify_user_idle()
            return response.lower()

        # Deferred decoder training on first turn (fast startup)
        if getattr(self, '_needs_seed_training', False):
            self._needs_seed_training = False
            try:
                passes = self._train_decoder_on_seed_corpus()
                if self._trace_enabled:
                    print(f"  [init] Seed corpus training: {passes} sentences")
            except Exception as e:
                if self._trace_enabled:
                    print(f"  [init] Seed corpus training error: {e}")
        if getattr(self, '_needs_synthetic_training', False):
            self._needs_synthetic_training = False
            try:
                # Freeze vocab to prevent template words from polluting embeddings
                self._freeze_decoder_vocab = True
                self._train_decoder_from_graph(min_synthetic=500)
                self._freeze_decoder_vocab = False
            except Exception as e:
                if self._trace_enabled:
                    print(f"  [init] Synthetic training error: {e}")

        # ── Cross-turn context accumulation (N400/P600 discourse integration) ──
        # Decay old context rather than wiping it — the brain maintains a
        # situation model across turns (Nature Human Behaviour 2025:
        # "shared representations at longer timescales support integration
        # of incoming conversational content with prior conversational context")
        old_ctx = getattr(self, '_current_context_vector', None)
        if old_ctx is not None:
            old_ctx *= 0.4  # Decay old context (forgets ~60% between turns)
        self._modulated_vectors.clear()
        if hasattr(self, '_prefrontal_buffer'):
            self._prefrontal_buffer = [self._prefrontal_buffer[-1]] if self._prefrontal_buffer else []

        # Signal background thread that user is active
                # Phase F: Reset per-turn surface realizer state (pronoun tracking)
        if hasattr(self, 'surface_realizer'):
            self.surface_realizer.reset_turn()
        # Phase 6a: Reset VerbLexicon refractory period (prevents verb perseveration)
        VerbLexicon.reset_refractory()
        self.notify_user_active()
        # Phase 15.2: Inter-turn episodic edge decay (forgetting between turns)
        self._decay_episodic_edges()
        # Phase 13.3: Decay activation fatigue between turns
        for _fk in list(self._activation_fatigue.keys()):
            self._activation_fatigue[_fk] *= 0.95
            if self._activation_fatigue[_fk] < 0.01:
                del self._activation_fatigue[_fk]
        # Phase 13.3: Reset _visited_concepts every 50 turns for novelty
        if self.turn_count > 1 and self.turn_count % 50 == 0:
            self._visited_concepts.clear()
            # Phase 18: Decay concept visit counts to prevent saturation
            for k in list(self._concept_visit_count.keys()):
                self._concept_visit_count[k] = max(0, self._concept_visit_count[k] - 2)

        # Wait for any background learning to finish its current cycle
        if self._bg_learning_active:
            with self._graph_lock:
                pass  # Ensure graph mutations aren't racing with chain walk
        
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

        # Phase 1.6: Extract causal relations from user statements
        # Creates causal graph edges from patterns like "when X, Y" and "if X, Y"
        # so the chain walk can follow them during response generation.
        causal_edges = self._extract_and_store_causal_relations(user_input)
        if causal_edges > 0:
            # Re-activate to include newly wired causal edges
            activated = self._activate_from_input(user_input)

        # Phase 7: Store activated IDs for strategy framework
        self._last_activated_ids = list(activated)

        # Step 1b.75: Phase 3.3 + 9c — Detect recall triggers with hippocampal reactivation
        # Cross-check with question type to avoid false positives:
        # introductions, greetings, what_is, tell_me, general queries should NOT
        # trigger recall mode — they are about NEW information, not past recall.
        recall_topic = self._detect_recall_trigger(user_input)
        if recall_topic:
            # Cross-check: get question type to filter false positives
            try:
                qtype, _ = self.pfc_workspace.detect_question_type(
                    user_input, concept_pos=getattr(self, '_concept_pos', None))
                # Only enter recall mode for non-introductory question types
                # Introduction/greeting/what_is/tell_me/general are about new info
                recall_blocklist = {"introduction", "greeting", "wellbeing", 
                                    "capability", "farewell", "what_is", "tell_me"}
                if qtype in recall_blocklist:
                    recall_topic = None
            except Exception:
                pass  # Don't break pipeline if qtype detection fails
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

        # Step 2: Extract topic with multi-strategy grounding
        subject, obj = self._extract_topic(user_input, activated)
        # Conditional / hypothetical cleanup: "if the sun disappeared…" yields a
        # junk subject "sun disappeared". Recover the real concept so grounding,
        # web search, and association spread all target "sun", not the frame.
        if self._is_conditional_query(user_input) and subject:
            cleaned = self._clean_scenario_subject(subject, user_input)
            if cleaned and cleaned != subject:
                subject = cleaned
                # Re-key primary IDs / graph activation to the cleaned subject.
                if subject in self._concept_keywords:
                    subject_ids = set()
                    for nid in self._concept_keywords[subject]:
                        subject_ids.add(nid)
                        self.graph.activate(nid, 0.8)
        if not subject:
            # Set default subject for chitchat/social queries to route them correctly
            t = user_input.lower().strip(" ?!.,")
            greetings = r"\b(hi|hello|hey|yo|sup|greetings|whats\s*up|howdy|good\s*morning|good\s*afternoon|good\s*evening)\b"
            wellbeing = r"\b(how\s*are\s*you|how\s*is\s*it\s*going|how\s*are\s*you\s*doing|how\s*have\s*you\s*been|hows\s*it\s*going|hows\s*life)\b"
            capabilities = r"\b(what\s*can\s*you\s*do|what\s*do\s*you\s*do|how\s*do\s*you\s*work|tell\s*me\s*about\s*yourself|who\s*are\s*you|what\s*is\s*your\s*name)\b"
            farewells = r"\b(bye|goodbye|see\s*you|good\s*night|farewell)\b"
            if re.search(greetings, t):
                subject = "hello"
            elif re.search(wellbeing, t):
                subject = "how"
            elif re.search(capabilities, t):
                subject = "ravana"
            elif re.search(farewells, t):
                subject = "bye"
        # Run grounding again to get confidence for auto-web-learning
        _grounded_subj, _gconf, _gmethod = self._ground_query(user_input)
        self._last_grounding_conf = _gconf
        self._last_grounding_method = _gmethod
        # Auto-trigger web learning for low-confidence multi-word queries
        if _gconf < 0.5 and _gmethod == "all_unknown" and _grounded_subj and self.baby_mode:
            with self._bg_lock:
                if _grounded_subj not in self._pending_learning_queue:
                    self._pending_learning_queue.append(_grounded_subj)
        relation = "is"

        # Step 2b: Primary IDs — only these concepts spread activation
        # (other input-matched concepts provide context but don't propagate)
        subject_ids = set()
        sl = subject.lower()
        if sl in self._concept_keywords:
            subject_ids.update(self._concept_keywords[sl])
        else:
            # Multi-word subject fallback: 'dark energy' won't match single-word
            # entries in _concept_keywords. Try each word individually so that
            # primary_ids is populated and the topic relevance gate works.
            for part in sl.split():
                if part in self._concept_keywords:
                    subject_ids.update(self._concept_keywords[part])

        # Step 2c: PFC-derived relation preference for spread activation.
        # The PFC determines what KIND of reasoning the question requires
        # (causal, contrastive, semantic, etc.) and biases the spread accordingly.
        # Using the PFC's task-set primary_relation, not the raw qtype string —
        # because "hypothetical" in GloVe space maps to analogical, but the PFC
        # correctly identifies it as a causal reasoning task.
        _qtype_for_spread, _ = self.pfc_workspace.detect_question_type(
            user_input, concept_pos=self._concept_pos)
        spread_pref = self._relation_modulation_for_word(
            self.pfc_workspace.get_primary_relation_for_qtype(_qtype_for_spread))

        # System 1 / System 2 Dual-Process integration
        try:
            settled_activations, s1_confidence = self.system1_attractor.settle(activated)
            if self.system1_attractor.should_escalate(s1_confidence) and subject:
                s2_trace = self.system2_simulator.simulate_forward(subject, steps=3)
                for state_a, cond, state_b in s2_trace:
                    for target in (state_a, state_b):
                        nids = self._concept_keywords.get(target.lower(), [])
                        for nid in nids:
                            if nid not in activated:
                                activated.append(nid)
                                self.graph.activate(nid, 0.7)
        except Exception as e:
            if getattr(self, '_trace_enabled', False):
                print(f"  [trace] S1/S2 settling error: {e}")

        # VSA Working Memory context gating & storage
        try:
            if subject:
                subj_nids = self._concept_keywords.get(subject.lower(), [])
                if subj_nids:
                    subj_node = self.graph.get_node(subj_nids[0])
                    if subj_node and subj_node.vector is not None:
                        if getattr(self, '_context_vector', None) is not None:
                            self.working_memory.set_context(self._context_vector)
                        self.working_memory.push(subj_node.vector, tag="subject")
        except Exception as e:
            if getattr(self, '_trace_enabled', False):
                print(f"  [trace] Working memory push error: {e}")

        associations = self._spread_and_collect(
            activated, primary_ids=subject_ids,
            relation_preference=spread_pref)

        # Filter associations to only contain nouns (and not grammatical/function words)
        filtered_associations = []
        for l, s in associations:
            ll = l.lower()
            if ll in self._GRAMMATICAL_CONCEPTS:
                continue
            pos = getattr(self, '_concept_pos', {}).get(ll, 'noun')
            if pos != 'noun':
                continue
            filtered_associations.append((l, s))
        associations = filtered_associations

        # Phase 2: Predictive coding update on activated nodes
        try:
            if activated and hasattr(self, '_current_context_vector') and self._current_context_vector is not None:
                for nid in activated[:10]:
                    node = self.graph.get_node(nid)
                    if node and node.vector is not None:
                        self.predictive_coding_learner.learn_node(
                            nid, self._current_context_vector, node.vector)
        except Exception as e:
            if getattr(self, '_trace_enabled', False):
                print(f"  [trace] PC learning error: {e}")

        # Step 4: Collect unknown words for deferred web learning
        # Phase 1.4: No hard lifetime cap — per-session rate limit (max 1 search per 3 turns)
        input_words = [w.strip(".,!?") for w in user_input.lower().split()
                      if len(w.strip(".,!?")) >= 3]
        known_words = sum(1 for w in input_words if w in self._concept_keywords)
        unknown_meaningful = [w for w in input_words
                              if w not in self._concept_keywords and w not in STOP_WORDS]
        # Phase 1.3: Collect unknown words into queue instead of searching synchronously
        if unknown_meaningful and self.baby_mode:
            with self._bg_lock:
                for w in unknown_meaningful:
                    if w not in self._pending_learning_queue:
                        self._pending_learning_queue.append(w)

        # Phase 11.1: Build context vector for this turn + sentence-level composition
        new_ctx = self._build_context_vector(subject) if subject else np.zeros(self.dim, dtype=np.float32)
        # Blend with decayed prior context (persistent situation model across turns)
        old_ctx = getattr(self, '_current_context_vector', None)
        if old_ctx is not None and np.any(old_ctx != 0):
            self._current_context_vector = new_ctx * 0.6 + old_ctx * 0.4
            n = np.linalg.norm(self._current_context_vector)
            if n > 0:
                self._current_context_vector /= n
        else:
            self._current_context_vector = new_ctx
        # Build sentence-level compositional vector from all input words (N400/P600)
        self._sentence_vector = self._build_sentence_vector(user_input)
        # Blend with accumulated discourse context (N400/P600 cross-turn integration)
        if hasattr(self, '_discourse_context') and self._discourse_context is not None:
            persistence = 0.6  # How much prior context persists
            self._sentence_vector = (
                persistence * self._discourse_context +
                (1.0 - persistence) * self._sentence_vector
            )
            n = np.linalg.norm(self._sentence_vector)
            if n > 0:
                self._sentence_vector /= n
        self._discourse_context = self._sentence_vector.copy() if self._sentence_vector is not None else None
        # Phase 11.4: Orthogonal content/context subspaces (PMC 2025)
        # Content = what we're talking about (sentence semantics)
        self._content_vector = self._sentence_vector.copy() if self._sentence_vector is not None else None
        # Context = how we're talking about it (discourse frame)
        raw_ctx = self._build_context_vector_from_input(user_input, subject)
        if self._content_vector is not None and np.any(self._content_vector != 0):
            self._context_vector = self._ensure_orthogonal(self._content_vector, raw_ctx)
        else:
            self._context_vector = raw_ctx
        
        # -- Situation Model Update (DMN-like continuous workspace) --
        try:
            if hasattr(self, "situation_model"):
                concept_embs = {}
                activations = {}
                for label, score in associations[:12]:
                    ll = label.lower()
                    nids = self._concept_keywords.get(ll, [])
                    if nids:
                        n = self.graph.get_node(nids[0])
                        if n and n.vector is not None:
                            concept_embs[ll] = n.vector
                            activations[ll] = max(activations.get(ll, 0), score)
                for nid in activated[:10]:
                    node = self.graph.get_node(nid)
                    if node and node.label and node.vector is not None:
                        ll = node.label.lower()
                        if ll not in concept_embs:
                            concept_embs[ll] = node.vector
                            activations[ll] = 0.5
                self.situation_model.update(
                    concept_embeddings=concept_embs,
                    activations=activations,
                    graph_get_vector_fn=self._glove_vector,
                    sentence_vector=self._sentence_vector,
                    context_vector_input=self._context_vector,
                )
        except Exception as e:
            if getattr(self, "_trace_enabled", False):
                print(f"  [trace] Situation model update error: {e}")



        # Step 5: Emotional modulation (with concept-specific tagging)
        self._update_emotion(user_input)
        for nid in activated:
            self._concept_vad[nid] = (
                self.emotion.state.valence,
                self.emotion.state.arousal,
                self.emotion.state.dominance,
            )

        # Step 5b: Update UserModel / Theory of Mind with this query
        self.user_model.observe_user_query(user_input, subject, self.emotion.state.valence)

        # Step 5c: P1 Theory of Mind — post-spread deep ToM update (roadmap §7)
        self._update_user_model(user_input, subject, associations)

        # ─── Conversational / Chit-Chat Check ───
        chitchat_response = self._handle_chitchat(user_input, subject)
        if chitchat_response:
            self._last_strategy = "chitchat"
            self._last_responses.append(chitchat_response)
            if len(self._last_responses) > 10:
                self._last_responses = self._last_responses[-10:]
            self.notify_user_idle()
            return chitchat_response

        # ─── Action / Impossible Request Check ───
        # "build me a python web scraper", "please send the email" — these are
        # requests to *do* something, not factual questions. RAVANA cannot
        # execute them, so answer honestly instead of confabulating a topic.
        action_verb = self._is_action_request(user_input)
        if action_verb:
            resp = self._handle_action_request(user_input, action_verb, subject)
            self._last_strategy = "action_request"
            self._last_responses.append(resp)
            if len(self._last_responses) > 10:
                self._last_responses = self._last_responses[-10:]
            self.notify_user_idle()
            return resp


        confidence = self.identity.state.strength * 0.5 + 0.2
        route = self.dual_process.decide_route(
            confidence=confidence,
            novelty=0.1 if associations else 0.6,
            stakes=0.15,
        )

        # Phase 1: CoherenceNetwork constraint satisfaction settling
        # Extract propositions from input and evaluate their coherence
        try:
            if subject and obj:
                input_words = [w.strip(".,!?") for w in user_input.lower().split()
                              if len(w.strip(".,!?")) >= 3]
                for w in input_words[:5]:
                    pid = f"{subject}_{w}_{self.turn_count}"
                    self.coherence_net.add_proposition(pid, initial_activation=0.1)
                    # Check existing beliefs for contradictions
                    existing = self.belief_store.query_belief(subject.lower(), w)
                    if existing is not None:
                        other_pid = f"{subject}_{w}_{self.turn_count - 1}"
                        self.coherence_net.add_proposition(other_pid, initial_activation=0.1)
                        self.coherence_net.add_contradiction(pid, other_pid, weight=-0.3)
            if self.coherence_net.propositions:
                settled = self.coherence_net.settle(max_iter=50)
                accepted = self.coherence_net.get_accepted(threshold=0.3)
                rejected = self.coherence_net.get_rejected(threshold=-0.3)
                if rejected and getattr(self, '_trace_enabled', False):
                    print(f"  [coherence] rejected {len(rejected)} propositions")
        except Exception as e:
            if getattr(self, '_trace_enabled', False):
                print(f"  [trace] CoherenceNetwork error: {e}")

        # Step 6b: Meta-cognitive assessment (if we have enough turns)
        if self.turn_count > 3 and self.turn_count % 3 == 0:
            bias_report = self.meta_cog.detect_reasoning_bias(self.turn_count)
            epistemic_mode = self.meta_cog.recommend_epistemic_mode(self.turn_count)
            # Phase 17.4: Metacognitive review every 5 turns
            if self.turn_count % 5 == 0:
                self._metacognitive_review()
        else:
            epistemic_mode = self.meta_cog.current_mode

        # Step 6c: Sleep pressure accumulation + scheduled sleep
        self._sleep_pressure += 0.02 + 0.01 * (1.0 - confidence)
        
        # Check for sleep triggers: pressure-based OR scheduled (turn-based)
        pressure_triggered = (self._sleep_pressure > 0.3 and (self.turn_count - self._last_sleep_episode) > 8)
        schedule_triggered = (self.turn_count - self._last_sleep_episode) >= self._sleep_schedule_turns
        
        if pressure_triggered or schedule_triggered:
            # Inject weak-response concepts into hippocampal replay BEFORE consolidation
            for iq in self._impossible_queries:
                if not iq.resolved and iq.subject:
                    try:
                        # Boost sleep pressure specifically for this concept
                        subj_ids = self._concept_keywords.get(iq.subject.lower(), [])
                        for nid in subj_ids:
                            node = self.graph.get_node(nid)
                            if node:
                                # Manually trigger SWS-like replay for this node
                                self.hippocampal_replay.add_experience(
                                    pair=(iq.subject, iq.subject),
                                    context=f"unknown concept: {iq.subject}",
                                    weight=2.0,  # Higher than normal (0.5 default)
                                    priority=1.0,  # Maximum priority
                                )
                    except Exception:
                        pass

            # Run a mini sleep cycle: consolidate knowledge
            metrics = self._sleep_consolidate()            # Mark impossible queries as resolved after sleep
            for iq in self._impossible_queries:
                if not iq.resolved:
                    iq.resolved = True

            self._last_sleep_episode = self.turn_count
            self._sleep_pressure = 0.0
            if self._trace_enabled and metrics:
                print(f"  [sleep] Cycle #{self._sleep_metrics['total_sleep_cycles']}: "
                      f"{metrics.get('edges_strengthened', 0)} edges strengthened, "
                      f"{metrics.get('edges_pruned', 0)} edges pruned")

        # Step 7: Past topics
        past = self._recall_past(subject, obj)

        # Phase FOK: Feeling-of-Knowing pre-check (metamemory analog)
        # Before generating, assess whether RAVANA has enough topic-specific 
        # knowledge to give a meaningful response. The brain does this via 
        # medial temporal lobe / hippocampal retrieval â€” low FOK = preparatory 
        # learning signal before the response is even generated.
        self._fok_pre_queued = False
        if subject and self.baby_mode:
            # Count strong noun associations specific to this subject
            strong_assocs = 0
            for label, score in associations[:12]:
                ll = label.lower()
                if (ll not in self._GRAMMATICAL_CONCEPTS and 
                    len(ll) >= 3 and 
                    ll != subject.lower() and
                    score > 0.2):
                    strong_assocs += 1
            
            # Count definitions and web-learned knowledge
            subj_lower = subject.lower()
            has_definition = subj_lower in self._definitions
            has_web_knowledge = subj_lower in getattr(self, '_concept_sources', {})
            has_schema = hasattr(self, 'event_schema_lib') and subj_lower in getattr(self.event_schema_lib, 'schemas', {})
            
            # RIHO model (Koriat & Levy-Sadot 2001): multi-word subjects require
            # configural integration — the brain checks if the COMBINATION of words
            # is familiar, not individual words (Reder 1987 cue-familiarity hypothesis).
            # If only individual words are known but the phrase isn't in _definitions —
            # the frontopolar cortex detects a mismatch and signals low FOK.
            phrase_known = has_definition or has_web_knowledge or has_schema
            is_multi_word = ' ' in subject.lower().strip()
            if (strong_assocs < 2 and not phrase_known) or (is_multi_word and not phrase_known):
                self._fok_pre_queued = True
                # Immediately queue this subject for learning (before response generation)
                with self._bg_lock:
                    topic = subject.lower()
                    if topic not in self._bg_learning_queue and topic not in self._pending_learning_queue:
                        self._pending_learning_queue.append(topic)
                if self._trace_enabled:
                    print(f"  [FOK] Low feeling-of-knowing for '{subject}' (assocs={strong_assocs}) â€” pre-queued learning")
                
                # LPFC pause: do synchronous web search on this subject NOW
                # instead of waiting for background learning. The brain's LPFC
                # buys time (~200-500ms) by inhibiting the prepotent generic
                # response while the hippocampus retrieves specific knowledge.
                if self.baby_mode and not self._fok_pause_done:
                    self._fok_pause_done = True
                    search_query = f"{subject} definition meaning explained"
                    if self._trace_enabled:
                        print(f"  [LPFC] Pausing generation â€” searching '{search_query}'...")
                    try:
                        self.learn_from_web(search_query, max_results=2)
                        if self._trace_enabled:
                            print(f"  [LPFC] Web search complete â€” re-activating concepts")
                        # Track recently learned concepts for dopamine novelty boost
                        # During re-spread, these will get 1.5x activation priority
                        self._recently_learned_labels.clear()
                        subj_lower = subject.lower()
                        # Add the subject itself
                        if subj_lower in self._concept_keywords:
                            self._recently_learned_labels.add(subj_lower)
                        # Add any concepts from _definitions that were just learned
                        for def_word in getattr(self, '_definitions', {}):
                            self._recently_learned_labels.add(def_word.lower())
                        # Add all concepts referenced in _concept_sources (web knowledge)
                        # These include every topic that was ever learned from the web
                        try:
                            learned_set = set()
                            for src_word in list(getattr(self, '_concept_sources', {}).keys()):
                                learned_set.add(src_word.lower())
                            # Take the most recently added ones (up to 20)
                            for lw in list(learned_set)[:20]:
                                self._recently_learned_labels.add(lw)
                        except Exception:
                            pass
                        self._recent_learn_turn = self.turn_count
                        # Directly boost activation of recently learned concepts
                        # for preferential spread (synaptic tag capture mechanism)
                        for label in self._recently_learned_labels:
                            nids = self._concept_keywords.get(label, [])
                            for nid in nids:
                                self.graph.activate(nid, 0.9)
                        # Re-activate concepts from the enriched graph
                        activated = self._activate_from_input(user_input)
                        # Re-auto-expand to wire new concepts
                        new_c = self._auto_expand_concepts(user_input)
                        if new_c > 0:
                            activated = self._activate_from_input(user_input)
                        # Re-spread activation with enriched knowledge
                        associations = self._spread_and_collect(
                            activated, primary_ids=subject_ids,
                            relation_preference=spread_pref)
                        # Re-filter associations
                        filtered = []
                        for l, s in associations:
                            ll = l.lower()
                            if ll in self._GRAMMATICAL_CONCEPTS:
                                continue
                            pos = getattr(self, '_concept_pos', {}).get(ll, 'noun')
                            if pos != 'noun':
                                continue
                            filtered.append((l, s))
                        associations = filtered
                        # Re-check FOK after learning
                        strong_assocs = sum(1 for _, sc in associations[:12] if sc > 0.2)
                        if strong_assocs >= 2:
                            self._fok_pre_queued = False  # FOK resolved!
                            if self._trace_enabled:
                                print(f"  [FOK] Knowledge acquired! {strong_assocs} associations now available")
                    except Exception as e:
                        if self._trace_enabled:
                            print(f"  [LPFC] Search failed: {e} (continuing with existing knowledge)")

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
            sentence_vector=self._sentence_vector,
            discourse_context=" | ".join(self._topic_list[-5:]) if self._topic_list else "",
            content_vector=self._content_vector,
            context_vector=self._context_vector,
            situation_vector=self.situation_model.get_blended_vector() if hasattr(self, "situation_model") else None,
            situation_narrative=self.situation_model.get_narrative_suggestions() if hasattr(self, "situation_model") else {},
        )

        # Step 11a: Store episodic memory BEFORE generating response
        # (Issue #7-8: store-before-recall fix — new facts must exist before recall check)
        try:
            if user_input and subject:
                content_words = [w.strip(".,!?") for w in user_input.lower().split()
                               if len(w.strip(".,!?")) >= 3 and w.strip(".,!?").isalpha()]
                aliases = list(content_words)
                if hasattr(self, 'user_model') and self.user_model.user_name:
                    aliases.append(self.user_model.user_name.lower())
                self.hippocampal_buffer.store(
                    subject=subject,
                    predicate="is_about",
                    object=user_input[:80],
                    confidence=0.6,
                    aliases=aliases[:10]
                )
        except Exception:
            pass

        # Acquire graph lock during generation to prevent background learning
        # from mutating graph structures during iteration. RLock is reentrant-safe.
        self._graph_lock.acquire()
        try:
            # Retry on the rare "dictionary changed size during iteration" race
            # with the background-learning thread: a live dict may be mutated
            # mid-iteration despite the lock, so re-run the turn up to 3 times.
            _attempts = 0
            while True:
                try:
                    response, strategy = self._generate_response(ctx)
                    break
                except RuntimeError as e:
                    if "dictionary changed size" in str(e) and _attempts < 3:
                        _attempts += 1
                        continue
                    raise
        finally:
            self._graph_lock.release()
        self._last_strategy = strategy

        # ─── Self-Improvement Loop: Learn from Weak Responses (ERN -> ACC -> LC-NE -> Hippocampus) ───
        quality_score = self._assess_response_quality(response, strategy, ctx)

        # Brain-inspired Syntactic & Construction Grammar Feedback Learning
        user_understood = (quality_score >= 0.55)
        if hasattr(self.syntactic_assembly, '_last_frame') and self.syntactic_assembly._last_frame:
            self.syntactic_assembly.learn_from_feedback(self.syntactic_assembly._last_frame, user_understood=user_understood)
            self.syntactic_assembly._last_frame = None
        if hasattr(self.surface_realizer, '_last_variant_name') and self.surface_realizer._last_variant_name is not None:
            self.surface_realizer.learn_from_feedback(self.surface_realizer._last_variant_name, success=user_understood)
            self.surface_realizer._last_variant_name = None

        if quality_score < 0.55 and ctx.subject and self.baby_mode:
            # Weak response detected -- boost curiosity and queue immediate learning

            # 1. Boost curiosity weight 5x-10x via the impossible queries registry
            if not any(iq.subject == ctx.subject for iq in self._impossible_queries):
                self._impossible_queries.append(FailedQuery(
                    subject=ctx.subject,
                    query=ctx.raw_input,
                    turn=self.turn_count,
                    resolved=False,
                    response_quality=quality_score,
                    strategy=strategy,
                ))

            # 2. Raise sleep pressure more aggressively (NREM tagging)
            self._sleep_pressure += 0.15 * (1.0 - quality_score)

            # 3. Queue for immediate web learning (not just background idle)
            with self._bg_lock:
                topic = ctx.subject.lower()
                if topic not in self._bg_learning_queue and topic not in self._pending_learning_queue:
                    self._pending_learning_queue.append(topic)

            # 4. Tag for higher hippocampal replay weight by forcing high prediction error
            try:
                subj_ids = self._concept_keywords.get(ctx.subject.lower(), [])
                for nid in subj_ids:
                    node = self.graph.get_node(nid)
                    if node:
                        node.prediction_free_energy = 0.8  # Force high PE -> curiosity spike
            except Exception:
                pass

            # 5. Emergency queue for background learning (wake thread immediately)
            try:
                self._queue_weak_concept_for_learning(ctx.subject, quality_score)
            except Exception:
                pass

            if self._trace_enabled:
                print(f"  [self-learn] Weak response ({quality_score:.2f}) for '{ctx.subject}' -- queued for learning")

        # ─── Correction Detection & Processing (ACC -> DA -> BG -> Hippocampus -> PFC) ───
        # Check if user is correcting RAVANA. This runs AFTER response generation.
        # The UserModel._detect_correction() was called during observe_user_query.
        # If detected, process the full 6-stage correction circuit.
        # Use the PREVIOUS turn's response as the one being corrected,
        # since the user's correction on this turn refers to what RAVANA
        # said last turn, not the response just generated.
        prev_response = self._last_responses[-1] if self._last_responses else response
        prev_strategy = getattr(self, '_last_strategy', strategy) or strategy
        self.user_model.store_response_for_correction(
            prev_response, prev_strategy, self.emotion.state.valence if hasattr(self, 'emotion') else 0.0)
        correction_ack = self._detect_and_handle_correction(
            user_input, ctx.subject, response, strategy, quality_score)
        if correction_ack:
            response = correction_ack

        # Phase 3: Register-controlled production — couple VAD + relationship
        # state into the register knobs, then apply them to the final text.
        # This is the previously-missing link: the register controller was
        # instantiated and updated by feedback, but never driven by emotion or
        # user relationship, and its apply_certainty_hedge was a no-op.
        try:
            if response:
                um = self.user_model
                rel_depth = getattr(um, "relationship_depth", 0.0)
                conv_depth = getattr(um, "conversation_depth", 0.0)
                uncer = float(getattr(ctx, "uncertainty", 0.0) or 0.0)
                self.register_controller.apply_affective_state(
                    self.emotion.state,
                    relationship_depth=rel_depth,
                    conversation_depth=conv_depth,
                    uncertainty=uncer,
                )
                conf = self.identity.state.strength * 0.5 + 0.3
                response = self.register_controller.compose(response, conf)
        except Exception:
            pass


        try:
            for hops_list in self._last_chain_hops:
                for f, t in hops_list:
                    self.hippocampal_replay.add_experience(
                        pair=(f, t), context=subject or "",
                        weight=0.5, priority=confidence)
        except Exception:
            pass

        # Phase 3: Feed prediction errors to curiosity engine
        try:
            for nid in activated[:5]:
                node = self.graph.get_node(nid)
                if node and node.label:
                    pe = getattr(node, 'prediction_free_energy', 0.0)
                    self.curiosity_engine.update_prediction_error(node.label.lower(), pe)
                    self.curiosity_engine.record_visit(node.label.lower())
        except Exception:
            pass

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
            if any(t.lower() == sl for t in self._topic_list):
                self._topic_list = [t for t in self._topic_list if t.lower() != sl]
            self._topic_list.append(subject)
        # Keep last 50 topics
        if len(self._topic_list) > 50:
            removed = self._topic_list[:-50]
            self._topic_list = self._topic_list[-50:]
            for r in removed:
                self._topic_store.pop(r.lower(), None)

        # Step 11: Skip post-response hippocampal store — already done before _generate_response
        # to ensure new facts exist before recall check (store-before-recall fix).
        # The pre-response store at Step 11a handles this. Post-response would create duplicates.
        pass
        self.hippocampal_buffer.advance_turn()

        self._store_episodic(subject, associations)

        if response is not None:
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

        # Step 12: Queue unknown words for background learning
        # Instead of rate-limited synchronous search, queue for background thread
        if self._pending_learning_queue and self._bg_learning_active:
            with self._bg_lock:
                for w in self._pending_learning_queue:
                    if w not in self._bg_learning_queue:
                        self._bg_learning_queue.append(w)
                self._pending_learning_queue.clear()
            # Background thread will wake when queue has items (see _bg_learn_loop)
        # Phase 18b: Track user query topics for curiosity priming
        if subject:
            sl = subject.lower()
            if sl != self._user_last_topic and len(sl) >= 3:
                self._user_query_topics.append(sl)
                if len(self._user_query_topics) > 10:
                    self._user_query_topics = self._user_query_topics[-10:]
                self._user_last_topic = sl

        # Phase 18: Track concept visits for curiosity/novelty scoring
        if subject:
            sl = subject.lower()
            self._concept_visit_count[sl] = self._concept_visit_count.get(sl, 0) + 1
        for label, _ in ctx.associated_concepts[:5]:
            ll = label.lower()
            self._concept_visit_count[ll] = self._concept_visit_count.get(ll, 0) + 1

        # Phase 18: Update concept learning progress from edge PE
        self._update_concept_learning_progress()

        # Phase 18: Compute curiosity urgency for autonomous exploration
        self._compute_curiosity_urgency()

        self.notify_user_idle()  # wake background thread after response

        # Post-turn context decay to prevent cross-turn bleeding
        if self._current_context_vector is not None:
            self._current_context_vector *= 0.3

        # P1 Theory of Mind: personalized greeting when relationship warrants it (roadmap §9)
        try:
            greeting = self._personalized_greeting()
            if greeting and response:
                response = greeting + response
        except Exception:
            pass  # Never break the pipeline for a greeting

        self._pending_quantity_result = None
        # NOTE: previously this returned ``response.lower()``. That destroyed
        # proper-noun casing in the final output (e.g. "France" -> "france",
        # "NASA" -> "nasa"), making RAVANA look broken. All generators already
        # produce correctly-cased text, and quality/scoring functions lowercase
        # internally where needed, so we return the response as-is.
        return response



    def _decay_episodic_edges(self):
        """Phase 15.2: Inter-turn episodic edge decay (forgetting between turns)."""
        if not hasattr(self, '_episodic_edges') or not self._episodic_edges:
            return
        for pair in list(self._episodic_edges.keys()):
            edge = self._episodic_edges[pair]
            edge.weight *= 0.95
            if edge.weight < 0.05:
                del self._episodic_edges[pair]


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
        """Activate concepts using N400/P600 sequential per-word processing.

        For each word in the input (in order):
        N400 phase: Retrieve the word's concept vector and compute prediction error
          (surprise = 1 - cosine similarity with accumulated context). High surprise
          = stronger retrieval activation (larger N400 amplitude).
        P600 phase: Integrate the retrieved meaning into the evolving sentence context.
          Propagate activation to graph neighbors, modulated by how well the word fits.

        Neuroscience basis:
        - Brouwer Retrieval-Integration theory (2012, 2017): every word elicits
          N400 (retrieval) followed by P600 (integration)
        - Nature 2024: single-neuron responses are context-dependent, not fixed
        - PMC 2023: representational dimensionality ramps across sentence
        """
        words = re.findall(r"[a-zA-Z']{1,}", text.lower())
        scores: Dict[int, float] = {}
        # Accumulated sentence context for N400 modulation
        acc_ctx = np.zeros(self.dim, dtype=np.float32)
        word_count = 0

        for w in words:
            if w in STOP_WORDS and word_count > 0:
                continue

            # === N400: Retrieve word meaning ===
            w_nids = self._concept_keywords.get(w, [])
            w_vec = None
            if w_nids:
                node = self.graph.get_node(w_nids[0])
                if node and node.vector is not None:
                    w_vec = node.vector

            # Compute N400 amplitude (surprise): how predictable is this word?
            n400_surprise = 0.5  # baseline
            if w_vec is not None and word_count > 0:
                n_acc = np.linalg.norm(acc_ctx)
                n_w = np.linalg.norm(w_vec)
                if n_acc > 1e-8 and n_w > 1e-8:
                    cos_sim = float(np.dot(acc_ctx, w_vec)) / (n_acc * n_w)
                    n400_surprise = 1.0 - max(0.0, min(1.0, cos_sim))  # 0=expected, 1=surprising
            elif word_count == 0:
                n400_surprise = 0.8  # First word is always somewhat surprising

            # Activate with N400-modulated strength
            if w_nids:
                for nid in w_nids:
                    node = self.graph.nodes.get(nid)
                    pos_boost = 0.5 if node and node.label and self._concept_pos.get(node.label.lower()) == 'noun' else 0.0
                    base = 5.0 + pos_boost
                    # N400 surprise amplifies activation for unexpected words
                    n400_boost = 1.0 + n400_surprise * 2.0
                    scores[nid] = scores.get(nid, 0) + base * n400_boost

            # Label matching (same as before)
            for nid, node in self.graph.nodes.items():
                if not node or not node.label:
                    continue
                label = node.label.lower()
                s = scores.get(nid, 0.0)
                if label == w:
                    s += 5.0 * (1.0 + n400_surprise)
                elif len(w) >= 3 and (label == w or label.startswith(w + " ") or (" " + w + " ") in label or label.endswith(" " + w)):
                    s += 3.0
                elif len(label) >= 3 and label in w:
                    s += 2.0
                if s > 0:
                    scores[nid] = s

            # === P600: Integrate into evolving sentence representation ===
            if w_vec is not None:
                # Integration: blend word vector into accumulated context
                # Gate is lower for surprising words (harder to integrate)
                integration_gate = 0.5 + 0.3 * (1.0 - n400_surprise)
                if word_count == 0:
                    acc_ctx = w_vec.copy()
                else:
                    acc_ctx = integration_gate * w_vec + (1.0 - integration_gate) * acc_ctx
                n = np.linalg.norm(acc_ctx)
                if n > 0:
                    acc_ctx /= n
            word_count += 1

            # Propagate activation to graph neighbors (P600 spread)
            if w_nids:
                for src_nid in w_nids:
                    for tgt_id, edge in self.graph.get_outgoing(src_nid):
                        if tgt_id not in scores:
                            # Weaker propagation for surprising words
                            prop_strength = 2.0 * (1.0 - n400_surprise * 0.5)
                            scores[tgt_id] = scores.get(tgt_id, 0) + edge.weight * prop_strength

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        activated = []
        # Keep more activations for richer sentence context (up to 8 vs 5)
        for nid, sc in sorted_scores[:8]:
            self.graph.activate(nid, min(1.0, sc * 0.12))
            activated.append(nid)
        return activated

    def _get_user_model(self):
        """Get or create user model for the current session."""
        if not hasattr(self, 'user_model'):
            from .user_model import UserModel
            self.user_model = UserModel()
        return self.user_model





    def _decompose_for_search(self, query: str, subject: str, assocs: List[Tuple]) -> List[str]:
        queries = [subject]
        q_lower = query.lower()
        if "blueprint" in q_lower or "create" in q_lower or "build" in q_lower or "design" in q_lower:
            queries.extend([f"{subject} design principles", f"{subject} components architecture",
                           f"how to build {subject}", f"{subject} engineering guide"])
        elif "how" in q_lower and "work" in q_lower:
            queries.extend([f"how does {subject} work", f"{subject} mechanism explained",
                           f"{subject} operating principles"])
        elif "why" in q_lower:
            queries.extend([f"why {subject} importance", f"{subject} purpose explained"])
        elif "explain" in q_lower or "detail" in q_lower or "comprehensive" in q_lower:
            queries.extend([f"{subject} explained in detail", f"{subject} comprehensive overview",
                           f"{subject} deep dive"])
        for label, _ in assocs[:3]:
            if label.lower() != subject.lower():
                queries.append(f"{subject} {label}")
        return queries

    def _reasoning_loop(self, ctx: CognitiveResponseContext) -> Tuple[str, str]:
        """Reasoning loop: web search + syntactic pipeline only.

        Stripped of the neural decoder (CE ~3.9, always produced word salad).
        Web search enriches graph knowledge, then the syntactic pipeline
        generates the response via SurfaceRealizer.
        """
        subject = ctx.subject
        query = ctx.raw_input
        assocs = ctx.associated_concepts

        subj_lower = subject.lower()
        subj_known = subj_lower in self._concept_keywords or subj_lower in self._concept_labels
        assoc_known = len(assocs) > 0

        is_complex = any(w in query.lower() for w in
                        ["how", "why", "create", "build", "design", "blueprint",
                         "explain", "detail", "comprehensive", "step by step",
                         "architecture", "implementation", "guide", "tutorial"])
        is_unknown = not subj_known or not assoc_known

        search_queries = []
        if is_complex or is_unknown:
            search_queries = self._decompose_for_search(query, subject, assocs)

        search_queries = [sq for sq in search_queries if sq][:4]

        for sq in search_queries:
            try:
                self.learn_from_web(sq)
            except Exception:
                continue

        # Syntactic pipeline only — no neural decoder generation
        try:
            syntax_response = self._generate_with_decoder_and_syntax(ctx)
            if syntax_response and len(syntax_response) > 10 and not _is_word_salad(syntax_response, subject=ctx.subject):
                _words = syntax_response.lower().split()
                _unique_ratio = len(set(_words)) / max(1, len(_words))
                if _unique_ratio >= 0.35:
                    return (syntax_response, "dorsal_reasoned")
        except Exception:
            pass

        return self._graph_fallback_response(ctx)

    # ═════════════════════════════════════════════════════════════════════════
    # Web-grounded direct answer (hippocampal fact retrieval from live search)
    # ─────────────────────────────────────────────────────────────────────────
    # When RAVANA has no stored definition for a factual query, the live search
    # engine (localhost:4000) already returns clean, answer-bearing snippets
    # (e.g. "Argentina won the 2022 FIFA World Cup..."). Earlier code only
    # stored these as loose graph associations and then emitted weak
    # association salad. This method fetches the snippets and STATES the fact
    # directly — no LLM, no templated hardcoding: we surface the best real
    # snippet (cleaned) so the answer is grounded in retrieved evidence.
    # ═════════════════════════════════════════════════════════════════════════

    # Words that frame a *conditional / hypothetical* scenario rather than naming
    # the subject itself. When a query is "if the sun disappeared…" or "suppose
    # gravity turned off…", these wrappers must be stripped before we search or
    # ground, otherwise the subject becomes junk like "sun disappeared" and the
    # web answer never lands. (No hardcoding: this is morphological cleanup of
    # the question frame, not a fact table.)
    _CONDITIONAL_FRAME = {
        "if", "suppose", "supposing", "assume", "assuming", "what", "would",
        "could", "can", "happen", "happens", "happened", "occur", "occurs",
        "occurred", "suddenly", "sudden", "instantly", "immediately", "one",
        "second", "seconds", "minute", "minutes", "day", "days", "moment",
        "were", "was", "is", "are", "be", "being", "turned", "turns", "turn",
        "switched", "switch", "gone", "vanished", "disappeared", "disappears",
        "removed", "removed", "stopped", "stops", "ceased", "shut", "off",
        "the", "a", "an", "of", "to", "on", "for", "and", "or", "but", "then",
    }

    # Sources whose snippets are reliably encyclopedic / answer-like. Used only
    # as a *tie-breaker* in snippet scoring (never a hard gate) so a great
    # answer from an unknown domain still wins over a poor one from a known one.
    _PREFERRED_SNIPPET_SOURCES = (
        "wikipedia", "britannica", "nasa", "nih", "nature", "science",
        "merriam-webster", "dictionary", "cambridge", "oxford", "britannica",
        "nasa.gov", "noaa", "smithsonian", "gov", "edu", "khan", "physics",
        "howstuffworks", "nationalgeographic", "stanford", "mit", "berkeley",
    )

    # Domains that NEVER carry a real answer for our kinds of queries — they are
    # crossword solvers, thesauri, SEO/spam, or generic UI shells. Hard-rejected
    # at the result level so their snippets can't masquerade as answers.
    # (Heuristic blocklist, not a fact table — it only drops sources that are
    # structurally incapable of answering "what is X" / "what if X".)
    _JUNK_SNIPPET_DOMAINS = (
        "crossword", "word.tips", "wordtips", "thesaurus.com", "definder",
        "elgoog", "techcrunch", "buzzfeed", "quiz", "sporcle", "puzz",
        "support.microsoft.com", "support.google.com", "support.apple.com",
        "sun-sentinel", "thesun.co.uk", "the-sun.com", "news", "reddit.com",
        "quora", "pinterest", "youtube.com", "youtu.be", "bible", "ecclesiastes",
        "linuxfoundation", "digital-trust", "azure.microsoft.com",
    )

    # Regex of sentence *shapes* that are navigation / meta / boilerplate, NOT
    # answers. These are targeted at specific junk patterns observed in the
    # search results (definder's meta-text, crossword UI, news shells). They are
    # deliberately narrow so a real dictionary definition ("The meaning of TRUST
    # is assured reliance…") is NOT rejected.
    _SNIPPET_REJECT_SHAPES = (
        r"^(what does .* mean in text|"
        r"definition of .* is where open source|"
        r"get the latest|sign in to your|applies to|"
        r"sun synonyms, sun pronunciation|"
        r"how to use .* in a sentence|"
        r"crossword solver)",
    )

    @staticmethod
    def _domain_of(url: str) -> str:
        from urllib.parse import urlparse
        try:
            net = urlparse(url).netloc.lower()
            return net[4:] if net.startswith("www.") else net
        except Exception:
            return (url or "").lower()

    def _is_conditional_query(self, text: str) -> bool:
        """Detect a hypothetical / counterfactual scenario ('if X happened…').

        Heuristic, not a fact table: any of the conditional lead-ins
        (if / suppose / assume / what if / what would happen) marks a scenario
        the user wants *reasoned out*, ideally from the web — not a generic
        reflective 'what does it mean to you?' turn.
        """
        t = text.lower().strip(" ?!.")
        if re.search(r"\b(if|suppose|supposing|assume|assuming|what if|"
                     r"what would happen|what happens if|imagine if|"
                     r"pretend that|in a world without)\b", t):
            return True
        return False

    def _clean_scenario_subject(self, subject: str, raw_input: str) -> str:
        """Reduce a conditional-query subject to the real concept being asked about.

        'sun disappeared' -> 'sun', 'gravity turned off' -> 'gravity'.
        Done by keeping only the noun-ish content words that are NOT part of the
        conditional frame, preferring known graph concepts when present. This is
        pure token filtering, so it never invents or hardcodes an answer.
        """
        subj = subject.lower().strip()
        if not subj:
            return subject
        # If the raw query is conditional, prefer the cleaned scenario as subject.
        if self._is_conditional_query(raw_input):
            words = [w.strip(".,!?") for w in raw_input.lower().split()
                     if w.strip(".,!?") not in self._CONDITIONAL_FRAME
                     and w.strip(".,!?") not in STOP_WORDS
                     and len(w.strip(".,!?")) >= 2]
            # Prefer a known graph concept among the remaining words (e.g. 'sun',
            # 'gravity'); otherwise use the longest remaining content word.
            known = [w for w in words if w in self._concept_keywords
                      or w in self._concept_labels]
            if known:
                # pick the most 'central' known concept: first that isn't a
                # generic relation word
                for w in words:
                    if w in known and w not in ("happen", "what", "would"):
                        return w
                return known[0]
            if words:
                # drop trailing auxiliaries / light verbs, keep the head noun
                for w in reversed(words):
                    if w in ("happen", "would", "could", "do", "does"):
                        continue
                    return w
                return words[0]
        # Non-conditional: still strip a trailing frame word if the whole subject
        # is just 'sun disappeared' style junk.
        parts = [w for w in subj.split()
                 if w not in self._CONDITIONAL_FRAME]
        if parts:
            return " ".join(parts)
        return subject

    def _rewrite_query_for_web(self, raw_input: str, subject: str) -> str:
        """Rewrite a user query into a search string that returns real answers.

        Problems this fixes (observed against localhost:4000):
          - 'what would happen if the sun disappeared' -> junk (Bluetooth fix).
            Better: 'sun disappeared what would happen'.
          - 'what is trust' -> 'Linux Foundation Digital Trust' junk.
            Better: 'trust definition' (dictionary sources win).
        Strategy (heuristic, no LLM):
          1. Conditional scenario -> drop the 'if/suppose/what would happen'
             frame, keep the scenario phrase, append 'what would happen'.
          2. Plain informational 'what is X'/'who is X' -> 'X definition' for
             single-word subjects so encyclopedic/dictionary hits surface.
          3. Otherwise keep the raw query.
        """
        t = raw_input.lower().strip(" ?!.")
        subj = subject.lower().strip()
        if self._is_conditional_query(raw_input):
            scenario_words = [w.strip(".,!?") for w in raw_input.lower().split()
                              if w.strip(".,!?") not in self._CONDITIONAL_FRAME
                              and w.strip(".,!?") not in STOP_WORDS
                              and len(w.strip(".,!?")) >= 2
                              # drop vague time/duration words that add no signal
                              and w.strip(".,!?") not in (
                                  "suddenly", "instantly", "immediately", "one",
                                  "second", "seconds", "minute", "minutes", "moment",
                                  "would", "could", "what", "happen", "happens",
                                  "there", "then", "away")]
            scenario = " ".join(scenario_words)
            if scenario:
                return f"{scenario} what would happen"
            return raw_input
        # Informational single-word definition queries: bias toward dictionary.
        if re.match(r"^(what|who|which)\s+(is|are|was|were)\b", t) and " " not in subj:
            return f"{subj} definition"
        # Abstract single-word concept ("trust", "love", "freedom"): dictionary
        # + "meaning" phrasing surfaces concept definitions rather than entities
        # named with the same word (e.g. "Linux Foundation Digital Trust").
        if " " not in subj and re.match(r"^(what|who|which|how)\b", t):
            return f"{subj} meaning"
        return raw_input

    def _web_query_variants(self, query: str, subject: str, is_conditional: bool):
        """Generate candidate search queries, best first.

        Heuristic, no LLM. For conditional scenarios we try several framings
        empirically known to surface *hypothetical* content on the local engine
        (the raw frame and a single rewrite often return junk because the engine
        ranks encyclopedic 'what is X' pages above 'what if X' reasoning).
        """
        variants = []
        if is_conditional:
            subj = subject
            scenario_words = [w.strip(".,!?") for w in query.lower().split()
                              if w.strip(".,!?") not in self._CONDITIONAL_FRAME
                              and w.strip(".,!?") not in STOP_WORDS
                              and len(w.strip(".,!?")) >= 2
                              and w.strip(".,!?") not in (
                                  "suddenly", "instantly", "immediately", "one",
                                  "second", "seconds", "minute", "minutes", "moment",
                                  "would", "could", "what", "happen", "happens",
                                  "there", "then", "away")]
            scenario = " ".join(scenario_words)
            if scenario:
                # Empirically, the local semantic engine answers these framings
                # with real hypothetical content (NASA-style reasoning) far more
                # reliably than the raw "what happens if X" frame, which it often
                # mismatches to topically-distant pages (e.g. "what happens if…"
                # → death articles). Order them FIRST so the loop prefers them.
                variants.append(f"if {scenario} ceased to exist")
                variants.append(f"{scenario} suddenly disappears what happens")
                variants.append(f"what if {scenario} stopped")
                variants.append(f"what happens if {scenario}")
                variants.append(f"{scenario} what would happen to earth")
                variants.append(f"if {scenario} earth effects")
                variants.append(f"what would happen to earth if {scenario}")
        # Generic fallbacks: the rewritten query, then raw, then bare subject.
        primary = self._rewrite_query_for_web(query, subject)
        for v in (primary, query, subject):
            if v and v not in variants:
                variants.append(v)
        # de-dup preserving order
        seen = set()
        out = []
        for v in variants:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    _SNIPPET_NOISE = (
        "from wikipedia", "from wikimedia", "wikiwand", "britannica",
        "redirected from", "jump to", "citation needed", "edit source",
        "view source", "listen to this article", "this article is about",
        "for other uses", "this page is about", "retrieved", "archived",
        "©", "all rights reserved", "privacy policy",
        # Boilerplate / navigation that ships inside search snippets
        "sign up", "subscribe", "newsletter", "photograph:", "photo:",
        "download the app", "cookie", "our site", "terms of service",
        "advertisement", "sponsored", "watch live", "listen now",
        "follow us", "more from", "get the", "app store", "google play",
        # News/aggregator boilerplate that is not an answer
        "recap", "bracket", "tactics", "highlights", "box score",
        "live updates", "watch:", "read more", "see also", "related:",
        "full coverage", "breaking:", "trending", "latest news",
        # HTML / photo-credit fragments that leak through the search API
        "<img", "getty", "ap images", "alt=", "loading=", "data-nimg",
        "border-top", "border-radius", "stuart", "buda mendes",
        # Wikipedia/encyclopedia language-list & navigation junk
        "toggle the table of contents", "table of contents",
        "afrikaans", "español", "العربية", "日本語", "繁體", "한국어",
        "47 languages", "languages", "read edit", "view history",
        # arXiv / listing / aggregator pages that aren't answers
        "arxiv", "recent submissions", "authors and titles", "showing up to",
        "entries per page", "see today's", "total of", "rss feed",
        # non-English / discussion-page navigation junk (e.g. Czech Wikipedia
        # "Diskuse"/"Přidat jazyky" boilerplate that outranks the real article)
        "diskuse", "přidat", "obsah stránky", "diskuze", "stránky",
        "přispět", "talk page", "discussion page", "not supported in other languages",
        "cs.wik", "de.wik", "fr.wik", "es.wik", "ru.wik", "pl.wik",
        # Promotional / SEO blurbs that aren't real answers
        "discover everything", "everything there is to know", "let me know if you",
        "let us know", "book now", "sign up", "subscribe to", "read more about",
        "find out more", "learn more about", "all you need to know", "click here",
    )

    # Irregular verb forms mapped to their base (for snippet-subject matching,
    # so "sink" in the query matches "sank"/"sunk" in a snippet, etc.).
    _IRREGULAR_VERBS = {
        "sank": "sink", "sunk": "sink", "sung": "sing", "sang": "sing",
        "rang": "ring", "rung": "ring", "began": "begin", "begun": "begin",
        "drank": "drink", "drunk": "drink", "swam": "swim", "swum": "swim",
        "ran": "run", "came": "come", "became": "become", "found": "find",
        "held": "hold", "told": "tell", "sold": "sell", "got": "get",
        "sat": "sit", "met": "meet", "led": "lead", "ate": "eat",
        "gave": "give", "took": "take", "made": "make", "saw": "see",
        "went": "go", "did": "do", "had": "have", "knew": "know",
        "grew": "grow", "threw": "throw", "drew": "draw", "fell": "fall",
        "broke": "break", "spoke": "speak", "wore": "wear", "wrote": "write",
        "rose": "rise", "drove": "drive", "flew": "fly", "froze": "freeze",
        "chose": "choose", "hid": "hide", "bit": "bite", "lit": "light",
        "built": "build", "felt": "feel", "kept": "keep", "left": "leave",
        "meant": "mean", "paid": "pay", "said": "say", "sent": "send",
        "slept": "sleep", "spent": "spend", "stood": "stand", "taught": "teach",
        "thought": "think", "understood": "understand", "won": "win",
        "caught": "catch", "bought": "buy", "brought": "bring", "fought": "fight",
        "lost": "lose", "put": "put", "set": "set", "shut": "shut",
        "cut": "cut", "hit": "hit", "read": "read", "burnt": "burn",
        "dreamt": "dream", "learnt": "learn", "spelt": "spell", "smelt": "smell",
        "spoilt": "spoil", "told": "tell", "dealt": "deal", "meant": "mean",
    }

    @staticmethod
    def _norm_word(w: str) -> str:
        """Reduce a word to a comparable base: irregular-verb map, then strip
        common inflectional suffixes."""
        w = w.lower()
        if w in CognitiveChatEngine._IRREGULAR_VERBS:
            return CognitiveChatEngine._IRREGULAR_VERBS[w]
        for suf in ("ing", "ed", "es", "s", "er", "est"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                return w[: -len(suf)]
        return w

    @staticmethod
    def _tok_match(token: str, wordset) -> bool:
        """Does `token` (a subject/query word) appear in `wordset` allowing for
        verb inflection (sink↔sank, train↔trained, immune↔immunity, ...)?"""
        t = CognitiveChatEngine._norm_word(token)
        for w in wordset:
            if w == token or w == t:
                return True
            nw = CognitiveChatEngine._norm_word(w)
            if nw == t:
                return True
            # prefix/root overlap for partial forms (immunity ~ immune)
            if len(t) >= 4 and (nw.startswith(t) or t.startswith(nw)):
                return True
        return False

    def _clean_snippet(self, text: str) -> str:
        """Strip wiki/markup noise and reduce a snippet to a clean statement."""
        if not text:
            return ""
        # Remove reference markers like [1], [23], [edit]
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\[edit\]", "", text, flags=re.IGNORECASE)
        # Remove markdown-ish wiki artifacts
        text = re.sub(r"\{\{[^}]*\}\}", "", text)
        text = re.sub(r"<[^>]+>", "", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _strip_title_echo(self, text: str, subject: str) -> str:
        """Remove a redundant leading title echo from a snippet.

        Search snippets sometimes arrive as 'Capital of Japan The capital of
        Japan is Tokyo.' — the article title is echoed before the real sentence.
        If the subject word appears (case-insensitively) more than once, keep
        from the word that starts the *second* mention (backing up to a
        preceding capitalized word, e.g. 'The'). Heuristic, no LLM."""
        if not subject or not text:
            return text
        subj0 = subject.lower().split()[0]
        matches = [(m.start(), m.group(0)) for m in re.finditer(r"\b" + re.escape(subj0) + r"\b", text, flags=re.IGNORECASE)]
        if len(matches) < 2:
            return text
        second_pos = matches[1][0]
        # Find word boundaries around the second occurrence and back up to a
        # preceding capitalized word (within 6 words) to preserve 'The ...'.
        words = re.findall(r"\S+", text)
        cursor = 0
        second_word_idx = None
        for i, w in enumerate(words):
            wstart = cursor
            wend = cursor + len(w)
            if wstart <= second_pos <= wend:
                second_word_idx = i
                break
            cursor = wend + 1
        if second_word_idx is None:
            return text
        start_idx = second_word_idx
        for j in range(second_word_idx - 1, max(-1, second_word_idx - 7), -1):
            if words[j] and words[j][0].isupper():
                start_idx = j
                break
        trimmed = " ".join(words[start_idx:])
        if len(trimmed.strip()) >= 15:
            return trimmed.strip()
        return text

    def _best_answer_snippet(self, results, subject: str, query: str,
                             is_conditional: bool = False) -> Optional[str]:
        """Pick the most answer-like snippet for a factual query.

        Heuristic (no LLM): prefer a snippet that is (a) a complete sentence,
        (b) mentions the subject or a salient query keyword, (c) reasonably
        concise (40-320 chars), and (d) free of boilerplate noise.
        """
        if not results:
            return None
        subj = (subject or "").lower()
        qkw = set(w for w in query.lower().split()
                  if len(w) > 3 and w not in STOP_WORDS)
        is_conditional = is_conditional or self._is_conditional_query(query)
        candidates = []
        for r in results[:6]:
            content = r.get("content", "") or ""
            title = r.get("title", "") or ""
            # Hard-reject sources that are structurally incapable of answering
            # our query types (crossword solvers, thesauri, spam, UI shells).
            # This is what previously let "The Sun Two Speed" crossword and
            # "How to use trust in a sentence" beat Merriam-Webster. The list is
            # a heuristic domain blocklist, NOT a fact table.
            _dom = self._domain_of(r.get("url", "") or "")
            if any(j in _dom for j in self._JUNK_SNIPPET_DOMAINS):
                continue
            # Skip results that are mostly HTML / CSS fragments (whole-result
            # junk). Photo-credit words like "getty" are handled at the
            # sentence level below, NOT here — otherwise we'd throw away a good
            # article that merely contains a "© ... via Getty Images" credit.
            raw_low = content.lower()
            # Skip non-English / discussion-page junk (e.g. a Czech Wikipedia
            # "Diskuse" page sneaks in with mostly non-ASCII navigation text).
            if content and sum(1 for c in content if ord(c) > 127) / max(1, len(content)) > 0.35:
                continue
            if ("<img" in raw_low or raw_low.count("<") > 3
                    or "{" in content or "}" in content or "url(" in raw_low
                    or "@font" in raw_low or "src:" in raw_low):
                continue
            blob = self._clean_snippet(content)
            if not blob:
                continue
            # Reject blobs that are largely non-English (discussion/nav pages in
            # other scripts slip past the whole-result check when the raw
            # content mixes ASCII boilerplate with a non-ASCII lead sentence).
            if sum(1 for c in blob if ord(c) > 127) / max(1, len(blob)) > 0.20:
                continue
            low = blob.lower()
            # Subject-relevance gate: the snippet must actually be about the
            # subject. For multi-word subjects require the full phrase or all
            # tokens (so "dark matter" doesn't match a "Dark" TV-series article
            # that only contains the first token). For single-word subjects
            # require a REAL token match (not a bare substring, otherwise
            # "trust" matches "Linux Foundation Digital Trust" and "sun" matches
            # "…under the sun" Bible verses). tolerant of verb inflection.
            subj_tokens = subj.split()
            wordset = set(low.split())
            if len(subj_tokens) >= 2:
                _phrase_ok = subj in low
                _all_tokens = all(self._tok_match(t, wordset) for t in subj_tokens)
                if not (_phrase_ok or _all_tokens):
                    continue
            elif subj:
                if not self._tok_match(subj, wordset):
                    continue
            # Reject obvious navigation/boilerplate
            if any(n in low for n in self._SNIPPET_NOISE):
                # still keep if it also directly answers; otherwise skip
                if not (subj and subj in low) and not (qkw and qkw & set(low.split())):
                    continue
            # Split into sentences, keep the ones that look like statements
            sents = re.split(r"(?<=[.!?])\s+", blob)
            for s in sents:
                s = s.strip()
                if len(s) < 20 or len(s) > 400:
                    continue
                if not re.search(r"[a-z]", s):
                    continue
                # Reject headline/question fragments (they are not answers)
                if "?" in s:
                    continue
                sl = s.lower()
                # Reject truncated / incomplete sentences
                if s.rstrip().endswith(("…", "...", "…")):
                    continue
                # Strongly reject boilerplate sentences (navigation, promos).
                if any(n in sl for n in self._SNIPPET_NOISE):
                    continue
                # Reject sentences that merely *look* like content but are
                # navigation / meta ("How to use trust in a sentence",
                # "Definition of sun", "Get the latest news…").
                if any(re.match(p, sl) for p in self._SNIPPET_REJECT_SHAPES):
                    continue
                # Score: mentions subject or query keywords, prefer answer shape
                score = 0.0
                sl_words = set(sl.split())
                # Definition / answer-shape detection for factual queries: a
                # sentence that directly defines the subject ("X is the…",
                # "X refers to…", "The meaning of X is…") is the answer we want,
                # so weight it strongly and don't let a stray mention outrank it.
                _first_words = sl.split()[:4]
                _def_verb = re.search(
                    r"\b(is|are|was|were|refers to|means|denotes|describes|"
                    r"explains|consists of|is the|is a|is an|means that|"
                    r"refers)\b", sl)
                _subj0 = subj.split()[0] if subj else ""
                _subj_is_topic = bool(_subj0) and (
                    self._tok_match(_subj0, set(_first_words))
                    or _subj0 in sl.split()[:2])
                if subj and (subj in sl or self._tok_match(_subj0, sl_words)):
                    score += 2.0
                score += 0.5 * len(qkw & sl_words)
                # Answer-pattern bonus: the subject is the grammatical subject
                # of the sentence with a copula/role verb ("France is the...",
                # "Argentina won the...", "Tokyo is the capital..."). These are
                # the sentences that actually answer "who/what is X".
                if _subj_is_topic and _def_verb:
                    score += 3.0
                elif _subj_is_topic:
                    score += 1.5
                # Conditional / hypothetical queries: the user wants a reasoned
                # *consequence*, not a dictionary definition of the subject. A
                # pure definition ("Gravity is the word used to describe a
                # physical law…") must NOT outrank an actual hypothetical
                # answer ("Imagine everything… floating midair…"). So for
                # conditionals we reverse the usual bias: penalize the bare
                # subject-definition shape and strongly boost causal /
                # consequence / scenario sentences.
                if is_conditional:
                    if _subj_is_topic and _def_verb:
                        score -= 3.0
                    if re.search(r"\b(would|could|imagine|without|no longer|"
                                 r"if .* (disappear|vanish|turn|stop|gone|"
                                 r"cease|removed|turned)|plunge|drift|freeze|"
                                 r"darkness|crash|float|orbit|fall|launch|"
                                 r"expand|escape|lost|everyone|everything|"
                                 r"people|oceans|planes|earth|planet|"
                                 r"seconds?|instant|midair)\b", sl):
                        score += 4.0
                if re.match(r"^(who|what|where|when|which|how|why)\b", query.lower()) and \
                        _def_verb:
                    score += 1.0
                # Role-answer bonus for "who is X" (person/creator queries):
                # presidents, founders, painters, authors, inventors, etc.
                if re.match(r"^who\b", query.lower()) and \
                        re.search(r"\b(president|prime minister|leader|head of state|monarch|"
                                  r"king|queen|chancellor|governor|ceo|founder|creator|"
                                  r"author|director|commander|painter|artist|writer|"
                                  r"composer|inventor|scientist|discoverer|musician|"
                                  r"novelist|poet|film maker|filmmaker|designer)\b", sl):
                    score += 2.5
                # Person-name bonus for "who" queries: a capitalized proper noun
                # near the subject is usually the answer (e.g. "Leonardo da Vinci").
                if re.match(r"^who\b", query.lower()) and re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", s):
                    score += 1.5
                # Penalize list/colon fragments and overly promotional text
                if sl.endswith((":", "•", "-", "…")):
                    score -= 0.5
                # Source-quality tie-breaker (never a hard gate): encyclopedic /
                # dictionary domains tend to give cleaner definitions than a
                # random blog, so nudge their snippets up without overriding a
                # genuinely better-scoring answer from elsewhere.
                rurl = (r.get("url", "") or "").lower()
                if any(src in rurl for src in self._PREFERRED_SNIPPET_SOURCES):
                    score += 0.3
                candidates.append((score, s))
        if not candidates:
            # Fallback: scan results for the first clean, non-boilerplate
            # snippet and return its first real sentence. Skip HTML/CSS/photo
            # junk and headline-only results.
            for r0 in results[:6]:
                content = r0.get("content", "") or ""
                raw_low = content.lower()
                if ("<img" in raw_low or "getty" in raw_low or raw_low.count("<") > 3
                        or "{" in content or "}" in content or "url(" in raw_low
                        or "@font" in raw_low or "src:" in raw_low):
                    continue
                blob = self._clean_snippet(content)
                if not blob:
                    continue
                if any(n in blob.lower() for n in self._SNIPPET_NOISE):
                    continue
                # Fallback must still be about the subject
                blow = blob.lower()
                _fb_tokens = subj.split()
                if len(_fb_tokens) >= 2:
                    if not (subj in blow or all(t in blow for t in _fb_tokens)):
                        continue
                elif subj and subj not in blow:
                    continue
                first = re.split(r"(?<=[.!?])\s+", blob)[0] if blob else ""
                if first and "?" not in first and len(first) >= 20:
                    return first.strip()
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _snippet_quality(self, snippet: str, subject: str, term: str,
                         is_conditional: bool = False) -> float:
        """Heuristic quality signal for a candidate answer snippet.

        Used to pick the best snippet across multiple search-query variants.
        Pure signal, no LLM, no fact table: rewards answer shape + encyclopedic
        source, penalizes residue from junk domains / meta sentences.
        """
        if not snippet:
            return -1.0
        s = snippet.lower().strip()
        score = 1.0
        # Answer shape: subject is the topic with a copula/definition verb.
        subj0 = (subject.split()[0] if subject else "")
        def_verb = re.search(
            r"\b(is|are|was|were|refers to|means|denotes|describes|explains|"
            r"consists of|is the|is a|is an|refers)\b", s)
        first_words = s.split()[:4]
        subj_is_topic = bool(subj0) and (
            self._tok_match(subj0, set(first_words)) or subj0 in s.split()[:2])
        if subj_is_topic and def_verb:
            score += 2.0
        elif subj_is_topic:
            score += 1.0
        # Strongly reward a SUBSTANTIVE definition sentence: the subject as topic
        # with a definition verb AND a real predicate (not just a title fragment
        # like "Definition of sun noun in Oxford…Dictionary."). A substantive
        # definition has several content words after the verb.
        if subj_is_topic and def_verb and len(s.split()) >= 6:
            # Penalize title-like fragments that end in "dictionary." / "nouns" /
            # "glossary" — these are page titles, not answers.
            if re.search(r"(dictionary\.|advanced learner|glossary|noun\b.*dictionary|"
                         r"definition of .* (noun|verb|adjective)|\bapi\b)", s):
                score -= 4.0
            else:
                score += 2.0
        # Real dictionary-concept definition shape (good for abstract concepts
        # like "trust", "love"): "The meaning of X is…".
        if re.search(r"the meaning of .* is\b", s):
            score += 2.0
        # Hypothetical / causal answer shape (good for "what if X" queries).
        if re.search(r"\b(would|without|if .* disappeared|if .* vanished|"
                     r"turned off|no longer|plunge|drift|freeze|darkness|"
                     r"crash|float|orbit)\\b", s):
            score += 1.5
        # For conditional / hypothetical queries, reverse the usual definition
        # bias: a bare dictionary definition of the subject ("Earth's gravity is
        # what keeps you on the ground") must NOT outrank a real hypothetical
        # answer ("Without gravity, Earth would be flung out into space"). The
        # user asked for a *consequence*, so weight consequence sentences up and
        # the pure subject-definition down.
        if is_conditional:
            if subj_is_topic and def_verb:
                score -= 3.0
            if re.search(r"\b(would|without|imagine|no longer|if .* (disappear|"
                         r"vanish|turn|stop|gone|cease|removed|turned|lost)|"
                         r"flung|float|drift|freeze|darkness|crash|orbit|fall|"
                         r"launch|expand|escape|everyone|everything|people|"
                         r"oceans|planes|earth|planet|spaces?|midair)\b", s):
                score += 3.0
        # Penalize page-TITLE fragments (not answers): "Definition of sun noun
        # in Oxford…Dictionary.", "The Free Dictionary", etc. These contain the
        # subject but no real predicate, so they must never beat a definition.
        if re.search(r"(dictionary\.|advanced learner|learner's dictionary|"
                     r"definition of .* (noun|verb|adjective)|\bglossary\b|"
                     r"\bapis?\b|the free dictionary|collins dictionary)", s):
            score -= 4.0
        # Penalize residual meta / junk phrasing.
        if any(re.match(p, s) for p in self._SNIPPET_REJECT_SHAPES):
            score -= 3.0
        if any(n in s for n in ("under the sun", "crossword", "how to use",
                                "get the latest", "sign in to your",
                                "applies to", "chapter", "verse")):
            score -= 3.0
        return score


    def _web_direct_answer(self, ctx: CognitiveResponseContext) -> Optional[Tuple[str, str]]:
        """Answer an unknown factual query directly from live web snippets.

        Returns (answer_text, strategy) or None if no usable snippet.
        """
        if not ctx.subject:
            return None
        # NOTE: Do NOT bail out just because a stored definition exists — the
        # live web snippet is fresher and often more accurate than a loosely
        # learned stored definition (and _generate_response already prefers web
        # over the stale def). Let web have its chance; _best_answer_snippet
        # returns None if the snippet doesn't actually back the claim.
        if not ctx.subject:
            return None
        query = ctx.raw_input.strip()
        if not query:
            return None
        is_conditional = self._is_conditional_query(query)
        if not (self._is_informational_query(query, ctx.subject) or is_conditional):
            return None
        variants = self._web_query_variants(query, ctx.subject, is_conditional)
        if getattr(self, '_trace_enabled', False):
            print(f"  [webans] informational query '{query}' subj='{ctx.subject}' "
                  f"variants={variants}")
        try:
            import time as _time
            _budget = 12.0  # hard wall-clock cap on the whole variant search
            _deadline = _time.time() + _budget
            best_overall = None
            best_overall_score = -1.0
            best_overall_term = None
            # Try every candidate query. Across all of them we keep the single
            # highest-VALUE snippet (junk-free + answer-shaped ranks above a
            # mere mention), so a good hypothetical answer on variant #3 beats
            # a junk first hit on variant #1.
            for term in variants:
                if _time.time() > _deadline:
                    break
                try:
                    # For conditionals the local engine (localhost:4000) is
                    # instant and reliably returns hypothetical content; skip the
                    # slower remote APIs so a hung DuckDuckGo/oxiverse call can
                    # never stall the turn for ~90s (observed empirically).
                    local_only = is_conditional
                    res = self.search_engine.search(term, max_results=6,
                                                    local_only=local_only)
                except Exception as ex:
                    if getattr(self, '_trace_enabled', False):
                        print(f"  [webans] search failed for {term!r}: {ex!r}")
                    continue
                if not res:
                    continue
                cand = self._best_answer_snippet(res, ctx.subject, query,
                                                is_conditional=is_conditional)
                if not cand:
                    continue
                # Value the candidate: penalize junk-ish residues, reward a real
                # answer shape and encyclopedic source. We don't need an exact
                # score from _best_answer_snippet; re-derive a quick quality signal.
                quality = self._snippet_quality(cand, ctx.subject, term,
                                                is_conditional=is_conditional)
                # Demote the bare-subject fallback variant: it often surfaces a
                # random page literally titled with the word (e.g. "Revocable
                # living trust") rather than a real definition, so prefer a
                # definition-framed variant of equal quality.
                if term == ctx.subject:
                    quality -= 1.0
                if getattr(self, '_trace_enabled', False):
                    print(f"  [webans] '{term}' -> {cand[:70]!r} (q={quality:.2f})")
                # Quality floor: only accept a snippet that is at least an actual
                # answer shape (substantive definition / hypothetical reasoning).
                # Junk like "Hotel Indigo hours" scores at/above the base 1.0 but
                # below this floor, so it is discarded instead of emitted.
                if quality < 1.5:
                    continue
                if quality > best_overall_score:
                    best_overall_score = quality
                    best_overall = cand
                    best_overall_term = term
            results = best_overall
        except Exception as ex:
            if getattr(self, '_trace_enabled', False):
                print(f"  [webans] search failed: {ex!r}")
            return None
        if not results:
            return None
        # We already picked the best snippet across all variants above.
        best = results
        if getattr(self, '_trace_enabled', False):
            print(f"  [webans] best snippet (via '{best_overall_term}'): {(best or 'NONE')[:80]}")
        if not best:
            return None
        best = self._strip_title_echo(best.strip(), ctx.subject)
        if not best.endswith((".", "!", "?")):
            best = best + "."
        # Light conversational close (generative, not hardcoded fact)
        closers = [
            "",
            " (that's what i found, at least.)",
            " let me know if you want more detail.",
            " hope that helps.",
        ]
        closer = ""
        if random.random() < 0.4:
            closer = random.choice(closers[1:])
        return (best + closer, "web_direct_answer")


    def _try_hippocampal_retrieval(self, ctx) -> Optional[str]:
        """Try to retrieve from hippocampal buffer for recall triggers."""
        if not ctx.subject or not self._recall_mode:
            return None
        facts = self.hippocampal_buffer.retrieve(ctx.subject)
        if not facts:
            return None
        # Return the highest-confidence fact
        best_fact = max(facts, key=lambda f: f.confidence)
        return None


    def _generate_acknowledgment(self, ctx, implicature) -> str:
        """Generate an acknowledgment response for pragmatic implicature.
        
        Replaces hardcoded template with SurfaceRealizer generative call.
        The acknowledgment is driven by free energy (uncertainty about the
        implicature), not a fixed string.
        """
        # Generative acknowledgment via SurfaceRealizer
        if ctx.subject:
            try:
                s = self._try_surface_realize(
                    subject=ctx.subject, target=ctx.subject,
                    discourse_type="reflect", free_energy=0.4, min_len=5)
                if s:
                    return s
            except Exception:
                pass
        return "that is interesting."


    def _generate_with_decoder_and_syntax(self, ctx: CognitiveResponseContext) -> Optional[str]:
        """Generate using full syntactic pipeline (P1: Production-Grade Syntactic Pipeline).

        Pipeline:
        1. PrefrontalWorkspace → Discourse Plan (structured intents from graph)
        2. SyntacticCellAssembly → Syntactic Frames (bind concepts to grammatical roles)
        3. BasalGangliaGate → Candidate Selection (Go/NoGo gating)
        4. CerebellarNgram → Fluent Completion (learned transitions)
        5. SurfaceRealizer → Final Text (morphology, agreement, punctuation)
        """
        if not ctx.subject:
            return None

        try:
            # Step 0: Seed language modules with POS info if not already seeded
            if not self.syntactic_assembly.subject_role and self._concept_pos:
                self.syntactic_assembly.seed_from_pos(self._concept_pos)
            if not self.cerebellar_ngram._pos_agreement and self._concept_pos:
                self.cerebellar_ngram.seed_from_pos(self._concept_pos)

            # Step 1: CausalSchema query for hypothetical/causal questions (Issues #2, #10)
            _causal_prediction = None
            if hasattr(self, 'causal_schema') and ctx.subject:
                qtype = self.pfc_workspace.detect_question_type(ctx.raw_input, self._concept_pos)[0]
                if qtype in ('hypothetical', 'why', 'how'):
                    # Try to predict what happens with the subject
                    pred, conf = self.causal_schema.predict(ctx.subject, 'change')
                    if pred and conf > 0.3:
                        _causal_prediction = (pred, conf)
                        # Record prediction for free-energy tracking
                        self.causal_schema.record_prediction(ctx.subject, 'change', pred, True)

            # Step 2: Check relational memory for transitive/comparison queries (Issue #9)
            _relation_result = None
            if hasattr(self, 'relation_memory') and ctx.subject:
                qtype = self.pfc_workspace.detect_question_type(ctx.raw_input, self._concept_pos)[0]
                if qtype == 'compare':
                    # Try to find comparative relations involving the subject
                    transitive_results = self.relation_memory.transitive_query(ctx.subject, 'taller')
                    if transitive_results:
                        _relation_result = transitive_results[0]

            # Step 3: Quantity comparison response (Issue #5)
            if hasattr(self, '_pending_quantity_result') and self._pending_quantity_result:
                qa, qb, q_result, q_conf = self._pending_quantity_result
                if q_result == 'equal':
                    return f"{qa.concept.capitalize()} and {qb.concept} have the same quantity. They are equal."
                elif q_result == 'a_greater':
                    return f"{qa.concept.capitalize()} has more than {qb.concept} ({qa.value} vs {qb.value})."
                elif q_result == 'b_greater':
                    return f"{qb.concept.capitalize()} has more than {qa.concept} ({qb.value} vs {qa.value})."

            # Discourse Planning
            is_follow_up = self._is_follow_up(ctx.raw_input)
            # Emotional mirror modulation: adjust associations and verbosity
            mirror_mod = self.mirror_engine.get_modulation(self.emotion.state)
            bm = mirror_mod['breadth_mult']
            max_assocs = max(2, min(10, int(round(5 * bm))))
            reduced_assocs = ctx.associated_concepts[:max_assocs]
            discourse_plan = self.pfc_workspace.plan_discourse(
                user_input=ctx.raw_input,
                subject=ctx.subject,
                concept_pos=self._concept_pos,
                associations=reduced_assocs,
                past_topics=ctx.past_topics,
                is_follow_up=is_follow_up,
            )

            vm = mirror_mod['verbosity_mult']
            target_verbosity = max(1, min(5, int(round(3 * vm))))
            if target_verbosity < len(discourse_plan.intents):
                discourse_plan.intents = discourse_plan.intents[:target_verbosity]
            elif target_verbosity > len(discourse_plan.intents):
                most_recent = self._topic_list[-1] if self._topic_list else ctx.subject
                for _ in range(target_verbosity - len(discourse_plan.intents)):
                    discourse_plan.intents.append(DiscourseIntent(
                        type=DiscourseType.ELABORATE,
                        subject=most_recent,
                        primary_relation="semantic",
                        seen_so_far=set(),
                    ))

            # Step 2-5: Build and realize each sentence from the discourse plan
            utterances = []
            
            # Inject relational memory and causal schema info into discourse context (Issues #2, #9)
            _relational_info = ""
            if hasattr(self, 'relation_memory') and ctx.subject:
                transitive_results = self.relation_memory.transitive_query(ctx.subject, 'taller')
                if transitive_results:
                    _relational_info = f" {ctx.subject} is {transitive_results[0][1]} than {transitive_results[0][0]}"

            discourse_context = DiscourseState(
                sentence_index=0,
                discourse_type=discourse_plan.question_type,
                total_sentences=len(discourse_plan.intents),
                free_energy=self._free_energy,
            )

            # Generate all frames first
            frames = []
            for intent in discourse_plan.intents:
                if not intent.target_concept:
                    continue
                relation = intent.primary_relation
                if intent.type == DiscourseType.CAUSAL_EXPLAIN:
                    relation = "causal"
                elif intent.type == DiscourseType.CONTRAST:
                    relation = "contrastive"
                
                frame = self.syntactic_assembly.bind_to_sentence(
                    subject=intent.subject,
                    relation=relation,
                    target=intent.target_concept,
                    pos_map=self._concept_pos,
                    chain_concepts=None,
                    chain_connectors=None,
                    depth=0,
                )
                frame._discourse_intent = intent
                frames.append(frame)

            # Merging / Nesting Pass (Broca's area hierarchy building)
            merged_frames = []
            skip_indices = set()
            for i in range(len(frames)):
                if i in skip_indices:
                    continue
                frame = frames[i]
                
                # Check if we can nest the next frame inside this one
                if i + 1 < len(frames) and (i + 1) not in skip_indices:
                    next_frame = frames[i + 1]
                    # If the next frame's subject is the same as the current frame's object (or very similar)
                    if (next_frame.subject_concept.lower() == frame.object_concept.lower() or 
                        frame.object_concept.lower() in next_frame.subject_concept.lower()) and next_frame.object_concept:
                        
                        # Set embedded relation
                        if next_frame.relation_type == "causal":
                            frame.embedded_relation = "because"
                        elif next_frame.relation_type == "contrastive":
                            frame.embedded_relation = "although"
                        else:
                            frame.embedded_relation = "which"
                        
                        # Prepare next_frame to be realized as a relative/nested clause
                        next_frame.pronoun_subject = ""
                        next_frame.article_subject = ""
                        next_frame.subject_concept = "" # realizes without subject prefix
                        
                        frame.embedded_frame = next_frame
                        skip_indices.add(i + 1)
                        
                merged_frames.append(frame)

            discourse_context.total_sentences = len(merged_frames)

            for i, frame in enumerate(merged_frames):
                intent = frame._discourse_intent
                relation = frame.relation_type

                # Step 3: Basal Ganglia Gating (Go/NoGo)
                candidates = [(frame.subject_concept or frame.object_concept, 1.0, 1.0, relation)]
                self.basal_ganglia.set_all_from_modulators({
                    "arousal": ctx.arousal,
                    "novelty": 0.3 if ctx.learned_recently else 0.1,
                    "exploration_drive": ctx.exploration_drive,
                    "prediction_error": 0.2,
                    "identity_strength": ctx.identity_strength,
                    "fatigue_level": 0.1,
                    "prefrontal_boost": 0.5,
                    "thalamic_salience": 0.7,
                    "subject_proximity_bonus": 0.3,
                    "contradiction_penalty": 0.3,
                    "dopamine_tone": self._dopamine_tone,
                })
                selected_label, selected_rel, go_score = self.basal_ganglia.select_concept(candidates)

                if not selected_label:
                    continue

                # Step 5: Surface Realization
                discourse_context.sentence_index = i
                discourse_context.previous_subject = utterances[-1].split()[0] if utterances else None
                discourse_context.discourse_type = intent.type

                sentence = self.surface_realizer.realize(
                    frame=frame,
                    discourse_context=discourse_context,
                    dopamine_tone=self._dopamine_tone,
                    cerebellar_ngram=self.cerebellar_ngram,
                    discourse_marker=intent.discourse_marker,
                )

                # VSA Schema Library realization integration
                try:
                    required_roles = ["subject", "verb"]
                    if frame.object_concept:
                        required_roles.append("object")
                    
                    vsa_schema = self.schema_library.select_schema(required_roles, dopamine_tone=self._dopamine_tone)
                    
                    fillers = {
                        "subject": frame.subject_concept,
                        "verb": frame.verb_phrase,
                        "object": frame.object_concept
                    }
                    
                    embeddings = {}
                    for w in fillers.values():
                        if w:
                            vec = self._decoder_word_to_embed.get(w.lower())
                            if vec is not None:
                                embeddings[w] = vec
                            else:
                                embeddings[w] = self.vsa_manager.generate_vector()
                                
                    vsa_sentence = self.schema_library.realize_sentence(vsa_schema, fillers, embeddings)
                    if vsa_sentence and getattr(self, '_trace_enabled', False):
                        print(f"  [trace] VSA Schema realized: {vsa_sentence}")
                except Exception as e:
                    if getattr(self, '_trace_enabled', False):
                        print(f"  [trace] VSA realization skipped: {e}")

                utterances.append(sentence)

                # Learn from this chain for cerebellar n-gram
                if len(utterances) > 0:
                    chain_labels = [frame.subject_concept or "it", frame.verb_phrase, frame.object_concept]
                    self.cerebellar_ngram.learn_chain(chain_labels, successful=True)
                    VerbLexicon.reinforce(relation, frame.verb_phrase, success=1.0)

            if utterances:
                return " ".join(utterances)

        except Exception as e:
            if self._trace_enabled:
                import traceback
                print(f"  [trace] syntactic pipeline error: {e}")
                traceback.print_exc()

        return None


    QUESTION_WORDS = {"what", "why", "how", "when", "where", "who", "which",
                        "does", "do", "is", "are", "can", "will", "would",
                        "could", "should", "did", "have", "has", "had"}

    # Words that signal the user is asking for more on a previous topic
    FOLLOW_UP_WORDS = {"more", "else", "another", "also", "further",
                       "other", "additionally", "favorite"}

    # Phase 3.3: Recall detection — semantic vector-based (no hardcoded patterns)
    # Uses GloVe cosine similarity to detect recall intent.
    # These are the semantic seeds — the actual matching uses vector similarity
    # so any semantically similar word (like "forgot", "previous", "previously")
    # will also trigger recall detection.
    _RECALL_SEED_CONCEPTS = [
        "remember", "recall", "recollect", "earlier", "previous",
        "said", "mentioned", "discussed", "talked", "before",
        "last", "prior", "past",
    ]
    _RECALL_DETECTION_THRESHOLD: float = 0.55  # Min cosine sim to be considered recall



    def _detect_brain_state(self) -> str:
        confidence = self.identity.state.strength * 0.5 + 0.2
        pe = getattr(self, '_mean_prediction_error', 0.3)
        novelty = 0.1 if len(self._last_responses) > 0 else 0.6
        if confidence < 0.3 or pe > 0.4 or novelty > 0.6:
            new_state = "heteromodal"
        elif confidence > 0.5 and pe < 0.2 and novelty < 0.3:
            new_state = "unimodal"
        else:
            new_state = "default"
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


    def _activate_schema(self, subject: str) -> Set[int]:
        subj_nids = self._concept_keywords.get(subject.lower(), [])
        if not subj_nids:
            return set()
        subj_nid = subj_nids[0]
        subj_node = self.graph.get_node(subj_nid)
        if subj_node is None or subj_node.vector is None:
            return set()
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


    def _build_context_vector(self, subject: str) -> np.ndarray:
        components = []
        weights = []
        subj_nids = self._concept_keywords.get(subject.lower(), [])
        if subj_nids:
            subj_node = self.graph.get_node(subj_nids[0])
            if subj_node and subj_node.vector is not None:
                v = subj_node.vector.ravel()
                if len(v) != self.dim:
                    v = np.resize(v, self.dim)
                components.append(v)
                weights.append(0.4)
        recent_vecs = []
        for resp in self._last_responses[-3:]:
            if resp is None:
                continue
            for w in resp.split():
                wn = self._concept_keywords.get(w.lower(), [])
                if wn:
                    wn_node = self.graph.get_node(wn[0])
                    if wn_node and wn_node.vector is not None:
                        v = wn_node.vector.ravel()
                        if len(v) != self.dim:
                            v = np.resize(v, self.dim)
                        recent_vecs.append(v)
        if recent_vecs:
            components.append(np.mean(recent_vecs, axis=0))
            weights.append(0.3)
        pfc_vecs = []
        for bl in self._prefrontal_buffer:
            bn = self._concept_keywords.get(bl, [])
            if bn:
                bn_node = self.graph.get_node(bn[0])
                if bn_node and bn_node.vector is not None:
                    v = bn_node.vector.ravel()
                    if len(v) != self.dim:
                        v = np.resize(v, self.dim)
                    pfc_vecs.append(v)
        if pfc_vecs:
            components.append(np.mean(pfc_vecs, axis=0))
            weights.append(0.2)
        e_vec = np.array([self.emotion.state.valence, self.emotion.state.arousal, self.emotion.state.dominance], dtype=np.float32)
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
    def _build_sentence_vector(self, user_input: str) -> np.ndarray:
        """Build compositional sentence-level vector from all input words (N400/P600)."""
        words = re.findall(r"[a-zA-Z']{2,}", user_input.lower())
        vecs = []
        for w in words:
            if w in STOP_WORDS:
                continue
            gv = self._glove_vector(w)
            if gv is not None:
                vecs.append(gv)
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)
        result = np.mean(vecs, axis=0).astype(np.float32)
        norm = np.linalg.norm(result)
        if norm > 0:
            result /= norm
        return result

    def _build_context_vector_from_input(self, user_input: str, subject: str) -> np.ndarray:
        """Build discourse context vector from input and subject."""
        sl = subject.lower()
        subj_vec = None
        if sl in self._concept_keywords:
            nids = self._concept_keywords[sl]
            if nids:
                node = self.graph.get_node(nids[0])
                if node and node.vector is not None:
                    subj_vec = node.vector.copy()
        words = re.findall(r"[a-zA-Z']{3,}", user_input.lower())
        word_vecs = []
        for w in words:
            if w in STOP_WORDS:
                continue
            gv = self._glove_vector(w)
            if gv is not None:
                word_vecs.append(gv)
        result = np.zeros(self.dim, dtype=np.float32)
        weight_sum = 0.0
        if subj_vec is not None:
            result += subj_vec * 0.4
            weight_sum += 0.4
        if word_vecs:
            ctx_mean = np.mean(word_vecs, axis=0).astype(np.float32)
            n = np.linalg.norm(ctx_mean)
            if n > 0:
                ctx_mean /= n
            result += ctx_mean * 0.4
            weight_sum += 0.4
        emotion_vec = np.zeros(self.dim, dtype=np.float32)
        emotion_vec[:3] = np.array([
            self.emotion.state.valence,
            self.emotion.state.arousal,
            self.emotion.state.dominance,
        ], dtype=np.float32)
        result += emotion_vec * 0.2
        weight_sum += 0.2
        if weight_sum > 0:
            result /= weight_sum
        norm = np.linalg.norm(result)
        if norm > 0:
            result /= norm
        return result.astype(np.float32)

    def _ensure_orthogonal(self, content_vector: np.ndarray, raw_ctx: np.ndarray) -> np.ndarray:
        """Ensure context vector is orthogonal to content vector via Gram-Schmidt."""
        c = content_vector.copy()
        r = raw_ctx.copy()
        nc = np.linalg.norm(c)
        if nc < 1e-8:
            nr = np.linalg.norm(r)
            if nr > 0:
                r /= nr
            return r.astype(np.float32)
        c /= nc
        dot = float(np.dot(r, c))
        r_orth = r - dot * c
        norm = np.linalg.norm(r_orth)
        if norm > 1e-8:
            r_orth /= norm
        else:
            rng = np.random.RandomState(42)
            rand_vec = rng.randn(self.dim).astype(np.float32)
            rand_vec -= np.dot(rand_vec, c) * c
            rn = np.linalg.norm(rand_vec)
            if rn > 0:
                rand_vec /= rn
            r_orth = rand_vec
        return r_orth.astype(np.float32)




    def _metacognitive_review(self):
        pass

    def _update_cerebellar_ngram(self, hops_list):
        chain_labels = []
        for hop in hops_list:
            if isinstance(hop, tuple) and len(hop) >= 2:
                chain_labels.append(hop[0])
                chain_labels.append(hop[1])
        self.cerebellar_ngram.learn_chain(chain_labels, successful=True, chain_hops=hops_list)

    def _assess_response_quality(self, response: str, strategy: str, ctx) -> float:
        """
        Rate the quality of the just-generated response (ERN analog).

        Returns 0.0 (terrible) to 1.0 (excellent).

        Factors:
        - Strategy used: narrative (0.7-1.0), syntax (0.4-0.6), gist_fallback (0.2-0.3), word_salad (0.0-0.1)
        - Number of unique content nouns (>=3 = good, 0-1 = weak)
        - Length: 15-60 chars sweet spot, <10 = too short
        - Has noun_assocs: 0 = very weak, >=2 = strong
        - Was schema_used: True = higher quality
        - _is_word_salad check: True = quality 0.0
        """
        if not response or not isinstance(response, str):
            return 0.0

        # Base score from strategy
        strategy_scores = {
            "situation_model_narrative": 0.60,
            "situation_model_decoder": 0.55,
            "situation_model_syntax": 0.50,
            "dorsal_reasoned": 0.45,
            "fast_ventral": 0.35,
            "graph_fallback": 0.25,
            "gist_fallback": 0.20,
            "chitchat": 0.40,
        }
        base_score = strategy_scores.get(strategy, 0.3)

        # Word salad check: immediate 0
        if _is_word_salad(response, subject=ctx.subject):
            return 0.0

        # Length penalty: too short or too long
        resp_len = len(response.strip())
        if resp_len < 8:
            length_factor = 0.0  # Too short to be meaningful
        elif resp_len < 15:
            length_factor = 0.3  # Short but acceptable
        elif resp_len <= 60:
            length_factor = 1.0  # Sweet spot
        elif resp_len <= 120:
            length_factor = 0.8
        else:
            length_factor = 0.5  # Too long, likely rambling

        # Content noun diversity
        words = re.findall(r"[a-zA-Z']{3,}", response.lower())
        content_words = [w for w in words if w not in STOP_WORDS and len(w) >= 3]
        unique_content = len(set(content_words))
        if unique_content >= 4:
            content_factor = 1.0
        elif unique_content >= 3:
            content_factor = 0.7
        elif unique_content >= 2:
            content_factor = 0.4
        elif unique_content >= 1:
            content_factor = 0.15
        else:
            content_factor = 0.0

        # Association richness from context
        noun_assocs = 0
        if ctx and hasattr(ctx, 'associated_concepts') and ctx.associated_concepts:
            for label, _ in ctx.associated_concepts:
                ll = label.lower()
                if ll not in self._GRAMMATICAL_CONCEPTS:
                    noun_assocs += 1
        assoc_factor = min(1.0, noun_assocs / 4.0)  # 4+ assocs = max

        # Knowledge-density penalty: penalize generic responses that lack subject-specific content
        knowledge_density_penalty = 0.0
        if ctx and hasattr(ctx, 'subject') and ctx.subject:
            subj = ctx.subject.lower()
            # Use only the top (strongest) associations â€” not all of them
            assoc_labels = {a[0].lower() for a in getattr(ctx, 'associated_concepts', [])[:8]}
            subject_related = sum(1 for w in content_words if w == subj or w in assoc_labels)
            total = max(len(content_words), 1)
            specificity = subject_related / total
            if specificity < 0.15:
                knowledge_density_penalty = 0.18
            elif specificity < 0.3:
                knowledge_density_penalty = 0.10

        # Filler-word penalty: responses heavy on stop words signal low information density
        total_words = max(len(words), 1)
        stop_ratio = len([w for w in words if w in STOP_WORDS]) / total_words
        filler_penalty = 0.1 if stop_ratio > 0.6 else 0.0

        # Schema usage bonus
        schema_bonus = 0.05 if strategy and ('schema' in strategy.lower() or 'narrative' in strategy.lower()) else 0.0

        # â”€â”€ Specificity-based template detection (Hippocampal specificity signal) â”€â”€
        # The brain detects generic responses by checking for the absence of specific
        # episodic details from the posterior hippocampus (Ramey 2022, MasÃ­s-Obando 2022).
        # If no specific memory trace is retrieved, the response defaults to the
        # prevailing schematic pattern stored in the neocortex (DMN).
        # In RAVANA: check if the response references any web-learned knowledge
        # (_definitions, _concept_sources) vs. only generic GloVe words.
        template_penalty = 0.0
        if ctx and hasattr(ctx, 'subject') and ctx.subject and len(response) > 15:
            subj = ctx.subject.lower()
            # Build the set of 'specific knowledge' words â€” words that RAVANA
            # has actually learned from web searches (not just GloVe neighbors)
            # This is the hippocampal specificity signal: has the hippocampus
            # contributed any unique details to this response?
            knowledge_words = set()
            # 1. Definitions learned from web
            for def_word in getattr(self, '_definitions', {}):
                knowledge_words.add(def_word.lower())
            # 2. Web-sourced concepts (from _concept_sources)
            for src_word in getattr(self, '_concept_sources', {}):
                knowledge_words.add(src_word.lower())
            # 3. Subject itself is always 'known'
            knowledge_words.add(subj)
            
            # Extract content words from the response
            resp_words_lower = response.lower().split()
            resp_content = {w.strip(".,!?") for w in resp_words_lower 
                          if len(w.strip(".,!?")) >= 3 
                          and w.strip(".,!?") not in STOP_WORDS}
            
            # Count how many response content words are 'specific knowledge'
            # vs. generic structural words that appear in any template response
            generic_structural = {'begins', 'unfolds', 'grows', 'deepens', 'emerges',
                'drives', 'shapes', 'follows', 'parallels', 'relates',
                'journey', 'naturally', 'deliberately', 'gradually',
                'connected', 'similar', 'leads', 'gives', 'rises', 'akin',
                'ultimately', 'spark', 'recognition', 'interest', 'vulnerability',
                'shared', 'experiences', 'something', 'different', 'direction',
                'pursue', 'desire', 'observation', 'study', 'gathered',
                'begins', 'unfolds', 'transforms', 'evolves'}
            
            specific_words = resp_content - generic_structural
            total_content = max(len(resp_content), 1)
            specificity_ratio = len(specific_words) / total_content
            
            # Also check: does the subject itself have any web knowledge?
            subj_has_knowledge = (subj in getattr(self, '_definitions', {}) or 
                                subj in getattr(self, '_concept_sources', {}) or
                                subj in getattr(self, '_recently_learned_labels', set()))
            
            # Penalty logic:
            # - If the subject HAS web knowledge but the response doesn't use it â†’ heavy penalty
            # - If the subject has NO web knowledge AND response is generic â†’ moderate penalty
            # - If specificity ratio is very low â†’ moderate penalty
            if subj_has_knowledge and specificity_ratio < 0.3:
                template_penalty = 0.20  # Has knowledge but didn't use it!
                if getattr(self, '_trace_enabled', False):
                    print(f"  [spec] '{subj}' has web knowledge but response is generic (ratio={specificity_ratio:.2f})")
            elif specificity_ratio < 0.2:
                template_penalty = 0.12  # Very low specificity
                if getattr(self, '_trace_enabled', False) and False:  # Noisy, keep quiet
                    pass

        # FOK note: if we pre-queued learning (brain knows it doesn't know),
        # the weak response is expected â€” learning is already queued,
        # no additional penalty needed beyond the template check

        # Combine: weighted average with penalties
        quality = (
            base_score * 0.35 +
            length_factor * 0.15 +
            content_factor * 0.25 +
            assoc_factor * 0.25 +
            schema_bonus -
            knowledge_density_penalty -
            filler_penalty -
            template_penalty
        )

        # Trace logging for quality score â€” shows even when loop doesn't trigger
        if getattr(self, '_trace_enabled', False):
            subj_name = ctx.subject if ctx and hasattr(ctx, 'subject') else '?'
            spec_info = f", spec_pen={template_penalty:.2f}" if template_penalty > 0 else ""
            fok_info = " [FOK]" if getattr(self, '_fok_pre_queued', False) else ""
            lpfc_info = " [LPFC]" if getattr(self, '_fok_pause_done', False) else ""
            print(f"  [trace]   quality_score={quality:.2f} for '{subj_name}'{fok_info}{lpfc_info} "
                  f"(strategy={strategy}, content={unique_content}, assoc={noun_assocs}, "
                  f"len={resp_len}, kd_pen={knowledge_density_penalty:.2f}, "
                  f"fill_pen={filler_penalty:.2f}{spec_info})")

        return max(0.0, min(1.0, quality))


    def _sleep_consolidate(self) -> Dict[str, int]:
        result = self.sleep_engine.run_cycle(
            graph=self.graph,
            episodic_buffer=[],
            episodic_triples=self.plasticity._episodic_triples if hasattr(self, 'plasticity') else [],
            belief_store=self.belief_store,
            topic_list=self._topic_list,
            user_model=self._get_user_model(),
            impossible_queries=(self.web_learner._impossible_queries if hasattr(self, 'web_learner') and hasattr(self.web_learner, '_impossible_queries') else []),
            contradiction_map=self._contradiction_map,
            drift_defense_threshold=0.7,
            drift_pull=0.05,
            concept_vad=self._concept_vad if hasattr(self, '_concept_vad') else None,
        )
        self.sleep_cycles_completed += 1
        # Phase 3: Hippocampal replay consolidation
        try:
            replay_metrics = self.hippocampal_replay.sleep_cycle(
                replay_count=100, interleave_count=50, prune_threshold=0.1)
            result['hippocampal_replays'] = replay_metrics.get('nrem_replays', 0)
            result['hippocampal_pruned'] = replay_metrics.get('pruned', 0)
        except Exception as e:
            if getattr(self, '_trace_enabled', False):
                print(f"  [trace] Hippocampal replay error: {e}")
        # Phase 5: Consolidate corrections from the correction log
        try:
            correction_metrics = self._consolidate_corrections_in_sleep()
            result.update(correction_metrics)
        except Exception as e:
            if getattr(self, '_trace_enabled', False):
                print(f"  [trace] Correction consolidation error: {e}")
        return result

    def _update_state(self, ctx: CognitiveResponseContext):
        """Update cognitive state post-response: free energy, meaning, identity, global workspace."""
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
            resolution_streak=sum(1 for r in self._last_responses if r is not None and len(r) > 20),
            correctness=correct,
        )
        self.identity.apply_update(new_s)
        if self.identity.state.strength > 1.0:
            self.identity.state.strength = 1.0
        if self.identity.state.strength < 0.0:
            self.identity.state.strength = 0.0
        self.gw.submit_bid(source="dialogue",
            payload={"subject": ctx.subject, "turn": self.turn_count},
            urgency=0.3 + 0.15 * min(len(ctx.associated_concepts), 4) / 4.0,
            valence=self.emotion.state.valence, episode=self.turn_count)
        self.gw.compete()


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

    # Generic/vague nouns that should never be returned as the primary subject
    # on their own ("system", "process", "thing", "matter", ...). When grounding
    # lands on one of these, prefer a more specific co-occurring word or the
    # full multi-word phrase so web grounding stays on-topic.
    _GENERIC_NOUNS = {
        "system", "systems", "process", "processes", "thing", "things",
        "matter", "stuff", "concept", "concepts", "idea", "ideas",
        "object", "objects", "item", "items", "person", "people",
        "place", "places", "world", "universe", "life", "reason",
        "fact", "facts", "way", "ways", "kind", "kinds", "type", "types",
        "form", "forms", "level", "levels", "part", "parts", "state", "states",
        "effect", "effects", "result", "results", "change", "changes",
        "point", "points", "number", "numbers", "word", "words",
        "language", "thought", "thoughts", "time", "question", "questions",
        "answer", "answers", "problem", "problems", "method", "methods",
    }

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
                        "explain", "describe", "define", "discuss", "show", "list",
                        "write", "read", "learn", "compare", "contrast", "introduce",
                        "suggest", "recommend",
                        # Generic adjectives & filler that make poor conversation topics
                        "good", "bad", "big", "small", "always", "never", "maybe",
                        "if", "but", "in", "out", "up", "down",
                        "point", "way", "thing", "stuff",
                        "and", "so"}

    # Words that should never become the conversation TOPIC. They are either
    # question/verb/sentence glue or generic "role" words whose object is the
    # real subject ("the president OF france" -> france, "what is the MEANING
    # OF life" -> life, "what HAPPENED IN 1923" -> 1923). Stripping these from
    # the extracted subject phrase stops RAVANA from answering a malformed
    # topic like "president france" or "happened 1923" and producing abstract,
    # ungrammatical replies.
    _SUBJECT_CONTEXT_WORDS = {
        # verbs / sentence glue that leak in from question parsing
        "happen", "happened", "happening", "occur", "occurred", "occur",
        "mean", "means", "meaning", "build", "builds", "building",
        "make", "makes", "made", "create", "creates", "created",
        "do", "does", "did", "done", "get", "gets", "got", "go", "goes",
        "went", "use", "uses", "used", "write", "writes", "wrote",
        "explain", "explains", "describe", "describes", "tell", "tells",
        "show", "shows", "give", "gives", "find", "finds", "help", "helps",
        "know", "knows", "think", "thinks", "feel", "feels", "want", "wants",
        "need", "needs", "like", "likes", "love", "loves", "hate", "hates",
        "become", "becomes", "became", "call", "calls", "called", "name",
        "named", "term", "termed", "say", "says", "said",
        # query-intent verbs whose object is the real topic
        "cause", "causes", "caused", "brew", "brews", "brewed",
        "teach", "teaches", "taught", "train", "trains", "trained",
        "sing", "sings", "sang", "sung",
        "learn", "learns", "study", "studies", "read", "reads",
        "write", "writes", "cook", "cooks", "bake", "bakes", "play", "plays",
        "draw", "draws", "make", "makes", "find", "finds", "get", "gets",
        "give", "gives", "show", "shows", "explain", "explains",
        "describe", "describes", "tell", "tells", "understand", "understands",
        "avoid", "prevent", "prevents", "stop", "stops", "keep", "keeps",
        "stay", "stays", "remain", "remains", "become", "becomes",
        "turn", "turns", "turned", "switch", "switches", "switched",
        "open", "opens", "close", "closes", "start", "starts", "stop", "stops",
        # question-frame residuals: "what YEAR did X fall/occur", "when did X happen"
        "year", "years", "fall", "falls", "fell", "occur", "occurs", "occurred",
        "did", "does", "do", "take", "takes", "took", "place", "happen",
        "happened", "happening", "become", "became", "mean", "means",
        # conditional / hypothetical markers whose payload is the real topic
        "suppose", "supposing", "assume", "assuming", "imagine", "pretend",
        "suddenly", "sudden", "instantly", "immediately", "briefly",
        # generic role / relation words whose object is the real subject
        "president", "prime", "minister", "capital", "king", "queen",
        "emperor", "author", "creator", "founder", "inventor", "leader",
        "owner", "winner", "captain", "mayor", "governor", "director",
        "chief", "boss", "head", "ceo", "population", "population of",
        # generic quantifiers / category words
        "best", "worst", "good", "bad", "better", "worse", "most", "least",
        "types", "type", "kind", "kinds", "sort", "sorts", "example",
        "examples", "difference", "differences", "definition", "definition of",
        "meaning of", "reason", "reasons", "fact", "facts", "history",
        "background", "overview", "summary",
        # vague filler nouns
        "some", "many", "much", "thing", "things", "stuff", "way", "ways",
        "point", "idea", "ideas", "something", "anything", "everything",
    }



    def _compute_phrase_embedding(self, phrase: str) -> Optional[np.ndarray]:
        """Compute a phrase embedding as the mean of its word vectors.
        Returns unit vector or None if no words have embeddings."""
        words = re.findall(r"[a-zA-Z']{2,}", phrase.lower())
        vecs = []
        for w in words:
            v = self._glove_vector(w)
            if v is not None:
                vecs.append(v)
        if not vecs:
            return None
        mean_vec = np.mean(vecs, axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec /= norm
        return mean_vec

    QUERY_PATTERNS = [
        (r"(?:what\s+happens\s+(?:if|when))\s+(.+)", 1),         # what happens if X (must be BEFORE generic what pattern)
        (r"(?:what|who)'?s?\s+(?:is\s+|are\s+)?(.+)", 1),       # what is X / who are X
        (r"(?:tell|show)\s+me\s+about\s+(.+)", 1),              # tell me about X
        (r"(?:explain|describe)\s+(.+)", 1),                     # explain X / describe X
        (r"(?:what|which)\s+(.+)\s+(?:is|are|mean)", 1),         # what X is / what X means
        (r"how\s+(?:do|does|did|can|to|would|should)\s+(.+)", 1), # how do X / how to X
        (r"(?:do you know|have you heard of)\s+(.+)", 1),        # do you know X
        (r"why\s+(?:is\s+|are\s+)?(.+)", 1),                    # why is X / why does X
    ]

    def save(self) -> str:
        """Save full cognitive state to disk. Returns path to save file."""
        import time
        import pickle
        import os
        import glob
        
        def _safe_dict_copy(d):
            for _ in range(5):
                try:
                    return dict(d)
                except RuntimeError:
                    time.sleep(0.01)
            try:
                return {k: v for k, v in list(d.items())}
            except Exception:
                return {}

        def _safe_concept_sources_copy():
            for _ in range(5):
                try:
                    return {k: list(v) for k, v in self._concept_sources.items()}
                except RuntimeError:
                    time.sleep(0.01)
            try:
                return {k: list(v) for k, v in list(self._concept_sources.items())}
            except Exception:
                return {}

        def _safe_set_copy(s):
            for _ in range(5):
                try:
                    return list(s)
                except RuntimeError:
                    time.sleep(0.01)
            try:
                return list(list(s))
            except Exception:
                return []

        with self._vocab_lock, self._graph_lock:
            _graph_snapshot = self.graph
            _decoder_w2i = _safe_dict_copy(self._decoder_word_to_idx)
            _decoder_i2w = _safe_dict_copy(self._decoder_idx_to_word)
            _decoder_w2e = _safe_dict_copy(self._decoder_word_to_embed)
            _ck_snapshot = _safe_dict_copy(self._concept_keywords)
            _cl_snapshot = _safe_set_copy(self._concept_labels)
            _vc_snapshot = _safe_set_copy(self._visited_concepts)
            _af_snapshot = _safe_dict_copy(self._activation_fatigue)
            _rt_snapshot = _safe_set_copy(self._recent_traversals)
            _rtm_snapshot = _safe_dict_copy(self._recent_traversal_map)
            _cv_snapshot = _safe_dict_copy(self._concept_vad)
            _td_snapshot = list(self._td_error_history[-50:])
            _cc_snapshot = _safe_dict_copy(self._concept_confidence)
            
            _topic_store = _safe_dict_copy(self._topic_store)
            _concept_visit_count = _safe_dict_copy(self._concept_visit_count)
            _concept_learning_progress = _safe_dict_copy(self._concept_learning_progress)
            _concept_pe_delta = _safe_dict_copy(self._concept_pe_delta)
            _concept_sources = _safe_concept_sources_copy()
            _explored_contradictions = [list(p) for p in _safe_set_copy(self._explored_contradictions)]
            _episodic_edges = _safe_dict_copy(self._episodic_edges)
            _semantic_edges = _safe_dict_copy(self._semantic_edges)
            _sentence_schema = _safe_dict_copy(self._sentence_schema)
            _cerebellar_ngram = _safe_dict_copy(self._cerebellar_ngram)
            _cerebellar_depth = _safe_dict_copy(self._cerebellar_depth)
            _concept_pos = _safe_dict_copy(self._concept_pos)

            state = {
                'graph': _graph_snapshot,
                'concept_keywords': _ck_snapshot,
                'turn_count': self.turn_count,
                'topic_list': list(self._topic_list),
                'topic_store': _topic_store,
                'response_context': list(self._response_context),
                'last_responses': list(self._last_responses),
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
                'concept_vad': _cv_snapshot,
                'meta_mode': self.meta_cog.current_mode.value,
                'contradiction_map': _safe_dict_copy(self._contradiction_map),
                'user_model': self.user_model,
                'use_vad': getattr(self, 'use_vad', True),
                'use_rlm': getattr(self, 'use_rlm', True),
                'use_beliefs': getattr(self, 'use_beliefs', True),
                'belief_store_state': getattr(self, 'belief_store', BeliefStore()).get_state(),
                # Background learning state
                'bg_learning_queue': list(self._bg_learning_queue),
                'bg_search_count': self._bg_search_count,
                'bg_multi_search_max': self._bg_multi_search_max,
                # Curiosity Drive state
                'curiosity_drive_enabled': self._curiosity_drive_enabled,
                'concept_visit_count': _concept_visit_count,
                'concept_learning_progress': _concept_learning_progress,
                'concept_pe_delta': _concept_pe_delta,
                'curiosity_topics_queue': list(self._curiosity_topics_queue),
                'last_auto_learn_turn': self._last_auto_learn_turn,
                'curiosity_urgency': self._curiosity_urgency,
                'user_query_topics': list(self._user_query_topics),
                'user_last_topic': self._user_last_topic,
                'concept_sources': _concept_sources,
                'explored_contradictions': _explored_contradictions,
                # Dual stores
                'episodic_edges': _episodic_edges,
                'semantic_edges': _semantic_edges,
                # Phase 10-17 state
                'sentence_schema': _sentence_schema,
                'mean_sentence_pe': self._mean_sentence_pe,
                'dopamine_tone': self._dopamine_tone,
                'td_error_history': _td_snapshot,
                'concept_confidence': _cc_snapshot,
                'cerebellar_ngram': _cerebellar_ngram,
                'cerebellar_ngram_state': self.cerebellar_ngram.get_state() if hasattr(self, 'cerebellar_ngram') else {},
                'cerebellar_depth': _cerebellar_depth,
                'concept_pos': _concept_pos,
                'concept_labels': list(_cl_snapshot),
                'visited_concepts': list(_vc_snapshot),
                'activation_fatigue': _af_snapshot,
                'recent_traversals': _rt_snapshot,
                'recent_traversal_map': _rtm_snapshot,
                'cognitive_state': self._cognitive_state,
                'state_duration': self._state_duration,
                'prefrontal_buffer': list(self._prefrontal_buffer),
                'mean_prediction_error': self._mean_prediction_error,
                'prediction_error_count': self._prediction_error_count,
                # Neural decoder
                'hippocampal_buffer_state': self.hippocampal_buffer.get_state(),
                'causal_schema_state': self.causal_schema.get_state(),
                'relation_memory_state': self.relation_memory.get_state(),
                'decoder_state_dict': self.neural_decoder.state_dict() if self.neural_decoder is not None else None,
                'decoder_training_count': self._decoder_training_count,
                'decoder_web_training_count': self._decoder_web_training_count,
                'decoder_seed_training_count': self._decoder_seed_training_count,
                'decoder_word_to_idx': _decoder_w2i,
                'decoder_idx_to_word': _decoder_i2w,
                'decoder_word_to_embed': _decoder_w2e,
                'definitions': self._definitions,
                # Curiosity diversity state
                'bg_learning_cycles': getattr(self, '_bg_learning_cycles', 0),
                'recent_curiosity_selections': list(getattr(self, '_recent_curiosity_selections', [])),
                'curiosity_selection_cooldown': getattr(self, '_curiosity_selection_cooldown', 5),
                # Phase 3 state
                'curiosity_engine_state': self.curiosity_engine.get_state(),
                'hippocampal_replay_state': self.hippocampal_replay.get_state(),
                'register_controller_state': self.register_controller.get_state(),
                # Neuromodulator state
                'neuromodulator_state': self.neuromodulator_engine.get_state()
                    if hasattr(self, 'neuromodulator_engine') and self.neuromodulator_engine is not None else None,
            }
            # Phase 1: Write graph to SQLite database for ACID persistence
            try:
                self.db.save_graph(self.graph)
            except Exception as e:
                if getattr(self, '_trace_enabled', False):
                    print(f"  [db] SQLite save failed: {e}")

            try:
                # Phase 6.1: Checkpoint rotation — save every 25 turns
                if self.turn_count > 0 and self.turn_count % 25 == 0:
                    checkpoint_path = self._save_path.replace('.pkl', f'_{self.turn_count}.pkl')
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump(state, f)
                    # Keep last 3 checkpoints, remove older ones
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
            # Use a custom unpickler that handles both 'ravana_chat' and
            # 'scripts.ravana_chat' module name references (pickle may store
            # either depending on how the module was imported when saved).
            class _RavanaUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    try:
                        return super().find_class(module, name)
                    except (ModuleNotFoundError, AttributeError):
                        if module == 'ravana_chat':
                            return super().find_class('scripts.ravana_chat', name)
                        elif module == 'scripts.ravana_chat':
                            return super().find_class('ravana_chat', name)
                        elif module == '__main__':
                            # Saved from direct `python ravana_chat.py` run
                            try:
                                return super().find_class('scripts.ravana_chat', name)
                            except (ModuleNotFoundError, AttributeError):
                                return super().find_class('ravana_chat', name)
                        raise
            with open(self._save_path, 'rb') as f:
                state = _RavanaUnpickler(f).load()

            # Restore graph
            loaded_graph = state['graph']
            if loaded_graph and loaded_graph.nodes:
                first_node = next(iter(loaded_graph.nodes.values()))
                if first_node.vector is not None and len(first_node.vector) != self.dim:
                    print(f"  [Load warning] Dimension mismatch: loaded graph has dim {len(first_node.vector)} but engine has dim {self.dim}. Discarding saved state.")
                    return False
            self.graph = loaded_graph
            self._concept_keywords = state['concept_keywords']
            self.turn_count = state['turn_count']

            # Restore decoder vocab mapping
            self._decoder_word_to_idx = state.get('decoder_word_to_idx', {})
            self._decoder_idx_to_word = state.get('decoder_idx_to_word', {})
            self._decoder_word_to_embed = state.get('decoder_word_to_embed', {})
            self._topic_list = state.get('topic_list', [])
            self._topic_store = state.get('topic_store', {})
            self._response_context = state.get('response_context', [])
            self._last_responses = [r for r in state['last_responses'] if r is not None]
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

            # Restore teen state (optional â€” may not exist in old saves)
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
            loaded_user_model = state.get('user_model', UserModel())
            # Upgrade old UserModel to new Theory of Mind version if needed
            if not hasattr(loaded_user_model, 'topic_interaction_count'):
                # Old UserModel - upgrade it
                upgraded = UserModel()
                upgraded.edge_reactivations = loaded_user_model.edge_reactivations
                upgraded.query_concepts = loaded_user_model.query_concepts
                upgraded.user_name = getattr(loaded_user_model, 'user_name', "")
                loaded_user_model = upgraded
            # Ensure P1 ToM fields exist (backward-compatible migration)
            if not hasattr(loaded_user_model, 'user_name'):
                loaded_user_model.user_name = ""
            if not hasattr(loaded_user_model, 'preferences'):
                loaded_user_model.preferences = {}
            if not hasattr(loaded_user_model, 'interaction_count'):
                loaded_user_model.interaction_count = sum(
                    getattr(loaded_user_model, 'topic_interaction_count', {}).values())
                loaded_user_model.relationship_depth = min(
                    1.0, loaded_user_model.interaction_count / 20.0)
                loaded_user_model.goals = []
                loaded_user_model.last_goal = 'EXPLORING'
            # Ensure P2 Emotional State Tracking fields exist (backward-compatible)
            if not hasattr(loaded_user_model, 'emotional_state'):
                loaded_user_model.emotional_state = {
                    'valence': 0.0, 'arousal': 0.3, 'dominance': 0.5,
                }
                loaded_user_model.belief_state = {}
                loaded_user_model.interaction_history = []
            self.user_model = loaded_user_model
            # A reask/correction is only meaningful within a single session.
            # The previous query is persisted in the saved state, which would
            # otherwise make the very first message of a new session look like a
            # repeat of the last message of the previous session and trigger a
            # false "reask correction". Reset it on load so reask detection only
            # fires between turns of the *same* run.
            if hasattr(self.user_model, '_previous_user_query'):
                self.user_model._previous_user_query = None
            # Restore belief store
            bs_state = state.get('belief_store_state', None)
            if bs_state:
                self.belief_store.set_state(bs_state)

            # Restore prefrontal buffer
            self._prefrontal_buffer = state.get('prefrontal_buffer', [])
            # Restore prediction error state
            self._mean_prediction_error = state.get('mean_prediction_error', 0.0)
            self._prediction_error_count = state.get('prediction_error_count', 0)

            # Restore background learning state
            self._bg_learning_queue = state.get('bg_learning_queue', [])
            self._bg_search_count = state.get('bg_search_count', 0)
            self._bg_multi_search_max = state.get('bg_multi_search_max', 3)

            # Restore curiosity drive state
            self._curiosity_drive_enabled = state.get('curiosity_drive_enabled', True)
            self._concept_visit_count = state.get('concept_visit_count', {})
            self._concept_learning_progress = state.get('concept_learning_progress', {})
            self._concept_pe_delta = state.get('concept_pe_delta', {})
            self._curiosity_topics_queue = state.get('curiosity_topics_queue', [])
            self._last_auto_learn_turn = state.get('last_auto_learn_turn', 0)
            self._curiosity_urgency = state.get('curiosity_urgency', 0.0)
            self._user_query_topics = state.get('user_query_topics', [])
            self._user_last_topic = state.get('user_last_topic', '')
            raw_sources = state.get('concept_sources', {})
            self._concept_sources = {k: set(v) for k, v in raw_sources.items()}
            raw_contra = state.get('explored_contradictions', [])
            self._explored_contradictions = {tuple(p) for p in raw_contra}
            # Restore curiosity diversity state
            self._bg_learning_cycles = state.get('bg_learning_cycles', 0)
            self._recent_curiosity_selections = state.get('recent_curiosity_selections', [])
            self._curiosity_selection_cooldown = state.get('curiosity_selection_cooldown', 5)

            # Restore dual stores
            epi_state = state.get('episodic_edges', {})
            if epi_state:
                self._episodic_edges = epi_state
                self._episodic_by_src.clear()
                for (s, t), e in self._episodic_edges.items():
                    self._episodic_by_src.setdefault(s, []).append((t, e))
            sem_state = state.get('semantic_edges', {})
            if sem_state:
                self._semantic_edges = sem_state
                self._semantic_by_src.clear()
                for (s, t), e in self._semantic_edges.items():
                    self._semantic_by_src.setdefault(s, []).append((t, e))

            # Restore Phase 10-17 state
            self._sentence_schema = state.get('sentence_schema', {})
            self._mean_sentence_pe = state.get('mean_sentence_pe', 0.0)
            self._dopamine_tone = state.get('dopamine_tone', 0.5)
            self._td_error_history = state.get('td_error_history', [])
            self._concept_confidence = state.get('concept_confidence', {})
            self._cerebellar_ngram = state.get('cerebellar_ngram', {})
            self._cerebellar_depth = state.get('cerebellar_depth', {})
            # Restore CerebellarNgram object state
            cng_state = state.get('cerebellar_ngram_state', {})
            if cng_state and hasattr(self, 'cerebellar_ngram'):
                self.cerebellar_ngram.set_state(cng_state)

            self._visited_concepts = set(state.get('visited_concepts', []))
            self._activation_fatigue = state.get('activation_fatigue', {})
            self._recent_traversals = state.get('recent_traversals', [])
            self._recent_traversal_map = state.get('recent_traversal_map', {})
            self._cognitive_state = state.get('cognitive_state', 'default')
            self._state_duration = state.get('state_duration', 0)
            self._impossible_queries = state.get('impossible_queries', [])
            self._concept_pos = ConceptPosDict(state.get('concept_pos', {}))
            self._concept_labels = set(state.get('concept_labels', []))

            # Stash decoder state dict to be loaded after _build_decoder_vocab() 
            # (neural decoder doesn't exist yet â€” created later in __init__)
            hb_state = state.get('hippocampal_buffer_state', None)
            if hb_state:
                self.hippocampal_buffer.set_state(hb_state)
            cs_state = state.get('causal_schema_state', None)
            if cs_state:
                self.causal_schema.set_state(cs_state)
            rm_state = state.get('relation_memory_state', None)
            if rm_state:
                self.relation_memory.set_state(rm_state)
            decoder_sd = state.get('decoder_state_dict', None)
            self._saved_decoder_state = decoder_sd if decoder_sd is not None else {}
            # Stash neuromodulator state â€” will be loaded after _build_decoder_vocab()
            # creates the neuromodulator engine.
            self._saved_neuromodulator_state = state.get('neuromodulator_state', None)
            self._decoder_training_count = state.get('decoder_training_count', 0)
            self._decoder_web_training_count = state.get('decoder_web_training_count', 0)
            self._decoder_seed_training_count = state.get('decoder_seed_training_count', 0)
            self._definitions = state.get('definitions', {})
            # Purge polluted definition keys on load. Earlier versions stored
            # incoherent web fragments under generic/pronoun words (e.g.
            # "you" -> "the stronger player...", "life" -> "war zone"). Drop
            # those so RAVANA stops answering every mention of them with
            # abstract, off-topic text.
            _DEF_PURGE = {
                "you", "i", "we", "they", "he", "she", "it", "me", "my", "your",
                "our", "their", "us", "them", "him", "her", "this", "that",
                "life", "lives", "death", "love", "hate", "god", "time", "thing",
                "things", "world", "people", "person", "day", "year", "mind",
                "soul", "self", "meaning", "purpose", "truth", "beauty",
                "knowledge", "wisdom", "freedom", "happiness", "success",
                "power", "nature", "art", "music", "science", "dream", "dreams",
                "hope", "fear", "war", "peace", "friend", "family", "home",
                "money", "work", "play", "game", "book", "story", "idea",
                "thought", "word", "language", "number", "system", "process",
            }
            if isinstance(self._definitions, dict):
                _clean_defs = {}
                for k, v in self._definitions.items():
                    # Drop generic/pronoun words (above).
                    if k in _DEF_PURGE:
                        continue
                    # Drop junk fragment keys: multi-word phrases, quoted keys,
                    # or keys that begin with a stopword (e.g. "the quokka's
                    # range", "won 2022 world", "fast facts quokkas"). Real
                    # concept keys are single clean tokens.
                    if (" " in k) or ("'" in k) or ('"' in k):
                        continue
                    first = k.split()[0] if k.split() else k
                    if first in STOP_WORDS:
                        continue
                    _clean_defs[k] = v
                self._definitions = _clean_defs

            # Restore Phase 3 state
            ce_state = state.get('curiosity_engine_state', None)
            if ce_state and hasattr(self, 'curiosity_engine'):
                self.curiosity_engine.set_state(ce_state)
            hr_state = state.get('hippocampal_replay_state', None)
            if hr_state and hasattr(self, 'hippocampal_replay'):
                self.hippocampal_replay.set_state(hr_state)
            rc_state = state.get('register_controller_state', None)
            if rc_state and hasattr(self, 'register_controller'):
                self.register_controller.set_state(rc_state)

            # Restart background learning
            self.start_background_learning()

            # Always rebuild POS tags on load to ensure accuracy with latest lexicon rules
            self._build_concept_pos()
            if hasattr(self, 'cerebellar_ngram'):
                self.cerebellar_ngram.seed_from_pos(self._concept_pos)
            if hasattr(self, 'syntactic_assembly'):
                self.syntactic_assembly.seed_from_pos(self._concept_pos)
            
            return True
        except Exception as e:
            print(f"  [Load error] {e}")
            return False


    def _adapt_verbosity_for_user(self, plan: 'DiscoursePlan', subject: str) -> 'DiscoursePlan':
        """Adaptive language complexity (roadmap Â§7 deliverable A).

        Modulates the PFC discourse plan based on user familiarity:
        - Low familiarity (< 0.3): keep all 3 intents â€” user needs full explanation
        - Medium familiarity (0.3-0.7): keep 2-3 intents
        - High familiarity (> 0.7) + LEARNING goal: trim to 2 â€” user knows the basics

        P2 Emotional Mirroring: user arousal also modulates verbosity â€” excited users
        get slightly longer, more engaged responses; calm users get more concise ones.

        This respects the PrefrontalWorkspace capacity (7Â±2 items, Baddeley & Hitch 1974)
        by avoiding unnecessary verbal load for expert users.
        """
        um = self.user_model
        familiarity = um.infer_user_knows(subject)
        goal = um.last_goal

        # P2: User arousal modulates base verbosity (mirroring loop)
        user_arousal = um.emotional_state.get('arousal', 0.3)
        target_intents = 3
        if user_arousal < 0.25:
            target_intents = 2  # calm user â†’ concise
        elif user_arousal > 0.6:
            target_intents = 4  # excited user â†’ more engagement

        if familiarity < 0.3:
            # Novice: full explanation
            if len(plan.intents) > target_intents:
                plan.intents = plan.intents[:target_intents]
            return plan
        elif familiarity > 0.7 and goal == "LEARNING":
            # Expert learner: trim â€” skip the generic ELABORATE
            if len(plan.intents) > 2:
                plan.intents = [plan.intents[0], plan.intents[1]]
            return plan
        # Medium: keep original plan, but cap at target
        if len(plan.intents) > target_intents:
            plan.intents = plan.intents[:target_intents]
        return plan






    def _detect_recall_trigger(self, text: str) -> Optional[str]:
        """Phase 3.3: Detect if user is recalling a past topic using vector semantics.
        
        Uses GloVe vector similarity between query words and recall-related seed concepts.
        If any query word has a cosine similarity >= _RECALL_DETECTION_THRESHOLD to any
        recall seed concept, the query is treated as a recall attempt.
        
        This avoids hardcoded trigger patterns and naturally generalizes to any
        semantically similar phrasing (e.g., "forgot", "previously", "what did I")."""
        text_lower = text.lower()
        words = [w.strip(".,!?") for w in text_lower.split() if len(w.strip(".,!?")) >= 3]
        
        # Pre-compute GloVe vectors for recall seeds (lazy cache)
        if not hasattr(self, '_recall_seed_vecs'):
            seed_vecs = {}
            for seed in self._RECALL_SEED_CONCEPTS:
                v = self._glove_vector(seed)
                if v is not None:
                    seed_vecs[seed] = v
            self._recall_seed_vecs = seed_vecs
        
        if not self._recall_seed_vecs:
            return None
        
        # Check each content word in the query for semantic similarity to recall seeds
        is_recall = False
        for word in words:
            wv = self._glove_vector(word)
            if wv is None:
                continue
            for seed, sv in self._recall_seed_vecs.items():
                sim = float(np.dot(wv, sv))
                if sim >= self._RECALL_DETECTION_THRESHOLD:
                    is_recall = True
                    break
            if is_recall:
                break
        
        if not is_recall:
            return None
        
        # If recall detected, find the most relevant past topic
        if self._topic_list:
            # Score each past topic by semantic similarity to the query
            best_topic = None
            best_score = 0.0
            for topic in reversed(self._topic_list):
                tv = self._glove_vector(topic)
                if tv is None:
                    continue
                score = 0.0
                for word in words:
                    wv = self._glove_vector(word)
                    if wv is not None:
                        score += float(np.dot(wv, tv))
                if score > best_score:
                    best_score = score
                    best_topic = topic
            if best_topic:
                return best_topic
            return self._topic_list[-1]
        
        return text_lower.split()[0] if words else None




    def _extract_topic(self, text: str, activated: List[int]) -> Tuple[str, str]:
        """Extract the main topic from input. Uses graph-activated concepts
        first, then falls back to pattern detection.

        For 'what is trust' -> 'trust'
        For 'you know i was thinking about trust' -> 'trust' (skips 'you', 'i')
        For 'does learning change your brain' -> 'learning'
        """
        # Use the multi-strategy query grounder
        topic, confidence, method = self._ground_query(text)
        if topic and confidence >= 0.5:
            return (topic, text)
        # Prefer low-confidence ground_query result over question/skip words
        if topic and method != "all_unknown" and method != "no_pattern" and method != "no_match":
            return (topic, text)

        # Fallback: best activated concept (skip question, topic-skip, and short words)
        # Prefer nouns over adjectives/verbs using POS tags
        # CRITICAL: Prefer labels that actually appear in the user's input text
        # over GloVe-neighbor activated words. Prevents extracting 'because'
        # for 'what is blockchain' when 'blockchain' has no GloVe vector.
        if activated:
            input_words = set(w.strip(".,!?").lower() for w in text.split() 
                             if len(w.strip(".,!?")) > 2)
            best_real = None
            best_noun = None
            in_input_noun = None
            in_input_real = None
            for nid in activated:
                node = self.graph.get_node(nid)
                if node and node.label:
                    lbl = node.label.lower()
                    if (len(lbl) > 2 and lbl not in self.QUESTION_WORDS
                            and lbl not in self.TOPIC_SKIP_WORDS):
                        pos = self._concept_pos.get(lbl, 'noun')
                        # Prioritize: input-text words > arbitrary GloVe neighbors
                        appears_in_input = lbl in input_words or any(lbl in w for w in input_words)
                        if appears_in_input and pos == 'noun' and in_input_noun is None:
                            in_input_noun = (node.label, text)
                        if appears_in_input and in_input_real is None:
                            in_input_real = (node.label, text)
                        if pos == 'noun' and best_noun is None:
                            best_noun = (node.label, text)
                        if best_real is None:
                            best_real = (node.label, text)
            # Prefer: input-matching noun > input-matching any > graph noun > graph any
            if in_input_noun:
                if getattr(self, '_trace_enabled', False):
                    print(f"  [trace]   topic='{in_input_noun[0]}' (input-match noun from activated)")
                return in_input_noun
            if in_input_real:
                if getattr(self, '_trace_enabled', False):
                    print(f"  [trace]   topic='{in_input_real[0]}' (input-match from activated)")
                return in_input_real
            # CRITICAL: No activated concept matched user input â€” skip to raw text
            # processing instead of picking an unrelated GloVe neighbor.
            # This prevents extracting 'because' for 'what is blockchain'.
            if getattr(self, '_trace_enabled', False):
                input_vs = ', '.join(sorted(input_words)) if input_words else '(empty)'
                print(f"  [trace]   no input-match in activated â€” falling through to raw text (input_words={{{input_vs}}})")
            # Fall through to raw text processing below (don't use best_noun/best_real)

        # Fallback: find meaningful words
        words = [w.strip(".,!?") for w in text.lower().split()
                 if len(w.strip(".,!?")) > 2
                 and w.strip(".,!?") not in self.QUESTION_WORDS
                 and w.strip(".,!?") not in self.TOPIC_SKIP_WORDS
                 and w.strip(".,!?") not in STOP_WORDS]
        if words:
            # Prefer words that are actually in the graph (known concepts) over unknown ones
            known_words = [w for w in reversed(words) if w in self._concept_labels or w in self._concept_keywords]
            # Prefer nouns among known words
            noun_words = [w for w in known_words if self._concept_pos.get(w, 'noun') == 'noun']
            if noun_words:
                return (noun_words[0], text)
            if known_words:
                return (known_words[0], text)
            # Prefer nouns among unknown words
            unknown_nouns = [w for w in words if self._concept_pos.get(w, 'noun') == 'noun']
            if unknown_nouns:
                return (unknown_nouns[0], text)
            return (words[-1], text)

        first = text.split()[0] if text.split() else ""
        first_stripped = first.strip(".,!?").lower()
        if first_stripped and len(first_stripped) > 2 and first_stripped not in self.QUESTION_WORDS and first_stripped not in self.TOPIC_SKIP_WORDS:
            return (first_stripped, text)
        return ("", text)




    @classmethod
    def _clean_subject_phrase(cls, phrase: str) -> str:
        """Strip question/verb/role words so the real topic survives.

        'happened 1923' -> '1923', 'build python web' -> 'python web',
        'meaning life' -> 'life', 'president france' -> 'france'.
        Falls back to the original phrase if everything gets stripped (so we
        never return an empty subject for a genuinely single-word topic).
        """
        words = [w.strip(".,!?") for w in phrase.lower().split()
                 if w.strip(".,!?") not in STOP_WORDS
                 and w.strip(".,!?") not in cls.QUESTION_WORDS]
        kept = [w for w in words if w not in cls._SUBJECT_CONTEXT_WORDS]
        if kept:
            return " ".join(kept)
        return phrase.strip(".,!?")

    def _ground_query(self, text: str) -> Tuple[str, float, str]:
        """Multi-strategy query grounding. Returns (subject, confidence, method).

        Strategies (tried in order):
        a) PrefrontalWorkspace question type parsing & exact phrase matching
        b) Compositional â€” split phrase, count known vs unknown words
        c) Phrase embedding similarity â€” mean word vec â†’ nearest concept (cosine > 0.75)
        d) Best single word fallback â€” last meaningful non-stop word
        """
        # Strategy A: Use PrefrontalWorkspace question type detection to parse semantic payload
        qtype = "general"
        query_phrase = ""
        try:
            if hasattr(self, 'pfc_workspace'):
                qtype, groups = self.pfc_workspace.detect_question_type(text, self._concept_pos)
                if groups:
                    query_phrase = groups[0].strip()
        except Exception:
            pass

        if not query_phrase:
            # Fallback to custom patterns
            text_lower = text.lower().strip(" ?!.")
            for pattern, group_idx in self.QUERY_PATTERNS:
                m = re.match(pattern, text_lower)
                if m:
                    query_phrase = m.group(group_idx).strip()
                    break

        if not query_phrase:
            return ("", 0.0, "no_pattern")

        # Strategy A2: Exact multi-word phrase match (domain concepts, seeded multi-word)
        phrase_clean = query_phrase.strip(".,!?")
        if phrase_clean in self._concept_labels:
            return (phrase_clean, 0.95, "exact_label")
        if phrase_clean in self._concept_keywords:
            return (phrase_clean, 0.90, "exact_keyword")

        # Strategy C (moved before B): Compositional â€” score words by known/unknown ratio
        words = [w.strip(".,!?") for w in query_phrase.split()
                 if len(w.strip(".,!?")) > 2
                 and w.strip(".,!?") not in self.QUESTION_WORDS
                 and w.strip(".,!?") not in self.TOPIC_SKIP_WORDS
                 and w.strip(".,!?") not in STOP_WORDS]
        if words:
            if len(words) >= 2:
                # For scenario/hypothetical/causal queries (e.g. hypothetical, why, how),
                # the last content/entity word represents the target scenario.
                # NOTE: only for "hypothetical" — for "why"/"how"/"compare" the
                # trailing word is usually a predicate ("salty" in "why is the
                # ocean salty", an adjective), not the actual topic. For those
                # we keep the multi-word phrase below so web grounding stays
                # on the real subject ("ocean salty").
                if qtype == "hypothetical" and len(words) >= 2:
                    last_word = words[-1]
                    if last_word in self._concept_labels or last_word in self._concept_keywords:
                        if last_word not in self._GENERIC_NOUNS:
                            return (last_word, 0.7, "scenario_last_entity")
                # Use the first 2-3 content words as the search subject (e.g. "time machine")
                clean_subj = " ".join(words[:3]) if len(words) >= 3 else " ".join(words)
                clean_subj = self._clean_subject_phrase(clean_subj)
                return (clean_subj, 0.45, "multi_word_unconnected")

            known_words = [w for w in words if w in self._concept_labels or w in self._concept_keywords]
            unknown_words = [w for w in words if w not in known_words]
            if known_words:
                if unknown_words:
                    # If there's any unknown word in the multi-word query, keep the clean subject phrase
                    # and return a low confidence to trigger web learning for the whole phrase
                    clean_subj = " ".join(words[:3]) if len(words) >= 3 else " ".join(words)
                    clean_subj = self._clean_subject_phrase(clean_subj)
                    return (clean_subj, 0.35, "partial_unknown")
                ratio = len(known_words) / len(words)
                # Prefer the FIRST known word over the last: in English the head
                # noun typically trails last, but compositional grounding here
                # keeps returning trailing generic nouns ("system", "process",
                # "matter") that collapse the subject. Pick the earliest known
                # word that isn't a generic/vague concept; fall back to the
                # whole cleaned phrase if only generic nouns are known.
                _generic = self._GENERIC_NOUNS
                specific = [w for w in known_words if w not in _generic]
                topic = specific[0] if specific else known_words[0]
                # If the chosen topic is a generic noun but other words exist,
                # keep the multi-word phrase so web grounding stays on-topic.
                if topic in _generic and len(words) > 1:
                    clean_subj = " ".join(words[:3]) if len(words) >= 3 else " ".join(words)
                    clean_subj = self._clean_subject_phrase(clean_subj)
                    if clean_subj:
                        return (clean_subj, 0.4, "compositional_generic_topic")
                return (topic, min(0.85, 0.5 + ratio * 0.4), f"compositional_{ratio:.2f}")
            # All unknown — will trigger web learning for the full phrase
            if words:
                clean_subj = " ".join(words[:3]) if len(words) >= 3 else " ".join(words)
                clean_subj = self._clean_subject_phrase(clean_subj)
                return (clean_subj, 0.2, "all_unknown")


        # Strategy B: Phrase embedding similarity search (fallback for short queries)
        phrase_vec = self._compute_phrase_embedding(query_phrase)
        if phrase_vec is not None:
            best_sim = 0.0
            best_label = None
            for nid, node in self.graph.nodes.items():
                if node.label and node.vector is not None:
                    sim = float(np.dot(phrase_vec, node.vector))
                    if sim > best_sim:
                        best_sim = sim
                        best_label = node.label
            # Higher threshold + reject TOPIC_SKIP_WORDS matches
            if best_label and best_sim > 0.75 and best_label.lower() not in self.TOPIC_SKIP_WORDS:
                return (best_label, best_sim, f"phrase_sim_{best_sim:.2f}")

        # Strategy D: Spelling-tolerant close match (handles typos like "intellegence")
        if words:
            close_matches = []
            for w in words:
                wl = w.lower()
                for label in list(self._concept_labels):
                    if (label.startswith(wl[:3]) and abs(len(label) - len(wl)) <= 2) or                        (len(wl) >= 4 and label.startswith(wl[:4])):
                        close_matches.append(label)
                        break
            if close_matches:
                topic = close_matches[-1]
                return (topic, 0.5, f"close_match_{topic}")

        return ("", 0.0, "no_match")




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




    def _personalized_greeting(self) -> str:
        """Return a personalized greeting prefix when relationship_depth warrants it.

        Neuroscience basis: repeated social interaction builds rapport, modeled
        as relationship_depth âˆˆ [0, 1]. Above 0.5, reference the last topic to
        demonstrate memory and continuity (roadmap Â§9).

        Returns empty string if relationship is too new or no prior topic exists.
        """
        um = self.user_model
        if um.relationship_depth < 0.5:
            return ""
        if not um.last_topic:
            return ""
        # Only greet every ~10 interactions to avoid repetition
        if um.interaction_count % 10 != 0 and um.interaction_count > 1:
            return ""
        past = um.last_topic.capitalize()
        if um.relationship_depth > 0.8:
            return f"Great to see you! I remember we were talking about {past}. "
        return f"Welcome back! Last time we discussed {past}. "




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

    # â”€â”€â”€ Graph-Driven Response Generation â”€â”€â”€
    # NO hardcoded strings. ALL content words are concept labels from the graph.
    # Edge relation types (stored in graph edge data) are mapped to graph concept
    # labels that already exist in the seeded vocabulary â€” no external text.

    # Tiered connectors: (min_weight, [word, ...])
    # Auto-wired edges: min=0.30, ~0.30-0.40 common, max=0.50
    # Weight â‰¥ 0.33 (~top 25%): stronger connector (e.g. "link", "and")
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
        "emotional": [
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
        "episodic": "connect",
    }

    # Edge type â†’ discourse starter (all labels exist in seeded TEEN_CONCEPTS)
    _EDGE_TO_STARTER = {
    }

    # â”€â”€ Solution #5: Relation Type Inference â”€â”€
    # Known label-pair patterns for heuristic relation type assignment.
    # All pairs stored as sorted tuples for consistent lookup.
    CONTRASTIVE_PAIRS = {
        tuple(sorted(["good", "bad"])), tuple(sorted(["love", "hate"])),
        tuple(sorted(["life", "death"])), tuple(sorted(["truth", "lie"])),
        tuple(sorted(["freedom", "oppression"])), tuple(sorted(["courage", "fear"])),
        tuple(sorted(["hope", "despair"])), tuple(sorted(["knowledge", "ignorance"])),
        tuple(sorted(["justice", "injustice"])), tuple(sorted(["create", "destroy"])),
        tuple(sorted(["accept", "reject"])), tuple(sorted(["always", "never"])),
        tuple(sorted(["happy", "sad"])), tuple(sorted(["joy", "grief"])),
        tuple(sorted(["excited", "bored"])), tuple(sorted(["big", "small"])),
        tuple(sorted(["hot", "cold"])), tuple(sorted(["up", "down"])),
        tuple(sorted(["in", "out"])), tuple(sorted(["here", "there"])),
        tuple(sorted(["now", "later"])), tuple(sorted(["yes", "no"])),
        tuple(sorted(["more", "less"])), tuple(sorted(["possible", "impossible"])),
        tuple(sorted(["trust", "hypocrisy"])), tuple(sorted(["freedom", "control"])),
    }
    CAUSAL_PAIRS = {
        tuple(sorted(["learn", "knowledge"])), tuple(sorted(["study", "understanding"])),
        tuple(sorted(["practice", "skill"])), tuple(sorted(["challenge", "struggle"])),
        tuple(sorted(["struggle", "growth"])), tuple(sorted(["question", "curiosity"])),
        tuple(sorted(["explore", "discovery"])), tuple(sorted(["trust", "friendship"])),
        tuple(sorted(["hypocrisy", "distrust"])), tuple(sorted(["grief", "sadness"])),
        tuple(sorted(["hope", "motivation"])), tuple(sorted(["rejection", "loneliness"])),
        tuple(sorted(["acceptance", "belonging"])), tuple(sorted(["anger", "conflict"])),
        tuple(sorted(["empathy", "understanding"])), tuple(sorted(["criticism", "growth"])),
        tuple(sorted(["failure", "learning"])), tuple(sorted(["change", "growth"])),
        tuple(sorted(["knowledge", "wisdom"])), tuple(sorted(["experience", "wisdom"])),
        tuple(sorted(["art", "expression"])), tuple(sorted(["science", "progress"])),
        tuple(sorted(["imagination", "invention"])), tuple(sorted(["experiment", "knowledge"])),
        tuple(sorted(["sleep", "tired"])), tuple(sorted(["play", "happy"])),
        tuple(sorted(["cause", "change"])), tuple(sorted(["sun", "hot"])),
        tuple(sorted(["sun", "light"])), tuple(sorted(["eat", "food"])),
        tuple(sorted(["drink", "water"])),
    }
    IS_A_PAIRS = {
        tuple(sorted(["dog", "animal"])), tuple(sorted(["cat", "animal"])),
        tuple(sorted(["bird", "animal"])), tuple(sorted(["rose", "flower"])),
        tuple(sorted(["oak", "tree"])), tuple(sorted(["oxiverse", "ecosystem"])),
        tuple(sorted(["intentforge", "search engine"])),
        tuple(sorted(["ravana", "cognitive architecture"])),
    }




    def _recall_past(self, subj: str, obj: str) -> List[str]:
        related = []
        for t in self._topic_list:
            pl = t.lower()
            sl = subj.lower()
            if pl != sl and (pl in sl or sl in pl or len(set(pl.split()) & set(sl.split())) > 0):
                related.append(t)
        return related[:3]

    # â”€â”€â”€ Phase 9c: Hippocampal Indexing â”€â”€â”€
    # Based on: Teyler & Rudy (2007) hippocampal indexing theory.
    # The hippocampus stores an INDEX to distributed neocortical patterns,
    # not the memory content itself. Reactivation of the index â†’ reactivation
    # of the distributed pattern â†’ memory experience.
    #
    # Instead of storing full topic content, we store a sparse index:
    # which concept IDs were activated, which edges were traversed.
    # During recall, the index reactivates the distributed graph pattern.




    def _update_emotion(self, text: str):
        """More nuanced emotional processing â€” teenage range of emotions."""
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
        # Phase 7.5: Curiosity drive â€” boost arousal for impossible queries
        if getattr(self, '_last_strategy_used', '') in ('G_uncertainty', 'F_web_research'):
            sa += 0.6  # strong curiosity arousal for impossible query
            sv += 0.1  # slight positive valence for curiosity
        self.emotion.update(stimulus_valence=sv, stimulus_arousal=sa,
                           stimulus_dominance=self.identity.state.strength * 0.4 + 0.2,
                           uncertainty=self._free_energy * 0.5, dt=1.0)

    # â”€â”€ P1 Theory of Mind â”€â”€




    def _update_user_model(self, text: str, subject: str,
                           associations: List[Tuple[str, float]]):
        """Deep Theory of Mind update after turn processing (roadmap Â§7).

        Extends the lightweight observe_user_query (which runs early in
        process_turn) with post-spread-activation updates:
        - Update topic familiarity from graph associations
        - Track inferred goals alongside cognitive style
        - Build personalized greeting eligibility
        """
        um = self.user_model
        # Update topic familiarity from the spread-activation associations.
        # Each association the user's query touched becomes slightly more
        # familiar (exponential moving average, rate 0.1).
        for concept, confidence in associations:
            cl = concept.lower()
            um.knowledge_model[cl] = (
                0.9 * um.knowledge_model.get(cl, 0.0)
                + 0.1 * min(1.0, confidence + 0.3)
            )
        # Goal is already inferred inside observe_user_query via infer_user_goal.
        # Store the last goal for adaptive verbosity check.
        self._last_user_goal = getattr(self, '_last_user_goal', 'EXPLORING')
        self._last_user_goal = um.last_goal




    def print_traces(self, label: str):
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
            unresolved = sum(1 for iq in self._impossible_queries if not iq.resolved)
            print(f"  [trace]   impossible queries: {len(self._impossible_queries)} total, {unresolved} unresolved")
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

    # === Background Web Learning ===
    # RAVANA can learn from the web between user messages, performing
    # multiple related searches per query to build richer knowledge.




    def start_background_learning(self):
        """Start the background learning thread. Called once at engine creation or CLI start."""
        if self._bg_learning_active and self._bg_learning_thread and self._bg_learning_thread.is_alive():
            return
        self._bg_learning_active = True
        self._bg_learning_thread = threading.Thread(target=self._bg_learn_loop, daemon=True)
        self._bg_learning_thread.start()
        if self._trace_enabled:
            print('  [bg] background learning thread started')




    def stop_background_learning(self):
        """Stop the background learning thread gracefully.""" 
        # Final curiosity sync - ensure latest diversity state is captured
        # Must run BEFORE _bg_learning_active is set to False
        try:
            self._auto_select_curiosity_topics(max_topics=0)  # just sync state
        except Exception:
            pass
        
        self._bg_learning_active = False
        self._cascade_for_quality = False
        self._bg_idle_event.set()  # wake up the thread so it can exit
        if self._bg_learning_thread and self._bg_learning_thread.is_alive():
            self._bg_learning_thread.join(timeout=30)
        if self._trace_enabled:
            print(f'  [bg] background learning stopped (performed {self._bg_search_count} searches)')




    def _is_informational_query(self, query: str, subject: str) -> bool:
        """Determines if a query is informational/fact-seeking (asks for a definition,
        factual knowledge, or explanation of an unknown concept) rather than
        conversational, logical, relational, or conditional.
        """
        q = query.lower().strip(" ?!.")
        
        # 1. Statements are never informational queries
        is_question = query.strip().endswith('?') or any(w in q for w in ["what", "who", "where", "when", "why", "how", "define", "explain", "describe", "tell me about"])
        if not is_question:
            return False
            
        # 2. Logic puzzles, conditional scenarios, riddles, comparison queries are NOT simple definition/fact-seeking queries.
        # These require cognitive reasoning, which should be processed internally.
        reasoning_patterns = [
            r"\b(if|suppose|assume|predict)\b",  # conditional/scenario (note: 'when' at sentence start is a question, not conditional)
            r"\b(why|how does|how do|how to)\b",      # causal/procedural reasoning
            r"\b(compare|contrast|difference|similar|opposite|antonym|synonym|analogy|analogies)\b", # relation/comparison
            r"\b(taller|shorter|heavier|lighter|older|younger|better|worse|biggest|tallest|heaviest|smartest)\b", # comparison/ordering
            r"\b(riddle|puzzle|logic|math|solve|calculation)\b", # logic/riddle
            r"\bis to\b", # analogy
            r"\b(you|your|yourself|think|opinion|feel|friendship|love|meaning of life)\b", # personal, opinion, or open philosophical
            r"\b(joke|riddle|story|poem|tale|fun fact|quote|fact about)\b", # entertainment/trivia — not definition-seeking
        ]
        for pattern in reasoning_patterns:
            if re.search(pattern, q):
                return False
                
        # 3. Check if the query matches a pattern asking for a definition/fact
        info_patterns = [
            r"^(what|who) (is|are|was|were|refers to|means)\b",
            r"^(what|who|where|when|which|how) \w+\b",  # "who won...", "where is...", "when was X built", "which city...", "how do X..."
            r"^define\b",
            r"^explain\b",
            r"^tell me about\b",
            r"^do you know\b",
            r"^what do you know about\b",
        ]
        if any(re.match(pat, q) for pat in info_patterns):
            return True
            
        # If it's a question but didn't match the reasoning patterns or explicit informational patterns,
        # we err on the side of conversational/reasoning to let RAVANA chat like a human.
        return False



    def _needs_web_search(self, subject: str) -> bool:
        """Check if a subject needs web search to enrich its associations.

        Returns True if:
        - The subject is not in the graph at all (completely unknown)
        - The subject IS in the graph but has fewer than 3 meaningful
          associations (edges with weight > 0.3). This catches abstract
          concepts like "consciousness" that are seeded with weak teenage
          associations and need web enrichment to produce useful responses.

        Returns False only if the concept has >= 3 strong graph edges
        (enough knowledge to generate a meaningful response via the
        ventral path alone).
        """
        if not subject:
            return False
        subj_lower = subject.lower().strip()

        # Not in graph at all → definitely needs web search
        if subj_lower not in self._concept_keywords and subj_lower not in self._concept_labels:
            with self._graph_lock:
                for nid, node in self.graph.nodes.items():
                    if node.label and node.label.lower() == subj_lower:
                        break
                else:
                    return True

        # Subject is in the graph — count strong outgoing edges (weight > 0.3)
        strong_edges = 0
        subj_nids = self._concept_keywords.get(subj_lower, [])
        for nid in subj_nids:
            for tid, edge in self.graph.get_outgoing(nid):
                if edge.weight > 0.3:
                    strong_edges += 1
            for src, edge in self.graph.get_incoming(nid):
                if edge.weight > 0.3:
                    strong_edges += 1

        # Need >= 3 strong associations to have enough knowledge
        return strong_edges < 3


    def _is_informational_query(self, query: str, subject: str) -> bool:
        """Determines if a query is informational/fact-seeking (asks for a definition,
        factual knowledge, or explanation of an unknown concept) rather than
        conversational, logical, relational, or conditional.
        """
        q = query.lower().strip(" ?!.")
        
        # 1. Statements are never informational queries
        is_question = query.strip().endswith('?') or any(w in q for w in ["what", "who", "where", "when", "why", "how", "define", "explain", "describe", "tell me about"])
        if not is_question:
            return False
            
        # 2. Logic puzzles, conditional scenarios, riddles, comparison queries are NOT simple definition/fact-seeking queries.
        # These require cognitive reasoning, which should be processed internally.
        # NOTE: 'when' at sentence start is a QUESTION word, not a conditional
        # ("when was X built"), so it must not be treated as a scenario here.
        reasoning_patterns = [
            r"\b(if|suppose|assume|predict)\b",  # conditional/scenario
            r"\b(compare|contrast|difference|similar|opposite|antonym|synonym|analogy|analogies)\b", # relation/comparison
            r"\b(taller|shorter|heavier|lighter|older|younger|better|worse|biggest|tallest|heaviest|smartest)\b", # comparison/ordering
            r"\b(riddle|puzzle|logic|math|solve|calculation)\b", # logic/riddle
            r"\bis to\b", # analogy
            r"\b(you|your|yourself|think|opinion|feel|friendship|love|meaning of life)\b", # personal, opinion, or open philosophical
        ]
        for pattern in reasoning_patterns:
            if re.search(pattern, q):
                return False
                
        # 3. Check if the query matches a pattern asking for a definition/fact
        info_patterns = [
            r"^(what|who) (is|are|was|were|refers to|means)\b",
            r"^(what|who|where|when|which|how) \w+\b",  # "who won...", "where is...", "when was X built", "which city...", "how do X..."
            r"^define\b",
            r"^explain\b",
            r"^tell me about\b",
            r"^do you know\b",
            r"^what do you know about\b",
        ]
        if any(re.match(pat, q) for pat in info_patterns):
            return True
            
        # If it's a question but didn't match the reasoning patterns or explicit informational patterns,
        # we err on the side of conversational/reasoning to let RAVANA chat like a human.
        return False


    # ===================================================================
    # PHASE: ERROR CORRECTION CIRCUIT (ACC/DA VTA/Hippocampus/PFC analog)
    # ===================================================================

    def _process_correction_feedback(self, correction):
        """Phase 2-4: Convert detected correction into system-wide negative prediction error.

        DA VTA/SNc analog: dopamine dip signals the pathway that produced the
        incorrect response should be weakened.

        1. Raise free energy (uncertainty spikes)
        2. Set basal ganglia prediction error (raises Go threshold)
        3. Weaken identity confidence
        4. Weaken graph edges that led to the incorrect response
        5. Trigger epistemic mode switch
        6. Log the correction
        """
        if self._trace_enabled:
            print(f"  [correction] Processing feedback: {correction.correction_type.value} "
                  f"severity={correction.severity:.2f} subject='{correction.subject}'")

        # Phase 2a: Free energy spike (uncertainty increases)
        self._free_energy = max(0.5, self._free_energy + 0.3 * correction.severity)

        # Phase 2b: Basal Ganglia prediction error (NoGo gate raised)
        error_signal = min(0.95, 0.8 * correction.severity)
        self.basal_ganglia.set_prediction_error(error_signal)
        if self._trace_enabled:
            print(f"  [correction][BG] set_prediction_error({error_signal:.2f}) - NoGo threshold raised")

        # Phase 2c: Identity confidence decreases
        identity_delta = -0.1 * correction.severity
        if hasattr(self, 'identity') and self.identity is not None:
            old_strength = self.identity.state.strength
            self.identity.state.strength = max(0.05, self.identity.state.strength + identity_delta)
            if self._trace_enabled:
                print(f"  [correction][ID] strength {old_strength:.2f} -> {self.identity.state.strength:.2f}")

        # Phase 2d: Weaken edges used to generate the response
        self._weaken_edges_for_response(correction)

        # Phase 4: Epistemic mode switch (PFC behavioral adjustment)
        # Switch to CAUTIOUS or RECOVERY after correction
        correction_current_mode = getattr(self.meta_cog, 'current_mode', None)
        if correction_current_mode and correction_current_mode not in (
            EpistemicMode.CAUTIOUS, EpistemicMode.RECOVERY):
            self.meta_cog.current_mode = EpistemicMode.CAUTIOUS if correction.severity < 0.7 else EpistemicMode.RECOVERY

        # Mark correction as processed
        correction.resolved = True

    def _weaken_edges_for_response(self, correction):
        """Phase 3: Weaken graph edges that contributed to the incorrect response.

        Hippocampal reconsolidation analog: retrieve the memory (edges),
        destabilize (weaken), prepare for update.

        Strategy: find edges from the subject concept to its top associations
        and weaken them proportionally to correction severity.
        """
        subj_lower = correction.subject.lower()
        subj_ids = self._concept_keywords.get(subj_lower, [])
        if not subj_ids:
            if self._trace_enabled:
                print(f"  [correction] No graph nodes for '{subj_lower}', skipping edge weakening")
            return

        # Also delegate to chain_walker methods for additional edge weakening
        try:
            self._weaken_edges_for_correction(subj_lower, correction.severity * 0.5)
            last_hops = self._last_chain_hops[-1] if self._last_chain_hops else []
            self._mark_edges_as_incorrect(subj_lower, last_hops)
        except Exception:
            pass

        weaken_factor = 1.0 - 0.5 * correction.severity  # e.g. 0.7 for severity=0.6
        weakened_count = 0

        for src_nid in subj_ids:
            # Weaken outgoing edges
            for tgt_nid, edge in list(self.graph.get_outgoing(src_nid)):
                old_weight = edge.weight
                edge.weight *= weaken_factor
                tgt_node = self.graph.get_node(tgt_nid)
                if tgt_node and tgt_node.label:
                    correction.weakened_edges.append((src_nid, tgt_nid))
                weakened_count += 1
                if self._trace_enabled and weakened_count <= 3:
                    tgt_label = self.graph.get_node(tgt_nid).label if self.graph.get_node(tgt_nid) else '?'
                    print(f"  [correction][edge] {subj_lower}->{tgt_label}: {old_weight:.3f} -> {edge.weight:.3f}")

            # Weaken incoming edges
            for src, edge in list(self.graph.get_incoming(src_nid)):
                edge.weight *= weaken_factor
                weakened_count += 1

        if self._trace_enabled:
            print(f"  [correction] Weakened {weakened_count} edges for '{subj_lower}'")

        # Queue the subject for web learning to get corrected knowledge
        with self._bg_lock:
            if subj_lower not in self._bg_learning_queue and subj_lower not in self._pending_learning_queue:
                self._pending_learning_queue.append(subj_lower)

        # Add corrected fact to belief store if available
        if correction.corrected_fact:
            fact_subj, fact_rel, fact_val = correction.corrected_fact
            self.belief_store.assert_belief(fact_subj, fact_rel, fact_val, confidence=0.9)
            # Add graph edge for corrected fact
            fact_nids = self._concept_keywords.get(fact_subj.lower(), [])
            val_nids = self._concept_keywords.get(fact_val.lower(), [])
            if fact_nids and val_nids:
                existing = self.graph.get_edge(fact_nids[0], val_nids[0])
                if existing is None:
                    self.graph.add_edge(fact_nids[0], val_nids[0],
                                        weight=0.5, relation_type="semantic")
                    correction.added_edges.append((fact_nids[0], val_nids[0]))
                    if self._trace_enabled:
                        print(f"  [correction] Added corrected edge: {fact_subj} -> {fact_val}")
                else:
                    existing.weight = min(0.7, existing.weight + 0.2)
                    if self._trace_enabled:
                        print(f"  [correction] Boosted existing edge: {fact_subj} -> {fact_val}")

    def _detect_and_handle_correction(self, user_input, subject, response, strategy, quality_score):
        """Phase 6: Full correction detection and processing pipeline.

        Called after response generation to check if the user is correcting RAVANA.
        Returns the correction apology/acknowledgment response or None.
        """
        if not self.user_model.detected_correction:
            return None

        correction_type = self.user_model.detected_correction_type
        severity = self.user_model.correction_severity

        # Build correction record
        correction = Correction(
            turn=self.turn_count,
            correction_type=correction_type,
            subject=subject or self.user_model.correction_subject,
            incorrect_response=response or "",
            user_correction_text=user_input,
            corrected_fact=self.user_model.detected_correction_fact,
            severity=severity,
        )

        # Log the correction
        self._correction_log.append(correction)
        if self._trace_enabled:
            print(f"  [correction] Detected {correction_type.value} correction "
                  f"(severity={severity:.2f})")

        # Process the correction through the full circuit
        self._process_correction_feedback(correction)

        # Store for sleep consolidation
        correction.resolved = False

        # Generate acknowledgment response
        # If user provided corrected fact, acknowledge it specifically
        if correction.corrected_fact:
            fact_subj, fact_rel, fact_val = correction.corrected_fact
            ack = f"thanks for correcting me. i'll remember that {fact_subj} {fact_rel} {fact_val}."
        elif severity > 0.6:
            ack = "thanks for the correction. i'm still learning and appreciate your feedback."
        else:
            ack = "got it, thanks for the feedback. i'll keep that in mind."

        # Reset user model correction flags for next turn
        self.user_model.reset_correction_flags()

        if self._trace_enabled:
            print(f"  [correction] Generated acknowledgment: '{ack}'")

        return ack.lower()


    def _consolidate_corrections_in_sleep(self):
        """Phase 5: During sleep, consolidate corrections into long-term memory.
Strengthen newly added correct edges (Hebbian replay).
Further weaken old incorrect edges (synaptic pruning).
If a concept has been corrected 3+ times, mark for priority web learning.
        """
        if not self._correction_log:
            return {'corrections_consolidated': 0}

        consolidated = 0
        correction_strengthened = 0
        correction_pruned = 0

        # Count corrections per subject
        subject_correction_count = {}
        for c in self._correction_log:
            if c.resolved:
                continue
            subj = c.subject.lower()
            subject_correction_count[subj] = subject_correction_count.get(subj, 0) + 1

            # Strengthen newly added correct edges (Hebbian replay)
            for src, tgt in c.added_edges:
                edge = self.graph.get_edge(src, tgt)
                if edge:
                    edge.weight = min(0.7, edge.weight * 1.3)
                    correction_strengthened += 1

            # Further weaken old incorrect edges (synaptic pruning)
            for src, tgt in c.weakened_edges:
                edge = self.graph.get_edge(src, tgt)
                if edge:
                    edge.weight *= 0.7
                    if edge.weight < 0.05:
                        self.graph.remove_edge(src, tgt)
                        correction_pruned += 1

            c.resolved = True
            consolidated += 1

        # Mark concepts corrected 3+ times for priority web learning
        for subj, count in subject_correction_count.items():
            if count >= 3 and subj not in self._pending_learning_queue:
                with self._bg_lock:
                    self._pending_learning_queue.append(subj)
                if self._trace_enabled:
                    print(f"  [sleep] Concept '{subj}' corrected {count}x - priority web learning queued")

        if self._trace_enabled and consolidated > 0:
            print(f"  [sleep] Consolidated {consolidated} corrections: "
                  f"{correction_strengthened} edges strengthened, "
                  f"{correction_pruned} edges pruned")

        # Clean up resolved corrections (keep last 50)
        self._correction_log = [c for c in self._correction_log if not c.resolved]
        if len(self._correction_log) > 50:
            self._correction_log = self._correction_log[-50:]

        return {
            'corrections_consolidated': consolidated,
            'correction_edges_strengthened': correction_strengthened,
            'correction_edges_pruned': correction_pruned,
        }
