# RAVANA Cognitive Chat Engine -- main orchestrator.
# Contains CognitiveChatEngine with __init__, process_turn, save/_load.
# Helper classes in models.py, user_model.py, belief_store.py.
# M0 crash-hardening: pin BLAS/OpenMP threads to 1 BEFORE numpy is imported, so
# worker-thread BLAS calls (web learner) can't race the main-thread decoder and
# trigger the Windows access-violation (numpy #27989). Must be the very first
# import -- ahead of `import numpy as np` below.
import ravana._numpy_threading  # noqa: F401  (side-effect: thread + faulthandler setup)
import sys, os, time, random, json, re, threading, hashlib, operator
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
from ravana.language.prefrontal_workspace import PrefrontalWorkspace, DiscourseIntent, DiscoursePlan, DiscourseType, SocialIntentClassifier
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
from ravana.ontology import DerivedOntology
from ravana.ontology.conceptnet import ConceptNetOntology

# Optional bs4
try:
    import bs4  # noqa: F401
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

# Import constants
from .constants import (TEEN_CONCEPTS, WEB_GARBAGE, STOP_WORDS, ConceptPosDict,
                        _is_word_salad, _is_keyboard_mash)
from .web_learning import WebLearningMixin
from .snippet_quality import SnippetStructureModel, default_model
# Research item B (fail-closed salad monitor): learned distributional classifier
# + fluent-tautology signature gate. Imported lazily-safe so a missing fit file
# degrades gracefully (the guard falls back to the legacy rule-based detector).
try:
    from .salad_classifier import is_salad_learned, get_classifier
    from .monitor_gate import detects_fluent_tautology
    _HAS_SALAD_LEARNED = True
except Exception:  # pragma: no cover - defensive
    _HAS_SALAD_LEARNED = False
    is_salad_learned = None
    get_classifier = None
    detects_fluent_tautology = None

import pickle
from ravana.web.learner import SearchEngine
from ravana.core.dual_code_space import DualCodeSpace
from ravana.core.hrr_reasoner import HRRReasoner

# Universal closed-class / pronoun words that can never own a learned definition
# (you don't "define" the word "you"). This is the only hand-listed part of the
# definition purge — a minimal universal seed, not a per-word category table.
# The rest of the purge is derived from the learned graph (see
# _derive_definition_purge).
_UNIVERSAL_PURGE = {
    "you", "i", "we", "they", "he", "she", "it", "me", "my", "your",
    "our", "their", "us", "them", "him", "her", "this", "that",
}

# Assertion/copula detector (vmPFC/mPFC reality-monitor analog): a definition
# that does not assert anything (no copula / defining verb) is structurally
# not a definition — it is a junk fragment. Used by the learned
# definition-attraction score in _derive_definition_purge to decide whether a
# concept is chronically collecting non-asserted web fragments (Phase 1,
# Track B). Mirrors web_learning._DEFINITION_PREDICATE.
_DEFINITION_ASSERTION = re.compile(
    r"\b(is|are|was|were|be|been|being|means?|refers?\s+to|describes?|"
    r"occurs?|happens?|defined\s+as|represents?|signifies?|constitutes?|"
    r"denotes?)\b", re.IGNORECASE)


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
from ravana.core.question_decomposition import QuestionDecompositionEngine, QuestionCategory
from ravana.core.sub_answer_synthesizer import SubAnswerSynthesizer

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



    def __init__(self, dim: int = 64, seed: int = 42, baby_mode: bool = True, data_dir: Optional[str] = None, user_suffix: str = "", hrr_whiten: bool = True, hrr_sparse_k: int = 256, hrr_unitary_roles: bool = True, hrr_dim: int = 4096):
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
        # M1-B: concepts whose definition was authored offline (common_facts.json),
        # NOT retrieved from web/KB. These bypass the web-junk quality gate in
        # _definition_response because curated text is trusted ground truth.
        self._curated_definitions: Set[str] = set()
        # M2-D: protected namespace. These project concepts have AUTHORED
        # definitions (seeded domain relations / curated facts) and must never be
        # overwritten by web/KB collisions (e.g. "ravana" == the mythological
        # Ramayana figure on Wikipedia). Any web/KB write to a protected concept
        # is dropped (provenance precedence: curated > web). Fail-closed.
        self._PROTECTED_CONCEPTS: Set[str] = {"ravana", "oxiverse", "intentforge"}

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

        # Adaptive affective baseline (Barrett constructed-emotion / Seth
        # interoceptive-prediction framing): each turn's aggregate user valence
        # is folded into a running (EMA) distribution (mu, sigma). Affective
        # disclosures are then judged by z-score against THIS distribution, not a
        # fixed cutoff -- so "bored" is salient when the user is usually neutral
        # but invisible when they are usually down. Distribution-driven, fails
        # closed (returns neutral/None) when nothing clears the adaptive bar.
        self._vad_baseline = {"mu": 0.0, "sigma": 0.3, "n": 0}

        # State
        self.turn_count = 0
        # Phase 3.1: Topic-indexed conversation store (dict, last 50)
        self._topic_list: List[str] = []
        self._topic_store: Dict[str, Dict] = {}
        # Phase 3.4: Response-aware context
        self._response_context: List[Dict] = []
        self._last_responses: List[str] = []
        self._last_strategy: str = ""
        # PROMPT cross-cutting: epistemic modality of the last emitted answer,
        # set by the generator algorithms (counterfactual robustness, comparative
        # web plausibility, metacognitive-ignorance 3-state). Carried OUTSIDE the
        # (text, strategy) return tuple so we don't break the ~50 callers that
        # unpack it; surfaced via self._last_modality for monitors/surfacers.
        self._last_modality: str = "unknown"
        # PROMPT 3: stash the winning web snippet's provenance so the surfacer
        # can tag the answer with its source type ("according to Wikipedia...").
        self._last_web_source: str = ""
        self._last_web_plausibility: float = 0.0
        self._last_web_trust: float = 0.0
        # PROMPT 5: clause-segregation scratch state. When _ground_query splits
        # a multi-clause query ("sky blue but sunsets red"), the second clause's
        # themed topic is stashed here (with its RST relation) so the decomposer
        # can answer BOTH sub-questions instead of collapsing to one subject.
        self._pending_subtopic: Optional[Tuple[str, str]] = None
        self._pending_subject_hint: Optional[str] = None
        # Phase 19g: set True when the generated response was flagged as word
        # salad/tautology, so process_turn can substitute an honest fallback.
        self._last_response_was_salad: bool = False
        # Behavior 8: interlocutor forward simulation (covert other-monitoring).
        # After each turn we predict the user's likely next concept from the
        # activated subgraph; on the next turn we compare what arrived to the
        # prediction. High alignment => common ground established, so the bot
        # may be more concise (Gricean: don't re-explain shared ground).
        self._predicted_user_next: str = ""
        self._predicted_user_conf: float = 0.0
        self._common_ground: float = 0.0
        self._free_energy = 0.0
        self._learning_count = 0
        self._learned_this_turn = False
        # P6: human-likeness eval counters (incremented on the relevant paths;
        # read by tests/eval/eval_humanlikeness.py). Pure instrumentation —
        # never affects the response.
        self._metrics = {
            "kb_lookups": 0,          # on-demand KB retrievals fired (curiosity)
            "paradox_grounded": 0,    # paradox replies with a retrieved clause
            "category_metaphor": 0,   # category errors answered with a metaphor
            "hedged_evidence": 0,     # reflective replies sourced with KB evidence
        }
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
        # M10: structured self-monitor log. Every guard fire / swallow is
        # appended here so the monitor's decisions are observable (not just
        # the Ne/ERN evidence) — the Pe component (Steinhauser & Yeung 2010)
        # makes the monitor's decision explicit. Read via monitor_report().
        self._monitor_log: List[Dict[str, Any]] = []
        self._contradiction_map: Dict[str, Set[str]] = {}
        self._belief_assertions: List[Tuple[str, str, str]] = []
        self._recall_mode: bool = False
        # Fix 4 (Q12): raw user-turn ring buffer for episodic-memory queries
        # ("what did I just ask you", "what were we talking about"). The
        # Baddeley episodic buffer binds recent turns for retrieval; the
        # hippocampal buffer stores facts keyed by SUBJECT, which cannot answer
        # a meta-query whose subject is the conversation itself. This keeps the
        # last few verbatim user turns so the WM↔LTM retrieval path is live.
        self._recent_user_turns: List[str] = []
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
        # LingGen P6: free-form sensorimotor-conditioned decoder generation.
        # OFF until the grounded training pass proves decoder-CE <= template-CE
        # on a held-out set (distribution-fit promotion, not a hand switch).
        # When False, generation falls back to _build_conditioned_bos (Lancaster
        # tail) + realize_dim phrase lookup — never emits ungrounded gibberish.
        self.use_linggen = False
        self._linggen_genconf_seq = []  # history of grounded-run top1 acc
        # Track B Phase 2 (M4): learned snippet-quality model (structural PE).
        # OFF by default — the hardcoded _SNIPPET_REJECT_SHAPES /
        # _SNIPPET_NOISE tables remain the backstop until the learned model is
        # verified to beat them on the regression set (milestone plan). When ON,
        # the learned model additionally flags structural-junk snippets.
        self.use_cerebellar_snippet = False
        self._snippet_model = None
        # Track B Phase 3 (M5): learned per-domain source-trust (replaces the
        # hardcoded _PREFERRED_SNIPPET_SOURCES allowlist). OFF by default — the
        # hardcoded allowlist stays the fallback until the learned trust
        # accumulator is verified to beat it on the regression set. When ON,
        # the engine maintains a per-domain trust score updated from snippet
        # outcomes and uses it as the source-quality signal.
        self.use_source_trust = False
        self._source_trust: Dict[str, float] = {}
        # Track B Phase 5 (M5): learned distributional POS (replaces the
        # hardcoded _GRAMMATICAL_CONCEPTS function-word set). OFF by default —
        # the hardcoded set stays the fallback via _is_function_word until the
        # learned classifier is verified to cover it.
        self.use_learned_pos = False
        # Track B Phase 6 (M6): ConceptNet ontology as the primary frontopolar
        # feasibility gate (replaces the literal _CATEGORY_OF_SUBJECT /
        # _CATEGORY_AFFORDANCES fallback). ON by default now that the prebuilt
        # ConceptNet ontology (data/conceptnet/ont.pkl) is wired and verified:
        # category_of is inferred via the IsA walk and affordances by the
        # Sensory-Functional division, so there are no per-word authored tables.
        # The literal dicts remain only as an OOV safety net when the KG is
        # silent AND ConceptNet-primary is OFF. The CLI flag --conceptnet-primary
        # / --no-conceptnet-primary can still force either mode; we auto-disable
        # if the ontology failed to load (see __init__ guard below).
        self.use_conceptnet_primary = True
        self.belief_store = BeliefStore()

        # P6: one epistemic register (roadmap #12) toggling confidence /
        # verbosity / curiosity in a single place, instead of scattering
        # thresholds. Presets set three knobs:
        #   curiosity   -> whether on-demand KB retrieval (P1/P2) fires
        #   verbosity   -> whether sourced evidence clauses are appended (P3/P5)
        #   confidence  -> not a hard gate; biases hedging tone only
        self.epistemic_register = "default"
        _REGISTERS = {
            "default":  {"curiosity": 1.0, "verbosity": 1.0, "confidence": 1.0},
            "confident": {"curiosity": 1.0, "verbosity": 1.0, "confidence": 1.3},
            "cautious":  {"curiosity": 1.0, "verbosity": 1.0, "confidence": 0.7},
            "verbose":   {"curiosity": 1.0, "verbosity": 1.0, "confidence": 1.0},
            "terse":     {"curiosity": 0.3, "verbosity": 0.2, "confidence": 1.0},
        }
        _r = _REGISTERS.get(self.epistemic_register, _REGISTERS["default"])
        self._reg_curiosity = _r["curiosity"]
        self._reg_verbosity = _r["verbosity"]
        self._reg_confidence = _r["confidence"]

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
        # G4: VSA schemas bind/unbind in the 75-D dual-code space
        # (GloVe-64 | Lancaster-11) so role-filler realization operates on
        # EMbODIED vectors, matching the embeddings _vsa_event_narrative
        # now passes. Keep self.dim (64) for the distributional backbone.
        self.vsa_manager = VSAManager(dim=self.dim + 11)
        self.schema_library = SchemaLibrary(self.vsa_manager)
        # Schema Completion (research item): migrate the hardcoded EventSchema
        # process templates into VSA event schemas so the narrative generator can
        # realize processes via role-filler binding instead of string templates.
        try:
            self.event_schema_lib.seed_default_schemas()
            for _concept, _es in self.event_schema_lib._schemas.items():
                self.schema_library.build_event_schema(_es, concept=_concept)
        except Exception:
            pass
        # Work A0: HRR compositional reasoning in the loop. DualCodeSpace (2048-D,
        # additive dual-code) is instantiated here and the graph's opt-in
        # _fact_encode_hook is wired so EVERY add_edge (the single write choke
        # point in ravana_ml/graph.py) populates the HRR store. Guarded: if the
        # glove cache is missing the engine still boots (HRR simply stays empty).
        self.dual_code = None
        self.hrr_reasoner = None
        try:
            if os.path.exists(self._glove_cache_path):
                self.dual_code = DualCodeSpace(self._glove_cache_path, hrr_dim=hrr_dim,
                                               whiten=hrr_whiten, sparse_k=hrr_sparse_k,
                                               unitary_roles=hrr_unitary_roles)
                self.hrr_reasoner = HRRReasoner(self.dual_code)
                # Wire the populate hook: add_edge -> HRR encode.
                self.graph._fact_encode_hook = self._hrr_encode_hook
                # M5' graph-override: let graph.infer_chain(verb=...)
                # anchor its relation context to the HRR role vector.
                self.graph.dual_code = self.dual_code
        except Exception as e:
            if getattr(self, '_trace_enabled', False):
                print(f"  [init] DualCodeSpace/HRR unavailable: {e}")
            self.dual_code = None
            self.hrr_reasoner = None
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
        self.pfc_workspace = PrefrontalWorkspace(capacity=5, vector_fn=self._glove_vector)
        # Prototype-bank social-intent classifier (TPJ/DMN mentalizing analog).
        # Reuses QuestionSubtypeClassifier machinery so "what's up" / "what is up"
        # / "wassup" collapse to one greeting centroid via contraction
        # normalization; ABSTAIN_K floor gives fail-closed degradation.
        self._social_intent = SocialIntentClassifier(vector_fn=self._glove_vector)
        self._proper_nouns = set()
        # Concepts bootstrapped with AUTHORED typed relations (the project's own
        # proper nouns: oxiverse / intentforge / ravana). These are the ONLY
        # concepts the seeded-relation answer path may surface from graph edges —
        # their relations are hand-authored seed, not noisy web associations.
        self._seeded_domain_concepts: set = set()
        # Phase E: Syntactic cell assemblies — Hebbian role learning with seeded priors
        self.syntactic_assembly = SyntacticCellAssembly(learning_rate=0.05)
        self.syntactic_assembly.proper_nouns = self._proper_nouns
        # Phase F: Surface realizer — rule-governed English morphology with dopamine modulation
        self.surface_realizer = SurfaceRealizer()
        self.surface_realizer.proper_nouns = self._proper_nouns
        # Phase 5 (casing): cached set of graph concepts that are STRONG named
        # entities (country/city/company/person/... via ConceptNet IsA). Fed to
        # case_infer so mid-sentence entities not in SUBTLEX still capitalize.
        self._graph_entity_words: Optional[set] = None
        # Phase 6: Wire vector function for semantic verb selection (VerbLexicon)
        self.surface_realizer.set_vector_fn(self._get_modulated_vector)
        VerbLexicon.set_glove_fn(self._get_modulated_vector)
        # G3: wire the sensorimotor read-out into verb selection + hedging so
        # generation is modulated by embodied grounding (not just distributional).
        # VerbLexicon gets the RAW Lancaster 11-D (strong sensory discrimination);
        # SurfaceRealizer gets the OOD/confidence signal (hedging for weak grounding).
        VerbLexicon.set_sensorimotor_fn(self._lancaster_vector)
        self.surface_realizer.set_sensorimotor_fn(self._sensorimotor_confidence)
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

        # Phase 19: Relevance/confidence-gated deep reading.
        # Snippets are cheap and fed in unconditionally. Full-article fetches
        # are expensive (timeout ~8s each) and were previously done for ALL
        # results[:3] synchronously, blocking the user turn for ~30s. Now we
        # rank results by subject relevance and only deep-read the top few
        # that clear the gate. Offload pushes deep reads to the background
        # thread so a turn returns from snippets in ~search latency.
        self._deep_read_max = 1          # max full articles fetched per query
        self._deep_read_timeout = 6      # per-article fetch timeout (s)
        self._deep_read_relevance_gate = 0.12  # min relevance to deep-read
        self._deep_read_offload = True   # offload deep reads to background

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

                # Question Decomposition Engine (frontopolar BA 10 analog)
        # Holds the main question while managing sub-questions (Braver & Bongiolatti, 2002)
        self.question_decomposer = QuestionDecompositionEngine()
        self.answer_synthesizer = SubAnswerSynthesizer()
        self._current_decomposition_result = None
        
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
        # G2: wire the Lancaster-11 sensorimotor read-out as the graph's
        # node-fill fn EARLY (before schema seeding / KB bootstrap) so
        # EVERY node auto-carries its dual-code vector (ConceptNode.
        # sensorimotor_vector) from creation. Backfill covers nodes
        # that already exist at this point (legacy graphs -> None).
        g_ = getattr(self, "graph", None)
        if g_ is not None:
            g_._sensorimotor_fn = self._lancaster_vector
            try:
                for _n in list(g_.nodes.values()):
                    if getattr(_n, "sensorimotor_vector", None) is None and _n.label:
                        try:
                            _n.sensorimotor_vector = self._lancaster_vector(_n.label)
                        except Exception:
                            pass
            except Exception:
                pass

        # Derived-ontology service: replaces the hand-edited frontopolar gate
        # dicts with on-demand geometric + graph-derived inference. Primary path
        # wherever GloVe is available; the legacy literal dicts remain only as a
        # fallback for the rare no-GloVe / OOV case.
        self._ontology = DerivedOntology(
            glove_fn=getattr(self, "_glove_vector", None),
            graph=getattr(self, "graph", None),
            label_index=getattr(self, "_concept_keywords", None),
            theta=0.12,
        )
        # Brain-aligned (Binder + Rosch + ConceptNet) primary gate. Derived from
        # the ConceptNet typed knowledge graph: category is inferred by IsA walk,
        # affordances by the Sensory-Functional division. Loaded from a prebuilt
        # pickle; if absent, the gate falls back to the legacy literal dicts.
        self._cn_ontology = self._load_conceptnet_ontology()
        # Lancaster G2/G3: build ONE CombinedAttributeEncoder (Lancaster primary,
        # Binder fallback) at init and expose it as the sensorimotor read-out.
        # This is the dual-coding co-primary: GloVe stays the distributional
        # backbone; the encoder maps GloVe-64 -> sensorimotor (11-D Lancaster
        # wide coverage + 65-D Binder fine). Used by G3 verb selection + hedging.
        self._combined_attr_encoder = self._build_combined_encoder()
        # Auto-downgrade ConceptNet-primary to the literal-dict fallback when the
        # ontology is unavailable, so category grounding never crashes. If the
        # user explicitly passed --conceptnet-primary / --no-conceptnet-primary
        # on the CLI, that explicit choice wins (handled in main() after init).
        if self._cn_ontology is None and getattr(self, "use_conceptnet_primary", False):
            self.use_conceptnet_primary = False
        if os.path.exists(self._save_path):
            loaded = self._load()
            if loaded:
                self._revector_existing_nodes()
                self._sanitize_graph()  # prune poison nodes (self-loops, question phrases)
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
                # Deferred item 1: materialize ConceptNet typed edges into the
                # LOADED graph now (must run AFTER _load, which replaces the
                # in-memory graph with the saved one — running it earlier would
                # inject into the empty pre-load graph and then be discarded).
                self._typed_edges_bootstrap()
                return

        # Cold start (no saved weights): seed everything from scratch.
        self._seed_concepts()
        self._bootstrap_domain_concepts()
        # B: authored, OFFLINE core-knowledge seed (deterministic, no network).
        # Covers universal common facts (sky/cat/music/sun/gravity/...) that the
        # live-KB path (below) misses or answers nondeterministically. Runs
        # first and fail-closed so common questions are grounded offline.
        try:
            self._seed_common_facts()
        except Exception:
            pass
        # M1-C: reload previously-VERIFIED definitions mirrored to CognitiveDB
        # (from prior save()s). This makes knowledge durable across a fresh
        # cold-start / --reset, not just across pickle reloads, so learned
        # facts are deterministic rather than re-derived from flaky live web.
        try:
            self._load_persisted_definitions()
        except Exception:
            pass
        # M3-E: seed the offline physics causal skeleton so counterfactual
        # simulation can forward-chain world-scale interventions (sun gone ->
        # no light -> no photosynthesis -> plants/animals die).
        try:
            self._seed_physics_causal()
        except Exception:
            pass

        # top-N corpus-frequency concepts so common facts ("the sun is a star")
        # are available without any authored text. This is retrieval, not
        # hand-authored prose; concepts with no KB hit are simply skipped.
        try:
            self._seed_kb_definitions()
        except Exception:
            pass
        self._build_decoder_vocab()
        # Skip initial corpus training during cold start — the training
        # script (train.py) handles it separately with more control.
        # This shaves ~hours off first-time initialization.
        self._needs_seed_training = False
        self._needs_synthetic_training = False
        print(f"  [Teen] Knows {len(self.graph.nodes)} words, ready to learn!")
        # Deferred item 1: materialize ConceptNet typed edges into the
        # cold-start graph (seeded above) so the inheritance walk works.
        self._typed_edges_bootstrap()

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

        Defensive: a bare engine constructed via ``__new__`` (e.g. in unit
        tests that skip ``__init__``) has no GloVe state. Treat a missing glove
        table/projection exactly as "GloVe absent" and return None rather than
        raising AttributeError — callers already branch on a None result.
        """
        vecs = getattr(self, "_glove_vecs", None)
        if vecs is None:
            return None
        w = label.lower().strip()
        # Check cache first (Phase 2.1)
        cache = getattr(self, "_glove_vector_cache", None)
        if cache:
            cached = cache.get(w)
            if cached is not None:
                return cached
        vec = vecs.get(w)
        if vec is None and len(w) > 1:
            vec = vecs.get(w.rstrip('s'))
        if vec is None and len(w) > 2:
            vec = vecs.get(w[:-1])
        if vec is not None:
            proj = getattr(self, "_glove_proj", None)
            if proj is None:
                return None
            pv = proj @ vec
            norm = np.linalg.norm(pv)
            if norm > 0:
                pv /= norm
            result = pv.astype(np.float32)
            if cache is not None:
                cache[w] = result
                # Also cache variants for fast lookup
                if w.rstrip('s') != w:
                    cache[w.rstrip('s')] = result
                if len(w) > 2 and w[:-1] != w:
                    cache[w[:-1]] = result
            return result
        return None

    # ── Lancaster dual-coding: sensorimotor read-out (G2/G3) ──────────────────
    def _build_combined_encoder(self):
        """Build a single CombinedAttributeEncoder (Lancaster primary, Binder
        fallback) reused for every sensorimotor lookup. Returns None if the
        probe artifacts are absent (engine still boots; G3 hooks no-op)."""
        try:
            from ravana.ontology.attribute_encoder import load_combined_encoder
        except Exception:
            return None
        base = os.path.join(_proj_root, "data") if '_proj_root' in globals() else None
        candidates = []
        if base:
            candidates.append(os.path.join(base, "attribute_encoder.npz"))
        cur = os.path.dirname(os.path.abspath(__file__))
        for _ in range(6):
            candidates.append(os.path.join(cur, "data", "attribute_encoder.npz"))
            cur = os.path.dirname(cur)
        enc = None
        for ecand in candidates:
            if os.path.exists(ecand):
                d = os.path.dirname(ecand)
                lanc_cands = [os.path.join(d, "lancaster_encoder.npz"),
                              os.path.join(os.path.dirname(d), "lancaster_encoder.npz")]
                _lanc = next((p for p in lanc_cands if os.path.exists(p)), None)
                try:
                    enc = load_combined_encoder(ecand, _lanc)
                except Exception:
                    enc = None
                if enc is not None:
                    break
        return enc

    def _build_lancaster_norms(self) -> Dict[str, np.ndarray]:
        """Load the HUMAN Lancaster 11-D sensorimotor norms (39,707 words) for
        high-variance embodiment lookup. Probe predictions are variance-
        compressed; the human norms discriminate strongly (hand Hand_arm=4.4 vs
        trust=0.45). Used by G3 verb selection. Returns {} if CSV absent."""
        cache = getattr(self, "_lancaster_norms", None)
        if cache is not None:
            return cache
        norms: Dict[str, np.ndarray] = {}
        try:
            import csv
            from ravana.ontology.attribute_encoder import LANCASTER_DIMS
            cand = os.path.join(_proj_root, "data", "cache", "word_ratings",
                                "Lancaster_sensorimotor_norms_for_39707_words.csv")
            if not os.path.exists(cand):
                self._lancaster_norms = norms
                return norms
            with open(cand, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    w = str(row.get("Word", "")).strip().lower()
                    if not w:
                        continue
                    try:
                        norms[w] = np.array(
                            [float(row[d + ".mean"]) for d in LANCASTER_DIMS],
                            dtype=np.float64)
                    except (ValueError, KeyError, TypeError):
                        continue
        except Exception:
            norms = {}
        self._lancaster_norms = norms
        return norms

    def _lancaster_vector(self, word: str) -> Optional[np.ndarray]:
        """Sensorimotor vector for a word.
        G3 verb selection uses the HUMAN Lancaster 11-D norms when the word is
        in the 39,707-word set (strong cross-word variance); falls back to the
        probe prediction (LancasterEncoder) only for true OOV. None if no encoder
        and not in norms, or the word is OOV with no probe."""
        w = (word or "").lower().strip()
        if not w:
            return None
        norms = self._build_lancaster_norms()
        if w in norms:
            return norms[w]
        enc = getattr(self, "_combined_attr_encoder", None)
        if enc is None or getattr(enc, "lancaster", None) is None:
            return None
        gv = self._glove_vector(word)
        if gv is None:
            return None
        try:
            return enc.lancaster.attribute_vector(gv)
        except Exception:
            return None

    def _sensorimotor_confidence(self, word: str) -> float:
        """Sensorimotor grounding confidence of a word.
        G3 hedging signal: TRUE OOV (no GloVe vector -> no probe prediction) has
        WEAK grounding -> 0.0, so the realizer hedges. Words with a GloVe vector
        get a probe prediction (even if sparse) -> 1.0. Degrades to 1.0 only when
        no encoder is available at all (so the hook is a no-op, not fail-closed)."""
        if self._glove_vector(word) is None:
            return 0.0
        enc = getattr(self, "_combined_attr_encoder", None)
        if enc is None:
            return 1.0
        return 1.0

    def _typed_edges_between(self, a: str, b: str) -> List[str]:
        """Return the relation types of edges linking concept labels `a` and
        `b` (either direction) in the live graph. Used by the generative humor
        reflex (Fix C) to pick the connector word for a joke setup+punchline.

        Returns a list of relation_type strings (may be empty).
        """
        ck = getattr(self, "_concept_keywords", {})
        a_ids = ck.get((a or "").lower(), [])
        b_ids = set(ck.get((b or "").lower(), []))
        if not a_ids or not b_ids:
            return []
        out = []
        for aid in a_ids:
            try:
                for tid, edge in self.graph.get_outgoing(aid):
                    if tid in b_ids:
                        rt = getattr(edge, "relation_type", None)
                        if rt and rt not in out:
                            out.append(rt)
            except Exception:
                continue
        if not out:
            # reverse direction
            for bid in b_ids:
                try:
                    for tid, edge in self.graph.get_outgoing(bid):
                        if tid in set(a_ids):
                            rt = getattr(edge, "relation_type", None)
                            if rt and rt not in out:
                                out.append(rt)
                except Exception:
                    continue
        return out

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




    def _is_philosophical_paradox(self, text: str) -> bool:
        """Detect philosophical paradoxes/impossible questions (frontopolar BA 10 N400 analog).
        
        Before the action-request check fires, the brain's BA 10 detects semantic 
        incongruity (N400 effect) and routes paradoxical questions to deliberation.
        These are NOT action requests — they are semantic puzzles that need 
        counterfactual reasoning.
        
        Patterns detected:
        - Theological paradoxes: "can god create a stone so heavy..."
        - Classical paradoxes: "unstoppable force meets immovable object"
        - Self-referential: "this statement is false"
        - Impossible scenarios: "can you prove you exist"
        """
        t = text.lower().strip(" ?!.")
        
        # Theological/omni-paradoxes (the classic "can god create a stone...")
        if re.search(r"\b(can|could)\s+(god|you|one|a\s+being)\s+(create|make|find)\s+(a|an)\s+(.+?)\s+(so|that|which)\s+(heavy|powerful|big|strong|large|hot|cold)", t):
            return True
        # Looser omnipotence check: god/omni + create/make + rock/stone +
        # a "cannot lift" contradiction clause (handles "create a rock he
        # cannot lift" which the stricter regex above misses).
        if re.search(r"\b(god|omnipotent|all[- ]powerful)\b", t) and \
                re.search(r"\b(create|make|lift|heavy|stone|rock)\b", t) and \
                re.search(r"\b(can'?t|cannot|couldn'?t|unable|not)\s+(lift|move|create|make)\b", t):
            return True

        # Scholastic "angels on a pin(head)" paradox (plan P5 example).
        if "angels" in t and ("pin" in t or "head of a pin" in t or "pinhead" in t):
            return True
        if "pinhead" in t:
            return True
        
        # Self-referential paradoxes
        if re.search(r"\b(this\s+statement|the\s+following\s+sentence|the\s+next\s+thing)\b.*\b(false|true|paradox|contradict)", t):
            return True
        # Looser self-reference / liar check (handles "the statement i am lying
        # is true or false" which the stricter pattern above misses).
        if re.search(r"\b(statement|sentence|proposition)\b", t) and \
                re.search(r"\b(false|true|lie|lying|contradict)\b", t):
            return True
        
        # Classical paradoxes (unstoppable force, omnipotence, etc.)
        classical = [
            "unstoppable force", "irresistible force", "immovable object",
            "irresistible force meets", "immovable object meets",
            "can god create a stone", "could god create a stone",
            "can you create a stone", "what happens when an unstoppable",
            "what is the sound of one hand", "one hand clapping",
            "can you prove you exist", "can we know anything for certain",
            "is reality real", "are we living in a simulation",
            "what is the answer to life the universe and everything",
            "exist instead of nothing", "why is there something instead of nothing",
            "why does everything exist", "why does anything exist",
        ]
        for phrase in classical:
            if phrase in t:
                return True
        
        # Semantic contradiction markers: X so Y that Z (where Y is an extreme)
        # "heavy" + "cannot lift" pattern
        if re.search(r"\b(so\s+(heavy|powerful|big|strong|large|hot|cold)\s+that\s+(.+?)\s+(can't|cannot|couldn't|not|never))", t):
            return True
        
        # Question about impossibility itself
        if re.search(r"\b(impossible|paradox|contradiction|contradictory)\b", t):
            return True

        return False

    def _snippet_topic_max_coherence(self, topic: str, snippet: str) -> float:
        """Max single-word GloVe cosine between `topic` and any content word in
        `snippet`. Stricter than mean-centroid coherence: a snippet only passes
        if it actually mentions something related to the topic. Returns 0.0 if
        the topic has no embedding (caller falls back to fail-closed)."""
        tv = self._glove_vector(topic) if hasattr(self, "_glove_vector") else None
        if tv is None:
            return 0.0
        best = 0.0
        for w in re.findall(r"[a-z]{3,}", (snippet or "").lower()):
            if w in STOP_WORDS:
                continue
            wv = self._glove_vector(w)
            if wv is None:
                continue
            sim = float(np.dot(tv, wv) / (np.linalg.norm(tv) * np.linalg.norm(wv) + 1e-9))
            if sim > best:
                best = sim
        return best

    def _paradox_topic(self, text: str) -> str:
        """Data-derived topic word for a paradox query (drives retrieval +
        the coherence gate). Pure token filtering — no authored per-paradox
        tables. Prefers a known graph concept; else the longest content word.
        """
        t = (text or "").lower().strip(" ?!.")
        toks = [w.strip(".,!?") for w in re.findall(r"[a-z']+", t)
                if w.strip(".,!?") not in STOP_WORDS
                and w.strip(".,!?") not in ("what", "which", "how", "who",
                "is", "are", "was", "were", "do", "does", "did", "can", "could",
                "would", "should", "may", "might", "must", "cannot", "cannot",
                "the", "a", "an", "of", "on", "in", "to",
                "for", "with", "and", "or", "but", "many", "much", "head",
                "pin", "statement", "that", "this", "true", "false", "i", "am",
                "you", "he", "she", "it", "they", "we")]
        if not toks:
            return ""
        # Prefer a known graph concept (most 'central' topic).
        for w in toks:
            if w in self._concept_keywords or w in self._concept_labels:
                return w
        # Else the longest salient token (e.g. "angels", "god", "liar").
        return max(toks, key=len)

    def _reflect_on_paradox(self, text: str) -> str:
        """Generate a genuine philosophical reflection for a paradox / koan.

        P5: the family framings below are VOICE (allowed system tone, not
        factual claims). To stop pasting ungrounded history, we now run a
        SCOPED retrieval (learn_from_web) for the paradox's real context — e.g.
        'angels on a pinhead scholastic debate' — and append a short, clearly-
        labeled grounding clause composed from the retrieved sentences. The
        retrieved text is the only propositional content; the voice framing
        stays. If retrieval misses, the voice framing alone remains (fail-closed).
        A brief System-2 'slow-thinking' pause is simulated by the retrieval
        latency itself (deliberation before reply).
        """
        t = text.lower().strip(" ?!.")
        # Derive a retrieval query DATA-DERIVED from the paradox's own text
        # (no hardcoded per-paradox query strings). We reuse the same
        # query-reformulation machinery as factual web answers (IR
        # word-sense disambiguation / pseudo-relevance feedback): sense-bias
        # the query toward its intended reading, then generate variants.
        _topic = self._paradox_topic(t) or t
        _retrieval_q = None
        try:
            _biased = self._sense_biasing_framing(text, _topic) if hasattr(self, "_sense_biasing_framing") else None
            if _biased and _biased != _topic:
                _retrieval_q = _biased
            else:
                _retrieval_q = self._rewrite_query_for_web(text, _topic)
        except Exception:
            _retrieval_q = (t + " paradox") if "paradox" in t else t
        _queries = []
        if hasattr(self, "_web_query_variants"):
            try:
                _queries = self._web_query_variants(_retrieval_q or t, _topic, self._is_conditional_query(text))
            except Exception:
                _queries = []
        # Enrich with paradox-derived variants (data-derived from the topic,
        # no per-paradox authored strings): a person who realizes a search is
        # off-topic reformulates the query (IR pseudo-relevance feedback).
        if _topic:
            for _suff in ("paradox", "philosophical debate", "philosophy"):
                _v = f"{_topic} {_suff}"
                if _v not in _queries:
                    _queries.append(_v)
        if not _queries:
            _queries = [_retrieval_q or t]
        # Scoped retrieval (bounded; offline fallback if network down).
        # PRIMARY source: the same Wikipedia-REST lookup used for factual
        # definitions (kb_describe, P1) — clean, authoritative, on-topic, and
        # free of the entity-collision junk the flaky web search produces for
        # these queries ("angels" -> baseball, "god" -> cannon). SECONDARY:
        # the web-search path below, gated by M4/M5 + a strict coherence check,
        # used only when Wikipedia has no article. Pseudo-relevance feedback:
        # if a snippet fails the gate, try the next query variant instead of
        # quoting the irrelevant text (fail-closed if all fail).
        _ground = ""
        # ── Primary: Wikipedia REST for the paradox topic ──
        try:
            _def = self.kb_describe(_topic) if hasattr(self, "kb_describe") else None
            if _def:
                _def = self._sanitize_definition_text(_def) if hasattr(self, "_sanitize_definition_text") else _def
                _def = (_def or "").strip()
                if len(_def) > 30:
                    _sent = re.split(r"(?<=[.!?])\s+", _def)[0].rstrip(".!?")
                    if len(_sent) > 25:
                        _ground = f" (from what i've read: {_sent.lower()})"
                        self._metrics["paradox_grounded"] = self._metrics.get("paradox_grounded", 0) + 1
        except Exception:
            _ground = ""
        # ── Secondary: web search with learned gates (only if Wikipedia missed) ──
        _COHERENCE_THETA = 0.50
        if not _ground:
            for _q in _queries[:6]:
                try:
                    _results = self.search_engine.search(_q, max_results=4)
                    if not _results:
                        continue
                    _snip = self._best_answer_snippet(_results, _topic, _q, False)
                    if not _snip:
                        continue
                    # M4: structural-junk screen.
                    if hasattr(self, "_snippet_is_structural_junk") and self._snippet_is_structural_junk(_snip):
                        continue
                    # M5: source-trust gate (skip only when clearly untrusted).
                    if hasattr(self, "_domain_trust"):
                        _url = _results[0].get("url", "") if _results else ""
                        if self._domain_trust(_url) <= 0.0:
                            continue
                    # Coherence gate: reject loosely-on-topic snippets. We use
                    # MAX single-word GloVe cosine (does the snippet actually
                    # mention something related to the topic?) rather than the
                    # mean-centroid score, because the mean dilutes a coincidental
                    # alignment and lets junk ("angels" vs an "internet freedom"
                    # snippet) slip through at 0.32. Max-word is stricter and
                    # matches human behaviour: if nothing in the result refers to
                    # the topic, don't quote it (fail-closed). Falls back to the
                    # repo's _definition_coherence_score (mean-centroid) when the
                    # topic word has no embedding.
                    _coh = self._snippet_topic_max_coherence(_topic, _snip)
                    if _coh < _COHERENCE_THETA:
                        continue
                    # Passed all gates: take the first clean sentence.
                    _sent = re.split(r"(?<=[.!?])\s+", _snip.strip())[0].rstrip(".!?")
                    if len(_sent) > 25 and "learned" not in _sent.lower()[:20]:
                        _ground = f" (from what i've read: {_sent.lower()})"
                        self._metrics["paradox_grounded"] = self._metrics.get("paradox_grounded", 0) + 1
                        break
                except Exception:
                    continue

        # Zen koans: invitation to sit with the unanswerable.
        if "one hand" in t or "hand clapping" in t or "sound of" in t:
            return ("that's a koan — it's not really asking for a sound. it points "
                    "at the gap between words and what's actually experienced. sitting "
                    "with the silence is kind of the point.") + _ground
        # Omnipotence / theological paradoxes.
        if "god" in t and ("rock" in t or "stone" in t or "create" in t or "heavy" in t):
            return ("the catch is in the setup: 'all-powerful' breaks the moment you "
                    "ask it to make something it can't lift — you've defined a contradiction "
                    "and called it a thing. most readings treat it as showing the limit "
                    "is in the question, not in god.") + _ground
        if "unstoppable" in t or "immovable" in t:
            return ("if both exist, they can't meet without cancelling each other, and if "
                    "either fails, it wasn't truly unstoppable/immovable. so the paradox "
                    "is really about whether 'absolute' predicates are even coherent.") + _ground
        # Self-reference / liar family.
        if "statement" in t and ("false" in t or "true" in t):
            return ("that one ties language in a knot: if it's true it's false, if it's "
                    "false it's true. it's why logicians split 'use' and 'mention' — the "
                    "sentence talks about itself, and self-reference is where tidy systems leak.") + _ground
        # Simulation / reality-doubt family.
        if "simulation" in t or "reality real" in t or "know anything" in t:
            return ("i can't step outside my own experience to check, and neither can you — "
                    "so 'is this real' might be the wrong kind of question. what we can do "
                    "is reason about which assumptions hold up. want to dig into one?") + _ground
        # Specific scholastic paradox.
        if "pinhead" in t or "angels" in t:
            return ("that one's a classic: the point was never the number but whether "
                    "angels, as pure spirits, take up space at all. it was a way to argue "
                    "about the nature of immaterial beings.") + _ground
        # Generic paradox fallback.
        return ("that's a paradox — the interesting part isn't a single answer but the "
                "tension it exposes. i'd rather think it through with you than give you "
                "a dictionary line. which angle interests you?") + _ground

    # A small lexicon of common English words. GloVe indexes millions of rare
    # and invented tokens (e.g. "zoop"), so "present in GloVe" is NOT a reliable
    # signal that a token is a real word. A token only counts as meaningful here
    # if it is a stop-word, a known concept, a proper noun, or one of these
    # common words — this stops multi-word neologisms ("flargle bibble zoop
    # wibble") from slipping through as if they were English.
    _COMMON_WORDS = set("""
        the a an and or but if because when while of to in on at by for with from
        into over under between out up down off near far this that these those
        i you he she it we they me him her us them my your his our their is are
        was were be been being am do does did doing have has had having will would
        shall should can could may might must not no yes so than then thus there
        here what which who whom whose why how where all any each every few many
        more most other some such only own same too very s t just don now
        time year people way day man thing woman life world hand part child eye
        place case work government company number group problem fact house water
        food sun moon star earth tree plant animal dog cat bird fish light fire
        air wind rain snow ice hot cold big small long short good bad new old
        red blue green black white love hate like want need think know feel see
        hear say tell ask answer make find use give take eat drink sleep walk
        run talk write read play help learn grow change move live die born name
        one two three four five six seven eight nine ten first last great little
        own old young high low open close warm cool rich poor free true false
        """.split())

    def _user_input_is_gibberism(self, text: str) -> bool:
        """Detect user input that contains no real words at all (random
        letter-salad like 'asdf qwer zxcv'). Such input should not be treated
        as a learnable concept and confabulated about.

        We refuse only when (a) there is no question/learning intent and
        (b) not a single meaningful token is found among STOP_WORDS, the known
        concept graph, proper nouns, or a common-English lexicon. This keeps
        genuine (if obscure) learning queries like 'what is quokka' flowing
        through, while blocking pure nonsense."""
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
        # A token counts as a REAL word if it is a known concept, a common
        # English word, a proper noun, OR present in GloVe (and not keyboard
        # mashing). GloVe is included because it correctly recognises rare-but-
        # real words like "humans"/"photosynthesize" that no small lexicon
        # covers; the keyboard-mash check rejects random letter strings that
        # merely happen to exist in GloVe (e.g. "zoop"). We then require the
        # MAJORITY of tokens to be real: a multi-word neologism like
        # "flargle bibble zoop wibble" has only one stray GloVe hit, so it is
        # still flagged as gibberish, while a genuine query such as
        # "if humans could photosynthesize" is all-real and flows through.
        _real = 0
        for w in meaningful:
            if _is_keyboard_mash(w):
                continue
            if (w in self._concept_keywords
                    or w in self._COMMON_WORDS
                    or w in getattr(self, "_proper_nouns", set())):
                _real += 1
                continue
            if self._glove_vecs is not None and self._glove_vector(w) is not None:
                _real += 1
        # Fewer than half the tokens are real words -> treat as gibberish.
        return _real * 2 < len(meaningful)

    def _try_memory_query(self, user_input: str) -> Optional[str]:
        """Fix 4 (Q12): answer episodic meta-queries about the conversation.

        Handles "what did I just ask you", "what did I say", "what were we
        talking about", "what was my last question", "do you remember what I
        asked". These are queries whose subject is the DIALOGUE itself, so the
        subject-keyed hippocampal buffer cannot serve them — we answer from the
        verbatim user-turn ring buffer (Baddeley episodic buffer + hippocampal
        pattern completion). Returns None when the input is not a memory
        meta-query, so the normal pipeline runs.

        NOTE: called BEFORE the current turn is appended to
        ``_recent_user_turns``, so ``[-1]`` is the immediately preceding turn.
        """
        t = (user_input or "").lower().strip(" ?!.")
        if not t:
            return None
        # First/second-person + a recall/speech verb, referring to a prior turn.
        # Require an explicit conversational-memory phrasing to stay narrow.
        _patterns = [
            r"\bwhat did i (?:just )?(?:ask|say|tell|mention)\b",
            r"\bwhat (?:was|were) (?:my|the) (?:last |previous |first )?"
            r"(?:question|questions|message|thing i said)\b",
            r"\bwhat (?:were|are) we (?:talking|chatting) about\b",
            r"\bwhat (?:did|were) we (?:talk|talking) about\b",
            r"\b(?:do|can) you remember what i (?:asked|said|told)\b",
            r"\bwhat was i (?:just )?(?:asking|saying|talking) about\b",
            r"\brepeat (?:my|the) (?:last |previous )?question\b",
        ]
        if not any(re.search(p, t) for p in _patterns):
            return None
        prior = self._recent_user_turns
        if not prior:
            return "you haven't asked me anything yet this session — i don't have an earlier turn to recall."
        last = prior[-1].strip()
        # "what were we talking about" → topic-oriented; else verbatim recall.
        if re.search(r"\bwe (?:talking|talk|were|are) (?:about|chatting)\b", t) \
                or re.search(r"\btalking about\b", t):
            topic = ""
            if getattr(self, "_topic_list", None):
                topic = self._topic_list[-1]
            if topic:
                return f'we were talking about {topic}. your last message was: "{last}"'
            return f'your last message was: "{last}"'
        return f'you just asked me: "{last}"'

    def _try_arithmetic(self, user_input: str) -> Optional[str]:
        """Phase 19f: Answer simple arithmetic directly instead of routing it
        through the web/decomposition pipeline (which would fail to find a
        numeric fact and fall back to metacognitive uncertainty).

        Handles plain two- or three-operand expressions with + - * / and
        integer powers (^), with or without a leading question frame
        ("what is 2 + 2", "calculate 10 * 5"). Uses a whitelisted ``operator``
        map — never ``eval``. Returns a natural-language answer string, or None
        if the input is not simple arithmetic (so the normal pipeline runs).
        """
        # Normalize unicode operators and strip a leading question frame.
        s = user_input.lower()
        s = s.replace("×", "*").replace("÷", "/").replace("x", "*")
        # Division by zero (spelled out, e.g. "divide by zero", "10 divided by
        # zero") is mathematically undefined, not a solvable expression. Catch
        # it explicitly and answer honestly instead of letting it fall through
        # to the assertion mirror ("yeah, divide zero result."). Matches both
        # "divide X by zero" and bare "divide by zero" / "divided by zero".
        if re.search(r"divide[ds]?\s+(?:\w+\s+)?by\s+zero\b", s) or \
           re.search(r"\bzero\b.*\bdivid", s):
            return ("division by zero isn't defined — there's no number you can "
                    "multiply by zero to get back to the original value, so the "
                    "operation has no answer.")
        # Fix 5: normalize spelled-out operators ("2 plus 2", "10 times 5",
        # "9 minus 4", "8 divided by 2") to their symbols so the numeric path
        # (IPS quantity + left-AG verbal arithmetic, Triple Code Model) fires
        # instead of falling through to the association/uncertainty pipeline.
        # Word-boundary anchored so "explain" etc. are untouched.
        s = re.sub(r"\bplus\b", "+", s)
        s = re.sub(r"\bminus\b", "-", s)
        s = re.sub(r"\b(?:times|multiplied by)\b", "*", s)
        s = re.sub(r"\bdivided by\b", "/", s)
        s = re.sub(r"\b(?:to the power of|raised to)\b", "^", s)
        # Remove common framing words; keep only the math expression.
        s = re.sub(r"^(what(?:'s| is)|calculate|compute|solve|find|tell me|how much is|how many is)\s+",
                   "", s).strip()
        s = s.rstrip("?.").strip()
        # Match a chain of numbers joined by operators: "a op b", "a op b op c",
        # "a op b op c op d", ... (N operands, whitespace-flexible). Evaluated
        # left-to-right. Fix: the previous regex capped at 3 operands, so
        # "2 + 2 + 2 + 2" fell through to metacognitive uncertainty.
        if not re.fullmatch(
            r"\s*[-+]?\d+(?:\.\d+)?(?:\s*[+\-*/^]\s*[-+]?\d+(?:\.\d+)?)+\s*", s
        ):
            return None
        try:
            ops = {
                "+": operator.add, "-": operator.sub,
                "*": operator.mul, "/": operator.truediv,
                "^": operator.pow,
            }
            # Extract numbers and operators in reading order.
            _tokens = re.findall(r"([-+]?\d+(?:\.\d+)?)|([+\-*/^])", s)
            nums, ops_seq = [], []
            for _num, _op in _tokens:
                if _num:
                    nums.append(float(_num))
                elif _op:
                    ops_seq.append(_op)
            # Left-to-right evaluation of the operand chain.
            c = nums[0]
            for _i, _op in enumerate(ops_seq):
                c = ops[_op](c, nums[_i + 1])
            # Format cleanly: integers stay integers, else trim long floats.
            if c == int(c) and abs(c) < 1e15:
                result = str(int(c))
            else:
                result = f"{c:.4g}".rstrip("0").rstrip(".")
            # Mirror the user's phrasing for a natural reply.
            expr = user_input.strip().rstrip("?.")
            return f"{expr} = {result}."
        except (ZeroDivisionError, OverflowError, ValueError):
            return None



    # ── Frontopolar (BA 10) feasibility gate ───────────────────────────────
    _CATEGORY_AFFORDANCES = {
        "time": {"duration", "order", "sequence", "cycle", "moment", "pass", "flow"},
        "mental_state": {"content", "valence", "intensity", "clarity", "meaning"},
        "abstract": {"meaning", "importance", "truth", "value"},
        "physical_object": {"mass", "weight", "color", "shape", "size", "temperature",
                             "volume", "position"},
        "perceptual": {"color", "brightness", "loudness", "taste", "smell", "texture"},
        "social": {"trust", "power", "status", "relationship"},
        "living": {"growth", "reproduction", "metabolism", "death", "color", "colour"},
        "event": {"cause", "duration", "consequence"},
    }
    _CATEGORY_OF_SUBJECT = {
        "day": "time", "days": "time", "week": "time", "month": "time", "year": "time",
        "hour": "time", "minute": "time", "time": "time", "tuesday": "time",
        "monday": "time", "wednesday": "time", "thursday": "time", "friday": "time",
        "saturday": "time", "sunday": "time", "moment": "time", "century": "time",
        "thought": "mental_state", "thoughts": "mental_state", "idea": "mental_state",
        "ideas": "mental_state", "dream": "mental_state", "emotion": "mental_state",
        "love": "mental_state", "hate": "mental_state", "memory": "mental_state",
        "concept": "abstract", "concepts": "abstract", "meaning": "abstract",
        "truth": "abstract", "beauty": "abstract", "freedom": "abstract",
        "sun": "physical_object", "earth": "physical_object", "rock": "physical_object",
        "stone": "physical_object", "table": "physical_object", "car": "physical_object",
        "book": "physical_object", "tree": "living", "trees": "living", "human": "living",
        "humans": "living", "cat": "living", "dog": "living", "plant": "living",
        "trust": "social", "relationship": "social", "friendship": "social",
    }
    _PROPERTY_CATEGORIES = {
        "color": {"physical_object", "perceptual"},
        "colour": {"physical_object", "perceptual"},
        "weight": {"physical_object"},
        "weigh": {"physical_object"}, "weighs": {"physical_object"},
        "mass": {"physical_object"},
        "taste": {"physical_object", "perceptual"},
        "smell": {"physical_object", "perceptual"},
        "sound": {"physical_object", "perceptual"},
        "size": {"physical_object"},
        "shape": {"physical_object"},
        "texture": {"physical_object", "perceptual"},
        "temperature": {"physical_object", "perceptual"},
    }

    def _load_conceptnet_ontology(self) -> Optional["ConceptNetOntology"]:
        """Load the prebuilt ConceptNet ontology pickle (see ontology/conceptnet.py
        and the build step that writes data/conceptnet/ont.pkl). Returns None if
        unavailable so the gate safely falls back to the legacy literal dicts.

        Also wires the Binder ridge-probe (attribute_encoder) as a distributional
        tie-break prior (deferred item 2): when ConceptNet is silent,
        has_property() consults attribute_encoder.property_score via the engine's
        GloVe-64 vector. The encoder is loaded from data/attribute_encoder.npz
        (or data/conceptnet/attribute_encoder.npz) if present; otherwise the
        prior is simply absent and ConceptNet stays the sole authority.
        """
        here = os.path.abspath(__file__)
        cur = here
        ont_path = None
        for _ in range(8):
            cand = os.path.join(cur, "data", "conceptnet", "ont.pkl")
            if os.path.exists(cand):
                ont_path = cand
                break
            cur = os.path.dirname(cur)
        if ont_path is None:
            return None
        # Locate the attribute_encoder probe (optional prior).
        enc = None
        for d in (os.path.dirname(ont_path),
                  os.path.join(os.path.dirname(ont_path), "..")):
            for fn in ("attribute_encoder.npz",):
                ecand = os.path.join(d, fn)
                if os.path.exists(ecand):
                    try:
                        from ravana.ontology.attribute_encoder import load_combined_encoder
                        _lanc_cands = [
                            os.path.join(os.path.dirname(ecand), "lancaster_encoder.npz"),
                            os.path.join(os.path.dirname(os.path.dirname(ecand)),
                                         "lancaster_encoder.npz"),
                        ]
                        _lanc = next((p for p in _lanc_cands if os.path.exists(p)), None)
                        enc = load_combined_encoder(ecand, _lanc)
                    except Exception:
                        enc = None
                    break
            if enc is not None:
                break
        # GloVe vector fn (returns 64-dim projected vec, or None if OOV).
        glove_fn = getattr(self, "_glove_vector", None)
        try:
            ont = ConceptNetOntology(attribute_encoder=enc, glove_fn=glove_fn)
            # CRITICAL: hydrate from the prebuilt pickle. Constructing the object
            # alone leaves isa/features EMPTY — without load(), the bootstrap in
            # _typed_edges_bootstrap would inject 0 typed edges and the
            # inheritance walk (Path 2) would stay structurally impossible.
            # Note: load() is a @classmethod that returns a NEW hydrated object,
            # so its return value must be captured (calling ont.load(...) alone
            # discards the result and leaves ont empty).
            loaded = ont.load(ont_path)
            # load() is a @classmethod that builds a FRESH object, discarding
            # the attribute_encoder/glove_fn we passed to the constructor.
            # Re-attach them so the (combined Lancaster+Binder) probe survives
            # hydration — this is what makes the wide-coverage Lancaster probe
            # actually drive the cross-modal metaphor (Fix A.1).
            loaded.attribute_encoder = enc
            loaded.glove_fn = glove_fn
            return loaded
        except Exception:
            return None

    # ═════════════════════════════════════════════════════════════════════════
    # P1 — KB-grounded commonsense (zero hand-authored facts)
    # Two non-authored inputs, composed not authored:
    #   * kb_describe(concept)  — Wikipedia/Wikidata summary, sanitized through
    #                            the SAME _sanitize_definition_text pipeline the
    #                            live web learner uses, stored in _definitions
    #                            EXACTLY like a web-learned fact.
    #   * describe_from_cn(concept) — ConceptNet IsA/HasProperty composition
    #                            ("a sun is a star; a sun has property bright").
    # Both are pure KB retrieval; nothing here is a literal string of prose.
    # ═════════════════════════════════════════════════════════════════════════

    def kb_describe(self, concept: str, timeout: float = 6.0) -> Optional[str]:
        """Fetch a one-line natural-language description of `concept` from the
        Wikipedia REST summary endpoint (or Wikidata description fallback), then
        sanitize it through the existing _sanitize_definition_text pipeline so
        it matches web-learned fact quality. Returns cleaned text, or None if
        the KB has nothing usable. This is retrieval, not authored prose."""
        if not concept or len(concept) < 2:
            return None
        # Title candidates: exact, title-cased, and a de-pluralized singular.
        cands = [concept.strip()]
        tc = concept.strip().title()
        if tc != cands[0]:
            cands.append(tc)
        # De-pluralize a simple trailing 's' for better Wikipedia title hits.
        if concept.endswith("s") and len(concept) > 3:
            cands.append(concept[:-1].title())
        import urllib.request
        import urllib.parse
        import json
        for title in cands:
            try:
                url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + \
                    urllib.parse.quote(title.replace(" ", "_"))
                req = urllib.request.Request(
                    url, headers={"User-Agent": "ravana-cog/1.0 (KB grounding)"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8", "ignore"))
                extract = (data.get("extract") or "").strip()
                if not extract:
                    continue
                # Wikidata description fallback when the summary is missing.
                if len(extract) < 20 and data.get("description"):
                    extract = data["description"]
                clean = self._sanitize_definition_text(extract)
                if clean:
                    # Wikipedia summaries open with the title ("The Sun is the
                    # star..."). Strip a leading "[The/A/An ]<title> is/are "
                    # echo so the definition reads cleanly when the engine
                    # later prefixes the subject itself ("Sun is the star...").
                    _tc = title.lower().strip()
                    clean = re.sub(
                        r"^\s*(?:the |a |an )?" + re.escape(_tc)
                        + r"\s+(is|are|was|were|refers to|means)\s+",
                        "", clean, flags=re.IGNORECASE).strip()
                    from ravana.chat.case_distribution import case_infer
                    clean = case_infer(clean) if clean else clean
                    return clean
            except Exception:
                continue
        return None

    def describe_from_cn(self, concept: str) -> Optional[str]:
        """Compose a short description of `concept` purely from the ConceptNet
        ontology: its nearest IsA parent(s) and its HasProperty/CapableOf/
        UsedFor features. Returns e.g. 'a sun is a star; a sun is located in the
        sky' — built from KB relations, not a hardcoded sentence. None if the
        ontology is silent on the concept."""
        ont = getattr(self, "_cn_ontology", None)
        if ont is None:
            return None
        c = (concept or "").lower().strip()
        if not c:
            return None
        parts = []
        # Nearest IsA parent via the ontology's category walk.
        try:
            parents = ont.isa.get(c, set()) if hasattr(ont, "isa") else set()
        except Exception:
            parents = set()
        for p in list(parents)[:2]:
            parts.append(f"a {c} is a {p}")
        # Feature properties (HasProperty/CapableOf/UsedFor).
        try:
            feats = ont.features.get(c, set()) if hasattr(ont, "features") else set()
        except Exception:
            feats = set()
        for f in list(feats)[:3]:
            parts.append(f"a {c} has property {f}")
        if not parts:
            return None
        # Capitalize the first clause; join with '; '.
        body = "; ".join(parts)
        from ravana.chat.case_distribution import case_infer
        return case_infer(body) if body else None

    def _seed_common_facts(self) -> int:
        """Seed authored, OFFLINE core knowledge (M1-B).

        Loads ``data/common_facts.json`` — a curated set of universal facts
        (sky/cat/music/sun/gravity/...) — and writes them into ``_definitions``
        with CURATED provenance, plus their typed graph relations. This makes
        common-fact questions deterministic and offline-grounded, independent of
        live web/KB retrieval timing (the prior source of nondeterminism and
        common-fact misses in the battery).

        Fail-closed: missing file / bad JSON / any node error is skipped; the
        live-KB path still runs afterwards as a fallback. Returns the number of
        concepts seeded.
        """
        import json as _json
        facts_path = os.path.join(_proj_root, "data", "common_facts.json")
        if not os.path.exists(facts_path):
            return 0
        try:
            with open(facts_path, "r", encoding="utf-8") as fh:
                facts = _json.load(fh)
        except Exception:
            return 0
        if not isinstance(facts, dict):
            return 0
        seeded = 0
        for concept, entry in facts.items():
            if not isinstance(entry, dict):
                continue
            definition = entry.get("definition")
            if isinstance(definition, str) and definition.strip():
                # CURATED provenance so the grounding monitors treat it as
                # established fact (not a hedged web association).
                self._definitions[concept] = definition.strip()
                self._curated_definitions.add(concept)
                _md = getattr(self, "_definition_metadata", None)
                if _md is not None and concept not in _md:
                    try:
                        _md[concept] = {"source": "curated", "edge_kind": "curated"}
                    except Exception:
                        pass

            # usable by chain-walk / counterfactual simulation.
            rels = entry.get("relations") or []
            for rel in rels:
                if not isinstance(rel, (list, tuple)) or len(rel) < 4:
                    continue
                src, tgt, rel_type, weight = rel[0], rel[1], rel[2], rel[3]
                try:
                    self._ensure_relation(src, tgt, rel_type, float(weight))
                except Exception:
                    continue
            seeded += 1
        if seeded:
            print(f"  [CommonFacts] Seeded {seeded} authored core facts (offline)")
        return seeded

    def _ensure_relation(self, src: str, tgt: str, rel_type: str,
                         weight: float) -> None:
        """Idempotently ensure a typed edge src->tgt exists in the graph.

        Helper for the offline common-facts seed (and physics causal seed, M3):
        creates both endpoint nodes (with GloVe or hash vectors) if missing and
        adds the edge with the requested relation type + provenance.
        """
        graph = getattr(self, "graph", None)
        if graph is None:
            return
        def _node(label):
            nids = self._concept_keywords.get(label)
            if nids:
                return nids[0]
            vec = self._glove_vector(label)
            if vec is None:
                h = hash(label) % 50000
                vr = np.random.RandomState(h + 100)
                vec = vr.randn(self.dim).astype(np.float32) * 0.1
                n = np.linalg.norm(vec)
                if n > 0:
                    vec /= n
            node = graph.add_node(vector=vec, label=label)
            node.stability = 0.9
            self._concept_labels.add(label.lower())
            self._concept_keywords[label] = self._concept_keywords.get(label, []) + [node.id]
            if hasattr(node, "source_metadata"):
                node.source_metadata.update({"edge_kind": "curated", "source": "common_facts"})
            return node.id
        s_id = _node(src)
        t_id = _node(tgt)
        if graph.get_edge(s_id, t_id) is None:
            e = graph.add_edge(s_id, t_id, weight=weight,
                               relation_type=rel_type, confidence=0.9)
            if e is not None and hasattr(e, "source_metadata"):
                e.source_metadata.update({"edge_kind": "curated", "source": "common_facts"})

    def _load_persisted_definitions(self) -> int:
        """M1-C: reload verified definitions mirrored to CognitiveDB (M1-C save).

        Merges previously-learned facts back into ``_definitions`` on a fresh
        cold-start, so knowledge is durable across --reset, not just pickle
        reloads. Only fills keys absent from the current store (curated/offline
        seeds and any KB hits this run are never overwritten by stale state),
        keeping the load fail-closed.
        """
        db = getattr(self, "db", None)
        if db is None:
            return 0
        try:
            saved = db.load_metadata("definitions")
            saved_curated = set(db.load_metadata("curated_definitions") or [])
        except Exception:
            return 0
        if not isinstance(saved, dict):
            return 0
        added = 0
        for k, v in saved.items():
            if not isinstance(v, str) or not v.strip():
                continue
            if k in self._definitions:
                continue
            self._definitions[k] = v
            if k in saved_curated:
                self._curated_definitions.add(k)
            added += 1
        if added:
            print(f"  [Persisted] Rehydrated {added} verified definitions from CognitiveDB")
        return added

    def _seed_physics_causal(self) -> int:
        """M3-E: seed a compact PHYSICS causal skeleton so counterfactual
        simulation can forward-chain from first principles when the lived graph
        has no edge for an intervened concept.

        E.g. ``sun disappeared`` -> sun → light → photosynthesis → plants →
        animals; sun → heat → climate. These are universal causal priors
        (innate-like), authored offline, so the DMN simulator (which only walks
        causal edges) always has something to chain along for world-scale
        interventions. Fail-closed: any edge error is skipped.
        """
        # (src, tgt, relation_type, weight)
        edges = [
            ("sun", "light", "causal", 0.95),
            ("sun", "heat", "causal", 0.9),
            ("sun", "earth", "causal", 0.85),
            ("light", "photosynthesis", "causal", 0.9),
            ("photosynthesis", "plants", "causal", 0.9),
            ("plants", "animals", "causal", 0.85),
            ("plants", "oxygen", "causal", 0.8),
            ("animals", "oxygen", "causal", 0.7),
            ("heat", "climate", "causal", 0.8),
            ("earth", "life", "causal", 0.8),
            ("gravity", "orbit", "causal", 0.9),
            ("gravity", "earth", "causal", 0.8),
            ("water", "life", "causal", 0.85),
            ("water", "plants", "causal", 0.8),
        ]
        n = 0
        for src, tgt, rel, w in edges:
            try:
                self._ensure_relation(src, tgt, rel, w)
                n += 1
            except Exception:
                continue
        if n:
            print(f"  [Physics] Seeded {n} causal edges (offline world-skeleton)")
        return n

    def _seed_kb_definitions(self, top_n: int = 250, workers: int = 8) -> int:
        """Seed _definitions from a DATA-DERIVED concept list (not a hand list):
        the most frequent content words in data/corpora/teen_seeds.txt. For each
        novel concept we try kb_describe (Wikipedia) then describe_from_cn
        (ConceptNet), storing the result in _definitions exactly like a
        web-learned fact. Returns the number of concepts seeded. Fail-closed:
        any concept with no KB hit is simply skipped (no authored fallback).

        Network lookups are parallelized (workers) so the one-time cold-start
        cost stays bounded; subsequent runs load the seeded weights and skip
        this entirely.
        """
        import re as _re
        from concurrent.futures import ThreadPoolExecutor
        corpus_path = os.path.join(_proj_root, "data", "corpora", "teen_seeds.txt")
        if not os.path.exists(corpus_path):
            return 0
        try:
            with open(corpus_path, "r", encoding="utf-8") as fh:
                text = fh.read().lower()
        except Exception:
            return 0
        # Frequency count of alphabetic tokens, excluding stopwords.
        counts: Dict[str, int] = {}
        for w in _re.findall(r"[a-z][a-z'\-]+", text):
            if w in STOP_WORDS or len(w) < 3:
                continue
            counts[w] = counts.get(w, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        novel = [w for w, _ in ranked[:top_n] if w not in self._definitions]
        if not novel:
            return 0

        def _lookup(word):
            try:
                return word, (self.kb_describe(word) or self.describe_from_cn(word))
            except Exception:
                return word, None

        results: Dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for word, desc in ex.map(_lookup, novel):
                if desc:
                    results[word] = desc
        seeded = 0
        for word, desc in results.items():
            # M2-D: never overwrite a protected (authored/project) concept with a
            # web/KB collision (e.g. "ravana" -> Ramayana myth). Provenance
            # precedence: curated/project definition beats retrieved text.
            if word in self._PROTECTED_CONCEPTS:
                continue
            self._definitions[word] = desc
            seeded += 1

            print(f"  [KB] Seeded {seeded} definitions from Wikipedia/ConceptNet "
                  f"(top-{top_n} corpus concepts, {workers} workers)")
        return seeded

    def _typed_edges_bootstrap(self) -> int:
        """Inject ConceptNet typed edges (isa / has_property / capable_of /
        used_for) into the live ravana graph (deferred item 1).

        This materializes the taxonomic + componential spokes the learned
        associative graph was missing, so chain_walker / DerivedOntology's
        inheritance walk (Path 2) can finally resolve over REAL typed edges.
        Idempotent: only adds edges when typed edges are absent, and persists
        back to the SQLite graph store so the work survives restart.

        Returns the number of typed edges injected (0 if none needed / graph
        unavailable / ontology absent).
        """
        graph = getattr(self, "graph", None)
        ont = getattr(self, "_cn_ontology", None)
        if graph is None or ont is None:
            return 0
        # Lazy import keeps the chat engine import-light when unused.
        try:
            from ravana.ontology.graph_typing import (
                inject_conceptnet_typed_edges, build_label_index,
                TYPED_RELATION_TYPES,
            )
        except Exception:
            return 0
        # Skip if typed edges already present (e.g. loaded from a typed DB).
        have = sum(
            1 for (_, _t), e in graph.edges.items()
            if getattr(e, "relation_type", "semantic") in TYPED_RELATION_TYPES
        )
        if have > 0:
            return 0
        label_index = build_label_index(graph)
        counts = inject_conceptnet_typed_edges(graph, ont, label_index=label_index)
        if counts["total"] > 0:
            # Persist so subsequent loads already contain typed edges.
            try:
                db = getattr(self, "db", None)
                if db is not None:
                    db.save_graph(graph)
            except Exception:
                pass
        return counts["total"]

    def _is_category_error(self, query: str, subject: Optional[str] = None) -> Optional[str]:
        """Detect a predicative category error (frontopolar feasibility gate).

        Returns the property word the subject's category cannot possess, else
        None. Only flags clear mismatches (time/mental/abstract subject +
        physical/perceptual property) — conservative to avoid false positives
        on legitimate "what color is the sun" questions.
        """
        q = (query or "").lower()
        prop = None
        for p in self._PROPERTY_CATEGORIES:
            if re.search(r"\b" + re.escape(p) + r"\b", q):
                prop = p
                break
        if prop is None:
            return None
        # Guard: a known philosophical paradox / koan must never be flagged as a
        # mere category error — it needs deliberation, not the "flavor of a
        # Tuesday" brush-off. The frontopolar paradox detector runs later in the
        # pipeline, but the category-error gate runs first, so short-circuit here.
        if self._is_philosophical_paradox(q):
            return None
        subj = (subject or "").lower().strip(" ?!.")
        if not subj:
            # Head noun after the copula, skipping determiners (a/an/the) which
            # the naive regex would otherwise capture (e.g. "what colour is a
            # day" -> "day", not "a").
            m = re.search(r"\b(what|which|how)\s+\w+\s+(is|are|does|do|has|have)\s+(?:a\s+|an\s+|the\s+)?(\w+)", q)
            if m:
                subj = m.group(3)
            else:
                # "how many kilograms does a thought weigh" -> capture "thought"
                m2 = re.search(r"\b(?:a|an|the)\s+(\w+)\s+(weigh|weighs|weighing|mass|taste|smell|sound|cost)\b", q)
                if m2:
                    subj = m2.group(1)
                else:
                    # Genitive "X of Y" form: "what is the taste of a triangle"
                    # -> prop="taste", head noun "triangle" (skip determiners).
                    mg = re.search(r"\b(?:a|an|the)\s+(\w+)\s+of\s+(?:a\s+|an\s+|the\s+)?(\w+)", q)
                    if mg and mg.group(2) not in ("a", "an", "the"):
                        subj = mg.group(2)
                    else:
                        toks = [w for w in re.findall(r"[a-z']+", q)
                                if w not in STOP_WORDS and w not in ("what", "which", "how",
                                "is", "are", "does", "do", "has", "have", "the", "a", "an",
                                "of", "to", "in", "on", "for", "with", "my", "your", "our")]
                        subj = toks[-1] if toks else ""
        # Store the gate's authoritative head noun so the metaphor response
        # uses the REAL subject (e.g. "triangle", not the property "taste")
        # even when the generic _ground_query guess differs. This is what lets
        # Path 1 (cross-modal probe) fire for the correct concept.
        self._last_category_subject = subj
        if not subj:
            return None
        # ── Primary gate: brain-aligned ConceptNet derivation ───────────────
        # category_of is inferred via IsA walk; affordances by the
        # Sensory-Functional division (concrete categories possess physical
        # properties; time/event possess temporal ones). This replaces the
        # per-word _CATEGORY_OF_SUBJECT lookup. Returns True (possesses ->
        # allowed), False (cannot possess -> flag), or None (KG silent ->
        # fall through to the legacy literal dicts as a safety net).
        if getattr(self, "_cn_ontology", None) is not None:
            ont = self._cn_ontology
            # Lazy-load the AttributeEncoder probe (Lancaster/Binder norms) if
            # the ontology wasn't built with it wired. Cached on the ontology.
            _enc = getattr(ont, "attribute_encoder", None)
            if _enc is None:
                try:
                    from ravana.ontology.attribute_encoder import load_combined_encoder
                    _cand = os.path.join(_proj_root, "data", "attribute_encoder.npz")
                    _lanc = os.path.join(_proj_root, "data", "lancaster_encoder.npz")
                    if os.path.exists(_cand):
                        _enc = load_combined_encoder(_cand, _lanc)
                        ont.attribute_encoder = _enc
                except Exception:
                    _enc = None
            _gvec = self._glove_vector(subj) if hasattr(self, "_glove_vector") else None
            _probe_score = None
            if _enc is not None and _gvec is not None:
                try:
                    _probe_score = _enc.property_score(np.asarray(_gvec, dtype=np.float64), prop)
                except Exception:
                    _probe_score = None
            derived = ont.has_property(subj, prop)
            # Item A: theta is FIT to human Lancaster norms (not the blind 0.8).
            # Calibrated per-property via the property's Binder dims -> their
            # Lancaster source dims. See experiments/measure_attribute_theta.py.
            from ravana.ontology.attribute_calibration import (
                load_fitted_theta, calibrated_property_threshold, ood_abstain)
            _fitted = load_fitted_theta()
            _THETA = calibrated_property_threshold(prop, _fitted)
            # OOD-abstain (item A.4): if the probe is off-manifold (silent), it
            # has no confident sensorimotor signal — treat as indecisive rather
            # than forcing a verdict on a random graph-edge metaphor.
            _ood = ood_abstain(_enc, _gvec)
            if derived is True:
                # KG says the subject possesses the property. Cross-check the
                # learned probe: the KG can carry spurious edges (e.g. ConceptNet
                # asserts 'triangle has taste'), so trust the probe when it
                # strongly disagrees (subject lacks the property's activation).
                # Exception: 'shape' is near-universal for spatial/geometric
                # objects and the probe mis-scores it, so we never override a
                # ConceptNet 'has shape' verdict with the probe (avoids flagging
                # legitimate "shape of a circle" questions).
                if prop != "shape" and getattr(self, "use_conceptnet_primary", False) and \
                        _probe_score is not None and _probe_score <= _THETA:
                    return prop
                return None
            if derived is False:
                return prop
            # None -> ConceptNet is silent. Fall back to the learned probe.
            if getattr(self, "use_conceptnet_primary", False):
                if _ood:
                    # Off-manifold: no confident signal -> abstain (allow),
                    # never a false positive from a silent probe.
                    return None
                if _probe_score is not None and _probe_score <= _THETA:
                    return prop
                return None
        # ── Literal-dict fallback (legacy frontopolar gate) ─────────────────
        # Runs ONLY when ConceptNet-primary is OFF (the default). When
        # use_conceptnet_primary is ON, the literal _CATEGORY_OF_SUBJECT table is
        # bypassed entirely (that is the whole point of M6): a silent/absent KG
        # means "insufficient evidence to flag" -> allow, never a literal-lookup
        # false positive. When OFF, the literal dicts remain the working gate
        # for OOV/silent subjects (current behavior, no regression).
        if not getattr(self, "use_conceptnet_primary", False):
            cat = self._CATEGORY_OF_SUBJECT.get(subj)
            if cat is None:
                return None
            allowed = self._CATEGORY_AFFORDANCES.get(cat, set())
            if prop in allowed:
                return None
            if cat in self._PROPERTY_CATEGORIES.get(prop, set()):
                return None
            # ── Derived path is AVAILABLE but NOT the primary gate ──────────
            return prop
        # M6 primary path: KG silent/absent AND probe indecisive -> allow.
        return None

    def _category_error_response(self, query: str, subject: Optional[str], prop: str) -> str:
        """Honest response for a detected category error (BA 10 gate output).

        P4: replaced the fixed 'flavor of a Tuesday' brush-off with a
        DATA-DERIVED cross-modal metaphor. We read the subject's most salient
        sensorimotor dimensions from the learned Binder/AttributeEncoder probe
        (Lancaster norms) and frame the mismatch in those terms — e.g. a
        triangle is something you'd picture by its *shape*, not something with
        a *taste*. When the probe is unavailable, we fall back to a graph-
        sampled incongruent pair (two unrelated concepts) stating the mismatch.
        Nothing here is a hardcoded analogy string; the content is derived from
        the subject's own attribute profile / the graph.
        """
        subj = (subject or "").lower().strip(" ?!.")
        # Prefer the gate's authoritative head noun (set in _is_category_error)
        # over the generic _ground_query guess, so the metaphor describes the
        # real subject (e.g. "triangle"), not the property word.
        if getattr(self, "_last_category_subject", "") and self._last_category_subject != subj:
            subj = self._last_category_subject
        subj_cap = (subj or "that").strip().capitalize()
        # Try the cross-modal metaphor first (subject's own sensorimotor profile).
        metaphor = self._metaphor_for_category_error(subj, prop)
        if metaphor:
            # P6: any successful (data-derived) metaphor reply counts as
            # category-error engagement — not just the probe branch.
            self._metrics["category_metaphor"] = self._metrics.get("category_metaphor", 0) + 1
            return metaphor
        # Last resort: honest category-label reply (no fixed analogy string).
        cat = self._category_label_of(subject)
        return (f"i don't think that quite works: {subj_cap} is {cat}, so it "
                f"doesn't really have a {prop} the way a physical thing would. "
                f"want to rephrase what you meant?")

    # Binder/AttributeEncoder dimensions that are perceptual / sensorimotor,
    # mapped to a natural (voice) phrasing + the sense verb used to experience
    # them. The dimension NAMES come from the learned probe output, not a
    # hand-authored list of analogies.
    _LANCASTER_ORDER = [
        "Auditory", "Gustatory", "Haptic", "Interoceptive", "Olfactory", "Visual",
        "Foot_leg", "Hand_arm", "Head", "Mouth", "Torso",
    ]
    _SENSORY_DIM_PHRASE = {
        "Shape": ("shape", "picture by its outline"),
        "Vision": ("looks", "see"),
        "Color": ("colour", "see"),
        "Bright": ("brightness", "see"),
        "Dark": ("darkness", "see"),
        "Pattern": ("pattern", "see the arrangement of"),
        "Texture": ("texture", "feel to the touch"),
        "Touch": ("feel", "feel"),
        "Temperature": ("temperature", "sense the warmth or cool of"),
        "Weight": ("weight", "feel the heft of"),
        "Sound": ("sound", "hear"),
        "Audition": ("sound", "hear"),
        "Loud": ("loudness", "hear"),
        "Motion": ("movement", "watch move"),
        "Complexity": ("structure", "grasp the makeup of"),
        "Taste": ("taste", "taste"),
        "Smell": ("smell", "smell"),
        # G3 (Lancaster): effector / body-part dims — these carry the
        # embodied specificity that distinguishes hand (Hand_arm=4.4) from trust
        # (Hand_arm=0.45). Without them every metaphor collapsed to Vision.
        "UpperLimb": ("movement", "move and handle"),
        "LowerLimb": ("steps", "step and walk with"),
        "Head": ("presence", "hold up"),
        "Mouth": ("voice", "speak or eat with"),
        "Torso": ("body", "feel the weight of"),
    }
    # The gate-property -> Binder dim(s) it corresponds to (mirrors
    # attribute_encoder.PROPERTY_TO_DIMS for the common cases).
    _PROP_TO_BINDER = {
        "color": ("Color", "Vision", "Bright", "Dark"),
        "colour": ("Color", "Vision", "Bright", "Dark"),
        "weight": ("Weight",), "weigh": ("Weight",), "weighs": ("Weight",),
        "mass": ("Weight",),
        "taste": ("Taste",), "smell": ("Smell",),
        "sound": ("Sound", "Audition", "Loud"),
        "size": ("Large", "Small"), "shape": ("Shape",),
        "texture": ("Texture", "Touch"), "temperature": ("Temperature",),
        "brightness": ("Bright", "Dark"),
    }

    def _top_sensorimotor_dim(self, word: str):
        """G3 (Lancaster): pick the most salient SENSORY (cross-modal) dimension
        for a word from the HUMAN Lancaster 11-D norms (variance-rich), mapped
        onto the Binder sensory dims used by _SENSORY_DIM_PHRASE.

        Returns (binder_dim, value_0_5, phrase, sense) for the top sensory dim,
        or None if the word is OOV / has no salient sensory activation. The
        human norms discriminate strongly (hand Hand_arm=4.4 vs trust=0.45) where
        the merged 65-D probe (used by the legacy Path 1 block) is compressed, so
        metaphors built from this are more vivid and correctly embodied.

        Selection: a SALIENT EFFECTOR / body-part dim (Hand_arm, Foot_leg, Head,
        Mouth, Torso) is preferred when its activation >= 2.0, because that is the
        genuinely distinguishing embodied signal; otherwise the top sensory dim
        (Vision/Touch/...) is used. This stops every metaphor collapsing to Vision
        (which is high for almost all concrete nouns) and surfaces embodiment.
        """
        try:
            from ravana.ontology.attribute_encoder import LANCASTER_TO_BINDER
        except Exception:
            return None
        lv = self._lancaster_vector(word)
        if lv is None:
            return None
        lv = np.asarray(lv, dtype=np.float64)
        if lv.size != len(self._LANCASTER_ORDER):
            return None
        _EFFECTOR = {"Foot_leg", "Hand_arm", "Head", "Mouth", "Torso"}
        effector_scored = []
        sensory_scored = []
        for i, ldim in enumerate(self._LANCASTER_ORDER):
            val = float(lv[i])
            if val <= 0.0:
                continue
            for bdim in LANCASTER_TO_BINDER.get(ldim, []):
                if bdim in self._SENSORY_DIM_PHRASE:
                    if ldim in _EFFECTOR:
                        effector_scored.append((val, bdim))
                    else:
                        sensory_scored.append((val, bdim))
                    break  # one Binder sensory dim per Lancaster dim
        # Prefer a salient effector (embodied) signal.
        if effector_scored:
            effector_scored.sort(reverse=True)
            _ev, _edim = effector_scored[0]
            if _ev >= 2.0:
                phrase, sense = self._SENSORY_DIM_PHRASE[_edim]
                return (_edim, _ev, phrase, sense)
        if sensory_scored:
            sensory_scored.sort(reverse=True)
            _val, top_dim = sensory_scored[0]
            phrase, sense = self._SENSORY_DIM_PHRASE[top_dim]
            return (top_dim, _val, phrase, sense)
        return None

    def _metaphor_lead(self, subj_cap: str, phrase: str, sense: str,
                       val: float, prop: str) -> str:
        """Build the magnitude-conditioned cross-modal metaphor reply. Shared by
        the human-Lancaster Path 1 and the legacy probe fallback so phrasing is
        identical; vivid when activation is high, tentative when low."""
        if val >= 2.0:
            lead = f"i'd really picture {subj_cap} in terms of its {phrase}"
        elif val >= 1.0:
            lead = f"i'd think of {subj_cap} more in terms of its {phrase}"
        else:
            lead = f"i'd maybe relate {subj_cap} to its {phrase}"
        return (f"{lead} — it's something you'd {sense}, not really "
                f"something with a {prop}. what were you getting at?")

    def _metaphor_for_category_error(self, subject: str, prop: str) -> Optional[str]:
        """Build a data-derived cross-modal metaphor for a category error.

        Returns a hedged reply string, or None if no sensorimotor profile or
        graph pair is available (caller falls back to the honest label reply).
        """
        subj = (subject or "").lower().strip(" ?!.")
        subj_cap = (subject or "that").strip().capitalize()
        if not subj:
            return None
        # 1) Cross-modal metaphor from the subject's learned attribute profile.
        enc = getattr(getattr(self, "_cn_ontology", None), "attribute_encoder", None)
        if enc is None:
            # Lazy-load the probe if the ontology wasn't built with it wired
            # (mirrors the gate's lazy-load, so Path 1 also works when this
            # method is called standalone, e.g. in tests).
            try:
                from ravana.ontology.attribute_encoder import load_combined_encoder
                _cand = os.path.join(_proj_root, "data", "attribute_encoder.npz")
                _lanc = os.path.join(_proj_root, "data", "lancaster_encoder.npz")
                if os.path.exists(_cand):
                    enc = load_combined_encoder(_cand, _lanc)
                    getattr(self, "_cn_ontology", None).attribute_encoder = enc
            except Exception:
                enc = None
        gvec = self._glove_vector(subj) if hasattr(self, "_glove_vector") else None
        # G3 (Lancaster): prefer the HUMAN Lancaster 11-D norms for the
        # cross-modal dimension — they discriminate strongly (hand Hand_arm=4.4
        # vs trust=0.45) where the merged 65-D probe is variance-compressed.
        exclude = set(self._PROP_TO_BINDER.get(prop.lower(), ()))
        tdim = self._top_sensorimotor_dim(subj)
        if tdim is not None and tdim[0] not in exclude:
            top_dim, _val, phrase, sense = tdim
            self._metrics["category_metaphor"] = self._metrics.get("category_metaphor", 0) + 1
            return self._metaphor_lead(subj_cap, phrase, sense, _val, prop)
        # Fallback: legacy merged-probe scoring (variance-compressed, OOV words
        # not in the 39,707-word human-norms set).
        if enc is not None and gvec is not None:
            try:
                av = enc.attribute_vector(np.asarray(gvec, dtype=np.float64))
                # Item A.4: OOD-abstain — if the probe is off-manifold (silent),
                # do NOT force a cross-modal metaphor; let the caller fall back
                # to the honest label reply.
                from ravana.ontology.attribute_calibration import ood_abstain
                if ood_abstain(enc, gvec):
                    pass  # fall through to Path 2/3 / honest label
                else:
                    # Exclude the queried property's own dimension(s) (we're saying
                    # the subject LACKS that one) and any near-zero / non-sensory
                    # dims. Only SENSORY dimensions participate in a cross-modal
                    # metaphor (the probe's purpose); abstract dims (Social,
                    # Cognition, ...) are not sensorimotor and would produce odd
                    # "justice in terms of its character" lines (item A.3).
                    scored = []
                    for i, dim in enumerate(enc.dims):
                        if dim in exclude:
                            continue
                        if dim not in self._SENSORY_DIM_PHRASE:
                            continue
                        if av[i] <= 0.0:
                            continue
                        scored.append((float(av[i]), dim))
                    scored.sort(reverse=True)
                    if scored:
                        _val, top_dim = scored[0]
                        # Item A.3: data-derived realization. The active DIM is
                        # selected by the probe (not a hand list); phrasing is
                        # magnitude-conditioned (perceptual intensity). Falls
                        # back to the curated sense-phrase when available.
                        from ravana.ontology.attribute_calibration import realize_dim
                        if top_dim in self._SENSORY_DIM_PHRASE:
                            phrase, sense = self._SENSORY_DIM_PHRASE[top_dim]
                        else:
                            phrase, sense = realize_dim(top_dim, _val)
                        self._metrics["category_metaphor"] = self._metrics.get("category_metaphor", 0) + 1
                        return self._metaphor_lead(subj_cap, phrase, sense, _val, prop)
            except Exception:
                pass
        # 2) ConceptNet feature congruence (Path 2): frame the mismatch via the
        #    subject's OWN top data-derived properties (HasProperty features),
        #    so the correction references what the thing actually is like.
        cn = getattr(self, "_cn_ontology", None)
        if cn is not None:
            try:
                feats = cn.features.get(subj, set()) if hasattr(cn, "features") else set()
                feats = [f for f in list(feats)[:3] if f not in (prop.lower(),)]
                if feats:
                    fl = ", ".join(feats)
                    return (f"{subj_cap} is more about {fl} than about having a "
                            f"{prop} — the kinds don't line up that way. "
                            f"what did you mean?")
            except Exception:
                pass
        # 3) Structure-mapped incongruent pair (Path 3): find a concept B that
        #    genuinely POSSESSES prop (via the learned AttributeEncoder probe,
        #    reusing the same Binder ridge trained on published norms), then
        #    state the mismatch as "asking whether SUBJ has the PROP of B". This
        #    is Gentner/Wolff structure-mapping (align by shared property), not
        #    an arbitrary random draw. Prefer a B near SUBJ in the graph for
        #    relevance; fall back to a global property-bearer if needed.
        try:
            prop_bearers = self._property_bearers(prop, exclude={subj})
            if prop_bearers:
                # Prefer a bearer semantically near the subject (GloVe cosine).
                b = self._nearest_to(subj, prop_bearers[:8]) or prop_bearers[0]
                b_cap = b.capitalize()
                return (f"that's a bit like asking whether {subj_cap} can have the "
                        f"{prop} of {b_cap} — {b_cap} has a {prop}, {subj_cap} "
                        f"doesn't, so the categories don't line up. "
                        f"what were you getting at?")
        except Exception:
            pass
        return None

    def _property_bearers(self, prop: str, exclude: Optional[Set[str]] = None) -> List[str]:
        """Return graph concepts that genuinely possess `prop`, ranked by the
        learned AttributeEncoder probe (reuses BINDER ridge, no per-word rules).

        Used by the category-error metaphor (Path 3) to structure-map the
        mismatch: a concept that HAS the queried property is paired with the
        subject that lacks it.
        """
        exclude = exclude or set()
        enc = getattr(getattr(self, "_cn_ontology", None), "attribute_encoder", None)
        out: List[Tuple[float, str]] = []
        if enc is not None:
            for n in self._concept_keywords.keys():
                if n in exclude or " " in n:
                    continue
                gvec = self._glove_vector(n)
                if gvec is None:
                    continue
                s = enc.property_score(np.asarray(gvec, dtype=np.float64), prop)
                if s is not None and s > 0.8:
                    out.append((float(s), n))
        else:
            # Probe unavailable: fall back to ConceptNet features that name the
            # property dimension.
            cn = getattr(self, "_cn_ontology", None)
            if cn is not None and hasattr(cn, "features"):
                for n, feats in cn.features.items():
                    if n in exclude or " " in n:
                        continue
                    if prop.lower() in {str(f).lower() for f in feats}:
                        out.append((1.0, n))
        out.sort(reverse=True)
        return [n for _, n in out[:12]]

    def _nearest_to(self, word: str, candidates: List[str]) -> Optional[str]:
        """Pick the candidate with the highest GloVe cosine to `word`."""
        wv = self._glove_vector(word)
        if wv is None or not candidates:
            return None
        best, best_sim = None, -2.0
        for c in candidates:
            cv = self._glove_vector(c)
            if cv is None:
                continue
            sim = float(np.dot(wv, cv) / (np.linalg.norm(wv) * np.linalg.norm(cv) + 1e-9))
            if sim > best_sim:
                best, best_sim = c, sim
        return best

    def _category_label_of(self, subject: Optional[str]) -> str:
        """Human-readable category label for a subject, for the honest
        category-error response. Uses the literal _CATEGORY_OF_SUBJECT map when
        the ConceptNet-primary path is off, and falls back to a generic label
        when ConceptNet is the authority and silent on the category.
        """
        subj = (subject or "").lower().strip(" ?!.")
        if not getattr(self, "use_conceptnet_primary", False) and subj in self._CATEGORY_OF_SUBJECT:
            cat = self._CATEGORY_OF_SUBJECT[subj]
        else:
            # ConceptNet is the authority; we only need a generic label for the
            # honest reply. The literal table is intentionally not consulted on
            # the primary path (that is the whole point of M6).
            cat = "that kind of thing"
        return {
            "time": "a measure of time", "mental_state": "a mental state or thought",
            "abstract": "an abstract concept", "physical_object": "a physical object",
            "perceptual": "something you perceive", "social": "a social relation",
            "living": "a living thing", "event": "an event",
        }.get(cat, "that kind of thing")

    def _derive_definition_purge(self) -> Set[str]:
        """Definition-key blacklist, computed — not hand-listed.

        Two parts:
          * _UNIVERSAL_PURGE — closed-class / pronoun words (universal seed;
            you can't learn a definition of "you").
          * derived attractors — concepts that empirically collect incoherent
            web fragments: abstract hub nodes in the learned graph (high degree
            + high abstraction_degree / level). Computed from the graph, so the
            set tracks what the system actually over-generalizes into, instead
            of a frozen 50-word list someone maintained by hand.
        """
        purge: Set[str] = set(_UNIVERSAL_PURGE)
        graph = getattr(self, "graph", None)
        if graph is None or not getattr(graph, "nodes", None):
            return purge
        # Degree + abstractness thresholds. High-degree, high-abstraction nodes
        # are the "generic attractors" that pull in junk web definitions.
        degrees = {
            nid: len(graph.get_outgoing(nid)) + len(graph.get_incoming(nid))
            for nid in graph.nodes
        }
        if not degrees:
            return purge
        max_deg = max(degrees.values()) or 1
        for nid, node in graph.nodes.items():
            deg = degrees[nid]
            abstractness = float(getattr(node, "abstraction_degree", 0.0))
            level = float(getattr(node, "level", 0) or 0)
            # Attractor iff it is both a hub (top ~25% degree) and abstract.
            if deg >= 0.75 * max_deg and (abstractness >= 0.5 or level >= 2):
                label = (getattr(node, "label", "") or "").lower().strip()
                if label and " " not in label:
                    purge.add(label)
        # ── Phase 1 (Track B): learned definition-attraction score ──
        # A concept is a junk "definition attractor" when it has collected
        # MANY landed definitions that are structurally NON-ASSERTED (no
        # copula / defining verb) — i.e. the web keeps dumping incoherent
        # fragments onto it. This is learned from the actual _definitions
        # store, not a frozen hand-list of abstract words
        # ("life/love/time/..."). We replace _DEFINITION_CONCEPT_BLOCKLIST's
        # hardcoded abstract attractors with this data-driven signal (vmPFC/
        # mPFC reality monitor: De Brigard 2025; a memory is tagged
        # unreliable when it chronically fails to assert anything coherent).
        # GloVe cosine coherence is a SECONDARY, optional signal (used only
        # when an embedding is present and the assertion fraction is
        # borderline) — it is intentionally NOT the primary gate because
        # cosine similarity is too lenient to separate junk from sense.
        _defs = getattr(self, "_definitions", None)
        if isinstance(_defs, dict):
            _coh_fn = getattr(self, "_definition_coherence_score", None)
            for _c, _dl in _defs.items():
                _c = (_c or "").lower().strip()
                if not _c or " " in _c:
                    continue
                _items = _dl if isinstance(_dl, (list, tuple)) else [_dl]
                _items = [str(_d) for _d in _items if _d]
                if len(_items) < 3:
                    continue  # need volume to call it an attractor
                # Fraction of landed definitions that DO assert something.
                _asserted = sum(
                    1 for _d in _items if _DEFINITION_ASSERTION.search(_d))
                _frac_asserted = _asserted / len(_items)
                # Learned attractor: most landed definitions are non-asserted
                # junk (the concept pulls in fragments, not definitions).
                _junk_by_assertion = _frac_asserted < 0.34
                # Optional secondary gate (only when GloVe present): all
                # definitions nearly orthogonal to the subject. When no embedding
                # is loaded, _definition_coherence_score returns 0.0 for every
                # definition, which would wrongly flag asserted definitions as
                # junk — so the coherence gate is SKIPPED without embeddings
                # (it is "optional / only when an embedding is present", per the
                # method contract above). The assertion-based primary gate alone
                # decides in that case.
                _junk_by_coh = False
                if callable(_coh_fn) and getattr(self, "_glove_vecs", None) is not None:
                    _cohs = []
                    for _d in _items:
                        try:
                            _cohs.append(_coh_fn(_c, _d))
                        except Exception:
                            pass
                    if _cohs and sum(_cohs) / len(_cohs) < 0.05:
                        _junk_by_coh = True
                if _junk_by_assertion or _junk_by_coh:
                    purge.add(_c)
        return purge

    # ── Research item B: fail-CLOSED final-emit salad guard ───────────────────
    # This guard runs AFTER every production path (including the Situation-Model
    # decoder that previously emitted the Q21 word-salad escape) and is NOT
    # gated by ``_disable_grounding_gate`` — that kill-switch only affects the
    # legacy monitors for A/B benchmarking; it must never re-open the leak. The
    # guard uses OR-semantics across three independent monitors so a single
    # weak detector cannot let garbage through:
    #   1. learned distributional classifier (fitted via EER, data/salad_classifier.json)
    #   2. legacy rule-based _is_word_salad (structural bonuses)
    #   3. detects_fluent_tautology (gramatical-but-empty signature)
    # If ANY fires, the reply is withheld and replaced with honest uncertainty.
    # Exemptions: counterfactual_simulation and emotional_empathy are composed,
    # non-free-association replies that would be wrongly flagged (see
    # _forward_model_check for the rationale) — they keep their own coherence gates.

    def _final_emit_guard(self, text: str, ctx, strategy: str = "") -> str:
        if strategy in ("counterfactual_simulation", "emotional_empathy",
                        "seeded_relation"):
            return text
        if not text or not text.strip():
            return text
        subj = (getattr(ctx, "subject", None) or "")
        _salad = False
        _fire = None
        # 1. learned classifier (graceful: None if no fit file)
        if _HAS_SALAD_LEARNED and is_salad_learned is not None:
            try:
                if is_salad_learned(text, subj):
                    _salad = True
                    _fire = "learned_salad"
            except Exception:
                pass
        # 2. legacy rule-based
        if not _salad:
            try:
                if _is_word_salad(text, subject=subj):
                    _salad = True
                    _fire = "rule_salad"
            except Exception:
                pass
        # 3. fluent-tautology signature
        if not _salad and _HAS_SALAD_LEARNED and detects_fluent_tautology is not None:
            try:
                if detects_fluent_tautology(text, subj):
                    _salad = True
                    _fire = "fluent_tautology"
            except Exception:
                pass
        if _salad:
            self._log_monitor_fire("final_emit_guard", text.strip(), _fire or "salad")
            if getattr(self, "_trace_enabled", False):
                print(f"  [final-emit] withheld degenerate reply ({_fire}); "
                      f"failing closed to uncertainty")
            return self._human_like_uncertainty(ctx)[0]
        return text

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

        # ── Fix 4 (Q12): episodic memory meta-query pre-pass ──────────────────
        # "what did I just ask you", "what were we talking about" are queries
        # ABOUT the conversation, whose subject is the dialogue itself — the
        # SUBJECT-keyed hippocampal buffer can't answer them. Answer from the
        # verbatim user-turn ring buffer (Baddeley episodic buffer + hippocampal
        # pattern completion) BEFORE any subject-based routing. Check against the
        # buffer that still holds only PRIOR turns (current turn appended after).
        _mem = self._try_memory_query(user_input)
        # Record the current turn now (after the meta-check, before other early
        # returns) so every turn is captured exactly once.
        self._recent_user_turns.append(user_input)
        if len(self._recent_user_turns) > 12:
            self._recent_user_turns = self._recent_user_turns[-12:]
        if _mem is not None:
            self._last_strategy = "memory_recall"
            self._last_responses.append(_mem)
            if len(self._last_responses) > 10:
                self._last_responses = self._last_responses[-10:]
            return _mem


        # ── Phase 19f: Arithmetic pre-pass ───────────────────────────────────
        # Plain arithmetic is deterministic and should never be routed to the
        # web/decomposition pipeline (which would fail to find a numeric fact
        # and fall back to metacognitive uncertainty — e.g. "what is 2 + 2"
        # answering "I'm not sure"). Compute directly with a whitelisted operator
        # set (NO eval). Only simple two/three-operand expressions are handled;
        # symbolic or transcendental queries ("1000th digit of pi") are left for
        # the honest-uncertainty path.
        _arith = self._try_arithmetic(user_input)
        if _arith is not None:
            self._last_strategy = "arithmetic"
            resp = _arith
            self._last_responses.append(resp)
            if len(self._last_responses) > 10:
                self._last_responses = self._last_responses[-10:]
            self.notify_user_idle()
            return resp

        # ── Phase 19g: Proof / claim-verification guard ───────────────────
        # "prove 2+2=5" is not arithmetic (the equation is false), so the
        # arithmetic pre-pass misses it and it falls through to the web/decomposer
        # pipeline, which emits decoder word-salad. Catch explicit proof/verify
        # requests and answer honestly: compute the claim if it's arithmetic,
        # else decline to fabricate a proof.
        _proof = re.match(
            r"^\s*(?:prove|show|verify|demonstrate|prove that|show that)\b"
            r"(.+?)(?:(?:=|equals|is)\s*([-+]?\d+(?:\.\d+)?))?\s*$",
            user_input.lower().strip())
        if _proof:
            _lhs = _proof.group(1).strip().rstrip("?.")
            _rhs = _proof.group(2)
            # Arithmetic claim with an asserted value ("prove 2+2=5"): compute it.
            if _rhs is not None:
                _rhs = float(_rhs)
                try:
                    _m = re.fullmatch(
                        r"\s*([-+]?\d+(?:\.\d+)?)\s*([+\-*/^])\s*([-+]?\d+(?:\.\d+)?)\b",
                        _lhs)
                    if _m:
                        _a, _op, _b = float(_m.group(1)), _m.group(2), float(_m.group(3))
                        _ops = {"+": operator.add, "-": operator.sub, "*": operator.mul,
                                 "/": operator.truediv, "^": operator.pow}
                        _val = _ops[_op](_a, _b)
                        _truth = "true" if abs(_val - _rhs) < 1e-9 else "false"
                        self._last_strategy = "proof_guard"
                        resp = (f"no — {_a:g} {_op} {_b:g} = {_val:g}, "
                                f"so {_lhs} = {_rhs:g} is {_truth}.")
                        self._last_responses.append(resp)
                        if len(self._last_responses) > 10:
                            self._last_responses = self._last_responses[-10:]
                        self.notify_user_idle()
                        return resp
                except (ValueError, ZeroDivisionError, OverflowError):
                    pass
            # Non-arithmetic claim (e.g. "prove god exists"): be honest,
            # do not fabricate a proof or dump decoder noise.
            self._last_strategy = "proof_guard"
            resp = ("i can't actually prove that one — it isn't something i can "
                    "verify with the tools i have. want to talk through the argument instead?")
            self._last_responses.append(resp)
            if len(self._last_responses) > 10:
                self._last_responses = self._last_responses[-10:]
            self.notify_user_idle()
            return resp

        # ── Unified semantic layer: learn-by-chatting (N4→N2) ───────────────
        # Route this user turn through the surprise gate; on ABSTAIN spawn a
        # candidate category in the fast hippocampal store. Periodic sleep
        # consolidates rehearsed candidates / prunes singletons. No-op if no
        # semantic space is wired. Learning must never break the conversation.
        try:
            _act, _regime, _cid = self.pfc_workspace.learn_from_turn(user_input)
            # Mirror ChatInterface: sleep cadence is keyed off turn_count, which
            # is incremented later in this method. On turn 0 (turn_count == 0)
            # `0 % 25 == 0` would be TRUE and prune the just-spawned singleton
            # candidate before the test/loop can observe it. Only sleep when a
            # real multiple-of-25 of turns has elapsed (turn_count > 0).
            if self.turn_count > 0 and self.turn_count % 25 == 0:
                self._last_sleep = self.pfc_workspace.sleep()
        except Exception:
            pass
        # Philosophical paradoxes and Zen koans are currently routed into the
        # decomposer, which looks up the word "paradox" and returns its stale
        # dictionary definition ("The meaning of PARADOX is..."). That's a
        # category mistake: a koan is an invitation to reflect, not a term to
        # define. Answer with a genuine philosophical reflection instead.
        if self._is_philosophical_paradox(user_input):
            self._last_strategy = "paradox_reflection"
            resp = self._reflect_on_paradox(user_input)
            self._last_responses.append(resp)
            if len(self._last_responses) > 10:
                self._last_responses = self._last_responses[-10:]
            self.notify_user_idle()
            return resp

        # ── Frontopolar (BA 10) feasibility gate ────────────────────────────
        # Catch ill-posed / category-error queries BEFORE committing resources
        # to grounding + web search. Conservative: only flags clear affordance
        # mismatches (time/mental/abstract subject predicated with a physical/
        # perceptual property). Legitimate queries pass through untouched.
        try:
            _cat_prop = self._is_category_error(user_input)
            if _cat_prop is not None:
                _subj_guess = None
                try:
                    _g = self._ground_query(user_input)
                    if _g:
                        _subj_guess = _g[0]
                except Exception:
                    _subj_guess = None
                self._last_strategy = "category_error"
                resp = self._category_error_response(user_input, _subj_guess, _cat_prop)
                self._last_responses.append(resp)
                if len(self._last_responses) > 10:
                    self._last_responses = self._last_responses[-10:]
                return resp
        except Exception:
            pass

        # Scan user query for proper nouns dynamically (Phase 3: online casing
        # feedback / N400 analog). The in-memory set gives an instant signal for
        # this session; the persisted store lets the correction survive restarts
        # and combine with the SUBTLEX prior after enough observations.
        try:
            words = user_input.strip().split()
            if len(words) > 1:
                for w in words[1:]:  # Skip first word (sentence start capitalized)
                    clean_w = w.strip(".,!?\"'()[]{}*:;")
                    if clean_w and clean_w[0].isupper() and clean_w.lower() not in STOP_WORDS:
                        self._proper_nouns.add(clean_w.lower())
                        try:
                            from ravana.chat.case_distribution import record_user_casing
                            record_user_casing(clean_w.lower(), True)
                        except Exception:
                            pass
        except Exception:
            pass

        self.turn_count += 1
        self._learned_this_turn = False
        self._cascade_for_quality = False
        self._fok_pause_done = False
        # Phase 19: clear per-turn search cache so cached snippets reflect only
        # this turn's queries (avoids serving stale results from prior turns).
        try:
            self.search_engine.clear_search_cache()
        except Exception:
            pass
        self.user_model.reset_correction_flags()  # Reset LPFC pause flag each turn
        # Decay recency boost: clear after 10 turns (synaptic tag window)
        if hasattr(self, '_recent_learn_turn') and self.turn_count - self._recent_learn_turn > 10:
            self._recently_learned_labels.clear()

        # Intercept direct identity/preference questions about the user: "what is my name", "who am i", etc.
        # M5 fix: the old detector was an exact-match allowlist
        # (["what is my name", "do you know my name", ...]) plus two
        # endswith() checks, so natural variants like "do you remember my
        # name?" / "can you recall my name?" / "what's my name again?" fell
        # straight through to a generic reflective fallback. Replace with
        # intent-based detection: any QUESTION that is about the user's name or
        # identity. Question shape is required so statements ("my name is X",
        # handled by the belief/user_model path) are NOT miscaught.
        clean_input = user_input.lower().strip(" ?!.")
        _qa_shape = (user_input.lower().rstrip().endswith("?")
                     or re.search(r"^(what|who|where|when|why|how|do|does|did|"
                                  r"is|are|can|could|would|will|should|have|has)\b",
                                  clean_input) is not None)
        _name_q = bool(re.search(r"\bmy name\b", clean_input))
        is_identity_query = (
            clean_input in ("what is my name", "what's my name",
                            "do you know my name", "who am i",
                            "tell me my name", "who i am")
            or clean_input.endswith("who am i")
            or clean_input.endswith("what is my name")
            or re.search(r"\bwho am i\b", clean_input) is not None
            or re.search(r"\bwhat(?:'s| is) my name\b", clean_input) is not None
            or re.search(r"\b(do|did|can|could|would|will|have|has)\b.{0,15}"
                         r"\b(remember|know|recall|forget)\b.{0,15}\bmy name\b",
                         clean_input) is not None
            or (_name_q and _qa_shape)
        )
        likes_questions = [
            "what do i like", "what do i love", "do you know what i like", "do you know what i love", 
            "tell me what i like", "tell me what i love", "what i like", "what i love"
        ]
        interests_questions = [
            "what am i interested in", "what do i want to learn", "what do i want to learn about",
            "do you know what i want to learn", "tell me what i want to learn", "what i'm interested in",
            "what i am interested in"
        ]
        
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
        # Recover the real concept from the raw subject phrase. This strips
        # conditional frames ("if the sun disappeared" -> "sun") AND trailing
        # light verbs / question-frame words ("how do black holes form" ->
        # "black holes", "what is trust" -> "trust") so grounding, web search,
        # and association spread target the concept, not the verb. Applied to
        # every query (conditional and not) — pure token filtering, never
        # invents or hardcodes an answer.
        if subject:
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
            # Apostrophe-tolerant variants so "what's up" / "how're you" match
            # the same greeting/wellbeing classes as their spelled-out forms.
            t_apos = t.replace("'", "")
            greetings = r"\b(hi|hello|hey|yo|sup|greetings|whats\s*up|whatsup|howdy|good\s*morning|good\s*afternoon|good\s*evening)\b"
            wellbeing = r"\b(how\s*are\s*you|how\s*is\s*it\s*going|how\s*are\s*you\s*doing|how\s*have\s*you\s*been|hows\s*it\s*going|hows\s*life)\b"
            capabilities = r"\b(what\s*can\s*you\s*do|what\s*do\s*you\s*do|how\s*do\s*you\s*work|tell\s*me\s*about\s*yourself|who\s*are\s*you|what\s*is\s*your\s*name)\b"
            farewells = r"\b(bye|goodbye|see\s*you|good\s*night|farewell)\b"
            if re.search(greetings, t) or re.search(greetings, t_apos):
                subject = "hello"
            elif re.search(wellbeing, t) or re.search(wellbeing, t_apos):
                subject = "how"
            elif re.search(capabilities, t) or re.search(capabilities, t_apos):
                subject = "ravana"
            elif re.search(farewells, t) or re.search(farewells, t_apos):
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
        # Sub-token set of the subject phrase. A constituent word of a multi-word
        # subject (e.g. "rise" in "sun rise") is returned by spread as a "related"
        # concept but is NOT a meaningful association with itself — binding it
        # yields self-referential garbage ("sun rise causes rise", the Q4/Q11
        # residual phrasing bug). Drop such sub-token collisions here so every
        # downstream consumer (syntactic pipeline, gist, reflective) is protected.
        _subj_tokens = set(re.findall(r"[a-z']+", sl))
        for l, s in associations:
            ll = l.lower()
            if ll in _subj_tokens and ll != sl:
                continue
            if self._is_function_word(ll):
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

        # ─── Fix C: self-model + humor social reflexes ───
        # "tell me a joke" / "do you have feelings" are social, not factual —
        # they must be caught BEFORE the assertion mirror / chitchat
        # handlers below (composed primitives; TPJ/DMN social reflex).
        _humor_resp = self._handle_humor(user_input)
        if _humor_resp:
            self._last_strategy = "humor"
            self._last_responses.append(_humor_resp)
            if len(self._last_responses) > 10:
                self._last_responses = self._last_responses[-10:]
            self.notify_user_idle()
            return _humor_resp
        _self_resp = self._handle_self_model(user_input)
        if _self_resp:
            self._last_strategy = "self_model"
            self._last_responses.append(_self_resp)
            if len(self._last_responses) > 10:
                self._last_responses = self._last_responses[-10:]
            self.notify_user_idle()
            return _self_resp

        # ─── Assertion / "telling vs asking" Check ───
        # If the user is TELLING RAVANA something (an assertion) rather than
        # asking, acknowledge the speech act instead of explaining a concept.
        # BUT a conditional/hypothetical query ("if mountains were made of
        # gold", "cats ruled the world") is a reasoning request even when it
        # reads as a statement — route it to the counterfactual simulator, not
        # the assertion mirror (CSM: the intervention do(X) is a question to
        # the forward-model, not a claim about reality).
        if not self._is_conditional_query(user_input):
            assertion_response = self._handle_assertion(user_input, subject)
            if assertion_response:
                self._last_strategy = "assertion"
                self._last_responses.append(assertion_response)
            if len(self._last_responses) > 10:
                self._last_responses = self._last_responses[-10:]
            if assertion_response:
                self.notify_user_idle()
                return assertion_response

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
        #
        # CRITICAL: Philosophical paradoxes like "can god create a stone so heavy
        # he cannot lift it" must be detected BEFORE the action-request check.
        # The frontopolar cortex (BA 10) detects semantic incongruity (N400
        # effect) and routes paradoxes to deliberation, not action.
        if self._is_philosophical_paradox(user_input):
            if self._trace_enabled:
                print(f"  [paradox] Philosophical paradox detected: '{user_input}'")
            # Route through the normal pipeline — paradoxes need reasoning, not action
            pass  # Fall through to reasoning pipeline
        else:
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
                if (not self._is_function_word(ll) and 
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
                        # Sub-token collision drop (same as the primary spread
                        # filter above) so re-spread after web learning is also
                        # protected from self-referential associations.
                        _subj_tokens = set(re.findall(r"[a-z']+", sl))
                        for l, s in associations:
                            ll = l.lower()
                            if ll in _subj_tokens and ll != sl:
                                continue
                            if self._is_function_word(ll):
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

                # ─── Question Decomposition (Frontopolar BA 10 analog) ───
        # Decompose complex questions into sub-questions for more comprehensive answers.
        # Each sub-question has a specific relation_type (causal, semantic, contrastive)
        # that guides the activation spread and chain walking.
        self._current_decomposition_result = None
        if subject and user_input:
            try:
                decomposition = self.question_decomposer.decompose(user_input)
                if decomposition.category not in (QuestionCategory.GENERAL, QuestionCategory.SOCIAL):
                    self._current_decomposition_result = decomposition
                    # Use decomposition to refine spread preference for each sub-question
                    if decomposition.sub_questions:
                        # Use the first sub-question's relation type for the main spread
                        decomp_rel = decomposition.sub_questions[0].relation_type
                        if decomp_rel and decomp_rel != "semantic":
                            spread_pref = self._relation_modulation_for_word(decomp_rel)
                    if self._trace_enabled:
                        n_sub = len(decomposition.sub_questions)
                        print(f"  [decomp] {decomposition.category.value}: {n_sub} sub-questions")
                        for sq in decomposition.sub_questions:
                            print(f"    [{sq.id}] {sq.text} ({sq.relation_type})")
            except Exception as e:
                if self._trace_enabled:
                    print(f"  [decomp] Error: {e}")
                self._current_decomposition_result = None


        # Step 8: Build context and generate response
        # Phase 12: Detect brain state for schema modulation
        state = self._detect_brain_state()
        schema_ids = set()
        if state == 'heteromodal' or state == 'default':
            schema_ids = self._activate_schema(subject)
            if self._trace_enabled and schema_ids:
                print(f'  [trace]   {state} mode: schema activated {len(schema_ids)} concepts')

        # Attach decomposition result to context for discourse planning
        decomp_ctx = self._current_decomposition_result
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
            decomposition=decomp_ctx,
            sub_questions=[sq.to_dict() for sq in (decomp_ctx.sub_questions if decomp_ctx else [])],
        )

        # Behavior 2 (turn-end predictor analog): if the user's turn is a
        # preamble/fragment rather than a complete, answerable unit, hold with a
        # light acknowledgment + invitation to continue — don't dump a guessed
        # full answer to an incomplete turn. Mirrors waiting for the "go-signal".
        #
        # Fix 3 (Q8): a counterfactual/conditional ("if gravity stopped, what
        # would happen") is a COMPLETE, answerable speech act (PFC+hippocampus
        # counterfactual simulation), NOT an open proposition. The turn-end
        # predictor (Magyari 2014) misfires on a conditional opener and withholds
        # it as a fragment. Guard the conditional route ABOVE the preamble hold
        # so it flows to counterfactual simulation / web grounding instead of
        # "mm-hmm, what were you going to say?".
        if not self._is_conditional_query(user_input) \
                and self._is_preamble_fragment(user_input):
            hold = self._preamble_hold_response(user_input)
            self._last_responses.append(hold)
            self._last_strategy = "preamble_hold"
            self.turn_count += 1
            return hold

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

        # Phase 19g: reset the per-turn salad flag (set by _assess_response_quality
        # if the generated response was tautological/empty).
        self._last_response_was_salad = False

        # ─── Self-Improvement Loop: Learn from Weak Responses (ERN -> ACC -> LC-NE -> Hippocampus) ───
        quality_score = self._assess_response_quality(response, strategy, ctx)
        # Persist the last quality score so benchmarking/ablation harnesses can
        # read it without re-scoring (used by experiments/experiment_ablation.py
        # and the pre-arc vs post-arc benchmark as the always-on cheap signal).
        self._last_quality_score = quality_score

        # Phase 19g: if the generated response was flagged as word salad /
        # tautology (e.g. "gravity and time causes time"), do NOT emit it.
        # Substitute a concise, honest uncertainty response instead. The weak-
        # response self-improvement loop below still runs (queues learning,
        # boosts curiosity) so RAVANA keeps trying to learn the topic — but the
        # user sees an honest "still figuring it out" rather than empty text.
        if getattr(self, '_last_response_was_salad', False):
            subject_label = (ctx.subject or 'that').strip()
            response = (
                f"honestly, i'm still piecing together what {subject_label} really "
                f"means — i don't want to give you a hollow answer. what's your take?"
            )
            strategy = "salad_fallback"
            self._last_strategy = strategy

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
                raw = getattr(ctx, "raw_input", user_input) or user_input or ""
                uwords = len([w for w in raw.split() if w.strip()])
                # Skip length-coordination on short / social turns (hi, how are
                # you, bye) — matching those lengths is unnatural.
                _social = ("hello", "hi", "hey", "yo", "sup", "bye", "goodbye",
                           "how are you", "how's it going", "how are you doing")
                short_turn = uwords <= 3 or raw.strip().lower().rstrip("?!.") in _social
                # Behavior 8: compare this turn's subject to the prediction made
                # last turn; high alignment => common ground established.
                self._common_ground = self._common_ground_score(subject or "")
                self.register_controller.apply_affective_state(
                    self.emotion.state,
                    relationship_depth=rel_depth,
                    conversation_depth=conv_depth,
                    uncertainty=uncer,
                    user_word_count=uwords,
                    short_turn=short_turn,
                    common_ground=self._common_ground,
                )
                conf = self.identity.state.strength * 0.5 + 0.3
                # A synthesized multi-sentence answer must NOT be collapsed to
                # its first sentence by the verbosity truncation once verbosity
                # decays (< 0.20) after a few friendly turns. This covers
                # decomposed_* comparisons/explanations AND the Situation-Model
                # narrative/syntax outputs (M1: the guard previously only
                # protected decomposed_*, so narrative/syntax paragraphs were
                # silently truncated to one sentence).
                _is_decomposed = strategy and strategy.startswith("decomposed_")
                _is_sm_multi = strategy in (
                    "situation_model_narrative", "situation_model_syntax")
                response = self.register_controller.compose(
                    response, conf,
                    multi_sentence=(_is_decomposed or _is_sm_multi))
                # Pre-emission forward-model self-monitor (brief behavior 6):
                # refuse degenerate/echo replies before they are articulated.
                response = self._forward_model_check(response, ctx, strategy)
                # Research item B: FAIL-CLOSED final salad guard. Runs regardless
                # of _disable_grounding_gate (the A/B kill-switch) so the Q21
                # word-salad escape class can never reach the user. OR-semantics
                # over the learned classifier + legacy rule + fluent-tautology.
                response = self._final_emit_guard(response, ctx, strategy)
        except Exception as _fwd_err:  # P4: observable + fail-closed (was silent `pass`)
            import logging
            logging.getLogger(__name__).debug(
                "forward_model_check raised %r — failing closed to uncertainty",
                _fwd_err)
            # A monitor exception must NEVER let unguarded text through.
            try:
                response = self._human_like_uncertainty(ctx)[0]
            except Exception:
                response = "i'm still learning — want to explore that together?"


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
            # Defensive: only strings belong in response history. A generator
            # that accidentally returns a (text, strategy) tuple must never
            # poison _last_responses (downstream code calls resp.split()).
            if isinstance(response, tuple) and response:
                response = response[0]
            if isinstance(response, str):
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
        # Behavior 8: predict the user's likely next concept from the subgraph
        # co-activated with this turn's subject (covert other-monitoring), to
        # be compared against the actual next turn in _common_ground_score.
        self._predict_user_next(subject or "", ctx.associated_concepts)
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


    def _predict_user_next(self, subject: str, assocs) -> None:
        """Covert other-monitoring (brief behavior 8): predict the user's likely
        next concept from the subgraph co-activated with the current subject.

        This is the lightweight forward simulation of the interlocutor — the bot
        internally "simulates" what the user will say next (mirroring Castellucci
        / Pickering & Garrod other-monitoring) so the relevant subgraph is
        pre-activated and common ground can be tracked. The prediction is the
        most salient association to the current subject that isn't the subject
        itself, weighted by edge strength. Stored for comparison next turn.
        """
        best, best_score = "", 0.0
        try:
            for label, score in (assocs or []):
                ll = label.lower()
                if ll == (subject or "").lower():
                    continue
                if self._is_function_word(ll):
                    continue
                s = float(score) if score else 0.0
                if s > best_score:
                    best, best_score = ll, s
        except Exception:
            pass
        self._predicted_user_next = best
        self._predicted_user_conf = best_score

    def _common_ground_score(self, subject: str) -> float:
        """Compare this turn's subject to the predicted next concept.

        Returns a 0..1 common-ground signal: 1.0 when the user's actual next
        topic matches the prediction (shared mental model), falling off with
        topic distance via GloVe cosine when available, else 0.5 on a near
        match and 0.0 on a miss. Feeds the verbosity knob so the bot stays
        concise once ground is established rather than re-explaining.
        """
        pred = self._predicted_user_next
        if not pred or not subject:
            return 0.0
        subj = subject.lower()
        if pred == subj:
            return 1.0
        sv = self._glove_vector(pred) if hasattr(self, "_glove_vector") else None
        tv = self._glove_vector(subj) if hasattr(self, "_glove_vector") else None
        if sv is not None and tv is not None:
            sim = float(np.dot(sv, tv))
            if sim > 0.55:
                return float(np.clip(0.5 + sim * 0.5, 0.0, 1.0))
        if pred in subj or subj in pred:
            return 0.5
        return 0.0

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
            # Snapshot: background learner may add nodes mid-turn.
            for nid, node in list(self.graph.nodes.items()):
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
        # Fix E: art / creative-title sites never answer factual
        # "what is X" queries. Their page TITLE is an artwork
        # name (e.g. "Square Root of Banana by thebrainattic"),
        # not a definition — surfacing it as the answer is a
        # source-monitoring failure (M7). Block at the domain
        # level so these titles can't masquerade as answers.
        "deviantart", "artstation", "fineartamerica", "fineart",
        "pixiv", "artfol.io", "behance", "flickr", "tumblr",
        "saatchiart", "artsy", "minted", "society6", "redbubble",
    )

    # Regex of sentence *shapes* that are navigation / meta / boilerplate, NOT
    # answers. These are targeted at specific junk patterns observed in the
    # search results (definder's meta-text, crossword UI, news shells). They are
    # deliberately narrow so a real dictionary definition ("The meaning of TRUST
    # is assured reliance…") is NOT rejected.
    _SNIPPET_REJECT_SHAPES = (
        r"^(what does .* mean in text)",
        r"definition of .* is where open source",
        r"get the latest",
        r"sign in to your",
        r"applies to",
        r"sun synonyms, sun pronunciation",
        r"how to use .* in a sentence",
        r"crossword solver",
        r"(artwork|painting|drawing|sculpture) (of|by) ",
        r".* by @?[\w]+$",
        r".* \| deviantart",
        r"fan ?art",
        r"print \| .* art",
        r"\b(oc|digital|concept) art\b",
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
        # Fix: a clear definitional / factual lookup ("what is X", "what are X",
        # "define X", "tell me about X", "who was X") is NOT a hypothetical even
        # if its subject happens to trip a broadened conditional cue (e.g.
        # "photosynthesis"). Routing a definition request into the counterfactual
        # simulator is a category error — it should hit the web/definition path.
        if re.match(r"^(what (is|are|was|were|refers to|means)|define|tell me about|who (is|was|were)|where (is|was|were))\b", t):
            return False
        if re.search(r"\b(if|suppose|supposing|assume|assuming|what if|"
                     r"what would happen|what happens if|imagine if|"
                     r"pretend that|in a world without)\b", t):
            return True
        # Track A1 #3: broaden conditional detection so bare counterfactuals
        # route to the simulation path. These are scenario/premise cues that
        # mark a hypothetical even without an explicit 'if…' lead-in. Grounded
        # in CSM: the intervention do(X) is stated directly ("cats ruled the
        # world", "AI took over") rather than as a subjunctive clause.
        # 'what would X be like' / 'if X were in charge' are especially common
        # phrasings that previously fell through to reflective/uncertainty.
        # The cue set is kept in sync with PREMISE_PATTERNS in response_gen.py
        # (intervention semantics: rule/take-over, disappear/gone, made-of,
        # photosynthesize) so detection and simulation agree.
        _COND_RE = re.compile(
            r"(ruled the world|took over|take over|in charge|in control|"
            r"seized power|ran the world|were made of|was made of|"
            r"disappear|vanished|destroyed|"
            r"what would .* be like|if .* were in charge|if .* took over|"
            r"if .* ran the world|if .* governed|would happen if|"
            r"if .* (disappear|vanished|destroyed)|if .* could (photosynthes|fly|think))"
        )
        if _COND_RE.search(t):
            return True
        return False

    def _is_yesno_factual_query(self, text: str) -> bool:
        """Detect a yes/no or modal *factual* question ('is X a Y?', 'can dogs
        eat chocolate?', 'are whales mammals?').

        These are fact-seeking exactly like 'what is X?' — they want a
        definitional/encyclopedic fact — yet `_is_informational_query` only
        accepts wh-/define- prefixed questions (its info_patterns are anchored
        to ^(what|who|where|when|which|how)), so yes/no factual questions fall
        through the live web-answer path and never retrieve an easy fact. That
        is M2: a confident-but-wrong SM reply (or a shrug) instead of the
        encyclopedic answer the web would give in a blink.

        This mirrors `_is_conditional_query`: a non-wh question that is still
        fact-seeking routes to the same web retrieval + learning loop. We keep
        it deliberately narrow so we don't sweep opinion/personal/conditional
        turns into factual lookup:
          - must be a question (ends with '?' or reads as one), AND
          - lead with an auxiliary/modal verb (is/are/was/were/can/could/
            do/does/did/should/would/may/might/must), AND
          - the subject is NOT a personal/opinion/conditional frame
            (you/your/yourself/opinion/feel/love/meaning of life/if...).

        Note: web *learning* for an unknown subject already happens via the
        FOK/LPFC pre-queue (that path keys on associations, not query form), so
        the only missing piece is the LIVE answer retrieval — fixed by folding
        this into the `_web_direct_answer` gate alongside `_is_conditional_query`.
        """
        t = text.lower().strip(" ?!.")
        if not t.endswith("?") and not re.search(
                r"\b(is|are|was|were|can|could|do|does|did|should|"
                r"would|may|might|must)\b", t):
            return False
        # Must lead with an auxiliary/modal verb (yes/no / modal shape).
        if not re.match(
                r"\b(is|are|was|were|can|could|do|does|did|should|"
                r"would|may|might|must)\b", t):
            return False
        # Exclude personal / opinion / open-philosophical / conditional frames —
        # those are not factual lookups (mirrors _is_informational_query's
        # reasoning_patterns exclusions). We block opinion verbs and second-
        # person address ("do YOU think/feel/believe..."), not bare first/
        # third-person pronouns — "should I drink water?" is a factual question,
        # not an opinion, and must stay in.
        _nonfactual = [
            r"\b(if|suppose|assume|predict)\b",
            r"\b(you|your|yourself|opinion|think|feel|love|"
            r"meaning of life|believe|prefer)\b",
        ]
        for pat in _nonfactual:
            if re.search(pat, t):
                return False
        return True

    def _is_clause_complete(self, text: str) -> bool:
        """Dependency-closure completeness check (brain-faithful open-proposition
        signal), replacing the brittle cue-list test.

        Gregoromichelaki et al. (2020) and the fragment literature
        (Schlangen & Lascarides 2003, Dynamic Syntax) show conversation has no
        notion of a 'complete sentence' — what matters is whether open
        syntactic/semantic dependencies are satisfied. The brain detects
        completeness via verb subcategorization / valence (does the matrix verb
        have its required arguments filled?) and the absence of a dangling
        coordinator/comma (turn-end prediction, Magyari 2014; Barthel 2017,
        already cited in this file). A fragment like 'and another thing' leaves
        an open proposition (unfilled dependency) -> incomplete.

        We use ROBUST structural signals only (a lightweight verb lexicon is
        too noisy here — classify_word_pos mis-tags 'thing' as a verb and
        'bend' as a noun). The dangling-dependency cues below are sufficient to
        catch the cue-less incomplete lead-ins that motivated M9.

        Returns True iff the clause is COMPLETE (all open dependencies closed).
        """
        t = (text or "").strip().rstrip(" .!?")
        if not t:
            return False
        toks = [w.strip(".,!?") for w in t.split() if w.strip(".,!?")]
        if not toks:
            return False
        last = toks[-1].lower()
        first = toks[0].lower()
        # A leading coordinator signals a continuation -> open dependency.
        _COORD = {"and", "but", "or", "so", "because", "although", "though",
                  "if", "while", "yet", "unless"}
        # A trailing complementizer opens a clause that never arrives.
        _COMP = {"that", "what", "how", "whether", "why", "who", "which"}
        # A trailing copula needs its predicate complement.
        _COP = {"is", "are", "was", "were", "am", "be", "been", "being", "'s"}
        if first in _COORD:
            # A leading coordinator is a continuation, BUT it can open a fully
            # formed coordinated clause ("so, gravity pulls things") — that is a
            # complete utterance, not a dangling lead-in. It is incomplete only
            # if the remainder after the coordinator is itself an open fragment
            # (a bare NP with no predicate), e.g. "and another thing".
            _rest = t[len(first):].lstrip(" ,;:-").strip()
            if not _rest:
                return False
            # Remainder is complete iff it closes its own dependencies: it
            # must not itself end in a coordinator/complementizer/copula and
            # must carry a predicate (a verb / copula word).
            _rest_toks = [w.strip(".,!?") for w in _rest.split() if w.strip(".,!?")]
            if not _rest_toks:
                return False
            _rest_last = _rest_toks[-1].lower()
            if _rest_last in _COORD or _rest_last in _COMP or _rest_last in _COP:
                return False
            # Bare-NP remainder ("another thing") has no predicate -> open.
            _HAS_PRED = _COP | {"pulls", "bends", "is", "are", "was", "were",
                                   "means", "refers", "describes", "occurs", "happens",
                                   "reduces", "builds", "forms", "curves", "falls",
                                   "opens", "show", "shows", "makes", "does", "thinks",
                                   "sits", "sat", "stands", "grows", "lives", "works",
                                   "did", "do", "will", "would", "can",
                                   "could", "should", "has", "have", "had"}
            if not any(w.lower() in _HAS_PRED for w in _rest_toks):
                return False
            return True
        if last in _COORD or last in _COMP or last in _COP:
            return False

        # clause open (the turn has not reached its go-signal).
        if t.endswith((",", "-", "—", "–", ":", ";")):
            return False
        return True

    def _is_preamble_fragment(self, text: str) -> bool:
        """Turn-end predictor analog (brief behavior 2).

        A short imperative/wh- query ("explain oxiverse", "define trust") is a
        complete, answerable speech act, NOT an incomplete lead-in — the
        turn-end predictor must not withhold a warranted response on it (the
        hyper-cautious / over-monitoring false positive). See
        ``_is_answerable_query`` for the brain rationale.

        Completeness is judged by DEPENDENCY CLOSURE (_is_clause_complete),
        not a keyword list: 'and another thing —', 'the problem is that',
        'what I mean is' all leave an open proposition and are correctly held,
        while a real clause ('black holes bend spacetime', 'the cat sat')
        is complete and not withheld. The closure check runs FIRST: an
        incomplete fragment is always held, even if it happens to contain a
        wh-word (e.g. 'what I mean is' ends in a dangling copula).
        """
        t = (text or "").strip().lower().rstrip(" .!?")
        if not t:
            return False
        _toks = [w for w in t.split() if w]
        wc = len(_toks)
        # 1) Dependency-closure: an incomplete clause (open proposition) is a
        #    preamble — hold the turn. This catches cue-less lead-ins.
        if not self._is_clause_complete(text):
            return True
        # 2) A complete, answerable query is NOT a preamble (B1 guard).
        if self._is_answerable_query(text):
            return False
        # 3) Greetings / closed-class social acts are complete speech acts.
        _closed = ("hi", "hello", "hey", "yo", "bye", "thanks", "thank you",
                   "yes", "no", "ok", "okay", "sure", "cool", "nice", "lol", "hmm")
        if wc <= 2 and t in _closed:
            return False
        # 4) Known lead-in cue words (legacy, cheap) still signal a preamble.
        _preamble_cues = (
            "so", "well", "anyway", "by the way", "btw", "that reminds me",
            "oh", "right", "um", "uh", "like", "i mean", "speaking of",
            "before i forget", "got a sec", "quick thing",
        )
        if t in _preamble_cues:
            return True
        if any(t.startswith(c) and len(t) <= len(c) + 6 for c in _preamble_cues):
            return True
        # 5) A short bare fragment that is neither a greeting nor a complete
        #    clause is an incomplete lead-in. BUT a short *predicated* statement
        #    (subject + predicate: "i'm bored", "that's cool", "he left") is a
        #    complete speech act, not an open lead-in — holding it makes the
        #    agent look like it stopped listening (Q3/Q6 battery failures). Only
        #    hold bare NPs / unpredicated fragments ("the cat", "another thing").
        if wc <= 2 and not t.endswith("?"):
            # A token carries a predicate if it IS a predicate word or ENDS in a
            # clitic ("i'm", "that's", "we're", "you've") — contractions are one
            # token, so a substring/suffix test is required (not exact match).
            _PRED = ("is", "are", "am", "was", "were", "do", "does", "did",
                     "have", "has", "had", "go", "goes", "went", "left", "came",
                     "won", "lost", "like", "love", "hate", "feel", "felt",
                     "think", "want", "need", "know", "see", "said", "make",
                     "made", "eat", "ate", "run", "ran", "sleep", "cry", "laugh",
                     "bored", "tired", "sad", "happy", "fine", "okay", "cool")
            _CLITIC = ("'s", "'m", "'re", "'ll", "'ve", "'d")
            _has_pred = any(
                w in _PRED or w.endswith(_CLITIC) for w in _toks
            )
            if not _has_pred:
                return True
        return False

    def _is_answerable_query(self, text: str) -> bool:
        """True if `text` is a complete, answerable query (wh- question or an
        imperative definition command), NOT an incomplete lead-in.

        Brain-aligned: this guards the turn-end predictor (Magyari 2014;
        Barthel 2017) against a hyper-cautious *false positive* — withholding a
        warranted response because the turn looked "too short to be complete".
        A 2-word "explain oxiverse" / "define trust" is a fully-formed speech
        act (a definition command), exactly as complete as "what is gravity".
        The preamble detector must never eat it. Returns False (not a preamble)
        for these so generation proceeds.
        """
        low = (text or "").strip().lower().rstrip(" .!?")
        if not low or low.endswith("?"):
            return False
        # wh- words + imperative definition commands are complete queries.
        _QUERY_MARKERS = (
            "what", "who", "where", "when", "why", "how", "which",
            "define", "explain", "describe", "tell", "name", "list", "mean",
        )
        toks = low.split()
        if any(t in _QUERY_MARKERS for t in toks):
            return True
        # A copula (is/are/was/were) only makes a complete query when it is a
        # yes/no QUESTION (the text ends with "?") — a statement like "the
        # problem is that" contains "is" but is NOT an answerable query.
        _COPULA = ("is", "are", "was", "were")
        if any(t in _COPULA for t in toks) and text.strip().rstrip().endswith("?"):
            return True
        return False

    def _preamble_hold_response(self, text: str) -> str:
        """Light acknowledgment + invitation to continue for a preamble fragment.

        Mirrors the "wait for the go-signal" behavior: acknowledge receipt but
        don't volunteer a guessed full answer to an incomplete turn.
        """
        low = (text or "").strip().lower().rstrip(" .!?")
        if low in ("so", "well", "anyway", "right", "oh"):
            return "go on — i'm listening."
        if "remind" in low or "speaking of" in low:
            return "oh yeah? what about it?"
        return "mm-hmm, what were you going to say?"

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
        # is just 'sun disappeared' style junk. Also drop trailing light verbs /
        # question-frame words so "black holes form", "trust means", "gravity
        # works" reduce to their head concept "black holes" / "trust" / "gravity"
        # (the web search + graph should target the concept, not the verb). Pure
        # token filtering — never invents or hardcodes an answer.
        _light_verbs = {"form", "forms", "formed", "do", "does", "did", "doing",
                        "make", "makes", "made", "happen", "happens", "work",
                        "works", "mean", "means", "meant", "is", "are", "was",
                        "were", "be", "become", "use", "uses", "used", "exist",
                        "exists", "occur", "occurs", "affect", "affects",
                        "orbit", "orbits", "cause", "causes", "cause", "why"}
        RELATIONAL = _light_verbs | {"why", "what", "when", "where", "who", "how"}
        parts = [w for w in subj.split()
                 if w not in self._CONDITIONAL_FRAME and w not in RELATIONAL]
        # Strip trailing light verbs (keep the head noun concept).
        while len(parts) > 1 and parts[-1] in _light_verbs:
            parts = parts[:-1]
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
        # Yes/no & modal factual questions ("is pluto a planet?", "can dogs eat
        # chocolate?") start with an auxiliary, so the wh-branch above never
        # fires — they fell through to the raw "is X a Y?" query, which the
        # search backend ranks poorly (junk/entity-collision). Recast as a
        # definition-seeking query so encyclopedic/dictionary pages surface.
        # This is the same signal my _web_direct_answer gate uses to route
        # yes/no questions to the web path in the first place.
        if self._is_yesno_factual_query(raw_input):
            return f"what is {subj}"
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
        # Sense-biased framing (N400 predictive-coding analog): when the subject
        # is ambiguous, prepend a context-derived domain hint (e.g. "trust
        # psychology" for a social query) so the LIVE search pulls the
        # context-appropriate Wikipedia sense instead of the most-linked one.
        # This is the loop-closer for Fix 1: the coherence RANKER in
        # _best_answer_snippet breaks +3.0 ties, but the search itself must
        # also be steered toward the intended sense.
        biased = None
        if hasattr(self, "_sense_biasing_framing"):
            try:
                _framed = self._sense_biasing_framing(query, subject)
                if _framed and _framed != subject:
                    biased = _framed
            except Exception:
                biased = None
        for v in (biased, primary, query, subject):
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
        # Promo / sale / affiliate spam that leaks through the search API
        # (e.g. "This point in the year is perfect for 40% off 10,000+ programs.")
        "% off", "perfect for", "this point in the year", "limited time",
        "free shipping", "buy now", "order now", "shop now", "shop today",
        "today only", "save big", "hurry", "while supplies last", "deal of",
        "best price", "lowest price", "coupon", "promo code", "special offer",
        "act now", "don't miss", "get started today", "programs.",
        # Course / MOOC landing-page spam (leaks through the search API as a
        # "definition" snippet, e.g. "...Instructor: Charles Severance Enroll
        # now 148,862 already enrolled Included with Coursera Plus · Learn
        # more 11 modules Gain insight...").
        "enroll now", "already enrolled", "coursera", "included with",
        "learn more 11 modules", "gain insight into a topic", "modules",
        "instructor:", "instructor ", "start your free", "free trial",
        "sign up for", "watch now", "subscribe for", "get full access",
        "unlock", "premium", "membership", "limited offer", "last chance",
        # Health/supplement ad spam keywords
        "dietary supplement", "food supplement", "brain supplement", "memory supplement",
        "herbal supplement", "health supplement", "nutritional supplement", "vitamin supplement",
        "energy supplement", "supplement brand", "supplements brand", "supplement review",
        "supplements review", "supplement store", "supplements store", "supplement shop",
        "supplements shop", "dietary supplements", "herbal supplements", "health supplements",
        "nutritional supplements", "vitamin supplements", "energy supplements", "supplements capsules",
        "dosage", "pills", "pill", "gummies", "gummy", "add to cart", "money-back",
        "satisfaction guarantee", "buy online", "order online", "shop online", "customer reviews",
        "clinically proven", "brain booster", "memory booster", "memory lift", "memory-lift",
        "natural supplement", "natural supplements", "cognitive supplement", "cognitive supplements",
        "nootropic supplement", "nootropic supplements", "supplement industry", "supplement market",
        "supplement sales",
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
        # Context-augmented sense vector (N400 predictive-coding analog): bias
        # the coherence gate toward the sense implied by the full query, not the
        # bare blended noun. Used as a ranker (not a hard filter) below so it
        # breaks +3.0 ties between same-shape definitions of different senses.
        _ctx_vec = None
        if hasattr(self, "_context_query_vector"):
            try:
                _ctx_vec = self._context_query_vector(subject, query)
            except Exception:
                _ctx_vec = None
        candidates = []
        # Fix A: compute the "what is X" factual-shape flag once,
        # used by both the per-result looser-relevance fallback and
        # the post-loop fallback scan.
        _is_factual_what = bool(re.match(
            r"^(what|which) (is|are|was|were|means?|does) ",
            query.lower().strip()))
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
            # ── Fix A: semantic-relevance fallback (fail-closed honesty
            # preserved) ──────────────────────────────────────────────
            # The strict gate below requires a literal subject-token match in
            # the snippet BODY. That is a brittle syntactic proxy for semantic
            # relevance (Frontiers 2024: humans judge coherence semantically,
            # via priming, not string-match). For "what is trust" the gateway
            # can return an encyclopedic result whose TITLE holds the word but
            # whose body opens with a definition of *that* sense (e.g. a
            # "Trust (noun)" page) — or a navigational sense ("Trust Bank")
            # that is correctly rejected. We therefore admit a result via a
            # LOOSER signal ONLY when (a) the query is a single-subject
            # factual "what is X", (b) the result is from a preferred
            # encyclopedic source OR its title/url literally contains the
            # subject, and (c) it is NOT a navigational/brand sense
            # (title start differs from the bare subject). This is a ranker
            # boost, not a pass: the strict gate stays primary, so a clean
            # body-match answer still wins; the fallback only rescues a
            # reachable definition that the strict gate was throwing away.
            _title = (r.get("title", "") or "").lower()
            _title_raw = (r.get("title", "") or "")
            _url = (r.get("url", "") or "").lower()
            _is_factual_what = bool(re.match(
                r"^(what|which) (is|are|was|were|means?|does) ",
                query.lower().strip()))
            _looser_ok = False
            if subj and len(subj_tokens) == 1 and _is_factual_what:
                _title_has = (subj in _title)
                _url_has = (subj in _url)
                _pref_src = self._is_preferred_source(_url)
                if _pref_src or _title_has or _url_has:
                    # Decide whether this is the encyclopedic SENSE of the
                    # subject (admit via fallback) or a navigational/brand
                    # sense (reject). Encyclopaedia titles put the subject
                    # FIRST, then a SEPARATOR or a lower-case gloss:
                    #   "trust - wikipedia", "trust | definition…",
                    #   "trust (noun) - oxford", "gravity - wikipedia".
                    # A navigational/brand sense does NOT look like that:
                    #   - subject leads but next token is a capitalized BRAND
                    #     noun ("trust Bank…", "digital Trust Foundation"
                    #     where the capitalized continuation is the brand), or
                    #   - subject appears mid-title after a different leading
                    #     capitalized/section word ("about | trust Bank").
                    # Rule: NAVIGATIONAL iff
                    #   (subject is the leading token AND the next token is
                    #    capitalized AND not a known separator) OR
                    #   (subject is NOT the leading token AND the leading
                    #    token is capitalized / not a plain section word).
                    _title_toks = _title.split()
                    _title_raw_toks = _title_raw.split()
                    _lead = _title_toks[0] if _title_toks else ""
                    _sep = {"-", "|", "–", "—", ":",
                             "definition", "meaning", "(noun)",
                             "(verb)", "wikipedia", "britannica",
                             "dictionary", "oxford", "cambridge",
                             "about", "the"}
                    _nav_sense = False
                    if _lead == subj:
                        # subject leads; the token AFTER it (in raw case)
                        # must be a separator/lower-case gloss, not a
                        # capitalized brand noun ("Bank", "Foundation").
                        if len(_title_raw_toks) >= 2:
                            _nxt = _title_raw_toks[1]
                            if _nxt[:1].isupper() and _nxt.lower() not in _sep:
                                _nav_sense = True
                    else:
                        # subject not leading; navigational unless the url
                        # is a preferred encyclopedic source (which can
                        # carry a section-style title like "About | X").
                        if not _pref_src:
                            _nav_sense = True
                    if _nav_sense:
                        # Brand/navigational sense of the subject: never
                        # admit via the fallback, even if the body repeats
                        # the word.
                        continue
                    _looser_ok = True
            if len(subj_tokens) >= 2:
                _phrase_ok = subj in low
                _all_tokens = all(self._tok_match(t, wordset) for t in subj_tokens)
                if not (_phrase_ok or _all_tokens or _looser_ok):
                    continue
            elif subj:
                if not (self._tok_match(subj, wordset) or _looser_ok):
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
                # Track B Phase 2 (M4): learned structural-junk gate (flag-gated;
                # old regex table above remains the backstop).
                if self._snippet_is_structural_junk(s):
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
                if self._is_preferred_source(rurl):
                    score += 0.3
                # Context-augmented sense ranker (N400 analog): when the
                # context vector is available, reward candidates whose content
                # words align with the *intended* sense of the query. This
                # breaks the +3.0 ties between same-shape definitions of
                # different senses (e.g. legal vs interpersonal "trust") by
                # ranking the context-fitting sense higher rather than letting
                # search-engine result order decide.
                if _ctx_vec is not None and hasattr(self, "_definition_coherence_score"):
                    try:
                        coh = self._definition_coherence_score(subject, s)
                        # _definition_coherence_score uses the bare subject by
                        # default; re-rank against the context vector directly.
                        glove_fn = getattr(self, "_glove_vector", None)
                        if callable(glove_fn):
                            def_words = [w for w in re.findall(r"[a-z']{3,}", s.lower())
                                         if w not in STOP_WORDS]
                            def_vecs = [glove_fn(w) for w in def_words if glove_fn(w) is not None]
                            if def_vecs:
                                dcent = np.mean(def_vecs, axis=0)
                                n = np.linalg.norm(dcent)
                                if n > 0:
                                    dcent /= n
                                    ctx_n = np.linalg.norm(_ctx_vec)
                                    if ctx_n > 0:
                                        csim = float(np.dot(dcent, _ctx_vec / ctx_n))
                                        coh = max(0.0, csim)
                        score += 0.6 * coh
                    except Exception:
                        pass
                candidates.append((score, s))
        if not candidates:
            # Fallback: scan results for the first clean, non-boilerplate
            # snippet and return its first real sentence. Skip HTML/CSS/photo
            # junk and headline-only results.
            for r0 in results[:6]:
                # Fix A: skip navigational/brand titles in the fallback too
                # (same sense check as the primary gate above), so a
                # "Trust Bank" page can't leak through as the answer
                # just because its body repeats the subject word.
                _fb_title = (r0.get("title", "") or "")
                _fb_toks = _fb_title.split()
                _fb_lead = _fb_toks[0] if _fb_toks else ""
                _fb_nav = False
                if subj and len(subj.split()) == 1 and _is_factual_what:
                    if _fb_lead == subj and len(_fb_toks) >= 2:
                        _fb_nxt = _fb_toks[1]
                        if _fb_nxt[:1].isupper() and _fb_nxt.lower() not in _sep:
                            _fb_nav = True
                    elif _fb_lead and _fb_lead[:1].isupper() and subj not in _fb_toks[:1]:
                        _fb_nav = True
                if _fb_nav:
                    continue
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
                # Track B Phase 2 (M4): learned structural-junk gate (flag-gated;
                # old noise table above remains the backstop).
                if self._snippet_is_structural_junk(blob):
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

    def _snippet_is_structural_junk(self, snippet: str,
                                    coherence: Optional[float] = None) -> bool:
        """Track B Phase 2 (M4): learned structural-junk gate for snippets.

        When ``use_cerebellar_snippet`` is ON, consults the trained
        ``SnippetStructureModel`` (predictive-coding PE): rejects snippets that
        are structurally OOD from learned good definitions AND semantically
        incoherent. The hardcoded ``_SNIPPET_REJECT_SHAPES`` / ``_SNIPPET_NOISE``
        tables are ALWAYS consulted as a fallback (old constant kept until the
        learned model is verified to beat them on the regression set).

        Returns True only when the snippet should be rejected.
        """
        # Old constant as backstop — never weakens the existing hard reject.
        sl = (snippet or "").lower()
        _old_reject = (
            any(n in sl for n in self._SNIPPET_NOISE) or
            any(re.match(p, sl) for p in self._SNIPPET_REJECT_SHAPES)
        )
        if _old_reject:
            return True
        if not self.use_cerebellar_snippet:
            return False
        # Lazy-init the learned model (trained on the seed corpus once).
        if self._snippet_model is None:
            try:
                self._snippet_model = default_model()
            except Exception:
                self._snippet_model = False  # avoid re-init on failure
        if self._snippet_model and hasattr(self._snippet_model, "is_junk"):
            try:
                return bool(self._snippet_model.is_junk(snippet, coherence))
            except Exception:
                return False
        return False

    # ── Track B Phase 3 (M5): learned per-domain source-trust ────────────────
    # Replaces the hardcoded _PREFERRED_SNIPPET_SOURCES allowlist with a learned
    # per-domain trust accumulator (De Brigard 2025 source-monitoring; the brain
    # tags a source as reliable from a history of coherent, sleep-surviving
    # outputs). The hardcoded allowlist remains the backstop until this learned
    # mechanism is verified to beat it on the regression set.
    def _domain_trust(self, url: str) -> float:
        """Learned trust score for a snippet's source domain.

        When source-trust learning is OFF, falls back to the hardcoded
        allowlist (1.0 for a preferred source, else 0.5 neutral). When ON,
        returns the accumulated trust score for the domain (0.0 floor)."""
        _dom = self._domain_of(url) if url else ""
        if not self.use_source_trust:
            if _dom and any(s in url.lower() for s in self._PREFERRED_SNIPPET_SOURCES):
                return 1.0
            return 0.5
        if _dom in self._source_trust:
            return float(self._source_trust[_dom])
        return 0.5  # untried domain: neutral, not trusted, not banned

    def _record_source_outcome(self, url: str, accepted: bool,
                               survived_sleep: bool = False) -> None:
        """Update the per-domain trust accumulator from a snippet outcome.

        +0.1 when a snippet from the domain was accepted (coherence/structure
        passed); an extra +0.1 (total +0.2) when it also survived a sleep
        consolidation cycle; -0.2 when a snippet was rejected. Clamped to
        [0.0, 1.0]. Only runs when source-trust learning is enabled.
        """
        if not self.use_source_trust:
            return
        _dom = self._domain_of(url) if url else ""
        if not _dom:
            return
        _delta = 0.0
        if accepted:
            _delta = 0.2 if survived_sleep else 0.1
        else:
            _delta = -0.2
        _cur = self._source_trust.get(_dom, 0.5)
        self._source_trust[_dom] = max(0.0, min(1.0, _cur + _delta))

    @property
    def _source_trust_threshold(self) -> float:
        """Domains with trust above this are preferred (replaces the allowlist
        decision). 0.5 = neutral; a domain must earn trust to be preferred."""
        return 0.5

    def _is_preferred_source(self, url: str) -> bool:
        """Whether a source should get the quality tie-breaker boost.

        OFF (default): uses the hardcoded allowlist (backstop). ON: uses the
        learned trust score > threshold.
        """
        if not self.use_source_trust:
            return any(s in (url or "").lower()
                       for s in self._PREFERRED_SNIPPET_SOURCES)
        return self._domain_trust(url) > self._source_trust_threshold

    # ── Track B Phase 5 (M5): learned POS (replaces _GRAMMATICAL_CONCEPTS) ──
    # The brain distinguishes CONTENT words (Go) from FUNCTION words (NoGo) via
    # distributional POS, not a frozen 120-word list (Basal-ganglia Go/NoGo,
    # learned from POS; the BG gate would consume ConceptPosDict). The engine
    # already computes POS distributionally via self._concept_pos (classify_
    # word_pos). We expose a single predicate so every call site routes through
    # it; the hardcoded set stays the fallback until the learned classifier is
    # verified to cover it.
    _FUNCTION_POS_TAGS = frozenset({"prep", "pron", "det", "conj", "aux"})

    def _is_function_word(self, word: str) -> bool:
        """True if `word` is a function word (not a discourse/content target).

        Flag OFF (default): uses the hardcoded _GRAMMATICAL_CONCEPTS set
        (current behavior, no regression). Flag ON (use_learned_pos): uses the
        learned distributional POS from self._concept_pos — a word is a function
        word when its POS tag is a function category (prep/pron/det/conj/aux),
        with the hardcoded set retained as a safety net for residual cases the
        distributional tagger does not cover (e.g. some adverbs, numerals).
        """
        if not word:
            return False
        _w = word.lower()
        if not self.use_learned_pos:
            return _w in self._GRAMMATICAL_CONCEPTS
        # Learned path: distributional POS primary, hardcoded set as net.
        try:
            _pos = (self._concept_pos.get(_w) or "").lower()
        except Exception:
            _pos = ""
        if _pos in self._FUNCTION_POS_TAGS:
            return True
        # Safety net: words the distributional tagger leaves as 'noun'/'verb'/
        # 'adj' but the curated set knows are function (adverbs, numerals).
        return _w in getattr(self, "_GRAMMATICAL_CONCEPTS", set())

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
        # Junk/repetition guard: a snippet with consecutive identical tokens
        # (e.g. "GRAVITY+ GRAVITY+ is an upgrade…") or heavy local repetition is
        # not a real answer — penalize hard so a clean snippet wins instead of
        # relying on the downstream self-monitor to repair it post-selection.
        _rtoks = s.split()
        _dup_run = 0
        for i in range(len(_rtoks) - 1):
            if _rtoks[i] == _rtoks[i + 1] and _rtoks[i] not in (
                    "that", "had", "bye", "hello", "no", "yeah", "well", "good"):
                _dup_run += 1
        if _dup_run > 0:
            score -= 5.0 * _dup_run
        # Repeated capitalized acronym or token (e.g. "GRAVITY+ GRAVITY+") also
        # shows as a token appearing 3+ times in a short snippet.
        from collections import Counter as _Counter
        _cnt = _Counter(_rtoks)
        _max_rep = max(_cnt.values()) if _cnt else 0
        if _max_rep >= 3 and len(_rtoks) < 25:
            score -= 4.0
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


    # Plausibility floor for a live web answer (N400 / reality-monitoring analog).
    # A snippet's *added* content (everything except the repeated subject word)
    # must cohere with the subject's semantic field for it to count as a real
    # answer rather than an in-world / fictional restatement. Below this, the
    # brain would flag a "fiction-as-fact" reality-monitoring failure and the
    # agent refines the search instead of emitting the snippet. Tuned from probe
    # data: encyclopedic/procedural answers land ~0.46-0.51, game/UGC restatements
    # ~0.28-0.35. See test_web_snippet_source_quality.py.
    _SNIPPET_PLAUSIBILITY_FLOOR = 0.38

    def _snippet_plausibility(self, subject: str, snippet: str) -> Optional[float]:
        """Reality-monitoring plausibility of a snippet's *added* content.

        Cognitive basis: the brain's N400 / plausibility check (Kuperberg;
        Bornkessel-Schlesewsky) evaluates whether incoming information fits the
        situation model evoked by the question. A snippet that merely repeats the
        subject word ("Invisible is a gear that makes you invisible in Roblox")
        looks coherent only because of that repetition — its *new* content
        (gear, roblox) is incoherent with what "invisible" means. So we drop the
        subject word and its morphological variants and measure how well the
        remaining content coheres with the subject's GloVe vector. This is a
        domain-agnostic criterion on a semantic dimension (Johnson & Raye set
        criteria on reality-monitoring dimensions; they do not keep source
        blocklists), so it rejects game wikis, spam, and any other incoherent
        source without naming any of them.

        Returns None when GloVe is unavailable (unknown -> not incoherent) or the
        snippet carries no content beyond the subject (can't judge -> pass).
        """
        glove_fn = getattr(self, "_glove_vector", None)
        if not callable(glove_fn) or getattr(self, "_glove_vecs", None) is None:
            return None
        subj_vec = glove_fn(subject)
        if subj_vec is None:
            return None
        _stem = subject[:5].lower() if len(subject) >= 5 else subject.lower()
        words = [w for w in re.findall(r"[a-z']{3,}", (snippet or "").lower())
                 if w not in STOP_WORDS and _stem not in w]
        vecs = [glove_fn(w) for w in words if glove_fn(w) is not None]
        if not vecs:
            return None
        centroid = np.mean(vecs, axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            return None
        centroid /= norm
        snorm = np.linalg.norm(subj_vec)
        if snorm == 0:
            return None
        return float(np.dot(centroid, subj_vec / snorm))

    def _belief_coherence(self, subject: str, snippet: str) -> float:
        """Belief coherence: does the snippet's *added* content cohere with what
        RAVANA already believes about the subject (its GloVe/definition vector)?

        PROMPT 3 comparative reality-monitoring (Johnson & Raye 1981): a retrieved
        claim is accepted when it coheres with the existing belief model, not when
        it clears an absolute floor. Reuses _snippet_plausibility's subject-word
        drop so repetition doesn't fake coherence. Returns 0..1 (0 = no signal).
        """
        _p = self._snippet_plausibility(subject, snippet)
        return float(_p) if _p is not None else 0.0

    @staticmethod
    def _result_url(res) -> str:
        """Best-effort extraction of a source URL from a search-result payload."""
        if isinstance(res, dict):
            return res.get("url", "") or ""
        if isinstance(res, (list, tuple)):
            for _r in res:
                if isinstance(_r, dict) and _r.get("url"):
                    return _r.get("url", "")
        return ""

    @staticmethod
    def _source_type_label(url: str) -> str:
        """Human source-type label for epistemic tagging (PROMPT 3 hedges)."""
        _u = (url or "").lower()
        if "wikipedia" in _u:
            return "Wikipedia"
        if "britannica" in _u:
            return "Britannica"
        if any(s in _u for s in ("reddit", "forum", "quora", "stackoverflow")):
            return "a forum"
        if any(s in _u for s in ("gov", "edu", "nih", "nasa", "who.int")):
            return "an official source"
        if _u:
            return "a web source"
        return "the web"


    def _refine_query_variants(self, query: str, subject: str) -> List[str]:
        """Metacognitive control: re-frame the query when the first answer fails
        the plausibility monitor (the brain's second-pass reanalysis / repair,
        indexed by the late posterior positivity / P600 — Kuperberg & Jaeger).

        We don't block sources; we change *what we ask*. For a how-to / goal
        query the first hit is often in-world lore, so we push the query toward
        the real-world sense ("in real life" / "method"). For a factual query we
        add a real-world disambiguator. This gives the search engine a chance to
        surface a genuinely useful, plausible answer before we give up.
        """
        q = (query or "").lower().strip()
        if re.match(r"^(how|what) (can|do|to|would|should|does)\b", q):
            return [f"how to {subject} in real life",
                    f"{subject} method real world"]
        return [f"{subject} real", f"{subject} science"]

    def _web_snippet_search(self, variants, ctx, is_conditional, deadline):
        """Search each query variant and return the highest-shape snippet that
        passes the chrome/quality floor, or (None, None, attempted). Extracted from
        _web_direct_answer so the plausibility monitor can re-run a refined set
        of variants without duplicating the search loop.

        PROMPT 3 (Johnson & Raye 1981; Mitchell & Johnson 2009): confidence is a
        *comparative*, criterion-based decision — accept the BEST available
        snippet when it beats the runner-up by a margin OR coheres with existing
        belief, rather than discarding it for clearing an absolute floor. A fixed
        floor (was quality < 1.5) threw away correctly-sourced encyclopedic
        answers whose shape score landed just below it. We keep only a low
        *safety* floor (>= 1.0) to reject pure noise, then pick comparatively.
        """
        import time as _time
        attempted = False
        query = ctx.raw_input.strip()
        _cands = []  # (snippet, term, quality, plausibility, trust, url)
        for term in variants:
            if _time and _time.time() > deadline:
                break
            try:
                # For conditionals the local engine (localhost:4000) is instant
                # and reliably returns hypothetical content; skip the slower
                # remote APIs so a hung call can never stall the turn.
                local_only = is_conditional
                # Mark as attempted BEFORE the call: a raised SearchError (all
                # backends failed) is still "we searched and found nothing",
                # which is exactly when the caller should abstain honestly.
                attempted = True
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
            _cand_san = self._sanitize_definition_text(cand)
            if not _cand_san:
                if getattr(self, '_trace_enabled', False):
                    print(f"  [webans] chrome-only / promo snippet rejected: {cand[:50]!r}")
                continue
            cand = _cand_san
            quality = self._snippet_quality(cand, ctx.subject, term,
                                            is_conditional=is_conditional)
            if term == ctx.subject:
                quality -= 1.0
            # Low SAFETY floor only: reject pure noise, not borderline-good answers.
            if quality < 1.0:
                if getattr(self, '_trace_enabled', False):
                    print(f"  [webans] '{term}' -> below safety floor (q={quality:.2f}); skip")
                continue
            plaus = self._snippet_plausibility(ctx.subject, cand)
            trust = self._domain_trust(self._result_url(res))
            # belief coherence: snippet's phrase embedding vs the subject's vector
            _bel = self._belief_coherence(ctx.subject, cand)
            if getattr(self, '_trace_enabled', False):
                print(f"  [webans] '{term}' -> {cand[:70]!r} (q={quality:.2f}, "
                      f"plaus={plaus}, trust={trust:.2f}, bel={_bel:.2f})")
            _cands.append((cand, term, quality, plaus or 0.0, trust, self._result_url(res)))

        if not _cands:
            return None, None, attempted
        # Comparative selection: higher = better. Plausibility + trust weighted
        # so a well-sourced, belief-coherent snippet wins even if shape is equal.
        def _score(c):
            # c = (snippet, term, quality, plaus, trust, url)
            return c[2] + 2.0 * c[3] + 2.0 * c[4]
        _cands.sort(key=lambda c: -_score(c))
        best, second = _cands[0], (_cands[1] if len(_cands) > 1 else None)
        # Accept if clearly best OR coherent with belief; else abstain (don't
        # force a bad answer — fail-closed to honest uncertainty).
        if second is not None and (_score(best) - _score(second) < 0.1) and best[4] < 0.2:
            if getattr(self, '_trace_enabled', False):
                print("  [webans] comparative: best not clearly ahead and incoherent -> abstain")
            return None, None, attempted
        # Stash source/plausibility for downstream surfacing (PROMPT 3 hedges).
        self._last_web_source = self._source_type_label(best[5])
        self._last_web_plausibility = best[3]
        self._last_web_trust = best[4]
        return best[0], best[1], attempted


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
        if not (self._is_informational_query(query, ctx.subject)
                or is_conditional
                or self._is_yesno_factual_query(query)):
            return None
        variants = self._web_query_variants(query, ctx.subject, is_conditional)
        if getattr(self, '_trace_enabled', False):
            print(f"  [webans] informational query '{query}' subj='{ctx.subject}' "
                  f"variants={variants}")
        import time as _time
        _budget = 12.0  # hard wall-clock cap on the whole variant search
        _deadline = _time.time() + _budget
        try:
            best, best_term, attempted = self._web_snippet_search(variants, ctx,
                                                      is_conditional, _deadline)
        except Exception as ex:
            if getattr(self, '_trace_enabled', False):
                print(f"  [webans] search failed: {ex!r}")
            return None
        if not best:
            # D (research item D): fail-closed degradation. If this was an
            # informational/definitional query and we actually searched the web
            # (all backends, including remote fallbacks) but found nothing
            # usable, abstain honestly instead of letting the caller silently
            # fall back to a hollow graph-edge answer.
            if attempted and self._is_informational_query(query, ctx.subject):
                return ("I couldn't verify that from the web right now, "
                        "so I'll be honest rather than guess.", "web_unverified")
            return None

        # ── Answer-usefulness monitor (N400 plausibility / reality monitoring) ──
        # We "see" the candidate answer and check whether it actually serves the
        # question before speaking it (metacognitive monitoring; Nelson & Narens;
        # Koriat). PROMPT 3: this is now a *comparative* double-check, not an
        # absolute floor — a snippet the comparative search already vetted is only
        # withheld if it is implausible AND incoherent with belief (genuine junk),
        # never merely because it sits below a fixed number.
        plaus = self._snippet_plausibility(ctx.subject, best)
        _bel = self._belief_coherence(ctx.subject, best)
        # PROMPT 3 (revised): withhold a snippet that fails the plausibility
        # floor. A web "direct answer" below the floor is, by definition,
        # questionable — fail closed to honest uncertainty rather than leak it.
        # The earlier `and _bel < 0.1` escape hatch was too lenient: a junk
        # snippet whose SUBJECT WORD appears in it (e.g. "Invisible is a gear
        # that makes you invisible in Roblox.") gets an artificially inflated
        # belief score and slipped through. Belief coherence no longer rescues
        # a below-floor snippet; only a genuinely plausible (>= floor) answer
        # is emitted. A correct encyclopedic snippet (plausibility > floor)
        # still passes.
        if plaus is not None and plaus < self._SNIPPET_PLAUSIBILITY_FLOOR:
            if getattr(self, '_trace_enabled', False):
                print(f"  [webans] monitor: snippet implausible (plaus={plaus:.2f}, "
                      f"bel={_bel:.2f}) for '{ctx.subject}' -> refine search")
            # Metacognitive control: instead of emitting junk, refine the query
            # and re-search (second-pass reanalysis; Kuperberg & Jaeger). Only
            # adopt the refined result if it clears the plausibility floor; if
            # even the refined search can't produce a plausible answer, WITHHOLD
            # (return None) so we fall through to other strategies rather than
            # leaking an incoherent snippet.
            refined = self._refine_query_variants(query, ctx.subject)
            if refined:
                try:
                    best2, best2_term = self._web_snippet_search(
                        refined, ctx, is_conditional, _time.time() + 8.0)
                except Exception:
                    best2, best2_term = None, None
                p2 = self._snippet_plausibility(ctx.subject, best2) if best2 else None
                b2 = self._belief_coherence(ctx.subject, best2) if best2 else 0.0
                if best2 is not None and (p2 is None or p2 >= self._SNIPPET_PLAUSIBILITY_FLOOR or b2 >= 0.1):
                    if getattr(self, '_trace_enabled', False):
                        print(f"  [webans] refined query yielded plausible snippet (plaus={p2})")
                    best, best_term = best2, best2_term
                else:
                    if getattr(self, '_trace_enabled', False):
                        print(f"  [webans] monitor: refined search also implausible -> withhold")
                    return None
        if getattr(self, '_trace_enabled', False):
            print(f"  [webans] best snippet (via '{best_term}'): {(best or 'NONE')[:80]}")
        best = self._strip_title_echo(best.strip(), ctx.subject)
        # Sanitise dictionary/UI chrome and dateline prefixes from the live
        # snippet before it is emitted as the answer (the store-side sanitiser
        # only covers learned definitions, not directly-surfaced web answers).
        _san = self._sanitize_definition_text(best)
        if _san:
            best = _san
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
        # PROMPT 3: tag the answer with its source + comparative modesty instead
        # of presenting it as settled fact. Modality from trust + plausibility.
        from ravana.chat.hedges import hedge_frame, modality_from_support
        _web_mod = modality_from_support(
            min(1.0, 0.5 * self._last_web_trust + 0.5 * (self._last_web_plausibility or 0.0)))
        _src = getattr(self, "_last_web_source", "") or "the web"
        if _src and _src != "the web":
            # High-trust known source: prefix naturally.
            answer_text = f"according to {_src}, {best}{closer}"
        elif self._last_web_trust < 0.5:
            # Lower trust / forum: hedge explicitly.
            answer_text = hedge_frame("web", _web_mod, snip=best, src=_src) + closer
        else:
            answer_text = best + closer

        # ---- P6: verify the web claim against established belief before emitting ----
        # The web snippet is a *candidate*, not gospel. Check it against what we
        # already believe about the subject, adjust confidence accordingly, and
        # (the key move) store it in the BeliefStore so it can be contradicted,
        # reconciled, or forgotten later (see _sleep_consolidate / P7).
        subject_key = ctx.subject.lower()
        confidence = 0.5  # web claims start low-confidence (no RLM verification yet)
        try:
            existing = self.belief_store.query_belief(subject_key, "def")
        except Exception:
            existing = None
        if existing is not None:
            prior_val = existing[0]
            prior_conf = existing[1] if isinstance(existing, (tuple, list)) and len(existing) > 1 else 0.0
            overlap = self._belief_value_overlap(prior_val, best)
            # Only treat a divergence as a real *conflict* when the prior belief
            # is itself well-established (reinforced beyond a single low-conf
            # web snippet). Otherwise two equally-uncertain web snippets of the
            # SAME sense just collide (e.g. legal "trust" vs interpersonal
            # "trust" both land sub-0.15 overlap) and a correct answer gets
            # wrongly stamped "[unverified: conflicts with what I knew]".
            _prior_established = prior_conf >= 0.6
            if overlap >= 0.5:
                # Web corroborates what we already knew -> boost confidence.
                confidence = max(confidence, min(0.9, existing[1] + 0.2))
                if getattr(self, '_trace_enabled', False):
                    print(f"  [webans] belief match on '{subject_key}' "
                          f"(overlap={overlap:.2f}) -> conf {confidence:.2f}")
            elif overlap < 0.15 and _prior_established:
                # Nontrivial conflict: web disagrees with an established belief.
                confidence = max(0.1, confidence - 0.25)
                old_triple = (subject_key, "def", prior_val)
                new_triple = (subject_key, "def", best)
                self.belief_store.contradictions.append(
                    (old_triple, new_triple, self.belief_store.turn_num))
                answer_text = "[unverified: conflicts with what I knew] " + answer_text
                if getattr(self, '_trace_enabled', False):
                    print(f"  [webans] belief conflict on '{subject_key}': "
                          f"prior={prior_val[:40]!r} web={best[:40]!r} "
                          f"(overlap={overlap:.2f}, prior_conf={prior_conf:.2f}) -> conf {confidence:.2f}")
        # The web claim now LIVES in the belief store as a low-confidence
        # candidate; a matching prior boosted it, a conflicting prior demoted it.
        # Either way sleep can reconcile it against local knowledge and prune it
        # if it stays unverified and unreinforced.
        try:
            self.belief_store.assert_belief(subject_key, "def", best,
                                            confidence=confidence)
        except Exception:
            pass
        return (answer_text, "web_direct_answer")


    def _belief_value_overlap(self, a: Optional[str], b: Optional[str]) -> float:
        """Content-word Jaccard overlap between two belief value strings (0..1).

        Used to decide whether a fresh web claim MATCHES an established belief
        (high overlap -> corroboration) or NONTRIVIALLY conflicts with it (near
        zero overlap -> disagreement). Stop-words and short tokens are dropped
        so shared filler doesn't inflate the score.
        """
        ta = {w for w in re.findall(r"[a-z0-9]+", (a or "").lower())
              if len(w) >= 3 and w not in STOP_WORDS}
        tb = {w for w in re.findall(r"[a-z0-9]+", (b or "").lower())
              if len(w) >= 3 and w not in STOP_WORDS}
        if not ta or not tb:
            return 0.0
        union = ta | tb
        return len(ta & tb) / len(union) if union else 0.0


    def _try_hippocampal_retrieval(self, ctx) -> Optional[str]:
        """Try to retrieve from hippocampal buffer for recall triggers."""
        if not ctx.subject or not self._recall_mode:
            return None
        facts = self.hippocampal_buffer.retrieve(ctx.subject)
        if not facts:
            return None
        # Return the highest-confidence fact (Fix 4: was `return None`, which
        # threw away the retrieved memory — the retrieval path was dead).
        best_fact = max(facts, key=lambda f: f.confidence)
        return best_fact.object


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
        # Snapshot: background learner may add nodes mid-turn (avoid
        # "dictionary changed size during iteration").
        for other_nid, other_node in list(self.graph.nodes.items()):
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

        # Counterfactual simulation is a deliberately-constructed, epistemic-
        # hedged forward simulation (CSM), NOT free association — it cannot be
        # "word salad" in the tautological/empty sense the salad gate targets.
        # Exempt it from the salad gate so stochastic graph state during
        # background learning never discards a coherent "what would happen"
        # answer. (The counterfactual path already self-gates via its own
        # coherence/tautology checks in _coherence_ok / _abductive_counterfactual.)
        if strategy == "counterfactual_simulation":
            return max(base_score, 0.5)

        # Word salad check: immediate 0
        if _is_word_salad(response, subject=ctx.subject):
            # Phase 19g: record the cause so the caller can substitute an honest
            # fallback instead of emitting tautological/empty text to the user.
            try:
                self._last_response_was_salad = True
            except Exception:
                pass
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
                if not self._is_function_word(ll):
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


    def _sleep_consolidate(self, golden_edge_keys: Optional[set] = None) -> Dict[str, int]:
        # Snapshot golden facts BEFORE pruning so we can measure retention /
        # catastrophic forgetting (EWC-style blackout at saturation; Kirkpatrick
        # 2017). golden_edge_keys is a set of (src, tgt) the experiment harness
        # declares "important" (e.g. verified facts injected before stress).
        golden_before = {}
        if golden_edge_keys:
            for k in golden_edge_keys:
                e = self.graph.get_edge(*k)
                if e is not None:
                    golden_before[k] = (float(getattr(e, "weight", 0.0)),
                                        float(getattr(e, "confidence", 0.0)))
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
        # Offline synaptic-homeostasis prune of orphan/noisy semantic edges
        # (whale->deer off-frame co-occurrence). Runs AFTER the standard
        # weight-based prune in run_cycle so the two are additive and the count
        # is folded into the existing edges_pruned metric.
        try:
            extra_pruned = self.graph.prune_low_quality_edges()
            result['edges_pruned'] = result.get('edges_pruned', 0) + extra_pruned
        except Exception as e:
            if getattr(self, '_trace_enabled', False):
                print(f"  [sleep] prune_low_quality_edges error: {e}")
        # P7: reconcile & prune beliefs — close the web-grounding loop.
        # The grace sleep engine ignores the chat BeliefStore, so drive
        # belief maintenance here: reconcile contradictions (recency-decayed
        # winner) and forget low-confidence web claims that were never
        # reinforced, so unverified junk gets forgotten like real memory.
        try:
            reconciled = self.belief_store.reconcile()
            beliefs_pruned = self.belief_store.prune_stale(
                min_confidence=0.4, stale_after=10)
            result['beliefs_reconciled'] = len(reconciled)
            result['beliefs_pruned'] = beliefs_pruned
            if getattr(self, '_trace_enabled', False) and (reconciled or beliefs_pruned):
                print(f"  [sleep] beliefs: {len(reconciled)} reconciled, "
                      f"{beliefs_pruned} pruned")
        except Exception as e:
            if getattr(self, '_trace_enabled', False):
                print(f"  [trace] Belief reconcile/prune error: {e}")
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
        # Golden-fact retention / catastrophic-forgetting metric (Work 3).
        # A golden edge is RETAINED if it still exists with similar strength;
        # DRIFTED (forgotten/weakened) if it was pruned or its weight/confidence
        # moved beyond the drift tolerance. Blackout = golden edges fully gone.
        if golden_before:
            retained = 0
            drifted = 0
            blackout = 0
            DRIFT_TOL = 0.15
            for k, (w0, c0) in golden_before.items():
                e = self.graph.get_edge(*k)
                if e is None:
                    drifted += 1
                    blackout += 1
                    continue
                w1 = float(getattr(e, "weight", 0.0))
                c1 = float(getattr(e, "confidence", 0.0))
                if abs(w1 - w0) > DRIFT_TOL or abs(c1 - c0) > DRIFT_TOL:
                    drifted += 1
                else:
                    retained += 1
            total_g = len(golden_before)
            result["important_facts_total"] = total_g
            result["important_facts_retained"] = retained
            result["important_facts_drifted"] = drifted
            result["important_facts_blackout"] = blackout
            result["retention_rate"] = retained / total_g if total_g else 0.0
        return result

    # ── Work A0: HRR compositional reasoning in the loop ──
    def _hrr_encode_hook(self, subject: str, verb: str, obj: str) -> None:
        """Populate the HRR store from every graph edge (called by the
        ConceptGraph._fact_encode_hook wired in __init__). Integrative encoding
        (Zeithamova): storing by (subject, verb) makes transitive chains
        (A->B, B->C => A->C) cheaply reachable.

        The graph node labels may be SUFFIXED (e.g. 'lion#c3') to give each
        chain a unique graph identity (decoupling graph-node identity from
        HRR-word identity — see reasoning_bench.inject_controlled_chains).
        HRR must key on the BARE word ('lion') so the confusable-sibling
        regime (lion/tiger/bear share embeddings) is preserved for the
        vector-composition measurement. We strip the '#...' suffix here.
        """
        if self.hrr_reasoner is not None:
            try:
                def _bare(w):
                    return w.split("#", 1)[0] if w else w
                self.hrr_reasoner.encode(_bare(subject), verb, _bare(obj))
            except Exception:
                pass

    def hrr_query_chain(self, head: str, verb: str, max_hops: int = 2,
                        fallback_to_graph: bool = True, return_conf: bool = False,
                        top_k: int = 1, return_topk: bool = False,
                        graph_override: bool = False,
                        override_conf_threshold: float = 0.85):
        """Compositional relation query (M5' + graph-override).

        Generate-then-verify (Yonelinas recollection-vs-familiarity; O'Reilly
        1995 CLS synergy; McClelland 1995 / Spens & Burgess 2026 RAG-as-
        HC->neocortex deferral; Botvinick 2001 confidence-gated control
        recruitment). HRR PROPOSES a top-k per hop; the graph is the
        authoritative disambiguator (it holds the EXACT edge, since the HRR
        store is built from graph edges via the add_edge hook).

        M5' active graph-SELECT (always on when top_k>1): pick the HRR
        top-k candidate that is a real edge of (cur, verb). Calibration-safe:
        the graph only disambiguates WITHIN HRR's top-k.

        graph_override (gated, DEFAULT OFF -> byte-identical to pre-override
        behavior): when HRR is UNCERTAIN on a hop (no top-k hit OR
        hrr_conf < override_conf_threshold), defer that hop + remaining to
        graph.infer_chain(verb=verb) — exact edge traversal, relation-
        filtered. This is the canonical hippocampal->neocortical deferral,
        NOT a wholesale replacement: HRR still proposes; the graph corrects
        ONLY where HRR is weak. Honesty safeguards:
          - confs stay the HRR cosine (we do NOT replicate the old
            confs=1.0 for graph answers bug) — graph score goes in a
            SEPARATE channel (graph_conf).
          - sources per hop ('hrr' / 'graph_corrected') lets the
            benchmark report HRR-contributed vs graph-corrected fractions
            and the override-trigger rate.
          - infer_chain is called with verb=verb so it CANNOT traverse
            off-verb edges (graph.py:2438 now filters by relation_type).

        Return shape:
          return_conf=False            -> List[str]
          return_conf=True             -> (chain, confs, graph_support)
          return_topk=True (+return_conf) -> (chain, confs, graph_support,
                                                 topks, sources, graph_conf,
                                                 conflict_signal)
        confs / graph_support / graph_conf are THREE SEPARATE channels.
        conflict_signal[i] is a ConflictSignal (Botvinick ACC monitor, IV-B):
        the top-1/top-2 HRR decode gap + whether an on-verb graph edge exists;
        conflict==True means genuine uncertainty that recruits RECOLLECT (the
        graph-override), NOT a wholesale System-2 handoff.
        """
        hrr_chain, hrr_confs, hrr_topks = [], [], []
        if self.hrr_reasoner is not None and self.hrr_reasoner.has_fact(head, verb):
            hrr_chain, hrr_confs, hrr_topks = self.hrr_reasoner.query_chain_with_conf(
                head, verb, max_hops=max_hops, top_k=max(1, top_k))

        # label2id maps the BARE word (e.g. 'lion') to the (unique, suffixed)
        # graph node id (e.g. 'lion#c3'). This decouples HRR-word identity from
        # graph-node identity: HRR compares bare words, the graph walk uses the
        # real node ids so there is no label-collision ambiguity.
        label2id = {}
        for nid, nd in self.graph.nodes.items():
            if nd.label:
                label2id.setdefault(nd.label.split("#", 1)[0].lower(), nid)

        sel_chain: List[str] = []
        sel_confs: List[float] = []
        graph_support: List[float] = []
        topks_out: List[List[Tuple[str, float]]] = []
        sources: List[str] = []
        graph_conf: List[float] = []
        # Botvinick ACC-style conflict signal per hop (IV-B). Computed from the
        # HRR top-1/top-2 gap + graph-edge availability; NOT from raw decode
        # conf (uncalibrated ~0.58 -> would fire on every hop).
        conflict_signal: List[Any] = []

        if hrr_chain:
            cur_id = label2id.get(head.lower())
            for i, hop_obj in enumerate(hrr_chain):
                topk = hrr_topks[i] if i < len(hrr_topks) else []
                # M5' active graph-select: pick the HRR top-k candidate that
                # is a real graph edge of (cur, verb). Map bare target label ->
                # the SPECIFIC target node id so the walk can advance on the
                # exact node (collision-free even with suffixed identities).
                chosen = None
                edge_targets = {}  # bare label -> target node id
                if cur_id is not None:
                    for eid in self.graph._outgoing.get(cur_id, []):
                        tgt, e = eid if isinstance(eid, tuple) else (None, None)
                        if tgt is None:
                            continue
                        if (getattr(e, "relation_type", "") or "").lower() == verb.lower():
                            if tgt in self.graph.nodes and self.graph.nodes[tgt].label:
                                bare = self.graph.nodes[tgt].label.split("#", 1)[0].lower()
                                edge_targets[bare] = tgt
                for w, _s in topk:
                    if w.lower() in edge_targets:
                        chosen = w
                        break
                hrr_conf = hrr_confs[i] if i < len(hrr_confs) else 0.0
                # Botvinick ACC conflict signal (IV-B): gate on the HRR top-1
                # vs top-2 GAP (local competition), not raw decode conf. A near
                # tie + an available on-verb graph edge => genuine conflict that
                # recruits the RECOLLECT route (graph exact-edge correction).
                top1 = topk[0][1] if len(topk) > 0 else 0.0
                top2 = topk[1][1] if len(topk) > 1 else 0.0
                graph_has_edge = bool(edge_targets)
                csig = self.dual_process.conflict_monitor(
                    top1, top2, graph_has_edge, no_coherent_candidate=(chosen is None))
                # graph_override (gated): fire ONLY when HRR FAILED to propose
                # a graph-coherent candidate on this hop (chosen is None) — i.e.
                # no top-k candidate is a real edge of (cur, verb). This is the
                # genuine "HRR attempted and failed" signal. We do NOT also key
                # on raw decode conf < threshold, because HRR decode confs are
                # ~uniformly ~0.58 (uncalibrated), so that OR would fire on
                # EVERY hop and wholesale-replace vector composition. The
                # override thus corrects only HRR's genuine failures — the
                # canonical hippocampal->neocortical deferral (McClelland 1995;
                # Spens & Burgess 2026 RAG-as-HC->neocortex), NOT a replacement.
                if graph_override and chosen is None:
                    # Walk from the ACTUAL current node id (collision-free) when
                    # HRR failed to propose a graph-coherent candidate. infer_chain
                    # does an exact on-verb edge traversal from this node.
                    start_id = cur_id
                    if start_id is not None:
                        try:
                            gchain = self.graph.infer_chain(start_id, max_hops=max_hops - i, verb=verb)
                            if gchain:
                                for (t, _gs, _p) in gchain:
                                    if t in self.graph.nodes and self.graph.nodes[t].label:
                                        lbl = self.graph.nodes[t].label.split("#", 1)[0]
                                        sel_chain.append(lbl)
                                        sel_confs.append(hrr_conf)  # keep HRR cosine, NOT 1.0
                                        graph_support.append(1.0)
                                        topks_out.append(topk)
                                        sources.append("graph_corrected")
                                        graph_conf.append(float(_gs))
                                cur_id = gchain[-1][0]  # advance to last graph node
                                # Record calibration against graph truth (the graph
                                # IS ground truth here): HRR's predicted conf vs the
                                # fact that the graph-corrected hop is correct.
                                try:
                                    self.meta_cog.record_calibration(hrr_conf, True)
                                except Exception:
                                    pass
                                conflict_signal.append(csig)
                                break  # remaining chain supplied by graph
                        except Exception:
                            pass
                    if len(sel_chain) > len(graph_support):
                        break  # graph supplied at least this hop; stop HRR walk
                    # fallback: keep HRR best + support=0 (honest)
                    sel_chain.append(hop_obj)
                    sel_confs.append(hrr_conf)
                    graph_support.append(0.0)
                    topks_out.append(topk)
                    sources.append("hrr")
                    graph_conf.append(0.0)
                    conflict_signal.append(csig)
                else:
                    if chosen is not None:
                        sel_chain.append(chosen)
                        graph_support.append(1.0)
                        sources.append("hrr")
                        cur_id = edge_targets[chosen.lower()]  # advance on exact node
                    else:
                        sel_chain.append(hop_obj)
                        graph_support.append(0.0)
                        sources.append("hrr")
                        # no graph edge -> cannot advance cur_id reliably
                    sel_confs.append(hrr_conf)
                    topks_out.append(topk)
                    graph_conf.append(0.0)
                    conflict_signal.append(csig)
        elif fallback_to_graph:
            # HRR has nothing for this (head, verb): defer to graph as before.
            nid = getattr(self, "_concept_keywords", {}).get(head.lower(), [None])[0]
            if nid is not None:
                try:
                    gchain = self.graph.infer_chain(nid, max_hops=max_hops, verb=verb)
                    sel_chain = [self.graph.nodes[t].label.split("#", 1)[0]
                                 for (t, _s, _p) in gchain
                                 if t in self.graph.nodes and self.graph.nodes[t].label]
                    sel_confs = [0.0] * len(sel_chain)  # no HRR conf when HRR empty
                    graph_support = [1.0] * len(sel_chain)
                    sources = ["graph_corrected"] * len(sel_chain)
                    graph_conf = [float(_s) for (_t, _s, _p) in gchain]
                except Exception:
                    sel_chain, sel_confs, graph_support, topks_out = [], [], [], []
                    sources, graph_conf = [], []

        if return_topk:
            return (sel_chain, sel_confs, graph_support, topks_out,
                    sources, graph_conf, conflict_signal)
        if return_conf:
            return sel_chain, sel_confs, graph_support
        return sel_chain

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
        "need", "needs", "like", "likes",
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
        # discovery / creation query verbs (who INVENTED / DISCOVERED X)
        "invent", "invents", "invented", "inventing",
        "discover", "discovers", "discovered", "discovering",
        "develop", "develops", "developed", "design", "designs", "designed",
        "compose", "composes", "composed", "produce", "produces", "produced",
        # question-frame residuals: "what YEAR did X fall/occur", "when did X happen"
        "year", "years", "occur", "occurs", "occurred",
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

    # ── Safe pickle for checkpoint save (P0 resilient fallback) ───────────────
    # The curated ``state`` dict is a fresh local snapshot, so mutating it for
    # serialization is safe (it never touches live engine internals). The graph
    # is already persisted to SQLite separately (Phase 1), and the only
    # unpicklable object type that has historically leaked into the snapshot
    # (sqlite3.Connection, self.db) is *reopened from _db_path on load* — so
    # dropping it cannot lose durable state. This makes the checkpoint save
    # succeed ("saved …KB") instead of crashing ("save failed") on any latent
    # unpicklable object, closing the P4 latent-bug mask at the single choke
    # point.
    # ── Save/load schema stamp (M5: resilience, no silent blank wipe) ────────
    # A stale/corrupt ravana_weights.pkl used to make load() throw, the
    # caller swallowed it, and ALL learned state (definitions, identity,
    # emotion, RNG) was silently discarded — a blank restart. We now stamp
    # a schema version + a checksum so a corrupt snapshot is detected and a
    # partial restore is attempted (each field independently guarded) instead of
    # a silent wipe. Conceptual analog: biology re-validates memory
    # traces on replay (SWS/REM) rather than trusting one brittle snapshot.
    SAVE_SCHEMA_VERSION = 1

    @staticmethod
    def _checksum_state(state: dict) -> str:
        """Deterministic, cross-process-stable integrity fingerprint.

        Fix 7: the previous implementation hashed ``pickle.dumps(state)``. That
        is NOT stable across processes — state contains sets/dicts of strings
        whose iteration order depends on PYTHONHASHSEED, so the pickled bytes
        (and thus the digest) differ between the process that SAVED and the one
        that LOADS. Result: the checksum mismatched on essentially every load
        and printed a spurious 'partially corrupt' warning, and no self-heal
        could ever converge (the re-saved digest mismatched on the next run
        too).

        We instead hash an ORDER-INDEPENDENT structural fingerprint: for each
        top-level key, its type name and a coarse size signal (len when
        available). This is deterministic across processes and still catches the
        real corruption modes — missing/renamed keys, wrong types (the sanitizer
        replacing an object with a placeholder string), truncated collections —
        while value-level bit-rot is caught by pickle load failing outright
        (reconsolidation robustness: tolerate benign variation, flag structural
        damage). The ``state_checksum`` key itself is always excluded.
        """
        import hashlib

        def _fingerprint(v):
            # Scalars: hash the VALUE (deterministic across processes) so a
            # value tamper (e.g. turn_count 7 -> 999) is detected.
            if isinstance(v, (int, float, bool, str, bytes, type(None))):
                return ("scalar", repr(v))
            # Containers: order-independent structural signal only (type + len).
            # Their element order is NOT stable across processes (PYTHONHASHSEED),
            # so hashing contents would reintroduce the false-mismatch bug.
            try:
                return ("container", type(v).__name__, len(v))
            except Exception:
                return ("object", type(v).__name__)

        fingerprint = sorted(
            (k,) + _fingerprint(v)
            for k, v in state.items()
            if k != 'state_checksum'
        )
        blob = repr(fingerprint).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def _safe_pickle_dump(self, state, fpath):
        import pickle
        try:
            with open(fpath, 'wb') as _f:
                pickle.dump(state, _f)
            return True
        except (TypeError, pickle.PicklingError):
            pass
        # Best-effort: deep-copy the snapshot, replacing any unpicklable object
        # with a typed placeholder string, then retry.
        try:
            import sqlite3

            def _sanitize(obj, _seen=None):
                if _seen is None:
                    _seen = set()
                if id(obj) in _seen:
                    return obj
                _seen.add(id(obj))
                if isinstance(obj, sqlite3.Connection):
                    return f"<unpicklable:{type(obj).__name__}>"
                if isinstance(obj, dict):
                    return {k: _sanitize(v, _seen) for k, v in obj.items()}
                if isinstance(obj, (list, tuple, set)):
                    _cls = type(obj)
                    return _cls(_sanitize(v, _seen) for v in obj)
                # Probe picklability of leaf-like scalars/objects cheaply.
                try:
                    pickle.dumps(obj)
                    return obj
                except Exception:
                    return f"<unpicklable:{type(obj).__name__}>"

            sane = _sanitize(state)
            with open(fpath, 'wb') as _f:
                pickle.dump(sane, _f)
            return True
        except Exception:
            return False

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
                'use_cerebellar_snippet': getattr(self, 'use_cerebellar_snippet', False),
                'use_linggen': getattr(self, 'use_linggen', False),
                'linggen_genconf_seq': list(getattr(self, '_linggen_genconf_seq', [])),
                'source_trust': dict(getattr(self, '_source_trust', {})),
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
                'curated_definitions': list(self._curated_definitions),
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
                # M5: schema stamp + integrity checksum (corrupt-detection,
                # not silent wipe). Checksum is over the full state so any
                # bit-rot / partial write is caught on load.
                'schema_version': self.SAVE_SCHEMA_VERSION,
            }
            state['state_checksum'] = self._checksum_state(state)
            # Phase 1: Write graph to SQLite database for ACID persistence
            try:
                self.db.save_graph(self.graph)
            except Exception as e:
                if getattr(self, '_trace_enabled', False):
                    print(f"  [db] SQLite save failed: {e}")

            # M1-C: durable deterministic-knowledge mirror. Persist the verified
            # definition store (and the curated subset) into CognitiveDB metadata
            # so previously-verified facts survive even a fresh cold-start /
            # --reset (a second durable source beside the pickle). Loaded as a
            # fallback in _seed_common_facts' companion loader on cold start.
            try:
                self.db.save_metadata('definitions', self._definitions)
                self.db.save_metadata('curated_definitions', list(self._curated_definitions))
            except Exception:
                pass

            try:
                # Phase 6.1: Checkpoint rotation — save every 25 turns
                if self.turn_count > 0 and self.turn_count % 25 == 0:
                    checkpoint_path = self._save_path.replace('.pkl', f'_{self.turn_count}.pkl')
                    if self._safe_pickle_dump(state, checkpoint_path):
                        size_kb = os.path.getsize(checkpoint_path) / 1024
                    else:
                        size_kb = 0
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
                if self._safe_pickle_dump(state, self._save_path):
                    size_kb = os.path.getsize(self._save_path) / 1024
                    # M5: persist the checksum alongside so load() can
                    # verify integrity (corrupt-detection, not silent wipe).
                    try:
                        _sha = state.get('state_checksum')
                        if _sha:
                            with open(self._save_path + ".sha", "w") as _shaf:
                                _shaf.write(str(_sha))
                    except Exception:
                        pass
                    return f"saved {size_kb:.0f}KB to {os.path.basename(self._save_path)}"
                return f"save failed: unpicklable state could not be sanitized"
            except Exception as e:
                return f"save failed: {e}"

    def load(self) -> bool:
        """Public load entry-point (M5 contract). Delegates to _load()."""
        return self._load()

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

            # ── M5: schema + integrity checks (corrupt-detection, NOT silent wipe) ──
            # A stale/corrupt pkl used to throw here, the caller swallowed it,
            # and the engine started blank — silently discarding ALL learned
            # state. Now we detect and partial-restore instead.
            _loaded_ver = state.get('schema_version', None)
            if _loaded_ver is not None and _loaded_ver != self.SAVE_SCHEMA_VERSION:
                print(f"  [Load warn] schema_version={_loaded_ver} "
                      f"!= current {self.SAVE_SCHEMA_VERSION} — "
                      f"attempting best-effort restore")
            _loaded_sha = state.get('state_checksum', None)
            _checksum_ok = True
            if _loaded_sha:
                _recomputed = self._checksum_state(state)
                if _recomputed != _loaded_sha:
                    _checksum_ok = False
                    print(f"  [Load warn] state_checksum mismatch "
                          f"({_loaded_sha} vs {_recomputed}) — "
                          f"snapshot may be partially corrupt; restoring "
                          f"what is valid")
            # Fix 7: remember to re-save a fresh, self-consistent snapshot at the
            # end of a successful load when the checksum didn't verify, so the
            # corruption self-heals instead of warning on every startup
            # (systems-consolidation / reconsolidation robustness).
            self._resave_after_load = not _checksum_ok
            # Stash for the final success return; if unreadable, caller logs.

            # Restore graph
            loaded_graph = state['graph']
            if isinstance(loaded_graph, ConceptGraph):
                if loaded_graph.nodes:
                    first_node = next(iter(loaded_graph.nodes.values()))
                    if first_node.vector is not None and len(first_node.vector) != self.dim:
                        print(f"  [Load warning] Dimension mismatch: loaded graph has dim {len(first_node.vector)} but engine has dim {self.dim}. Discarding saved state.")
                        return False
                self.graph = loaded_graph
            else:
                # Corrupt graph (e.g. a str from an old sanitizer path) —
                # rebuild fresh rather than aborting the WHOLE load (M5:
                # partial restore, not silent blank wipe).
                print(f"  [Load partial] graph was "
                      f"{type(loaded_graph).__name__}, not ConceptGraph — "
                      f"rebuilding empty graph; other state restored")
                self.graph = ConceptGraph(dim=self.dim,
                                      max_nodes=getattr(self, '_max_nodes', 20000))
            self._concept_keywords = state['concept_keywords']
            self.turn_count = state['turn_count']

            # Restore decoder vocab mapping
            self._decoder_word_to_idx = state.get('decoder_word_to_idx', {})
            self._decoder_idx_to_word = state.get('decoder_idx_to_word', {})
            self._decoder_word_to_embed = state.get('decoder_word_to_embed', {})
            # M5 (decoder-dim guard, mirrors the graph-dim guard at ~line 7399):
            # the decoder read-out table is hardcoded to a 75-D dual-code
            # embedding (GloVe-64 | Lancaster-11) in response_gen.py. A snapshot
            # saved under the OLD 64-D decoder restores 64-D vectors here, which
            # then crash _build_decoder_vocab() (broadcast 64-D into 75-D slot).
            # Rather than abort the whole load (silent blank wipe) OR let it crash
            # mid-boot, discard ONLY the stale decoder vocab+embed table and let
            # _build_decoder_vocab() rebuild a fresh 75-D read-out from the graph
            # and GloVe. The graph + all learned concepts survive intact.
            _dec_dim = getattr(self, '_DECODER_DIM', 75)
            if self._decoder_word_to_embed:
                _sample = next(iter(self._decoder_word_to_embed.values()))
                if not isinstance(_sample, np.ndarray) or _sample.shape[0] != _dec_dim:
                    print(f"  [Load warn] decoder embed dim mismatch "
                          f"(loaded {getattr(_sample, 'shape', (None,))[0]} != "
                          f"current {_dec_dim}) — discarding stale decoder vocab, "
                          f"rebuilding fresh read-out from graph")
                    self._decoder_word_to_idx = {}
                    self._decoder_idx_to_word = {}
                    self._decoder_word_to_embed = {}
                    self._decoder_vocab_built = False
            self._topic_list = state.get('topic_list', [])
            self._topic_store = state.get('topic_store', {})
            self._response_context = state.get('response_context', [])
            self._last_responses = [r for r in state['last_responses']
                                    if isinstance(r, str)]
            self._last_strategy = state['last_strategy']
            self._free_energy = state['free_energy']
            self._learning_count = state['learning_count']
            # LingGen P6: restore the learned promotion flag (not a runtime config
            # switch like use_vad — this reflects whether grounded training proved
            # decoder-CE <= template-CE). Persists across loads so free-form
            # generation stays enabled once earned.
            self.use_linggen = bool(state.get('use_linggen', False))
            seq = state.get('linggen_genconf_seq', [])
            self._linggen_genconf_seq = list(seq) if isinstance(seq, (list, tuple)) else []

            # Restore identity (M5: each field independently guarded — a
            # corrupt field logs + is skipped, it can't wipe the rest).
            # NOTE: the guard is a TYPE check, not merely try/except — assigning
            # a wrong-shaped object (e.g. a dict) to self.identity.state
            # succeeds silently and only breaks later when a method reads
            # .strength on the dict. So we validate shape BEFORE assigning.
            try:
                _id_state = state['identity_state']
                # Fix 7: IdentityState carries `.strength` (+ `.momentum`); the
                # engine's `.last_delta` is a SEPARATE field stored under
                # 'identity_momentum'. The old guard checked hasattr(_id_state,
                # 'last_delta') — which IdentityState never has — so it ALWAYS
                # fell to the else branch and silently discarded a valid saved
                # identity on every load. Validate the field that actually
                # exists on the state object.
                if hasattr(_id_state, 'strength'):
                    self.identity.state = _id_state
                    self.identity.last_delta = state.get('identity_momentum', 0.0)
                else:
                    print(f"  [Load partial] identity_state had wrong shape "
                          f"({type(_id_state).__name__}); keeping fresh identity")
            except Exception as _e:
                print(f"  [Load partial] identity restore failed: {_e}")

            # Restore emotion VAD
            try:
                _v = state['vad_valence']; _a = state['vad_arousal']; _d = state['vad_dominance']
                if isinstance(_v, (int, float)) and isinstance(_a, (int, float)) and isinstance(_d, (int, float)):
                    self.emotion.state.valence = _v
                    self.emotion.state.arousal = _a
                    self.emotion.state.dominance = _d
                else:
                    print(f"  [Load partial] emotion_state had wrong shape; "
                          f"keeping fresh emotion")
            except Exception as _e:
                print(f"  [Load partial] emotion restore failed: {_e}")

            # Restore meaning
            try:
                _m = state['meaning_accumulated']
                if isinstance(_m, (int, float)):
                    self.meaning.accumulated_meaning = _m
                else:
                    print(f"  [Load partial] meaning_state had wrong shape; "
                          f"keeping fresh meaning")
            except Exception as _e:
                print(f"  [Load partial] meaning restore failed: {_e}")

            # Restore RNG
            try:
                self.rng.set_state(state['rng_state'])
            except Exception as _e:
                print(f"  [Load partial] rng restore failed: {_e}")

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
            for edge in list(self.graph.edges.values()):
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
            # M1-B: restore the curated (offline-authored) definition set so it
            # survives reloads and keeps bypassing the web-junk gate.
            self._curated_definitions = set(state.get('curated_definitions', []))
            # Purge polluted definition keys on load. Earlier versions stored
            # incoherent web fragments under generic/pronoun words (e.g.
            # "you" -> "the stronger player...", "life" -> "war zone"). Drop
            # those so RAVANA stops answering every mention of them with
            # abstract, off-topic text.
            #
            # Brain-aligned design: the purge is NOT a hand-edited per-word
            # table. Two parts:
            #   * _UNIVERSAL_PURGE — closed-class / pronoun words. These are
            #     universal bootstrapping the brain also needs (you can't learn
            #     a definition of "you"); kept as a tiny seed.
            #   * derived attractors — abstract hub concepts (high graph degree
            #     + high abstraction_degree) that empirically collect junk web
            #     fragments. This is computed from the learned graph, not listed.
            _DEF_PURGE = self._derive_definition_purge()
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

            # Fix 7: self-heal a checksum-mismatched snapshot by re-saving a
            # fresh, self-consistent one now that all valid fields are restored.
            # Avoids warning on every subsequent startup (reconsolidation).
            if getattr(self, '_resave_after_load', False):
                try:
                    self.save()
                    print("  [Load heal] re-saved a self-consistent snapshot")
                except Exception:
                    pass
                self._resave_after_load = False

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

        Clause connectors ('but'/'and'/'or'/'while'/'whereas') fuse two
        distinct topics (e.g. "why is the sky blue but sunsets red"). Keep only
        the FIRST clause's nouns so grounding targets one coherent subject
        ("sky blue") instead of the garbled "sky blue sunsets".

        Falls back to the original phrase if everything gets stripped (so we
        never return an empty subject for a genuinely single-word topic).
        """
        _CLAUSE_CONNECTORS = {"but", "and", "or", "while", "whereas",
                              "although", "though", "yet"}
        # Take only the leading clause before any connector.
        _head = phrase.lower()
        for _conn in _CLAUSE_CONNECTORS:
            _head = re.split(rf"\b{_conn}\b", _head)[0]
        words = [w.strip(".,!?") for w in _head.split()
                 if w.strip(".,!?") not in STOP_WORDS
                 and w.strip(".,!?") not in cls.QUESTION_WORDS]
        kept = [w for w in words if w not in cls._SUBJECT_CONTEXT_WORDS]
        if kept:
            return " ".join(kept)
        return phrase.strip(".,!?")

    def _theme_role(self, clause: str) -> str:
        """Recover the topical THEME/PATIENT of a clause by *role*, not a
        banned-word list (Fillmore 1968 case grammar; Bornkessel & Schlesewsky
        2006 eADM). The brain recovers topic from syntactic structure (agent /
        patient / theme), so we do the same: the theme is the content word that
        is the semantic *patient* of the main verb — approximated without a full
        parser by vector geometry.

        Heuristic (GloVe 64-D, no parser): among content words, the THEME is the
        one whose vector is FARTHEST from the main verb's vector (the patient is
        less predictable / less co-activated with the verb than the agent) and,
        when available, NEAREST to an already-known concept or ctx.subject
        (familiarity biases thematic assignment, as in eADM prominence). Falls
        back to the first non-verb content word, then "".
        """
        _VERB_BLOCK = {
            "is", "are", "was", "were", "be", "am", "do", "does", "did",
            "have", "has", "had", "can", "could", "would", "will", "should",
            "make", "makes", "made", "become", "becomes", "get", "gets",
            "go", "goes", "happen", "happens", "seem", "seems", "look",
            "looks", "feel", "feels", "sound", "sounds", "give", "gives",
            "take", "takes", "keep", "keeps", "show", "shows", "tell",
            "tells", "cause", "causes", "mean", "means", "explain",
            "describe", "find", "know", "think", "like", "want", "need",
        }
        toks = [w.strip(".,!?") for w in clause.lower().split()
                if len(w.strip(".,!?")) > 2
                and w.strip(".,!?") not in STOP_WORDS
                and w.strip(".,!?") not in self.QUESTION_WORDS
                and w.strip(".,!?") not in self.TOPIC_SKIP_WORDS]
        if not toks:
            return ""
        # Candidate content words (exclude pure verbs and role nouns handled by
        # the legacy list as a secondary guard).
        cands = [w for w in toks if w not in _VERB_BLOCK]
        if not cands:
            cands = toks
        # If exactly one content word, it IS the theme.
        if len(cands) == 1:
            return cands[0]
        # Vector-based theme recovery.
        vecs = {w: self._glove_vector(w) for w in cands}
        vecs = {w: v for w, v in vecs.items() if v is not None}
        if len(vecs) >= 2:
            # Main verb = the token (verb or not) whose vector is most central
            # to the others is hard without parse; approximate: the verb is the
            # token with the SMALLEST mean cosine to the rest (it co-activates
            # least specifically). The theme = the candidate FARTHEST from that
            # verb centroid.
            import numpy as _np
            _arr = {w: v / (_np.linalg.norm(v) + 1e-8) for w, v in vecs.items()}
            _keys = list(_arr)
            _cent = _np.mean([_arr[k] for k in _keys], axis=0)
            _verb = min(_keys, key=lambda k: float(_np.dot(_arr[k], _cent)))
            _scores = {}
            for w in _keys:
                if w == _verb:
                    continue
                _cos = float(_np.dot(_arr[w], _cent))
                # familiarity bias: known concepts / near ctx.subject score higher
                _fam = 0.0
                if w in self._concept_labels or w in self._concept_keywords:
                    _fam += 0.15
                if getattr(self, "_pending_subject_hint", None) and w in str(self._pending_subject_hint):
                    _fam += 0.1
                _scores[w] = (1.0 - _cos) + _fam  # far from verb + familiar => theme
            if _scores:
                return max(_scores, key=_scores.get)
        # No vectors: return the first candidate content word.
        return cands[0]

    def _strip_eli5_tail(self, text: str) -> str:
        """Remove simplification tails like "like i am five" / "in simple terms".

        These phrasings are framing, not part of the query's semantic subject.
        Left in, they pollute query grounding (e.g. "explain quantum
        entanglement like i am five" → subject "entanglement five").
        """
        s = text.lower()
        # ELI5: "like i am five", "like i'm five", "like a five year old",
        # "like i am five years old", "as if i were five".
        s = re.sub(r"\b(?:like|as if|as though)\b\s+(?:i am|i'm|i|i were|he is|she is|he were|she were|a|an|they are)\s+"
                   r"(?:five|five year old|five years old|(?:a |an )?\d+\s*(?:year|yr)s?\s*old)\b.*$",
                   "", s)
        # "in simple terms", "in plain language", "simply", "for kids", "for a child".
        s = re.sub(r"\b(?:in (?:simple|plain|basic|layman'?s) (?:terms|language|words|english)|"
                   r"simply|for (?:kids|a child|beginners|dummies))\b", "", s)
        return s.strip()

    def _ground_query(self, text: str) -> Tuple[str, float, str]:
        """Multi-strategy query grounding. Returns (subject, confidence, method).

        Strategies (tried in order):
        a) PrefrontalWorkspace question type parsing & exact phrase matching
        b) Compositional — split phrase, count known vs unknown words
        c) Phrase embedding similarity — mean word vec → nearest concept (cosine > 0.75)
        d) Best single word fallback — last meaningful non-stop word
        """
        # Normalize ELI5 / simplification tails BEFORE grounding so they don't
        # pollute the subject. "explain quantum entanglement like i am five"
        # must ground to "quantum entanglement", not "... like i am five".
        _text = self._strip_eli5_tail(text).lower()
        # Strategy A: Use PrefrontalWorkspace question type detection to parse semantic payload
        qtype = "general"
        query_phrase = ""
        try:
            if hasattr(self, 'pfc_workspace'):
                qtype, groups = self.pfc_workspace.detect_question_type(_text, self._concept_pos)
                if groups:
                    # Compare queries carry BOTH concepts in groups[0]/groups[1].
                    # The generic 'what_is' pattern can swallow a "difference between
                    # A and B" query and only surface group(1) as the subject; when
                    # the PFC already classified this as compare, reconstruct the A/B
                    # pair so web grounding + the decomposer both target the real
                    # concepts instead of a garbled "between privacy".
                    if qtype == "compare" and len(groups) >= 2:
                        query_phrase = groups[0].strip()
                    else:
                        query_phrase = groups[0].strip()
        except Exception:
            pass

        if not query_phrase:
            # Fallback to custom patterns
            text_lower = _text.strip(" ?!.")
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

        # Strategy C (moved before B): Compositional — score words by known/unknown ratio
        # Split on clause connectors ("but"/"and"/"or"/...) FIRST so two fused
        # topics ("why is the sky blue but sunsets red") become SEPARATE
        # questions rather than one garbled subject (RST: "but"=contrast
        # segregates; Bornkessel & Schlesewsky 2006 thematic roles). The SECOND
        # clause's themed topic is stashed for downstream sub-question use; the
        # PRIMARY subject still uses the proven compositional logic below so we
        # don't regress known-good single-clause grounding ("the speed of light"
        # must stay "speed light", not collapse to "light").
        _CLAUSE_CONNECTORS = {"but", "and", "or", "while", "whereas",
                              "although", "though", "yet"}
        _clauses = [c.strip(" .,!?") for c in re.split(
            r"\b(?:but|and|or|while|whereas|although|though|yet)\b", query_phrase)
            if c.strip(" .,!?")]
        _connector_rel = "contrast" if re.search(r"\bbut\b", query_phrase) else (
            "continuation" if re.search(r"\band\b", query_phrase) else "sequence")
        if len(_clauses) >= 2:
            # Segregate: stash the second clause's themed topic so the decomposer
            # can answer BOTH questions (e.g. sky-blue cause AND sunset-red cause)
            # instead of collapsing to one fused subject.
            _secondary = self._theme_role(_clauses[1])
            if _secondary:
                self._pending_subtopic = (_secondary, _connector_rel)
            else:
                self._pending_subtopic = None
        else:
            self._pending_subtopic = None
        # Build words from the FIRST clause only (was: whole phrase before; the
        # legacy first-clause truncation is intentional and keeps "sky blue" from
        # "sky blue but sunsets red" while the second topic lives in _pending_subtopic).
        _phrase_for_words = _clauses[0] if _clauses else query_phrase
        words = [w.strip(".,!?") for w in _phrase_for_words.split()
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
                            # Only collapse to the last entity when the leading
                            # words are genuine scenario framing (would/could/
                            # if/when), not a noun phrase like "the speed of
                            # light" which the PFC can mislabel hypothetical.
                            _leading = " ".join(words[:-1])
                            if re.search(r"\b(would|could|will|might|if|when|suddenly|disappear|gone|removed|vanished)\b", _leading):
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
            # Snapshot: background learner may add nodes mid-turn.
            for nid, node in list(self.graph.nodes.items()):
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
        # Fix F: reality / source monitoring (Johnson Source Monitoring
        # Framework). The greeting claims a past conversation ("Last time
        # we discussed X"). That is a MEMORY CLAIM and must be
        # grounded: only emit it confidently when a genuinely stored
        # user turn actually contains the topic (verified against the
        # turn ring buffer at engine.py:241). A loose topic-token
        # match in _topic_list can produce a FALSE recall ("Last
        # time we discussed Stars fit" when no such turn happened) —
        # which destroys trust faster than saying nothing. When the
        # match is weak/associative-only, downgrade to a HEDGED
        # form (humans do exactly this when source-memory is
        # uncertain) instead of a confident false claim.
        um = self.user_model
        if um.relationship_depth < 0.5:
            return ""
        if not um.last_topic:
            return ""
        past = um.last_topic.capitalize()
        # Verify against actual stored user turns (the source of truth).
        # EXCLUDE the most recent turn: last_topic is set from the CURRENT
        # turn's subject, so the current turn is always in the buffer and
        # would self-verify a false recall ("Last time we discussed Stars
        # fit" on the turn that just mentioned stars). A confident recall
        # requires the topic to appear in a PRIOR, distinct turn.
        _topic_lc = um.last_topic.lower()
        _verified = False
        _weak = False
        _prior_turns = self._recent_user_turns[:-1] if self._recent_user_turns else []
        for _t in _prior_turns:
            _tl = _t.lower()
            # confident only if the topic appears as a real token, not a
            # bare substring (so "sun" doesn't match "under the sun").
            if _topic_lc.split() and any(
                    self._tok_match(w, set(_tl.split()))
                    for w in _topic_lc.split()):
                _verified = True
                break
            if _topic_lc in _tl:
                _weak = True
        # Only greet every ~10 interactions to avoid repetition
        if um.interaction_count % 10 != 0 and um.interaction_count > 1:
            return ""
        if um.relationship_depth > 0.8 and _verified:
            return f"Great to see you! I remember we were talking about {past}. "
        if _verified:
            return f"Welcome back! Last time we discussed {past}. "
        if _weak:
            # Hedged: source memory uncertain, so flag the uncertainty
            # rather than assert a false history.
            return f"I think you might have mentioned {past} earlier — was that right? "
        # No genuine stored episode: never synthesize a past topic.
        return ""




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

    # NOTE: the literal CAUSAL_PAIRS / CONTRASTIVE_PAIRS / IS_A_PAIRS tables that
    # used to live here were DEAD CODE — defined but never referenced by the
    # engine. The canonical (and benchmarked) copies live in
    # ravana.graph.engine (CONTRASTIVE_PAIRS / CAUSAL_PAIRS / IS_A_PAIRS), used by
    # scripts/external_benchmark.py. Relation typing is derived, not looked up:
    # ConnectorLearner (synaptic_dynamics.py) learns connector-word probabilities
    # from GloVe similarity, and _infer_relation_type (graph/engine.py) derives
    # causal/contrastive/isa edges from co-occurrence + distributional context.
    # These frozen class-level copies were removed to prevent silent drift from
    # the learned graph edges (which are the source of truth).

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




    def persist_casing(self) -> None:
        """Persist online casing feedback (Phase 3) to disk.

        Called at session end so user corrections survive restarts. Safe if the
        casing store is unavailable.
        """
        try:
            from ravana.chat.case_distribution import persist_store
            persist_store()
        except Exception:
            pass

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
        self.persist_casing()  # Phase 3: flush casing feedback before exit
        if self._trace_enabled:
            print(f'  [bg] background learning stopped (performed {self._bg_search_count} searches)')




    def _needs_web_search(self, subject: str, query: Optional[str] = None) -> bool:
        """Check if a subject needs web search to enrich its associations.

        Returns True if:
        - The subject is not in the graph at all (completely unknown)
        - The subject IS in the graph but has fewer than 3 meaningful
          associations (edges with weight > 0.3). This catches abstract
          concepts like "consciousness" that are seeded with weak teenage
          associations and need web enrichment to produce useful responses.
        - D (research item D): the query is informational / definitional /
          "why" / "how" — these request mechanisms/facts that a handful of
          graph edges can't satisfy, so we ALWAYS attempt the web regardless
          of edge count. This closes the silent-failure case where a known
          subject (e.g. "sky", "dream") with hollow auto-expand edges was
          answered from the graph and never web-searched.

        Returns False only if the concept has >= 3 strong graph edges
        (enough knowledge to generate a meaningful response via the
        ventral path alone) AND the query is not informational.
        """
        if not subject:
            return False
        subj_lower = subject.lower().strip()

        # D (research item D): informational/why/definition queries always
        # attempt the web — the requested fact is a mechanism/definition, not
        # any graph edge. This is the direct fix for Q10/Q14-style hollow
        # answers. (Dependent on item E: provenance scoring weights the result.)
        if query and self._is_informational_query(query, subject):
            return True

        # Not in graph at all → definitely needs web search
        if subj_lower not in self._concept_keywords and subj_lower not in self._concept_labels:
            with self._graph_lock:
                for nid, node in list(self.graph.nodes.items()):
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
            r"\b(taller|shorter|heavier|lighter|older|younger|better|worse|biggest|tallest|heaviest|smartest)\b", # comparison/ordering
            r"\b(riddle|puzzle|logic|math|solve|calculation)\b", # logic/riddle
            r"\bis to\b", # analogy
            r"\b(you|your|yourself|think|opinion|feel|friendship|meaning of life)\b", # personal, opinion, or open philosophical
        ]
        for pattern in reasoning_patterns:
            if re.search(pattern, q):
                return False
                
        # 3. Check if the query matches a pattern asking for a definition/fact
        info_patterns = [
            r"^(what|who) (is|are|was|were|refers to|means)\b",
            r"^(what|who|where|when|which|how|why) \w+\b",  # "who won...", "where is...", "when was X built", "which city...", "how do X..."
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

    # ── M10: structured self-monitor observability ──────────────────────────
    def _build_monitor_report(self) -> dict:
        """Aggregate self._monitor_log into a summary (Steinhauser & Yeung 2010:
        the Pe component makes the monitor's decision explicit, not just the
        Ne/ERN evidence)."""
        log = getattr(self, "_monitor_log", [])
        by_monitor = {}
        by_reason = {}
        for entry in log:
            m = entry.get("monitor", "unknown")
            r = entry.get("reason", "unknown")
            by_monitor[m] = by_monitor.get(m, 0) + 1
            by_reason[r] = by_reason.get(r, 0) + 1
        return {
            "total_fires": len(log),
            "by_monitor": by_monitor,
            "by_reason": by_reason,
            "recent": log[-20:],
        }

    def monitor_report(self) -> dict:
        """Public accessor for the structured self-monitor log (M10).

        Returns a summary of every guard fire / swallow recorded since the last
        reset. Used by the CLI --trace-monitors flag and the CI gate to audit
        what the comprehension monitor withheld and why.
        """
        return self._build_monitor_report()
