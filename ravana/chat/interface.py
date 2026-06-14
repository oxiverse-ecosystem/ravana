"""
Chat Interface - RAVANA's main entry point for interactive chat.
Ties together: GraphEngine, DecoderEngine, WebLearner, CognitiveCore, RLM
"""
import sys
import os
import time
import json
import re
import pickle
import argparse
import threading
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import deque
import hashlib

# Import all refactored modules
from ravana.core import (VADEmotionEngine, VADConfig, IdentityEngine, IdentityState,
                          MeaningEngine, MeaningConfig, DualProcessController, DualProcessConfig, Route,
                          GlobalWorkspace, GWConfig, MetaCognition, MetaCognitiveConfig, EpistemicMode,
                          SleepConsolidation, SleepConfig, BeliefStore, UserBeliefProfile, BeliefConfig)
from ravana.graph import GraphEngine
from ravana.decoder import DecoderEngine, DecoderConfig
from ravana.web import WebLearner, SearchEngine, SearchConfig, SearchError
from ravana.bootstrap import BootstrapManager
from ravana.nn.rlm import RelationPredictor, PropagationEngine, Plasticity
from ravana_ml.nn.neural_decoder import NeuralDecoder

# Language modules
from ravana.language.basal_ganglia import BasalGangliaGate
from ravana.language.cerebellar_ngram import CerebellarNgram
from ravana.language.prefrontal_workspace import PrefrontalWorkspace, DiscourseIntent, DiscoursePlan, DiscourseType
from ravana.language.syntactic_cell_assembly import SyntacticCellAssembly, SyntacticFrame
from ravana.language.surface_realizer import SurfaceRealizer, DiscourseState

# Constants
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

QUESTION_WORDS = {"what", "why", "how", "when", "where", "who", "which",
                  "does", "do", "is", "are", "can", "will", "would",
                  "could", "should", "did", "have", "has", "had"}

FOLLOW_UP_WORDS = {"more", "else", "another", "also", "further",
                   "other", "additionally", "favorite"}

RECALL_TRIGGERS = [
    "remember when", "remember we", "earlier you", "you said",
    "you mentioned", "we talked about", "we discussed", "before you",
    "what did you say about", "what did we say about",
    "recall", "previously", "last time",
]

TOPIC_SKIP_WORDS = {"i", "you", "we", "they", "he", "she", "it", "me", "my",
                    "your", "our", "their", "him", "her", "its", "this", "that",
                    "these", "those", "there", "here", "some", "any", "all",
                    "each", "every", "both", "one", "more", "most", "few",
                    "very", "too", "just", "about", "also", "then", "than",
                    "now", "then", "well", "like", "such", "same", "still",
                    "even", "much", "really", "quite",
                    "think", "know", "feel", "want", "need", "go", "come",
                    "get", "say", "make", "take", "see", "hear", "tell",
                    "give", "let", "put", "keep", "look", "find", "ask",
                    "good", "bad", "big", "small", "always", "never", "maybe",
                    "if", "but", "in", "out", "up", "down",
                    "point", "way", "thing", "stuff", "and", "so"}


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
    learned_recently: bool = False
    recall_mode: bool = False


@dataclass
class ChatConfig:
    """Configuration for chat engine."""
    dim: int = 64
    seed: int = 42
    baby_mode: bool = True
    data_dir: Optional[str] = None
    user_suffix: str = ""
    use_vad: bool = True
    use_rlm: bool = True
    use_beliefs: bool = True
    trace_enabled: bool = False
    reasoning_mode: str = "stochastic"


class ChatInterface:
    """Main chat interface integrating all RAVANA components."""

    def __init__(self, config: Optional[ChatConfig] = None):
        self.config = config or ChatConfig()
        self._proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.rng = np.random.RandomState(self.config.seed)

        # Set up paths
        if self.config.data_dir:
            self._save_path = os.path.join(self.config.data_dir, f"ravana_weights{self.config.user_suffix}.pkl")
            self._glove_cache_path = os.path.join(self.config.data_dir, "ravana_glove_cache.npz")
        else:
            self._save_path = os.path.join(self._proj_root, f"ravana_weights{self.config.user_suffix}.pkl")
            self._glove_cache_path = os.path.join(self._proj_root, "ravana_glove_cache.npz")

        # Initialize components
        self._init_components()

        # Try loading saved state
        self._load_or_initialize()

    def _init_components(self):
        """Initialize all cognitive components."""
        dim = self.config.dim

        # Graph engine (handles GloVe internally)
        self.graph_engine = GraphEngine(dim=dim, seed=self.config.seed)

        # Decoder engine
        self.decoder_engine = DecoderEngine(DecoderConfig(embed_dim=dim))

        # Web learner
        self.web_learner = WebLearner(
            graph_engine=self.graph_engine,
            decoder_engine=self.decoder_engine,
            glove_vector_fn=self.graph_engine._glove_vector,
            data_dir=self.config.data_dir,
            trace_enabled=self.config.trace_enabled
        )

        # Bootstrap manager
        self.bootstrap_manager = BootstrapManager(self.graph_engine, self.web_learner)

        # Cognitive core
        self.emotion = VADEmotionEngine(VADConfig(eta_valence=0.3, eta_arousal=0.4, eta_dominance=0.25))
        self.identity = IdentityEngine(IdentityConfig(initial_strength=0.25, momentum_factor=0.3, recovery_bias=0.15))
        self.meaning = MeaningEngine(MeaningConfig(w_dissonance_reduction=0.3, w_identity_coherence=0.3, w_predictive_power=0.4, effort_kappa=0.5))
        self.dual_process = DualProcessController(DualProcessConfig(system2_confidence_threshold=0.25, system2_novelty_threshold=0.4, max_consecutive_system2=5))
        self.gw = GlobalWorkspace(GWConfig(capacity=7, broadcast_threshold=0.3, decay_rate=0.1))
        self.meta_cog = MetaCognition(MetaCognitiveConfig(probe_failure_threshold=0.4, confidence_calibration_window=15))
        self.sleep_engine = SleepConsolidation(SleepConfig(pressure_threshold=0.3, counterfactual_rate=0.15, emotional_flip_rate=0.08))
        self.belief_store = BeliefStore(BeliefConfig(recency_decay=0.1))

        # RLM components
        self.relation_predictor = RelationPredictor(vocab_size=50000, embed_dim=dim, concept_dim=dim)
        self.propagation = PropagationEngine(self.graph_engine.graph)
        self.plasticity = Plasticity(self.graph_engine.graph, base_lr=0.005)

        # Language modules
        self.basal_ganglia = BasalGangliaGate()
        self.cerebellar_ngram = CerebellarNgram()
        self.pfc_workspace = PrefrontalWorkspace(capacity=5)
        self.syntactic_assembly = SyntacticCellAssembly(learning_rate=0.05)
        self.surface_realizer = SurfaceRealizer()

        # State
        self.turn_count = 0
        self._topic_list: List[str] = []
        self._topic_store: Dict[str, Dict] = {}
        self._response_context: List[Dict] = []
        self._last_responses: List[str] = []
        self._last_strategy: str = ""
        self._free_energy = 0.0
        self._learning_count = 0
        self._learned_this_turn = False
        self._pending_learning_queue: List[str] = []
        self._network_available: Optional[bool] = None
        self._network_retry_turn: int = 0
        self._turns_since_last_search: int = 0
        self._concept_pos: Dict[str, str] = {}
        self._chain_traces: List = []
        self._impossible_queries: List = []
        self._last_strategy_used: str = ""
        self._last_activated_ids: List[int] = []
        self._trace_enabled = self.config.trace_enabled
        self._contradiction_map: Dict[str, Set[str]] = {}
        self._belief_assertions: List[Tuple[str, str, str]] = []
        self._recall_mode: bool = False
        self._prefrontal_buffer: List[str] = []
        self._pfc_gating_enabled = True
        self._pfc_buffer_capacity = 7
        self._mean_prediction_error = 0.0
        self._prediction_error_count = 0
        self._activation_fatigue: Dict[int, float] = {}
        self._recent_traversals: List[Tuple[int, int]] = []
        self._recent_traversal_map: Dict[Tuple[int, int], int] = {}
        self._visited_concepts: Set[str] = set()
        self._dopamine_tone: float = 0.5
        self._td_error_history: List[float] = []
        self._expected_strength: float = 0.25
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
        self.reasoning_mode: str = self.config.reasoning_mode
        self._consistency_paths: Dict[int, List[str]] = {}
        self._consistency_trace: List[str] = []
        self._concept_sources: Dict[str, Set[str]] = {}
        self._concept_visit_count: Dict[str, int] = {}
        self._concept_learning_progress: Dict[str, float] = {}
        self._concept_pe_delta: Dict[str, float] = {}
        self._curiosity_topics_queue: List[str] = []
        self._last_auto_learn_turn: int = 0
        self._curiosity_urgency: float = 0.0
        self._user_query_topics: List[str] = []
        self._user_last_topic: str = ""
        self._concept_confidence: Dict[str, float] = {}
        self._calibration_error: float = 0.0
        self._metacognitive_review_turn: int = 0
        self._explored_contradictions: Set[Tuple[str, str]] = set()
        self._cerebellar_ngram: Dict[str, Dict[str, float]] = {}
        self._cerebellar_depth: Dict[str, float] = {}

        # Sleep metrics
        self._sleep_pressure = 0.0
        self._last_sleep_episode = 0
        self._sleep_schedule_turns = 20
        self._sleep_schedule_time = 300
        self._sleep_metrics = {
            "edges_strengthened": 0, "edges_pruned": 0, "episodic_consolidated": 0,
            "impossible_queries_resolved": 0, "total_sleep_cycles": 0,
            "last_sleep_turn": 0, "last_sleep_metrics": {},
        }

    def _load_or_initialize(self):
        """Load saved weights or initialize from scratch."""
        if os.path.exists(self._save_path):
            loaded = self._load()
            if loaded:
                self.graph_engine._revector_existing_nodes = lambda: None  # Skip - already loaded
                self.bootstrap_manager.graph_engine.bootstrap_domain_concepts()
                self.decoder_engine.build_vocab(self.graph_engine, self.graph_engine._glove_vector)
                if hasattr(self.decoder_engine, '_saved_decoder_state') and self.decoder_engine._saved_decoder_state:
                    try:
                        self.decoder_engine.neural_decoder.load_state_dict(self.decoder_engine._saved_decoder_state)
                        self.decoder_engine._saved_decoder_state = {}
                    except Exception:
                        self.decoder_engine._saved_decoder_state = {}
                print(f"  [Loaded] Remembered {len(self.graph_engine.graph.nodes)} words from before!")
                return

        # Cold start
        self.bootstrap_manager.bootstrap_all()
        self.decoder_engine.build_vocab(self.graph_engine, self.graph_engine._glove_vector)
        print(f"  [Teen] Knows {len(self.graph_engine.graph.nodes)} words, ready to learn!")

    def process_turn(self, user_input: str) -> str:
        """Process input and generate a response, auto-learning when needed."""
        self.turn_count += 1
        self._learned_this_turn = False
        self._cascade_for_quality = False

        # Turn-scoped context isolation
        self._current_context_vector = None
        self._modulated_vectors.clear()
        if hasattr(self, '_prefrontal_buffer'):
            self._prefrontal_buffer = [self._prefrontal_buffer[-1]] if self._prefrontal_buffer else []

        self.web_learner.notify_user_active()
        if hasattr(self, 'surface_realizer'):
            self.surface_realizer.reset_turn()

        # Episodic decay between turns
        self.web_learner._decay_episodic_edges()

        # Decay activation fatigue
        for fk in list(self._activation_fatigue.keys()):
            self._activation_fatigue[fk] *= 0.95
            if self._activation_fatigue[fk] < 0.01:
                del self._activation_fatigue[fk]

        # Reset visited concepts every 50 turns
        if self.turn_count > 1 and self.turn_count % 50 == 0:
            self._visited_concepts.clear()
            for k in list(self._concept_visit_count.keys()):
                self._concept_visit_count[k] = max(0, self._concept_visit_count[k] - 2)

        # Activate from input
        activated = self._activate_from_input(user_input)

        # Auto-expand concepts
        new_concepts = self.graph_engine.auto_expand_concepts(user_input)
        if new_concepts > 0 and self._trace_enabled:
            print(f"  [trace]   auto-expanded {new_concepts} new concepts from input")
        if new_concepts > 0:
            activated = self._activate_from_input(user_input)

        self._last_activated_ids = list(activated)

        # Recall trigger detection
        recall_topic = self._detect_recall_trigger(user_input)
        self._recall_mode = recall_topic is not None
        if recall_topic:
            reactivated = self.graph_engine.recall_hippocampal(recall_topic)
            if reactivated:
                for nid in reactivated:
                    if nid not in activated:
                        activated.append(nid)

        # Follow-up boost
        self._activation_boost: Optional[Dict[str, float]] = None
        if self._is_follow_up(user_input) and self._topic_list:
            last_topic = self._topic_list[-1]
            lt_nids = self.graph_engine._concept_keywords.get(last_topic.lower(), [])
            for nid in lt_nids:
                if nid not in activated:
                    activated.append(nid)
                    self.graph_engine.graph.activate(nid, 0.6)
            self._activation_boost = self._get_user_model().activation_boost_for(last_topic)

        # Extract topic
        subject, obj = self._extract_topic(user_input, activated)
        _grounded_subj, _gconf, _gmethod = self._ground_query(user_input)
        self._last_grounding_conf = _gconf
        self._last_grounding_method = _gmethod
        if _gconf < 0.5 and _gmethod == "all_unknown" and _grounded_subj and self.config.baby_mode:
            if _grounded_subj not in self._pending_learning_queue:
                self._pending_learning_queue.append(_grounded_subj)
        relation = "is"

        # Primary IDs
        subject_ids = set()
        sl = subject.lower()
        if sl in self.graph_engine._concept_keywords:
            subject_ids.update(self.graph_engine._concept_keywords[sl])

        # Spread activation
        associations = self.graph_engine.spread_and_collect(activated, primary_ids=subject_ids)

        # Collect unknown words for deferred web learning
        input_words = [w.strip(".,!?") for w in user_input.lower().split() if len(w.strip(".,!?")) >= 3]
        unknown_meaningful = [w for w in input_words if w not in self.graph_engine._concept_keywords and w not in STOP_WORDS]
        if unknown_meaningful and self.config.baby_mode:
            for w in unknown_meaningful:
                if w not in self._pending_learning_queue:
                    self._pending_learning_queue.append(w)

        # Build context vector
        if subject:
            self._current_context_vector = self._build_context_vector(subject)

        # Emotional modulation
        self._update_emotion(user_input)
        for nid in activated:
            self._concept_vad[nid] = (self.emotion.state.valence, self.emotion.state.arousal, self.emotion.state.dominance)

        # Dual-process route
        confidence = self.identity.state.strength * 0.5 + 0.2
        route = self.dual_process.decide_route(confidence=confidence, novelty=0.1 if associations else 0.6, stakes=0.15)

        # Meta-cognitive assessment
        if self.turn_count > 3 and self.turn_count % 3 == 0:
            bias_report = self.meta_cog.detect_reasoning_bias(self.turn_count)
            epistemic_mode = self.meta_cog.recommend_epistemic_mode(self.turn_count)
            if self.turn_count % 5 == 0:
                self._metacognitive_review()
        else:
            epistemic_mode = self.meta_cog.current_mode

        # Sleep pressure + scheduled sleep
        self._sleep_pressure += 0.02 + 0.01 * (1.0 - confidence)
        pressure_triggered = (self._sleep_pressure > 0.3 and (self.turn_count - self._last_sleep_episode) > 8)
        schedule_triggered = (self.turn_count - self._last_sleep_episode) >= self._sleep_schedule_turns

        if pressure_triggered or schedule_triggered:
            metrics = self._sleep_consolidate()
            self._last_sleep_episode = self.turn_count
            self._sleep_pressure = 0.0
            if self._trace_enabled and metrics:
                print(f"  [sleep] Cycle #{self._sleep_metrics['total_sleep_cycles']}: "
                      f"{metrics.get('edges_strengthened', 0)} edges strengthened, "
                      f"{metrics.get('edges_pruned', 0)} edges pruned")

        # Past topics
        past = self._recall_past(subject, obj)

        # Brain state detection
        state = self._detect_brain_state()
        schema_ids = set()
        if state == 'heteromodal' or state == 'default':
            schema_ids = self._activate_schema(subject)

        # Build response context
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
            recall_mode=self._recall_mode,
        )

        # Generate response
        response, strategy = self._generate_response(ctx)
        self._last_strategy = strategy

        # Update cognitive state
        self._update_state(ctx)

        # Track topics
        if subject:
            sl = subject.lower()
            hop_labels = []
            for hops_list in self._last_chain_hops:
                for f, t in hops_list:
                    hop_labels.append((f, t))
            self.graph_engine.hippocampal_index_topic(subject, list(activated) if activated else [], hop_labels)
            if not any(t.lower() == sl for t in self._topic_list):
                self._topic_list.append(subject)
        if len(self._topic_list) > 50:
            removed = self._topic_list[:-50]
            self._topic_list = self._topic_list[-50:]
            for r in removed:
                self._topic_store.pop(r.lower(), None)

        # Store episodic memory
        self._store_episodic(subject, associations)

        if response is not None:
            self._last_responses.append(response)
            if len(self._last_responses) > 10:
                self._last_responses = self._last_responses[-10:]

        # Update cerebellar n-gram
        for hops_list in self._last_chain_hops:
            self._update_cerebellar_ngram(hops_list)

        # Store response context
        hop_labels = []
        for hops_list in self._last_chain_hops:
            for f, t in hops_list:
                hop_labels.append((f, t))
        self._response_context.append({'subject': subject, 'response': response, 'hops': hop_labels, 'turn': self.turn_count})
        if len(self._response_context) > 10:
            self._response_context = self._response_context[-10:]

        # Queue background learning
        if self._pending_learning_queue:
            with self.web_learner._bg_lock:
                for w in self._pending_learning_queue:
                    if w not in self.web_learner._bg_learning_queue:
                        self.web_learner._bg_learning_queue.append(w)
                self._pending_learning_queue.clear()

        # Track user query topics for curiosity priming
        if subject:
            sl = subject.lower()
            if sl != self._user_last_topic and len(sl) >= 3:
                self._user_query_topics.append(sl)
                if len(self._user_query_topics) > 10:
                    self._user_query_topics = self._user_query_topics[-10:]
                self._user_last_topic = sl

        # Track concept visits
        if subject:
            sl = subject.lower()
            self._concept_visit_count[sl] = self._concept_visit_count.get(sl, 0) + 1
        for label, _ in ctx.associated_concepts[:5]:
            ll = label.lower()
            self._concept_visit_count[ll] = self._concept_visit_count.get(ll, 0) + 1

        self.web_learner._update_concept_learning_progress()
        self.web_learner._compute_curiosity_urgency()

        self.web_learner.notify_user_idle()

        # Post-turn context decay
        if self._current_context_vector is not None:
            self._current_context_vector *= 0.3

        return response

    def _generate_response(self, ctx: CognitiveResponseContext) -> Tuple[str, str]:
        """Generate response using neural decoder with reasoning loop."""
        # Try neural decoder + syntactic pipeline
        _have_web = self.decoder_engine.web_training_count >= 500
        _total = self.decoder_engine.training_count
        if _have_web and _total >= 1000:
            try:
                decoder_response = self._generate_with_decoder_and_syntax(ctx)
                if decoder_response and len(decoder_response) > 10:
                    return (decoder_response, "neural_decoder")
            except Exception:
                pass

        # Reasoning loop
        try:
            return self._reasoning_loop(ctx)
        except Exception as e:
            if self._trace_enabled:
                print(f"  [trace] reasoning loop error: {e}")

        # Graph fallback
        return self._graph_fallback_response(ctx)

    # ... (other helper methods would be here - abbreviated for space)
    # The full implementation would include all the helper methods from the original ravana_chat.py

    def _get_user_model(self):
        """Get or create user model for the current session."""
        if not hasattr(self, 'user_model'):
            from scripts.ravana_chat import UserModel
            self.user_model = UserModel()
        return self.user_model

    def _activate_from_input(self, text: str) -> List[int]:
        """Find matching concepts using keywords and label matching."""
        words = re.findall(r"[a-zA-Z']{1,}", text.lower())
        scores: Dict[int, float] = {}
        for w in words:
            if w in self.graph_engine._concept_keywords:
                for nid in self.graph_engine._concept_keywords[w]:
                    node = self.graph_engine.graph.nodes.get(nid)
                    pos_boost = 0.5 if node and node.label and self._concept_pos.get(node.label.lower()) == 'noun' else 0.0
                    scores[nid] = scores.get(nid, 0) + 5.0 + pos_boost
        for nid, node in self.graph_engine.graph.nodes.items():
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
            self.graph_engine.graph.activate(nid, min(1.0, sc * 0.15))
            activated.append(nid)
        return activated

    def _is_follow_up(self, text: str) -> bool:
        words = set(w.lower().strip(".,!?") for w in text.split())
        return bool(words & FOLLOW_UP_WORDS)

    def _extract_topic(self, text: str, activated: List[int]) -> Tuple[str, str]:
        topic, confidence, method = self._ground_query(text)
        if topic and confidence >= 0.5:
            return (topic, text)
        if topic and method != "all_unknown" and method != "no_pattern" and method != "no_match":
            return (topic, text)
        if activated:
            best_real = None
            best_noun = None
            for nid in activated:
                node = self.graph_engine.graph.get_node(nid)
                if node and node.label:
                    lbl = node.label.lower()
                    if (len(lbl) > 2 and lbl not in QUESTION_WORDS and lbl not in TOPIC_SKIP_WORDS):
                        pos = self._concept_pos.get(lbl, 'noun')
                        if pos == 'noun' and best_noun is None:
                            best_noun = (node.label, text)
                        if best_real is None:
                            best_real = (node.label, text)
            if best_noun:
                return best_noun
            if best_real:
                return best_real
        words = [w.strip(".,!?") for w in text.lower().split()
                 if len(w.strip(".,!?")) > 2 and w.strip(".,!?") not in QUESTION_WORDS
                 and w.strip(".,!?") not in TOPIC_SKIP_WORDS and w.strip(".,!?") not in STOP_WORDS]
        if words:
            known_words = [w for w in reversed(words) if w in self.graph_engine._concept_labels or w in self.graph_engine._concept_keywords]
            noun_words = [w for w in known_words if self._concept_pos.get(w, 'noun') == 'noun']
            if noun_words:
                return (noun_words[0], text)
            if known_words:
                return (known_words[0], text)
            unknown_nouns = [w for w in words if self._concept_pos.get(w, 'noun') == 'noun']
            if unknown_nouns:
                return (unknown_nouns[0], text)
            return (words[-1], text)
        first = text.split()[0] if text.split() else ""
        first_stripped = first.strip(".,!?").lower()
        if first_stripped and len(first_stripped) > 2 and first_stripped not in QUESTION_WORDS and first_stripped not in TOPIC_SKIP_WORDS:
            return (first_stripped, text)
        return ("", text)

    def _ground_query(self, text: str) -> Tuple[str, float, str]:
        text_lower = text.lower().strip(" ?!.")
        QUERY_PATTERNS = [
            (r"(?:what|who)'?s?\s+(?:is\s+|are\s+)?(.+)", 1),
            (r"(?:tell|show)\s+me\s+about\s+(.+)", 1),
            (r"(?:explain|describe)\s+(.+)", 1),
            (r"(?:what|which)\s+(.+)\s+(?:is|are|mean)", 1),
            (r"(?:do you know|have you heard of)\s+(.+)", 1),
        ]
        query_phrase = ""
        for pattern, group_idx in QUERY_PATTERNS:
            m = re.match(pattern, text_lower)
            if m:
                query_phrase = m.group(group_idx).strip()
                break
        if not query_phrase:
            return ("", 0.0, "no_pattern")
        phrase_clean = query_phrase.strip(".,!?")
        if phrase_clean in self.graph_engine._concept_labels:
            return (phrase_clean, 0.95, "exact_label")
        if phrase_clean in self.graph_engine._concept_keywords:
            return (phrase_clean, 0.90, "exact_keyword")
        words = [w.strip(".,!?") for w in query_phrase.split()
                 if len(w.strip(".,!?")) > 2 and w.strip(".,!?") not in QUESTION_WORDS
                 and w.strip(".,!?") not in TOPIC_SKIP_WORDS and w.strip(".,!?") not in STOP_WORDS]
        if words:
            known_words = [w for w in words if w in self.graph_engine._concept_labels or w in self.graph_engine._concept_keywords]
            unknown_words = [w for w in words if w not in known_words]
            if known_words:
                ratio = len(known_words) / len(words)
                topic = known_words[-1]
                return (topic, min(0.85, 0.5 + ratio * 0.4), f"compositional_{ratio:.2f}")
            if words:
                return (words[-1], 0.2, "all_unknown")
        return ("", 0.0, "no_match")

    def _recall_past(self, subj: str, obj: str) -> List[str]:
        related = []
        for t in self._topic_list:
            pl = t.lower()
            sl = subj.lower()
            if pl != sl and (pl in sl or sl in pl or len(set(pl.split()) & set(sl.split())) > 0):
                related.append(t)
        return related[:3]

    def _store_episodic(self, subject: str, associations: List[Tuple[str, float]]):
        if not subject or not associations:
            return
        subj_nids = self.graph_engine._concept_keywords.get(subject.lower(), [])
        if not subj_nids:
            return
        subj_nid = subj_nids[0]
        for assoc_label, _ in associations[:3]:
            assoc_nids = self.graph_engine._concept_keywords.get(assoc_label.lower(), [])
            if not assoc_nids:
                continue
            assoc_nid = assoc_nids[0]
            existing = self.graph_engine.graph.get_edge(subj_nid, assoc_nid)
            if existing is None:
                self.graph_engine.graph.add_edge(subj_nid, assoc_nid, weight=0.15, relation_type="episodic")
            elif existing.relation_type == "episodic":
                sl = subject.lower()
                entry = self._topic_store.get(sl, {})
                visits = entry.get('visit_count', 1) if isinstance(entry, dict) else 1
                if visits >= 3:
                    existing.relation_type = "semantic"
                    existing.weight = min(0.40, existing.weight + 0.15)
                elif visits >= 2:
                    existing.weight = min(0.30, existing.weight + 0.10)

    def _find_bridge(self, assoc: List[Tuple[str, float]], subj: str) -> str:
        if not assoc:
            return subj if subj else ""
        best = assoc[0][0]
        if best.lower() == subj.lower() and len(assoc) > 1:
            return assoc[1][0]
        return best

    def _update_emotion(self, text: str):
        positive = {"good", "great", "happy", "love", "nice", "fun", "yay", "wow",
                    "cool", "amazing", "awesome", "wonderful", "beautiful", "excited",
                    "grateful", "proud", "hopeful", "joy", "interesting"}
        negative = {"bad", "sad", "scared", "angry", "hurt", "cry", "mean",
                    "terrible", "awful", "upset", "frustrated", "anxious",
                    "worried", "disappointed", "lonely", "guilty", "afraid"}
        curious = {"why", "how", "what", "wonder", "curious", "interesting",
                    "really", "tell me", "explain", "mean"}
        words = set(w.lower().strip(".,!?") for w in text.split())
        sv, sa = 0.0, 0.2
        if words & positive:
            sv += 0.4; sa += 0.2
        if words & negative:
            sv -= 0.4; sa += 0.25
        if words & curious:
            sa += 0.3
            if sv == 0.0:
                sv += 0.05
        if self._learned_this_turn:
            sa += 0.3; sv += 0.2
        input_words = [w for w in words if len(w) >= 3]
        known = sum(1 for w in input_words if w in self.graph_engine._concept_keywords)
        if input_words and known / len(input_words) < 0.5:
            sa += 0.15
        if self._prediction_error_count > 5:
            pe_surprise = min(0.4, self._mean_prediction_error * 2.0)
            sa += pe_surprise
        self.emotion.update(stimulus_valence=sv, stimulus_arousal=sa,
                           stimulus_dominance=self.identity.state.strength * 0.4 + 0.2,
                           uncertainty=self._free_energy * 0.5, dt=1.0)

    def _detect_recall_trigger(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        for trigger in RECALL_TRIGGERS:
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
        subj_nids = self.graph_engine._concept_keywords.get(subject.lower(), [])
        if not subj_nids:
            return set()
        subj_nid = subj_nids[0]
        subj_node = self.graph_engine.graph.get_node(subj_nid)
        if subj_node is None or subj_node.vector is None:
            return set()
        pe = getattr(self, '_mean_prediction_error', 0.3)
        threshold = 0.6 if pe < 0.2 else (0.4 if pe > 0.5 else 0.5)
        schema_ids = {subj_nid}
        for other_nid, other_node in self.graph_engine.graph.nodes.items():
            if other_nid == subj_nid or other_node.vector is None:
                continue
            cos = float(np.dot(subj_node.vector, other_node.vector))
            if cos > threshold:
                schema_ids.add(other_nid)
                self.graph_engine.graph.activate(other_nid, 0.6)
        return schema_ids

    def _build_context_vector(self, subject: str) -> np.ndarray:
        components = []
        weights = []
        subj_nids = self.graph_engine._concept_keywords.get(subject.lower(), [])
        if subj_nids:
            subj_node = self.graph_engine.graph.get_node(subj_nids[0])
            if subj_node and subj_node.vector is not None:
                components.append(subj_node.vector)
                weights.append(0.4)
        recent_vecs = []
        for resp in self._last_responses[-3:]:
            if resp is None:
                continue
            for w in resp.split():
                wn = self.graph_engine._concept_keywords.get(w.lower(), [])
                if wn:
                    wn_node = self.graph_engine.graph.get_node(wn[0])
                    if wn_node and wn_node.vector is not None:
                        recent_vecs.append(wn_node.vector)
        if recent_vecs:
            components.append(np.mean(recent_vecs, axis=0))
            weights.append(0.3)
        pfc_vecs = []
        for bl in self._prefrontal_buffer:
            bn = self.graph_engine._concept_keywords.get(bl, [])
            if bn:
                bn_node = self.graph_engine.graph.get_node(bn[0])
                if bn_node and bn_node.vector is not None:
                    pfc_vecs.append(bn_node.vector)
        if pfc_vecs:
            components.append(np.mean(pfc_vecs, axis=0))
            weights.append(0.2)
        e_vec = np.array([self.emotion.state.valence, self.emotion.state.arousal, self.emotion.state.dominance], dtype=np.float32)
        e_pad = np.zeros(self.config.dim, dtype=np.float32)
        e_pad[:3] = e_vec
        components.append(e_pad)
        weights.append(0.1)
        if not components:
            return np.zeros(self.config.dim, dtype=np.float32)
        ctx = np.average(np.array(components), axis=0, weights=np.array(weights))
        norm = np.linalg.norm(ctx)
        if norm > 0:
            ctx /= norm
        return ctx.astype(np.float32)

    def _get_temperature(self, sentence_idx: int, base_temps: Optional[List[float]] = None) -> float:
        if not self.config.use_vad:
            base = (base_temps or [0.08, 0.15, 0.25])[min(sentence_idx, 2)]
            return base
        base_temps = base_temps or [0.08, 0.15, 0.25]
        base = base_temps[min(sentence_idx, 2)]
        arousal = self.emotion.state.arousal
        arousal_factor = 0.7 + (arousal * 0.6)
        temp = base * arousal_factor
        state = getattr(self, '_cognitive_state', 'default')
        if state == "heteromodal":
            temp *= 1.3
        elif state == "unimodal":
            temp *= 0.7
        dt = getattr(self, '_dopamine_tone', 0.5)
        dt_factor = 0.7 + (dt - 0.5) * 1.0
        temp *= dt_factor
        return temp

    def _sleep_consolidate(self) -> Dict[str, int]:
        return self.sleep_engine.run_cycle(
            graph=self.graph_engine.graph,
            episodic_buffer=[],
            episodic_triples=self.plasticity._episodic_triples,
            belief_store=self.belief_store,
            topic_list=self._topic_list,
            user_model=self._get_user_model(),
            impossible_queries=self.web_learner._impossible_queries if hasattr(self.web_learner, '_impossible_queries') else [],
            contradiction_map=self.graph_engine._contradiction_map,
            drift_defense_threshold=0.7,
            drift_pull=0.05
        )

    def _reasoning_loop(self, ctx: CognitiveResponseContext) -> Tuple[str, str]:
        subject = ctx.subject
        query = ctx.raw_input
        assocs = ctx.associated_concepts

        subj_lower = subject.lower()
        subj_known = subj_lower in self.graph_engine._concept_keywords or subj_lower in self.graph_engine._concept_labels
        assoc_known = len(assocs) > 0

        is_complex = any(w in query.lower() for w in
                        ["how", "why", "create", "build", "design", "blueprint",
                         "explain", "detail", "comprehensive", "step by step",
                         "architecture", "implementation", "guide", "tutorial"])
        is_unknown = not subj_known or not assoc_known

        search_queries = []
        if is_complex or is_unknown:
            search_queries = self._decompose_for_search(query, subject, assocs)
        elif self.decoder_engine.training_count < 2000:
            search_queries = [subject]

        search_queries = search_queries[:4]

        all_learned_text = ""
        for sq in search_queries:
            try:
                result = self.web_learner.learn_from_web(sq)
            except Exception:
                continue

        if all_learned_text and self.decoder_engine.neural_decoder and self.decoder_engine._decoder_vocab_built:
            self.decoder_engine.train_on_text(all_learned_text, subject, self.graph_engine,
                                               self.graph_engine._glove_vector, passes=3)

        try:
            decoder_response = self._generate_with_decoder_and_syntax(ctx)
            if decoder_response and len(decoder_response) > 10:
                return (decoder_response, "neural_decoder_reasoned")
        except Exception:
            pass

        return self._graph_fallback_response(ctx)

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
        seen = set()
        unique = []
        for q in queries:
            ql = q.lower()
            if ql not in seen:
                seen.add(ql)
                unique.append(q)
        return unique

    def _graph_fallback_response(self, ctx: CognitiveResponseContext) -> Tuple[str, str]:
        subject = ctx.subject
        assocs = ctx.associated_concepts
        if not assocs:
            if subject:
                subj_lower = subject.lower()
                if subj_lower in self.graph_engine._concept_keywords or subj_lower in self.graph_engine._concept_labels:
                    neighbor = self.graph_engine.find_vector_neighbor(subject)
                    if neighbor and neighbor.lower() != subject.lower():
                        return (f"{subject} connects with {neighbor}.", "associative")
                    return (f"{subject.capitalize()} is something I know about.", "associative")
                return (f"I don't know about {subject} yet.", "unknown_subject")
            return ("...", "associative")
        temps = [self._get_temperature(0), self._get_temperature(1), self._get_temperature(2)]
        sentences = []
        seen = {subject.lower()}
        for i, temp in enumerate(temps):
            chain = self._walk_chain_simple(subject, seen, max_hops=2 if i > 0 else 1, temperature=temp)
            if chain:
                concepts = [p for p in chain.split() if p.lower() not in self._CONNECTOR_SET]
                if len(concepts) >= 2:
                    sentences.append(f"{subject.capitalize()} relates to {concepts[1]}.")
                elif concepts:
                    sentences.append(f"{concepts[0].capitalize()} is interesting.")
                seen.update(p.lower() for p in concepts)
        if not sentences:
            return (f"I don't know much about {subject} yet.", "associative")
        return (" ".join(sentences), "graph_fallback")

    def _walk_chain_simple(self, label: str, seen: Set[str], max_hops: int,
                            temperature: float = 0.25) -> Optional[str]:
        nids = self.graph_engine._concept_keywords.get(label.lower(), [])
        if not nids:
            return None
        chain = [label]
        chain_labels = {label.lower()}
        cur_nid = nids[0]
        cur_label = label
        hops = 0
        while hops < max_hops:
            candidates = []
            for tid, edge in self.graph_engine.graph.get_outgoing(cur_nid):
                tn = self.graph_engine.graph.nodes.get(tid)
                if tn is None or tn.label is None or tn.label.lower() in seen:
                    continue
                if tn.label.lower() in chain_labels:
                    continue
                if edge.weight < 0.35:
                    continue
                candidates.append((edge.weight * edge.confidence, tn.label, edge, "out"))
            for src, edge in self.graph_engine.graph.get_incoming(cur_nid):
                sn = self.graph_engine.graph.nodes.get(src)
                if sn is None or sn.label is None or sn.label.lower() in seen:
                    continue
                if sn.label.lower() in chain_labels:
                    continue
                if edge.weight < 0.35:
                    continue
                candidates.append((edge.weight * edge.confidence, sn.label, edge, "in"))
            if not candidates:
                break
            # Simple selection
            best_sig, best_label, best_edge, direction = max(candidates, key=lambda x: x[0])
            connector = self._EDGE_CONNECTORS.get(best_edge.relation_type, [("", [])])[0][1][0] if best_edge.relation_type in self._EDGE_CONNECTORS else ""
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
            cur_label = best_label
            cur_nid = self.graph_engine._concept_keywords.get(best_label.lower(), [None])[0]
            if cur_nid is None:
                break
            hops += 1
        if len(chain) <= 1:
            return None
        return " ".join(chain)

    def _generate_with_decoder_and_syntax(self, ctx: CognitiveResponseContext) -> Optional[str]:
        """Generate using decoder + syntactic pipeline - placeholder."""
        # This would integrate the full syntactic pipeline
        # For now, use basic decoder generation
        return self.decoder_engine.generate(ctx, self.graph_engine) if hasattr(self.decoder_engine, 'generate') else None

    def save(self) -> str:
        state = {
            'graph': self.graph_engine.graph,
            'concept_keywords': self.graph_engine._concept_keywords,
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
            'dim': self.config.dim,
            'rng_state': self.rng.get_state(),
            'sleep_pressure': self._sleep_pressure,
            'last_sleep_episode': self._last_sleep_episode,
            'sleep_cycles_completed': self.sleep_engine.metrics.get('total_sleep_cycles', 0),
            'concept_vad': self._concept_vad,
            'meta_mode': self.meta_cog.current_mode.value,
            'contradiction_map': self.graph_engine._contradiction_map,
            'user_model': self._get_user_model(),
            'use_vad': self.config.use_vad,
            'use_rlm': self.config.use_rlm,
            'use_beliefs': self.config.use_beliefs,
            'belief_store_state': self.belief_store.get_state(),
            'bg_learning_queue': list(self.web_learner._bg_learning_queue),
            'bg_search_count': self.web_learner._bg_search_count,
            'bg_multi_search_max': self.web_learner._bg_multi_search_max,
            'curiosity_drive_enabled': self.web_learner._curiosity_drive_enabled,
            'concept_visit_count': self._concept_visit_count,
            'concept_learning_progress': self._concept_learning_progress,
            'concept_pe_delta': self._concept_pe_delta,
            'curiosity_topics_queue': self._curiosity_topics_queue,
            'last_auto_learn_turn': self._last_auto_learn_turn,
            'curiosity_urgency': self.web_learner._curiosity_urgency,
            'user_query_topics': self._user_query_topics,
            'user_last_topic': self._user_last_topic,
            'concept_sources': {k: list(v) for k, v in self._concept_sources.items()},
            'explored_contradictions': [list(p) for p in self._explored_contradictions],
            'decoder_state_dict': self.decoder_engine.neural_decoder.state_dict() if self.decoder_engine.neural_decoder else None,
            'decoder_training_count': self.decoder_engine.training_count,
            'decoder_web_training_count': self.decoder_engine.web_training_count,
        }
        try:
            if self.turn_count > 0 and self.turn_count % 25 == 0:
                checkpoint_path = self._save_path.replace('.pkl', f'_{self.turn_count}.pkl')
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(state, f)
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
        try:
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
                            try:
                                return super().find_class('scripts.ravana_chat', name)
                            except (ModuleNotFoundError, AttributeError):
                                return super().find_class('ravana_chat', name)
                        raise
            with open(self._save_path, 'rb') as f:
                state = _RavanaUnpickler(f).load()

            self.graph_engine.graph = state['graph']
            self.graph_engine._concept_keywords = state['concept_keywords']
            self.turn_count = state['turn_count']
            self._topic_list = state.get('topic_list', [])
            self._topic_store = state.get('topic_store', {})
            self._response_context = state.get('response_context', [])
            self._last_responses = [r for r in state['last_responses'] if r is not None]
            self._last_strategy = state['last_strategy']
            self._free_energy = state['free_energy']
            self._learning_count = state['learning_count']
            self.identity.state = state['identity_state']
            self.identity.last_delta = state['identity_momentum']
            self.emotion.state.valence = state['vad_valence']
            self.emotion.state.arousal = state['vad_arousal']
            self.emotion.state.dominance = state['vad_dominance']
            self.meaning.accumulated_meaning = state['meaning_accumulated']
            self.rng.set_state(state['rng_state'])
            self._sleep_pressure = state.get('sleep_pressure', 0.0)
            self._last_sleep_episode = state.get('last_sleep_episode', 0)
            self.sleep_engine.metrics['total_sleep_cycles'] = state.get('sleep_cycles_completed', 0)
            self._concept_vad = state.get('concept_vad', {})
            meta_mode_str = state.get('meta_mode', 'exploratory')
            try:
                self.meta_cog.current_mode = EpistemicMode(meta_mode_str)
            except ValueError:
                self.meta_cog.current_mode = EpistemicMode.EXPLORATORY
            for edge in self.graph_engine.graph.edges.values():
                edge.parent_graph = self.graph_engine.graph
            self.graph_engine._contradiction_map = state.get('contradiction_map', {})
            user_model = state.get('user_model', None)
            if user_model:
                self._user_model = user_model
            bs_state = state.get('belief_store_state', None)
            if bs_state:
                self.belief_store.set_state(bs_state)
            self._prefrontal_buffer = state.get('prefrontal_buffer', [])
            self._mean_prediction_error = state.get('mean_prediction_error', 0.0)
            self._prediction_error_count = state.get('prediction_error_count', 0)
            self.web_learner._bg_learning_queue = state.get('bg_learning_queue', [])
            self.web_learner._bg_search_count = state.get('bg_search_count', 0)
            self.web_learner._bg_multi_search_max = state.get('bg_multi_search_max', 3)
            self.web_learner._curiosity_drive_enabled = state.get('curiosity_drive_enabled', True)
            self._concept_visit_count = state.get('concept_visit_count', {})
            self._concept_learning_progress = state.get('concept_learning_progress', {})
            self._concept_pe_delta = state.get('concept_pe_delta', {})
            self._curiosity_topics_queue = state.get('curiosity_topics_queue', [])
            self._last_auto_learn_turn = state.get('last_auto_learn_turn', 0)
            self.web_learner._curiosity_urgency = state.get('curiosity_urgency', 0.0)
            self._user_query_topics = state.get('user_query_topics', [])
            self._user_last_topic = state.get('user_last_topic', '')
            raw_sources = state.get('concept_sources', {})
            self._concept_sources = {k: set(v) for k, v in raw_sources.items()}
            raw_contra = state.get('explored_contradictions', [])
            self._explored_contradictions = {tuple(p) for p in raw_contra}
            decoder_sd = state.get('decoder_state_dict', None)
            self.decoder_engine._saved_decoder_state = decoder_sd if decoder_sd is not None else {}
            self.decoder_engine._decoder_training_count = state.get('decoder_training_count', 0)
            self.decoder_engine._decoder_web_training_count = state.get('decoder_web_training_count', 0)
            self.web_learner.start_background_learning()
            return True
        except Exception as e:
            print(f"  [Load error] {e}")
            return False

    def start_background_learning(self):
        self.web_learner.start_background_learning()

    def stop_background_learning(self):
        self.web_learner.stop_background_learning()


# CLI Entry Point
def main():
    parser = argparse.ArgumentParser(description="RAVANA - teenage mind on the web")
    parser.add_argument("--dim", type=int, default=64, help="Graph dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--reset", action="store_true", help="Delete saved weights and start fresh")
    parser.add_argument("--chat", type=str, default=None, help='Send queries in batch mode')
    parser.add_argument("--strategy", action="store_true", help="Include strategy name in --chat output")
    parser.add_argument("--trace", action="store_true", help="Print edge-level chain traces")
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD emotion modulation")
    parser.add_argument("--no-rlm", action="store_true", help="Disable RLMv2 triple verification")
    parser.add_argument("--no-beliefs", action="store_true", help="Disable belief store")
    parser.add_argument("--no-curiosity", action="store_true", help="Disable autonomous curiosity-driven learning")
    parser.add_argument("--mode", type=str, default="stochastic", choices=["stochastic", "deterministic", "exploratory"])
    parser.add_argument("--user", type=str, default=None, help="User name for multi-user isolation")
    parser.add_argument("--data-dir", type=str, default=None, help="Custom data directory")
    parser.add_argument("--export-graph", type=str, default=None, help="Export graph to JSON")
    parser.add_argument("--import-graph", type=str, default=None, help="Import graph from JSON")
    parser.add_argument("--stats", action="store_true", help="Print graph statistics")
    parser.add_argument("--concept", type=str, default=None, help="Show what RAVANA knows about a concept")
    args = parser.parse_args()

    reset_suffix = args.user or ""
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"ravana_weights{reset_suffix}.pkl")
    if args.reset:
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f"  [Reset] Deleted {os.path.basename(save_path)}, starting fresh!")
        else:
            print(f"  [Reset] No saved weights found, starting fresh!")

    config = ChatConfig(
        dim=args.dim, seed=args.seed, baby_mode=True,
        data_dir=args.data_dir, user_suffix=reset_suffix,
        trace_enabled=args.trace
    )
    engine = ChatInterface(config)

    if args.no_vad:
        engine.config.use_vad = False
        print("  [Config] VAD modulation disabled")
    if args.no_rlm:
        engine.config.use_rlm = False
        print("  [Config] RLMv2 triple verification disabled")
    if args.no_beliefs:
        engine.config.use_beliefs = False
        print("  [Config] Belief store disabled")
    if args.no_curiosity:
        engine.web_learner._curiosity_drive_enabled = False
        print('  [Curiosity] Autonomous learning disabled')

    if args.mode != "stochastic":
        engine.reasoning_mode = args.mode
        print(f"  [Mode] Reasoning mode set to '{args.mode}'")

    if args.export_graph:
        import json
        g = engine.graph_engine.graph
        data = {"nodes": [{"id": n.id, "label": n.label} for n in g.nodes.values()],
                "edges": [{"source": src, "target": tgt, "relation": e.relation_type, "weight": e.weight}
                         for (src, tgt), e in g.edges.items()]}
        with open(args.export_graph, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  [Export] Exported {len(data['nodes'])} nodes + {len(data['edges'])} edges to {args.export_graph}")
        engine.save()
        return

    if args.import_graph:
        import json
        with open(args.import_graph, "r") as f:
            data = json.load(f)
        g = engine.graph_engine.graph
        for node_data in data.get("nodes", []):
            nid = node_data.get("id")
            label = node_data.get("label", "")
            if label:
                vec = engine.graph_engine._glove_vector(label) if engine.graph_engine._glove_vecs is not None else None
                g.add_node(vector=vec, label=label)
        for edge_data in data.get("edges", []):
            src = edge_data.get("source")
            tgt = edge_data.get("target")
            rel = edge_data.get("relation", "related")
            w = edge_data.get("weight", 0.5)
            if src is not None and tgt is not None:
                g.add_edge(src, tgt, relation_type=rel, weight=w)
        engine.save()
        return

    if args.stats:
        print(f"  [Stats] Graph has {len(engine.graph_engine.graph.nodes)} nodes and {len(engine.graph_engine.graph.edges)} edges")
        print(f"  [Stats] Turn count: {engine.turn_count}")
        return

    if args.concept:
        nids = engine.graph_engine._concept_keywords.get(args.concept.lower(), [])
        if nids:
            node = engine.graph_engine.graph.get_node(nids[0])
            if node:
                outgoing = engine.graph_engine.graph.get_outgoing(nids[0])
                print(f"  [Concept] '{args.concept}': {len(outgoing)} edges, vector dim={len(node.vector) if node.vector is not None else 0}")
                for tgt_node, e in outgoing[:10]:
                    tgt_label = engine.graph_engine.graph.get_node(tgt_node).label if engine.graph_engine.graph.get_node(tgt_node) else "?"
                    print(f"    -> {tgt_label} [{e.relation_type}] w={e.weight:.3f}")
            else:
                print(f"  [Concept] '{args.concept}' found but no node data")
        else:
            print(f"  [Concept] '{args.concept}' not found in graph")
        return

    if args.chat is not None:
        queries = [q.strip() for q in args.chat.split("|") if q.strip()]
        results = []
        for i, q in enumerate(queries):
            t0 = time.time()
            try:
                resp = engine.process_turn(q)
            except Exception as e:
                resp = f"[error: {e}]"
            elapsed = time.time() - t0
            strategy = engine._last_strategy if args.strategy else ""
            results.append((q, resp, elapsed))
            print(f"Q{i+1}: {q}")
            print(f"A{i+1}: {resp}{' [' + strategy + ']' if strategy else ''}")
            if elapsed > 0.5:
                print(f"     [...{elapsed:.1f}s]")
            if args.trace:
                print(f"  [trace] Q{i+1}")
            print()
        print(f"  [{engine.save()}]")
        print(f"  [Stats] Turns: {engine.turn_count}, Words: {len(engine.graph_engine.graph.nodes)}, Sleeps: {engine.sleep_engine.metrics.get('total_sleep_cycles', 0)}")
        return

    # Interactive mode
    print()
    print("  ============================================")
    print("   RAVANA - teenage mind, learning from the web...")
    print("  ============================================")
    print()
    if engine.turn_count == 0:
        print("  Hey! I'm a teenage mind — I know some things but")
        print("  I'm always curious to learn more. I can think about")
        print("  causes, patterns, and different perspectives.")
        print("  Talk to me about anything!")
    else:
        print(f"  Welcome back! I now know {len(engine.graph_engine.graph.nodes)} words across {len(engine.graph_engine.graph.edges)} connections.")
        print(f"  I've slept {engine.sleep_engine.metrics.get('total_sleep_cycles', 0)} times to consolidate my learning.")
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
            if user_input.lower() in ("bye", "goodbye", "see you", "good night"):
                print(f"\n  RAVANA: Bye bye! I'll remember what you taught me!")
                break
            try:
                t0 = time.time()
                response = engine.process_turn(user_input)
                elapsed = time.time() - t0
                print(f"\n  RAVANA: {response}")
                if engine._learning_count > 0 and engine.turn_count % 5 == 0:
                    print(f"  [I've learned {engine._learning_count} times from the web and know {len(engine.graph_engine.graph.nodes)} words now!]")
                if elapsed > 0.5:
                    print(f"  [...took a moment to think...]")
            except Exception as e:
                print(f"\n  RAVANA: Hmm, I got confused. Let me try again!")
                if "--debug" in sys.argv:
                    import traceback
                    traceback.print_exc()
    finally:
        engine.stop_background_learning()
        print(f"  [{engine.save()}]")


if __name__ == "__main__":
    main()