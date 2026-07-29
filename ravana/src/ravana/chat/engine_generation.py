"""Auto-generated mixin module for CognitiveChatEngine.
Response generation mixin — neural decoder + syntax realization, context vectors, metaphors, sleep consolidation, HRR chains.
"""
from __future__ import annotations

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
                        _is_word_salad, _is_keyboard_mash,
                        _UNIVERSAL_PURGE, _DEFINITION_ASSERTION)
from .web_learning import WebLearningMixin
# Defect F: learned structural-PE snippet model (contrastive gap). Imported
# lazily-safe so a missing module degrades gracefully (the gate stays None and
# the old heuristic floor remains the backstop, never weakened).
try:
    from .snippet_quality import SnippetStructureModel, default_model
    _HAS_SNIPPET_MODEL = True
except Exception:  # pragma: no cover - defensive
    SnippetStructureModel = None  # type: ignore
    default_model = None  # type: ignore
    _HAS_SNIPPET_MODEL = False
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

# Stage 5a (de-hardcoding plan): snippet-PE gate parameters live in a fit file
# (data/snippet_pe.json) rather than inline constants. Fails open to seed
# constants when the fit file is absent.
try:
    from .snippet_pe_config import default_config as _default_pe_config
    _HAS_PE_CONFIG = True
except Exception:  # pragma: no cover - defensive
    _HAS_PE_CONFIG = False
    _default_pe_config = None

# Stage 5b-i (de-hardcoding plan): learned distributional POS classifier
# (data/pos_model.json), replacing the rule-based classify_word_pos +
# KNOWN_VERBS/ADJS/FUNCTION_WORDS lists when --learned-pos is enabled.
try:
    from .pos_model import PosModel, _seed_from_constants as _pos_seed_from_constants
    _HAS_POS_MODEL = True
except Exception:  # pragma: no cover - defensive
    _HAS_POS_MODEL = False
    PosModel = None
    _pos_seed_from_constants = None

# Stage 3 (de-hardcoding plan): Semantic Prototype Router (M-A) — replaces the
# ~15 hardcoded routing lists with one learned centroid router. Flag-gated
# (use_intent_router, OFF by default); regex path stays the default until the
# router is verified on the regression suite and promoted.
try:
    from .intent_router import IntentRouter
    _HAS_INTENT_ROUTER = True
except Exception:  # pragma: no cover - defensive
    _HAS_INTENT_ROUTER = False
    IntentRouter = None

# Stage 5b-ii (de-hardcoding plan): the duplicated closed-class functional
# lexicons (_generic / _FRAMING / _bare_moral / _INC/_DEC/_REM) collapse into one
# data-driven source of truth (data/functional_lexicon.json). Fails open.
try:
    from .functional_lexicon import default_lexicon as _default_lexicon
    _HAS_FUNC_LEX = True
except Exception:  # pragma: no cover - defensive
    _HAS_FUNC_LEX = False
    _default_lexicon = None

import pickle
from ravana.web.learner import SearchEngine
from ravana.core.dual_code_space import DualCodeSpace
from ravana.core.hrr_reasoner import HRRReasoner

# Universal closed-class / pronoun words that can never own a learned definition
# (you don't "define" the word "you"). This is the only hand-listed part of the
# definition purge — a minimal universal seed, not a per-word category table.
# The rest of the purge is derived from the learned graph (see
# _derive_definition_purge).

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




class GenerationMixin:
    """Response generation mixin — neural decoder + syntax realization, context vectors, metaphors, sleep consolidation, HRR chains."""

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
        _gate_key = "schema_cos_hi" if pe < 0.2 else ("schema_cos_lo" if pe > 0.5 else "schema_cos")
        schema_ids = {subj_nid}
        # Snapshot: background learner may add nodes mid-turn (avoid
        # "dictionary changed size during iteration").
        for other_nid, other_node in list(self.graph.nodes.items()):
            if other_nid == subj_nid or other_node.vector is None:
                continue
            cos = float(np.dot(subj_node.vector, other_node.vector))
            if self._adaptive_gate(_gate_key, cos, strict=True):
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
            # A verified web-sourced answer (snippet passed the safety floor +
            # trust/plausibility gate) is by construction a high-quality,
            # grounded reply — score it like a narrative answer, NOT the weak
            # 0.3 default that previously pushed it below the 0.55 emit
            # threshold and forced a fail-closed retreat to uncertainty.
            "web_direct_answer": 0.70,
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

    def _final_emit_guard(self, text: str, ctx, strategy: str = "") -> str:
        if strategy in ("counterfactual_simulation", "emotional_empathy",
                        "creative_generation", "seeded_relation"):
            return text
        # Research item B fix: a response surfaced verbatim from a verified web
        # source (web_direct_answer) is externally-grounded content, NOT RAVANA's
        # own free-association output. The salad/tautology monitors were built to
        # censor RAVANA's *self-generated* degenerate text (e.g. "trust is trust",
        # "X leads to Y"); they must not re-litigate a quoted encyclopedic fact.
        # Exempting it is brain-faithful: source monitoring distinguishes
        # internally-generated from retrieved memory (Johnson & Raye 1981), and a
        # vetted external quote is by construction not word-salad. Without this,
        # legitimate definitions ("X is the process by which...") were being
        # withheld as "fluent_tautology" — exactly the confident-garbage-in-reverse
        # failure mode (honest uncertainty shown INSTEAD of a correct answer).
        # Exempt externally-grounded / knowledge-backed strategies from the
        # salad + fluent-tautology censors. These monitors were built to catch
        # RAVANA's OWN degenerate free-association text, NOT legitimate answers:
        #   - web_direct_answer: vetted live web snippet
        #   - definition_with_assoc: a real stored/curated definition (e.g.
        #     "Gravity is the force by which a planet...") — the fluent_tautology
        #     detector false-positives on definitional prose ("X is the process
        #     by which...") and was wrongly withholding correct answers.
        #   - seeded_relation: authored project-knowledge relation (trusted ground)
        # Source monitoring (Johnson & Raye 1981): a retrieved/known fact is by
        # construction not word-salad, so the self-monitor must not veto it.
        if strategy in ("web_direct_answer", "definition_with_assoc", "seeded_relation"):
            return text
        # Research item B fix (cont.): a response that *quotes a verified web
        # source* — the "according to <src>," / "from the web" framing that
        # _web_direct_answer stamps onto grounded snippets — is externally
        # retrieved content, not RAVANA's own free-association output. The
        # salad/tautology monitors must not re-litigate it (they were built to
        # censor RAVANA's self-generated degenerate text). This catches the
        # decomposition path too: a "why" turn may surface a web snippet via
        # sub-question search and wrap it in a decomposed_* strategy, and that
        # grounded clause must survive the final-emit guard exactly like a
        # web_direct_answer does. Source monitoring (Johnson & Raye 1981)
        # distinguishes internally-generated from retrieved memory; a vetted
        # external quote is by construction not word-salad.
        _WEB_MARKERS = ("according to", "from the web", "i read that",
                        "according to a web source", "per the web")
        if text and text.strip().lower().startswith(_WEB_MARKERS):
            return text
        if not text or not text.strip():
            return text
        subj = (getattr(ctx, "subject", None) or "")
        _salad = False
        _fire = None
        # 1. learned classifier (graceful: None verdict if no fit file). When
        #    the learned gate is present AND returns a definite verdict it is
        #    AUTHORITATIVE — the legacy rule-based salad check below is skipped.
        #    The rule runs only as a fallback when the learned classifier is
        #    unavailable (not imported, is_salad_learned is None) or returned
        #    None (no fit); this keeps the backstop without weakening the
        #    learned gate.
        _learned_verdict = None
        if _HAS_SALAD_LEARNED and is_salad_learned is not None:
            try:
                _learned_verdict = is_salad_learned(text, subj)
            except Exception:
                _learned_verdict = None
            if _learned_verdict is not None:
                _salad = bool(_learned_verdict)
                _fire = "learned_salad"
        # 2. legacy rule-based — fallback ONLY when the learned gate is absent
        #    or produced no verdict.
        if is_salad_learned is None or _learned_verdict is None:
            try:
                if _is_word_salad(text, subject=subj):
                    _salad = True
                    _fire = _fire or "rule_salad"
            except Exception:
                pass
        # 3. fluent-tautology signature (independent learned detector)
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

    def _metaphor_lead(self, subj_cap: str, phrase: str, sense: str,
                       val: float, prop: str) -> str:
        """Build the magnitude-conditioned cross-modal metaphor reply. Shared by
        the human-Lancaster Path 1 and the legacy probe fallback so phrasing is
        identical; vivid when activation is high, tentative when low.

        B3 (semantic control; Lambon Ralph 2016): ANCHOR on the property the
        user actually asked about, THEN offer the cross-modal substitute. The
        old form led with the substituted dimension ("i'd picture Tuesday in
        terms of its shape") when the user asked about COLOR — which reads as
        answering a different question. The brain's semantic-control network
        keeps the assigned query (the asked property) in focus, so we name the
        mismatch on {prop} first, then bridge to the sensory dimension we DO
        have a profile for.
        """
        # Bridge phrasing to the sensory dimension we actually have a read on.
        if val >= 2.0:
            bridge = f"if anything, i'd picture it more by its {phrase}"
        elif val >= 1.0:
            bridge = f"if anything, i'd think of it by its {phrase}"
        else:
            bridge = f"maybe i'd loosely relate it to its {phrase}"
        return (f"{subj_cap} doesn't really have a {prop} — that's not the "
                f"kind of thing it is. {bridge} — something you'd {sense}. "
                f"what were you getting at?")

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
    def _prop_binder_exclude(self, prop: str) -> set:
        """Exclusion set of binder dims for a property (Plan: learned, Item 6).

        Starts from the hand-authored ``_PROP_TO_BINDER`` map (the stable seed)
        and extends it with the DOMINANT Lancaster sensorimotor dimension of the
        property, projected from its GloVe vector via the trained
        LancasterEncoder (data/lancaster_encoder.npz). For any property word the
        probe's top 11-D dim IS the binder — so the exclusion generalizes beyond
        the ~11 hand-authored properties. Falls back to the hand map alone when
        the probe is unavailable (day-one behavior preserved).
        """
        _excl = set(self._PROP_TO_BINDER.get((prop or "").lower(), ()))
        try:
            from ravana.ontology.attribute_encoder import LANCASTER_TO_BINDER
            _lv = self._lancaster_vector(prop) if hasattr(self, "_lancaster_vector") else None
            if _lv is not None and len(_lv) == len(self._LANCASTER_ORDER):
                import numpy as np
                _i = int(np.argmax(_lv))
                _ldim = self._LANCASTER_ORDER[_i]
                for _bdim in LANCASTER_TO_BINDER.get(_ldim, []):
                    _excl.add(_bdim)
        except Exception:
            pass
        return _excl


        # G3 (Lancaster): prefer the HUMAN Lancaster 11-D norms for the
        # cross-modal dimension — they discriminate strongly (hand Hand_arm=4.4
        # vs trust=0.45) where the merged 65-D probe is variance-compressed.
        exclude = self._prop_binder_exclude(prop)
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
        # Triplet-inference sleep stage (Phase 4): NREM batch schema
        # extraction folds replayed relational statistics into the profiles
        # (0.7/0.3 slow integration) + bounded REM sabotage. Additive.
        if getattr(self, "triplet_op", None) is not None:
            try:
                from ravana.core.triplet_inference import SleepSchemaExtractor
                _tse = SleepSchemaExtractor()
                result['triplet_schemas'] = _tse.extract_schemas(
                    self.triplet_op.memory)
                _tse.rem_sabotage(self.triplet_op.memory)
            except Exception:
                pass
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
        # Round 4 (C1): sleep-time junk-NODE pruning (synaptic homeostasis).
        # Edge-only prune leaves low-degree junk singletons behind; this removes
        # them (unless adjacent to a hub or in the protected core set). Count
        # folded into the sleep metrics for observability.
        try:
            _tau_low = getattr(self, "_sleep_prune_tau_low", 1)
            _tau_high = getattr(self, "_sleep_prune_tau_high", 8)
            _pruned = self.graph.prune_low_degree_junk_nodes(
                tau_low=_tau_low, tau_high=_tau_high)
            nodes_pruned, removed_labels, hub_labels = _pruned
            result['junk_nodes_pruned'] = nodes_pruned
            if getattr(self, '_trace_enabled', False) and nodes_pruned:
                print(f"  [sleep] pruned {nodes_pruned} low-degree junk nodes")
            # Round 5 (D1): self-label the sleep oracle. Removed nodes = decayed
            # traces (negative); surviving hubs = consolidated (positive). These
            # feed the self-supervised junk classifier's weak-label buffer.
            try:
                from ravana.chat.junk_scorer import record_label, get_buffer, refit_now
                for _lbl in removed_labels:
                    record_label(_lbl, "pruned",
                                 meta={"degree": 0, "source_count": 1, "glove_mag": None})
                # Sample a few surviving hubs as positives (avoid label flood).
                for _lbl in hub_labels[:50]:
                    record_label(_lbl, "hub",
                                 meta={"degree": _tau_high, "source_count": 5,
                                       "glove_mag": 1.0})
                # Advance the consolidation cycle and refit periodically.
                _buf = get_buffer(getattr(self, "data_dir", None))
                _buf.tick()
                _refit_every = getattr(self, "_junk_refit_every", 1)
                if self.sleep_cycles_completed % _refit_every == 0:
                    _delta = refit_now()
                    if _delta and getattr(self, '_trace_enabled', False):
                        print(f"  [junk-clf] refit n={_delta['n']} "
                              f"brier={_delta['brier_after']:.3f} "
                              f"theta={_delta['theta']:.3f} kappa={_delta['kappa']:.3f}")
            except Exception as e:
                if getattr(self, '_trace_enabled', False):
                    print(f"  [sleep] junk-self-label error: {e}")
        except Exception as e:
            if getattr(self, '_trace_enabled', False):
                print(f"  [sleep] prune_low_degree_junk_nodes error: {e}")
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
        # Phase 3b: drain hippocampal buffer facts into the neocortical graph.
        # get_consolidation_candidates() (hippocampal_buffer.py:292) was never
        # called anywhere in the engine — buffer facts stayed raw forever
        # (Item 2, P1). Here, after replay, high-confidence rehearsed facts
        # graduate to durable graph edges (complementary learning systems:
        # hippocampal -> neocortex during sleep) and are marked consolidated so
        # they are not re-drained next cycle.
        try:
            _cands = self.hippocampal_buffer.get_consolidation_candidates()
            _graduated = 0
            _user_skipped = 0
            for _ft in _cands:
                try:
                    # Source monitoring (investigation Gap 2): user self-
                    # disclosures must NOT become entity-keyed world edges
                    # ("my cat is pixel" -> edge about cats-in-general). They
                    # already graduate through the personal_facts drain below.
                    # Mark consolidated so they stop appearing as candidates.
                    if getattr(_ft, 'user_fact', False):
                        self.hippocampal_buffer.mark_consolidated(_ft)
                        _user_skipped += 1
                        continue
                    self._ensure_relation(_ft.subject, _ft.object,
                                          _ft.predicate,
                                          weight=float(getattr(_ft, 'confidence', 0.8)))
                    self.hippocampal_buffer.mark_consolidated(_ft)
                    _graduated += 1
                except Exception:
                    continue
            result['buffer_facts_graduated'] = _graduated
            result['user_facts_withheld'] = _user_skipped
            if getattr(self, '_trace_enabled', False) and _graduated:
                print(f"  [sleep] graduated {_graduated} hippocampal facts to graph")
        except Exception as e:
            if getattr(self, '_trace_enabled', False):
                print(f"  [trace] buffer->graph consolidation error: {e}")
        # Phase 3b2: drain the learned personal-fact store into the graph
        # (B5). Confident + rehearsed user-profile facts ("my cat is Pixel")
        # graduate to durable edges tagged source="personal_fact" so they
        # become stable semantic memory rather than only living in the
        # per-session user_model. Same CLS design as the hippocampal drain.
        try:
            _pf = getattr(self, 'user_model', None)
            _pf_store = getattr(_pf, 'personal_facts', None) if _pf else None
            _pf_grad = 0
            if _pf_store is not None:
                for _ft in _pf_store.get_consolidation_candidates():
                    try:
                        self._ensure_relation(
                            _ft.subject, _ft.value,
                            _ft.attribute,
                            weight=float(getattr(_ft, 'confidence', 0.7)))
                        _pf_grad += 1
                    except Exception:
                        continue
            result['personal_facts_graduated'] = _pf_grad
            if getattr(self, '_trace_enabled', False) and _pf_grad:
                print(f"  [sleep] graduated {_pf_grad} personal facts to graph")
        except Exception as e:
            if getattr(self, '_trace_enabled', False):
                print(f"  [trace] personal-fact drain error: {e}")
        # Phase 3b3: drain the learned opinion store into OPINION edges (C4).
        # Stances become graph edges tagged relation_type="opinion" with weight
        # = polarity * confidence, so the graph treats "user thinks X is great"
        # as a subjective stance, never as a semantic fact about X.
        try:
            _op = getattr(self, 'user_model', None)
            _op_store = getattr(_op, 'opinions', None) if _op else None
            _op_grad = 0
            if _op_store is not None:
                for _st in _op_store.get_consolidation_candidates():
                    try:
                        self._ensure_relation(
                            "user", _st.topic,
                            "opinion",
                            weight=float(_st.polarity * _st.confidence))
                        _op_grad += 1
                    except Exception:
                        continue
            result['opinions_graduated'] = _op_grad
            if getattr(self, '_trace_enabled', False) and _op_grad:
                print(f"  [sleep] graduated {_op_grad} opinion stances to graph")
        except Exception as e:
            if getattr(self, '_trace_enabled', False):
                print(f"  [trace] opinion drain error: {e}")
        # Phase 3c: Hebbian reinforcement of the ConnectorLearner (Item 3, P1).
        # Re-affirm each confirmed connector->relation association from the
        # learner's own discovered set, nudging prototype centroids toward the
        # connectors' vectors so retrieval generalizes to the observed lexical
        # neighborhood. Runs during sleep (offline, per consolidation cycle).
        try:
            _cl = getattr(self, '_connector_learner', None)
            if _cl is not None and _cl._is_initialized:
                for _w, _rt in list(_cl._connector_to_rel.items()):
                    _v = self._glove_vector(_w) if hasattr(self, '_glove_vector') else None
                    _cl.hebbian_update(_w, _rt, vec=_v, learning_rate=0.05)
        except Exception as e:
            if getattr(self, '_trace_enabled', False):
                print(f"  [trace] connector hebbian error: {e}")
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

    def _topic_set_gate(self, candidates: List[Tuple[str, float]],
                         subject: str,
                         min_coherence: float = 0.20) -> List[Tuple[str, float]]:
        """PFC topic-set maintenance gate: admit only associations whose GloVe
        cosine to the grounded subject is above a coherence threshold.

        Brain basis: the PFC maintains an active topic representation and
        inhibits associations that are not coherent with it (Miller & Cohen
        2001; 'global workspace' gating in Dehaene 2011). This is the same
        mechanism whether the downstream path is reflective, creative, or
        definitional — a single reusable gate.

        Fail-closed: if the subject has no embedding, use the syntactic head
        noun; if that also fails, return the subject itself as the sole anchor.
        """
        if not candidates or not subject:
            return candidates
        _vec = self._glove_vector(subject) if hasattr(self, "_glove_vector") else None
        if _vec is None:
            _head = self._subject_head(subject, subject) if hasattr(self, "_subject_head") else ""
            if _head:
                _vec = self._glove_vector(_head) if hasattr(self, "_glove_vector") else None
        if _vec is None:
            # No embedding for subject — return candidates unfiltered;
            # the downstream coherence gate will catch bad output.
            return candidates
        _norm = float(np.linalg.norm(_vec))
        if _norm <= 0:
            return candidates
        _subj_tokens = set(re.findall(r"[a-z']+", subject.lower()))
        filtered = []
        for label, score in candidates:
            ll = label.lower()
            if ll in _subj_tokens:
                continue
            lv = self._glove_vector(ll) if hasattr(self, "_glove_vector") else None
            if lv is None:
                continue
            _cos = float(np.dot(_vec, lv) / (_norm * float(np.linalg.norm(lv)) + 1e-9))
            if _cos >= min_coherence:
                filtered.append((label, score))
        if not filtered:
            # Self-anchor: return the subject as the sole association so the
            # generator stays on-topic rather than emitting nothing.
            return [(subject, 1.0)]
        return filtered

    def _ground_query(self, text: str) -> Tuple[str, float, str]:
        """Multi-strategy query grounding. Returns (subject, confidence, method).

        Strategies (tried in order):
        a) PrefrontalWorkspace question type parsing & exact phrase matching
        b) Compositional — split phrase, count known vs unknown words
        c) Phrase embedding similarity — mean word vec → nearest concept (cosine > 0.75)
        d) Best single word fallback — last meaningful non-stop word
        """
        print(f"  [ground_query] input={text!r}")
        # Normalize ELI5 / simplification tails BEFORE grounding so they don't

        # pollute the subject. "explain quantum entanglement like i am five"
        # must ground to "quantum entanglement", not "... like i am five".
        _text = self._strip_eli5_tail(text).lower()
        # Strategy A: Use PrefrontalWorkspace question type detection to parse semantic payload
        qtype = "general"
        query_phrase = ""
        groups = []
        try:
            if hasattr(self, 'pfc_workspace'):
                qtype, groups = self.pfc_workspace.detect_question_type(_text, self._concept_pos)
                print(f"  [ground_query] pfc qtype={qtype!r} groups={groups!r}")
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
        except Exception as e:
            print(f"  [ground_query] pfc failed: {e!r}")
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

        # Strategy C (moved before B)

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
        print(f"  [ground_query] query_phrase={query_phrase!r} words={words!r}")
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
                        if not self._is_generic_noun(last_word):
                            _leading = " ".join(words[:-1])
                            if re.search(r"\b(would|could|will|might|if|when|suddenly|disappear|gone|removed|vanished)\b", _leading):
                                return (last_word, 0.7, "scenario_last_entity")

                clean_subj = " ".join(words[:3]) if len(words) >= 3 else " ".join(words)
                clean_subj = self._clean_subject_phrase(clean_subj)
                # Malformed-grounding guard (fixes "is it ever okay to break a
                # promise" -> "ever okay break"): when the phrase is dominated by
                # generic framing words (ever/okay/break/...), it is NOT a real
                # multi-word subject — re-derive the salient head noun from the
                # query via the distributional _subject_head (e.g. "promise").
                # Only triggers on framing-laden phrases, so genuine subjects
                # like "ocean salty" / "time machine" are untouched. The framing
                # set is sourced from the consolidated functional lexicon.
                _FRAMING = (tuple(self._func_lex.framing)
                            if self._func_lex is not None
                            else {"ever", "okay", "ok", "break", "make",
                                  "really", "right", "wrong", "thing",
                                  "things", "actually"})
                if any(w in _FRAMING for w in clean_subj.split()):
                    _head = self._subject_head(clean_subj, _text)
                    if _head and _head not in _FRAMING:
                        return (_head, 0.6, "subject_head")
                return (clean_subj, 0.45, "multi_word_unconnected")

            known_words = [w for w in words if w in self._concept_labels or w in self._concept_keywords]
            unknown_words = [w for w in words if w not in known_words]
            if known_words:
                if unknown_words:
                    clean_subj = " ".join(words[:3]) if len(words) >= 3 else " ".join(words)
                    clean_subj = self._clean_subject_phrase(clean_subj)
                    return (clean_subj, 0.35, "partial_unknown")
                ratio = len(known_words) / len(words)
                _generic = self._GENERIC_NOUNS
                specific = [w for w in known_words if w not in _generic]
                topic = specific[0] if specific else known_words[0]
                if topic in _generic and len(words) > 1:
                    clean_subj = " ".join(words[:3]) if len(words) >= 3 else " ".join(words)
                    clean_subj = self._clean_subject_phrase(clean_subj)
                    if clean_subj:
                        return (clean_subj, 0.4, "compositional_generic_topic")
                return (topic, min(0.85, 0.5 + ratio * 0.4), f"compositional_{ratio:.2f}")
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
            if best_label and self._adaptive_gate("phrase_sim", best_sim, strict=True) and best_label.lower() not in self.TOPIC_SKIP_WORDS:
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

        print(f"  [ground_query] no_match fallback")
        return ("", 0.0, "no_match")

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
        s = re.sub(r"\b(?:like|as if|as though)\b\s+(?:i am|i'm|im|i|i were|he is|she is|he were|she were|a|an|they are)\s+"
                   r"(?:five|five year old|five years old|(?:a |an )?\d+\s*(?:year|yr)s?\s*old)\b.*$",
                   "", s)
        # "in simple terms", "in plain language", "simply", "for kids", "for a child".
        s = re.sub(r"\b(?:in (?:simple|plain|basic|layman'?s) (?:terms|language|words|english)|"
                   r"simply|for (?:kids|a child|beginners|dummies))\b", "", s)
        return s.strip()

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
        _gate_key = "schema_cos_hi" if pe < 0.2 else ("schema_cos_lo" if pe > 0.5 else "schema_cos")
        schema_ids = {subj_nid}
        # Snapshot: background learner may add nodes mid-turn (avoid
        # "dictionary changed size during iteration").
        for other_nid, other_node in list(self.graph.nodes.items()):
            if other_nid == subj_nid or other_node.vector is None:
                continue
            cos = float(np.dot(subj_node.vector, other_node.vector))
            if self._adaptive_gate(_gate_key, cos, strict=True):
                schema_ids.add(other_nid)
                self.graph.activate(other_nid, 0.6)
        return schema_ids

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

    def _is_informational_query(self, query: str, subject: str) -> bool:
        """Determines if a query is informational/fact-seeking (asks for a definition,
        factual knowledge, or explanation of an unknown concept) rather than
        conversational, logical, relational, or conditional.
        """
        # Stage 3 (M-A) promoted route: router drives `definition_seeking`
        # when promoted; falls through to the regex below.
        if self._router_says("definition_seeking", query):
            return True
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

