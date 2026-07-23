"""Auto-generated mixin module for CognitiveChatEngine.
Persistence & correction mixin — checksum, pickle safety, correction detection/consolidation.
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




class PersistenceMixin:
    """Persistence & correction mixin — checksum, pickle safety, correction detection/consolidation."""

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

