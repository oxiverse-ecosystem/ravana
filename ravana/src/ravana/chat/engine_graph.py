"""Auto-generated mixin module for CognitiveChatEngine.
Graph & concept-vector mixin — GloVe loading, edge bootstrap, category error detection, definition purge, conceptnet ontology.
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




class GraphMixin:
    """Graph & concept-vector mixin — GloVe loading, edge bootstrap, category error detection, definition purge, conceptnet ontology."""

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

    def _init_glove(self):
        """Load GloVe 100D vectors and build projection to self.dim.

        Phase 2.3: Warm-start — tries to load pre-computed projected vectors
        from 'ravana_glove_cache.npz' first. Falls back to reading the raw
        GloVe file and caching the result for next time.

        Search order for the cache:
        1. self._glove_cache_path (may be data_dir-specific, e.g. a temp dir)
        2. _proj_root/data/ravana_glove_cache.npz (repo-level, committed via LFS)
        """
        # Phase 2.3: Try warm-start cache first.
        # Check the instance path first, then fall back to the repo-level cache
        # (important when data_dir is a temp dir, as in tests).
        _repo_cache = os.path.join(_proj_root, "data", "ravana_glove_cache.npz")
        _cache_path = self._glove_cache_path
        if not os.path.exists(_cache_path) and os.path.exists(_repo_cache):
            _cache_path = _repo_cache
        if os.path.exists(_cache_path):
            try:
                data = np.load(_cache_path, allow_pickle=True)
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
        """Download GloVe 6B vectors from HuggingFace (Stanford host is down).

        Downloads glove.6B.zip (~822 MB), extracts glove.6B.100d.txt and glove.6B.50d.txt.
        Uses streaming download with progress indicator.

        Returns True on success, False on failure.

        Offline guard: an 822 MB fetch must never happen implicitly inside a
        test/CI process. When RAVANA_OFFLINE=1 (set by conftest and by the CI
        workflow) this returns False immediately, so a cache miss degrades to
        "no GloVe vectors" in milliseconds instead of stalling the job for
        minutes on a slow or 503-ing mirror. Set RAVANA_ALLOW_GLOVE_DOWNLOAD=1
        to opt back in explicitly (e.g. scripts/download_datasets.py).
        """
        import zipfile

        if os.environ.get("RAVANA_OFFLINE") == "1" and \
                os.environ.get("RAVANA_ALLOW_GLOVE_DOWNLOAD") != "1":
            print("  [GloVe] Offline mode (RAVANA_OFFLINE=1) — skipping 822MB "
                  "auto-download; running without GloVe vectors.")
            return False

        # Stanford NLP host frequently returns 503/404; use the HuggingFace
        # mirror as primary (reliable CDN) with Stanford as fallback.
        glove_urls = [
            "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip",
            "https://nlp.stanford.edu/data/glove.6B.zip",
        ]
        zip_path = os.path.join(glove_dir, "glove.6B.zip")

        try:
            os.makedirs(glove_dir, exist_ok=True)

            # Stream download with progress
            for glove_url in glove_urls:
                try:
                    print(f"  [GloVe] Downloading from {glove_url}...")
                    req = urllib.request.Request(glove_url, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) RAVANA/1.0'
                    })

                    with urllib.request.urlopen(req, timeout=60) as resp:
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
                    print(f"  [GloVe] Download from {glove_url} failed: {e}")
                    # Clean up partial download
                    try:
                        if os.path.exists(zip_path):
                            os.remove(zip_path)
                    except Exception:
                        pass
                    continue

            # All mirrors failed
            print("  [GloVe] All download mirrors exhausted.")
            return False
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

    def _domain_of(self, url: str) -> str:
        from urllib.parse import urlparse
        try:
            net = urlparse(url).netloc.lower()
            return net[4:] if net.startswith("www.") else net
        except Exception:
            return (url or "").lower()

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
        # Remove dangling identifiers / reference handles that leak into answers
        # (observed: "according to an official source, doi: 10." — a truncated
        # DOI/identifier fragment from the source markup that is NOT part of any
        # sentence). This is morphological chrome cleanup, not a fact edit — it
        # only removes non-linguistic residue shaped exactly like an identifier
        # handle, never ordinary words or numbers.
        # 1) A handle token: "doi: 10.1234/abc", "ISBN 978-...", "arxiv:1234.56",
        #    "PMID: 123456" — delete the whole handle + its id. The id pattern
        #    includes a bare "10." fragment (the truncated-DOI leak case) but is
        #    ONLY consumed when immediately preceded by the handle word, so a
        #    real "version 10.2" in prose is never touched.
        text = re.sub(r"\b(?:doi|isbn|issn|arxiv|pmid)\b\s*[:=]?\s*"
                      r"(?:\d+\.\S+|\d{3,})?", "", text,
                      flags=re.IGNORECASE)
        # 2) Any bare URL.
        text = re.sub(r"\bhttps?://\S+", "", text)
        # 3) Stray standalone "doi"/"isbn" word with no following id (the
        #    "according to an official source, doi" case where the id got cut).
        text = re.sub(r"\b(?:doi|isbn|issn|arxiv|pmid)\b", "", text,
                      flags=re.IGNORECASE)
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

