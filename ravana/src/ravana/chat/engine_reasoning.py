"""Auto-generated mixin module for CognitiveChatEngine.
Reasoning & query-classification mixin — paradox, causal/temporal/arithmetic reasoning, query type detection, scenario cleaning.
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




class ReasoningMixin:
    """Reasoning & query-classification mixin — paradox, causal/temporal/arithmetic reasoning, query type detection, scenario cleaning."""

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
                    # M4: structural-junk screen. Pass the snippet's topic
                    # coherence so the dual-gate (high PE AND low coherence)
                    # fires on real junk rather than merely surprising-but-on-
                    # topic prose (the model never rejects without coherence).
                    _coh = self._snippet_topic_max_coherence(_topic, _snip)
                    if hasattr(self, "_snippet_is_structural_junk") and \
                            self._snippet_is_structural_junk(_snip, _coh):
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
                    or w in self._closed_class("common_words")
                    or w in getattr(self, "_proper_nouns", set())):
                _real += 1
                continue
            if self._glove_vecs is not None and self._glove_vector(w) is not None:
                _real += 1
        # Fewer than half the tokens are real words -> treat as gibberish.
        return _real * 2 < len(meaningful)

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
        from .brain_regions import parse_number_phrase, count_sequence

        # ── §6 cerebellar / number-line module ──────────────────────────────
        # Counting and number-word arithmetic are DETERMINISTIC procedural
        # sub-routines (cerebellum), not semantic associations. They must fire
        # BEFORE the decoder/web pipeline, which would otherwise produce
        # word-salad ("count to ten" -> associative noise) or an honesty punt
        # ("two plus two" -> "outside what i know").
        #
        # (a) COUNT TO N  — "count to ten" / "count from 1 to 10". Pure ordered
        #     sequence generation over the number line (no graph walk).
        _count = re.match(
            r"^\s*(?:can you |please )?(?:count|list|give me|say|recite)\b.*?\b"
            r"(?:to|up to|from 1 to|1 to)\b\s*([a-z0-9 ]+?)\s*[.!?]?$",
            user_input.lower().strip())
        if _count:
            n = parse_number_phrase(_count.group(1), self._glove_vector)
            seq = count_sequence(n)
            if seq:
                joined = ", ".join(str(x) for x in seq)
                return f"{joined}." if len(seq) <= 20 else f"{seq[0]}, {seq[1]}, … up to {seq[-1]}."
        # (a.5) ROOTS / RADICALS — "square root of X", "cube root of X",
        # "sqrt(X)". Defect D fix: the old arithmetic pass only handled
        # + - * / ^, so "square root of negative one" fell through to the web
        # path and surfaced a misleading snippet ("a positive number"). Roots
        # are deterministic procedural math (cerebellum), evaluate them directly
        # with the stdlib `math` module and a sign check — imaginary results
        # are reported as i·√|x|, never asserted as a real number. No web
        # lookup, no confabulation.
        import math as _math
        # Capture the radicand as a phrase AFTER "root (of/from)" — this covers
        # both literal digits ("square root of -1", "sqrt(9)") AND spelled-out
        # numbers ("square root of negative one", "cube root of eight"). We
        # parse the phrase with parse_number_phrase (GloVe-derived number map,
        # not a hardcoded table) so no topic list is needed.
        _root_degree = re.search(
            r"\b(cube root|fourth root|square root|sqrt|root)\b",
            user_input.lower().replace("×", "*").replace("x", "*"))
        _radicand_phrase = None
        if _root_degree:
            # find the phrase following "of"/"from"/"(" after the root word
            _m = re.search(
                r"\b(?:square root|cube root|fourth root|sqrt|root)\b"
                r"\s*(?:of|from|\(|\s)*\s*([a-z0-9 +\-]+?)\s*(?:\)|\?|\.|,|$)",
                user_input.lower().replace("×", "*").replace("x", "*"))
            if _m:
                _radicand_phrase = _m.group(1).strip().rstrip(").")
        if _root_degree and _radicand_phrase is not None:
            _degree = 2
            _raw = user_input.lower()
            if re.search(r"cube root", _raw):
                _degree = 3
            elif re.search(r"fourth root", _raw):
                _degree = 4
            # Parse the radicand: digits first, else spelled-out number words.
            # Handle an explicit sign word ("negative one" / "minus eight") so
            # the sign check below reports imaginary roots correctly.
            _x = None
            _phrase = _radicand_phrase
            _neg = False
            if re.match(r"^(negative|minus)\b", _phrase):
                _neg = True
                _phrase = re.sub(r"^(negative|minus)\s+", "", _phrase).strip()
            try:
                _x = float(_phrase)
            except (ValueError, TypeError):
                try:
                    _x = parse_number_phrase(_phrase, self._glove_vector)
                except Exception:
                    _x = None
            if _x is not None and _neg:
                _x = -abs(_x)
            if _x is not None:
                if _x < 0 and _degree % 2 == 1:
                    # odd root of a negative number is real and negative
                    _val = -_math.pow(-_x, 1.0 / _degree)
                    return (f"{user_input.strip().rstrip('?.')} = {_val:.4g} "
                            f"(negative, since an odd root of a negative is real).")
                if _x < 0 and _degree % 2 == 0:
                    # even root of a negative number is imaginary: i·ⁿ√|x|
                    _mag = _math.pow(-_x, 1.0 / _degree)
                    return (f"{user_input.strip().rstrip('?.')} = i·{_mag:.4g} "
                            f"— that's imaginary (no real number squares to a "
                            f"negative).")
                _val = _math.pow(_x, 1.0 / _degree)
                return (f"{user_input.strip().rstrip('?.')} = {_val:.4g}.")
        # (b) NUMBER-WORD ARITHMETIC — "two plus two" / "ten times five". The
        #     operands are recovered from the GloVe-derived number ordinal map,
        #     not a hardcoded table, then evaluated with the whitelisted ops.
        if re.search(r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|"
                     r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
                     r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|"
                     r"sixty|seventy|eighty|ninety|hundred)\b", user_input.lower()):
            # Translate spelled-out numbers to digits, preserving operators.
            _expr = user_input.lower()
            # operators already normalized below; first map number words.
            _word_to_digit = {
                "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
                "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
                "fourteen": "14", "fifteen": "15", "sixteen": "16",
                "seventeen": "17", "eighteen": "18", "nineteen": "19",
                "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
                "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
                "hundred": "100",
            }
            # Compound tens ("twenty one") -> "20 1" then merge to "21".
            def _num_words_to_digits(txt):
                toks = re.findall(r"[a-z]+", txt)
                out, seen_tens = [], None
                for tk in toks:
                    if tk in ("plus", "minus", "times", "multiplied", "divided",
                              "by", "to", "the", "power", "of", "raised"):
                        continue
                    if tk in _word_to_digit:
                        v = _word_to_digit[tk]
                        if seen_tens is not None:
                            out.append(str(int(seen_tens) + int(v)))
                            seen_tens = None
                        else:
                            out.append(v)
                    elif tk in ("twenty", "thirty", "forty", "fifty", "sixty",
                                "seventy", "eighty", "ninety"):
                        seen_tens = {"twenty": 20, "thirty": 30, "forty": 40,
                                     "fifty": 50, "sixty": 60, "seventy": 70,
                                     "eighty": 80, "ninety": 90}[tk]
                return " ".join(out)
            # Re-assemble: replace only the number-word spans with digits, keep
            # operator words ("plus"/"times"...) which are normalized below.
            _assembled = _expr
            for w, d in _word_to_digit.items():
                _assembled = re.sub(rf"\b{w}\b", d, _assembled)
            # Now normalize operator words.
            _assembled = re.sub(r"\bplus\b", "+", _assembled)
            _assembled = re.sub(r"\bminus\b", "-", _assembled)
            _assembled = re.sub(r"\b(?:times|multiplied by)\b", "*", _assembled)
            _assembled = re.sub(r"\bdivided by\b", "/", _assembled)
            _assembled = re.sub(r"\b(?:to the power of|raised to)\b", "^", _assembled)
            _assembled = re.sub(r"^(what(?:'s| is)|calculate|compute|solve|"
                                r"find|tell me|how much is|how many is)\s+", "",
                                _assembled).strip()
            _assembled = _assembled.rstrip("?. ").strip()
            if re.fullmatch(r"\s*[-+]?\d+(?:\.\d+)?(?:\s*[+\-*/^]\s*[-+]?\d+"
                             r"(?:\.\d+)?)+\s*", _assembled):
                try:
                    _ops = {"+": operator.add, "-": operator.sub,
                            "*": operator.mul, "/": operator.truediv, "^": operator.pow}
                    _toks = re.findall(r"([-+]?\d+(?:\.\d+)?)|([+\-*/^])", _assembled)
                    _nums2, _seq = [], []
                    for _n, _o in _toks:
                        if _n:
                            _nums2.append(float(_n))
                        elif _o:
                            _seq.append(_o)
                    _c = _nums2[0]
                    for _i, _op in enumerate(_seq):
                        _c = _ops[_op](_c, _nums2[_i + 1])
                    if _c == int(_c) and abs(_c) < 1e15:
                        _res = str(int(_c))
                    else:
                        _res = f"{_c:.4g}".rstrip("0").rstrip(".")
                    return f"{user_input.strip().rstrip('?.')} = {_res}."
                except (ZeroDivisionError, OverflowError, ValueError):
                    pass

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

    def _try_hippocampal_retrieval(self, ctx) -> Optional[str]:
        """Try to retrieve a fact the user stated earlier this conversation.

        Fix (LoCoMo / LongMemEval): this method used to hard-require
        ``self._recall_mode`` (set only by the narrow _detect_recall_trigger,
        which fires on EXPLICIT recall phrasing like "do you remember"). Direct
        questions about a previously-mentioned subject ("what is wrong with my
        car?", "when did I go to X?") never set recall_mode, so a stored fact was
        never surfaced and the query fell through to a generic dictionary
        definition. We now attempt retrieval whenever the query subject matches a
        stored episodic fact; ``_recall_mode`` only boosts confidence. Returns
        None (fail-open) when nothing matches, so fresh-engine benchmarks with an
        empty buffer are unaffected.
        """
        if not getattr(ctx, "subject", None):
            return None
        try:
            facts = self.hippocampal_buffer.retrieve(ctx.subject)
        except Exception:
            return None
        if not facts:
            return None
        # Highest-confidence fact wins (Fix 4: was `return None`, dead path).
        best_fact = max(facts, key=lambda f: f.confidence)
        return best_fact.object

    def _phrase_recalled_fact(self, user_input: str, subject: str,
                              fact_object: str) -> str:
        """Phrase a recalled episodic fact as a natural answer.

        ``fact_object`` is the stored utterance (e.g. "i went to the lgbtq
        support group on 7 may 2023"). We echo it back conversationally rather
        than dumping the raw triple, and normalise a leading first-person "i/my"
        to "you/your" since the fact was something the USER told us.
        """
        text = (fact_object or "").strip()
        if not text:
            return "you mentioned that earlier, but i don't have the details."
        # Flip first-person -> second-person so the recall reads naturally.
        import re as _re
        flipped = text
        flipped = _re.sub(r"^\s*i\s+", "you ", flipped, flags=_re.IGNORECASE)
        flipped = _re.sub(r"^\s*my\s+", "your ", flipped, flags=_re.IGNORECASE)
        flipped = _re.sub(r"\bmy\b", "your", flipped)
        flipped = _re.sub(r"\bI\b", "you", flipped)
        return f"you told me earlier: {flipped}"


    def _is_self_disclosure_stmt(self, user_input: str) -> bool:
        """vmPFC-mimetic gate: detect first-person self-disclosure STATEMENTS.

        Self-referential processing is orthogonal to semantic-feasibility
        checking (Suzuki 2022: the mPFC values self-relevant info and routes it
        to autobiographical storage, NOT to the dACC category-error channel).
        This MUST fire BEFORE the frontopolar feasibility gate so a disclosure
        like "my favorite color is purple" is stored + acknowledged instead of
        being misrouted into the "color of Tuesday" cross-modal metaphor.

        PROMOTED-ROUTE FAST PATH: if the fused prototype router (schema v4,
        reference-target axis) confidently classifies this as `self_disclosure`
        and that route is promoted, trust it and return True. The router only
        returns `self_disclosure` when its margin over the runner-up is cleared
        AND the route is in the `promoted` allow-list, so this is safe; when the
        router is silent or unpromoted it falls through to the legacy regex
        below (fail-open, no behavior change).

        Catches the STATEMENT forms the identity block (which only matches
        QUESTION forms) misses:
          - "my favorite X is Y"  (favorite)
          - "my X is Y" / "i am X" / "i'm X"  (self-description / possession)
          - "i love/like/hate X"  (affect preference)
          - "my name is X" / "i am called X" / "call me X"  (name)
        Does NOT catch questions ("what is my favorite X" — handled by the
        identity block), nor creative/request frames ("tell me a story about a
        lonely robot" — handled by the TPJ frame-guard, A2).
        """
        if self.use_intent_router and self._router_says("self_disclosure", user_input):
            return True

        q = (user_input or "").lower().strip()
        if not q:
            return False
        # Exclude request/creative frames (the TPJ gate, A2) — a self-disclosure
        # is never embedded in an imperative "tell/write/imagine me a ..." frame.
        if re.search(
            r"\b(tell|write|create|make|imagine|describe|teach|draw|compose|give)\b"
            r".*\b(me|us|him|her|them)\b", q):
            return False
        # Request-frame with explicit artifact ("a story about", "a poem about")
        if re.search(r"\b(a|an|the)\s+(story|poem|song|haiku|joke|tale|letter)\s+(about|of|for)\b", q):
            return False
        # M-E (plan Stage 1): "remember X = store X" is an INTENTIONAL
        # declarative-encoding directive (the self-reference effect — info linked
        # to the self is encoded most richly; Squire's hippocampal episode-
        # specific neurons bind *what happened*). It is the OPPOSITE of a recall
        # query. So: a "remember"-framed utterance that ALSO contains a
        # first-person self-disclosure proposition ("remember i love
        # stargazing", "remember my favorite color is blue") is a high-salience
        # STORE command — route it into the self-disclosure path (return True).
        # Pure recall phrasings ("remember what i told you", "do you remember my
        # cat") carry no new disclosure proposition and still fall through to
        # the episodic-remember recall path below.
        _has_self_disclosure_prop = bool(re.search(
            r"\b(i\s+(love|like|hate|enjoy|prefer|am|feel|have)|my\s+(favorite|"
            r"name)\s+is|my\s+\w+\s+is)\b", q))
        if re.search(r"\b(remember|remind me)\b", q) and _has_self_disclosure_prop:
            # Intentional encode directive: treat as a self-disclosure statement.
            return True
        # Exclude memory/recall frames (Human-Likeness Plan C): "remember what
        # i told you about my cat" / "what did i tell you" are EPISODIC RECALL
        # queries, not self-disclosure statements to store. They contain "my
        # cat" etc. and would otherwise be miscaught here and acked, never
        # reaching the _episodic_remember recall path. Route them past this gate.
        if re.search(
            r"\b(remember|recall|remind me|what did i|what was i|"
            r"what have i|do you remember)\b", q):
            return False
        # First-person possession / self-description / affect / name.
        _self_pat = re.compile(
            r"\b(my\s+(favorite\s+)?\w+|i\s+am|i'm|i\s+love|i\s+like|i\s+hate|"
            r"i\s+have|call\s+me|my\s+name\s+is)\b")
        if not _self_pat.search(q):
            return False
        # Reject interrogatives AND imperatives — both are the USER directing
        # the AGENT (asking or commanding), not reporting a self-fact to store.
        # "explain quantum computing like i'm five" matches the "i'm" self-pattern
        # but is an imperative request, never a disclosure. An imperative opens
        # with a verb of address (explain/write/tell/...). Fail-closed: if it
        # reads as a command or a question, it is not a disclosure statement.
        _imperative_vb = re.compile(
            r"^(explain|write|tell|create|make|imagine|describe|teach|draw|"
            r"compose|give|show|help|suggest|recommend|list|generate|find|"
            r"what|who|whom|which|when|where|why|how|can|could|should|would|"
            r"do|does|did|is|are|was|were|will|have|has|am|please)\b")
        if (re.search(r"\?$", q) or _imperative_vb.match(q)):
            return False
        return True

    def _process_self_disclosure_stmt(self, user_input: str) -> str:
        """Store a self-disclosure statement (hippocampal binding, Yonelinas 2019)
        and acknowledge it naturally.

        Grounded in the SAME parsing that UserModel.observe_user_query uses, so
        the statement path and the question path share one store
        (preferences["favorites"] / preferences["likes"] / user_name). The
        acknowledgment is composed from the parsed fact — never hardcoded.
        """
        q = (user_input or "").lower().strip(" ?!.")
        parsed = None  # (kind, key, val)
        # favorite
        m = re.search(r"\bmy\s+favorite\s+(.+?)\s+is\s+(.+)", q, re.IGNORECASE)
        if m:
            parsed = ("favorite", m.group(1).strip(" .!?"), m.group(2).strip(" .!?"))
        else:
            # name — capture a proper-noun phrase, stop at a clause
            # boundary ("and"/"but"/comma/period) so "my name is
            # alex and i live in berlin" does NOT swallow the rest
            # of the sentence into the name.
            mn = re.search(
                r"\b(?:my\s+name\s+is|i\s+am\s+called|call\s+me)\s+"
                r"([^.,!?]+?)(?:\s+(?:and|but|,|\.|$))", q, re.IGNORECASE)
            if mn:
                name_cand = mn.group(1).strip()
                # Drop a trailing clause tail the lazy match may have
                # included (e.g. "alex" is already clean; this is
                # a defensive trim).
                name_cand = re.split(r"\s+(?:and|but|,|\.)\s*", name_cand)[0].strip()
                nw = name_cand.split()
                if nw and nw[0].lower() in ("is", "are", "was", "were"):
                    nw = nw[1:]
                name_cap = " ".join(w.capitalize() for w in nw)
                if name_cap and name_cap.lower() not in (
                        "happy", "sad", "tired", "busy", "fine", "good",
                        "what", "who", "why", "how"):
                    parsed = ("name", None, name_cap)
            else:
                # like/love
                ml = re.search(r"\bi\s+(like|love|hate)\s+(.+)", q, re.IGNORECASE)
                if ml:
                    parsed = ("like", None, ml.group(2).strip(" .!?"))

        # Persist via the existing UserModel store (single source of truth).
        try:
            _subj = (parsed[2] if parsed else "") or "self"
            self.user_model.observe_user_query(
                user_input, _subj,
                float(getattr(self.emotion.state, "valence", 0.5))
                if hasattr(self, "emotion") else 0.5)
        except Exception:
            # Fallback: write directly so storage is never lost even if
            # observe_user_query changes upstream.
            prefs = getattr(self.user_model, "preferences", None)
            if prefs is None:
                prefs = self.user_model.preferences = {}
            if parsed and parsed[0] == "favorite":
                prefs.setdefault("favorites", {})[parsed[1]] = parsed[2]
            elif parsed and parsed[0] == "like":
                prefs.setdefault("likes", [])
                if parsed[2] not in prefs["likes"]:
                    prefs["likes"].append(parsed[2])
            elif parsed and parsed[0] == "name":
                self.user_model.user_name = parsed[2]
        # Mirror the salient self-facts into the hippocampal entity
        # index so the EXISTING recall path (_retrieve_episodic,
        # which reads self._episodic_index) can answer later
        # "where do I live?" / "what is my name?" without a
        # separate code path. Entity "i" carries the user's own
        # biographical attributes (location / name / background).
        # Runs UNCONDITIONALLY (after both the normal and fallback
        # persist paths) so the index is always populated.
        _ei = getattr(self, "_episodic_index", None)
        if _ei is not None:
            _me = _ei.setdefault("i", {})
            if getattr(self.user_model, "user_name", ""):
                _me["name"] = self.user_model.user_name
            _loc = getattr(self.user_model, "user_location", "")
            if _loc:
                _me["location"] = _loc
            _bg = getattr(self.user_model, "user_background", "")
            if _bg:
                _me["background"] = _bg

        # Compose a gist-based acknowledgment (no templates: derived from the
        # parsed fact so it reads as a person who just heard you).
        if parsed is None:
            ack = "got it — thanks for telling me."
        elif parsed[0] == "favorite":
            ack = f"noted! i'll remember your favorite {parsed[1]} is {parsed[2]}."
        elif parsed[0] == "name":
            ack = f"nice to meet you, {parsed[2]}! i'll remember that."
        elif parsed[0] == "like":
            _obj = parsed[2]
            # §7 deictic resolution: "i love you" -> the user's 1st-person
            # declaration is addressed to the agent, so the agent reciprocates
            # ("i love you too"), never echoes it back as "you love you". This
            # is a structural I<->user, you<->agent map, not content.
            if _obj.strip() in ("you", "u", "ur", "your"):
                _verb = "love" if "love" in q else "like"
                ack = f"aw, i {_verb} you too."
            else:
                ack = f"good to know — you {'love' if 'love' in q else 'like'} {_obj}. i'll keep that in mind."
        else:
            ack = "got it — thanks for telling me."

        # Episodic transcript already captured this turn in _record_episode;
        # mark it stored so the fail-closed path doesn't double-fire a web lookup.
        self._episodic_miss = False
        return ack

    def _ensure_self_model(self) -> "SelfModel":
        """Lazily derive the self-model from the seeded graph (vmPFC content)."""
        from .brain_regions import SelfModel
        if self.self_model is None:
            try:
                self.self_model = SelfModel.from_graph(self.graph_engine)
            except Exception:
                self.self_model = SelfModel()
        return self.self_model

    def _affect_is_relevant(self, target: str, tvec) -> bool:
        """Issue 1 — Emotion–Topic Congruence Gate (vmPFC value integration).

        Decides whether the agent's CURRENT ambient mood may surface as a
        first-person affective echo for `target`. Brain-faithful: the feeling
        is only "about the topic" when the topic's embedding is near the
        mood's affect anchor. A neutral ambient mood (no salient valence) has
        NO affect anchor, so congruence is undefined -> the echo is suppressed
        (the reported leak). The user opening the emotional channel
        (_emotional_channel_active) is an OR-override (they invited affect).

        Returns True (emit echo) / False (suppress, use neutral reason).
        """
        # Explicit emotional channel: user invited affect — permit echo.
        if getattr(self, "_emotional_channel_active", False):
            return True
        # No GloVe target vector -> cannot assess congruence; default suppress
        # (fail-closed: don't leak feeling without a topic to attach it to).
        if tvec is None:
            return False
        valence = 0.5
        try:
            valence = float(getattr(self.emotion.state, "valence", 0.0))
        except Exception:
            valence = 0.0
        # Map valence to a mood anchor. VAD valence is neutral at 0.0 (range
        # -1..1), so the dead-band is symmetric around 0, NOT around 0.5. A
        # near-neutral ambient mood (|valence| below the salience threshold)
        # has NO affect anchor -> the echo is suppressed (the reported leak:
        # stale neutral mood must not bleed "how i'm feeling" into every
        # reply). Only a salient positive/negative mood can anchor an echo, and
        # even then only when congruent with the topic.
        _salience = 0.25
        if valence >= _salience:
            _anchor_word = "happy"
        elif valence <= -_salience:
            _anchor_word = "sad"
        else:
            return False  # neutral ambient mood -> gate closed, no echo
        glove_fn = getattr(self, "_glove_vector", None)
        if not callable(glove_fn):
            return False
        _fv = glove_fn(_anchor_word)
        if _fv is None or tvec is None:
            return False
        _denom = (float(np.linalg.norm(_fv)) * float(np.linalg.norm(tvec)) + 1e-9)
        _c = float(np.dot(_fv, tvec) / _denom)
        return _c > getattr(self, "_affect_congruence_tau", 0.35)

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
        same web retrieval + learning loop. We keep
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
        # Stage 3 (M-A) promoted route: router drives `factual_yesno` when
        # promoted; falls through to the regex below.
        if self._router_says("factual_yesno", text):
            return True
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

    def _is_conditional_query(self, text: str) -> bool:
        """Detect a hypothetical / counterfactual scenario ('if X happened…').

        Heuristic, not a fact table: any of the conditional lead-ins
        (if / suppose / assume / what if / what would happen) marks a scenario
        the user wants *reasoned out*, ideally from the web — not a generic
        reflective 'what does it mean to you?' turn.
        """
        # Stage 3 (M-A) promoted route: the fused prototype router drives the
        # decision for `conditional` when promoted; falls through to regex below.
        if self._router_says("conditional", text):
            return True
        t = text.lower().strip(" ?!.")

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
                     if w.strip(".,!?") not in self._closed_class("conditional_frame")
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
                 if w not in self._closed_class("conditional_frame") and w not in RELATIONAL]
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
                              if w.strip(".,!?") not in self._closed_class("conditional_frame")
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
                              if w.strip(".,!?") not in self._closed_class("conditional_frame")
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
        # M3-E: world-skeleton (above) + Human-Likeness Plan (A2): the classic
        # "tree falls in a forest" counterfactual has no causal edges in the
        # lived graph, so the forward-simulator returns [] and the query fell
        # into the category-error metaphor dead-end. Seed the physical causal
        # chain so the simulator can forward-chain a REAL causal answer:
        #   tree → fall → vibrate → air → sound ; and observer → perceive
        ("tree", "fall", "causal", 0.95),
        ("fall", "vibrate", "causal", 0.9),
        ("vibrate", "air", "causal", 0.85),
        ("air", "sound", "causal", 0.9),
        ("sound", "hear", "causal", 0.85),
        ("sound", "perceive", "causal", 0.8),
        ("observer", "perceive", "causal", 0.8),
        ("fall", "sound", "causal", 0.85),
        ("vibration", "air", "causal", 0.8),
        ("vibration", "sound", "causal", 0.85),
        ("noise", "hear", "causal", 0.8),
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

