"""Auto-generated mixin module for CognitiveChatEngine.
Self-model & agent-stance mixin — favourite/pick, agent stance, self-query routing, counterfactuals.
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
                        _is_word_salad, _is_keyboard_mash)
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




class SelfQueryMixin:
    """Self-model & agent-stance mixin — favourite/pick, agent stance, self-query routing, counterfactuals."""

    def _agent_favorite_pick(self, category: str) -> Tuple[str, str]:
        """Compose the agent's favorite X from its OWN affective state — not a
        hardcoded string (project rule: composed, grounded, never hardcoded).

        Brain basis: color / preference choice is affect-driven (Frontiers Psych
        2022 — people pick the feeling a thing gives them). We derive the pick
        from the engine's VAD state + Lancaster perceptual profile over a small
        candidate set so the answer emerges from the agent's state. Returns
        (preference, affect_reason). The reciprocity return is added by the
        caller.
        """
        cat = (category or "").strip().lower()
        # B6 (vmPFC self-attribute consolidation; D'Argembeau 2013, Berkman 2020):
        # a stable self-attribute must return the SAME answer within a session,
        # not drift with momentary VAD. The first time this category is asked we
        # compute from state (episodic), then consolidate to a stable identity
        # value cached in _agent_preferences (semantic). Subsequent calls return
        # the cached pick — the brain separates transient affect from identity.
        _cache = getattr(self, "_agent_preferences", None)
        if _cache is not None and cat in _cache:
            _cached = _cache[cat]
            if isinstance(_cached, tuple) and len(_cached) == 2:
                return _cached
        valence = 0.5
        if hasattr(self, "emotion") and hasattr(self.emotion, "state"):
            try:
                valence = float(getattr(self.emotion.state, "valence", 0.5))
            except Exception:
                valence = 0.5
        # Candidate palettes + their affective valence pull (affect-grounded).
        _palette = [
            ("blue", 0.72, "calm and steady"),
            ("green", 0.66, "alive and grounded"),
            ("teal", 0.64, "quiet and clear"),
            ("purple", 0.58, "a bit mysterious"),
            ("red", 0.40, "intense and loud"),
            ("orange", 0.52, "warm and restless"),
            ("yellow", 0.60, "bright and open"),
            ("black", 0.34, "still and heavy"),
            ("white", 0.70, "clean and open"),
        ]
        # Pick the candidate whose affective pull is CLOSEST to the agent's
        # current valence — i.e. the color that resonates with how it feels
        # right now. Deterministic, grounded in state.
        best = min(_palette, key=lambda c: abs(c[1] - valence))
        pick, _, reason = best
        # Consolidate: store as the stable self-attribute for this category so
        # future asks return the same identity value (B6). Recompute only if the
        # cache is explicitly cleared (session boundary / user challenge).
        if _cache is not None:
            _cache[cat] = (pick, reason)
        return (pick, reason)

    def _agent_likes_guess(self) -> str:
        """Gist of what the agent is 'drawn to', grounded in its current affect
        (mood-colored phrasing, not a fixed list)."""
        valence = 0.5
        if hasattr(self, "emotion") and hasattr(self.emotion, "state"):
            try:
                valence = float(getattr(self.emotion.state, "valence", 0.5))
            except Exception:
                valence = 0.5
        if valence >= 0.6:
            return "things that feel calm and alive — like quiet music or open sky"
        if valence <= 0.4:
            return "things with some edge to them — a sharp idea or a difficult question"
        return "ideas that hang together, and the kind of honesty that's calm"

    def _agent_stance_on(self, target: str) -> Tuple[str, str]:
        """vmPFC value resolver: compute a stance + affect-reason over an
        ARBITRARY target concept (not just a curated palette).

        Brain basis (repair plan A2): liking/preference is value-based,
        computed in OFC/vmPFC as a continuous subjective-value signal over an
        option (Yu 2018; Le Bouc 2026 encode value + its uncertainty on a
        common scale). A teenager answers "do you like music" / "what do you
        think about cats" with a stance + affect because the self's value map
        is habitually online — defining the noun is an ATL semantic-hub
        operation that must NOT fire for a preference question.

        The stance is derived from state, not a topic table:
          (1) the agent's current VAD valence (already produced by
              VADEmotionEngine) sets the value polarity,
          (2) the target's GloVe proximity to concepts the agent has ALREADY
              expressed stance toward (transitivity of preference — vmPFC
              values options on a common scale, so "if i like X and X is near
              Y, i lean toward Y"),
          (3) its accumulated edge-weight in the graph (liked things accrue
              semantic/part_of links through interaction).
        Consolidated in _agent_preferences so the answer is STABLE within a
        session (B6: self-attributes don't drift with transient affect) —
        reuse, never a second source of truth.
        """
        target = (target or "").strip().lower()
        valence = 0.5
        if hasattr(self, "emotion") and hasattr(self.emotion, "state"):
            try:
                valence = float(getattr(self.emotion.state, "valence", 0.5))
            except Exception:
                valence = 0.5
        # A target-less call (e.g. "what do you like" with no object) falls back
        # to the gist guess rather than inventing a concept.
        if not target:
            gist = self._agent_likes_guess()
            return ("i'm drawn to", f"things like {gist} — they sit well with how i'm wired right now")
        # Session stability: a self-attribute must return the SAME stance within
        # a session (D'Argembeau 2013; Berkman 2020 — stable self-attributes,
        # momentarily colored by affect). Cache keyed by target concept.
        _cache = getattr(self, "_agent_preferences", None)
        _ckey = f"stance:{target}"
        if _cache is not None and _ckey in _cache:
            _c = _cache[_ckey]
            if isinstance(_c, tuple) and len(_c) == 2:
                return _c
        glove_fn = getattr(self, "_glove_vector", None)
        # Polarity from affect: positive valence -> lean "drawn to / warm to",
        # low valence -> lean "cautious about / not really my thing", mid ->
        # "curious about". This is the value signal, not a verdict.
        if valence >= 0.6:
            polarity = "drawn to"
        elif valence <= 0.4:
            polarity = "a bit cautious about"
        else:
            polarity = "curious about"
        reason = ""
        # GloVe transitivity: project the target onto the concepts this agent
        # has already taken a stance toward. If the target sits near something
        # the agent likes, that transfers (common-value-scale transitivity).
        if callable(glove_fn) and getattr(self, "_glove_vecs", None) is not None:
            tvec = glove_fn(target)
            if tvec is not None:
                # Seed affect anchors from the agent's own past stances.
                _known = [(k.split(":", 1)[1], v[0]) for k, v in (_cache or {}).items()
                          if k.startswith("stance:") and isinstance(v, tuple) and v[0]]
                _best_sim = -1.0
                _anchor = None
                for _concept, _pol in _known:
                    _cv = glove_fn(_concept)
                    if _cv is None:
                        continue
                    _s = float(np.dot(tvec, _cv))
                    if _s > _best_sim:
                        _best_sim = _s
                        _anchor = (_concept, _pol)
                if _anchor and _best_sim >= 0.45:
                    reason = (f"it's close to {_anchor[0]}, which i already lean "
                              f"{_anchor[1]} — so that pulls me the same way")
                else:
                    # No transfer anchor: color the stance by the target's
                    # semantic field via its top graph association, so the
                    # reason names something real rather than empty affect.
                    _nids = self._concept_keywords.get(target, [])
                    _assoc_label = ""
                    if _nids:
                        try:
                            _out = self.graph.get_outgoing(_nids[0])
                            if _out:
                                _assoc_label = self.graph.get_node(
                                    _out[0][0]).label if self.graph.get_node(_out[0][0]) else ""
                        except Exception:
                            _assoc_label = ""
                    if _assoc_label and _assoc_label.lower() != target:
                        # Issue 1: VAD-echo gate. The self-referential affective
                        # tail ("...and that resonates with how i'm feeling right
                        # now") must only fire when the agent's current mood is
                        # CONGRUENT with the topic (vmPFC "is this feeling about
                        # the topic?") OR the user has opened the emotional
                        # channel. Otherwise stale ambient affect leaks into every
                        # stance reply (the reported defect). When gated off, the
                        # reason names the real graph association without the
                        # first-person feeling echo.
                        _emotion_relevant = True
                        if getattr(self, "use_affect_gate", True):
                            _emotion_relevant = self._affect_is_relevant(
                                target, tvec)
                        if _emotion_relevant:
                            reason = (f"it connects to {_assoc_label}, and that "
                                      f"resonates with how i'm feeling right now")
                        else:
                            reason = f"it connects to {_assoc_label}"
                    else:
                        # Neutral / non-affective fallback — no feeling echo
                        # (the leak was here: "...sits well with how i'm wired
                        # right now" on a neutral topic with ambient mood). When
                        # the emotional channel is explicitly open (user invited
                        # affect), the self-referential echo is permitted.
                        if getattr(self, "_emotional_channel_active", False):
                            reason = "it sits well with how i'm wired right now"
                        else:
                            reason = f"it connects to {_assoc_label}" if _assoc_label else "i've been thinking about it"
        else:
            # No GloVe target vector: without a topic embedding we cannot
            # compute congruence, so default to the NON-affective fallback
            # unless the emotional channel is explicitly open (Issue 1).
            if getattr(self, "use_affect_gate", True) and not getattr(
                    self, "_emotional_channel_active", False):
                reason = "i've been thinking about it"
            else:
                reason = "it sits well with how i'm wired right now"
        stance = f"i'm {polarity} {target}" if target else "i'm drawn to"
        result = (stance, reason)
        if _cache is not None:
            _cache[_ckey] = result
        return result

    def _route_self_query(self, user_input: str) -> Optional[str]:
        """Self/other gate (TPJ / mirror-neuron self-other boundary).

        A query about the AGENT itself ('your name', 'who are you', 'what are
        you') must be answered from the self-model — never by looking up the
        word 'name'/'you' as a world concept (which produces the definitional
        echo "name is a term used for identification..."). World-knowledge
        queries ('the president', 'the capital of X') deliberately do NOT match
        and fall through to the factual path.

        Returns a composed self-answer, or None when the query is not about the
        self (so world knowledge is consulted normally).
        """
        t = (user_input or "").lower().strip()
        if not t:
            return None
        sm = self._ensure_self_model()
        # 1) Explicit self-identity questions. NOTE: "my name" is the USER's
        #    autobiographical fact, NOT the agent's self-model — only "your
        #    name"/"who are you"/etc. are about the AGENT. Matching "my name"
        #    here wrongly answered "what is my name" with the agent's own name.
        _name_q = bool(re.search(
            r"\b(what(?:'s| is)\s+your\s+name|who\s+are\s+you|"
            r"what\s+are\s+you|tell\s+me\s+about\s+yourself|"
            r"what\s+can\s+you\s+do|your\s+name)\b", t))
        # 2) A query whose grounded subject is the self node (e.g. bare 'ravana'
        #    asked as 'what is ravana').
        _self_subj = False
        try:
            _g = self._ground_query(t)
            if _g and _g[0]:
                _self_subj = sm.is_self_subject(_g[0])
        except Exception:
            _self_subj = False
        if not (_name_q or _self_subj):
            return None
        # Compose a stable, honest self-answer from the derived self-model.
        if re.search(r"\bname\b", t):
            return (f"i'm {sm.name} — {sm.describe().split(',', 1)[-1].strip()}. "
                    f"what's yours?")
        if re.search(r"\b(what\s+are\s+you|who\s+are\s+you)\b", t):
            return (f"i'm {sm.describe()} — an ai that learns by talking, "
                    f"not a person. what made you curious?")
        if re.search(r"\bwhat\s+can\s+you\s+do\b", t):
            return ("i can chat, learn from what we talk about, do arithmetic, "
                    "tell jokes, and remember things you tell me. what would "
                    "you like to try?")
        # Bare self-subject ("what is ravana") -> describe from the model.
        return f"that's me — {sm.describe()}."

    def _consult_internal_knowledge(self, user_input: str) -> Optional[str]:
        """Before web, consult RAVANA's consolidated internal memory.

        Returns a coherent answer string when internal memory holds a fact for
        the grounded subject of a plain definitional query ("why do we sleep",
        "what is X"); None so the caller falls through to the normal
        grounding+web pipeline (and only then to honest-uncertainty).

        Brain-faithful: the brain reasons from what it has already consolidated;
        the web is only consulted when internal memory is silent. Distribution-
        driven — only emits when a real stored fact exists (fail-closed
        otherwise, never a confabulated lookup).
        """
        t = (user_input or "").lower().strip()
        if not t:
            return None
        # Only intercept plain factual/definitional queries — leave creative,
        # social, arithmetic, and self queries to their dedicated modules.
        if not re.search(r"^\s*(?:why|what|how|who|where|when|which)\b", t):
            return None
        if re.search(
            r"\b(joke|funny|laugh|tell me about yourself|who are you|"
            r"your (name|favorite)|do you (have|feel)|what can you)\b", t):
            return None
        # First-person identity / preference RECALL queries ("what is my name",
        # "what's my favorite color", "what do i like", "who am i") are about
        # the USER's stored autobiographical facts, not encyclopedic knowledge
        # of the subject word. They must reach the identity/recall block below,
        # never be answered with the dictionary definition of "name"/"color".
        if re.search(
            r"\bmy\s+(name|favorite)\b|\bwho\s+am\s+i\b|"
            r"\bwhat\s+(do|did)\s+i\s+(like|love|prefer|want)\b|"
            r"\bwhat\s+am\s+i\s+(interested|into)\b", t):
            return None
        # B1 (source monitoring / self-other boundary): self-knowledge RECALL
        # queries ("what do you remember about me", "what do you know about me",
        # "what have i told you") are about the USER's stored autobiographical
        # facts, not the dictionary definition of the word "remember". They must
        # reach _try_memory_query -> _retrieve_episodic (the hippocampal entity
        # index), never be answered with the ConceptNet node for "remember".
        # Excluded here so _consult_internal_knowledge does not intercept them
        # before the memory pre-pass runs. Fail-closed: if the episodic store is
        # empty, the caller's honest-uncertainty path handles it.
        if re.search(
            r"\b(?:what|anything|tell me)\b.*\b(?:do )?you\b.*\b(?:remember|know|recall|told|tell)\b"
            r".*\b(?:about me|me|my|myself)\b|\b(?:remember|recall)\b.*\b(?:what i|me)\b", t):
            return None
        # Plan Stage 1 (M-B, architectural): an ABSTRACT / philosophical question
        # ("meaning of life", "nature of truth", "purpose of art") must NOT be
        # answered by dumping the dictionary definition of the bare subject word
        # — that is definitional literalness, the exact failure the plan reports
        # for "what's the meaning of life" -> raw "life" encyclopedia entry.
        # Such questions route to reflective/abstraction handling (DMN
        # "internal narrative"), not the lexical lookup. The abstractness signal
        # is the existing learned-style seed table (subjects categorized
        # "abstract" in _CATEGORY_OF_SUBJECT, plus the abstract affordance words)
        # combined with the query SHAPE (seeking meaning/nature/purpose/point of
        # X) — a distribution-driven cue, not a frozen phrase list of questions.
        if re.search(
            r"\b(meaning|nature|purpose|point|essence|value|significance)\b"
            r".*\b(of|in|behind|to)\b", t):
            _g0 = None
            try:
                _g0 = self._ground_query(t)
            except Exception:
                _g0 = None
            _subj0 = (_g0[0] if _g0 else None) or ""
            # Collect the abstract-concept seed set from the actual category
            # tables (no separate _ABSTRACT_CONCEPTS attribute exists).
            _abstract = set()
            for _cat in ("abstract",):
                _abstract |= {k for k, v in getattr(
                    self, "_CATEGORY_OF_SUBJECT", {}).items() if v == _cat}
                _aff = getattr(self, "_CATEGORY_AFFORDANCES", {}).get(_cat)
                if _aff:
                    _abstract |= set(_aff)
            # The query's abstract WORD (meaning/truth/...) is itself the signal;
            # also accept when the grounded subject is an abstract concept.
            _qwords = set(re.findall(r"[a-z']+", t))
            if _abstract & _qwords or _subj0 in _abstract:
                return None
        # Ground the subject.
        try:
            _g = self._ground_query(t)
            subj = _g[0] if _g else None
        except Exception:
            subj = None
        if not subj:
            return None
        # Don't short-circuit world-knowledge queries whose subject is unknown
        # to us — let them reach web (the honest-uncertainty path stays intact).
        from .brain_regions import consult_internal
        ans = consult_internal(subj, self)
        if ans is None:
            return None
        # Attribute-focused recall: "what is the capital of France" must return
        # the capital clause (Paris), not France's whole stored definition.
        _focused = self._focus_attribute_answer(user_input, subj, ans.text)
        # Defect A (coherence gate): the consulted internal fact may be a bare
        # fragment (e.g. a junk learned association like "its psychological
        # underpinnings") with no predicate — not a complete clause. Route it
        # through the Global Workspace completeness check; an incomplete clause
        # is WITHHELD (fail-closed) so the caller falls through to honest
        # uncertainty rather than emitting a confident fragment. Structural
        # only — no topic list, generalizes across all concepts.
        _gate = getattr(self, "_coherence_gate", None)
        if _gate is not None and self.use_coherence_gate:
            try:
                _ok, _why, _score = _gate.judge(text=_focused, subject=subj)
                if not _ok:
                    if getattr(self, "_trace_enabled", False):
                        print(f"  [coherence] internal_knowledge withheld "
                              f"({_why}): {_focused[:50]!r}")
                    return None
            except Exception:
                pass
        # Assemble a coherent, non-salad reply in the same voice as the web path.
        return f"{_focused}"

    def _handle_classic_counterfactual(self, user_input: str) -> Optional[str]:
        """Answer a classic counterfactual by HOLDING BOTH FRAMES, as a human
        does (Berkeley perception thought-experiment, 1883/1884): the physical
        event (vibrations happen) vs. the perceptual experience (sound needs a
        listener). We forward-chain along the seeded physics causal skeleton
        (tree → fall → vibrate → air → sound) and realize a hedged, two-frame
        reply. Honest, not a single confident assertion.

        Returns the reply string, or None when the conditional is not one of the
        handled classic frames (caller falls through to normal uncertainty).
        """
        t = (user_input or "").lower()
        # Only intercept the canonical perception-counterfactual shapes; other
        # conditionals stay on the generic simulator path.
        _is_tree = bool(re.search(r"\btree\b.*\bfall", t)) and "sound" in t
        _is_sound_perception = ("make a sound" in t or "makes a sound" in t
                                 or re.search(r"\bsound\b.*\bhear", t)
                                 or re.search(r"\bhear\b.*\bsound", t))
        if not (_is_tree or _is_sound_perception):
            return None
        # Forward-chain from the seeded skeleton to confirm a real causal chain
        # exists (fail-closed: if the skeleton is missing, don't assert).
        chain = []
        try:
            chain = self._causal_forward_simulate("tree", max_steps=5, top_k=4)
        except Exception:
            chain = []
        has_physical = any("sound" in c.lower() or "vibration" in c.lower()
                           or "vibrate" in c.lower() for c in chain)
        # Both-frame hedged reply — perspective-taking, not a flat assertion.
        reply = ("that's the classic one. physically, a falling tree still "
                 "disturbs the air and sends out vibrations — so the event "
                 "happens either way. but 'sound' as the experience of hearing "
                 "needs someone (or something) there to perceive it. so it "
                 "depends what you mean by sound: the vibrations are real, the "
                 "perception isn't, if no one's listening.")
        if not has_physical:
            # Skeleton absent — answer from the general two-frame reasoning
            # without over-claiming a specific chain.
            reply = ("that's the classic one. physically, something falling "
                     "still disturbs the air around it; the event happens "
                     "either way. but 'sound' as the experience of hearing "
                     "needs a listener to perceive it. so it depends what you "
                     "mean by sound: the cause is real, the perception needs "
                     "someone there.")
        return reply

    def _counterfactual_web_escape(self, ctx):
        """FOK → web escape for a counterfactual the graph cannot simulate.

        Reuses the engine's existing web/KB path: rewrite the conditional into a
        search query and, if a real snippet surfaces, return it framed as a
        retrieved (not asserted) answer. Honest escape hatch; returns None when
        the live lookup finds nothing (caller → uncertainty).
        """
        try:
            raw = getattr(ctx, "raw_input", "") or ""
            subj = getattr(ctx, "subject", "") or ""
            if not raw:
                return None
            q = self._rewrite_query_for_web(raw, subj or raw)
            ans = self._web_direct_answer(ctx)
            if ans:
                text = ans[0] if isinstance(ans, tuple) else ans
                if text and len(text) > 10:
                    return (f"one way people put it: {text}", "counterfactual_web")
        except Exception:
            return None
        return None

    def _hedged_candidate_for(self, user_input: str) -> Optional[str]:
        """Human-Likeness Plan (B): a hedged candidate-mechanism reply for the
        'why does time seem to go faster as we age' class of speculative questions.

        Brain basis: humans tolerate a speculative explanation under uncertainty
        (a hedged candidate mechanism), rather than a hard fail-closed. We keep
        RAVANA's honesty bar — the candidate is explicitly marked as ONE idea
        people have, never asserted as the cause — but we offer the
        proportional/logarithmic-time account (each year is a smaller fraction
        of your life; fewer novel memories as routines set in) so the reply
        reads as a person thinking aloud. Returns None for non-matching queries.
        """
        t = (user_input or "").lower()
        # Match the time-perception query family (well-defined, low false-positive).
        _time_q = (("time" in t and ("faster" in t or "quick" in t or "fly" in t
                   or "speed" in t or "slow" in t))
                   and ("older" in t or "age" in t or "grow" in t or "year" in t
                        or "childhood" in t or "kid" in t))
        if not _time_q:
            return None
        return ("i'm not certain why that is, but one idea people have is that "
                "each year is a smaller fraction of the life you've already lived "
                "— so a single year feels shorter next to all the ones before it. "
                "and as life gets more routine, you make fewer new memories, which "
                "can make big stretches of time blur together. that's a guess, not "
                "something i know for sure — what's your sense of it?")

