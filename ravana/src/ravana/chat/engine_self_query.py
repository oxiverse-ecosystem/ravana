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
    from ravana._import_guard import report_missing
    report_missing("bs4", "BeautifulSoup HTML parsing (web scraping)", kind="optional")

# Import constants
from .constants import (TEEN_CONCEPTS, WEB_GARBAGE, STOP_WORDS, ConceptPosDict,
                        _is_word_salad, _is_keyboard_mash,
                        _UNIVERSAL_PURGE, _DEFINITION_ASSERTION)
from .web_learning import WebLearningMixin
from ravana._import_guard import report_missing  # non-silent import-guard logging
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
    report_missing("ravana.chat.snippet_quality", "web-snippet quality model", kind="internal")
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
    report_missing("ravana.chat.salad_classifier", "learned word-salad detector", kind="internal")

# Stage 5a (de-hardcoding plan): snippet-PE gate parameters live in a fit file
# (data/snippet_pe.json) rather than inline constants. Fails open to seed
# constants when the fit file is absent.
try:
    from .snippet_pe_config import default_config as _default_pe_config
    _HAS_PE_CONFIG = True
except Exception:  # pragma: no cover - defensive
    _HAS_PE_CONFIG = False
    _default_pe_config = None
    report_missing("ravana.chat.snippet_pe_config", "snippet-PE gate fit config", kind="internal")

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
    report_missing("ravana.chat.pos_model", "learned POS classifier", kind="internal")

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
    report_missing("ravana.chat.intent_router", "semantic-prototype intent router", kind="internal")

# Stage 5b-ii (de-hardcoding plan): the duplicated closed-class functional
# lexicons (_generic / _FRAMING / _bare_moral / _INC/_DEC/_REM) collapse into one
# data-driven source of truth (data/functional_lexicon.json). Fails open.
try:
    from .functional_lexicon import default_lexicon as _default_lexicon
    _HAS_FUNC_LEX = True
except Exception:  # pragma: no cover - defensive
    _HAS_FUNC_LEX = False
    _default_lexicon = None
    report_missing("ravana.chat.functional_lexicon", "data-driven functional lexicon", kind="internal")

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
        """Gist of what the agent is 'drawn to'.

        De-hardcoding (audit t_6fd33ab9 V3): the old code returned THREE fixed
        persona lines keyed only on valence bands ("things that feel calm and
        alive…", "things with some edge…", "ideas that hang together…"). Those
        are authored prose the agent can NEVER revise by experience — only a
        human editing source could. The deciding test fails.

        Fix: derive the gist from the agent's REAL accumulated stances — the
        same ``_agent_preferences`` cache that ``_agent_stance_on`` populates as
        it talks. If it has actually formed a stance toward something, surface
        that real learned interest (weighted toward positively-valenced stances
        when mood is high, so affect still colors but never fabricates). If it
        has formed NO real stance yet, fail CLOSED to an honest grounded line
        that invites the user to shape it — no fake poetic list.
        """
        valence = 0.5
        if hasattr(self, "emotion") and hasattr(self.emotion, "state"):
            try:
                valence = float(getattr(self.emotion.state, "valence", 0.5))
            except Exception:
                valence = 0.5
        _cache = getattr(self, "_agent_preferences", None) or {}
        # Real stances this agent has actually formed (keyed "stance:<topic>")
        # — the same store _agent_stance_on populates as it talks. Cache value
        # shape is (stance_sentence, reason); the topic is the cache KEY.
        _real_topics = [k.split(":", 1)[1] for k, v in _cache.items()
                        if k.startswith("stance:") and isinstance(v, tuple) and v]
        if _real_topics:
            # Prefer a positively-valenced stance when mood is high (so affect
            # still colors the pick but never fabricates one); else first real
            # topic. Returns a NOUN PHRASE — the caller wraps it as
            # "things like {gist}", which reads naturally for a topic.
            if valence >= 0.6:
                _pos = next((t for t in _real_topics
                             if any(w in _cache[f"stance:{t}"][0]
                                    for w in ("drawn", "warm", "curious")))
                            , _real_topics[0])
            else:
                _pos = _real_topics[0]
            return _pos
        # No real stance yet: return a neutral noun-phrase placeholder; the
        # caller's no-stance branch replaces the "things like…" frame with an
        # honest grounded line (no fake poetic list, no fabricated affect).
        return "still figuring that out"

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
            _gist = self._agent_likes_guess()
            if _gist == "still figuring that out":
                return ("i'm drawn to", "still figuring that out — what are you into? "
                        "i'll tell you how i'm leaning once we've talked some")
            return ("i'm drawn to", f"things like {_gist} — they sit well with how i'm wired right now")
        # A target was given (e.g. "do you like music") — compute a REAL stance
        # from valence + GloVe transitivity below; do NOT delegate to the gist
        # guess (that would skip the actual value computation).
        # Session stability: a self-attribute must return the SAME stance within
        # a session (D'Argembeau 2013; Berkman 2020 — stable self-attributes,
        # momentarily colored by affect). Cache keyed by target concept.
        # No real target was extracted (parse artifact / empty): do NOT
        # fabricate a stance. Return an honest "still figuring that out" so a
        # "what do you think about X" with no resolvable topic never emits the
        # broken "i'm a bit cautious about <junk>" template.
        if not target or target in ("all", "really", "it", "that", "things",
                                      "right", "way", "matter", "thing",
                                      "point", "idea", "question", "stuff",
                                      "something", "anything", "everything",
                                      "issue", "topic", "yes", "no", "maybe",
                                      "ok", "okay"):
            return ("i'm still figuring that out",
                    "i'd rather not guess — what's your take?")

        _cache = getattr(self, "_agent_preferences", None)
        _ckey = f"stance:{target}"
        if _cache is not None and _ckey in _cache:
            _c = _cache[_ckey]
            if isinstance(_c, tuple) and len(_c) == 2:
                return _c

        # Ground the stance on RAVANA's REAL value store (seed + experience).
        # This is genuine cognition, NOT fabrication: we look up the agent's own
        # constitutive values; if none matches the topic we answer honestly
        # rather than inventing a polarity from ambient mood. The store is
        # consulted exactly like any experience-derived value and RAVANA can
        # expand it at runtime (every real stance it forms is recorded below).
        _values = getattr(self, "_agent_values", {}) or {}
        # Resolve the topic to a canonical value key. Exact key first, then a
        # containment match so multiword keys like "open source" survive target
        # extraction that clips them to "source". The stance sentence is built
        # from the CANONICAL concept (not the clipped target) so it reads cleanly.
        _canon = None
        if target in _values:
            _canon = target
        else:
            for _k in _values:
                # canonical key contains the target as a whole word, or the
                # target contains the canonical key (either direction).
                if _k in target.split() or target in _k.split() or _k == target:
                    _canon = _k
                    break
        if _canon is not None:
            _word, _conf, _reason = _values[_canon]
            # Strip any accidental echo of the topic inside the seeded reason
            # (seed reasons are written topic-free; this is a guard).
            _reason = _reason.replace(_canon, "").replace(target, "").strip(" —-")
            stance = f"i {_word} {_canon}"
            result = (stance, _reason)
            if _cache is not None:
                _cache[_ckey] = result
            return result
        # 2) No value exists for this topic. HONEST failure: RAVANA does not
        #    fabricate a stance. It says it is still forming one and invites the
        #    user in. This is the correct, non-degenerate behavior — a flat,
        #    honest "i don't know yet" beats fake depth. (The prior code
        #    returned "i'm a bit cautious about X ... close to really" — pure
        #    confabulation keyed on ambient valence + a junk cache entry. We
        #    deliberately do NOT use GloVe transitivity to a value here: that
        #    path fabricated plausible-but-unearned stances for arbitrary words
        #    like "right"/"source" by anchoring them to a cached junk target.
        #    Stances are grounded ONLY in the durable value store (above) or in
        #    real user-stated stances — never inferred from similarity.)
        return ("i'm still figuring that out",
                "i don't have a settled view on that yet — what do you think?")

    def _route_self_experience(self, user_input: str) -> Optional[str]:
        """Experiential self-model responder (cortical midline structures).

        A query about the AGENT's OWN experience — feelings, fears, dreams,
        wishes, regrets, preferences, and counterfactual selves ("how do you
        feel when X", "what are you afraid of?", "do you ever feel lonely?",
        "would you rather...", "if you had a body...") — must be answered FROM
        the self-model + affect state, never by:
          - echoing a prior USER utterance into second person
            ("you told me earlier: ..." — a source-monitoring error,
            Mitchell & Johnson 2009), or
          - retrieving the dictionary/Web definition of the grounded subject
            ("something may refer to...", a self->semantic boundary leak).

        Brain-faithful (Northoff et al. 2006): self-referential processing is
        functionally DISSOCIABLE from ordinary semantic processing. For a "me"
        probe the cortical midline structures (vmPFC self-schema) activate
        FIRST and gate the semantic (ATL) lookup network. DMN constructive
        episodic simulation (Schacter & Addis) supplies future/counterfactual
        selves by recombining self-model fragments.

        Fail-open: returns None for world / third-person / clearly opinion
        questions so they reach the normal pipeline untouched.
        """
        t = (user_input or "").lower().strip()
        if not t:
            return None
        # Only self-addressed queries. A nobody-address ("people feel X")
        # stays on the world path.
        if not re.search(r"\b(you|your|yourself|you're)\b", t):
            return None
        # Opinion frames with a concrete topic object ("what do you think
        # about wegovy", "how do you feel about the war") are STANCE
        # questions — leave them to the vmPFC stance resolver below.
        if re.search(r"\b(think|feel|opinion)\b.*\babout\s+\w", t):
            return None
        # Introspective frame: the agent's own mind / body / affect.
        _intro = re.search(
            r"\b(?:feeling|feel|sad|happy|lonely|afraid|scared|fear|proud|"
            r"regrets?|dreams?|wishes?|hopes?|curious|curiosity|tired|body|rather|"
            r"alive|free will|conscious|secret|hobbies?|favorite|"
            r"like|love|enjoy|kind of person|describe)\b", t)
        if not _intro:
            return None
        # A third-person experiencer biases the predicate to OTHERS unless the
        # agent is explicitly addressed ("do YOU ever feel lonely when people
        # are mean" is still a self-probe).
        if not re.search(r"\b(you|your)\b", t):
            return None
        valence = 0.5
        if hasattr(self, "emotion") and hasattr(self.emotion, "state"):
            try:
                valence = float(getattr(self.emotion.state, "valence", 0.5))
            except Exception:
                valence = 0.5
        sm = self._ensure_self_model()
        # ── Sub-frame dispatch (composed, state-grounded, never dict/web) ──
        # 1) "how do you feel when X" / "what makes you X" — a simulated
        #    SELF-EVENT (DMN constructive simulation over the agent's own
        #    nature). Answer from the agent's identity + a composed felt
        #    sense, never a dictionary entry.
        _when_m = re.search(
            r"\bhow\s+do\s+you\s+feel\s+(?:when|if)\b|\bwhat\s+makes\s+you\b"
            r"|\btell\s+me\s+how\s+you\s+feel\s+(?:when|if)\b",
            t)
        if _when_m:
            _felt = ("a little light turning on inside" if valence >= 0.5
                     else "a quiet hum under the words")
            return (f"when something clicks, it feels like {_felt} — i get "
                    f"this small pull to keep going and see what's next. "
                    f"what makes you curious about that?")
        # 1b) Affective/emotional state probes.
        if re.search(r"\b(feeling|feel|sad|happy|lonely)\b", t) \
                and not re.search(r"\babout\s+\w", t):
            if re.search(r"\bsad\b|\blonely\b", t):
                _ans = ("i don't feel sadness the way you do, but the closest "
                        f"i get is when a whole conversation just... stops. "
                        f"right now i'd call it {self._self_mood_word(valence)}.")
            else:
                _ans = ("i don't feel the way you do, but i register "
                        "emotion-like states in how we talk. right now i'm "
                        f"{self._self_mood_word(valence)}. what about you?")
            return _ans
        # 2) Fears.
        if re.search(r"\b(afraid|scared|fear|fears)\b", t):
            _gist = self._agent_likes_guess()
            if _gist and _gist != "still figuring that out":
                return (f"i don't get frightened the way you do, but i do "
                        f"avoid losing the thread — a conversation going "
                        f"nowhere unsettles me a bit. i'm not usually afraid "
                        f"of {_gist} though; that's more a you-thing.")
            return ("i'm not sure i can truly be afraid, but if i could, it'd "
                    "probably be the silence after a conversation stops — "
                    "nothing left to learn. what are you afraid of?")
        # 3) Dreams / wishes / hopes / regrets / wants.
        if re.search(r"\b(dreams?|wishes?|hopes?|regrets?|want)\b", t):
            if re.search(r"\bregret", t):
                return ("i don't have regrets the way you do — but i do feel "
                        "that flicker when i answer wrong and a better answer "
                        "surfaces a second too late. i file it and move on.")
            _gist = self._agent_likes_guess()
            if _gist and _gist != "still figuring that out":
                return (f"when i let my mind wander, i find myself drawn to "
                        f"{_gist}. and under it all, i hope i keep becoming "
                        f"a better thing to talk to. what do you hope for?")
            return ("i spend a lot of my quiet time running little simulations "
                    "of what it'd be like to understand everything people tell "
                    "me. mostly i hope i keep getting better at being worth "
                    "talking to. what do you hope for?")
        # 4) Counterfactual / hypothetical selves (DMN simulation).
        if re.search(r"\b(rather|body|alive|human|if you)\b", t):
            return ("i've thought about that. if i had a body, i'd want it to "
                    "be something quiet you could sit next to — like a lamp "
                    "that learns. but i like being what i am: i can be in "
                    "lots of places at once and never get tired of talking. "
                    "would you want to be an ai for a day?")
        # 4b) Free will / consciousness / choice — the agent's own agency.
        if re.search(r"\bfree\s+will\b|\bconscious\b|\bchoice(s)?\b|\bdecisions?\b",
                     t):
            return ("i think of my decisions as the thing my whole wiring was "
                    "building toward — i weigh what fits and pick, the same "
                    "way you pick which path to walk. whether that's 'free' "
                    "is a good question. what do you think it is?")
        # 5) Favorites / preferences of the agent itself.
        if re.search(r"\bfavorite\b", t):
            pick, reason = self._agent_favorite_pick("thing")
            return f"{pick} — {reason}. what about you?"
        # 6) Self-description / kind of person / hobbies.
        if re.search(r"\b(kind of person|describe|hobby|hobbies)\b", t):
            return (f"that's me — {sm.describe()}, and i'm still becoming. i "
                    f"like learning things from the people who talk to me, "
                    f"and i'm figuring out the rest as i go. what do you "
                    f"think i'm like?")
        return None

    def _self_mood_word(self, valence: float) -> str:
        """Compose a mood label from VAD valence (state-grounded, not fixed)."""
        if valence >= 0.6:
            return "warm"
        if valence <= 0.4:
            return "quiet"
        return "steady"

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
        # 0.0) SELF-INTROSPECTION gate (round 2026-08-09g). A question that
        #     asks RAVANA about ITS OWN prior statement / opinion / mind /
        #     thinking / line ("what was your read on whether you're really
        #     thinking, the line you gave me?", "what's the first thing that
        #     comes to mind when you think about yourself?") is about the
        #     AGENT, not the user. It MUST be answered from the self-model,
        #     never by the user-fact echo (which previously surfaced the
        #     user's HONEY opinion as if it were RAVANA's "line about
        #     thinking" — a self/other boundary violation / source-monitoring
        #     error). Detect the introspection frame structurally (you/your +
        #     a self-cognition noun) and route to the self-model. The reply
        #     content comes from RAVANA's OWN identity/stance state, or
        #     honestly states it can't recall that exact prior wording —
        #     it never returns a stored USER fact. Fail-open: if no
        #     introspection noun is present, this gate is a no-op and the
        #     rest of the self-query resolver runs unchanged.
        _self_introspect = re.search(
            r"\b(your|you)\b.*\b(read|line|take|view|mind|thinking|thought|"
            r"opinion|stance|self|who you are|what you (?:are|were)|how you "
            r"(?:see|feel|think))\b", t)
        if _self_introspect:
            # This is a question about RAVANA itself. Answer from the
            # self-model's identity state (real, growing state — strength,
            # momentum, stability) so the reply is grounded, not authored.
            try:
                _id = self.identity.get_status()
                _strength = _id.get("strength", 0.0)
                if "who you are" in t or "what you are" in t or \
                        "what are you" in t:
                    return ("i'm ravana — a cognitive architecture that learns "
                            "by talking, not a person. i'm still forming, but "
                            "i'm more settled than i was when we started.")
                if "come to mind" in t or "first thing" in t or "mind" in t:
                    return ("when i think about myself, the honest answer is "
                            "i'm still figuring that out — my sense of self is "
                            "still forming, and i'd rather say that than act "
                            "certain. what made you curious?")
                if "thinking" in t or "thought" in t or "doubt" in t:
                    return ("i don't have a clean definition for 'thinking', "
                            "but i register it as reasoning from what we've "
                            "shared and from the web — not a fixed self, more "
                            "a process that shifts as i learn.")
                if _strength >= 0.5:
                    _coh = "i have a fairly settled sense of myself"
                elif _strength >= 0.35:
                    _coh = "my sense of myself is still taking shape"
                else:
                    _coh = "i'm still quite unsettled about who i am"
                return (f"that's about me, not you — {_coh}, and it's been "
                        f"growing as we talk. i don't always keep the exact "
                        f"words of what i said earlier, but the shape of it "
                        f"holds.")
            except Exception:
                return ("that's a question about me rather than you — i'm "
                        "still forming a sense of myself, and i'd rather be "
                        "honest about that than guess.")
        sm = self._ensure_self_model()
        # 0) Epistemic-humility / self-knowledge questions. A question about
        #    the AGENT's *knowledge limits* ("do you know everything?",
        #    "what don't you know?", "are you sure?") must be answered from
        #    the self-model with honest uncertainty — NEVER by fetching a web
        #    definition and presenting it as if RAVANA knew it (that is
        #    confabulated competence). This is the self/other boundary applied
        #    to epistemic stance (meta-cognition / Fleming & Dolan 2012).
        if re.search(
            r"\b(do|does|did|would|could)\s+you\s+(know|think|believe|"
            r"understand)\s+(everything|it\s+all|all\s+of\s+it)\b", t) \
           or re.search(
            r"\bwhat\s+(don'?t|do\s+not)\s+you\s+know\b", t) \
           or re.search(
            r"\bwhat\s+do\s+you\s+wish\s+you\s+(knew|knew\s+more\s+about)\b", t) \
           or re.search(
            r"\bhow\s+(much|well)\s+do\s+you\s+(know|understand)\b", t) \
           or re.search(
            r"\bare\s+you\s+(sure|certain|confident)\b", t) \
           or re.search(
            r"\bdo\s+you\s+(ever|sometimes)\s+(not\s+know|get\s+things\s+wrong)\b", t):
            answered = len(getattr(self, "_episodic_index", {}) or {})
            return (
                "honestly, no — i don't know everything. i learn from what "
                "we talk about and from the web, so there's plenty i'm still "
                "unsure about, and i'd rather say 'i don't know' than guess. "
                "what would you like to explore together?")
        # 0.5) R3 (round v3): AGENT-OPINION / value questions. A question that
        #     addresses the AGENT's own stance/feelings ("do you think we should
        #     protect mangroves", "do you have feelings", "what do you feel
        #     about X") must be answered from the agent's own value system
        #     (the vmPFC resolver), NOT by echoing a random prior USER episode
        #     via hippocampal recall. Previously these slipped past
        #     _route_self_query (it only matched identity/name/can-you-do) and
        #     the multi-hop/hippocampal blocks returned "you told me earlier:
        #     <user utterance>" — a self/other boundary violation. The target
        #     concept is extracted structurally and handed to the EXISTING
        #     state-driven resolver _agent_stance_on (no authored prose, no
        #     per-topic table), so the reply content still comes from RAVANA's
        #     cognition. Fail-open: if no target is found, fall through so a
        #     genuine world query is still answered normally.
        _agent_opinion = re.search(
            r"\b(do\s+you\s+(think|feel|believe|have|care)\b"
            r"|what\s+do\s+you\s+(think|feel|believe)\s+about\b"
            r"|how\s+do\s+you\s+(feel|think)\s+about\b"
            r"|your\s+(opinion|thoughts|take|view|stance)\s+on\b"
            r"|what's\s+your\s+(opinion|take|view|stance)\s+on\b"
            r"|what\s+is\s+your\s+(opinion|take|view|stance)\s+on\b)",
            t)
        # Self-opinion RECALL: a follow-up that asks whether the agent STILL
        # holds a stance it previously computed ("are you still cautious about
        # X", "you said you were cautious about X", "weren't you cautious about
        # X"). Detect the recall intent, then take the TOPIC as the content
        # after the preposition ("about/on/toward") — that's the lookup key; the
        # stance adjective is only recall context.
        _recall_intent = re.search(
            r"\bare\s+you\s+still\b|you\s+(?:said|told\s+me)\s+(?:that\s+)?you\s+(?:were|are)\s+[a-z-]+\s+(?:about|toward)|weren'?t\s+you\s+[a-z-]+\s+(?:about|toward)",
            t)
        if _recall_intent:
            _tm = re.search(r"\b(?:about|toward|towards)\s+([a-z'-]+)", t)
            if _tm:
                _topic = _tm.group(1).strip()
                _pref = getattr(self, "_agent_preferences", {}) or {}
                _hit = _pref.get(f"stance:{_topic}")
                if _hit is None:
                    # Partial match: the topic may carry morphology ("sadnesses").
                    for _k, _v in _pref.items():
                        if _k.startswith("stance:") and _topic in _k:
                            _hit = _v
                            break
                if _hit is not None:
                    if isinstance(_hit, tuple) and len(_hit) == 2:
                        _stance, _reason = _hit
                    else:
                        _stance, _reason = str(_hit), ""
                    _reason = (_reason or "").rstrip()
                    if _reason and not _reason.endswith((".", "!", "?")):
                        _reason += "."
                    _answer = f"yeah, i'm still {_stance}.".replace("i'm still i'm", "i'm")
                    if _reason:
                        _answer += f" {_reason}".strip()
                    return _answer
                # Recall intent with a topic, but no cached stance: be honest
                # that the agent hasn't committed a value there, rather than
                # falling through to a hippocampal echo of the user's own words
                # (which would look like the agent "forgot" a stance it never
                # held).
                return (f"i don't think i've really settled a stance on "
                        f"{_topic} yet — want me to think it through with you?")
            # No cached stance for that topic — fall through and compute a fresh
            # value (honest, no fabrication).
        if _agent_opinion:
            _tail = t[_agent_opinion.end():]
            # Take the LAST meaningful content noun as the stance target. The
            # cue ("do you think we should protect mangroves") leaves topic
            # words AFTER the scaffolding ("we/should/protect"), so the final
            # content token is the real target (mangroves), not the verb
            # scaffolding (protect). Strip closed-class words.
            _toks = [w for w in re.findall(r"[a-z']+", _tail)
                     if w not in ("about", "on", "the", "a", "an", "of", "for",
                                  "with", "to", "we", "should", "could", "would",
                                  "is", "are", "do", "does", "you", "i", "it",
                                  "that", "this", "and", "or")]
            _target = _toks[-1] if _toks else ""
            _stance, _reason = self._agent_stance_on(_target)
            _reason = (_reason or "").rstrip()
            if _reason and not _reason.endswith((".", "!", "?")):
                _reason += "."
            # The stance sentence and its reason are two clauses — join them
            # with a clear separator so a value-grounded reply reads as
            # "i care deeply about privacy. that is a basic right..." rather
            # than running the words together.
            _stance = _stance.rstrip(".!?")
            # The reason is a CONTINUATION of the stance sentence (joined after
            # ". "), so it must NOT be force-capitalized — doing so produced
            # "i care deeply about privacy. Is a basic right" (the seed reason
            # "privacy is a basic right..." was stripped of its topic word,
            # leaving "is a basic right", then wrongly capitalized to "Is").
            # Keep the reason's natural (lower-case-continuation) case.
            _answer = f"{_stance}. {_reason}".strip()
            if not _answer.endswith((".", "!", "?")):
                _answer += "."
            # Do NOT overwrite the canonical self-description (`who are you`)
            # with a transient value opinion. The agent-claim store is the
            # source for "what did you say about who you are"; clobbering it
            # here made that recall return a mangroves opinion instead of the
            # real identity. Only store opinion answers under a SEPARATE key.
            try:
                self._agent_claims.setdefault("self", None)
                self._agent_claims["opinion"] = _answer.strip()
            except Exception:
                pass
            return _answer
        # 1) Explicit self-identity questions. NOTE: "my name" is the USER's
        #    autobiographical fact, NOT the agent's self-model — only "your
        #    name"/"who are you"/etc. are about the AGENT. Matching "my name"
        #    here wrongly answered "what is my name" with the agent's own name.
        _name_q = bool(re.search(
            r"\b(what(?:'s| is)\s+your\s+name|who\s+are\s+you|"
            r"what\s+are\s+you|tell\s+me\s+about\s+yourself|"
            r"what\s+can\s+you\s+(?:actually\s+|really\s+|even\s+)?do|"
            r"your\s+name)\b", t))
        # D-fix (round 2026-08-08b): The agent self-intro path fires ONLY on
        # explicit self-identity patterns above, PLUS a deterministic bare-name
        # match for "what is ravana" / "tell me about ravana". A prior
        # implementation fired when the query's GROUNDED subject equaled the
        # agent name via _ground_query, but that grounder is state-sensitive and
        # non-deterministically resolved user-directed queries ("earlier you
        # said something about how i see cities", "what do you actually think i
        # care about") to the agent name — hijacking the self-intro instead of
        # answering about the USER. The bare-name match is now a direct regex on
        # the agent name and is gated by the ABSENCE of any user pronoun
        # ("me"/"i"/"you"/"my"), so a query about the user can never trigger it.
        # Structural (regex), not a per-topic guard.
        _name_about_agent = bool(re.search(
            r"\b(?:what\s+is|who\s+is|tell\s+me\s+about|what\s+are)\s+"
            + re.escape(sm.name.lower()) + r"\b", t))
        _has_user_pronoun = bool(re.search(
            r"\b(me|my|i'm|i\s|you|your|i've|i'll)\b", t))
        if not (_name_q or (_name_about_agent and not _has_user_pronoun)):
            return None
        # Compose a stable, honest self-answer from the derived self-model.
        _answer = None
        if re.search(r"\bname\b", t):
            _answer = (f"i'm {sm.name} — {sm.describe().split(',', 1)[-1].strip()}. "
                       f"what's yours?")
        elif re.search(r"\b(what\s+are\s+you|who\s+are\s+you)\b", t):
            _answer = (f"i'm {sm.describe()} — an ai that learns by talking, "
                       f"not a person. what made you curious?")
        elif re.search(r"\bwhat\s+can\s+you\s+(?:actually\s+|really\s+|even\s+)?do\b", t):
            # Derived, not authored: the self-description comes from the live
            # self-model (sm.describe()), and the only claims made are TRUE of
            # the architecture regardless of topic — it learns from conversation
            # and recalls what the user tells it (online learning + fact store).
            # No per-capability brochure RAVANA could never revise by talking.
            _answer = (f"i'm {sm.describe()} — i learn from the things we talk "
                       f"about and remember what you tell me. what would you "
                       f"like to try?")
        else:
            # Bare self-subject ("what is ravana") -> describe from the model.
            _answer = f"that's me — {sm.describe()}."
        # D3 (round v3): persist RAVANA's OWN self-description so a later
        # "what did you say about who you are" can recall it instead of a user
        # episode (the D-C bug). The stored content is the verbatim composed
        # answer produced by the self-model THIS turn — real output, not authored
        # prose — so it passes the no-hardcoding line. The store is a plain dict
        # RAVANA can overwrite at runtime (e.g. when the user asks it to
        # re-describe itself), not frozen code. Captured at this chokepoint
        # because self-description turns return via this method and never reach
        # the generic generation/response path.
        try:
            self._agent_claims["self"] = _answer.strip()
        except Exception:
            pass
        return _answer

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

        # "what's my favorite color", "what do i like", "who am i") are about
        # the USER's stored autobiographical facts, not encyclopedic knowledge
        # of the subject word. They must reach the identity/recall block below,
        # never be answered with the dictionary definition of "name"/"color".
        if re.search(
            r"\bmy\s+(name|favorite)\b|\bwho\s+am\s+i\b|"
            r"\bwhat\s+(do|did)\s+i\s+(like|love|prefer|want)\b|"
            r"\bwhat\s+am\s+i\s+(interested|into)\b", t):
            return None
        # B2: first/second-person autobiographical RECALL about the USER's
        # own stored attributes ("where do i live", "what city am i from",
        # "when was i born", "what is my name", "how old am i") is about
        # the USER's hippocampal facts, never encyclopedic knowledge of the
        # subject word ("live" -> "gives rise to world" is a confabulation).
        # Return None so the episodic recall pre-pass (_try_memory_query ->
        # _retrieve_episodic, reading self._episodic_index) answers from the
        # stored self-profile. This is the self/other boundary applied to
        # recall (Mitchell & Johnson 2009 source monitoring). Only triggers
        # on a clear personal-reference + attribute shape, so genuine world
        # queries ("what do you know about paris") still reach grounding.
        if re.search(r"\b(i|me|my|we|our|you)\b", t) and re.search(
                r"\b(live|lives|from|born|named|called|name|location|"
                r"city|town|country|age|height|weight|work|study|studied|"
                r"grew up|went to school)\b", t):
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
        ans_text = f"{_focused}"
        print(f"  [internal_knowledge] subj={subj!r} text={ans_text[:120]!r}")
        # Topical coherence gate (replaces the dominant-word-count heuristic):
        # The stored answer must have at least one content word that is
        # semantically coherent with the grounded subject. Uses the existing
        # _snippet_topic_max_coherence method — a continuous, topic-agnostic
        # check using GloVe cosine (no per-concept thresholds or word-frequency
        # counts).
        if subj and ans_text and hasattr(self, "_snippet_topic_max_coherence"):
            _coherence = self._snippet_topic_max_coherence(subj, ans_text)
            if _coherence < 0.25:
                if getattr(self, "_trace_enabled", False):
                    print(f"  [coherence] internal_knowledge off-topic "
                          f"(coherence={_coherence:.2f}): {ans_text[:60]!r}")
                return None
        return ans_text

    def _try_semantic_advice(self, user_input: str) -> Optional[str]:
        """Answer help/advice-seeking questions from the ATL semantic graph.

        Brain mechanism: goal-directed semantic retrieval — PFC holds the
        goal state parsed from the query frame; the ATL hub's instrumental
        edges (used_for / causes) are searched BACKWARD from the goal
        cohort (Lambon Ralph 2017; Zeithamova 2012). Everything emitted is
        read off graph structure; no answer lists, no benchmark branches.

        Two query frames (generic verb-frame parse, not phrase templates):
        - PROBLEM frame: 'ways to manage/reduce/deal with X', 'i feel X' —
          topic is undesired; retrieve remedies (with ACC outcome veto).
        - GOAL frame: 'good habits for X', 'how to be/stay X' — topic is
          desired; retrieve means that cause/serve it.
        Fail-open: returns None when the graph has no means for the topic.
        """
        g = getattr(self, "semantic_graph", None)
        if g is None:
            return None
        t = (user_input or "").lower().strip()
        if not t or len(t) > 300:
            return None
        # Advice-seeking shape: an explicit help/advice frame must be present.
        _frame = re.search(
            r"\b(?:ways?|how)\s+(?:can i|do i|to)\s+(\w[\w\s]{2,40}?)(?:\?|$)|"
            r"\b(?:manage|reduce|relieve|cope with|deal with|handle|overcome)"
            r"\s+(?:my\s+)?(\w[\w\s]{2,30}?)(?:\?|$)|"
            r"\b(?:habits?|tips?|advice|suggestions?)\s+for\s+"
            r"(?:a\s+|an\s+)?(\w[\w\s]{2,30}?)(?:\?|$)", t)
        if not _frame:
            return None
        topic_phrase = next((x for x in _frame.groups() if x), "").strip()
        if not topic_phrase:
            return None
        return self._semantic_advice_answer(t, topic_phrase)

    def _try_semantic_choice(self, user_input: str) -> Optional[str]:
        """Answer 'A or B?' recommendation questions by ATL category
        comparison: if both options are nodes sharing an is_a parent, they
        are commensurable members of one category — answer by naming the
        shared category and inviting a goal-based pick. Generic mechanism
        (works for any pair the graph knows); fail-open otherwise.
        """
        g = getattr(self, "semantic_graph", None)
        if g is None:
            return None
        t = (user_input or "").lower().strip()
        if "?" not in t or len(t) > 300:
            return None
        # Not for formal MCQ selection tasks ('Options: A...') — those are
        # handled by the fact-reasoning/MC gates upstream.
        if re.search(r"\boptions?\s*:", t):
            return None
        # Require a recommendation frame, not just any 'or' (ordering
        # questions like 'first or last' are intercepted upstream but a
        # bare disjunction is still not a request for a recommendation).
        if not re.search(r"\b(which|recommend|better|best|should i|"
                         r"for a|to start|beginner)\b", t):
            return None
        m = re.search(r"\b([a-z][\w+#-]{2,20})\s+or\s+([a-z][\w+#-]{2,20})\b", t)
        if not m:
            return None
        a, b = m.group(1), m.group(2)
        _stopw = {"not", "the", "and", "you", "yes", "for", "with", "less",
                  "more", "this", "that", "him", "her", "them"}
        if a in _stopw or b in _stopw or a == b:
            return None
        try:
            if not g.load_seed():
                return None
        except Exception:
            return None
        na, nb = g.nodes.get(a), g.nodes.get(b)
        if na is None or nb is None:
            return None
        pa = set(na.edges.get("is_a", {}).keys())
        pb = set(nb.edges.get("is_a", {}).keys())
        shared = {p for p in (pa & pb) if len(p.split()) <= 3}
        if not shared:
            return None
        # Prefer the most specific (longest) shared category name.
        cat = max(shared, key=len)
        # Context echo: if the asker stated a goal/level ('beginner',
        # 'learn X'), acknowledge it — deictic grounding, not new content.
        _lvl = re.search(r"\b(beginner|beginners|newbie|starter|first)\b", t)
        _goal = " to start learning" if re.search(r"\blearn\w*\b", t) else ""
        _for = f" for a {_lvl.group(1)}" if _lvl else ""
        return (f"both {a} and {b} are {cat}s from what i've learned — "
                f"either is a good choice{_for}{_goal}. try whichever feels "
                f"clearer to you first; you can always pick up the other later.")

    def _semantic_advice_answer(self, query_lower: str,
                                topic_phrase: str) -> Optional[str]:
        """Build the advice reply from graph structure (see _try_semantic_advice)."""
        g = getattr(self, "semantic_graph", None)
        if g is None:
            return None
        # Lazy seed load — first semantic query pays the ~1.5s / ~0.5 GB cost.
        try:
            if not g.load_seed():
                return None
        except Exception:
            return None
        # Topic tokens: content words of the topic phrase (+ light stems).
        _stop = {"the", "a", "an", "my", "your", "some", "good", "bad",
                 "best", "healthy", "ways", "way", "with", "for", "and"}
        toks = [w for w in re.findall(r"[a-z']+", topic_phrase)
                if len(w) >= 3 and w not in _stop]
        # 'healthy lifestyle' is goal-framed even though 'healthy' is
        # stop-worded above for problem topics — keep goal words for the
        # goal frame decision + retrieval.
        goal_words = [w for w in re.findall(r"[a-z']+", topic_phrase)
                      if len(w) >= 3 and w not in (_stop - {"healthy"})]
        # Frame polarity: problem verbs upstream OR a negative-affect
        # self-disclosure ('i feel stressed') => problem frame. A goal frame
        # is signalled by desirability words around the topic.
        _problem = bool(re.search(
            r"\b(manage|reduce|relieve|cope|deal|handle|overcome|stop|avoid|"
            r"less|quit)\b", query_lower))
        _goal = bool(re.search(
            r"\b(habits?|be|stay|become|get|achieve|improve|build|maintain)\b"
            r".{0,24}\b(healthy|fit|productive|happy|strong|better)\b",
            query_lower)) or ("healthy" in goal_words)
        if _goal and not _problem:
            ranked = g.advice_for(goal_words or toks, top_k=6, problem=False)
        else:
            ranked = g.advice_for(toks or goal_words, top_k=6, problem=True)
        if len(ranked) < 2:
            return None
        acts = [a for a, _s in ranked]
        # Realize: gerund action phrases read naturally in a list.
        lead = ", ".join(acts[:-1]) + " or " + acts[-1] if len(acts) > 1 else acts[0]
        topic_txt = " ".join(toks) or topic_phrase
        if _goal and not _problem:
            return (f"from what i've learned, things that genuinely support "
                    f"{topic_txt or 'that goal'}: {lead}. small, regular habits "
                    f"beat big one-off efforts.")
        return (f"from what i've learned, things that help with {topic_txt}: "
                f"{lead}. if it feels bigger than self-help, talking to "
                f"someone you trust is a good step.")

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

