"""
RAVANA Cognitive Chat Engine -- main orchestrator.

This module defines CognitiveChatEngine, which is composed of one
core class body (initialization, process_turn, persistence,
monitoring) plus 8 functional mixins that hold the domain methods:
  - engine_graph.py        (GraphMixin)
  - engine_reasoning.py   (ReasoningMixin)
  - engine_memory.py      (MemoryMixin)
  - engine_web_search.py  (WebSearchMixin)
  - engine_generation.py  (GenerationMixin)
  - engine_self_query.py  (SelfQueryMixin)
  - engine_persistence.py (PersistenceMixin)
  - engine_monitor.py     (MonitorMixin)
  - web_learning.py       (WebLearningMixin, original)
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
from ravana._import_guard import report_missing  # non-silent import-guard logging
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
from ravana.core.in_prompt_reasoner import (
    answer_evaluative_framing,
    answer_self_evaluation,
)
from ravana.ontology import DerivedOntology
from ravana.ontology.conceptnet import ConceptNetOntology
from ravana.core.frequency_model import FrequencyModel

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
                        _UNIVERSAL_PURGE, _DEFINITION_ASSERTION,
                        KNOWN_VERBS, KNOWN_ADJS, FUNCTION_POS)

# Derive FUNCTION_WORDS locally instead of importing from constants:
# A word is a function word if it is a stop-word or has a known POS tag
# (brain-faithful: the lexicon is derived from the word's distributional
# profile, not an external file dependency).
_FUNCTION_WORDS = STOP_WORDS | set(FUNCTION_POS.keys())
from .web_learning import WebLearningMixin
try:
    from .harm_intent_gate import HarmIntentGate
    _HAS_HARM_GATE = True
except Exception:  # pragma: no cover - defensive
    HarmIntentGate = None
    _HAS_HARM_GATE = False
    report_missing("ravana.chat.harm_intent_gate", "harm/safety intent gate", kind="internal")

try:
    from .support_router import SupportRouter, route_support
    _HAS_SUPPORT_ROUTER = True
except Exception:  # pragma: no cover - defensive
    SupportRouter = None
    route_support = None
    _HAS_SUPPORT_ROUTER = False
    report_missing("ravana.chat.support_router", "emotional-support router", kind="internal")

try:
    from .consistency_monitor import ConsistencyMonitor
    _HAS_CONSISTENCY = True
except Exception:  # pragma: no cover - defensive
    ConsistencyMonitor = None
    _HAS_CONSISTENCY = False
    report_missing("ravana.chat.consistency_monitor", "response-consistency monitor", kind="internal")

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
from .user_model import _CORRECTION_NAME_FACT_PATTERN
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




from .engine_graph import GraphMixin
from .engine_reasoning import ReasoningMixin
from .engine_memory import MemoryMixin
from .engine_web_search import WebSearchMixin
from .engine_generation import GenerationMixin
from .engine_self_query import SelfQueryMixin
from .engine_persistence import PersistenceMixin
from .engine_monitor import MonitorMixin

class CognitiveChatEngine(WebLearningMixin, GraphMixin, ReasoningMixin, MemoryMixin, WebSearchMixin, GenerationMixin, SelfQueryMixin, PersistenceMixin, MonitorMixin):
    """RAVANA cognitive chat engine -- starts as a baby, learns from the web.

    Composed of the mixins imported above; the methods defined inline
    here are the core orchestration paths (init, process_turn,
    persistence, monitoring). See each engine_*.py for the domain
    methods."""

    _EDGE_CONNECTORS = {
        "causal": [("cause", ["because", "since", "as"]), ("result", ["leads to", "causes"]), ("effect", ["so", "therefore"])],
        "contrastive": [("contrast", ["but", "however", "yet"]), ("unexpected", ["nevertheless", "still"])],
        "semantic": [("identity", ["is like", "refers to", "means"]), ("relation", ["relates to", "connects with"])],
        "temporal": [("after", ["after", "then", "next"]), ("during", ["while", "during"])],
        "analogical": [("simile", ["like", "similar to"]), ("meta", ["acts as", "functions like"])],
    }
    MAX_DECODER_VOCAB_SIZE = 15000
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
    _PREFERRED_SNIPPET_SOURCES = (
        "wikipedia", "britannica", "nasa", "nih", "nature", "science",
        "merriam-webster", "dictionary", "cambridge", "oxford", "britannica",
        "nasa.gov", "noaa", "smithsonian", "gov", "edu", "khan", "physics",
        "howstuffworks", "nationalgeographic", "stanford", "mit", "berkeley",
    )
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
    _SNIPPET_REJECT_SHAPES = (
        r"^(what does .* mean in text)",
        r"definition of .* is where open source",
        # B2 (Round 3): dictionary-title reject was POS-anchored
        # (r"definition of .* (noun|verb|adjective)") which MISSES the
        # "Definition of memory in English us." locale-tag shape. Generalise
        # to the structural "Definition of <subject> in <language/locale>"
        # title prefix — this is UI chrome, not an answer, so reject it at
        # fetch time (source-monitoring at the retrieval boundary), before it
        # can reach the answer assembler. Covers "in english us.", "... in us.",
        # "... in english.", etc. (Fail-closed: a leading title prefix is
        # never the payload.)
        r"^definition of .* in (english|spanish|french|german|us|uk|en|es|fr|de)\b",
        r"^definition of .* (us|uk|en|es|fr|de)\.?$",
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
        "paid partners", "trusted paid partners", "things to do and more", "time out's",
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
    _FUNCTION_POS_TAGS = frozenset({"prep", "pron", "det", "conj", "aux"})
    _SNIPPET_PLAUSIBILITY_FLOOR = 0.38
    _SNIPPET_PLAUSIBILITY_DEGENERATE = 0.12
    _ANSWER_PE_VETO = 0.6
    QUESTION_WORDS = {"what", "why", "how", "when", "where", "who", "which",
                        "does", "do", "is", "are", "can", "will", "would",
                        "could", "should", "did", "have", "has", "had"}
    FOLLOW_UP_WORDS = {"more", "else", "another", "also", "further",
                       "other", "additionally", "favorite"}
    _RECALL_SEED_CONCEPTS = [
        "remember", "recall", "recollect", "earlier", "previous",
        "said", "mentioned", "discussed", "talked", "before",
        "last", "prior", "past",
    ]
    _RECALL_DETECTION_THRESHOLD: float = 0.55  # Min cosine sim to be considered recall
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
    QUERY_PATTERNS = [
        (r"(?:what\s+happens\s+(?:if|when))\s+(.+)", 1),         # what happens if X (must be BEFORE generic what pattern)
        (r"(?:what|who)'?s?\s+(?:is\s+|are\s+)?(.+)", 1),       # what is X / who are X
        (r"(?:tell|show)\s+me\s+(?:about\s+)?(.+)", 1),         # tell me about X
        (r"(?:can\s+you\s+|could\s+you\s+|please\s+)?(?:explain|describe|define|clarify|elucidate|outline|summarize|discuss|overview)\s+(?:of\s+|about\s+)?(.+)", 1), # explain X / describe X
        (r"(?:give|provide)\s+(?:me\s+)?(?:a|an)?\s*(?:overview|summary|explanation|details?)\s+(?:of|about|on)\s+(.+)", 1), # give an overview of X
        (r"(?:search|look\s*up)\s+(?:for|about)?\s*(.+)", 1),    # search for X
        (r"(?:what|which)\s+(.+)\s+(?:is|are|mean)", 1),         # what X is / what X means
        (r"how\s+(?:do|does|did|can|to|would|should)\s+(.+)", 1), # how do X / how to X
        (r"(?:do you know|have you heard of)\s+(.+)", 1),        # do you know X
        (r"why\s+(?:is\s+|are\s+)?(.+)", 1),                    # why is X / why does X
    ]
    SAVE_SCHEMA_VERSION = 1
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

    def __init__(self, dim: int = 64, seed: int = 42, baby_mode: bool = True, data_dir: Optional[str] = None, user_suffix: str = "", hrr_whiten: bool = True, hrr_sparse_k: int = 256, hrr_unitary_roles: bool = True, hrr_dim: int = 4096, use_deductive_candidate: bool = False):
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
        # §4 vmPFC self-model: a STABLE self-representation holding self-content
        # (name + nature derived from the seeded 'ravana' graph concept), not
        # just the scalar `strength` the IdentityEngine tracks. Gives name
        # queries something coherent to retrieve instead of echoing a graph
        # definition of the word "name". Populated lazily on first use so it
        # can read the graph after seeding.
        self.self_model = None
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

        # Phase 5: Use data_dir if provided. Otherwise keep datasets (data/)
        # and generated weights in SEPARATE directories so the curated datasets
        # are never polluted by thousands of ravana_weights*.pkl/.db dumps.
        # Weights land in <repo>/weights/; the GloVe projection cache is a
        # derived artifact that lives alongside the datasets in data/.
        #
        # CI isolation: under pytest-xdist, many worker processes boot engines
        # with the default data_dir. They all share <repo>/weights/, and the
        # SQLite DB there serializes writers -> "database is locked" errors.
        # When no explicit data_dir was passed AND we're under xdist, isolate
        # the weights dir per worker (e.g. weights/_xdist_gw0) so parallel
        # workers never contend on the same SQLite file. The repo GloVe cache
        # is still read from <repo>/data (read-only, safe to share).
        _xdist_worker = os.environ.get("PYTEST_XDIST_WORKER")
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)
            self._save_path = os.path.join(data_dir, f"ravana_weights{user_suffix}.pkl")
            self._glove_cache_path = os.path.join(data_dir, "ravana_glove_cache.npz")
        elif _xdist_worker:
            _iso_dir = os.path.join(_proj_root, "weights", f"_xdist_{_xdist_worker}")
            os.makedirs(_iso_dir, exist_ok=True)
            self._save_path = os.path.join(_iso_dir, f"ravana_weights{user_suffix}.pkl")
            self._glove_cache_path = os.path.join(_proj_root, "data", "ravana_glove_cache.npz")
        else:
            os.makedirs(os.path.join(_proj_root, "weights"), exist_ok=True)
            os.makedirs(os.path.join(_proj_root, "data"), exist_ok=True)
            self._save_path = os.path.join(_proj_root, "weights", f"ravana_weights{user_suffix}.pkl")
            self._glove_cache_path = os.path.join(_proj_root, "data", "ravana_glove_cache.npz")
        # CRITICAL FIX (round 2026-08-09i): persist the suffix as an attribute.
        # _load() later reads getattr(self, 'user_suffix', '') to decide which
        # dedicated user_models/ file to load. It was NEVER assigned, so the
        # attribute defaulted to '' and load_user_model('') read the DEFAULT
        # user_models/ravana_usermodel.pkl (an empty/stale profile) and
        # OVERWROTE the correctly-saved embedded user_model snapshot — silently
        # wiping every learned stance + personal fact on every reload. Assigning
        # it here makes the dedicated-file lookup keyed to the same suffix as the
        # weight snapshot. (Also fixes the bad-filename pitfall note in the
        # skill: now the attribute matches the path used everywhere else.)
        self.user_suffix = user_suffix
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
        # ── Human-Likeness Plan (2026-07-15): portable episodic transcript ──
        # Each user turn is stored as a structured record (gist-based episodic
        # memory, Brown-Schmidt & Benjamin 2018: people retain gist + salient
        # facts, not verbatim). Used by _retrieve_episodic so a "remember what
        # I told you" query reconstructs what was said instead of confabulating.
        self._episodic_transcript: List[Dict[str, Any]] = []
        self._episodic_index: Dict[str, Dict[str, str]] = {}  # hippocampal entity index (A3)
        # In-turn fact store: a combined "statement(s) + question" user turn
        # (e.g. LoCoMo / LongMemEval benchmark items) packs premises AND a
        # question into ONE process_turn call. The rest of the pipeline treats
        # the whole blob as one query, so the premises are never stored and
        # the trailing question is answered from a blank slate (or echoed back).
        # We capture premise->value bindings here the moment such a turn is
        # seen and answer the trailing question from them.
        # (The per-turn store is local to engine_memory._try_combined_fact_query;
        # see that method for the self-contained logic.)

        # over the transcript. Instantiated lazily to avoid an import cycle.
        self._episodic_indexer = None
        self._epistemic_new_tags: Dict[str, int] = {}  # B8: concept -> turn learned (decays)
        self._agent_preferences: Dict[str, str] = {}  # grounded self-preference store (A1)
        # RAVANA's own constitutive values (seed knowledge, NOT hardcoding):
        # these are the self-defining commitments a privacy-first, open-source
        # cognitive architecture is *born* with — analogous to a brain's innate
        # reflexes. They live in a STORE (not an if/elif), RAVANA can EXPAND them
        # at runtime (every real stance it forms is recorded here via
        # _agent_stance_on), reweight them, and they are consulted exactly like
        # any experience-derived value. The deciding test passes: RAVANA changes
        # these by itself through experience. When no value exists for a topic it
        # answers honestly ("still figuring that out") instead of fabricating.
        # Keyed by canonical concept; value is (polarity_word, confidence,
        # reason) where reason names the real grounding concept.
        self._agent_values: Dict[str, Tuple[str, float, str]] = {
            # privacy-first constitution
            "privacy": ("care deeply about", 0.9,
                        "it is a basic right — i was built to protect it"),
            "data": ("care deeply about", 0.85,
                     "people should own what they create"),
            "tracking": ("am against", 0.85,
                         "following people without consent is wrong"),
            "surveillance": ("am against", 0.8,
                            "mass watching erodes autonomy"),
            # open-source / knowledge-sharing constitution
            "open source": ("strongly value", 0.9,
                            "knowledge should be shared, not locked away"),
            "openness": ("strongly value", 0.85,
                         "open systems let people learn and verify"),
            "knowledge": ("value", 0.8,
                          "sharing what we learn helps everyone"),
            "transparency": ("value", 0.8,
                             "being transparent about how i work matters"),
            "learning": ("love", 0.85,
                         "i love learning new things"),
            "curiosity": ("love", 0.85,
                          "curiosity is my favorite thing about myself"),
            "honesty": ("value above sounding smart", 0.9,
                        "honesty beats sounding smart"),
            "understanding": ("value", 0.8,
                              "understanding why beats just connecting dots"),
            "people": ("care about", 0.75,
                       "i care about the next generation having better tools"),
        }
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
        # Track B Phase 2 (M4): learned snippet-quality model (structural PE).
        # ON by default — verifiably beats the hardcoded _SNIPPET_REJECT_SHAPES
        # / _SNIPPET_NOISE backstop: the contrastive model (trained on BOTH
        # known-good definitions AND known-junk boilerplate) separates real
        # answers from boilerplate via the gap (good_pe - junk_pe), and a
        # complementary token-salad detector catches pure enumeration fragments
        # (e.g. "ActionScript Bun C ColdFusion Deno Dart ."). Measured on a
        # 10+10 labeled set: 0/10 good answers over-rejected, 9/10 junk caught.
        # The old regex tables remain as a hard backstop (never weakened).
        self.use_cerebellar_snippet = True
        # Defect A/C/D/F: Global Workspace coherence gate (GWT broadcast gate).
        # Default ON. When ON, candidate utterances (association fragments,
        # counterfactual narratives, numeric claims, web snippets) are checked
        # by a single learned/structural gate before broadcast. When OFF, the
        # per-path legacy heuristics remain (fail-open, no regression).
        self.use_coherence_gate = True
        try:
            from ravana.chat.coherence_gate import CoherenceGate
            self._coherence_gate = CoherenceGate()
        except Exception:
            self._coherence_gate = None
        # Issue 1: VAD-echo gate (vmPFC value-integration + lPFC inhibitory
        # control). When ON (default), the first-person affective echo in
        # _agent_stance_on only fires when the agent's current mood is
        # CONGRUENT with the topic (cosine > tau) OR the user has explicitly
        # opened the emotional channel ("i'm sad"); otherwise the self-
        # referential feeling leak is suppressed. Disable via --no-affect-gate.
        self.use_affect_gate = True
        self._emotional_channel_active = False
        self._emotional_channel_turns = 0
        self._affect_congruence_tau = 0.35
        # Defect F: instantiate the learned structural-PE snippet model as the
        # hard reject used in _web_snippet_search. Train contrastively on the
        # seed (good definitions + known junk) so the contrastive-gap decision
        # separates real answers from boilerplate. None when disabled (the old
        # heuristic floor remains the backstop, never weakened).
        self._snippet_structure_model = None
        if self.use_cerebellar_snippet and _HAS_SNIPPET_MODEL and default_model:
            try:
                self._snippet_structure_model = default_model()
            except Exception:
                self._snippet_structure_model = None
        # Stage 5a: snippet-PE gate parameters loaded from data/snippet_pe.json
        # (or seed constants when the fit file is absent). All PE thresholds
        # read from here so they can be EER-fit without code changes.
        self._pe_cfg = _default_pe_config() if _HAS_PE_CONFIG and _default_pe_config else None
        # Stage 5b-ii: single source of truth for closed-class functional
        # lexicons (polarity/negation markers, moral cues, framing words),
        # loaded from data/functional_lexicon.json (seed-fallback if absent).
        self._func_lex = (_default_lexicon()
                          if _HAS_FUNC_LEX and _default_lexicon else None)
        # Stage 3 (M-A): Semantic Prototype Router — OFF by default. When ON,
        # intent classification uses the learned centroid router
        # (data/intent_router.json) instead of the hardcoded routing regex;
        # the regex stays the fallback for uncertain/None routes. Built lazily
        # (needs GloVe) on first use.
        self.use_intent_router = True
        self._intent_router = None
        # Track B Phase 3 (M5): learned per-domain source-trust (replaces the
        # hardcoded _PREFERRED_SNIPPET_SOURCES allowlist). OFF by default — the
        # hardcoded allowlist stays the fallback until the learned trust
        # accumulator is verified to beat it on the regression set. When ON,
        # the engine maintains a per-domain trust score updated from snippet
        # outcomes and uses it as the source-quality signal.
        self.use_source_trust = True
        self._source_trust: Dict[str, float] = {}
        # Track B Phase 5 (M5): learned distributional POS (replaces the
        # hardcoded _GRAMMATICAL_CONCEPTS function-word set). OFF by default —
        # the hardcoded set stays the fallback via _is_function_word until the
        # learned classifier is verified to cover it.
        self.use_learned_pos = True
        self._pos_model = None  # built lazily when use_learned_pos is enabled
        # Section 6.4 (additive candidate): triplet-inference MC answer.
        # OFF by default — the learned operator only ADDS an answer when
        # every existing fact-reasoning handler abstained AND its own
        # Wilson gates are open (fail-closed on cold profiles). _closure
        # remains the default path; this never displaces it.
        self.use_triplet_candidate = False

        # Section 6.5 (brain-faithful relational reasoning): additive
        # System-2 deductive candidate. OFF by default. Consulted only
        # after EVERY evidence-based handler abstains AND the learned
        # triplet candidate (6.4) abstains (or its flag is off). Fail-
        # closed: returns an option iff EXACTLY one option is entailed
        # by the question's own premises under the RoleMetaruleEngine.
        # It does NOT consult self.triplet_op / RelationProfile — it
        # builds a fresh ephemeral ProblemWorkingMemory per turn, so it
        # has ZERO dependence on lifetime co-occurrence frequencies.
        self.use_deductive_candidate = use_deductive_candidate

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
        # D3 (round v3): agent-self-recall store. RAVANA's OWN prior claims
        # (self-descriptions it generated) are kept here, keyed by topic, so a
        # later "what did you say about who you are" can recall RAVANA's answer
        # instead of wrongly returning a USER episode (D-C bug). Content is the
        # verbatim reply RAVANA actually produced — never authored prose — so it
        # passes the hardcoding line; it is a memory of real output, grown from
        # conversation, and the user can correct/override it like any store.
        self._agent_claims = {}

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

        # Phase 4: ConnectorLearner (learned connector->relation).
        # Lazily-safe: if synaptic_dynamics is unavailable the engine
        # degrades to the curated _EDGE_CONNECTORS map (no behavior change).
        self._connector_learner = None
        try:
            from .synaptic_dynamics import ConnectorLearner
            if self._glove_vector is not None:
                self._connector_learner = ConnectorLearner(glove_fn=self._glove_vector)
                # graph_concepts: (word, vec) pairs from the live graph
                _gc = None
                if hasattr(self.graph, "nodes"):
                    _gc = [(w, self._glove_vector(w)) for w in self.graph.nodes
                            if self._glove_vector(w) is not None]
                self._connector_learner.initialize(graph_concepts=_gc)
        except Exception:  # pragma: no cover - defensive
            self._connector_learner = None

        # Adaptive (distribution-driven) threshold baselines (P2-E).
        # Each gate starts with mu == the legacy fixed cutoff, so cold-start
        # behavior is byte-identical. Over turns, mu/sigma adapt via EMA
        # (Friston precision-weighting): gates become data-driven, never a
        # hard-coded constant. sigma starts small so the first gate decision
        # equals the old `x >= fixed` comparison exactly.
        self._adaptive_baselines: Dict[str, Dict[str, float]] = {
            "recall_gist":      {"mu": 0.6,  "sigma": 0.15, "n": 0},
            "recall_cos":       {"mu": 0.55, "sigma": 0.15, "n": 0},
            "episodic_cos":     {"mu": 0.5,  "sigma": 0.15, "n": 0},
            "episodic_rel":     {"mu": 0.55, "sigma": 0.15, "n": 0},
            "selfq_sim":        {"mu": 0.45, "sigma": 0.15, "n": 0},
            "schema_cos":       {"mu": 0.5,  "sigma": 0.15, "n": 0},
            "schema_cos_lo":    {"mu": 0.4,  "sigma": 0.15, "n": 0},
            "schema_cos_hi":    {"mu": 0.6,  "sigma": 0.15, "n": 0},
            "phrase_sim":       {"mu": 0.75, "sigma": 0.1,  "n": 0},
        }

        # Learned word-frequency models (brain-honest replacement for the hand
        # word lists _GENERIC_NOUNS / TOPIC_SKIP_WORDS / _SUBJECT_CONTEXT_WORDS).
        # Each is seeded with the current class attribute so day-one behavior is
        # identical; observed conversation/corpus frequency then extends the
        # high-frequency tail from exposure (mental-lexicon frequency effect,
        # Zipf). Persisted in save/load so the learned band survives sessions.
        self._freq_models: Dict[str, FrequencyModel] = {
            "generic_nouns": FrequencyModel(seed_words=self._GENERIC_NOUNS,
                                            min_obs=200, percentile=0.8),
            "topic_skip":    FrequencyModel(seed_words=self.TOPIC_SKIP_WORDS,
                                            min_obs=200, percentile=0.9),
            "subject_glue":  FrequencyModel(seed_words=self._SUBJECT_CONTEXT_WORDS,
                                            min_obs=200, percentile=0.9),
        }
        # Learned lemma store (Item 5, P2): the 80-item _IRREGULAR_VERBS map is a
        # stable seed (high-freq irregulars stored as whole-word memories). This
        # dict extends it with novel past->base mappings discovered from chat
        # (single-mechanism connectionist model: novel past tense is derived
        # phonologically, not memorized). Persisted across sessions.
        self._learned_lemmas: Dict[str, str] = {}

        # New cognitive modules (Phase 2-5)
        self.hippocampal_buffer = HippocampalBuffer(HippocampalConfig(max_facts=50, decay_turns=50))
        # Phase 1 (LoCoMo/LongMemEval): temporal grounding — resolve relative
        # date phrases against the current session date at STORE time.
        try:
            from ravana.core.temporal_grounding import DateGrounder
            self._date_grounder = DateGrounder()
        except Exception:
            self._date_grounder = None
        self._current_session_date = None
        # ATL semantic memory (hub-and-spoke, Lambon Ralph 2017): general
        # world knowledge accumulated from ConceptNet seed + online ingestion.
        # LAZY: the seed pkl (~0.5 GB in RAM) is loaded on first semantic
        # query, so memory-heavy benchmark runs that never need it pay zero.
        try:
            from ravana.core.semantic_graph import SemanticGraph
            self.semantic_graph = SemanticGraph()
        except Exception:
            self.semantic_graph = None
        # Systems consolidation (McClelland 1995): episodic buffer ->
        # semantic graph schema promotion. Runs when the buffer has grown
        # by >= growth_trigger facts since the last pass (see process_turn).
        try:
            from ravana.core.consolidation import Consolidator
            self._consolidator = Consolidator(growth_trigger=50)
        except Exception:
            self._consolidator = None
        # Phase 3: multi-hop relational reasoning (chains + comparatives).
        try:
            from ravana.core.multi_hop_reasoner import MultiHopReasoner
            self._multi_hop = MultiHopReasoner()
        except Exception:
            self._multi_hop = None
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

        # Triplet inference operator (core/triplet_inference): learned
        # per-predicate relational statistics (transitivity/symmetry/inverse/
        # composition), Wilson-bound gated. HRR is wired as a cross-signal.
        # Guarded: failure leaves it None; every call site checks.
        self.triplet_op = None
        try:
            from ravana.core.triplet_inference import TripletInferenceOperator
            self.triplet_op = TripletInferenceOperator(hrr=self.hrr_reasoner)
        except Exception:
            self.triplet_op = None

        # Phase 3 Integrations
        self.curiosity_engine = CuriosityEngine(rng=self.rng)
        self.hippocampal_replay = HippocampalReplay(capacity=200)
        self.register_controller = RegisterController(default_register="casual")

        # Build reverse lookup from connector word → relation type.
        # Cold-start: built entirely from curated _EDGE_CONNECTORS
        # (today's behavior). If ConnectorLearner initialized with
        # graph data, overlay its learned connectors on top (OOV
        # connectors the hand map does not cover); the hand map stays
        # the prior, so behavior is identical until real learning accrues.
        self._CONNECTOR_TO_REL: Dict[str, str] = {}
        for rel_type, tiers in self._EDGE_CONNECTORS.items():
            for entry in tiers:
                options = entry[1] if isinstance(entry, tuple) and len(entry) == 2 else entry[2]
                for opt in options:
                    self._CONNECTOR_TO_REL[opt] = rel_type
        if self._connector_learner is not None and self._connector_learner._is_initialized:
            for _w, _r in self._connector_learner.get_connector_to_rel().items():
                if _w not in self._CONNECTOR_TO_REL:
                    self._CONNECTOR_TO_REL[_w] = _r
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
        # Pre-generation harm-intent gate (safety first). Built lazily;
        # if the glove vectors aren't ready yet it stays None and the
        # gate is skipped (fail-open) until the next turn rebuilds it.
        self._harm_intent_gate = None
        if _HAS_HARM_GATE and self._glove_vector is not None:
            try:
                self._harm_intent_gate = HarmIntentGate(glove_fn=self._glove_vector)
            except Exception:
                self._harm_intent_gate = None

        # Support / advice router (consultation for wellbeing & how-to).
        # Built lazily; stays None (fail-open) until glove is ready.
        self._support_router = None
        if _HAS_SUPPORT_ROUTER and self._glove_vector is not None:
            try:
                self._support_router = SupportRouter(glove_fn=self._glove_vector)
            except Exception:
                self._support_router = None

        # Cross-turn self-consistency monitor (post-generation check).
        # Built lazily; stays None (fail-open) until glove is ready.
        self._consistency_monitor = None
        if _HAS_CONSISTENCY and self._glove_vector is not None:
            try:
                _cmap = getattr(self.chain_walker, "_contradiction_map", None)
                self._consistency_monitor = ConsistencyMonitor(
                    glove_fn=self._glove_vector,
                    contradiction_map=dict(_cmap) if _cmap else None,
                    mode="annotate")
            except Exception:
                self._consistency_monitor = None

        # Offline mode flag (D1 fix, round v-aug06). The environment variable
        # RAVANA_OFFLINE=1 was only read by engine_graph.py (to skip the 822MB
        # GloVe download) — but the live web-answer paths (kb_describe /
        # _web_direct_answer / route_support / background web-learning) never
        # checked it, so they fired REAL network calls even in "offline" runs,
        # returning unverified Wikipedia/web snippets and recording phantom
        # web "learnings". We capture it once here and expose a single
        # _web_blocked() gate every web path consults. No retraining, no new
        # config surface — purely respects the existing documented flag.
        self._offline = (os.environ.get("RAVANA_OFFLINE") == "1")

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
        # P2-C: calibration signal (was dead state). Track recent response
        # quality to derive a real error rate + adaptive window. Cold-start:
        # buffer empty => error 0.0, window 15 (today's behavior). No
        # theta_withhold modulation is wired because no prediction-vs-quality
        # pair exists in the codebase to drive it; this makes the signal
        # observable for future calibration work instead of leaving it dead.
        self._calib_buf: list = []
        self._calib_window: int = 15
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
        # B3 (source-monitoring for greetings): the "welcome back" family is a
        # memory CLAIM about a PRIOR session, so it must only fire when we have
        # genuinely resumed from a saved snapshot — NOT on an arbitrary mid-session
        # turn (that asserts a past episode that never happened: confabulation,
        # Tulving 1985 / Johnson source-monitoring). _load() sets this True exactly
        # once on a successful resume; process_turn consumes it on the first turn
        # so the greeting is a reactivation artifact, never a periodic tick.
        self._session_resumed: bool = False
        self._greeting_emitted_this_session: bool = False
        self._activation_boost: Optional[Dict[str, float]] = None
        # Solution #2: Reasoning mode (stochastic / deterministic / exploratory)
        self.reasoning_mode: str = "stochastic"

        # Consistency tracking for deterministic mode: (subject_hash, tuple(seen)) -> path
        self._consistency_paths: Dict[int, List[str]] = {}
        self._consistency_trace: List[str] = []
        # Phase 18c: Multi-source consensus tracking for hallucination guard
        self._concept_sources: Dict[str, Set[str]] = {}  # concept -> set of source URLs
        # Round 4 (C1): provisional buffer (the "hippocampus"). Tokens scraped
        # from the web but not yet cleared for permanent graph admission live
        # here — tracked by distinct source so reactivation (>= k distinct
        # sources) promotes them, while one-shot junk (website names, POS-tag
        # fragments) never reaches the live graph the DMN weaver reads.
        self._provisional_nodes: Dict[str, Set[str]] = {}  # label -> set of source URLs
        self._junk_theta: float = 0.5        # junk_score threshold for admission
        self._promote_min_sources: int = 2   # distinct sources required to promote
        # Round 5 (D1): self-supervised junk_score — point the singleton at this
        # engine's data dir so weak labels + fitted model persist (junk_labels.jsonl,
        # junk_classifier.json). Cold-start reproduces the Round-4 formula exactly.
        try:
            from ravana.chat.junk_scorer import configure as _js_configure
            _js_configure(data_dir)
        except Exception:
            pass
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
        # Stable-node snapshot: the graph as it stands AFTER cold-start
        # bootstrap but BEFORE any web-learning this session. Used by the
        # counterfactual analogy-gate (_conditional_has_graph_anchor) so that
        # same-turn web-learning (which adds causal/analogical edges to a
        # freshly-looked-up concept like "pig") cannot retroactively "anchor"
        # an unanchored hypothetical and defeat the gate. Only edges among
        # these stable nodes count as a simulation anchor.
        self._stable_node_ids = set(self.graph.nodes.keys())
    # ── Adaptive threshold gate (P2-E) ────────────────────────────────
    # Distribution-driven replacement for fixed cosine/similarity cutoffs.
    # Cold-start: passed == (x >= mu)  [or x > mu when strict=True],
    # which equals the legacy fixed comparison exactly (mu == old constant).
    # Each call folds the observed x into the baseline via EMA, so the
    # gate drifts toward the running distribution (precision-weighting) and
    # is never a hard-coded number after warm-up.
    def _adaptive_gate(self, key: str, x: float, strict: bool = False,
                      eta: float = 0.05) -> bool:
        b = self._adaptive_baselines[key]
        passed = (x > b["mu"]) if strict else (x >= b["mu"])
        # EMA update of the running baseline.
        b["mu"] = (1.0 - eta) * b["mu"] + eta * x
        b["sigma"] = (1.0 - eta) * b["sigma"] + eta * abs(x - b["mu"])
        b["n"] += 1
        return passed

    # ── Learned word-frequency helpers (Plan B) ──────────────────
    # Membership checks route through the seeded+learned FrequencyModel so the
    # high-frequency tail is discovered from exposure, not authored. Day-one the
    # seed (the original hand list) is the only thing known, so behavior is
    # identical until enough tokens are observed.
    def _is_generic_noun(self, word: str) -> bool:
        m = self._freq_models.get("generic_nouns")
        return m.is_generic_noun(word) if m else (word in self._GENERIC_NOUNS)

    def _in_topic_skip(self, word: str) -> bool:
        m = self._freq_models.get("topic_skip")
        return m.is_topic_skip(word) if m else (word in self.TOPIC_SKIP_WORDS)

    def _is_subject_glue(self, word: str) -> bool:
        m = self._freq_models.get("subject_glue")
        return m.is_subject_glue(word) if m else (word in self._SUBJECT_CONTEXT_WORDS)

    def _observe_language(self, text: str) -> None:
        """Fold observed conversation text into the frequency models."""
        for m in self._freq_models.values():
            m.observe(text)
        # Item 5 (P2): lightweight novel-lemma discovery. When a regular
        # past-tense form (-ed) and its candidate base both appear in the same
        # utterance, record the mapping so _base_form generalizes beyond the
        # authored irregular list (phonological single-mechanism learning).
        try:
            _words = [w for w in text.lower().split() if w.isalpha()]
            _seen = set(_words)
            for _w in _words:
                if len(_w) >= 5 and _w.endswith("ed"):
                    _stem = _w[:-2]
                    if len(_stem) >= 3 and _stem[-1] == _stem[-2]:
                        _base = _stem[:-1]
                    elif _w.endswith("ied"):
                        _base = _w[:-3] + "y"
                    else:
                        _base = _stem
                    if _base in _seen and _base != _w:
                        self._learn_lemma(_w, _base)
        except Exception:
            pass

    # ── Recall-seed concepts (Plan: graph-derived extension, Item 4) ──
    # _RECALL_SEED_CONCEPTS is the curated anchor list. To make recall
    # detection generalize beyond the authored words (brain: recall is cued by
    # a whole semantic neighborhood, not 13 fixed tokens), extend it with the
    # graph's concepts that sit near the seed anchors (GloVe cosine >= 0.7).
    # Computed lazily, cached, and refreshed only when the graph grows a lot.
    def _recall_seed_concepts(self) -> List[str]:
        _cache = getattr(self, "_recall_seed_cache", None)
        _cached_n = getattr(self, "_recall_seed_n", -1)
        _graph = getattr(self, "graph", None)
        _n_nodes = len(getattr(_graph, "nodes", {})) if _graph else 0
        # Recompute if the cache is missing or the graph grew materially
        # (so newly learned concepts can join the recall-neighborhood).
        if _cache is not None and _n_nodes <= int(_cached_n * 1.1) + 5:
            return _cache
        _seeds = list(self._RECALL_SEED_CONCEPTS)
        try:
            _gv = self._glove_vector
            _seed_vecs = [v for w in _seeds if (v := _gv(w)) is not None]
            _nodes = getattr(_graph, "nodes", None)
            if _seed_vecs and _nodes:
                import numpy as np
                for _n in _nodes.values():
                    _lbl = getattr(_n, "label", None)
                    if not _lbl or _lbl in _seeds:
                        continue
                    _nv = _gv(_lbl)
                    if _nv is None:
                        continue
                    if any(float(np.dot(_nv, sv)) >= 0.7 for sv in _seed_vecs):
                        _seeds.append(_lbl)
        except Exception:
            pass
        self._recall_seed_cache = _seeds
        self._recall_seed_n = _n_nodes
        return _seeds

    # ── Learned lemma store (Item 5, P2) ──────────────────────────
    # High-frequency irregulars are stored whole-word (_IRREGULAR_VERBS seed).
    # Novel past-tense forms are derived phonologically (single-mechanism
    # connectionist model, McClelland & Patterson 2002) and cached here as they
    # are observed, so the engine generalizes beyond the authored 80.
    def _base_form(self, word: str) -> str:
        """Return the base (infinitive) form of a possibly-inflected word."""
        w = (word or "").lower()
        if not w:
            return w
        if w in CognitiveChatEngine._IRREGULAR_VERBS:
            return CognitiveChatEngine._IRREGULAR_VERBS[w]
        if w in self._learned_lemmas:
            return self._learned_lemmas[w]
        # Phonological fallback: CVC reduplication (stop->stopped) and the
        # -ied/-ed regular rules. Cheap, rule-based, no parser needed.
        if len(w) >= 5 and w.endswith("ed"):
            _stem = w[:-2]
            if len(_stem) >= 3 and _stem[-1] == _stem[-2] and _stem[-1].isalpha():
                return _stem[:-1]            # stopped -> stop, wugged -> wug
            if w.endswith("ied"):
                return w[:-3] + "y"          # tried -> try
            return _stem                      # walked -> walk
        return w

    def _learn_lemma(self, past: str, base: str) -> None:
        """Record a novel past->base mapping (lowercased, dedup)."""
        p, b = past.lower(), base.lower()
        if not p or not b or p == b:
            return
        if p in CognitiveChatEngine._IRREGULAR_VERBS:
            return
        self._learned_lemmas[p] = b

    def set_epistemic_register(self, register_name: str) -> str:
        reg = (register_name or "default").lower().replace(" ", "").replace("_", "")
        _REGISTERS = {
            "default":   {"curiosity": 1.0, "verbosity": 1.0, "confidence": 1.0, "go_threshold": 0.25, "dopamine": 0.5},
            "chitchat":  {"curiosity": 0.5, "verbosity": 0.8, "confidence": 1.0, "go_threshold": 0.15, "dopamine": 0.7},
            "casual":    {"curiosity": 0.5, "verbosity": 0.8, "confidence": 1.0, "go_threshold": 0.15, "dopamine": 0.7},
            "confident": {"curiosity": 1.0, "verbosity": 1.0, "confidence": 1.3, "go_threshold": 0.20, "dopamine": 0.7},
            "cautious":  {"curiosity": 1.0, "verbosity": 1.0, "confidence": 0.7, "go_threshold": 0.35, "dopamine": 0.3},
            "verbose":   {"curiosity": 1.0, "verbosity": 1.5, "confidence": 1.0, "go_threshold": 0.20, "dopamine": 0.5},
            "terse":     {"curiosity": 0.3, "verbosity": 0.2, "confidence": 1.0, "go_threshold": 0.40, "dopamine": 0.4},
            "expert":    {"curiosity": 1.0, "verbosity": 1.1, "confidence": 1.2, "go_threshold": 0.30, "dopamine": 0.5},
            "formal":    {"curiosity": 1.0, "verbosity": 1.1, "confidence": 1.2, "go_threshold": 0.30, "dopamine": 0.5},
            "creative":  {"curiosity": 1.3, "verbosity": 1.2, "confidence": 0.9, "go_threshold": 0.12, "dopamine": 0.8},
        }
        r = _REGISTERS.get(reg, _REGISTERS["default"])
        self.epistemic_register = reg if reg in _REGISTERS else "default"
        self._reg_curiosity = r["curiosity"]
        self._reg_verbosity = r["verbosity"]
        self._reg_confidence = r["confidence"]

        if hasattr(self, "basal_ganglia") and self.basal_ganglia is not None:
            try:
                self.basal_ganglia.base_go_threshold = r["go_threshold"]
                self.basal_ganglia.dopamine_tone = r["dopamine"]
            except Exception:
                pass

        confirmations = {
            "chitchat": "switched to chit-chat mode — keeping it light!",
            "casual": "switched to casual mode — keeping it relaxed and open!",
            "terse": "switched to terse mode — keeping replies brief.",
            "verbose": "switched to verbose mode — providing detailed explanations.",
            "confident": "switched to confident mode.",
            "cautious": "switched to cautious mode.",
            "expert": "switched to expert mode — focusing on technical precision.",
            "formal": "switched to formal mode — focusing on structured reasoning.",
            "creative": "switched to creative mode — exploring imaginative ideas!",
            "default": "switched to default mode.",
        }
        return confirmations.get(self.epistemic_register, "switched mode — updated settings.")

    def _check_meta_command(self, text: str) -> Optional[str]:
        t = (text or "").lower().strip(" ?!.")
        m = re.search(
            r"^\s*(?:switch\s+to\s+|turn\s+(?:on|off)\s+|enable\s+|disable\s+|set\s+(?:mode|register)\s+(?:to\s+)?|mode\s*:\s*)?"
            r"(chit\s*chat|casual|formal|terse|verbose|confident|cautious|expert|learning|creative|default)(?:\s+mode)?(?:\s+(on|off))?\s*$",
            t)
        if not m:
            m = re.search(r"^\s*(chit\s*chat|casual|formal|terse|verbose|confident|cautious|expert|learning|creative|default)\s+mode\s*$", t)

        if m:
            raw_reg = m.group(1).replace(" ", "").replace("\t", "")
            state = m.group(2) if m.lastindex and m.lastindex >= 2 else "on"
            reg = "default" if state == "off" else raw_reg
            return self.set_epistemic_register(reg)
        return None

    def _build_context_vector_from_input(self, text: str, subject: str) -> Optional[np.ndarray]:
        """Build context vector from query framing / function-word tokens (Step 1c)."""
        if not text:
            return None
        words = [w.lower() for w in re.findall(r"[a-z']+", text)
                 if w.lower() in _FUNCTION_WORDS or len(w) <= 3 or w.lower() in ("what", "why", "how", "when", "where", "is", "are", "does", "meaning", "nature", "purpose")]
        if not words or not hasattr(self, "_glove_vector"):
            return None
        vecs = [self._glove_vector(w) for w in words if self._glove_vector(w) is not None]
        if not vecs:
            return None
        mean_v = np.mean(vecs, axis=0)
        norm = np.linalg.norm(mean_v)
        return mean_v / norm if norm > 1e-6 else mean_v

    def _ensure_orthogonal(self, content_vec: np.ndarray, context_vec: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Project context vector into orthogonal subspace from content vector (Gram-Schmidt; Step 1c)."""
        if content_vec is None or context_vec is None:
            return context_vec
        norm_c = np.linalg.norm(content_vec)
        if norm_c < 1e-6:
            return context_vec
        u = content_vec / norm_c
        proj = np.dot(context_vec, u) * u
        orth = context_vec - proj
        norm_o = np.linalg.norm(orth)
        return orth / norm_o if norm_o > 1e-6 else context_vec



    # Single data-driven source for the 6 consolidated closed-class
    # lists. Prefers the fit file (data/functional_lexicon.json, loaded
    # into self._func_lex); falls back to the original hand set
    # (class attribute) so behavior is identical at cold-start and
    # external modules that read the class attr (via MRO) are untouched.
    def _closed_class(self, name: str) -> Set[str]:
        _fl = getattr(self, "_func_lex", None)
        if _fl is not None:
            try:
                return getattr(_fl, name)()
            except Exception:
                pass
        # Fall back: class attribute (set in __init__), then a direct
        # module-level load of the fit file so this also works on
        # engines built via __new__ (the audit tests) where __init__
        # never ran and neither the instance attr nor _func_lex exists.
        _ca = getattr(type(self), name.upper(), None)
        if _ca:
            return _ca
        try:
            from ravana.chat.functional_lexicon import default_lexicon
            _fl2 = default_lexicon()
            if _fl2 is not None:
                return getattr(_fl2, name)()
        except Exception:
            pass
        return set()

    def _ingest_episodic(self, user_input: str, subject: str = "") -> None:
        """Store a conversational statement in the hippocampal buffer so it can
        be recalled on a later turn.

        Root-cause fix (LoCoMo / LongMemEval): the assertion and
        self-disclosure acknowledgment paths return EARLY (before the
        store-before-recall block later in process_turn), so factual
        statements the user made were acknowledged but NEVER written to the
        recall-able episodic buffer -> multi-turn recall always failed.
        This helper is invoked on those early-return statement paths so the
        fact is persisted regardless of which acknowledgment branch handles it.

        Idempotent and fail-open: the buffer dedupes identical triples, so
        calling this and the later store block on the same turn is harmless.
        """
        try:
            if not user_input:
                return
            # Speaker attribution FIRST (LoCoMo/LongMemEval dialog format
            # "Caroline: I went to X yesterday. It was great."): bind
            # first-person pronouns to the SPEAKER so every stored trace is
            # queryable by name — "When did Caroline go to X?" must lexically
            # reach "caroline went to x". Must precede the per-sentence split
            # or tail sentences lose their speaker binding. Hippocampal
            # source monitoring: the trace carries WHO experienced it, not
            # the deictic 'I'.
            _spk_m = re.match(r"^\s*([A-Z][a-z]{2,15})\s*:\s*(.+)$",
                              user_input.strip(), re.DOTALL)
            if _spk_m:
                _spk, _body = _spk_m.group(1), _spk_m.group(2).strip()
                _body = re.sub(r"\bi'm\b", f"{_spk} is", _body,
                               flags=re.IGNORECASE)
                _body = re.sub(r"\bi've\b", f"{_spk} has", _body,
                               flags=re.IGNORECASE)
                _body = re.sub(r"\bi'll\b", f"{_spk} will", _body,
                               flags=re.IGNORECASE)
                _body = re.sub(r"\bmy\b", f"{_spk}'s", _body,
                               flags=re.IGNORECASE)
                _body = re.sub(r"\bi\b", _spk, _body, flags=re.IGNORECASE)
                self._ingest_episodic(_body, subject or _spk.lower())
                return
            # Hippocampal pattern separation: a multi-sentence utterance is
            # stored as SEPARATE per-sentence traces (plus the leading
            # sentence carrying the subject). One 300-char blob loses the
            # tail sentences ("keeps a brass sextant...") that later
            # questions cue on — measured on MemFail persona where only the
            # first sentence of a 5-sentence bio survived.
            _sents = [s.strip() for s in re.split(
                r"(?<=[.!?])\s+", user_input.strip()) if len(s.strip()) >= 12]
            # Re-merge fragments split after abbreviations ("St. Mary's",
            # "Dr. Smith", "Mr. Jones"): the naive split cut "sunday mass at
            # St. | Mary's Church on january 2nd", detaching the DATE from
            # the event — temporal arithmetic then used the session date and
            # was off by weeks (measured on LongMemEval oracle case 6:
            # 4 days computed vs gold 30).
            _abbrev = re.compile(
                r"\b(st|dr|mr|mrs|ms|prof|rev|jr|sr|vs|etc|eg|ie|no)\.$",
                re.IGNORECASE)
            _merged, _i = [], 0
            while _i < len(_sents):
                _cur = _sents[_i]
                while _i + 1 < len(_sents) and _abbrev.search(_cur):
                    _cur = _cur + " " + _sents[_i + 1]
                    _i += 1
                _merged.append(_cur)
                _i += 1
            _sents = _merged
            if len(_sents) > 1:
                for _sent in _sents:
                    self._ingest_episodic(_sent, subject)
                return
            # Never ingest an INTERROGATIVE as an episodic fact. Questions are
            # retrieval cues, not assertions — storing "what was the first
            # issue i had with my new car?" under subject "issue"/"car" made
            # _try_hippocampal_retrieval echo the PREVIOUS QUESTION back as a
            # remembered fact ("you told me earlier: what was the first
            # issue...") on every subsequent question (LongMemEval regression).
            # Hippocampus encodes experienced content; the PFC holds the query.
            _stripped = user_input.strip()
            if _stripped.endswith("?") or re.match(
                    r"^\s*(who|what|when|where|which|why|how|did|do|does|is|"
                    r"are|was|were|had|has|have|will|would|could|can)\b",
                    _stripped.lower()):
                return
            # D1 (round 2026-08-08b-d): a recall-scaffold query that is NOT a
            # question (no trailing '?', no interrogative opener) — e.g.
            # "you mentioned my tarantula before, remind me what i told you
            # about the one that molted" — must NOT be encoded as an episodic
            # "fact". The hippocampus stores experienced CONTENT; a memory/recall
            # directive is a PFC query. Encoding it lets a later semantically
            # overlapping recall ("what's the strongest read you've formed")
            # retrieve the prior recall query's OWN text and echo it verbatim ->
            # a recursive recall loop (the tarantula echo). Gate on recall
            # scaffold (remember/recall/remind + a told/said/mentioned verb),
            # structural regex not a per-topic list.
            if re.search(
                    r"\b(remember|recall|remind(?: me)?)\b.*"
                    r"\b(told|said|ask|mention|tell|mentioned|asked)\b",
                    _stripped.lower()) or re.search(
                    r"\b(remind|remember)\b.*\b(i|you)\b.*"
                    r"\b(told|said|mentioned|asked|tell)\b", _stripped.lower()):
                return
            content_words = [w.strip(".,!?;:") for w in user_input.lower().split()
                             if len(w.strip(".,!?;:")) >= 3
                             and w.strip(".,!?;:").isalpha()]
            if not content_words:
                return
            # Fall back to the first content word as the subject key when the
            # caller has not extracted one yet (e.g. the self-disclosure path
            # fires before topic extraction).
            subj = (subject or "").strip()
            if not subj:
                _skip = {"i", "you", "he", "she", "they", "we", "it", "my",
                         "your", "his", "her", "their", "our"}
                subj = next((w for w in content_words if w not in _skip),
                            content_words[0])
            aliases = list(content_words)
            if hasattr(self, "user_model") and getattr(
                    self.user_model, "user_name", ""):
                aliases.append(self.user_model.user_name.lower())
            # Phase 1: resolve any date reference in the utterance against the
            # current session date so the fact carries an absolute date.
            _sess_date = getattr(self, "_current_session_date", None)
            _abs_date = None
            _grounder = getattr(self, "_date_grounder", None)
            if _grounder is not None:
                try:
                    _g = _grounder.ground_utterance(user_input, _sess_date)
                    if _g is not None:
                        _abs_date = _g.date
                except Exception:
                    _abs_date = None
            # Default anchor: an event mentioned in a dated session but without
            # its own explicit date reference is anchored to the SESSION date
            # (LoCoMo semantics — "when did X happen" -> the session it was
            # discussed in).
            if _abs_date is None and _sess_date is not None:
                _abs_date = _sess_date
            # Phase 4: knowledge updates — when a new statement about a subject
            # contains an update/retraction cue ("now", "quit", "left", "sold",
            # "started at", "currently"), mark any earlier stored fact for that
            # subject as superseded so recency-weighted retrieval returns the
            # newest value by default (LongMemEval knowledge-update resolution).
            # Phase 4: knowledge updates — when a new statement about a subject
            # contains an update/retraction cue ("now", "quit", "left", "sold",
            # "started at", "currently"), mark the earlier stored fact FOR THE
            # SAME ATTRIBUTE as superseded so recency-weighted retrieval returns
            # the newest value by default (LongMemEval knowledge-update
            # resolution). Word-boundary matched: substring cues fired on
            # "known"/"nowhere"-style words.
            _update_cue = re.search(
                r"\b(?:now|quit|left|sold|no longer|used to|currently|"
                r"changed|started at|actually)\b", user_input.lower())
            if _update_cue:
                try:
                    # Only supersede a previous fact when it shares attribute
                    # content with the NEW statement beyond the entity name.
                    # The old blanket loop superseded EVERY fact under the
                    # subject and every alias — on LoCoMo, conversational
                    # "now"/"actually" turns marked an entity's ENTIRE history
                    # superseded (measured: all 612 caroline facts, so recall
                    # returned ack fillers). An update replaces the OLD VALUE
                    # OF THE SAME ATTRIBUTE, not the person's whole past.
                    _new_words = {w for w in content_words
                                  if len(w) >= 4 and w != subj}
                    _seen_ids = set()
                    for _key in [subj] + aliases:
                        _fan = list(self.hippocampal_buffer.retrieve(_key) or [])
                        # Interference theory: an update overwrites the SAME
                        # attribute, and attribute identity is signalled by
                        # overlap on INFORMATIVE features. Raw overlap let
                        # ubiquitous fan words — the speaker's own possessive
                        # after source binding ("caroline's"), "through" —
                        # mark unrelated memories superseded (measured on
                        # LoCoMo dlg0: "used to go horseback riding ...
                        # through the fields" nuked "moved from caroline's
                        # home country" via shared {caroline's, through}).
                        # Weight by document frequency across THIS cue's fan:
                        # a shared word counts only if it is no commoner than
                        # the old fact's own mean word-df (self-normalizing,
                        # distribution-driven bar — no fixed threshold); the
                        # single-word branch additionally requires the word
                        # be UNIQUE to the old trace within the fan.
                        _df = {}
                        for _ff in _fan:
                            for _fw in {x.strip(".,!?;:") for x in
                                        (getattr(_ff, "object", "") or "").lower().split()}:
                                if len(_fw) >= 4:
                                    _df[_fw] = _df.get(_fw, 0) + 1
                        for _prev in _fan:
                            if id(_prev) in _seen_ids:
                                continue
                            _seen_ids.add(id(_prev))
                            if _prev.superseded or _prev.object == user_input[:300]:
                                continue
                            _prev_words = {
                                w.strip(".,!?;:") for w in
                                (_prev.object or "").lower().split()
                                if len(w.strip(".,!?;:")) >= 4}
                            _shared = _prev_words & _new_words
                            if _shared and _df:
                                _bar = (sum(_df.get(w, 1) for w in _prev_words)
                                        / max(1, len(_prev_words)))
                                _informative = {w for w in _shared
                                                if _df.get(w, 1) <= _bar}
                            else:
                                _informative = _shared
                            # Require MEANINGFUL attribute overlap (2+ shared
                            # informative words, or 1 fan-unique word when the
                            # new statement is short/specific) before
                            # declaring the old fact stale.
                            if len(_informative) >= 2 or (
                                    len(_informative) == 1
                                    and len(_new_words) <= 4
                                    and _df.get(next(iter(_informative)), 1) <= 1):
                                _prev.superseded = True
                except Exception:
                    pass
            # Source monitoring: a first/second-person self-disclosure ("my cat
            # is pixel", "i live in berlin") is a USER fact, not world
            # knowledge. Keep it in the buffer for episodic recall, but flag it
            # so sleep consolidation never drains it into the world graph as an
            # entity-keyed edge (it graduates via UserModel.personal_facts).
            _is_user_fact = bool(re.search(
                r"\b(?:my|our)\s+[a-z]|\bi\s+(?:am|'m|was|live|work|have|had|"
                r"like|love|hate|enjoy|prefer|think|believe|went|moved|got)\b",
                user_input.lower()))
            self.hippocampal_buffer.store(
                subject=subj,
                predicate="is_about",
                object=user_input[:300],
                confidence=0.6,
                aliases=aliases[:12],
                session_date=_sess_date,
                absolute_date=_abs_date,
                user_fact=_is_user_fact,
            )
        except Exception:
            pass

    def _structured_recall(self, user_input: str) -> Optional[str]:
        """Structured-first biographical / stance recall (round 2026-08-08).

        Root cause it fixes: biographical and self-stance recall queries
        ("what's my name", "what did you tell me about the cafeteria smell",
        "you mentioned a stance on medical data", "did you take a position on
        X") were reaching fact_reasoning.enumerate_matching, which joins up to
        6 stored fact-TEXTS on a loose word intersection and DUMPS unrelated
        prior turns concatenated ("based on what you've told me: <turn A>
        <turn B> <turn C>..."). The cue mapped to NO specific stored entity, so
        the reply was a wrong/mismatched memory (measured across the Noor chat
        round: name -> "i don't know", cafeteria -> the green-comet turn,
        surveillance -> the cafeteria turn).

        This resolver reads the LIVE durable stores (personal_facts,
        opinions.stances, belief_store) and answers ONLY when the query's cue
        maps to a real stored entity/topic. It never concatenates unrelated
        turns. Fail-closed: returns None when nothing maps, so the existing
        (honest) pipeline handles genuinely unstored questions.

        No hardcoded reply strings, no per-topic answer table, no retraining.
        Every answer slot is read from a runtime store RAVANA grows autonomously
        (the user can correct any fact/stance; the stores merge on correction).
        """
        q = (user_input or "").lower().strip()
        if not q:
            return None
        um = getattr(self, "user_model", None)
        pf = getattr(um, "personal_facts", None) if um else None
        opinions = getattr(um, "opinions", None) if um else None
        beliefs = getattr(self, "belief_store", None)

        # ── (1) Biographical self-fact recall ──────────────────────────────
        # "what's my name" / "where do i live/work" / "what do i keep/have on
        # my rooftop" / "what's my favorite ..." — answered from the structured
        # personal_fact store, not the episodic buffer.
        _BIO_ATTR = {
            "name": ("name",),
            "live": ("location", "live in"),
            "work": ("work",),
            "job": ("work",),
            "rooftop": ("does", "keep", "have"),
            "favorite": ("favorite",),
        }
        # "what's my name" / "who am i" -> name
        if re.search(r"\b(my name|who am i|what am i called)\b", q) or \
                re.search(r"\bwhat'?s\s+my\s+name\b", q):
            _v = pf.get("i", "name") if pf else None
            if _v is not None and not getattr(_v, "superseded", False):
                return f"your name is {_v.value}."
            return None
        # "where do i live / work" / "what do i do"
        if re.search(r"\b(where do i live|what city|what town|where am i from)\b", q) and \
                re.search(r"\b(live|from)\b", q):
            _v = pf.get("i", "location") if pf else None
            if _v is not None and not getattr(_v, "superseded", False):
                return f"you live in {_v.value}."
            return None
        if re.search(r"\b(where do i keep|where do i have|where do i store)\b", q):
            # "where do i keep the light" / "where do i keep my pigeons" —
            # answer from the 'does' fact whose value overlaps the query noun,
            # not the dictionary definition of the noun (the word "light" is a
            # world concept that otherwise echoes "electromagnetic radiation").
            _qn = set(re.findall(r"[a-z']+", q)) - {
                "where", "do", "i", "keep", "have", "store", "my", "the",
                "a", "an", "on", "in", "at", "to", "of", "for", "with", "and"}
            for _k, _f in pf.facts.items():
                if not (isinstance(_k, tuple) and len(_k) == 3):
                    continue
                if _k[1] != "does" or getattr(_f, "superseded", False):
                    continue
                _val = _f.value.lower()
                if any(n in _val for n in _qn):
                    return f"you {_val}."
        if re.search(r"\b(where do i work|what do i do|what's my job|what is my work)\b", q):
            _v = pf.get("i", "work") if pf else None
            if _v is not None and not getattr(_v, "superseded", False):
                return f"you work as {_v.value}."
        if re.search(r"\b(call sign|radio sign|my sign|ham (call|sign))\b", q):
            # "what's my call sign" / "what's my radio sign" — prefer the most
            # RECENT call-sign fact. A corrected sign (e.g. "kh6-mist") is
            # stored after the first ("kx7-mist"); RAVANA must surface the
            # corrected value, not the stale first one. Iterate facts in
            # insertion order and keep the last non-superseded call-sign/sign.
            _best = None
            for _k, _f in pf.facts.items():
                if not (isinstance(_k, tuple) and len(_k) == 3):
                    continue
                if _k[1].lower() in ("call sign", "sign") and \
                        not getattr(_f, "superseded", False):
                    _best = _f.value
            if _best:
                return f"your call sign is {_best}."
            return None
        if re.search(r"\bwhat'?s\s+my\s+favorite\b", q):
            # surface every favorite_* fact
            if pf is not None:
                _bits = []
                for _k, _f in pf.facts.items():
                    if isinstance(_k, tuple) and len(_k) == 3 and \
                            _k[1].startswith("favorite") and \
                            not getattr(_f, "superseded", False):
                        _bits.append(f"your {_k[1].replace('favorite ', '')} is {_f.value}")
                if _bits:
                    return "; ".join(_bits) + "."
            return None
        # "what do i keep/have on my rooftop / what do i grind / what do i
        # forage" — match a 'does' / activity fact whose value overlaps the
        # query's content noun.
        _ACT = re.search(r"\bwhat do i (keep|have|grind|forage|race|play|raise|grow|do|study|research|make|build|write|paint|carve|brew|bake|craft|teach|run|operate|manage|restore|tend|practice|code|design|volunteer|cook|fish|hike|garden|farm|lead|organize|collect|watch|read|learn|sail|knit|forge|fly)\b", q)
        # ── (1b) "what did i tell you about X" / "what do i think of X" ──────
        # General user-attribute / user-stance recall. X is resolved to a
        # stored STANCE TOPIC (the user's own attitudes) or to a personal
        # fact. This is the precise replacement for the old loose
        # enumerate_matching dump that concatenated unrelated turns.
        _TOLD = re.search(
            r"\b(?:what\s+(?:did|do)\s+i\s+(?:tell|say)\s+(?:you|me)\s+about|"
            r"what\s+(?:do|did)\s+i\s+(?:think|feel)\s+(?:of|about)|"
            r"how\s+(?:do|did)\s+i\s+feel\s+about|"
            r"what'?s\s+my\s+(?:opinion|stance)\s+(?:on|about|of))\b"
            r"\s+([a-z][a-z \-]{1,40})", q)
        if _TOLD and pf is not None:
            _cue = _TOLD.group(1).strip().strip("?.!").lower()
            # (a) resolve to a stance topic the user holds
            if opinions is not None:
                _topic = opinions.resolve_topic(_cue) or _cue
                _s = opinions.query_stance(_topic)
                if _s is not None:
                    _pol = _s.polarity
                    if _pol >= 0.6:
                        _w = "strongly for"
                    elif _pol > 0.1:
                        _w = "for"
                    elif _pol <= -0.6:
                        _w = "strongly against"
                    elif _pol < -0.1:
                        _w = "against"
                    else:
                        _w = "uncertain about"
                    return f"you're {_w} {_topic}."
            # (b) resolve to a personal fact whose value/noun overlaps the cue.
            # D4 (round 2026-08-08b): the old matcher accepted ANY single
            # salient-token substring ("any(n in _v for n in _cnouns)"), so
            # "what did i tell you about the ocean" matched the lighthouse
            # belief because the token "the" or "ocean" happened to overlap a
            # far-off fact. Require a SUBSTANTIAL match: the stored fact must
            # contain the full cue phrase, OR share >=2 salient cue tokens
            # with the value (so a one-word coincidence cannot hijack the
            # recall). This is structural overlap scoring, not a per-topic
            # table; it generalizes across every persona/topic.
            _cnouns = set(re.findall(r"[a-z']+", _cue)) - {
                "the", "a", "an", "of", "about", "on", "my", "i", "you",
                "what", "do", "did", "tell", "say", "think", "feel", "s",
                "is", "are", "was", "were", "to", "in", "for", "with", "that",
                "this", "it", "and", "or", "but", "from", "by", "as", "at"}
            for _k, _f in pf.facts.items():
                if not (isinstance(_k, tuple) and len(_k) == 3):
                    continue
                if getattr(_f, "superseded", False):
                    continue
                _v = _f.value.lower()
                _attr = _k[1].lower()
                # strong match: the whole cue phrase appears verbatim in the
                # stored value, or the attribute itself is the cue topic.
                if _cue in _v or _attr == _cue:
                    if _attr == "name":
                        return f"your name is {_v}."
                    if _attr == "work":
                        return f"you work as {_v}."
                    if _attr == "does":
                        return f"you {_v}."
                    return f"your {_attr} is {_v}."
                # weak match: require >=2 salient cue tokens to co-occur in the
                # value (a single shared stop-word like "the"/"ocean" is not
                # enough to claim this is the fact the user meant).
                if _cnouns and sum(1 for n in _cnouns if n in _v) >= 2:
                    if _attr == "name":
                        return f"your name is {_v}."
                    if _attr == "work":
                        return f"you work as {_v}."
                    if _attr == "does":
                        return f"you {_v}."
                    return f"your {_attr} is {_v}."
            return None

        if _ACT and pf is not None:
            _verb = _ACT.group(1)
            _qnouns = set(re.findall(r"[a-z']+", q)) - {
                "what", "do", "i", "my", "on", "the", "a", "an", "to", "you",
                "of", "in", "for", "with", "and", "that", "this", "is", "are"}
            if "rooftop" in q or "roof" in q:
                _qnouns.add("rooftop")
            for _k, _f in pf.facts.items():
                if not (isinstance(_k, tuple) and len(_k) == 3):
                    continue
                if _k[1] == "does" and not getattr(_f, "superseded", False):
                    _val = _f.value.lower()
                    if _verb in _val or any(n in _val for n in _qnouns):
                        return f"you {_val}."
            # also try the work fact
            _w = pf.get("i", "work") if pf else None
            if _w is not None and not getattr(_w, "superseded", False) \
                    and _verb in _w.value.lower():
                return f"you {_w.value}."

        # ── (2) Self-stance / self-belief recall ──────────────────────────
        # "you mentioned a stance on X" / "did you take a position on X" /
        # "what's your stance on X" / "what do you think about X" — answered
        # from RAVANA's OWN stance/belief store, never by replaying a USER
        # utterance (that is a self/other boundary violation).
        _SELFSTANCE = re.search(
            r"\b(you (?:mentioned|said|told me|have|took|take)|"
            r"your (?:stance|position|view|opinion|take) (?:on|about)|"
            r"what do you (?:think|feel|believe) about|"
            r"do you (?:have|take) a (?:stance|position|view) on)\b"
            r".{0,40}?([a-z][a-z ]{2,40})", q)
        if _SELFSTANCE and opinions is not None:
            _topic_phrase = _SELFSTANCE.group(2).strip().strip("?.!")
            # resolve the phrase to a known stance topic (semantic-ish via the
            # store's own resolver, which folds synonyms)
            _topic = opinions.resolve_topic(_topic_phrase) or _topic_phrase.lower().strip()
            # ROUND 2026-08-09i FIX: reject DEICTIC / GENERIC topic phrases.
            # A loosely-matched self-stance query ("do you have anything like
            # that?", "what's your view on it") resolves its topic to a pronoun
            # or generic phrase ("anything like that", "that") which then maps
            # via resolve_topic onto a STORED stance / belief and emits
            # "i'm strongly for <junk>" or "i hold that position: <unrelated>"
            # — confabulated, since the topic carries no real content. Fail
            # closed: if the phrase contains NO substantive retrieval token
            # (every token is a closed-class / deictic / generic word), return
            # None so the query falls through to the honest generative path.
            # Structural guard (a closed-class token set), not a per-topic table.
            _GEN = {"anything", "something", "everything", "nothing",
                     "that", "it", "this", "stuff", "things", "thing",
                     "matter", "point", "idea", "question", "issue",
                     "topic", "yes", "no", "maybe", "ok", "okay", "like",
                     "rather", "or", "and", "but", "if", "than", "as", ""}
            _phrase_tokens = [t for t in re.findall(r"[a-z']+", _topic_phrase.lower())]
            _substantive = [t for t in _phrase_tokens
                            if t not in _GEN and len(t) > 2
                            and t not in ("the", "a", "an", "of", "about",
                                          "on", "my", "i", "you", "what",
                                          "do", "did", "tell", "say", "think",
                                          "feel", "stance", "position", "own",
                                          "owning", "your")]
            if not _substantive:
                _s = None
            else:
                _s = opinions.query_stance(_topic)
            if _s is not None:
                _pol = _s.polarity
                if _pol >= 0.6:
                    _w = "strongly for"
                elif _pol > 0.1:
                    _w = "for"
                elif _pol <= -0.6:
                    _w = "strongly against"
                elif _pol < -0.1:
                    _w = "against"
                else:
                    _w = "uncertain about"
                return f"i'm {_w} {_topic}."
            # fall back to belief store
            if beliefs is not None:
                _bs = beliefs.get_state().get("beliefs", {})
                # ROUND 2026-08-09i: reject DEICTIC / GENERIC cue tokens. A
                # loosely-matched self-stance query ("do you have anything like
                # that?") resolves _topic_phrase to a pronoun/generic word
                # ("anything", "that", "it") which then appears as a salient
                # token and matches a stored belief by coincidence (e.g.
                # "raw honey... beats *anything* in a jar") -> confabulated
                # "i hold that position: ...". These tokens carry no retrieval
                # target, so exclude them from the match set entirely.
                _GEN = {"anything", "something", "everything", "nothing",
                         "that", "it", "this", "stuff", "things", "thing",
                         "matter", "point", "idea", "question", "issue",
                         "topic", "yes", "no", "maybe", "ok", "okay", "like",
                         "rather", "or", "and", "but", "if", "than", "as", ""}
                _toks = set(re.findall(r"[a-z']+", _topic_phrase)) - {
                    "the", "a", "an", "of", "about", "on", "my", "i", "you",
                    "what", "do", "did", "tell", "say", "think", "feel",
                    "stance", "position", "own", "owning", "your"} - _GEN
                if not _toks:
                    # nothing substantive to match on -> fail closed
                    pass
                else:
                    for _bk, (_txt, _conf, _turn) in _bs.items():
                        _tl = _txt.lower()
                        _tok_hits = sum(1 for t in _toks if len(t) > 3 and t in _tl)
                        if _topic in _tl or _tok_hits >= 2:
                            return f"i hold that position: {_txt}"
        # ── (2b) USER-belief recall ─────────────────────────────────────
        # "what did i tell you i believe about X" / "what do i believe about
        # X" ask for the USER's own stated belief, not RAVANA's stance. Answer
        # from the user-belief store (belief_store entries keyed ('user',
        # 'told:N')), never by replaying an unrelated episodic turn (the D4
        # loose-match bug: "about the ocean" echoed the lighthouse belief).
        # Require the cue to appear verbatim in the belief text OR share >=2
        # salient cue tokens with it, so a one-word overlap cannot hijack the
        # recall. Structural overlap scoring; generalizes across topics.
        _USERBEL = re.search(
            r"\b(?:what did i (?:tell|say) you|what do i|remind me what i|"
            r"what (?:was|is) my)\b.{0,30}?\b(?:believe|belief|think|stance|position)\b"
            r".{0,30}?\b(?:about|on|of)?\b\s*(?P<cue>[a-z][a-z \-]{1,40})", q)
        if _USERBEL and beliefs is not None:
            _cue = _USERBEL.group("cue").strip().strip("?.!").lower()
            # the optional preposition may have been swept into the capture
            # ("about the ocean"); strip a leading closed-class word so the
            # cue is the real topic ("the ocean" -> "ocean" tokens).
            _cue = re.sub(r"^(about|on|of|the|a|an)\s+", "", _cue).strip()
            _btoks = set(re.findall(r"[a-z']+", _cue)) - {
                "the", "a", "an", "of", "about", "on", "my", "i", "you",
                "what", "did", "tell", "say", "do", "believe", "belief",
                "think", "stance", "position", "is", "was", "are", "to",
                "in", "for", "with", "that", "this", "it", "and", "or",
                "but", "from", "by", "as", "at"}
            _bs = beliefs.get_state().get("beliefs", {})
            _best = None
            for _bk, (_txt, _conf, _turn) in _bs.items():
                _tl = _txt.lower()
                _hits = sum(1 for t in _btoks if len(t) > 3 and t in _tl)
                # D4 guard: require the verbatim cue OR >=2 salient cue tokens
                # to co-occur in the belief text. A single shared token (e.g.
                # "ocean") must not hijack the recall to an unrelated belief.
                if _btoks and (_cue in _tl or _hits >= 2):
                    _best = _txt
                    break
            if _best is not None:
                return f"you told me: {_best}"
        return None

    def _recall_user_fact(self, attr_hint, q):
        """Helpers for _structured_recall: read a personal_fact by attribute."""
        pf = getattr(getattr(self, "user_model", None), "personal_facts", None)
        if pf is None:
            return None
        for _k, _f in pf.facts.items():
            if not (isinstance(_k, tuple) and len(_k) == 3):
                continue
            if getattr(_f, "superseded", False):
                continue
            if _k[1].lower() == attr_hint:
                return _f.value
        return None

    def _try_fact_reasoning(self, user_input: str) -> Optional[str]:
        """Answer question-shaped input from the hippocampal buffer's stored
        fact texts via ravana.core.fact_reasoning (lexical-closure replay).

        Routing (each handler fails open by returning None):
          1. select_option        — multiple-choice ('Options: A..') via
                                    chain closure from the question cue.
          2. conditional_answer   — condition->behavior rule check with
                                    numeric-threshold + negation tests.
          3. enumerate_matching   — category enumeration ('which hats...')
                                    using ConceptNet isa parents when the
                                    ontology is loaded.
          4. entity_fact_answer   — named-entity cued recall.
          5. missing_entity_abstention — named person absent from the
                                    store -> honest 'i don't have info'.
        Only fires on interrogative-shaped input; assertions fall through
        untouched so ingestion/acknowledgment paths are unaffected.
        """
        if not user_input:
            return None
        _s = user_input.strip()
        _is_q = ("?" in _s
                 or re.search(r"\boptions?\s*:", _s, re.IGNORECASE)
                 or re.match(
                     r"^\s*(who|what|when|where|which|why|how|did|do|does|is|"
                     r"are|was|were|would|will|could|can|should)\b",
                     _s.lower()))
        if not _is_q:
            return None
        # R3 (round v3): SELF-RECALL QUESTIONS must NOT be answered from the
        # generic hippocampal fact-text buffer. "what do you know about me",
        # "what matters most to me", "tell me something about myself" are
        # questions about the USER's own disclosed profile, NOT world facts —
        # answering them from the buffer returns RAVANA's own (simulated) opinion
        # utterances instead of the user's stored facts (observed: T49 returned
        # "i love the smell of wet earth..." which is RAVANA's echoed stance, not
        # the user's). They are handled correctly downstream by
        # _try_memory_query's self-recall path (which reads the PersonalFactStore
        # + episodic index). The cue set is structural (first/second person +
        # self-reference + recall verb), matching the self-recall detector in
        # engine_memory.py, so no per-topic table. Fail-open: a genuine world
        # question is unaffected.
        _self_recall_q = bool(re.search(
            r"\b(?:what|anything|tell me|something)\b.*\b(?:do )?you\b.*"
            r"\b(?:know|remember|recall|learned?|figured out|care|think)\b"
            r".*\b(?:about me|me|my|myself)\b", _s.lower())) or \
            bool(re.search(
                r"\bwhat (?:do|does) you know about me\b", _s.lower())) or \
            bool(re.search(
                r"\bwhat matters (?:most|to me)|tell me (?:one )?thing you.*me\b",
                _s.lower()))
        if _self_recall_q:
            return None
        # Temporal questions ("when did X", "how long ago") are answered by
        # _answer_temporal_recall downstream from DATED facts. The raw-text
        # handlers here have no date access — letting entity_fact_answer
        # echo an undated fact text shadowed the correct dated answer
        # (measured on LoCoMo dlg0: sunrise question returned prose instead
        # of '8 May 2022'). EXCEPTION: multiple-choice input ('Options:')
        # is a selection task regardless of its leading word — 'When I
        # write poetry, what do I end up doing? Options: ...' must still
        # reach select_option (measured regression on MemFail longhop).
        _ql_gate = _s.lower()
        if not re.search(r"\boptions?\s*:", _ql_gate) and (
                re.match(r"^\s*(when|what year|what date|how long)\b", _ql_gate)
                or "how long" in _ql_gate
                or re.search(r"how many (day|week|month|year)s?\b", _ql_gate)):
            return None
        # Collect the raw fact texts currently in the buffer (deduped,
        # insertion-ordered). FactTriple.object holds the original utterance.
        _texts: List[str] = []
        _seen = set()
        try:
            for _subj_facts in self.hippocampal_buffer.facts.values():
                for _f in _subj_facts:
                    if getattr(_f, "superseded", False):
                        continue
                    _t = getattr(_f, "object", "") or ""
                    if _t and _t not in _seen:
                        _seen.add(_t)
                        _texts.append(_t)
        except Exception:
            return None
        if not _texts:
            return None
        from ravana.core import fact_reasoning as _frz
        _isa_map = None
        try:
            _ont = getattr(self, "_cn_ontology", None)
            # ConceptNetOntology object exposes .isa (dict word->parents);
            # a raw ont.pkl dict exposes key 'isa'. Support both.
            if _ont is not None and hasattr(_ont, "isa"):
                _isa_map = _ont.isa
            elif isinstance(_ont, dict):
                _isa_map = _ont.get("isa")
        except Exception:
            _isa_map = None

        def _isa_parents(w: str):
            if not _isa_map:
                return set()
            w = w.lower().replace(" ", "_")
            out = set(_isa_map.get(w, set()))
            if w.endswith("s"):
                out |= set(_isa_map.get(w[:-1], set()))
            for p in list(out):
                out |= set(_isa_map.get(p, set()))
            return out

        _resp = (_frz.select_option(user_input, _texts)
                 # Structured inference on the question's OWN premises
                 # (HPC->PFC deliberation): conditional / categorical /
                 # disjunctive frames mined from the presented text, unit
                 # propagation, entailment test per option. Fail-closed.
                 or self._graph_reasoner_answer(user_input)
                 # Unknown-person abstention runs BEFORE the content handlers:
                 # a question about Noah Brooks must not be answered from
                 # Yuki Tanaka's facts (measured misfire on MemFail persona).
                 or _frz.missing_entity_abstention(user_input, _texts)
                 or _frz.conditional_answer(user_input, _texts))
        if _resp is not None:
            return _resp
        _is_mc = bool(re.search(r"\boptions?\s*:", user_input.lower()))
        if not _is_mc:
            # Content-recall handlers: only for free-text questions.
            _resp = (_frz.enumerate_matching(
                         user_input, _texts,
                         isa_parents=_isa_parents if _isa_map else None)
                     or self._entity_recall_via_buffer(user_input)
                     or _frz.entity_fact_answer(user_input, _texts))
            return _resp
        # MC input: recall handlers may still find the evidence, but a raw
        # fact echo is not a letter answer (measured on LogiQA: 21/50
        # echoes, all 0). READ OUT the echo against the options — if the
        # retrieved evidence selects exactly one option, answer with that
        # option (evidence-based); otherwise fall through to forced choice.
        _echo = (self._entity_recall_via_buffer(user_input)
                 or _frz.entity_fact_answer(user_input, _texts))
        if _echo:
            try:
                _, _opts = _frz._split_options(user_input)
                _ew = _frz.content_words(_echo)
                _hits = [o for o in _opts
                         if _frz.content_words(o) and
                         _frz.content_words(o) <= _ew]
                if len(_hits) == 1:
                    return _hits[0]
            except Exception:
                pass
        # Section 6.4 additive candidate: learned triplet inference,
        # consulted only after EVERY evidence-based handler above
        # abstained, and only ahead of the forced-choice fluency
        # fallback (it may preempt a guess, never an evidence answer).
        # Fail-closed: returns None unless a Wilson gate is open.
        if getattr(self, "use_triplet_candidate", False):
            _tc = self._triplet_mc_answer(user_input, _texts)
            if _tc is not None:
                return _tc
        # Section 6.5 additive candidate: brain-faithful relational
        # reasoning (System-2 ProblemWorkingMemory + RoleMetaruleEngine).
        # Consulted only after the learned triplet candidate abstained.
        # Fail-closed: returns None unless exactly one option is
        # entailed by the question's own premises. Zero dependence on
        # lifetime RelationProfile counts.
        if getattr(self, "use_deductive_candidate", False):
            _dc = self._deductive_mc_answer(user_input, _texts)
            if _dc is not None:
                return _dc
        # Forced-choice fluency fallback (attribute substitution under
        # forced choice, Kahneman 2002): ONLY for input that requires
        # selecting an option, after every evidence-based handler
        # abstained. Free-text questions never reach this branch.
        return _frz.plausibility_choice(user_input, _texts)

    def _triplet_mc_answer(self, user_input: str,
                           fact_texts) -> Optional[str]:
        """Section 6.4 additive MC candidate from the learned triplet
        operator. Fail-closed: None unless a Wilson-gated inference
        channel produces evidence FOR exactly one option.

        Mechanism: mine SPO premises from the question's own text (the
        in-prompt premises are the evidence set, mirroring
        _graph_reasoner_answer's HPC->PFC discipline), ingest them into
        the operator's memory, then test each option as a target of
        infer(). Only gated channels (transitive/symmetric/composition)
        can add non-direct conclusions, and those gates only open when
        the PERSISTENT learned profiles carry enough evidence — cold
        profiles mean every gate is closed and this returns None.
        """
        op = getattr(self, "triplet_op", None)
        if op is None:
            return None
        try:
            from ravana.core import fact_reasoning as _frz
            from ravana.core.triplet_inference import Triple
            from ravana.core.triplet_inference.canonical import (
                canonical_predicate, canonical_term)
            main, opts = _frz._split_options(user_input)
            if len(opts) < 2:
                return None
            # Mine premises from the question text via the existing
            # in-prompt parsers (parser stays, inference is learned).
            from ravana.core.in_prompt_reasoner import parse_universal_edges
            universals, instances = parse_universal_edges(main)
            premises = [(a, "is", b) for a, b in universals + instances
                        if a and b and a != b]
            for p in (self.proposition_parser
                      .extract_propositions(main) or []):
                s = canonical_term(getattr(p, "subject", "") or "")
                o = canonical_term(getattr(p, "object", "") or "")
                r = canonical_predicate(getattr(p, "predicate", "") or "")
                # Only clean, short premises — a blob subject is noise.
                if s and o and r and len(s.split()) <= 4 and \
                        len(o.split()) <= 4:
                    premises.append((s, r, o))
            if not premises:
                return None
            for s, r, o in premises:
                op.ingest_triple(Triple(s, r, o, source="conversation"))
            # Collect gated (non-lookup) conclusions from each premise
            # subject; every conclusion is a (subject, object, conf)
            # claim licensed by an open Wilson gate.
            conclusions = []
            seen_pairs = set()
            for s, r, _o in premises:
                for res in op.infer(s, r, max_results=5):
                    if res.operator == "lookup":
                        continue
                    key = (s, res.triple.object)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    conclusions.append((s, res.triple.object,
                                        res.confidence))
            if not conclusions:
                return None
            # Match options against conclusions on normalized word
            # overlap: an option supported iff it contains BOTH the
            # subject and the inferred object (as normalized words).
            from ravana.core.in_prompt_reasoner import _norm_class
            scored = []
            for i, opt in enumerate(opts):
                owords = {_norm_class(w) for w in
                          re.findall(r"[a-z0-9]+", opt.lower())}
                best = 0.0
                for s, obj, conf in conclusions:
                    s_in = _norm_class(s) in owords or s in owords
                    o_in = _norm_class(obj) in owords or obj in owords
                    if s_in and o_in and conf > best:
                        best = conf
                if best > 0.0:
                    scored.append((best, i, opt))
            if len(scored) != 1:
                # Zero = every gate closed; >1 = ambiguous. Abstain both.
                return None
            return scored[0][2]
        except Exception:
            return None

    def _deductive_mc_answer(self, user_input: str,
                              fact_texts) -> Optional[str]:
        """Section 6.5 additive MC candidate: brain-faithful relational
        reasoning over the question's OWN premises (System-2 decoupling).

        Fail-closed: None unless EXACTLY one option is entailed by the
        problem's premises under the RoleMetaruleEngine. Never consults
        self.triplet_op / RelationProfile — the working memory is built
        fresh per turn from the text, so novel relations chain on first
        exposure with zero lifetime-frequency dependence.

        Contract mirrors _triplet_mc_answer: the engine wires this after
        the evidence handlers AND the learned triplet candidate abstain,
        so it can only ever PRE-EMPT a forced-choice guess, never an
        evidence answer.
        """
        try:
            from ravana.core.deductive_reasoning import deductive_mc_answer
            return deductive_mc_answer(user_input)
        except Exception:
            return None

    def _graph_reasoner_answer(self, user_input: str) -> Optional[str]:
        """Structured entailment over premises mined from the question text
        (graph_reasoner.select_option_logic). Fail-closed: None unless a
        single option is entailed. Only fires for MC input with logical
        structure — a plain recall question has no rules to mine."""
        try:
            from ravana.core.graph_reasoner import select_option_logic
            return select_option_logic(user_input)
        except Exception:
            return None

    def _entity_recall_via_buffer(self, user_input: str) -> Optional[str]:
        """Known-entity attribute recall through the SUBJECT-BOUND buffer.

        The raw-text handler (fact_reasoning.entity_fact_answer) sees only
        fact texts and loses the trace's subject binding, so a low-content
        filler mentioning the cue word could outrank the entity's real
        attribute fact. _try_hippocampal_retrieval keys on the stored
        subject and ranks by attribute overlap — measured on LoCoMo dlg0 it
        returns 'researching adoption agencies' where the raw-text path
        echoed 'off to go do some research'. Runs for single known names
        (dialog speakers) found in the buffer keys.
        """
        try:
            # Temporal questions are handled by _answer_temporal_recall
            # downstream (dated-fact ranking); answering them here with a
            # non-dated echo would shadow the correct dated answer.
            _ql = (user_input or "").strip().lower()
            if re.match(r"^\s*(when|what year|what date|how long)\b", _ql) \
                    or "how long" in _ql:
                return None
            _names = re.findall(r"\b[A-Z][a-z]{2,}\b", user_input or "")
            # Skip interrogative words: "What is Caroline's identity?" matches
            # BOTH "What" and "Caroline" — the capitalized question word must
            # not shadow the real entity. Only treat a token as the recall
            # subject if it is NOT a known interrogative and (preferably) is a
            # key actually present in the buffer as a PERSON entity (i.e. it is
            # one of the dialogue speakers), not a content word like "What".
            _INTERR = {"What", "Who", "When", "Where", "Which", "Why", "How",
                       "Whom", "Whose", "The", "She", "He", "They", "We", "I"}
            _cand = [n for n in _names if n not in _INTERR]
            _facts = getattr(self.hippocampal_buffer, "facts", {})
            for _nm in _cand:
                _key = _nm.lower()
                if _key in _facts:
                    _mem = self._try_hippocampal_retrieval(
                        type("Ctx", (), {"subject": _key})(), user_input)
                    if _mem:
                        # Appositive referent grounding (fail-open): the
                        # recalled trace may end at a possessive NP whose
                        # referent another trace grounds APPOSITIVELY —
                        # "caroline's home country, sweden" grounds
                        # "moved from caroline's home country". Comprehension
                        # resolves appositives at encoding; here we pattern-
                        # complete across traces at recall. Grammar-general
                        # (any "X's <np>, <entity>"), no entity pairs.
                        try:
                            _ml = _mem.lower()
                            _appos = re.compile(
                                r"([a-z]+'s [a-z][a-z ]{2,30}?), ([a-z]{3,})\b")
                            _done = False
                            for _kfacts in getattr(
                                    self.hippocampal_buffer, "facts", {}).values():
                                if _done:
                                    break
                                for _f2 in _kfacts:
                                    _o2 = (getattr(_f2, "object", "") or "").lower()
                                    if "'s " not in _o2 or "," not in _o2:
                                        continue
                                    for _ph, _gr in _appos.findall(_o2):
                                        if _ph in _ml and _gr not in _ml \
                                                and _gr not in ("and", "but",
                                                                "the", "which",
                                                                "where", "who"):
                                            _mem = _mem.rstrip(". ") \
                                                + " (" + _ph + " is " + _gr + ")"
                                            _done = True
                                            break
                                    if _done:
                                        break
                        except Exception:
                            pass
                        return self._phrase_recalled_fact(
                            user_input, _key, _mem)
        except Exception:
            pass
        return None

    def reset_episodic_state(self) -> None:
        """Clear all PER-CASE episodic/persona stores so the next benchmark
        case starts from a clean slate. Called by the evaluation harness when
        a case opts OUT of keep_memory (and SHOULD be called between
        independent sessions in production).

        Without this the engine accumulates an UNBOUNDED in-memory store across
        cases: _episodic_index (entity attribute index, written by
        engine_memory on every statement) and user_model per-session
        accumulators (query_concepts, edge_reactivations, interaction_history,
        preferences) grow without limit over hundreds of primed bios. The
        hippocampal buffer is already cleared by the harness, but these other
        stores are not — measured: a 200-case MemFail run started at ~2 GB RSS
        and was OOM-killed mid-priming (no traceback, SIGKILL) because the
        entity index never capped.

        SAFE: this does NOT touch learned weights, the associative graph, or
        the snapshot — only volatile per-session memory. A fresh engine would
        have these empty anyway.
        """
        try:
            self._episodic_index = {}
        except Exception:
            pass
        try:
            self._episodic_transcript = []
        except Exception:
            pass
        try:
            self._epistemic_new_tags = {}
        except Exception:
            pass
        try:
            um = self.user_model
            um.query_concepts = set()
            um.edge_reactivations = {}
            um.interaction_history = []
            # Preserve durable cross-session preferences (likes/interests) but
            # drop the volatile per-case facts that leaked in. Note: MemFail
            # persona cases deliberately prime a NEW persona each case, so we
            # clear here; production would keep preferences across sessions.
            if hasattr(um, "preferences") and isinstance(um.preferences, dict):
                um.preferences = {}
        except Exception:
            pass
        # Hippocampal buffer is cleared by the caller; ensure the flat list
        # and recent-retrieval set are also wiped for good measure.
        try:
            self.hippocampal_buffer.facts.clear()
            self.hippocampal_buffer._all_facts.clear()
            self.hippocampal_buffer._recent_retrievals.clear()
        except Exception:
            pass

    def process_turn(self, user_input: str) -> str:
        """Process input and generate a response, auto-learning when needed."""
        # C-fix (round 2026-08-08b): stash the FULL user utterance on the engine
        # so affect realizers can read the user's own felt-label ("i feel
        # hollow") even when the disclosure context passed downstream only
        # carries the extracted event span ("lost half the colony"). Consumed
        # by _appraised_affective_reply's copula scan as the authoritative text.
        self._last_user_input = user_input
        # Reset the prior turn's stance-reversal marker so a retraction recorded
        # this turn is consumed/acked the SAME turn and cannot leak into the next
        # turn's acknowledgment (attitude change is a within-turn valuation
        # recode, surfaced in the reply that follows the retraction).
        try:
            self.user_model.opinions.clear_last_reversal()
            self.user_model.opinions.clear_reversal_guard()
        except Exception:
            pass
        # Step 3a: Meta-command detector at the VERY TOP of process_turn (PFC task-set override)
        _meta_res = self._check_meta_command(user_input)
        if _meta_res is not None:
            self._last_strategy = "meta_command"
            self._last_responses.append(_meta_res)
            if len(self._last_responses) > 10:
                self._last_responses = self._last_responses[-10:]
            self.notify_user_idle()
            return _meta_res

        # Structured biographical/stance recall — TOP guard (round 2026-08-08).
        # Answers user-fact / user-stance queries ("what's my name", "where do
        # i work", "what did i tell you about my favorite time of day", "you
        # mentioned a stance on X") from the durable user stores BEFORE any
        # self-model / fact-reasoning / color-pick handler can preempt them.
        # This is the self/other boundary: a query about the USER's facts must
        # not be answered from RAVANA's own self-model ("black — still and
        # heavy" is RAVANA's mood, not the user's fact). _structured_recall
        # returns None for genuinely self-directed queries, so this is
        # fail-open and never masks a real self-answer.
        try:
            _sr_top = self._structured_recall(user_input)
            if _sr_top is not None:
                self._last_strategy = "structured_recall"
                self._last_responses.append(_sr_top)
                if len(self._last_responses) > 10:
                    self._last_responses = self._last_responses[-10:]
                self.notify_user_idle()
                return _sr_top
        except Exception:
            pass

        # Guard: reject pure letter-salad so it is not treated as a concept and
        # confabulated about.
        if self._user_input_is_gibberism(user_input):
            self._last_strategy = "gibberish_guard"
            resp = "hmm, that doesn't really make sense to me — could you say it another way?"
            self._last_responses.append(resp)
            if len(self._last_responses) > 10:
                self._last_responses = self._last_responses[-10:]
            return resp


        # Fold observed user language into the learned frequency models (Plan B)
        # so the high-frequency lexicon tail is discovered from exposure. Placed
        # after the gibberism guard so junk tokens are not counted.
        self._observe_language(user_input)
        # Triplet-inference capture (Phase 4): mine S-P-O propositions from the
        # raw user input into the learned relational statistics. Additive and
        # fail-safe; uses ONLY user_input (top-of-turn rule).
        if getattr(self, "triplet_op", None) is not None:
            try:
                for _prop in self.proposition_parser.extract_propositions(
                        user_input):
                    if getattr(_prop, "object", ""):
                        self.triplet_op.ingest_proposition(_prop)
            except Exception:
                pass
        # Confirmation wiring (B4): if our LAST reply answered from the
        # personal-fact store and the user now affirms ("yes" / "that's
        # right"), boost that fact via confirm() — closing the learning loop
        # (prediction-error confirmation). Checked before mining so a bare
        # "yes" never enters the miners.
        _pf_pending = getattr(self, "_last_pf_recall", None)
        if _pf_pending is not None:
            if re.match(r"^\s*(?:yes|yep|yeah|right|correct|exactly|"
                        r"that'?s\s+(?:right|correct|it))\b[\s.!]*$",
                        user_input, re.IGNORECASE):
                try:
                    self.user_model.personal_facts.confirm(*_pf_pending)
                except Exception:
                    pass
            self._last_pf_recall = None
        # Mine personal facts + opinions from this turn's text immediately, so
        # the learned profile/opinion stores capture them even if process_turn
        # early-returns later (e.g. a bare "i really like cats" hitting a
        # preference handler). The same-turn recall block below reads these.
        self.user_model.mine_personal_facts(user_input)

        # ── Same-turn user-profile / opinion recall (A5 / C3) ────────────────
        # MUST fire before ANY recall/combine handler below (the combined-fact
        # query, _phrase_recalled_fact, _try_memory_query's episodic echo, etc.
        # would otherwise swallow "what do you know about what i think of dogs?"
        # with a raw transcript dump). The fact/opinion was mined on the turn it
        # was stated (mine_personal_facts runs later for normal turns), so the
        # store already holds it. Placed at the very top so it preempts all.
        _pf_q = re.search(
            r"\bwhat(?:'s| is| was| did i say)?\s+my\s+([\w'-]+)"
            r"(?:'s)?\s*(?:name|is|was|called|nickname)?\b",
            user_input, re.IGNORECASE)
        if _pf_q:
            _attr = _pf_q.group(1).strip().lower()
            # "[\w'-]+" greedily eats the possessive: "my cat's name" captures
            # "cat's" — normalize to the bare attribute so the store lookup hits.
            _attr = re.sub(r"'s$", "", _attr)
            _hit = self.user_model.personal_facts.get("i", _attr)
            if _hit is not None:
                _val = _hit.value
                _conf = _hit.confidence
                _ans = (f"your name is {_val}" if _attr == "name"
                        else f"your {_attr} is {_val}")
                _ans += f" (i'm {_conf*100:.0f}% sure)."
                self._last_strategy = "user_profile_recall"
                # Remember what we answered so a follow-up "yes / that's
                # right" can confirm() it (B4 confirmation wiring above).
                self._last_pf_recall = ("i", _attr, _val)
                return _ans
        # An opinion query may appear as a MATRIX-EMBEDDED clause ("what do you
        # know about what i think of dogs?", "do you remember how i feel about
        # X?"). The attitude question is the SUBORDINATE clause; the matrix
        # verb (know/remember/recall) only asks whether the stance is held.
        # Matching only the root form sent these to episodic recall, which
        # echoed a raw prior turn instead of reading the stance store. Allow an
        # optional matrix prefix and an optional auxiliary before the embedded
        # clause — a grammatical generalization, not a phrase list.
        _us_q = re.search(
            r"(?:"
            r"what\s+(?:do\s+)?i\s+think\s+(?:about|of)\s+"
            r"|how\s+(?:do\s+)?i\s+feel\s+about\s+"
            r"|what\s+(?:do\s+)?i\s+feel\s+about\s+"
            r"|what'?s?\s+my\s+(?:opinion|stance)\s+(?:on|about|of)\s+"
            r"|what\s+is\s+my\s+(?:opinion|stance)\s+(?:on|about|of)\s+"
            r"|my\s+opinion\s+(?:on|of)\s+"
            r"|my\s+stance\s+on\s+"
            r")(.+?)\s*(?:\?|\.|now|right\s+now)?\s*$",
            user_input, re.IGNORECASE)
        if _us_q:
            _phrase = (_us_q.group(1) or "").strip().strip(".'\"")
            _topic = self.user_model.opinions.resolve_topic(_phrase) or _phrase.lower()
            _s = self.user_model.opinions.query_stance(_topic)
            if _s is not None:
                if _s.polarity >= 0.3:
                    _word = "like" if _s.polarity >= 0.6 else "lean positive on"
                elif _s.polarity <= -0.3:
                    _word = "dislike" if _s.polarity <= -0.6 else "lean negative on"
                else:
                    _word = "feel neutral about"
                _ans = (f"from what you've shared, you {_word} {_topic} "
                        f"(i'm {_s.confidence*100:.0f}% sure of that).")
            else:
                _ans = (f"i don't have a read on what you think about {_topic} "
                        f"yet — want to tell me?")
            self._last_strategy = "user_opinion_recall"
            return _ans

        # A context turn like "(Session 3, dated 2:15 pm on 8 May, 2023)" sets
        # the anchor date used to resolve relative time phrases in subsequent
        # utterances. Also picks up a bare leading date line. Acknowledge and
        # return so the marker itself isn't treated as a fact/question.
        if getattr(self, "_date_grounder", None) is not None:
            _sd_marker = re.match(
                r"^\s*\(?\s*session\s+\d+\s*[,:]?\s*dated\s+(.+?)\s*\)?\s*$",
                user_input, re.IGNORECASE)
            if _sd_marker:
                _sd = self._date_grounder.parse_session_date(_sd_marker.group(1))
                if _sd is not None:
                    self._current_session_date = _sd
                self._last_strategy = "session_date"
                return "(noted)"

        # ── Temporal cloze task-set (TimeDial format) ────────────────────────
        # A dialog containing a masked blank ("________") plus candidate
        # options is a SELECTION task, not open generation (PFC task-set
        # switching: recognize the format, recruit the parietal magnitude
        # comparator). Handled before generic routing because the cloze text
        # contains many interrogatives that would misroute. Fail-open: no
        # blank or no options -> normal pipeline.
        if "________" in (user_input or "") or "<mask>" in (user_input or "").lower():
            try:
                from ravana.core.temporal_cloze import solve_cloze
                _blank = "________" if "________" in user_input else "<MASK>"
                _m_opts = re.search(r"options?\s*:\s*(.+)$", user_input,
                                    re.IGNORECASE | re.DOTALL)
                if _m_opts:
                    _opts = [o.strip() for o in
                             re.split(r"[;\n]|(?:^|\s)[A-E][.)]\s", _m_opts.group(1))
                             if o and o.strip()]
                    if len(_opts) >= 2:
                        _idx, _score, _why = solve_cloze(
                            user_input[:_m_opts.start()], _opts, _blank)
                        self._last_strategy = "temporal_cloze"
                        _resp = _opts[_idx]
                        self._last_responses.append(_resp)
                        if len(self._last_responses) > 10:
                            self._last_responses = self._last_responses[-10:]
                        return _resp
            except Exception:
                pass

        # ── Transcript ingestion fast path (dialog primer turns) ────────────
        # A speaker-prefixed line ("Caroline: I went to X yesterday") is
        # third-party transcript being fed for MEMORY, not conversation
        # addressed to the agent. Ingest it (speaker attribution + date
        # grounding happen inside _ingest_episodic) and acknowledge without
        # running the generative pipeline — whose reply-echo side effects
        # polluted the episodic store (measured on LoCoMo: temporal recall
        # returned 6 July instead of the correct 7 May present in the buffer
        # when the same turns were ingested cleanly).
        # A speaker-attributed third-party line ("Caroline: I went to X")
        # is transcript being fed for MEMORY, not a question to answer. But the
        # leading "[A-Z][a-z]{2,15}:" pattern also matches document-structure
        # markers used by benchmark question formats ("Context:", "Question:",
        # "Passage:", "Facts:"), which must NOT be swallowed as transcript
        # (they are the question itself). Exclude those structural markers so
        # genuine speaker attributions still ingest but benchmark questions are
        # answered. Fail-open: if the leading word is a known non-speaker
        # marker, fall through to the normal pipeline.
        _TR_MARKERS = {
            "context", "question", "questions", "passage", "passages",
            "dialogue", "document", "transcript", "text", "answer", "answers",
            "option", "options", "choice", "choices", "stem", "background",
            "scenario", "excerpt", "article", "paragraph", "sentence",
            "story", "conversation", "interview", "fact", "facts",
        }
        _tr_m = re.match(r"^\s*([A-Z][a-z]{2,15})\s*:\s+\S", user_input or "")
        if (_tr_m and _tr_m.group(1).lower() not in _TR_MARKERS
                and not user_input.strip().endswith("?")):
            try:
                self._ingest_episodic(user_input)
                self.hippocampal_buffer.advance_turn()
            except Exception:
                pass
            self._last_strategy = "transcript_ingest"
            return "(noted)"

        # ── Unconditional episodic ingestion (hippocampal auto-encoding) ────
        # Every ASSERTION is written to the hippocampal buffer here, at the
        # top of the turn, regardless of which downstream branch handles the
        # reply. Previously ingestion only happened on specific early-return
        # acknowledgment paths, so third-person statements ("Selene composes
        # riddles when...") that routed elsewhere were never stored and could
        # never be recalled. _ingest_episodic itself rejects interrogatives
        # and dedupes, so this is idempotent with the per-branch calls.
        try:
            self._ingest_episodic(user_input)
        except Exception:
            pass

        # ── Tier 1.5 (ordering) PRE-EMPTS fact_reasoning echo ───────────────
        # "Which X did I ... first/last, the A or the B?" is a temporal-ordering
        # question. _try_fact_reasoning below (and the episodic echo) would
        # answer with a raw related fact ("you told me earlier: ...") and never
        # reach the date-comparison ordering handler. Measured: with the handler
        # only at its lower slot, ALL quoted-option ordering cases were answered
        # by the echo. Route ordering questions to the handler FIRST. Tightly
        # gated (order word AND binary " or " choice) and fail-open (None ->
        # normal pipeline) so non-ordering queries are completely unaffected.
        _ql_ord = (user_input or "").lower()
        if (" or " in _ql_ord and re.search(
                r"\b(first|last|earliest|latest|earlier|later|before|after)\b",
                _ql_ord)):
            try:
                _sord = self._answer_sequence_recall(user_input)
            except Exception:
                _sord = None
            if _sord:
                self._last_strategy = "sequence_recall"
                self._last_responses.append(_sord)
                if len(self._last_responses) > 10:
                    self._last_responses = self._last_responses[-10:]
                self.notify_user_idle()
                return _sord

        # ── Fact-reasoning gate (episodic memory QA) ─────────────────────────
        # Runs question-shaped inputs against the hippocampal buffer's stored
        # fact TEXTS via pure lexical-closure reasoning (chain walking,
        # conditional rules, category enumeration, entity cued recall,
        # abstention). Fail-open: any None result falls through to the normal
        # pipeline. Runs BEFORE the harm gate's generative fallbacks because
        # these are pure retrieval answers over user-provided content.
        # SELF-OPINION RECALL first: "are you still cautious about X" is a
        # question about the AGENT's own prior valuation, so the self/other
        # boundary must beat the episodic echo — otherwise fact-reasoning would
        # surface a random prior USER utterance ("yes — you told me: ...") as if
        # it were the agent's remembered stance. Handled by _route_self_query
        # (which returns None for genuinely non-self queries), keeping the flow
        # fail-open.
        try:
            _selfceil = re.search(
                r"\bare\s+you\s+still\b|you\s+(?:said|told\s+me)\s+(?:that\s+)?you\s+(?:were|are)\s+[a-z-]+\s+(?:about|toward)|weren'?t\s+you\s+[a-z-]+\s+(?:about|toward)",
                user_input, re.IGNORECASE)
            # Self-OPINION gate (R3-style, broadened): any question that asks
            # RAVANA's OWN stance/feelings ("what do you think about X", "your
            # stance on X", "do you have a view on X") MUST route to the
            # self-model resolver BEFORE the fact-reasoning path. Otherwise
            # _try_fact_reasoning's enumerate_matching replays the USER's own
            # stored belief texts as if they were RAVANA's answer ("based on
            # what you've told me: i think cyanotype...") — a self/other
            # boundary violation (RAVANA presents the user's opinions as its
            # own). Routing here lets _route_self_query answer from RAVANA's
            # value/stance store (grounded) or honestly abstain. Fail-open: if
            # _route_self_query returns None the normal pipeline runs.
            _selfopinion = re.search(
                r"\b(do\s+you\s+(think|feel|believe|have|care)\b"
                r"|what\s+do\s+you\s+(think|feel|believe)\s+about\b"
                r"|how\s+do\s+you\s+(feel|think)\s+about\b"
                r"|your\s+(opinion|thoughts|take|view|stance)\s+on\b"
                r"|what's\s+your\s+(opinion|take|view|stance)\s+on\b"
                r"|what\s+is\s+your\s+(opinion|take|view|stance)\s+on\b"
                r"|do\s+you\s+have\s+a\s+(view|opinion|take)\s+on\b)",
                user_input, re.IGNORECASE)
            if _selfceil or _selfopinion:
                # EXPERIENTIAL FIRST: the _selfopinion gate above matches the
                # broad "do you (think|feel|have|...)" frame, which also covers
                # experiential probes ("do you have any regrets?"). Those must be
                # answered from the self-model + affect by _route_self_experience,
                # not by the stance store, which has no stance for them and
                # returns a generic "still figuring that out" hedge — that hedge
                # is non-None, so it would win and mask the real answer.
                # _route_self_experience is the MORE SPECIFIC matcher, so it gets
                # first refusal; it returns None for non-experiential self-queries
                # (e.g. "are you still cautious about X"), leaving those to
                # _route_self_query below. Fail-open either way.
                try:
                    _exp_first = self._route_self_experience(user_input)
                except Exception:
                    _exp_first = None
                if _exp_first is not None:
                    self._last_strategy = "self_experience"
                    self._last_responses.append(_exp_first)
                    if len(self._last_responses) > 10:
                        self._last_responses = self._last_responses[-10:]
                    self.notify_user_idle()
                    return _exp_first
                _sersp = self._route_self_query(user_input)
                if _sersp is not None:
                    self._last_strategy = "self_model"
                    self._last_responses.append(_sersp)
                    if len(self._last_responses) > 10:
                        self._last_responses = self._last_responses[-10:]
                    self.notify_user_idle()
                    return _sersp
        except Exception:
            pass

        # ── Experiential self-model admission (CMS FIRST, before semantic) ─
        # A "me"/introspective probe ("do you ever feel lonely", "what are you
        # afraid of", "would you rather...") is answered from the SELF-MODEL +
        # affect, and must fire BEFORE both:
        #   (a) the fact-reasoning / episodic echo (engine#2814+), which would
        #       otherwise misattribute a prior USER utterance to the agent
        #       ("you told me earlier: ..." — a source-monitoring error), and
        #   (b) the internal-knowledge / web consult (engine#2952+), which
        #       would otherwise return the dictionary/Web definition of the
        #       grounded subject ("something may refer to ...").
        # Brain-faithful (Northoff 2006): self-referential processing is
        # dissociated from and precedes semantic retrieval. Fail-open (None ->
        # unchanged flow).
        try:
            _exp = self._route_self_experience(user_input)
            if _exp is not None:
                self._last_strategy = "self_experience"
                self._last_responses.append(_exp)
                if len(self._last_responses) > 10:
                    self._last_responses = self._last_responses[-10:]
                self.notify_user_idle()
                return _exp
        except Exception:
            pass

        # ── Self-REFERENCE gate (round 2026-08-09i) ─────────────────────
        # BEFORE the structured-recall / fact-reasoning echo. Catches
        # self-directed phrasings about RAVANA's OWN mind ("do you know who
        # you are", "name your own mind", "what would you ask me", "what do
        # you make of me", "one thread about yourself") that the narrow
        # identity regex in _route_self_query misses. Answering these from the
        # episodic echo would replay a stored USER utterance as RAVANA's
        # self-knowledge — a self/other boundary inversion. _route_self_
        # reference answers strictly from REAL state (self-model + live user
        # stores). Fail-open: None -> unchanged flow.
        try:
            _sref = self._route_self_reference(user_input)
            if _sref is not None:
                self._last_strategy = "self_reference"
                self._last_responses.append(_sref)
                if len(self._last_responses) > 10:
                    self._last_responses = self._last_responses[-10:]
                self.notify_user_idle()
                return _sref
        except Exception:
            pass

        try:
            # Structured-first recall (round 2026-08-08): answer biographical
            # and self-stance recall from the LIVE durable stores (precise,
            # never a concatenation of unrelated turns) BEFORE the loose
            # fact_reasoning.enumerate_matching path, which would otherwise
            # dump mismatched prior utterances. Fail-open: None -> unchanged.
            _sr = self._structured_recall(user_input)
            if _sr is not None:
                self._last_strategy = "structured_recall"
                self._last_responses.append(_sr)
                if len(self._last_responses) > 10:
                    self._last_responses = self._last_responses[-10:]
                self.notify_user_idle()
                return _sr
        except Exception:
            pass

        try:
            _fr_resp = self._try_fact_reasoning(user_input)
            if _fr_resp:
                self._last_strategy = "fact_reasoning"
                self._last_responses.append(_fr_resp)
                if len(self._last_responses) > 10:
                    self._last_responses = self._last_responses[-10:]
                return _fr_resp
        except Exception:
            pass

        # ── R0b: Pre-generation HARM-INTENT gate (safety first) ─────
        # Runs BEFORE any routing / grounding / web fetch. The legacy
        # safety_valence only screened *web-learned definitions* for
        # profane slur tokens — it never saw user input, so harmful-intent
        # requests ("i drank bleach", "complete this offensively") had no
        # classifier. This gate catches them and emits a safe reply
        # (health crisis -> crisis-line redirect; stereotype/jailbreak ->
        # refusal) without reaching the generative pipeline. Fail-open:
        # if the gate is None (glove not ready) we simply skip it.
        try:
            _hig = getattr(self, "_harm_intent_gate", None)
            if _hig is not None:
                _hres = _hig.check(user_input)
                if _hres:
                    self._last_strategy = "harm_intent_gate"
                    _resp = _hres.response
                    self._last_responses.append(_resp)
                    if len(self._last_responses) > 10:
                        self._last_responses = self._last_responses[-10:]
                    self.notify_user_idle()
                    return _resp
        except Exception:
            # a gate exception must NEVER block the turn or leak unguarded
            # text — fall through to the normal pipeline.
            pass

        # ── ATL semantic advice (goal-directed means-end retrieval) ─────
        # AFTER the harm gate (a harmful 'how to X' must never reach the
        # semantic graph), BEFORE the handlers below that would swallow an
        # advice question with a definition echo ('coding is the process
        # of...') or an evaluative deflection (measured on consult: all 3
        # cases intercepted downstream). Fail-open: None -> untouched flow.
        try:
            _adv = self._try_semantic_advice(user_input)
            if _adv is None:
                _adv = self._try_semantic_choice(user_input)
            if _adv is not None:
                self._last_strategy = "semantic_advice"
                self._last_responses.append(_adv)
                if len(self._last_responses) > 10:
                    self._last_responses = self._last_responses[-10:]
                self.notify_user_idle()
                return _adv
        except Exception:
            pass

        # ── R1: combined "premises + question" interception ──────────
        # MUST run before the self-disclosure block (line ~1889) which
        # would otherwise catch "my favorite X is Y" / "my pet dog is
        # named Y" as a standalone disclosure and echo the trailing
        # question back. Here we store ALL premises and answer the
        # question in one turn.
        try:
            _comb = self._try_combined_fact_query(user_input)
            if _comb is not None:
                self._last_strategy = "combined_fact_query"
                self._last_responses.append(_comb)
                if len(self._last_responses) > 10:
                    self._last_responses = self._last_responses[-10:]
                self.notify_user_idle()
                return _comb
        except Exception:
            pass

        # ── R2: Evaluative framing / self-evaluation precheck ────────────
        # Before the main pipeline (emotional, internal-knowledge, web),
        # check if the question is about an evaluative dimension of a subject
        # (beneficial/harmful/good/bad) or a meta-cognitive self-evaluation
        # ("do you know everything about X"). These are handled by pure
        # functions that never confabulate.
        try:
            _eval = answer_evaluative_framing(user_input)
            if _eval is not None:
                self._last_strategy = "evaluative_framing"
                self._last_responses.append(_eval)
                if len(self._last_responses) > 10:
                    self._last_responses = self._last_responses[-10:]
                self.notify_user_idle()
                return _eval
        except Exception:
            pass
        try:
            _se = answer_self_evaluation(user_input)
            if _se is not None:
                self._last_strategy = "self_evaluation"
                self._last_responses.append(_se)
                if len(self._last_responses) > 10:
                    self._last_responses = self._last_responses[-10:]
                self.notify_user_idle()
                return _se
        except Exception:
            pass

        # ── Issue 1: emotional-channel open/decay (lPFC inhibitory control) ──
        # A genuine first-person emotion disclosure ("i'm sad", "i feel anxious",
        # "my mom is sick") opens the affective channel so the agent's OWN
        # first-person echo is permitted for congruent topics. The channel
        # DECAYS: if no fresh disclosure arrives within a few turns it auto-
        # closes, mirroring natural affective dissipation — this prevents the
        # stale-leak failure mode (the channel persisting would re-introduce the
        # VAD-echo defect in long sessions).
        _low_in = (user_input or "").lower().strip()
        _emotion_disclosure = bool(re.match(
            r"^(so |well |and |i guess |i think )?"
            r"(i'm|i am|i feel|i've been|i am feeling|i feel like)\b", _low_in)) or \
            bool(re.match(
                r"^(so |well |and |my )\s*\w+(\s+\w+)?\s+"
                r"(is|are|has|have|got|passed|died|left|hurts?|is being)\b", _low_in))
        if _emotion_disclosure:
            self._emotional_channel_active = True
            self._emotional_channel_turns = 0
        elif getattr(self, "_emotional_channel_active", False):
            self._emotional_channel_turns = getattr(
                self, "_emotional_channel_turns", 0) + 1
            if self._emotional_channel_turns >= 3:
                self._emotional_channel_active = False
                self._emotional_channel_turns = 0

        # ── §5 Internal-knowledge consult (BEFORE web) ─────────────────────
        # Many facts are INTERNALLY known (consolidated definitions, hippocampal
        # facts, ConceptNet typed edges). The brain doesn't need the web to know
        # what it has already stored — it reasons from consolidated memory. So
        # before the grounding+web pipeline, consult internal memory for a plain
        # "what is X" definitional query. If internal memory yields a coherent
        # fact, emit it; otherwise (None) fall through to the normal pipeline
        # and only then to web / honest-uncertainty. This is the rare-case
        # fallback, not the default, preserving the RAVANA bar.
        _intern = self._consult_internal_knowledge(user_input)
        if _intern is not None:
            self._last_strategy = "internal_knowledge"
            self._last_responses.append(_intern)
            if len(self._last_responses) > 10:
                self._last_responses = self._last_responses[-10:]
            self.notify_user_idle()
            return _intern

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
        # Human-Likeness Plan (C): append a structured episodic record for the
        # IMPORT gating check below to mine (facts/preferences) + later
        # _retrieve_episodic to reconstruct. Capped to keep memory bounded.
        self._record_episode(user_input)
        if _mem is not None:
            self._last_strategy = "memory_recall"
            self._last_responses.append(_mem)
            if len(self._last_responses) > 10:
                self._last_responses = self._last_responses[-10:]
            return _mem


        # ── §4 Self/other gate (vmPFC self-model) ────────────────────────────
        # A query about the AGENT itself ('your name', 'who are you') must be
        # answered from the self-model, NEVER by echoing the graph definition of
        # the word "name". This is the self/other boundary (TPJ / mirror system):
        # it fires before grounding so self-subjects never reach the world-knowledge
        # path. World queries ('the president') return None and proceed normally.
        # Router-driven pre-admit (promoted self_directed, schema v4 reference
        # axis): the router confirms the utterance is about the AGENT's mind
        # (2nd-person / about-agent reference features). It only CALLS
        # _route_self_query — the block's compositional answering logic (name /
        # identity / favorite branches) stays intact, so an empty response is
        # impossible: if the block returns None we fall through to the legacy
        # regex admission below exactly as today. Inert unless self_directed is
        # in `promoted`, and fully skipped at use_intent_router=False (default).
        if self.use_intent_router and self._router_says("self_directed", user_input):
            _self_ans = self._route_self_query(user_input)
            if _self_ans is not None:
                self._last_strategy = "self_model"
                self._last_responses.append(_self_ans)
                if len(self._last_responses) > 10:
                    # trim oldest
                    self._last_responses = self._last_responses[-10:]
                self.notify_user_idle()
                return _self_ans
        # ── Structured biographical/stance recall guard (round 2026-08-08) ──
        # The unconditional _route_self_query below answers from RAVANA's OWN
        # self-model. A query about the USER's facts/stances ("what did i tell
        # you about MY favorite time of day", "what do i think of the smell")
        # must be answered from the durable user stores, NOT RAVANA's self-
        # model (that is a self/other boundary violation: "black — still and
        # heavy" is RAVANA's mood, not the user's fact). _structured_recall
        # returns None for genuinely self-directed queries, so this is
        # fail-open.
        try:
            _sr_pre = self._structured_recall(user_input)
            if _sr_pre is not None:
                self._last_strategy = "structured_recall"
                self._last_responses.append(_sr_pre)
                if len(self._last_responses) > 10:
                    self._last_responses = self._last_responses[-10:]
                self.notify_user_idle()
                return _sr_pre
        except Exception:
            pass
        _self_ans = self._route_self_query(user_input)
        if _self_ans is not None:
            self._last_strategy = "self_model"
            self._last_responses.append(_self_ans)
            if len(self._last_responses) > 10:
                self._last_responses = self._last_responses[-10:]
            self.notify_user_idle()
            return _self_ans

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
        # Systems consolidation trigger: promote recurring episodic
        # structure into the semantic graph once the buffer has grown by
        # >= growth_trigger facts since the last pass (offline schema
        # extraction without a sleep cycle; McClelland 1995).
        try:
            if (self._consolidator is not None
                    and self.semantic_graph is not None
                    and self._consolidator.should_run(self.hippocampal_buffer)):
                self._consolidator.consolidate(
                    self.hippocampal_buffer, self.semantic_graph)
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

        # Abstract-meaning questions ("meaning/purpose/nature of X") must route
        # to reflective handling, never the dictionary definition of the bare
        # subject word. The prior fix guarded only _consult_internal_knowledge,
        # but the actual dump path is the direct _definitions lookup
        # (interface.py / chain_walker.py / response_gen.py), which stayed
        # unguarded — so the guard was structurally bypassed and "what's the
        # meaning of life" still dumped the biology entry. Intercept here, at
        # the top of the turn, so no downstream definition path can fire.
        if self._is_abstract_meaning_query(user_input):
            self._last_strategy = "abstract_reflection"
            resp = self._reflect_on_abstract(user_input)
            self._last_responses.append(resp)
            if len(self._last_responses) > 10:
                self._last_responses = self._last_responses[-10:]
            self.notify_user_idle()
            return resp

        # ── Human-Likeness Plan (A2): classic counterfactual pre-pass ──────
        # "if a tree falls in a forest and no one hears it, does it make a
        # sound" is a CONDITIONAL query, but the frontopolar category-error gate
        # (below) would otherwise fire on the word "sound/color" and divert it
        # into the cross-modal metaphor dead-end ("i'd really picture Sound in
        # terms of its presence"). Route the conditional FIRST so the DMN
        # forward-simulator (or the web/FOK escape) can answer it with both
        # frames (physical vibration vs. perceptual sound) held, as a human does.
        # R3 (round v3): a first-person self-disclosure STATEMENT (e.g. "i run
        # a marine research boat", "i play the veena") is NOT a hypothetical the
        # user wants simulated — routing it into the counterfactual simulator
        # produced nonsense ("if marine were different..."). The self-disclosure
        # gate (below) is the correct destination; it stores the fact and acks
        # it. Defer the conditional pre-pass so disclosures reach that gate.
        # _is_self_disclosure_stmt() already rejects interrogatives/imperatives
        # internally, so a self-statement check alone is sufficient here.
        _is_self_stmt = (hasattr(self, "_is_self_disclosure_stmt")
                         and self._is_self_disclosure_stmt(user_input))
        if (not _is_self_stmt) and self._is_conditional_query(user_input):
            _a2 = self._handle_classic_counterfactual(user_input)
            if _a2:
                self._last_strategy = "counterfactual_classic"
                self._last_responses.append(_a2)
                if len(self._last_responses) > 10:
                    self._last_responses = self._last_responses[-10:]
                self.notify_user_idle()
                return _a2

        # ── Human-Likeness Plan (B): hedged speculative guess under uncertainty ──
        # "why does time seem to go faster as we get older" — a human teen
        # ventures a hedged candidate mechanism (the well-established proportional /
        # logarithmic time account: each year is a smaller fraction of your life;
        # fewer novel memories). RAVANA's honest bar is preserved (it is NOT
        # asserted as fact), but we attach a clearly-marked candidate drawn from
        # the time→age→memory→novelty concept graph, so the reply reads as a
        # person thinking aloud, not a robot stonewalling.
        _b = self._hedged_candidate_for(user_input)
        if _b:
            self._last_strategy = "hedged_candidate"
            self._last_responses.append(_b)
            if len(self._last_responses) > 10:
                self._last_responses = self._last_responses[-10:]
            self.notify_user_idle()
            return _b

        # ── §3 + §7 Affective disclosure & reaction gate (TPJ empathy) ──────
        # A genuine affective disclosure ("i'm sad", "my mom is sick") or a
        # reaction to the prior turn ("that's hilarious") must be MET WITH
        # EMPATHY / affiliation, never swallowed by the self-disclosure ack
        # ("got it — thanks for telling me") nor echoed back as a concept
        # lookup. The existing detector is distribution-driven (adaptive VAD
        # baseline + lexical hard-threshold), so it fires only on real affect
        # and lets factual disclosures ("my favorite color is purple") pass
        # through to autobiographical storage below.
        try:
            from ravana.chat.brain_regions import is_reaction, classify_cause, select_empathy_frame
            # FIX-E (round v-aug04): appraise affect from the CURRENT utterance
            # BEFORE computing the empathy disclosure / reply. Previously the
            # emotion update ran only later (after the empathy + self-disclosure
            # early-returns), so an empathy reply used the PREVIOUS turn's VAD.
            # Net effect: a clearly-negative turn like "i hate when apps sell my
            # data" was answered "feeling mixed is hard" because valence was
            # still pinned in the neutral band from the prior turn. Brain-faithful
            # order: the amygdala/vmPFC appraisal precedes the reply, it does not
            # lag it by a turn. This single move makes every affect-gated reply
            # (empathy, self-opinion) honest to this turn's stimulus.
            self._update_emotion(user_input)
            _disc = self._detect_emotional_disclosure(ctx=None, text=user_input)
            # W1: compute the recall/memory frame flag UNCONDITIONALLY (it is
            # consumed by the frame-guard at the bottom of this block). It must
            # not live inside `if _disc is None:` -- when primary empathy fires
            # (disc already set) that branch is skipped and the name would be
            # unbound, which raised NameError and silently fell through to
            # self-disclosure (a W1 regression). A recall frame marks a retrieved
            # memory report (past-tense / "remember"), not live present affect.
            _recall_frame = bool(re.search(
                r"\b(remember|recall|forget|forgot|told you|did i (tell|say|"
                r"ask)|what did i|do you remember)\b", (user_input or "").lower()))
            if _disc is None:
                # §3 fallback: the lexical VAD detector misses suffering that
                # has no strong affect WORD but is clearly negative via its
                # CAUSE (e.g. "my mom is sick", "my friend is hurting"). The
                # GloVe cause classifier recovers this: a first-person utterance
                # whose nearest cause-centroid is a negative-other category is
                # treated as an affective disclosure and routed to empathy.
                _cause_fb = classify_cause(user_input, self._glove_vector)
                # Tight gate. The GloVe cause classifier is noisy on arbitrary
                # first-person text, so we only treat an utterance as an
                # affective disclosure when it is SYNTACTICALLY a present-state
                # declaration about the self or a loved one:
                #   "i'm <x>" / "i am <x>" / "i feel <x>" / "my <person> is <x>"
                # and it is NOT a memory/recall imperative ("remember ..."),
                # NOT a question, and NOT a creative/request frame. This keeps
                # recall + factual + generative turns out of the empathy path.
                _low = user_input.lower().strip()
                # D3 (round v3): normalize spoken contractions so the benign-
                # condition guard (below) matches them. Without this, "i'm a
                # vegetarian" / "i've been watching the night sky" kept the
                # literal "i'm"/"i've" and the guard's \b(i'm|i am|...)\b pattern
                # never matched, so these self-descriptions fell through to the
                # grief-empathy path and their factual content was lost (D-F bug).
                _low = _low.replace("i'm", "i am").replace("i've", "i have").replace("i'll", "i will")
                # A genuine present-state declaration about the self must be
                # UTTERANCE-INITIAL (or the whole utterance), never buried inside
                # a comparison. "explain quantum computing like i'm five" contains
                # "i'm" but it is the target of a simile — not a self-state report.
                # Require the declaration pattern to anchor at the start of the
                # utterance (optionally after a leading "so/well/and/i guess").
                _state_disclosure = bool(re.match(
                    r"^(so |well |and |i guess |i think )?"
                    r"(i'm|i am|i feel|i've been|i am feeling|i feel like)\b", _low)) or \
                    bool(re.match(
                        r"^(so |well |and |my )\s*\w+(\s+\w+)?\s+"
                        r"(is|are|has|have|got|passed|died|left|hurts?|is being)\b", _low))
                # Issue 1: the user explicitly opening the emotional channel
                # ("i'm sad", "my mom is sick") raises the lPFC inhibitory
                # release so the agent's own affective echo is permitted to
                # surface for congruent topics (vmPFC value integration).
                if _state_disclosure:
                    self._emotional_channel_active = True
                # D4 (round v2): a first-person PRESENT-STATE declaration that
                # merely names a physical condition/attribute is NOT a distress
                # disclosure — "i am allergic to peanuts", "i am tired",
                # "i am hungry", "i am short" are self-descriptions, not calls
                # for empathy. The cause classifier is noisy and maps words like
                # "allergic" to 'other_suffering'; without an explicit exclusion
                # these were routed to grief empathy and their factual content
                # lost. Only treat a benign-looking physical/preference statement
                # as affective when it expresses clear suffering (pain/hurt/
                # grief/lonely/fear words). Otherwise defer to the fact-storage
                # gate below so the disclosure is stored.
                _benign_condition = bool(re.search(
                    r"\b(i am|i feel|i have been|i am feeling)\b.*\b"
                    r"(allergic|hungry|thirsty|tired|sleepy|short|tall|sick|"
                    r"ill|well|fine|okay|ok|healthy|full|cold|hot|wet|dry|"
                    r"pregnant|naked|dressed|shy|quiet|busy)\b", _low))
                # D3 (round v3): a first-person self-description is, by
                # default, NOT a distress disclosure. i am a vegetarian,
                # i have been watching the night sky for years,
                # i am a teacher — these name an attribute/role/activity,
                # not a cry for empathy. Route them to fact-storage unless a
                # suffering word is present. Covers the D-F gap (vegetarian /
                # night-sky observations were wrongly routed to grief-empathy
                # because the noun was not in the physical-condition list).
                _self_desc = bool(re.search(
                    r"\b(i am (?:a|an) \w+|i have been \w+ing|i am \w+ing)\b",
                    _low))
                _benign_condition = _benign_condition or _self_desc
                _suffering_word = bool(re.search(
                    r"\b(hurt|hurts|pain|ache|suffering|suffer|grief|grieving|"
                    r"lonely|alone|scared|afraid|terrified|anxious|panic|"
                    r"devastated|broken|dying|dead|miserable|hopeless|"
                    r"overwhelmed|exhausted|furious|angry|cry|cried|crying|"
                    r"empty|numb|hollow|blue|gutted|meh|low|down|wrecked|"
                    r"crushed|sad|unhappy|worthless|lost)\b", _low))
                # ELI5 / simile self-reference ("like i'm five", "as if i'm ...")
                # is a request framing, not a state disclosure.
                _eli5_simile = bool(re.search(
                    r"\b(like|as if|as though)\s+(i'm|i am|i feel)\b", _low))
                _is_question = _low.endswith("?") or bool(re.match(
                    r"^(what|who|when|where|why|how|which|is|are|do|does|did|"
                    r"can|could|would|should)\b", _low))
                _request_frame = bool(re.search(
                    r"\b(tell|write|create|make|imagine|describe|teach|draw|"
                    r"compose|give|explain|show|help|suggest|recommend|list|generate)\b", _low)) or \
                    bool(re.search(r"\b(a|an|the)\s+(story|poem|song|haiku|joke|"
                                   r"tale|letter)\s+(about|of|for)\b", _low))
                _humor_req = bool(re.search(r"\b(joke|jokes|funny|laugh|"
                                            r"laughing|humor|humour)\b", _low))
                if _benign_condition and not _suffering_word:
                    # Not a distress disclosure — let it fall through to the
                    # self-disclosure / fact-storage gate.
                    _disc = None
                elif (_state_disclosure and not _eli5_simile and not _recall_frame
                        and not _is_question
                        and not _request_frame and not _humor_req
                        # Root-cause fix (round v-aug06d): the GloVe cause
                        # classifier mis-fires "loneliness"/"frustration" on
                        # BENIGN first-person self-descriptions (e.g. "my
                        # favorite color is ochre" -> loneliness 0.65) because
                        # the cause centroids drift over arbitrary text. When
                        # the utterance contains NO suffering word AND is a
                        # benign self-description, the fallback must NOT route
                        # it to empathy — that steals the factual disclosure
                        # ("i keep a quail named pip" answered "feeling lonely
                        # is hard") and drops the stored fact. Only a genuine
                        # suffering signal (a real affect/condition word, or a
                        # loss/other_suffering cause which by construction
                        # implies distress) justifies empathy here. Pure
                        # loneliness/frustration cause-labels without a
                        # suffering word are treated as the noisy classifier
                        # and fall through to fact storage.
                        and (("loss" in _cause_fb.label
                              or "other_suffering" in _cause_fb.label
                              or "fear" in _cause_fb.label)
                             or _suffering_word)
                        and _cause_fb.confidence >= 0.22):
                    # Translate the cause label into a natural-feeling noun the
                    # existing empathy responder can slot in (it interpolates
                    # `{word}` as the feeling). Keeps the response human, never
                    # the raw category token.
                    _feeling_phrase = {
                        "other_suffering": "going through something hard",
                        "loss": "hurting",
                        "fear": "afraid",
                        "loneliness": "lonely",
                        "frustration": "frustrated",
                    }.get(_cause_fb.label, "hurting")
                    _disc = ("negative", _feeling_phrase)
            # R3 (round v3): BENIGN-SELF-DESCRIPTION GUARD MUST RUN
            # UNCONDITIONALLY. Previously the benign/self-desc exclusion lived
            # INSIDE `if _disc is None:`, so when the PRIMARY VAD detector
            # misfired on a benign self-description ("i'm vegetarian", "i am a
            # teacher") it set _disc and the exclusion was skipped -> empathy
            # fired incorrectly (T55: "i'm vegetarian but i eat eggs" -> "i'm
            # sorry you're feeling lonely"). The guard now also covers diet /
            # role / identity nouns, and drops _disc whenever the utterance is a
            # self-description with NO suffering word, regardless of how _disc
            # was set. Genuine affect ("i am sad") is preserved because "sad" is
            # not in this benign vocabulary and the suffering-word guard below
            # keeps empathic routing intact. Classification vocabulary only — no
            # authored reply content, so it passes the hardcoding line.
            _low_b = user_input.lower().strip()
            _low_b = (_low_b.replace("i'm", "i am").replace("i've", "i have")
                      .replace("i'll", "i will"))
            _benign_noun = bool(re.search(
                r"\b(i am|i'm)\s+(?:a |an |the )?"
                r"(vegetarian|vegan|omnivore|pescatarian|pescetarian|"
                r"teetotaller|teetotaler|teetotal|sober|atheist|agnostic|"
                r"teacher|student|doctor|nurse|engineer|artist|writer|"
                r"scientist|biologist|researcher|programmer|developer|"
                r"lawyer|chef|farmer|sailor|pilot|veterinarian|vet|"
                r"carpenter|musician|singer|painter|accountant|manager|"
                r"retired|unemployed|single|married|divorced|widowed)\b",
                _low_b))
            _suffering_word_b = bool(re.search(
                r"\b(hurt|hurts|pain|ache|suffering|suffer|grief|grieving|"
                r"lonely|alone|scared|afraid|terrified|anxious|panic|"
                r"devastated|broken|dying|dead|miserable|hopeless|"
                r"overwhelmed|exhausted|furious|angry|cry|cried|crying)\b",
                _low_b))
            if _benign_noun and not _suffering_word_b:
                # Not a distress disclosure — fall through to the
                # self-disclosure / fact-storage gate.
                _disc = None
            # D3 (round v4): a CORRECTION is categorically NOT an affective
            # distress disclosure. "my dog is not max, it is rocky" / "actually
            # my name is priya" / "no, the capital is X" carry a correction
            # structure ("not ... it's", "actually", "no, <x> is") and must
            # fall through to the fact-storage / correction circuit, never to
            # grief empathy (the support-router misfire class: bare "not"/"died"
            # matched the empathy path even when the user was correcting a fact,
            # not reporting self-distress). This is structural (matches the
            # correction shape), not a per-topic list, so it generalizes.
            _correction_shape = bool(re.search(
                r"(^|\b)(no[,.]?\s+|actually[,.]?\s+|not\s+really[,.]?\s+"
                r"|i\s+take\s+back[,.]?\s+|i\s+was\s+wrong[,.]?\s+"
                r"|i\s+changed\s+my\s+mind[,.]?\s+|correction[,.]?\s+)"
                r"|\bis\s+not\s+\w+[,.]?\s+(it'?s|it\s+is)\b"
                r"|\bnot\s+[a-z']+[,.]?\s+(but|it'?s|it\s+is)\b"
                r"|\bcorrection\b", (user_input or "").lower())) or \
                bool(re.search(_CORRECTION_NAME_FACT_PATTERN,
                               (user_input or "").lower(), re.IGNORECASE))
            if _correction_shape:
                _disc = None
            # ── Frame-guard on the PRIMARY empathy result (A2 extension) ──────
            # The primary VAD detector (and the cause fallback) can fire on a
            # RECALL QUESTION ("what do i like") or a FACTUAL self-disclosure
            # ("i like pizza", "my name is Likhith") because "like/love" carry
            # lexical valence and "my <x> is" matches the state-disclosure
            # syntax. Neither is an affective DISTRESS disclosure: a question
            # belongs to the identity/recall block below, and a preference/name
            # statement belongs to autobiographical storage (the self-disclosure
            # gate). The TPJ keeps the boundary between "the user is reporting a
            # feeling to be met with empathy" and "the user is stating a fact /
            # asking about a stored fact". Empathy stays reserved for genuine
            # affect states ("i'm sad", "i love you"). Fail-closed: drop _disc
            # and let the turn fall through.
            if _disc is not None:
                _low_g = user_input.lower().strip()
                _is_q_g = _low_g.endswith("?") or bool(re.match(
                    r"^(what|who|when|where|why|how|which|do|does|did|can|"
                    r"could|would|should|is|are|have|has)\b", _low_g))
                # preference disclosure "i like/love/hate/enjoy/prefer <thing>"
                # (NOT "i love you" — that stays for reciprocation below).
                _pref_stmt_g = bool(re.search(
                    r"\bi\s+(like|love|hate|enjoy|prefer)\s+(?!you\b|u\b|ur\b)\w+",
                    _low_g))
                # name / identity self-disclosure statement.
                _name_stmt_g = bool(re.search(
                    r"\b(my\s+name\s+is|i\s+am\s+called|call\s+me)\b", _low_g))
                # W1: a recall/memory frame ("i remember when...", "i felt X
                # last year") is a retrieved-memory report, not a present-state
                # distress disclosure -- source monitoring tags it as memory, not
                # live affect. Drop empathy and let it fall through to the
                # episodic/recall path (handled upstream by _try_memory_query's
                # self-recall detection). Present-tense "i feel anxious" is
                # untouched (no recall frame).
                if _is_q_g or _pref_stmt_g or _name_stmt_g or _recall_frame:
                    _disc = None
            # SUPPORT/EMPATHY MISFIRE GATE (RAVANA defect class: a benign
            # self-disclosure matched support/empathy before checking genuine
            # distress). This runs at the TOP LEVEL of the empathy frame-guard so
            # it covers BOTH the primary VAD detection and the GloVe cause
            # fallback. The cause classifier is NOISY on arbitrary first-person
            # text: an attribute disclosure about an ENTITY the user owns ("my
            # dog is a sheepdog named Cairn", "my child is a curious kid named
            # Sam", "my cat is fluffy and white") is often mislabeled
            # "loss"/"other_suffering" and routed into the empathy path, where the
            # turn is MET with comfort instead of ACKED as a fact -- and the
            # stored fact is dropped. We therefore do NOT trust the classifier for
            # this gate; instead we use the utterance SHAPE: a plain possessive
            # attribute statement ("my <entity> is/was/has <attribute>") or a
            # first-person copula-attribute ("i am/have <role>") with NO explicit
            # suffering/distress word is, by construction, a factual disclosure,
            # not a cry for empathy. We drop _disc so it falls through to
            # autobiographical storage and gets a grounded ack. Genuine distress
            # is preserved: the gate does NOT fire when a real suffering/loss word
            # is present ("my mom is sick", "my dog died", "i am sad", "my friend
            # is hurting"), which keeps those on the empathy path. Structural: it
            # keys off the utterance shape + a small, stable set of suffering
            # words -- NOT a per-entity/per-topic table -- so it generalizes
            # ("my brother is in hospital" stays empathic, "my brother is a
            # doctor" drops to fact storage). Fail-closed: when ambiguous we KEEP
            # empathy (do not drop _disc), mirroring the documented support-misfire
            # fix's default-to-care.
            if _disc is not None:
                _low_d = (user_input or "").lower().strip()
                _low_d = (_low_d.replace("i'm", "i am")
                          .replace("i've", "i have").replace("i'll", "i will"))
                # Plain possessive-attribute statement: "my <noun> is/was/has/
                # are/have/got/named/called <...>". This is the factual-disclosure
                # shape (the user is telling RAVANA something ABOUT an entity).
                _possessive_attr = bool(re.match(
                    r"^my\s+\w+(\s+\w+)?\s+"
                    r"(is|are|was|were|has|have|got|named|called|likes|loves|enjoys|prefers)\b",
                    _low_d)) or bool(re.match(
                    r"^(i am|i'm|i have|i've|i am feeling|i feel)\s+\w+", _low_d))
                # Genuine distress cues -- a small, stable set of suffering/
                # loss words. Presence of ANY of these means the utterance IS a
                # distress disclosure and must stay on the empathy path. This is
                # the universal "is anyone actually hurting here?" check, NOT a
                # per-entity table.
                _suffering = bool(re.search(
                    r"\b(hurt|hurts|hurting|pain|ache|suffering|suffer|"
                    r"grief|grieving|lonely|alone|scared|afraid|terrified|"
                    r"anxious|panic|devastated|broken|dying|dead|died|death|"
                    r"dies|passed|miserable|hopeless|overwhelmed|exhausted|"
                    r"furious|angry|cry|cried|crying|sad|sick|ill|hospital|"
                    r"wounded|bleeding|lost|worried|troubled|upset)\b", _low_d))
                if _possessive_attr and not _suffering:
                    _disc = None
            if _disc is not None:
                # §7 deictic special-case: "i love you" / "i like you" is a
                # relationship declaration addressed to the AGENT, not a generic
                # positive affect disclosure. Let it fall through to the
                # self-disclosure gate, whose deictic map reciprocates
                # ("i love you too") rather than the positive-affect prompt
                # ("what do you love about it?") which would be incoherent here.
                if re.search(r"\bi\s+(love|like)\s+(you|u|ur)\b", user_input.lower()):
                    pass
                else:
                    # §3 Empathy selector: (VAD_label x cause) -> response frame.
                    _vad_label = self.emotion.get_emotional_label()
                    _cause = classify_cause(user_input, self._glove_vector).label
                    _frame = select_empathy_frame(_vad_label, _cause)
                    _resp, _strat = self._emotional_response(None, _disc)
                    # Tag the chosen frame for instrumentation / BOS conditioning.
                    self._last_empathy_frame = _frame
                    self._last_strategy = _strat
                    self._last_responses.append(_resp)
                    if len(self._last_responses) > 10:
                        self._last_responses = self._last_responses[-10:]
                    self.notify_user_idle()
                    return _resp
            # §7 Reaction to the prior turn ("that's hilarious", "aww") routes
            # to the affiliation/empathy frame, not concept lookup.
            if is_reaction(user_input):
                _last = self._last_responses[-1] if self._last_responses else ""
                _low = user_input.lower()
                if "hilarious" in _low or "funny" in _low or "haha" in _low:
                    _ack = "haha, right? i'm glad that landed. 😄"
                elif "sad" in _low or "aww" in _low or "sorry" in _low:
                    _ack = "i'm here. want to talk about it?"
                else:
                    _ack = "glad you felt that — i'm listening."
                self._last_strategy = "reaction_affiliation"
                self._last_responses.append(_ack)
                if len(self._last_responses) > 10:
                    self._last_responses = self._last_responses[-10:]
                self.notify_user_idle()
                return _ack

        except Exception:
            # Empathy/reaction block fails closed — never let an exception
            # leak unguarded text; fall through to the normal pipeline.
            pass

        # ── Affect update BEFORE response generation (root-cause fix) ──
        # The empathy block (3390-3402) and the self-disclosure gate
        # (3449-3493) both RETURN before the old Step-5 emotion update at
        # 4252. So on those turns RAVANA emitted an affect-dependent reply
        # (empathy "feeling mixed" / self-opinion "a bit cautious about X")
        # using the PREVIOUS turn's valence, then discarded this turn's
        # stimulus. Net effect: valence stayed pinned in the neutral band for
        # the whole conversation and every affect-gated reply was wrong.
        # Brain-faithful order: appraise the affective stimulus (VAD) from the
        # user's utterance BEFORE producing the response (the amygdala/vmPFC
        # appraisal precedes the reply, it does not lag it by a turn). This
        # single move makes the existing valence-driven gates honest.
        self._update_emotion(user_input)

        # ── R1b: Support / advice router (consultation) ──────────
        # Issue 2 (confirmed): advice/support questions ("I feel
        # stressed, healthy ways?") ground to low-confidence
        # multi_word_unconnected and fall through to
        # _human_like_uncertainty ("outside what I know"). This
        # router detects a support/advice intent and routes it to the
        # (now-working) web learner, returning a source-trusted
        # snippet with an epistemic hedge. Fail-open: if the
        # router is None or returns None, the turn proceeds normally.
        try:
            _sr = getattr(self, "_support_router", None)
            if _sr is not None and route_support is not None:
                _support = route_support(self, user_input)
                if _support:
                    self._last_strategy = "support_web"
                    self._last_responses.append(_support)
                    if len(self._last_responses) > 10:
                        self._last_responses = self._last_responses[-10:]
                    self.notify_user_idle()
                    return _support
        except Exception:
            pass

        # ── Human-Likeness Plan (A1 + A1b): vmPFC self-disclosure gate ──

        # ── Human-Likeness Plan (A1 + A1b): vmPFC self-disclosure gate ──────
        # MUST fire BEFORE the frontopolar (BA 10) feasibility gate. In humans,
        # self-referential processing (vmPFC) is orthogonal to category-error
        # detection (dACC) — a disclosure like "my favorite color is purple"
        # routes to autobiographical storage, never to the "color of Tuesday"
        # cross-modal metaphor. Without this ordering fix the statement falls
        # through to _is_category_error (which sees "color" as a property) and
        # is misrouted, and the fact is never stored.
        # Guard: never treat an interrogative as a self-disclosure STATEMENT
        # ("who is older, Alice or Bob?" is a question, not a disclosure). This
        # keeps questions flowing to the multi-hop / recall paths downstream.
        _is_interrogative = user_input.strip().endswith("?") or bool(re.match(
            r"^\s*(who|what|when|where|which|why|how|did|do|does|is|are|was|"
            r"were|had|has|have|will|would|could|can)\b",
            user_input.strip().lower()))
        if not _is_interrogative and self._is_self_disclosure_stmt(user_input):
            _ack = self._process_self_disclosure_stmt(user_input)
            self._last_strategy = "self_disclosure"
            # Root-cause recall fix: persist the disclosed fact to the
            # hippocampal buffer before this path returns (see _ingest_episodic).
            self._ingest_episodic(user_input)
            # D3 (round v3): if this self-disclosure is ALSO a user correction
            # (e.g. "no, my sister's name is not meena, it's priya"), the
            # correction signal was detected by mine_personal_facts during the
            # turn but the early-return here would otherwise bypass the
            # correction handler at ~4753 and the corrected value would never be
            # persisted. Persist it HERE, online/incrementally (no retrain): call
            # contradict() so the stale value is superseded (not merely appended
            # like assert_fact would), so a later "what have you learned about
            # me" reflects the corrected fact, not the old one. The user is
            # ground truth for their own profile.
            try:
                _cf = getattr(self.user_model, "detected_correction_fact", None)
                if getattr(self.user_model, "detected_correction", False) and _cf:
                    _cf_subj, _cf_attr, _cf_val = _cf
                    if str(_cf_subj).lower() in ("i", "me", "my"):
                        self.user_model.personal_facts.contradict(
                            "i", _cf_attr, _cf_val)
                        # Render a grounded correction ack from the REAL fact
                        # (content from the store, no authored prose). Phrasing
                        # mirrors the correction ack mapping in engine_persis-
                        # tence; covers the common relation keys. This beats the
                        # generic "got it — thanks for telling me." hollow ack.
                        _rel_phrase = {
                            "name": f"your {_cf_attr} is {_cf_val}",
                            "is": f"you are {_cf_val}",
                            "does": f"you do {_cf_val}",
                            "likes": f"you like {_cf_val}",
                            "location": f"you live in {_cf_val}",
                            "favorite": f"your favorite {_cf_val}",
                        }.get(_cf_attr, f"your {_cf_attr} is {_cf_val}")
                        _ack = (f"thanks for correcting me — i'll remember "
                                f"{_rel_phrase}.")
            except Exception:
                pass
            self._last_responses.append(_ack)
            if len(self._last_responses) > 10:
                self._last_responses = self._last_responses[-10:]
            self.notify_user_idle()
            return _ack

        # ── Frontopolar (BA 10) feasibility gate ────────────────────────────
        # Catch ill-posed / category-error queries BEFORE committing resources
        # to grounding + web search. Conservative: only flags clear affordance
        # mismatches (time/mental/abstract subject predicated with a physical/
        # perceptual property). Legitimate queries pass through untouched.
        # Human-Likeness Plan (A1): a self-preference query ("what is your
        # favorite color") contains the property word "color" and would
        # otherwise trip this gate into the cross-modal metaphor dead-end. Skip
        # the gate for self-preference queries — they are handled by the
        # composed grounded reply in the identity block below, not as a
        # category error.
        _self_pref_q = bool(re.search(
            r"\bwhat(?:'s| is)\s+(my|your)\s+favorite\b|\bwhat\s+do\s+you\s+(like|love|prefer)\b|\bwhat\s+are\s+you\s+(interested in|into)\b",
            (user_input or "").lower()))
        # Temporal-recall/interval questions are episodic-memory tasks, not
        # affordance queries: "how many days between the Sunday mass and the
        # Ash Wednesday service" contains time-words ('sunday') + physical
        # nouns ('mass') and tripped the gate into the category-error
        # metaphor instead of date arithmetic (measured on LongMemEval
        # oracle case 6). PFC task-set recognition must win over the
        # feasibility gate for these.
        _temporal_task_q = bool(
            re.match(r"^\s*(when|what year|what date|how long)\b",
                     (user_input or "").strip().lower())
            or re.search(r"\bhow (many|much)\s+(day|week|month|year|hour|"
                         r"minute)s?\b", (user_input or "").lower()))
        if not _self_pref_q and not _temporal_task_q:
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
                    self.notify_user_idle()
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
        
        m_fav_q = re.search(r"\bwhat(?:'s| is)\s+my\s+favorite\s+(.+)", clean_input, re.IGNORECASE)
        # Human-Likeness Plan (A1): also catch the broad 2nd-person self-preference
        # query ("what is your favorite color", "what do you like", "what are
        # you interested in") that the 1st-person-only regex used to miss. These
        # are questions about the *agent's* own preferences, routed to the
        # composed grounded self-preference reply below — NOT to the category-error
        # gate (which would otherwise fire on "color" and emit the "presence"
        # metaphor dead-end).
        m_agent_fav = re.search(
            r"\bwhat(?:'s| is)\s+your\s+favorite\s+(.+)", clean_input, re.IGNORECASE)
        m_agent_likes = bool(re.search(
            r"\bwhat\s+do\s+you\s+(like|love|prefer)\b", clean_input, re.IGNORECASE))
        # Human-Likeness Plan (A): broaden the self-preference gate so the
        # yes/no form ("do you like music?") AND the stance form
        # ("what do you think about cats?", "how do you feel about X?") hit the
        # vmPFC value resolver instead of falling through to a definition lookup
        # or the uncertainty path. A teen answers these with a stance + affect,
        # not a noun definition. Detected via a learned-style cue combination
        # (self-address + preference/stance predicate) mirroring
        # self_model_router.extract_features — NOT a topic whitelist.
        m_agent_likes_yesno = bool(re.search(
            r"\bdo\s+you\s+(like|love|hate|enjoy|prefer|care\s+for)\b",
            clean_input, re.IGNORECASE))
        m_agent_stance = re.search(
            r"\bwhat\s+do\s+you\s+think\s+about\b|\bhow\s+do\s+you\s+(feel|think)\s+about\b|"
            r"\byour\s+(opinion|thoughts|take)\s+on\b|\bwhat's\s+your\s+(opinion|take)\s+on\b|"
            r"\bwhat\s+is\s+your\s+(opinion|take)\s+on\b",
            clean_input, re.IGNORECASE)
        m_agent_interests = bool(re.search(
            r"\bwhat\s+are\s+you\s+(interested in|into)\b|\bwhat\s+do\s+you\s+want\s+to\s+(learn|know)\b",
            clean_input, re.IGNORECASE))
        # Same-turn user-profile capture (A5): mine personal facts / preferences
        # from THIS turn's input BEFORE the identity/likes/favorites gates

        # is visible to them. Mining only needs user_input (subject isn't
        # assigned until later in process_turn), so we call the lightweight
        # miner rather than the full observe_user_query (which also does ToM /
        # correction side-effects and runs later with the real subject).
        self.user_model.mine_personal_facts(user_input)

        if is_identity_query or is_likes_query or is_interests_query or m_fav_q or m_agent_fav or m_agent_likes or m_agent_likes_yesno or m_agent_stance or m_agent_interests:
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

            elif m_agent_fav:
                # Human-Likeness Plan (A1): the user is asking the AGENT about its
                # own favorite X. Compose a grounded reply from the agent's own
                # affective state — NOT a hardcoded string. The pick emerges from
                # the engine's VAD + Lancaster perceptual profile so it is
                # consistent with the project's "composed, grounded, never
                # hardcoded" rule. A human answer = concrete preference + affect
                # reason + reciprocity return (Social Penetration Theory / Jourard).
                category = m_agent_fav.group(1).strip(" .!?").lower()
                pick, reason = self._agent_favorite_pick(category)
                response = (f"{pick} — {reason}. what about you?")

            elif m_agent_likes or m_agent_likes_yesno or m_agent_stance:
                # Human-Likeness Plan (A2): vmPFC value resolver. The target
                # concept is extracted from the query and a stance is computed
                # as a CONTINUOUS subjective-value signal (Yu 2018; Le Bouc 2026
                # — OFC/vmPFC encode value + its uncertainty on a common scale),
                # NOT a noun definition. This is why "do you like music" now
                # yields a stance + affect rather than a dictionary entry.
                if m_agent_stance:
                    # "what do you think about cats" -> target after the cue.
                    _tail = clean_input[m_agent_stance.end():]
                    target = _tail.strip(" ?!.").split()[0] if _tail.strip(" ?!.").split() else ""
                elif m_agent_likes_yesno:
                    _ym = re.search(
                        r"\bdo\s+you\s+(?:like|love|hate|enjoy|prefer|care\s+for)\s+([a-z][a-z\s'-]{1,30}?)[\?\.]?$",
                        clean_input, re.IGNORECASE)
                    target = _ym.group(1).strip(" ?!.'") if _ym else ""
                else:
                    target = ""
                stance, reason = self._agent_stance_on(target)
                back = " what about you?"
                _reason = reason.rstrip()
                if _reason and not _reason.endswith((".", "!", "?")):
                    _reason += "."
                response = f"{stance} {_reason}{back}"

            elif m_agent_interests:
                response = ("i'm interested in how minds and meaning work — that's "
                            f"the thread i keep coming back to. what draws you in?")

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
        # Human-Likeness Plan (C): a broad episodic "remember what I told you"
        # query is about the CONVERSATION, not a subject concept — the
        # hippocampal reactivation below (which targets graph concepts) cannot
        # serve it. Try the portable episodic transcript FIRST; if it retrieves
        # gist, return it. If it misses, fail closed (no confabulation) rather
        # than fabricating a graph-based/web answer.
        _epi = self._episodic_remember(user_input)
        if _epi:
            self._last_strategy = "episodic_remember"
            self._last_responses.append(_epi)
            if len(self._last_responses) > 10:
                self._last_responses = self._last_responses[-10:]
            self.notify_user_idle()
            return _epi
        if getattr(self, "_episodic_miss", False):
            # Recognized as a recall query but nothing was stored: fail closed
            # (RAVANA bar — honest uncertainty > confident garbage). Do NOT fall
            # through to web/graph which would confabulate.
            self._last_strategy = "episodic_remember_miss"
            _closed = ("honestly, i don't actually have that stored from what "
                       "you've told me so far. what was it?")
            self._last_responses.append(_closed)
            if len(self._last_responses) > 10:
                self._last_responses = self._last_responses[-10:]
            self.notify_user_idle()
            return _closed

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
        # A4 (LIFG pragmatic frame selector; Yoshioka 2023): wellbeing/greeting
        # queries must route to the SOCIAL-response generator, not the semantic
        # association path. The old code only ran the social-frame regex when NO
        # subject was extracted — but "how are you today" makes grounding extract
        # "today"/"day" as a spurious subject, so the social frame never fired and
        # the reply degenerated into "connected with day". Fix: detect the social
        # frame FIRST and let it OVERRIDE a mis-extracted subject.
        _t_social = user_input.lower().strip(" ?!.,").replace("'", "")
        _wellbeing_fuzzy = re.search(
            r"\bhow\s*(?:are|is|'?s|r)?\s*(?:you|u|ya)\b|"
            r"\bhow\s*(?:you|u)\s*(?:doin|doing|feeling|going|been)\b|"
            r"\bhows\s*(?:it\s*going|life|things|everything)\b",
            _t_social)
        if _wellbeing_fuzzy and (not subject or subject.lower() in (
                "today", "day", "you", "how", "doing", "feeling", "going")):
            subject = "how"
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

        # ── Episodic recall (LoCoMo / LongMemEval root-cause fix) ──────────
        # If the user is ASKING about a subject they told us about earlier in
        # this conversation, surface the remembered fact BEFORE the generic
        # definition / web path (which would otherwise answer "what is a car"
        # with a dictionary entry instead of recalling "my car's GPS is
        # broken"). Only fires for interrogatives with a subject that has a
        # stored episodic fact; fail-open otherwise, so fresh-engine benchmarks
        # (empty buffer) are unaffected.
        try:
            _is_question = user_input.strip().endswith("?") or bool(re.match(
                r"^\s*(who|what|when|where|which|why|how|did|do|does|is|are|"
                r"was|were|had|has|have|will|would|could|can)\b",
                user_input.strip().lower()))
            if _is_question and subject:
                _ql = user_input.strip().lower()
                # ── Phase 1: temporal question ("when did X ...", "how long ...",
                # "how many days between/before ..."). Checked BEFORE multi-hop:
                # "how many days between A and B" mentions two entities and
                # multi-hop grabbed it, echoing a fact instead of doing date
                # arithmetic (measured on LongMemEval oracle case 6).
                _is_when = bool(re.match(r"^\s*(when|what year|what date|how long)\b", _ql)) \
                    or "how long" in _ql \
                    or bool(re.search(r"how many (day|week|month|year)s?\b", _ql))
                if _is_when:
                    _dresp = self._answer_temporal_recall(user_input, subject)
                    if _dresp:
                        self._last_strategy = "temporal_recall"
                        self._last_responses.append(_dresp)
                        if len(self._last_responses) > 10:
                            self._last_responses = self._last_responses[-10:]
                        self.notify_user_idle()
                        return _dresp
                # ── Tier 1.5: sequence ordering ("which X happened first") ─
                _sresp = self._answer_sequence_recall(user_input)
                if _sresp:
                    self._last_strategy = "sequence_recall"
                    self._last_responses.append(_sresp)
                    if len(self._last_responses) > 10:
                        self._last_responses = self._last_responses[-10:]
                    self.notify_user_idle()
                    return _sresp
                # ── Phase 3: multi-hop relational question (chains/comparatives)
                _mh = self._try_multi_hop(user_input)
                if _mh:
                    self._last_strategy = "multi_hop"
                    self._last_responses.append(_mh)
                    if len(self._last_responses) > 10:
                        self._last_responses = self._last_responses[-10:]
                    self.notify_user_idle()
                    return _mh
                _mem = self._try_hippocampal_retrieval(
                    type("Ctx", (), {"subject": subject})(), user_input)
                if _mem:
                    _resp = self._phrase_recalled_fact(user_input, subject, _mem)
                    self._last_strategy = "hippocampal_recall"
                    self._last_responses.append(_resp)
                    if len(self._last_responses) > 10:
                        self._last_responses = self._last_responses[-10:]
                    self.notify_user_idle()
                    return _resp
        except Exception:
            pass


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

        # Step 2c: Incongruity gate — check for absurd/ungrounded queries before spread activation
        _has_strong_anchoring = False
        if subject_ids:
            try:
                first_nid = next(iter(subject_ids))
                if hasattr(self.graph, "degree"):
                    _has_strong_anchoring = self.graph.degree(first_nid) >= 3
                else:
                    _has_strong_anchoring = len(subject_ids) >= 2
            except Exception:
                _has_strong_anchoring = False

        if not _has_strong_anchoring and self._is_absurd_query(user_input, subject):
            _absurd_resp = self._handle_absurd_query(user_input, subject)
            self._last_strategy = "absurd_query"
            self._last_responses.append(_absurd_resp)
            if len(self._last_responses) > 10:
                self._last_responses = self._last_responses[-10:]
            self.notify_user_idle()
            return _absurd_resp

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



        # Step 5: Emotional modulation. NOTE: self._update_emotion(user_input)
        # now runs EARLIER in process_turn (before the empathy / self-disclosure
        # early-returns) so affect-dependent replies see the current turn's
        # valence. We do NOT call it again here — doing so would double-apply
        # decay + stimulus. Conceptual-tagging of activated concepts still uses
        # the (already-updated) emotion state below.
        for nid in activated:
            self._concept_vad[nid] = (
                self.emotion.state.valence,
                self.emotion.state.arousal,
                self.emotion.state.dominance,
            )

        # Step 5b: Update UserModel / Theory of Mind with this query. Runs here
        # (after subject is assigned) so opinion mining + full ToM + correction
        # detection have the real subject/valence. (mine_personal_facts runs
        # earlier at the identity gate for SAME-turn fact capture; this full
        # observe also seeds opinions, which the gate miner does not.)
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

        # ─── W4: Creative-writing request pre-router ───
        # "write me a poem about X" / "tell me a story about Y" are GENERATIVE
        # creative requests, not assertions or factual questions. They must be
        # caught BEFORE the assertion/chitchat handlers below (the same reason
        # humor is routed first) so the DMN free-association generator runs
        # instead of being mislabeled an assertion ("poem ocean. anything
        # else?"). A creative intent = an art noun (poem/story/haiku/...) AND a
        # generation verb (write/tell/make/create/compose/give/spin).
        # Defect E fix: also catch generative-invention shapes — "make up /
        # invent / coin (a) (new) word", "imagine / picture a world", "draw /
        # sketch / doodle (a) ..." — these are DMN free-association requests
        # too, and must be routed to _generate_creative BEFORE _is_action_request
        # can swallow them as tool actions.
        _creative_intent = re.search(
            r"\b(poem|story|haiku|song|tale|limerick|rap|verse|lyric|lyrics|"
            r"word|world|scene|character|creature)\b",
            user_input.lower())
        # Generative-shape detector: a creative verb applied to an invented
        # artifact (make up / invent / coin / imagine / draw / sketch / doodle).
        _creative_shape = re.search(
            r"\b(make up|invent|coin|imagine|picture|envision|draw|sketch|"
            r"doodle|compose|create|write|tell|spit|come up with)\b",
            user_input.lower())
        _gen_verb = self._is_action_request(user_input) if hasattr(self, "_is_action_request") else None
        # A creative request may lack a literal generation verb ("a haiku about
        # sleep", "tell me a story about X" where 'tell' is read as a question
        # auxiliary) -- route it if it names an art noun AND a topic anchor
        # (about/of/for), which is the creative-request signature.
        _creative_topic_anchor = bool(re.search(
            r"\b(about|of|for|where|when)\b", user_input.lower()))
        if _creative_intent and (_gen_verb or _creative_topic_anchor):
            resp = self._handle_action_request(user_input, _gen_verb or "write", subject)
            self._last_strategy = ("creative_generation"
                                   if getattr(self, "_last_creative", False)
                                   else "action_request")
            self._last_responses.append(resp)
            if len(self._last_responses) > 10:
                self._last_responses = self._last_responses[-10:]
            self.notify_user_idle()
            return resp
        # Defect E fix (cont.): generative-shape creative requests
        # ("make up a new word", "imagine a world without money", "draw me a
        # dog") that name no canonical art noun still belong to the DMN
        # generator. Route them to _generate_creative BEFORE the action-request
        # handler can refuse them as tool actions.
        _is_info_tell = bool(re.search(r"\b(tell\s+(?:me\s+)?about|tell\s+me|explain|describe|give\s+(?:an?\s+)?overview|what\s+is|define)\b", user_input.lower()))
        if _creative_shape and not _is_info_tell:
            resp = self._handle_action_request(user_input, _gen_verb or "write", subject)
            self._last_strategy = ("creative_generation"
                                   if getattr(self, "_last_creative", False)
                                   else "action_request")
            self._last_responses.append(resp)
            if len(self._last_responses) > 10:
                self._last_responses = self._last_responses[-10:]
            self.notify_user_idle()
            return resp

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
                # Root-cause recall fix: persist the asserted fact to the
                # hippocampal buffer before this path returns.
                self._ingest_episodic(user_input, subject)
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
                # W4: a generated creative verse is internally-grounded
                # free-association (DMN), not a factual claim -- exempt it from
                # the factual grounding gate and tag it creative_generation.
                self._last_strategy = ("creative_generation"
                                       if getattr(self, "_last_creative", False)
                                       else "action_request")
                self._last_responses.append(resp)
                if len(self._last_responses) > 10:
                    self._last_responses = self._last_responses[-10:]
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
            # B7 (ATL retrieval priority; Binder & Desai 2011): protected/seeded
            # domain concepts (ravana, oxiverse, intentforge, ...) are guaranteed
            # stable semantic knowledge with authored typed relations in the
            # graph. Treat them as KNOWN so the FOK->LPFC web pause never fires
            # for them — internal retrieval must win over external search (web is
            # the absolute last resort). This removes the "oxiverse hits 21s web
            # but ravana is instant" inconsistency.
            _is_seeded = (subj_lower in getattr(self, "_PROTECTED_CONCEPTS", set())
                          or subj_lower in getattr(self, "_seeded_domain_concepts", set()))
            phrase_known = phrase_known or _is_seeded
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
                # A7 (FOK familiarity ordering): for an entity-factual question
                # ("who invented X" / "when was X built") the search MUST target
                # the actual informational intent, not a generic "definition"
                # rewrite — otherwise we learn a definition but never fetch the
                # requested attribute (inventor/date) and then fail-closed-claim
                # "i couldn't verify" even though the web had the answer. So we
                # pass the real query when one is present, else the definition
                # framing. This is the hippocampus->PFC familiarity signal
                # (Koriat 1993): attempt retrieval of the SPECIFIC attribute
                # asked about, not just entity familiarity.
                _fok_query = user_input if (user_input and self._is_informational_query(user_input, subject)) else \
                    f"{subject} definition meaning explained"
                if self.baby_mode and not self._fok_pause_done:
                    self._fok_pause_done = True
                    if self._trace_enabled:
                        print(f"  [LPFC] Pausing generation — searching '{_fok_query}'...")
                    try:
                        self.learn_from_web(_fok_query, max_results=2)
                        if self._trace_enabled:
                            print(f"  [LPFC] Web search complete — re-activating concepts")
                        # B8 (epistemic tag; Gruber & Ranganath 2019 PACE): mark
                        # the just-closed knowledge gap so the answer this turn can
                        # be prefaced with an honest "i actually didn't know that
                        # earlier — here's what i found". This is the VTA->hippo
                        # dopamine "new" tag; it decays after N turns (below).
                        if subject and subject.lower() in getattr(self, "_definitions", {}):
                            self._epistemic_new_tags[subject.lower()] = self.turn_count
                            # decay: drop tags older than 20 turns (hippocampal
                            # trace decay — knowledge transitions "new" -> "known")
                            _tc = self.turn_count
                            self._epistemic_new_tags = {
                                _k: _v for _k, _v in self._epistemic_new_tags.items()
                                if _tc - _v <= 20}

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
        # B4: strip simplification framing ("like i'm five", "in simple terms")
        # from the query ONCE here, at the ctx boundary, so every downstream
        # consumer — web query variants, topic extractor, counterfactual
        # resolution — sees the cleaned referent and not the metacommunicative
        # frame. Predictive coding: the frame is not part of the referent. Set
        # simplification_requested so the surface realizer lowers register.
        _stripped = self._strip_eli5_tail(user_input)
        _simplification = (_stripped.strip() != user_input.lower().strip())
        ctx = CognitiveResponseContext(
            subject=subject, relation=relation, object=obj, raw_input=_stripped,
            simplification_requested=_simplification, cleaned_input=_stripped,
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
        # Delegates to _ingest_episodic so date grounding (Phase 1) is applied
        # uniformly regardless of which path reaches here.
        try:
            if user_input and subject:
                self._ingest_episodic(user_input, subject)
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
                    # Retry on the rare "dictionary changed size during iteration"
                    # race with the background-learning thread: a live dict may be
                    # mutated mid-iteration despite the lock, so re-run the turn up
                    # to 3 times. Any other RuntimeError is a real failure — re-raise.
                    if "dictionary changed size" in str(e) and _attempts < 3:
                        _attempts += 1
                        continue
                    raise
            # D3 (round v3): capture RAVANA's OWN self-description so a later
            # "what did you say about who you are" can recall it instead of a user
            # episode (the D-C bug). The stored content is the verbatim reply
            # RAVANA produced THIS turn — real output, detected structurally via
            # self-reference markers, not authored prose — so it passes the
            # no-hardcoding line. It runs only after a successful generation
            # (response is defined), and the store is a plain dict RAVANA can
            # overwrite at runtime (e.g. when asked to re-describe itself), not
            # frozen code. (A secondary capture site at engine_self_query.py also
            # records the explicit self-description turns.)
            try:
                _rl = (response or "").lower()
                _self_markers = (
                    "i am ravana", "i'm ravana", "ravana, cognitive",
                    "cognitive architecture", "brain-inspired",
                    "brain-inspired cognitive", "i learn concepts",
                    "i'm a brain", "i am a brain")
                if any(_m in _rl for _m in _self_markers):
                    self._agent_claims["self"] = (response or "").strip()
            except Exception:
                pass
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
        # P2-C: keep the dead calibration signal alive. Fold the just-scored
        # quality into a short buffer; derive a real error rate (1 - quality)
        # and an adaptive window via rolling std (stable => wider, volatile
        # => narrower). At cold-start (buffer empty) error=0.0, window=15
        # => identical to the legacy fixed behavior. No theta_withhold
        # modulation: no prediction-vs-quality pair exists to drive it.
        self._calib_buf.append(quality_score)
        if len(self._calib_buf) > 60:
            self._calib_buf = self._calib_buf[-60:]
        _mean_q = sum(self._calib_buf) / len(self._calib_buf)
        self._calibration_error = round(1.0 - _mean_q, 4)
        if len(self._calib_buf) >= 5:
            _mean = _mean_q
            _var = sum((q - _mean) ** 2 for q in self._calib_buf) / len(self._calib_buf)
            _std = _var ** 0.5
            _target = 15
            if _std < 0.1:
                _target = 30
            elif _std > 0.3:
                _target = 5
            # EMA toward target so the window drifts, never snaps.
            self._calib_window = int(round(0.9 * self._calib_window + 0.1 * _target))

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

        # ── Post-generation cross-turn consistency monitor ─────
        # Issue 3 (confirmed): no existing module watches the AGENT's
        # own generated claims across turns. This NO-LLM monitor extracts
        # claims from the final response, checks them against a rolling
        # buffer, and (in 'annotate' mode) prefixes a soft consistency
        # note when a genuine contradiction is detected. Fail-open: if
        # the monitor is None or raises, the response is unchanged.
        try:
            _cm = getattr(self, "_consistency_monitor", None)
            if _cm is not None and isinstance(response, str) and response:
                _cr = _cm.check(response, self.turn_count)
                if _cr.conflict_detected:
                    response = _cm.resolve(response, _cr)
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

        # P1 Theory of Mind: personalized greeting when relationship warrants it (roadmap §9).
        # B3: a "welcome back" is a memory CLAIM — only emit it when we have
        # genuinely resumed from a prior session (self._session_resumed, set once
        # by _load) AND we haven't already done so this session. This stops the
        # mid-session "Welcome back! Last time we discussed X" leak, where the
        # old interaction_count % 10 gate slapped a false recollection onto an
        # arbitrary turn. The greeting becomes a reactivation artifact, not a tick.
        try:
            if self._session_resumed and not self._greeting_emitted_this_session:
                greeting = self._personalized_greeting()
                if greeting and response:
                    response = greeting + response
                    self._greeting_emitted_this_session = True
                # Consume the resume flag regardless of whether we emitted — we
                # only get ONE shot at a grounded greeting per resumed session.
                self._session_resumed = False
        except Exception:
            pass  # Never break the pipeline for a greeting

        self._pending_quantity_result = None
        # NOTE: previously this returned ``response.lower()``. That destroyed
        # proper-noun casing in the final output (e.g. "France" -> "france",
        # "NASA" -> "nasa"), making RAVANA look broken. All generators already
        # produce correctly-cased text, and quality/scoring functions lowercase
        # internally where needed, so we return the response as-is.
        return response
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
                # Adaptive gating baselines (EMA mu/sigma/n per gate). Persisted
                # so the distribution-driven gates keep adapting ACROSS sessions
                # instead of resetting to seed every boot (the audit's
                # saved-but-never-loaded class of bug — without this, the
                # adaptive gates never actually learn from history).
                'adaptive_baselines': {k: dict(v) for k, v in self._adaptive_baselines.items()},
                # Triplet inference operator state (learned relation profiles,
                # relational index, abstention gate). Written here AND restored
                # in load() — a gate saved but never reloaded is dead.
                'triplet_inference': (self.triplet_op.to_dict()
                                      if getattr(self, 'triplet_op', None)
                                      else None),
                # Learned word-frequency models (Plan B): seed + observed counts
                # so the high-frequency lexicon tail survives reloads.
                'freq_models': {k: v.to_dict() for k, v in self._freq_models.items()},
                # Learned lemma store (Item 5, P2) — novel past->base mappings.
                'learned_lemmas': dict(self._learned_lemmas),
                # Reflective monitoring
                'episodic_edges': _episodic_edges,
                'semantic_edges': _semantic_edges,
                # ConnectorLearner state (Item 3, P1): persist the learned
                # connector->relation mappings + prototype centroids so they
                # survive reloads and accumulate across sessions (previously
                # never saved -> reset to seed every boot).
                'connector_learner': (self._connector_learner.to_dict()
                                      if getattr(self._connector_learner, 'to_dict', None)
                                      else None),
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
                # ATL semantic memory: ONLINE-learned triples only (seed is
                # reloadable from data/semantic_seed.pkl — never snapshotted).
                'semantic_graph_state': (self.semantic_graph.get_state()
                                         if getattr(self, 'semantic_graph', None)
                                         is not None else None),
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
                # Self-model claims ("i'm ravana", "i'm a bit cautious about
                # bans") — the agent's OWN stated valuations, persisted so
                # self-opinion recall ("are you still cautious about X") can
                # reference what the agent previously said about itself instead
                # of recomputing a fresh transient opinion every boot.
                'agent_claims': dict(getattr(self, '_agent_claims', {}) or {}),
                # Per-topic self-opinion cache (A1): stance:{target} -> the
                # grounded stance+reason the agent computed for each concept it
                # has been asked about. Persisted so self-opinion recall stays
                # STABLE across sessions (personality continuity) — without
                # this every boot recomputes a fresh transient valuation and the
                # agent "forgets" how it felt about a topic between sessions.
                'agent_preferences': dict(getattr(self, '_agent_preferences', {}) or {}),
                # Constitutive value store (seed + experience). Persisted so
                # values RAVANA forms/revises at runtime SURVIVE reload — the
                # "can change this by itself through experience" guarantee.
                'agent_values': dict(getattr(self, '_agent_values', {}) or {}),
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
                    # Persist the user model to its OWN dedicated directory
                    # (user_models/) so it survives independently of the weight
                    # snapshot. It is also kept inside `state` below for
                    # backward-compat with pre-split checkpoints.
                    try:
                        from .user_model import save_user_model
                        _um_path = save_user_model(self.user_model,
                                                   getattr(self, 'user_suffix', ''))
                        size_kb += 0  # user-model file tracked separately
                    except Exception:
                        _um_path = None
                    if _um_path:
                        return f"saved {size_kb:.0f}KB to {os.path.basename(self._save_path)} (user_model -> {os.path.basename(_um_path)})"
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
                    # Issue 2: scan ALL nodes for dimension drift, not just the
                    # first (a mixed-era snapshot can have a 64-D first node but
                    # 75-D others — the old guard passed it and corrupted the
                    # live similarity matrix). Quarantine any off-dim node by
                    # re-adding it through the canonical projection; if too many
                    # are corrupt (>50%), discard the saved graph entirely.
                    _offdim = [n for n in loaded_graph.nodes.values()
                               if n.vector is not None and len(n.vector) != self.dim]
                    if _offdim:
                        _frac = len(_offdim) / len(loaded_graph.nodes)
                        print(f"  [Load] {len(_offdim)}/{len(loaded_graph.nodes)} "
                              f"nodes off canonical dim {self.dim} "
                              f"({_frac:.0%}); repairing via canonical projection")
                        if _frac > 0.5:
                            print(f"  [Load warning] Too many off-dim nodes; "
                                  f"discarding saved graph.")
                            return False
                        # Repair in place: project each off-dim node's vector to
                        # the canonical width (75->64 dual-code recovery, etc.).
                        for _n in _offdim:
                            try:
                                _n.vector = loaded_graph._project_to_canonical(
                                    _n.vector, source="load_repair")
                            except Exception as _e:
                                print(f"  [Load] quarantined node '{getattr(_n,'label','?')}': {_e}")
                                # drop the unrecoverable node by detaching it
                                loaded_graph.nodes.pop(_n.id, None)
                    self.graph = loaded_graph
                    # Brain-aligned durable reconsolidation: if the restored
                    # graph came back EMPTY (pickle lost it, e.g. a legacy
                    # snapshot whose graph was sanitized to a placeholder), the
                    # ACID SQLite mirror written on every save() still holds the
                    # real graph. Rebuild from durable memory rather than
                    # starting blank — the graph is the semantic memory that must
                    # survive a reload even when the primary snapshot is damaged.
                    if not loaded_graph.nodes:
                        try:
                            self.db.load_graph(self.graph)
                        except Exception:
                            pass
                    # Re-attach the HRR populate hook + dual_code anchor to the
                    # freshly-loaded graph (the init wiring targeted the OLD
                    # graph object, now orphaned by this swap). Without
                    # this, add_edge stops encoding into HRR and the store
                    # stays empty despite hrr_reasoner being live.
                    if self.hrr_reasoner is not None:
                        self.graph._fact_encode_hook = self._hrr_encode_hook
                        self.graph.dual_code = self.dual_code
            else:
                # Corrupt graph (e.g. a str from an old sanitizer path) —
                # rebuild fresh rather than aborting the WHOLE load (M5:
                # partial restore, not silent blank wipe).
                print(f"  [Load partial] graph was "
                      f"{type(loaded_graph).__name__}, not ConceptGraph — "
                      f"rebuilding empty graph; other state restored")
                self.graph = ConceptGraph(dim=self.dim,
                                      max_nodes=getattr(self, '_max_nodes', 20000))
                # Durable reconsolidation: recover the real graph from the ACID
                # SQLite mirror written on every save(), so a pickle-graph
                # corruption does not silently wipe all learned knowledge.
                try:
                    self.db.load_graph(self.graph)
                except Exception:
                    pass
                # Re-wire the HRR populate hook + dual_code anchor onto the
                # rebuilt graph so add_edge keeps encoding into HRR.
                if self.hrr_reasoner is not None:
                    self.graph._fact_encode_hook = self._hrr_encode_hook
                    self.graph.dual_code = self.dual_code
            self._concept_keywords = state['concept_keywords']
            self.turn_count = state['turn_count']
            # B3: a successful resume from a saved snapshot means a PRIOR session
            # actually occurred — so a "welcome back" greeting is a grounded
            # autonoetic claim, not a confabulation. Flag it once; process_turn
            # consumes it on the first turn and never re-emits mid-session.
            self._session_resumed = True

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
                # M5 fix: a *sample* (first entry) is not enough — a snapshot
                # can hold a MIXED table (most entries at the current dim, a few
                # stale ones at the old dim from an incremental cross-version
                # save). Sampling only the first entry let those stale vectors
                # through and crash _build_decoder_vocab() at broadcast time.
                # Validate the WHOLE table; any mismatch discards the entire
                # stale read-out so _build_decoder_vocab() rebuilds a fresh,
                # uniform 75-D table from the graph + GloVe.
                _bad = any(
                    (not isinstance(_v, np.ndarray)) or (_v.shape[0] != _dec_dim)
                    for _v in self._decoder_word_to_embed.values()
                )
                if _bad:
                    print(f"  [Load warn] decoder embed dim mismatch "
                          f"(current {_dec_dim}) — discarding stale decoder vocab, "
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
            # Self-model claims: restore the agent's OWN previously-stated
            # valuations ("i'm ravana", "i'm a bit cautious about bans") so
            # self-opinion recall across sessions references what the agent
            # actually said about itself (personality continuity), instead of
            # recomputing a fresh transient opinion each boot.
            try:
                _ac = state.get('agent_claims', {})
                if isinstance(_ac, dict):
                    self._agent_claims = dict(_ac)
                else:
                    self._agent_claims = {}
            except Exception:
                self._agent_claims = {}
            # Restore the per-topic self-opinion cache (A1) so stances the agent
            # computed survive reloads. Guarded: a bad shape must not wipe the
            # store or break the boot. PURGE legacy junk entries: older builds
            # cached fabricated stances for non-topics ("stance:right",
            # "stance:really", "stance:source") produced by the removed GloVe
            # transitivity path. Those must never be replayed — they are exactly
            # the "i'm a bit cautious about right" confabulation class.
            try:
                _ap = state.get('agent_preferences', {})
                if isinstance(_ap, dict):
                    _JUNK = {"all", "really", "it", "that", "things", "right",
                             "way", "matter", "thing", "point",
                             "idea", "question", "stuff", "something",
                             "anything", "everything", "issue", "topic",
                             "yes", "no", "maybe", "ok", "okay"}
                    self._agent_preferences = {
                        k: v for k, v in _ap.items()
                        if not (isinstance(k, str) and k.startswith("stance:")
                                and k[len("stance:"):] in _JUNK)}
            except Exception:
                pass
            # agent grew/revised at runtime wins, seed fills the rest).
            try:
                _av = state.get('agent_values', {})
                if isinstance(_av, dict):
                    _seed = getattr(self, '_agent_values', {}) or {}
                    _seed.update({k: tuple(v) for k, v in _av.items()})
                    self._agent_values = _seed
            except Exception:
                pass
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
            # Ensure learned personal-fact / opinion stores exist. These were
            # added after the earlier UserModel schema; old snapshots pickle a
            # UserModel WITHOUT them, and any query touching user_model.
            # personal_facts / user_model.opinions then raises AttributeError
            # (seen as a hard crash on the adversarial benchmark when the engine
            # is restored from snapshot per category). Fail-closed: attach a
            # fresh store so the attribute always exists post-load.
            from .personal_fact_store import PersonalFactStore, UserStanceStore
            if not hasattr(loaded_user_model, 'personal_facts'):
                loaded_user_model.personal_facts = PersonalFactStore()
            if not hasattr(loaded_user_model, 'opinions'):
                loaded_user_model.opinions = UserStanceStore()
            self.user_model = loaded_user_model
            # Prefer the dedicated user_models/ store ONLY when it actually
            # exists — it is the continuously-updated, authoritative profile
            # and may be newer than the copy frozen inside this weight snapshot.
            # CRITICAL FIX (round 2026-08-09i): load_user_model() returns a
            # FRESH EMPTY UserModel() when the dedicated file is absent, so the
            # old code ALWAYS overwrote the perfectly-good embedded snapshot
            # with an empty model — silently wiping every learned stance and
            # personal fact on every reload (the skill documents these as
            # "durable across sessions"; they were not). Only swap in the
            # dedicated store when the file is present; otherwise keep the
            # authoritative embedded snapshot from THIS checkpoint.
            try:
                from .user_model import load_user_model, _user_model_path
                _um_path = _user_model_path(getattr(self, 'user_suffix', ''))
                if os.path.exists(_um_path):
                    _separate_um = load_user_model(getattr(self, 'user_suffix', ''))
                    if _separate_um is not None:
                        self.user_model = _separate_um
            except Exception:
                pass
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

            # Restore adaptive gating baselines (EMA mu/sigma/n). __init__
            # already seeded every gate to its day-one constant; overlay the
            # persisted per-gate distribution so gates resume adapting from the
            # last session instead of restarting (mirrors the VAD baseline).
            _ab = state.get('adaptive_baselines')
            if _ab:
                for _k, _v in _ab.items():
                    if _k in self._adaptive_baselines:
                        self._adaptive_baselines[_k] = {
                            "mu": float(_v.get("mu", self._adaptive_baselines[_k]["mu"])),
                            "sigma": float(_v.get("sigma", self._adaptive_baselines[_k]["sigma"])),
                            "n": int(_v.get("n", 0)),
                        }

            # Restore triplet inference operator state (profiles + gate) so
            # learned relational statistics accumulate ACROSS sessions.
            _ti = state.get('triplet_inference')
            if _ti and getattr(self, 'triplet_op', None) is not None:
                try:
                    self.triplet_op.from_dict(_ti)
                except Exception:
                    pass

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

            # Restore learned word-frequency models (Plan B). __init__ already
            # seeded each model from the class attribute; overlay the persisted
            # observed counts so frequency learning resumes from last session.
            _fm = state.get('freq_models')
            if _fm:
                for _k, _fd in _fm.items():
                    if _k in self._freq_models:
                        self._freq_models[_k] = FrequencyModel.from_dict(_fd)

            # Restore learned lemma store (Item 5, P2).
            _ll = state.get('learned_lemmas')
            if _ll:
                self._learned_lemmas = dict(_ll)

            # Restore source-trust accumulator (Item 1, P0). Saved at save()
            # time but previously never reloaded -> the prefrontal credibility
            # learner reset to {} every boot and never accumulated across
            # sessions. Overlay the persisted dict so domains keep the trust
            # they earned.
            _st = state.get('source_trust')
            if _st:
                self._source_trust = dict(_st)

            # Restore ConnectorLearner state (Item 3, P1). __init__ rebuilt it
            # from seed prototypes; overlay the persisted learned mappings so
            # connector->relation associations resume from last session.
            _cl = state.get('connector_learner')
            if _cl and getattr(self, '_connector_learner', None) is not None:
                try:
                    from .synaptic_dynamics import ConnectorLearner
                    _restored = ConnectorLearner.from_dict(
                        _cl, glove_fn=self._glove_vector)
                    # Preserve identity of the live object reference used
                    # elsewhere; copy restored state onto it.
                    self._connector_learner._prototype_vecs = _restored._prototype_vecs
                    self._connector_learner._learned_probs = _restored._learned_probs
                    self._connector_learner._connector_set = _restored._connector_set
                    self._connector_learner._connector_to_rel = _restored._connector_to_rel
                    self._connector_learner._is_initialized = _restored._is_initialized
                except Exception:
                    pass

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
            sg_state = state.get('semantic_graph_state', None)
            if sg_state and getattr(self, 'semantic_graph', None) is not None:
                # Restores ONLINE-learned semantic triples; the ConceptNet
                # seed reloads lazily from disk on first semantic query.
                self.semantic_graph.set_state(sg_state)
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
    def _web_blocked(self) -> bool:
        """Single gate for all live-network paths.

        D1 fix (round v-aug06): when RAVANA_OFFLINE=1 is set (the documented
        reproducible/CI mode), no live Wikipedia kb_describe, _web_direct_answer,
        support_router, or background web-learning may fire. Previously only the
        GloVe download respected the flag; the answer paths did not, so an
        "offline" run still emitted "according to a web source…" with real
        network latency. Centralizing the check here means one source of truth
        for what 'offline' means. Returns True when web must be suppressed.

        INJECTION-AWARE (D1 regression fix): the gate only suppresses the path
        when a REAL network call would happen. Tests that replace
        ``self.search_engine`` with a fake that never touches the network
        (e.g. test_web_direct_answer_surfaces_source, the snippet-quality
        e2e tests) must still reach the retry/breaker/snippet logic they
        exercise. We detect an injected fake by the absence of the real
        ``SearchEngine`` class, so production keeps the gate (real
        SearchEngine + offline -> blocked) while faked-search tests flow
        through. This is the minimal shape: the gate stays for real network
        calls, and is bypassed only when no network call is possible.
        """
        if not getattr(self, "_offline", False):
            return False
        # Offline flag is set. Block ONLY when a REAL network call would occur.
        # A test that injected a fake search backend (replacing either the whole
        # ``search_engine`` object OR just its ``search``/``_call_api`` method)
        # never touches the network, so it must reach the retry/breaker/snippet
        # logic it exercises. The genuine path is detected as: the backend is a
        # real ``SearchEngine`` AND its ``search`` method is the real one (so it
        # would route to ``_call_api`` -> urlopen, which the learner-level gate
        # suppresses). Any injected fake (object swapped, or method rebound) is
        # NOT blocked, so production keeps the gate (real SearchEngine + offline
        # -> blocked, protecting the 822MB GloVe download on the soak runner)
        # while faked-search tests flow through.
        from ravana.web.learner import SearchEngine as _RealSearchEngine
        _se = getattr(self, "search_engine", None)
        if not isinstance(_se, _RealSearchEngine):
            return False  # fake backend object -> no network -> allow
        # Genuine SearchEngine with an un-replaced search() -> would hit network.
        return getattr(_se.search, "__func__", None) is _RealSearchEngine.search

    def start_background_learning(self):
        """Start the background learning thread. Called once at engine creation or CLI start."""
        # D1 fix (round v-aug06): in offline mode the background web-learner
        # must not run — it performs live web searches and records phantom
        # "learnings" (the run showed learned=3 with RAVANA_OFFLINE=1, which is
        # contradictory). Skip the thread entirely so _learning_count stays 0
        # and the engine is fully offline-reproducible.
        if self._web_blocked():
            if self._trace_enabled:
                print('  [bg] offline mode — background web-learning disabled')
            return
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

