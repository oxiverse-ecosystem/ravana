#!/usr/bin/env python3
"""
RAVANA Baby — cognitive chat that starts like a baby and learns from the web
============================================================================
No commands. No LLM. Pure RAVANA cognitive architecture.
Starts knowing ~180 teen-level concepts with GloVe semantic embeddings
and typed graph edges (causal, contrastive, analogical, temporal, semantic),
auto-learns from the internet when it doesn't know something.

Usage:
    python scripts/ravana_chat.py
"""

import sys, os, time, random, json, re, argparse, pickle, threading, hashlib
import urllib.request
from urllib.error import URLError
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Insert in REVERSE order of priority (last insert ends up first in sys.path)
# NOTE: modular `ravana` package (with language/core/chat) lives at ravana/src/ravana/.
# It MUST shadow the stale root ravana/ dir (whose __init__.py is broken), so we add
# ravana/src LAST (highest priority) and keep _proj_root for `scripts` package discovery.
sys.path.insert(0, _proj_root)
sys.path.insert(0, os.path.join(_proj_root, "ravana", "src"))
sys.path.insert(0, os.path.join(_proj_root, "ravana-v2"))

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
from ravana.core import UserEmotionDetector


# Try importing beautifulsoup4 for HTML parsing (optional but recommended)
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


class SearchEngine:
    """
    Multi-API search engine with circuit breaker fallback.
    
    Tries multiple search APIs in order:
    1. Oxiverse API (primary)
    2. DuckDuckGo HTML scrape (fallback)
    3. Bing API (optional, requires key)
    4. Google Custom Search (optional, requires key)
    
    Circuit breaker tracks consecutive failures per API and skips
    failed APIs for a cooldown period.
    """
    
    def __init__(self):
        # API configurations: (name, url_template, timeout, max_results)
        self.apis = [
            ("oxiverse", "https://api.oxiverse.com/search?q={}", 10, 3),
            ("duckduckgo", "https://html.duckduckgo.com/html/?q={}", 10, 3),
            # Bing and Google would need API keys - placeholders for future
            # ("bing", "https://api.bing.microsoft.com/v7.0/search?q={}", 5, 10),
            # ("google", "https://www.googleapis.com/customsearch/v1?q={}&key={}&cx={}", 5, 10),
        ]
        
        # Circuit breaker state per API
        self._api_failure_counts = {name: 0 for name, _, _, _ in self.apis}
        self._api_last_failure_time = {name: 0 for name, _, _, _ in self.apis}
        self._api_cooldown = 10  # seconds before retrying failed API (was 60)
        self._max_failures_before_break = 10  # allow more retries (was 3)
        
        # Headers for requests
        self._headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 RAVANA/1.0'
        }
    
    def _is_api_available(self, api_name: str) -> bool:
        """Check if API is available (not in circuit breaker cooldown)."""
        if self._api_failure_counts[api_name] < self._max_failures_before_break:
            return True
        elapsed = time.time() - self._api_last_failure_time[api_name]
        if elapsed > self._api_cooldown:
            # Reset after cooldown
            self._api_failure_counts[api_name] = 0
            return True
        return False
    
    def _record_success(self, api_name: str):
        """Record successful API call."""
        self._api_failure_counts[api_name] = 0
    
    def _record_failure(self, api_name: str):
        """Record failed API call."""
        self._api_failure_counts[api_name] += 1
        self._api_last_failure_time[api_name] = time.time()
    
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search with automatic fallback across APIs.
        
        Returns normalized results: list of dicts with keys:
        - title, url, content (snippet)
        """
        query_encoded = quote(query)
        errors = []
        
        for api_name, url_template, timeout, api_max_results in self.apis:
            if not self._is_api_available(api_name):
                continue
                
            try:
                url = url_template.format(query_encoded)
                results = self._call_api(api_name, url, timeout, max_results)
                if results:
                    self._record_success(api_name)
                    return results[:max_results]
            except Exception as e:
                self._record_failure(api_name)
                errors.append(f"{api_name}: {e}")
                continue
        
        # All APIs failed
        raise SearchError(f"All search APIs failed: {'; '.join(errors)}")
    
    def _call_api(self, api_name: str, url: str, timeout: int, max_results: int) -> List[Dict[str, Any]]:
        """Call specific API and parse results."""
        req = urllib.request.Request(url, headers=self._headers)
        resp = urllib.request.urlopen(req, timeout=timeout)
        content = resp.read().decode('utf-8', errors='replace')
        
        if api_name == "oxiverse":
            data = json.loads(content)
            results = data.get('results', [])
            return [
                {
                    'title': r.get('title', ''),
                    'url': r.get('url', ''),
                    'content': r.get('content', r.get('snippet', ''))
                }
                for r in results if r.get('url')
            ]
        
        elif api_name == "duckduckgo":
            # Parse HTML results from DuckDuckGo
            return self._parse_duckduckgo_html(content, max_results)
        
        return []
    
    def _parse_duckduckgo_html(self, html: str, max_results: int) -> List[Dict[str, Any]]:
        """Parse DuckDuckGo HTML results."""
        results = []
        if HAS_BS4:
            soup = BeautifulSoup(html, 'html.parser')
            # DuckDuckGo result selectors
            for result in soup.select('.result__snippet, .result__title, a.result__url')[:max_results * 3]:
                # Try to extract title, url, snippet from result containers
                container = result.find_parent('div', class_='result')
                if container:
                    title_elem = container.select_one('.result__title a, h2 a')
                    url_elem = container.select_one('.result__url, a.result__url')
                    snippet_elem = container.select_one('.result__snippet')
                    
                    title = title_elem.get_text(strip=True) if title_elem else ''
                    url = title_elem.get('href', '') if title_elem else (url_elem.get_text(strip=True) if url_elem else '')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else title
                    
                    if url and not url.startswith('/'):
                        results.append({'title': title, 'url': url, 'content': snippet})
                    
                    if len(results) >= max_results:
                        break
        else:
            # Regex fallback if BeautifulSoup not available
            import re
            # Basic pattern for DDG results
            snippets = re.findall(r'<a[^>]*class="result__snippet"[^>]*>([^<]+)</a>', html)
            titles = re.findall(r'<a[^>]*class="result__title"[^>]*>([^<]+)</a>', html)
            urls = re.findall(r'<a[^>]*class="result__url"[^>]*>([^<]+)</a>', html)
            
            for i in range(min(len(snippets), len(titles), len(urls), max_results)):
                results.append({
                    'title': titles[i],
                    'url': urls[i],
                    'content': snippets[i]
                })
        
        return results


class SearchError(Exception):
    """Raised when all search APIs fail."""
    pass


# Global search engine instance
SEARCH_ENGINE = SearchEngine()


# ─── Teen Vocabulary: what a teenager knows (~180 words) ───
DOMAIN_CONCEPTS = {
    "oxiverse": {
        "keywords": "oxiverse privacy-first source-available ecosystem alternative big-tech",
        "relations": [
            ("oxiverse", "ecosystem", "is_a", 0.7),
            ("oxiverse", "privacy", "causal", 0.65),
            ("oxiverse", "big tech", "contrastive", 0.55),
        ],
        "stability": 0.9,
    },
    "intentforge": {
        "keywords": "intentforge intent-driven semantic search engine discovery",
        "relations": [
            ("intentforge", "search engine", "is_a", 0.7),
            ("intentforge", "discovery", "causal", 0.6),
            ("intentforge", "oxiverse", "part_of", 0.65),
        ],
        "stability": 0.9,
    },
    "ravana": {
        "keywords": "ravana cognitive architecture backprop-free hebbian reasoning",
        "relations": [
            ("ravana", "cognitive architecture", "is_a", 0.7),
            ("ravana", "hebbian learning", "causal", 0.6),
            ("ravana", "analogical reasoning", "causal", 0.55),
            ("ravana", "oxiverse", "part_of", 0.5),
        ],
        "stability": 0.9,
    },
}

TEEN_CONCEPTS = [
    # ── Social & Identity ──
    ("hello", "hi hey greeting sup"),
    ("bye", "goodbye farewell later"),
    ("yes", "okay yeah agree absolutely"),
    ("no", "nope negative disagree"),
    ("please", "polite request kindly"),
    ("thanks", "thank appreciate gratitude"),
    ("sorry", "apologize forgive regret"),
    ("i", "me myself my own"),
    ("you", "your yourself thou"),
    ("we", "us our together group"),
    ("they", "them their other people"),
    ("friend", "buddy pal companion ally"),
    ("people", "human society community crowd"),
    ("person", "individual human being someone"),

    # ── Advanced Social Concepts ──
    ("trust", "rely faith confidence belief"),
    ("justice", "fairness equality right moral"),
    ("hypocrisy", "contradict inconsistent double standard fake"),
    ("empathy", "compassion understanding feeling care"),
    ("respect", "admire honor esteem regard"),
    ("identity", "self character personality who"),
    ("culture", "tradition society custom heritage"),
    ("power", "control influence authority strength"),
    ("freedom", "liberty choice independence right"),
    ("responsibility", "duty obligation accountability burden"),

    # ── Epistemic & Abstract ──
    ("truth", "real fact honest genuine accurate"),
    ("belief", "faith opinion conviction view"),
    ("knowledge", "wisdom understanding awareness learning"),
    ("meaning", "purpose significance essence point"),
    ("pattern", "structure repetition system cycle"),
    ("system", "network framework structure organization"),
    ("perspective", "viewpoint angle lens outlook"),
    ("context", "situation background setting circumstance"),
    ("paradox", "contradiction puzzle ironic dilemma"),
    ("principle", "rule value standard moral axiom"),
    ("theory", "hypothesis idea framework explanation"),
    ("evidence", "proof data fact support clue"),
    ("analysis", "examination study breakdown evaluation"),
    ("conclusion", "result inference deduction summation"),
    ("logic", "reason rational sense coherence"),
    ("intuition", "instinct gut feeling hunch"),
    ("wisdom", "insight knowledge judgment prudence"),

    # ── Abstract Qualities ──
    ("complex", "complicated intricate sophisticated layered"),
    ("significant", "important meaningful major notable"),
    ("fundamental", "basic essential core foundation"),
    ("inevitable", "unavoidable certain destined fated"),
    ("possible", "maybe potential feasible plausible"),
    ("obvious", "clear apparent evident obvious"),
    ("subtle", "nuanced delicate faint indirect"),
    ("profound", "deep meaningful significant thoughtful"),
    ("ignorance", "unawareness blindness obliviousness inexperience"),
    ("injustice", "unfairness inequality oppression bias"),
    ("oppression", "tyranny suppression persecution subjugation"),

    # ── Core Verbs (expanded) ──
    ("want", "wish desire need crave"),
    ("like", "enjoy love prefer appreciate"),
    ("go", "move leave walk proceed"),
    ("come", "arrive approach appear"),
    ("see", "look watch observe perceive"),
    ("hear", "listen sound overhear"),
    ("eat", "food meal consume devour"),
    ("drink", "water thirsty sip beverage"),
    ("sleep", "rest nap bed unconscious"),
    ("play", "fun game toy recreation"),
    ("help", "assist aid support serve"),
    ("make", "create build produce cause"),
    ("get", "receive obtain acquire understand"),
    ("know", "understand aware learn recognize"),
    ("think", "believe consider wonder reason"),
    ("say", "tell speak talk express"),
    ("feel", "sense emotion touch experience"),
    ("love", "care affection adore cherish"),
    ("give", "share present offer donate"),
    ("take", "grab seize accept choose"),

    # ── Advanced Verbs ──
    ("analyze", "examine study evaluate break down"),
    ("conclude", "decide infer deduce determine"),
    ("reflect", "ponder contemplate meditate consider"),
    ("question", "challenge doubt inquire interrogate"),
    ("explore", "discover investigate venture search"),
    ("understand", "comprehend grasp realize fathom"),
    ("compare", "contrast relate match evaluate"),
    ("criticize", "judge critique evaluate assess"),
    ("assume", "presume suppose guess speculate"),
    ("imagine", "envision dream visualize conceive"),
    ("connect", "relate associate link bridge"),
    ("influence", "affect shape impact sway"),
    ("struggle", "fight conflict strive contend"),
    ("challenge", "dare confront test oppose"),

    # ── Core Nouns (expanded) ──
    ("water", "drink wet rain liquid"),
    ("food", "eat meal snack nutrition"),
    ("home", "house room family shelter"),
    ("sun", "light warm day star"),
    ("moon", "night star dark lunar"),
    ("tree", "plant leaf flower forest"),
    ("bird", "fly animal feather wing"),
    ("dog", "pet puppy bark canine"),
    ("cat", "kitten meow pet feline"),
    ("book", "read story page novel"),
    ("song", "music sing melody rhythm"),
    ("world", "earth globe planet universe"),
    ("nature", "environment wild natural earth"),
    ("time", "clock moment age duration"),
    ("life", "living existence being survive"),
    ("death", "die end mortality passing"),
    ("mind", "brain thought consciousness psyche"),
    ("heart", "organ emotion core center"),
    ("science", "study research knowledge method"),
    ("history", "past story legacy record"),
    ("art", "creative expression beauty culture"),

    # ── Abstract Verbs ──
    ("cause", "produce create generate result"),
    ("change", "transform shift modify evolve"),
    ("grow", "develop expand mature increase"),
    ("learn", "study discover understand master"),
    ("teach", "educate instruct explain mentor"),
    ("create", "make invent produce generate"),
    ("destroy", "ruin break eliminate devastate"),
    ("protect", "defend guard shield secure"),
    ("accept", "embrace welcome acknowledge agree"),
    ("reject", "refuse deny dismiss decline"),

    # ── Adjectives (expanded) ──
    ("good", "nice great fine positive"),
    ("bad", "wrong negative evil harmful"),
    ("big", "large huge giant massive"),
    ("small", "tiny little mini slight"),
    ("hot", "warm burn fire heated"),
    ("cold", "cool freeze ice chilly"),
    ("happy", "joy glad smile content"),
    ("sad", "cry unhappy upset sorrow"),
    ("scared", "afraid fear frighten anxious"),
    ("angry", "furious mad frustrated rage"),
    ("tired", "sleepy exhausted fatigue drained"),
    ("excited", "eager enthusiastic thrilled pumped"),
    ("curious", "interested inquisitive nosy wonder"),
    ("confused", "lost puzzled baffled uncertain"),
    ("bored", "uninterested dull tired weary"),
    ("proud", "accomplished satisfied dignified confident"),
    ("lonely", "isolated alone abandoned disconnected"),
    ("grateful", "thankful appreciative indebted blessed"),

    # ── Emotion Words ──
    ("anxiety", "worry nervous tension stress"),
    ("excitement", "thrill enthusiasm anticipation energy"),
    ("frustration", "annoyance irritation aggravation anger"),
    ("hope", "optimism aspiration wish dream"),
    ("fear", "terror dread panic horror"),
    ("joy", "delight happiness bliss pleasure"),
    ("grief", "sorrow loss mourning lament"),
    ("sadness", "sorrow unhappiness melancholy grief"),
    ("surprise", "shock amazement astonishment wonder"),
    ("guilt", "remorse regret shame blame"),
    ("disappointment", "letdown regret dissatisfaction dismay"),
    ("hate", "detest loathe despise abhor"),
    ("despair", "hopelessness misery anguish desolation"),
    ("distrust", "suspicion doubt mistrust wariness"),
    ("motivation", "drive inspiration ambition determination"),

    # ── Imagination & Future ──
    ("future", "tomorrow ahead later coming"),
    ("past", "history ago previous yesterday"),
    ("machine", "device engine mechanism robot"),
    ("invention", "creation innovation discovery breakthrough"),
    ("invent", "create design devise pioneer"),
    ("possibility", "potential chance opportunity likelihood"),
    ("imagination", "creativity fantasy vision dream"),
    ("impossible", "unlikely hopeless absurd ridiculous"),
    ("journey", "travel adventure voyage quest"),
    ("secret", "hidden mystery private unknown"),
    ("experiment", "trial test attempt investigation"),

    # ── Relational / Meta ──
    ("and", "also plus together"),
    ("so", "therefore thus hence"),
    ("then", "next after afterwards"),
    ("link", "connect join bond tie"),
    ("why", "reason because explanation cause"),
    ("how", "method way process means"),
    ("what", "which thing object identity"),
    ("if", "suppose whether maybe perhaps"),
    ("but", "however yet although though"),
    ("because", "since due cause reason"),
    ("maybe", "perhaps possibly probably could"),
    ("always", "forever constant perpetual eternal"),
    ("never", "not once zero none"),

    # ── Basic concepts (retained) ──
    ("up", "above high sky rise"),
    ("down", "below low ground fall"),
    ("in", "inside within interior"),
    ("out", "outside exit exterior"),
    ("here", "this place near present"),
    ("there", "that place far distant"),
    ("now", "today present moment current"),
    ("later", "soon future after eventual"),
    ("more", "extra additional plus further"),
    ("all", "every everything whole total"),
    ("some", "few several part partial"),
    ("one", "single first unique individual"),
    ("two", "second pair both double"),
    ("many", "multiple numerous several abundant"),
]

# ─── Stop words to filter during article reading ───
# Web garbage words that should never become graph concepts
# These come from HTML/CSS/JS, URLs, and programming artifacts
WEB_GARBAGE = {
    # HTML/CSS artifacts
    'html', 'css', 'js', 'div', 'span', 'class', 'style', 'script', 'meta',
    'href', 'src', 'img', 'br', 'hr', 'head', 'body', 'font', 'nav',
    'header', 'footer', 'section', 'article', 'aside', 'main',
    'width', 'height', 'color', 'margin', 'padding', 'border',
    'background', 'display', 'position', 'float', 'clear', 'overflow',
    'block', 'inline', 'flex', 'grid', 'align', 'justify', 'content',
    # URL/web artifacts
    'http', 'https', 'www', 'com', 'org', 'net', 'io', 'gov', 'edu',
    'url', 'uri', 'link', 'domain', 'cookie',
    'query', 'params', 'token', 'oauth', 'api', 'json', 'xml',
    # JavaScript/programming
    'var', 'let', 'const', 'func', 'function', 'return', 'import',
    'export', 'module', 'require', 'console', 'log', 'debug',
    'undefined', 'null', 'true', 'false', 'typeof', 'instanceof',
    'array', 'object', 'string', 'number', 'boolean', 'promise',
    'async', 'await', 'callback', 'event', 'listener', 'handler',
    'prototype', 'constructor', 'length', 'index', 'value',
    # Common tech/platform names that are not general English words
    'gform', 'wordpress', 'joomla', 'drupal', 'shopify', 'squarespace',
    'wix', 'webflow', 'github', 'gitlab', 'bitbucket', 'heroku',
    'netlify', 'vercel', 'aws', 'azure', 'gcp', 'docker', 'kubernetes',
    'react', 'angular', 'vue', 'svelte', 'jquery', 'bootstrap',
    'tailwind', 'sass', 'less', 'webpack', 'vite',
    # Analytics/tracking
    'analytics', 'tracking', 'pixel', 'gtag', 'gaq',
    'utm', 'campaign', 'click', 'impression', 'conversion',
    # Date/time patterns that appear as words
    'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
    'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
}


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
    # Additional function/grammatical words that should not be concepts
    "such", "one", "other", "same", "new", "call", "make", "get", "take",
    "give", "find", "use", "go", "come", "see", "know", "think", "say",
    "tell", "ask", "want", "need", "like", "love", "feel", "look",
    "seem", "become", "turn", "start", "begin", "end", "stop", "keep",
    "let", "put", "set", "run", "walk", "talk", "hear", "listen",
    "show", "mean", "help", "work", "play", "try", "move", "change",
    # Prepositions and contractions that slip through
    "til", "until", "till", "upon", "within", "without", "across", "along",
    "among", "around", "beneath", "beyond", "despite", "except", "inside",
    "near", "since", "toward", "towards", "underneath", "versus", "via",
    "whether", "once", "twice",
}


@dataclass
class FailedQuery:
    query: str = ""
    subject: str = ""
    activated_concepts: List[str] = field(default_factory=list)
    strategies_tried: List[str] = field(default_factory=list)
    best_guess_response: str = ""
    turn: int = 0
    free_energy_at_time: float = 0.0
    resolved: bool = False

@dataclass
class ChainHop:
    from_label: str
    to_label: str
    relation_type: str
    weight: float
    confidence: float
    temperature: float
    candidates: int  # how many edges were considered at this hop
    rlm_confidence: float = 0.0  # RLMv2 triple verification score
    contradiction: str = ""  # contradiction detail if detected

@dataclass
class ChainTrace:
    hops: List[ChainHop] = field(default_factory=list)
    max_hops: int = 0
    completed: bool = False

@dataclass
class UserModel:
    """Tracks user interaction patterns and Theory of Mind.

    Theory of Mind components:
    - knowledge_model: What concepts the user appears to know (from their queries)
    - learning_goals: Topics the user has asked about repeatedly (goals they pursue)
    - emotional_rapport: User's emotional valence toward topics (positive/negative)
    - cognitive_style: Inferred style — 'curious', 'skeptical', 'practical', 'balanced'
    - engagement_level: How engaged the user is (0-1)
    - conversation_depth: Average number of follow-ups per topic
    - relationship_depth: 0.0 (stranger) → 1.0 (close) — grows with interaction_count
    - goals: Inferred current goal — LEARNING / DEBUGGING / EXPLORING

    No hardcoded patterns — preferences emerge from graph traversal stats.
    """
    edge_reactivations: Dict[Tuple[str, str], int] = field(default_factory=dict)
    query_concepts: Set[str] = field(default_factory=set)
    
    # Theory of Mind state
    knowledge_model: Dict[str, float] = field(default_factory=dict)      # concept -> confidence user knows it
    learning_goals: Dict[str, int] = field(default_factory=dict)        # topic -> frequency of queries
    emotional_rapport: Dict[str, float] = field(default_factory=dict)   # topic -> valence (-1 to 1)
    cognitive_style: str = "balanced"                                     # 'curious', 'skeptical', 'practical', 'balanced'
    engagement_level: float = 0.5                                         # 0-1
    conversation_depth: float = 0.0                                       # avg follow-ups per topic
    # P1 Theory of Mind: relationship + goals (roadmap §7)
    interaction_count: int = 0                                            # total user turns observed
    relationship_depth: float = 0.0                                       # 0.0=stranger, 1.0=close friend
    goals: List[str] = field(default_factory=list)                        # inferred goal history (LEARNING/DEBUGGING/EXPLORING)
    last_goal: str = "EXPLORING"                                          # most recently inferred goal
    
    # P2 Emotional State Tracking (roadmap §7)
    emotional_state: Dict[str, float] = field(default_factory=lambda: {
        'valence': 0.0, 'arousal': 0.3, 'dominance': 0.5,
    })
    belief_state: Dict[str, Dict] = field(default_factory=dict)
    interaction_history: List[Dict] = field(default_factory=list)
    _emotion_detector: Any = None  # Lazy-initialized UserEmotionDetector

    # Interaction tracking
    topic_interaction_count: Dict[str, int] = field(default_factory=dict)
    topic_followup_count: Dict[str, int] = field(default_factory=dict)
    last_topic: str = ""
    turn_since_topic_change: int = 0

    def observe_chain(self, hops: List[Tuple[str, str]], is_user_query: bool = False):
        """Record which edges were traversed during a chain walk."""
        for from_label, to_label in hops:
            key = (from_label.lower(), to_label.lower())
            self.edge_reactivations[key] = self.edge_reactivations.get(key, 0) + 1
        if is_user_query:
            for from_label, to_label in hops:
                self.query_concepts.add(from_label.lower())
                self.query_concepts.add(to_label.lower())
                # User asking about from -> to means they know 'from' and want to learn 'to'
                self.knowledge_model[from_label.lower()] = min(1.0, self.knowledge_model.get(from_label.lower(), 0.0) + 0.1)
                self.learning_goals[to_label.lower()] = self.learning_goals.get(to_label.lower(), 0) + 1

    def observe_user_query(self, query: str, subject: str, valence: float):
        """Update Theory of Mind from a user query."""
        subject_lower = subject.lower()
        self.topic_interaction_count[subject_lower] = self.topic_interaction_count.get(subject_lower, 0) + 1
        
        # Track learning goals: repeated queries about same topic = learning goal
        self.learning_goals[subject_lower] = self.learning_goals.get(subject_lower, 0) + 1
        
        # Track knowledge: user asking about a subject implies partial familiarity
        if subject_lower:
            self.knowledge_model[subject_lower] = min(
                1.0, self.knowledge_model.get(subject_lower, 0.0) + 0.1)
        
        # Track emotional rapport: user's valence toward topic
        current_rapport = self.emotional_rapport.get(subject_lower, 0.0)
        # Slowly adjust toward user's valence (rate 0.2)
        self.emotional_rapport[subject_lower] = current_rapport + 0.2 * (valence - current_rapport)
        
        # Track cognitive style from query patterns
        self._update_cognitive_style(query)
        
        # Track topic changes for engagement
        if subject_lower != self.last_topic and self.last_topic:
            self.topic_followup_count[self.last_topic] = max(0, self.topic_followup_count.get(self.last_topic, 0) - 1)
            self.turn_since_topic_change = 0
        else:
            self.turn_since_topic_change += 1
            if self.last_topic:
                self.topic_followup_count[self.last_topic] = self.topic_followup_count.get(self.last_topic, 0) + 1
        self.last_topic = subject_lower
        
        # Update engagement level: more follow-ups = higher engagement
        total_interactions = sum(self.topic_interaction_count.values())
        total_followups = sum(self.topic_followup_count.values())
        self.conversation_depth = total_followups / max(1, len(self.topic_interaction_count))
        self.engagement_level = min(1.0, 0.3 + 0.7 * (total_followups / max(1, total_interactions)))

        # P1 Theory of Mind: relationship depth grows with interaction count.
        # Neuroscience basis: familiarity/rapport builds incrementally with repeated
        # exposure (mere-exposure effect). Saturates at ~20 interactions per roadmap §9.
        self.interaction_count += 1
        self.relationship_depth = min(1.0, self.interaction_count / 20.0)

        # P1 Theory of Mind: infer the user's current goal from query phrasing.
        inferred = self.infer_user_goal(query)
        self.last_goal = inferred
        self.goals.append(inferred)
        if len(self.goals) > 50:
            self.goals = self.goals[-50:]

        # P2 Emotional State Tracking: infer user emotion from query text.
        emotion_vad = self._infer_user_emotion(query)
        self._record_interaction(query, subject, emotion_vad)

    def _ensure_emotion_detector(self):
        """Lazy-init the emotion detector to avoid import-order issues."""
        if self._emotion_detector is None:
            self._emotion_detector = UserEmotionDetector()

    def _infer_user_emotion(self, text: str) -> Tuple[float, float, float]:
        """Detect user emotional state from query text using VAD lexicon.

        Uses UserEmotionDetector with ANEW-based VAD lexicon + intensifier/
        negation modulation. Updates self.emotional_state with EMA blending.

        Returns (valence, arousal, dominance) tuple.
        Neuroscience basis: Barsalou (1999) semantic grounding of emotion
        concepts; Warriner et al. (2013) affective norms.
        """
        self._ensure_emotion_detector()
        v, a, d = self._emotion_detector.detect(text)
        # EMA blend with previous state for temporal coherence
        rate = 0.35
        prev = self.emotional_state
        self.emotional_state = {
            'valence': prev['valence'] + rate * (v - prev['valence']),
            'arousal': prev['arousal'] + rate * (a - prev['arousal']),
            'dominance': prev['dominance'] + rate * (d - prev['dominance']),
        }
        return (v, a, d)

    def _record_interaction(self, text: str, subject: str,
                            emotion_vad: Tuple[float, float, float]):
        """Record this interaction in history for belief/personality modeling."""
        self.interaction_history.append({
            'text': text[:200],
            'subject': subject,
            'valence': emotion_vad[0],
            'arousal': emotion_vad[1],
            'dominance': emotion_vad[2],
            'turn': len(self.interaction_history),
        })
        if len(self.interaction_history) > 100:
            self.interaction_history = self.interaction_history[-100:]

    def infer_user_goal(self, query: str) -> str:
        """Infer the user's current goal from query phrasing.

        Returns one of: LEARNING, DEBUGGING, EXPLORING.
        Roadmap §7 "Goal Inference":
          - "how does X work?" → LEARNING
          - "why is X broken?" → DEBUGGING
          - "tell me about X" → EXPLORING
        """
        q = query.lower().strip()
        # DEBUGGING: trouble-shooting phrasing — errors, breakage, frustration
        debug_markers = ('broken', "doesn't work", "doesn't work", 'error', 'fail',
                         'bug', 'crash', 'wrong', 'stuck', 'issue', 'fix', 'not working',
                         "isn't working", 'exception', 'traceback')
        if any(m in q for m in debug_markers) or q.startswith('why is') and any(
            m in q for m in ('broken', 'error', 'fail', 'wrong', 'crash')):
            return "DEBUGGING"
        # LEARNING: mechanism / explanation seeking
        learn_markers = ('how does', 'how do', 'how is', 'how are', 'what is', 'what are',
                         'explain', 'how come', 'why does', 'why do')
        if any(q.startswith(m) or (' ' + m) in q for m in learn_markers):
            return "LEARNING"
        # EXPLORING: open-ended topic introduction
        explore_markers = ('tell me about', "let's talk about", 'i want to know',
                           'i wonder', 'teach me', 'show me', 'describe')
        if any(m in q for m in explore_markers):
            return "EXPLORING"
        return "EXPLORING"

    def _update_cognitive_style(self, query: str):
        """Infer cognitive style from query patterns."""
        q_lower = query.lower()
        style_scores = {
            'curious': sum(1 for w in ['why', 'how', 'what', 'explain', 'understand', 'curious', 'wonder'] if w in q_lower),
            'skeptical': sum(1 for w in ['really', 'actually', 'prove', 'evidence', 'doubt', 'sure', 'fake', 'lie'] if w in q_lower),
            'practical': sum(1 for w in ['how to', 'build', 'make', 'create', 'step', 'guide', 'tutorial', 'implement'] if w in q_lower),
        }
        if style_scores:
            top_style = max(style_scores, key=style_scores.get)
            if style_scores[top_style] > 0:
                self.cognitive_style = top_style

    def infer_topic_interest(self, topic: str) -> float:
        """Return interest level for a topic (0-1).
        Combines learning goals, rapport, and interaction frequency."""
        t = topic.lower()
        goal_strength = min(1.0, self.learning_goals.get(t, 0) * 0.2)
        rapport = (self.emotional_rapport.get(t, 0.0) + 1.0) / 2.0  # normalize -1..1 to 0..1
        interaction = min(1.0, self.topic_interaction_count.get(t, 0) * 0.1)
        return (goal_strength * 0.4 + rapport * 0.4 + interaction * 0.2)

    def infer_user_knows(self, concept: str) -> float:
        """Probability user knows this concept (0-1)."""
        return self.knowledge_model.get(concept.lower(), 0.0)

    def infer_user_wants_to_learn(self, concept: str) -> float:
        """How much user wants to learn about this concept (0-1)."""
        t = concept.lower()
        goal = min(1.0, self.learning_goals.get(t, 0) * 0.15)
        rapport = (self.emotional_rapport.get(t, 0.0) + 1.0) / 2.0
        return max(0.0, goal * 0.6 + rapport * 0.4 - self.knowledge_model.get(t, 0.0) * 0.5)

    def get_preferred_relation_types(self) -> List[str]:
        """Infer preferred relation types from edge reactivations."""
        rel_counts = {}
        for (f, t), count in self.edge_reactivations.items():
            # Infer relation from concept pair (simplified)
            rel = 'semantic'  # default
            rel_counts[rel] = rel_counts.get(rel, 0) + count
        return sorted(rel_counts, key=rel_counts.get, reverse=True)[:3]

    def inferred_preferences(self, threshold: int = 2) -> Dict[Tuple[str, str], int]:
        """Return edges the user has reactivated >= threshold times.
        Used for display/debugging, NOT for boost calculation."""
        return {(f, t): c for (f, t), c in self.edge_reactivations.items()
                if c >= threshold}

    def activation_boost_for(self, concept: str) -> Dict[str, float]:
        """Continuous activation boost for edges reachable from concept.

        Uses sigmoid confidence: 0 visits → 1.0x, 1 visit → 1.15x,
        2 visits → 1.2x, saturating at 1.3x.
        No hard threshold — even a single visit provides a small boost."""
        boost: Dict[str, float] = {}
        cl = concept.lower()
        for (from_c, to_c), count in self.edge_reactivations.items():
            if from_c == cl:
                boost[to_c] = 1.0 + (count / (count + 1.0)) * 0.3
        return boost

    def get_state(self) -> Dict:
        """Return serializable state."""


        return {
            'edge_reactivations': {str(k): v for k, v in self.edge_reactivations.items()},
            'query_concepts': list(self.query_concepts),
            'knowledge_model': self.knowledge_model,
            'learning_goals': self.learning_goals,
            'emotional_rapport': self.emotional_rapport,
            'cognitive_style': self.cognitive_style,
            'engagement_level': self.engagement_level,
            'conversation_depth': self.conversation_depth,
            'topic_interaction_count': self.topic_interaction_count,
            'topic_followup_count': self.topic_followup_count,
            'last_topic': self.last_topic,
            'turn_since_topic_change': self.turn_since_topic_change,
            # P1 Theory of Mind: relationship + goals (roadmap §7)
            'interaction_count': self.interaction_count,
            'relationship_depth': self.relationship_depth,
            'goals': self.goals,
            'last_goal': self.last_goal,
            # P2 Emotional State Tracking (roadmap §7)
            'emotional_state': self.emotional_state,
            'belief_state': self.belief_state,
            'interaction_history': self.interaction_history,
        }
    def set_state(self, state: Dict):
        """Restore state from dict."""
        self.edge_reactivations = {eval(k): v for k, v in state.get('edge_reactivations', {}).items()}
        self.query_concepts = set(state.get('query_concepts', []))
        self.knowledge_model = state.get('knowledge_model', {})
        self.learning_goals = state.get('learning_goals', {})
        self.emotional_rapport = state.get('emotional_rapport', {})
        self.cognitive_style = state.get('cognitive_style', 'balanced')
        self.engagement_level = state.get('engagement_level', 0.5)
        self.conversation_depth = state.get('conversation_depth', 0.0)
        self.topic_interaction_count = state.get('topic_interaction_count', {})
        self.topic_followup_count = state.get('topic_followup_count', {})
        self.last_topic = state.get('last_topic', '')
        self.turn_since_topic_change = state.get('turn_since_topic_change', 0)
        # P1 Theory of Mind: relationship + goals (backward-compatible defaults)
        self.interaction_count = state.get('interaction_count', 0)
        self.relationship_depth = state.get('relationship_depth', 0.0)
        self.goals = state.get('goals', [])
        self.last_goal = state.get('last_goal', 'EXPLORING')
        # P2 Emotional State Tracking (backward-compatible defaults)
        self.emotional_state = state.get('emotional_state',
            {'valence': 0.0, 'arousal': 0.3, 'dominance': 0.5})
        self.belief_state = state.get('belief_state', {})
        self.interaction_history = state.get('interaction_history', [])

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
    learned_recently: bool = False  # Did we just learn something new?
    recall_mode: bool = False  # Episodic re-traversal vs generic chain walk
    sentence_vector: Any = None  # Compositional sentence-level vector (N400/P600 integration)
    discourse_context: str = ""  # Accumulated discourse context across turns


class BeliefStore:
    """Tracks asserted belief triples and contradiction history.

    Each belief is (subject, predicate, value) with confidence and timestamp.
    Contradictions are detected and recorded for edge-weight modulation.
    """
    def __init__(self):
        self.beliefs: Dict[Tuple[str, str], Tuple[str, float, int]] = {}
        # (subject, predicate) -> (value, confidence, turn)
        self.contradictions: List[Tuple[Tuple, Tuple, int]] = []
        # [(old_triple, new_triple, turn)]
        self.resolution_history: Dict[str, str] = {}
        # triple_str -> "accept_new" | "reject_new" | "both"
        self.turn_num = 0

    def advance_turn(self):
        self.turn_num += 1

    def assert_belief(self, subject_id: str, predicate: str,
                       value: str, confidence: float = 0.8):
        key = (subject_id, predicate)
        self.beliefs[key] = (value, confidence, self.turn_num)

    def query_belief(self, subject_id: str,
                     predicate: str) -> Optional[Tuple[str, float, int]]:
        return self.beliefs.get((subject_id, predicate))

    def detect_contradiction(self, subject_id: str, predicate: str,
                              new_value: str) -> Optional[Tuple]:
        existing = self.query_belief(subject_id, predicate)
        if existing and existing[0] != new_value:
            old_triple = (subject_id, predicate, existing[0])
            new_triple = (subject_id, predicate, new_value)
            self.contradictions.append((old_triple, new_triple, self.turn_num))
            return (existing, new_value, existing[2])
        return None

    def resolve_contradiction(self, old_triple: Tuple, new_triple: Tuple,
                               choice: str):
        self.contradictions.append((old_triple, new_triple, self.turn_num))
        key = str(new_triple)
        self.resolution_history[key] = choice

    def reconcile(self) -> Dict[Tuple[str, str], Tuple[str, float, int]]:
        """Resolve contradictions: pick winner by confidence * recency decay.

        The belief store keyed by (subject, predicate) holds only the latest
        value per pair, but contradiction history lists every flip. We scan
        contradictions to find each contested pair and pick the winner based
        on confidence × recency (later turns = higher recency).
        """
        resolved: Dict[Tuple[str, str], Tuple[str, float, int]] = {}
        # Group contradiction candidates by (subject, predicate)
        groups: Dict[Tuple[str, str], List[Tuple[str, float, int]]] = {}
        for old_triple, new_triple, c_turn in self.contradictions:
            # old_triple = (subject, predicate, old_value)
            # new_triple = (subject, predicate, new_value)
            subj, pred = old_triple[0], old_triple[1]
            key = (subj, pred)
            # Add both sides of the contradiction
            old_val, old_conf = old_triple[2], 0.0
            new_val, new_conf = new_triple[2], 0.0
            # Look up actual confidence from belief store if available
            cur = self.beliefs.get(key)
            if cur:
                if cur[0] == old_val:
                    old_conf = cur[1]
                elif cur[0] == new_val:
                    new_conf = cur[1]
            groups.setdefault(key, []).append(
                (old_val, old_conf if old_conf > 0 else 0.5, c_turn))
            groups.setdefault(key, []).append(
                (new_val, new_conf if new_conf > 0 else 0.5, c_turn))
        # Deduplicate within each group
        for key, candidates in groups.items():
            seen = set()
            unique = []
            for c in candidates:
                if c[0] not in seen:
                    seen.add(c[0])
                    unique.append(c)
            if len(unique) < 2:
                continue
            def decay_score(vc: Tuple) -> float:
                _, conf, turn = vc
                recency = 1.0 / (1.0 + (self.turn_num - turn) * 0.1)
                return conf * recency
            winner = max(unique, key=decay_score)
            # Update the belief store with the winner
            self.beliefs[key] = (winner[0], winner[1], self.turn_num)
            resolved[key] = self.beliefs[key]
        return resolved

    def get_state(self) -> Dict:
        return {
            'beliefs': self.beliefs,
            'contradictions': self.contradictions,
            'resolution_history': self.resolution_history,
            'turn_num': self.turn_num,
        }

    def set_state(self, state: Dict):
        self.beliefs = state.get('beliefs', {})
        self.contradictions = state.get('contradictions', [])
        self.resolution_history = state.get('resolution_history', {})
        self.turn_num = state.get('turn_num', 0)


class CognitiveChatEngine:
    """RAVANA cognitive chat engine — starts as a baby, learns from the web."""
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

        self.graph = ConceptGraph(dim=dim, max_nodes=10000)
        self.baby_mode = baby_mode
        self._concept_labels: Set[str] = set()  # set of primary concept labels

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
        self._concept_pos: Dict[str, str] = {}
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
        # Phase 9b: Prediction error tracking (surprise signal for Active Inference)
        self._mean_prediction_error = 0.0
        self._prediction_error_count = 0
        # Integration toggles (can be disabled via CLI)
        self.use_vad = True
        self.use_rlm = True
        self.use_beliefs = True
        self.belief_store = BeliefStore()

        # Fix 2: Dormant edge tracking — auto-wired GloVe edges are invisible
        # until the user model visits them at least once.
        self._dormant_edges: Set[Tuple[int, int]] = set()

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
            "where", "why", "how", "here", "there", "now", "then", "later",
            "soon", "ago", "back", "away", "forward", "backward", "inside",
            "outside", "near", "far", "high", "low", "deep", "shallow",
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
            "such", "same", "different", "new", "certain", "whole", "own", "particular",
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
            "here", "there", "where", "why", "how", "when",
            # Discourse markers / connectives
            "instead", "introduced", "alternatively", "conversely", "likewise",
            "similarly", "therefore", "however", "moreover", "furthermore",
            "besides", "nevertheless", "nonetheless", "accordingly", "consequently",
            "thus", "hence", "accordingly", "subsequently", "meanwhile",
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
        # Phase E: Syntactic cell assemblies — Hebbian role learning with seeded priors
        self.syntactic_assembly = SyntacticCellAssembly(learning_rate=0.05)
        # Phase F: Surface realizer — rule-governed English morphology with dopamine modulation
        self.surface_realizer = SurfaceRealizer()
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
        # Solution #2: Reasoning mode (stochastic / deterministic / exploratory)
        self.reasoning_mode: str = "stochastic"

        # Consistency tracking for deterministic mode: (subject_hash, tuple(seen)) -> path
        self._consistency_paths: Dict[int, List[str]] = {}
        self._consistency_trace: List[str] = []
        # Phase 18c: Multi-source consensus tracking for hallucination guard
        self._concept_sources: Dict[str, Set[str]] = {}  # concept -> set of source URLs
        # Phase 18d: Explored contradiction pairs (prevent re-queuing "good vs bad debate")
        self._explored_contradictions: Set[Tuple[str, str]] = set()

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
                # Decoder loaded successfully - no deferred training needed
                self._needs_seed_training = False
                self._needs_synthetic_training = False
                self._freeze_decoder_vocab = True  # Freeze decoder vocab during inference
                print(f"  [Loaded] Remembered {len(self.graph.nodes)} words from before!")
                return

        # Cold start (no saved weights): seed everything from scratch.
        self._seed_concepts()
        self._bootstrap_domain_concepts()
        self._build_decoder_vocab()
        print(f"  [Teen] Knows {len(self.graph.nodes)} words, ready to learn!")

    # ─── Neural Decoder Vocabulary ───

    MAX_DECODER_VOCAB_SIZE = 1500  # Stage 2: focused vocab for 5× speed

    def _build_decoder_vocab(self):
        """Build vocabulary for the NeuralDecoder from graph concepts + GloVe + function words.

        Maps every graph concept label and common English function words to
        vocab indices and embedding vectors. Initializes the NeuralDecoder.
        Stage 2: Capped at MAX_DECODER_VOCAB_SIZE for speed (5× fewer logits).
        """
        self._decoder_word_to_idx = {}
        self._decoder_idx_to_word = {}
        self._decoder_word_to_embed = {}

        # Special tokens
        special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
        for i, tok in enumerate(special_tokens):
            self._decoder_word_to_idx[tok] = i
            self._decoder_idx_to_word[i] = tok
            vec = np.random.randn(self.dim).astype(np.float32) * 0.05
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            self._decoder_word_to_embed[tok] = vec

        next_idx = len(special_tokens)
        max_vocab = self.MAX_DECODER_VOCAB_SIZE

        # Common function words for fluent generation
        function_words = [
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "not", "no", "nor", "so", "if", "then", "than", "too", "very",
            "just", "about", "also", "into", "over", "after", "before",
            "between", "through", "during", "because", "while", "which",
            "who", "whom", "what", "when", "where", "why", "how", "all",
            "each", "every", "both", "few", "more", "most", "some", "any",
            "this", "that", "these", "those", "it", "its", "they", "them",
            "their", "we", "our", "you", "your", "he", "she", "him", "her",
            "his", "i", "me", "my", "myself", "am",
            "of", "to", "for", "with", "from", "at", "by", "as", "on",
            "related", "connect", "connects", "mean", "means",
            "concept", "concepts", "idea", "ideas", "important",
            "lead", "leads", "talk", "think", "link", "links",
        ]

        # Add graph concept labels (highest priority after function words)
        with self._graph_lock:
            concept_labels = set()
            for nid, node in self.graph.nodes.items():
                if node.label:
                    concept_labels.add(node.label.lower())

        # Function words first (ensures fluent generation)
        for fw in function_words:
            if next_idx >= max_vocab:
                break
            if fw not in self._decoder_word_to_idx:
                self._decoder_word_to_idx[fw] = next_idx
                self._decoder_idx_to_word[next_idx] = fw
                vec = self._glove_vector(fw)
                if vec is None:
                    vec = np.random.randn(self.dim).astype(np.float32) * 0.1
                    norm_v = np.linalg.norm(vec)
                    if norm_v > 0:
                        vec /= norm_v
                self._decoder_word_to_embed[fw] = vec.astype(np.float32)
                next_idx += 1

        # Graph concepts next
        for label in sorted(concept_labels):
            if next_idx >= max_vocab:
                break
            if label not in self._decoder_word_to_idx:
                self._decoder_word_to_idx[label] = next_idx
                self._decoder_idx_to_word[next_idx] = label
                nids = self._concept_keywords.get(label, [])
                if nids:
                    node = self.graph.get_node(nids[0])
                    if node and node.vector is not None:
                        vec = node.vector.copy()
                    else:
                        vec = self._glove_vector(label)
                else:
                    vec = self._glove_vector(label)
                if vec is None:
                    vec = np.random.randn(self.dim).astype(np.float32) * 0.1
                    norm_v = np.linalg.norm(vec)
                    if norm_v > 0:
                        vec /= norm_v
                self._decoder_word_to_embed[label] = vec.astype(np.float32)
                next_idx += 1

        # Fill remaining slots with corpus words
        remaining = max_vocab - next_idx
        if remaining > 0:
            corpus_path = os.path.join(_proj_root, "data", "corpora", "teen_seeds.txt")
            if os.path.exists(corpus_path):
                import re
                with open(corpus_path, "r", encoding="utf-8") as f:
                    corpus_text = f.read()
                words_in_corpus = set(re.findall(r"[a-zA-Z']{3,}", corpus_text.lower()))
                for w in sorted(words_in_corpus):
                    if remaining <= 0:
                        break
                    if w not in self._decoder_word_to_idx:
                        self._decoder_word_to_idx[w] = next_idx
                        self._decoder_idx_to_word[next_idx] = w
                        vec = self._glove_vector(w)
                        if vec is None:
                            vec = np.random.randn(self.dim).astype(np.float32) * 0.1
                            norm_v = np.linalg.norm(vec)
                            if norm_v > 0:
                                vec /= norm_v
                        self._decoder_word_to_embed[w] = vec.astype(np.float32)
                        next_idx += 1
                        remaining -= 1

        # If forced vocab size is set (from saved state), match it for back-compat
        if hasattr(self, '_forced_vocab_size') and self._forced_vocab_size:
            target_size = min(self._forced_vocab_size, max_vocab)
            while len(self._decoder_word_to_idx) < target_size:
                pad_token = f"<pad_{next_idx}>"
                self._decoder_word_to_idx[pad_token] = next_idx
                self._decoder_idx_to_word[next_idx] = pad_token
                vec = np.random.randn(self.dim).astype(np.float32) * 0.05
                norm_v = np.linalg.norm(vec)
                if norm_v > 0:
                    vec /= norm_v
                self._decoder_word_to_embed[pad_token] = vec.astype(np.float32)
                next_idx += 1
            delattr(self, '_forced_vocab_size')

        vocab_size = len(self._decoder_word_to_idx)
        self.neural_decoder = NeuralDecoder(
            vocab_size=vocab_size,
            embed_dim=self.dim,
            hidden_dim=256,
            n_attention_heads=4,
            contrastive_weight=0.5,
            contrastive_negatives=8,
        )

        for word, idx in self._decoder_word_to_idx.items():
            if word in self._decoder_word_to_embed:
                self.neural_decoder.word_embedding.weight.data[idx] = \
                    self._decoder_word_to_embed[word]
        self.neural_decoder.rebuild_vocab_cache()
        self._decoder_vocab_built = True
        self._needs_seed_training = True
        self._needs_synthetic_training = True

    def _train_decoder_from_graph(self, min_synthetic: int = 2000):
        """Train neural decoder on natural sentences derived from graph relationships.

        Uses diverse, human-like phrasings instead of template artifacts like
        "X connects with Y" that were poisoning the decoder's output.
        
        Args:
            min_synthetic: Maximum number of synthetic sentences to generate (default 2000)
        """
        if self.neural_decoder is None or not self._decoder_vocab_built:
            return
        # Natural phrasings that sound like human conversation, not graph templates
        templates = [
            # Casual / conversational
            "{s} and {o} go hand in hand",
            "when you think about {s}, {o} comes to mind",
            "{s} is a big part of {o}",
            "{s} has a lot to do with {o}",
            "{o} matters a lot when it comes to {s}",

            # Explanatory
            "{s} is really about {o}",
            "at its core, {s} ties into {o}",
            "{s} plays a role in {o}",
            "one way to think about {s} is through {o}",
            "{s} shapes how we understand {o}",

            # Causal / influence
            "{s} can lead to {o}",
            "{o} often happens because of {s}",
            "{s} has an impact on {o}",
            "{s} contributes to {o} in important ways",
            "{o} is something that {s} can bring about",

            # Contrastive
            "{s} and {o} are quite different",
            "while {s} is one thing, {o} is another",
            "{s} stands apart from {o} in interesting ways",
            "unlike {s}, {o} takes a different path",

            # Analogical
            "{s} is a lot like {o} in some ways",
            "you can see {s} reflected in {o}",
            "{s} and {o} share something important",
            "thinking of {s} reminds me of {o}",

            # Temporal / process
            "{s} often comes before {o}",
            "{o} tends to follow {s}",
            "{s} sets the stage for {o}",
            "{s} paves the way for {o}",

            # Speculative / reflective
            "maybe {s} and {o} are more related than we think",
            "it is interesting how {s} ties into {o}",
            "{s} makes you wonder about {o}",
            "in a way, {s} and {o} feed into each other",

            # Personal / first person
            "i think {s} is closely tied to {o}",
            "to me, {s} says something about {o}",
            "i see {s} and {o} as deeply connected",
        ]

        sentences = []
        seen = set()
        for (src_id, tgt_id), edge in self.graph.edges.items():
            src_node = self.graph.get_node(src_id)
            tgt_node = self.graph.get_node(tgt_id)
            if src_node and tgt_node and src_node.label and tgt_node.label:
                s_label = src_node.label.lower()
                o_label = tgt_node.label.lower()
                if s_label == o_label:
                    continue
                key = (s_label, o_label)
                if key not in seen:
                    seen.add(key)
                    t = templates[len(sentences) % len(templates)]
                    sent = t.format(s=s_label, o=o_label)
                    sentences.append(sent)

        # Limit synthetic training to avoid overwhelming real English patterns
        sentences = sentences[:min_synthetic]
        total_trained = 0
        sleep_every = 200
        for sent in sentences:
            words = sent.split()
            err = self.neural_decoder.train_on_sentence(
                words, self._decoder_word_to_embed, self._decoder_word_to_idx)
            total_trained += 1
            if total_trained % sleep_every == 0:
                self.neural_decoder.sleep_cycle()
        self.neural_decoder.sleep_cycle()
        self._decoder_training_count = total_trained + self._decoder_web_training_count
        print(f"  [Decoder] Trained on {total_trained} natural graph sentences"
              f"{' + ' + str(self._decoder_web_training_count) + ' from web' if self._decoder_web_training_count > 0 else ''}")

    def _train_decoder_on_seed_corpus(self, max_sentences: int = 250) -> int:
        # Train the neural decoder on the bundled real-English corpus
        # (data/corpora/teen_seeds.txt). This is the difference between
        # a decoder that has seen only {s} connects with {o} template
        # garbage and a decoder that has actually seen English. Without
        # this, the decoder learns to imitate the very template phrases
        # that produce the trash output in chat.
        if self.neural_decoder is None or not self._decoder_vocab_built:
            return 0
        corpus_path = os.path.join(_proj_root, "data", "corpora", "teen_seeds.txt")
        if not os.path.exists(corpus_path):
            return 0
        try:
            with open(corpus_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            return 0
        # Pre-expand decoder vocab with corpus words so they can be
        # used as inputs/targets (GloVe-backed vectors).
        words_in_corpus = set(re.findall(r"[a-zA-Z\']{3,}", text.lower()))
        new_for_vocab = [w for w in words_in_corpus if w not in self._decoder_word_to_idx]
        if new_for_vocab:
            self._expand_decoder_vocab(new_for_vocab)

        # Pre-process corpus once, then loop with shuffled passes
        nd = self.neural_decoder
        all_sentences = nd.prepare_sentences(
            text, self._decoder_word_to_embed, self._decoder_word_to_idx,
            min_sentence_len=3,
        )
        n_available = len(all_sentences)
        sentences_per_pass = min(200, n_available)
        n_passes = 20
        sleep_interval = 5

        total_err = 0.0
        passes = 0
        rng = np.random.RandomState(42)

        for i in range(n_passes):
            idx = rng.permutation(n_available)
            batch = [all_sentences[idx[j]] for j in range(sentences_per_pass)]

            err_sum = 0.0
            for sent in batch:
                err = nd.train_on_sentence(
                    sent['words'], self._decoder_word_to_embed, self._decoder_word_to_idx,
                    word_indices=sent['word_indices'],
                    conditioning_embs=sent['conditioning_embs'],
                )
                err_sum += err
            avg_err = err_sum / sentences_per_pass
            total_err += avg_err
            passes += sentences_per_pass

            if (i + 1) % sleep_interval == 0:
                nd.sleep_cycle()

        if (n_passes % sleep_interval) != 0:
            nd.sleep_cycle()

        self._decoder_seed_training_count = passes
        self._decoder_training_count += passes
        return passes

    def _expand_decoder_vocab(self, new_words: List[str]):
        """Add new words to the decoder vocabulary (hippocampal replay)."

        New words get embeddings from GloVe (or random if unavailable).
        The decoder embedding table is resized and vocabulary caches rebuilt.
        Existing embeddings are preserved (neocortical consolidation).
        Stage 2: Respects MAX_DECODER_VOCAB_SIZE cap.
        """
        if not new_words or self.neural_decoder is None:
            return
        import sys
        if getattr(self, '_freeze_decoder_vocab', False):
            return
        max_vocab = self.MAX_DECODER_VOCAB_SIZE

        with self._vocab_lock:
            if len(self._decoder_word_to_idx) >= max_vocab:
                return
            added = 0
            for word in new_words:
                wl = word.lower().strip()
                if wl in self._decoder_word_to_idx or wl in ('<pad>', '<unk>', '<bos>', '<eos>'):
                    continue
                idx = len(self._decoder_word_to_idx)
                self._decoder_word_to_idx[wl] = idx
                self._decoder_idx_to_word[idx] = wl
                vec = self._glove_vector(wl)
                if vec is None:
                    vec = np.random.randn(self.dim).astype(np.float32) * 0.1
                    n = np.linalg.norm(vec)
                    if n > 0:
                        vec /= n
                self._decoder_word_to_embed[wl] = vec.astype(np.float32)
                added += 1

            if added == 0:
                return

            old_vocab_size = self.neural_decoder.vocab_size
            new_vocab_size = len(self._decoder_word_to_idx)

            # Resize decoder embedding table (preserve existing weights)
            old_weight = self.neural_decoder.word_embedding.weight.data
            new_weight = np.zeros((new_vocab_size, self.dim), dtype=np.float32)
            new_weight[:len(old_weight)] = old_weight
            for word, idx in list(self._decoder_word_to_idx.items()):
                if idx >= len(old_weight) and word in self._decoder_word_to_embed:
                    new_weight[idx] = self._decoder_word_to_embed[word]

        from ravana_ml.nn.module import Embedding, Linear
        new_emb = Embedding(new_vocab_size, self.dim)
        new_emb.weight.data = new_weight
        self.neural_decoder.word_embedding = new_emb

        # Resize output_proj weight and bias to match new vocab size
        old_out_w = self.neural_decoder.output_proj.weight.data
        new_out_w = np.zeros((new_vocab_size, self.neural_decoder.hidden_dim), dtype=np.float32)
        new_out_w[:old_vocab_size] = old_out_w
        new_out_proj = Linear(self.neural_decoder.hidden_dim, new_vocab_size)
        new_out_proj.weight.data = new_out_w
        if self.neural_decoder.output_proj.bias is not None:
            old_bias = self.neural_decoder.output_proj.bias.data
            new_bias = np.zeros(new_vocab_size, dtype=np.float32)
            new_bias[:old_vocab_size] = old_bias
            new_out_proj.bias.data = new_bias
        self.neural_decoder.output_proj = new_out_proj

        self.neural_decoder.vocab_size = new_vocab_size
        self.neural_decoder._vocab_dim = self.dim
        self.neural_decoder.rebuild_vocab_cache()

        # Hippocampal replay: mix new embeddings with similar existing ones
        # Build batch embedding matrix for existing words once (avoids O(V²) per new word)
        with self._vocab_lock:
            existing_labels = []
            existing_vecs = []
            new_set = set(w.lower().strip() for w in new_words)
            for ew, ei in list(self._decoder_word_to_idx.items()):
                if ew.startswith('<') or ew in new_set:
                    continue
                ev = self._decoder_word_to_embed.get(ew)
                if ev is not None:
                    existing_labels.append(ew)
                    existing_vecs.append(ev)
            if existing_vecs:
                existing_mat = np.stack(existing_vecs, axis=0).astype(np.float32)
            else:
                existing_mat = np.empty((0, self.dim), dtype=np.float32)

        for word in new_words:
            wl = word.lower().strip()
            idx = self._decoder_word_to_idx.get(wl)
            if idx is None:
                continue
            vec = self._decoder_word_to_embed.get(wl)
            if vec is None:
                continue
            if len(existing_labels) > 0:
                sims = existing_mat @ vec
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                best_label = existing_labels[best_idx]
                # Blend: 70% new + 30% nearest (hippocampal-neocortical consolidation)
                if best_sim > 0.3:
                    best_vec = self._decoder_word_to_embed[best_label]
                    blended = vec * 0.7 + best_vec * 0.3
                    blended /= np.linalg.norm(blended)
                    self._decoder_word_to_embed[wl] = blended
                    new_weight[idx] = blended
        self.neural_decoder.word_embedding.weight.data = new_weight

        # Mini training pass on new words using batch similarity
        for nw in new_words:
            nwl = nw.lower().strip()
            nidx = self._decoder_word_to_idx.get(nwl)
            if nidx is None:
                continue
            nvec = self._decoder_word_to_embed.get(nwl)
            if nvec is None:
                continue
            if len(existing_labels) > 0:
                sims = existing_mat @ nvec
                above_thresh = np.where(sims > 0.4)[0]
                if len(above_thresh) > 0:
                    top_k = min(3, len(above_thresh))
                    top_order = np.argsort(sims[above_thresh])[::-1][:top_k]
                    for ni in above_thresh[top_order]:
                        neighbor = existing_labels[ni]
                        bridge = f"{nwl} and {neighbor} are related"
                        self.neural_decoder.train_on_sentence(
                            bridge.split(), self._decoder_word_to_embed, self._decoder_word_to_idx)

        print(f"  [Decoder] Expanded vocab by {added} words (now {new_vocab_size})")

    def _build_conditioning_for_text(self, topic: str, text_words: List[str]) -> Optional[np.ndarray]:
        """Build graph walk conditioning embeddings for a text's important words.

        Walks from the topic through up to 5 of the most frequent content words
        in the text, collecting their concept embeddings from the graph.
        Falls back to text word embeddings if no graph concepts match.
        """
        concept_embs = []
        seen = set()

        # Collect from graph concepts first
        tl = topic.lower().strip()
        if tl in self._concept_keywords:
            nids = self._concept_keywords[tl]
            node = self.graph.get_node(nids[0])
            if node and node.vector is not None:
                concept_embs.append(node.vector.copy())
                seen.add(tl)

        counted = {}
        for w in text_words:
            wl = w.lower().strip(".,!?\"' ")
            if wl not in STOP_WORDS and len(wl) >= 3:
                counted[wl] = counted.get(wl, 0) + 1
        for w, _ in sorted(counted.items(), key=lambda x: x[1], reverse=True)[:5]:
            if w not in seen and w in self._concept_keywords:
                nids = self._concept_keywords[w]
                node = self.graph.get_node(nids[0])
                if node and node.vector is not None:
                    concept_embs.append(node.vector.copy())
                    seen.add(w)

        # Fall back to GloVe/decoder embeddings if no graph concepts found
        if len(concept_embs) < 1:
            tl = topic.lower().strip()
            if tl in self._decoder_word_to_embed:
                concept_embs.append(self._decoder_word_to_embed[tl].copy())
                seen.add(tl)
            for w, _ in sorted(counted.items(), key=lambda x: x[1], reverse=True)[:5]:
                if w not in seen and w in self._decoder_word_to_embed:
                    concept_embs.append(self._decoder_word_to_embed[w].copy())
                    seen.add(w)

        if len(concept_embs) < 1:
            return None
        return np.stack(concept_embs, axis=0).astype(np.float32)

    # ─── Neural Decoder Generation ───

    def _generate_with_decoder(self, ctx: CognitiveResponseContext) -> Optional[str]:
        """Generate response using NeuralDecoder with graph conditioning.

        Builds conditioning embeddings from the subject and associated concepts,
        then runs the decoder autoregressively with cerebellar n-gram bias and
        basal ganglia gating for biologically-plausible word selection.

        Returns None if decoder is not ready or generation fails.
        (Quality gate is now in _generate_response.)
        """
        if self.neural_decoder is None or not self._decoder_vocab_built:
            return None

        subject = ctx.subject
        if not subject or subject.lower() not in self._decoder_word_to_idx:
            return None

        # Build conditioning embeddings from graph concepts
        concept_embs = []
        seen_labels: Set[str] = set()

        # Subject embedding (core concept) — lock graph reads
        with self._graph_lock:
            subj_lower = subject.lower()
            if subj_lower in self._concept_keywords:
                nids = self._concept_keywords[subj_lower]
                node = self.graph.get_node(nids[0])
                if node and node.vector is not None:
                    concept_embs.append(node.vector.copy())
                    seen_labels.add(subj_lower)

            # Associated concepts with weighted significance
            for label, score in ctx.associated_concepts[:6]:
                ll = label.lower()
                if ll not in seen_labels and ll in self._concept_keywords:
                    nids = self._concept_keywords[ll]
                    node = self.graph.get_node(nids[0])
                    if node and node.vector is not None:
                        weight = 0.5 + min(score, 1.0) * 0.5
                        concept_embs.append(node.vector.copy() * weight)
                        seen_labels.add(ll)
                        if len(concept_embs) >= 6:
                            break

        # Fallback: use decoder word embeddings if no graph nodes found
        if len(concept_embs) < 1:
            if subj_lower in self._decoder_word_to_embed:
                concept_embs.append(self._decoder_word_to_embed[subj_lower].copy())
            else:
                return None

        # Add sentence-level compositional vector (N400/P600 integration) as conditioning
        sent_vec = getattr(ctx, 'sentence_vector', None)
        if sent_vec is not None and np.any(sent_vec != 0):
            concept_embs.append(sent_vec.astype(np.float32) * 0.6)

        conditioning_embs = np.stack(concept_embs, axis=0).astype(np.float32)

        # Special token indices
        bos_idx = self._decoder_word_to_idx.get("<bos>", 0)
        eos_idx = self._decoder_word_to_idx.get("<eos>", 2)

        # Dynamic temperature from cognitive state
        # Stage 2: Lower base temp for 1500-vocab model (more deterministic → content words)
        base_temp = 0.35
        arousal = ctx.arousal if hasattr(ctx, 'arousal') else 0.3
        temp = base_temp * (0.7 + arousal * 0.6)
        # Higher dopamine tone = more creative/exploratory
        dt = getattr(self, '_dopamine_tone', 0.5)
        temp *= (0.7 + dt * 0.6)
        temp = max(0.15, min(0.7, temp))

        # Build content word IDs (non-function, non-special tokens)
        # IMPORTANT: Do NOT include semantic connector words like "connects",
        # "relates", "links", "leads" here — those are content words whose
        # presence in the function-word set was causing the decoder to default
        # to template-like "X connects with Y" patterns.
        function_words_set = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "not", "no", "nor", "so", "if", "then", "than", "too", "very",
            "just", "about", "also", "into", "over", "after", "before",
            "between", "through", "during", "because", "while", "which",
            "who", "whom", "what", "when", "where", "why", "how", "all",
            "each", "every", "both", "few", "more", "most", "some", "any",
            "this", "that", "these", "those", "it", "its", "they", "them",
            "their", "we", "our", "you", "your", "he", "she", "him", "her",
            "his", "i", "me", "my", "myself", "am",
            "of", "to", "for", "with", "from", "at", "by", "as", "on",
            "<pad>", "?", "<bos>", "<eos>",
        }
        with self._vocab_lock:
            content_word_ids = {
                idx for word, idx in self._decoder_word_to_idx.items()
                if word not in function_words_set and not word.startswith("<")
            }

        # Stage 2: Boost subject concept in conditioning to seed content words
        subject_idx = self._decoder_word_to_idx.get(subject.lower())
        if subject_idx is not None:
            subject_boost = {subject_idx: 3.0}
        else:
            subject_boost = None

        try:
            generated = self.neural_decoder.generate(
                conditioning_embs=conditioning_embs,
                max_steps=28,
                bos_idx=bos_idx,
                eos_idx=eos_idx,
                temperature=temp,
                cerebellar_ngram=self.cerebellar_ngram,
                idx_to_word=self._decoder_idx_to_word,
                basal_ganglia=self.basal_ganglia,
                content_word_ids=content_word_ids,
                token_boost=subject_boost,
            )

            if not generated or len(generated) < 3:
                return None

            # Convert indices to words, filtering special tokens
            words = []
            for idx in generated:
                word = self._decoder_idx_to_word.get(idx, "")
                if word and not word.startswith("<"):
                    words.append(word)

            if len(words) < 3:
                return None

            text = " ".join(words)

            # Quality gate: if >60% of output is function words, fall through
            # to syntactic pipeline or reasoning loop (decoder not ready yet)
            func_set = {"a","an","the","is","are","was","were","be","been",
                "being","have","has","had","do","does","did","will",
                "would","could","should","may","might","shall","can",
                "not","no","nor","so","if","then","than","too","very",
                "just","about","also","into","over","after","before",
                "between","through","during","because","while","which",
                "who","whom","what","when","where","why","how","all",
                "each","every","both","few","more", "most", "some", "any",
                "this", "that", "these", "those", "it", "its", "they", "them",
                "their", "we", "our", "you", "your", "he", "she", "him", "her",
                "his", "i", "me", "my", "myself", "am",
                "of", "to", "for", "with", "from", "at", "by", "as", "on"}
            func_count = sum(1 for w in words if w.lower() in func_set)
            if len(words) > 0 and func_count / len(words) > 0.70:
                return None

            # Template-pattern gate: reject output that looks like "X connects with Y"
            # or "X relates to Y" — these are the synthetic graph training artifacts.
            template_verbs = {"connects", "connect", "relates", "relate", "links",
                               "link", "leads", "lead", "refers", "involves",
                               "associated"}
            template_preps = {"with", "to", "from", "into"}
            text_lower = text.lower()
            # Reject if pattern like "word connects/relates/links with/to word" appears
            for tv in template_verbs:
                for tp in template_preps:
                    if re.search(rf'\b\w+\s+{tv}\s+{tp}\s+\w+', text_lower):
                        return None

            # Bigram repetition gate: reject if any bigram appears 3+ times
            if len(words) >= 4:
                bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
                from collections import Counter
                bg_counts = Counter(bigrams)
                if any(c >= 3 for c in bg_counts.values()):
                    return None

            # Clean up basic punctuation issues
            text = text[0].upper() + text[1:]
            if not text.endswith((".", "?", "!")):
                text += "."
            # Remove double spaces
            text = re.sub(r'\s+', ' ', text)

            return text
        except Exception as e:
            if self._trace_enabled:
                print(f"  [trace] decoder generation error: {e}")
            return None

    def _seed_concepts(self):
        """Seed the graph with teenager-level vocabulary and typed relationships."""
        self._concept_labels = set()
        all_labels = {label.lower() for label, _ in TEEN_CONCEPTS}
        for label, keywords in TEEN_CONCEPTS:
            # GloVe vector if available, else hash-based random
            vec = self._glove_vector(label)
            if vec is None:
                h = hash(label) % 10000
                vr = np.random.RandomState(h + 42)
                vec = vr.randn(self.dim).astype(np.float32) * 0.15
                for kw in keywords.split():
                    kh = hash(kw) % 5000
                    kr = np.random.RandomState(kh)
                    vec += kr.randn(self.dim).astype(np.float32) * 0.05
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
            node = self.graph.add_node(vector=vec, label=label)
            self._concept_labels.add(label.lower())

            # Map keywords, skipping those that shadow another concept's label
            for kw in keywords.split():
                kl = kw.lower()
                if kl in all_labels:
                    continue
                self._concept_keywords.setdefault(kl, []).append(node.id)
            self._concept_keywords.setdefault(label, []).append(node.id)
            if "_" in label:
                for part in label.split("_"):
                    if len(part) >= 3:
                        self._concept_keywords.setdefault(part, []).append(node.id)

        # Build concept_id map after seeding
        label_to_id = {}
        for nid, node in self.graph.nodes.items():
            if node.label:
                label_to_id[node.label] = nid

        # ── Typed Edges: (source, target, relation_type, weight) ──
        # Semantic (is-a, similar to): baseline connections
        semantic_edges = [
            ("hello", "bye"), ("hello", "thanks"), ("thanks", "sorry"),
            ("yes", "no"), ("good", "bad"), ("big", "small"), ("hot", "cold"),
            ("up", "down"), ("in", "out"), ("here", "there"),
            ("now", "later"), ("more", "some"), ("one", "many"),
            ("happy", "sad"), ("joy", "grief"), ("excited", "bored"),
            ("love", "hate"), ("hope", "fear"),
            ("trust", "respect"), ("freedom", "responsibility"),
            ("knowledge", "wisdom"), ("knowledge", "truth"),
            ("sun", "moon"), ("sun", "light"), ("tree", "bird"),
            ("dog", "cat"), ("dog", "friend"),
            ("water", "rain"), ("food", "eat"),
            ("book", "read"), ("book", "story"), ("song", "music"),
            ("science", "knowledge"), ("history", "story"),
            ("life", "death"), ("mind", "heart"),
            ("time", "change"), ("world", "nature"),
            ("time", "future"), ("time", "past"), ("future", "past"),
            ("machine", "invention"), ("invention", "create"),
            ("imagination", "dream"), ("imagination", "create"),
            ("impossible", "possible"), ("possibility", "maybe"),
            ("future", "change"), ("past", "history"),
            ("invent", "create"), ("invent", "explore"),
            ("journey", "explore"), ("experiment", "discovery"),
            ("i", "we"), ("i", "you"),
            ("all", "many"), ("work", "effort"),
            ("play", "fun"), ("play", "game"),
            ("why", "because"), ("if", "maybe"),
        ]

        # Causal: A causes/creates/produces B
        causal_edges = [
            ("sun", "hot"), ("sun", "light"),
            ("eat", "food"), ("drink", "water"),
            ("sleep", "tired"), ("play", "happy"),
            ("love", "joy"), ("fear", "anxiety"),
            ("cause", "change"), ("learn", "knowledge"),
            ("study", "understanding"), ("practice", "skill"),
            ("challenge", "struggle"), ("struggle", "growth"),
            ("power", "responsibility"), ("freedom", "choice"),
            ("question", "curiosity"), ("explore", "discovery"),
            ("trust", "friendship"), ("hypocrisy", "distrust"),
            ("grief", "sadness"), ("hope", "motivation"),
            ("rejection", "loneliness"), ("acceptance", "belonging"),
            ("anger", "conflict"), ("empathy", "understanding"),
            ("criticism", "growth"), ("failure", "learning"),
            ("change", "growth"), ("time", "change"),
            ("knowledge", "wisdom"), ("experience", "wisdom"),
            ("art", "expression"), ("science", "progress"),
            ("imagination", "invention"), ("experiment", "knowledge"),
            ("machine", "future"), ("journey", "discovery"),
        ]

        # Temporal: A then/before/after B
        temporal_edges = [
            ("birth", "life"), ("life", "death"),
            ("morning", "noon"), ("noon", "night"),
            ("question", "answer"), ("cause", "effect"),
            ("learn", "understand"), ("struggle", "succeed"),
        ]

        # Contrastive: A vs B (opposites)
        contrastive_edges = [
            ("good", "bad"), ("love", "hate"), ("life", "death"),
            ("truth", "lie"), ("freedom", "oppression"),
            ("courage", "fear"), ("hope", "despair"),
            ("knowledge", "ignorance"), ("justice", "injustice"),
            ("create", "destroy"), ("accept", "reject"),
            ("always", "never"),
        ]

        # Analogical: A is like B (different domains, similar pattern)
        analogical_edges = [
            ("mind", "garden"), ("life", "journey"),
            ("knowledge", "light"), ("time", "river"),
            ("society", "organism"), ("identity", "building"),
            ("memory", "library"), ("heart", "engine"),
        ]

        # Apply all edges with their types
        self._apply_edges(label_to_id, semantic_edges, "semantic", 0.35)
        self._apply_edges(label_to_id, causal_edges, "causal", 0.45)
        self._apply_edges(label_to_id, temporal_edges, "temporal", 0.40)
        self._apply_edges(label_to_id, contrastive_edges, "contrastive", 0.50)
        self._apply_edges(label_to_id, analogical_edges, "analogical", 0.30)

        # ── Hub edges: connect self-referential and high-frequency concepts
        # This densifies the graph so response generation can walk richer paths
        hub_edges = [
            # Self → cognitive actions
            ("i", "think", "causal"), ("i", "feel", "causal"),
            ("i", "know", "causal"), ("i", "want", "causal"),
            ("i", "like", "causal"), ("i", "say", "causal"),
            ("we", "people", "semantic"), ("we", "they", "semantic"),
            ("you", "person", "semantic"), ("you", "friend", "semantic"),
            # Cognitive hubs → content
            ("think", "knowledge", "causal"), ("think", "question", "causal"),
            ("think", "analyze", "causal"), ("think", "explore", "causal"),
            ("feel", "love", "causal"), ("feel", "happy", "causal"),
            ("feel", "sad", "causal"), ("feel", "fear", "causal"),
            ("feel", "excited", "causal"),
            ("know", "learn", "causal"), ("know", "truth", "semantic"),
            ("know", "knowledge", "semantic"), ("know", "understand", "causal"),
            ("want", "freedom", "causal"), ("want", "love", "causal"),
            ("want", "more", "causal"), ("want", "power", "causal"),
            ("like", "love", "semantic"), ("like", "play", "causal"),
            ("like", "friend", "semantic"), ("like", "music", "semantic"),
            ("like", "art", "semantic"),
            # Extra connective density
            ("people", "society", "semantic"), ("people", "culture", "semantic"),
            ("learn", "knowledge", "causal"), ("learn", "understand", "causal"),
            ("understand", "knowledge", "semantic"),
            ("question", "curious", "causal"), ("explore", "discover", "causal"),
        ]
        for src, tgt, rel in hub_edges:
            sid = label_to_id.get(src)
            tid = label_to_id.get(tgt)
            if sid is not None and tid is not None and self.graph.get_edge(sid, tid) is None:
                bw = 0.35 if rel == "semantic" else 0.45
                self.graph.add_edge(sid, tid, weight=bw + self.rng.uniform(0, 0.15),
                                   relation_type=rel)

        # ── Auto-wire: connect GloVe-similar concepts (>0.6) as semantic edges
        auto_count = 0
        nids = list(label_to_id.values())
        for i in range(len(nids)):
            ni = self.graph.get_node(nids[i])
            if ni is None or ni.vector is None:
                continue
            for j in range(i + 1, len(nids)):
                if self.graph.get_edge(nids[i], nids[j]) is not None or self.graph.get_edge(nids[j], nids[i]) is not None:
                    continue
                nj = self.graph.get_node(nids[j])
                if nj is None or nj.vector is None:
                    continue
                sim = float(np.dot(ni.vector, nj.vector))
                if sim > 0.6:
                    weight = min(0.5, sim * 0.5)
                    # Infer proper relation type instead of always "semantic"
                    inf_type, _ = self._infer_relation_type(ni.label, nj.label, "semantic")
                    edge = self.graph.add_edge(nids[i], nids[j], weight=weight, relation_type=inf_type)
                    edge.confidence = 0.001  # dormant: invisible until visited
                    self._dormant_edges.add((nids[i], nids[j]))
                    auto_count += 1

        self._all_labels = label_to_id
        self._build_contradiction_map(contrastive_edges)
        # Fix seeded edge types using relation inference
        migrated = self._correct_relation_types()
        # Phase A: Build concept POS tags from seeded concepts
        self._build_concept_pos()
        # Phase C: Seed cerebellar n-gram from concept POS tags
        if hasattr(self, 'cerebellar_ngram') and hasattr(self, '_concept_pos'):
            self.cerebellar_ngram.seed_from_pos(self._concept_pos)
        # Phase E: Seed syntactic cell assemblies from concept POS tags
        if hasattr(self, 'syntactic_assembly') and hasattr(self, '_concept_pos'):
            self.syntactic_assembly.seed_from_pos(self._concept_pos)
        extra = f", {migrated} reclassified" if migrated else ""
        print(f"  [Teen] Seeded {len(self.graph.nodes)} concepts, {len(self.graph.edges)} connections ({auto_count} auto-wired) across 5 relation types{extra}")

    def _build_concept_pos(self):
        """Build POS tags for seeded concepts based on word characteristics."""
        verb_suffixes = ['ing', 'ed', 'ize', 'ify', 'ate', 'en', 'ish']
        adj_suffixes = ['able', 'ible', 'ful', 'less', 'ous', 'al', 'ic', 'ive']
        known_verbs = {'know', 'think', 'feel', 'want', 'need', 'like', 'love',
                      'go', 'come', 'see', 'hear', 'eat', 'drink', 'sleep', 'play',
                      'help', 'make', 'get', 'say', 'give', 'take', 'cause', 'change',
                      'grow', 'learn', 'teach', 'create', 'destroy', 'protect', 'accept',
                      'reject', 'invent', 'connect', 'influence', 'struggle', 'challenge',
                      'analyze', 'conclude', 'reflect', 'question', 'explore', 'understand',
                      'compare', 'criticize', 'assume', 'imagine'}
        known_adjs = {'good', 'bad', 'big', 'small', 'hot', 'cold', 'happy', 'sad',
                     'scared', 'angry', 'tired', 'excited', 'curious', 'confused',
                     'bored', 'proud', 'lonely', 'grateful', 'complex', 'significant',
                     'fundamental', 'inevitable', 'possible', 'obvious', 'subtle',
                     'profound'}
        # Iterate over all graph node labels, not just _concept_labels
        for node in self.graph.nodes.values():
            if node.label:
                ll = node.label.lower()
                if ll in known_verbs:
                    self._concept_pos[ll] = 'verb'
                elif ll in known_adjs:
                    self._concept_pos[ll] = 'adj'
                elif any(len(ll) > len(s) + 1 and ll.endswith(s) for s in verb_suffixes):
                    self._concept_pos[ll] = 'verb'
                elif any(len(ll) > len(s) + 1 and ll.endswith(s) for s in adj_suffixes):
                    self._concept_pos[ll] = 'adj'
                else:
                    self._concept_pos[ll] = 'noun'


    def _bootstrap_domain_concepts(self):
        """Seed domain-specific concepts (oxiverse, intentforge, ravana) with rich structure."""
        domain_relation_type_map = {
            "is_a": "semantic",
            "causal": "causal",
            "contrastive": "contrastive",
            "part_of": "contextual",
        }

        def _ensure_concept(label):
            for n in self.graph.nodes.values():
                if n.label == label:
                    return n.id
            vec = self._glove_vector(label)
            if vec is None:
                h = hash(label) % 50000
                vr = np.random.RandomState(h + 100)
                vec = vr.randn(self.dim).astype(np.float32) * 0.1
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
            n = self.graph.add_node(vector=vec, label=label)
            n.stability = 0.8
            self._concept_labels.add(label.lower())
            self._concept_keywords.setdefault(label, []).append(n.id)
            for part in label.split():
                if len(part) >= 3 and part not in self._concept_labels:
                    self._concept_keywords.setdefault(part, []).append(n.id)
            return n.id

        created = []
        for concept_name, meta in DOMAIN_CONCEPTS.items():
            if any(node.label == concept_name for node in self.graph.nodes.values()):
                continue
            node_id = _ensure_concept(concept_name)
            for kw in meta["keywords"].split():
                if kw not in self._concept_labels:
                    self._concept_keywords.setdefault(kw, []).append(node_id)
            for (src, tgt, rel_type, weight) in meta["relations"]:
                src_id = node_id if src == concept_name else _ensure_concept(src)
                tgt_id = node_id if tgt == concept_name else _ensure_concept(tgt)
                mapped_rel = domain_relation_type_map.get(rel_type, "semantic")
                if self.graph.get_edge(src_id, tgt_id) is None:
                    self.graph.add_edge(src_id, tgt_id, weight=weight, confidence=0.9, relation_type=mapped_rel)
            created.append(concept_name)
        if created:
            print(f"  [Domain] Bootstrapped {len(created)} domain concepts: {', '.join(created)}")

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
        for nid, node in self.graph.nodes.items():
            if node.label:
                pe = getattr(node, 'prediction_free_energy', 0.0)
                if pe > 0.1:
                    label = node.label.lower()
                    if label not in seen and len(label) >= 3:
                        scores[label] = scores.get(label, 0.0) + pe * 2.0  # weight node PE higher
                        seen.add(label)
        
        # Source 2: Edge-level prediction free energy (edges with high prediction error)
        for (src, tgt), edge in self.graph.edges.items():
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

    # ─── Phase 1: Auto-Expansion from Every Message ───

    def _auto_expand_concepts(self, text: str) -> int:
        """Phase 1.1+1.2: Auto-expand graph from user input.

        For each meaningful word in input that is not yet a graph concept:
        - If word has a GloVe vector: add as concept node, wire to top-5 GloVe
          nearest neighbors (Phase 1.1), then auto-wire to ALL existing concepts
          with cosine > 0.5 (Phase 1.2).
        - If word has no GloVe vector: skip (web search handles later).

        Returns number of new concepts added.
        """
        # Extract meaningful words from input
        words = re.findall(r"[a-zA-Z']{3,}", text.lower())
        meaningful = set()
        for w in words:
            wc = w.strip("'")
            if wc not in STOP_WORDS and wc not in WEB_GARBAGE and len(wc) >= 3:
                meaningful.add(wc)
        if not meaningful:
            return 0

        # Build set of existing graph labels for fast lookup
        existing_labels = set()
        for nid, node in self.graph.nodes.items():
            if node.label:
                existing_labels.add(node.label.lower())

        new_count = 0
        new_nodes: Dict[str, int] = {}  # word -> node_id

        # Phase 1.1: Add each new word as a concept (GloVe only, skip non-GloVe)
        for word in meaningful:
            if word in existing_labels:
                continue
            vec = self._glove_vector(word)
            if vec is None:
                continue  # Not in GloVe — web search can handle later
            node = self.graph.add_node(vector=vec, label=word)
            self._concept_keywords[word] = self._concept_keywords.get(word, []) + [node.id]
            self._concept_labels.add(word.lower())
            existing_labels.add(word)
            new_nodes[word] = node.id
            new_count += 1

        if not new_nodes:
            return 0

        # Phase 1.1+1.2: Wire new concepts to existing graph concepts via vector similarity
        for word, nid in new_nodes.items():
            node = self.graph.get_node(nid)
            if node is None or node.vector is None:
                continue

            # FAISS/vectorized: matrix multiply instead of per-node Python loop
            similarities = []
            # Ensure vector matrix is rebuilt if dirty (nodes may have been recently added)
            if self.graph._vectors_dirty or self.graph._vector_matrix_normed is None:
                self.graph._rebuild_vector_matrix()
            if self.graph._vector_matrix_normed is not None and len(self.graph._node_id_order) > 0:
                vec_norm = node.vector / (np.linalg.norm(node.vector) + 1e-15)
                all_sims = self.graph._vector_matrix_normed @ vec_norm.astype(np.float32)
                for idx in np.where(all_sims > 0.3)[0]:
                    other_nid = self.graph._node_id_order[idx]
                    if other_nid == nid:
                        continue
                    other_node = self.graph.get_node(other_nid)
                    if other_node is None or other_node.label is None or other_node.vector is None:
                        continue
                    if other_node.label.lower() in new_nodes:
                        continue
                    similarities.append((other_nid, float(all_sims[idx])))

            if not similarities:
                continue

            # Phase 2.4: Use np.argpartition for O(N) top-5 selection
            # instead of O(N log N) full sort
            sim_array = np.array([s[1] for s in similarities], dtype=np.float32)
            nid_array = np.array([s[0] for s in similarities], dtype=np.int32)

            # Phase 9b: Wire using prediction-error-based learning
            # Initial weight = raw GloVe similarity (no arbitrary cap)
            # Then one gradient step refines the weight toward accurate prediction
            k_top = min(5, len(sim_array))
            if k_top > 0:
                top_idx = np.argpartition(sim_array, -k_top)[-k_top:]
                top_order = np.argsort(-sim_array[top_idx])
                top_idx = top_idx[top_order]
                wired_11 = 0
                for idx in top_idx:
                    if wired_11 >= 5:
                        break
                    sim = float(sim_array[idx])
                    other_nid = int(nid_array[idx])
                    if sim > 0.4 and self.graph.get_edge(nid, other_nid) is None:
                        weight = min(0.6, sim * 0.6)
                        other_node = self.graph.get_node(other_nid)
                        other_label = other_node.label if other_node else word
                        inf_type, _ = self._infer_relation_type(word, other_label, "semantic")
                        self.graph.add_edge(nid, other_nid, weight=weight, relation_type=inf_type)
                        wired_11 += 1

            # Phase 1.2 + 9b: Auto-wire to ALL existing concepts where sim > 0.5
            for idx in range(len(similarities)):
                sim = float(sim_array[idx])
                other_nid = int(nid_array[idx])
                if sim > 0.5 and self.graph.get_edge(nid, other_nid) is None:
                    weight = min(0.6, sim * 0.6)
                    other_node = self.graph.get_node(other_nid)
                    other_label = other_node.label if other_node else word
                    inf_type, _ = self._infer_relation_type(word, other_label, "semantic")
                    self.graph.add_edge(nid, other_nid, weight=weight, relation_type=inf_type)

        return new_count

    # ─── Phase 9b: Active Inference / Prediction Error ───
    # Based on: Friston's Free Energy Principle & Active Inference
    # (Friston, Parr, Pezzulo 2022; Bogacz 2017 tutorial)
    #
    # Each edge weight is a PREDICTION: "from concept A, you can reach concept B
    # with this expected similarity." When the actual GloVe vector similarity differs
    # from the edge weight, prediction error drives learning — not additive increments.

    def _prediction_error(self, src_vec: np.ndarray, tgt_vec: np.ndarray,
                           current_weight: float) -> Tuple[float, float]:
        """Compute prediction error and gradient for an edge weight.

        Prediction: edge weight predicts cosine similarity between vectors.
        Error: MSE between predicted similarity and actual cosine similarity.
        Gradient: d(error)/d(weight) = 2 * (weight - cosine) * (-1)

        Returns (error, gradient).
        """
        actual_sim = float(np.dot(src_vec, tgt_vec))
        error = (current_weight - actual_sim) ** 2
        gradient = 2.0 * (current_weight - actual_sim)
        return error, gradient

    def _update_edge_from_error(self, edge, src_vec: np.ndarray,
                                 tgt_vec: np.ndarray, learning_rate: float = 0.15):
        """Update edge weight using gradient descent on prediction error.

        Low error → weight stays (prediction is accurate)
        High error → weight moves toward actual similarity (prediction improves)
        Confidence adjusts inversely to error (low error → high confidence)
        """
        error, gradient = self._prediction_error(src_vec, tgt_vec, edge.weight)
        # Gradient descent: move weight toward reducing error
        edge.weight -= learning_rate * gradient * 0.5
        edge.weight = np.clip(edge.weight, 0.01, 0.99)
        # Confidence: inverse of error (squashed to 0-1)
        edge.confidence = max(0.05, 1.0 - np.tanh(error * 3.0))
        # Track edge-level prediction free energy (for curiosity drive)
        # Exponential moving average of prediction error on this edge
        alpha = 0.1
        edge.prediction_free_energy = (1 - alpha) * edge.prediction_free_energy + alpha * error
        edge.prediction_count += 1
        # Track running mean prediction error (surprise signal)
        alpha_global = 0.05
        self._mean_prediction_error = (1 - alpha_global) * self._mean_prediction_error + alpha_global * error
        self._prediction_error_count += 1
        return error

    def _build_contradiction_map(self, contrastive_edges: List[Tuple[str, str]]):
        """Build a map of concept → set of antonym concepts from contrastive edges."""
        for a, b in contrastive_edges:
            al, bl = a.lower(), b.lower()
            self._contradiction_map.setdefault(al, set()).add(bl)
            self._contradiction_map.setdefault(bl, set()).add(al)

    def _rebuild_contradiction_map(self):
        """Rebuild contradiction map from graph edges (used after load)."""
        for (src, tgt), edge in self.graph.edges.items():
            if edge.relation_type != "contrastive":
                continue
            sn = self.graph.nodes.get(src)
            tn = self.graph.nodes.get(tgt)
            if sn and sn.label and tn and tn.label:
                al, bl = sn.label.lower(), tn.label.lower()
                self._contradiction_map.setdefault(al, set()).add(bl)
                self._contradiction_map.setdefault(bl, set()).add(al)

    def _is_contradictory(self, source_label: str, target_label: str,
                          relation_type: str) -> bool:
        """Check if asserting (source, relation, target) contradicts a logged belief.

        A contradiction occurs when source already asserted about concept X,
        and target is a known antonym of X (from the contradiction map).
        """
        tl = target_label.lower()
        sl = source_label.lower()
        for bel_src, bel_rel, bel_tgt in self._belief_assertions:
            if bel_src.lower() != sl:
                continue
            # Check if target is an antonym of the previously asserted object
            bel_antonyms = self._contradiction_map.get(bel_tgt.lower(), set())
            if tl in bel_antonyms:
                return True
        return False

    def _log_assertions(self, chains: List[str], subject: str):
        """Extract and log (subject, relation, object) triples from generated chains."""
        starter_values = set(self._EDGE_TO_STARTER.values())
        for chain_str in chains:
            parts = chain_str.strip(".").split()
            # Skip discourse starter if present (e.g. "but", "then", "like")
            start = 1 if parts and parts[0] in starter_values else 0
            if len(parts) - start < 3:
                continue
            if start == 0:
                cur_subj = parts[0]
            else:
                cur_subj = parts[1]
            # Walk (connector, object) pairs
            i = start + 1 if start == 0 else start + 2
            actual_start = start + 1 if start == 0 else start + 1
            for j in range(actual_start, len(parts) - 1, 2):
                connector = parts[j]
                obj = parts[j + 1]
                rel_type = "semantic"
                for rtype, glabel in self._EDGE_TO_GRAPH_LABEL.items():
                    if glabel == connector:
                        rel_type = rtype
                        break
                self._belief_assertions.append((cur_subj, rel_type, obj))
                cur_subj = obj

    def _apply_edges(self, label_to_id, edge_list, rel_type, base_weight):
        """Apply a list of (src, tgt) edges to the graph with a relation type."""
        for src, tgt in edge_list:
            sid = label_to_id.get(src)
            tid = label_to_id.get(tgt)
            if sid is not None and tid is not None and self.graph.get_edge(sid, tid) is None:
                self.graph.add_edge(sid, tid, weight=base_weight + self.rng.uniform(0, 0.15),
                                   relation_type=rel_type)

    # ─── Web Learning ───

    def learn_from_web(self, query: str, max_results: int = 3) -> Tuple[str, str]:
        """Search the web, fetch articles, extract concepts, and learn from them.

        Phase 5: Auto offline fallback. If the network API fails (timeout, DNS,
        HTTP error), sets _network_available = False and falls back silently.
        No error messages leak to the user — just returns a short summary.
        """
        self._learned_this_turn = True
        self._learning_count += 1

        # Phase 5: Skip API call if we already know the network is down
        # (unless it's time for a retry — check every 20 turns)
        if self._network_available is False:
            if self._network_retry_turn > 0 and self.turn_count >= self._network_retry_turn:
                self._network_available = None  # try again
                self._network_retry_turn = 0
            elif self._network_retry_turn == 0:
                # First time being offline — schedule a retry
                self._network_retry_turn = self.turn_count + 20
                known_count = self._learn_from_text(query + " " + query, query, source_url=query)
                if known_count > 0:
                    return f"learned {known_count} things about {query}"
                return f"offline - already knew about {query}"
            else:
                # Still try GloVe-only learning from the query text itself
                known_count = self._learn_from_text(query + " " + query, query, source_url=query)
                if known_count > 0:
                    return f"learned {known_count} things about {query}"
                return f"offline - already knew about {query}"

        query_clean = quote(query)

        try:
            # Step 1: Search with fallback engines (circuit breaker)
            try:
                results = self.search_engine.search(query, max_results=max_results)
                if self._network_available is None:
                    self._network_available = True
            except SearchError:
                # All search engines failed
                raise URLError("All search APIs failed")

            if not results:
                return self._learn_from_snippets(query, [])

            # Step 2: Get snippets and try to fetch article text
            snippets = []

            # Parallel article fetching (ThreadPoolExecutor)
            def _fetch_single(r):
                snippet = r.get("content", "") or ""
                title = r.get("title", "") or ""
                url = r.get("url", "") or ""
                text = f"{title}. {snippet}"
                if url:
                    try:
                        article_text = self._fetch_article_text(url)
                        if article_text and len(article_text) > len(text):
                            text = article_text[:5000]
                    except Exception:
                        pass
                return text
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(_fetch_single, r) for r in results[:3]]
                for future in as_completed(futures):
                    snippets.append(future.result())

            # Step 3: Extract and learn concepts from all gathered text
            combined_text = " ".join(snippets)
            first_url = results[0].get("url", "") if results else ""
            new_concepts_added = self._learn_from_text(combined_text, query, source_url=first_url if first_url else query)

            if new_concepts_added > 0:
                return f"learned {new_concepts_added} new things about {query}", combined_text
            else:
                return f"read about {query} but already knew the words", combined_text

        except (urllib.request.URLError, urllib.request.HTTPError,
                ConnectionError, TimeoutError, OSError, json.JSONDecodeError):
            # Phase 5: Network failure — mark as unavailable, fall back silently
            if self._trace_enabled:
                print(f"  [bg] Network error in learn_from_web: {type(e).__name__}")
            self._network_available = False
            # Still try GloVe-only learning from the query text
            known_count = self._learn_from_text(query + " " + query, query, source_url=query)
            if known_count > 0:
                return f"learned {known_count} things about {query}"
            return f"offline - already knew about {query}"
        except Exception as e:
            # Any other error — also fall back silently
            if self._trace_enabled:
                print(f"  [bg] Unexpected error in learn_from_web: {type(e).__name__}: {e}")
            self._network_available = False
            return f"offline"

    def _fetch_article_text(self, url: str) -> Optional[str]:
        """Fetch a URL and extract readable article text."""
        # Skip video/social media URLs that don't have extractable article content
        skip_domains = ('youtube.com', 'youtu.be', 'facebook.com', 'twitter.com', 'x.com',
                        'instagram.com', 'tiktok.com', 'reddit.com', 'linkedin.com',
                        'pinterest.com', 'vimeo.com', 'dailymotion.com', 'twitch.tv')
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if any(skip_domain in domain for skip_domain in skip_domains):
                return None
        except Exception:
            pass

        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            resp = urllib.request.urlopen(req, timeout=8)
            html = resp.read().decode('utf-8', errors='replace')

            if HAS_BS4:
                soup = BeautifulSoup(html, 'html.parser')
                # Remove script/style elements
                for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                    tag.decompose()
                # Try to find main content
                main = soup.find('article') or soup.find('main') or soup.find('body')
                if main:
                    text = main.get_text(separator=' ', strip=True)
                else:
                    text = soup.get_text(separator=' ', strip=True)
                return re.sub(r'\s+', ' ', text)[:5000]
            else:
                # Simple regex-based tag stripping
                text = re.sub(r'<[^>]+>', ' ', html)
                text = re.sub(r'\s+', ' ', text).strip()
                return text[:3000]
        except Exception:
            return None

    def _learn_from_text(self, text: str, topic: str, source_url: str = "") -> int:
        """Extract important keywords from text, add them as concepts, form connections.

        Returns number of new concepts added.
        """
        # Tokenize and count word frequencies
        words = re.findall(r"[a-zA-Z']{3,}", text.lower())
        word_counts = {}
        for w in words:
            wc = w.strip("'")
            if wc not in STOP_WORDS and len(wc) >= 3:
                word_counts[wc] = word_counts.get(wc, 0) + 1

        if not word_counts:
            return 0

        # Sort by frequency, take top 30
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        important_words = [w for w, c in sorted_words[:30]]

        # Always add the topic word
        topic_lower = topic.lower().strip()
        topic_words = [tw for tw in re.findall(r"[a-zA-Z']{3,}", topic_lower) if tw not in STOP_WORDS]
        important_words = topic_words + [w for w in important_words if w not in topic_words]

        # Check which words are new vs known (lock graph reads)
        with self._graph_lock:
            existing_labels = set()
            for nid, node in self.graph.nodes.items():
                if node.label:
                    existing_labels.add(node.label.lower())

            new_count = 0
            existing_labels_before = existing_labels.copy()
            label_to_id = {}
            for nid, node in self.graph.nodes.items():
                if node.label:
                    label_to_id[node.label] = nid

        # Phase 18c: Track source of this learning - use URL when available
        if source_url:
            source_id = hashlib.md5(source_url.encode()).hexdigest()[:16]
        else:
            source_id = hashlib.md5((source_url or topic).encode()).hexdigest()[:16]

        # Add new concepts (locked to prevent graph mutation during save iteration)
        for word in important_words:
            if word in existing_labels:
                # Concept already exists - reinforce by tracking additional source
                self._concept_sources.setdefault(word, set()).add(source_id)
                # Boost edge confidence if concept confirmed by multiple sources
                if len(self._concept_sources[word]) >= 2:
                    nids = self._concept_keywords.get(word, [])
                    for nid in nids:
                        for tid, edge in self.graph.get_outgoing(nid):
                            edge.confidence = min(0.8, edge.confidence + 0.05)
                continue
            # GloVe vector if available, else hash-based random
            vec = self._glove_vector(word)
            if vec is None:
                h = hash(word) % 50000
                vr = np.random.RandomState(h + 100)
                vec = vr.randn(self.dim).astype(np.float32) * 0.1
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
            with self._graph_lock:
                node = self.graph.add_node(vector=vec, label=word)
            label_to_id[word] = node.id
            self._concept_keywords[word] = self._concept_keywords.get(word, []) + [node.id]
            self._concept_labels.add(word.lower())
            existing_labels.add(word)
            new_count += 1
            # Phase 18c: Track source for new concepts
            self._concept_sources[word] = {source_id}
            # New concepts tracked by source (confidence set naturally by prediction error)

        # Form connections between co-occurring important words
        # Words that appear together in the article get edges
        article_words_lower = [w for w in re.findall(r"[a-zA-Z']{3,}", text.lower())
                              if w not in STOP_WORDS and len(w.strip("'")) >= 3]
        # Get unique important words that appear in the text
        present_important = list(set(w for w in article_words_lower if w in important_words))

        with self._graph_lock:
            for i in range(len(present_important)):
                for j in range(i + 1, len(present_important)):
                    w1, w2 = present_important[i], present_important[j]
                    nid1 = label_to_id.get(w1)
                    nid2 = label_to_id.get(w2)
                    if nid1 is not None and nid2 is not None:
                        # Don't duplicate existing edges, just boost weight
                        existing = self.graph.get_edge(nid1, nid2)
                        if existing is None:
                            weight = max(0.3, min(0.7, 0.2 + word_counts.get(w1, 1) * word_counts.get(w2, 1) * 0.001))
                            self.graph.add_edge(nid1, nid2, weight=weight, relation_type="semantic")
                        else:
                            existing.weight = min(0.9, existing.weight + 0.05)

        # Connect new words to existing related concepts via vector similarity
        # Only do this for the topic word and first 2 important words (others
        # rely on co-occurrence edges, which are more topically coherent)
        with self._graph_lock:
            for word in important_words[:10]:
                nid = label_to_id.get(word)
                if nid is None:
                    continue
                node = self.graph.get_node(nid)
                if node is None:
                    continue
                edge_count = 0
                # FAISS/vectorized: matrix multiply instead of per-node loop
                if self.graph._vectors_dirty or self.graph._vector_matrix_normed is None:
                    self.graph._rebuild_vector_matrix()
                if self.graph._vector_matrix_normed is not None and len(self.graph._node_id_order) > 0:
                    vec_norm = node.vector / (np.linalg.norm(node.vector) + 1e-15)
                    all_sims = self.graph._vector_matrix_normed @ vec_norm.astype(np.float32)
                    for idx in np.where(all_sims > 0.3)[0]:
                        existing_nid = self.graph._node_id_order[idx]
                        if existing_nid == nid or existing_nid not in self.graph.nodes:
                            continue
                        existing_node = self.graph.get_node(existing_nid)
                        if existing_node is None or existing_node.vector is None:
                            continue
                        if existing_node.label and existing_node.label in important_words:
                            continue
                        sim = float(all_sims[idx])
                        if self.graph.get_edge(nid, existing_nid) is None:
                            weight = max(0.25, min(0.5, sim * 0.5))
                            inf_type, _ = self._infer_relation_type(word, existing_node.label, "semantic")
                            self.graph.add_edge(nid, existing_nid, weight=weight,
                                                relation_type=inf_type)
                            edge_count += 1
                # Guarantee at least one connection for the topic word
                if edge_count == 0 and word == important_words[0]:
                    best_sim = 0.0
                    best_nid = None
                    best_label = None
                    for existing_nid, existing_node in list(self.graph.nodes.items()):
                        if existing_nid == nid or existing_node.label in important_words:
                            continue
                        if existing_node.vector is not None and node.vector is not None:
                            sim = float(np.dot(node.vector, existing_node.vector))
                            if sim > best_sim:
                                best_sim = sim
                                best_nid = existing_nid
                                best_label = existing_node.label
                    if best_nid is not None and best_label is not None:
                        weight = max(0.25, min(0.4, best_sim * 0.5))
                        inf_type, _ = self._infer_relation_type(word, best_label, "semantic")
                        self.graph.add_edge(nid, best_nid, weight=weight, relation_type=inf_type)

        # Train neural decoder on article text (unsupervised next-word prediction)
        if self.neural_decoder is not None and self._decoder_vocab_built:
            # Build graph walk conditioning from the article's important words
            article_words = re.findall(r"[a-zA-Z']{3,}", text.lower())
            cond_embs = self._build_conditioning_for_text(topic, article_words)

            # Expand decoder vocab with newly added concepts (lock graph read)
            with self._graph_lock:
                new_labels = [n.label for nid, n in self.graph.nodes.items()
                              if n.label and n.label.lower() not in self._decoder_word_to_idx
                              and n.label.lower() not in ('<pad>', '<unk>', '<bos>', '<eos>')]
            if new_labels:
                self._expand_decoder_vocab(new_labels)

            err, trained = self.neural_decoder.train_on_text(
                text, self._decoder_word_to_embed, self._decoder_word_to_idx,
                conditioning_embs=cond_embs)
            if trained > 0:
                self._decoder_web_training_count += trained
            # Multiple passes per article to strengthen Hebbian traces
            # More passes for web articles (10 instead of 4)
            if trained > 0:
                for _ in range(10):
                    err2, trained2 = self.neural_decoder.train_on_text(
                        text, self._decoder_word_to_embed, self._decoder_word_to_idx,
                        conditioning_embs=cond_embs)
                    self._decoder_web_training_count += trained2
                self.neural_decoder.sleep_cycle()

                # Train CerebellarNgram on article text (climbing-fibre error modulated)
                if hasattr(self, 'cerebellar_ngram'):
                    # Note: learn_from_text method doesn't exist, skipping
                    pass

        # Train neural decoder on real article text
        if self.neural_decoder is not None and self._decoder_vocab_built and len(text) > 20:
            new_labels = [w for w in important_words if w not in existing_labels_before]
            if new_labels:
                self._expand_decoder_vocab(new_labels)
            err, n_trained = self.neural_decoder.train_on_text(
                text, self._decoder_word_to_embed, self._decoder_word_to_idx
            )
            self._decoder_web_training_count += n_trained
            self._decoder_training_count += n_trained
        return new_count
    def _learn_from_snippets(self, query: str, snippets: List[str]) -> str:
        """Learn from search snippet text when article fetch fails."""
        combined = f"{query} " + " ".join(snippets[:3])
        count = self._learn_from_text(combined, query, source_url=query)
        if count > 0:
            return f"learned {count} new things about {query} from search snippets"
        return f"read about {query} but already knew those words"

    # ─── Core Response Pipeline ───

    def process_turn(self, user_input: str) -> str:
        """Process input and generate a response, auto-learning when needed."""
        self.turn_count += 1
        self._learned_this_turn = False
        self._cascade_for_quality = False

        # Deferred decoder training on first turn (fast startup)
        if getattr(self, '_needs_seed_training', False):
            self._needs_seed_training = False
            try:
                # Full training: 20 passes on the corpus (real English syntax)
                corpus_path = os.path.join(_proj_root, "data", "corpora", "teen_seeds.txt")
                if os.path.exists(corpus_path):
                    corpus_text = open(corpus_path, "r", encoding="utf-8").read()
                    total_err = 0.0
                    passes = 0
                    for _ in range(20):  # 20 passes for Hebbian consolidation
                        err, n = self.neural_decoder.train_on_text(
                            corpus_text,
                            self._decoder_word_to_embed, self._decoder_word_to_idx,
                            min_sentence_len=3, max_sentences=200
                        )
                        total_err += err
                        passes += n
                    self.neural_decoder.sleep_cycle()
                    self._decoder_seed_training_count = passes
                    self._decoder_training_count += passes
                    if self._trace_enabled:
                        print(f"  [init] Seed corpus training: {passes} sentences ({total_err/passes:.3f} avg err)")
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

        # Phase 7: Store activated IDs for strategy framework
        self._last_activated_ids = list(activated)

        # Step 1b.75: Phase 3.3 + 9c — Detect recall triggers with hippocampal reactivation
        recall_topic = self._detect_recall_trigger(user_input)
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
        # Run grounding again to get confidence for auto-web-learning
        _grounded_subj, _gconf, _gmethod = self._ground_query(user_input)
        self._last_grounding_conf = _gconf
        self._last_grounding_method = _gmethod
        # Auto-trigger web learning for low-confidence multi-word queries
        if _gconf < 0.5 and _gmethod == "all_unknown" and _grounded_subj and self.baby_mode:
            if _grounded_subj not in self._pending_learning_queue:
                self._pending_learning_queue.append(_grounded_subj)
        relation = "is"

        # Step 2b: Primary IDs — only these concepts spread activation
        # (other input-matched concepts provide context but don't propagate)
        subject_ids = set()
        sl = subject.lower()
        if sl in self._concept_keywords:
            subject_ids.update(self._concept_keywords[sl])

        # Step 3: Spread activation through graph
        associations = self._spread_and_collect(activated, primary_ids=subject_ids)

        # Step 4: Collect unknown words for deferred web learning
        # Phase 1.4: No hard lifetime cap — per-session rate limit (max 1 search per 3 turns)
        input_words = [w.strip(".,!?") for w in user_input.lower().split()
                      if len(w.strip(".,!?")) >= 3]
        known_words = sum(1 for w in input_words if w in self._concept_keywords)
        unknown_meaningful = [w for w in input_words
                              if w not in self._concept_keywords and w not in STOP_WORDS]
        # Phase 1.3: Collect unknown words into queue instead of searching synchronously
        if unknown_meaningful and self.baby_mode:
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

        # Step 6: Dual-process route
        confidence = self.identity.state.strength * 0.5 + 0.2
        route = self.dual_process.decide_route(
            confidence=confidence,
            novelty=0.1 if associations else 0.6,
            stakes=0.15,
        )

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
            # Run a mini sleep cycle: consolidate knowledge
            metrics = self._sleep_consolidate()
            self._last_sleep_episode = self.turn_count
            self._sleep_pressure = 0.0
            if self._trace_enabled and metrics:
                print(f"  [sleep] Cycle #{self._sleep_metrics['total_sleep_cycles']}: "
                      f"{metrics.get('edges_strengthened', 0)} edges strengthened, "
                      f"{metrics.get('edges_pruned', 0)} edges pruned")

        # Step 7: Past topics
        past = self._recall_past(subject, obj)

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
        )

        response, strategy = self._generate_response(ctx)
        self._last_strategy = strategy

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
            if not any(t.lower() == sl for t in self._topic_list):
                self._topic_list.append(subject)
        # Keep last 50 topics
        if len(self._topic_list) > 50:
            removed = self._topic_list[:-50]
            self._topic_list = self._topic_list[-50:]
            for r in removed:
                self._topic_store.pop(r.lower(), None)

        # Step 11: Store episodic memory — link subject to its associations
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

        return response

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
        """Find matching concepts using keywords and label matching."""
        words = re.findall(r"[a-zA-Z']{1,}", text.lower())
        scores: Dict[int, float] = {}

        for w in words:
            if w in self._concept_keywords:
                for nid in self._concept_keywords[w]:
                    # Boost nouns slightly for better subject extraction
                    node = self.graph.nodes.get(nid)
                    pos_boost = 0.5 if node and node.label and self._concept_pos.get(node.label.lower()) == 'noun' else 0.0
                    scores[nid] = scores.get(nid, 0) + 5.0 + pos_boost

        for nid, node in self.graph.nodes.items():
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
            self.graph.activate(nid, min(1.0, sc * 0.15))
            activated.append(nid)
        return activated

    QUESTION_WORDS = {"what", "why", "how", "when", "where", "who", "which",
                        "does", "do", "is", "are", "can", "will", "would",
                        "could", "should", "did", "have", "has", "had"}

    # Words that signal the user is asking for more on a previous topic
    FOLLOW_UP_WORDS = {"more", "else", "another", "also", "further",
                       "other", "additionally", "favorite"}

    # Phase 3.3: Recall trigger patterns
    RECALL_TRIGGERS = [
        "remember when", "remember we", "earlier you", "you said",
        "you mentioned", "we talked about", "we discussed", "before you",
        "what did you say about", "what did we say about",
        "recall", "previously", "last time",
    ]

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
                        # Generic adjectives & filler that make poor conversation topics
                        "good", "bad", "big", "small", "always", "never", "maybe",
                        "if", "but", "in", "out", "up", "down",
                        "point", "way", "thing", "stuff",
                        "and", "so"}

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
        (r"(?:what|who)'?s?\s+(?:is\s+|are\s+)?(.+)", 1),       # what is X / who are X
        (r"(?:tell|show)\s+me\s+about\s+(.+)", 1),              # tell me about X
        (r"(?:explain|describe)\s+(.+)", 1),                     # explain X / describe X
        (r"(?:what|which)\s+(.+)\s+(?:is|are|mean)", 1),         # what X is / what X means
        (r"(?:do you know|have you heard of)\s+(.+)", 1),        # do you know X
        (r"(?:what\s+happens\s+(?:if|when))\s+(.+)", 1),         # what happens if X / what happens when X
    ]

    def _ground_query(self, text: str) -> Tuple[str, float, str]:
        """Multi-strategy query grounding. Returns (subject, confidence, method).

        Strategies (tried in order):
        a) Exact phrase match — the full phrase after 'what is' matches a concept label
        b) Compositional — split phrase, count known vs unknown words; use best known word
        c) Phrase embedding similarity — mean word vec → nearest concept (cosine > 0.75)
        d) Best single word fallback — last meaningful non-stop word
        """
        # Detect query patterns and extract the semantic payload
        text_lower = text.lower().strip(" ?!.")
        query_phrase = ""
        for pattern, group_idx in self.QUERY_PATTERNS:
            m = re.match(pattern, text_lower)
            if m:
                query_phrase = m.group(group_idx).strip()
                break

        if not query_phrase:
            return ("", 0.0, "no_pattern")

        # Strategy A: Exact multi-word phrase match (domain concepts, seeded multi-word)
        phrase_clean = query_phrase.strip(".,!?")
        if phrase_clean in self._concept_labels:
            return (phrase_clean, 0.95, "exact_label")
        if phrase_clean in self._concept_keywords:
            return (phrase_clean, 0.90, "exact_keyword")

        # Strategy C (moved before B): Compositional — score words by known/unknown ratio
        # This is more reliable for multi-word queries than phrase embedding
        words = [w.strip(".,!?") for w in query_phrase.split()
                 if len(w.strip(".,!?")) > 2
                 and w.strip(".,!?") not in self.QUESTION_WORDS
                 and w.strip(".,!?") not in self.TOPIC_SKIP_WORDS
                 and w.strip(".,!?") not in STOP_WORDS]
        if words:
            known_words = [w for w in words if w in self._concept_labels or w in self._concept_keywords]
            unknown_words = [w for w in words if w not in known_words]
            if known_words:
                ratio = len(known_words) / len(words)
                # Prefer last known word (most specific in English)
                topic = known_words[-1]
                return (topic, min(0.85, 0.5 + ratio * 0.4), f"compositional_{ratio:.2f}")
            # All unknown — will trigger web learning
            if words:
                return (words[-1], 0.2, "all_unknown")

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
                for label in self._concept_labels:
                    if (label.startswith(wl[:3]) and abs(len(label) - len(wl)) <= 2) or                        (len(wl) >= 4 and label.startswith(wl[:4])):
                        close_matches.append(label)
                        break
            if close_matches:
                topic = close_matches[-1]
                return (topic, 0.5, f"close_match_{topic}")

        return ("", 0.0, "no_match")

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
        if activated:
            best_real = None
            best_noun = None
            for nid in activated:
                node = self.graph.get_node(nid)
                if node and node.label:
                    lbl = node.label.lower()
                    if (len(lbl) > 2 and lbl not in self.QUESTION_WORDS
                            and lbl not in self.TOPIC_SKIP_WORDS):
                        pos = self._concept_pos.get(lbl, 'noun')
                        if pos == 'noun' and best_noun is None:
                            best_noun = (node.label, text)
                        if best_real is None:
                            best_real = (node.label, text)
            # Prefer noun if available, else first valid
            if best_noun:
                return best_noun
            if best_real:
                return best_real

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

    def _spread_and_collect(self, seed_ids: List[int],
                             primary_ids: Optional[Set[int]] = None) -> List[Tuple[str, float]]:
        """Propagate activation through graph edges (3 hops).

        Only concepts in primary_ids (or all seed_ids if not specified)
        serve as activation sources. Other seed_ids get context activation
        (0.3) but don't propagate — they only prevent their neighbors from
        being collected as novel associations.
        """
        if not seed_ids:
            return []
        seed_set = set(seed_ids)
        spread_set = primary_ids if primary_ids else seed_set

        # Base activation: context (0.3) for all seeds, full (1.0) for spread sources
        for nid in seed_ids:
            self.graph.activate(nid, 1.0 if nid in spread_set else 0.3)

        for hop in range(3):
            new_acts: Dict[int, float] = {}
            decay = 0.7 ** (hop + 1)
            for nid in list(self.graph._active_nodes):
                node = self.graph.nodes.get(nid)
                if node is None or node.activation <= 0.01:
                    continue
                # Only propagate from primary spread sources
                if nid not in spread_set:
                    continue
                # Follow outgoing edges
                for tid, edge in self.graph.get_outgoing(nid):
                    if tid in seed_set:
                        continue
                    signal = node.activation * edge.weight * edge.confidence * decay
                    if signal > 0.01:
                        new_acts[tid] = new_acts.get(tid, 0.0) + signal
                # Follow incoming edges (semantic network is effectively undirected)
                for src, edge in self.graph.get_incoming(nid):
                    if src in seed_set:
                        continue
                    signal = node.activation * edge.weight * edge.confidence * decay
                    if signal > 0.01:
                        new_acts[src] = new_acts.get(src, 0.0) + signal
            for nid, sig in new_acts.items():
                if nid in self.graph.nodes:
                    self.graph.activate(nid, sig)

        collected = []
        for nid, node in self.graph.nodes.items():
            if nid not in seed_set and node.activation > 0.05:
                collected.append((node.label or f"c{nid}", float(node.activation)))

        collected.sort(key=lambda x: x[1], reverse=True)
        for node in self.graph.nodes.values():
            node.activation = 0.0
        return collected[:12]

    def _find_bridge(self, assoc: List[Tuple[str, float]], subj: str) -> str:
        if not assoc:
            return subj if subj else ""
        best = assoc[0][0]
        if best.lower() == subj.lower() and len(assoc) > 1:
            return assoc[1][0]
        return best

    def _update_emotion(self, text: str):
        """More nuanced emotional processing — teenage range of emotions."""
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
        # Phase 7.5: Curiosity drive — boost arousal for impossible queries
        if getattr(self, '_last_strategy_used', '') in ('G_uncertainty', 'F_web_research'):
            sa += 0.6  # strong curiosity arousal for impossible query
            sv += 0.1  # slight positive valence for curiosity
        self.emotion.update(stimulus_valence=sv, stimulus_arousal=sa,
                           stimulus_dominance=self.identity.state.strength * 0.4 + 0.2,
                           uncertainty=self._free_energy * 0.5, dt=1.0)

    # ── P1 Theory of Mind ──

    def _update_user_model(self, text: str, subject: str,
                           associations: List[Tuple[str, float]]):
        """Deep Theory of Mind update after turn processing (roadmap §7).

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

    def _personalized_greeting(self) -> str:
        """Return a personalized greeting prefix when relationship_depth warrants it.

        Neuroscience basis: repeated social interaction builds rapport, modeled
        as relationship_depth ∈ [0, 1]. Above 0.5, reference the last topic to
        demonstrate memory and continuity (roadmap §9).

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

    def _adapt_verbosity_for_user(self, plan: 'DiscoursePlan', subject: str) -> 'DiscoursePlan':
        """Adaptive language complexity (roadmap §7 deliverable A).

        Modulates the PFC discourse plan based on user familiarity:
        - Low familiarity (< 0.3): keep all 3 intents — user needs full explanation
        - Medium familiarity (0.3-0.7): keep 2-3 intents
        - High familiarity (> 0.7) + LEARNING goal: trim to 2 — user knows the basics

        P2 Emotional Mirroring: user arousal also modulates verbosity — excited users
        get slightly longer, more engaged responses; calm users get more concise ones.

        This respects the PrefrontalWorkspace capacity (7±2 items, Baddeley & Hitch 1974)
        by avoiding unnecessary verbal load for expert users.
        """
        um = self.user_model
        familiarity = um.infer_user_knows(subject)
        goal = um.last_goal

        # P2: User arousal modulates base verbosity (mirroring loop)
        user_arousal = um.emotional_state.get('arousal', 0.3)
        target_intents = 3
        if user_arousal < 0.25:
            target_intents = 2  # calm user → concise
        elif user_arousal > 0.6:
            target_intents = 4  # excited user → more engagement

        if familiarity < 0.3:
            # Novice: full explanation
            if len(plan.intents) > target_intents:
                plan.intents = plan.intents[:target_intents]
            return plan
        elif familiarity > 0.7 and goal == "LEARNING":
            # Expert learner: trim — skip the generic ELABORATE
            if len(plan.intents) > 2:
                plan.intents = [plan.intents[0], plan.intents[1]]
            return plan
        # Medium: keep original plan, but cap at target
        if len(plan.intents) > target_intents:
            plan.intents = plan.intents[:target_intents]
        return plan


    def _detect_recall_trigger(self, text: str) -> Optional[str]:
        """Phase 3.3: Detect if user is recalling a past topic."""
        text_lower = text.lower()
        for trigger in self.RECALL_TRIGGERS:
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

    def _recall_past(self, subj: str, obj: str) -> List[str]:
        related = []
        for t in self._topic_list:
            pl = t.lower()
            sl = subj.lower()
            if pl != sl and (pl in sl or sl in pl or len(set(pl.split()) & set(sl.split())) > 0):
                related.append(t)
        return related[:3]

    # ─── Phase 9c: Hippocampal Indexing ───
    # Based on: Teyler & Rudy (2007) hippocampal indexing theory.
    # The hippocampus stores an INDEX to distributed neocortical patterns,
    # not the memory content itself. Reactivation of the index → reactivation
    # of the distributed pattern → memory experience.
    #
    # Instead of storing full topic content, we store a sparse index:
    # which concept IDs were activated, which edges were traversed.
    # During recall, the index reactivates the distributed graph pattern.

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

    # ─── Graph-Driven Response Generation ───
    # NO hardcoded strings. ALL content words are concept labels from the graph.
    # Edge relation types (stored in graph edge data) are mapped to graph concept
    # labels that already exist in the seeded vocabulary — no external text.

    # Tiered connectors: (min_weight, [word, ...])
    # Auto-wired edges: min=0.30, ~0.30-0.40 common, max=0.50
    # Weight ≥ 0.33 (~top 25%): stronger connector (e.g. "link", "and")
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
        "analogical": [
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
        "causal": "cause",
        "analogical": "like",
        "semantic": "connect",
        "contrastive": "but",
        "temporal": "change",
        "episodic": "connect",
    }

    # Edge type → discourse starter (all labels exist in seeded TEEN_CONCEPTS)
    _EDGE_TO_STARTER = {
        "causal": "because",
        "contrastive": "but",
        "semantic": "and",
        "analogical": "like",
        "temporal": "then",
    }

    # ── Solution #5: Relation Type Inference ──
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

    def _infer_relation_type(self, src_label: str, tgt_label: str,
                              current_type: str = "semantic") -> Tuple[str, float]:
        """Infer the correct relation type for a concept pair.
        Returns (inferred_type, confidence).
        Uses label-pair heuristics first, then vector-based ranking."""
        sl = src_label.lower()
        tl = tgt_label.lower()
        pair = tuple(sorted([sl, tl]))
        # Heuristic: known contrastive pairs
        if pair in self.CONTRASTIVE_PAIRS:
            return ("contrastive", 0.85)
        if pair in self.CAUSAL_PAIRS:
            return ("causal", 0.80)
        if pair in self.IS_A_PAIRS:
            return ("semantic", 0.80)
        # Vector-based ranking
        ranked = self._rank_relations(src_label, tgt_label, current_type)
        if ranked:
            best_type, best_score = ranked[0]
            # Only reclassify if clearly a different type
            if len(ranked) > 1:
                second_score = ranked[1][1]
                margin = best_score - second_score
                if margin > 0.10 and best_type != current_type:
                    return (best_type, min(0.75, best_score))
            return (current_type, max(0.3, best_score))
        return (current_type, 0.3)

    def _correct_relation_types(self) -> int:
        """Iterate edges and reclassify mis-typed relations.
        Returns number of edges migrated."""
        migrated = 0
        for (sid, tid), edge in list(self.graph.edges.items()):
            src_node = self.graph.get_node(sid)
            tgt_node = self.graph.get_node(tid)
            if src_node is None or tgt_node is None or not src_node.label or not tgt_node.label:
                continue
            # Skip high-confidence edges (manually seeded or user-confirmed)
            if edge.confidence >= 0.8:
                continue
            inferred_type, inferred_conf = self._infer_relation_type(
                src_node.label, tgt_node.label, edge.relation_type)
            if inferred_type != edge.relation_type:
                old_type = edge.relation_type
                edge.relation_type = inferred_type
                edge.confidence = max(edge.confidence, inferred_conf * 0.8)
                migrated += 1
        return migrated

    def _get_edge_info(self, src_label: str, tgt_label: str) -> Optional[Dict]:
        """Get edge information between two concept labels from the graph."""
        src_ids = self._concept_keywords.get(src_label.lower(), [])
        tgt_ids = self._concept_keywords.get(tgt_label.lower(), [])
        for sid in src_ids:
            for tid in tgt_ids:
                edge = self.graph.get_edge(sid, tid)
                if edge:
                    return {'weight': edge.weight, 'confidence': edge.confidence,
                            'relation_type': edge.relation_type, 'reversed': False}
                edge = self.graph.get_edge(tid, sid)
                if edge:
                    return {'weight': edge.weight, 'confidence': edge.confidence,
                            'relation_type': edge.relation_type, 'reversed': True}
        return None

    def _group_associations(self, subject: str, assocs: List,
                             seen_labels: Set[str],
                             max_total: int = 5) -> Dict[str, List[str]]:
        """Group associations by their relation type to subject."""
        groups: Dict[str, List[str]] = {}
        for concept, _ in assocs:
            if len(seen_labels) > max_total:
                break
            key = concept.lower()
            if key in seen_labels:
                continue
            seen_labels.add(key)
            edge_info = self._get_edge_info(subject, concept)
            rel = edge_info['relation_type'] if edge_info else "semantic"
            groups.setdefault(rel, []).append(concept)
        return groups

    def _rel_clause(self, subject: str, rel: str, concepts: List[str]) -> str:
        """Build a clause like 'subject cause concept1 concept2'."""
        connector = self._pick_connector(rel, 0.5, 1.0, 0.25)
        concepts = [c for c in concepts if c.lower() != connector]
        if not concepts:
            return ""
        if connector:
            return f"{subject} {connector} {' '.join(concepts)}"
        return f"{subject} {' '.join(concepts)}"

    def _build_phrase(self, subject: str, groups: Dict[str, List[str]]) -> str:
        """Build a phrase from grouped associations: subject connector group1 group2..."""
        parts = [subject]
        for rel, concepts in groups.items():
            connector = self._EDGE_TO_GRAPH_LABEL.get(rel, "")
            if connector:
                if parts[-1] != connector:
                    parts.append(connector)
                concepts = [c for c in concepts if c.lower() != connector]
            parts.extend(concepts)
        return " ".join(parts)

    # ─── VAD Modulation ───

    def _get_temperature(self, sentence_idx: int,
                         base_temps: Optional[List[float]] = None) -> float:
        """Return temperature for sentence, scaled by current arousal.

        High arousal → more exploration (higher temp).
        Low arousal → more focused (lower temp).

        P2 Emotional Mirroring: also mirrors user's arousal — when the user
        is excited, RAVANA's language becomes more varied too.
        """
        if not getattr(self, 'use_vad', True):
            base = (base_temps or [0.08, 0.15, 0.25])[min(sentence_idx, 2)]
            return base
        base_temps = base_temps or [0.08, 0.15, 0.25]
        base = base_temps[min(sentence_idx, 2)]
        arousal = self.emotion.state.arousal  # 0..1
        arousal_factor = 0.7 + (arousal * 0.6)  # [0.7, 1.3]
        temp = base * arousal_factor
        # Phase 12.2: Brain state modulation of temperature
        state = getattr(self, '_cognitive_state', 'default')
        if state == "heteromodal":
            temp *= 1.3  # more exploration
        elif state == "unimodal":
            temp *= 0.7  # more precision
        # Phase 14.3: Dopamine tone modulation of temperature
        dt = getattr(self, '_dopamine_tone', 0.5)
        dt_factor = 0.7 + (dt - 0.5) * 1.0  # [0.7, 1.1] at [0.1, 0.9]
        temp *= dt_factor
        # P2 Emotional Mirroring: mirror user's arousal into temperature
        if hasattr(self, 'user_model') and self.user_model is not None:
            user_arousal = self.user_model.emotional_state.get('arousal', 0.3)
            # User arousal modulates temp by ±20%: calm user → tighter, excited user → looser
            user_arousal_factor = 0.8 + (user_arousal * 0.4)  # [0.8, 1.2]
            temp *= user_arousal_factor
        return temp

    def _vad_decay(self, rate: float = 0.95):
        """Return VAD toward neutral each turn."""
        if not getattr(self, 'use_vad', True):
            return
        s = self.emotion.state
        s.valence *= rate
        s.arousal = 0.3 + (s.arousal - 0.3) * rate
        s.dominance = 0.5 + (s.dominance - 0.5) * rate

    # ─── Relation Verifier (lightweight RLMv2 wrapper) ───

    def _classify_triple(self, source_label: str, target_label: str,
                         relation_type: str) -> Dict[str, Any]:
        """Verify a (source, relation, target) triple using graph + vector data.

        Returns:
            {'confidence': float (0..1),
             'top_relations': List[(rel_type, score)],
             'is_analogical': bool,
             'suggested_type': str}
        """
        src_nids = self._concept_keywords.get(source_label.lower(), [])
        tgt_nids = self._concept_keywords.get(target_label.lower(), [])
        if not src_nids or not tgt_nids:
            return {'confidence': 0.0, 'top_relations': [],
                    'is_analogical': False, 'suggested_type': relation_type}

        # Edge-based confidence
        edge_info = self._get_edge_info(source_label, target_label)
        if edge_info:
            edge_conf = edge_info['weight'] * edge_info['confidence']
            matches = edge_info['relation_type'] == relation_type
        else:
            edge_conf = 0.0
            matches = False

        # Vector similarity confidence (cosine)
        src_node = self.graph.get_node(src_nids[0])
        tgt_node = self.graph.get_node(tgt_nids[0])
        vec_conf = 0.0
        if src_node and tgt_node and src_node.vector is not None and tgt_node.vector is not None:
            vec_conf = float(np.dot(src_node.vector, tgt_node.vector))

        # Blend: edge (0.6) + vector (0.4), with boost if edge matches
        if matches:
            confidence = min(1.0, edge_conf * 0.7 + vec_conf * 0.3)
        elif edge_conf > 0:
            confidence = 0.3 + edge_conf * 0.3
        else:
            confidence = max(0.0, vec_conf * 0.4 - 0.1)

        # Rank all relation types by vector-based plausibility
        top_relations = self._rank_relations(source_label, target_label, relation_type)

        # Cross-domain detection: different concept categories
        is_analogical = self._is_cross_domain(source_label, target_label)

        return {
            'confidence': confidence,
            'top_relations': top_relations,
            'is_analogical': is_analogical,
            'suggested_type': top_relations[0][0] if top_relations else relation_type,
        }

    DOMAIN_MAP = {
        "emotion": {"love", "hate", "fear", "joy", "sad", "angry", "happy", "scared",
                     "excited", "curious", "confused", "bored", "proud", "lonely",
                     "grateful", "anxiety", "excitement", "frustration", "hope",
                     "grief", "surprise", "guilt", "disappointment",
                     "empathy", "lonely", "heart", "feel"},
        "abstract": {"truth", "justice", "freedom", "power", "knowledge", "wisdom",
                      "meaning", "pattern", "system", "perspective", "context",
                      "paradox", "principle", "theory", "evidence", "analysis",
                      "conclusion", "logic", "intuition", "identity", "culture",
                      "responsibility", "trust", "hypocrisy", "respect",
                      "belief", "significant", "fundamental", "inevitable",
                      "possible", "obvious", "subtle", "profound"},
        "social": {"friend", "people", "person", "trust", "respect",
                    "culture", "power", "freedom", "responsibility",
                    "we", "they", "you", "i", "friend"},
        "concrete": {"water", "food", "home", "sun", "moon", "tree", "bird",
                      "dog", "cat", "book", "song", "world", "nature",
                      "machine", "invention", "water", "food", "sun"},
        "action": {"make", "create", "destroy", "protect", "accept", "reject",
                    "analyze", "conclude", "reflect", "question", "explore",
                    "understand", "compare", "criticize", "assume", "imagine",
                    "cause", "change", "grow", "learn", "teach",
                    "go", "come", "see", "hear", "eat", "drink", "sleep",
                    "play", "help", "get", "know", "think", "say", "feel",
                    "give", "take"},
    }

    def _is_cross_domain(self, label_a: str, label_b: str) -> bool:
        """Check if two concepts belong to different domains."""
        la = label_a.lower()
        lb = label_b.lower()
        domain_a = None
        domain_b = None
        for domain, words in self.DOMAIN_MAP.items():
            if la in words:
                domain_a = domain
            if lb in words:
                domain_b = domain
        return domain_a is not None and domain_b is not None and domain_a != domain_b

    def _rank_relations(self, source_label: str, target_label: str,
                        default_type: str) -> List[Tuple[str, float]]:
        """Rank relation types by vector-based plausibility for a concept pair."""
        src_nids = self._concept_keywords.get(source_label.lower(), [])
        tgt_nids = self._concept_keywords.get(target_label.lower(), [])
        if not src_nids or not tgt_nids:
            return [(default_type, 0.5)]

        src_node = self.graph.get_node(src_nids[0])
        tgt_node = self.graph.get_node(tgt_nids[0])
        if src_node is None or tgt_node is None:
            return [(default_type, 0.5)]

        src_vec = src_node.vector
        tgt_vec = tgt_node.vector
        if src_vec is None or tgt_vec is None:
            return [(default_type, 0.5)]

        # Causal scoring: positive + strong magnitude = likely causal
        cosine = float(np.dot(src_vec, tgt_vec))
        mag_ratio = float(np.linalg.norm(tgt_vec) / max(np.linalg.norm(src_vec), 0.001))

        scores = {
            "semantic": max(0.0, cosine),
            "causal": max(0.0, 0.3 + 0.3 * mag_ratio - 0.2 * (1.0 - abs(cosine))),
            "contrastive": max(0.0, 0.5 - 0.5 * cosine),
            "analogical": max(0.0, 0.2 + 0.3 * cosine - 0.3 * abs(mag_ratio - 1.0)),
            "temporal": 0.1,
            "emotional": 0.1,
        }

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked

    def _find_vector_neighbor(self, label: str) -> Optional[str]:
        """Find the most semantically similar concept (GloVe cosine similarity)."""
        nids = self._concept_keywords.get(label.lower(), [])
        if not nids:
            return None
        nid = nids[0]
        node = self.graph.get_node(nid)
        if node is None or node.vector is None:
            return None
        best_sim = 0.45
        best_label = None
        for other_nid, other_node in self.graph.nodes.items():
            if other_nid == nid or other_node.label is None:
                continue
            if other_node.vector is None:
                continue
            sim = float(np.dot(node.vector, other_node.vector))
            if sim > best_sim:
                best_sim = sim
                best_label = other_node.label
        return best_label

    def _phrase_from_label(self, label: str, seen_labels: Set[str],
                           max_concepts: int = 4) -> Optional[str]:
        """Walk the graph from a single label and return a phrase."""
        nids = self._concept_keywords.get(label.lower(), [])
        if not nids:
            return None
        assocs = self._spread_and_collect(nids, primary_ids=set(nids))
        if not assocs:
            return None
        local_seen = set(seen_labels)
        local_seen.discard(label.lower())
        groups = self._group_associations(label, assocs, local_seen, max_total=max_concepts)
        if not groups:
            return None
        phrase = self._build_phrase(label, groups)
        if phrase == label:
            return None
        return phrase

    def _pick_connector(self, relation_type: str, weight: float,
                        confidence: float, temperature: float) -> str:
        """Pick a connector word based on edge weight, temp, and VAD state.

        Higher weight → stronger connector (e.g. "link" over "connect").
        Higher temperature → more random selection among options.
        High arousal → prefer stronger/assertive connectors.
        Low arousal → prefer weaker/softer connectors.
        All returned words are graph concept labels."""
        tiers = self._EDGE_CONNECTORS.get(relation_type, [])
        if not tiers:
            return ""
        for min_weight, options in tiers:
            if weight >= min_weight:
                # VAD modulation: arousal shifts preference within the tier
                if getattr(self, 'use_vad', True):
                    arousal = self.emotion.state.arousal
                    valence = self.emotion.state.valence
                    dominance = self.emotion.state.dominance
                    # High arousal + positive valence → prefer assertive (first) connector
                    if arousal > 0.50 and valence > 0.10 and len(options) > 1:
                        return options[0]
                    # Low arousal + negative valence → prefer weakest (last) connector
                    if arousal < 0.25 and valence < -0.10 and len(options) > 1:
                        return options[-1]
                # Standard temperature-driven choice
                if temperature < 0.2 or getattr(self, 'reasoning_mode', 'stochastic') == 'deterministic':
                    return options[0]
                return self.rng.choice(options)
        return tiers[-1][1][0]

    
    # ── Phase 10: Predictive Coding in Chain Walking ──
    # Neuroscience: N400 = lexico-semantic prediction error (Wang, Noureddine & Kuperberg 2025)
    # Each hop predicts the next concept; mismatch = prediction error

    def _compute_hop_prediction_error(self, cur_vec: np.ndarray,
                                       subj_vec: np.ndarray,
                                       pfc_centroid: np.ndarray,
                                       chosen_vec: np.ndarray) -> float:
        """Compute prediction error for a chain walk hop.
        
        Predicts next concept as weighted blend of current, subject, and PFC centroid.
        Error = 1.0 - cosine(predicted, actual)
        Returns 0.0 if vectors are invalid.
        """
        if cur_vec is None or chosen_vec is None:
            return 0.0
        if subj_vec is None:
            subj_vec = np.zeros_like(cur_vec)
        if pfc_centroid is None:
            pfc_centroid = np.zeros_like(cur_vec)
        
        predicted = cur_vec * 0.6 + subj_vec * 0.3 + pfc_centroid * 0.1
        norm = np.linalg.norm(predicted)
        if norm > 0:
            predicted /= norm
        actual_cos = float(np.dot(predicted, chosen_vec))
        error = 1.0 - actual_cos  # 0 = perfect, 1 = completely surprising
        return error

    def _update_sentence_schema(self, subject: str, new_concepts: List[str]):
        """Phase 10.3: Sparse-updating sentence schema.
        
        Schema only changes between sentences, not during chain walking.
        New concepts at weight 0.5, old concepts decay by 0.8.
        """
        # Decay existing
        for k in list(self._sentence_schema.keys()):
            self._sentence_schema[k] *= 0.8
            if self._sentence_schema[k] < 0.1:
                del self._sentence_schema[k]
        # Add subject always at high weight
        self._sentence_schema[subject.lower()] = 1.0
        # Add new concepts at moderate weight
        for c in new_concepts[:3]:
            cl = c.lower()
            if cl not in self._sentence_schema:
                self._sentence_schema[cl] = 0.5


    # ── Phase 11: Context-Dependent Vector Modulation ──
    # Neuroscience: PFC neurons represent SAME word differently per context (Nature 2024)
    # HPC broadcasts context state to OFC via theta synchronization (Nature Comms 2025)

    def _build_context_vector(self, subject: str) -> np.ndarray:
        """Build a context vector from topic, recent history, PFC, and emotion.
        
        Components:
        - Subject topic vector × 0.4
        - Mean of last 5 response concept vectors × 0.3
        - Current PFC buffer centroid × 0.2
        - Current VAD emotional vector × 0.1
        """
        components = []
        weights = []
        
        # Subject vector
        subj_nids = self._concept_keywords.get(subject.lower(), [])
        if subj_nids:
            subj_node = self.graph.get_node(subj_nids[0])
            if subj_node and subj_node.vector is not None:
                components.append(subj_node.vector)
                weights.append(0.4)
        
        # Recent response centroid
        recent_vecs = []
        for resp in self._last_responses[-3:]:
            if resp is None:
                continue
            for w in resp.split():
                wn = self._concept_keywords.get(w.lower(), [])
                if wn:
                    wn_node = self.graph.get_node(wn[0])
                    if wn_node and wn_node.vector is not None:
                        recent_vecs.append(wn_node.vector)
        if recent_vecs:
            components.append(np.mean(recent_vecs, axis=0))
            weights.append(0.3)
        
        # PFC buffer centroid
        pfc_vecs = []
        for bl in self._prefrontal_buffer:
            bn = self._concept_keywords.get(bl, [])
            if bn:
                bn_node = self.graph.get_node(bn[0])
                if bn_node and bn_node.vector is not None:
                    pfc_vecs.append(bn_node.vector)
        if pfc_vecs:
            components.append(np.mean(pfc_vecs, axis=0))
            weights.append(0.2)
        
        # Emotional vector
        e_vec = np.array([self.emotion.state.valence,
                          self.emotion.state.arousal,
                          self.emotion.state.dominance], dtype=np.float32)
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

    def _build_sentence_vector(self, text: str) -> np.ndarray:
        """Compose a sentence-level semantic vector from all input words.

        Mimics the N400/P600 retrieval-integration cycle:
        - Each word triggers retrieval of its concept vector (N400)
        - Each word's meaning is integrated into the evolving representation (P600)
        - Integration uses a simple gated composition: new = gate * word + (1-gate) * acc
        where gate is inversely proportional to how similar the word is to the current
        accumulated meaning (novel words contribute more).

        Neuroscience basis:
        - Brouwer et al. 2012/2017 Retrieval-Integration theory
        - Nature 2024: single-neuron semantic composition across sentences
        - Dimensionality ramping (PMC 2023): meaningful sentences increase representational dimensionality
        """
        words = [w.strip(".,!?").lower() for w in text.split()
                 if len(w.strip(".,!?")) >= 2]
        if not words:
            return np.zeros(self.dim, dtype=np.float32)

        acc = np.zeros(self.dim, dtype=np.float32)
        count = 0

        for w in words:
            # N400 phase: retrieve word vector from graph/decoder
            vec = None
            nids = self._concept_keywords.get(w, [])
            if nids:
                node = self.graph.get_node(nids[0])
                if node and node.vector is not None:
                    vec = node.vector.copy()
            if vec is None and hasattr(self, '_decoder_word_to_embed'):
                vec = self._decoder_word_to_embed.get(w, None)
            if vec is None:
                continue

            # P600 phase: integrate into evolving sentence representation
            # Novel words contribute more (gate is high for dissimilar words)
            if count == 0:
                acc = vec.copy()
            else:
                norm_acc = np.linalg.norm(acc)
                norm_vec = np.linalg.norm(vec)
                if norm_acc > 0 and norm_vec > 0:
                    cos_sim = float(np.dot(acc, vec)) / (norm_acc * norm_vec)
                else:
                    cos_sim = 0.0
                # Gate: 0.7 for novel (low sim) to 0.3 for very similar
                gate = 0.7 - 0.4 * max(0.0, min(1.0, cos_sim))
                acc = gate * vec + (1.0 - gate) * acc
            count += 1

            # Renormalize after integration
            n = np.linalg.norm(acc)
            if n > 0:
                acc /= n

        if count == 0:
            return np.zeros(self.dim, dtype=np.float32)

        # Final normalization
        n = np.linalg.norm(acc)
        if n > 0:
            acc /= n
        return acc.astype(np.float32)

    def _modulate_vector(self, node_id: int) -> Optional[np.ndarray]:
        """Phase 11.2: Return context-modulated vector for a concept node.
        
        Modulated = original_vector + 0.15 * context_vector, then normalized.
        Returns None if node or vector is missing.
        """
        node = self.graph.get_node(node_id)
        if node is None or node.vector is None:
            return None
        ctx = getattr(self, '_current_context_vector', None)
        if ctx is None or np.all(ctx == 0):
            return node.vector
        modulated = node.vector + 0.15 * ctx
        norm = np.linalg.norm(modulated)
        if norm > 0:
            modulated /= norm
        return modulated.astype(np.float32)

    def _starter_from_chain(self, chain: str, subject: str,
                             connector_counts: Optional[Dict[str, int]] = None) -> str:
        """Extract the first edge relation type from a chain string and return
        a discourse starter (graph label). The matched starter is weighted 3x
        higher but other starters still have a chance — prevents monotony when
        most edges are semantic."""
        if not chain:
            return "and"
        matched = "and"
        parts = chain.split()
        if len(parts) >= 3:
            conn = parts[1].lower()
            rel_type = self._CONNECTOR_TO_REL.get(conn)
            if rel_type:
                starter = self._EDGE_TO_STARTER.get(rel_type)
                if starter:
                    matched = starter
        candidates = list(self._EDGE_TO_STARTER.values())
        # Phase 4.4: If "connect" has been used >3 times, force a different starter
        if connector_counts and connector_counts.get('connect', 0) >= 3:
            # Force non-semantic starter (but, because, like, then)
            non_semantic = [s for s in candidates if s != 'and']
            if non_semantic:
                candidates = non_semantic
                # Boost the matched starter weight within non-semantic
                weights = np.array([3.0 if s == matched else 1.0 for s in candidates], dtype=np.float64)
                weights /= weights.sum()
                return candidates[self.rng.choice(len(candidates), p=weights)]
        weights = np.array([3.0 if s == matched else 1.0 for s in candidates], dtype=np.float64)
        weights /= weights.sum()
        return candidates[self.rng.choice(len(candidates), p=weights)]

    # ─── Phase 9: Prefrontal Gating (Neuroscience-Inspired Working Memory) ───

    def _prefrontal_gate_candidates(self, candidates: List[Tuple],
                                      subject: str) -> List[Tuple]:
        """Gate candidates through prefrontal working memory buffer.

        Boosts concepts actively held in the prefrontal buffer (working memory).
        Mildly suppresses unrelated concepts to maintain topic coherence.
        Gating strength is modulated by arousal — teens have weaker gating
        under high arousal (limbic > PFC).

        Based on: Ott & Nieder (2019) — Dopamine and Cognitive Control in PFC.
        PFC D1/D2 receptors gate sensory input into working memory.
        """
        if not self._prefrontal_buffer or subject.lower() not in self._prefrontal_buffer:
            return candidates

        # Arousal modulates gating strength (high arousal = leaky gate)
        arousal = self.emotion.state.arousal if getattr(self, 'use_vad', True) else 0.3
        gate_strength = max(0.1, 0.5 - arousal * 0.4)  # 0.5 at calm, 0.26 at peak arousal
        boost_factor = 1.0 + 0.5 * gate_strength       # 1.5 at calm, 1.13 at peak
        penalty_factor = 0.6 + 0.3 * gate_strength     # 0.75 at calm, 0.68 at peak

        gated = []
        for sig, tgt_lbl, edge, d in candidates:
            if tgt_lbl.lower() in self._prefrontal_buffer:
                sig *= boost_factor
            else:
                # Check vector similarity to buffer contents
                tgt_nids = self._concept_keywords.get(tgt_lbl.lower(), [])
                if tgt_nids:
                    tgt_node = self.graph.get_node(tgt_nids[0])
                    if tgt_node and tgt_node.vector is not None:
                        best_cos = 0.0
                        for b_label in self._prefrontal_buffer:
                            b_nids = self._concept_keywords.get(b_label.lower(), [])
                            if b_nids:
                                b_node = self.graph.get_node(b_nids[0])
                                if b_node and b_node.vector is not None:
                                    cos = float(np.dot(tgt_node.vector, b_node.vector))
                                    if cos > best_cos:
                                        best_cos = cos
                        if best_cos > 0.25:
                            sig *= (1.0 + 0.35 * best_cos * gate_strength)
                        else:
                            sig *= penalty_factor
                    else:
                        sig *= penalty_factor
                else:
                    sig *= penalty_factor
            gated.append((sig, tgt_lbl, edge, d))
        return gated

    def _prefrontal_maintain_buffer(self, subject: str, chain_concepts: List[str]):
        """Update prefrontal working memory buffer with recent chain concepts.

        Maintains the subject + most vector-similar recently-used concepts.
        Caps at ~7 items (typical working memory capacity).
        Old buffer entries decay unless they are still semantically relevant.

        Based on: Chatham, Frank & Badre (2014) — Corticostriatal output gating
        during selection from working memory. PFC + basal ganglia control
        what stays in working memory and what gets expressed.
        """
        buffer = {subject.lower()}

        for label in chain_concepts:
            if label.lower() == subject.lower():
                continue
            if len(buffer) >= self._pfc_buffer_capacity:
                break
            subj_nids = self._concept_keywords.get(subject.lower(), [])
            lbl_nids = self._concept_keywords.get(label.lower(), [])
            if subj_nids and lbl_nids:
                subj_node = self.graph.get_node(subj_nids[0])
                lbl_node = self.graph.get_node(lbl_nids[0])
                if (subj_node and subj_node.vector is not None
                        and lbl_node and lbl_node.vector is not None):
                    cos = float(np.dot(lbl_node.vector, subj_node.vector))
                    if cos > 0.25:
                        buffer.add(label.lower())

        # Merge with existing buffer (keep old entries if still relevant)
        for label in self._prefrontal_buffer:
            if len(buffer) >= self._pfc_buffer_capacity:
                break
            if label not in buffer:
                lbl_nids = self._concept_keywords.get(label.lower(), [])
                subj_nids = self._concept_keywords.get(subject.lower(), [])
                if lbl_nids and subj_nids:
                    lbl_node = self.graph.get_node(lbl_nids[0])
                    subj_node = self.graph.get_node(subj_nids[0])
                    if (lbl_node and lbl_node.vector is not None
                            and subj_node and subj_node.vector is not None):
                        cos = float(np.dot(lbl_node.vector, subj_node.vector))
                        if cos > 0.15:
                            buffer.add(label.lower())

        self._prefrontal_buffer = list(buffer)

    
    # ── Phase 12: Schema-Level Activation ──
    # Neuroscience: vmPFC activates whole semantic schemas from minimal input (Entropy 2026)
    # Two brain states: heteromodal (integrative) and unimodal (focused) (Comms Bio 2024)

    def _activate_schema(self, subject: str) -> Set[int]:
        """Phase 12.1: Activate a schema cluster around the subject.
        
        Schema = top-K GloVe neighbors (cosine > 0.5) around the subject.
        All concepts in the schema get activation × 1.3 during spread-and-collect.
        Returns set of concept IDs in the schema.
        """
        subj_nids = self._concept_keywords.get(subject.lower(), [])
        if not subj_nids:
            return set()
        subj_nid = subj_nids[0]
        subj_node = self.graph.get_node(subj_nid)
        if subj_node is None or subj_node.vector is None:
            return set()
        
        # Dynamic threshold based on prediction error
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

    def _detect_brain_state(self) -> str:
        """Phase 12.2: Detect whether RAVANA is in heteromodal (integrative) or unimodal (focused) state.
        
        Heteromodal: confidence < 0.3 OR mean_prediction_error > 0.4 OR novelty > 0.6
        Unimodal: confidence > 0.5 AND mean_prediction_error < 0.2 AND novelty < 0.3
        """
        confidence = self.identity.state.strength * 0.5 + 0.2
        pe = getattr(self, '_mean_prediction_error', 0.3)
        novelty = 0.1 if len(self._last_responses) > 0 else 0.6
        
        if confidence < 0.3 or pe > 0.4 or novelty > 0.6:
            new_state = "heteromodal"
        elif confidence > 0.5 and pe < 0.2 and novelty < 0.3:
            new_state = "unimodal"
        else:
            new_state = "default"
        
        # Damped state transitions: hold for 2 turns before switching
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


    # ── Phase 13: Adaptive Gain Control ──
    # Neuroscience: Semantic satiation = bottom-up gain reduction (Zhang, Comms Bio 2024)
    # Adolescent PFC has high Glu:GABA, more exploration, less stability (PMC 2024)

    def _apply_activation_fatigue(self, candidates: List[Tuple]) -> List[Tuple]:
        """Phase 13.1: Reduce signal for recently overused concepts.
        
        Fatigue accumulates by 0.15 per activation, decays 0.95 per turn.
        Signal *= max(0.3, 1.0 - fatigue). High-fatigue nodes fade after ~5 activations.
        """
        fatigued = []
        for item in candidates:
            sig, tgt_lbl, edge, d = item
            tgt_nids = self._concept_keywords.get(tgt_lbl.lower(), [])
            fatigue = 0.0
            for nid in tgt_nids:
                fatigue = max(fatigue, self._activation_fatigue.get(nid, 0.0))
            adjusted = sig * max(0.3, 1.0 - fatigue)
            fatigued.append((adjusted, tgt_lbl, edge, d))
        return fatigued

    def _apply_edge_repetition_penalty(self, candidates: List[Tuple]) -> List[Tuple]:
        """Phase 13.2: Penalize recently-traversed edges.
        
        Same edge within 1 turn: signal × 0.3
        Same edge within 2 turns: signal × 0.6
        Same edge within 3 turns: signal × 0.8
        """
        penalized = []
        for sig, tgt_lbl, edge, d in candidates:
            penalty = 1.0
            if edge is not None:
                key = (edge.source, edge.target)
                for i, (s, t) in enumerate(reversed(self._recent_traversals)):
                    if (s, t) == key:
                        turns_ago = i + 1
                        if turns_ago <= 1:
                            penalty = 0.3
                        elif turns_ago <= 2:
                            penalty = 0.6
                        elif turns_ago <= 3:
                            penalty = 0.8
                        break
            penalized.append((sig * penalty, tgt_lbl, edge, d))
        return penalized

    def _apply_novelty_bonus(self, candidates: List[Tuple]) -> List[Tuple]:
        """Phase 13.4: Bonus for unvisited concepts (teen exploration).
        
        First 50 turns: ×1.3 for novel concepts, after 50: ×1.05.
        """
        if self.turn_count < 50:
            novelty_mult = 1.3
        else:
            novelty_mult = 1.05
        
        boosted = []
        for sig, tgt_lbl, edge, d in candidates:
            if tgt_lbl.lower() not in self._visited_concepts:
                sig *= novelty_mult
            boosted.append((sig, tgt_lbl, edge, d))
        return boosted


    # ── Phase 14: Online Dopamine-Gated Learning ──
    # Neuroscience: Striatal DA signals prediction errors across ALL domains (Costa, Sci Adv 2025)
    # DA prediction errors drive internal model updates (Gershman, Nat Neurosci 2024)

    def _td_learn(self, cur_label: str, tgt_label: str, edge,
                   hop_prediction_error: float, td_lr: float = 0.1):
        """Phase 14.1: Update edge weight using temporal difference learning.
        
        TD_error = V_actual - V_expected
        V_expected = edge.weight * edge.confidence
        V_actual = 1.0 - hop_prediction_error
        Positive TD: edge better than expected → strengthen.
        Negative TD: edge worse than expected → weaken.
        """
        if edge is None:
            return
        V_expected = edge.weight * edge.confidence
        V_actual = 1.0 - hop_prediction_error
        td_error = V_actual - V_expected
        
        # Modulate learning rate by dopamine tone
        dt = getattr(self, '_dopamine_tone', 0.5)
        effective_lr = td_lr * (0.5 + dt)  # 0.06 at low, 0.14 at high
        
        delta = effective_lr * td_error
        edge.weight = np.clip(edge.weight + delta, 0.01, 0.99)
        
        # Update confidence based on absolute TD error
        abs_error = abs(td_error)
        if abs_error < 0.2:
            edge.confidence = min(1.0, edge.confidence + 0.05)
        elif abs_error > 0.5:
            edge.confidence = max(0.05, edge.confidence - 0.1)
        
        # Track TD error history
        self._td_error_history.append(td_error)
        if len(self._td_error_history) > 20:
            self._td_error_history = self._td_error_history[-20:]

    def _update_dopamine_tone(self):
        """Phase 14.2: Update dopamine tone based on mean TD error, novelty, and repetition.
        
        dopamine_tone = 0.5 + 0.3 * mean_TD + 0.2 * novelty - 0.1 * repetition_penalty
        Clamped to [0.1, 0.9].
        """
        if not self._td_error_history:
            return
        mean_TD = np.mean(np.abs(self._td_error_history[-5:]))
        
        # Novelty: fraction of unseen concepts in last response
        if self._last_responses:
            last_words = set(self._last_responses[-1].lower().split())
            novel = sum(1 for w in last_words if w not in self._visited_concepts)
            novelty = novel / max(1, len(last_words))
        else:
            novelty = 0.5
        
        # Repetition penalty
        if len(self._recent_traversals) > 5:
            unique = len(set(self._recent_traversals[-5:]))
            repetition = 1.0 - (unique / 5.0)
        else:
            repetition = 0.0
        
        self._dopamine_tone = np.clip(
            0.5 + 0.3 * mean_TD + 0.2 * novelty - 0.1 * repetition,
            0.1, 0.9
        )

    def _detect_latent_cause_switch(self) -> bool:
        """Phase 14.5: Detect when prediction error is consistently high across edges.
        
        If mean absolute TD > 0.4 for 3+ hops: signal latent cause switch.
        Returns True if switch detected.
        """
        if len(self._td_error_history) < 3:
            return False
        recent = np.abs(self._td_error_history[-3:])
        if float(np.mean(recent)) > 0.4:
            return True
        return False

    def _walk_chain(self, label: str, seen: Set[str], max_hops: int,
                    temperature: float = 0.25,
                    contradiction_penalty: float = 0.6,
                    activation_boost: Optional[Dict[str, float]] = None,
                    subject_proximity: Optional[str] = None,
                    episodic_first: bool = False) -> Optional[str]:
        import sys
        """Walk a path through the graph from label, temperature-weighted.
        temperature=0 → greedy (always strongest), temperature=1 → near-uniform.
        Each hop adds `connector concept` to the chain. Returns None if no path.

        Applies contradiction_penalty to edges whose target contradicts a
        previously asserted belief about the source concept.
        activation_boost: {target_label: multiplier} from user model preferences.

        When episodic_first=True, only episodic edges are considered for the
        first hop, falling through to all edge types if none exist.

        Records detailed hop info in self._chain_traces when self._trace_enabled."""
        if getattr(self, 'reasoning_mode', 'stochastic') == 'deterministic':
            self._consistency_trace = [label]

        nids = self._concept_keywords.get(label.lower(), [])
        if not nids:
            return None
        chain = [label]
        chain_labels = {label.lower()}  # path-level cycle detection
        cur_nid = nids[0]
        cur_label = label
        hops = 0
        local_hops: List[Tuple[str, str]] = []
        trace = ChainTrace(max_hops=max_hops) if self._trace_enabled else None
        while hops < max_hops:
            candidates = []
            for tid, edge in self.graph.get_outgoing(cur_nid):
                tn = self.graph.nodes.get(tid)
                if tn is None or tn.label is None or tn.label.lower() in seen:
                    continue
                if tn.label.lower() in chain_labels:
                    continue  # cycle detected within this chain
                # Filter: skip weak edges (below auto-wire minimum threshold)
                if edge.weight < 0.35:
                    continue
                # Self-penalty: penalize edges to concepts too semantically different from subject
                score = edge.weight * edge.confidence
                if label and tn.label and label.lower() != tn.label.lower():
                    src_node = self.graph.get_node(nids[0])
                    tgt_node = tn
                    if src_node and src_node.vector is not None and tgt_node and tgt_node.vector is not None:
                        semantic_sim = float(np.dot(src_node.vector, tgt_node.vector))
                        # Penalize edges where target is too semantically distant from subject
                        if semantic_sim < 0.3 and score > 0.1:
                            score *= 0.5
                candidates.append((score, tn.label, edge, "out"))
            for src, edge in self.graph.get_incoming(cur_nid):
                sn = self.graph.nodes.get(src)
                if sn is None or sn.label is None or sn.label.lower() in seen:
                    continue
                if sn.label.lower() in chain_labels:
                    continue  # cycle detected within this chain
                # Filter: skip weak edges
                if edge.weight < 0.35:
                    continue
                candidates.append((edge.weight * edge.confidence, sn.label, edge, "in"))
            # Phase 15.4: Blend candidates from episodic + semantic dual stores
            # Build dedup set from main graph candidates already gathered
            _main_concepts = {c[1].lower() for c in candidates}
            # Semantic edges (stable knowledge) - O(1) src-indexed
            for dtgt, dedge in self._semantic_by_src.get(cur_nid, []):
                dnode = self.graph.nodes.get(dtgt)
                if dnode and dnode.label and dnode.label.lower() not in seen and dnode.label.lower() not in chain_labels and dnode.label.lower() not in _main_concepts:
                    candidates.append((dedge.weight * dedge.confidence, dnode.label, dedge, "out"))
            # Episodic edges (conversation memory) - O(1) src-indexed, decay-modulated
            for dtgt, dedge in self._episodic_by_src.get(cur_nid, []):
                decay_mod = dedge.weight / 0.3 if dedge.weight > 0 else 1.0
                dnode = self.graph.nodes.get(dtgt)
                if dnode and dnode.label and dnode.label.lower() not in seen and dnode.label.lower() not in chain_labels and dnode.label.lower() not in _main_concepts:
                    candidates.append((dedge.weight * dedge.confidence * 0.7 * decay_mod, dnode.label, dedge, "out"))
            if not candidates:
                break
                        # ── Fused score accumulation (12 modifier loops fused into 1 pass) ──
            # Compute all multiplicative modifiers in a single O(candidates) iteration
            # instead of 12 sequential list-building loops (O(stages * candidates)).

            # Precompute global stats used by modifiers
            _recent_set = set(self._recent_traversals[-10:]) if self._recent_traversals else set()
            _use_vad = getattr(self, 'use_vad', True)
            if _use_vad:
                _valence = self.emotion.state.valence
                _dominance = self.emotion.state.dominance
            else:
                _valence = 0.0; _dominance = 0.5
            _max_w = max((c[0] for c in candidates), default=0.001)
            _prox_nids = self._concept_keywords.get(subject_proximity.lower(), []) if subject_proximity is not None else []
            _prox_node = self.graph.get_node(_prox_nids[0]) if _prox_nids else None
            _prox_vec = _prox_node.vector if _prox_node and _prox_node.vector is not None else None
            _ctx_mod_avail = self._current_context_vector is not None and not np.all(self._current_context_vector == 0)
            rlm_data = {}

            _fused = []
            for _sig, _tgt_lbl, _edge, _d in candidates:
                _pair = (cur_nid, _edge.target) if _d == "out" else (_edge.source, cur_nid)

                # 1. Dormant edge boost (side effect: may wake edges)
                if self._dormant_edges and _pair in self._dormant_edges:
                    _key = (cur_label.lower(), _tgt_lbl.lower())
                    _vc = self.user_model.edge_reactivations.get(_key, 0)
                    if _vc > 0 or _edge.confidence > 0.15:
                        self._dormant_edges.discard(_pair)
                        _edge.confidence = 0.3
                        _sig = _edge.weight * 0.3

                # 2. Contradiction penalty
                if contradiction_penalty > 0 and self._contradiction_map and                    self._is_contradictory(cur_label, _tgt_lbl, _edge.relation_type):
                    _sig *= (1.0 - contradiction_penalty)

                # 3. Preference boost (user model)
                if self.user_model.edge_reactivations:
                    _cnt = self.user_model.edge_reactivations.get((cur_label.lower(), _tgt_lbl.lower()), 0)
                    if _cnt > 0:
                        _sig *= (1.0 + (_cnt / (_cnt + 1.0)) * 0.3)

                # 4. Activation boost
                if activation_boost:
                    _sig *= activation_boost.get(_tgt_lbl.lower(), 1.0)

                # 5. Edge repetition penalty
                if self._recent_traversals and _pair in _recent_set:
                    _turn_idx = self._recent_traversal_map.get(_pair, 0)
                    _dist = len(self._recent_traversals) - _turn_idx
                    if _dist < 10:
                        _sig *= 0.3
                    elif _dist < 20:
                        _sig *= 0.6
                    else:
                        _sig *= 0.8

                # 6. VAD Edge Preference (uses raw edge.weight, not modified sig)
                if _use_vad:
                    if _valence > 0.10:
                        _sig *= (1.0 + 0.15 * (_edge.weight / _max_w))
                    elif _valence < -0.10:
                        _sig *= 0.9
                    if _dominance < 0.35:
                        _sig *= (0.6 + 0.4 * _edge.weight / _max_w)

                # 7. Context-Dependent Vector Modulation
                if _ctx_mod_avail:
                    _tgt_nids_m = self._concept_keywords.get(_tgt_lbl.lower(), [])
                    if _tgt_nids_m:
                        _mod_vec = self._modulate_vector(_tgt_nids_m[0])
                        if _mod_vec is not None:
                            _ctx_sim = float(np.dot(_mod_vec, self._current_context_vector))
                            _sig *= (1.0 + 0.1 * max(0.0, _ctx_sim))

                # 8. Subject Proximity Bonus
                if _prox_vec is not None:
                    _tgt_nids_c = self._concept_keywords.get(_tgt_lbl.lower(), [])
                    if _tgt_nids_c:
                        _tgt_node_c = self.graph.get_node(_tgt_nids_c[0])
                        if _tgt_node_c is not None and _tgt_node_c.vector is not None:
                            _cos = float(np.dot(_tgt_node_c.vector, _prox_vec))
                            _sig *= (1.0 + 0.15 * max(0.0, _cos))

                # 9. RLMv2 Confidence Modulation
                if getattr(self, 'use_rlm', True):
                    _triple = self._classify_triple(cur_label, _tgt_lbl, _edge.relation_type)
                    _rlm_conf = _triple['confidence']
                    rlm_data[_tgt_lbl.lower()] = _rlm_conf
                    if _rlm_conf < 0.4:
                        _sig *= 0.7
                    elif _rlm_conf > 0.75:
                        _sig *= 1.15
                    if _triple['is_analogical']:
                        _sig *= 0.85

                # 10. BeliefStore Confidence Weighting
                if getattr(self, 'use_beliefs', True):
                    _belief = self.belief_store.query_belief(cur_label, _edge.relation_type)
                    if _belief:
                        _value, _bconf, _ = _belief
                        if _value == _tgt_lbl:
                            _sig *= (1.0 + 0.2 * _bconf)
                        else:
                            _sig *= (1.0 - 0.4 * _bconf)

                _fused.append((_sig, _tgt_lbl, _edge, _d))
            candidates = _fused
            if not candidates:
                break
            # ── Filtering stages (keep separate — they remove candidates) ──
            # Episodic-first mode (recall)
            if episodic_first:
                _episodic_cands = [(s, l, e, d) for (s, l, e, d) in candidates if e.relation_type == "episodic"]
                if _episodic_cands:
                    candidates = _episodic_cands
            if not candidates:
                break
            # Prefrontal Gating
            if getattr(self, '_pfc_gating_enabled', True) and self._prefrontal_buffer:
                _gate_subject = subject_proximity if subject_proximity else label
                candidates = self._prefrontal_gate_candidates(candidates, _gate_subject)
            # Thalamic Gate
            candidates = self._thalamic_gate(candidates)
# Phase 13.3: Apply activation fatigue to penalize recently overused concepts
            if candidates and self._activation_fatigue:
                candidates = self._apply_activation_fatigue(candidates)

            # Phase B: Basal Ganglia Gate selection (replaces temperature-weighted softmax)
            # The 15+ modulators are reframed as inputs that set the gate's Go/NoGo parameters.
            
            # Build candidate list for BG Gate: (label, raw_score, confidence, relation_type)
            bg_input = []
            cerebellar_bias = 0.0
            for c in candidates:
                score = c[0]
                label_c = c[1]
                edge_c = c[2]
                conf_c = getattr(edge_c, 'confidence', 0.5)
                rel_c = getattr(edge_c, 'relation_type', 'semantic')
                cereb_str = self.cerebellar_ngram.get_transition_strength(label, label_c) if hasattr(self, "cerebellar_ngram") else 0.0
                cerebellar_bias = max(cerebellar_bias, cereb_str)
                bg_input.append((label_c, score + cereb_str * 0.15, conf_c, rel_c))
            
            # Set BG Gate parameters from the system's current modulators
            # All 15+ existing modulators are reframed as gate parameter inputs, not additive bonuses.
            arousal = float(self.emotion.state.arousal) if hasattr(self.emotion, 'state') and hasattr(self.emotion.state, 'arousal') else 0.3
            identity_str = float(self.identity.state.strength) if hasattr(self.identity, 'state') and hasattr(self.identity.state, 'strength') else 0.5
            # Exploration drive: computed from available fields (no ctx dependency)
            exploration_drive = 0.3 * (1.0 - identity_str) + 0.2 * arousal
            # Novelty: inline approximation from visited concepts
            n_visited = len([v for v in self._visited_concepts if v.lower() == label.lower()]) if hasattr(self, '_visited_concepts') else 0
            novelty = min(1.0, 1.0 - (n_visited / max(1, len(self._visited_concepts)))) if self._visited_concepts else 0.5
            # Fatigue: inline lookup into _activation_fatigue
            label_node_ids = self._concept_keywords.get(label.lower(), [])
            fatigue = sum(self._activation_fatigue.get(nid, 0.0) for nid in label_node_ids) if hasattr(self, '_activation_fatigue') else 0.0
            fatigue = min(1.0, fatigue)
            
            self.basal_ganglia.set_arousal(arousal)
            self.basal_ganglia.set_novelty(novelty)
            self.basal_ganglia.set_exploration_drive(exploration_drive)
            self.basal_ganglia.set_prediction_error(self._mean_prediction_error)
            self.basal_ganglia.set_identity_strength(identity_str)
            self.basal_ganglia.set_fatigue(fatigue)
            pfc_labels = [p.lower() for p in getattr(self, '_prefrontal_buffer', [])]
            self.basal_ganglia.set_prefrontal_boost(0.5 if label.lower() in pfc_labels else 0.0)
            self.basal_ganglia.set_thalamic_salience(0.5)
            if subject_proximity:
                self.basal_ganglia.set_subject_proximity_bonus(0.3)
            self.basal_ganglia.set_contradiction_penalty(contradiction_penalty)
            self.basal_ganglia.set_dopamine_tone(getattr(self, '_dopamine_tone', 0.5))
            
            # Phase C: Cerebellar n-gram transition bias — prefer grammatically-proven transitions
            # Phase C: cerebellar bias is applied per-candidate above (not post-hoc)
            # If cerebellar predicts this transition, boost the selection; if not, slight penalty
            self.basal_ganglia.set_subject_proximity_bonus(max(0.0, cerebellar_bias * 0.5))
            
            # Select via BG Gate
            bg_label, bg_rel, bg_go_score = self.basal_ganglia.select_concept(bg_input, self.rng)
            
            if bg_label:
                found_idx = -1
                for ci, (_, cl, _, _) in enumerate(candidates):
                    if cl.lower() == bg_label.lower():
                        found_idx = ci
                        break
                if found_idx >= 0:
                    idx = found_idx
                else:
                    idx = max(range(len(candidates)), key=lambda i: candidates[i][0])
            else:
                # Gate returned no winner — fallback to greedy
                idx = max(range(len(candidates)), key=lambda i: candidates[i][0])
            best_sig, best_label, best_edge, direction = candidates[idx]
            # Fix 2: Awaken dormant edge when first traversed
            if self._dormant_edges:
                be_pair = (cur_nid, best_edge.target) if direction == "out" else (best_edge.source, cur_nid)
                if be_pair in self._dormant_edges:
                    self._dormant_edges.discard(be_pair)
                    best_edge.confidence = 0.3  # awakened to normal level
            # Phase 10.1: Per-hop prediction error
            _cur_node_walk = self.graph.get_node(cur_nid)
            _tgt_node_walk = self.graph.get_node(best_edge.target) if best_edge.target is not None else self.graph.get_node(best_edge.source)
            _hop_pe = 0.0
            if _cur_node_walk is not None and _cur_node_walk.vector is not None and _tgt_node_walk is not None and _tgt_node_walk.vector is not None:
                _sn_walk = None
                if subject_proximity:
                    _snids_walk = self._concept_keywords.get(subject_proximity.lower(), [])
                    if _snids_walk:
                        _sn_node = self.graph.get_node(_snids_walk[0])
                        if _sn_node is not None and _sn_node.vector is not None:
                            _sn_walk = _sn_node.vector
                _subj_vec_walk = _sn_walk if _sn_walk is not None else _cur_node_walk.vector
                _pfc_vecs_walk = []
                for _bl in self._prefrontal_buffer:
                    _bn = self._concept_keywords.get(_bl, [])
                    if _bn:
                        _bnnode = self.graph.get_node(_bn[0])
                        if _bnnode is not None and _bnnode.vector is not None:
                            _pfc_vecs_walk.append(_bnnode.vector)
                _pfc_centroid_walk = np.mean(_pfc_vecs_walk, axis=0) if _pfc_vecs_walk else np.zeros(self.dim)
                _hop_pe = self._compute_hop_prediction_error(
                    _cur_node_walk.vector, _subj_vec_walk, _pfc_centroid_walk, _tgt_node_walk.vector)
                _alpha_pe = 0.05
                self._mean_sentence_pe = (1 - _alpha_pe) * self._mean_sentence_pe + _alpha_pe * _hop_pe
                self._sentence_pe_count += 1
            # Phase 14.1: TD learning on traversed edge
            if _hop_pe > 0:
                self._td_learn(cur_label, best_label, best_edge, _hop_pe)
            # Phase 14.2: Update dopamine tone from recent TD errors
            self._update_dopamine_tone()
            # Phase 17.1: Update per-concept confidence (only when PE computed)
            if _hop_pe > 0:
                self._update_concept_confidence(best_label, _hop_pe)
            # Phase 13.1: Activation fatigue for traversed nodes
            for _nid in [cur_nid, best_edge.source, best_edge.target]:
                if _nid is not None and _nid >= 0:
                    self._activation_fatigue[_nid] = self._activation_fatigue.get(_nid, 0.0) + 0.15
            # Phase 13.2: Track recent traversals
            if best_edge.source is not None and best_edge.target is not None:
                pair = (best_edge.source, best_edge.target)
                self._recent_traversals.append(pair)
                if len(self._recent_traversals) > 30:
                    self._recent_traversals = self._recent_traversals[-30:]
                # Rebuild map from pruned list (correct positions)
                self._recent_traversal_map = {p: len(self._recent_traversals) - i
                                               for i, p in enumerate(reversed(self._recent_traversals))}
            connector = self._pick_connector(best_edge.relation_type, best_edge.weight,
                                             best_edge.confidence, temperature)
            # Retry same hop if best neighbor IS the connector word
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
            # Phase 13.3/16.1: Track visited concepts for novelty scoring
            self._visited_concepts.add(best_label.lower())
            # Record belief assertion BEFORE ChainHop creation (contradiction_detail needed for trace)
            self._belief_assertions.append((cur_label, best_edge.relation_type, best_label))
            contradiction_detail: str = ""
            if getattr(self, 'use_beliefs', True):
                # Check for contradiction BEFORE asserting (detect overwrite)
                c = self.belief_store.detect_contradiction(
                    cur_label, best_edge.relation_type, best_label)
                if c is not None:
                    (old_val, old_conf, old_turn), new_val, _ = c
                    contradiction_detail = (
                        f"{cur_label} -> {old_val} (turn {old_turn}) vs "
                        f"{cur_label} -> {new_val} @ conf {best_edge.weight:.3f}")
                    if self._trace_enabled:
                        print(f"  [trace]     contradiction: {contradiction_detail}")
                self.belief_store.assert_belief(
                    cur_label, best_edge.relation_type, best_label,
                    confidence=best_edge.weight)
                self.belief_store.advance_turn()
            # RLM confidence for the selected edge
            rlm_hop_conf = rlm_data.get(best_label.lower(), 0.0) if rlm_data else 0.0
            if trace is not None:
                trace.hops.append(ChainHop(
                    from_label=cur_label, to_label=best_label,
                    relation_type=best_edge.relation_type,
                    weight=best_edge.weight, confidence=best_edge.confidence,
                    temperature=temperature, candidates=len(candidates),
                    rlm_confidence=rlm_hop_conf,
                    contradiction=contradiction_detail))
            local_hops.append((cur_label, best_label))
            if getattr(self, 'reasoning_mode', 'stochastic') == 'deterministic':
                self._consistency_trace.append(best_label)
            cur_label = best_label
            cur_nid = self._concept_keywords.get(best_label.lower(), [None])[0]
            if cur_nid is None:
                break
            hops += 1
        if trace is not None:
            trace.completed = hops >= max_hops
            self._chain_traces.append(trace)
        if local_hops:
            self._last_hops.append(local_hops)
        if len(chain) <= 1:
            return None
        return " ".join(chain)


    # ─── Phase 7: Multi-Strategy Reasoning ───

    def _bridge_prospecting(self, activated_ids: List[int], subject: str,
                             seen: Set[str], temperature: float) -> Optional[str]:
        """Strategy B: Find a bridge concept connecting multiple activated inputs.
        Reuses _spread_and_collect to find the highest-activation non-seed concept."""
        if len(activated_ids) < 2:
            return None
        assocs = self._spread_and_collect(activated_ids, primary_ids=set(activated_ids))
        if not assocs:
            return None
        bridge = assocs[0][0]
        if bridge.lower() in seen or bridge.lower() == subject.lower():
            if len(assocs) > 1:
                bridge = assocs[1][0]
            else:
                return None
        # Walk from the bridge concept
        return self._walk_chain(bridge, seen, max_hops=2, temperature=temperature,
                                subject_proximity=subject)

    def _analogical_detour(self, subject: str, seen: Set[str],
                            temperature: float) -> Optional[str]:
        """Strategy C: Walk from analogically-linked concepts to the subject."""
        subj_nids = self._concept_keywords.get(subject.lower(), [])
        if not subj_nids:
            return None
        subj_nid = subj_nids[0]
        analogical_targets = []
        for tid, edge in self.graph.get_outgoing(subj_nid):
            if edge.relation_type == "analogical":
                tn = self.graph.get_node(tid)
                if tn and tn.label and tn.label.lower() not in seen:
                    analogical_targets.append(tn.label)
        for src, edge in self.graph.get_incoming(subj_nid):
            if edge.relation_type == "analogical":
                sn = self.graph.get_node(src)
                if sn and sn.label and sn.label.lower() not in seen:
                    analogical_targets.append(sn.label)
        if not analogical_targets:
            return None
        # Walk from the first analogical target
        start = analogical_targets[0]
        seen.add(start.lower())
        return self._walk_chain(start, seen, max_hops=2, temperature=temperature,
                                subject_proximity=subject)

    def _contrastive_flip(self, subject: str, seen: Set[str],
                           temperature: float) -> Optional[str]:
        """Strategy D: Walk from the opposite of the subject (contrastive edges)."""
        antonyms = self._contradiction_map.get(subject.lower(), set())
        if not antonyms:
            return None
        for antonym in antonyms:
            if antonym not in seen:
                seen.add(antonym)
                result = self._walk_chain(antonym, seen, max_hops=2, temperature=temperature)
                if result:
                    return f"{subject} but {result}"
        return None

    def _sub_question_decompose(self, subject: str, query: str,
                                 seen: Set[str], temperature: float) -> Optional[str]:
        """Strategy E: Split multi-word queries into existing graph concepts."""
        words = re.findall(r"[a-zA-Z']{3,}", query.lower())
        existing = [w for w in words
                    if w in self._concept_keywords and w != subject.lower() and w not in seen]
        if not existing:
            return None
        # Pick the most connected existing concept and walk from it
        best_word = existing[0]
        seen.add(best_word)
        return self._walk_chain(best_word, seen, max_hops=2, temperature=temperature,
                                subject_proximity=subject)

    # Natural sentence templates for LLM-like output
        return " ".join(parts) + "."

    # ─── Phase 8: Sentence Formatting & Prefrontal Coherence ───

    def _is_low_quality_response(self, sentences: list) -> bool:
        """Detect circular/repetitive/low-quality responses.
        
        Stage 1: Loosened — natural chat allows concept carry-across and
        coreference. Only flag truly degenerate cases.
        """
        if not sentences:
            return True
        full = ' '.join(sentences).lower()
        words = full.split()
        if len(words) < 3:
            return True
        # Stage 1: Only flag extreme repetition (6+ occurrences)
        from collections import Counter
        word_counts = Counter(w for w in words if len(w) > 3)
        if any(c >= 6 for c in word_counts.values()):
            return True
        # Stage 1: Only flag exact word-set duplicates (same sentence verbatim)
        sentences_lower = [s.lower().strip() for s in sentences]
        if len(sentences_lower) >= 2:
            for i, s1 in enumerate(sentences_lower):
                for s2 in sentences_lower[i+1:]:
                    if s1 == s2 and len(s1) > 15:
                        return True
        return False

    # ─── Phase E+F+G: Syntactic assembly + Surface realization pipeline ───


    def _generate_response(self, ctx: CognitiveResponseContext) -> Tuple[str, str]:
        """Generate response using syntactic pipeline (primary), neural decoder
        (when proven fluent), then reasoning loop (last resort).

        Order reflects neuroscience:
        - Syntactic pipeline = P600-driven compositional integration (primary route)
        - Neural decoder = overlearned fluent speech (requires high quality)
        - Reasoning loop = web learning when knowledge is absent

        Neuroscience basis:
        - Brouwer Retrieval-Integration: composition (P600) is the default route
        - PMC 2023: hierarchical predictive coding across multiple timescales
        - Nature Human Behaviour 2025: multi-timescale discourse organization
        """
        subject = ctx.subject
        assocs = ctx.associated_concepts

        # Path 1: Syntactic pipeline (P600 compositional integration) — primary
        try:
            syntax_response = self._generate_with_decoder_and_syntax(ctx)
            if syntax_response and len(syntax_response) > 10:
                return (syntax_response, "syntactic_pipeline")
        except Exception:
            pass

        # Path 2: Neural decoder — only when proven fluent (strict quality gate)
        decoder_ready = False
        if self.neural_decoder is not None and self._decoder_vocab_built:
            nd = self.neural_decoder
            ce_ok = nd._avg_cross_entropy < 4.0 if nd._metric_examples > 10 else False
            t1_ok = nd._avg_top1_acc > 0.25 if nd._metric_examples > 10 else False
            trained_enough = self._decoder_training_count >= 2000
            decoder_ready = ce_ok and t1_ok and trained_enough
        if decoder_ready:
            try:
                decoder_response = self._generate_with_decoder(ctx)
                if decoder_response and len(decoder_response) > 10:
                    return (decoder_response, "neural_decoder")
            except Exception:
                pass

        # Path 3: Reasoning loop (web search + learn, then retry)
        try:
            return self._reasoning_loop(ctx)
        except Exception as e:
            if self._trace_enabled:
                print(f"  [trace] reasoning loop error: {e}")
            raise

    def _generate_with_decoder_and_syntax(self, ctx: CognitiveResponseContext) -> Optional[str]:
        """Generate response using full syntactic pipeline:

        1. PrefrontalWorkspace: Plan discourse (3-sentence intent arc)
        2. SyntacticCellAssembly: Bind concepts to grammatical frames
        3. BasalGangliaGate: Concept selection (Go/NoGo)
        4. CerebellarNgram: Completion hints for function words
        5. SurfaceRealizer: Final text with morphology, agreement, articles

        Uses graph walk chain concepts directly - decoder produces garbage.
        """
        subject = ctx.subject
        assocs = ctx.associated_concepts
        if not subject:
            return None
        # Provide fallback associations if empty - use subject and vector neighbors
        if not assocs:
            neighbor = self._find_vector_neighbor(subject)
            if neighbor:
                assocs = [(neighbor, 0.5)]
            else:
                assocs = [(subject, 1.0)]

        # ─── P2 Emotional Mirroring: concept breadth modulation ───
        # User arousal broadens concept exploration; low arousal narrows it.
        user_arousal = self.user_model.emotional_state.get('arousal', 0.3)
        breadth_mult = 0.5 + user_arousal  # [0.5, 1.5]
        max_assocs = max(3, min(15, int(round(8 * breadth_mult))))

        # ─── STEP 1: PrefrontalWorkspace — Discourse Planning ───
        plan = self.pfc_workspace.plan_discourse(
            user_input=ctx.raw_input,
            subject=subject,
            concept_pos=self._concept_pos,
            associations=assocs[:max_assocs],
            past_topics=ctx.past_topics,
            is_follow_up=self._is_follow_up(ctx.raw_input),
        )

        # P1 Theory of Mind: adaptive verbosity based on user familiarity (roadmap §7)
        try:
            plan = self._adapt_verbosity_for_user(plan, subject)
        except Exception:
            pass  # Never break the pipeline for ToM modulation
        
        if self._trace_enabled:
            print(f"  [trace] Discourse plan: {len(plan.intents)} intents")
            for i, intent in enumerate(plan.intents):
                print(f"    Intent {i}: type={intent.type}, subject={intent.subject}, target={intent.target_concept}, relation={intent.primary_relation}, marker={intent.discourse_marker}")
        
        # ─── STEP 2: For each discourse intent, generate a sentence ───
        sentences = []
        dopamine_tone = getattr(self, '_dopamine_tone', 0.5)
        # Stage 1: Allow a concept to carry across 2-3 sentences with pronoun/coreference.
        # This prevents the choppy "Trust… It… Likewise it…" pattern.
        used_targets = {}  # concept -> last sentence index used
        target_reuse_window = 3  # Allow reuse after 3 sentences (was 2)
        last_subject = None  # For pronoun coreference across sentences
        
        for sent_idx, intent in enumerate(plan.intents):
            # Generate concept sequence for this intent
            intent_subject = intent.subject
            intent_target = intent.target_concept
            intent_relation = intent.primary_relation
            
            # Filter out grammatical concepts from discourse plan target
            if intent_target and intent_target.lower() in self._GRAMMATICAL_CONCEPTS:
                intent_target = ""
            
            # Walk chain for this intent's relation type
            seen = {intent_subject.lower()}
            if intent_target:
                seen.add(intent_target.lower())
            
            # Dynamic graph walk: arousal + dopamine modulate depth/creativity
            _hops = 2 + int(dopamine_tone * 2) + (1 if user_arousal > 0.6 else 0)
            _hops = min(_hops, 5)
            _temp = 0.2 + user_arousal * 0.5 + dopamine_tone * 0.3
            _temp = min(_temp, 0.8)
            chain_result = self._walk_chain(
                label=intent_subject,
                seen=seen.copy(),
                max_hops=_hops,
                temperature=_temp,
                contradiction_penalty=0.6,
                activation_boost=self._activation_boost,
                subject_proximity=intent_subject,
            )
            
            # Parse chain into concepts
            chain_concepts = []
            chain_connectors = []
            if chain_result:
                for token in chain_result.split():
                    if token in self._CONNECTOR_SET:
                        chain_connectors.append(token)
                    else:
                        is_gram = token.lower() in self._GRAMMATICAL_CONCEPTS
                        if not is_gram:
                            chain_concepts.append(token)
            
            if self._trace_enabled:
                print(f"  [trace] Sent {sent_idx}: intent_subject={intent_subject}, intent_target={intent_target}, chain_result={chain_result}")
                print(f"    chain_concepts={chain_concepts}, chain_connectors={chain_connectors}")
                print(f"    used_targets={used_targets}, last_subject={last_subject}")
            
            # Use discourse target if chain didn't produce it, else use chain's top concept
            target_for_frame = intent_target
            
            # PRIORITY 1: Use valid chain concepts (grounded in graph structure)
            # Stage 1: Don't require disjoint targets — allow carry-across
            if not target_for_frame and chain_concepts:
                for c in chain_concepts:
                    cl = c.lower()
                    if cl not in seen and cl not in self._GRAMMATICAL_CONCEPTS:
                        target_for_frame = c
                        break
            
            # PRIORITY 2: Use discourse plan target if valid
            if not target_for_frame and intent_target:
                it_lower = intent_target.lower()
                if it_lower not in seen and it_lower not in self._GRAMMATICAL_CONCEPTS:
                    target_for_frame = intent_target
            
            # PRIORITY 3: Vector neighbor as last resort
            if not target_for_frame:
                neighbor = self._find_vector_neighbor(intent_subject)
                if self._trace_enabled:
                    print(f"    _find_vector_neighbor({intent_subject}) = {neighbor}")
                if neighbor:
                    nb_lower = neighbor.lower()
                    if nb_lower not in seen and nb_lower not in self._GRAMMATICAL_CONCEPTS:
                        target_for_frame = neighbor
            
            # Stage 1: Allow reuse if it's been at least target_reuse_window sentences,
            # or if this is adjacent sentence reuse (for coreference/continuity)
            can_reuse = False
            if target_for_frame:
                tfl = target_for_frame.lower()
                if tfl in used_targets:
                    last_used = used_targets[tfl]
                    if sent_idx - last_used >= target_reuse_window:
                        can_reuse = True
                    elif last_subject and last_subject.lower() == tfl:
                        can_reuse = True  # Allow subject to carry across
                else:
                    can_reuse = True
            else:
                can_reuse = False
            
            if not target_for_frame or not can_reuse or target_for_frame.lower() in self._GRAMMATICAL_CONCEPTS:
                if self._trace_enabled:
                    print(f"    SKIPPING sentence {sent_idx}: target_for_frame={target_for_frame}")
                continue
            
            used_targets[target_for_frame.lower()] = sent_idx
            last_subject = target_for_frame
            
            # ─── STEP 4: SyntacticCellAssembly — Bind to grammatical frame ───
            frame = self.syntactic_assembly.bind_to_sentence(
                subject=intent_subject,
                relation=intent_relation,
                target=target_for_frame,
                pos_map=self._concept_pos,
                chain_concepts=chain_concepts,
                chain_connectors=chain_connectors,
                depth=sent_idx,
            )
            
            # Apply discourse marker from plan
            discourse_marker = intent.discourse_marker
            
            # ─── STEP 5: SurfaceRealizer — Morphology, agreement, pronouns, articles ───
            discourse_state = DiscourseState(
                sentence_index=sent_idx,
                previous_subject=sentences[-1].split()[0].lower() if sentences else None,
                discourse_type=intent.type,
                total_sentences=len(plan.intents),
            )
            
            try:
                sentence = self.surface_realizer.realize(
                    frame=frame,
                    discourse_context=discourse_state,
                    dopamine_tone=dopamine_tone,
                    cerebellar_ngram=getattr(self, 'cerebellar_ngram', None),
                    discourse_marker=discourse_marker,
                )
                sentences.append(sentence)
            except Exception as e:
                if self._trace_enabled:
                    print(f"  [trace] SurfaceRealizer error: {e}")
                continue
        
        if not sentences:
            return None
        
        return " ".join(sentences)

    def _reasoning_loop(self, ctx: CognitiveResponseContext) -> Tuple[str, str]:
        """Multi-step reasoning with web search for complex queries.

        This replaces the cascade strategies (B-G) with a unified loop:
        1. Analyze query complexity
        2. Decompose into sub-questions if needed
        3. Search web for each sub-question
        4. Learn from full articles (trains decoder continuously)
        5. Synthesize and generate with decoder
        """
        subject = ctx.subject
        query = ctx.raw_input
        assocs = ctx.associated_concepts

        if self._trace_enabled:
            print(f"  [reasoning] Starting reasoning loop for: {subject}")

        # Step 1: Check if we need web search
        subj_lower = subject.lower()
        subj_known = subj_lower in self._concept_keywords or subj_lower in self._concept_labels
        assoc_known = len(assocs) > 0

        # Complex query indicators
        is_complex = any(w in query.lower() for w in
                        ["how", "why", "create", "build", "design", "blueprint",
                         "explain", "detail", "comprehensive", "step by step",
                         "architecture", "implementation", "guide", "tutorial"])
        is_unknown = not subj_known or not assoc_known

        # Determine search queries
        search_queries = []

        # Check for unknown multi-word phrases in the original query
        # Extract phrases that aren't fully grounded
        query_words = ctx.raw_input.lower().split()
        unknown_phrases = []
        for i in range(len(query_words) - 1):
            phrase = ' '.join(query_words[i:i+2])
            if phrase not in self._concept_keywords and phrase not in self._concept_labels:
                unknown_phrases.append(phrase)
        # Also check 3-grams
        for i in range(len(query_words) - 2):
            phrase = ' '.join(query_words[i:i+3])
            if phrase not in self._concept_keywords and phrase not in self._concept_labels:
                unknown_phrases.append(phrase)

        # Determine search queries
        search_queries = []

        # Check for unknown multi-word phrases in the original query
        query_words = ctx.raw_input.lower().split()
        unknown_phrases = []
        for i in range(len(query_words) - 1):
            phrase = ' '.join(query_words[i:i+2])
            if phrase not in self._concept_keywords and phrase not in self._concept_labels:
                unknown_phrases.append(phrase)
        # Also check 3-grams
        for i in range(len(query_words) - 2):
            phrase = ' '.join(query_words[i:i+3])
            if phrase not in self._concept_keywords and phrase not in self._concept_labels:
                unknown_phrases.append(phrase)

        # Complex query indicators
        is_complex = any(w in query.lower() for w in
                        ["how", "why", "create", "build", "design", "blueprint",
                         "explain", "detail", "comprehensive", "step by step",
                         "architecture", "implementation", "guide", "tutorial"])
        is_unknown = not subj_known or not assoc_known

        # Step 1: Check if we need web search
        subj_lower = subject.lower()
        subj_known = subj_lower in self._concept_keywords or subj_lower in self._concept_labels
        assoc_known = len(assocs) > 0

        # Complex query indicators
        is_complex = any(w in query.lower() for w in
                        ["how", "why", "create", "build", "design", "blueprint",
                         "explain", "detail", "comprehensive", "step by step",
                         "architecture", "implementation", "guide", "tutorial"])
        is_unknown = not subj_known or not assoc_known

        # Determine search queries
        search_queries = []

        # Check for unknown multi-word phrases in the original query
        query_words = ctx.raw_input.lower().split()
        unknown_phrases = []
        for i in range(len(query_words) - 1):
            phrase = ' '.join(query_words[i:i+2])
            if phrase not in self._concept_keywords and phrase not in self._concept_labels:
                unknown_phrases.append(phrase)
        # Also check 3-grams
        for i in range(len(query_words) - 2):
            phrase = ' '.join(query_words[i:i+3])
            if phrase not in self._concept_keywords and phrase not in self._concept_labels:
                unknown_phrases.append(phrase)

        # Determine search queries
        if is_complex or is_unknown:
            # Decompose into sub-questions
            search_queries = self._decompose_for_search(query, subject, assocs)
        elif self._decoder_training_count < 2000:
            # Even for known topics, search to expand knowledge if decoder needs training
            search_queries = [subject]

        # Add unknown multi-word phrases as search queries
        for phrase in unknown_phrases[:3]:  # Limit to top 3
            if phrase not in search_queries:
                search_queries.append(phrase)

        # Limit concurrent searches to prevent overload
        search_queries = search_queries[:3]  # Only 1 search max for speed in reasoning loop

        if self._trace_enabled:
            print(f"  [reasoning] Search queries: {search_queries}")

        all_learned_text = ""
        for sq in search_queries:
            if self._trace_enabled:
                print(f"  [reasoning] Searching: {sq}")
            try:
                # Synchronous with shorter timeout
                result, article_text = self.learn_from_web(sq, max_results=3)
                if self._trace_enabled:
                    print(f"  [reasoning]   Result: {result}")
                if article_text:
                    all_learned_text += " " + article_text
            except Exception as e:
                if self._trace_enabled:
                    print(f"  [reasoning]   Search failed: {e}")

        # Step 3: Train decoder on all gathered article text
        if all_learned_text and self.neural_decoder is not None and self._decoder_vocab_built:
            if self._trace_enabled:
                print(f"  [reasoning] Training decoder on {len(all_learned_text)} chars of article text")
            self._train_decoder_on_text(all_learned_text, subject)

        # Step 4: Generate response with decoder + syntactic pipeline
        try:
            # Prevent vocab expansion during generation
            self._decoder_vocab_built = True
            decoder_response = self._generate_with_decoder_and_syntax(ctx)
            if decoder_response and len(decoder_response) > 10:
                return (decoder_response, "neural_decoder_reasoned")
        except Exception as e:
            if self._trace_enabled:
                print(f"  [reasoning] Decoder generation failed: {e}")

        # Fallback to graph walk
        return self._graph_fallback_response(ctx)


    def _decompose_for_search(self, query: str, subject: str, assocs: List[Tuple]) -> List[str]:
        """Decompose a complex query into targeted search sub-questions."""
        queries = [subject]  # Always include main subject

        q_lower = query.lower()

        # Add specific sub-questions based on query type
        if "blueprint" in q_lower or "create" in q_lower or "build" in q_lower or "design" in q_lower:
            queries.extend([
                f"{subject} design principles",
                f"{subject} components architecture",
                f"how to build {subject}",
                f"{subject} engineering guide"
            ])
        elif "how" in q_lower and "work" in q_lower:
            queries.extend([
                f"how does {subject} work",
                f"{subject} mechanism explained",
                f"{subject} operating principles"
            ])
        elif "why" in q_lower:
            queries.extend([
                f"why {subject} importance",
                f"{subject} purpose explained"
            ])
        elif "explain" in q_lower or "detail" in q_lower or "comprehensive" in q_lower:
            queries.extend([
                f"{subject} explained in detail",
                f"{subject} comprehensive overview",
                f"{subject} deep dive"
            ])

        # Add associated concepts as searches
        for label, _ in assocs[:3]:
            if label.lower() != subject.lower():
                queries.append(f"{subject} {label}")

        # Deduplicate
        seen = set()
        unique = []
        for q in queries:
            ql = q.lower()
            if ql not in seen:
                seen.add(ql)
                unique.append(q)

        return unique


    def _fetch_full_article_text(self, query: str) -> str:
        """Fetch full article text from web search for decoder training."""
        try:
            try:
                results = self.search_engine.search(query, max_results=10)
            except SearchError:
                return ""

            full_texts = []
            for r in results[:5]:  # Fetch more articles
                url = r.get("url", "")
                if url:
                    try:
                        article = self._fetch_article_text(url)
                        if article and len(article) > 200:
                            full_texts.append(article)
                    except Exception:
                        continue

            return " ".join(full_texts)
        except Exception:
            return ""


    def _train_decoder_on_text(self, text: str, topic: str):
        """Train neural decoder on full article text with graph conditioning."""
        if self.neural_decoder is None or not self._decoder_vocab_built:
            return

        # Build conditioning from topic and related concepts
        cond_embs = self._build_conditioning_for_text(topic, re.findall(r"[a-zA-Z']{3,}", text.lower()))

        # Expand vocab with new words from text
        words_in_text = set(re.findall(r"[a-zA-Z']{3,}", text.lower()))
        new_for_vocab = [w for w in words_in_text
                         if w not in self._decoder_word_to_idx
                         and w not in STOP_WORDS
                         and len(w) >= 3]
        if new_for_vocab:
            self._expand_decoder_vocab(list(new_for_vocab)[:50])  # Limit expansion

        # Train on text (multiple passes for Hebbian consolidation)
        total_err = 0.0
        total_trained = 0
        for _ in range(3):  # More passes for article text
            err, n = self.neural_decoder.train_on_text(
                text, self._decoder_word_to_embed, self._decoder_word_to_idx,
                min_sentence_len=3, conditioning_embs=cond_embs
            )
            total_err += err
            total_trained += n

        self._decoder_training_count += total_trained
        self._decoder_web_training_count += total_trained

        # Also train cerebellar n-gram
        if hasattr(self, 'cerebellar_ngram') and self.cerebellar_ngram is not None:
            # Note: learn_from_text method doesn't exist, skipping
            pass
        self.neural_decoder.sleep_cycle()

        if self._trace_enabled:
            print(f"  [reasoning] Decoder trained on {total_trained} sentences from articles")


    def _safe_graph_node(self, label: str):
        """Safely look up a graph node by label, returning None if not found."""
        nids = self._concept_keywords.get(label.lower(), [])
        if nids:
            return self.graph.get_node(nids[0])
        return None

    def _graph_fallback_response(self, ctx: CognitiveResponseContext) -> Tuple[str, str]:
        """Graph-walk fallback when decoder / syntactic pipeline are not enough."""
        subject = ctx.subject
        assocs = ctx.associated_concepts

        if not subject:
            return ("...", "associative")

        # We still know notable entities that aren’t explained well yet -> ask.
        if not assocs:
            subj_lower = subject.lower()
            if subj_lower in self._concept_keywords or subj_lower in self._concept_labels:
                return (f"I recognize {subject}, but I can't explain it yet. Want me to keep exploring?", "associative")
            return (f"I don't know about {subject} yet.", "unknown_subject")

        temps = [self._get_temperature(0), self._get_temperature(1), self._get_temperature(2)]
        sentences_parts: List[str] = []
        seen: Set[str] = {subject.lower()}
        from random import Random
        rnd = Random(self.rng.randint(0, 10**9))

        def _one_rel(a: str, b: str) -> str:
            relation_phrases = [
                "reminds me of",
                "ties into",
                "leads toward",
                "feels related to",
                "points to",
                "shows up again with",
                "keeps coming back to",
            ]
            phrase = rnd.choice(relation_phrases)
            return f"{a} {phrase} {b}."

        seen_full: Set[str] = set()
        max_loops = max(3, min(8, len(assocs) * 2))
        loops = 0

        while len(sentences_parts) < 4 and loops < max_loops:
            loops += 1
            label, score = assocs[(self.turn_count + loops) % len(assocs)]
            if not label or label.lower() in seen:
                continue

            node = self._safe_graph_node(label)
            node_b = self._safe_graph_node(subject)
            if node and node_b and hasattr(node, "vector") and hasattr(node_b, "vector") and node.vector is not None and node_b.vector is not None:
                try:
                    from numpy import dot
                    from numpy.linalg import norm
                    relation_score = dot(node.vector, node_b.vector) / (norm(node.vector) * norm(node_b.vector) + 1e-9)
                except Exception:
                    relation_score = float(score)
            else:
                relation_score = float(score)

            relation_score = max(0.0, min(1.0, relation_score))
            sentence = _one_rel(subject.capitalize(), label.capitalize())
            sentence_key = " ".join(sorted([subject.lower(), label.lower()]))
            if sentence_key in seen_full:
                continue
            seen_full.add(sentence_key)
            sentences_parts.append(sentence)
            seen.add(label.lower())

        # Add one relevant follow-up.
        if len(sentences_parts) == 1:
            first_lower = sentences_parts[0].lower()
            follow_up = getattr(self, "_ASK_BACK", {}).get("default", "")
            if not follow_up:
                follow_up = "Does that match what you were getting at?"
            sentences_parts.append(follow_up)

        if not sentences_parts:
            return (f"I don't know much about {subject} yet.", "associative")

        return (" ".join(sentences_parts), "graph_fallback")



    def _update_state(self, ctx: CognitiveResponseContext):
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
            resolution_streak=sum(1 for r in self._last_responses if len(r) > 20),
            correctness=correct,
        )
        self.identity.apply_update(new_s)
        # Clamp identity
        if self.identity.state.strength > 1.0:
            self.identity.state.strength = 1.0
        if self.identity.state.strength < 0.0:
            self.identity.state.strength = 0.0

        self.gw.submit_bid(source="dialogue",
            payload={"subject": ctx.subject, "turn": self.turn_count},
            urgency=0.3 + 0.15 * min(len(ctx.associated_concepts), 4) / 4.0,
            valence=self.emotion.state.valence, episode=self.turn_count)
        self.gw.compete()

    # ── Sleep Consolidation ──

    
    # ── Phase 15: Complementary Learning Systems ──
    # Neuroscience: Hippocampus (fast, sparse) + Neocortex (slow, overlapping) (McClelland 1995)
    # Consolidation only when it aids generalization (Nat Neurosci 2023)

    def _add_episodic_edge(self, src: int, tgt: int, weight: float = 0.3, confidence: float = 0.5):
        """Phase 15.1: Add a fast-learning, high-decay episodic edge."""
        if not hasattr(self, '_episodic_edges'):
            self._episodic_edges = {}
        key = (src, tgt)
        if key not in self._episodic_edges:
            edge = ConceptEdge(src, tgt, weight=weight, confidence=confidence, relation_type="episodic")
            self._episodic_edges[key] = edge

    def _add_semantic_edge(self, src: int, tgt: int, weight: float = 0.4, confidence: float = 0.3):
        """Phase 15.1: Add a slow-learning, stable semantic edge."""
        if not hasattr(self, '_semantic_edges'):
            self._semantic_edges = {}
        key = (src, tgt)
        if key not in self._semantic_edges:
            edge = ConceptEdge(src, tgt, weight=weight, confidence=confidence, relation_type="semantic")
            self._semantic_edges[key] = edge
            self._semantic_by_src.setdefault(src, []).append((tgt, edge))

    def _decay_episodic_edges(self):
        """Phase 15.2: Decay episodic edges — 10% per turn, remove below 0.05."""
        if not hasattr(self, '_episodic_edges'):
            return
        to_remove = []
        for (src, tgt), edge in self._episodic_edges.items():
            edge.weight *= 0.90
            if edge.weight < 0.05:
                to_remove.append((src, tgt))
        for key in to_remove:
            del self._episodic_edges[key]
            # Clean up src-indexed lookup
            src_id, tgt_id = key
            if src_id in self._episodic_by_src:
                self._episodic_by_src[src_id] = [(t, e) for t, e in self._episodic_by_src[src_id] if t != tgt_id]
                if not self._episodic_by_src[src_id]:
                    del self._episodic_by_src[src_id]
            # Update src-indexed lookup
            src_id, tgt_id = key
            if src_id in self._episodic_by_src:
                self._episodic_by_src[src_id] = [(t, e) for t, e in self._episodic_by_src[src_id] if t != tgt_id]

    def _consolidate_to_semantic(self):
        """Phase 15.5: Transfer frequently-used episodic edges to semantic store.
        
        Edges traversed 2+ times get consolidated. If semantic equivalent exists,
        weighted average merge. Episodic weight halved after consolidation.
        Only consolidates if it aids generalization (Phase 15.6).
        """
        if not hasattr(self, '_episodic_edges') or not hasattr(self, '_semantic_edges'):
            return 0
        
        # Count traversals from edge_reactivations
        consolidated = 0
        for (src, tgt), edge in list(self._episodic_edges.items()):
            # Check if traversed 2+ times
            src_node = self.graph.get_node(src)
            tgt_node = self.graph.get_node(tgt)
            if src_node is None or tgt_node is None:
                continue
            key = (src_node.label.lower(), tgt_node.label.lower())
            count = self.user_model.edge_reactivations.get(key, 0)
            
            if count >= 2 and self._should_consolidate(src, tgt, edge.weight):
                sem_key = (src, tgt)
                if sem_key in self._semantic_edges:
                    sem_edge = self._semantic_edges[sem_key]
                    sem_edge.weight = (sem_edge.weight + edge.weight * count) / (1 + count)
                else:
                    self._add_semantic_edge(src, tgt, weight=edge.weight * 0.8, confidence=0.3)
                edge.weight *= 0.5
                consolidated += 1
        return consolidated

    def _should_consolidate(self, src: int, tgt: int, epi_weight: float) -> bool:
        """Phase 15.6: Only consolidate if it aids generalization.
        
        If existing semantic edges from src predict a similar weight, consolidate.
        If very different, don't consolidate (exception, not pattern).
        """
        if not hasattr(self, '_semantic_edges') or not self._semantic_edges:
            return True
        weights = [e.weight for (s, t), e in self._semantic_edges.items() if s == src]
        if not weights:
            return True
        pred_weight = np.mean(weights)
        delta = abs(epi_weight - pred_weight)
        return delta < 0.3


    # ── Phase 16: Thalamocortical Gating & Forward Model ──
    # Neuroscience: High-order thalamic nuclei gate conscious perception (Science 2024)
    # Cerebellar forward model predicts upcoming sensory consequences (Neuron 2026)

    def _thalamic_gate(self, candidates: List[Tuple]) -> List[Tuple]:
        """Phase 16.1: Gate candidates — only the most salient pass through.
        
        Salience = signal_ratio * 0.4 + novelty * 0.3 + schema_relevance * 0.3
        Gate threshold: salience > 0.2
        """
        if not candidates:
            return []
        max_sig = max(c[0] for c in candidates) or 0.001
        gated = []
        for sig, tgt_lbl, edge, d in candidates:
            novelty = 1.0 if tgt_lbl.lower() not in self._visited_concepts else 0.3
            schema_rel = 1.2 if tgt_lbl.lower() in self._sentence_schema else 0.8
            salience = (sig / max_sig) * 0.4 + novelty * 0.3 + schema_rel * 0.3
            if salience > 0.2:
                gated.append((sig, tgt_lbl, edge, d))
        return gated

    def _cerebellar_predictor(self, cur_label: str) -> Dict[str, float]:
        """Phase 16.3: Predict next concepts from past traversal patterns.
        
        Uses n-gram model over _last_chain_hops.
        Returns dict of {predicted_label: probability}.
        """
        predictions = {}
        for hops_list in self._last_chain_hops[-10:]:
            for i, (f, t) in enumerate(hops_list):
                if f.lower() == cur_label.lower():
                    predictions[t.lower()] = predictions.get(t.lower(), 0) + 1
        if predictions:
            total = sum(predictions.values())
            return {k: v / total for k, v in predictions.items()}
        return {}

    def _get_cerebellar_depth(self, label: str) -> float:
        """Phase 16.5: Get expected number of hops from past patterns."""
        return self._cerebellar_depth.get(label.lower(), 0.0)

    def _update_cerebellar_ngram(self, hops: List[Tuple[str, str]]):
        """Update cerebellar n-gram model with a completed chain.
        
        Uses both the legacy _cerebellar_ngram dict and the new CerebellarNgram module.
        """
        # Legacy update (backward compatible)
        for i, (f, t) in enumerate(hops):
            fl, tl = f.lower(), t.lower()
            if fl not in self._cerebellar_ngram:
                self._cerebellar_ngram[fl] = {}
            self._cerebellar_ngram[fl][tl] = self._cerebellar_ngram[fl].get(tl, 0) + 1
            
            # Also learn depth
            remaining = len(hops) - i - 1
            current_depth = self._cerebellar_depth.get(fl, 0.0)
            self._cerebellar_depth[fl] = current_depth * 0.7 + remaining * 0.3
        
        # Phase C: Also learn into CerebellarNgram module
        if hasattr(self, 'cerebellar_ngram'):
            chain_labels = []
            for f, t in hops:
                chain_labels.extend([f, t])
            self.cerebellar_ngram.learn_chain(
                chain_labels=chain_labels,
                successful=True,
                chain_hops=hops,
            )


    # ── Phase 17: Meta-Learning ──
    # Neuroscience: Brain tracks own uncertainty, calibrates confidence (PMC 2024)
    # Epistemic value = expected information gain (Free Energy Principle)

    def _update_concept_confidence(self, label: str, prediction_error: float):
        """Phase 17.1: Update per-concept confidence.
        
        Initial: 0.3 for seed, 0.1 for learned.
        Increases by 0.05 on use, decreases by 0.1 on high PE.
        """
        cl = label.lower()
        current = self._concept_confidence.get(cl, 0.3 if cl in self._concept_labels else 0.1)
        if prediction_error < 0.2:
            current += 0.05
        elif prediction_error > 0.5:
            current -= 0.1
        self._concept_confidence[cl] = np.clip(current, 0.05, 0.99)

    def _compute_epistemic_value(self, label: str) -> float:
        """Phase 17.2: Compute epistemic value of exploring a concept.
        
        epistemic_value = uncertainty * expected_information_gain
        """
        conf = self._concept_confidence.get(label.lower(), 0.1)
        uncertainty = 1.0 - conf
        pe = self._mean_prediction_error
        info_gain = 1.0 - pe  # High PE = high potential information gain
        return uncertainty * info_gain

    def _calibrated_confidence(self, label: str) -> float:
        """Phase 17.3: Get calibrated confidence for a concept.
        
        Adjusts raw confidence by calibration error (subtract overconfidence).
        """
        raw = self._concept_confidence.get(label.lower(), 0.1)
        calibration_penalty = getattr(self, '_calibration_error', 0.0) * 0.2
        return max(0.05, raw - calibration_penalty)

    def _metacognitive_review(self):
        """Phase 17.4: Run metacognitive review every 5 turns.
        
        1. Check response diversity (same connector used 3+ times)
        2. Check concept coverage (same 5 concepts keep appearing)
        3. Check prediction error trend
        """
        if self.turn_count < 5:
            return
        
        issues = []
        
        # 1. Connector diversity
        if hasattr(self, '_response_context') and len(self._response_context) >= 3:
            last_3 = ''.join(c['response'] for c in self._response_context[-3:])
            for connector in ['connect', 'cause', 'but', 'like', 'then']:
                if last_3.count(connector) >= 3:
                    issues.append(f"overused connector '{connector}'")
        
        # 2. Concept coverage
        if hasattr(self, '_last_responses') and len(self._last_responses) >= 3:
            all_words = []
            for r in self._last_responses[-3:]:
                all_words.extend(r.lower().split())
            word_counts = {}
            for w in all_words:
                word_counts[w] = word_counts.get(w, 0) + 1
            # Check if same concept appears 5+ times
            for w, c in word_counts.items():
                if c >= 5 and w not in {'connect', 'but', 'like', 'cause', 'then', 'and', 'the'}:
                    issues.append(f"repetitive concept '{w}' ({c}x)")
        
        # 3. Prediction error trend
        if len(self._td_error_history) >= 5:
            recent_pe = np.mean(np.abs(self._td_error_history[-3:]))
            older_pe = np.mean(np.abs(self._td_error_history[:-3])) if len(self._td_error_history) > 3 else 0.3
            if recent_pe > older_pe * 1.2:
                issues.append(f"increasing PE trend (recent={recent_pe:.2f}, older={older_pe:.2f})")
        
        # Log findings
        if issues and hasattr(self, '_trace_enabled') and self._trace_enabled:
            print(f"  [meta]   review: {', '.join(issues)}")

    def _sleep_consolidate(self) -> Dict[str, int]:
        """Run a mini sleep cycle to strengthen useful patterns and weaken noise.

        Teenagers' brains consolidate learning during sleep! This mimics that
        by replaying recent topics through the graph and applying Hebbian
        strengthening to co-activated concepts.
        
        Returns:
            Dict with sleep metrics: edges_strengthened, edges_pruned, etc.
        """
        if len(self.graph.edges) < 3:
            return {"edges_strengthened": 0, "edges_pruned": 0, "episodic_consolidated": 0}

        edges_strengthened = 0
        edges_pruned = 0
        episodic_consolidated = 0

        # Phase 15.2: Episodic decay already runs per-turn in process_turn; skip double-decay here

        # Phase 9b: Prediction-error-driven sleep consolidation
        # Instead of blind +0.02 increments, use prediction error to guide updates.
        # Edges that accurately predict vector similarity get strengthened.
        # Edges with high prediction error get corrected (not just weakened).
        for topic in self._topic_list[-5:]:
            tids = self._concept_keywords.get(topic.lower(), [])
            if len(tids) < 2:
                continue
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    edge = self.graph.get_edge(tids[i], tids[j])
                    if edge:
                        src_node = self.graph.get_node(tids[i])
                        tgt_node = self.graph.get_node(tids[j])
                        if (src_node and src_node.vector is not None
                                and tgt_node and tgt_node.vector is not None):
                            error = self._update_edge_from_error(
                                edge, src_node.vector, tgt_node.vector, learning_rate=0.08)
                            edges_strengthened += 1

        # Phase 15.5: Consolidate frequently-used episodic edges to semantic store
        consolidated = self._consolidate_to_semantic()
        if consolidated > 0:
            episodic_consolidated = consolidated
            if self._trace_enabled:
                print(f"  [trace]   sleep: consolidated {consolidated} edges to semantic")

        # Phase 2: Synaptic pruning (neuroscience-inspired: prune weak, unused connections)
        edges_to_prune = []
        for (src, tgt), edge in self.graph.edges.items():
            if edge.weight < 0.08 and edge.confidence < 0.10:
                edges_to_prune.append((src, tgt))
        for src, tgt in edges_to_prune:
            self.graph.remove_edge(src, tgt)
        edges_pruned = len(edges_to_prune)

        # Phase 3: Identity consolidation
        self.identity.state.strength = min(1.0, self.identity.state.strength + 0.02)

        # Phase 4: Meaning consolidation
        self.meaning.compute_meaning(
            episode=self.turn_count,
            pre_dissonance=self._free_energy + 0.02,
            post_dissonance=self._free_energy,
            pre_identity=self.identity.state.strength - 0.01,
            post_identity=self.identity.state.strength,
            predictive_gain=0.2,
            effort=0.1,
        )

        # Phase 5: Emotion regulation (calm down after sleep)
        self.emotion.update(
            stimulus_valence=0.0,
            stimulus_arousal=-0.1,
            stimulus_dominance=0.05,
            uncertainty=0.0,
        )

        self.sleep_engine.accumulate_pressure(-0.15)  # Reduce sleep pressure
        self.sleep_cycles_completed += 1

        # Phase 7.6: Sleep-replay impossible queries
        replayed = 0
        for iq in self._impossible_queries:
            if iq.resolved:
                continue
            subj_nids = self._concept_keywords.get(iq.subject.lower(), [])
            if subj_nids:
                subj_node = self.graph.get_node(subj_nids[0])
                if subj_node and subj_node.vector is not None:
                    # Try to auto-wire to all existing concepts with high similarity
                    new_edges = 0
                    for other_nid, other_node in self.graph.nodes.items():
                        if other_nid == subj_nids[0]:
                            continue
                        if other_node.vector is not None and other_node.label:
                            sim = float(np.dot(subj_node.vector, other_node.vector))
                            if sim > 0.5 and self.graph.get_edge(subj_nids[0], other_nid) is None:
                                weight = min(0.6, sim * 0.6)
                                ne = self.graph.add_edge(subj_nids[0], other_nid, weight=weight,
                                                     relation_type="semantic")
                                ne.confidence = 0.001  # dormant
                                self._dormant_edges.add((subj_nids[0], other_nid))
                                # Run prediction error correction for accuracy
                                if other_node.vector is not None:
                                    new_edge = self.graph.get_edge(subj_nids[0], other_nid)
                                    if new_edge:
                                        self._update_edge_from_error(
                                            new_edge, subj_node.vector, other_node.vector,
                                            learning_rate=0.08)

        if replayed > 0 and getattr(self, "_trace_enabled", False):
            print(f"  [trace]   sleep replay: resolved {replayed} impossible queries")

        # Fix 3: Belief reconciliation — resolve contradictions by recency x confidence
        if getattr(self, "use_beliefs", True) and self.belief_store.beliefs:
            before = len(self.belief_store.contradictions)
            resolved = self.belief_store.reconcile()
            after = len(self.belief_store.contradictions)
            if resolved:
                # Prune contradictory graph edges that conflict with resolved beliefs
                for (subj, pred), (value, conf, _) in resolved.items():
                    subj_nids = self._concept_keywords.get(subj.lower(), [])
                    for nid in subj_nids:
                        for tid, edge in list(self.graph.get_outgoing(nid)):
                            tgt_node = self.graph.get_node(tid)
                            if tgt_node and tgt_node.label and edge.relation_type == pred and tgt_node.label.lower() != value.lower():
                                edge.weight = max(0.01, edge.weight * 0.3)
                                edge.confidence = max(0.01, edge.confidence * 0.3)
                                edges_pruned += 1
                if getattr(self, "_trace_enabled", False):
                    print(f"  [trace]   sleep belief: reconciled {len(resolved)} contradictions (pruned {edges_pruned} conflicting edges)")

        # Solution #5: Correct mis-typed relations during sleep
        if self.turn_count > 0 and self.turn_count % 50 == 0:
            migrated = self._correct_relation_types()
            if migrated > 0:
                if getattr(self, "_trace_enabled", False):
                    print(f"  [trace]   sleep relation: reclassified {migrated} edges")

        # Update sleep metrics
        self._sleep_metrics["edges_strengthened"] += edges_strengthened
        self._sleep_metrics["edges_pruned"] += edges_pruned
        self._sleep_metrics["episodic_consolidated"] += episodic_consolidated
        self._sleep_metrics["total_sleep_cycles"] += 1
        self._sleep_metrics["last_sleep_turn"] = self.turn_count
        self._sleep_metrics["last_sleep_metrics"] = {
            "edges_strengthened": edges_strengthened,
            "edges_pruned": edges_pruned,
            "episodic_consolidated": episodic_consolidated,
            "sleep_cycle": self._sleep_metrics["total_sleep_cycles"],
            "turn": self.turn_count,
        }

        return {
            "edges_strengthened": edges_strengthened,
            "edges_pruned": edges_pruned,
            "episodic_consolidated": episodic_consolidated,
        }

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
            self._bg_learning_thread.join(timeout=5)
        if self._trace_enabled:
            print(f'  [bg] background learning stopped (performed {self._bg_search_count} searches)')

    def _bg_learn_loop(self):
        """Background learning thread: processes pending queue and related searches when idle."""
        while self._bg_learning_active:
            # Wake periodically (30s timeout) — event is set when user goes idle
            self._bg_idle_event.wait(timeout=30)
            self._bg_idle_event.clear()
            if not self._bg_learning_active:
                break

            # Periodic curiosity cycle reset for continuous exploration in background mode
            self._bg_idle_search_count += 1
            if self._bg_idle_search_count % 10 == 0:
                self._curiosity_cycles_this_session = 0
                if self._trace_enabled:
                    print('  [bg] curiosity budget refreshed for new exploration cycle')

            # Process pending queue items in background
            queries_to_process = []
            with self._bg_lock:
                queries_to_process = list(self._bg_learning_queue)
                self._bg_learning_queue.clear()
            # Also process deferred learning queue
            with self._bg_lock:
                deferred = list(self._pending_learning_queue)
                self._pending_learning_queue.clear()
            all_queries = queries_to_process + deferred
            # Phase 18: Autonomously select curiosity topics when queue runs low
            # Run curiosity selection if queue is small (<=3) or periodically every idle period
            queue_size = len(self._bg_learning_queue) + len(self._pending_learning_queue)
            should_run_curiosity = (not all_queries and self._curiosity_drive_enabled) or \
                                   (queue_size <= 3 and self._curiosity_drive_enabled and \
                         self._bg_idle_search_count % 2 == 0)  # Every other idle period
            
            if should_run_curiosity:
                if self._curiosity_cycles_this_session < 5:  # Increased from 3
                    self._curiosity_cycles_this_session += 1
                    self._bg_idle_search_count = 0
                    if self._trace_enabled:
                        reason = "queue empty" if not all_queries else "queue low"
                        print(f'  [bg] idle cycle {self._curiosity_cycles_this_session}/5 ({reason}) - selecting curiosity topics...')
                    self._auto_select_curiosity_topics(max_topics=2)
                    with self._bg_lock:
                        all_queries = list(self._bg_learning_queue)
                        self._bg_learning_queue.clear()
                elif self._trace_enabled:
                    if not all_queries:
                        print('  [bg] idle: waiting for more user input (curiosity budget used)')
                    else:
                        print(f'  [bg] idle: queue has {queue_size} items, processing...')
            try:
                for query in all_queries:
                    if not self._bg_learning_active:
                        break
                    self._bg_idle_search_count += 1
                    if self._trace_enabled:
                        print(f'  [bg] ({self._bg_idle_search_count}) researching: {query}')
                    self._bg_multi_search(query)
                # Save periodically after learning
                if all_queries:
                    try:
                        path = self.save()
                        if self._trace_enabled:
                            print(f'  [bg] saved to {os.path.basename(path)}')
                    except Exception:
                        pass
            except Exception:
                # Prevent background thread from dying silently
                pass

    def _bg_multi_search(self, query: str):
        """Perform multiple related searches for a single query.
        1. Search the original query
        2. Extract related terms from results
        3. Search top 2-3 related terms
        All done synchronously within the background thread."""
        if not self._bg_learning_active:
            return
        # Step 1: Search the original query
        try:
            result_summary = self.learn_from_web(query)
            self._bg_search_count += 1
            if self._trace_enabled:
                print(f'  [bg]   + {result_summary}')
        except Exception:
            if self._trace_enabled:
                print(f'  [bg]   ! failed to research: {query}')
            return
        # Step 2: Extract related search terms from the query itself
        related = self._extract_related_queries(query)
        # Step 3: Search top related terms
        for related_query in related[:self._bg_multi_search_max]:
            if not self._bg_learning_active:
                break
            time.sleep(1)  # polite delay between searches
            try:
                result_summary = self.learn_from_web(related_query)
                self._bg_search_count += 1
                if self._trace_enabled:
                    print(f'  [bg]   + {result_summary}')
            except Exception:
                continue

    def _generate_curiosity_query(self, concept: str, source_type: str = "default",
                                antonym: str = "") -> str:
        """Generate a targeted web search query based on curiosity source type.

        Templates:
        - default: concept name (already handled by _bg_multi_search)
        - high_pe: concept + 'explained' or 'mechanism'
        - contradiction: 'concept vs antonym' or 'concept controversy'
        - dormant_edge: concept + 'related topics' or 'what is'
        """
        if source_type == "contradiction" and antonym:
            # Target: expose why the contradiction exists
            return f"{concept} vs {antonym} debate"
        elif source_type == "high_pe":
            # Target: clarify the confusing topic
            return f"{concept} explained simply"
        elif source_type == "dormant_edge":
            # Target: discover what connects to this unexplored concept
            return f"{concept} related concepts what is"
        else:
            return concept

    def _extract_related_queries(self, query: str) -> List[str]:
        """Extract related search queries from the original query.
        Uses graph knowledge and word associations to generate related searches."""
        related = []
        words = [w for w in query.lower().split() if w not in STOP_WORDS and len(w) >= 3]
        # Strategy 1: Individual word deep-dives
        for w in words:
            if w not in self._concept_labels and w not in related:
                related.append(w)
        # Strategy 2: Graph neighbors - what concepts are linked to the query concepts?
        for w in words:
            nids = self._concept_keywords.get(w, [])
            for nid in nids:
                for tid, edge in self.graph.get_outgoing(nid):
                    tn = self.graph.nodes.get(tid)
                    if tn and tn.label:
                        neighbor = tn.label.lower()
                        if neighbor not in STOP_WORDS and len(neighbor) >= 3 and neighbor not in related:
                            related.append(f'{w} {neighbor}')
                for src, edge in self.graph.get_incoming(nid):
                    sn = self.graph.nodes.get(src)
                    if sn and sn.label:
                        neighbor = sn.label.lower()
                        if neighbor not in STOP_WORDS and len(neighbor) >= 3 and neighbor not in related:
                            related.append(f'{neighbor} {w}')
        # Strategy 3: Compound queries from multi-word topics
        if len(words) >= 2:
            related.append(f'{words[0]} explained')
            related.append(f'how does {words[0]} work')
        return related[:self._bg_multi_search_max + 2]

    def queue_background_search(self, query: str):
        """Queue a query for background research. Thread-safe."""
        with self._bg_lock:
            if query not in self._bg_learning_queue:
                self._bg_learning_queue.append(query)
        self._bg_idle_event.set()  # Wake the background thread

    def notify_user_active(self):
        """Signal that the user is actively chatting (pause background learning)."""
        self._bg_idle_event.clear()  # thread will block on wait

    def notify_user_idle(self):
        """Signal that the user is idle (resume background learning)."""
        self._bg_idle_event.set()  # thread will wake up and process
        self._curiosity_cycles_this_session = 0  # fresh curiosity budget per idle period

    # --- Phase 18: Curiosity Drive - Autonomous Topic Selection ---
    # Based on: Berlyne epistemic curiosity, Loewenstein information-gap theory,
    # Friston Active Inference (prediction error minimization), Oudeyer learning progress.

    def _update_concept_learning_progress(self):
        """Aggregate edge-level prediction error to concept-level learning progress.

        Uses EWMA with rolling window to track PE delta per concept.
        Positive delta = genuine learning (PE dropping).
        Flat/negative delta = stuck (curiosity should spike).
        """
        if not self._curiosity_drive_enabled:
            return

        # Build concept -> incident edges map
        concept_edges: Dict[str, List[float]] = {}
        for (src, tgt), edge in self.graph.edges.items():
            src_node = self.graph.get_node(src)
            tgt_node = self.graph.get_node(tgt)
            pe = getattr(edge, 'prediction_free_energy', 0.0)
            if src_node and src_node.label:
                concept_edges.setdefault(src_node.label.lower(), []).append(pe)
            if tgt_node and tgt_node.label:
                concept_edges.setdefault(tgt_node.label.lower(), []).append(pe)

        alpha = 0.3  # EWMA smoothing factor
        for label, pe_list in concept_edges.items():
            if not pe_list:
                continue
            mean_pe = sum(pe_list) / len(pe_list)
            prev = self._concept_learning_progress.get(label, mean_pe)
            # Compute delta: positive = PE dropping = genuine learning
            # Negative = PE rising = getting more confused = curiosity spike
            delta = prev - mean_pe  # positive = improvement
            # Store smoothed PE value for next comparison
            smoothed = (1 - alpha) * prev + alpha * mean_pe
            # Store delta as learning progress (positive = learning, negative = stuck/confused)
            self._concept_learning_progress[label] = smoothed
            # Also track the delta separately for curiosity scoring

            self._concept_pe_delta[label] = delta

    def _compute_curiosity_urgency(self) -> float:
        """Compute overall curiosity urgency from multiple signals.

        Returns (0-1) urgency score. High urgency = strong drive to explore."""
        if not self._curiosity_drive_enabled:
            self._curiosity_urgency = 0.0
            return 0.0

        urgency = 0.0

        # 1. Prediction error urgency (Active Inference: surprise drives exploration)
        if self._prediction_error_count > 5:
            pe_factor = min(0.4, self._mean_prediction_error * 2.0)
            urgency += pe_factor

        # 2. Information gap urgency (unresolved impossible queries)
        unresolved = sum(1 for iq in self._impossible_queries if not iq.resolved)
        if unresolved > 0:
            gap_urgency = min(0.5, unresolved * 0.05)
            urgency += gap_urgency

        # 3. Arousal modulation (multiplicative, coupling dopamine and norepinephrine)
        # Neuroscience: arousal amplifies exploration non-linearly
        # Even at minimal arousal (0.1), 55% of curiosity remains
        # At max arousal (1.0), curiosity is at 100%
        arousal_gain = 0.5 + 0.5 * self.emotion.state.arousal
        urgency *= arousal_gain

        # 4. Identity uncertainty (low identity = more exploration)
        identity_curiosity = (1.0 - self.identity.state.strength) * 0.15
        urgency += identity_curiosity

        # 5. Low-confidence concepts in the graph
        low_conf_count = sum(1 for c in self._concept_confidence.values() if c < 0.3)
        if low_conf_count > 0:
            urgency += min(0.2, low_conf_count * 0.01)

        self._curiosity_urgency = min(1.0, urgency)
        return self._curiosity_urgency

    def _get_concept_confidence(self, concept_label: str) -> float:
        """Get the confidence level of a concept from its incident edges."""
        nids = self._concept_keywords.get(concept_label.lower(), [])
        if not nids:
            return 0.0
        confidences = []
        for nid in nids:
            for tid, edge in self.graph.get_outgoing(nid):
                confidences.append(getattr(edge, 'confidence', 0.0))
            for src, edge in self.graph.get_incoming(nid):
                confidences.append(getattr(edge, 'confidence', 0.0))
        if not confidences:
            return 0.0
        return sum(confidences) / len(confidences)

    # Phase C: Epistemic hedging - produce hedges based on concept confidence and prediction error



        for nid, node in self.graph.nodes.items():
            if not node.label:
                continue
            total = 0
            dormant = 0
            for tid, edge in self.graph.get_outgoing(nid):
                total += 1
                if (nid, tid) in self._dormant_edges:
                    dormant += 1
            for src, edge in self.graph.get_incoming(nid):
                if src == nid:
                    continue
                pair = (src, nid)
                total += 1
                if pair in self._dormant_edges:
                    dormant += 1
            if total > 0:
                ratios[node.label.lower()] = dormant / total
        return ratios

    def _compute_dormant_edge_ratio(self) -> Dict[str, float]:
        """Compute dormant edge ratio per concept.

        Returns dict mapping concept label -> dormant_ratio (0-1).
        High ratio = most edges are dormant = high novelty.
        """
        ratios: Dict[str, float] = {}
        for nid, node in self.graph.nodes.items():
            if not node.label:
                continue
            total = 0
            dormant = 0
            for tid, edge in self.graph.get_outgoing(nid):
                total += 1
                if (nid, tid) in self._dormant_edges:
                    dormant += 1
            for src, edge in self.graph.get_incoming(nid):
                if src == nid:
                    continue
                pair = (src, nid)
                total += 1
                if pair in self._dormant_edges:
                    dormant += 1
            if total > 0:
                ratios[node.label.lower()] = dormant / total
        return ratios


    def _auto_select_curiosity_topics(self, max_topics: int = 3) -> List[str]:
        """Autonomously select topics for background research based on curiosity signals.

        Priority order:
        1. Unresolved impossible queries (direct knowledge gaps) - weight 5x
        2. High prediction error concepts (surprising) - weight 3x
        3. Contradiction pairs (cognitive dissonance) - weight 3x
        4. Least-visited concepts (novelty) - weight 1x
        5. Random graph walk from high-degree hubs (serendipity) - weight 1x

        Returns list of topic strings to research."""
        if not self._curiosity_drive_enabled or not self._bg_learning_active:
            return []

        candidates: List[Tuple[str, float]] = []
        seen_topics = set()

        # Source 1: Unresolved impossible queries (weight 5x)
        for iq in self._impossible_queries:
            if iq.resolved:
                continue
            topic = iq.subject.lower().strip()
            if topic and topic not in seen_topics and len(topic) >= 3:
                candidates.append((topic, 5.0))
                seen_topics.add(topic)

        # Source 2: High prediction error concepts (weight 3x, targeted query)
        high_pe_nodes = []
        for nid, node in self.graph.nodes.items():
            if node.label and node.prediction_free_energy > 0.3:
                label = node.label.lower()
                if label not in seen_topics and len(label) >= 3:
                    high_pe_nodes.append((label, node.prediction_free_energy))
        high_pe_nodes.sort(key=lambda x: -x[1])
        for label, pe in high_pe_nodes[:5]:
            candidates.append((label, 3.0 * min(1.0, pe)))
            seen_topics.add(label)

        # Source 3: Contradiction pairs with confidence-mismatch scoring
        # Higher confidence mismatch = more uncertain = more curious to resolve
        for concept, antonyms in self._contradiction_map.items():
            # Look up edge confidence for both sides of contradiction
            concept_conf = self._get_concept_confidence(concept)
            for ant in antonyms:
                ant_conf = self._get_concept_confidence(ant)
                # Mismatch score: how uncertain are we which side is correct?
                mismatch = abs(concept_conf - ant_conf) * 2.0  # 0 if equal, up to 2.0
                # Base weight: higher for high-mismatch (ambiguity) AND low-mismatch (both confident but opposing)
                # This captures both "don't know which is right" and "both sides claim truth"
                conf_level = (concept_conf + ant_conf) / 2.0
                if conf_level > 0.6:
                    # Both confident but contradictory - interesting!
                    weight = 3.0 + mismatch * 0.5
                else:
                    # At least one side uncertain - knowledge gap
                    weight = 2.0 + mismatch * 0.75
                if concept not in seen_topics and len(concept) >= 3:
                    candidates.append((concept, max(2.0, weight)))
                    seen_topics.add(concept)
                if ant not in seen_topics and len(ant) >= 3:
                    candidates.append((ant, max(1.5, weight * 0.8)))
                    seen_topics.add(ant)

        # Source 4: Novel concepts via dormant edge ratio (high dormant = unexplored)
        all_labels = set()
        dormant_ratios = self._compute_dormant_edge_ratio()
        for nid, node in self.graph.nodes.items():
            if node.label:
                all_labels.add(node.label.lower())
        unvisited = [l for l in all_labels if l not in seen_topics and len(l) >= 3]
        # Sort by dormant ratio descending (most unexplored first)
        unvisited.sort(key=lambda l: dormant_ratios.get(l, 0), reverse=True)
        for label in unvisited[:3]:
            dr = dormant_ratios.get(label, 0)
            novelty_weight = 0.5 + dr * 0.5  # range: 0.5 (fully explored) to 1.0 (fully dormant)
            candidates.append((label, novelty_weight))
            seen_topics.add(label)

        # Generate targeted queries for top contradiction candidates
        # Skip already-explored pairs to prevent "good vs bad debate" repetition
        contradiction_queries = []
        for topic, _ in candidates:
            if topic in self._contradiction_map:
                for ant in self._contradiction_map[topic]:
                    pair = (topic, ant) if topic < ant else (ant, topic)
                    if pair in self._explored_contradictions:
                        continue
                    targeted = self._generate_curiosity_query(
                        topic, source_type="contradiction", antonym=ant)
                    if targeted != topic:
                        contradiction_queries.append((targeted, pair))
        # Queue contradiction-specific queries directly
        for cq, pair in contradiction_queries[:2]:
            if self._bg_learning_active and cq not in self._bg_learning_queue:
                self.queue_background_search(cq)
                self._explored_contradictions.add(pair)

        # Source 5: Random graph walk from high-degree hubs (serendipity)
        if len(self.graph.nodes) > 0:
            degree_counts: Dict[int, int] = {}
            for src, tgt in self.graph.edges:
                degree_counts[src] = degree_counts.get(src, 0) + 1
                degree_counts[tgt] = degree_counts.get(tgt, 0) + 1
            if degree_counts:
                top_hub_id = max(degree_counts, key=degree_counts.get)
                hub_node = self.graph.get_node(top_hub_id)
                if hub_node and hub_node.label:
                    current_id = top_hub_id
                    for _ in range(2):
                        edges = list(self.graph.get_outgoing(current_id))
                        if not edges:
                            break
                        next_id = self.rng.choice([e[0] for e in edges]) if len(edges) > 1 else edges[0][0]
                        next_node = self.graph.get_node(next_id)
                        if next_node and next_node.label:
                            lbl = next_node.label.lower()
                            if lbl not in seen_topics and len(lbl) >= 3:
                                candidates.append((lbl, 0.8))
                                seen_topics.add(lbl)
                            current_id = next_id

        # Phase 18b: Boost candidates matching user's recent interests (priming)
        if self._user_query_topics:
            for i in range(len(candidates)):
                topic, weight = candidates[i]
                # Check if topic or any part matches user's recent queries
                for uq in self._user_query_topics[-3:]:  # last 3 topics
                    if uq in topic or topic in uq:
                        candidates[i] = (topic, min(2.0, weight * 1.5))  # 1.5x boost (capped at 2.0)
                        break
                    # Also check if the topic is connected to user's interests via graph
                    uq_nids = self._concept_keywords.get(uq, [])
                    topic_nids = self._concept_keywords.get(topic, [])
                    if uq_nids and topic_nids:
                        for un in uq_nids:
                            for tn in topic_nids:
                                if self.graph.get_edge(un, tn) or self.graph.get_edge(tn, un):
                                    candidates[i] = (topic, min(1.5, weight * 1.3))  # 1.3x boost (capped at 1.5)
                                    break

        # Initialize diversity tracking
        if not hasattr(self, '_recent_curiosity_selections'):
            self._recent_curiosity_selections = []  # list of (topic, turn_count)
        if not hasattr(self, '_curiosity_selection_cooldown'):
            self._curiosity_selection_cooldown = 10  # suppress selected topics for N turns (increased from 5)
        if not hasattr(self, '_bg_learning_cycles'):
            self._bg_learning_cycles = 0
        # Clear old selections from different turn domains (chat vs background)
        # This prevents old chat turns from permanently blocking background selections
        current_domain_turn = self._bg_learning_cycles if self._bg_learning_cycles > 0 else self.turn_count
        if self._recent_curiosity_selections:
            # Check if we're in a different turn domain than existing entries
            latest_turn = self._recent_curiosity_selections[-1][1]
            # If the latest turn is in the "other" domain (e.g., chat turns 40+ vs background 0+),
            # clear the history to allow fresh selections
            domain_threshold = 20  # turns threshold to consider different domains
            if latest_turn > domain_threshold and current_domain_turn <= domain_threshold:
                self._recent_curiosity_selections = []

        # ----- DIVERSITY: Penalize recently selected topics -----
        # Use background cycle count if turn_count isn't advancing
        current_turn = self._bg_learning_cycles if self._bg_learning_cycles > 0 else self.turn_count
        self._bg_learning_cycles += 1
        # Clean old entries
        self._recent_curiosity_selections = [
            (t, turn) for t, turn in self._recent_curiosity_selections
            if current_turn - turn < self._curiosity_selection_cooldown
        ]
        recently_selected = {t for t, _ in self._recent_curiosity_selections}
        
        # Penalize recently selected topics
        penalized_candidates = []
        for topic, weight in candidates:
            if topic in recently_selected:
                # Decay weight exponentially based on recency
                for t, turn in self._recent_curiosity_selections:
                    if t == topic:
                        recency = current_turn - turn
                        # Stronger penalty: 0.2^recency instead of 0.3^recency
                        penalty = 0.2 ** recency  # 0.2, 0.04, 0.008...
                        weight = weight * penalty
                        break
            penalized_candidates.append((topic, weight))
        candidates = penalized_candidates

        # EPSILON-GREEDY: 35% chance to pick random unvisited concept (pure exploration)
        epsilon = 0.35
        unvisited_labels = [l for l in all_labels if l not in seen_topics and len(l) >= 3]
        if unvisited_labels and self.rng.random() < epsilon:
            random_topic = self.rng.choice(unvisited_labels)
            candidates.append((random_topic, 0.5))  # low weight but could be picked
            if self._trace_enabled:
                print(f"  [curiosity] epsilon-exploration: {random_topic}")

        candidates.sort(key=lambda x: -x[1])
        selected = [topic for topic, _ in candidates[:max_topics]]

        # Track selections
        for topic in selected:
            self._recent_curiosity_selections.append((topic, current_turn))

        # For sync mode (max_topics=0), still track the top candidate for diversity
        if max_topics == 0 and candidates:
            top_topic = candidates[0][0]
            self._recent_curiosity_selections.append((top_topic, current_turn))

        if selected and self._trace_enabled:
            weights_str = ", ".join(f"{t}({w:.1f})" for t, w in candidates[:max_topics])
            print(f"  [curiosity] auto-selected: {weights_str} (urgency={self._curiosity_urgency:.2f})")

        self._last_auto_learn_turn = self.turn_count
        return selected

    def save(self) -> str:
        """Save full cognitive state to disk. Returns path to save file."""
        with self._vocab_lock, self._graph_lock:
            _graph_snapshot = self.graph
            _decoder_w2i = dict(self._decoder_word_to_idx)
            _decoder_i2w = dict(self._decoder_idx_to_word)
            _decoder_w2e = dict(self._decoder_word_to_embed)
            _ck_snapshot = dict(self._concept_keywords)
            _cl_snapshot = set(self._concept_labels)
            _vc_snapshot = set(self._visited_concepts)
            _af_snapshot = dict(self._activation_fatigue)
            _rt_snapshot = list(self._recent_traversals)
            _rtm_snapshot = dict(self._recent_traversal_map)
            _cv_snapshot = dict(self._concept_vad)
            _td_snapshot = list(self._td_error_history[-50:])
            _cc_snapshot = dict(self._concept_confidence)
        state = {
            'graph': _graph_snapshot,
            'concept_keywords': _ck_snapshot,
            'turn_count': self.turn_count,
            'topic_list': list(self._topic_list),
            'topic_store': dict(self._topic_store),
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
            'contradiction_map': dict(self._contradiction_map),
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
            'concept_visit_count': dict(self._concept_visit_count),
            'concept_learning_progress': dict(self._concept_learning_progress),
            'concept_pe_delta': dict(self._concept_pe_delta),
            'curiosity_topics_queue': list(self._curiosity_topics_queue),
            'last_auto_learn_turn': self._last_auto_learn_turn,
            'curiosity_urgency': self._curiosity_urgency,
            'user_query_topics': list(self._user_query_topics),
            'user_last_topic': self._user_last_topic,
            'concept_sources': {k: list(v) for k, v in self._concept_sources.items()},
            'explored_contradictions': [list(p) for p in list(self._explored_contradictions)],
            # Dual stores
            'episodic_edges': dict(self._episodic_edges),
            'semantic_edges': dict(self._semantic_edges),
            # Phase 10-17 state
            'sentence_schema': dict(self._sentence_schema),
            'mean_sentence_pe': self._mean_sentence_pe,
            'dopamine_tone': self._dopamine_tone,
            'td_error_history': _td_snapshot,
            'concept_confidence': _cc_snapshot,
            'cerebellar_ngram': dict(self._cerebellar_ngram),
            'cerebellar_ngram_state': self.cerebellar_ngram.get_state() if hasattr(self, 'cerebellar_ngram') else {},
            'cerebellar_depth': dict(self._cerebellar_depth),
            'concept_pos': dict(self._concept_pos),
            'concept_labels': list(self._concept_labels),
            'visited_concepts': list(self._visited_concepts),
            'activation_fatigue': _af_snapshot,
            'recent_traversals': _rt_snapshot,
            'recent_traversal_map': _rtm_snapshot,
            'cognitive_state': self._cognitive_state,
            'state_duration': self._state_duration,
            'prefrontal_buffer': list(self._prefrontal_buffer),
            'mean_prediction_error': self._mean_prediction_error,
            'prediction_error_count': self._prediction_error_count,
            # Neural decoder
            'decoder_state_dict': self.neural_decoder.state_dict() if self.neural_decoder is not None else None,
            'decoder_training_count': self._decoder_training_count,
            'decoder_web_training_count': self._decoder_web_training_count,
            # Curiosity diversity state
            'bg_learning_cycles': getattr(self, '_bg_learning_cycles', 0),
            'recent_curiosity_selections': list(getattr(self, '_recent_curiosity_selections', [])),
            'curiosity_selection_cooldown': getattr(self, '_curiosity_selection_cooldown', 5),
        }
        try:
            # Phase 6.1: Checkpoint rotation — save every 25 turns
            if self.turn_count > 0 and self.turn_count % 25 == 0:
                checkpoint_path = self._save_path.replace('.pkl', f'_{self.turn_count}.pkl')
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(state, f)
                # Keep last 3 checkpoints, remove older ones
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
            self.graph = state['graph']
            self._concept_keywords = state['concept_keywords']
            self.turn_count = state['turn_count']
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

            # Restore teen state (optional — may not exist in old saves)
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
                loaded_user_model = upgraded
            # Ensure P1 ToM fields exist (backward-compatible migration)
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
            self._concept_pos = state.get('concept_pos', {})
            self._concept_labels = set(state.get('concept_labels', []))

            # Stash decoder state dict to be loaded after _build_decoder_vocab() 
            # (neural decoder doesn't exist yet — created later in __init__)
            decoder_sd = state.get('decoder_state_dict', None)
            self._saved_decoder_state = decoder_sd if decoder_sd is not None else {}
            self._decoder_training_count = state.get('decoder_training_count', 0)
            self._decoder_web_training_count = state.get('decoder_web_training_count', 0)

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
            self._concept_pos = state.get('concept_pos', {})
            self._concept_labels = set(state.get('concept_labels', []))

            # Stash decoder state dict to be loaded after _build_decoder_vocab()
            # (neural decoder doesn't exist yet — created later in __init__)
            decoder_sd = state.get('decoder_state_dict', None)
            self._saved_decoder_state = decoder_sd if decoder_sd is not None else {}
            self._decoder_training_count = state.get('decoder_training_count', 0)
            self._decoder_web_training_count = state.get('decoder_web_training_count', 0)

            # Restart background learning
            self.start_background_learning()

            # Rebuild POS tags if missing (old saves didn't include concept_pos)
            if not self._concept_pos:
                self._build_concept_pos()
                # Re-seed dependent components
                if hasattr(self, 'cerebellar_ngram'):
                    self.cerebellar_ngram.seed_from_pos(self._concept_pos)
                if hasattr(self, 'syntactic_assembly'):
                    self.syntactic_assembly.seed_from_pos(self._concept_pos)
            
            return True
        except Exception as e:
            print(f"  [Load error] {e}")
            return False
# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Pure natural language chat, no commands
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RAVANA - teenage mind on the web")
    parser.add_argument("--dim", type=int, default=64, help="Graph dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--reset", action="store_true", help="Delete saved weights and start fresh")
    parser.add_argument("--chat", type=str, default=None,
        help='Send queries in batch mode. Use | to separate multiple queries. '
             'Outputs Q: and A: lines for easy parsing. E.g.: --chat "hi|what is trust|bye"')
    parser.add_argument("--strategy", action="store_true", help="Include strategy name in --chat output")
    parser.add_argument("--trace", action="store_true", help="Print edge-level chain traces")
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD emotion modulation")
    parser.add_argument("--no-rlm", action="store_true", help="Disable RLMv2 triple verification")
    parser.add_argument("--no-beliefs", action="store_true", help="Disable belief store")
    parser.add_argument("--no-curiosity", action="store_true", help="Disable autonomous curiosity-driven learning")
    parser.add_argument("--mode", type=str, default="stochastic", choices=["stochastic", "deterministic", "exploratory"],
                        help="Reasoning mode: stochastic (default), deterministic (reproducible), exploratory (high-temp)")

    parser.add_argument("--user", type=str, default=None,
                        help="User name for multi-user isolation (creates user-specific save files)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Custom data directory for weights and GloVe cache")
    parser.add_argument("--export-graph", type=str, default=None,
                        help="Export graph to JSON file (all concepts + edges)")
    parser.add_argument("--import-graph", type=str, default=None,
                        help="Import graph from JSON file (merge into existing)")
    parser.add_argument("--stats", action="store_true",
                        help="Print graph statistics")
    parser.add_argument("--concept", type=str, default=None,
                        help="Show what RAVANA knows about a concept")
    args = parser.parse_args()

    # Handle --reset
    reset_suffix = args.user or ""
    save_path = os.path.join(_proj_root, "data", f"ravana_weights{reset_suffix}.pkl")
    if args.reset:
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f"  [Reset] Deleted {os.path.basename(save_path)}, starting fresh!")
        else:
            print(f"  [Reset] No saved weights found, starting fresh!")

    data_dir = args.data_dir
    user_suffix = args.user or ""
    engine = CognitiveChatEngine(dim=args.dim, seed=args.seed, baby_mode=True, data_dir=data_dir, user_suffix=user_suffix)
    engine.start_background_learning()
    if args.trace:
        engine._trace_enabled = True
    if args.no_vad:
        engine.use_vad = False
        print("  [Config] VAD modulation disabled")
    if args.no_rlm:
        engine.use_rlm = False
        print("  [Config] RLMv2 triple verification disabled")
    if args.no_beliefs:
        engine.use_beliefs = False
        print("  [Config] Belief store disabled")
    if args.no_curiosity:
        engine._curiosity_drive_enabled = False
        print('  [Curiosity] Autonomous learning disabled')

    # Solution #2: Apply reasoning mode
    if args.mode != "stochastic":
        engine.reasoning_mode = args.mode
        print(f"  [Mode] Reasoning mode set to '{args.mode}'")


    # ── BATCH MODE (--chat) ──
    # ── Phase 6 CLI Actions ──
    if args.export_graph:
        try:
            import json
            g = engine.graph
            data = {
                "nodes": [{"id": n.id, "label": n.label} for n in g.nodes.values()],
                "edges": [{"source": src, "target": tgt,
                           "relation": e.relation_type, "weight": e.weight}
                          for (src, tgt), e in g.edges.items()],
            }
            with open(args.export_graph, "w") as f:
                json.dump(data, f, indent=2)
            ec = len(data["edges"])
            nc = len(data["nodes"])
            print(f"  [Export] Exported {nc} nodes + {ec} edges to {args.export_graph}")
        except Exception as e:
            print(f"  [Export] Failed: {e}")
        engine.save()
        return
    if args.import_graph:
        try:
            import json
            with open(args.import_graph, "r") as f:
                data = json.load(f)
            g = engine.graph
            count = 0
            for node_data in data.get("nodes", []):
                nid = node_data.get("id")
                label = node_data.get("label", "")
                if label:
                    # Use GloVe vector if available, otherwise let graph create random vector
                    vec = engine._glove_vector(label) if engine._glove_vecs is not None else None
                    added_node = g.add_node(vector=vec, label=label)
                    count += 1
            for edge_data in data.get("edges", []):
                src = edge_data.get("source")
                tgt = edge_data.get("target")
                rel = edge_data.get("relation", "related")
                w = edge_data.get("weight", 0.5)
                if src is not None and tgt is not None:
                    g.add_edge(src, tgt, relation_type=rel, weight=w)
            print(f"  [Import] Imported {len(data.get('nodes', []))} nodes + {len(data.get('edges', []))} edges")
        except Exception as e:
            print(f"  [Import] Failed: {e}")
        engine.save()
        return
    if args.stats:
        print(f"  [Stats] Graph has {len(engine.graph.nodes)} nodes and {len(engine.graph.edges)} edges")
        print(f"  [Stats] Turn count: {engine.turn_count}")
        return
    if args.concept:
        nids = engine._concept_keywords.get(args.concept.lower(), [])
        if nids:
            node = engine.graph.get_node(nids[0])
            if node:
                outgoing = engine.graph.get_outgoing(nids[0])
                print(f"  [Concept] '{args.concept}': {len(outgoing)} edges, vector dim={len(node.vector) if node.vector is not None else 0}")
                for tgt_node, e in outgoing[:10]:
                    tgt_label = engine.graph.get_node(tgt_node).label if engine.graph.get_node(tgt_node) else "?"
                    print(f"    -> {tgt_label} [{e.relation_type}] w={e.weight:.3f}")
            else:
                print(f"  [Concept] '{args.concept}' found but no node data")
        else:
            print(f"  [Concept] '{args.concept}' not found in graph")
        return

    if args.chat is not None:
        queries = [q.strip() for q in args.chat.split("|") if q.strip()]
        if not queries:
            return
        results = []
        for i, q in enumerate(queries):
            t0 = time.time()
            try:
                resp = engine.process_turn(q)
            except Exception as e:
                resp = f"[error: {e}]"
            elapsed = time.time() - t0
            strategy = engine._last_strategy if args.strategy else ""
            strat_tag = f" [{strategy}]" if strategy else ""
            results.append((q, resp, elapsed))
            print(f"Q{i+1}: {q}")
            print(f"A{i+1}: {resp}{strat_tag}")
            if elapsed > 0.5:
                print(f"     [...{elapsed:.1f}s]")
            if args.trace:
                engine.print_traces(f"Q{i+1}")
            print()
        result = engine.save()
        print(f"  [{result}]")
        print(f"  [Stats] Turns: {engine.turn_count}, Words: {len(engine.graph.nodes)}, Sleeps: {engine.sleep_cycles_completed}")
        return

    # ── INTERACTIVE MODE ──
    print()
    print("  ============================================")
    print("   RAVANA - teenage mind, learning from the web...")
    print("  ============================================")
    print()

    if engine.turn_count == 0:
        print()
        print("  Hey! I'm a teenage mind — I know some things but")
        print("  I'm always curious to learn more. I can think about")
        print("  causes, patterns, and different perspectives.")
        print("  Talk to me about anything!")
    else:
        print(f"  Welcome back! I now know {len(engine.graph.nodes)} words across {len(engine.graph.edges)} connections.")
        print(f"  I've slept {engine.sleep_cycles_completed} times to consolidate my learning.")
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

            # Detect quit naturally
            if user_input.lower() in ("bye", "goodbye", "see you", "good night"):
                print(f"\n  RAVANA: Bye bye! I'll remember what you taught me!")
                return

            try:
                t0 = time.time()
                response = engine.process_turn(user_input)
                elapsed = time.time() - t0
                print(f"\n  RAVANA: {response}")

                # Show learning stats every 5 turns
                if engine._learning_count > 0 and engine.turn_count % 5 == 0:
                    print(f"  [I've learned {engine._learning_count} times from the web and know "
                          f"{len(engine.graph.nodes)} words now!]")

                if elapsed > 0.5:
                    print(f"  [...took a moment to think...]")

            except Exception as e:
                print(f"\n  RAVANA: Hmm, I got confused. Let me try again!")
                if "--debug" in sys.argv:
                    import traceback
                    traceback.print_exc()
    finally:
        # Stop background learning before saving
        engine.stop_background_learning()
        # Auto-save on any exit
        result = engine.save()
        print(f"  [{result}]")


if __name__ == "__main__":
    main()
