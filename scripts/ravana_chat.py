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

import sys, os, time, random, json, re, argparse, pickle
import urllib.request
from urllib.parse import quote
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _proj_root)
sys.path.insert(0, os.path.join(_proj_root, "ravana-v2"))

from ravana_ml.graph import ConceptGraph
from core.emotion import VADEmotionEngine, VADConfig
from core.identity import IdentityEngine
from core.meaning import MeaningEngine, MeaningConfig
from core.dual_process import DualProcessController, DualProcessConfig
from core.global_workspace import GlobalWorkspace, GWConfig
from core.meta_cognition import MetaCognition, MetaCognitiveConfig, EpistemicMode
from core.sleep import SleepConsolidation, SleepConfig


# Try importing beautifulsoup4 for HTML parsing (optional but recommended)
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

SEARCH_API = "https://api.oxiverse.com/search?q="


# ─── Teen Vocabulary: what a teenager knows (~180 words) ───
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
    ("surprise", "shock amazement astonishment wonder"),
    ("guilt", "remorse regret shame blame"),
    ("disappointment", "letdown regret dissatisfaction dismay"),

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


@dataclass
class ChainHop:
    from_label: str
    to_label: str
    relation_type: str
    weight: float
    confidence: float
    temperature: float
    candidates: int  # how many edges were considered at this hop

@dataclass
class ChainTrace:
    hops: List[ChainHop] = field(default_factory=list)
    max_hops: int = 0
    completed: bool = False

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


class CognitiveChatEngine:
    """RAVANA cognitive chat engine — starts as a baby, learns from the web."""

    def __init__(self, dim: int = 64, seed: int = 42, baby_mode: bool = True):
        self.dim = dim
        self.rng = np.random.RandomState(seed)

        self.graph = ConceptGraph(dim=dim, max_nodes=10000)
        self.baby_mode = baby_mode
        self._concept_labels: Set[str] = set()  # set of primary concept labels

        # GloVe embeddings (loaded lazily during seeding)
        self._glove_vecs: Optional[Dict[str, np.ndarray]] = None
        self._glove_proj: Optional[np.ndarray] = None
        self._glove_dim: int = 100

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
        self._past_topics: List[str] = []
        self._last_responses: List[str] = []
        self._last_strategy: str = ""
        self._free_energy = 0.0
        self._learning_count = 0
        self._learned_this_turn = False
        self._concept_keywords: Dict[str, List[int]] = {}
        self._save_path = os.path.join(_proj_root, "ravana_weights.pkl")
        self.sleep_cycles_completed = 0
        self._chain_traces: List[ChainTrace] = []
        self._trace_enabled = False
        self._contradiction_map: Dict[str, Set[str]] = {}
        self._belief_assertions: List[Tuple[str, str, str]] = []
        self._user_interests: Dict[str, float] = {}  # concept -> strength (decays)

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

        # Concept-emotion tags
        self._concept_vad: Dict[int, Tuple[float, float, float]] = {}

        if os.path.exists(self._save_path):
            loaded = self._load()
            if loaded:
                print(f"  [Loaded] Remembered {len(self.graph.nodes)} words from before!")
                return

        # Seed teen concepts from scratch
        self._init_glove()
        self._seed_concepts()
        print(f"  [Teen] Knows {len(self.graph.nodes)} words, ready to learn!")

    # ─── Teen Graph Seeding ───

    def _seed_concepts(self):
        """Seed the graph with teenager-level vocabulary and typed relationships."""
        all_labels = []
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
            all_labels.append(label)
            self._concept_labels.add(label.lower())

            # Map keywords
            for kw in keywords.split():
                kl = kw.lower()
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
                if self.graph.get_edge(nids[i], nids[j]) is not None:
                    continue
                nj = self.graph.get_node(nids[j])
                if nj is None or nj.vector is None:
                    continue
                sim = float(np.dot(ni.vector, nj.vector))
                if sim > 0.6:
                    weight = min(0.5, sim * 0.5)
                    self.graph.add_edge(nids[i], nids[j], weight=weight, relation_type="semantic")
                    auto_count += 1

        self._all_labels = label_to_id
        self._build_contradiction_map(contrastive_edges)
        print(f"  [Teen] Seeded {len(self.graph.nodes)} concepts, {len(self.graph.edges)} connections ({auto_count} auto-wired) across 5 relation types")

    # ─── GloVe Semantic Vectors ───

    def _init_glove(self):
        """Load GloVe 100D vectors and build projection to self.dim."""
        for name in ['glove.6B.100d.txt', 'glove.6B.50d.txt']:
            path = os.path.join(_proj_root, 'data', 'glove', name)
            if os.path.exists(path):
                self._glove_dim = 100 if '100d' in name else 50
                break
        else:
            return
        glove_path = os.path.join(_proj_root, 'data', 'glove', f'glove.6B.{self._glove_dim}d.txt')
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

    def _glove_vector(self, label: str) -> Optional[np.ndarray]:
        """Look up a label in GloVe, project to self.dim, return unit vector."""
        if self._glove_vecs is None:
            return None
        w = label.lower().strip()
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
            return pv.astype(np.float32)
        return None

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

    def learn_from_web(self, query: str) -> str:
        """Search the web, fetch articles, extract concepts, and learn from them.
        
        Returns a summary of what was learned.
        """
        self._learned_this_turn = True
        self._learning_count += 1
        query_clean = quote(query)

        try:
            # Step 1: Search
            search_url = f"{SEARCH_API}{query_clean}"
            req = urllib.request.Request(search_url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) RAVANA-Baby/1.0'
            })
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read().decode('utf-8'))
            results = data.get('results', [])

            if not results:
                return self._learn_from_snippets(query, [])

            # Step 2: Get snippets and try to fetch article text
            new_concepts_added = 0
            snippets = []

            for i, r in enumerate(results[:3]):  # Top 3 results
                snippet = r.get('content', '') or ''
                title = r.get('title', '') or ''
                url = r.get('url', '') or ''
                text = f"{title}. {snippet}"

                # Try to fetch full article text
                if url:
                    try:
                        article_text = self._fetch_article_text(url)
                        if article_text and len(article_text) > len(text):
                            text = article_text[:3000]
                    except Exception:
                        pass

                snippets.append(text)

            # Step 3: Extract and learn concepts from all gathered text
            combined_text = " ".join(snippets)
            new_concepts_added = self._learn_from_text(combined_text, query)

            if new_concepts_added > 0:
                return f"learned {new_concepts_added} new things about {query}"
            else:
                return f"read about {query} but already knew the words"

        except Exception as e:
            return f"tried to learn but got confused: {e}"

    def _fetch_article_text(self, url: str) -> Optional[str]:
        """Fetch a URL and extract readable article text."""
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

    def _learn_from_text(self, text: str, topic: str) -> int:
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

        # Check which words are new vs known
        existing_labels = set()
        for nid, node in self.graph.nodes.items():
            if node.label:
                existing_labels.add(node.label.lower())

        new_count = 0
        label_to_id = {}
        for nid, node in self.graph.nodes.items():
            if node.label:
                label_to_id[node.label] = nid

        # Add new concepts
        for word in important_words:
            if word in existing_labels:
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
            node = self.graph.add_node(vector=vec, label=word)
            label_to_id[word] = node.id
            self._concept_keywords[word] = self._concept_keywords.get(word, []) + [node.id]
            self._concept_labels.add(word.lower())
            existing_labels.add(word)
            new_count += 1

        # Form connections between co-occurring important words
        # Words that appear together in the article get edges
        article_words_lower = [w for w in re.findall(r"[a-zA-Z']{3,}", text.lower())
                              if w not in STOP_WORDS and len(w.strip("'")) >= 3]
        # Get unique important words that appear in the text
        present_important = list(set(w for w in article_words_lower if w in important_words))

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
        for word in important_words[:3]:
            nid = label_to_id.get(word)
            if nid is None:
                continue
            node = self.graph.get_node(nid)
            if node is None:
                continue
            edge_count = 0
            for existing_nid, existing_node in self.graph.nodes.items():
                if existing_nid == nid or (existing_node.label and existing_node.label in important_words):
                    continue
                if existing_node.vector is not None and node.vector is not None:
                    sim = float(np.dot(node.vector, existing_node.vector))
                    if sim > 0.3:
                        if self.graph.get_edge(nid, existing_nid) is None:
                            weight = max(0.25, min(0.5, sim * 0.5))
                            self.graph.add_edge(nid, existing_nid, weight=weight,
                                                relation_type="semantic")
                            edge_count += 1
            # Guarantee at least one connection for the topic word
            if edge_count == 0 and word == important_words[0]:
                best_sim = 0.0
                best_nid = None
                for existing_nid, existing_node in self.graph.nodes.items():
                    if existing_nid == nid or existing_node.label in important_words:
                        continue
                    if existing_node.vector is not None and node.vector is not None:
                        sim = float(np.dot(node.vector, existing_node.vector))
                        if sim > best_sim:
                            best_sim = sim
                            best_nid = existing_nid
                if best_nid is not None:
                    weight = max(0.25, min(0.4, best_sim * 0.5))
                    self.graph.add_edge(nid, best_nid, weight=weight, relation_type="semantic")
        return new_count

    def _learn_from_snippets(self, query: str, snippets: List[str]) -> str:
        """Learn from search snippet text when article fetch fails."""
        combined = f"{query} " + " ".join(snippets[:3])
        count = self._learn_from_text(combined, query)
        if count > 0:
            return f"learned {count} new things about {query} from search snippets"
        return f"read about {query} but already knew those words"

    # ─── Core Response Pipeline ───

    def process_turn(self, user_input: str) -> str:
        """Process input and generate a response, auto-learning when needed."""
        self.turn_count += 1
        self._learned_this_turn = False

        # Step 1: Find matching concepts
        activated = self._activate_from_input(user_input)

        # Step 1b: If this is a follow-up (more/else/also), reactivate the latest
        # past topic so the graph walks find it naturally
        if self._is_follow_up(user_input) and self._past_topics:
            last_topic = self._past_topics[-1]
            lt_nids = self._concept_keywords.get(last_topic.lower(), [])
            for nid in lt_nids:
                if nid not in activated:
                    activated.append(nid)
                    self.graph.activate(nid, 0.6)

        # Step 1c: Extract user interests from "I like/love {concept}" patterns
        self._store_user_interests(user_input)

        # Step 2: Extract topic
        subject, obj = self._extract_topic(user_input, activated)
        relation = "is"

        # Step 2b: Primary IDs — only these concepts spread activation
        # (other input-matched concepts provide context but don't propagate)
        subject_ids = set()
        sl = subject.lower()
        if sl in self._concept_keywords:
            subject_ids.update(self._concept_keywords[sl])

        # Step 3: Spread activation through graph
        associations = self._spread_and_collect(activated, primary_ids=subject_ids)

        # Step 4: Decide if we need to learn from the web
        # Auto-learn when there are unknown meaningful words in the input
        input_words = [w.strip(".,!?") for w in user_input.lower().split()
                      if len(w.strip(".,!?")) >= 3]
        known_words = sum(1 for w in input_words if w in self._concept_keywords)
        unknown_meaningful = [w for w in input_words
                              if w not in self._concept_keywords and w not in STOP_WORDS]
        needs_learning = (len(unknown_meaningful) > 0 and
                         self._learning_count < 15)  # Limit to 15 searches per session

        if needs_learning and self.baby_mode:
            learn_query = self._extract_learning_query(user_input, activated)
            if learn_query:
                learn_summary = self.learn_from_web(learn_query)
                # Re-activate with new knowledge
                activated = self._activate_from_input(user_input)
                subject, obj = self._extract_topic(user_input, activated)
                # Recompute primary IDs for new subject
                subject_ids = set()
                sl2 = subject.lower()
                if sl2 in self._concept_keywords:
                    subject_ids.update(self._concept_keywords[sl2])
                associations = self._spread_and_collect(activated, primary_ids=subject_ids)

        # Step 5: Emotional modulation (with concept-specific tagging)
        self._update_emotion(user_input)
        for nid in activated:
            self._concept_vad[nid] = (
                self.emotion.state.valence,
                self.emotion.state.arousal,
                self.emotion.state.dominance,
            )

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
        else:
            epistemic_mode = self.meta_cog.current_mode

        # Step 6c: Sleep pressure accumulation
        self._sleep_pressure += 0.02 + 0.01 * (1.0 - confidence)
        if self._sleep_pressure > 0.3 and (self.turn_count - self._last_sleep_episode) > 8:
            # Run a mini sleep cycle: consolidate knowledge
            self._sleep_consolidate()
            self._last_sleep_episode = self.turn_count
            self._sleep_pressure = 0.0

        # Step 7: Past topics
        past = self._recall_past(subject, obj)

        # Step 8: Build context and generate response
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
        )

        response, strategy = self._generate_response(ctx)
        self._last_strategy = strategy

        # Step 9: Update cognitive state
        self._update_state(ctx)

        # Step 10: Track topics
        if subject and subject.lower() not in [t.lower() for t in self._past_topics]:
            self._past_topics.append(subject)
        if len(self._past_topics) > 30:
            self._past_topics = self._past_topics[-30:]

        # Step 11: Store episodic memory — link subject to its associations
        self._store_episodic(subject, associations)

        # Step 12: Decay user interests (fade 5% per turn)
        for k in list(self._user_interests):
            self._user_interests[k] *= 0.95
            if self._user_interests[k] < 0.1:
                del self._user_interests[k]

        self._last_responses.append(response)
        if len(self._last_responses) > 10:
            self._last_responses = self._last_responses[-10:]

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
                    scores[nid] = scores.get(nid, 0) + 5.0

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

    def _is_follow_up(self, text: str) -> bool:
        words = set(w.lower().strip(".,!?") for w in text.split())
        return bool(words & self.FOLLOW_UP_WORDS)

    def _store_episodic(self, subject: str, associations: List[Tuple[str, float]]):
        """Create episodic edges linking current subject to top associations."""
        if not subject or not associations:
            return
        subj_nids = self._concept_keywords.get(subject.lower(), [])
        if not subj_nids:
            return
        for assoc_label, _ in associations[:3]:
            assoc_nids = self._concept_keywords.get(assoc_label.lower(), [])
            if (assoc_nids and
                    not self.graph.get_edge(subj_nids[0], assoc_nids[0])):
                self.graph.add_edge(subj_nids[0], assoc_nids[0],
                                    weight=0.15, relation_type="episodic")

    # Verbs that signal user interest when used as "I {verb} {concept}"
    _INTEREST_VERBS = {"like", "love", "enjoy", "play", "listen", "watch",
                        "read", "study", "practice", "explore", "create"}

    def _store_user_interests(self, text: str):
        """Parse 'I {verb} {concept}' patterns and store as user interests."""
        words = text.lower().split()
        for i, w in enumerate(words):
            if w == "i" and i + 2 < len(words):
                verb = words[i + 1].strip(".,!?'")
                if verb in self._INTEREST_VERBS:
                    obj = words[i + 2].strip(".,!?'")
                    if obj in self._concept_keywords:
                        self._user_interests[obj] = min(1.0, self._user_interests.get(obj, 0) + 0.4)

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

    def _extract_topic(self, text: str, activated: List[int]) -> Tuple[str, str]:
        """Extract the main topic from input. Uses graph-activated concepts
        first, then falls back to pattern detection.
        
        For 'what is trust' -> 'trust'
        For 'you know i was thinking about trust' -> 'trust' (skips 'you', 'i')
        For 'does learning change your brain' -> 'learning'
        """
        # Check for "what is X" / "what's X" patterns first
        m = re.match(r"what'?s?\s+(?:is\s+|are\s+)?(.+)", text.lower().strip(" ?!."))
        if m:
            phrase = m.group(1).strip()
            words = phrase.split()
            # Prefer concepts that exist in the graph, fall back to last meaningful word
            for w in reversed(words):
                wc = w.strip(".,!?")
                if (len(wc) > 2 and wc not in self.QUESTION_WORDS
                        and wc not in self.TOPIC_SKIP_WORDS
                        and wc not in STOP_WORDS):
                    if wc in self._concept_labels:
                        return (wc, text)
            # Try keyword-matched concepts
            for w in reversed(words):
                wc = w.strip(".,!?")
                if (len(wc) > 2 and wc not in self.QUESTION_WORDS
                        and wc not in self.TOPIC_SKIP_WORDS
                        and wc not in STOP_WORDS
                        and wc in self._concept_keywords):
                    return (wc, text)
            # No graph concept found — return last meaningful word
            for w in reversed(words):
                wc = w.strip(".,!?")
                if (len(wc) > 2 and wc not in self.QUESTION_WORDS
                        and wc not in self.TOPIC_SKIP_WORDS
                        and wc not in STOP_WORDS):
                    return (wc, text)
            if words:
                return (words[-1].strip(".,!?"), text)

        # Use best activated concept (skip question, topic-skip, and short words)
        if activated:
            best_real = None
            for nid in activated:
                node = self.graph.get_node(nid)
                if node and node.label:
                    lbl = node.label.lower()
                    if (len(lbl) > 2 and lbl not in self.QUESTION_WORDS
                            and lbl not in self.TOPIC_SKIP_WORDS):
                        if best_real is None:
                            best_real = (node.label, text)
            if best_real:
                return best_real
            nid = activated[0]
            node = self.graph.get_node(nid)
            if node and node.label:
                return (node.label, text)

        # Fallback: find meaningful words
        words = [w.strip(".,!?") for w in text.lower().split()
                 if len(w.strip(".,!?")) > 2
                 and w.strip(".,!?") not in self.QUESTION_WORDS
                 and w.strip(".,!?") not in self.TOPIC_SKIP_WORDS
                 and w.strip(".,!?") not in STOP_WORDS]
        if words:
            return (words[-1], text)

        first = text.split()[0] if text.split() else ""
        return (first, text)

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
        sa = 0.15
        # Positive words boost valence
        if words & positive:
            sv += 0.25
            sa += 0.08
        # Negative words lower valence
        if words & negative:
            sv -= 0.25
            sa += 0.12
        # Curiosity words increase arousal (engagement)
        if words & curious:
            sa += 0.1
            if sv == 0.0:
                sv += 0.05  # slight positive bias for curiosity
        # Learning excitement
        if self._learned_this_turn:
            sa += 0.25
            sv += 0.1
        # Novelty-based arousal (unknown words = mild surprise)
        input_words = [w for w in words if len(w) >= 3]
        known = sum(1 for w in input_words if w in self._concept_keywords)
        if input_words and known / len(input_words) < 0.5:
            sa += 0.1  # novelty surprise
        self.emotion.update(stimulus_valence=sv, stimulus_arousal=sa,
                           stimulus_dominance=self.identity.state.strength * 0.4 + 0.2,
                           uncertainty=self._free_energy * 0.5)

    def _recall_past(self, subj: str, obj: str) -> List[str]:
        related = []
        for p in self._past_topics:
            pl = p.lower()
            sl = subj.lower()
            if pl != sl and (pl in sl or sl in pl or len(set(pl.split()) & set(sl.split())) > 0):
                related.append(p)
        return related[:3]

    # ─── Graph-Driven Response Generation ───
    # NO hardcoded strings. ALL content words are concept labels from the graph.
    # Edge relation types (stored in graph edge data) are mapped to graph concept
    # labels that already exist in the seeded vocabulary — no external text.

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
        connector = self._EDGE_TO_GRAPH_LABEL.get(rel, "")
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

    def _starter_from_chain(self, chain: str, subject: str) -> str:
        """Extract the first edge relation type from a chain string and return
        a discourse starter (graph label). The matched starter is weighted 3x
        higher but other starters still have a chance — prevents monotony when
        most edges are semantic."""
        if not chain:
            return "and"
        matched = "and"
        parts = chain.split()
        if len(parts) >= 3:
            conn = parts[1]
            for rel_type, graph_label in self._EDGE_TO_GRAPH_LABEL.items():
                if graph_label and conn == graph_label:
                    starter = self._EDGE_TO_STARTER.get(rel_type)
                    if starter:
                        matched = starter
                    break
        candidates = list(self._EDGE_TO_STARTER.values())
        weights = np.array([3.0 if s == matched else 1.0 for s in candidates], dtype=np.float64)
        weights /= weights.sum()
        return candidates[self.rng.choice(len(candidates), p=weights)]

    def _walk_chain(self, label: str, seen: Set[str], max_hops: int,
                    temperature: float = 0.25,
                    contradiction_penalty: float = 0.6) -> Optional[str]:
        """Walk a path through the graph from label, temperature-weighted.
        temperature=0 → greedy (always strongest), temperature=1 → near-uniform.
        Each hop adds `connector concept` to the chain. Returns None if no path.

        Applies contradiction_penalty to edges whose target contradicts a
        previously asserted belief about the source concept.

        Records detailed hop info in self._chain_traces when self._trace_enabled."""
        nids = self._concept_keywords.get(label.lower(), [])
        if not nids:
            return None
        chain = [label]
        chain_labels = {label.lower()}  # path-level cycle detection
        cur_nid = nids[0]
        cur_label = label
        hops = 0
        trace = ChainTrace(max_hops=max_hops) if self._trace_enabled else None
        while hops < max_hops:
            candidates = []
            for tid, edge in self.graph.get_outgoing(cur_nid):
                tn = self.graph.nodes.get(tid)
                if tn is None or tn.label is None or tn.label.lower() in seen:
                    continue
                if tn.label.lower() in chain_labels:
                    continue  # cycle detected within this chain
                candidates.append((edge.weight * edge.confidence, tn.label, edge, "out"))
            for src, edge in self.graph.get_incoming(cur_nid):
                sn = self.graph.nodes.get(src)
                if sn is None or sn.label is None or sn.label.lower() in seen:
                    continue
                if sn.label.lower() in chain_labels:
                    continue  # cycle detected within this chain
                candidates.append((edge.weight * edge.confidence, sn.label, edge, "in"))
            if not candidates:
                break
            # Apply contradiction penalty: if target contradicts a belief
            # about cur_label, reduce its score by penalty factor
            if contradiction_penalty > 0 and self._contradiction_map:
                penalized = []
                for sig, tgt_lbl, edge, d in candidates:
                    if self._is_contradictory(cur_label, tgt_lbl, edge.relation_type):
                        sig *= (1.0 - contradiction_penalty)
                    penalized.append((sig, tgt_lbl, edge, d))
                candidates = penalized
            # Personalization bonus: boost edges toward user interest concepts
            if self._user_interests:
                boosted = []
                for sig, tgt_lbl, edge, d in candidates:
                    interest_strength = self._user_interests.get(tgt_lbl.lower(), 0.0)
                    if interest_strength > 0:
                        sig += interest_strength * 0.2
                    boosted.append((sig, tgt_lbl, edge, d))
                candidates = boosted
            # Temperature-weighted selection
            if temperature > 0 and len(candidates) > 1:
                sigs = np.array([c[0] for c in candidates])
                # Clamp to avoid numerical issues with near-zero sigs
                sigs = np.clip(sigs, 1e-10, None)
                weights = np.exp(sigs / temperature)
                weights /= weights.sum()
                idx = self.rng.choice(len(candidates), p=weights)
            else:
                idx = max(range(len(candidates)), key=lambda i: candidates[i][0])
            best_sig, best_label, best_edge, direction = candidates[idx]
            connector = self._EDGE_TO_GRAPH_LABEL.get(best_edge.relation_type, "")
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
            if trace is not None:
                trace.hops.append(ChainHop(
                    from_label=cur_label, to_label=best_label,
                    relation_type=best_edge.relation_type,
                    weight=best_edge.weight, confidence=best_edge.confidence,
                    temperature=temperature, candidates=len(candidates)))
            cur_label = best_label
            cur_nid = self._concept_keywords.get(best_label.lower(), [None])[0]
            if cur_nid is None:
                break
            hops += 1
        if trace is not None:
            trace.completed = hops >= max_hops
            self._chain_traces.append(trace)
        if len(chain) <= 1:
            return None
        return " ".join(chain)

    def _generate_response(self, ctx: CognitiveResponseContext) -> Tuple[str, str]:
        """Walk a progressive chain through the graph: 1 hop, then 2 hops, then 3.
        Each sentence continues from where the previous left off, creating a coherent
        multi-hop narrative. All words from graph labels, connectors from edge data."""
        subject = ctx.subject
        assocs = ctx.associated_concepts

        if not assocs:
            if subject:
                neighbor = self._find_vector_neighbor(subject)
                if neighbor and neighbor.lower() != subject.lower():
                    return (f"{subject} connect {neighbor}.", "associative")
                return (subject + ".", "associative")
            return ("...", "associative")

        seen: Set[str] = {subject.lower()}
        sentences = []

        # Walk progressive depths: 1 hop, 2 more, 3 more
        # Each subsequent walk starts from the last concept of the previous walk
        # Vary temperature per sentence: first sentence mostly greedy,
        # later sentences more exploratory
        chain1 = self._walk_chain(subject, seen, max_hops=1, temperature=0.15)
        if not chain1:
            neighbor = self._find_vector_neighbor(subject)
            if neighbor and neighbor.lower() != subject.lower():
                return (f"{subject} connect {neighbor}.", "associative")
            return (subject + ".", "associative")
        sentences.append(chain1 + ".")

        # Extract last concept of chain1 as the new starting point
        c1_parts = chain1.split()
        start2 = c1_parts[-1] if c1_parts[-1] not in self._EDGE_TO_GRAPH_LABEL.values() and \
            c1_parts[-1].lower() not in self.TOPIC_SKIP_WORDS else subject
        chain2 = self._walk_chain(start2, seen, max_hops=2, temperature=0.3)
        if chain2:
            conn = self._starter_from_chain(chain2, subject)
            sentences.append(f"{conn} {chain2}.")
        else:
            # Multi-branch fallback: try subject, then other associations
            chain2 = self._walk_chain(subject, seen, max_hops=2, temperature=0.3)
            if not chain2:
                for alt_label, _ in assocs[:4]:
                    alc = alt_label.lower()
                    if alc not in seen and alc not in self.TOPIC_SKIP_WORDS:
                        chain2 = self._walk_chain(alt_label, seen, max_hops=2, temperature=0.3)
                        if chain2:
                            break
            if chain2:
                conn = self._starter_from_chain(chain2, subject)
                sentences.append(f"{conn} {chain2}.")

        if len(sentences) < 3:
            # Try extending from chain2's end
            if chain2:
                c2_parts = chain2.split()
                start3 = c2_parts[-1] if c2_parts[-1] not in self._EDGE_TO_GRAPH_LABEL.values() and \
                    c2_parts[-1].lower() not in self.TOPIC_SKIP_WORDS else subject
            else:
                start3 = subject
            chain3 = self._walk_chain(start3, seen, max_hops=3, temperature=0.4)
            if chain3:
                conn = self._starter_from_chain(chain3, subject)
                sentences.append(f"{conn} {chain3}.")
            else:
                for perspective in self.rng.permutation(["i", "we", "you"])[:2]:
                    per_phrase = self._phrase_from_label(perspective, seen, max_concepts=4)
                    if per_phrase:
                        conn = self._starter_from_chain(per_phrase, perspective)
                        sentences.append(f"{conn} {per_phrase}.")
                        break

        self._log_assertions(sentences, subject)
        return (" ".join(sentences), "associative")


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

    def _sleep_consolidate(self):
        """Run a mini sleep cycle to strengthen useful patterns and weaken noise.
        
        Teenagers' brains consolidate learning during sleep! This mimics that
        by replaying recent topics through the graph and applying Hebbian
        strengthening to co-activated concepts.
        """
        if len(self.graph.edges) < 3:
            return

        # Phase 1: Strengthen edges between concepts that co-occur in conversation
        for topic in self._past_topics[-5:]:
            tids = self._concept_keywords.get(topic.lower(), [])
            if len(tids) < 2:
                continue
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    edge = self.graph.get_edge(tids[i], tids[j])
                    if edge:
                        edge.weight = min(0.95, edge.weight + 0.02)
                        edge.confidence = min(0.95, edge.confidence + 0.01)

        # Phase 2: Gentle edge pruning (only prune truly dead edges)
        edges_to_prune = []
        for (src, tgt), edge in self.graph.edges.items():
            if edge.weight < 0.02 and edge.confidence < 0.05 and edge.prediction_count < 1:
                edges_to_prune.append((src, tgt))
        for src, tgt in edges_to_prune:
            self.graph.remove_edge(src, tgt)

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

    # ─── Save/Load Persistence ───

    # ─── Diagnostics ───

    def print_traces(self, label: str = ""):
        """Print all chain walk traces from the last response."""
        if not self._chain_traces:
            return
        print(f"  [trace] {label}: {len(self._chain_traces)} chains")
        for ci, t in enumerate(self._chain_traces):
            print(f"  [trace]   chain {ci}: {t.max_hops} max, {'done' if t.completed else 'short'}")
            for i, h in enumerate(t.hops):
                dir_sym = " -> " if h.relation_type != "episodic" else " ~~ "
                print(f"  [trace]     hop {i}: {h.from_label}{dir_sym}{h.to_label}  "
                      f"[{h.relation_type}] w={h.weight:.3f} c={h.confidence:.3f} "
                      f"t={h.temperature:.2f} ({h.candidates} cand)")
        self._chain_traces.clear()

    def save(self) -> str:
        """Save full cognitive state to disk. Returns path to save file."""
        state = {
            'graph': self.graph,
            'concept_keywords': self._concept_keywords,
            'turn_count': self.turn_count,
            'past_topics': self._past_topics,
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
            'dim': self.dim,
            'rng_state': self.rng.get_state(),
            # Teen additions
            'sleep_pressure': self._sleep_pressure,
            'last_sleep_episode': self._last_sleep_episode,
            'sleep_cycles_completed': self.sleep_cycles_completed,
            'concept_vad': self._concept_vad,
            'meta_mode': self.meta_cog.current_mode.value,
            'contradiction_map': self._contradiction_map,
        }
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
            with open(self._save_path, 'rb') as f:
                state = pickle.load(f)

            # Restore graph
            self.graph = state['graph']
            self._concept_keywords = state['concept_keywords']
            self.turn_count = state['turn_count']
            self._past_topics = state['past_topics']
            self._last_responses = state['last_responses']
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
    args = parser.parse_args()

    # Handle --reset
    save_path = os.path.join(_proj_root, "ravana_weights.pkl")
    if args.reset:
        if os.path.exists(save_path):
            os.remove(save_path)
            print("  [Reset] Deleted saved weights, starting fresh!")
        else:
            print("  [Reset] No saved weights found, starting fresh!")

    engine = CognitiveChatEngine(dim=args.dim, seed=args.seed, baby_mode=True)
    if args.trace:
        engine._trace_enabled = True

    # ── BATCH MODE (--chat) ──
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
        # Auto-save on any exit
        result = engine.save()
        print(f"  [{result}]")


if __name__ == "__main__":
    main()
