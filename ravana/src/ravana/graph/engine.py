"""
Graph Engine - RAVANA's concept graph operations.
Contains: ConceptGraph wrapper, seeding, auto-expansion, spreading activation, hippocampal indexing.
"""
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import re
import os
import hashlib

# Import from ravana_ml.graph
from ravana_ml.graph import ConceptGraph, ConceptEdge, ConceptBindingMap
from ravana_ml.embedder import LearnedEmbedder

# Constants from original
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

# Teen concepts (subset for initial seeding - full list in ravana_chat.py)
TEEN_CONCEPTS = [
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
    ("complex", "complicated intricate sophisticated layered"),
    ("significant", "important meaningful major notable"),
    ("fundamental", "basic essential core foundation"),
    ("inevitable", "unavoidable certain destined fated"),
    ("possible", "maybe potential feasible plausible"),
    ("obvious", "clear apparent evident"),
    ("subtle", "nuanced delicate faint indirect"),
    ("profound", "deep meaningful significant thoughtful"),
    ("ignorance", "unawareness blindness obliviousness inexperience"),
    ("injustice", "unfairness inequality oppression bias"),
    ("oppression", "tyranny suppression persecution subjugation"),
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

# Relation inference pairs
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

# Domain concepts for bootstrapping
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


class GraphEngine:
    """Concept graph operations: seeding, expansion, activation, indexing."""

    def __init__(self, dim: int = 64, seed: int = 42, glove_vecs: Optional[Dict] = None,
                 glove_proj: Optional[np.ndarray] = None, glove_dim: int = 100,
                 glove_vector_cache: Optional[Dict] = None):
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.graph = ConceptGraph(dim=dim, max_nodes=10000)

        # GloVe embeddings
        self._glove_vecs = glove_vecs
        self._glove_proj = glove_proj
        self._glove_dim = glove_dim
        self._glove_vector_cache = glove_vector_cache or {}

        # Concept tracking
        self._concept_labels: Set[str] = set()
        self._concept_keywords: Dict[str, List[int]] = {}
        self._concept_pos: Dict[str, str] = {}
        self._concept_sources: Dict[str, Set[str]] = {}
        self._all_labels: Dict[str, int] = {}

        # Dormant edges (auto-wired but not visited)
        self._dormant_edges: Set[Tuple[int, int]] = set()

        # Hippocampal indexing
        self._topic_store: Dict[str, Dict] = {}
        self._topic_list: List[str] = []
        self._hippocampal_index: Dict[str, Dict] = {}

        # Contradiction map
        self._contradiction_map: Dict[str, Set[str]] = {}

        # Edge type connectors
        self._EDGE_CONNECTORS = {
            "causal": [("cause", ["because", "since", "as"]), ("result", ["leads to", "causes"]), ("effect", ["so", "therefore"])],
            "contrastive": [("contrast", ["but", "however", "yet"]), ("unexpected", ["nevertheless", "still"])],
            "semantic": [("identity", ["is like", "refers to", "means"]), ("relation", ["relates to", "connects with"])],
            "temporal": [("after", ["after", "then", "next"]), ("during", ["while", "during"])],
            "analogical": [("simile", ["like", "similar to"]), ("meta", ["acts as", "functions like"])],
            "episodic": [("memory", ["remember", "recall"]), ("event", ["when", "at"])],
        }
        self._EDGE_TO_GRAPH_LABEL = {
            "causal": "cause", "analogical": "like", "semantic": "connect",
            "contrastive": "but", "temporal": "change", "episodic": "connect",
        }
        self._EDGE_TO_STARTER = {
            "causal": "because", "contrastive": "but", "semantic": "and",
            "analogical": "like", "temporal": "then", "episodic": "and",
        }
        self._CONNECTOR_TO_REL: Dict[str, str] = {}
        for rel_type, tiers in self._EDGE_CONNECTORS.items():
            for entry in tiers:
                options = entry[1] if isinstance(entry, tuple) and len(entry) == 2 else entry[2]
                for opt in options:
                    self._CONNECTOR_TO_REL[opt] = rel_type
        self._CONNECTOR_SET = set(self._CONNECTOR_TO_REL.keys())

        # Dual stores (Phase 15)
        self._episodic_edges: Dict[Tuple[int, int], Any] = {}
        self._semantic_edges: Dict[Tuple[int, int], Any] = {}
        self._episodic_by_src: Dict[int, List] = {}
        self._semantic_by_src: Dict[int, List] = {}

    def _glove_vector(self, label: str) -> Optional[np.ndarray]:
        """Look up a label in GloVe, project to self.dim, return unit vector."""
        if self._glove_vecs is None:
            return None
        w = label.lower().strip()
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
            if w.rstrip('s') != w:
                self._glove_vector_cache[w.rstrip('s')] = result
            if len(w) > 2 and w[:-1] != w:
                self._glove_vector_cache[w[:-1]] = result
            return result
        return None

    # ─── Seeding ───

    def seed_concepts(self):
        """Seed the graph with teenager-level vocabulary and typed relationships."""
        self._concept_labels = set()
        all_labels = {label.lower() for label, _ in TEEN_CONCEPTS}
        for label, keywords in TEEN_CONCEPTS:
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

        # Build label_to_id map
        label_to_id = {}
        for nid, node in self.graph.nodes.items():
            if node.label:
                label_to_id[node.label] = nid

        # Typed edges
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
        ]

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

        temporal_edges = [
            ("birth", "life"), ("life", "death"),
            ("morning", "noon"), ("noon", "night"),
            ("question", "answer"), ("cause", "effect"),
            ("learn", "understand"), ("struggle", "succeed"),
        ]

        contrastive_edges = [
            ("good", "bad"), ("love", "hate"), ("life", "death"),
            ("truth", "lie"), ("freedom", "oppression"),
            ("courage", "fear"), ("hope", "despair"),
            ("knowledge", "ignorance"), ("justice", "injustice"),
            ("create", "destroy"), ("accept", "reject"),
            ("always", "never"),
        ]

        analogical_edges = [
            ("mind", "garden"), ("life", "journey"),
            ("knowledge", "light"), ("time", "river"),
            ("society", "organism"), ("identity", "building"),
            ("memory", "library"), ("heart", "engine"),
        ]

        # Phase 6: New cognitive reasoning edge types (edge types registered in ConceptGraph)
        # Specific fact pairs should be learned via RLMv2 from data, not hardcoded
        # negation_edges, antonym_edges, transitive_edges, physical_cause_edges,
        # comparison_edges, pragmatic_edges will be learned during training

        self._apply_edges(label_to_id, semantic_edges, "semantic", 0.35)
        self._apply_edges(label_to_id, causal_edges, "causal", 0.45)
        self._apply_edges(label_to_id, temporal_edges, "temporal", 0.40)
        self._apply_edges(label_to_id, contrastive_edges, "contrastive", 0.50)
        self._apply_edges(label_to_id, analogical_edges, "analogical", 0.30)
        # Phase 6 edge types are registered in ConceptGraph but populated via learning

        # Hub edges
        hub_edges = [
            ("i", "think", "causal"), ("i", "feel", "causal"),
            ("i", "know", "causal"), ("i", "want", "causal"),
            ("i", "like", "causal"), ("i", "say", "causal"),
            ("we", "people", "semantic"), ("we", "they", "semantic"),
            ("you", "person", "semantic"), ("you", "friend", "semantic"),
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

        # Auto-wire: connect GloVe-similar concepts (>0.6) as semantic edges
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
                    inf_type, _ = self._infer_relation_type(ni.label, nj.label, "semantic")
                    edge = self.graph.add_edge(nids[i], nids[j], weight=weight, relation_type=inf_type)
                    edge.confidence = 0.001
                    self._dormant_edges.add((nids[i], nids[j]))
                    auto_count += 1

        self._all_labels = label_to_id
        self._build_contradiction_map(contrastive_edges)
        migrated = self._correct_relation_types()
        self._build_concept_pos()
        extra = f", {migrated} reclassified" if migrated else ""
        print(f"  [Graph] Seeded {len(self.graph.nodes)} concepts, {len(self.graph.edges)} connections ({auto_count} auto-wired) across 5 relation types{extra}")

    def _apply_edges(self, label_to_id: Dict[str, int], edge_list: List[Tuple[str, str]],
                     rel_type: str, base_weight: float):
        """Apply edges to graph."""
        for src, tgt in edge_list:
            sid = label_to_id.get(src)
            tid = label_to_id.get(tgt)
            if sid is not None and tid is not None and self.graph.get_edge(sid, tid) is None:
                self.graph.add_edge(sid, tid, weight=base_weight + self.rng.uniform(0, 0.15),
                                   relation_type=rel_type)

    def _build_contradiction_map(self, contrastive_edges: List[Tuple[str, str]]):
        """Build map of concept -> antonym concepts from contrastive edges."""
        for a, b in contrastive_edges:
            al, bl = a.lower(), b.lower()
            self._contradiction_map.setdefault(al, set()).add(bl)
            self._contradiction_map.setdefault(bl, set()).add(al)

    def _rebuild_contradiction_map(self):
        """Rebuild from graph edges after load."""
        for (src, tgt), edge in self.graph.edges.items():
            if edge.relation_type != "contrastive":
                continue
            sn = self.graph.nodes.get(src)
            tn = self.graph.nodes.get(tgt)
            if sn and sn.label and tn and tn.label:
                al, bl = sn.label.lower(), tn.label.lower()
                self._contradiction_map.setdefault(al, set()).add(bl)
                self._contradiction_map.setdefault(bl, set()).add(al)

    def _build_concept_pos(self):
        """Build POS tags for seeded concepts."""
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
        for label in self._concept_labels:
            ll = label.lower()
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

    # ─── Relation Type Inference ───

    def _infer_relation_type(self, src_label: str, tgt_label: str,
                              current_type: str = "semantic") -> Tuple[str, float]:
        """Infer correct relation type for a concept pair."""
        sl = src_label.lower()
        tl = tgt_label.lower()
        pair = tuple(sorted([sl, tl]))
        if pair in CONTRASTIVE_PAIRS:
            return ("contrastive", 0.85)
        if pair in CAUSAL_PAIRS:
            return ("causal", 0.80)
        if pair in IS_A_PAIRS:
            return ("semantic", 0.80)
        ranked = self._rank_relations(src_label, tgt_label, current_type)
        if ranked:
            best_type, best_score = ranked[0]
            if len(ranked) > 1:
                second_score = ranked[1][1]
                margin = best_score - second_score
                if margin > 0.10 and best_type != current_type:
                    return (best_type, min(0.75, best_score))
            return (current_type, max(0.3, best_score))
        return (current_type, 0.3)

    def _rank_relations(self, source_label: str, target_label: str,
                        default_type: str) -> List[Tuple[str, float]]:
        """Rank relation types by vector-based plausibility."""
        src_nids = self._concept_keywords.get(source_label.lower(), [])
        tgt_nids = self._concept_keywords.get(target_label.lower(), [])
        if not src_nids or not tgt_nids:
            return [(default_type, 0.5)]
        src_node = self.graph.get_node(src_nids[0])
        tgt_node = self.graph.get_node(tgt_nids[0])
        if src_node is None or tgt_node is None or src_node.vector is None or tgt_node.vector is None:
            return [(default_type, 0.5)]
        src_vec = src_node.vector
        tgt_vec = tgt_node.vector
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
        return sorted(scores.items(), key=lambda x: -x[1])

    def _correct_relation_types(self) -> int:
        """Iterate edges and reclassify mis-typed relations."""
        migrated = 0
        for (sid, tid), edge in list(self.graph.edges.items()):
            src_node = self.graph.get_node(sid)
            tgt_node = self.graph.get_node(tid)
            if src_node is None or tgt_node is None or not src_node.label or not tgt_node.label:
                continue
            if edge.confidence >= 0.8:
                continue
            inferred_type, inferred_conf = self._infer_relation_type(
                src_node.label, tgt_node.label, edge.relation_type)
            if inferred_type != edge.relation_type:
                edge.relation_type = inferred_type
                edge.confidence = max(edge.confidence, inferred_conf * 0.8)
                migrated += 1
        return migrated

    # ─── Auto-Expansion (Phase 1) ───

    def auto_expand_concepts(self, text: str) -> int:
        """Phase 1.1+1.2: Auto-expand graph from user input."""
        words = re.findall(r"[a-zA-Z']{3,}", text.lower())
        meaningful = set()
        for w in words:
            wc = w.strip("'")
            if wc not in STOP_WORDS and len(wc) >= 3:
                meaningful.add(wc)
        if not meaningful:
            return 0

        existing_labels = set()
        for nid, node in self.graph.nodes.items():
            if node.label:
                existing_labels.add(node.label.lower())

        new_count = 0
        new_nodes: Dict[str, int] = {}

        # Phase 1.1: Add each new word as a concept (GloVe only)
        for word in meaningful:
            if word in existing_labels:
                continue
            vec = self._glove_vector(word)
            if vec is None:
                continue
            node = self.graph.add_node(vector=vec, label=word)
            self._concept_keywords[word] = self._concept_keywords.get(word, []) + [node.id]
            self._concept_labels.add(word.lower())
            existing_labels.add(word)
            new_nodes[word] = node.id
            new_count += 1

        if not new_nodes:
            return 0

        # Phase 1.1+1.2: Wire new concepts to existing via vector similarity
        for word, nid in new_nodes.items():
            node = self.graph.get_node(nid)
            if node is None or node.vector is None:
                continue
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
                    sim = float(all_sims[idx])
                    # Top-5 wiring
                    if sim > 0.4 and self.graph.get_edge(nid, other_nid) is None:
                        weight = min(0.6, sim * 0.6)
                        inf_type, _ = self._infer_relation_type(word, other_node.label, "semantic")
                        self.graph.add_edge(nid, other_nid, weight=weight, relation_type=inf_type)
                    # Phase 1.2: Auto-wire ALL existing where sim > 0.5
                    if sim > 0.5 and self.graph.get_edge(nid, other_nid) is None:
                        weight = min(0.6, sim * 0.6)
                        inf_type, _ = self._infer_relation_type(word, other_node.label, "semantic")
                        self.graph.add_edge(nid, other_nid, weight=weight, relation_type=inf_type)

        return new_count

    def bootstrap_domain_concepts(self):
        """Seed domain-specific concepts (oxiverse, intentforge, ravana)."""
        domain_relation_type_map = {
            "is_a": "semantic", "causal": "causal",
            "contrastive": "contrastive", "part_of": "contextual",
        }

        def _ensure_concept(label: str) -> int:
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
            print(f"  [Graph] Bootstrapped {len(created)} domain concepts: {', '.join(created)}")

    # ─── Spreading Activation ───

    def spread_and_collect(self, seed_ids: List[int],
                           primary_ids: Optional[Set[int]] = None) -> List[Tuple[str, float]]:
        """Propagate activation through graph edges (3 hops)."""
        if not seed_ids:
            return []
        seed_set = set(seed_ids)
        spread_set = primary_ids if primary_ids else seed_set

        for nid in seed_ids:
            self.graph.activate(nid, 1.0 if nid in spread_set else 0.3)

        for hop in range(3):
            new_acts: Dict[int, float] = {}
            decay = 0.7 ** (hop + 1)
            for nid in list(self.graph._active_nodes):
                node = self.graph.nodes.get(nid)
                if node is None or node.activation <= 0.01:
                    continue
                if nid not in spread_set:
                    continue
                for tid, edge in self.graph.get_outgoing(nid):
                    if tid in seed_set:
                        continue
                    signal = node.activation * edge.weight * edge.confidence * decay
                    if signal > 0.01:
                        new_acts[tid] = new_acts.get(tid, 0.0) + signal
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

    # ─── Hippocampal Indexing ───

    def hippocampal_index_topic(self, subject: str, activated_ids: List[int],
                                 hop_labels: List[Tuple[str, str]]):
        """Create hippocampal index for a topic."""
        sl = subject.lower()
        indexed_concepts = list(set(activated_ids))
        indexed_edges = [(f.lower(), t.lower()) for f, t in hop_labels]

        index_entry = {
            'label': subject,
            'turn': len(self._topic_list),
            'indexed_concepts': indexed_concepts[:10],
            'indexed_edges': indexed_edges[:5],
            'visit_count': 1,
        }

        if sl not in self._topic_store:
            self._topic_store[sl] = index_entry
        else:
            entry = self._topic_store[sl]
            entry['visit_count'] += 1
            entry['turn'] = index_entry['turn']
            existing_cons = set(entry.get('indexed_concepts', []))
            existing_cons.update(indexed_concepts[:10])
            entry['indexed_concepts'] = list(existing_cons)[:15]

        if sl not in self._topic_list:
            self._topic_list.append(subject)
        if len(self._topic_list) > 50:
            removed = self._topic_list[:-50]
            self._topic_list = self._topic_list[-50:]
            for r in removed:
                self._topic_store.pop(r.lower(), None)

    def recall_hippocampal(self, topic: str) -> Optional[List[int]]:
        """Reactivate a hippocampal index."""
        entry = self._topic_store.get(topic.lower())
        if not entry:
            return None

        reactivated = []
        for nid in entry.get('indexed_concepts', []):
            node = self.graph.get_node(nid)
            if node and node.label:
                self.graph.activate(nid, 0.5)
                reactivated.append(nid)

        for f_label, t_label in entry.get('indexed_edges', []):
            f_nids = self._concept_keywords.get(f_label.lower(), [])
            t_nids = self._concept_keywords.get(t_label.lower(), [])
            for fn in f_nids:
                for tn in t_nids:
                    edge = self.graph.get_edge(fn, tn)
                    if edge:
                        if edge.relation_type == "episodic":
                            edge.weight = min(0.35, edge.weight + 0.05)
                        self.graph.activate(fn, 0.4)
                        self.graph.activate(tn, 0.4)
                        if fn not in reactivated:
                            reactivated.append(fn)
                        if tn not in reactivated:
                            reactivated.append(tn)

        subj_nids = self._concept_keywords.get(topic.lower(), [])
        for sn in subj_nids:
            self.graph.activate(sn, 0.7)
            if sn not in reactivated:
                reactivated.append(sn)

        return reactivated

    # ─── Curiosity Scoring ───

    def get_curiosity_scores(self, max_topics: int = 10) -> List[Tuple[str, float]]:
        """Compute curiosity scores for graph concepts."""
        scores = {}
        seen = set()

        # Node-level prediction free energy
        for nid, node in self.graph.nodes.items():
            if node.label:
                pe = getattr(node, 'prediction_free_energy', 0.0)
                if pe > 0.1:
                    label = node.label.lower()
                    if label not in seen and len(label) >= 3:
                        scores[label] = scores.get(label, 0.0) + pe * 2.0
                        seen.add(label)

        # Edge-level prediction free energy
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

        # Contradiction involvement
        for label in self._contradiction_map:
            l = label.lower()
            if len(l) >= 3:
                scores[l] = scores.get(l, 0.0) + 1.0

        # Dormant edges
        if self._dormant_edges:
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

        # Novelty (least visited)
        # This requires visit count from outside
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:max_topics]

    # ─── Prediction Error Updates ───

    def update_edge_from_error(self, edge, src_vec: np.ndarray,
                                tgt_vec: np.ndarray, lr: float = 0.15) -> float:
        """Update edge weight using gradient descent on prediction error."""
        actual_sim = float(np.dot(src_vec, tgt_vec))
        error = (edge.weight - actual_sim) ** 2
        gradient = 2.0 * (edge.weight - actual_sim)
        edge.weight -= lr * gradient * 0.5
        edge.weight = np.clip(edge.weight, 0.01, 0.99)
        edge.confidence = max(0.05, 1.0 - np.tanh(error * 3.0))
        alpha = 0.1
        edge.prediction_free_energy = (1 - alpha) * edge.prediction_free_energy + alpha * error
        edge.prediction_count += 1
        return error

    # ─── Vector Neighbor ───

    def find_vector_neighbor(self, label: str) -> Optional[str]:
        """Find most semantically similar concept."""
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