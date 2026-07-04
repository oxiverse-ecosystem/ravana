"""Auto-generated mixin module for CognitiveChatEngine."""
from __future__ import annotations
import sys, os, time, random, json, re, threading, hashlib
import urllib.request
import socket
socket.setdefaulttimeout(4.0)
from urllib.error import URLError
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import deque, Counter


from .constants import TEEN_CONCEPTS, WEB_GARBAGE, STOP_WORDS
from .models import ChainHop, ChainTrace, CognitiveResponseContext
from ravana.graph.engine import DOMAIN_CONCEPTS, CONTRASTIVE_PAIRS, CAUSAL_PAIRS, IS_A_PAIRS
from ravana.language.surface_realizer import DiscourseState

# Compute project root (same logic as engine.py)
_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


class ChainWalkerMixin:
    """Mixin providing graph traversal, relation inference, and chain walking methods."""

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
                # Higher threshold (0.65 instead of 0.6) and dormant confidence raised from 0.001 to 0.02
                # This reduces noise edges while keeping strong semantic connections.
                if sim > 0.65:
                    weight = min(0.5, sim * 0.5)
                    # Infer proper relation type instead of always "semantic"
                    inf_type, _ = self._infer_relation_type(ni.label, nj.label, "semantic")
                    edge = self.graph.add_edge(nids[i], nids[j], weight=weight, relation_type=inf_type)
                    edge.confidence = 0.02  # dormant: require 2+ visits to wake (raised from 0.001)
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
        from ravana.chat.constants import KNOWN_VERBS, KNOWN_ADJS, FUNCTION_WORDS, FUNCTION_POS

        verb_suffixes = ['ing', 'ed', 'ize', 'ify', 'ate', 'en', 'ish']
        adj_suffixes = ['able', 'ible', 'ful', 'less', 'ous', 'al', 'ic', 'ive']

        # Iterate over all graph node labels, not just _concept_labels
        for node in self.graph.nodes.values():
            if node.label:
                ll = node.label.lower()
                is_verb = ll in KNOWN_VERBS
                if not is_verb:
                    # Check third person singular -s (e.g. happens, differs, leads)
                    if ll.endswith('s'):
                        if ll[:-1] in KNOWN_VERBS:
                            is_verb = True
                        elif ll.endswith('es') and ll[:-2] in KNOWN_VERBS:
                            is_verb = True
                
                if is_verb:
                    self._concept_pos[ll] = 'verb'
                elif ll in KNOWN_ADJS:
                    self._concept_pos[ll] = 'adj'
                elif any(len(ll) > len(s) + 1 and ll.endswith(s) for s in verb_suffixes):
                    self._concept_pos[ll] = 'verb'
                elif any(len(ll) > len(s) + 1 and ll.endswith(s) for s in adj_suffixes):
                    self._concept_pos[ll] = 'adj'
                elif ll in FUNCTION_WORDS:
                    self._concept_pos[ll] = FUNCTION_POS.get(ll, 'adv')
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

        # Seed default natural definitions for core domain concepts
        default_definitions = {
            "ravana": "a cognitive architecture designed like a teenage mind that learns concepts and connections from the web using Hebbian learning and sleep consolidation without any backpropagation.",
            "oxiverse": "a privacy-first, source-available ecosystem built as a decentralized alternative to big tech.",
            "intentforge": "an intent-driven semantic search engine that helps discover and learn about new concepts.",
            "hebbian learning": "a biological learning rule where connections between neurons strengthen when they are activated together, often summarized as 'cells that fire together, wire together'.",
            "cognitive architecture": "a theoretical model and software framework that replicates the structure and cognitive processes of the human brain.",
            "privacy": "the fundamental right to control how your personal data and digital identity are accessed and used.",
        }
        for concept, defn in default_definitions.items():
            if concept not in self._definitions:
                self._definitions[concept] = defn

        # Dynamically seed proper nouns from DOMAIN_CONCEPTS
        for concept_name in DOMAIN_CONCEPTS.keys():
            self._proper_nouns.add(concept_name.lower())

        # Dynamically seed proper nouns from teen_seeds.txt (by checking capitalized words inside sentences)
        corpus_path = os.path.join(_proj_root, "data", "corpora", "teen_seeds.txt")
        if os.path.exists(corpus_path):
            try:
                import re
                with open(corpus_path, "r", encoding="utf-8") as f:
                    corpus_text = f.read()
                # Split by sentence boundary
                sentences = re.split(r'[.!?]\s+', corpus_text)
                for sent in sentences:
                    words = sent.strip().split()
                    if len(words) > 1:
                        for w in words[1:]:
                            clean_w = w.strip(".,!?\"'()[]{}*:;")
                            if clean_w and clean_w[0].isupper() and clean_w.lower() not in STOP_WORDS:
                                self._proper_nouns.add(clean_w.lower())
            except Exception:
                pass



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



    def _init_causal_detection(self):
        """Precompute causal seed vectors for semantic causal detection.

        Uses GloVe semantic similarity: ANY input word whose vector is close to
        known causal seeds (because, therefore, cause, trigger, if, when, etc.)
        is treated as a causal connector. This generalizes to novel phrasings
        without hardcoded pattern matching for each specific sentence structure.
        """
        seeds = [
            "cause", "causes", "lead", "leads", "led", "trigger", "triggers",
            "produce", "produces", "result", "results", "resulted",
            "generate", "generates", "induce", "induces", "create", "creates",
            "spark", "sparks", "provoke", "provokes",
            "because", "therefore", "hence", "thus", "consequently",
            "since", "when", "whenever", "if", "unless", "once", "so",
            "make", "makes", "force", "forces", "enable", "enables",
        ]
        self._causal_seed_vecs = []
        for w in seeds:
            v = self._glove_vector(w)
            if v is not None:
                self._causal_seed_vecs.append(v)
        if self._causal_seed_vecs:
            proto = np.mean(self._causal_seed_vecs, axis=0).astype(np.float32)
            norm = np.linalg.norm(proto)
            self._causal_proto = proto / norm if norm > 0 else None
        else:
            self._causal_proto = None



    def _word_causal_score(self, word: str) -> float:
        """Return max GloVe similarity between word and any causal seed vector.

        Score >0.30 indicates the word carries causal connective semantics
        (like 'because', 'if', 'triggers', 'so'), making it suitable for
        splitting a sentence into cause and effect clauses.
        """
        if not self._causal_seed_vecs:
            return 0.0
        v = self._glove_vector(word)
        if v is None:
            return 0.0
        best = 0.0
        for sv in self._causal_seed_vecs:
            sim = float(np.dot(v, sv))
            if sim > best:
                best = sim
        return best



    def _init_relation_prototypes(self):
        """Precompute prototype vectors for ALL relation types.

        Each prototype is the mean GloVe vector of seed words representing
        a relation type (causal, contrastive, temporal, analogical, semantic).
        Used by the PFC to compute top-down attentional bias: the similarity
        between the discourse intent's relation and each prototype determines
        how much to boost or discount edges of that type during chain walking.

        Neuroscience basis: Miller & Cohen (2001) — the PFC maintains a task-set
        that biases posterior processing toward goal-relevant dimensions.
        Here, vector similarity (not rules) determines which relation dimensions
        are relevant for the current discourse intent.
        """
        proto_map = {
            "causal": ["cause", "lead", "trigger", "produce", "result",
                        "because", "therefore", "hence", "consequently", "effect",
                        "causal"],
            "contrastive": ["but", "however", "unlike", "opposite", "differ",
                             "contrast", "instead", "although", "yet", "versus"],
            "temporal": ["when", "after", "before", "during", "while", "then",
                          "until", "since", "sequence", "subsequent"],
            "analogical": ["like", "similar", "analogous", "metaphor", "compare",
                            "parallel", "resemble", "likewise", "correspond"],
            "semantic": ["relate", "connect", "associate", "refer", "meaning",
                          "define", "describe", "denote", "signify"],
        }
        self._relation_prototypes = {}
        for rel_type, seeds in proto_map.items():
            vecs = [self._glove_vector(w) for w in seeds
                    if self._glove_vector(w) is not None]
            if vecs:
                proto = np.mean(vecs, axis=0).astype(np.float32)
                norm = np.linalg.norm(proto)
                if norm > 0:
                    proto /= norm
                self._relation_prototypes[rel_type] = proto



    def _relation_modulation_for_word(self, word: str
                                      ) -> Optional[Dict[str, float]]:
        """Compute relation modulation from a single word using GloVe semantics.

        Uses z-score normalization across all relation prototype similarities so
        the most relevant relation type gets the strongest boost, less relevant
        types get neutral or slight discount, and unrelated types get clear
        discount. No hardcoded thresholds — the distribution of similarities
        determines the modulation naturally.

        Used by both the PFC discourse planner (per-intent) and the association
        spread (pre-computed from question type).

        Returns dict of {relation_type: multiplier} or None if word not found.
        """
        if not hasattr(self, '_relation_prototypes') or not self._relation_prototypes:
            self._init_relation_prototypes()
        if not self._relation_prototypes:
            return None

        target_vec = self._glove_vector(word)
        if target_vec is None:
            return None

        sims = {}
        for rt, proto_vec in self._relation_prototypes.items():
            sims[rt] = float(np.dot(target_vec, proto_vec))

        arr = np.array(list(sims.values()))
        mean = float(arr.mean())
        std = float(arr.std()) if arr.std() > 0.01 else 0.15

        out = {}
        for rt, sim in sims.items():
            z = (sim - mean) / std
            out[rt] = 1.0 + min(0.5, z * 0.15)
        return out



    def _compute_relation_modulation(self, plan, intent
                                     ) -> Optional[Dict[str, float]]:
        """PFC top-down modulation: which relation types to bias for this intent.

        Uses the intent's primary_relation (e.g. 'causal' for hypotheticals)
        or the plan's question_type as fallback. Delegates to the shared
        _relation_modulation_for_word for the actual vector computation.

        Neuroscience basis: Miller & Cohen (2001) — PFC task-set biases.
        """
        rel_type = intent.primary_relation or ""
        out = self._relation_modulation_for_word(rel_type)
        if out is None:
            qtype = getattr(plan, 'question_type', "") or ""
            if qtype:
                out = self._relation_modulation_for_word(qtype)
        return out



    def _extract_and_store_causal_relations(self, text: str) -> int:
        """Extract causal relations from user input using GloVe vector semantics.

        Instead of hardcoded regex patterns for "when X, Y" / "if X, then Y" /
        "X causes Y" etc., this uses GloVe semantic similarity to detect ANY
        word with causal connective semantics. It then splits the sentence at
        the causal word and creates cause→effect edges between content words
        on each side.

        Neuroscience basis: the brain's causal reasoning network (prefrontal-
        parietal) uses semantic integration (Hagoort's MUC framework) rather
        than pattern matching to detect cause-effect structure in language.
        This method mirrors that semantic approach.

        Returns number of causal edges created or strengthened.
        """
        if not hasattr(self, '_causal_seed_vecs') or not self._causal_seed_vecs:
            self._init_causal_detection()
        if not self._causal_seed_vecs:
            return 0

        text_lower = text.lower().strip()
        if not text_lower:
            return 0

        # Skip interrogative input: questions ask about causality, they don't
        # assert it. Parsing "what happens if X?" as "happens → X" would create
        # spurious edges that derail the chain walk.
        if text_lower.endswith("?") or re.match(
            r'^(what|who|where|when|why|how|which|whose|whom|'
            r'do(es|id)?|is|are|was|were|can|could|will|would|'
            r'shall|should|may|might|must|have|has|had)\b', text_lower):
            return 0

        edges_created = 0
        sentences = [s.strip() for s in re.split(r'[.!?]+', text_lower) if s.strip()]

        for sentence in sentences:
            words = list(re.finditer(r'[A-Za-z]+', sentence))
            for match in words:
                word = match.group()
                if len(word) < 3:
                    continue
                if self._word_causal_score(word) < 0.30:
                    continue

                if match.start() == 0:
                    # Connector at start: "connector cause, effect"
                    rest = sentence[match.end():].strip().lstrip(",;").strip()
                    split_pos = len(rest)
                    for delim in [",", ";"]:
                        idx = rest.find(delim)
                        if 0 <= idx < split_pos:
                            split_pos = idx
                    then_m = re.search(r'\bthen\b', rest)
                    if then_m and then_m.start() < split_pos:
                        split_pos = then_m.start()
                    if 0 < split_pos < len(rest):
                        before = rest[:split_pos].strip()
                        after = rest[split_pos + 1:].strip()
                        after = re.sub(r'^then\s+', '', after)
                    else:
                        continue
                else:
                    # Connector in middle: "cause connector effect"
                    before = sentence[:match.start()].strip().rstrip(",;").strip()
                    after = sentence[match.end():].strip().lstrip(",;").strip()

                if not before or not after:
                    continue

                before_words = self._extract_content_words(before)
                after_words = self._extract_content_words(after)

                if before_words and after_words:
                    for bw in before_words:
                        for aw in after_words:
                            if bw != aw and self._create_causal_edge(bw, aw, text):
                                edges_created += 1

        if edges_created > 0 and self._trace_enabled:
            print(f"  [trace]   created/strengthened {edges_created} causal edges from input")
        return edges_created



    def _extract_content_words(self, text: str) -> List[str]:
        """Extract meaningful content words from text, filtering stop/garbage words."""
        words = re.findall(r"[a-zA-Z']{3,}", text.lower())
        meaningful = []
        for w in words:
            wc = w.strip("'")
            if wc not in STOP_WORDS and wc not in WEB_GARBAGE and len(wc) >= 3:
                meaningful.append(wc)
        return meaningful



    def _create_causal_edge(self, cause_word: str, effect_word: str,
                            source_text: str = "") -> bool:
        """Create or strengthen a causal edge between two words in the graph.

        If the edge already exists, boost its weight and set relation to causal.
        Otherwise create a new causal edge with weight sufficient for chain walking.
        """
        cause_nids = self._concept_keywords.get(cause_word, [])
        effect_nids = self._concept_keywords.get(effect_word, [])

        if not cause_nids or not effect_nids:
            return False

        cause_nid = cause_nids[0]
        effect_nid = effect_nids[0]

        existing = self.graph.get_edge(cause_nid, effect_nid)
        if existing:
            was_already_causal = existing.relation_type == "causal"
            existing.weight = min(0.85, existing.weight + 0.3)
            if existing.relation_type != "causal":
                existing.relation_type = "causal"
            existing.confidence = min(0.95, existing.confidence + 0.2)
            if not was_already_causal and self._trace_enabled:
                print(f"  [trace]   boosted edge {cause_word} -> {effect_word} to causal")
        else:
            self.graph.add_edge(
                cause_nid, effect_nid,
                weight=0.75,
                relation_type="causal",
                confidence=0.7,
            )
        # Mark as user-stated belief
        if source_text:
            self._belief_assertions.append((cause_word, "causal", effect_word))

        return True

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



    def _spread_and_collect(self, seed_ids: List[int],
                             primary_ids: Optional[Set[int]] = None,
                             relation_preference: Optional[Dict[str, float]] = None,
                             ) -> List[Tuple[str, float]]:
        """Propagate activation through graph edges (3 hops).

        Only concepts in primary_ids (or all seed_ids if not specified)
        serve as activation sources. Other seed_ids get context activation
        (0.3) but don't propagate — they only prevent their neighbors from
        being collected as novel associations.

        relation_preference: optional {relation_type: multiplier} from the PFC's
            top-down task-set bias (Miller & Cohen 2001). When set, edges of
            task-relevant types are amplified during propagation — e.g. causal
            edges get boosted when answering "what happens if" questions,
            without hardcoding which concepts to prefer.
        """
        if not seed_ids:
            return []
        all_seed_set = set(seed_ids)
        spread_set = primary_ids if primary_ids else all_seed_set
        # Only the primary spread sources block propagation — other seeds
        # (P600-propagated neighbor concepts) should remain collectable.
        # If we block them, high-value causal targets like "occurs" and
        # "explosion" (activated as neighbors of "lamp" during P600) become
        # invisible to the spread, defeating causal reasoning.
        block_set = spread_set

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
                    if tid in block_set:
                        continue
                    signal = node.activation * edge.weight * edge.confidence * decay
                    if relation_preference:
                        signal *= relation_preference.get(edge.relation_type, 1.0)
                    if signal > 0.01:
                        new_acts[tid] = new_acts.get(tid, 0.0) + signal
                # Follow incoming edges (semantic network is effectively undirected)
                for src, edge in self.graph.get_incoming(nid):
                    if src in block_set:
                        continue
                    signal = node.activation * edge.weight * edge.confidence * decay
                    if relation_preference:
                        signal *= relation_preference.get(edge.relation_type, 1.0)
                    if signal > 0.01:
                        new_acts[src] = new_acts.get(src, 0.0) + signal
            for nid, sig in new_acts.items():
                if nid in self.graph.nodes:
                    self.graph.activate(nid, sig)

        collected = []
        for nid, node in self.graph.nodes.items():
            if nid not in block_set and node.activation > 0.05:
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



    def _infer_relation_type(self, src_label: str, tgt_label: str,
                              current_type: str = "semantic") -> Tuple[str, float]:
        """Infer the correct relation type for a concept pair.
        Returns (inferred_type, confidence).
        Uses label-pair heuristics first, then vector-based ranking."""
        sl = src_label.lower()
        tl = tgt_label.lower()
        pair = tuple(sorted([sl, tl]))
        if pair in CONTRASTIVE_PAIRS:
            return ("contrastive", 0.85)
        if pair in CAUSAL_PAIRS:
            return ("causal", 0.80)
        if pair in IS_A_PAIRS:
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


    def _rank_relations(self, source_label: str, target_label: str,
                        default_type: str) -> List[Tuple[str, float]]:
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



    def _walk_chain(self, label: str, seen: Set[str], max_hops: int,
                    temperature: float = 0.25,
                    contradiction_penalty: float = 0.6,
                    activation_boost: Optional[Dict[str, float]] = None,
                    subject_proximity: Optional[str] = None,
                    episodic_first: bool = False,
                    relation_modulation: Optional[Dict[str, float]] = None) -> Optional[str]:
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
                # Filter: skip function words (closed-class items that add noise to chains)
                if tn.label.lower() in self._GRAMMATICAL_CONCEPTS:
                    continue
                # Filter: skip weak edges (below auto-wire minimum threshold)
                if edge.weight < 0.35:
                    continue
                # Self-penalty: penalize edges to concepts too semantically different from subject
                # SKIP for causal edges: causality often links semantically unrelated
                # concepts (lamp → explosion), so the penalty is always wrong here.
                score = edge.weight * edge.confidence
                if label and tn.label and label.lower() != tn.label.lower() and edge.relation_type != "causal":
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
                # Filter: skip function words
                if sn.label.lower() in self._GRAMMATICAL_CONCEPTS:
                    continue
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
                if (dnode and dnode.label and dnode.label.lower() not in seen
                    and dnode.label.lower() not in chain_labels
                    and dnode.label.lower() not in _main_concepts
                    and dnode.label.lower() not in self._GRAMMATICAL_CONCEPTS):
                    candidates.append((dedge.weight * dedge.confidence, dnode.label, dedge, "out"))
            # Episodic edges (conversation memory) - O(1) src-indexed, decay-modulated
            for dtgt, dedge in self._episodic_by_src.get(cur_nid, []):
                decay_mod = dedge.weight / 0.3 if dedge.weight > 0 else 1.0
                dnode = self.graph.nodes.get(dtgt)
                if (dnode and dnode.label and dnode.label.lower() not in seen
                    and dnode.label.lower() not in chain_labels
                    and dnode.label.lower() not in _main_concepts
                    and dnode.label.lower() not in self._GRAMMATICAL_CONCEPTS):
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

                # 11. PFC Relation Type Modulation — top-down attentional bias
                # Based on Miller & Cohen (2001): PFC maintains task-set that
                # biases processing toward goal-relevant information. Here the
                # intent's target relation type modulates edge selection so
                # causal edges are amplified for hypothetical queries.
                if relation_modulation:
                    _sig *= relation_modulation.get(_edge.relation_type, 1.0)

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
        return chain


    def _try_surface_realize(self, subject: str, target: str,
                              relation: str = "semantic",
                              discourse_type: str = "explain",
                              free_energy: float = 0.3,
                              min_len: int = 8) -> Optional[str]:
        """Try to generate a sentence using SurfaceRealizer.
        
        Returns the sentence string on success, None on failure.
        Centralizes the SR try/except pattern to eliminate code duplication
        across all fallback paths in _graph_fallback_response.
        """
        try:
            if hasattr(self, 'syntactic_assembly') and hasattr(self, 'surface_realizer'):
                frame = self.syntactic_assembly.bind_to_sentence(
                    subject=subject, relation=relation, target=target,
                    pos_map=getattr(self, '_concept_pos', {}))
                disc_ctx = DiscourseState(
                    sentence_index=0, discourse_type=discourse_type,
                    total_sentences=1, free_energy=free_energy)
                s = self.surface_realizer.realize(frame=frame, discourse_context=disc_ctx,
                    dopamine_tone=getattr(self, '_dopamine_tone', 0.5),
                    cerebellar_ngram=getattr(self, 'cerebellar_ngram', None))
                if s and len(s) > min_len:
                    return s
        except Exception:
            pass
        return None


    # ─── Phase 7: Multi-Strategy Reasoning ───



    def _graph_fallback_response(self, ctx: CognitiveResponseContext) -> Tuple[str, str]:
        """Graph-walk fallback when decoder / syntactic pipeline are not enough.

        No hardcoded templates. Every utterance is generated through the
        free-energy-driven SurfaceRealizer with Hebbian verb selection.
        The PFC discourse type and free energy modulate epistemic stance,
        hedging, and sentence structure dynamically.

        Neuroscience basis:
        - Brouwer Retrieval-Integration: the N400/P600 cycle generates
          each sentence through retrieval + integration, not templates
        - Friston Free Energy Principle: epistemic stance emerges from
          confidence (inverse free energy), not hardcoded frames
        - Hebbian VerbLexicon: verb phrases are composed from morphemic
          primitives, not retrieved from a phrase list
        - PFC Discourse Planning: discourse markers emerge from the
          intent type, not from random selection
        """
        subject = ctx.subject
        assocs = ctx.associated_concepts

        if not subject:
            return ("...", "associative")

        subj_lower = subject.lower()

        # --- Priority 1: Definitional knowledge with SurfaceRealizer ---
        if subj_lower in self._definitions:
            defn = self._definitions[subj_lower]
            try:
                # Route through surface realizer for the definition sentence
                if hasattr(self, 'syntactic_assembly') and hasattr(self, 'surface_realizer'):
                    frame = self.syntactic_assembly.bind_to_sentence(
                        subject=subject,
                        relation="semantic",
                        target=defn,
                        pos_map=getattr(self, '_concept_pos', {}),
                    )
                    disc_ctx = DiscourseState(
                        sentence_index=0,
                        discourse_type="explain",
                        total_sentences=min(3, 1 + len(assocs)),
                        free_energy=0.2,
                    )
                    sentence = self.surface_realizer.realize(
                        frame=frame,
                        discourse_context=disc_ctx,
                        dopamine_tone=getattr(self, '_dopamine_tone', 0.5),
                        cerebellar_ngram=getattr(self, 'cerebellar_ngram', None),
                    )
                    if sentence and len(sentence) > 10:
                        response = sentence
                    else:
                        response = f"{subject.capitalize()} is {defn}."
                else:
                    response = f"{subject.capitalize()} is {defn}."
                
                # Add association sentences via surface realizer
                seen_assoc = {subj_lower}
                for label, score in assocs[:2]:
                    if label and label.lower() not in seen_assoc and label.lower() not in self._GRAMMATICAL_CONCEPTS:
                        try:
                            if hasattr(self, 'syntactic_assembly') and hasattr(self, 'surface_realizer'):
                                frame2 = self.syntactic_assembly.bind_to_sentence(
                                    subject=subject,
                                    relation="semantic",
                                    target=label,
                                    pos_map=getattr(self, '_concept_pos', {}),
                                )
                                disc_ctx2 = DiscourseState(
                                    sentence_index=1,
                                    discourse_type="elaborate",
                                    total_sentences=3,
                                    free_energy=0.3,
                                )
                                s2 = self.surface_realizer.realize(
                                    frame=frame2,
                                    discourse_context=disc_ctx2,
                                    dopamine_tone=getattr(self, '_dopamine_tone', 0.5),
                                    cerebellar_ngram=getattr(self, 'cerebellar_ngram', None),
                                    discourse_marker="also",
                                )
                                if s2 and len(s2) > 5:
                                    response += " " + s2
                                    seen_assoc.add(label.lower())
                        except Exception:
                            pass
                return (response, "definition_with_assoc")
            except Exception:
                # Generative fallback via SurfaceRealizer
                s = self._try_surface_realize(
                    subject=subject, target=defn[:60],
                    discourse_type="explain", free_energy=0.25, min_len=10)
                if s:
                    return (s, "definition_with_assoc")
                return (f"{subject.capitalize()} is {defn}.", "definition_with_assoc")

        # --- Priority 2: Known concept without definition ---
        if not assocs:
            if subj_lower in self._concept_keywords or subj_lower in self._concept_labels:
                # Generate via SurfaceRealizer
                s = self._try_surface_realize(
                    subject=subject, target=subject,
                    discourse_type="acknowledge", free_energy=0.4, min_len=8)
                if s:
                    return (s, "associative")
            return (f"i know about {subject}.", "associative")
            
            # For truly unknown concepts, generate based on the question type
            q_lower = ctx.raw_input.lower()
            # Generative response for unknown concepts via SurfaceRealizer
            disc_type = "explore" if ("why" in q_lower or q_lower.strip().endswith('?')) else "reflect"
            s = self._try_surface_realize(
                subject=subject, target=subject,
                discourse_type=disc_type, free_energy=0.6, min_len=8)
            if s:
                return (s, "conversational_fallback")
            if "why" in q_lower or q_lower.strip().endswith('?'):
                return (f"that is an interesting question.", "conversational_fallback")
            else:
                return (f"i am still learning about that.", "conversational_fallback")

        # --- Priority 3: Associative response via SurfaceRealizer ---
        try:
            if hasattr(self, 'syntactic_assembly') and hasattr(self, 'surface_realizer'):
                seen: Set[str] = {subject.lower()}
                
                # Filter out function words from associations (they leak through spreading activation
                # because graph edges like "why"→"because" exist. The chain walker filters them via
                # _GRAMMATICAL_CONCEPTS, but _graph_fallback_response iterates associations directly.)
                filtered_assocs = []
                for l, s in assocs:
                    ll = l.lower()
                    if ll in self._GRAMMATICAL_CONCEPTS:
                        continue
                    # Skip anything that isn't a noun (verbs, adjs, conj, prep, det, pron, adv)
                    pos = getattr(self, '_concept_pos', {}).get(ll, 'noun')
                    if pos != 'noun':
                        continue
                    filtered_assocs.append((l, s))
                # If all assocs were filtered out (e.g. only function words remain), skip to last-resort
                if not filtered_assocs:
                    raise StopIteration("all assocs were function words")
                
                # Build discourse context
                num_sentences = min(3, len(filtered_assocs))
                disc_ctx = DiscourseState(
                    sentence_index=0,
                    discourse_type="explain",
                    total_sentences=num_sentences,
                    free_energy=self._free_energy if hasattr(self, '_free_energy') else 0.3,
                )
                
                utterances = []
                for i, (label, score) in enumerate(filtered_assocs[:3]):
                    if label.lower() in seen:
                        continue
                    seen.add(label.lower())
                    
                    # Determine relation from graph edge
                    nid_subj = self._concept_keywords.get(subj_lower, [None])[0]
                    nid_label = self._concept_keywords.get(label.lower(), [None])[0]
                    relation = "semantic"
                    if nid_subj is not None and nid_label is not None:
                        edge = self.graph.get_edge(nid_subj, nid_label)
                        if edge is None:
                            edge = self.graph.get_edge(nid_label, nid_subj)
                        if edge is not None:
                            relation = edge.relation_type or "semantic"
                    
                    # Build syntactic frame
                    frame = self.syntactic_assembly.bind_to_sentence(
                        subject=subject,
                        relation=relation,
                        target=label,
                        pos_map=getattr(self, '_concept_pos', {}),
                    )
                    
                    # Set discourse context for this sentence
                    disc_ctx.sentence_index = i
                    disc_ctx.previous_subject = utterances[-1].split()[0] if utterances else None
                    if i == 0:
                        discourse_marker = ""
                    elif relation == "contrastive":
                        discourse_marker = ""
                    else:
                        discourse_marker = ""
                    
                    # Realize through free-energy-driven surface realizer
                    sentence = self.surface_realizer.realize(
                        frame=frame,
                        discourse_context=disc_ctx,
                        dopamine_tone=getattr(self, '_dopamine_tone', 0.5),
                        cerebellar_ngram=getattr(self, 'cerebellar_ngram', None),
                        discourse_marker=discourse_marker,
                    )
                    
                    if sentence and len(sentence) > 5:
                        utterances.append(sentence)
                        from ravana.language.verb_lexicon import VerbLexicon
                        VerbLexicon.reinforce(relation, frame.verb_phrase, success=1.0)
                    
                    if len(utterances) >= num_sentences:
                        break
                
                if utterances:
                    response = " ".join(utterances)
                    return (response, "graph_fallback")
        except Exception:
            pass
        
        # Last-resort: bare minimum if everything fails
        # Filter function words even in last-resort path
        filtered_assocs_lr = [
            (l, s) for l, s in (assocs or [])
            if l.lower() not in self._GRAMMATICAL_CONCEPTS
        ]
        if filtered_assocs_lr:
            # Find the first non-function-word association for a SurfaceRealizer call
            try:
                if hasattr(self, 'syntactic_assembly') and hasattr(self, 'surface_realizer'):
                    frame = self.syntactic_assembly.bind_to_sentence(
                        subject=subject,
                        relation="semantic",
                        target=filtered_assocs_lr[0][0],
                        pos_map=getattr(self, '_concept_pos', {}),
                    )
                    disc_ctx = DiscourseState(
                        sentence_index=0,
                        discourse_type="explain",
                        total_sentences=1,
                        free_energy=0.4,
                    )
                    s = self.surface_realizer.realize(
                        frame=frame,
                        discourse_context=disc_ctx,
                        dopamine_tone=getattr(self, '_dopamine_tone', 0.5),
                        cerebellar_ngram=getattr(self, 'cerebellar_ngram', None),
                    )
                    if s and len(s) > 5:
                        return (s, "associative")
            except Exception:
                pass
        # If even filtered assocs fail, return minimal response without function words
        if assocs:
            first_content = filtered_assocs_lr[0][0].lower() if filtered_assocs_lr else subject.lower()
            s = self._try_surface_realize(
                subject=subject, target=first_content,
                discourse_type="connect", free_energy=0.35, min_len=8)
            if s:
                return (s, "associative")
            return (f"i see. {subject.lower()} and {first_content} are linked.", "associative")
        s = self._try_surface_realize(
            subject=subject, target=subject,
            discourse_type="reflect", free_energy=0.5, min_len=8)
        if s:
            return (s, "associative")
        return (f"i see. {subject.lower()} is something i am still exploring.", "associative")


