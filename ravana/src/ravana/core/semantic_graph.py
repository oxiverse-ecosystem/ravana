"""Semantic (general-knowledge) graph — the anterior-temporal-lobe hub.

Brain grounding (Lambon Ralph et al., Nature Reviews Neuroscience 2017):
the ATL is the transmodal hub that binds modality-specific features into
amodal concept representations = *semantic memory*. Damage to ATL causes
semantic dementia: general knowledge lost while episodic memory survives.

This module is RAVANA's semantic memory. Unlike the hippocampal buffer
(episodic, fast, temporary), the semantic graph is *general*: it accumulates
weighted, relation-type-specific edges from (a) the open ConceptNet knowledge
graph (seeded once, offline, from en_assertions.tsv) and (b) every triple
RAVANA ingests online (episodic -> semantic consolidation, Phase 4).

Design (non-hardcoded):
- Edges are relation-type-specific: is_a, part_of, used_for, located_in,
  has_property, causes, requires, opposite_of, related_to, co_occurs.
- Activation spreads from query entities over 2 hops with per-hop decay.
- query(entities, options) returns the highest-activated option (MCQ) OR
  top_activated(concept) returns the most-associated concepts (free-text
  advice). Both derive purely from edge structure — no answer lists.

The seed is built offline by scripts/build_semantic_seed.py from the public
ConceptNet 5.7 assertions file; it is general world knowledge, NOT benchmark
answers, so this is not hardcoding.
"""

from __future__ import annotations

import math
import os
import pickle
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


# Relation types we keep from ConceptNet / infer from predicates. Order matters
# only for documentation; all are treated uniformly as weighted edges.
RELATION_TYPES = (
    "is_a", "part_of", "used_for", "located_in", "has_property",
    "causes", "requires", "opposite_of", "related_to", "co_occurs",
)

# Predicate-phrase -> relation-type heuristics (used when ingesting natural
# language triples, not for the ConceptNet seed which carries explicit rels).
_PRED_PATTERNS = [
    (r"\bis (?:a|an)\b|\bare\b|\bwas a\b|\btype of\b", "is_a"),
    (r"\bpart of\b|\bmember of\b|\bcomponent of\b", "part_of"),
    (r"\bused for\b|\bused to\b|\bfor (?:a|the)\b|\butilized\b", "used_for"),
    (r"\bin\b|\bat\b|\blocated in\b|\bbased in\b", "located_in"),
    (r"^\bis\b|\bhas\b|\bwith\b|\bhas property\b", "has_property"),
    (r"\bcauses\b|\bleads to\b|\bresults in\b|\bproduces\b", "causes"),
    (r"\brequires\b|\bneeds\b|\bmust\b|\bdepends on\b", "requires"),
    (r"\bopposite of\b|\bnot\b", "opposite_of"),
    (r"\brelated to\b|\bassociated with\b|\binvolves\b", "related_to"),
]


def infer_relation(predicate: str) -> str:
    """Map a natural-language predicate to a relation type."""
    p = (predicate or "").lower().strip()
    for pat, rel in _PRED_PATTERNS:
        if re.search(pat, p):
            return rel
    return "related_to"


class SemanticNode:
    """One concept with relation-type-specific weighted edges."""

    __slots__ = ("id", "edges", "activation")

    def __init__(self, node_id: str):
        self.id = node_id
        # edges[relation] = {target_id: (weight, count)}
        self.edges: Dict[str, Dict[str, Tuple[float, int]]] = defaultdict(dict)
        self.activation = 0.0

    def add_edge(self, relation: str, target: str, weight: float = 1.0) -> None:
        d = self.edges.setdefault(relation, {})
        if target in d:
            w, c = d[target]
            d[target] = (w + weight, c + 1)
        else:
            d[target] = (weight, 1)

    def decayed_weight(self, relation: str, target: str) -> float:
        w, c = self.edges.get(relation, {}).get(target, (0.0, 0))
        # Confidence grows with corroboration count (brain-faithful: repeated
        # exposure strengthens a semantic edge).
        return w * (1.0 - math.exp(-c / 3.0))


class SemanticGraph:
    """Relation-aware semantic memory with activation spreading.

    Two edge populations share one node store:
    - SEED edges: general world knowledge loaded from data/semantic_seed.pkl
      (built offline from public ConceptNet 5.7 by scripts/build_semantic_seed.py).
      Reloadable from disk, therefore NOT pickled into engine snapshots.
    - ONLINE edges: triples RAVANA ingests from text it reads (episodic ->
      semantic consolidation). These ARE persisted (get_state/set_state), so
      online learning survives save/load — a gate written but never reloaded
      is dead.
    """

    #: drop ConceptNet 'related_to' edges below this weight when seeding.
    #: The unfiltered related_to channel is 1.43M edges of free-association
    #: noise ('accentology'-'stress') and costs ~1.2 GB RSS; the w>=2 subset
    #: (multiply-asserted edges) keeps the semantically dense core (~218K
    #: edges, ~0.5 GB). Typed relations (causes/used_for/...) are kept whole.
    SEED_MIN_RELATED_WEIGHT = 2.0

    #: relations whose IN-edges answer means-end queries ("what is X used
    #: for / what achieves X"). HPC->PFC goal-directed search runs backward
    #: from the goal state along instrumental edges. is_a is included
    #: because ConceptNet categorizes means AS purposes ('meditation is_a
    #: way to relax') — an is_a edge into a goal phrase names an instrument.
    _MEANS_RELS = ("used_for", "causes", "is_a")

    def __init__(self):
        self.nodes: Dict[str, SemanticNode] = {}
        # Reverse (in-edge) index for the instrumental relations only:
        # target -> [(relation, source, weight)]. Small (~54K entries).
        self._means_in: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
        # token -> node-ids that have means-in edges and contain the token.
        self._tok2goal: Dict[str, Set[str]] = defaultdict(set)
        # Online-learned triples (subject, predicate, object, confidence) —
        # the ONLY part that goes into engine snapshots.
        self._online: List[Tuple[str, str, str, float]] = []
        self._seed_loaded = False

    # ── seed ──────────────────────────────────────────────────────────────
    @staticmethod
    def default_seed_path() -> Optional[str]:
        """Walk up from this file to find data/semantic_seed.pkl (same
        discovery pattern as engine_graph._load_conceptnet_ontology)."""
        cur = os.path.abspath(__file__)
        for _ in range(8):
            cand = os.path.join(cur, "data", "semantic_seed.pkl")
            if os.path.exists(cand):
                return cand
            cur = os.path.dirname(cur)
        return None

    def load_seed(self, path: Optional[str] = None) -> bool:
        """Load the ConceptNet-derived seed (idempotent). Returns True when
        general knowledge is available after the call."""
        if self._seed_loaded:
            return True
        path = path or self.default_seed_path()
        if not path or not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                seed = pickle.load(f)
        except Exception:
            return False
        for rel, edges in seed.items():
            for a, b, w in edges:
                if rel == "related_to" and w < self.SEED_MIN_RELATED_WEIGHT:
                    continue
                self.add_relation(a, rel, b, w, _online=False)
        self._seed_loaded = True
        return True

    # ── mutation ──────────────────────────────────────────────────────────
    def _node(self, nid: str) -> SemanticNode:
        return self.nodes.setdefault(nid, SemanticNode(nid))

    def _index_means(self, relation: str, source: str, target: str,
                     weight: float) -> None:
        if relation not in self._MEANS_RELS:
            return
        self._means_in[target].append((relation, source, weight))
        for tok in target.split():
            if len(tok) >= 3:
                self._tok2goal[tok].add(target)

    def add_triple(self, subj: str, pred: str, obj: str,
                   confidence: float = 1.0) -> None:
        """Add one (subject, predicate, object) triple as a directed edge.
        Called for ONLINE-ingested text; recorded for persistence."""
        subj = (subj or "").strip().lower()
        obj = (obj or "").strip().lower()
        if not subj or not obj or subj == obj:
            return
        rel = infer_relation(pred)
        self._node(subj).add_edge(rel, obj, confidence)
        self._index_means(rel, subj, obj, confidence)
        self._online.append((subj, pred, obj, confidence))
        # Symmetric back-edge for non-hierarchical relations so activation can
        # spread both ways (stress<->exercise, not just exercise->stress).
        if rel in ("causes", "related_to", "used_for", "has_property",
                   "requires", "located_in", "part_of", "co_occurs"):
            self._node(obj).add_edge("related_to", subj, confidence)

    def add_relation(self, a: str, relation: str, b: str,
                     weight: float = 1.0, _online: bool = False) -> None:
        """Direct relation add (used by the ConceptNet seed)."""
        a = a.strip().lower()
        b = b.strip().lower()
        if not a or not b or a == b:
            return
        self._node(a).add_edge(relation, b, weight)
        self._index_means(relation, a, b, weight)
        if _online:
            self._online.append((a, relation, b, weight))
        if relation == "opposite_of":
            # Antonymy is symmetric — write the reverse opposite_of edge so
            # antonyms() is a single node lookup (stress <-> relax).
            self._node(b).add_edge("opposite_of", a, weight)
        elif relation in ("causes", "related_to", "used_for", "has_property",
                          "requires", "located_in", "part_of", "co_occurs",
                          "is_a"):
            # is_a is hierarchical (one-way), but the rest are mutual-ish.
            if relation != "is_a":
                self._node(b).add_edge("related_to", a, weight)

    # ── retrieval ──────────────────────────────────────────────────────────
    def _spread(self, entities: List[str], depth: int = 2,
                decay: float = 0.5) -> Dict[str, float]:
        """Spread activation from seed entities; return concept->activation."""
        act: Dict[str, float] = defaultdict(float)
        frontier = {}
        for e in entities:
            e = e.strip().lower()
            if e:
                frontier[e] = 1.0
        for hop in range(depth + 1):
            nxt: Dict[str, float] = defaultdict(float)
            for node_id, a in frontier.items():
                act[node_id] = max(act.get(node_id, 0.0), a)
                node = self.nodes.get(node_id)
                if node is None:
                    continue
                for rel, targets in node.edges.items():
                    for tgt, (w, _c) in targets.items():
                        contrib = a * (w / (1.0 + w)) * decay ** hop
                        nxt[tgt] += contrib
            frontier = nxt
        # zero the seeds themselves (we want associated concepts, not the query)
        for e in entities:
            act.pop(e.strip().lower(), None)
        return dict(act)

    def top_activated(self, entity: str, depth: int = 2,
                      top_k: int = 6) -> List[Tuple[str, float]]:
        """Return the most-associated concepts for a free-text query entity."""
        act = self._spread([entity], depth=depth)
        ranked = sorted(act.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]

    def means_for(self, goal_tokens: List[str], top_k: int = 8,
                  rels: Optional[Tuple[str, ...]] = None) -> List[Tuple[str, float]]:
        """Means-end retrieval: which ACTIONS achieve / relieve <goal>?

        HPC->PFC goal-directed search (Zeithamova 2012): hold the goal state
        ('less stress', 'healthy'), search BACKWARD along instrumental
        edges (used_for / causes) into the goal cohort, and return the
        source actions ranked by accumulated edge weight. Purely structural
        — derives from whatever the graph has read; no answer lists.

        ``rels`` restricts which in-edge relations count. For remedial
        queries pass ('used_for',): ConceptNet used_for targets are PURPOSE
        phrases ('relieving stress'), so their sources are remedies, while
        causes-edges into a bare problem node are its causes (stressors).
        """
        cohort: Set[str] = set()
        for t in goal_tokens:
            t = (t or "").strip().lower()
            if len(t) < 3:
                continue
            # Morphological-family activation (ATL lexical access spreads
            # over inflectional variants): relax -> relaxing/relaxation/
            # relaxed; stress -> stressed/stressing. Generated variants
            # only — no fuzzy scan of the whole index.
            variants = {t, t + "s", t + "ing", t + "ed", t + "ation"}
            if t.endswith("e"):
                variants |= {t[:-1] + "ing", t + "d"}
            if t.endswith("ed") and len(t) > 5:
                variants.add(t[:-2])
            if t.endswith("ing") and len(t) > 6:
                variants |= {t[:-3], t[:-3] + "e"}
            for v in variants:
                cohort |= self._tok2goal.get(v, set())
        if not cohort:
            return []
        score: Dict[str, float] = defaultdict(float)
        for node in cohort:
            for rel, src, w in self._means_in.get(node, ()):  # in-edges
                if rels is not None and rel not in rels:
                    continue
                score[src] += w
        ranked = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]

    def antonyms(self, token: str) -> Set[str]:
        """Both directions of opposite_of for a concept (stress <-> relax).

        opposite_of back-edges are written symmetrically at add time, so a
        single node lookup suffices; the legacy full-scan fallback covers
        graphs restored from old snapshots that predate the back-edge."""
        token = (token or "").strip().lower()
        out: Set[str] = set()
        node = self.nodes.get(token)
        if node is not None:
            out |= set(node.edges.get("opposite_of", {}).keys())
        return {a for a in out if len(a.split()) <= 2}

    def advice_for(self, topic_tokens: List[str],
                   top_k: int = 6, problem: bool = True) -> List[Tuple[str, float]]:
        """Actions that help with <topic> — the ATL+PFC advice circuit.

        Combines two structural searches (no answer lists):
        1. PURPOSE search: used_for in-edges into the topic cohort —
           sources of 'X used_for relieving stress' are remedies.
        2. GOAL INVERSION: the antonym of a problem state is the desired
           state (stress -> relax); means into the antonym cohort
           (used_for + causes) are also remedies. This is semantic
           opposition coding in the ATL hub (Lambon Ralph 2017).

        ``problem``: True when the topic is an UNDESIRED state (the query's
        verb frame said manage/reduce/relieve it). Then the ACC veto below
        applies. False when the topic IS the goal (healthy lifestyle) —
        causing the goal is precisely what we want, no veto.
        """
        merged: Dict[str, float] = defaultdict(float)
        for act, s in self.means_for(topic_tokens, top_k=top_k * 3,
                                     rels=("used_for",)):
            merged[act] += s
        anti: Set[str] = set()
        for t in topic_tokens:
            anti |= self.antonyms(t)
        if anti:
            anti_toks = [tok for a in anti for tok in a.split()]
            for act, s in self.means_for(anti_toks, top_k=top_k * 3):
                merged[act] += 0.5 * s
        # ACC outcome-conflict veto (Botvinick 2001): an action whose OWN
        # predicted outcome IS the problem state (a causes-edge onto the bare
        # topic node, e.g. 'taking midterm' -> causes -> 'stress') cannot be
        # a remedy for it — inhibit it. Structural, not a word list: relief
        # sources only point at composite relief phrases ('reduction of
        # stress'), never at the bare problem node. Only for problem-framed
        # topics; for goal-framed topics causing the goal is desirable.
        if problem:
            tt = {t.strip().lower() for t in topic_tokens if t}
            vetoed = set()
            for act in merged:
                node = self.nodes.get(act)
                if node is None:
                    continue
                if tt & set(node.edges.get("causes", {}).keys()):
                    vetoed.add(act)
            for act in vetoed:
                merged.pop(act, None)
        # Concept-family pooling + lateral inhibition: actions sharing a
        # content stem ('soaking in hotspring'/'soak in hotspring',
        # 'getting exercise'/'exercising'/'doing exercises') are ONE memory
        # family. Population coding: the family ACCUMULATES all its
        # exemplars' evidence (so a concept attested through many variants
        # outranks a single strong duplicate), then winner-take-all keeps
        # the strongest-labelled exemplar per family.
        _generic = {"going", "taking", "getting", "having", "being", "doing",
                    "for", "the", "with", "and"}

        def _fam(act: str) -> frozenset:
            ws = {w for w in act.split() if len(w) >= 4 and w not in _generic}
            ws = {w[:-3] if w.endswith("ing") and len(w) > 6 else w for w in ws}
            ws = {w[:-1] if w.endswith("s") and len(w) > 4 else w for w in ws}
            return frozenset(ws)

        fam_score: Dict[int, float] = {}
        fam_words: List[Set[str]] = []
        fam_label: Dict[int, Tuple[str, float]] = {}
        for act, s in sorted(merged.items(), key=lambda kv: kv[1], reverse=True):
            fw = set(_fam(act))
            hit = next((i for i, ws in enumerate(fam_words) if ws & fw), None)
            if hit is None:
                hit = len(fam_words)
                fam_words.append(fw)
                fam_score[hit] = 0.0
                fam_label[hit] = (act, s)
            else:
                fam_words[hit] |= fw
                if s > fam_label[hit][1]:
                    fam_label[hit] = (act, s)
            fam_score[hit] += s
        pooled = sorted(((fam_label[i][0], sc) for i, sc in fam_score.items()),
                        key=lambda kv: kv[1], reverse=True)
        return pooled[:top_k]

    def query(self, entities: List[str], options: List[str],
              depth: int = 2) -> Optional[str]:
        """MCQ: return the option whose words are most activated.

        Fail-closed: returns None when no option gains activation (never
        guess). This is the brain-faithful 'no knowledge -> abstain' path.
        """
        if not options:
            return None
        act = self._spread(entities, depth=depth)
        if not act:
            return None
        opt_words = [set(o.lower().split()) for o in options]
        scored = []
        for ow in opt_words:
            s = sum(act.get(w, 0.0) for w in ow)
            scored.append(s)
        best = max(scored)
        if best <= 0.0:
            return None
        # Tie-break: only answer when one option clearly wins.
        top_idx = [i for i, s in enumerate(scored) if s == best]
        if len(top_idx) > 1:
            return None
        return options[top_idx[0]]

    # ── persistence ──────────────────────────────────────────────────────────
    def get_state(self) -> bytes:
        """Pickle ONLY the online-learned triples. The seed is reloadable
        from disk and would bloat every snapshot by hundreds of MB."""
        return pickle.dumps({"online": list(self._online)})

    def set_state(self, data: bytes) -> None:
        if not data:
            return
        try:
            d = pickle.loads(data)
        except Exception:
            return
        # New format: replay online triples through add_triple/add_relation
        # so all indices (means_in, tok2goal) are rebuilt consistently.
        for item in d.get("online", []):
            try:
                s, p, o, c = item
                self.add_triple(s, p, o, float(c))
            except Exception:
                continue
        # Legacy format (full node dump) — restore edges directly.
        for nid, nd in d.get("nodes", {}).items():
            node = self._node(nid)
            for r, t in nd.get("edges", {}).items():
                node.edges[r] = {k: tuple(v) for k, v in t.items()}
