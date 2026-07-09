"""
DerivedOntology — brain-inspired replacement for the frontopolar literal gate.

The old gate hard-coded three frozen dicts
(_CATEGORY_AFFORDANCES / _CATEGORY_OF_SUBJECT / _PROPERTY_CATEGORIES) and a
person had to edit them per word. The brain does none of that:

  1. Concepts are distributed feature bundles (McRae, Cree & Seidenberg;
     Binder et al. 2016, "Toward a Brain-Based Componential Semantic
     Representation" — 65-dim attribute vectors, semantic neighborhoods by
     cosine similarity). "Color applies to X" is a GEOMETRIC fact: X's vector
     overlaps the color dimension. Nobody hard-codes `living: {color}`.
  2. Embodied simulators (Barsalou 1999): verifying "does X have property P"
     = can we co-activate the P simulator within X's simulator. This is a
     similarity test, not a dictionary lookup.
  3. Taxonomic inheritance is learned from feature-overlap / clustering
     (Rosch prototypes; Collins & Loftus spreading activation), not a frozen
     `_CATEGORY_OF_SUBJECT` table.

So this service derives affordances instead of listing them:

  * Property-probe vectors: a property P is characterized by a handful of
    canonical seed words; the probe is the (mean of their GloVe vectors).
    Cheap, computed once — not per word.
  * has_property(subject, P):
        cosine(glove(subject), P_probe) > theta
        OR a learned graph walk subject ->(isa/attribute)-> ... -> P.
  * No literal per-word tables. category_of() / has_property() are computed
    on demand, so "color into living" hacks become impossible.

The SAME manifold the brain uses is already in the codebase (GloVe vectors +
ConceptGraph with typed edges), so this routes through existing infrastructure
rather than inventing anything new.

Fallback policy: if GloVe is unavailable (glove_fn returns None) OR the subject
word is OOV, has_property() returns None ("cannot determine") and the caller is
expected to fall back to the legacy literal dicts. This keeps behaviour intact
on the rare no-GloVe path while making the derived path the default everywhere
GloVe is present.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Set, Tuple

# ── Property-probe seeds ────────────────────────────────────────────────────
# Each property P is described by a small set of canonical words that
# canonically possess it. The probe vector is the mean of their GloVe vectors
# (projected to the model dim, unit-normalized). This is the "color dimension"
# the brain re-activates when asked "does X have a color?".
#
# These seeds are NOT a category table — they are ~8 prototypical exemplars per
# property, used only to point the probe into the right region of the manifold.
# They are never consulted per subject word.
_PROPERTY_PROBE_SEEDS: Dict[str, List[str]] = {
    "color":  ["red", "blue", "green", "yellow", "black", "white", "purple",
               "orange", "color", "colour", "pink", "brown"],
    "colour": ["red", "blue", "green", "yellow", "black", "white", "purple",
               "orange", "color", "colour", "pink", "brown"],
    "mass":   ["heavy", "light", "kilogram", "weigh", "weight", "mass", "gram",
               "ton", "density"],
    "weight": ["heavy", "light", "kilogram", "weigh", "weight", "mass", "gram",
               "pound", "density"],
    "weigh":  ["heavy", "light", "kilogram", "weigh", "weight", "mass", "gram",
               "ton"],
    "weighs": ["heavy", "light", "kilogram", "weigh", "weight", "mass", "gram",
               "ton"],
    "size":   ["big", "small", "large", "tiny", "huge", "enormous", "size",
               "length", "width", "tall", "short"],
    "shape":  ["round", "square", "circular", "triangular", "oval", "shape",
               "geometric", "rectangular"],
    "texture": ["smooth", "rough", "soft", "hard", "bumpy", "texture", "silky",
                "coarse"],
    "temperature": ["hot", "cold", "warm", "cool", "temperature", "freezing",
                    "boiling", "heat"],
    "taste": ["sweet", "sour", "bitter", "salty", "tasty", "taste", "flavor",
              "flavour"],
    "smell": ["fragrant", "smelly", "odour", "scent", "smell", "aroma",
              "stench"],
    "sound": ["loud", "quiet", "noisy", "silent", "sound", "noise", "pitch",
              "volume"],
    "loudness": ["loud", "quiet", "noisy", "silent", "sound", "noise", "pitch"],
    "brightness": ["bright", "dim", "glow", "luminous", "dark", "brightness",
                   "radiant"],
    "duration": ["hour", "minute", "year", "second", "brief", "long",
                 "duration", "century", "moment", "instant"],
    "order": ["first", "last", "sequence", "rank", "order", "before", "after"],
    "cycle": ["repeat", "loop", "season", "rhythm", "cycle", "periodic"],
}

# Properties whose possession is what the gate actually cares about.
_KNOWN_PROPERTIES: Set[str] = set(_PROPERTY_PROBE_SEEDS.keys())

# Graph edge relation-types that license a taxonomic / partonomic inheritance
# (subject can possess P if an ancestor or attribute edge leads to P).
_INHERITANCE_EDGE_TYPES: Set[str] = {
    "isa", "is_a", "hypernym", "hyponym",
    "attribute", "has_property", "part_of", "composed_of",
}


class DerivedOntology:
    """Derived category/attribute knowledge over a GloVe manifold + concept graph.

    Compute-on-demand, never hand-edited per word. This is the single source of
    truth the engine routes through.
    """

    def __init__(
        self,
        glove_fn,
        graph=None,
        label_index: Optional[Dict[str, List[int]]] = None,
        theta: float = 0.12,
        max_graph_depth: int = 3,
    ):
        """
        Args:
            glove_fn: callable(word) -> Optional[np.ndarray (unit, model dim)].
                      Returns None when the word is OOV / GloVe absent.
            graph: Optional ConceptGraph for the isa/attribute inheritance walk.
            label_index: Optional dict mapping lowercased label -> list of node
                         ids (the engine's _concept_keywords). Required for the
                         graph walk; ignored if graph is None.
            theta: cosine threshold for "subject overlaps the property probe".
            max_graph_depth: BFS depth for the inheritance walk.
        """
        self.glove_fn = glove_fn
        self.graph = graph
        self.label_index = label_index or {}
        self.theta = float(theta)
        self.max_graph_depth = max_graph_depth
        self._probes: Dict[str, Optional[np.ndarray]] = {}
        self._build_probes()

    # ── probe construction ──────────────────────────────────────────────────
    def _build_probes(self):
        for prop, seeds in _PROPERTY_PROBE_SEEDS.items():
            vecs = []
            for s in seeds:
                v = self.glove_fn(s) if self.glove_fn else None
                if v is not None:
                    vecs.append(np.asarray(v, dtype=np.float32))
            if not vecs:
                self._probes[prop] = None
                continue
            stack = np.stack(vecs, axis=0)            # (n, dim)
            mean = stack.mean(axis=0)
            norm = float(np.linalg.norm(mean))
            self._probes[prop] = (mean / norm).astype(np.float32) if norm > 0 else None

    def probe(self, prop: str) -> Optional[np.ndarray]:
        """Return the unit probe vector for a property (or None if unknown)."""
        return self._probes.get(prop.lower())

    # ── core derived queries ─────────────────────────────────────────────────
    def has_property(self, subject: str, prop: str) -> Optional[bool]:
        """Does `subject` possess property `prop`?

        Returns:
            True  — subject's vector overlaps the property probe, or a learned
                    graph isa/attribute path reaches the property.
            False — subject clearly cannot possess it.
            None  — cannot determine (GloVe absent / subject OOV / probe absent);
                    caller should fall back to legacy literal tables.

        Brain analog: attempt to co-activate the property simulator inside the
        subject's simulator (cosine of feature vectors); if that fails, check
        whether an inherited simulator (isa/attribute edge) already carries it.
        """
        if prop.lower() not in _KNOWN_PROPERTIES:
            return None
        pvec = self._probes.get(prop.lower())
        svec = self.glove_fn(subject) if self.glove_fn else None
        # Both vectors available -> geometric test.
        if pvec is not None and svec is not None:
            cos = float(np.dot(np.asarray(svec, dtype=np.float32), pvec))
            if cos >= self.theta:
                return True
            # Not geometrically close — give the learned graph a chance to
            # inherit the property through isa/attribute edges.
            if self.graph is not None and self._graph_path_to_property(subject, prop):
                return True
            return False
        # Cannot determine via geometry -> signal caller to use legacy fallback.
        return None

    def _graph_path_to_property(self, subject: str, prop: str) -> bool:
        """BFS over the concept graph: does an isa/attribute path from subject
        reach a concept that itself possesses `prop` (recursive probe test)?"""
        if self.graph is None:
            return False
        subj_ids = self.label_index.get(subject.lower(), [])
        if not subj_ids:
            return False
        visited: Set[int] = set()
        frontier = list(subj_ids)
        depth = 0
        while frontier and depth < self.max_graph_depth:
            nxt = []
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)
                node = self.graph.nodes.get(nid)
                if node is None or not node.label:
                    continue
                # A direct ancestor/attribute that itself has the property => inherit.
                if node.label.lower() != subject.lower():
                    res = self.has_property(node.label, prop)
                    if res is True:
                        return True
                for _, edge in self.graph.get_outgoing(nid):
                    if edge.relation_type in _INHERITANCE_EDGE_TYPES:
                        if edge.target not in visited:
                            nxt.append(edge.target)
                for src, edge in self.graph.get_incoming(nid):
                    if edge.relation_type in _INHERITANCE_EDGE_TYPES:
                        if src not in visited:
                            nxt.append(src)
            frontier = nxt
            depth += 1
        return False

    def category_of(self, subject: str, category_seeds: Dict[str, List[str]],
                    theta: Optional[float] = None) -> Optional[str]:
        """Infer the (soft) category of `subject` by cosine to category
        centroids derived from seed exemplars — NOT a frozen lookup table.

        Args:
            subject: word to classify.
            category_seeds: {category_name: [exemplar words]}; centroids are
                            computed on demand from GloVe.
            theta: optional override for the membership threshold.

        Returns the best-matching category name if cosine >= theta, else None.
        """
        t = self.theta if theta is None else theta
        svec = self.glove_fn(subject) if self.glove_fn else None
        if svec is None:
            return None
        svec = np.asarray(svec, dtype=np.float32)
        best_cat, best_cos = None, -2.0
        for cat, seeds in category_seeds.items():
            cvec = self._centroid(seeds)
            if cvec is None:
                continue
            cos = float(np.dot(svec, cvec))
            if cos > best_cos:
                best_cos, best_cat = cos, cat
        if best_cat is not None and best_cos >= t:
            return best_cat
        return None

    def _centroid(self, seeds: List[str]) -> Optional[np.ndarray]:
        vecs = [np.asarray(self.glove_fn(s), dtype=np.float32)
                for s in seeds]
        vecs = [v for v in vecs if v is not None]
        if not vecs:
            return None
        mean = np.stack(vecs, axis=0).mean(axis=0)
        norm = float(np.linalg.norm(mean))
        return (mean / norm).astype(np.float32) if norm > 0 else None

    # ── gate helper (the frontopolar feasibility test) ──────────────────────
    @staticmethod
    def known_properties() -> Set[str]:
        return set(_KNOWN_PROPERTIES)
