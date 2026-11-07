"""Schema-consistency-gated knowledge intake.

The brain's answer to "how do you ingest new knowledge without corrupting what
you already know" is a WRITE GATE, not a write. New information is matched
against existing structure first, and the match quality decides the route:

  * schema-consistent  -> fast merge into the structured store
  * schema-inconsistent -> buffer (episodic) and interleave during replay
  * contradictory       -> revise the incumbent by entrenchment, or buffer both

Grounding (see docs/BRAIN_TO_RAVANA_LEARNING.md for the full mapping):

  * Kintsch (1988, 1994) construction-integration: comprehension is
    construction + integration, and "learning from a text" is a matter of how
    elaborate a SITUATION MODEL is built -- not of storing the text. This
    module is the write side of that claim: it stores propositions, never text.
  * Tse et al. (2011, Science) 333:891-895; McClelland (2013, JEPG) 142(4):
    schema-consistent information is learned in one trial without interference;
    schema-inconsistent information must be interleaved or it catastrophically
    interferes (McCloskey & Cohen 1989). -> the consistency gate below.
  * Kumaran, Hassabis & McClelland (2016, TiCS) 20(7):512-534: replay is
    goal-dependent and "neocortical learning can be rapid for information that
    is consistent with known structure". -> fast path is not a shortcut, it is
    the mechanism.
  * Craik & Lockhart (1972); Craik & Tulving (1975): trace persistence is a
    positive function of the DEPTH and ELABORATION of processing, not of
    exposure time. -> elaboration score drives the retention prior.
  * Karpicke & Roediger (2008, Science) 319:966-968: retrieval is itself the
    learning event; re-reading without retrieval is the weakest condition.
    -> every intake schedules a self-retrieval at an expanding interval.
  * Alchourron, Gardenfors & Makinson (1985): revision removes the minimum
    necessary to restore consistency, guided by an entrenchment ordering.
    -> contradiction resolution below.
  * Anderson, Bjork & Bjork (1994, JEP:LMC) 20(5):1063-1087: retrieving one
    item suppresses its competitors (retrieval-induced forgetting).
    -> lateral inhibition for near-duplicate nodes.
  * Nader, Schafe & LeDoux (2000, Nature) 406:722-726: a reactivated memory
    becomes labile and must be re-stabilised, and the update is gated by
    prediction error (Sevenster et al. 2013). -> the revise route opens a
    bounded labile window rather than appending a duplicate.
  * Johnson, Hashtroudi & Lindsay (1993, Psych Bull) 114(1):3-28: source is
    not a stored tag but an attribution computed at remember-time from
    features of the record. -> Provenance.trust() is computed, not stored.

Everything here is a deterministic graph operation: no gradient training, no
retraining from scratch, CPU-resident.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ravana.web.openie import Fact


# ─────────────────────────────────────────────────────────────────────────────
# Provenance: source monitoring as a computed attribution, not a stored tag
# ─────────────────────────────────────────────────────────────────────────────

# Source reliability priors. Deliberately coarse: the brain does not store a
# trust label, it re-derives trust from record features at query time
# (Johnson et al. 1993). These are the prior term of that derivation.
SOURCE_RELIABILITY: Dict[str, float] = {
    "user": 0.9,          # a first-person disclosure about the user
    "seed": 0.8,          # curated seed knowledge
    "web": 0.5,           # an arbitrary web page
    "inference": 0.4,     # derived by RAVANA, not read
    "retrieved": 0.45,    # recalled from an earlier read
}


@dataclass
class Provenance:
    """Where a fact came from, and what was done to it on the way in.

    Mirrors the source-monitoring framework: the record keeps the features
    (source, modality, operations) from which trust is later ATTRIBUTED, and
    keeps `inferred` distinct from `read` (reality monitoring).
    """

    source: str = ""
    source_type: str = "web"
    url: str = ""
    ingested_at: float = field(default_factory=time.time)
    extractor: str = "openie"
    # Encoding operations applied (e.g. "copula-splice", "coreference"). The
    # brain's reality-monitoring distinction is exactly "which operations
    # produced this record".
    operations: Tuple[str, ...] = ()
    corroboration: float = 0.0     # 0..1, fraction of independent sources seen
    reliability: Optional[float] = None

    @property
    def is_inferred(self) -> bool:
        return self.source_type in ("inference", "retrieved")

    def base_reliability(self) -> float:
        if self.reliability is not None:
            return float(self.reliability)
        return SOURCE_RELIABILITY.get(self.source_type, 0.5)

    def trust(self, recency_halflife: float = 604800.0) -> float:
        """Compute trust at query time from record features.

        Not a stored tag: source reliability + corroboration + recency, with
        internally-generated records discounted (reality monitoring).
        """
        age = max(0.0, time.time() - self.ingested_at)
        # recency_halflife <= 0 means "no recency credit at all" (used by the
        # tests and by a caller that wants pure source+corroboration trust).
        recency = math.exp(-age / recency_halflife) if recency_halflife > 0 else 0.0
        raw = (0.45 * self.base_reliability()
               + 0.35 * float(self.corroboration)
               + 0.20 * recency)
        if self.is_inferred:
            raw *= 0.85
        return clamp01(raw)

    def as_metadata(self) -> Dict[str, Any]:
        """Durable, serialisable form (for edge.source_metadata)."""
        return {
            "source": self.source,
            "source_type": self.source_type,
            "url": self.url,
            "ingested_at": self.ingested_at,
            "extractor": self.extractor,
            "operations": list(self.operations),
            "corroboration": self.corroboration,
        }

    @classmethod
    def from_metadata(cls, meta: Optional[Dict[str, Any]]) -> "Provenance":
        if not meta:
            return cls()
        return cls(
            source=str(meta.get("source", "")),
            source_type=str(meta.get("source_type", "web")),
            url=str(meta.get("url", "")),
            ingested_at=float(meta.get("ingested_at", time.time())),
            extractor=str(meta.get("extractor", "openie")),
            operations=tuple(meta.get("operations", ()) or ()),
            corroboration=float(meta.get("corroboration", 0.0) or 0.0),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Verdict types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Contradiction:
    """Two values compete for the same (subject, predicate) slot."""

    subject: str
    predicate: str
    old_value: str
    new_value: str
    old_entrenchment: float
    new_entrenchment: float
    resolution: str = "pending"   # pending | revise | buffer

    @property
    def winner_is_new(self) -> bool:
        return self.new_entrenchment > self.old_entrenchment


@dataclass
class IntakeVerdict:
    """The gate's decision for one fact."""

    fact: Fact
    consistency: float          # 0..1  does it fill slots that already exist?
    elaboration: float          # 0..1  how much existing structure it touches?
    entrenchment: float         # 0..1  how supported is the incumbent?
    salience: float             # 0..1  neuromodulatory write gate
    route: str                  # fast_merge | episodic_buffer | revise
    contradiction: Optional[Contradiction]
    retention_prior: float      # 0..1  elaboration-driven durability prior
    review_interval: int        # turns until the scheduled self-retrieval
    provenance: Provenance
    reason: str

    @property
    def effective_write(self) -> float:
        """The gated update magnitude actually applied.

        Salience gates plasticity: a predictable, low-surprise fact barely
        moves any weight (Friston 2010; Chang, Dell & Bock 2006). This is the
        scalar the engine multiplies its confidence/weight delta by.
        """
        return clamp01(self.salience * (0.35 + 0.65 * self.elaboration))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def clamp01(x: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if v != v:  # NaN
        return 0.0
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _norm_label(s: str) -> str:
    return " ".join((s or "").lower().strip().split())


def _token_set(s: str) -> set:
    return {t for t in _norm_label(s).split() if len(t) > 2}


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# Expanding-interval retrieval practice schedule (Cepeda et al. 2006, 2008;
# Karpicke & Roediger 2008). The interval grows with each successful retrieval
# and is capped so a fact is revisited a bounded number of times.
REVIEW_SCHEDULE: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)


def review_interval_for(prior_retention: float, reviews_done: int = 0) -> int:
    """Next inter-retrieval interval in turns.

    Better-elaborated facts get a longer interval (they are more durable, so
    the next retrieval can be pushed further out). Poorly-elaborated facts are
    revisited sooner -- the spacing effect.
    """
    idx = min(max(0, int(reviews_done)), len(REVIEW_SCHEDULE) - 1)
    base = REVIEW_SCHEDULE[idx]
    # Elaboration scales the interval: a well-elaborated fact is more durable,
    # so its next retrieval can be pushed further out (spacing effect). The
    # factor is deliberately strong enough that the FIRST interval already
    # differs between a shallow and a deep encoding.
    scaled = base * (0.5 + 2.0 * clamp01(prior_retention))
    return max(1, int(round(scaled)))


def edge_type_for_relation(relation: str) -> str:
    """Map an OpenIE relation onto the graph's relation_type vocabulary."""
    return {
        "is_a": "is_a",
        "has_property": "has_property",
        "located_in": "located_in",
        "part_of": "part_of",
        "causes": "causes",
        "related_to": "semantic",
    }.get(relation, "semantic")


# ─────────────────────────────────────────────────────────────────────────────
# The gate
# ─────────────────────────────────────────────────────────────────────────────

class SchemaConsistencyGate:
    """Decides, per fact, whether to fast-merge, buffer, or revise.

    The gate is a pure function of (fact, current graph structure, provenance,
    salience). It never writes. `KnowledgeIntake` applies the verdict.
    """

    def __init__(self,
                 fast_merge_threshold: float = 0.55,
                 revise_margin: float = 0.10,
                 revise_min_entrenchment: float = 0.35):
        self.fast_merge_threshold = float(fast_merge_threshold)
        self.revise_margin = float(revise_margin)
        # Absolute floor for a challenger: a weakly-supported new value may not
        # overturn an incumbent no matter how weak the incumbent is. Without
        # this, a noisy extractor can displace a real value with a bad one.
        self.revise_min_entrenchment = float(revise_min_entrenchment)

    # ── structure access (duck-typed over GraphEngine) ──────────────────────

    @staticmethod
    def _node_id(ge, label: str) -> Optional[int]:
        labels = getattr(ge, "_all_labels", None)
        if not labels:
            return None
        return labels.get(_norm_label(label))

    @staticmethod
    def _node(ge, nid: Optional[int]):
        if nid is None:
            return None
        graph = getattr(ge, "graph", None)
        return graph.get_node(nid) if graph is not None else None

    @classmethod
    def _out_edges(cls, ge, nid: int) -> List[Any]:
        graph = getattr(ge, "graph", None)
        if graph is None:
            return []
        try:
            return [e for _, e in graph.get_outgoing(nid)]
        except Exception:
            return []

    @classmethod
    def _in_edges(cls, ge, nid: int) -> List[Any]:
        graph = getattr(ge, "graph", None)
        if graph is None:
            return []
        try:
            return [e for _, e in graph.get_incoming(nid)]
        except Exception:
            return []

    @classmethod
    def _degree(cls, ge, nid: int) -> int:
        return len(cls._out_edges(ge, nid)) + len(cls._in_edges(ge, nid))

    @classmethod
    def _edge_relation(cls, edge) -> str:
        rt = getattr(edge, "relation_type", None)
        if rt:
            return str(rt)
        return str(getattr(edge, "edge_type", "semantic") or "semantic")

    @classmethod
    def _edge_confidence(cls, edge) -> float:
        try:
            return clamp01(edge.confidence)
        except Exception:
            return 0.5

    @staticmethod
    def _target_label(ge, edge) -> str:
        graph = getattr(ge, "graph", None)
        tgt = getattr(edge, "target", None)
        node = graph.get_node(tgt) if (graph is not None and tgt is not None) else None
        return _norm_label(getattr(node, "label", "") if node is not None else "")

    @staticmethod
    def _source_label(ge, edge) -> str:
        graph = getattr(ge, "graph", None)
        src = getattr(edge, "source", None)
        node = graph.get_node(src) if (graph is not None and src is not None) else None
        return _norm_label(getattr(node, "label", "") if node is not None else "")

    # ── the three scores ─────────────────────────────────────────────────────

    def consistency(self, fact: Fact, ge) -> float:
        """How well does this fact fill slots that ALREADY exist?

        Tse et al. (2011); McClelland (2013): a fact that fits existing
        structure is learned in one trial. A fact that fits nothing must be
        interleaved. The score is the fraction of structural preconditions
        already satisfied.
        """
        subj_id = self._node_id(ge, fact.subject)
        obj_id = self._node_id(ge, fact.obj)
        edge_type = self._edge_type_for(fact)

        s = 0.0
        # 1. the subject is already a known concept
        if subj_id is not None:
            s += 0.30
        # 2. the object is already a known concept
        if obj_id is not None:
            s += 0.20
        # 3. the subject already participates in this relation type -- i.e.
        #    there is a slot for it to fill.
        if subj_id is not None:
            same_rel = [e for e in self._out_edges(ge, subj_id)
                        if self._edge_relation(e) == edge_type]
            if same_rel:
                s += 0.30
        # 4. both endpoints known and already connected (the strongest signal:
        #    the fact is a re-sighting or a refinement of existing structure)
        if subj_id is not None and obj_id is not None:
            graph = getattr(ge, "graph", None)
            if graph is not None and graph.get_edge(subj_id, obj_id) is not None:
                s += 0.10
        return clamp01(s)

    def elaboration(self, fact: Fact, ge) -> float:
        """How much existing structure does this fact touch?

        Craik & Lockhart (1972); Craik & Tulving (1975): durability is a
        function of the depth and elaboration of processing. Operationally:
        how many existing connections are activated, and does the fact bridge
        two otherwise separate regions?
        """
        subj_id = self._node_id(ge, fact.subject)
        obj_id = self._node_id(ge, fact.obj)
        if subj_id is None and obj_id is None:
            return 0.0

        subj_deg = self._degree(ge, subj_id) if subj_id is not None else 0
        obj_deg = self._degree(ge, obj_id) if obj_id is not None else 0

        # Shared neighbourhood: the fact closes a triangle with existing
        # knowledge -- the cheapest possible elaboration.
        shared = 0
        bridging = 0.0
        if subj_id is not None and obj_id is not None:
            s_nbrs = {e.target for e in self._out_edges(ge, subj_id)}
            s_nbrs |= {e.source for e in self._in_edges(ge, subj_id)}
            o_nbrs = {e.target for e in self._out_edges(ge, obj_id)}
            o_nbrs |= {e.source for e in self._in_edges(ge, obj_id)}
            shared = len(s_nbrs & o_nbrs)
            # Bridging: both endpoints known but NOT already linked -> the fact
            # connects two regions that were separate.
            graph = getattr(ge, "graph", None)
            if graph is not None and graph.get_edge(subj_id, obj_id) is None:
                bridging = 0.25

        # Saturation: a 4-degree node is "well connected"; beyond that the
        # marginal elaboration falls off (bounded working memory, Cowan 2001).
        deg_term = clamp01((subj_deg + obj_deg) / 8.0)
        shared_term = clamp01(shared / 3.0)
        return clamp01(0.55 * deg_term + 0.30 * shared_term + bridging)

    def entrenchment(self, fact: Fact, ge, provenance: Provenance) -> float:
        """How dispensable is the incumbent? (AGM entrenchment ordering)

        Support count, source reliability, corroboration and recency. This is
        the score that decides a contradiction: the more entrenched value wins
        unless the challenger clears `revise_margin` above it.
        """
        subj_id = self._node_id(ge, fact.subject)
        obj_id = self._node_id(ge, fact.obj)
        edge_type = self._edge_type_for(fact)

        support = 0.0
        if subj_id is not None:
            for e in self._out_edges(ge, subj_id):
                if self._edge_relation(e) != edge_type:
                    continue
                conf = self._edge_confidence(e)
                # The incumbent is the strongest same-slot edge.
                if conf > support:
                    support = conf
        # Corroboration is read off the edge's stored provenance when present.
        corroboration = 0.0
        if subj_id is not None and obj_id is not None:
            graph = getattr(ge, "graph", None)
            e = graph.get_edge(subj_id, obj_id) if graph is not None else None
            if e is not None:
                meta = getattr(e, "source_metadata", None) or {}
                try:
                    corroboration = clamp01(len(meta.get("contexts", []) or [])) * 0.34
                except Exception:
                    corroboration = 0.0
        trust = provenance.trust()
        return clamp01(0.45 * support + 0.35 * trust + 0.20 * corroboration)

    def challenger_entrenchment(self, fact: Fact, ge, provenance: Provenance) -> float:
        """Entrenchment of the NEW value, for contradiction comparison.

        Combines the extractor's own confidence in the triple with the source
        trust and whether the value is already a connected concept. A
        extractor that is unsure of the triple must not be able to overturn an
        established value.
        """
        trust = provenance.trust()
        obj_id = self._node_id(ge, fact.obj)
        known = 0.0
        if obj_id is not None:
            known = clamp01(self._degree(ge, obj_id) / 6.0) * 0.3
        return clamp01(0.45 * clamp01(fact.confidence)
                       + 0.35 * trust + known)

    # ── contradiction ────────────────────────────────────────────────────────

    def detect_contradiction(self, fact: Fact, ge,
                             provenance: Provenance) -> Optional[Contradiction]:
        """Two values competing for the same (subject, predicate) slot.

        Returns None when there is no incumbent, or when the incumbent IS this
        value (a re-sighting, not a contradiction).
        """
        subj_id = self._node_id(ge, fact.subject)
        if subj_id is None:
            return None
        edge_type = self._edge_type_for(fact)
        new_label = _norm_label(fact.obj)
        for e in self._out_edges(ge, subj_id):
            if self._edge_relation(e) != edge_type:
                continue
            old_label = self._target_label(ge, e)
            if not old_label or old_label == new_label:
                continue
            # Same slot, different value -> contradiction.
            old_ent = self._edge_confidence(e)
            new_ent = self.challenger_entrenchment(fact, ge, provenance)
            return Contradiction(
                subject=_norm_label(fact.subject),
                predicate=edge_type,
                old_value=old_label,
                new_value=new_label,
                old_entrenchment=old_ent,
                new_entrenchment=new_ent,
            )
        return None

    # ── routing ──────────────────────────────────────────────────────────────

    def retention_prior(self, consistency: float, elaboration: float) -> float:
        """Elaboration-driven durability prior (Craik & Lockhart)."""
        return clamp01(0.45 * elaboration + 0.35 * consistency + 0.20)

    def route(self, consistency: float, contradiction: Optional[Contradiction],
              salience: float,
              provenance: Optional[Provenance] = None) -> Tuple[str, str]:
        """Pick the route and a human-readable reason."""
        if contradiction is not None:
            margin = contradiction.new_entrenchment - contradiction.old_entrenchment
            strong_enough = (contradiction.new_entrenchment
                             >= self.revise_min_entrenchment)
            # Source-monitoring guard: a single arbitrary page does not get to
            # overturn an incumbent. Revision requires either a high-reliability
            # source (the user, curated seed) or independent corroboration.
            can_overturn = True
            if provenance is not None:
                can_overturn = (provenance.base_reliability() >= 0.7
                                or provenance.corroboration >= 0.5)
            if can_overturn and strong_enough and margin > self.revise_margin \
                    and salience >= 0.3:
                # Reconsolidation: the incumbent is labile for a bounded
                # window, and the challenger clears the bar.
                return "revise", (
                    f"contradiction on ({contradiction.subject}, "
                    f"{contradiction.predicate}): challenger entrenchment "
                    f"{contradiction.new_entrenchment:.2f} clears incumbent "
                    f"{contradiction.old_entrenchment:.2f} by {margin:.2f}")
            if not can_overturn:
                return "episodic_buffer", (
                    f"contradiction on ({contradiction.subject}, "
                    f"{contradiction.predicate}) held: a single uncorroborated "
                    f"source may not overturn an incumbent")
            if not strong_enough:
                return "episodic_buffer", (
                    f"contradiction on ({contradiction.subject}, "
                    f"{contradiction.predicate}) rejected: challenger "
                    f"{contradiction.new_entrenchment:.2f} is below the "
                    f"{self.revise_min_entrenchment:.2f} support floor")
            return "episodic_buffer", (
                f"contradiction on ({contradiction.subject}, "
                f"{contradiction.predicate}) not resolved "
                f"(margin {margin:.2f}); both held for replay")
        if consistency >= self.fast_merge_threshold:
            return "fast_merge", (
                f"schema-consistent ({consistency:.2f} >= "
                f"{self.fast_merge_threshold:.2f}); one-trial merge")
        return "episodic_buffer", (
            f"schema-inconsistent ({consistency:.2f} < "
            f"{self.fast_merge_threshold:.2f}); buffer for interleaved replay")

    def evaluate(self, fact: Fact, ge, provenance: Optional[Provenance] = None,
                 salience: float = 1.0) -> IntakeVerdict:
        """Full gate evaluation for one fact."""
        prov = provenance or Provenance()
        sal = clamp01(salience)
        c = self.consistency(fact, ge)
        e = self.elaboration(fact, ge)
        ent = self.entrenchment(fact, ge, prov)
        con = self.detect_contradiction(fact, ge, prov)
        route, reason = self.route(c, con, sal, prov)
        prior = self.retention_prior(c, e)
        return IntakeVerdict(
            fact=fact, consistency=c, elaboration=e, entrenchment=ent,
            salience=sal, route=route, contradiction=con,
            retention_prior=prior,
            review_interval=review_interval_for(prior, 0),
            provenance=prov, reason=reason)

    # ── relation mapping ─────────────────────────────────────────────────────

    @staticmethod
    def _edge_type_for(fact: Fact) -> str:
        """Map an OpenIE relation onto the graph's relation_type vocabulary."""
        return edge_type_for_relation(fact.relation)


# ─────────────────────────────────────────────────────────────────────────────
# The orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeIntake:
    """Apply gate verdicts to the two-tier store.

    Routes:
      * fast_merge      -> typed edge in the ConceptGraph (slow store), with
                           provenance attached and the update magnitude gated
                           by salience x elaboration.
      * episodic_buffer -> FactTriple in the HippocampalBuffer (fast store),
                           with a scheduled self-retrieval.
      * revise          -> reconsolidation: open a bounded labile window on the
                           incumbent edge, apply the challenger, record the
                           revision, and protect the neighbours.

    Also owns the retrieval-practice schedule: every accepted fact gets a
    due-turn, and `due_for_review()` returns the ones whose interval elapsed.
    """

    def __init__(self, gate: Optional[SchemaConsistencyGate] = None,
                 ge=None, buffer=None, clock: Optional[Any] = None):
        self.gate = gate or SchemaConsistencyGate()
        self.ge = ge
        self.buffer = buffer
        self._now = clock or (lambda: time.time())
        # (subject, predicate, object) -> {"due": turn, "reviews": n, ...}
        self._review_queue: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._turn = 0
        self.stats: Dict[str, int] = {
            "fast_merge": 0, "episodic_buffer": 0, "revise": 0,
            "fallback_merge": 0, "rejected": 0, "reviewed": 0,
        }

    # ── turn bookkeeping ─────────────────────────────────────────────────────

    def advance_turn(self) -> None:
        self._turn += 1

    @property
    def turn(self) -> int:
        return self._turn

    # ── write ────────────────────────────────────────────────────────────────

    def intake(self, fact: Fact, provenance: Optional[Provenance] = None,
               salience: float = 1.0) -> IntakeVerdict:
        if self.ge is None:
            raise RuntimeError("KnowledgeIntake requires a graph engine")
        v = self.gate.evaluate(fact, self.ge, provenance, salience)
        if v.route == "fast_merge":
            self._fast_merge(v)
        elif v.route == "revise":
            self._revise(v)
        else:
            self._buffer(v)
        self._schedule_review(v)
        return v

    def _write_edge(self, v: IntakeVerdict, confidence: float) -> bool:
        """Write one typed edge with durable provenance. Returns success."""
        ge = self.ge
        graph = getattr(ge, "graph", None)
        if graph is None:
            return False
        subj_id = self.gate._node_id(ge, v.fact.subject)
        obj_id = self.gate._node_id(ge, v.fact.obj)
        if subj_id is None or obj_id is None:
            return False
        edge_type = self.gate._edge_type_for(v.fact)
        existing = graph.get_edge(subj_id, obj_id)
        delta = v.effective_write
        if existing is not None:
            # Locate-then-edit (Meng et al. 2022): a bounded update to THIS edge
            # and its neighbourhood only -- never a global overwrite.
            new_conf = clamp01(self.gate._edge_confidence(existing)
                               + 0.05 * delta)
            try:
                existing.confidence = new_conf
            except Exception:
                pass
            meta = getattr(existing, "source_metadata", None)
            if isinstance(meta, dict):
                meta.update(v.provenance.as_metadata())
                meta["last_revised"] = self._now()
                meta["revisions"] = int(meta.get("revisions", 0)) + 1
            return True
        edge = graph.add_edge(
            source=subj_id, target=obj_id, weight=0.5,
            relation_type=edge_type, confidence=clamp01(confidence))
        if edge is not None and hasattr(edge, "source_metadata"):
            edge.source_metadata.update(v.provenance.as_metadata())
            edge.source_metadata["revisions"] = 0
        return edge is not None

    def _fast_merge(self, v: IntakeVerdict) -> None:
        # Retention prior (elaboration) sets the initial confidence; salience
        # scales how far toward it we move on this turn.
        target = v.retention_prior
        current = 0.5
        subj_id = self.gate._node_id(self.ge, v.fact.subject)
        obj_id = self.gate._node_id(self.ge, v.fact.obj)
        graph = getattr(self.ge, "graph", None)
        if graph is not None and subj_id is not None and obj_id is not None:
            e = graph.get_edge(subj_id, obj_id)
            if e is not None:
                current = self.gate._edge_confidence(e)
        confidence = clamp01(current + (target - current) * v.effective_write)
        if self._write_edge(v, confidence):
            self.stats["fast_merge"] += 1
        else:
            self.stats["rejected"] += 1

    def _revise(self, v: IntakeVerdict) -> None:
        """Reconsolidation: update the incumbent in place, bounded.

        Nader et al. (2000): reactivation makes the trace labile; the update is
        applied inside the window and the rest of the graph is untouched.
        """
        con = v.contradiction
        assert con is not None
        subj_id = self.gate._node_id(self.ge, con.subject)
        if subj_id is None:
            self._buffer(v)
            return
        # Locate the incumbent edge (same slot, old value) and demote it; then
        # write the challenger. The old value is NOT deleted -- it is kept as
        # a superseded record so "you said X before" stays answerable.
        edge_type = self.gate._edge_type_for(v.fact)
        graph = getattr(self.ge, "graph", None)
        for e in self.gate._out_edges(self.ge, subj_id):
            if self.gate._edge_relation(e) != edge_type:
                continue
            if self.gate._target_label(self.ge, e) != con.old_value:
                continue
            try:
                e.confidence = clamp01(self.gate._edge_confidence(e) * 0.5)
            except Exception:
                pass
            meta = getattr(e, "source_metadata", None)
            if isinstance(meta, dict):
                meta["superseded_by"] = con.new_value
                meta["superseded_at"] = self._now()
            break
        if self._write_edge(v, clamp01(0.5 + 0.5 * v.retention_prior)):
            self.stats["revise"] += 1
        else:
            self._buffer(v)

    def _buffer(self, v: IntakeVerdict) -> None:
        """Route to the fast episodic store.

        If no buffer is attached, retain the fact at LOW confidence instead of
        dropping it: the confidence encodes how poorly it integrated, so sleep
        pruning and the low-quality-edge gate will remove it if it never
        consolidates. This is the brain-faithful fallback -- with hippocampal
        damage, semantic learning still proceeds, slowly and poorly.
        """
        if self.buffer is not None:
            try:
                self.buffer.store(
                    v.fact.subject, v.fact.relation, v.fact.obj,
                    confidence=clamp01(0.4 + 0.6 * v.retention_prior),
                    aliases=list(_token_set(v.fact.obj))[:3],
                )
                self.stats["episodic_buffer"] += 1
                return
            except Exception:
                pass
        if self._write_edge(v, clamp01(0.25 * v.retention_prior)):
            self.stats["fallback_merge"] += 1
        else:
            self.stats["rejected"] += 1

    # ── retrieval practice ───────────────────────────────────────────────────

    def _schedule_review(self, v: IntakeVerdict) -> None:
        key = (_norm_label(v.fact.subject), v.fact.relation,
               _norm_label(v.fact.obj))
        rec = self._review_queue.get(key)
        if rec is None:
            self._review_queue[key] = {
                "due": self._turn + v.review_interval,
                "interval": v.review_interval,
                "reviews": 0,
                "retention": v.retention_prior,
                "provenance": v.provenance,
            }
        else:
            # Already scheduled: keep the earliest due date, refresh retention.
            rec["retention"] = max(rec.get("retention", 0.0), v.retention_prior)

    def due_for_review(self, limit: int = 8) -> List[Tuple[str, str, str]]:
        """Facts whose retrieval-practice interval has elapsed.

        Retrieval practice, not re-reading, is the learning event
        (Karpicke & Roediger 2008). The engine should attempt to answer from
        the graph alone and only consult the source on failure.
        """
        due = [(k, rec) for k, rec in self._review_queue.items()
               if rec["due"] <= self._turn]
        due.sort(key=lambda kv: (kv[1]["retention"], kv[1]["due"]))
        return [k for k, _ in due[:limit]]

    def record_review(self, key: Tuple[str, str, str], recalled: bool) -> None:
        """Close a retrieval-practice trial and reschedule.

        Success expands the interval (spacing effect); failure resets it to
        the shortest interval, because a fact that could not be recalled is
        exactly the one that needs the next attempt soonest.
        """
        rec = self._review_queue.get(key)
        if rec is None:
            return
        if recalled:
            rec["reviews"] = int(rec.get("reviews", 0)) + 1
            rec["retention"] = clamp01(rec.get("retention", 0.5) + 0.1)
            nxt = review_interval_for(rec["retention"], rec["reviews"])
        else:
            rec["reviews"] = 0
            rec["retention"] = clamp01(rec.get("retention", 0.5) * 0.6)
            nxt = REVIEW_SCHEDULE[0]
        rec["interval"] = nxt
        rec["due"] = self._turn + nxt
        self.stats["reviewed"] += 1

    def drop_review(self, key: Tuple[str, str, str]) -> None:
        self._review_queue.pop(key, None)

    # ── serialisation ────────────────────────────────────────────────────────

    def get_state(self) -> Dict[str, Any]:
        return {
            "turn": self._turn,
            "stats": dict(self.stats),
            "review_queue": {
                "|".join(k): {
                    "due": rec["due"], "interval": rec["interval"],
                    "reviews": rec["reviews"], "retention": rec["retention"],
                }
                for k, rec in self._review_queue.items()
            },
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self._turn = int(state.get("turn", 0) or 0)
        self.stats = dict(state.get("stats", self.stats))
        self._review_queue = {}
        for flat, rec in (state.get("review_queue", {}) or {}).items():
            parts = tuple(flat.split("|"))
            if len(parts) != 3:
                continue
            self._review_queue[parts] = {
                "due": int(rec.get("due", 0)),
                "interval": int(rec.get("interval", 1)),
                "reviews": int(rec.get("reviews", 0)),
                "retention": float(rec.get("retention", 0.5)),
                "provenance": Provenance(),
            }
