"""Assemble and grade multi-step causal explanation chains.

Implements stages 2-3 of `docs/BRAIN_EXPLANATION_AND_REASONING.md` §5.2.

WHY
---
Asked `why do you think it rains`, RAVANA answers `i'm drawn to rains. it sits
close to people, which i care about` -- an affective association, not a causal
explanation. Asked `explain how a battery works`, it says `i don't have a clean
definition for battery, but it is tied to plasma and cells`. The graph holds
causal edges and nothing walks them into a chain.

THE MECHANISM (per the spec)
----------------------------
* **Assemble**: enumerate simple paths of 2-4 hops over `relation_type ==
  "causal"` edges only, on a SNAPSHOT of the graph (the B5 stability fix at
  `chain_walker.py:2215` -- never simulate on the live graph).
* **Direction**: `why <X>` prefers backward (cause -> X); `how does <X> work`
  prefers forward (X -> effect); `explain <X>` takes bidirectional.
* **Score**: `sum(log(w*c)) - lambda*len - mu*unprovenanced` -- explanation is
  effective because it is SELECTIVE (Lombrozo 2007).
* **Monitor**: a separate reader (Nelson & Narens 1990) grades coverage, typed
  fraction, provenance, gap count and the margin over the runner-up.

FAIL-CLOSED: returns None when no typed chain clears the bar, so the caller
falls back to the honest hedge rather than asserting an association.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# Relation types that assert a causal link. `causal` is the graph's own type;
# `physical_cause` is its mechanism-simulation sibling. Everything else
# (semantic, analogical, inferred) is association, not causation -- the
# measured defect was an association being offered as an explanation.
CAUSAL_RELATIONS = frozenset({"causal", "physical_cause"})

# Simplicity penalty (Lombrozo 2007: explanation is selective). A longer chain
# must earn its extra hops.
LAMBDA_LENGTH = 0.35
# Penalty per hop with no provenance (a web/learning source vs an unknown one).
MU_UNPROVENANCED = 0.25

# Adequacy thresholds (spec §5.2 stage 3).
THETA_ASSERT = 0.70
THETA_HEDGE = 0.40


@dataclass
class Hop:
    from_label: str
    relation_type: str
    to_label: str
    weight: float = 0.0
    confidence: float = 0.0
    provenance: str = ""          # '', 'web', 'definition', 'conversation'
    # The graph's real provenance record (`edge.source_metadata`), a dict with
    # an `edge_kind` field. Optional so hand-built hops in tests stay simple.
    source_metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def provenanced(self) -> bool:
        return bool(self.provenance)


@dataclass
class ExplanationChain:
    subject: str
    direction: str                              # 'backward' | 'forward' | 'bidirectional'
    hops: List[Hop] = field(default_factory=list)
    score: float = 0.0
    runner_up_score: float = 0.0

    @property
    def labels(self) -> List[str]:
        if not self.hops:
            return [self.subject]
        # Hops are stored in WALK order (walk node -> neighbour) and are
        # contiguous by construction. Build the walk-order label list, then
        # reverse the LABELS for a backward walk.
        #
        # Do NOT reverse the hop list: hops share endpoints, so reversing them
        # breaks adjacency (measured: hops sun->earth, earth->gravity reversed
        # gave "sun leads to sun, and sun leads to light").
        walk_order = [self.hops[0].from_label]
        walk_order.extend(h.to_label for h in self.hops)
        if self.direction == "backward":
            walk_order.reverse()
        return walk_order

    @property
    def margin(self) -> float:
        return self.score - self.runner_up_score


def _graph_view(graph):
    """Return the object that actually exposes `get_outgoing`/`get_node`.

    `GraphEngine` WRAPS a `ConceptGraph` at `.graph`; the chat engine's mixin
    exposes the `ConceptGraph` directly. Both are passed to this module in
    practice, so accept either rather than forcing callers to unwrap.
    """
    inner = getattr(graph, "graph", None)
    if inner is not None and hasattr(inner, "get_outgoing"):
        return inner
    return graph


def _edge_metadata(edge) -> Dict[str, object]:
    """Read an edge's provenance dict, tolerating its absence."""
    meta = getattr(edge, "source_metadata", None)
    return dict(meta) if isinstance(meta, dict) else {}


def _as_label(value) -> str:
    """Coerce a node label to a string.

    A label is normally a string, but a multi-word or list-valued label occurs
    in the wild; joining is the least-surprising coercion and never raises.
    """
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value).strip()
    return str(value or "").strip()


def _edge_is_causal(relation_type) -> bool:
    """`relation_type` may be a string OR a list of strings.

    The graph stores a single string, but a caller can pass a multi-valued
    relation; a list is causal only if every member is causal (fail-closed).
    """
    if isinstance(relation_type, (list, tuple, set)):
        vals = list(relation_type)
        return bool(vals) and all(
            str(v).strip().lower() in CAUSAL_RELATIONS for v in vals)
    return str(relation_type or "").strip().lower() in CAUSAL_RELATIONS


def _hop_score(hop: Hop) -> float:
    w = max(1e-6, float(hop.weight or 0.0))
    c = max(1e-6, float(hop.confidence or 0.0))
    return math.log(w * c)


def score_chain(hops: Sequence[Hop]) -> float:
    """`sum(log(w*c)) - lambda*len - mu*unprovenanced`."""
    if not hops:
        return -1e9
    raw = sum(_hop_score(h) for h in hops)
    unprov = sum(1 for h in hops if not h.provenanced)
    return raw - LAMBDA_LENGTH * len(hops) - MU_UNPROVENANCED * unprov


def enumerate_chains(subject_ids: Sequence[int],
                     graph,
                     *,
                     direction: str = "forward",
                     min_steps: int = 2,
                     max_steps: int = 4,
                     limit: int = 8) -> List[ExplanationChain]:
    """Enumerate simple causal paths from `subject_ids`.

    `graph` must expose `get_outgoing(node_id) -> [(nid, edge)]` and
    `get_node(nid) -> node` with `.label`. A snapshot is taken by the caller;
    this function never mutates the graph.
    """
    graph = _graph_view(graph)
    out: List[ExplanationChain] = []
    seen_paths = set()

    def _walk(start_id: int, forward: bool) -> None:
        # stack: (node_id, [Hop], visited_node_ids)
        #
        # MEASURED BUG: the simple-path check tested only the neighbour's
        # LABEL, so a RECIPROCAL edge pair (`sun -> light` and
        # `light -> sun`) was not detected as a cycle: walking sun -> light
        # then light -> sun revisits `sun`, but the check looked at the
        # neighbour label against `hops[*].to_label`, and `sun` appears only
        # as `hops[0].from_label`. Measured output:
        #   "light leads to sun, and sun leads to light"
        # Track node IDS, not labels -- two distinct labels can even denote
        # the same node, and the same label can recur legitimately only once.
        stack = [(start_id, [], {start_id})]
        while stack:
            nid, hops, visited = stack.pop()
            if len(hops) >= max_steps:
                continue
            # MEASURED BUG: the `forward` flag was accepted and then IGNORED --
            # both directions called `get_outgoing`, so a "backward" search was
            # silently a forward one. The comment claimed the link was
            # "traversed in reverse", but nothing reversed it. Consequence:
            # `why <X>` (which needs causes, i.e. INCOMING edges) searched
            # X's EFFECTS instead, and the chain the reader gets runs the wrong
            # way -- or finds nothing, because most concepts have few outgoing
            # causal edges.
            #
            # Measured: with two trusted causal edges
            # (evaporate -> condense -> rainfall) injected,
            # `enumerate_chains([condense], direction="backward")` returned
            # ZERO chains, because it looked for outgoing edges of `condense`
            # and the cause (`evaporate`) is incoming.
            if forward:
                neighbours = graph.get_outgoing(nid) or []
            else:
                neighbours = graph.get_incoming(nid) or []
            for other_id, edge in neighbours:
                if not _edge_is_causal(getattr(edge, "relation_type", "")):
                    continue
                node = graph.get_node(other_id)
                label = _as_label(getattr(node, "label", ""))
                if not label:
                    continue
                if any(h.to_label == label for h in hops):
                    continue                      # simple path: no repeats
                # MEASURED BUG: the simple-path check above only tested
                # `to_label`, so a backward walk could return to its START
                # node (which appears only as a `from_label`) and emit a
                # CYCLE -- measured: backward from `zz_rainfall` returned
                # ['zz_condense','zz_rainfall','zz_condense']. Test both ends.
                if any(label in (h.from_label, h.to_label) for h in hops):
                    continue
                if other_id in visited:
                    continue                  # reciprocal edge -> cycle
                from_node = graph.get_node(nid)
                from_label = _as_label(getattr(from_node, "label", ""))
                # Hops are recorded in WALK order (walk node -> neighbour) so
                # the chain is contiguous. For a backward walk that order is
                # effect -> cause, so the WHOLE list is reversed at the end
                # (see `labels`). Flipping individual hops instead breaks the
                # chain, because neighbouring hops share endpoints --
                # measured: "light leads to photosynthesis, and photosynthesis
                # leads to light".
                hop = Hop(
                    from_label=from_label,
                    relation_type=getattr(edge, "relation_type", "causal"),
                    to_label=label,
                    weight=float(getattr(edge, "weight", 0.0) or 0.0),
                    confidence=float(getattr(edge, "confidence", 0.0) or 0.0),
                    provenance=getattr(edge, "provenance", "") or "",
                    source_metadata=_edge_metadata(edge),
                )
                new_hops = hops + [hop]
                if len(new_hops) >= min_steps:
                    key = tuple((h.from_label, h.to_label) for h in new_hops)
                    if key not in seen_paths:
                        seen_paths.add(key)
                        out.append(ExplanationChain(
                            subject=str(start_id), direction=direction,
                            hops=list(new_hops),
                            score=score_chain(new_hops)))
                stack.append((other_id, new_hops, visited | {other_id}))

    for sid in subject_ids:
        if direction in ("forward", "bidirectional"):
            _walk(sid, forward=True)
        if direction in ("backward", "bidirectional"):
            _walk(sid, forward=False)

    # F3 (trust-first candidate selection): partition by assertable_chain BEFORE
    # ranking, so a high-weight UNtrusted chain cannot crowd a taught trusted
    # chain out of the limited traversal budget (measured: honey's curated chain
    # was present at limit 256 but absent at 64 because untrusted co-occurrence
    # chains out-scored it on log-weight). Within each partition rank by the
    # existing score; the score VALUE, grade_chain and every threshold are
    # unchanged. The runner-up is taken within the SAME trust partition so the
    # margin compares like-with-like. Brain basis: inhibitory selection of the
    # target over prepotent competitors (Anderson & Spellman 1995, Psychol Rev
    # 102:68-100); fast-store episode traces must not be drowned by slow-store
    # statistics (McClelland, McNaughton & O'Reilly 1995, Psychol Rev 102:419-457).
    _decorated = [(assertable_chain(c), c.score, c) for c in out]
    _decorated.sort(key=lambda t: (t[0], t[1]), reverse=True)
    out = [t[2] for t in _decorated]
    _i, _n = 0, len(out)
    while _i < _n:
        _flag = _decorated[_i][0]
        _j = _i
        while _j < _n and _decorated[_j][0] == _flag:
            _j += 1
        _part = out[_i:_j]
        if len(_part) > 1:
            _top, _second = _part[0].score, _part[1].score
            for _k, _c in enumerate(_part):
                _c.runner_up_score = _second if _k == 0 else _top
        else:
            _part[0].runner_up_score = _part[0].score
        _i = _j
    return out[:limit]


def _hop_provenance(hop: Hop) -> str:
    """Normalise a hop's provenance to a trusted-source tag.

    Reads the graph's ACTUAL provenance record, which lives in
    `edge.source_metadata` (a dict) with an `edge_kind` field -- not in an
    edge attribute called `provenance`. `web_to_graph.py:290` stamps
    `edge_kind="web_fact"` on every OpenIE fact; the co-occurrence path stamps
    `edge_kind="co_occurrence"` (`web/learner.py:926`).

    The first version of this module read `edge.provenance`, which does not
    exist, so every hop looked unprovenanced and the gate declined everything
    -- including web facts that ARE provenance-stamped.
    """
    if hop.provenance:
        return hop.provenance
    meta = hop.source_metadata if isinstance(hop.source_metadata, dict) else {}
    kind = str(meta.get("edge_kind") or "").strip().lower()
    if kind == "web_fact":
        return "web"
    # `curated` is what `_ensure_relation` stamps on the authored offline
    # skeleton (common_facts / physics causal priors). It is a hand-written
    # source, so it is trusted -- but see docs/AUTHORED_CAUSAL_SKELETON.md:
    # the list contains at least one edge that is not a prior at all
    # (`gravity -> earth`), so a blanket stamp is not a truth guarantee.
    if kind in ("definition", "curriculum", "curated"):
        return kind
    if kind == "co_occurrence":
        return ""                    # association, not causation
    src = str(meta.get("source") or "").strip().lower()
    if src in ("web", "wikipedia", "conceptnet"):
        return "web"
    return ""


def provenance_trusted(hop: Hop) -> bool:
    """True when a hop's source is one the system can assert from.

    MEASURED (docs/EXPLANATION_CHAINS_MEASURED.md): the live graph's causal
    edges are co-occurrence-derived and not truthful -- `gravity -> earth`,
    `air -> sound`, `earth -> sun`, all at w>=0.80 c=0.90. Asserting a chain of
    those would INCREASE confident wrong output.

    So a hop is assertable only when it carries provenance from a source that
    states the relation explicitly: a stored definition, or a web fact that
    OpenIE extracted with an explicit causal predicate. An unprovenanced
    co-occurrence edge is association, not causation.
    """
    return _hop_provenance(hop) in ("web", "definition", "conversation",
                                    "curriculum", "curated")


def assertable_chain(chain: Optional[ExplanationChain]) -> bool:
    """True when EVERY hop is provenance-trusted.

    Fail-closed: one untrusted hop makes the whole chain unassertable, because
    a chain is only as strong as its weakest link and the reader cannot tell
    which link a listener will check.
    """
    if chain is None or not chain.hops:
        return False
    return all(provenance_trusted(h) for h in chain.hops)


def grade_chain(chain: Optional[ExplanationChain], *,
                theta_assert: float = THETA_ASSERT,
                theta_hedge: float = THETA_HEDGE) -> Dict[str, object]:
    """Stage 3: a READER grades the chain (Nelson & Narens 1990).

    Returns the components and a verdict: 'assert' | 'hedge' | 'decline'.
    """
    if chain is None or not chain.hops:
        return {"adequacy": 0.0, "verdict": "decline", "coverage": 0.0,
                "typed_fraction": 0.0, "provenance": 0.0, "gap_count": 0.0,
                "margin": 0.0}
    hops = chain.hops
    typed = sum(1 for h in hops
                if _edge_is_causal(h.relation_type)) / len(hops)
    prov = min(float(h.confidence or 0.0) for h in hops)
    # A gap is a hop pair where the link is missing or inverted. Every hop here
    # is a real edge, so gaps are counted as hops with no provenance AND low
    # confidence -- the closest structural analogue available.
    gaps = sum(1 for h in hops
               if not h.provenanced and float(h.confidence or 0.0) < 0.3)
    margin = max(0.0, min(1.0, chain.margin))
    adequacy = (1.0 * typed + 1.0 * prov - 0.5 * gaps + 0.5 * margin)
    adequacy = max(0.0, min(1.0, adequacy))
    verdict = ("assert" if adequacy >= theta_assert
               else "hedge" if adequacy >= theta_hedge
               else "decline")
    # The structural grade is necessary but NOT sufficient: a chain of
    # unprovenanced co-occurrence edges grades 1.000 and is still false
    # (measured). Provenance is the gate that structure cannot be.
    trusted = assertable_chain(chain)
    if verdict == "assert" and not trusted:
        verdict = "decline"
    return {"adequacy": adequacy, "verdict": verdict, "coverage": 1.0,
            "typed_fraction": typed, "provenance": prov,
            "gap_count": float(gaps), "margin": margin,
            "provenance_trusted": trusted}
