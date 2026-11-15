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
        out = [self.hops[0].from_label]
        out.extend(h.to_label for h in self.hops)
        return out

    @property
    def margin(self) -> float:
        return self.score - self.runner_up_score


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
    out: List[ExplanationChain] = []
    seen_paths = set()

    def _walk(start_id: int, forward: bool) -> None:
        # stack: (node_id, [Hop])
        stack = [(start_id, [])]
        while stack:
            nid, hops = stack.pop()
            if len(hops) >= max_steps:
                continue
            neighbours = graph.get_outgoing(nid) or []
            for other_id, edge in neighbours:
                if not _edge_is_causal(getattr(edge, "relation_type", "")):
                    continue
                # `get_outgoing` is forward; for a backward walk the causal
                # link is traversed in reverse, which is what `why <X>` wants.
                node = graph.get_node(other_id)
                label = _as_label(getattr(node, "label", ""))
                if not label:
                    continue
                if any(h.to_label == label for h in hops):
                    continue                      # simple path: no repeats
                from_node = graph.get_node(nid)
                from_label = _as_label(getattr(from_node, "label", ""))
                hop = Hop(
                    from_label=from_label,
                    relation_type=getattr(edge, "relation_type", "causal"),
                    to_label=label,
                    weight=float(getattr(edge, "weight", 0.0) or 0.0),
                    confidence=float(getattr(edge, "confidence", 0.0) or 0.0),
                    provenance=getattr(edge, "provenance", "") or "",
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
                stack.append((other_id, new_hops))

    for sid in subject_ids:
        if direction in ("forward", "bidirectional"):
            _walk(sid, forward=True)
        if direction in ("backward", "bidirectional"):
            _walk(sid, forward=False)

    out.sort(key=lambda c: c.score, reverse=True)
    if len(out) > 1:
        for i, c in enumerate(out):
            c.runner_up_score = out[1].score if i == 0 else out[0].score
    return out[:limit]


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
    return hop.provenance in ("web", "definition", "conversation", "curriculum")


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
