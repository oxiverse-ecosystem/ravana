"""Online learning loop — updates RelationProfiles from each triple.

Every inference property is a learned Bernoulli statistic:

- transitivity: when a chain A-r->B-r->C is observable, does A-r->C
  hold? (positive/negative counts; Lippl 2024 conjunctivity alpha =
  1 - rate)
- symmetry: when (A,r,B) is observed, does (B,r,A) hold?
- inverse: which OTHER predicate r2 most often satisfies (B,r2,A)?
- composition: which (r2, r3) satisfy (A,r1,B),(B,r2,C) => (A,r3,C)?

Gating philosophy (house rule — no fixed behavior thresholds): a
property fires when the Wilson 95% LOWER bound of its rate exceeds
0.5. "0.5" is not a tuned knob; it is the more-likely-than-not
decision boundary for a Bernoulli parameter, and the Wilson bound
prices in evidence volume, so small-n predicates fail closed without
any separate MIN_OBSERVATIONS constant.

The HRR cross-signal (observe_hrr_chain) lets the vector system's
compositional successes/failures feed the same statistics, bridging
symbolic and vector inference.
"""
from __future__ import annotations

from typing import Optional

from .core import Triple
from .memory import TripletMemory

# Bernoulli decision boundary (see module docstring — not a tuned knob).
DECISION_BOUNDARY = 0.5


class ProfileLearner:
    """Stateless-per-call learner over a TripletMemory."""

    def __init__(self, memory: TripletMemory):
        self.memory = memory

    # ── main entry: called for every ingested triple ──────────────────
    def observe(self, triple: Triple) -> None:
        m = self.memory
        s, p, o = triple.subject, triple.predicate, triple.object
        prof = m.profile(p)

        # 1. Transitivity — enumerate chains this triple participates in.
        # (a) The new triple closes/extends chains as the FIRST leg:
        #     (s,p,o) + (o,p,c): is (s,p,c) known?
        for c in m.objects_of(o, p):
            if c == s:
                continue
            if m.has_fact(s, p, c):
                prof.transitivity_pos += 1
            else:
                prof.transitivity_neg += 1
        # (b) The new triple as the SECOND leg: (a,p,s) + (s,p,o): (a,p,o)?
        for a in m.subjects_of(s, p):
            if a == o:
                continue
            if m.has_fact(a, p, o):
                prof.transitivity_pos += 1
            else:
                prof.transitivity_neg += 1
        # (c) The new triple as the CLOSING edge: (s,p,b) + (b,p,o) means
        # the chain s->b->o existed and s->o just arrived — that converts
        # prior negative evidence into positive. Count the confirmation.
        for b in m.find_mediators(s, p, o):
            prof.transitivity_pos += 1
            # Roll back one pessimistic count if any was recorded — the
            # open chain that just closed had been counted as negative.
            if prof.transitivity_neg > 0:
                prof.transitivity_neg -= 1

        # 2. Symmetry — does the reverse fact exist?
        if m.has_fact(o, p, s):
            prof.symmetry_pos += 1
            # The earlier observation of (o,p,s) recorded a negative
            # (its reverse didn't exist yet). Reconcile it.
            if prof.symmetry_neg > 0:
                prof.symmetry_neg -= 1
        else:
            prof.symmetry_neg += 1

        # 3. Inverse detection — any OTHER predicate linking o back to s.
        for t in m.triples_about(o):
            if t.object == s and t.predicate != p:
                prof.inverse_counts[t.predicate] = (
                    prof.inverse_counts.get(t.predicate, 0) + 1)
                # Symmetric bookkeeping on the other profile.
                other = m.profile(t.predicate)
                other.inverse_counts[p] = other.inverse_counts.get(p, 0) + 1

        # 4. Composition — (s,p,o) + (o,r2,c) with a known (s,r3,c).
        for t2 in m.triples_about(o):
            r2, c = t2.predicate, t2.object
            if c == s:
                continue
            for t3 in m.triples_about(s):
                if t3.object == c:
                    bucket = prof.composition_counts.setdefault(r2, {})
                    bucket[t3.predicate] = bucket.get(t3.predicate, 0) + 1

        # 5. Hierarchy depth — length of the maximal same-predicate chain
        # through this triple (bounded walk, cycle-guarded).
        depth = self._chain_depth(s, p, o)
        prof.depth_sum += depth
        prof.depth_n += 1

    def _chain_depth(self, s: str, p: str, o: str, cap: int = 8) -> int:
        m = self.memory
        depth = 1
        seen = {s, o}
        cur = o
        while depth < cap:
            nxt = None
            for c in m.objects_of(cur, p):
                if c not in seen:
                    nxt = c
                    break
            if nxt is None:
                break
            seen.add(nxt)
            cur = nxt
            depth += 1
        cur = s
        while depth < cap:
            prv = None
            for a in m.subjects_of(cur, p):
                if a not in seen:
                    prv = a
                    break
            if prv is None:
                break
            seen.add(prv)
            cur = prv
            depth += 1
        return depth

    # ── HRR cross-signal ──────────────────────────────────────────────
    def observe_hrr_chain(self, predicate: str, success: bool) -> None:
        """Feed an HRRReasoner.query_chain outcome into transitivity
        evidence: a successful vector chain over a predicate is positive
        evidence that its representation is additive (transitive);
        a failure is (weak) negative evidence."""
        prof = self.memory.profile(predicate)
        if success:
            prof.transitivity_pos += 1
        else:
            prof.transitivity_neg += 1

    # ── gate predicates (used by operators) ───────────────────────────
    def is_transitive(self, predicate: str) -> bool:
        prof = self.memory.profiles.get(
            self.memory.profile(predicate).predicate)
        return (prof is not None
                and prof.transitivity_lower() > DECISION_BOUNDARY)

    def is_symmetric(self, predicate: str) -> bool:
        prof = self.memory.profiles.get(
            self.memory.profile(predicate).predicate)
        return (prof is not None
                and prof.symmetry_lower() > DECISION_BOUNDARY)
