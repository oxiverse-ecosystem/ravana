"""HRR compositional reasoner over the DualCodeSpace (Work A0).

Brain basis (per the plan):
  - Cohen & Eichenbaum 1993: the graph IS the relational store.
  - Dusek & Eichenbaum 1997: transitive inference (A->B, B->C => A->C) needs
    INTEGRATED structure; hippocampal disconnection kills inference but not the
    bare associations -> test composing edges never seen together.
  - Zeithamova & Preston: integrative encoding (reactivate AB when adding BC)
    makes chaining cheap.
  - Plate 2003: HRR capacity is linear in D, so budget ~2-3 terms and prefer the
    graph for long chains; clean-up after every unbind; role vectors random+unitary.

Design decision (faithful to the plan's caution): the 2048-D HRR space does
BINDING / COMPOSITION / ANALOGY; the DISCRETE GRAPH is the clean-up codebook.
We never emit an HRR vector as an answer — we unbind through recover_role_filler
over the concrete graph-atom set, then ground the result in the graph. GloVe
atoms are correlated (not random), so DualCodeSpace already lifts them through a
random JL projection (_lift) before binding; the discrete decode step is what
makes the recovery unambiguous regardless.

Fact store: keyed by (subject, verb) -> bundled HRR structure. Transitive query
chains via recover_role_filler(struct, "object", atoms): look up (head, verb),
recover the object, recurse. The graph.infer_chain path is the authoritative
multi-hop fallback for long chains.
"""
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

from ravana.core.dual_code_space import DualCodeSpace


class HRRReasoner:
    def __init__(self, dual: DualCodeSpace):
        self.dual = dual
        # (subject, verb) -> bundled HRR structure (the integrated fact).
        self._facts: Dict[Tuple[str, str], np.ndarray] = {}
        # Set of concrete concept labels used as clean-up candidates.
        self._atoms: Set[str] = set()
        # Integrative index: subject -> list of (verb, object) for chaining.
        self._by_subject: Dict[str, List[Tuple[str, str]]] = {}
        # Relation-conditioned candidate index (M1, Tulving encoding specificity /
        # Anderson fan-effect attenuation): verb -> set of object labels. Restricts
        # the clean-up NN search to the real object pool of a typed relation,
        # shrinking the wrong-sibling set from all-atoms to the relation's objects.
        self._by_verb: Dict[str, Set[str]] = {}

    # ── populate ──
    def encode(self, subject: str, verb: str, obj: str) -> None:
        """Encode one fact. Integrative encoding (Zeithamova): storing by
        (subject, verb) means adding (B, rel, C) automatically makes A->C
        reachable once A->B was stored, because query_chain walks the subject
        index."""
        key = (subject.lower(), verb.lower())
        struct = self.dual.encode_fact(subject, verb, obj)
        self._facts[key] = struct
        self._atoms.add(subject.lower())
        self._atoms.add(obj.lower())
        self._by_subject.setdefault(subject.lower(), []).append((verb.lower(), obj.lower()))
        # M1: index objects per verb for relation-conditioned clean-up.
        self._by_verb.setdefault(verb.lower(), set()).add(obj.lower())

    def _candidates(self, verb: Optional[str] = None) -> List[str]:
        """Clean-up candidate set. M1: if the relation is known, restrict to its
        real object pool (encoding specificity); fall back to all atoms only for
        an unseen relation. Restricting the SET (not the score) is calibration-safe."""
        if verb is not None:
            pool = self._by_verb.get(verb.lower())
            if pool:
                return list(pool)
        return list(self._atoms)

    # ── single-hop HRR recovery (clean-up through the discrete atom set) ──
    def recover_object(self, subject: str, verb: str) -> Optional[str]:
        struct = self._facts.get((subject.lower(), verb.lower()))
        if struct is None:
            return None
        return self.dual.recover_role_filler(struct, "object", self._candidates(verb))

    # ── transitive chain: A->B, B->C => recover C from A ──
    def query_chain_with_conf(self, head: str, verb: str, max_hops: int = 2):
        """Like query_chain but returns (chain, confs) where confs[i] is the
        decode cosine for the i-th recovered hop — the confidence proxy
        (recollection strength; Yonelinas). Hop 1 = stored (direct associate),
        hop>1 = inferred (composed)."""
        chain: List[str] = []
        confs: List[float] = []
        cur = head.lower()
        seen = {cur}
        for _ in range(max_hops):
            struct = self._facts.get((cur, verb.lower()))
            if struct is None:
                break
            nxt, c = self.dual.recover_role_filler_with_conf(struct, "object", self._candidates(verb))
            if nxt is None or nxt.lower() in seen:
                break
            chain.append(nxt)
            confs.append(c)
            seen.add(nxt.lower())
            cur = nxt
        return chain, confs

    def query_chain(self, head: str, verb: str, max_hops: int = 2) -> List[str]:
        """Return the chain of objects reachable from `head` via `verb`, walking
        the integrated fact store. Decodes every step through the atom set
        (clean-up), never emitting a raw vector."""
        chain, _ = self.query_chain_with_conf(head, verb, max_hops=max_hops)
        return chain

    # ── relation composition (causes ∘ enables): bundle role vectors ──
    def compose_relations(self, *verbs: str) -> np.ndarray:
        """Compose multiple relations into one query role (for multi-relational
        hops). Bundling the role vectors yields an approximate composed role;
        Plate (2003): keep to ~2-3 terms — HRR capacity is linear in D."""
        vecs = [self.dual.role(v.lower()) for v in verbs]
        if not vecs:
            return np.zeros(self.dual.hrr_dim, dtype=np.float32)
        s = np.sum(vecs, axis=0)
        ns = np.linalg.norm(s)
        return (s / ns).astype(np.float32) if ns > 0 else s.astype(np.float32)

    def __len__(self) -> int:
        return len(self._facts)

    def has_fact(self, subject: str, verb: str) -> bool:
        return (subject.lower(), verb.lower()) in self._facts
