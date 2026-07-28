"""TripletMemory — the dual-system store for triplet inference.

Fast episodic side: an optional facade over the existing
HippocampalBuffer (which is EPHEMERAL — ~50 facts, turn-gated decay).
Because profile learning needs far more history than the buffer keeps,
TripletMemory maintains its own bounded relational index of ACTIVE
triples for chain mining, plus the persistent per-predicate
RelationProfiles (the slow, neocortical statistics).

Persistence contract (hard lesson from the de-hardcoding audit): both
``to_dict`` and ``from_dict`` exist and the host engine must call BOTH
in save() and load() — a gate written but never reloaded is dead.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .canonical import canonical_predicate
from .core import RelationProfile, RelationalSchema, Triple


class TripletMemory:
    """Bounded relational index + learned relation profiles + schemas."""

    def __init__(self, hippocampal_buffer=None, max_triples: int = 5000):
        # Optional facade: also mirror ingested triples into the existing
        # episodic buffer so the rest of the engine sees them.
        self.episodic = hippocampal_buffer
        self.max_triples = int(max_triples)

        # Relational index over ACTIVE triples (superseded ones excluded).
        self.triples: List[Triple] = []
        self._by_key: Dict[Tuple[str, str, str], Triple] = {}
        self._by_sp: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self._by_op: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self._by_subj: Dict[str, List[Triple]] = defaultdict(list)

        # Slow stores.
        self.profiles: Dict[str, RelationProfile] = {}
        self.schemas: Dict[str, RelationalSchema] = {}

    # ── profile access (lazy creation — predicates are discovered) ────
    def profile(self, predicate: str) -> RelationProfile:
        p = canonical_predicate(predicate)
        prof = self.profiles.get(p)
        if prof is None:
            prof = RelationProfile(predicate=p)
            self.profiles[p] = prof
        return prof

    # ── storage ───────────────────────────────────────────────────────
    def add(self, triple: Triple) -> Triple:
        """Index a triple (canonicalizing its predicate). Dedupes by key;
        a re-observation strengthens confidence instead of duplicating."""
        s = triple.subject.strip().lower()
        p = canonical_predicate(triple.predicate)
        o = triple.object.strip().lower()
        if not s or not p or not o:
            return triple
        key = (s, p, o)
        existing = self._by_key.get(key)
        if existing is not None:
            existing.confidence = min(1.0, existing.confidence + 0.05)
            return existing
        t = Triple(subject=s, predicate=p, object=o,
                   confidence=triple.confidence, source=triple.source,
                   session_date=triple.session_date,
                   absolute_date=triple.absolute_date,
                   superseded=triple.superseded,
                   turn_number=triple.turn_number)
        self.triples.append(t)
        self._by_key[key] = t
        self._by_sp[(s, p)].add(o)
        self._by_op[(o, p)].add(s)
        self._by_subj[s].append(t)
        if len(self.triples) > self.max_triples:
            self._evict_oldest()
        return t

    def supersede(self, subject: str, predicate: str, object: str) -> bool:
        key = (subject.strip().lower(), canonical_predicate(predicate),
               object.strip().lower())
        t = self._by_key.get(key)
        if t is None:
            return False
        t.superseded = True
        self._by_sp[(key[0], key[1])].discard(key[2])
        self._by_op[(key[2], key[1])].discard(key[0])
        return True

    def _evict_oldest(self):
        """Drop the oldest superseded triples first, then plain oldest."""
        n_drop = max(1, len(self.triples) - self.max_triples)
        keep: List[Triple] = []
        dropped = 0
        # First pass: drop superseded.
        for t in self.triples:
            if dropped < n_drop and t.superseded:
                self._unindex(t)
                dropped += 1
            else:
                keep.append(t)
        # Second pass: drop from the front (oldest) if still over.
        while dropped < n_drop and keep:
            t = keep.pop(0)
            self._unindex(t)
            dropped += 1
        self.triples = keep

    def _unindex(self, t: Triple):
        key = t.key()
        self._by_key.pop(key, None)
        self._by_sp[(t.subject, t.predicate)].discard(t.object)
        self._by_op[(t.object, t.predicate)].discard(t.subject)
        subj_list = self._by_subj.get(t.subject)
        if subj_list and t in subj_list:
            subj_list.remove(t)

    # ── lookups used by learning + operators ──────────────────────────
    def has_fact(self, subject: str, predicate: str, object: str) -> bool:
        t = self._by_key.get((subject.strip().lower(),
                              canonical_predicate(predicate),
                              object.strip().lower()))
        return t is not None and not t.superseded

    def objects_of(self, subject: str, predicate: str) -> Set[str]:
        return set(self._by_sp.get(
            (subject.strip().lower(), canonical_predicate(predicate)), ()))

    def subjects_of(self, object: str, predicate: str) -> Set[str]:
        return set(self._by_op.get(
            (object.strip().lower(), canonical_predicate(predicate)), ()))

    def find_mediators(self, a: str, predicate: str, c: str) -> Set[str]:
        """B such that (a, r, B) and (B, r, c) are both active facts."""
        return self.objects_of(a, predicate) & self.subjects_of(c, predicate)

    def triples_about(self, subject: str) -> List[Triple]:
        return [t for t in self._by_subj.get(subject.strip().lower(), [])
                if not t.superseded]

    def active_triples(self) -> Iterable[Triple]:
        return (t for t in self.triples if not t.superseded)

    # ── persistence ───────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "triples": [t.to_dict() for t in self.triples],
            "profiles": {k: v.to_dict() for k, v in self.profiles.items()},
            "schemas": {k: v.to_dict() for k, v in self.schemas.items()},
        }

    def from_dict(self, d: dict) -> None:
        self.triples = []
        self._by_key.clear()
        self._by_sp.clear()
        self._by_op.clear()
        self._by_subj.clear()
        for td in d.get("triples", []):
            t = Triple.from_dict(td)
            key = t.key()
            if key in self._by_key:
                continue
            self.triples.append(t)
            self._by_key[key] = t
            if not t.superseded:
                self._by_sp[(t.subject, t.predicate)].add(t.object)
                self._by_op[(t.object, t.predicate)].add(t.subject)
            self._by_subj[t.subject].append(t)
        self.profiles = {
            k: RelationProfile.from_dict(v)
            for k, v in d.get("profiles", {}).items()}
        self.schemas = {
            k: RelationalSchema.from_dict(v)
            for k, v in d.get("schemas", {}).items()}
