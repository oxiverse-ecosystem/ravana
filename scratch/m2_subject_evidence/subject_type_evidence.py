"""Default-OFF subject-side type evidence channel (scratch prototype).

Pure, engine-free, answer-key-free. Produces type evidence for the **actual
query subject** from independently licensed external text/ontology records;
rejects negated, contradicted, unlicensed and low-credibility assertions.
Nothing runs unless RAVANA_SUBJECT_TYPE_EVIDENCE=1.

Brain grounding (see M2_SUBJECT_TYPE_EVIDENCE_PREREG.md): a new word's type
comes from subject-specific contrastive input (Carey & Bartlett 1978), inside
a structured hypothesis space (Xu & Tenenbaum 2007), assimilated only into an
existing schema (Tse et al. 2007), with source-controlled admission because
schema-congruence also causes false memories (Johnson et al. 1993).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

ENV_FLAG = "RAVANA_SUBJECT_TYPE_EVIDENCE"
CREDIBILITY_FLOOR = 0.50
ADMITTED_RELATIONS = frozenset({"is_a", "instance_of"})

# Credibility is a function of the SOURCE LICENSE only, never of any label.
LICENSE_CREDIBILITY = {
    "CC0-1.0": 0.7,       # Wikidata structured data / item descriptions
    "WordNet-3.0": 0.9,   # Princeton WordNet 3.0 noun hypernyms
}
LICENSE_URLS = {
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "WordNet-3.0": "https://wordnet.princeton.edu/wordnet/license/",
}

_NEGATION = re.compile(
    r"\b(?:not|never|no|isn't|isnt|aren't|arent|wasn't|wasnt|weren't|werent"
    r"|cannot|can't|cant|doesn't|doesnt|didn't|didnt|without|non|false|except)\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[a-z0-9]+")

__all__ = [
    "ENV_FLAG", "CREDIBILITY_FLOOR", "Assertion", "Evidence", "Rejection",
    "enabled", "effective_credibility", "is_negated", "admit",
    "evidence_for_subject", "signature",
]


def enabled() -> bool:
    """Inert by default; the only way to turn this path on is the env flag."""
    return os.environ.get(ENV_FLAG, "") == "1"


@dataclass(frozen=True)
class Assertion:
    subject: str
    relation: str
    type_name: str
    text: str
    source: str
    license: str
    source_url: str = ""
    snapshot_sha256: str = ""
    credibility: float = 0.0


@dataclass(frozen=True)
class Evidence:
    subject: str
    relation: str
    type_name: str
    assertion_text: str
    source_text: str
    source: str
    license: str
    license_url: str
    source_url: str
    snapshot_sha256: str
    credibility: float


@dataclass(frozen=True)
class Rejection:
    subject: str
    type_name: str
    reason: str


def _norm(value: str) -> str:
    return " ".join(_WORD.findall((value or "").lower()))


def effective_credibility(a: Assertion) -> float:
    """Credibility from the license allow-list, capped by the declared value.

    A caller cannot inflate credibility by declaring a large number: the
    allow-list is the ceiling. An unlicensed record scores 0.0.
    """
    ceiling = LICENSE_CREDIBILITY.get((a.license or "").strip(), 0.0)
    try:
        declared = float(a.credibility)
    except (TypeError, ValueError):
        declared = 0.0
    if declared != declared:
        declared = 0.0
    # A non-positive declaration means "not stated": inherit the licence's
    # ceiling. A positive declaration is a conservative cap, never an uplift.
    if not ceiling:
        return 0.0
    if declared <= 0.0:
        return ceiling
    return max(0.0, min(declared, ceiling))


def is_negated(text: str, subject: str, type_name: str) -> bool:
    """True if negation scopes over the typed predicate for this subject.

    The cue must occur in the clause that mentions the type (not merely
    somewhere in an unrelated sentence). Fail-closed: ambiguous text is
    treated as negated.
    """
    raw = text or ""
    low = raw.lower()
    if not low.strip():
        return True
    target = _norm(type_name)
    if not target:
        return True
    # Locate the type mention by normalised token span.
    toks = [(m.group(0).lower(), m.start(), m.end())
            for m in _WORD.finditer(raw)]
    hits = [i for i, (t, _s, _e) in enumerate(toks) if t == target.split()[0]]
    if not hits:
        # No explicit type mention: still fail closed on a negation cue in
        # the subject's own clause.
        return bool(_NEGATION.search(low))
    for i in hits:
        start = toks[max(0, i - 12)][1]
        end = toks[min(len(toks) - 1, i + 4)][2]
        clause = low[start:end]
        if _NEGATION.search(clause):
            return True
    return False


def admit(subject: str, assertions: Iterable[Assertion],
          floor: float = CREDIBILITY_FLOOR
          ) -> Tuple[Tuple[Evidence, ...], Tuple[Rejection, ...]]:
    """Admit only sourced, non-negated, non-contradicted subject evidence."""
    who = _norm(subject)
    if not who:
        return (), (Rejection(subject, "", "empty_subject"),)
    items = list(assertions)
    admitted: list[Evidence] = []
    rejected: list[Rejection] = []
    # Higher-credibility negations are the contradiction set.
    negations = {}
    for a in items:
        if _norm(a.subject) != who or not a.type_name:
            continue
        if is_negated(a.text, a.subject, a.type_name):
            negations.setdefault(_norm(a.type_name), []).append(
                effective_credibility(a))
    for a in items:
        t = a.type_name or ""
        key = _norm(t)
        if _norm(a.subject) != who:
            rejected.append(Rejection(subject, t, "wrong_subject"))
            continue
        if a.relation not in ADMITTED_RELATIONS:
            rejected.append(Rejection(subject, t, "bad_relation"))
            continue
        if not key or not (a.text or "").strip():
            rejected.append(Rejection(subject, t, "empty_assertion"))
            continue
        if not LICENSE_CREDIBILITY.get((a.license or "").strip(), 0.0):
            rejected.append(Rejection(subject, t, "unlicensed"))
            continue
        cred = effective_credibility(a)
        if cred < floor:
            rejected.append(Rejection(subject, t, "low_credibility"))
            continue
        if is_negated(a.text, a.subject, t):
            rejected.append(Rejection(subject, t, "negated"))
            continue
        if any(other > cred + 1e-12 for other in negations.get(key, ())):
            rejected.append(Rejection(subject, t, "contradicted"))
            continue
        admitted.append(Evidence(
            subject=subject,
            relation=a.relation,
            type_name=t,
            assertion_text=a.text,
            source_text=a.text,
            source=a.source,
            license=a.license,
            license_url=LICENSE_URLS.get(a.license, ""),
            source_url=a.source_url,
            snapshot_sha256=a.snapshot_sha256,
            credibility=cred,
        ))
    # De-duplicate by (source, type), keeping the highest credibility.
    best: dict[Tuple[str, str], Evidence] = {}
    for ev in admitted:
        k = (ev.source, _norm(ev.type_name))
        if k not in best or ev.credibility > best[k].credibility:
            best[k] = ev
    out = sorted(best.values(),
                 key=lambda e: (-e.credibility, e.type_name, e.source))
    return tuple(out), tuple(rejected)


def evidence_for_subject(subject: str, snapshot: dict,
                         extra: Sequence[Assertion] = (),
                         floor: float = CREDIBILITY_FLOOR
                         ) -> Tuple[Tuple[Evidence, ...], Tuple[Rejection, ...]]:
    """Evidence for the ACTUAL query subject, or nothing. Default-OFF."""
    if not enabled():
        return (), ()
    who = _norm(subject)
    snap_sha = str(snapshot.get("provenance", {}).get("facts_sha256", ""))
    assertions = []
    for f in snapshot.get("facts", ()):
        if _norm(f.get("subject", "")) != who:
            continue
        assertions.append(Assertion(
            subject=f.get("subject", ""),
            relation=f.get("relation", ""),
            type_name=f.get("type_name", ""),
            text=f.get("assertion", "") or f.get("text_evidence", ""),
            source=f.get("source", ""),
            license=f.get("license", ""),
            source_url=f.get("source_url", ""),
            snapshot_sha256=snap_sha,
            credibility=0.0,  # derived from license inside admit()
        ))
    assertions.extend(extra)
    return admit(subject, assertions, floor=floor)


def signature(evidence: Sequence[Evidence]) -> tuple:
    """Comparable, content-addressed signature of an evidence set."""
    return tuple(sorted((e.subject, e.relation, e.type_name, e.source,
                         e.assertion_text) for e in evidence))
