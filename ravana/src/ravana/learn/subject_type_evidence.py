"""Default-OFF, non-leaking subject-side type evidence channel for TRA/RSI.

This module is pure and engine-free. It does not boot an engine, call
``process_turn``, write a store, import a benchmark or answer key, or change any
gate, threshold, item, prompt or scoring routine. Nothing happens unless
``RAVANA_SUBJECT_TYPE_EVIDENCE=1`` is set explicitly.

It produces type evidence for the **actual query subject** from an independent
licensed external snapshot (Wikidata structured data, CC0-1.0) and admits an
assertion only when it is sourced, licensed, above the credibility floor,
non-negated and not contradicted by a higher-credibility negation. Every
emitted record carries full provenance.

Brain grounding (pre-registration: ``docs/investigations/M2_SUBJECT_TYPE_EVIDENCE_PREREG.md``):
a new word's type comes from a subject-specific contrastive input (Carey &
Bartlett 1978; Carey 2010, doi:10.1080/15475441.2010.484379), inside a
structured hypothesis space (Xu & Tenenbaum 2007, doi:10.1037/0033-295X.114.2.245),
assimilated only into an existing schema (Tse et al. 2007, doi:10.1126/science.1135935),
with source-controlled admission because schema-congruence also produces false
memories (Johnson, Hashtroudi & Lindsay 1993, doi:10.1037/0033-2909.114.1.3).
The mapping to a provenance filter is an extrapolation, not a claim that the
brain reads an ontology.

Source licence notices
----------------------
Wikidata structured data and item descriptions: CC0-1.0 (public domain);
official policy https://www.wikidata.org/wiki/Wikidata:Licensing and
https://creativecommons.org/publicdomain/zero/1.0/ .
WordNet 3.0 (optional cross-check source): Copyright (c) 1995-2006 Princeton
University; WordNet License 3.0, https://wordnet.princeton.edu/wordnet/license/ .
Miller (1995), CACM 38(11):39-41, doi:10.1145/219717.219748.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
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
SNAPSHOT_RELPATH = Path("data") / "subject_type_evidence" / "snapshot.json"

_NEGATION = re.compile(
    r"\b(?:not|never|no|isn't|isnt|aren't|arent|wasn't|wasnt|weren't|werent"
    r"|cannot|can't|cant|doesn't|doesnt|didn't|didnt|without|non|false|except)\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[a-z0-9]+")

__all__ = [
    "ENV_FLAG", "CREDIBILITY_FLOOR", "Assertion", "Evidence", "Rejection",
    "enabled", "default_snapshot_path", "load_snapshot",
    "effective_credibility", "is_negated", "admit",
    "evidence_for_subject", "signature",
]


def enabled() -> bool:
    """Inert by default; the only way to turn this path on is the env flag."""
    return os.environ.get(ENV_FLAG, "") == "1"


def default_snapshot_path() -> Path:
    """Path to the checked-in frozen licensed snapshot (no network)."""
    return Path(__file__).resolve().parents[4] / SNAPSHOT_RELPATH


def load_snapshot(path: str | os.PathLike | None = None) -> dict:
    """Read the frozen snapshot from disk; never fetch from the network."""
    target = Path(path) if path is not None else default_snapshot_path()
    with open(target, "r", encoding="utf-8", newline="") as fh:
        return json.load(fh)


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
    """Credibility from the license allow-list, capped by a declared value.

    A non-positive declaration means "not stated" and inherits the license
    ceiling. A positive declaration is a conservative cap, never an uplift, so
    a caller cannot forge a high-credibility record by declaring one.
    """
    ceiling = LICENSE_CREDIBILITY.get((a.license or "").strip(), 0.0)
    try:
        declared = float(a.credibility)
    except (TypeError, ValueError):
        declared = 0.0
    if declared != declared:
        declared = 0.0
    if not ceiling:
        return 0.0
    if declared <= 0.0:
        return ceiling
    return max(0.0, min(declared, ceiling))


def is_negated(text: str, subject: str, type_name: str) -> bool:
    """True if negation scopes over the typed predicate for this subject.

    The cue must occur in the clause that mentions the type, not merely in an
    unrelated sentence. Fail-closed: missing or ambiguous text is negated.
    """
    raw = text or ""
    low = raw.lower()
    if not low.strip():
        return True
    target = _norm(type_name)
    if not target:
        return True
    toks = [(m.group(0).lower(), m.start(), m.end())
            for m in _WORD.finditer(raw)]
    hits = [i for i, (t, _s, _e) in enumerate(toks) if t == target.split()[0]]
    if not hits:
        return bool(_NEGATION.search(low))
    for i in hits:
        start = toks[max(0, i - 12)][1]
        end = toks[min(len(toks) - 1, i + 4)][2]
        if _NEGATION.search(low[start:end]):
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
    negations: dict[str, list[float]] = {}
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
            credibility=0.0,
        ))
    assertions.extend(extra)
    return admit(subject, assertions, floor=floor)


def signature(evidence: Sequence[Evidence]) -> tuple:
    """Comparable, content-addressed signature of an evidence set."""
    return tuple(sorted((e.subject, e.relation, e.type_name, e.source,
                         e.assertion_text) for e in evidence))
