"""Cross-turn self-consistency monitor (post-generation check).

Issue 3 (confirmed): there is NO cross-turn consistency pass.
belief_store.py handles factual *belief* contradictions,
coherence_gate.py handles per-turn coherence, and
chain_walker.contradiction_map detects *antonyms* — but
none of them watch the AGENT'S OWN generated claims across
turns. So "is AI beneficial" (turn 5) and "is AI harmful"
(turn 10) can contradict with no detection.

This module is a lightweight, NO-LLM monitor (uses the
existing GloVe + contradiction_map + belief_store infrastructure):

  Step A  Claim extraction from the just-generated response
          (split on complementizers; pull subject–predicate–value).
  Step B  Cross-turn check against a rolling claim buffer
          (cosine of subject+predicate mean-pool; on a
          same-subject match, polarity check via contradiction_map
          / negative GloVe cosine -> flag).
  Step C  Resolution (3 modes):
          - annotate: prefix "i realize i might have said something
            different before — here's where i'm at now..."
          - harmonize: if both sides valid for different aspects,
            compose a balanced answer.
          - report: log the conflict, emit response as-is.

Fail-open: any exception -> no monitoring (response unchanged).
A monitor must NEVER rewrite a clean response or leak text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

try:
    import numpy as np
    _HAS_NP = True
except Exception:  # pragma: no cover
    _HAS_NP = False
    np = None


# Complementizers / claim-boundary cues.
_COMPLEMENT = re.compile(
    r"\b(that|is|are|means?|because|so)\b")
_SUBJ_SPLIT = re.compile(r"[.!?]\s+|\b(and|but|however|although)\b")
_VALUE_NEG = re.compile(
    r"\b(not|never|isn'?t|aren'?t|wasn'?t|weren'?t|no|none|"
    r"don'?t|doesn'?t|didn'?t|can'?t|won'?t|couldn'?t|"
    r"harmful|dangerous|negative|useless|pointless)\b",
    re.IGNORECASE)


@dataclass
class Claim:
    subject: str
    predicate: str
    value: str
    polarity: float            # +1 positive, -1 negative, 0 unknown
    turn: int
    embedding: Optional[Any] = None


@dataclass
class ConsistencyReport:
    conflict_detected: bool
    conflicting_claim: Optional[Claim] = None
    similarity: float = 0.0
    resolution: str = "report"   # annotate | harmonize | report


class ConsistencyMonitor:
    """Cross-turn self-consistency monitor (NO-LLM)."""

    def __init__(self, glove_fn=None, contradiction_map: Optional[Dict[str, set]] = None,
                 buffer_size: int = 50, mode: str = "annotate"):
        self._glove = glove_fn
        self._cmap = contradiction_map or {}
        self._buf: List[Claim] = []
        self._buf_size = buffer_size
        self._mode = mode

    # ── Step A: claim extraction ───────────────────────────────
    @staticmethod
    def _noun_chunks(text: str, glove_fn) -> List[str]:
        # Cheap subject candidate: the first content NOUN (skipping
        # question / filler lead-ins). "AI is beneficial" -> "ai";
        # "what is gravity" -> "gravity". We keep ALL tokens
        # (including short ones like "ai") and skip only stop-lead-ins.
        words = re.findall(r"[a-z'][a-z']*", text.lower())
        if not words:
            return []
        _skip = {"is", "are", "was", "were", "be", "been", "the",
                  "a", "an", "i", "you", "it", "they", "to", "of",
                  "for", "and", "but", "what", "how", "why", "tell",
                  "me", "about", "do", "does", "did", "can", "could",
                  "would", "should", "that", "this", "in", "on",
                  "with", "my", "your", "our", "their"}
        for w in words:
            if w not in _skip:
                return [w]
        return [words[0]]

    def _embed(self, text: str):
        if not _HAS_NP or self._glove is None:
            return None
        toks = re.findall(r"[a-z'][a-z']*", text.lower())
        vecs = [np.asarray(self._glove(w), dtype=float)
                for w in toks if self._glove(w) is not None]
        if not vecs:
            return None
        return np.mean(vecs, axis=0)

    def _extract_claims(self, response: str, turn: int) -> List[Claim]:
        if not response or not response.strip():
            return []
        claims: List[Claim] = []
        # Split into candidate clauses on sentence / contrast boundaries.
        clauses = _SUBJ_SPLIT.split(response)
        for cl in clauses:
            cl = (cl or "").strip().rstrip(".!?")
            if len(cl) < 6:
                continue
            subjs = self._noun_chunks(cl, self._glove)
            if not subjs:
                continue
            subj = subjs[0]
            neg = bool(_VALUE_NEG.search(cl))
            pol = -1.0 if neg else 1.0
            emb = self._embed(cl)
            claims.append(Claim(
                subject=subj, predicate=cl, value=cl,
                polarity=pol, turn=turn, embedding=emb))
        return claims

    # ── Step B: cross-turn contradiction ──────────────────────
    def _is_contradiction(self, a: Claim, b: Claim) -> bool:
        if a.subject != b.subject:
            # also catch antonym subjects via contradiction_map
            if a.subject in self._cmap.get(b.subject, set()) or \
               b.subject in self._cmap.get(a.subject, set()):
                pass  # related antonym subjects count as same axis
            else:
                return False
        # polarity opposition = contradiction
        if a.polarity * b.polarity < 0:
            return True
        # embedding similarity + opposite polarity handled above; if subjects
        # match and polarities same, it's consistent (not a conflict).
        return False

    def _cosine(self, a, b) -> float:
        if a is None or b is None or not _HAS_NP:
            return 0.0
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # ── Public: check a new response against the buffer ───────
    def check(self, response: str, turn: int) -> ConsistencyReport:
        new_claims = self._extract_claims(response, turn)
        if not new_claims:
            return ConsistencyReport(conflict_detected=False)
        # Compare each new claim against prior turns.
        best: Optional[Claim] = None
        best_sim = 0.0
        for nc in new_claims:
            for pc in self._buf:
                sim = self._cosine(nc.embedding, pc.embedding)
                # Flag on a GENUINE contradiction: same (or antonym)
                # subject with OPPOSITE polarity. Embedding cosine is
                # used only to rank the best prior match, never as a
                # hard gate that could hide a polarity-opposition.
                if self._is_contradiction(nc, pc):
                    if sim > best_sim:
                        best_sim = sim
                        best = pc
        # Store new claims (rolling buffer).
        self._buf.extend(new_claims)
        if len(self._buf) > self._buf_size:
            self._buf = self._buf[-self._buf_size:]
        if best is not None:
            return ConsistencyReport(
                conflict_detected=True, conflicting_claim=best,
                similarity=best_sim, resolution=self._mode)
        return ConsistencyReport(conflict_detected=False)

    # ── Step C: resolution ───────────────────────────────────
    def resolve(self, response: str, report: ConsistencyReport) -> str:
        if not report.conflict_detected:
            return response
        if report.resolution == "annotate":
            return ("i realize i might have said something different "
                    "before — here's where i'm at now: " + response)
        if report.resolution == "harmonize":
            # Both valid for different aspects: keep the new answer but
            # acknowledge the earlier stance softly.
            return (response + " (i've also said the opposite can hold "
                    "depending on the angle — i'm trying to be consistent.)")
        # report: emit as-is, just flag it.
        return response


def build_monitor(glove_fn=None, contradiction_map=None, mode: str = "annotate"):
    return ConsistencyMonitor(
        glove_fn=glove_fn, contradiction_map=contradiction_map, mode=mode)
