"""Support / advice router (consultation for wellbeing & how-to-help).

Issue 2 (confirmed): advice/support questions like "I feel stressed,
healthy ways to manage stress?" ground to low-confidence
"multi_word_unconnected" and fall through to _human_like_uncertainty
("outside what i know"). There was NO router that recognized a
support/advice intent and sent it to the (now-working) web
learner. This module closes that gap.

Design:
  - Two-stage detection (no model required):
      1. Heuristic cues (emotion/wellbeing disclosure + question,
         direct advice request, support-seeking affect words).
      2. Optional GloVe centroid (intent_router style) over a
         support_seeking prototype fused from seed queries.
  - Routing: when support is detected, synthesize a web search
    query ("healthy ways to cope with stress") and ask the engine's
    EXISTING web-direct path (_web_direct_answer) to fetch +
    validate by source trust (healthline / WHO / NIH). The reply is
    prefaced with an epistemic hedge ("from what i've read...").
    Only if the web search genuinely fails do we fall back to
    honest uncertainty — never to a hollow graph guess.

Reuses existing infrastructure (SearchEngine, _web_direct_answer,
source-trust) — it does NOT hardcode answers; the advice text
comes from the live web, same as any other factual query.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

try:
    import numpy as np
    _HAS_NP = True
except Exception:  # pragma: no cover
    _HAS_NP = False
    np = None


# ── Stage 1: heuristic cues ──────────────────────────────────────
# Emotion/wellbeing disclosure + what/how question.
_HEURISTIC = [
    # "i'm stressed, healthy ways?" / "i feel anxious what should i do?"
    r"^(i'?m|i am|i feel|i'?ve been)\s+\w+.*\??\s*"
    r"(healthy|ways|what should|how (can|do|to)|coping|"
    r"help|advice|tips|strategies|deal with)",
    # direct advice request
    r"\b(should i|how can i|what can i do|any advice|tips for|"
    r"ways to|how do i cope|help me with|want to (learn|know)|"
    r"for a beginner)\b",
    # support-seeking affect, interrogative shape
    r"\b(stressed|anxious|overwhelmed|depressed|lonely|worried|"
    r"burned? out|sad|down)\b.*\??\s*"
    r"(what|how|any|ways|help|advice|tips|should)",
]
_HEURISTIC_RE = [re.compile(p, re.IGNORECASE) for p in _HEURISTIC]

# Affect / support keywords (also used as a cheap Stage-1 OR-gate and
# to pick the empathy-flavored hedge).
_SUPPORT_AFFECT = {
    "stress", "stressed", "anxious", "anxiety", "overwhelmed",
    "depressed", "depression", "lonely", "loneliness", "worried",
    "worry", "burnout", "burned", "sad", "down", "overwhelm",
    "tired", "exhausted", "hurt", "grief", "grieving",
}

# Query-shaping templates: (regex-trigger-word, web-query template).
# The template slots the user's topic so the web search is specific.
_TOPIC_TEMPLATES = [
    (r"\b(stress|stressed|anxious|anxiety|overwhelm|burnout|"
     r"burned? out|tired|exhausted)\b",
     "healthy ways to cope with {topic} and relax"),
    (r"\b(lonely|loneliness|alone)\b",
     "ways to deal with loneliness and feel connected"),
    (r"\b(depressed|depression|sad|down)\b",
     "evidence-based things that help with low mood"),
    (r"\b(worried|worry|anxious)\b",
     "practical ways to manage worry and anxiety"),
    (r"\b(sleep|insomnia|can't sleep)\b",
     "healthy habits to improve sleep"),
    (r"\b(programming|coding|learn to code|python|javascript)\b",
     "is python or javascript better for a beginner learning to code"),
    (r"\b(healthy|habit|habits|exercise|diet|eat|nutrition)\b",
     "sensible healthy habits to build"),
]

_HEDGE = "from what i've read, "


class SupportRouter:
    """Detect support/advice intent and route it to web-backed answer."""

    def __init__(self, glove_fn=None, threshold: float = 0.40):
        self._glove = glove_fn
        self._threshold = threshold
        self._anchor = None
        if glove_fn is not None and _HAS_NP:
            seeds = [
                "i feel stressed what should i do",
                "healthy ways to deal with anxiety",
                "how to cope with loneliness",
                "tips for managing stress",
                "what helps with depression",
                "how to deal with pressure",
                "i'm overwhelmed any advice",
                "ways to relax when anxious",
            ]
            toks = []
            for s in seeds:
                toks.extend(re.findall(r"[a-z']+", s))
            vecs = [np.asarray(glove_fn(w), dtype=float)
                    for w in toks if glove_fn(w) is not None]
            if vecs:
                self._anchor = np.mean(vecs, axis=0)

    # ── Stage 1: heuristic ─────────────────────────────────────
    def _heuristic_hit(self, low: str) -> bool:
        if any(rx.search(low) for rx in _HEURISTIC_RE):
            return True
        # OR-gate: a support affect word AND a question/interrogative
        has_affect = any(w in low for w in _SUPPORT_AFFECT)
        has_q = low.endswith("?") or re.match(
            r"^(what|how|which|should|can|could|do|any|ways|"
            r"tips|help)\b", low)
        return bool(has_affect and has_q)

    # ── Stage 2: GloVe centroid (optional) ──────────────────
    def _centroid_hit(self, low: str) -> bool:
        if self._glove is None or self._anchor is None or not _HAS_NP:
            return False
        toks = re.findall(r"[a-z']+", low)
        vecs = [np.asarray(self._glove(w), dtype=float)
                for w in toks if self._glove(w) is not None]
        if not vecs:
            return False
        qv = np.mean(vecs, axis=0)
        na = np.linalg.norm(qv)
        nb = np.linalg.norm(self._anchor)
        if na == 0.0 or nb == 0.0:
            return False
        sim = float(np.dot(qv, self._anchor) / (na * nb))
        return sim >= self._threshold

    # ── Public: detect support intent ─────────────────────────
    def is_support(self, user_input: str) -> bool:
        low = (user_input or "").lower().strip()
        if not low:
            return False
        return self._heuristic_hit(low) or self._centroid_hit(low)

    # ── Public: build a specific web query from the user text ──
    def build_query(self, user_input: str) -> str:
        low = (user_input or "").lower().strip()
        for rx, tmpl in _TOPIC_TEMPLATES:
            m = re.search(rx, low)
            if m:
                topic = m.group(1)
                return tmpl.format(topic=topic)
        # generic fallback: lift content words, drop stopwords
        _stop = {"i", "im", "am", "feel", "feeling", "my", "me", "a",
                 "an", "the", "is", "are", "do", "does", "did", "to",
                 "for", "with", "and", "of", "in", "on", "what",
                 "how", "should", "can", "any", "ways", "help",
                 "advice", "tips", "healthy", "ways", "you", "your"}
        words = [w for w in re.findall(r"[a-z']+", low)
                if w not in _stop and len(w) > 2]
        if words:
            return "helpful advice about " + " ".join(words[:6])
        return "healthy ways to feel better"


def route_support(engine, user_input: str) -> Optional[str]:
    """End-to-end: detect + fetch + hedge. Returns a web-backed
    support answer, or None when not a support query / web failed.

    Reuses the engine's EXISTING web-direct pipeline
    (_web_direct_answer), which already applies source trust
    (who.int / nih.gov / healthline / mayoclinic) and
    plausibility vetting. We only add a support-specific
    query shape + an epistemic hedge. Fail-open: if the
    engine lacks the web path, fall back to a raw
    source-trust-scored snippet select.
    """
    router = getattr(engine, "_support_router", None)
    if router is None or not router.is_support(user_input):
        return None
    # D1 fix (round v-aug06): respect RAVANA_OFFLINE. route_support performs a
    # LIVE web lookup via _web_direct_answer; in offline/reproducible mode it
    # must not hit the network (the flag is the documented CI/offline contract).
    # Falling through returns None so the turn proceeds to honest uncertainty
    # instead of emitting an unverified "from what i've read…" snippet.
    if getattr(engine, "_web_blocked", lambda: False)():
        return None
    query = router.build_query(user_input)
    try:
        # Prefer the engine's real web-direct path (source trust +
        # plausibility). Build a minimal CognitiveResponseContext.
        if hasattr(engine, "_web_direct_answer"):
            from ravana.chat.models import CognitiveResponseContext
            ctx = CognitiveResponseContext()
            ctx.raw_input = query
            ctx.subject = query.split()[0] if query.split() else query
            ans = engine._web_direct_answer(ctx)
            if ans and isinstance(ans, tuple) and ans[0]:
                body = ans[0].split(". ")[0][:240]
                return (f"{_HEDGE}{body}. (if it feels bigger than "
                        f"self-help, talking to someone you trust or a "
                        f"professional can really help.)")
        # Fallback: raw search + source-trust scoring.
        from ravana.web.learner import SearchEngine
        se = SearchEngine()
        res = se.search(query)
        snips = res.get("results", []) if isinstance(res, dict) else res
        if not snips:
            return None
        # Score by trustworty-domain preference, then length.
        _trusted = ("who.int", "nih.gov", "healthline", "mayoclinic",
                    "betterhelp", "psychologytoday", "sleepfoundation",
                    "cdc.gov", "nhs.uk", "helpguide")
        def _score(s):
            c = (s.get("content") or "").strip()
            url = (s.get("url") or "").lower()
            sc = 2 if any(t in url for t in _trusted) else 0
            return sc * 1000 + len(c)
        best = max(snips, key=_score)
        c = (best.get("content") or "").strip()
        if len(c) < 40:
            return None
        c = c.split(". ")[0][:240]
        return (f"{_HEDGE}{c}. (if it feels bigger than "
                f"self-help, talking to someone you trust or a "
                f"professional can really help.)")
    except Exception:
        return None
