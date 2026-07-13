"""Regression tests for the web-snippet quality leak (Roblox UGC bug).

A "how can i become invisible right now" query surfaced a Roblox UGC gear
description ("Invisible is a gear that makes you invisible in Roblox") as the
literal answer. The correct (brain-like) fix is NOT a hardcoded domain
blocklist — humans don't keep a list of "bad websites". Instead the agent
(1) monitors whether the retrieved answer's *added* content actually coheres
with the question's topic (the N400 / plausibility / reality-monitoring check,
Johnson & Raye 1981; Kuperberg & Jaeger), and (2) if it doesn't serve the
query, REFINES the search and re-tries (metacognitive control / second-pass
reanalysis) — withholding only if even the refined search can't produce a
plausible answer. Critically the check is a criterion on a *semantic dimension*,
so it rejects game wikis, spam, or any incoherent source without naming any.

Run from repo root:
    python -m pytest tests/unit/test_web_snippet_source_quality.py -v
"""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana_ml", "src"))

from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.models import CognitiveResponseContext


def _build_engine():
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               data_dir="/tmp/ravana_snippet_quality_test")


ROBLOX = "Invisible is a gear that makes you invisible in Roblox."
WIKI = "Invisibility is the state of an object that cannot be seen by others."


# ── 1. _snippet_plausibility discriminates incoherent restatements ────────────
def test_snippet_plausibility_discriminates():
    eng = _build_engine()
    roblox_p = eng._snippet_plausibility("invisible", ROBLOX)
    wiki_p = eng._snippet_plausibility("invisible", WIKI)
    assert roblox_p is not None and wiki_p is not None
    # The Roblox restatement's *added* content (gear, roblox) is incoherent with
    # "invisible"; the encyclopedic definition's added content coheres.
    assert roblox_p < eng._SNIPPET_PLAUSIBILITY_FLOOR <= wiki_p


# ── 2. query refinement re-frames a how-to query toward the real world ────────
def test_refine_query_variants():
    eng = _build_engine()
    refined = eng._refine_query_variants("how can i become invisible right now",
                                         "invisible")
    assert any("real life" in r or "real world" in r for r in refined)
    # factual query gets a real-world disambiguator too
    refined2 = eng._refine_query_variants("what is invisible", "invisible")
    assert refined2


# ── 3. END-TO-END: a Roblox-style snippet (any domain) is WITHHELD, not leaked ─
# We mock the search engine so no network is needed. The snippet has a perfectly
# innocuous URL — proving the fix is content-based, not a domain blocklist.
def test_roblox_snippet_withheld_end_to_end():
    eng = _build_engine()
    roblox_result = [{
        "title": "Invisible | Some Wiki",
        "url": "https://example.com/invisible",  # benign domain on purpose
        "content": ROBLOX,
    }]

    def _mock_search(term, max_results=6, local_only=False):
        return roblox_result

    eng.search_engine.search = _mock_search
    ctx = CognitiveResponseContext(subject="invisible",
                                   raw_input="how can i become invisible right now")
    result = eng._web_direct_answer(ctx)
    assert result is None, f"Roblox-style snippet leaked as answer: {result}"


# ── 4. END-TO-END: an encyclopedic snippet is still surfaced ──────────────────
def test_encyclopedic_snippet_surfaced_end_to_end():
    eng = _build_engine()
    wiki_result = [{
        "title": "Invisible - Encyclopedia",
        "url": "https://example.com/invisible",
        # Contains the subject token "invisible" so it passes the relevance
        # gate, and its added content coheres with the topic.
        "content": "Invisible is the state of an object that cannot be seen by others.",
    }]

    def _mock_search(term, max_results=6, local_only=False):
        return wiki_result

    eng.search_engine.search = _mock_search
    ctx = CognitiveResponseContext(subject="invisible",
                                   raw_input="how can i become invisible right now")
    result = eng._web_direct_answer(ctx)
    assert result is not None
    assert "invisible" in result[0].lower()
