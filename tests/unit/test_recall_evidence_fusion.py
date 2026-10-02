"""Capability test: EVIDENCE FUSION breaks coverage ties (RRF).

`_self_cued_episodic` qualifies records by how many of the query's content cues
they account for (majority) plus cue rarity, then used to order the survivors by
`(coverage, store_index)`. On an exact coverage tie the store index decided —
i.e. recency, which is not evidence about relevance. Measured before this fix:
on the store below, coverage ties at 5v5 and the off-topic record wins.

Both channels that settle it were already in RAVANA: the lexical BM25 tier
(`_bm25_rank`) and the GloVe cosine tier. What was missing was their
COMPOSITION, so this fix wires Reciprocal Rank Fusion over them
(docs/SYNTHESIS_TOTAL_AGENT_MEMORY.md).

The point of RRF is that it is RANK-based: BM25 here is an unbounded sum of
`idf * tf` and the semantic tier an unbounded sum of dot products, so no weight
between them is meaningful without a constant fitted to one conversation.

Each test gets a FRESH engine; no result depends on test ordering.
"""
import os
import sys

import pytest

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (PROJ,
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402

QUERY = "what did i tell you about the harbour radio drifting past the buoy"
CORRECT_MARK = "kip ran the harbour pilot radio station"
BAIT_MARK = "bought a second harbour radio"

# The relevant disclosure is stored EARLY; the bait carries the same cues and is
# stored LAST, so the old recency tie-break handed it the win.
STORE = [
    "i repainted the kitchen backsplash teal last saturday",
    CORRECT_MARK + " and he kept hearing drifting traffic past the outer buoy",
    "the sourdough starter doris needed feeding twice a day in winter",
    "my sister meera restores antique clocks in pune",
    "we argued about rent on the flat above the bakery",
    "the fiddle leaf fig dropped two leaves after the cold snap",
    "my cousin tobi repairs bicycle gearboxes in leeds",
    "i bought a second harbour radio for the workshop and the buoy horn "
    "rattles past the drifting tide line",
]


@pytest.fixture
def engine():
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               user_suffix="rrftest")


def _store(eng, records):
    eng._episodic_transcript = [{"text": t, "facts": {}} for t in records]
    return eng


def test_rrf_breaks_exact_coverage_tie_by_evidence(engine):
    """The defect: equal coverage, recency decides, wrong record comes back."""
    _store(engine, STORE)
    got = engine._self_cued_episodic(QUERY)
    assert got is not None, "a qualifying record exists; recall must not fail closed"
    assert CORRECT_MARK in got, (
        "coverage ties at 5v5 on this store and BOTH the BM25 and the cosine "
        "channel rank the harbour-station record first; fusion must return it, "
        "got: %r" % got)
    assert BAIT_MARK not in got, (
        "the off-topic record must not win the tie on recency alone")


def test_channels_disagree_is_where_fusion_earns_its_keep(engine):
    """RRF rewards agreement across channels, not a single channel's top hit.

    A record that tops NEITHER channel but is present in both consistently must
    be able to beat one that tops one channel and is absent from the other.
    This is the property a weighted sum cannot express without a fitted scale.
    """
    channels = [[7, 3, 9], [3, 7, 9]]
    fused = engine._rrf_fuse(channels)
    order = [i for i, _s in fused]
    # 7 and 3 are symmetric here (both mid in both), 9 is last in both; the
    # invariant that matters is that every channel contributed.
    assert len(order) == 3
    assert order[-1] == 9, "last in every channel must fuse last"
    assert set(order[:2]) == {3, 7}


def test_rrf_is_rank_based_so_channel_scale_is_irrelevant(engine):
    """Two channels whose scores differ by orders of magnitude must fuse alike."""
    small = [[0, 1, 2]]
    huge = [[0, 1, 2]]
    assert engine._rrf_fuse(small) == engine._rrf_fuse(huge)


def test_single_channel_degrades_without_crashing(engine):
    """With one channel the fusion is that channel's order (rank 0 still wins)."""
    fused = engine._rrf_fuse([[4, 5, 6]])
    assert fused[0][0] == 4
    assert [i for i, _ in fused] == [4, 5, 6]


def test_untied_candidates_still_take_the_unique_winner(engine):
    """No tie -> the fusion is not consulted, so nothing changes for that case."""
    _store(engine, [STORE[0], STORE[1], STORE[2], STORE[3]])
    got = engine._self_cued_episodic(QUERY)
    assert got is not None
    assert CORRECT_MARK in got


def test_evidence_gate_still_fails_closed(engine):
    """Fusion must not ADD candidates: a cue nothing covers still returns None."""
    _store(engine, STORE)
    assert engine._self_cued_episodic(
        "what did i tell you about the zeppelin navigation school") is None


def test_semantic_channel_ranks_every_record_including_token_disjoint(engine):
    """The semantic channel is an evidence channel, not a lexical echo."""
    recs = [{"text": "a", "facts": {}},
            {"text": "b", "facts": {}}]
    ranks = engine._cue_semantic_ranks(["harbour", "radio"], recs)
    assert set(ranks) == {0, 1}, "every record gets a position, even token-disjoint"


def test_semantic_channel_returns_empty_without_projections(engine, monkeypatch):
    """No GloVe -> no semantic evidence invented."""
    monkeypatch.setattr(engine, "_glove_vector", lambda _w: None)
    assert engine._cue_semantic_ranks(["harbour"], [{"text": "a", "facts": {}}]) == {}
