# Multi-Signal Recall Relevance Scorer (MRS)

## What it is

The Multi-Signal Recall Relevance Scorer (MRS) ranks recall candidates from
the `PersonalFactStore` using multiple weighted signals — content overlap,
graph/semantic distance, recency, type alignment, entity match, confidence,
and query-type weighting — instead of returning the first match.

It replaces the old first-match-wins behavior in the recall pipeline with a
scored, threshold-gated selection that adapts online from conversation
feedback.

## Why it exists

Before MRS, recall candidates were returned on a first-match basis. When
multiple facts matched a query (e.g. two different pets' names), the system
returned whichever fact was encountered first in the store — not necessarily
the most relevant one. There was no mechanism to prefer a higher-confidence
fact, a more recent fact, or a fact whose entity matched the query.

MRS addresses this by scoring every candidate against the query using
multiple signals, then filtering by a learned threshold. The threshold
adapts incrementally from false-positive and false-negative feedback.

## How it grew from the conversation

The capability was identified as limitation L1 (recall relevance) during the
2026-10-12 autonomous round. The round's analysis found that recall queries
with multiple matching facts returned inconsistent results — sometimes the
right fact, sometimes not — because there was no relevance scoring layer.

The implementation adds:

1. **`RecallRanker` class** (`ravana/src/ravana/chat/recall_ranker.py`) —
   the core ranking engine. Stateless; all parameters live in the
   `RelevanceThreshold` data store.

2. **Query-type detection** — classifies queries into 9 types (biographical,
   entity, topical, stance, activity, possessive, material, count,
   whereabouts) using the same regex patterns already used by
   `_structured_recall`. No new patterns.

3. **Entity match bonus** — if the query contains an entity word that appears
   in the candidate's fact key, the candidate gets a bonus (capped at 0.15).

4. **Confidence weighting** — facts with higher confidence (from
   `PersonalFactStore`) get a small bonus (capped at 0.10).

5. **Query-type weighting** — each query type has per-signal weight
   adjustments. For example, biographical queries weight content overlap
   higher (1.2) and recency lower (0.6), while stance queries weight graph
   distance higher (1.3).

6. **Threshold adaptation** — the `RelevanceThreshold.base_threshold` adapts
   incrementally via `adapt_threshold()`: false positives raise it, false
   negatives lower it, bounded to [0.05, 0.5]. The adaptation rate is 0.01
   per step.

7. **Relevance gate** (`engine.py:_relevance_gate`) — wraps the existing
   `_structured_recall` path. After a candidate answer is produced, MRS
   scores it against the query. If the score is below the threshold, the
   gate returns `None` (fail-closed: honest uncertainty beats confabulation).

8. **Feedback wiring** (`engine_persistence.py`) — when a user correction is
   processed, `wire_feedback()` is called on the ranker, which records a
   false positive and triggers threshold adaptation.

## Architecture

```
query ──▶ detect_query_type() ──▶ query_type
                                        │
candidates ──▶ collect_candidates() ──▶ [RecallCandidate]
                                        │
                                        ▼
                              rank() per candidate:
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
              content overlap    graph distance      recency decay
              (Jaccard)          (GloVe cosine)     (exp(-0.01 * Δturn))
                    │                   │                   │
                    └───────────┬───────┘                   │
                                │                           │
                          type alignment                    │
                                │                           │
                                ▼                           │
                    weighted sum × query_type_weights       │
                                │                           │
                    ┌───────────┴───────────┐               │
                    │                       │               │
              entity match bonus    confidence weight       │
              (≤ 0.15)              (≤ 0.10)               │
                    │                       │               │
                    └───────────┬───────────┘               │
                                │                           │
                                ▼                           ▼
                              score ∈ [0, 1]
                                │
                                ▼
                    rank_and_filter(threshold)
                                │
                                ▼
                    filtered candidates
```

## SEED compliance

All ranking parameters live in the `RelevanceThreshold` data store
(`ravana/src/ravana/chat/relevance_store.py:122`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_threshold` | 0.15 | Minimum score to accept a candidate |
| `content_weight` | 0.40 | Weight for content overlap signal |
| `graph_weight` | 0.25 | Weight for graph/semantic distance |
| `recency_weight` | 0.20 | Weight for recency decay |
| `type_weight` | 0.15 | Weight for type alignment |
| `adaptation_rate` | 0.01 | Step size for threshold adaptation |

The threshold persists via `to_dict()` / `from_dict()` and survives
save/load cycles. No hardcoded values, no per-topic tables, no retraining.

## Online learning

The threshold adapts via `adapt_threshold()`
(`ravana/src/ravana/chat/relevance_store.py:263`):

- **False positive** (returned irrelevant result): `base_threshold += 0.01`
  (capped at 0.5)
- **False negative** (missed relevant result): `base_threshold -= 0.01`
  (capped at 0.05)
- Adaptation only fires when `total_scored >= 10` (enough data)
- The adaptation is incremental — one step per feedback event, not a batch
  re-fit

## Query-type weights

Defined in `recall_ranker.py:56-66`:

| Query type | Content | Graph | Recency | Type |
|------------|---------|-------|---------|------|
| biographical | 1.2 | 0.8 | 0.6 | 1.3 |
| entity | 1.3 | 0.7 | 0.5 | 1.4 |
| topical | 1.1 | 1.2 | 0.8 | 0.9 |
| stance | 0.9 | 1.3 | 0.7 | 1.2 |
| activity | 1.0 | 1.0 | 0.9 | 1.0 |
| possessive | 1.2 | 0.8 | 0.6 | 1.3 |
| material | 1.1 | 1.0 | 0.7 | 1.1 |
| count | 1.0 | 0.9 | 0.8 | 1.0 |
| whereabouts | 1.1 | 0.9 | 0.7 | 1.1 |

Unknown query types use uniform weights (1.0 for all signals).

## Test coverage

65 tests across two files:

- `tests/test_recall_ranker.py` — 38 unit tests covering query-type
  detection, candidate ranking, entity match bonus, confidence weighting,
  query-type weights, score_text, feedback wiring, candidate collection,
  and threshold persistence.
- `tests/test_mrs_capability.py` — 27 acceptance tests covering SEED data
  store compliance, online learning, multi-signal ranking, and end-to-end
  pipeline behavior.

All 65 tests pass (verified: `pytest tests/test_recall_ranker.py
tests/test_mrs_capability.py` → 65 passed in ~1.5s).

## Key source locations

- `ravana/src/ravana/chat/recall_ranker.py` — RecallRanker class, query-type
  patterns, entity match bonus, confidence weighting, query-type weights
- `ravana/src/ravana/chat/relevance_store.py:122` — RelevanceThreshold data
  store (SEED parameters)
- `ravana/src/ravana/chat/relevance_store.py:263` — adapt_threshold (online
  learning)
- `ravana/src/ravana/chat/engine.py` — _relevance_gate (recall pipeline
  integration)
- `ravana/src/ravana/chat/engine_persistence.py` — wire_feedback (correction
  → threshold adaptation)
