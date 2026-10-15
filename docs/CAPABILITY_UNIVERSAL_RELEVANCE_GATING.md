# Capability: Universal Relevance Gating

## What it does

Universal Relevance Gating ensures that **every** recall path in RAVANA's
cognitive engine verifies a candidate answer's relevance to the query before
returning it. Without this gate, recall paths that match by content-word
overlap alone could confabulate — returning an answer that shares words with
the query but does not actually address it.

The gate is **fail-closed**: if the multi-signal relevance score is below the
learned threshold, the candidate is rejected and the turn falls through to
honest uncertainty or another strategy. This is the same principle as
RAVANA's abstention behavior — confident-wrong is treated as high free-energy
that would poison the graph.

## How it grew from the conversation

The Multi-Signal Recall Relevance Scorer (MRS) infrastructure was added in
round 2026-10-13T1411Z (PR #118) and the `_relevance_gate` was wired into
`_structured_recall`. However, the episodic recall path
(`_self_cued_episodic`) and the agent-own recall path
(`_route_agent_own_recall`) were **not** gated — they could still confabulate
by matching on content-word overlap without verifying the answer actually
addresses the query.

This capability wires the same gate into the episodic recall path at the
`process_turn` call site, so **all** recall paths are relevance-gated.

## Design

### 1. Call-site wiring (not inside the recall method)

The gate is applied at the `process_turn` call site
(`ravana/src/ravana/chat/engine.py:7180-7181`), NOT inside
`_self_cued_episodic` itself. This is the same pattern used for
`_structured_recall` (line 7108). The gate wraps the caller's use of the
recall method, so the recall method itself stays pure (returns the best
match) and the gate decides whether to accept it.

```python
# ravana/src/ravana/chat/engine.py:7180-7181
if _cued_self is not None:
    _cued_self = self._relevance_gate(user_input, _cued_self)
```

### 2. Fail-closed behavior

If the MRS score is below the threshold, the gate returns `None`, causing the
turn to fall through to honest uncertainty or another strategy. This is the
same fail-closed behavior as the structured recall gate.

### 3. No new store, no new data structure

This is a pure wiring change that applies the existing MRS scorer
(`RecallRanker.score_text`) to the episodic recall path. The scorer,
threshold, and adaptation mechanism are all from the existing
`relevance_store.py` infrastructure.

## Why this is SEED (not hardcoding)

- The relevance threshold (`base_threshold`) is a **persisted data store**
  value (`RelevanceThreshold` in `ravana/src/ravana/chat/relevance_store.py:122`),
  not a constant in code.
- The MRS scorer reads from the `RelevanceThreshold` data store, which
  **adapts incrementally** from conversation feedback via `adapt_threshold()`
  (`ravana/src/ravana/chat/relevance_store.py:263`).
- No per-topic logic, no hardcoded reply strings, no keyword-to-response
  tables.
- Online/incremental: no retraining, no offline rebuild.
- The test "Can RAVANA change this by itself, through experience?" — **YES**:
  the threshold adapts via `adapt_threshold()` based on false
  positive/negative signals.

## The gate mechanism

The `_relevance_gate` method (`ravana/src/ravana/chat/engine.py:2994-3017`):

1. Calls `RecallRanker.score_text(query, answer)` to compute a multi-signal
   relevance score in [0, 1].
2. Compares the score against `self._relevance_threshold.base_threshold`.
3. If score >= threshold: return the answer (pass).
4. If score < threshold: return `None` (reject — fail closed).
5. On scorer error: return the answer (fail-open — never break recall).

The `RecallRanker` (`ravana/src/ravana/chat/recall_ranker.py:96`) computes
the score from:
- Content-word overlap (Jaccard)
- Graph/semantic distance (GloVe cosine)
- Recency (exponential decay by turn distance)
- Type match (entity/attribute alignment)
- Entity match bonus
- Confidence weighting
- Query-type weighting

All learned parameters live in the `RelevanceThreshold` data store. The
ranker itself is stateless.

## Online learning: threshold adaptation

The threshold adapts via `adapt_threshold()`
(`ravana/src/ravana/chat/relevance_store.py:263-289`):

- **False positives** (returned irrelevant results) raise the threshold
  (be more selective).
- **False negatives** (missed relevant results) lower the threshold
  (be more permissive).
- Simple gradient descent on the error rate — no retraining, just
  incremental parameter adjustment.
- Adaptation only fires after `total_scored >= 10` (enough data).

Signals are recorded via `record_false_positive()` and
`record_false_negative()` (`relevance_store.py:292-305`).

## Tests

`tests/unit/test_universal_relevance_gating.py` — 3 tests:

1. **`test_self_cued_episodic_answer_is_relevance_gated`** — verifies the
   gate rejects low-relevance episodic answers at high threshold and passes
   them at default threshold.
2. **`test_relevance_gate_is_applied_at_process_turn_level`** — verifies the
   gate is applied at the `process_turn` call site, not just when
   `_self_cued_episodic` is called directly.
3. **`test_structured_recall_still_gated`** — regression: verifies the
   existing structured recall gating still works.

## What this does NOT do

- Does NOT add an LLM.
- Does NOT add hardcoded reply strings.
- Does NOT use per-topic lookup tables.
- Does NOT require retraining or offline rebuild.
- Does NOT replace the existing recall pipeline — it adds a relevance gate
  layer on top.
- Does NOT change `_self_cued_episodic` itself — the gate is applied by the
  caller.
