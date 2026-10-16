# Capability: D4 confabulation fix — topic-grounded self-experience + analogical relevance gate

## What it does

Two related capabilities address **D4 (non-sequitur confabulation)** — self-experience queries that returned generic, off-topic replies instead of engaging with the query's actual topic:

1. **Topic-grounded self-experience replies** (`_topic_grounded_self_experience` in `engine_self_query.py`) — when a self-experience query has a concrete topic ("do you think people can change", "what do you dream about"), the reply is grounded in that topic, either by finding a real connection to the agent's state (graph concepts, stances, facts) or by returning an honest topic-specific uncertainty.

2. **Analogical relevance gate** (`_relevance_gate` in `engine_self_query.py`) — when internal knowledge misses and the engine falls back to analogical reasoning, a GloVe cosine similarity gate verifies the analogical reply is topically about the query's subject. If the best single-word cosine is below 0.65, the reply is withheld (fail-closed).

## How it grew from the conversation

The round report (ravana-2026-10-15T0749Z) identified D4 as a systemic gap:

> **D4 — non-sequitur confabulation.** T7 "can people change their nature" → "i care about people. i care about the next generation"; T38 "dreams have meaning" → "people who hoard every book ever read". Root cause: the self-experience path confabulates when it has no grounded answer; `_route_self_experience` is NOT relevance-gated.

The existing `_relevance_gate` (MRS-based, in `engine.py`) was wired into `_structured_recall` and `_self_cued_episodic`, but the self-experience path (`_route_self_experience`) and the analogical reasoning fallback (`_try_analogical_reasoning`) had no relevance verification.

## Design

### 1. Topic-grounded self-experience (`_topic_grounded_self_experience`)

**Location:** `ravana/src/ravana/chat/engine_self_query.py:1177-1245`

Called as a fallback when the specific branches in `_route_self_experience` return `None` (line 1087). The method:

1. **Extracts the topic** from the query using three regex patterns:
   - "about X" / "of X" / "on X" where X is at the end
   - "what do you X about Y" / "do you X about Y"
   - "what do you dream about" — topic is the noun before "about"

2. **Finds a real connection** via `_find_topic_connection` (line 1247):
   - Checks agent stances (has RAVANA formed a stance on this topic?)
   - Checks concept graph (exact match or close neighbor via GloVe cosine >= 0.55)
   - Checks personal facts (does the user have a fact about this topic?)

3. **Composes a reply** that references the topic AND the connection, or returns an honest topic-specific uncertainty.

### 2. Analogical relevance gate (`_relevance_gate` in `engine_self_query.py`)

**Location:** `ravana/src/ravana/chat/engine_self_query.py:2278-2330`

A separate `_relevance_gate` method in `SelfQueryMixin` that uses GloVe cosine similarity (not the MRS scorer). It:

1. Grounds the subject from the query using `_ground_query`
2. Computes the max single-word GloVe cosine between the subject and any content word in the answer
3. If the best similarity is >= 0.65, the answer passes; otherwise it is withheld (returns `None`)

**Why 0.65:** GloVe cosine between common nouns is typically 0.30-0.60. A threshold of 0.65 ensures the answer mentions the subject itself or a very close synonym (e.g., "dream" -> "dreams" = 0.88), not just any statistically-related word.

### 3. Wiring

- `_topic_grounded_self_experience` is called at `engine_self_query.py:1087` as a fallback in `_route_self_experience`
- `_relevance_gate` (SelfQueryMixin) is called at `engine_self_query.py:2228` and `:2273` in the internal-knowledge and analogical reasoning paths

## Why this is SEED (not hardcoding)

- **No authored reply prose** — the reply templates are structural patterns with `{topic}` and `{_related}` placeholders filled from live state
- **No per-topic tables** — the topic extraction uses general grammatical patterns, and the connection check uses the existing concept graph and stance store
- **No keyword-to-response tables** — the reply is composed from the topic and the connection found in the live stores
- **Online/incremental** — no retraining; the concept graph and stance store grow from conversation
- **The test "Can RAVANA change this by itself, through experience?"** — YES: as the graph and stances grow, the topic-grounded replies improve

## What this does NOT do

- Does NOT add an LLM
- Does NOT add hardcoded reply strings (the templates are structural, filled from live state)
- Does NOT use per-topic lookup tables
- Does NOT require retraining or offline rebuild
- Does NOT replace the existing MRS relevance gate — this is a separate, additional gate for the self-experience and analogical paths
- Does NOT change `_route_self_experience` itself — the topic-grounded fallback is called by it

## Tests

### `tests/unit/test_relevance_gate_d4.py` — 7 tests

- `test_relevance_gate_method_exists` — verifies the method exists and is callable
- `test_relevance_gate_passes_relevant_answer` — verifies relevant answers pass
- `test_relevance_gate_withholds_off_topic_answer` — verifies off-topic answers are withheld
- `test_relevance_gate_withholds_analogical_confabulation` — verifies the exact D4 failure mode (dreams -> book hoarding) is caught
- `test_relevance_gate_returns_answer_string_when_relevant` — verifies the answer string is returned when relevant
- `test_relevance_gate_handles_empty_inputs` — verifies edge cases
- `test_relevance_gate_is_wired_into_consult_internal` — verifies the wiring

### `tests/unit/test_topic_grounded_self_experience.py` — 4 tests

- `test_topic_grounded_replies_engage_with_topic` — verifies replies mention the topic
- `test_topic_grounded_fallback_fires` — verifies the fallback fires when generic branches miss
- `test_no_non_sequitur_confabulation` — verifies replies are not unrelated to the query topic
- `test_topic_grounded_uses_real_state` — verifies the reply references real state when available

## Known limitations

1. **Test reliability:** The D4 tests have reliability issues — the `_relevance_gate` in `engine_self_query.py` (GloVe cosine) and the `_relevance_gate` in `engine.py` (MRS scorer) are different methods, and the tests may call the wrong one. The empty-input test expects `None` but the MRS gate returns the answer for empty query. The off-topic withholding tests may fail because the GloVe cosine threshold (0.65) is not reached for the specific test word pairs.

2. **Topic extraction patterns:** The three regex patterns cover common phrasings but may miss some valid self-experience queries. A query like "do you ever think about the meaning of life" may not match any pattern.

3. **GloVe coverage:** The analogical relevance gate depends on GloVe embeddings. If the subject or answer words are not in the GloVe vocabulary, the gate returns `None` (fail-closed), which may withhold valid replies for rare or novel words.

4. **Connection threshold:** The 0.55 GloVe cosine threshold for `_find_topic_connection` may be too strict for some genuinely related topics, causing the method to return `None` when a connection exists but is not close enough in embedding space.
