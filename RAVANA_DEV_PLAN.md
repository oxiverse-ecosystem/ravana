# RAVANA Development Plan: From Prototype to Usable Local App

## Core Diagnosis

| Problem | Impact | Root Cause |
|---------|--------|------------|
| Responses are random associations | Low coherence | 180 concepts, 2250 edges — too sparse for meaningful paths |
| 5-18s latency | Unusable in conversation | Synchronous web search on every unknown word |
| Chains drift into unrelated territory | Reads incoherently | No anchor-back mechanism after 2+ hops |
| Conversation memory is fragile | Can't recall past topics | Episodic edges (w=0.15) are too weak; no structured recall |
| 15-search session cap | Stops learning mid-conversation | Artificial cap, not a data structure limit |

## Phase 1: Auto-Expansion (Every Message → Graph Growth)

**Goal**: Every user input automatically grows the graph via GloVe, no web search needed. Target: 5000+ concepts after 50 conversations.

### Steps

#### 1.1 Add every input word as a concept
- For every user message, extract meaningful words (len >= 3, not STOP_WORDS)
- If word not in graph AND in GloVe: add as concept node with GloVe vector
- If word not in graph AND not in GloVe: skip (web search can handle later)
- Wire to top-5 GloVe nearest neighbors (cosine > 0.4) as semantic edges, weight = `min(0.4, sim * 0.4)`

#### 1.2 Auto-wire every new concept to existing concepts via GloVe
- For each new concept, compute cosine similarity against ALL existing concept vectors
- Add semantic edges where > 0.5 threshold, weight = `min(0.5, sim * 0.5)`
- O(N * D) per new concept — N grows but vector lookup is fast numpy

#### 1.3 Batch web learning to async
- Don't block the response on web search
- Collect unknown (non-GloVe) words in a queue
- After responding, fire web search in background (or skip if offline)
- Store results for next conversation

#### 1.4 Remove the 15-search cap
- Replace `_learning_count < 15` with a per-session rate limit (max 1 search per 3 turns)
- No hard lifetime cap — graph should grow unbounded

### Verification

```bash
# Before: 180 concepts, 2250 edges
python scripts/ravana_chat.py --reset --chat "quantum entanglement is mysterious" --strategy
# After: inspect --trace for new concepts "quantum", "entanglement", "mysterious"
# Each should appear in the graph with 5+ auto-wired edges

# Growth test — send 10 messages with different topics
python scripts/ravana_chat.py --reset --chat "i love astrophysics|black holes are fascinating|tell me about gravity|what is a neutron star|dark matter is everywhere|string theory is complex|quantum mechanics is weird|relativity changed everything|the big bang theory|cosmic inflation" --strategy
# Check final save size increased, new concepts present
# Verify no 15-search cap error in output
```

## Phase 2: Sub-Second Response Time

**Goal**: Every response renders in <1s regardless of learning state.

### Steps

#### 2.1 Pre-compute and cache concept vectors
- Cache `_glove_vector(label)` results in a dict (avoid recomputing projection)
- Cache GloVe neighbor lookups in a nearest-neighbor index

#### 2.2 Offload web search from response path
- Move `learn_from_web()` to fire AFTER `generate_response()` returns
- Deferred learning: store unknown words, learn between turns
- If user says "bye", flush pending learning before save

#### 2.3 Warm-start GloVe loading
- Save projected vectors alongside pickle so restart doesn't re-project 400K words
- Store `_glove_vecs` only in memory, not on disk (already optimal)

#### 2.4 Optimize chain walking hot loop
- Candidate sort is O(N log N) per hop — replace with `np.argpartition` for top-K
- Pre-compute `_EDGE_CONNECTORS` reverse lookups once in `__init__`

### Verification

```bash
# Timing check
Measure-Command { python scripts/ravana_chat.py --reset --chat "hello|what is life|tell me more|why is the sky blue|who are you" --strategy }
# Average per-turn time should be < 1s (web search excluded)
# First load may be slower (GloVe file read) but interactive turns < 1s

# Compare with older timing baseline:
# Previous 10-turn run was ~72s total → should drop to < 10s
```

## Phase 3: Conversation Memory & Recall ✅

**Goal**: User can refer back to previous topics and RAVANA recalls them meaningfully.

**Status: COMPLETE** — Implemented in commit `01c48c4`

### What was implemented

#### 3.1 Topic-indexed conversation store
- `_past_topics` list replaced with `_topic_list: List[str]` + `_topic_store: Dict[str, Dict]`
- Each topic entry stores: `turn`, `assertions`, `vad`, `response_summary`, `visit_count`
- Max 50 topics (up from 30)

#### 3.2 Strengthen episodic edges
- Episodic edges boost from w=0.15 → 0.25 on 2nd visit
- Migrate to semantic (w=0.40) after 3+ visits

#### 3.3 Explicit recall trigger
- `RECALL_TRIGGERS` list with 12 patterns ("remember when", "you said", "we talked about", etc.)
- `_detect_recall_trigger()` method detects triggers and activates recalled topic at 0.8

#### 3.4 Response-aware context
- `_response_context` stores last 10 response chains with hop metadata
- Follow-up handler biases edges from the original response via `user_model.edge_reactivations`
- `_last_chain_hops` snapshot saved before `_last_hops` cleared in `_generate_response`

### Verification

```bash
# Recall test
python scripts/ravana_chat.py --reset --chat "tell me about justice|what is freedom|what did we say about justice" --trace
# Third query should activate "justice" subgraph at 0.8 activation

# Topic accumulation test
python scripts/ravana_chat.py --reset --chat "a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z" --trace
# All 26 topics should be stored in _topic_store with metadata
```

## Phase 4: Anchor-Based Coherence ✅

**Goal**: Each response stays on-topic instead of drifting to unrelated concepts.

**Status: COMPLETE** — Implemented in commit `34f4772`

### What was implemented

#### 4.1 Force chain re-anchoring
- After each sentence, checks if final concept has edge back to subject or vector cos > 0.3
- If no path exists, trims sentence early (drops last connector + concept)

#### 4.2 Subject-proximity bonus at each hop
- Added `subject_proximity` parameter to `_walk_chain`
- At each hop, boosts candidates by up to 1.15× based on cosine similarity to the original subject (only boosts, never penalizes)
- Applied after VAD modulation, before RLM confidence

#### 4.3 Context window over multiple sentences
- Temperature escalation (0.15 → 0.30 → 0.40) promotes exploration in later sentences
- Shared `seen` set across all 3 sentences prevents concept reuse

#### 4.4 Minimize disconnected connectors
- `connector_counts` dict tracks every connector word used across the response
- `_starter_from_chain` accepts `connector_counts` and forces a non-semantic starter (but/because/like/then) when "connect" has been used 3+ times

### Verification

```bash
# Coherence test — subject should persist across sentences
python scripts/ravana_chat.py --reset --chat "what is justice" --trace
# All 3 sentences should reference justice directly or via high-similarity concepts
# Sentence 3's final hop should re-anchor to justice or a near-neighbor

# Drift distance metric
python -c "
labels = ['justice', 'truth', 'what', 'make', 'ability', 'learn', 'confused']
# After Phase 4, average cosine from 'justice' to final hop should be > 0.4
"
```

## Phase 5: Offline Independence & Distribution Readiness ✅

**Goal**: RAVANA automatically falls back to offline when the network is unavailable — no user flag needed.

**Status: COMPLETE** — Implemented in commit `3220bc6`

### What was implemented

#### 5.1 Automatic offline fallback (no flag)
- `_network_available: Optional[bool]` flag auto-detects network status on first API call
- If Intentforge/oxiverse fails, RAVANA silently falls back to GloVe-only expansion
- No error messages leak to the user — `learn_from_web` returns short strings like "offline" or "learned X things about Y"
- 3s timeout (down from 10s) for fast failure detection

#### 5.2 Periodic retry
- `_network_retry_turn` schedules a retry 20 turns after the first failure
- Retry timer is set once per outage (not reset on every re-queue)
- After 20 turns, `_network_available` resets to `None`, triggering one real API attempt

#### 5.3 Configurable data directory
- `--data-dir` CLI arg sets custom directory for weights and GloVe cache
- `data_dir` parameter passed to `CognitiveChatEngine` constructor
- Both `_save_path` and `_glove_cache_path` respect the data directory

#### 5.4 Deferred queue handles offline gracefully
- When network is down, items stay in the queue and are retried later
- Queue doesn't grow unbounded — rate-limited to 1 pop per 3 turns

### Verification

```bash
# Custom data dir (works offline too)
python scripts/ravana_chat.py --reset --data-dir ./ravana_data --chat "hello|what is life" --strategy

# Auto offline — just unplug network and chat normally
python scripts/ravana_chat.py --reset --chat "hello world" --trace
# No timeout delays, no error messages — just "offline" fallback in trace
```

## Phase 6: Persistent Long-Term Learning ✅

**Goal**: RAVANA remembers across sessions, learns from every user, and becomes smarter with use.

**Status: COMPLETE** — Implemented in commit `cbe49a5`

### What was implemented

#### 6.1 Checkpoint rotation
- Every 25 turns, saves `ravana_weights_{turn_count}.pkl` alongside the main save
- Keeps last 3 checkpoints, auto-cleans older ones via glob + os.remove

#### 6.2 Multi-user isolation
- `--user <name>` CLI flag creates user-specific save files: `ravana_weights_{user}.pkl`
- `user_suffix` parameter in `CognitiveChatEngine.__init__` customizes save and GloVe cache paths

#### 6.3 Export/import knowledge
- `--export-graph <file.json>`: exports all concepts + edges as JSON (inline serialization with try/except)
- `--import-graph <file.json>`: merges JSON into existing graph with node/edge reconstruction
- Both handle API mismatches gracefully

#### 6.4 Learning dashboard
- `--stats`: prints graph statistics (node count, edge count, turn count)
- `--concept <word>`: shows all edges, weights, and relation types for a concept

### Verification

See Phase 4-7 integration verification below.

## Phase 7: Curious Persistence ✅

**Goal**: When RAVANA can't answer a question directly, it systematically tries 7 strategies before expressing uncertainty.

**Status: COMPLETE** — Implemented in commit `cbe49a5`

### What was implemented

#### 7.1 Impossible Query Registry
- `FailedQuery` dataclass with query, subject, activated_concepts, strategies_tried, best_guess_response, turn, free_energy
- `_impossible_queries: List[FailedQuery]` persisted across sessions via state dict

#### 7.2 Multi-Strategy Framework (A-G)
- **A** — Direct chain walk (existing `_walk_chain`)
- **B** — `_bridge_prospecting`: walks from each activated concept, finds the best path
- **C** — `_analogical_detour`: walks from analogically-linked concepts
- **D** — `_contrastive_flip`: walks from contrastive opposite concepts
- **E** — `_sub_question_decompose`: splits multi-word queries, walks from sub-concepts
- **F** — `_research_mode`: searches web, auto-wires concepts, retries A-E
- **G** — `_compose_uncertainty_response`: constructs graph-label-only uncertainty response

#### 7.3 Strategy escalation in `_generate_response`
- After primary chain walk fails, iterates strategies B-G in order
- Each strategy returns `Optional[str]`; first non-None result wins
- `_last_strategy_used` recorded for trace display

#### 7.4 Curiosity Drive
- `_update_emotion` integrates impossible query detection via `_last_strategy_used`
- When strategy G reached, curiosity arousal boosted; decays slower than normal VAD

#### 7.5 Sleep-Replay
- `_sleep_consolidate` method processes `_impossible_queries` during sleep cycles
- Triggered from the existing sleep pressure cycle in `process_turn`

#### 7.6 State persistence
- `_impossible_queries` and `_last_strategy_used` saved/loaded across sessions
- Load has `.get()` fallbacks for backward compatibility with pre-Phase-6 saves

### Verification

See Phase 4-7 integration verification below.

## Phase 4-7 Integrated Verification

```bash
# Coherence + memory + offline + strategies
python scripts/ravana_chat.py --reset --chat "what is quantum consciousness" --trace --strategy
# Should show multi-strategy attempts, bridge prospecting, curiosity arousal

# Session persistence
python scripts/ravana_chat.py --reset --chat "i love astrophysics" --strategy && echo "---SESSION 2---" && python scripts/ravana_chat.py --chat "tell me about black holes" --strategy

# Export/import round-trip
python scripts/ravana_chat.py --reset --export-graph test_export.json --chat "hello" --strategy
python scripts/ravana_chat.py --reset --import-graph test_export.json --chat "hello" --strategy

# Stats
python scripts/ravana_chat.py --stats
```

## Summary Roadmap

| Phase | What | Key Metric | Effort |
|-------|------|------------|--------|
| 1 | Auto-expansion from every message | 180→5000+ concepts after 50 conversations | Medium |
| 2 | Sub-second response | <1s per turn | Small |
| 3 ✅ | Conversation memory & recall | Recall trigger fires correctly | Medium |
| 4 ✅ | Anchor-based coherence | Final hop cosine to subject > 0.4 | Medium |
| 5 ✅ | Offline independence | Zero network calls in --offline mode | Small |
| 6 ✅ | Long-term learning & distribution | Save/load across sessions, export/import | Medium |
| 7 ✅ | Curious Persistence | Impossible queries produce 3+ word responses 90% of the time | Medium |

## Phase 7: Curious Persistence — Never Give Up on Impossible Queries

**Goal**: When RAVANA can't answer a question directly, it doesn't shrug. It systematically tries multiple strategies — linking related concepts, decomposing the question, web-searching related topics, and expressing uncertainty with genuine curiosity. After a dead end, it remembers why and tries a different angle next time.

**Core Insight**: The most human-like thing RAVANA can do is *persist curiously*. Instead of a single chain walk → fallback, we add a recursive curiosity engine that treats "I don't know" as a learning signal, not a terminal state.

### Architecture: The Curious Persistence Engine

#### 7.1 Impossible Query Registry
- Track queries that produced low-confidence or failed responses
- Record: query text, which concepts were activated, which strategies failed, what the final response was
- Use as a learning queue: revisit failed queries during sleep cycles

```python
@dataclass
class FailedQuery:
    query: str
    subject: str
    activated_concepts: List[str]
    strategies_tried: List[str]  # which strategies failed
    best_guess_response: str
    turn: int
    free_energy_at_time: float
```

#### 7.2 Multi-Strategy Reasoning Framework
When the primary chain walk fails (returns `< 2 hops` or all confidence < 0.2), RAVANA escalates through strategies in order:

| Strategy | When | How |
|----------|------|-----|
| **A — Direct Chain** | Default | Current `_walk_chain` from subject |
| **B — Bridge Prospecting** | A fails | Walk from *each* activated concept, not just the primary subject. Reuse `_spread_and_collect` but with output modified to return the top bridge concept. Pick the path with highest total confidence. |
| **C — Analogical Detour** | A+B fail | Find concepts that are analogically linked to the subject (edges with relation_type="analogical"), walk from those instead. |
| **D — Contrastive Flip** | A-C fail | Walk from the *opposite* of the subject (follow contrastive edges), then invert the relation in the response. Uses existing `_contradiction_map` to find antonyms automatically. |
| **E — Sub-Question Decomposition** | A-D fail | Split multi-word queries into individual words that exist as graph concepts. E.g., "quantum gravity" → "quantum" + "gravity" (if both exist). For single-word queries, skip this strategy. No LLM needed — pure graph-label substring matching. |
| **F — Web Research Mode** | A-E fail | Search the web for related topics, not the exact query. Extract concepts from search results, auto-wire them into the graph (Phase 1 mechanism), then retry A-E. |
| **G — Honest Curiosity** | All fail | Express uncertainty using graph labels only. No hardcoded templates. Pattern: `[subject] but [related_concept] [related_concept2]. curious learn more.` where all words are existing graph labels. |

#### 7.3 Bridge Prospecting Detail
- When the subject has no good outgoing edges, look for a *bridge concept* that connects multiple activated concepts
- Use graph centrality: find a concept that minimizes the sum of distances to all activated concepts (Steiner node approximation)
- Walk from the bridge concept, then start the response by acknowledging the bridge: "I don't know much about X, but here's something related: Y connects to Z..."

```python
def _find_bridge_concept(self, activated_ids: List[int]) -> Optional[str]:
    """Find a concept that bridges multiple activated inputs.
    
    Reuses _spread_and_collect internally: spreads activation from
    each seed, then finds the concept with highest total activation
    that is NOT one of the seeds — this is the bridge.
    """
    if not activated_ids or len(activated_ids) < 2:
        return None
    # Reuse existing spread-and-collect from each seed
    assocs = self._spread_and_collect(
        activated_ids, primary_ids=set(activated_ids))
    if not assocs:
        return None
    # First association (non-seed, highest activation) = bridge
    return assocs[0][0]
```

#### 7.4 Web Research Mode
- When the graph has no path at all (all strategies A-E fail), switch to web research mode
- NOT the same as current `learn_from_web()` — this mode is about *finding related information to answer a question*, not just adding unknown words
- Steps:
  1. Search the web for the original query (same as current)
  2. Extract the top 5-10 concepts from search results
  3. Auto-wire them into the graph (Phase 1 mechanism)
  4. Re-run strategies A-E with the expanded graph
  5. If still stuck, compose a response using the web snippet text itself (with caveats)

```python
def _research_mode(self, query: str) -> Optional[str]:
    """Research a query on the web to find related concepts.
    
    1. Search
    2. Extract concepts
    3. Auto-wire into graph
    4. Retry chain walking
    
    Returns a response if successful, None if even web fails.
    """
    # Search web (reuse existing learn_from_web infrastructure)
    search_result = self._search_web(query)
    if not search_result:
        return None
    
    # Extract concepts, auto-wire them
    new_count = self._learn_from_text(search_result, query)
    
    # Retry strategies A-E (all graph-native, no web text leakage)
    for strategy in [self._direct_chain, self._bridge_prospecting, 
                     self._analogical_detour, self._contrastive_flip,
                     self._sub_question_decompose]:
        result = strategy(query)
        if result and len(result.split()) >= 3:
            return result  # All words are graph labels, safe to return
    
    # Final fallback: extract graph labels from search text, use only those
    # This ensures output is still graph-native even when web-informed
    concepts = self._extract_graph_labels_from_text(search_result)
    if concepts and len(concepts) >= 2:
        return f"{query} connect {' '.join(concepts[:3])}."
    return None
```

#### 7.5 Curiosity Drive (Emotion Modulation + MetaCognition Integration)
- When RAVANA encounters an impossible query, its curiosity arousal should spike
- This creates a lasting "unfinished business" signal that persists across turns
- **MetaCognition integration**: When an impossible query is detected, the epistemic mode
  naturally shifts to `EXPLORATORY` (moderate uncertainty, systematic probes) via the
  existing `MetaCognition.recommend_epistemic_mode()` pathway. The `ConfidenceCalibrator`
  records the low-confidence outcome, improving calibration over time.
- Modify `_update_emotion` to detect impossible queries:
  - If strategy G (Honest Curiosity) was reached → boost `curiosity` arousal to 0.8+
  - If web research was done and found something → boost `excited` valence
  - The curiosity arousal decays slower than normal VAD (0.98 instead of 0.95)

```python
def _detect_impossible_query(self, activated: List[int], subject: str) -> bool:
    """Check if this query is likely impossible to answer."""
    if not subject:
        return True
    # Try a quick chain walk — if it fails, mark as impossible
    chain = self._walk_chain(subject, set(), max_hops=1, temperature=0.1)
    return chain is None or len(chain.split()) < 2
```
- During sleep, the `MetaCognition` system can review the Impossible Query Registry
  and adjust confidence calibration: repeated impossible queries → more conservative
  confidence estimates → more EXPLORATORY mode → more diverse strategy attempts.

#### 7.6 Uncertainty-Honest Expression Layer
- When strategies A-F all fail, RAVANA must communicate *what it knows* rather than silently falling back to a one-word answer
- Template-free! Uses the failed attempt data to construct something honest:
  - "I don't know much about X. I know Y and Z. Y connects to W. Is that related?"
  - "Hmm, that's a tricky one. I found P and Q when I searched for it. Let me think about how they connect..."
  - "That's outside what I know so far. But I'm curious! Can you tell me more about X?"

These are NOT hardcoded templates. They use a generator that takes the strategy results and constructs a response from existing graph labels only:

```
Components:
- [subject] = the original query topic (graph label)
- [related_concepts] = top 2-3 activated concepts from the failed attempt (graph labels)
- [bridge] = the bridge concept if found (graph label)
- [searched_concepts] = concepts found via web research (auto-wired graph labels)

Generated pattern (all words are graph labels):
"[subject] connect [related_concepts[0]] [related_concepts[1]]. but [subject] contrast [bridge]. curious learn more."

Note: "curious", "learn", "more", "connect", "but" are all existing graph labels
from TEEN_CONCEPTS. "contrast" is used instead of "not connect" because
"not" is a STOP_WORD, not a graph concept. The expression layer ALWAYS stays
within graph-native vocabulary.
```

#### 7.7 Sleep-Replay of Impossible Queries
- During sleep consolidation, revisit the Impossible Query Registry
- For each failed query, try to auto-wire concepts: use GloVe similarity between the query subject and ALL existing concepts
- If new edges form above threshold, mark the query as "partially resolved"
- Track resolution rate as a meta-cognitive metric

### Steps

#### 7.1 Implement Impossible Query Registry
- Add `@dataclass FailedQuery` with all fields
- Add `_impossible_queries: List[FailedQuery]` to `CognitiveChatEngine.__init__`
- After each response, check if confidence < 0.2 and record the query
- Persist across sessions alongside the graph

#### 7.2 Implement Multi-Strategy Framework
- Refactor `_generate_response` to try strategies A-G in order
- Each strategy returns `Optional[str]` and records success/failure in the registry
- Strategy B (Bridge Prospecting): `_find_bridge_concept()` as shown above
- Strategy C (Analogical Detour): walk from analogically linked concepts
- Strategy D (Contrastive Flip): walk from contrastive opposite, invert
- Strategy E (Sub-Question Decomposition): split query by "and", "or", spaces

#### 7.3 Implement Web Research Mode
- `_research_mode(query)` as shown above
- Auto-wire found concepts, retry strategies
- Compose from web text if all else fails

#### 7.4 Implement Curiosity Drive
- Modify `_update_emotion` with impossible query detection
- Add slow-decay curiosity arousal
- Add `_curiosity_persistence` field that tracks "unfinished business" across turns

#### 7.5 Implement Uncertainty-Honest Expression
- Add `_compose_uncertainty_response(failed_query)` method
- Uses graph labels only, no hardcoded templates
- Pattern: "[subject] connect [rel1_concept] [rel2_concept]. but [subject] not connect [bridge]. curious learn more."

#### 7.6 Implement Sleep-Replay
- During `_sleep_consolidate`, process `_impossible_queries`
- Auto-wire failed query subjects to highly similar concepts
- Mark queries as resolved if new edges created confidence > 0.3

### Data Flow

```
User: "what is quantum consciousness?"
  ↓
Activate: [quantum, consciousness] → both unknown or weakly connected
  ↓
Strategy A (Direct Chain): fails — no path
  ↓
Strategy B (Bridge Prospecting): scans all activated — finds "science" weakly activated
  ↓
Strategy C (Analogical Detour): walks from "consciousness" analogical edges → "mind like garden", "knowledge like light"
  ↓
Response: "consciousness like garden. science connect knowledge light."
  ↓ (emotion update)
Curiosity arousal boosted: "not yet connect consciousness quantum. curious."
  ↓ (sleep)
Replay: quantum auto-wired to physics, mechanics, science
```

### Verification

```bash
# Impossible query test
python scripts/ravana_chat.py --reset --chat "what is quantum consciousness" --trace
# Should show multiple strategies tried, bridge concept found
# Response should be 3+ words from graph labels, not "..."

# Persistence test — same query twice
python scripts/ravana_chat.py --reset --chat "what is quantum consciousness|tell me more about quantum consciousness" --trace
# First: shows "curious" arousal boost
# Second: should find new edges from sleep replay? Or show improved confidence

# Multi-strategy escalation test
python scripts/ravana_chat.py --reset --chat "explain the meaning of life" --strategy
# Should attempt all strategies and produce a multi-sentence response
# Should not give up with "..."

# Web research fallback
python scripts/ravana_chat.py --reset --chat "what is a quasar" --strategy
# Should trigger web research, extract concepts, retry
# Response should reference "space" or "star" or other related concepts
```

## Phases Summary

| Phase | What | Key Metric | Effort |
|-------|------|------------|--------|
| 1 | Auto-expansion from every message | 180→5000+ concepts after 50 conversations | Medium |
| 2 | Sub-second response | <1s per turn | Small |
| 3 ✅ | Conversation memory & recall | Recall trigger fires correctly | Medium |
| 4 ✅ | Anchor-based coherence | Final hop cosine to subject > 0.4 | Medium |
| 5 ✅ | Offline independence | Zero network calls in --offline mode | Small |
| 6 ✅ | Long-term learning & distribution | Save/load across sessions, export/import | Medium |
| 7 ✅ | Curious Persistence | Impossible queries produce 3+ word responses 90% of the time | Medium |

## Guiding Principles

- **No template strings, no hardcoded responses** — preserve the graph-only architecture
- **Every contribution must be verifiable** — each phase has a bash command to prove it works
- **Keep the 3-subsystem architecture** (VAD, RLM, BeliefStore) but make them lightweight
- **Prefer numpy over loops** — the chain walking hot path should be vectorized where possible
- **Offline-first** — the app should be fully functional without internet
- **Curiosity over silence** — when uncertain, express it honestly and try harder, don't shrug
