# total-agent-memory architecture research

Research snapshot: `vbcherepanov/total-agent-memory` at commit `95ea2b881f507cc303eda32fc1d9849c8cf228aa` (2026-09-25). The external repository was cloned read-only under `scratch/` for source inspection; no external code is vendored.

## Executive summary

total-agent-memory (TAM) is a local-first, server-backed memory service for coding agents. Its useful architectural lesson for RAVANA is not an LLM or a second graph. It is the separation of independent retrieval signals into ranked lists and their fusion by Reciprocal Rank Fusion (RRF). TAM combines a lexical BM25 tier, a dense local-embedding tier, and optional graph/reranking tiers. RAVANA already has an episodic transcript plus GloVe semantic scoring, but currently selects a single semantic winner after an earlier stem-cue shortcut. A dependency-free BM25 ranking fused with RAVANA's existing semantic ranking would improve retrieval when exact rare terms and semantic similarity disagree, without retraining, an LLM, or answer templates.

## MCP server layer

Verified mechanisms:

- `total_agent_memory/server.py:7-16` loads the implementation from `src/server.py` and re-exports `main` and `run` for the package entry point.
- `total_agent_memory/server.py:27-29` adapts the asynchronous `main` to the synchronous `main_sync` console-script entry point.
- `total_agent_memory/remote.py` is the standard-library remote MCP bridge documented by TAM; the repository README at `README.md:250-288` describes remote `memory_recopes`, `memory_save`, `memory_recall`, `memory_get`, `memory_update`, `memory_delete`, `memory_history`, and `memory_export` tools.
- `src/server.py:4108-4148` publishes the retrieval tool schema and explicitly documents the default `BM25 + semantic + RRF` mode.

What this means: MCP is TAM's integration boundary, not its cognition. RAVANA already has an internal cognitive engine and should not add an MCP layer merely to copy TAM.

## BM25 + FastEmbed recall

Verified mechanisms:

- `src/server.py:2935-2997` executes the first retrieval tier with SQLite FTS5 `bm25(...)`, normalizes scores relative to the batch maximum, and records the resulting document order in `tier_rankings["fts"]`.
- `src/embed_provider.py:142-217` lazily initializes a local FastEmbed `TextEmbedding` model, materializes generated vectors, and records vector dimensionality.
- `src/server.py:3020-3073` obtains dense candidates from `_search_spaces`, records them in `tier_rankings["semantic"]`, and orders the semantic tier by similarity.
- `src/server.py:2741-2781` implements weighted Reciprocal Rank Fusion: each source contributes `weight / (k + rank + 1)`.
- `src/server.py:3395-3428` applies RRF to the independent tiers, applies per-result decay/importance, and sorts the fused result.
- `src/config.py:734` labels the default profile as FastEmbed + FTS5 + vector + RRF with no hot-path LLM.

The key idea is rank fusion, not a fixed score threshold. Independent evidence sources vote on ordering; their raw score scales need not be calibrated against each other.

## Triple extraction and graph linkage

Verified mechanisms:

- `src/triple_extraction_queue.py:1-6` describes the durable background queue: saves enqueue quickly, while a worker performs deep extraction and persists subject-predicate-object graph edges.
- `src/triple_extraction_queue.py:43-67` enqueues idempotently by `knowledge_id`.
- `src/triple_extraction_queue.py:71-109` reclaims stale work, atomically claims the oldest pending item, and marks it processing.
- `src/ingestion/extractor.py:233-251` selects fast or deep extraction.
- `src/ingestion/extractor.py:253-292` ensures extracted concepts/entities exist as graph nodes and links them to the knowledge record.
- `src/ingestion/extractor.py:294-334` creates graph edges or reinforces an existing edge of the same relation.
- `src/temporal_kg.py:53-75` defines a bi-temporal `(subject, predicate, object)` assertion with validity intervals and supersession.
- `src/temporal_kg.py:87-135` deduplicates identical assertions and closes older same-predicate assertions when a newer object arrives.

TAM's deep extraction invokes Ollama, so RAVANA cannot adopt that exact implementation under its no-LLM constraint. RAVANA already has a dependency-free typed OpenIE extractor at `ravana/src/ravana/web/openie.py:139-175`, producing `(subject, relation, object)` facts, and already reinforces typed concept-graph edges. Re-adding queueing or a temporal graph would duplicate existing architecture.

## RAVANA comparison

### Similarities

- Both are local-first and privacy-preserving by default; TAM uses SQLite/local FastEmbed, while RAVANA uses local typed stores and GloVe projections.
- Both retain raw conversational records and derive structured graph/fact state.
- Both support incremental graph-edge reinforcement rather than rebuilding the entire representation on every write.

### Differences

| Concern | total-agent-memory | RAVANA |
|---|---|---|
| Primary role | persistent memory service for external coding agents | decoder-first cognitive architecture with identity, stance, affect, memory, web learning, and action |
| Retrieval | SQLite FTS5/BM25 + dense vectors + optional graph/cross-encoder/MMR, fused with RRF | entity/slot precision paths, stem-cue matching, and GloVe semantic scoring; no independent lexical ranking fused with semantic ranking |
| Dense embeddings | local FastEmbed model, stored vectors | deterministic GloVe vectors projected to the engine dimension |
| Triple extraction | optional Ollama-backed background extraction | dependency-free typed OpenIE (`web/openie.py:139-175`) |
| Knowledge lifecycle | bi-temporal assertions with explicit supersession | active/superseded fact metadata plus typed graph reinforcement |
| Integration | MCP tools and HTTP/server modes | in-process engine plus state-driven agent tools |
| Learning rule | external agent writes/searches memory; optional enrichment jobs | conversation/web inputs update live identity, stance, fact, belief, concept-graph, and episodic stores |

## Adopt, adapt, reject

### Adopt: one exact idea

Adopt TAM's **independent lexical ranking + rank fusion** for episodic retrieval. RAVANA currently has two pieces but not their fusion:

1. The stem-cue path in `ravana/src/ravana/chat/engine_memory.py:904-984` builds `_CUE_STOP` (`engine_memory.py:904`), filters query cues (`engine_memory.py:928-929`), and scores them by presence/fraction of query cue stems.
2. The semantic path in `ravana/src/ravana/chat/engine_memory.py:985-1084` sums GloVe cosine evidence (`engine_memory.py:1026-1032`) and accepts a winner through adaptive gating plus a word-boundary verbatim check (`engine_memory.py:1052-1081`).

A small dependency-free BM25 tier over the same in-memory/durable episodic records can produce an independent lexical ordering. Reciprocal Rank Fusion can combine that ordering with the existing semantic ordering. This is online and stateless per query: every new episode is immediately in the next lexical ranking, and no rebuild or retraining is required.

Adopted as `MemoryMixin._bm25_rank` at `ravana/src/ravana/chat/engine_memory.py:553-584`, called from the lexical tier at `engine_memory.py:968`.

### Adapt

- Use RAVANA's existing transcript/index records, not a new database.
- Use the existing query scaffolding and generic-cue filters so recall intent does not become lexical evidence.
- Keep entity/slot exact recall ahead of hybrid fuzzy recall.
- Preserve fail-closed behavior: return a fused candidate only when at least one real lexical or strong semantic cue exists.

### Reject

- Reject FastEmbed as a new model dependency: RAVANA already has deterministic GloVe vectors, and adding a model would violate the minimal incremental design.
- Reject MCP: it is an integration surface, not the adopted cognitive capability.
- Reject Ollama triple extraction: RAVANA has an existing no-LLM typed extractor.
- Reject a parallel temporal graph: RAVANA already has typed graph and fact lifecycle structures; this would be architecture duplication.
- Reject author-written answer behavior. The change may alter which stored episode is selected, but it must not add a question-to-answer mapping or authored reply.

## Constraint justification

RAVANA's four non-negotiable constraints, and how each recommendation above is checked against them.

| Constraint | Adopted BM25 tier | Rejected items |
|---|---|---|
| No LLM | Okapi BM25 is a closed-form term-weighting formula; `_bm25_rank` (`engine_memory.py:553-584`) uses only `re`, `math.log`, and `collections.Counter` over the stored episode text. No model inference on the query path. | TAM's Ollama triple extraction (`src/ingestion/extractor.py:233-251`) requires a local LLM process at query-independent time — a hard fail. |
| No retraining | Term frequency, document length, and IDF are computed from the current store on every query. Nothing is fitted, stored as a model parameter, or re-embedded. There is no checkpoint to rebuild. | TAM's FastEmbed dense tier (`src/embed_provider.py:142-217`) requires materializing a model and generating vectors; adopting it would add both a model artifact and a re-embedding step. |
| Online / incremental | A newly recorded episode participates in the next query's IDF and document-length statistics immediately. Learning is a single forward pass over live records. | A parallel bi-temporal graph (`src/temporal_kg.py:53-135`) and a durable extraction queue (`src/triple_extraction_queue.py:1-6`) are both large new subsystems whose value depends on extraction RAVANA already performs. |
| No hardcoding | The tier adds no reply string. It only reorders the candidates already returned by the existing lexical/semantic/entity paths, which then render stored state via `_reconstruct_gist` (`engine_memory.py:1507`). | Any question-to-answer table, keyword-to-reply branch, or authored fallback was excluded by design; there is nothing in the adopted tier to seed one. |

Privacy is the fifth constraint and is unaffected: BM25 statistics are derived in-process from RAVANA's own episodic records and are never transmitted, logged, or persisted to a new store.

## Citation audit

Every `path:line` in this document was re-verified by reading the cited lines on 2026-09-25:

- External citations were checked against a read-only clone of `vbcherepanov/total-agent-memory` pinned at `95ea2b881f507cc303eda32fc1d9849c8cf228aa`, matching the snapshot commit named above. No external code is vendored into this repository.
- RAVANA citations were checked against this working tree after the adopted change; the two retrieval ranges were corrected to `904-984` and `985-1084` (the previously written `895-987`/`995-1094` pointed at comment blocks outside the two paths).
