# total-agent-memory — architecture notes for RAVANA

Research record for backlog item #4 of `set-and-forget-loop/ravana.txt`.

**Source:** https://github.com/vbcherepanov/total-agent-memory
**Pinned at:** `de53c19b6ae650459c96f0d297c5d33fd77e77b4` (shallow clone, 2026-09-29)
**Method:** read the actual source in a local clone, not the README. Every claim
below carries `path:line` from that clone. Where TAM's own README makes a
marketing claim that the source does not support, that is called out explicitly.

TAM is MIT licensed (`LICENSE`, README "License" section).

---

## 1. The MCP server layer

A single `asyncio` process exposes the whole store as MCP tools
(`src/server.py`).

- **Tool catalogue.** `_tool_catalogue()` at `src/server.py:4237` builds a list
  of `Tool(...)` objects. The catalogue is a flat, explicit literal — 72 tools
  carry `name="..."` at that indentation on this SHA
  (`grep -c '^            name="' src/server.py` → `72`). The README's
  "74 MCP tools" figure is two ahead of this commit. First entries:
  `memory_recall` (`:4242`), `memory_timeline` (`:4304`),
  `memory_index_passages` (`:4321`), `memory_answer` (`:4329`), `memory_save`
  (`:4346`), `memory_update` (`:4398`).
- **Behaviour annotations.** `_annotate(tool)` at `src/server.py:4220` attaches
  `ToolAnnotations` to every tool using three explicit sets declared just above
  it: `_READ_ONLY_TOOLS` (`:4189`), `_DESTRUCTIVE_TOOLS` (`:4206`),
  `_IDEMPOTENT_TOOLS` (`:4213`). Every read-only tool is marked
  `readOnlyHint=True`, `destructiveHint=False`,
  `idempotentHint=True`; `openWorldHint=False` for all of them with the stated
  rationale that "Memory is the canonical closed world" (`:4216-4217`).
  `list_tools()` at `:4234` returns `[_annotate(t) for t in await _tool_catalogue()]`.
- **Transports.** `main()` at `src/server.py:7786` dispatches on `MCP_TRANSPORT`:
  `stdio`, `http`, or `streamable-http`. Multi-worker HTTP is a separate
  pre-fork path, `_serve_http_workers` (`:7743`), which stops the remaining
  children when any one exits.
- **Process entry.** `run()` at `src/server.py:7800` either becomes a socket
  worker, a worker group, or the single server.

The design point relevant to RAVANA is the **annotation discipline**: the
server does not infer safety from the tool body, it declares it once in a named
set and mechanically projects that onto every tool.

## 2. BM25 + embedding recall, fused by Reciprocal Rank Fusion

`src/memory_core/episodes/retriever.py` is the clearest statement of the
hybrid-recall design, and the part of TAM most worth studying.

- **Two independent channels.** `retrieve_episodes()` at `:44` runs
  `_bm25_search` (`:119`) and `_cosine_search` (`:251`) over the same candidate
  set, each returning an independently ranked `list[tuple[id, score]]`.
- **Fusion.** `_rrf_fuse(channels, k=...)` at `:334-343`:

  ```python
  scores: dict[int, float] = {}
  for _name, ranked in channels.items():
      for rank, (eid, _raw) in enumerate(ranked):
          scores[eid] = scores.get(eid, 0.0) + 1.0 / (k + rank + 1)
  return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
  ```

  `RRF_K = 60` at `:36`; `CANDIDATE_MULTIPLIER = 5` at `:38` (each channel
  returns `max(k * 5, 25)` candidates before fusion, `:73`).
- **Why rank-based, and why that matters.** The module docstring at `:12-15`
  gives the reason directly: "RRF is rank-based and ignores raw magnitude
  differences between BM25 and cosine, which is the right call here: BM25
  produces unbounded negative log scores while cosine sits in [-1, 1]." Two
  channels whose scores are not commensurable are combined by their *ordering*,
  not by summing their magnitudes.
- **Sign normalisation.** FTS5's `bm25()` is lower-is-better, so `_bm25_fts`
  (`:130`) negates it and `_bm25_pg` (`:164`) does the same for Postgres, so
  both channels hand RRF the same "first row is best" convention.
- **Degradation, not failure.** If FTS5 is unavailable, `_bm25_search` (`:119`)
  falls back to `_bm25_like` (`:186`), a distinct-token-count substring score,
  with the docstring stating it exists to "keep smoke tests sensible". A
  malformed MATCH expression after sanitising also falls back.
- **The lexical index itself.** `src/memory_core/fts_schema.py` declares
  `knowledge_fts` as an **external-content** FTS5 table over `knowledge`
  (DDL at `:18`), with `SCOPED_BM25_WEIGHTS = "1.0, 1.0, 1.0, 0.0"` at `:39`. The
  project-scoping trick is the interesting part (module docstring `:1-11`,
  `project_token` `:42`, `scoped_match` `:47`): project names are encoded as
  `"p" + hex(utf8)` in a generated column, so a
  project-scoped search ANDs one token into `MATCH` and lets FTS5 intersect
  doclists, with that column weighted 0 so it cannot affect the ranking. Scope
  is enforced by the index, not by post-filtering.

## 3. Triple extraction

Triples are `(subject, predicate, object)` rows written to `graph_edges`.

- **Extractor.** `src/ingestion/extractor.py` — `ConceptExtractor`. Its module
  docstring (`:1-9`) describes two modes: a fast local pass that matches text
  against *existing* graph nodes (<10 ms) and a deep pass that calls an LLM to
  create new concepts (~2-5 s). LLM traffic is routed through
  `llm_provider`, defaulting to Ollama at `http://localhost:11434` (`:24`).
- **Async shim.** `src/ai_layer/relation_extractor.py:30` exposes
  `extract_triples(text) -> list[tuple[str, str, str]]` as a pure
  text-in/triples-out function, with the docstring noting the DB-bound write
  path "stays where it is".
- **Queue.** `src/triple_extraction_queue.py` is the durability mechanism:
  `TripleExtractionQueue` at `:28`. Its docstring (`:1-7`) states the split —
  `memory_save` enqueues a `knowledge_id` in under 1 ms, and a background
  worker runs the slow deep extraction. `enqueue()` at `:43` is idempotent for
  *pending* rows but deliberately re-enqueues on content update (`:52-57`).
  `max_attempts=3` default (`:34`, clamped at `:37`).

## 4. Other mechanisms observed (not adopted — see the synthesis doc)

- `src/fusion.py:1-19` — Dempster-Shafer mass-function combination over
  hypothesis strings, with Θ as the universal set and conflict mass normalised
  out.
- `src/embed_provider.py:1-15` — a `Protocol`-based pluggable embedding
  abstraction (FastEmbed local, OpenAI-compatible, Cohere, DashScope). Note the
  docstring is honest that it is "Scaffolding only" and the FastEmbed flow still
  lives inside `server.py`.

---

## What TAM is and is not

TAM is an **MCP server that stores an external agent's memory in SQLite**. Its
"brain" is the coding agent it serves (Claude Code, Codex, Cursor) — the
memory substrate is retrieval infrastructure for someone else's language model.
RAVANA is the reverse: the retrieval machinery *is* the mind, and the decoder
is trained on the concept graph those mechanisms produce. That difference is
the reason the synthesis doc adopts the fusion algorithm and not the storage
layer.
