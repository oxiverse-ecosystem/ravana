# RAVANA vs total-agent-memory — what to adopt, what to reject

Synthesis for backlog item #5 of `set-and-forget-loop/ravana.txt`.
Evidence base: `docs/EXTERNAL_TOTAL_AGENT_MEMORY.md` (TAM read at `de53c19`).
Every RAVANA claim below is cited to `path:line` in this repo, and the central
claim is backed by a measured run, not an argument.

---

## The framing difference that decides everything

TAM is memory infrastructure for *someone else's* language model. Its retrieval
layer exists to serve a decoder that already knows how to speak; the graph is a
side store that improves what gets put in that decoder's context.

RAVANA has no such decoder to serve. The retrieval path *is* the cognition: what
`_self_cued_episodic` returns is what RAVANA believes the user said. So a
retrieval bug in TAM is a relevance-quality issue; the same bug in RAVANA is a
false-memory issue, and RAVANA's stated bar is to fail closed rather than
confabulate.

That asymmetry sets the filter for everything below: **adopt mechanisms that
change WHICH record RAVANA retrieves, because that is a belief-formation
decision. Reject mechanisms that change how RAVANA stores or transports data,
because RAVANA already has a persistence layer and a transport layer.**

---

## ADOPT — Reciprocal Rank Fusion of independent retrieval channels

### The measured defect

`_self_cued_episodic` selects the record to recall with:

```python
_cand.append((len(_matched), _i))     # engine_memory.py:791
...
_cand.sort(reverse=True)               # engine_memory.py:794
return self._reconstruct_gist(eligible[_cand[0][1]])   # engine_memory.py:795
```

Candidates are ordered by **cue-coverage count**, tie-broken by **store index**
(recency). That is the entire ranking mechanism. Nothing else participates.

RAVANA nonetheless already owns two more evidence channels over the same
records, and uses neither here:

- a lexical BM25 tier, `_bm25_rank` (`engine_memory.py:624`), used by the
  sibling path at `engine_memory.py:1206`;
- a GloVe cosine tier, computed at `engine_memory.py:1253-1276` in the sibling
  path and used there for a binary accept/reject gate.

So the capability is not missing. It is **not composed**. When two records
cover the same cues — which is precisely the ambiguous case where a tie-break is
needed — the decision falls to recency, which is not evidence about relevance.

### Measured, not argued

Probe: `scratch/_path_probe.py`, run on the round branch
`auto/round-2026-09-29T0823Z` (base `4287916a`) with `RAVANA_OFFLINE=1`.

Query `what did i tell you about the harbour radio drifting past the buoy`
against an 8-turn store containing one relevant disclosure (index 1, about an
uncle who ran a harbour pilot radio station) and one off-topic record (index 7,
about buying a second radio for a workshop) that happens to carry every query
cue.

```
cues: ['harbour', 'radio', 'drifting', 'past', 'buoy'] need: 3
eligible: 8 max_df: 4
  idx=1 cov=5 'kip ran the harbour pilot radio station and he kept hearin'
  idx=7 cov=5 'i bought a second harbour radio for the workshop and the b'

SHIPPED _self_cued_episodic -> "you mentioned: \"i bought a second harbour
                               radio for the workshop and the buoy horn
                               rattles past the drifting tide line\""
```

Coverage ties at 5. `_cand.sort(reverse=True)` then prefers the larger index,
so the recency tie-break returns the off-topic record. The channels RAVANA
already computes both rank the correct record first on that same store:

```
BM25 channel : [('CORRECT(kip/harbour)', 5.724), ('BAIT(newer, off-topic)', 3.869)]
COS channel  : [('CORRECT(kip/harbour)', 16.71), ('BAIT(newer, off-topic)', 13.84)]
```

The evidence was present and discarded. That is the defect: not a missing
capability, a missing **composition**.

An honest note on probe design: two earlier probe shapes failed to reproduce
this (`scratch/_fusion_probe3.py`, `scratch/_fusion_probe4.py`). In both, the
correct record also won the cosine channel, so count-plus-recency landed on it
by luck. A probe that a fix happens to pass is not a probe. Only the exact-tie
shape exposes the missing composition.

### Why RRF is the right fix, and not a weight

TAM's own justification (`retriever.py:12-15`) is that RRF "ignores raw
magnitude differences between BM25 and cosine... BM25 produces unbounded
negative log scores while cosine sits in [-1, 1]". That is exactly RAVANA's
situation: BM25 scores here are unbounded positive sums of `idf * tf`, and the
cosine tier is an unbounded sum of dot products. Any attempt to combine them by
weighting would require inventing a normalisation constant tuned against this
store — a magic number that would not survive a different conversation.

RRF combines by **rank position** only:

```python
scores[eid] += 1.0 / (k + rank + 1)     # k = 60, TAM retriever.py:36
```

No threshold, no scale, no per-channel weight. A record that is mid-ranked in
both channels beats one that is first in a single channel and absent from the
other — the correct behaviour, because the first record has two independent
pieces of evidence and the second has one.

### Why this satisfies RAVANA's constraints

| Constraint | How RRF satisfies it |
| --- | --- |
| No LLM | Pure arithmetic over two orderings. No generation anywhere. |
| No retraining | Reads the live store per query. Nothing is fit; nothing is persisted. |
| Online / incremental | A record disclosed this turn is rankable the next turn because the channels already read the live store. |
| No hardcoding | No topic table, no keyword list, no reply string. The only constant is `k = 60`, TAM's published value, not tuned on this store. |
| Fails closed | The existing evidence bar (majority coverage + rarity `df <= n//2`) still gates admission. RRF only re-ORDERS records that already passed. A query with no qualifying record still returns `None`. |

The last row matters most: RRF cannot make RAVANA more confabulating. It never
adds a candidate. It changes which of the already-qualified candidates wins, and
it changes that using evidence instead of a counter.

---

## REJECT — and why, in each case

**The SQLite / FTS5 storage substrate** (`fts_schema.py:18`, `KNOWLEDGE_FTS_DDL`).
RAVANA persists through pickle checkpoints and an episodic indexer. Adopting
external-content FTS5 means a storage migration with no capability gain — the
BM25 ranking is already implemented in-process at `engine_memory.py:624`.

**The project-token index trick** (`fts_schema.py:1-11`, `project_token:42`).
Encoding a scope as `"p" + hex()` so FTS5 intersects doclists is elegant and
specific to having a multi-tenant index. RAVANA is single-user; its equivalent
scoping is the self/other boundary in the fact store, which is already enforced
at read time.

**The async triple-extraction queue** (`triple_extraction_queue.py:28`). The
queue exists to keep a slow LLM call off a hot save path — the docstring is
explicit that `extract_and_link(deep=True)` calls Ollama. RAVANA has no LLM, so
there is no slow call to defer. The *idea* worth keeping is separating
enrichment from the hot path, but RAVANA's enrichment is already incremental and
online, so there is nothing to rebuild.

**Deep extraction via LLM** (`ingestion/extractor.py:1-9`, "create new
concepts", ~2-5 s). Directly forbidden by RAVANA's architecture. Named here so
the rejection is on record.

**Dempster-Shafer fusion** (`fusion.py:1-19`). Genuinely interesting, and
rejected on evidence, not taste: DS requires every hypothesis to be
pairwise-disjoint and assigned a prior mass. RAVANA's records are natural
language, not hypothesis strings — the same disclosure appears in many surface
forms, so the "frame of discernment" has no stable membership. Forcing it would
mean building a normaliser that decides when two strings are the same claim,
which is a harder problem than the one DS would solve.

**The pluggable embedding provider** (`embed_provider.py:1-15`). RAVANA is
deliberately GloVe-on-CPU. Its own docstring calls itself "Scaffolding only" —
adopting scaffolding is not an improvement.

---

## Deferred — genuinely valuable, not this round

**Tool-behaviour annotation** (`server.py:4189-4234`). TAM declares each tool's
safety in a named set and mechanically projects it onto the tool object, rather
than inferring it from the body. RAVANA's agentic layer (`ravana/agent/`) has
four tools and its own hard guards. Adopting the *declaration* discipline —
safety declared once in a named set, not re-derived per call site — is a real
improvement, and it generalizes beyond recall. It is not a retrieval-mechanism
adoption, so it does not belong in this round's implement-one step; it is filed
as the next candidate.
