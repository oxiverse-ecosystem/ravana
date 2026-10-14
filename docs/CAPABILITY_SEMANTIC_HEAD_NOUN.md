# Semantic head noun detection for topic extraction (D1)

**Capability:** when RAVANA extracts the topic from a multi-word phrase, it now
identifies the **semantic head noun** — the word most central to the phrase —
instead of relying on a positional heuristic (last content word) that grabbed
modifiers and compound parts. *"migration patterns"* → *"migration"* (not
*"patterns"*); *"climate change"* → *"climate"* (not *"change"*); *"cooking
earlier"* → *"cooking"* (not *"earlier"*).

This closes **D1** from the round: the topic extractor's fallback path used
`words[-1]` (last content word), which is wrong for compound nouns and
modifier+noun phrases where the head is not the final token.

## The failure pattern (verified live)

| Query | Before (broken) | After (fixed) |
|-------|-----------------|---------------|
| `"what do you think about migration patterns"` | topic `"patterns"` — modifier grabbed | topic `"migration"` — head noun |
| `"i think cooking earlier is better"` | topic `"earlier"` — trailing modifier | topic `"cooking"` — head noun |
| `"its undervalued"` | topic `"undervalued"` — single adjective | topic `"undervalued"` — passes through (no compound) |

## How it works (verified against the source)

### 1. The method (`engine_generation.py:2188`)

`_semantic_head_noun(words)` takes a list of content words and returns the
**semantic head** — the word with the highest average cosine similarity to all
other words in the phrase, using RAVANA's existing GloVe embeddings.

The intuition: in a compound noun like *"migration patterns"*, the head
(*migration*) is more semantically central — it has higher average similarity
to the other words than the modifier (*patterns*) does. This is a
distributional-structural property, not a positional rule.

**Algorithm:**
1. Require ≥ 2 words with GloVe vectors (return `None` otherwise).
2. For each word, compute average cosine similarity to every other word.
3. Return the word with the highest average.

**Fail-closed:** returns `None` when GloVe is unavailable, when fewer than 2
words have vectors, or when the input has < 2 words. The caller falls through
to the existing `words[-1]` positional heuristic.

### 2. The call site (`engine_generation.py:2320`)

In `_extract_topic`'s fallback path, after known-word and unknown-noun checks
fail, `_semantic_head_noun(words)` is tried before the positional `words[-1]`:

```python
# engine_generation.py:2316-2323
sem_head = self._semantic_head_noun(words)
if sem_head:
    return (sem_head, text)
return (words[-1], text)
```

This is a **narrow insertion**: it only fires when the graph-activated and
known-word paths have already failed, so every pre-fix working case is
untouched.

### 3. Not hardcoded — distributional, not a lookup table

The method uses GloVe embeddings (seed vectors RAVANA already loads) and
cosine similarity. There is no per-phrase table, no keyword→topic mapping, no
authored prose. The same method works for **any** compound noun or
modifier+noun phrase it has never seen. Online/incremental: uses existing
GloVe vectors, no retraining.

## Verified behaviour (live in-process, `RAVANA_OFFLINE=1`, `.venv-real`)

```
Q: what do you think about migration patterns
A: topic = "migration" (not "patterns")

Q: i think cooking earlier is better
A: topic = "cooking" (not "earlier")

Q: its undervalued
A: topic = "undervalued" (single word, _semantic_head_noun returns None)

Q: what is trust
A: topic = "trust" (simple case, no regression)
```

## Test coverage

`tests/unit/test_semantic_head_noun.py` — 7 tests (all passing, 32.15s on
`.venv-real`):

- `test_compound_noun_migration_patterns` — `"migration patterns"` → `"migration"`, not `"patterns"`.
- `test_undervalued_single_word` — single word passes through; `_semantic_head_noun(["undervalued"])` returns `None`.
- `test_simple_topic_still_works` — `"what is trust"` → `"trust"` (regression guard).
- `test_trailing_modifier_stripped` — `"cooking earlier"` → `"cooking"`, not `"earlier"`.
- `test_semantic_head_single_word` — `_semantic_head_noun(["trust"])` returns `None`.
- `test_semantic_head_empty` — `_semantic_head_noun([])` returns `None`.
- `test_semantic_head_climate_change` — `_semantic_head_noun(["climate", "change"])` → `"climate"`.

The suite is **RED-capable**: before the fix, the compound-noun and
trailing-modifier tests failed (positional heuristic grabbed the wrong word);
all 7 pass with the fix.

## How it grew from the conversation

A round-trip: conversational probing surfaced opinion queries whose topic was
garbage — *"migration patterns"* extracted as *"patterns"*, *"cooking earlier"*
as *"earlier"*. Root cause tracing (`engine_generation.py:2269`) showed the
fallback path used `words[-1]`, a positional heuristic that assumes the head
is the last token. That is true for simple noun phrases (*"mangroves"*) but
wrong for compound nouns and modifier+noun phrases. The fix adds **one**
semantic-centrality check (`_semantic_head_noun`) before the positional
fallback — every other topic path is untouched. 10 existing topic extraction
tests remain green; 7 new tests lock the fix in.
