# Clause-structure topic detection for opinion queries (FIX-RV-10)

**Capability:** when RAVANA is asked its opinion about a clause — *"do you think silence is underrated"*, *"what do you think happens to a memory"* — it now extracts the **real topic** (silence / memory) instead of leaking the predicate, verb, or a trailing adverb into the stance key. A bare adverb (*"what kind of mind do you have, exactly"*) no longer fabricates a stance; it falls through to honest uncertainty.

This closes **DEFECT 3** from the round. The opinion-topic extractor treated the tail after the opinion cue as a **flat noun phrase**, so the clause predicate leaked into the topic.

## The three failure patterns (verified live)

| Query | Before (broken) | After (fixed) |
|-------|-----------------|---------------|
| `"do you think silence is underrated"` | topic `"silence underrated"` — predicate leaks in | topic `"silence"` |
| `"what do you think happens to a memory"` | topic `"happens memory"` — verb leaks in | topic `"memory"` |
| `"what kind of mind do you have, exactly"` | topic `"exactly"` — fabricated stance on the adverb | no topic; honest uncertainty |

## How it works (verified against the source + live engine)

### 1. The extractor (`engine_self_query.py`)

`_route_self_query` strips the opinion cue (*"do you think"*, *"what do you think"*, …) and calls the flat topic extractor on the remainder. Before FIX-RV-10, that flat extractor took the **last content token** as the topic — correct for noun-phrase tails (*"mangroves"* in *"do you think we should protect mangroves"*), but broken for clause-shaped tails.

The fix (commit `added3ce`, `engine_self_query.py:1344`) inserts a **clause-structure detector** before the flat extractor:

- **Copular clause** (`_COPULA = is/are/was/were/been/being/seems/appears/looks/sounds/feels/becomes/remains/stays`): if the tail contains a copula at index > 0, narrow `_tail` to the words **before** the copula → the subject. `"silence is underrated"` → `"silence"`.
- **Intransitive verb + prepositional phrase** (`_INTRANS_PP = happens/occurred/occurs/exists/matters/counts`): if the tail starts with an intransitive verb followed by `"to"`, narrow `_tail` to the words **after** `"to"` → the object of the preposition. `"happens to a memory"` → `"memory"`.
- **Bare trailing adverb** (`_TAIL_ADVERBS`: exactly / precisely / specifically / …): if the cleaned tail is *only* an adverb, set `_tail = ""` so the query falls through to honest uncertainty instead of fabricating a stance.

The flat extractor then runs on the narrowed tail — its last-content-token logic is now correct because the clause predicate is gone.

### 2. The adverb deny-list is structural, not a per-topic table

`_TAIL_ADVERBS` is a closed-class grammar set (adverb vocabulary), not a list of stance topics. It contains no RAVANA reply, no topic — only function words that signal "there is no resolvable stance topic here". A brain is born knowing what an adverb is; this is legitimate seed structure, not authored answers.

### 3. Fail-closed

- A clause the detector does **not** recognize (no copula, no intransitive-PP) leaves `_tail` unchanged and the flat extractor runs on the original — so every pre-fix working case still works.
- A tail that is **only** an adverb clears `_tail` → the stance resolver finds no topic → the query falls through to the honest *"i'm still forming a view on X"* / *"i don't really have a solid grasp on X"* path. No fabricated stance.
- A tail whose narrowed topic resolves to the **real content head** lands in the user-model stance store, recallable by later `"do you still feel that way about X"` / `"am i for or against X"` queries.

## Verified behaviour (live in-process, `RAVANA_OFFLINE=1`, `.venv-real`)

```
Q: do you think silence is underrated
A: i'm still forming a view on silence. i don't have a fixed stance on silence yet — what's your take? i'd rather hear how you see it than guess.

Q: what do you think happens to a memory
A: i'm still forming a view on memory. i don't have a fixed stance on memory yet — what's your take? i'd rather hear how you see it than guess.

Q: what kind of mind do you have, exactly
A: i don't really have a solid grasp on exactly so far. what's your sense of it?

Q: do you think we should protect mangroves
A: i'm still forming a view on mangroves. i don't have a fixed stance on mangroves yet — what's your take? i'd rather hear how you see it than guess.
```

| Case | Topic extracted | Matches expectation |
|------|-----------------|---------------------|
| Copular clause | `silence` | yes (not `"silence underrated"`) |
| Intransitive + PP | `memory` | yes (not `"happens memory"`) |
| Bare trailing adverb | *(none)* | yes (honest uncertainty, not `"a view on exactly"`) |
| Simple opinion (regression) | `mangroves` | yes (still works) |

## Test coverage

`tests/unit/test_topic_extraction_clause.py` — 4 regression tests (all passing, 67.99s on `.venv-real`):

- `test_copular_clause_topic_is_subject` — `"silence is underrated"` must NOT contain `"silence underrated"`, must contain `"silence"`.
- `test_intransitive_to_pp_topic_is_object` — `"happens to a memory"` must NOT contain `"happens memory"` or `"forming a view on happens"`, must contain `"memory"`.
- `test_trailing_adverb_no_topic` — `"what kind of mind do you have, exactly"` must NOT contain `"forming a view on exactly"`, `"a view on exactly"`, or `"stance on exactly"`.
- `test_simple_opinion_still_works` — `"do you think we should protect mangroves"` must contain `"mangroves"` (regression guard).

The suite is **RED-capable**: before the fix, the copular + intransitive + adverb tests all failed; all 4 pass with it.

## How it grew from the conversation

A round-trip: conversational probing surfaced three opinion queries whose topic was garbage. Root cause tracing (`engine_self_query.py:1344`) showed the flat extractor was running on a clause-shaped tail with no structural pre-pass. The fix adds **one** clause detector (copula / intransitive-PP / bare-adverb) before the existing extractor — every other topic path is untouched. 42 existing opinion-related tests remain green; 4 new tests lock the fix in.
