# Structured Quantity Memory (count-bearing disclosures)

How RAVANA captures *"I keep twelve racing pigeons"* as a **number**, retrieves it
on *"how many racing pigeons do I keep?"*, **aggregates** across species on
*"…in total"*, and **corrects** itself on *"it's seven hives now"* — without an
LLM, without retraining, and without a per-topic answer table.

This capability was added in round **2026-08-11T0521Z** (feature `t_c05047a3`).
All claims below are backed by a live in-process probe (`dim=64`, offline) and the
unit suite `tests/unit/test_quantity_memory_2026_08_11.py` (6 passed).

## The gap it fills

The activity/event miner already preserved the *count* inside the free-text
`'does'` fact (e.g. `"keep twelve racing pigeons"`). But there was **no structured
count store**, so RAVANA could not:

1. **Synthesize** a clean count answer for a *multi-word* noun phrase — the old
   recall regex matched only a single noun word with a fixed verb list, so
   *"how many racing pigeons do i keep"* fell through to a raw-gist echo.
2. **Aggregate** counts across the store (*"how many pets do i have in total"*
   was simply impossible — there was nothing to sum).

`QuantityMemory` adds the missing structured store and wires it into the miner and
the recall path.

## What it does (verified)

1. **Captures a structured count** `(subject, kind, count, noun_phrase,
   noun_canonical, category)` from first-person disclosures — *"i keep twelve
   racing pigeons"*, *"i have three cats"*, *"i bake two sourdough loaves"*,
   *"i lost five hens"*. Verbs are normalized to a *family* (`keep`/`kept` →
   `keep`, `make`/`made`/`bake`/`cook` → `made`, `lose`/`lost`/`drop` → `loss`),
   and the noun is reduced to a **canonical singular** so *"racing pigeons"*
   aggregates with *"homing pigeons"* under `pigeon`.
2. **Answers single lookups**, including multi-word nouns:
   *"how many racing pigeons do i keep"* →
   **`you have twelve racing pigeons.`** (verified live).
3. **Aggregates by category** on an *"in total"* / *"altogether"* / *"all told"*
   cue. The `possession` category sums every species the user disclosed; the
   `loss` category (e.g. *"i lost five hens"*) is **excluded** from a pet total so
   a loss never inflates the count. Verified live: 12 pigeons + 3 cats + 2 dogs +
   4 crabs → **`you have 21 pets in total.`** (the 5 lost hens are not in that
   total).
4. **Corrects online.** After a prior count fact exists, *"it's seven hives now"*
   supersedes the earlier *"i keep six hives"*; a follow-up *"how many hives do i
   keep"* returns **`you have seven hives.`** (verified live). A correction never
   requires a retrain — the corrected count is written straight into the store.
5. **Fails closed on unknown counts.** *"how many goats do i keep"* (never
   disclosed) returns an honest non-answer, never a fabricated number.
6. **Does not mine a question as a fact.** *"how many racing pigeons do i keep"*
   is a question; the interrogative guard means it is **never** seeded as a
   quantity record (asserted by `test_question_not_stored_as_fact`).

## How it grew from the conversation (source citations)

### The store — `ravana/chat/personal_fact_store.py`

`QuantityMemory` is a `dataclass`-record store keyed by
`(subject, kind, noun_canonical, count)` so an update opens a contradiction the
same way `PersonalFactStore` does (active value wins; the user is ground truth):

```python
# ravana/src/ravana/chat/personal_fact_store.py  (class at L527)
class QuantityMemory:
    # verb family -> (kind, category). Seed; covers everyday disclosures.
    # Past-tense forms are included (the verb is normalized to its present
    # family before lookup). Extendable, not a per-topic table.
    VERB_KIND = {                                       # L543
        "keep": ("keep", "possession"), "kept": ("keep", "possession"),
        "have": ("have", "possession"), "had": ("have", "possession"),
        "own": ("own", "possession"), "raise": ("raise", "possession"),
        ...
        "bake": ("bake", "made"), "make": ("make", "made"),
        "lose": ("lose", "loss"), "lost": ("lose", "loss"),
        ...
    }
```

`assert_quantity` (L572) raises the confidence of a repeated disclosure and
**retires any conflicting prior count** (`superseded = True`) so only one active
record per `(subject, kind, noun)` remains. `query_count` (L641) matches by token
overlap — *"racing pigeons"* resolves a stored `pigeon` record — and returns
`None` on no match (never fabricates). `aggregate` (L673) sums the active records
of a category for *"in total"*. `get_state`/`set_state` (L697/L706) ride the
engine's existing pickle/SQLite persistence — **no new database**.

Two helpers keep replies honest and natural:

```python
# ravana/src/ravana/chat/personal_fact_store.py
def number_to_int(token: str) -> Optional[int]: ...   # L493: "twelve"->12, "3"->3, "a"->1
def render_count(count: int) -> str: ...               # L503: 1-12 as words, else digits
```

`render_count` echoes the *mined* surface form — the miner stores counts as words
(*"keep six hives"*), so recall says *"six"*, not *"6"*.

### Mining — `ravana/chat/user_model.py`, `UserModel.mine_personal_facts`

The miner is interrogative-guarded so a first-person *count question* is never
seeded as a fact (mirrors the stance guard):

```python
# ravana/src/ravana/chat/user_model.py  (L1416)
_qty_is_question = (q_clean.rstrip().endswith("?")
                    or bool(re.match(r"^(what|who|when|...|have|has|had)\b", q_clean)))
...
# ravana/src/ravana/chat/user_model.py  (_qty_pat at L1421; assert at L1459)
_qty_pat = re.compile(
    r"\bi\s+(?:also\s+|really\s+|...)?"
    r"(" + "|".join(QuantityMemory.VERB_KIND.keys()) + r")"
    r"(?:s|es|ing|ed|[a-z]ed|[a-z]d)?\s+"
    r"(a|an|one|two|...|twenty|\d+)\s+(.+?)...", re.IGNORECASE)
...
if not _qty_is_question:
    for _qm in _qty_pat.finditer(q_clean):
        ...
        self.quantity_memory.assert_quantity(         # L1459
            "i", _kind, _cnt, _noun_raw, _canon, category=_cat,
            confidence=0.6, source="seed_regex")
```

The canonical noun resolves an animal word to its species via
`pet_slots.species_of` (L1454) so plural/species variants aggregate correctly.

### Correction — `ravana/chat/user_model.py` (L570)

A plain update like *"it's seven hives now"* carries no negation or *"my X is Y"*
shape, so the name-correction paths miss it. A dedicated count-correction detector
matches `cardinal + entity + update-cue` (e.g. *now*, *split*, *added*), finds the
prior count/activity fact for that entity, sets `detected_correction_fact`, and
**mirrors the corrected count into `QuantityMemory`** so recall reflects the new
number:

```python
# ravana/src/ravana/chat/user_model.py  (L624)
self.quantity_memory.correct(         # supersedes the prior record, writes the new count
    subject="i", noun=_ent, count=_qcount)
```

`QuantityMemory.correct` (L596) retires any active record matching the noun and
re-asserts the corrected count at higher confidence (`source="correction"`).

### Recall — `ravana/chat/engine.py`, the count/quantity block

The recall path parses *"how many …"*, distinguishes an aggregation cue from a
single lookup, and renders every reply from `QuantityMemory` state:

```python
# ravana/src/ravana/chat/engine.py  (COUNT block at L2519)
if _agg_word and _qm is not None:
    _total = _qm.aggregate(category="possession")      # L2540
    if _total > 0:
        return f"you have {render_count(_total)} {_label} in total."   # L2546
...
_rec = _qm.query_count(_cn, kind=_kind)                # L2555
if _rec is not None:
    return f"you have {render_count(_rec.count)} {_cn}."   # L2557
```

There is **no authored reply pool** — the words "you have", the count, and the noun
all come from the store; `render_count` is a thin connector.

## Why this is not hardcoding

- Counts are **stored**, not matched to pre-written sentences. A new disclosure
  (any verb in `VERB_KIND`, any noun) is captured with zero code change.
- Every reply reads `count` / `noun` / `total` from state. The only free text is
  the connective *"you have … in total."* — a pure function of the numbers.
- The verb family and number-words are **seed vocabulary RAVANA can grow at
  runtime**; missing an entry degrades to "unknown", never to a wrong answer.
- Corrections are **online** (`correct()` writes the store); no retrain, no
  regeneration step.

## Tests

`tests/unit/test_quantity_memory_2026_08_11.py` (6 tests, all pass):

- `test_count_captured_for_multiword_noun` — *"i keep twelve racing pigeons"* →
  count 12, canonical `pigeon`.
- `test_count_captured_for_varied_verbs_and_nouns` — `have`/`bake`/`lose` families.
- `test_aggregation_across_species` — possession sums, `loss` excluded.
- `test_question_not_stored_as_fact` — a count question seeds no record.
- `test_number_word_parser` — `number_to_int` word/digit handling.
- `test_correction_supersedes_prior_count` — *"no, i have four cats actually"*
  supersedes the earlier *"i have three cats"*.

Engine-level surface recall (the *"you have N X."* string) is exercised by the
end-to-end probe described above; it is intentionally not a heavy engine-init unit
test (full engine boot ≈ 27 s).
