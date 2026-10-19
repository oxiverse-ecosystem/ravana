# Capability: duration mining attaches to the nearest preceding activity verb

**Status:** shipped (commit `e57d44e5`, branch `auto/round-2026-10-19T0513Z`).
**Verified:** 5/5 new tests pass (`tests/unit/test_duration_attaches_to_nearest.py`);
27 existing tests unaffected.

## What it does

When the user says they have been doing something for a span of time across
multiple clauses, RAVANA now attaches the duration to the **nearest preceding
activity verb** — not the first verb in the clause.

Before the fix, a sentence like:

> my favourite instrument is the veena and i have been playing it for five years

would mine `since(is …)` — attaching "for five years" to "is" (the first
activity-ok verb in the clause) rather than "playing" (the nearest one). The
duration was effectively dropped from fact recall because "is" is not a real
activity.

After the fix, the same sentence mines `since(play … five years)` — the
duration correctly attaches to "playing", the nearest preceding activity verb.

## How it grew from the conversation

The parent chat round (`t_135e88a0`, 2026-10-19T0513Z) surfaced a residual
limitation (T06): the duration "for five years" was dropped from fact recall in
multi-clause sentences. The root cause was a `break` statement in the duration
mining block (b) in `user_model.py` that caused the first activity-ok verb to
be picked instead of the last.

The fix removes the `break` so the loop continues to the nearest preceding
activity verb. An object guard was also added to skip words that immediately
follow another activity-ok word (they are objects, not separate activities) —
e.g. "practicing yoga for three years" correctly attaches to "practicing", not
"yoga".

## Implementation

**File:** `ravana/src/ravana/chat/user_model.py`, block (b), lines 4909-4968.

The fix is in the activity-verb scan loop (lines 4943-4957):

```python
for _i, _v in enumerate(_av):
    _vl = _v.lower()
    if _vl in _ASPECTUAL_VERBS:
        continue
    if _activity_verb_ok(_vl):
        # Skip a word that immediately follows another activity-ok
        # word — it is the OBJECT of that verb, not a separate
        # activity. "practicing yoga for three years" — "yoga" is
        # the object of "practicing", not a separate activity, so
        # the duration attaches to "practicing".
        if _i > 0 and _av[_i - 1].lower() not in _ASPECTUAL_VERBS \
                and _activity_verb_ok(_av[_i - 1].lower()):
            continue
        _verb = _vl
        _vidx = _i
```

Key changes:
1. **Removed `break`** — the loop now continues scanning all words before the
   duration marker, so `_verb` ends up as the LAST (nearest) activity-ok verb,
   not the first.
2. **Object guard** (lines 4948-4955) — if the current word immediately follows
   another activity-ok word, it is the object of that verb, not a separate
   activity, so it is skipped.

## Tests

`tests/unit/test_duration_attaches_to_nearest.py` — 5 tests, all pass:

- `test_duration_attaches_to_nearest_activity` — "for five years" attaches to
  "playing", not "is".
- `test_duration_recall_renders_correct_activity` — recall query returns the
  correct activity with duration.
- `test_duration_recall_start_year` — recall query for start year returns the
  correct year (2021 for "five years" from 2026).
- `test_simple_duration_still_works` — simple single-clause duration still works
  (no regression).
- `test_multiclause_duration_picks_last_verb` — in a multi-clause sentence,
  duration picks the nearest verb ("practicing", not "live").

## Design compliance

- **Seed knowledge only.** The fix modifies the verb-selection logic in block
  (b); no new verb lists or per-topic tables were added. The existing
  `_activity_verb_ok` and `_ASPECTUAL_VERBS` sets are reused.
- **Online / incremental, no retraining.** The fix is a structural change to
  the verb-scan loop; nothing requires a rebuild.
- **Zero authored reply prose.** The reply is rendered from mined `since`
  facts by the pre-existing resolver.
