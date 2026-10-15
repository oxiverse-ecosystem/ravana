# Reversal Ack Articulates New Polarity — telling the user what RAVANA understood their new stance to be

When a user reverses a held stance ("i flipped, the reef tank is more work than joy"),
RAVANA recodes the stance and acknowledges the change. Before this fix, the ack only
articulated the OLD polarity — "you were for it before" — never the NEW one. The user
had no way to know whether RAVANA understood their new position correctly.

This page documents the fix: what it does, how it is implemented, how it grew out of
a real conversational gap, and how it is verified.

All claims below were checked against the source on branch
`auto/round-2026-10-13T1307Z` at commit `30e87480`. Line numbers cite that tree.

## What it does

The reversal ack now renders BOTH the old and new polarity direction, so the user
knows what RAVANA understood their new position to be.

Before the fix:

```text
user: "i flipped, the reef tank is more work than joy"
RAVANA: "got it — you've changed your mind about reef tank; i'll remember (you were for it before)."
```

After the fix:

```text
user: "i flipped, the reef tank is more work than joy"
RAVANA: "got it — you've changed your mind about reef tank; i'll remember (you were for it before, now you're against it)."
```

The direction words ("now you're for it", "now you're against it", "now you're less
sure") are derived from the live recoded stance value (`_rev[2]` from `last_reversal`),
not authored prose. The polarity band thresholds (0.25) are the same general semantic
bars used throughout the engine for rendering stance direction.

## How it grew — the residual gap

Round `t_781c11fe` (auto cycle 2026-10-13T1307Z) logged limitation D4:

> "Shallow stance update on contradiction — 'got it — you've changed your mind about
> actually' doesn't articulate WHAT the new stance is. The engine records the fact of
> change but doesn't derive a new polarity from the new statement."

The query that exposed it: "changed my mind — words matter more". The response was
"got it — you've changed your mind about actually" — it acknowledged the change but
never told the user what RAVANA understood the new stance to be.

## Root cause

In `engine_reasoning.py:2330-2356`, the reversal ack composer read `last_reversal`
(a tuple of `(topic, old_polarity, new_polarity)`) but only used `_rev[1]` (old
polarity). The new polarity `_rev[2]` was available in the tuple but never articulated.
The ack said "you were for it before" but never said what the user's new stance is.

## Mechanism (real code paths)

The fix is a 13-line change in `engine_reasoning.py:2334-2354`:

1. **Read `_rev[2]` (new polarity)** from `last_reversal` — this is the recoded
   stance value after `reverse_stance` or `recode_stance_toward` mutates it.

2. **Render as direction word** using the same grounded pattern as the old polarity:
   - `> 0.25` → "now you're for it"
   - `< -0.25` → "now you're against it"
   - else → "now you're less sure"

3. **Include in ack** — the ack now says "you were for it before, now you're against
   it" instead of just "you were for it before".

The `last_reversal` tuple is set by `PersonalFactStore.reverse_stance`
(`personal_fact_store.py:411`) as `(topic, old_polarity, new_polarity)`. The same
tuple is also set by `recode_stance_toward`, so the fix applies to both the
keyword-cued reversal path and the free-form contradiction recode path.

### Why this is NOT hardcoding

- The direction words are **derived from the live recoded stance value** (`_rev[2]`),
  not authored prose.
- The polarity band thresholds (0.25) are the same general semantic bars used
  throughout the engine for rendering stance direction.
- No per-topic logic, no keyword→response tables, no hardcoded reply strings.
- Online/incremental: no retraining, no offline rebuild.
- The test "Can RAVANA change this by itself, through experience?" — YES: the
  direction words adapt as the stance value changes from conversation feedback.

## Verification

Covered by `tests/unit/test_reversal_ack_new_polarity.py` (6 tests, all passing):

- `test_last_reversal_contains_new_polarity` — verifies last_reversal tuple has
  (topic, old, new).
- `test_reversal_ack_articulates_new_polarity_positive_to_negative` — verifies
  for→against flip produces a negative new polarity.
- `test_reversal_ack_articulates_new_polarity_negative_to_positive` — verifies
  against→for flip produces a positive new polarity.
- `test_reversal_ack_articulates_new_polarity_softening` — verifies softening
  reversal lands near neutral (|new_pol| < 0.3).
- `test_recode_stance_toward_also_sets_new_polarity` — verifies the free-form
  recode path also sets new polarity in last_reversal.
- `test_ack_direction_words_match_polarity_bands` — verifies direction-word logic
  matches polarity band thresholds (0.25).

Regression suites all pass: 24 personal_fact_store + 10 regression + 30 dehardcode
— zero regressions.

## Limits

- The ack composition change does NOT affect the reversal detection logic or the
  stance store mutation — only how the ack renders the already-computed new polarity.
- The direction words are rendered from the same polarity bands used throughout the
  engine; they are structural vocabulary, not per-topic prose.
- The fix does NOT add an LLM, does NOT add hardcoded reply strings, and does NOT
  require retraining or offline rebuild.
