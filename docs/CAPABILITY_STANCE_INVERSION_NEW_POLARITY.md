# Capability: stance inversion with new-polarity derivation (D4 fix)

**Status:** IN PROGRESS — partial implementation, test failing (round
`2026-10-13T0958Z`, feature `t_969da1c2`).
**Verified:** `_mine_new_opinion_polarity` works correctly in isolation
(returns +0.52 for "wonderful", -1.0 for "terrible"). The end-to-end
routing does NOT reach `_execute_stance_inversion` for natural
contradiction inputs. The test
`tests/unit/test_stance_inversion_new_polarity.py` FAILS (3/3 checks).

## What it is supposed to do

When a user contradicts a previously-held stance with a new opinion
(e.g. "changed my mind — silence is actually wonderful"), the engine
should derive the NEW polarity from the post-contradiction clause and
use `recode_stance_toward` to move the stance toward that value. The
old behavior was to blindly flip via `reverse_stance`, which doesn't
reflect WHAT the user now believes — only that they changed their mind.

This addresses **defect D4** from the round report
`tmp/reports/ravana-2026-10-15T0342Z.md`:

> **D4: Shallow stance update on contradiction** — "got it — you've
> changed your mind about actually" doesn't articulate the new stance.
> The engine records the fact of change but doesn't derive a new
> polarity.

## What was implemented

Three changes to `ravana/src/ravana/chat/engine_self_query.py`:

### 1. `_mine_new_opinion_polarity()` (line 717)

New method that extracts the post-contradiction clause and derives a
polarity from it. Uses the `EvaluativePolarityModel` (GloVe-based) to
score evaluative predicates in "X is Y" constructions, with a
sentiment-word fallback. Returns a float in [-1, +1] or None.

Verified in isolation:
- `_mine_new_opinion_polarity("changed my mind — silence is actually wonderful", "silence")` → `+0.52`
- `_mine_new_opinion_polarity("changed my mind — talking is actually terrible", "talking")` → `-1.0`

### 2. `_execute_stance_inversion()` modified (line 799)

When `_mine_new_opinion_polarity` returns a non-None value, uses
`recode_stance_toward(topic, new_pol, blend=0.7)` instead of
`reverse_stance(topic)`. Falls back to `reverse_stance` when no
new opinion can be mined.

### 3. Contradiction regex broadened (line 519)

The natural-contradiction regex now matches "changed my mind" without
requiring a preceding "i", and strips em-dashes/colons from the
post-contradiction clause.

### 4. `_VERB_SCAFFOLD` made local (line 1554)

Replaced the import from `.engine` with a local tuple to avoid a
circular dependency. No behavior change.

## Why the test fails

The test inputs ("changed my mind — silence is actually wonderful" /
"changed my mind — talking is actually terrible") are handled by a
different path in `process_turn` — NOT by `_route_own_stance_inversion`.
The PFC classifies these as user opinion disclosures, not as
self-queries about RAVANA's own stance. The stance inversion path is
only reached when the PFC routes to `_route_self_query` →
`_route_own_stance_inversion`, which requires the input to be
classified as a question about RAVANA's own view.

The `_mine_new_opinion_polarity` method exists and works, but it is
never called because `_execute_stance_inversion` is never reached.

Evidence (traced live):
```
process_turn("i hate silence")  → stance 'silence' = -0.95
process_turn("changed my mind — silence is actually wonderful")
  → _route_own_stance_inversion NOT called
  → _execute_stance_inversion NOT called
  → stance 'silence' = -0.53 (moved by free-form recode path, not inversion)
  → reply: "it sounds open."
```

## What needs to happen next

The routing gap needs to be closed. Either:

1. The PFC's prototype classifier needs to recognize "changed my mind"
   as a self-query about RAVANA's own stance (not just a user opinion
   disclosure), OR
2. A new routing path needs to catch natural-contradiction utterances
   BEFORE the opinion-mining path, when RAVANA holds a stance on the
   same topic, OR
3. The free-form contradiction recode path (which DID move the stance
   from -0.95 to -0.53) needs to be extended to use
   `_mine_new_opinion_polarity` instead of its current blind-flip
   logic.

Option 3 is the most targeted: the free-form recode path already
detects the contradiction and moves the stance — it just doesn't
derive the new polarity from the post-contradiction clause.

## Hardcoding audit

No reply-producing strings were added. The `_mine_new_opinion_polarity`
method uses the existing `EvaluativePolarityModel` (GloVe-based, not a
lookup table) and a small positive/negative word set as fallback. The
word set is seed vocabulary (like `_OPINION_STOP`), not a per-topic
answer table. No Q→A dictionary, no keyword→reply table.

## Where it lives

| Concern | Location |
|---------|----------|
| `_mine_new_opinion_polarity` method | `ravana/src/ravana/chat/engine_self_query.py:717` |
| `_execute_stance_inversion` (modified) | `ravana/src/ravana/chat/engine_self_query.py:799` |
| `_route_own_stance_inversion` (routing) | `ravana/src/ravana/chat/engine_self_query.py:485` |
| Natural contradiction regex | `ravana/src/ravana/chat/engine_self_query.py:519-531` |
| `recode_stance_toward` (target API) | `ravana/src/ravana/chat/personal_fact_store.py:678` |
| `reverse_stance` (fallback) | `ravana/src/ravana/chat/personal_fact_store.py:607` |
| Test (failing) | `tests/unit/test_stance_inversion_new_polarity.py` |
