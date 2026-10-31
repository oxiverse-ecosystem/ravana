# Capability: cross-turn subject repetition drives anaphora (M17 consumer)

**Status:** shipped (commit `673fef10`, branch `auto/round-2026-10-31T0417Z`).
**Verified:** `ravana/src/ravana/chat/engine_generation.py` compiles; import of
`CognitiveChatEngine` succeeds (`RAVANA_OFFLINE=1`, `.venv-real`).

## What it does

When the same subject recurs across two consecutive turns, RAVANA's surface
realizer **pronounizes (anaphorizes)** the repeated subject instead of
re-emitting the full noun phrase. The decision is driven by RAVANA's own
discourse state, not an authored reply:

- `DiscourseState.subject_repetitions` (in
  `ravana/src/ravana/language/surface_realizer.py:53`) already existed as the
  counter the realizer reads in `_resolve_pronoun` (line 869) to choose a
  pronoun ("it"/"they"/variants) instead of the full noun.
- The `previous_*` referents on `GenerationMixin` (`_discourse_prev_subject`,
  `_discourse_prev_object`, `_discourse_prev_verb`) were populated on every turn
  but the `subject_repetitions` field of the `DiscourseState` was left at its
  default `0` — so the realizer never knew a repetition had happened.
- This capability wires the two together: in `engine_generation.py` around the
  `DiscourseState(...)` construction (lines 283-304), it compares the previous
  turn's subject with this turn's subject (`ctx.subject`) and sets
  `subject_repetitions=1` when they match (case-insensitive, trimmed). That is
  the single signal `_resolve_pronoun` needs to switch from the full noun to a
  pronoun.

Generalizes to any subject — there is no subject allowlist, no per-topic
pronoun table, and no retraining step. The only state read is the discourse
referent that RAVANA already tracks.

## How it grew from the conversation

The prior commit (`1b898f53`) wired the `DiscourseState previous_*` referents
into the generation path (cross-sentence continuity) but left `subject_repetitions`
unused — an "orphaned" population. This feature closes that gap: the M17
consumer now reads the referent it was already given. No new authored prose; the
diff is net +10 lines of wiring.

## Source references

- `ravana/src/ravana/chat/engine_generation.py:283-304` — subject comparison +
  `subject_repetitions=_cross_turn_repeat`.
- `ravana/src/ravana/language/surface_realizer.py:48-57` — `DiscourseState`
  dataclass (`subject_repetitions: int = 0`).
- `ravana/src/ravana/language/surface_realizer.py:869-...` — `_resolve_pronoun`
  reads `context.subject_repetitions` to pick the pronoun variant.

## Hardcoding self-audit

Zero authored reply strings added. The only added logic is a structural
string-equality check on already-tracked discourse referents. Honest fallback
unchanged.
