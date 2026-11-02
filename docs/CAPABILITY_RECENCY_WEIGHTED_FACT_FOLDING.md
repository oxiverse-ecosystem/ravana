# Capability: recency-weighted fact folding in episodic recall

**Status:** shipped (commit `e1261ef6`, branch `auto/round-2026-11-02T0737Z`). NOT pushed.
**Feature card:** `t_dcc60ddc` (round `2026-11-02T0737Z`).
**Docs card:** `t_4f19ad19`.
**Verified:** episodic recall tests pass on this branch; the fold logic reads
`confidence` and `turn_number` from the live `PersonalFact` objects.

## What it does

RAVANA's episodic recall index (`self._episodic_index`) is an
`entity -> attribute -> value` map rebuilt each recall turn from two sources:
the stored episode records (`store`) and the live `PersonalFactStore`
(`self.user_model.personal_facts.facts`). When several facts compete for the
same `(entity, attribute)` slot — e.g. a user corrects "my cat's name is
milo" with "no, my cat is nora", or accumulates repeated disclosures with
slightly different values — the index used to keep whichever value happened
to land last in dict iteration order. That ordering is not semantic, so a
stale value could overwrite a fresh, higher-confidence one.

The fix folds facts into the index with a per-(entity, attr) score:

```
score = confidence * (1.0 / (1.0 + max(0, turn_num - fact.turn_number) * 0.1))
```

A new fact only replaces the current slot value when its score strictly
exceeds the stored score for that slot. Older / lower-confidence values are
suppressed instead of clobbering the fresher one. Superseded facts (user
corrections) are still skipped outright before folding, so a retired value
never races the active one.

## Where it lives

- `ravana/src/ravana/chat/engine_memory.py`, inside the episodic recall
  builder (the `MemoryMixin` method that constructs `_entity_idx`):
  - `_entity_score = {}` tracked per `(entity, attr)` key
    (engine_memory.py ~L1176).
  - `_fold_fact(_e, _a, _v)` helper performs the score comparison and
    conditional write (engine_memory.py ~L1220-1225).
  - The three fold sites: pet facts (`_fold_fact(_sp, _idxnum, _val)`,
    ~L1230), possession-name facts (`_fold_fact(_ent, _attr, _val)`,
    ~L1243), and possession-attribute facts (`_fold_fact(_ent, _attr, _val)`,
    ~L1259).

## How it grew from the conversation

The feature was motivated by the "stale accumulated values" defect class in
episodic recall: the index is rebuilt from a dict whose iteration order is
not guaranteed to be recency-ordered, so a high-confidence recent disclosure
could be silently overwritten by an older, lower-confidence one landing later
in iteration. The fold makes the winning value a function of the fact's own
`confidence` and `turn_number` rather than of dict ordering.

## Verified claims

- The score formula uses `getattr(_fact, "confidence", 0.6)` and
  `getattr(_fact, "turn_number", 0)` (real attributes on `PersonalFact`).
- Superseded facts are filtered before folding (`getattr(_fact, "superseded", False)`).
- The change is confined to `engine_memory.py`; no new public API.
- Episodic recall tests pass on `auto/round-2026-11-02T0737Z`.

## Hardcoding audit

No authored reply prose added. The change is a ranking/fold rule over live
fact attributes; no reply strings were introduced.
