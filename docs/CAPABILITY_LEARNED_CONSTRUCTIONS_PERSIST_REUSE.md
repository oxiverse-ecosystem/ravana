# Capability: learned syntactic constructions persist + reuse (M16b)

## What it does

RAVANA's `SyntacticCellAssembly` keeps a bounded inventory of learned
grammatical constructions (`learned_frames`). Before M16b, those learned
constructions were lost on `save()`/`load()` and `bind_to_sentence` always
recomputed a frame even when an identical construction had already been
learned.

After M16b:

1. **Persistence.** `get_state()` serializes `learned_frames` (via
   `dataclasses.asdict`) plus `max_learned_frames`; `set_state()` restores
   them (filtering unknown keys, capping to the persisted max). The engine
   round-trips this under the `syntactic_assembly_state` key in its
   `get_state()`/`set_state()` (`engine.py` ~11388 / ~12081).

2. **Reuse.** `bind_to_sentence(subject, relation, target, ...)` first calls
   `find_construction(subject, relation, target)`; if a matching learned
   construction exists it returns a shallow copy of that frame instead of
   recomputing role weights / verb phrase / articles.

**Locations:**
- `ravana/src/ravana/language/syntactic_cell_assembly.py:229` (reuse branch in
  `bind_to_sentence`)
- `ravana/src/ravana/language/syntactic_cell_assembly.py:517` (`find_construction`)
- `ravana/src/ravana/language/syntactic_cell_assembly.py:661` (`get_state`)
- `ravana/src/ravana/language/syntactic_cell_assembly.py:683` (`set_state`)
- `ravana/src/ravana/chat/engine.py:11388` (state key written)
- `ravana/src/ravana/chat/engine.py:12081` (state restored)

## How it grew from the conversation

The M16 usage-based construction substrate (round 2026-10-27T0548Z) added
`allocate_construction` / `find_construction` / `reinforce_construction` over
a bounded `learned_frames` inventory. M16b closes the loop: a construction
learned during a turn must survive the session boundary (save/load) and must
be preferred over recomputation on the next matching turn. Before M16b the
learned inventory was ephemeral, so every session re-learned from scratch and
`bind_to_sentence` ignored what had been learned.

## Design

- Persistence uses the existing `get_state`/`set_state` contract (same as
  `cerebellar_ngram`, `concept_confidence`, etc.); no new storage format.
- `find_construction` matches on `(subject, object, relation)` with
  case-insensitive concept comparison; on a hit `bind_to_sentence` returns
  `copy.copy(learned)` so callers can mutate without corrupting the stored
  frame.
- On load, frames are filtered to known `SyntacticFrame` dataclass fields and
  capped to the persisted `max_learned_frames`, so a stale/larger dump cannot
  overflow the inventory.
- No new reply strings, no authored prose; only data keys and learned frames.
  Online/incremental: a single successful use allocates a construction.

## Verification

- Source change: `git show c8975126` (2 files, +30/-2).
- `get_state`/`set_state` round-trip exercised against the real class; see
  `tests/unit/test_ravana_language.py` `TestSyntacticCellAssembly`.
- Engine boot + save/load restores `syntactic_assembly_state` when present.
