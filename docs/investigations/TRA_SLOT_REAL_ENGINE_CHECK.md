# TRA slot primitives — real-engine verification on main

After merging the narrow default-off implementation (#513), a bounded real-engine
check was run with `RAVANA_TRA_SLOTS=1` on a fresh isolated scratch store.

- 6 natural `is_a` assertions were taught via `process_turn`.
- `typed_slot_views_from_graph` found 489 typed views in-process and **489 after
  `save()` + cold reload** (retention, TRA C9).
- `slot_keyed_forced_choice(views, "mist", ["weather phenomenon", "civic event",
  "celestial event", "animal"])` returned `pick="weather phenomenon"`,
  `reason="trusted_typed_slot_strict_winner"`, score 0.3889 from the stored edge
  confidence (no GloVe cosine, no gold key).
- Disjoint options `["apple","banana","chair","table"]` returned `pick=None`,
  `reason="no_trusted_typed_slot_above_floor"` — TRA C10 fail-closed.
- The production response hook `_typed_slot_candidate` returned
  `"weather phenomenon"` on the forced-choice question.

This verifies only the two primitives on the real engine. It does **not** assess
TRA gates C1–C8, C11 or C13, and is **not** a TRA PASS. The full 400-family
benchmark still has to be implemented and run before any capability claim.
