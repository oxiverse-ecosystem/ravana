# TRA typed role-slot primitives — narrow default-OFF production port

**Status: two primitives only. This is NOT a TRA PASS and NOT a benchmark result.**

This ports the POSITIVE result of `TRA_SLOT_WRITE_READ_EXPERIMENT.md` into
production as a narrow, default-OFF change. No threshold, trust predicate,
gate, benchmark, `models.py` or `ravana_ml/graph.py` was edited.

## What was ported

| Half | Scratch source | Production location |
|---|---|---|
| Parse natural `is_a` assertions | `parse_typed_assertion` / `IS_A_RE` / `REFUSE_RE` | `chain_walker.py:995` `_TYPED_IS_A_RE`, `:1003` `_TYPED_REFUSE_RE`, `:1008` `_typed_slot_assertions` |
| Write through the unchanged intake API | `scratch_write_typed_assertion` | `chain_walker.py:1171` `ChainWalkerMixin._consolidate_asserted_typed_slots` |
| Call the writer at `process_turn` | scratch called it directly | `engine.py:10874` defensive `getattr(self, "_consolidate_asserted_typed_slots", None)` hook beside the existing relation hook |
| Read typed slots from stored edges | `_rec` + `iter_edges` | `schema_induction.py:171` `typed_slot_views_from_graph` |
| Fail-closed slot scorer | `slot_keyed_forced_choice` | `schema_induction.py:208` `slot_keyed_forced_choice` |
| Answer hook | scratch had none (standalone) | `response_gen.py:7683` `ResponseGenMixin._typed_slot_candidate`, called from `_schema_candidate` at `:7720` |

### Exact functions

- `schema_induction.slots_enabled()` (`:136`) — reads `RAVANA_TRA_SLOTS`;
  only the literal value `1` enables. `SLOT_ENV_FLAG` at `:130` is distinct
  from `RAVANA_SCHEMA`, so the two paths can be A/B'd independently.
- `schema_induction.TypedSlotEdgeView` (`:150`) — frozen read shape carrying
  `source`, `relation_type`, `target`, `type_label`, `slot_label`, `edge_kind`,
  `edge_source`, `source_type`, `confidence`.
- `schema_induction.typed_slot_views_from_graph(graph)` (`:171`) — pure read
  adapter. Provenance is read from `source_metadata` (with an attribute
  fallback for duck-typed fakes), never from a bare `getattr` on a live edge.
- `schema_induction.slot_keyed_forced_choice(slot_views, subject, options, *, congruence_floor=0.20, strict_margin=0.10, typed_rel="is_a")` (`:208`)
  — pure, stateless. An option is eligible only when a **trusted** typed edge
  with a dedicated `type_label` stamp goes from the subject to that exact
  option; score = **stored edge confidence only** (no GloVe, no item metadata,
  no gold answer, no `world_model`). Returns `pick=None` with
  `reason=no_trusted_typed_slot_above_floor` for disjoint options / unrelated
  subjects / untrusted edges / below-floor scores, and
  `reason=strict_margin_not_met` for a tie.
- `ChainWalkerMixin._typed_slot_assertions(text)` (`:1008`) — returns
  `(IS_A, subject, object, type_label=object, slot_label="type")`. Hedges,
  negation, questions and non-`is_a` clauses are refused. The scratch word-
  boundary contraction rule is used verbatim; the shared
  `_ASSERTION_NEGATION_RE` was **not** reused because its `[a-z]n...t\b`
  pattern falsely rejects ordinary words such as `event`.
- `ChainWalkerMixin._consolidate_asserted_typed_slots(text)` (`:1171`) —
  returns `0` before any graph access when the flag is off. When on it calls
  `intake.intake(Fact(...), Provenance(source="conversation",
  source_type="user", ...), salience=1.0)`, verifies `_is_trusted_edge` on the
  resulting edge, and stamps `relation_type="is_a"`, `asserted_relation`,
  `type_label`, `slot_label` and `slot_writer` on the live `source_metadata`.
  No trust threshold is widened; an untrusted edge is skipped.
- `engine.py:10874` — hook calls the consolidator via `getattr` and ignores
  absence/errors. It is deliberately **not** added to the `consolidated`
  counter, so the downstream `weak_associations` signal is unchanged.
- `ResponseGenMixin._typed_slot_candidate` (`:7683`) — flag-gated, forced-
  choice-only, uses `parse_forced_choice` plus the two read primitives.
  `_schema_candidate` (`:7720`) calls it via `getattr` **before** the
  `RAVANA_SCHEMA` guard, so the new flag is independent and duck-typed engines
  without the hook stay inert.

## Flag-off equivalence

`RAVANA_TRA_SLOTS` defaults to `0`. With it unset the writer returns `0`
without touching the intake or the graph, and the reader hook returns `None`
without touching the graph. This is verified by two differential tests:
`test_write_flag_off_is_byte_identical` (no intake calls, empty graph) and
`test_hook_flag_off_is_byte_identical_and_flag_on_picks` (same graph and
prompts give `None` off, a typed pick on, and still `None` on disjoint options
with the flag on).

## Tests

Command (worktree, main repo venv, worktree `PYTHONPATH`):

```
cd C:/Users/Likhith/Documents/projects/ravana_tra_impl
export PYTHONPATH="C:/Users/Likhith/Documents/projects/ravana_tra_impl/scripts;C:/Users/Likhith/Documents/projects/ravana_tra_impl/ravana/src;C:/Users/Likhith/Documents/projects/ravana_tra_impl/ravana_ml/src;."
RAVANA_OFFLINE=1 C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe \
  -m pytest tests/unit/test_tra_slots.py -q
```

- `tests/unit/test_tra_slots.py`: **14 passed** (14 test functions). Coverage:
  6-write-through-intake with provenance assertions; flag-off byte-identity;
  parser refusal; cold-reload simulation (pickle roundtrip) with 6/6 slots;
  positive pick + permutation stability; C10 disjoint → `None`; unrelated
  subject → `None`; untrusted typed edge → `None` while a trusted control in
  the same graph still picks; tie → `strict_margin_not_met`; below-floor →
  `no_trusted_typed_slot_above_floor`; signature/gold-leakage check; flag-off
  vs flag-on differential; `RAVANA_SCHEMA` independence; duck-typed fake
  without the hook stays inert; no cross-turn state.
- Related regression set in the **same** pytest process:
  `test_tra_slots.py`, `test_schema_induction.py`, `test_schema_wiring.py`,
  `test_schema_boundary.py`, `test_trusted_edge_choice.py`,
  `test_rsi_all_relations.py` → **87 passed**.
- Module resolution was asserted at runtime:
  `ravana.chat.engine.__file__`, `chain_walker.__file__`,
  `schema_induction.__file__` and `response_gen.__file__` all resolved under
  `C:/Users/Likhith/Documents/projects/ravana_tra_impl`, not the main install.

No live engine was booted and no benchmark was run.

## NOT assessed — no TRA PASS

**TRA benchmark gates C1–C8, C11 and C13 are NOT assessed. This change is NOT
a TRA PASS.** Only the write/read primitives and their unit-level controls
were exercised. In particular: no 400-family item bank, no k-sweep, no
near/far/foil/negative items, no live fake-arm or API-counter harness (C11),
and no end-to-end benchmark gate C1–C8/C13. The shipped `RAVANA_SCHEMA=1`
answer path is unchanged; the disjoint-option failure it emits (`banana`) was
a contrast observation in the scratch experiment, not something this change
claims to fix in that path. Promotion to a default-on TRA capability requires
the full TRA harness with the disjoint control wired as a hard gate.

## Honest caveats

- The cold-reload test is a pickle roundtrip of a fake graph, not a real
  `CognitiveChatEngine` save + cold boot; the real 6/6 retention evidence
  remains from the scratch experiment.
- The write fake mirrors the `intake.intake` call shape and stamping rules; it
  is not the real `KnowledgeIntake`, which was exercised in the scratch run.
- Thresholds (`0.20` floor, `0.10` margin) are designer choices fixed in the
  scratch experiment, not TRA-pre-registered values; no sensitivity sweep was
  run here.
- Only `is_a`/category assertions are handled. `part_of`, `precedes`,
  `resembles` and other families are untouched.
