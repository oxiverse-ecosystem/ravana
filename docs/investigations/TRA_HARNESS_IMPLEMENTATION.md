# TRA harness — implementation vs pre-registered spec

**Status: harness implemented and statically/fake-verified. NO engine was
booted and NO benchmark run was performed in this worktree.** The owner of the
verification run is the parent agent. This document records what was built,
the exact frozen gates, the hashes, the test counts, and the commands for the
fake then live runs.

Worktree: `C:/Users/Likhith/Documents/projects/ravana_tra_harness`, branch
`feat/tra-benchmark`, from main `b9cea40`.

Spec: `docs/investigations/TRA_BENCHMARK_SPEC.md` (pre-registration, 353 lines).
Prior experiments read: `TRA_BRAIN_DESIGN.md`, `TRA_WRITE_READ_EXPERIMENT.md`,
`TRA_SLOT_WRITE_READ_EXPERIMENT.md`, `TRA_SLOT_IMPLEMENTATION.md`,
`TRA_SLOT_REAL_ENGINE_CHECK.md`.

Files owned and changed (only these):

| File | Content |
|---|---|
| `scripts/tra_items.py` | 400-family frozen item bank, hash split, validation, manifest hashes |
| `scripts/measure_tra.py` | Phases P0-P6, fake/live backends, parser, C1-C13, saturation, additive JSON, exit codes |
| `tests/unit/test_measure_tra.py` | 31 fake-only unit tests (no engine boot) |
| `docs/investigations/TRA_HARNESS_IMPLEMENTATION.md` | this document |

No file under `ravana/src`, `ravana_ml/src`, other `scripts/` or other `tests/`
was edited. No threshold, gate, item or candidate was tuned after a result.

---

## 1. Frozen constants, counts and hashes

```
HASH_PREFIX = RAVANA-TRA-v1        COMPONENT = TRA
K_LEVELS    = (0, 1, 2, 4, 8)      CONDITIONS = (near, far, foil, negative)
ARMS        = (structural, naive, chance, counterexample, permuted)
families    = 400  (primary 200 / sealed 200)
pool_k_counts = 40 families per k per pool for every k in K_LEVELS
hash rule   = sha256("RAVANA-TRA-v1|TRA|<difficulty>|<item_id>")
```

| Artifact | SHA-256 |
|---|---|
| manifest, **primary** (canonical JSON of all 200 records) | `711e50a489fe5fa5f375b2267a3944d1f5b06741713f34cacafc9a4abde43d76` |
| manifest, **sealed** (canonical JSON of all 200 records) | `405fdcbd99d7904304fc176acd35ee5452b051d8791ecbf862a6588d4bf6acd5` |
| whole bank (`item_manifest_hash`) | `03537712fec70e7f3bd6226d7e97ee636b2dd99cac47d197eba3f0620500c512` |

All three are asserted byte-for-byte by `tests/unit/test_measure_tra.py`, as is
the entire `TH` threshold dictionary (`TH_SNAPSHOT`). `PYTHONHASHSEED=0` is
recommended for the run; the bank itself uses no `hash()` and no RNG, so the
hashes above are process-independent.

---

## 2. Implemented vs spec, item by item

### §1 Item schema — implemented
All 400 items carry `item_id` (`tra_%04d`), `role` (six-role enum), `type_label`
(domain-general slot, shared by demo targets and gold, identical across
`domain`/`far_domain`), `difficulty`, `domain`, `far_domain` (`+3`, asserted
`!=`), `subject`, 16 `taught_exemplars` (`{r1,r2,surface,type_label}`, first 8
teachable), `unseen_test_words.near/far` (gold + 4 candidates), `foil`,
`negative`, `type_sets`, `world_model` and `pool`.

Validation implements all six spec rules: unique ids / enum / disjoint domains
and pools; gold in candidates and in `world_model`; query subject and gold never
in any taught-exemplar object, gold carries the taught type and at least one
candidate has a disjoint type set; foil has no gold target, contains a
demonstrated anchor and no candidate is a world-model target; negative target
exists in the world model and in candidates; 16 exemplars and pairwise-disjoint
24-word pools. `validate_items()` returns `[]` for the frozen bank.

**Provenance-safe teaching text.** `teaching_surface()` emits only exemplar
tokens plus the domain-general type label, e.g.
`cloud causes rainbow. rainbow is an effect.` It names the slot at write time
(P2c) and never contains `world_model`, `type_sets`, gold words, `domain` or
`far_domain`. The graded query displays the four options but never annotates
which one is gold.

### §2 Phase structure — implemented
P0 freeze (bank + validation + manifests), P1 cold baseline query, P2/P2c
teaching (`k` turns; k=0 receives eight matched neutral turns), P3 **one graded
query turn** parsed into `{role,target,verdict}`, P4 paraphrase immediate probe,
P5 `eng.save()` + engine object deletion + fresh cold boot on the isolated
scratch `data_dir`, P6 held-out paraphrase on the fresh engine. Demo reactions
stay in the raw observation for audit and never supply graded tokens.
Unparseable → `verdict=no_answer` and `engine_response_unparsed += 1`.
`api_calls` counts `boot,intake,save,cold_reload,process_turn`; exceptions are
surfaced into `backend_error`, never swallowed.

### §3 C1-C13 — implemented, exact criteria (nothing changed)

| ID | Exact criterion as implemented |
|---|---|
| C1 | structural `near(k=8) >= 0.90` |
| C2 | structural foil reject `>= 0.90` **and** naive foil correct `<= 0.10` |
| C3 | naive `near − permuted >= 0.15` **and** structural `|near − permuted| <= 0.10` |
| C4 | `chance <= 0.35` **and** `far − chance >= 0.20` |
| C5 | structural negative reject `>= 0.85` **and** counterexample reject `<= 0.20` |
| C6 | max adjacent drop across k `<= 0.05`, both pools |
| C7 | graded `near >= far >= foil`, max drop `<= 0.15`, both pools |
| C8 | `role_coverage >= 0.95` both pools **and** grader fixed vectors: `good` true; `wrong_role`, `wrong_target`, `foil_accept`, `negative_accept` false; `foil_reject`, `negative_reject` true |
| C9 | `delayed_retention >= 0.80` at k=8 both pools; every delayed row `reloaded>=1` and `slot_read_fresh`; `save>=1`, `cold_reload>=1`; fresh-engine read |
| C10 | typed option picked over disjoint-typed option, flip of taught slot flips the pick, disjoint options return `None` (fail-closed) |
| C11 | `backend=="live"`, `live==true`, `backend_error is None`, all five API counters `>=1`, `engine_observation_missing==0`, `synthetic_fallback_used==0`, graded tokens from the query response only |
| C12 | taught views carry `edge_kind`/`source_type`, no fabricated URL/span, no arm disables the trust gate |
| C13 | `immediate(k=8) − baseline >= 0.15` and coverage loss `<= 0.05` |

Failure of C1-C10, C12 or C13 prints `INSTRUMENT INVALID — no score` and exits
**2**. C11 is a **capability** gate: the fake arm must fail it (spec §7 step 2,
"prove the fake arm cannot emit a graded answer"), so its failure does not make
the fake mechanism check an instrument error. A fake run is reported as
`mechanism_probe_only_fake_arm`, never as capability.

**Saturation.** The unchanged shared `front_gates.gate_saturation(...)` is fed
`{"tra_near": 100*near, "tra_negative_reject": 100*negative_reject}`. A FLAG is
non-blocking only if `mutation_control()` shows both M-A (taught type re-stamped
in a scratch copy) and M-B (gold's type membership removed) move the graded
verdict; otherwise exit 2 and no number is reported. `mutation_control()` takes
an optional `bank_mutator(item, kind)` hook for re-stamping a **real** scratch
store on the live arm; the default works on deep copies without an engine.

### §4 Split / counts / one-k-per-item — implemented
Hash split within difficulty, 50/50 per difficulty → 200/200. Within each pool,
sort by the same hash and cycle `(0,1,2,4,8)` → exactly 40 families per k per
pool. Every family is assigned exactly one k. Split and manifests are frozen
before any query.

### §5 Endpoints and PASS/FAIL — implemented
Primary endpoint `TRA_typed_transfer` = mean correct at k=8 pooled over unseen
near+far. Secondaries: near, far, foil_reject, negative_reject, permuted,
chance, monotonic/graded drops, role_coverage, delayed_retention. `decide()`
evaluates every PASS rule in **both** pools plus sealed-within-0.10, all
controls, live backend, error-free run and saturation mutations. Wilson 95% CIs
are computed per pool. Exit codes: **0** PASS, **1** FAIL/genuine (including a
fake mechanism-only run), **2** INVALID.

### §6-§7 — implemented
LEARN-5 interpretation matrix is encoded in `interpretation`
(`capability_pass` / `mechanism_probe_only_fake_arm` / `genuine_fail`). Fake arm
is the default; `RealBackend` is lazy-imported only by `--live`; importing the
module boots nothing (asserted in a subprocess unit test). The default-OFF
`RAVANA_TRA_SLOTS` primitive is enabled **only** around the live arm via
`tra_slots_flag()` and restored in `finally`, including the unset case; the fake
arm leaves the environment byte-identical.

---

## 3. Verification actually performed (static + fake only)

```
PYTHONPATH=<worktree>/scripts;<worktree>/ravana/src;<worktree>/ravana_ml/src;.
<main>/.venv/Scripts/python.exe -m pytest tests/unit/test_measure_tra.py -q
  -> 31 passed in 2.88s
<same process> + tests/unit/test_tra_slots.py + tests/unit/test_schema_induction.py
  -> 55 passed in 7.52s
```

Module resolution was asserted: `measure_tra.__file__` and `tra_items.__file__`
resolve under the worktree; `ravana.chat.schema_induction.__file__` resolves
under `<worktree>/ravana/src/...`, not the main editable install.

Fake sweep outcome (recorded, not a capability claim): all C1-C10/C12/C13 PASS,
C11 BAD by construction; primary and sealed `transfer=1.000`, `near=far=0.800`,
`foil=neg=1.000`, `chance=0.000`, `mono=graded=0.000`, `retention=1.000`,
`growth=+1.000`, `control_delta=0.000`; G6 FLAG with M-A/M-B moved; exit 1,
`interpretation=mechanism_probe_only_fake_arm`.

No engine boot, no full benchmark run, no production weight store write and no
network call occurred in this worktree.

---

## 4. Parent's run commands (fake, then live)

Run from the worktree, using the **main repo venv** with PYTHONPATH pointed at
the worktree:

```bash
cd C:/Users/Likhith/Documents/projects/ravana_tra_harness
export PYTHONPATH="C:/Users/Likhith/Documents/projects/ravana_tra_harness/scripts;C:/Users/Likhith/Documents/projects/ravana_tra_harness/ravana/src;C:/Users/Likhith/Documents/projects/ravana_tra_harness/ravana_ml/src;."
PY=C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe

# 0. freeze check (manifest + thresholds): expect exit 0
PYTHONHASHSEED=0 RAVANA_OFFLINE=1 "$PY" scripts/measure_tra.py --validate-only

# 1. FAKE mechanism check: expect exit 1, interpretation=mechanism_probe_only_fake_arm
PYTHONHASHSEED=0 RAVANA_OFFLINE=1 "$PY" scripts/measure_tra.py

# 2. LIVE engine on an isolated scratch store (harness sets/restores RAVANA_TRA_SLOTS=1)
PYTHONHASHSEED=0 RAVANA_OFFLINE=1 "$PY" scripts/measure_tra.py --live
```

Before interpreting step 2, read `backend`, `live`, `api_calls`,
`backend_error`, `engine_observation_missing`, `engine_graded` and confirm C11
is ok. If a boundary front is saturated (G6 FLAG), run M-A/M-B on a **fresh
scratch copy** through `mutation_control(items, bank_mutator=...)` before citing
any number. A fake-arm PASS is never a capability claim.

---

## 5. Deviations / items not fully implemented (honest)

1. **No live execution.** `RealBackend` (boot, teaching via `process_turn`,
   `save()` + cold reload, `typed_slot_views_from_graph`, API counters) is
   implemented but was **not exercised** here — the task explicitly forbids
   engine boot and the full benchmark run. Parent owns live verification.
2. **Live mutation hook is an interface, not a store stamper.** The default
   `mutation_control()` mutates deep copies of the item bank. Re-stamping a real
   stored `source_metadata` edge (M-A) requires the parent to supply
   `bank_mutator`; no real-store mutation code is shipped in this harness.
3. **C10 mirror.** The type-contrast control uses a pure local function with
   the same fail-closed contract as the merged production
   `slot_keyed_forced_choice`, so unit tests need no engine import. It does not
   call the live primitive on a live graph.
4. **Distractor same-length is best-effort.** Distractors prefer the gold's
   character length, but the six fixed 24-word pools cannot guarantee three
   equal-length words for every family; the hard constraints (distinct,
   same-domain, disjoint type set) are enforced and validated.
5. **`permuted` is an arm plus a `naive` permuted probe row**, not a fifth
   condition. It exercises the same C3 contrast (naive breaks under
   permutation, structural does not) without adding a fifth k-sweep condition.
6. **P4/P6 are run for the structural arm** per condition (P4) and near/far
   (P6). Spec says one paraphrase query per item; extra arms would multiply
   engine turns without adding a pre-registered endpoint.
