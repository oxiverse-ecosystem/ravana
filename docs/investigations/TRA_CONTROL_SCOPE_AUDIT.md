# TRA control-scope and provenance-seam audit

**Scope of this audit.** Static reads plus **fake-arm-only** execution in worktree
`C:/Users/Likhith/Documents/projects/ravana_tra_controls` (branch
`docs/tra-control-audit`, from `main` `f86e470`). No engine boot, no
`process_turn`, no store write, no change to any threshold, gate, item,
`scripts/measure_tra.py` or `ravana/src`. Live numbers quoted below are read
from pre-existing artifacts in the main worktree:
`scratch/tra_live_focused2.log`, `scratch/tra_one_item_diag.log`,
`scratch/tra_fake_arm.log`, `scratch/tra_focused_fake.log`.

**Headline.** The focused live run did not measure capability: it exited 2
`INSTRUMENT INVALID` because capability endpoints were wired into the
instrument-gate path. The three root causes are (1) **control-scope inversion**,
(2) **the intended unseen-subject capability gap, misread as a construction
defect**, and (3) **a provenance seam that returns a cache-shaped, unscoped
edge census instead of the taught-edge observations C12 is defined to check**.

---

## 0. Evidence base (reproduced here, not assumed)

| ID | What was done | Result |
|---|---|---|
| E1 | `PYTHONHASHSEED=0 python scripts/measure_tra.py` in this worktree (full 400-family bank, `FakeBackend`, no boot) | C1–C10, C12, C13 all `ok`; only C11 `BAD` (by construction). `invalid_controls=[]`, `commit=f86e470`. |
| E2 | Static, no-boot probe importing `measure_tra` and `schema_induction` (`numpy` from the repo `.venv`) | `_provenance_ok([trusted])=True`; `_provenance_ok([trusted, seeded])=False`; chatter -> `no_answer` but `role_present=True`; production slot scorer returns `pick=None` for TRA word options even for a **taught** subject, and picks correctly only when options are type labels. |
| E3 | `scratch/tra_live_focused2.log:49702-49727` | focused live (`max_per_pool=20`, 40 families, `RAVANA_TRA_SLOTS=1`): C1 near8=0.0, C2 0.1/0.4, C3 gap 0/0, C4 chance 0 / far−chance 0, C5 0.45/0.55, C6 0.2/0.3334, C9 retention=None/None, C12 `views=1599 fabricated=False`, C13 growth=0, C7/C8/C10/C11 ok, then `INSTRUMENT INVALID — no score (failed: C1, C2, C3, C4, C5, C6, C9, C12, C13)`, exit 2. |
| E4 | `scratch/tra_one_item_diag.log` + `scratch/tra_one_item_diag.py:11-27` | item `tra_0003`, role SIMILAR, subject `dew`; teaching surfaces `cloud resembles storm. storm is a similar-form.` ×8; query `what is true of dew? Options: sunset, squall, breeze, shower.`; answer chatter; `views 0 []`; `calls {'boot':1,'intake':8,'save':0,'cold_reload':0,'process_turn':9}`; `typed_views_prod 565`; `hook None`. |

---

## 1. Are C1–C6 / C9 / C12 / C13 instrument controls or mis-scoped capability gates?

### 1.1 EVIDENCE — what the instrument actually wires

- `scripts/measure_tra.py:721-845` defines **one** `instrument_controls()` that
  computes C1–C13 from the *same* `rows`/`summary` produced by whichever
  backend ran `run_sweep()` (`scripts/measure_tra.py:519-546`).
- `scripts/measure_tra.py:1035-1039` calls that function once with the live
  backend; `scripts/measure_tra.py:1039` builds
  `invalid = [c for c in checks if c["id"] != "C11" and not c["ok"]]`;
  `scripts/measure_tra.py:1047-1050` prints `INSTRUMENT INVALID — no score` and
  returns exit 2 on **any** such failure — before saturation (`:1052`), before
  `decide()` (`:1066`), before any row is written (`:1069`) and before any
  endpoint is printed (`:1080-1096`).
- `scripts/measure_tra.py:919-920` makes `instrument_controls_ok` part of the
  overall PASS conjunction; `:921-922` alone isolates C11. The source comment at
  `scripts/measure_tra.py:1037-1038` explicitly says C11 is the capability
  gate and C1–C10/C12/C13 are instrument gates.
- The criteria themselves are capability thresholds, not instrument-shape
  checks: C1 `near(k=8)>=0.90` (`:730-732`), C2 foil reject / naive accept
  (`:733-739`), C3 permutation power (`:740-747`), C4 chance/far (`:748-753`),
  C5 negative power (`:754-760`), C6 monotonic k-curve (`:761-766`), C9 cold
  retention (`:798-812`), C13 `immediate−baseline>=0.15` (`:839-844`).
  These are the *same quantities* re-declared as PASS rules in `decide()`
  (`:890-918`, notably `:911-915` for retention and control deltas). A miss is
  therefore counted twice: once as a FAIL rule and once as an INVALID control.
- The spec says C1–C10 are certified on the oracle arm:
  `docs/investigations/TRA_BENCHMARK_SPEC.md:300` — "Run the fake sweep: all
  C1–C10 must pass on the oracle"; `:301` puts `RealBackend` after that step.
  E1 confirms it: on the full fake arm every control but C11 passes.
- Comparator harnesses keep the arms separate by dependency injection and
  validate the instrument on the fake/fixture side:
  `scripts/measure_learn5_rsi.py:24-31` (FakeBackend "default … used by unit
  tests and the default main()", RealBackend only under `--live`),
  `scripts/measure_learn5_pdr.py:23-39` (same convention; `:41-46` says the
  real arm censuses the live store and **never** returns inherited synthetic
  observations), `scripts/measure_learn5_wfa.py:997-1036` (controls are
  known-good/known-bad vectors and static probes, not thresholds on the live
  capability endpoints; responders are built from a separate `source` at
  `:2413-2420`). RSI also deliberately declines to gate on growth
  (`measure_learn5_rsi.py:21-22`), unlike TRA C13.
- Focused-subset artefact: `scratch/tra_focused_fake.log:13-31` shows the *fake*
  arm itself throwing C1/C9/C13 when the bank is reduced. The fake
  certification must therefore use the full frozen 400-family bank (spec
  `:210-221`), not a matched small subset.

### 1.2 EVIDENCE — what passes vacuously on the live arm

`grade()` at `scripts/measure_tra.py:500` accepts `role is None` as
`role_present`, and `_parse_engine_answer()` (`:170-247`) returns `None` for
chatter. E2 shows the live substitution at `:410-413` yields
`{"role": None, "verdict": "no_answer"}` with `role_present=True` and
`correct=False`. With all endpoints at zero, C7 (`:767-772`) computes a graded
drop of 0 and C8 (`:788-797`) computes coverage 1.0. So C7/C8 "ok" on the live
arm (`scratch/tra_live_focused2.log:49715-49716`) is **not** evidence that the
live observation path is healthy; it is the all-`no_answer` floor. C10 (`:685-697`)
is a static synthetic probe and C11 (`:817-831`) is the genuine live-arm gate.

### 1.3 INFERENCE — classification

| Control | Correct classification | Where it must be evaluated |
|---|---|---|
| C1–C6 | Capability thresholds wearing an instrument label. On the fake arm they certify the oracle's k-curve; on live they are reported endpoints. | Certify on fake; **report on live, never INVALID on live** |
| C9 (retention), C13 (growth) | Capability endpoints (durability / change) already re-declared in `decide()` at `:911-915`; duplicating them as INVALID gates converts a genuine FAIL into exit 2. | Certify the counters/save-reload shape on fake; **report retention/growth on live** |
| C12 (provenance integrity) | A **genuine instrument-integrity control**: it must stay on the live arm, because it is the only check of the observation seam. Its current failure is a seam defect (§3), not a capability result. | Fix the seam input; keep the gate live |
| C7, C8 | Instrument-shape checks, but blind to the all-`no_answer` floor — read beside printed answers (measurement-validity §"print the probes with their outputs") | Fake certification; live only as diagnostic |
| C10, C11 | C10 static; C11 the real anti-fake capability gate. Correctly placed today. | C10 static; C11 live |

**Conclusion (INFERENCE, from §1.1–1.2):** C1–C6, C9 and C13 are **mis-scoped
capability gates**. The focused live run should have been a scored `FAIL`, not
`INSTRUMENT INVALID`. C12 is correctly an instrument gate but is fed a broken
observation. The mis-scoping is in the *wiring* (`main()`, `decide()`), not in
any threshold or criterion.

### 1.4 Minimal re-scoping (thresholds and gate definitions unchanged)

No edit to `TH` (`scripts/measure_tra.py:74-100`), to any criterion in
`instrument_controls()` (`:721-845`), to `decide()` (`:884-928`), to the item
bank, or to `ravana/src`. Only the **arm supplying each observation** changes,
in `main()`:

1. At `:1017-1022`, when `live`, run the default `FakeBackend` sweep over the
   **full** 400-family bank for instrument certification
   (`fake_rows`, `fake_summary`), and keep the existing `RealBackend` sweep
   (`rows`, `summary`) for capability. The fake arm boots nothing and is
   byte-identical to the CI default (`:9-12, :29`).
2. At `:1035`, call `instrument_controls()` once with the fake arm
   (`fake_rows`, `fake_summary`, `fake_backend`) to certify C1–C10/C12/C13;
   its C11 is `BAD` by construction and is already excluded at `:919-920` and
   `:1039`. Take C11 (`:817-831`) from a call with the live backend (or factor
   that block into a helper) and combine:
   `checks = [c for c in fake_checks if c["id"] != "C11"] + [live_c11]`.
3. Point `invalid` and the `INSTRUMENT INVALID` branch (`:1039`, `:1047-1050`)
   at the fake-certified checks. INVALID then means "the oracle-shaped
   instrument is broken", never "the engine failed a capability threshold".
4. Leave `decide()` untouched (`:884-928`): `instrument_controls_ok` now comes
   from the fake arm, `c11_anti_fake_ok` from the live arm, and every endpoint
   rule from the live `summary`. Live misses flow to `passed` (`:928`),
   `interpretation="genuine_fail"` (`:1073-1075`) and exit 1 (`:1128`).
5. Keep C12 live and repair its seam per §3 — waiving it to fake would hide
   the only live observation-seam check.

---

## 2. "No typed edge for unseen subject" — construction defect or intended capability gap?

### 2.1 EVIDENCE — the ban on teaching the query subject's type

- `TRA_BENCHMARK_SPEC.md:47`: `subject` is an **undemonstrated anchor word**,
  "Never appears in any taught exemplar"; validation rule `:101` re-checks
  that the query subject and gold are not in any taught-exemplar object set.
- `TRA_BENCHMARK_SPEC.md:53`: `type_sets` are an authored ontology used to
  construct items and as the **ORACLE ceiling only**, "never sent to the engine
  at query time"; `:54` says the same for `world_model` ("Leakage if exposed to
  the graded engine path"). The harness docstring repeats both at
  `scripts/measure_tra.py:23-26`; `:125-142` builds every query surface from
  only the subject + candidate list.
- P2c (`TRA_BENCHMARK_SPEC.md:124`) teaches the slot **inside the same k
  turns**, with an exemplar surface that names the slot ("mist is an effect of
  drizzle"). `scripts/measure_tra.py:149-166` implements exactly this:
  `teaching_surfaces()` returns `ex["surface"]` for each exemplar; the slot is
  asserted on the **r2 token** of a demonstration, never on the query subject.
- `scratch/tra_one_item_diag.py:15-20` and E4 confirm the implementation: the
  eight surfaces assert `storm/frost/… is a similar-form`, while the query
  subject is `dew`. The production writer stamps exactly that shape:
  `ravana/src/ravana/chat/chain_walker.py:1008-1022` parses
  `"storm is a similar-form."` into `(IS_A, "storm", "similar-form", …)` and
  `:1204-1237` writes a provenanced `is_a` edge with `type_label` stamped on
  the edge.
- C11 anti-leakage (`TRA_BENCHMARK_SPEC.md:166`; `scripts/measure_tra.py:817-831`,
  `:170-247`, `:476-487`) requires graded tokens to come from the query
  response, never demo triples, `world_model` or `type_sets`.
- The capability under test is stated at `TRA_BENCHMARK_SPEC.md:11-22` and
  `:275-277`: a **typed, domain-general role slot shared by demonstrated and
  unseen words**; transfer to unseen words is the measured capability.

### 2.2 EVIDENCE — why the production hook still returns None

`ravana/src/ravana/chat/response_gen.py:7683-7705` resolves a subject from the
forced-choice prompt and calls `slot_keyed_forced_choice`. That scorer
(`ravana/src/ravana/chat/schema_induction.py:226-273`) requires, for a pick:
`e.source == subject` (`:241`), `type_label == e.target` (`:244`), and
`slot in options` (`:246`). E4's `hook None` is the first condition: no stored
edge has `source == "dew"`. E2 adds a second, independent condition: with a
real trusted view `storm → similar-form` (type label `similar-form`), the
production scorer returns `pick=None` for TRA's option list
`["sunset","squall","breeze","shower"]` (`no_trusted_typed_slot_above_floor`),
and returns `"similar-form"` only when options are **type labels**
`["similar-form","effect"]`. The harness's mirrored C10 probe uses exactly
type-label options (`scripts/measure_tra.py:685-697`), so C10 cannot see this
contract mismatch.

### 2.3 INFERENCE — verdict

**Not a construction defect.** Adding a turn that names `dew`'s type would
violate `TRA_BENCHMARK_SPEC.md:47/:101`, convert transfer into recall of a
directly taught subject, and change the item/gate definition after results —
explicitly forbidden by `:170-172` and `:314`. The absent typed edge for the
unseen subject is the **genuine capability gap the spec sets out to measure**.
It is correct fail-closed behaviour by the production hook, not an INVALID
instrument.

The option-space mismatch in §2.2 is a **separate mechanism finding**
(INFERENCE, not a reason to change the run's verdict): under the current
prompt contract the hook cannot emit a domain-word target even for a subject
that *was* taught. That belongs in the capability-gap list, not in the
instrument-validity list. Whether the benchmark *should* pose type-label
options instead is a design change and is out of scope here.

---

## 3. Why `RealBackend.provenance_views()` returns 0 while the graph has 565 views

### 3.1 EVIDENCE — exact cause

- `scripts/measure_tra.py:444-445`: `provenance_views()` is a pure getter over
  `self.provenance_views_cache`. It never reads the engine.
- `scripts/measure_tra.py:273` (inherited `FakeBackend.__init__`) initialises
  that cache to `[]`. In `RealBackend` it is assigned in **one place only**:
  `scripts/measure_tra.py:433-440`, inside `save_reload()`, from
  `typed_slot_views_from_graph(fresh.graph)` after `eng.save()` and a cold
  reload. `teach()` (`:379-393`) and `query()` (`:395-416`) never populate it.
- `scratch/tra_one_item_diag.py:22` calls `b.provenance_views()` immediately
  after teach+query and **never calls `save_reload()`**; E4's counters
  (`save=0, cold_reload=0`) confirm it. The getter therefore returns the
  initial `[]` — `views 0 []`. The same script at `:26` calls
  `si.typed_slot_views_from_graph(eng.graph)` directly on the still-booted
  engine and gets `565`. **"0 vs 565" is a cache-population/timing artefact,
  not evidence that the graph lacks typed edges.**
- The 565 is also not 565 *typed slot* edges:
  `ravana/src/ravana/chat/schema_induction.py:184-205` appends a view for
  **every** `graph.edges` entry, reading provenance from `source_metadata`
  (`:187-202`). The boot log in E4 shows seeded PMI/GloVe/CommonFacts/Physics
  edges, so the census is the whole store.
- In the full live run `save_reload()` did run (C9 detail: `save=40 reload=40`,
  E3), so C12 sees `views=1599`. It still fails because
  `_provenance_ok()` (`scripts/measure_tra.py:700-710`) requires **every** view
  to have an allowed `edge_kind`, `source_type == "user"`, and no URL/span.
  E2 proves one non-user seeded view is sufficient to return `False`.

### 3.2 INFERENCE — minimal fix, no fabricated observations

1. Populate the cache at the observation point, not only after a cold reload:
   in `RealBackend.query()`/`teach()` (`scripts/measure_tra.py:379-416`) read
   `typed_slot_views_from_graph(self._engine.graph)` into the cache, and keep
   the fresh-engine census at `:433-440`. Then `provenance_views()` at `:444`
   returns real views even for a teach→query probe. (Mirror PDR's documented
   rule, `measure_learn5_pdr.py:41-46`: the real arm censuses the actual store
   and never falls back to the inherited synthetic list,
   `measure_tra.py:318,449-454`.)
2. Scope the returned census to the **taught demo edges** C12 is defined to
   check (`TRA_BENCHMARK_SPEC.md:167`): filter the real views by the item's
   taught r1/r2 tokens and `type_label`, using fields the adapter already
   exposes (`source`, `target`, `type_label`, `edge_kind`, `source_type`;
   `schema_induction.py:192-204`). Keep `_provenance_ok()` (`:700-710`),
   `TRUSTED_KINDS` (`:101`) and `TH` byte-identical — only the input list is
   corrected. This is filtering real observations, not fabricating them.
3. Call `save_reload()` before any provenance assertion in diagnostics
   (`scratch/tra_one_item_diag.py:22`), or expose the refresh seam above.

**Unverified (do not overstate):** the focused log prints `views=1599` but not
which view broke `_provenance_ok`; the identity of the specific non-user edge
is inferred from the seed lines in E4 and the all-views predicate, not read
back from the store in this audit.

---

## 4. Pre-registered output of a corrected live run

With §1.4 wiring and §3 seam repair in place, and **no threshold or gate
definition changed**, a live run in which the capability is absent must print:

- Header: `backend=live`, `commit=f86e470` (or the SHA at run time), full
  400-family bank / both pools, or the explicit
  `FOCUSED SUBSET … mechanism probe, NOT a full-benchmark verdict` banner from
  `scripts/measure_tra.py:971-985`; `RAVANA_TRA_SLOTS=1` enabled only inside
  `tra_slots_flag()` (`:105-121, :1019-1020`).
- Instrument controls: C1–C10, C12, C13 **ok on the fake/oracle arm** (full
  bank; E1); C11 **ok on the live arm** (`backend=live`, all five API counters
  ≥1, `engine_observation_missing==0`, `synthetic_fallback_used==0`,
  `graded_from_query=True`). No `INSTRUMENT INVALID` line from a low endpoint.
- Decision rules printed (`scripts/measure_tra.py:1077-1079`) with the
  capability rules FAIL: `transfer`, `near`, `far`, `far−chance`,
  `foil_reject`, `negative_reject`, `permuted`, retention/growth and the
  primary-vs-sealed checks at `:890-918`. Expected shape from E3:
  `near≈0`, `far≈0`, `transfer≈0`, `foil_reject≈0–0.1`,
  `negative_reject≈0–0.45`, `chance≈0`, `far−chance≈0`, `permuted≈near`,
  `monotonic_max_drop>0.05`, `growth≈0`, `delayed_retention=None` because
  `immediate=0` makes the ratio at `:629-630` undefined.
- Both pool lines (`:1080-1090`) with all endpoints, plus Wilson 95% CIs from
  `statistics()` (`:941-955`), and every raw query answer beside its verdict
  — the all-`no_answer` floor must be visible, not hidden behind
  `role_coverage=1.0` (§1.2).
- Saturation: `front_gates.gate_saturation()` (`:1052-1056`) fed
  `tra_near = 100*near`, `tra_negative_reject = 100*negative_reject`. With
  both at the low band (`SATURATION_LOW=5`, `scripts/front_gates.py:24-25,
  435-448`) expect `G6 FLAG — 2 front(s) non-informative: tra_near=0,
  tra_negative_reject=0`; with `negative_reject≈0.45` expect a one-front FLAG.
  G6 FLAG is **non-blocking** (`front_gates.py:446,452-454`; spec `:176-193`),
  so FLAG is not INVALID. Mutation control M-A/M-B (`:856-880`) must move on
  the scratch bank; if a FLAG coincides with disconnected mutations, that
  **is** `INSTRUMENT INVALID` (`:1060-1064`) — a different, legitimate reason.
- `interpretation: genuine_fail` (`:1073-1075`), `OVERALL: FAIL`, **exit 1**
  (`:1125-1128`). The strings `INSTRUMENT INVALID — no score` and exit 2 are
  absent. Summary JSON records `"pass": false`, `"invalid_controls": []`,
  live `api_calls`, `backend_error: null`.
- Do **not** loosen a threshold, gate, item or gold to recover a nonzero
  number (`TRA_BENCHMARK_SPEC.md:170-172, :260-264, :314`). A zero that
  survives an honest instrument is the finding: the typed, domain-general
  role slot is absent.

---

## 5. Three root causes (for the parent report)

1. **Control-scope inversion** — `scripts/measure_tra.py:721-845, 1035-1050,
   919-920` feed live capability endpoints (C1–C6, C9, C13) into the
   instrument-gate path, so a genuine capability failure prints
   `INSTRUMENT INVALID`/exit 2 instead of a scored `FAIL`/exit 1. Fix is wiring
   only (§1.4); all definitions stay frozen.
2. **Unseen-subject type absence is the intended capability gap, not a
   construction defect** — `TRA_BENCHMARK_SPEC.md:47,53,124,166,275-277` and
   `scripts/measure_tra.py:149-166` forbid teaching the query subject's type;
   `ravana/src/ravana/chat/schema_induction.py:241,244,246` also requires the
   option space to be type labels. Teaching `dew`'s type would leak the answer
   and change the benchmark after results.
3. **Provenance seam** — `scripts/measure_tra.py:273,433-445,700-710` caches
   views only inside `save_reload()` and checks every graph edge, so a
   teach→query probe reads `0` while a direct census reads `565`, and the full
   run's `1599`-view unscoped census fails on boot-seeded non-user edges.
   Fix the seam input (populate at observation, scope to taught demo edges);
   do not fabricate observations.
