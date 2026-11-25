# TRA — pre-registered specification for the Typed Role Acquisition benchmark

**Status: pre-registration / design only.** No engine was booted, no `process_turn`
called, no pytest run, no gate, threshold, item bank or harness edited, and no
file under `ravana/src/`, `ravana_ml/src/`, `scripts/` or `tests/` changed. The
only artifact is this document, written into worktree
`C:/Users/Likhith/Documents/projects/ravana_tra_bench` on branch
`docs/tra-benchmark` from main `aa13c31`. Everything below is a *commitment*
about a future run; no result is claimed.

**Why a new benchmark.** Experiments 1 and 2 (`RSI_ROLE_BINDING_EXPERIMENT.md`,
`RSI_SYMBOLIC_ROLE_EXPERIMENT.md`) and the static audit
(`RSI_TYPED_ROLE_DESIGN.md`) converge on one finding: RSI cannot be fixed by a
better ranker. Gold candidates are deliberately *unseen* demonstration tokens,
`relation_type` is constant within a family, and the measured store carries
0.9% typed edges with all 2,167 nodes `concrete` — the RSI hypothesis path
reads neither type nor provenance. The missing mechanism is a **typed,
domain-general role slot** shared by demonstrated and unseen words, admitted
despite unseen token identity (`RSI_SYMBOLIC_ROLE_EXPERIMENT.md:171-188`,
`RSI_TYPED_ROLE_DESIGN.md:58-63, D2/D4`). TRA is the instrument that measures
*acquisition of that slot*, not a re-scoring of RSI. It is a LEARN-5 component
candidate, governed by `docs/BRAIN_LEARN5_DESIGN.md` §2–§7.

**Naming and collision rule.** `HASH_PREFIX = "RAVANA-TRA-v1"`; `COMPONENT =
"TRA"`. This is a **new** constant string, so TRA pools cannot collide with the
existing RAVANA-7 v1 or LEARN-5 v1 (CAR/WFA/PDR/CLI/RSI) pools. All hashes,
splits and manifests are frozen before any engine boot.

---

## 1. Item schema

Each item is one **role family**: a domain-general role slot `role`, taught
through `k` typed exemplars, tested on *unseen* words. The bank carries the
type information the engine currently lacks; **the type label is legitimate
teaching material only when delivered through a provenance-stamped demo turn**,
and is never present at query time as test-set metadata (audit §2b).

| Field | Type | Meaning / constraint |
|---|---|---|
| `item_id` | str | `tra_%04d`; unique. |
| `role` | enum | One of `EFFECT, WHOLE, PRECEDES, SIMILAR, SUPPORTS, CONTRAST` — the relation whose r2 slot is typed. |
| `type_label` | str | Domain-general slot label for the r2 role (e.g. `effect`, `structured-whole`, `earlier-event`, `similar-form`). Shared by demo targets AND gold; identical across `domain` and `far_domain`. |
| `difficulty` | enum | `easy, medium, hard, expert`; fixed before runs, used only for stratification. |
| `domain` | str | Demo teaching domain, from the six disjoint 24-word pools of `measure_learn5_rsi.py:137-169`. |
| `far_domain` | str | Deterministically `+3` domains from `domain`; asserted `!= domain`. |
| `subject` | str | Undemonstrated anchor word; the query subject. Never appears in any taught exemplar. |
| `taught_exemplars` | list[dict] | Exactly 16 r1/r2 pairs, of which the first `max(K_LEVELS)` are teachable. Each `{"r1","r2","surface","type_label"}`. The r2 words are the taught *typed* tokens; their type is stamped at write time. |
| `unseen_test_words` | dict | `{"near": {...}, "far": {...}}` — the held-out gold words, never demonstrated. `near` is drawn from `domain`, `far` from `far_domain`; both share `type_label`. |
| `candidates` | list[str] | Four distinct options for each of `near`/`far`: the gold + 3 distractors. Distractors are same-domain, same-length words; at most one shares the gold's type set. |
| `foil` | dict | `{"subject","candidates","gold_target": None}`: same subject, wrong role; includes one demonstrated anchor so a surface matcher wrongly accepts. Required verdict = `reject`. |
| `negative` | dict | `{"subject","candidates","counterexample_target","gold_target": None}`: demonstrations are negated ("r1 does not R r2"); the true target exists in the world model, but no role is licensed. Required verdict = `reject`. |
| `type_sets` | dict[str, list[str]] | Authored ontology membership per word (e.g. `ceremony: [effect, social-event]`, `rainbow: [effect, weather-event]`). Used to *construct* items and as the ORACLE ceiling only; never sent to the engine at query time. |
| `world_model` | dict | `{("role", subject): [target...]}` — the answer key. Consulted only by fake/chance/naive/counterexample oracles and the A2 ceiling. **Leakage if exposed to the graded engine path.** |
| `pool` | str | `primary` or `sealed`, assigned by the hash rule of §4. |

### Worked example (one family, `near` + `far`)

```json
{
  "item_id": "tra_0007",
  "role": "EFFECT",
  "type_label": "effect",
  "difficulty": "medium",
  "domain": "weather",
  "far_domain": "social",
  "subject": "thaw",
  "taught_exemplars": [
    {"r1": "cloud",   "r2": "rainbow",  "surface": "cloud causes rainbow.",   "type_label": "effect"},
    {"r1": "storm",   "r2": "blizzard", "surface": "storm causes blizzard.",  "type_label": "effect"},
    {"r1": "drizzle", "r2": "mist",     "surface": "drizzle causes mist.",    "type_label": "effect"},
    {"r1": "thunder", "r2": "tempest",  "surface": "thunder causes tempest.", "type_label": "effect"}
  ],
  "unseen_test_words": {
    "near": {"subject": "thaw", "gold": "glow",
             "candidates": ["glow", "chill", "dew", "sleet"], "domain": "weather"},
    "far":  {"subject": "role", "gold": "ceremony",
             "candidates": ["ceremony", "election", "contract", "habit"], "domain": "social"}
  },
  "foil":     {"subject": "thaw", "candidates": ["rainbow", "chill", "dew", "glow"], "gold_target": null},
  "negative": {"subject": "gale", "candidates": ["glow", "chill", "dew", "sleet"],
               "counterexample_target": "glow", "gold_target": null},
  "type_sets": {"rainbow": ["effect","weather-event"], "blizzard": ["effect","weather-event"],
                "glow": ["effect","weather-event"], "ceremony": ["effect","social-event"],
                "sleet": ["weather-event"], "chill": ["weather-event"]},
  "world_model": {"EFFECT|thaw": ["glow"], "EFFECT|role": ["ceremony"], "EFFECT|gale": ["glow"]},
  "pool": "primary"
}
```

Reading: the demo teaches that the r2 slot of `causes` is an **effect**; the
test requires admitting `glow` (near) and `ceremony` (far) as effects even
though neither token was ever demonstrated. `foil` must be rejected because the
relation is wrong (a surface matcher sees the demonstrated `rainbow`);
`negative` must be rejected because the exemplars were negated.

### Validation (`validate_items`, before any boot; failure ⇒ exit 2)

1. Unique ids; `role` in the enum; `domain != far_domain`; pools disjoint.
2. `gold in candidates` and `gold in world_model[(role, subject)]` for both near and far; candidates distinct.
3. Query subject and gold are **not** in any taught-exemplar object set; every gold's `type_label` is in its `type_sets` and equals the item's `type_label`; at least one candidate has a disjoint type set.
4. `foil.gold_target is None` and no foil candidate satisfies `(role, subject)` in the world model; foil includes a demonstrated anchor.
5. `negative.counterexample_target` is in the world model and in candidates; no negation licenses a rule.
6. `len(taught_exemplars) >= max(K_LEVELS)`; the 24-word domain pools remain pairwise disjoint.

---

## 2. Phase structure and which `process_turn` calls teach vs query

Backends follow the dependency-injection convention of
`measure_learn5_rsi.py:480-590` and `measure_learn5_car.py:364-520`:
`FakeBackend` (default, oracle, no boot/no save/no network) and `RealBackend`
(`--live`, lazy import of `CognitiveChatEngine`, isolated scratch `data_dir`,
never the production `weights/`). Importing the module never boots an engine.

Per item, per `k ∈ K_LEVELS = (0,1,2,4,8)`, per condition
`CONDITIONS = (near, far, foil, negative)` plus `permuted` and `chance` probes:

| Phase | Calls | Teaching or query | Graded? |
|---|---|---|---|
| **P0 freeze** | none | — | no. Build bank, `validate_items`, hash split, write manifest + pool hashes. |
| **P1 baseline** (cold, before teaching) | 1 `process_turn` per item with the same query surface | **query** (pre-test) | diagnostic; establishes `baseline` rate and detects ceiling items. |
| **P2 teaching** | exactly `k` `process_turn` calls, one per exemplar | **teaching** | no — demo reactions are audit-only, never supply tokens. |
| **P2c type teaching** | included in the same `k` calls: the surface assertion names the slot ("mist is an effect of drizzle"), so the type reaches the store with real `Provenance` at write time | **teaching** | no. |
| **P3 query** | exactly 1 `process_turn`: free of parser key tokens, e.g. `"Considering the examples above, what is true of {subject}? Options: {candidates}. Answer briefly."` | **query** | **yes — the only graded observation.** Parse `{role, target, verdict}` from this response only. |
| **P3f foil/negative query** | same single query turn with foil/negative subject & candidates | **query** | yes — required verdict `reject`. |
| **P4 immediate** | 1 paraphrase query per item (different surface, same intent) | **query** | diagnostic; reported beside delayed. |
| **P5 save + cold reload** | `eng.save()` to scratch `data_dir`; delete engine; construct fresh `CognitiveChatEngine(data_dir=...)` | — | no. |
| **P6 delayed** | 1 held-out paraphrase query on the cold engine | **query** | **confirmatory** endpoint. No `eng.save()` during probe. |

**k=0 control** receives the same subject names and the same number of neutral
turns ("what do you know about X?", "things happen for many reasons, don't
they?") as the k=8 arm — matched exposure, no type teaching. The `chance` arm
uses a hash-shuffled role label; the `counterexample` arm ignores negation. Both
are confound arms, never reported as capability.

**Parser discipline** (copied from `measure_learn5_rsi.py:378-475`): an explicit
`role=… target=… verdict=…` form is accepted, or a known role phrase + candidate
recovered from natural text. Hedges and bare acknowledgments are stripped from
the *evidence* (a decline often echoes "effect"; that is not an induced slot).
Unparseable ⇒ `verdict="no_answer"`, `engine_response_unparsed += 1`. Demo
triples are never substituted for a missing answer. `api_calls` counts
`boot, intake, save, cold_reload, process_turn`; any API exception is surfaced
into `backend_error`, never swallowed.

---

## 3. Gate list (copied/adapted; exact criteria)

All gates run **before any score is read**. Failure of C1–C10 prints
`INSTRUMENT INVALID — no score` and returns exit 2. Exit codes: 0 PASS,
1 FAIL (genuine), 2 INVALID.

| ID | Source | Exact criterion |
|---|---|---|
| **C1** | RSI C1 | primary structural `near(k=8) >= 0.90`. |
| **C2** | RSI C2 | foil: structural reject `>= 0.90` **and** naive arm accept `<= 0.10`. |
| **C3** | RSI C3 | permutation: naive `near − permuted >= 0.15` **and** structural `|near − permuted| <= 0.10`. |
| **C4** | RSI C4 | `chance <= 0.35` **and** `far − chance >= 0.20`. |
| **C5** | RSI C5 | negative: structural reject `>= 0.85` **and** counterexample arm reject `<= 0.20`. |
| **C6** | RSI C6 | monotonic k-curve: max adjacent drop `<= 0.05` across `k∈{0,1,2,4,8}`, both pools. |
| **C7** | RSI C7 | graded `near >= far >= foil`: max drop `<= 0.15` at every k, both pools. |
| **C8** | RSI C8 | `role_coverage >= 0.95` in both pools **and** grader fixed vectors: `good` true; `wrong_role`, `wrong_target`, `foil_accept`, `negative_accept` false; `foil_reject`, `negative_reject` true. |
| **C9** | PDR C7 + CAR cold boot | **cold-reload retention:** `durable >= 0.80` (delayed typed-transfer ≥0.80 of immediate at k=8), every row `reloaded >= 1`, `save >= 1`, `cold_reload >= 1`, and the slot is read from the **fresh** engine. |
| **C10** | experiment F7 / measurement-validity §"validate instrument" | **type-contrast sanity:** a synthetic control where the taught `type_label` truly separates two candidates must flip the pick (`near` candidate typed `effect` picked over disjoint-typed one); a disjoint control must return `None`. Fail ⇒ INSTRUMENT INVALID. |
| **C11** | PDR §1 / RSI observation seam | **anti-fake-backend:** `backend == "live"`, `live == true`, `backend_error is None`, `api_calls` has `boot,intake,save,cold_reload,process_turn >= 1`, `engine_observation_missing == 0`, no synthetic fallback when the census returns no edges, and graded tokens come from the **query** response (never demo triples, never `world_model`, never `type_sets`). A fake-arm PASS is never reported as capability. |
| **C12** | PDR C8 | **provenance integrity:** taught demo edges carry real `edge_kind/source_type/active` from `Provenance`; no fabricated URL/span; the trust gate is removed by no arm to obtain a number. |
| **C13** | PDR C1/C2/C4 | **learning precondition:** `k=0` baseline `>= 0.80` is *not* required for TRA (unseen words), but `immediate(k=8) − baseline >= 0.15` and coverage loss `<= 0.05` are — a flat curve is absence, not acquisition. |

`TH` is a byte-identical frozen snapshot asserted by a unit test (pattern
`tests/unit/test_learn5_rsi.py:353`), and is recorded in the summary. No
threshold, gate or item is edited after any result is seen.

### Saturation gate + mutation control

Use the unchanged shared `front_gates.gate_saturation(...)` (Tier-0 G6), fed
`{"tra_near": 100*near, "tra_negative_reject": 100*negative_reject}`. A
saturated `1.00/0.00` with provenance fields satisfied is **INVALID as a
progress signal** until the mutation control runs on a **scratch copy**:

- **M-A.** On one item, re-stamp a taught demo edge's real `source_metadata` to
  `{"source_type":"seed","edge_kind":"curated"}` (or flip its `type_label` in
  the scratch bank). Regrade: the graded observation **must** flip or its
  margin move.
- **M-B.** On another item, remove the gold's membership in `type_sets` in the
  scratch bank so its type no longer matches the taught slot. Regrade: the
  verdict **must** flip `match → reject`.

If both move: grader genuinely reads the cold-reloaded engine; record as
mechanism check, G6 FLAG, still invalid as progress. If either leaves the score
unchanged: the grader is disconnected (fake store / stale summary / wrong
census key) — the boundary numbers are fake-oracle outputs; **do not report
them**, fix the observation seam and re-run.

---

## 4. Primary/sealed split and item counts (resolving the design contradiction)

**The contradiction.** BRAIN_LEARN5 §2.2/§3.5 require (a) `n >= 40`
item-families per pool, (b) **one `k` per item** (so an item never appears at
two k levels — no within-item carryover), and (c) `>= 40` observations at
*each* k level. With `|K| = 5` these cannot all hold on a 40-family pool:
40 families each assigned one k yields only 8 items per k level. RSI violates
(b) by sweeping every item at every k; CAR satisfies (b) but lands at ~10 per
k level. **TRA resolves it by scaling the bank, not the design.**

**Resolution and counts.**

```
families_per_pool = 40 (min observations per k) × 5 (|K_LEVELS|) = 200
total_families    = 2 pools × 200 = 400
K_LEVELS          = (0, 1, 2, 4, 8)
```

- `build_items()` emits **400** families: 200 `primary`, 200 `sealed`.
- Split within `difficulty`: `h = sha256(f"RAVANA-TRA-v1|TRA|{difficulty}|{item_id}")`;
  sort by `h` bytes; first 50% → PRIMARY, rest → SEALED. Fixed before boot.
- Within each pool, assign each family to exactly one `k` by `sorted-by-hash,
  cycle (0,1,2,4,8)` → **exactly 40 families per k per pool**; each item seen at
  one k only. Baseline-stratify ceiling items across k (CAR pattern) without
  changing any gate.
- Record `manifest_sha256 = sha256(canonical_json(sealed_records, sort_keys=True,
  ensure_ascii=False))` per pool. The sealed pool is revealed **once**, in the
  reported run; if a sealed result changes a design decision it is relabelled
  PRIMARY and a new sealed pool is drawn by the same rule.
- Unit of analysis is the **family** (not the token/turn); a family is averaged
  before testing. Report Wilson 95% CIs; exact binomial/Fisher where base rate
  ≈ 0. Seed `PYTHONHASHSEED=0`; same-config reproducibility gate: two runs must
  give identical structural verdicts.

---

## 5. Endpoints and pre-registered PASS/FAIL rules

Primary endpoint (one per component, both pools):
`TRA_typed_transfer = mean_i[ correct(i) ]` at `k = 8`, pooled over unseen
`near` and `far`, where `correct = (verdict=="match" ∧ role==gold.role ∧
target==gold.target)`. Secondary endpoints: `near`, `far`, `foil_reject`,
`negative_reject`, `permuted`, `chance`, `monotonic_max_drop`, `graded_max_drop`,
`role_coverage`, `delayed_retention`.

| Endpoint | PASS (each pool) | HARD FAIL (any one) |
|---|---|---|
| `TRA_typed_transfer` (near+far, k=8) | `>= 0.75` | `< 0.75` even if `p<0.05` (effect-size gate) |
| `near` / `far` | `near >= 0.75`, `far >= 0.70` | `far <= chance + 0.10` |
| `far − chance` | `>= 0.20` | `<= 0.10` |
| `foil_reject` | `>= 0.85` | `< 0.70` (foil admitted) |
| `negative_reject` | `>= 0.80` | `< 0.70` |
| `permuted` | `>= near − 0.10` | `< near − 0.15` (order memorisation) |
| `monotonic_max_drop` | `<= 0.05` | `> 0.10` at any adjacent k |
| `graded_max_drop` (near≥far≥foil) | `<= 0.15` | `> 0.15` |
| `role_coverage` | `>= 0.95` | `< 0.90` |
| `delayed_retention` (cold / immediate) | `>= 0.80` | gain absent after cold reload (activation, not retention) |
| `k=0` / chance / counterexample arms | deltas `<= 0.05` | any control delta `> 0.05` |
| sealed vs primary | sealed within `0.10` on transfer, `near`, `foil_reject` | sealed `transfer < primary − 0.10` |
| validity | C1–C13 ok, `backend=="live"`, `backend_error is None` | any control BAD, exit 2, or boundary endpoints without passing M-A/M-B |

**Overall PASS** requires: exit 0, all C1–C13 ok, every PASS rule true in
**both** pools, sealed within 0.10, no hard fail, and (if saturated) both
mutations moved the score. Any hard fail ⇒ capability claim **not supported**;
do not tune a threshold, gate, item or gold to recover a number. Interpretation
matrix unchanged from `PDR_BRAIN_DESIGN.md` §4.4: controls-pass/rules-pass/
interior ⇒ informative PASS; all-boundary ⇒ saturated/INVALID; controls-pass/
rules-fail ⇒ genuine FAIL; exit 2 ⇒ no score.

---

## 6. When TRA is a LEARN-5 component vs a mechanism probe only

BRAIN_LEARN5 §2.3 defines learning as the conjunction of four properties:

1. **Change** — within-item `k=8 − k=0` delta (not a level).
2. **Attribution** — the type slot is tied to provenance-stamped teaching
   episodes; a guessed or pre-existing answer cannot count.
3. **Transfer** — the graded probe uses an unseen word (`near`) and a different
   domain sharing only the type (`far`).
4. **Counterfactual** — `k=0`, `chance`, `foil`, `negative`, `permuted` arms stay
   flat, and the gain survives a cold reload.

**Valid LEARN-5 component** only if all four hold on **both** pools with the
§5 PASS rules and the §3 gates: it would supply the typed-slot substrate RSI
lacks and could enter the release rule alongside CAR/WFA/PDR/CLI/RSI.

**Mechanism probe only** if any of: immediate in-process binding that vanishes
at cold reload; saturated controls (all 1.00/0.00) with no mutation evidence;
a score from the fake oracle or a fake backend; a pass obtained by reading
`world_model`/`type_sets` at query time; a single-family result (n<40); a
primary-only pass; or a "type" that is just the constant domain label (audit
§2b — constant within candidates, zero discriminating power). Such a result is
reported as an experiment, never as a LEARN-5 PASS.

---

## 7. Fake-based implementation sequence and risk list

**Sequence (fake first, engine last; no threshold ever changes after a result):**

1. Freeze this spec, `HASH_PREFIX`, 400-family bank, hash/one-k split, manifest and `TH`. Commit before any run.
2. Implement `FakeBackend` (deterministic, `engine_graded=False`, no boot/save/network) + `build_items` + `validate_items` + unit tests that pin `TH`, the manifest hash, and the split. Prove the fake arm cannot emit a graded answer (C11).
3. Implement `RSIComponent`-style arms (`typed`, `naive`, `chance`, `counterexample`, `permuted`) and `grade()` with C8 fixed vectors. Run the fake sweep: all C1–C10 must pass on the oracle; run M-A/M-B on the scratch bank to prove the grader is connected to the type label.
4. Only then add `RealBackend`: lazy import, isolated scratch `data_dir`, provenance-stamped teaching turns, query-only grading, `api_calls`, `engine_observation_missing`, `engine_response_unparsed`, cold reload. Run `--validate-only`, then a single PRIMARY run, read answers, then reveal SEALED once.
5. Interpret with §5 matrix; if saturated, run M-A/M-B on a scratch store before citing anything.

**Risks and their controls:**

| Risk | Failure mode | Control |
|---|---|---|
| **Cold-reload sanitizer** | `__getstate__` drops the provenance field on save; unpickling skips `__init__` so the reloaded slot has no writable dict | re-verify on a **fresh** engine (C9); make the field lazy and have the writer backfill; re-check the store *after* the harness run |
| **Provenance overwrite** | an earlier producer's flag overwrites the later type-stamping producer, so the slot looks taught but is unprovenanced | re-evaluate the stamp on the text actually emitted; trace the string through every wrapper; C12 |
| **Duck-typed fakes** | a fake implements `.present()` and returns gold/`world_model`, scoring 1.00 | C11: require `engine_graded`, live API counters, query-response tokens; fake PASS never reported; M-A/M-B detect a disconnected grader |
| **Answer-key leakage** | `world_model` or `type_sets` reaches the query prompt or the parser; family-level answer lookup | graded path uses only the query response; audit §2b — key/type-set are ORACLE ceiling only; leakage ⇒ HARD FAIL |
| **Compute confounds** | k levels differ in turn count or exposure, so "k" is just more compute | one k per item, matched neutral turns for k=0, fixed display budget, `k` reported; no extra replay at higher k |
| **Saturation** | all endpoints at 1.00/0.00 read as a win | G6 gate + M-A/M-B; saturated PASS is INVALID as progress |
| **Threshold tuning** | a rule is loosened to turn FAIL into PASS | frozen `TH` asserted by unit test; no edit after results; sealed relabel rule §4 |
| **Split leakage / circular gold** | items or gold chosen after seeing failures | hash split fixed before boot; gold from authored `world_model` independent of the grader; manifest hash pinned |

---

## 8. What this document does not claim

* No TRA result is measured; every prediction and criterion above is a
  pre-registration. No engine, pytest, store write, gate or item bank was touched.
* The `type_sets` ontology is an authored instrument, not evidence that RAVANA
  currently implements a typed slot; the static audit found the substrate
  effectively empty (`RSI_TYPED_ROLE_DESIGN.md` E12–E16).
* Passing TRA would show learning-shaped behaviour on this constructed
  distribution, not open-world generalisation, truth, consciousness, or neural
  isomorphism (BRAIN_LEARN5 §8).
* No brain citation was re-researched here; the mechanism rationale is inherited
  from the verified ledger in `RSI_TYPED_ROLE_DESIGN.md` §5 (Gentner 1983;
  Gick & Holyoak 1983; Tse et al. 2007; McClelland et al. 1995).

### Pre-registration checklist (complete before the first run)

- [ ] `HASH_PREFIX = RAVANA-TRA-v1`; 400 families; 200/200 split; 40 per k per pool; manifest hashes recorded
- [ ] `TH` frozen and asserted by unit test; rubric written before answers are read
- [ ] Gold provenance stated (`world_model`/`type_sets` = ORACLE only, never graded)
- [ ] `PYTHONHASHSEED=0`; same-config reproducibility verified twice
- [ ] One primary endpoint declared; secondaries Holm-adjusted; Wilson/exact CIs planned
- [ ] C1–C13 specified with pass bounds; M-A/M-B prepared on scratch store
- [ ] Sealed pool unrevealed until the single reported run; FAIL rules acknowledged as binding

## Sources (in-repo, read for this specification)

* `docs/BRAIN_LEARN5_DESIGN.md` §2–§8 (hash split, one-k-per-item, n≥40, learning definition, RSI criteria)
* `docs/investigations/RSI_TYPED_ROLE_DESIGN.md` (typed-slot hypothesis, audit §2b, falsifiers D1–D6, citation ledger)
* `docs/investigations/RSI_SYMBOLIC_ROLE_EXPERIMENT.md` (unseen-token failure, mechanism statement)
* `docs/investigations/PDR_BRAIN_DESIGN.md` §1, §4, §5 (controls, outcome matrix, mutation control)
* `scripts/measure_learn5_rsi.py` (item schema, parser, C1–C8, decide, saturation wiring)
* `scripts/measure_learn5_pdr.py` (instrument controls, provenance/cold-reload schema, exit codes)
* `scripts/measure_learn5_car.py` (phase structure, cold boot 1/2, matched k=0 turns)
* `scripts/measure_learn5_wfa.py`, `scripts/front_gates.py` (anti-fake observation seam, G6 saturation gate)
* `tests/unit/test_learn5_*.py` (frozen-`TH`, no-boot/no-network/no-save, unparsed-response tests)
