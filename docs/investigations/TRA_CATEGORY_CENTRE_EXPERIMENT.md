# TRA category-centre experiment — category centre from taught exemplar slots

**Status: negative experiment, scratch + docs only. DO NOT IMPLEMENT.**
No file under `ravana/src/`, `ravana_ml/src/`, `scripts/` or `tests/` was edited.
No `CognitiveChatEngine` was constructed, no `process_turn` called, no store
written, no engine booted. Worktree
`C:/Users/Likhith/Documents/projects/ravana_tracentre` on
`experiment/tra-category-centre` from main `bfe99a6`. The production GloVe
loader was imported read-only from the unmodified main tree and its source
hash was asserted identical to the worktree copy before any number was read.

This tests the one route not covered by the three earlier failures: a
**category centre** formed from several provenance-taught exemplar slots
(mean unit GloVe vector of the taught `r2` objects carrying a slot), rather
than a single label vector, a token signature, or a gold-key oracle. It is a
static/recorded-case mechanism probe, **not** the 400-family TRA benchmark and
**not** a LEARN-5 component claim.

## Brain grounding (verified in the prior literature pass, re-cited here)

- Fast mapping / sparse lexical type acquisition: Carey & Bartlett 1978;
  Dollaghan 1985, *J Speech Hear Res* 28(3), doi:10.1044/jshr.2803.454;
  Medina et al. 2011, *PNAS* 108, doi:10.1073/pnas.1105040108.
- Bayesian category induction from scattered exemplars: Xu & Tenenbaum 2007,
  *Psychol Rev* 114(2), doi:10.1037/0033-295X.114.2.245.
- Schema assimilation from repeated instances: Tse 2007, *Science* 317,
  doi:10.1126/science.1135935.

Prediction tested: if the brain forms a category from multiple taught
instances, a mean of several taught exemplar vectors should let an unseen
subject be admitted to the taught slot, with foil/negative still rejected.

## What was run

```
cd C:/Users/Likhith/Documents/projects/ravana_tracentre
C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe \
  scratch/tra_category_centre_experiment.py > scratch/tra_category_centre_run.log 2>&1
```

`EXIT=0`, 160 graded rows, 3.43 s, `errors=[]`. Two read-only inputs outside
the worktree: `data/ravana_glove_cache.npz` (sha256 `92b3c65ddb779ac9…`) and
`scratch/tra_harness/tra_rows.jsonl`. No threshold was changed after a result
was seen (`floor=0.20`, `strict_margin=0.10`, fixed in the script before run 1).

### Instrument validation before the number

| Check | Result |
|---|---|
| Production loader SHA-256 (main = worktree) | `295610726018e5f8…` identical; loader `__file__` asserted under main |
| Item bank SHA-256 (main = worktree) | `7bc55db8637fb4b2…` identical; `build_items()` = 400, `validate_items()` = 0 errors |
| GloVe lookup | 400,001 words × 64 D via unmodified `build_glove64_lookup` |
| Recorded rows | 1,120 `backend=live` used; **12,320 fake rows excluded**; all 1,120 subjects/candidates/golds cross-checked against the regenerated bank (0 mismatches) |
| Word coverage | all 102 recorded words have vectors; **only 1 of 6 type labels (`effect`) is in GloVe**; 5 are OOV |
| Probe hand-check | `cos(thaw, effect)=0.3231`, reproduces the value recorded in `TRA_WRITE_READ_EXPERIMENT.md` E7 |

### Retracted runs (instrument defects, retained for audit, never averaged in)

| Run | Defect | Effect |
|---|---|---|
| 1 (`tra_category_centre_run_invalid_r1.log`) | confidence-only scalars fed to a cosine scorer → `TypeError` | `EXIT=2`; no numbers produced |
| 2 (`tra_category_centre_run_invalid_r2.log`) | no centre-separation / wiring diagnostic, so the exact-centre sanity failure could not be classified | arm counts superseded |
| 3 (`tra_category_centre_run_invalid_r3.log`, `..._results_invalid_r3.json`) | secondary domain arm emitted one below-floor pick (`below_floor_but_emitted`) — a fail-open branch | fixed to return `None`; 1 domain row changed |
| 4 (`tra_category_centre_run_pre_g6.log`, `..._results_pre_g6.json`) | valid but superseded: G6 gate not yet wired into the run | retained; counts identical to final |

## Inputs and n (all teaching material)

- 40 recorded live items (`tra_0000`–`tra_0039`), **20 primary / 20 sealed**;
  `k` distribution `{0:6, 1:8, 2:11, 4:9, 8:6}`.
- Only the item bank's `type_label` and `taught_exemplars` (the `r2` values)
  enter any centre. `type_sets`, `world_model`, gold and the query subject's
  own type are **never** passed to a scorer; gold is read only to grade after
  scoring. The teaching items are the frozen bank, not oracle answers.
- Stored confidences from `scratch/tra_slot_results.json`: 6 trusted typed
  edges for `mist/hail/drizzle→weather phenomenon` and
  `election/ceremony/contract→civic event` (0.3638–0.4298). These are **not**
  TRA type labels; only 10 taught-exemplar occurrences across the 40 items
  carry any recorded confidence.
- 6 centres: `contrast-case` (10 contributions), `earlier-event` (23),
  `effect` (22), `similar-form` (21), `structured-whole` (24),
  `supported-outcome` (14); one-k-per-item, so a k=0 item contributes nothing.

## Arms

| Arm | Scoring surface |
|---|---|
| `C_pooled_cos` | options = 6 type labels; score = `cos(unit(vec(subject)), unit(centre[label]))`, pooled centres |
| `C_local_cos` | same, but the queried item's own slot centre uses **only its own first-k taught `r2` vectors** (k=0 ⇒ no centre); other slots pooled |
| `C_label_tiebreak` | `C_pooled_cos` plus `cos(centre, vec(label))` only to break a top-2 tie inside the margin |
| `C_conf_weighted` | centre vectors weighted by recorded confidences where available |
| `C_conf_only` | stored confidence **is** the score, no vectors at all |
| `C_domain_membership` | secondary comparability surface: `cos(vec(candidate), slot_centre)`, taught `r2` excluded; near/far correct = gold word, foil/negative correct = `None` |

Gates on every arm: congruence floor 0.20, strict margin 0.10, two deterministic
permutation shuffles (any changed pick ⇒ `None`), disjoint option set ⇒ `None`,
missing subject vector ⇒ `None`.

## EVIDENCE (measured)

**E1 — Centre geometry.** Pairwise centre cosine ranges **0.729–0.980**
(min `contrast-case`↔`structured-whole`, max `earlier-event`↔`supported-outcome`).
With the subject vector set exactly equal to a known centre, top-1 is the
correct label **6/6 at score 1.0000**, but the top-2 margin is
**0.0198–0.0873**, below the frozen 0.10 gate: the exact-centre sanity
**FAILS** at the gate for every label. An instrument-only wiring check with
the margin disabled solely in that probe picks the correct label **6/6** —
the scorer is connected; the veto is the geometry.

**E2 — Per-arm correct counts** (n = 40 items × 4 conditions = 160 cases per
arm; foil/negative correct = `None`):

| Arm | near | far | foil (reject) | negative (reject) |
|---|---|---|---|---|
| `C_pooled_cos` | **1/40** | **1/40** | 36/40 | 32/40 |
| `C_local_cos` | **0/40** | **0/40** | 38/40 | 33/40 |
| `C_label_tiebreak` | **1/40** | **1/40** | 36/40 | 32/40 |
| `C_conf_weighted` | **1/40** | **0/40** | 30/40 | 31/40 |
| `C_conf_only` | **0/40** | **0/40** | 40/40 | 40/40 |
| `C_domain_membership` | **4/40** | **2/40** | 25/40 | 31/40 |

near+far by pool (40 graded cases per pool per arm): `C_pooled_cos` primary
0/40, sealed 2/40; `C_local_cos` 0/40 and 0/40; `C_label_tiebreak` 0/40 and
2/40; `C_conf_weighted` 0/40 and 1/40; `C_conf_only` 0/40 and 0/40;
`C_domain_membership` 3/40 and 3/40.

**E3 — Hit audit: none of the label-space hits use this item's teaching.**
Both `C_pooled_cos` hits (`tra_0011` near `storm`→`contrast-case`,
`tra_0039` far `arc`→`similar-form`) and the single `C_conf_weighted` hit are
**k=0 items**, whose own slot received zero taught exemplars; they are
cross-item centre artefacts, not within-item transfer. `C_local_cos`, the arm
that actually uses the item's own k taught exemplars for its own slot, scores
0/40 near and 0/40 far. Of the six `C_domain_membership` hits, four are the
same token `cartilage` (`tra_0025`, `tra_0031`, `tra_0035`, `tra_0038`) — a
single taught body-domain word, not general transfer.

**E4 — Abstention is the dominant outcome.** Reasons across all 160 label-space
cases: `C_pooled_cos` 133 `strict_margin_not_met`, 8 `no_category_above_floor`,
19 strict winners; `C_local_cos` 139/8/13; `C_conf_only` **160/160**
`strict_margin_not_met`. High foil/negative "correct" counts are therefore
mostly abstentions, not selective rejection: `C_conf_only` returns `None` on
all 160 cases, trivially correct on all 80 foil/negative cases and useless on
all 80 near/far cases.

**E5 — Fail-closed controls.** Disjoint options `apple/banana/chair/table` ⇒
`pick=None`, reason `disjoint_or_unknown_option_set` (PASS). Unrelated subject
`zeppelin` ⇒ `None` via `strict_margin_not_met` (scores 0.146–0.248), PASS.
Six k=0 items contribute no teaching by construction.

**E6 — Confidence signal cannot ground TRA slots.** `C_conf_only` scores are
near-uniform (sample row: `contrast-case` 0.4298, `earlier-event` 0.4198,
`effect` 0.4231, `similar-form` 0.4095, `supported-outcome` 0.4298;
`structured-whole` has no recorded confidence at all); no margin clears 0.10.

**E7 — G6 saturation (shipped `scripts/front_gates.gate_saturation`, called
unmodified; non-blocking FLAG):**

| Arm | `tra_near` | `tra_negative_reject` | G6 |
|---|---|---|---|
| `C_pooled_cos` | 2.5 | 80.0 | **FLAG** — `tra_near` low band |
| `C_local_cos` | 0.0 | 82.5 | **FLAG** — `tra_near` low band |
| `C_label_tiebreak` | 2.5 | 80.0 | **FLAG** — `tra_near` low band |
| `C_conf_weighted` | 2.5 | 77.5 | **FLAG** — `tra_near` low band |
| `C_conf_only` | 0.0 | 100.0 | **FLAG** — both fronts non-informative |
| `C_domain_membership` | 10.0 | 77.5 | PASS (no front in saturated band) |

**E8 — Comparators (cited, not recomputed).** Production focused live TRA run:
`TRA_typed_transfer = 0.000` (`TRA_CONTROL_SCOPE_AUDIT.md` E3). Earlier vector
role-centre: near 1/4, far 1/4, foil 0/4, negative 4/4
(`RSI_ROLE_BINDING_EXPERIMENT.md`). Symbolic signatures: V1/V2 near/far 0/4,
V3/V4 sym-floor near 2/4 only on already-correct items with 3–4 foil leaks
(`RSI_SYMBOLIC_ROLE_EXPERIMENT.md`). Those used the RSI item bank at n=1 per
family/condition, so they are context, **not** a like-for-like numerical
comparison with this 40-item TRA replay.

## INFERENCE (not established by the measurements)

- **I1** — The route fails at representation, not at storage or trust: the six
  mean vectors occupy a tight cone (0.73–0.98 cosine), so a GloVe centroid of
  taught exemplar objects does not separate the slots. A perfect-centre query
  is vetoed by the frozen margin. This is a property of the vector space, not
  of the floor/margin choice alone; no threshold was re-tuned to show it.
- **I2** — The tie-break variant is effectively inert: 5 of 6 type labels are
  OOV in GloVe, so `cos(centre, label_vec)` is unavailable for them. The
  primary arm does not use label vectors, so its failure is not a coverage
  artefact.
- **I3** — The stored-confidence variant cannot supply the missing signal:
  the recorded edges concern other category strings, so on TRA slots the
  confidences are nearly constant and fail-closed everywhere.
- **I4** — The positive flips in the pooled arm are not acquisition: they
  occur on k=0 items with no teaching from that item. Counting them as wins
  would reward cross-item lexical similarity, exactly the failure F3 warns of.
- **I5** — There is no negation channel in a static centroid. `negative`
  demonstrations write no trusted slot in the real engine, but this static
  replay has no way to see that; its high negative "correct" count is
  abstention, not negation handling.
- **I6** — The brain findings support *that* categories can be induced from
  exemplars; they do not imply that a mean of distributional word vectors is
  the brain's category format. The data are consistent with a typed,
  structurally constrained slot rather than an embedding neighbourhood —
  the same mechanism RSI experiments said was missing.
- **I7** — This does not prove typed transfer is impossible in RAVANA; it
  rules out this specific centroid route on this bank with these inputs.

## Recommendation tied to TRA C10 and G6

**DO NOT IMPLEMENT — no production PR, no default-on flag change.**

- **C10 (type-contrast sanity).** The *fail-closed half* holds here: disjoint
  options and an unrelated subject both return `None`, so the mechanism does
  not admit everything. But C10 also requires that a taught type genuinely
  separate two candidates and that a disjoint control return `None`.
  The exact-centre probe — the strongest possible type contrast — **fails**
  the margin for 6/6 slots, and `C_local_cos` (the arm using this item's own
  teaching) scores 0/40 near, 0/40 far. Under the spec's "contrast sanity fail
  ⇒ INSTRUMENT INVALID" reading, this route cannot be certified as a working
  type contrast. The shipped flag-on path also still fails C10 by emitting
  `banana` (`TRA_SLOT_WRITE_READ_EXPERIMENT.md` E11).
- **G6.** Five of six arms are FLAG (non-informative low band). The only G6
  PASS (domain membership, `tra_near=10.0`) is far below TRA's `near >= 0.75`
  PASS rule and is dominated by one repeated token. A G6 FLAG is non-blocking
  but is explicitly **not** capability evidence; a G6 PASS at 10% is a scored
  FAIL, not a win.
- The honest outcome is a **negative deliverable**: a third distinct
  representation (after GloVe role centroids and symbolic token signatures)
  fails to supply unseen-subject typed transfer while preserving fail-closed
  abstention. Keep this branch as evidence; name the still-missing mechanism:
  a typed, domain-general role slot with a decision surface over domain-word
  options, a negation channel, and non-embedding membership — not a mean of
  taught word vectors.

## Honest caveats

- n = 40 recorded items (20 primary / 20 sealed), 4 conditions each — a
  mechanism replay, not the 400-family benchmark, with no Wilson CI or k-curve.
- Only 1 of 6 type labels has a GloVe vector; the tie-break variant is
  unassessable on 5 labels and no substitute vector was fabricated.
- Recorded confidences cover 10 exemplar occurrences, not a TRA type label.
- The source store is one `weights/` snapshot; a different history changes the
  centroid geometry.
- Retracted runs 1–3 and superseded run 4 are retained, not averaged in.

## Files

- `scratch/tra_category_centre_experiment.py` — scratch-only script
- `scratch/tra_category_centre_results.json` — 160 rows, controls, G6, hit audit
- `scratch/tra_category_centre_run.log` — final captured stdout, `EXIT=0`
- `scratch/tra_category_centre_run_invalid_r{1,2,3}.log`,
  `scratch/tra_category_centre_results_invalid_r3.json`,
  `scratch/tra_category_centre_run_pre_g6.log`,
  `scratch/tra_category_centre_results_pre_g6.json` — retained defect/supersession audit
- this document
