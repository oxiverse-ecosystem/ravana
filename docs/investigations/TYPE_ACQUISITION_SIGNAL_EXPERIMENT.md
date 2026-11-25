# Type-acquisition signal search — is ANY teaching-turn signal enough?

**Status: negative experiment, scratch + docs only. DO NOT IMPLEMENT.**
No file under `ravana/src/`, `ravana_ml/src/`, `scripts/` or `tests/` was edited.
No `CognitiveChatEngine` was constructed, no `process_turn` called, no store
written, no engine booted (a parent WFA run was active). Worktree
`C:/Users/Likhith/Documents/projects/ravana_typelearn_exp` on
`experiment/type-acquisition-signal` from main `bfe99a6`. The unmodified
production GloVe loader and the frozen item bank were imported read-only and
their source hashes were asserted identical between main and worktree before
any number was read.

**Question.** On the 40 recorded live TRA cases, can a signal that is
*legitimately present in the teaching turns* classify an unseen subject
(`near`/`far`) while still rejecting `foil` and `negative`? This is not the
400-family TRA benchmark and not a LEARN-5 component claim — it is a
static/recorded-case search over candidate mechanisms.

## Brain grounding (inherited, re-cited)

Fast mapping / sparse lexical type acquisition: Carey & Bartlett 1978;
Dollaghan 1985, *J Speech Hear Res* 28(3), doi:10.1044/jshr.2803.454; Medina
et al. 2011, *PNAS* 108, doi:10.1073/pnas.1105040108. Category induction from
scattered exemplars: Xu & Tenenbaum 2007, *Psychol Rev* 114(2),
doi:10.1037/0033-295X.114.2.245. Schema assimilation: Tse 2007, *Science* 317,
doi:10.1126/science.1135935. The prediction tested is that *some* surface or
co-occurrence statistic of multiple teaching turns should let a new word enter
the taught slot; this doc tests four concrete candidates, not the general
neuroscientific claim.

## What was run

```
cd C:/Users/Likhith/Documents/projects/ravana_typelearn_exp
C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe \
  scratch/type_acquisition_signal.py > scratch/type_acquisition_run.log 2>&1
```

`EXIT=0`, 160 graded rows, zero errors. Read-only inputs outside the worktree:
`data/ravana_glove_cache.npz` (sha256 `92b3c65ddb779ac9…`),
`scratch/tra_harness/tra_rows.jsonl` (sha256 recorded in the results), the
production loader (`295610726018e5f8…`) and `scripts/tra_items.py`
(`7bc55db8637fb4b2…`). No threshold was changed after a result was seen:
congruence floor `0.20`, strict margin `0.10`, two deterministic permutation
shuffles, all fixed in the script before run 1.

### Instrument validation before the number

| Check | Result |
|---|---|
| Production loader SHA-256 (main = worktree) | `295610726018e5f8…` identical; `__file__` asserted under main |
| Item bank SHA-256 (main = worktree) | `7bc55db8637fb4b2…` identical; `build_items()` 400, `validate_items()` 0 errors |
| Recorded rows | 13,440 total; **12,320 fake rows excluded**, 1,120 `backend=live` used |
| Graded selection | one structural/query/non-permuted row per (item, condition) = **160 rows**, 40 items × 4 conditions |
| Bank cross-check | all 160 subjects, candidate sets and gold targets match the regenerated bank (0 mismatches) |
| GloVe lookup | 400,001 words × 64 D via unmodified `build_glove64_lookup` |
| Production baseline recomputed from recorded flags | near **0/40**, far **0/40** (confirms cited `TRA_typed_transfer = 0.000`); foil 11/40, negative 18/40 |
| Teaching material | 114 teaching surfaces regenerated with `teaching_surface()` (recorded rows carry empty `teaching_surfaces`); pooled co-occurrence vocab = 29 words |

## Arms

| Arm | Scoring surface | Deployable? |
|---|---|---|
| `A0` | Production baseline, recorded focused live (near/far recomputed from `flags.correct`) | n/a — comparator |
| `A1` | GloVe **item-own category centre**: mean unit vector of this item's first-k taught `r2` words; candidates scored by `cos(vec(candidate), centre)` | yes (control — already known to fail) |
| `A2` | **Teaching-corpus co-occurrence**: per-word context vectors from words co-occurring in the actual teaching surfaces only; options ranked by `cos(ctx(query_subject), ctx(option))`; pooled corpus primary, item-local reported | yes |
| `A3` | **Character-3gram fast mapping**: taught `r2` word nearest the query subject by n-gram Jaccard becomes the prototype; options ranked by Jaccard to it; subject admission uses the same 0.20 floor | yes |
| `A4` | **ORACLE ceiling**: item-bank `world_model` gold for near/far, `None` for foil/negative | **NO — contamination** |

All deployable scorers see only `type_label`, `taught_exemplars` (first-k `r2`
words), the query subject and the candidate list. `type_sets`, `world_model`,
`domain`, `far_domain` and gold are never passed to A1–A3; gold is read only to
grade after scoring. Every arm fails closed on floor, strict margin, two
permutation shuffles, and options absent from its known universe.

## EVIDENCE (measured)

**E1 — Correct counts (n = 40 per condition; foil/negative correct = `None`).**

| Arm | near | far | foil (reject) | negative (reject) | foil leaks |
|---|---|---|---|---|---|
| `A0` (recorded production) | **0/40** | **0/40** | 11/40 | 18/40 | 0 |
| `A1` item-own centre | **4/40** | **4/40** | 13/40 | 32/40 | **27/40** |
| `A2` teaching co-occurrence | **1/40** | **0/40** | 31/40 | 30/40 | **9/40** |
| `A3` char-ngram fast map | **0/40** | **0/40** | 40/40 | 40/40 | 0 (all abstain) |
| `A4` ORACLE (non-deployable) | 40/40 | 40/40 | 40/40 | 40/40 | 0 |

Diagnostics (not headline arms): `A2_local` 0/40 near, 0/40 far;
`A3_no_subject_gate` 0/40 near, 0/40 far with **15/40 foil leaks**;
`A3_subject_to_option` 0/40 near, 0/40 far.

**E2 — near+far by pool** (40 graded cases per pool per arm): `A0` 0 primary /
0 sealed; `A1` 2 / 6; `A2` 0 / 1; `A3` 0 / 0; `A4` 40 / 40. The only nonzero
deployable cells are small and dominated by artefacts (E3).

**E3 — Hit audit: none of the deployable near/far hits is within-item transfer.**

- `A1`: 8 hits. Three are the same token `cartilage` (`tra_0025`, `tra_0031`,
  `tra_0038`) — one repeated body-domain word; the other five (`drizzle`,
  `signal`, `ritual`, `election`, `radius`) are cross-item lexical coincidences.
  No hit is at k=0, but no hit is caused by the item's own slot centre separating
  its gold from distractors: the centre is a tight embedding neighbourhood.
- `A2`: its single hit (`tra_0021` near, `cloud`→`frost`) is a **k=0 item**,
  i.e. that item contributed zero teaching turns. It is a cross-item corpus
  artefact, exactly the failure mode E3 of the category-centre doc warns about.
- `A3`: zero hits, zero near/far picks.

**E4 — The only arms with any near/far hits leak foils.** `A1` emits a candidate
on 27/40 foil cases (e.g. `tra_0000` foil subject `tempest` → `storm`, GloVe
cosine 0.763; `tra_0004` → `breeze`, cosine 1.000) — a surface matcher accepting
the demonstrated anchor, which the spec requires be rejected. `A2` emits on
9/40 foils. `A3`'s zero foil leaks are trivial abstention, not selective
rejection; its own no-subject-gate diagnostic leaks 15/40.

**E5 — Abstention dominates the deployable arms.** Reasons across all 160 cases:
`A1` 74 `strict_margin_not_met`, 59 `strict_winner`, 24
`no_subject_vector_or_own_centre`, 3 `no_option_above_floor`; `A2` 114
`subject_absent_from_pooled_teaching_corpus`, 27 `strict_winner`, 12
`strict_margin_not_met`, 7 `no_option_above_floor`; `A3` 136
`subject_not_fast_mapped_to_taught_form` + 24 k=0 `no_taught_r2_word_at_k0`.
So `A3`'s perfect-looking 40/40 foil and negative rejection is a gate property,
not a capability.

**E6 — Fail-closed controls (all PASS, with a caveat).** Disjoint options
`apple/banana/chair/table` and unrelated subject `zeppelin` return `None` for
A1/A2/A3. A1's `None` comes from `strict_margin_not_met` (scores exist but no
winner) — a genuine separation failure. A2's and A3's come from
`subject_absent_from_pooled_teaching_corpus` /
`subject_not_fast_mapped_to_taught_form`: the mechanism has no representation
for the unrelated subject at all. Both fail closed; neither demonstrates that a
*taught* type separates two options.

**E7 — G6 saturation (shipped `front_gates.gate_saturation`, unmodified,
non-blocking FLAG).**

| Arm | `tra_near` | `tra_negative_reject` | G6 |
|---|---|---|---|
| `A0` | 0.0 | 45.0 | FLAG — near low band |
| `A1` | 10.0 | 80.0 | PASS (no saturated band) — but a **scored FAIL** at 10% near with 27 foil leaks |
| `A2` | 2.5 | 75.0 | FLAG — near low band |
| `A3` | 0.0 | 100.0 | FLAG — both fronts non-informative |
| `A4` | 100.0 | 100.0 | FLAG — both fronts saturated (oracle) |

A G6 PASS is not capability evidence, and a G6 FLAG is non-blocking; neither
changes the verdict below.

**E8 — What the oracle actually needs.** Across all 160 cases, the query
subject's own authored `type_set` contains the taught `type_label` **0/160**
times; for all 80 near/far cases the **gold candidate's** type set contains it
**80/80** times. A4 therefore scores 40/40 on every condition only by reading
candidate-side membership / `world_model`. That is a contamination ceiling, not
a mechanism: the information that solves the task lives on the *answer options*,
not in the subject or in any teaching-side statistic tested above.

## INFERENCE (not established by the measurements)

- **I1** — The search was broad but shallow: a distributional centroid, a
  teaching-corpus co-occurrence graph, and a subword surface cue all fail. This
  is the same result shape as the earlier GloVe role-centre and symbolic token
  experiments, now over 40 recorded items rather than n=1 per family: the
  failure is not specific to one vector space or one hand-picked threshold.
- **I2** — A2's coverage is a hard ceiling, not just a bad scorer: only 114
  teaching surfaces were generated, the pooled vocab is 29 words, the query
  subject has a teaching-corpus context in only 46/160 cases, and any candidate
  in the corpus in 119/160. Even a perfect co-occurrence reader cannot classify
  a subject with no corpus entry.
- **I3** — A3 fails because character similarity between an unseen domain word
  and taught exemplar words is ~0; the near-exemplar prototype is undefined in
  practice. Its diagnostic without the subject gate also fails at 0/80 and
  leaks foils, so the gate is not the sole problem.
- **I4** — The small positive flips in A1/A2 are overfitting signatures
  (repeated token, cross-item k=0 artefact), not acquisition. Counting them as
  wins would reward lexical memorisation, which the spec's F3 and C10 forbid.
- **I5** — No deployable arm has both near/far > 0 *and* zero foil leakage. That
  conjunction is the task's decision rule, and it fails. The honest result is
  **no deployable teaching-side signal was found** on these recorded cases.
- **I6** — This does not prove typed transfer is impossible in RAVANA. It rules
  out these three teaching-side statistics on this bank with these recorded
  cases. The brain findings support that categories *can* be induced from
  exemplars; they do not imply that a co-occurrence or n-gram statistic is the
  format the engine needs.

## Recommendation tied to C10 and G6

**DO NOT IMPLEMENT — no production PR, no default-on flag change, no TRA PASS
claim.**

- **C10 (type-contrast sanity).** The fail-closed half holds (E6). The positive
  half does not: no deployable arm shows a taught type genuinely separating two
  candidate options on unseen near/far words. Under the spec's "contrast sanity
  fail ⇒ INSTRUMENT INVALID" reading, none of A1–A3 can be certified as a
  working type contrast. The shipped opt-in path still fails C10 by emitting
  `banana` (`TRA_SLOT_WRITE_READ_EXPERIMENT.md` E11).
- **G6.** Three arms are FLAG; A1's PASS is a 10% near score with 27/40 foil
  leakage — a scored FAIL, not a win. A4's 100/100 is saturated by construction
  and is explicitly non-deployable.
- The **smallest additional knowledge** the engine would need is a
  *candidate-side type-membership source*: a function that maps both taught
  exemplar words and unseen candidate words to the same domain-general slot
  (e.g. `rainbow`/`glow`/`ceremony` → `effect`), available at query time and
  independent of the answer key. In this bank that function is exactly
  `type_sets`/`world_model` (E8), so using it would be contamination, not
  learning. A legitimate version would have to come from an external ontology
  or an independently learned type lexicon, not from the item bank's gold.
  The only separately measured partial primitive is the scratch trusted
  `is_a` write + fail-closed slot-keyed read in
  `TRA_SLOT_WRITE_READ_EXPERIMENT.md` (6/6 cold-reload retention, disjoint ⇒
  `None`), but it never tested unseen near/far words and its production
  scorer's option space is type labels, not domain words
  (`TRA_CONTROL_SCOPE_AUDIT.md` §2.2). It is a substrate lead, not a fix.
- Next work item, in order: (1) supply a non-contaminating candidate type
  source; (2) measure it with the same near/far/foil/negative discipline and
  the disjoint control wired as a hard instrument gate; (3) only then consider
  a default-OFF production prototype. Re-run this scratch search if a new
  source is proposed — do not promote a mechanism from its oracle arm.

## Honest caveats

- n = 40 recorded items (20 primary / 20 sealed), 4 conditions each — a
  mechanism replay, not the 400-family benchmark; no Wilson CI, k-curve or
  cold-reload endpoint.
- Teaching surfaces were regenerated from the item bank because the recorded
  rows carry empty `teaching_surfaces`; they are the exact surfaces the bank
  defines, but they are not a transcript of what the live engine stored.
- 5 of 6 type labels have no GloVe vector; A1 does not depend on label vectors,
  so its failure is not a coverage artefact, but the earlier label-space
  variants are unassessable here.
- Floor 0.20 and margin 0.10 are designer choices, NOT TRA-pre-registered
  thresholds (as in the category-centre experiment); no sensitivity sweep was
  run, and none was tuned after a result.
- A4 is saturated by design and is reported only to size the gap.

## Files

- `scratch/type_acquisition_signal.py` — scratch-only script (no engine, no key)
- `scratch/type_acquisition_results.json` — 160 rows, controls, G6, hit audit
- `scratch/type_acquisition_run.log` — captured stdout, `EXIT=0`
- this document
