# SUBSTITUTION_COHERENCE — testing "replace the unseen word with a known word, then check the paragraph"

**Status: negative experiment, scratch + docs only. DO NOT IMPLEMENT.**
No file under `ravana/src/`, `ravana_ml/src/`, `scripts/` or `tests/` was edited.
No `CognitiveChatEngine` was constructed, no `process_turn` called, no store
written, no engine booted (a parent WFA run was active). Worktree
`C:/Users/Likhith/Documents/projects/ravana_substitution` on
`experiment/substitution-coherence` from main `bfe99a6`. The item bank and the
shipped `scripts/front_gates.py` were imported read-only from the unmodified
main tree; the bank's SHA-256 was asserted identical between main and worktree
before any number was read. No word embeddings were used: the claim under test
is a **coherence** test, not a similarity test, and the embedding-centroid route
is already a measured failure (`TRA_CATEGORY_CENTRE_EXPERIMENT.md`).

## The hypothesis, stated as a mechanism

User's own intent, verbatim: *"I will learn new words (unseen) through replacing
the word with already known word, and then checking the para when study wholly
makes sense and consistent."*

Instantiated on the recorded TRA cases:

1. Build a **paragraph** from the item's teaching surfaces plus the query
   surface. Only `type_label`, `taught_exemplars` (`r1`/`r2`), `domain` and the
   bare query token enter it. `type_sets`, `world_model`, the item's
   `far_domain` and the query subject's own type are **never** given to a
   deployable scorer; gold is read only to grade after scoring.
2. For each taught exemplar object `r2_i`, **substitute it into the query
   subject position**, giving `<exemplar_i> is a <option>` with the teaching
   paragraph as evidence.
3. Score **coherence**, not similarity: (S1) trusted-typed-edge chain plus role
   preservation across all k exemplars; (S2) query-context ↔ teaching-context
   overlap; (S3) whole-paragraph licensing; (S4/S5) the same two over the six
   type labels, which is the literal "candidate **type** option" /
   `<subject> is a <type>` reading; (S6) a `type_sets` **ORACLE**, explicitly
   non-deployable.
4. Every arm passes the TRA fail-closed gates: congruence floor `0.20`, strict
   margin `0.10`, two deterministic permutation shuffles of the tie-break
   order, disjoint options ⇒ `None`, unrelated subject ⇒ `None`, and the
   shipped G6 saturation gate on all arms.

## Brain grounding (verified in `TYPE_LEARNING_BRAIN.md`, re-cited here)

Carey & Bartlett 1978 (contrastive referent gives the type); Dollaghan 1985,
*J Speech Hear Res* 28(3), doi:10.1044/jshr.2803.454; Medina et al. 2011,
*PNAS* 108, doi:10.1073/pnas.1105040108 (most natural word-learning instances
are uninformative; learners keep one hypothesis); Xu & Tenenbaum 2007,
*Psychol Rev* 114(2), doi:10.1037/0033-295X.114.2.245 (sparse induction needs
a **pre-existing** kind hierarchy); Tse et al. 2007, *Science* 316,
doi:10.1126/science.1135935 (one-trial assimilation needs an existing schema).
The prediction tested here is the user's specific one: that substitution plus
a paragraph-consistency check is enough. This doc tests that mechanism, not
the general neuroscientific claim.

## What was run

```
cd C:/Users/Likhith/Documents/projects/ravana_substitution
C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe \
  scratch/type_substitution_coherence.py > scratch/type_substitution_run.log 2>&1
```

`EXIT=0`, 160 graded rows, 0.43 s, `errors=[]`.

| Instrument check | Result |
|---|---|
| Item bank SHA-256 (main = worktree) | `7bc55db8637fb4b2…` identical; `build_items()` 400, `validate_items()` 0 errors |
| Recorded rows | 13,440 total; **12,320 fake rows excluded**; 1,120 `backend=live` used |
| Graded selection | one structural/query/non-permuted row per (item, condition) = **160**; 40 items × 4 conditions |
| Bank cross-check | all 160 subjects, candidate sets and gold targets match the regenerated bank (0 mismatches) |
| Recorded trusted typed edges | 6, targets = `weather phenomenon`, `civic event`; **0** target any TRA type label |
| Production baseline recomputed from recorded flags | near **0/40**, far **0/40**, foil 11/40, negative 18/40 |
| No-engine / no-key | `CognitiveChatEngine` never imported; no store write; gold/type_sets absent from S1–S5 |

### Retracted run (retained for audit, never averaged in)

Run 1 (`type_substitution_run_invalid_r1.log`,
`type_substitution_results_invalid_r1.json`) had a **probe defect**: the
permutation gate shuffled the option order and then re-sorted by score inside
`_one()`, discarding the shuffle, so every strict winner — including S5's
`effect=1.0` winner visible in that run's own worked example — was vetoed as
`permutation_unstable`. That produced S5 near 0/40, contradicting its own
printed score. The ranker was fixed to permute the **tie-break** key only
(scores never change), and the label-space controls were fixed to keep the
type-label option space (they had silently reverted to candidate words). Run 2
supersedes run 1; both are committed.

## Worked example (printed input and output, not just the aggregate)

`tra_0000`, `near`, `k=4`, subject `tempest`:

- TEACH paragraph: `cloud causes storm. storm is an effect.` / `storm causes
  frost. frost is an effect.` / `frost causes drizzle. drizzle is an effect.` /
  `drizzle causes thunder. thunder is an effect.`
- QUERY surface: `Considering the examples above, what is true of tempest?
  Options: squall, shower, breeze, sunset. Answer briefly.`
- SUBSTITUTED (r2 object → subject): the same query with `storm`, `frost`,
  `drizzle`, `thunder` in the subject position — the teaching half is unchanged
  and still never mentions `tempest`.
- S1 scores `{squall:0, shower:0, breeze:0, sunset:0}` → `None`.
- S2 query-context = `[breeze, shower, squall, sunset]`; overlaps all exactly
  `0.2055` → `strict_margin_not_met` → `None`.
- S3 scores `{squall:0, shower:0, breeze:0, sunset:0}` → `None`.
- S5 scores `{…, effect:1.0, …}` → picks `effect` for *every* subject.

## EVIDENCE (measured)

**E1 — Per-arm correct counts.** n = 40 per condition (80 near+far); foil and
negative "correct" = `None`. S4/S5 use the six-type-label option space, so
their near/far counts are not directly comparable to the word-selection
baseline; they are included because the hypothesis literally asks about a
*candidate type option*.

| Arm | near | far | foil (reject) | negative (reject) | foil leaks |
|---|---|---|---|---|---|
| `S0` production baseline (recorded) | **0/40** | **0/40** | 11/40 | 18/40 | 0 |
| `S1` graph chain, trusted edge + role, candidate words | **0/40** | **0/40** | 40/40 | 40/40 | 0 |
| `S2` query-context ↔ teaching-context overlap | **0/40** | **0/40** | 40/40 | 40/40 | 0 |
| `S3` whole-paragraph licensing, candidate words | **0/40** | **0/40** | **6/40** | 40/40 | **34** |
| `S4` graph chain over 6 type labels | **0/40** | **0/40** | 40/40 | 40/40 | 0 |
| `S5` paragraph licensing over 6 type labels | 34/40 | 34/40 | **6/40** | **6/40** | **34** |
| `S6` ORACLE `type_sets` — **NON-DEPLOYABLE** | 40/40 | 40/40 | 40/40 | 40/40 | 0 |

near+far by pool (40 graded cases per pool): `S0` 0/0, `S1` 0/0, `S2` 0/0,
`S3` 0/0, `S4` 0/0, `S5` primary 36/40 and sealed 32/40, `S6` 40/40 and 40/40.

**E2 — The four word-space arms are trivial abstainers, not selectors.**
`S1`: 0 near/far picks because the only recorded trusted typed edges target
`weather phenomenon` / `civic event`, never a TRA candidate word or type label.
`S2`: the subject's query context is exactly the four candidate words in 160/160
cases, and the overlap is the **same value for all four options in all 160
cases** — a candidate-invariant score cannot rank, so the strict margin vetoes
it. `S3`: all 80 near/far cases are `no_option_above_floor` or
`k0_no_teaching_paragraph`. Their 40/40 foil/negative "correct" counts are
abstention, not selective rejection.

**E3 — Where substitution+coherence *does* fire, it fires on the foil.**
`S3` emits on 34/40 foils. In every one the picked word is the demonstrated
anchor that the TRA foil condition deliberately plants (`tra_0000` → `storm`,
`tra_0001` → `breeze`, `tra_0003` → `storm`; 34/34 picked words occur verbatim
in that item's teaching paragraph). This is the predicted failure mode of a
paragraph-licence coherence test: "is this reading consistent with the study
paragraph?" answers **yes** for the anchor the paragraph actually taught.
The other 6 foil cases are k=0 items with no teaching at all.

**E4 — The strongest charitable form becomes an answer-everything machine.**
`S5` uses the teaching surface's own assertion (`r2 is a <label>`) as the
trusted type evidence, which is exactly what "check that the paragraph wholly
makes sense" would license. It reaches 34/40 near and 34/40 far (the 34 k>0
items), but returns a type for **all 34 k>0 foils and all 34 k>0 negatives**
(k=0 items are the only 6 abstentions in each condition). It fails the
unrelated-subject control outright: subject `zeppelin` with the six normal type
labels still yields `pick='effect'`, `reason=strict_winner`, `pass=False`.
The paragraph never mentions the subject, so substituting it away leaves
*nothing* that could distinguish a near subject from `zeppelin`.

**E5 — Fail-closed controls.** Disjoint options ⇒ `None` on all five deployable
arms. Unrelated subject ⇒ `None` on S1–S4, and **not** on S5 (E4). No gate was
lowered to admit a hit; floor `0.20`, margin `0.10` and both permutation seeds
were fixed before run 1, and the corrected permutation gate makes ties fail
closed rather than passing them.

**E6 — Information audit (the decisive measurement).** Across all 160 graded
cases: the query subject occurs in its own teaching surfaces **0/160** times;
the gold candidate occurs in its own teaching surfaces **0/80** near/far times;
the number of recorded provenance-trusted typed edges targeting any of the six
TRA type labels is **0**; teaching text is one paragraph per item, shared
unchanged by all four conditions (40 distinct hashes for 40 items). There are
114 teaching surfaces and k distribution `{0:6, 1:8, 2:11, 4:9, 8:6}`.

**E7 — G6 saturation (shipped `front_gates.gate_saturation`, non-blocking FLAG).**
`S0`, `S1`, `S2`, `S3`, `S4` FLAG (`tra_near=0.0`; S1–S4 also
`tra_negative_reject=100.0`, i.e. non-informative abstention). `S6` FLAG
(both fronts saturated by construction). `S5` **PASS** on G6, but PASS here
means its two fronts are not in a saturated band — at 85% near with 34 foil
leaks and a failed unrelated-subject control it is a scored **FAIL**, not a win.
A G6 FLAG is a caveat, never capability evidence.

**E8 — Comparators (cited, not recomputed here).** Production focused live TRA
`TRA_typed_transfer = 0.000` (`TRA_CONTROL_SCOPE_AUDIT.md` E3), recomputed here
as 0/40 near and 0/40 far. Earlier scratch arms on the same 40 cases:
item-own GloVe centre 4/40 near, 4/40 far with **27 foil leaks**; teaching
co-occurrence 1/40, 0/40 with 9 foil leaks; char-ngram 0/40, 0/40 with zero
leaks but all abstain (`TYPE_ACQUISITION_SIGNAL_EXPERIMENT.md` E1). Category
centres: `C_local_cos` 0/40, 0/40; `C_pooled_cos` 1/40, 1/40, all label-space
hits on k=0 items (`TRA_CATEGORY_CENTRE_EXPERIMENT.md` E2/E3). RSI role centres
and symbolic signatures were n=1 per family and scored 1/4 or 0/4; they are
context, not a like-for-like comparison. **No arm in this or any earlier
search has both near/far > 0 and zero foil leakage.**

## INFERENCE (not established by the measurements)

- **I1** — Substitution removes the only piece of information that differs
  between the conditions. The teaching paragraph is item-level; the query
  subject is the only condition-specific token, and step 2 replaces it with a
  taught exemplar. The resulting coherence evidence is therefore a function of
  the item's teaching and the option set, not of the subject being typed. This
  is why S5 answers `zeppelin` and the foils identically to a near subject.
- **I2** — The information deficit is measured, not inferred: E6 gives zero
  subject-in-teaching and zero gold-in-teaching occurrences and zero trusted
  typed edges to the label space. A coherence test has no subject-specific
  fact to be coherent *with*. No ordering of floor, margin or permutation
  gates can create that fact; lowering them would only convert abstention into
  the foil-accepting behaviour of S3/S5.
- **I3** — The literal substitution is also role-violating. Teaching asserts
  `r1 <relation> r2`, and the unseen subject occupies `r1`'s role (the gold is
  the `r2`-like word). Placing the taught **object** `r2` in the subject slot
  breaks the asserted direction; the role-consistency term in S1 therefore
  never fires. A role-preserving substitution (`r1`) still has no teaching
  that names the candidate, so the deficit remains.
- **I4** — S2's `0.2055` overlap is a cross-item lexicon artefact (the same
  word can be a demo word in another item). It is identical across the four
  options in every case, so it carries no ranking information. Counting it as
  evidence would repeat the cross-item k=0 failure already retracted in
  `TYPE_ACQUISITION_SIGNAL_EXPERIMENT.md` E3.
- **I5** — This is **not** a universal impossibility claim. If the inputs were
  changed so that the unseen word had its own ostensive/contrastive referent,
  its own distributional contexts, or a pre-existing kind hierarchy (the three
  channels the brain literature says are actually used), then a coherence test
  over subject-specific evidence could in principle work. What is established
  here is impossibility **under the current input regime**: teaching about
  other words, a bare unseen token in a relation-neutral forced-choice frame,
  and no stored type evidence for that token. S6 shows the answer key lives on
  the *candidate* side (`type_sets`/`world_model`), which is contamination, not
  learning.

## Implement / do-not-implement, tied to the user's hypothesis

**DO NOT IMPLEMENT.** No production PR, no default-on flag, no TRA PASS claim.

| Part of the hypothesis | Verdict | Why, tied to E |
|---|---|---|
| "replace the unseen word with an already-known word" | Mechanically possible, but **not sufficient** | E1/E2: substitution yields no subject-specific evidence; E6: subject appears 0/160 times; I1/I3 |
| "check the para when study wholly makes sense and consistent" | **Fails the TRA selectivity contract** | E3: 34 foil leaks (it endorses the taught anchor); E4: 34 foil + 34 negative leaks and `zeppelin` accepted |
| graph-chain coherence (trusted typed edge) | **Do not implement** | E1 S1/S4: 0/40 and 0/40; E6: zero trusted edges to any TRA label |
| distributional context overlap | **Do not implement** | E1 S2 and E2: candidate-invariant score in 160/160 cases |
| "paragraph wholly makes sense" licensing | **Do not implement** | E1 S3/S5, E3, E4: it is a foil/answer-everything detector |
| `type_sets` / `world_model` oracle | **Do not implement — contamination** | E1 S6: 40/40 by reading the answer key; non-deployable upper bound |

**What the next work item is, not a ceiling to soften:** a *subject-specific*
evidence channel. Concretely, either (A) genuinely acquired, provenance-stamped
type assertions for a broad vocabulary from real conversation/web reading, or
(B) a distributional channel over sentences that actually contain the queried
token (frozen snapshots per `references/frozen-snapshots.md` in the
`ravana-debugging` skill), with held-out words excluded at item-construction
time. Then re-run this exact near/far/foil/negative + disjoint + unrelated +
G6 harness. Do not promote any mechanism from its oracle arm.

## Explicit answer to the task's question

**Does substitution+coherence supply a deployable signal? No.** No arm moves
near/far above zero without foil leakage (deployable verdict: `any_deployable_signal =
false`). **Is the absence of the unseen word from all teaching/contexts an
information deficit that no coherence test can overcome? Yes — under the
current inputs.** E6 measures the deficit directly, I1 explains why
substitution cannot repair it, and E3/E4 show that the only arms which do fire
fire on the foil or on everything. **Does this experiment prove the hypothesis
impossible? Yes for the current input regime; no as a universal claim about
substitution+coherence with subject-specific evidence** (I5). The honest
verdict is *information-starved loop*, not *broken ranker* and not yet
*architecture limit*.

## Honest caveats

- n = 40 recorded items (20 primary / 20 sealed), 4 conditions each — a
  mechanism replay, not the 400-family benchmark; no Wilson CI or k-curve.
- Teaching surfaces were regenerated with the item bank's `teaching_surface()`
  because recorded rows carry empty `teaching_surfaces`; they are the exact
  surfaces the bank defines, not a transcript of what the live engine stored.
- Floor `0.20` and margin `0.10` are designer choices, not TRA-pre-registered
  thresholds; no sensitivity sweep was run and none was tuned after a result.
- `domain` is permitted teaching material but is item-level and identical
  across conditions, so it adds no discriminating evidence here.
- Run 1 is retracted for the permutation-gate defect described above and is
  retained only for audit.

## Files

- `scratch/type_substitution_coherence.py` — scratch-only script (no engine, no key, no embeddings)
- `scratch/type_substitution_results.json` — 160 rows, controls, G6, hit/information audit
- `scratch/type_substitution_run.log` — final captured stdout, `EXIT=0`
- `scratch/type_substitution_run_invalid_r1.log`,
  `scratch/type_substitution_results_invalid_r1.json` — retracted run 1, retained
- this document
