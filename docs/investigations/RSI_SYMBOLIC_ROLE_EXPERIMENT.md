# RSI symbolic role-signature matcher — scratch experiment 2 (no production edits)

**Status: experiment only.** No file under `ravana/src/`, `scripts/` or `tests/`
was edited. No `CognitiveChatEngine` was constructed, no `process_turn` was
called, no live engine was booted and no pytest suite was run. The only
production-behaviour change is an **in-process monkeypatch** in the scratch
script that adds one defaulted `role_scores=None` argument to the existing
ranker; the congruence floor and strict-margin gate are the unmodified main
code. Gold is used only to score a pick after ranking.

Worktree: `C:/Users/Likhith/Documents/projects/ravana_rsi_symbolic` on
`experiment/rsi-symbolic-roles` from main `ca26488`. The two production files
used (`schema_induction.py`, `response_gen.py`) are byte-identical between
experiment 1's base `dbb3e78` and `ca26488` (`git diff` empty), so the
baseline is directly comparable.

## What was run

```
cd C:/Users/Likhith/Documents/projects/ravana_rsi_symbolic
PYTHONPATH=".../ravana_rsi_symbolic/ravana/src;.../ravana_rsi_symbolic/ravana_ml/src;.../ravana_rsi_symbolic" \
  C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe \
  scratch/role_binding_symbolic_experiment.py > scratch/role_binding_symbolic_run.log 2>&1
```

The script printed and asserted that `ravana.chat.schema_induction.__file__`
and `ravana.chat.response_gen.__file__` resolve to **this worktree**, not the
main editable install, and that the captured ranker is the unmodified
production function. `EXIT=0`.

Inputs (all read-only, from the main repo):

| Input | Use |
|---|---|
| `scratch/rsi_focus_live_rows.json` (384 rows) | the SAME recorded focused-live items as experiment 1: demonstrations, subject, options, gold |
| `scratch/rsi_teach_form_probe2.log` | recorded real-engine census: 8 trusted CAUSE edges |
| `data/ravana_glove_cache.npz` | engine's GloVe-64 cache, used only for the production baseline and the V4 tie-break |
| `scratch/role_binding_results.json` | experiment 1's recorded vector-role picks/scores (comparator) |

16 cases were replayed: 4 families × {near, far, foil, negative}, k=8, one
item per family per condition (`rsi_000` CAUSE, `rsi_001` PART_OF, `rsi_002`
PRECEDES, `rsi_005` RESEMBLES), same selection/dedup rule as experiment 1.

## Instrument validation (before any score)

* **Baseline reproduction:** the recomputed vector-only production baseline matches
  experiment 1's recorded baseline on **16/16** cases — the probe is stable.
* **Sanity control (F7):** a synthetic control in which the demonstrated r2
  tokens genuinely include a candidate gives symbolic scores
  `car=1.0, petal=0.0`, ranker pick `car` (right); a disjoint control
  (`petal` only) gives `None` (fail-closed). `passes: true`. The symbolic
  scorer/ranker wiring is live, so the nulls below are informative, not a
  wiring failure.

## Method

Each trusted demonstration is represented as the symbolic triple
`(relation_type, r1_label_tokens, r2_label_tokens)`; the query relation is the
demonstrated family, and the r2 role is the multiset of demonstration *target*
tokens. A candidate is scored by:

* **(a) relation-type match** — exact within a family, therefore a constant
  `1.0` for every candidate in a case (no candidate-discriminating power on
  its own; with a strict margin all candidates tie and the ranker emits `None`);
* **(b) r2-token overlap** — fraction of the candidate's tokens present in the
  r2 role multiset;
* **(c) V4 only: a small GloVe fallback used ONLY to break an exact top tie**
  (a `1e-6` nudge to the highest `cos(subject, candidate)` among tied leaders);
  it can never override a strict symbolic winner.
* **V3/V4 add an identity-free overlap**: max character-3gram Jaccard of the
  candidate against any demonstrated r2 token — no word identity, no vectors.

Every arm is fed to the **same unmodified production ranker** through the
`role_scores` argument. Arms are reported under two floor treatments:
`*_prodfloor` keeps the production congruence floor (min GloVe cosine over
demo pairs), exactly as experiment 1 did; `*_symfloor` rescales the floor to
the weakest within-r2-role symbolic similarity, to give the non-vector scores
a fair chance. The strict margin (`min_margin=0`) is unchanged everywhere.

Trust model is identical to experiment 1: CAUSE uses the 8 actually recorded
trusted engine edges; PART_OF/PRECEDES/RESEMBLES use synthetic trusted edges
from the recorded demonstrations (post-write-fix model, not backed by a
recorded live census for those families); negated demonstrations write no
trusted edges in either baseline or symbolic arms.

## Result — per-condition totals (correct out of 4; foil/negative correct = reject)

| Arm | near | far | foil (reject) | negative (reject) |
|---|---|---|---|---|
| production baseline (recomputed) | 2/4 | 0/4 | 0/4 | 4/4 |
| exp-1 vector role centres (recorded) | 1/4 | 1/4 | 0/4 | 4/4 |
| V1 exact relation only — prod floor | 0/4 | 0/4 | 4/4 | 4/4 |
| V2 relation + r2-token overlap — prod floor | 0/4 | 0/4 | 4/4 | 4/4 |
| V3 + identity-free overlap — prod floor | 0/4 | 0/4 | 3/4 | 4/4 |
| V4 full (GloVe tie-break) — prod floor | 0/4 | 0/4 | 3/4 | 4/4 |
| V1 exact relation only — sym floor | 0/4 | 0/4 | 4/4 | 4/4 |
| V2 relation + r2-token overlap — sym floor | 0/4 | 0/4 | 4/4 | 4/4 |
| V3 + identity-free overlap — sym floor | 2/4 | 0/4 | 1/4 | 4/4 |
| V4 full (GloVe tie-break) — sym floor | 2/4 | 0/4 | 0/4 | 4/4 |

**Foil leakage** (of 4 foils that should be rejected but were not):
production baseline 4, exp-1 vector 4, V1/V2 0, V3/V4 prod-floor 1 (PART_OF
`frost`), V3 sym-floor 3 (CAUSE `mist`, PART_OF `frost`, RESEMBLES `frost`),
V4 sym-floor 4 (adds PRECEDES `breeze` from the GloVe tie-break).

**Negative rejection is 4/4 for every arm** — but this is a gate property
(negated demos admit no hypothesis, so the ranker returns `None`), not a
matcher success.

### Key result: PART_OF / PRECEDES near and far

| Arm | PART_OF/PRECEDES near+far correct |
|---|---|
| production baseline | 0/4 |
| exp-1 vector role | 1/4 (PRECEDES far, accidental `ceremony`) |
| V1, V2, V3, V4 under EITHER floor | **0/4** |

**No symbolic variant moves PART_OF or PRECEDES near or far off zero.**

### Per-case near/far picks (gold in parentheses)

| Family | Cond | Subject | Gold | baseline | exp-1 vector | V3 sym-floor |
|---|---|---|---|---|---|---|
| CAUSE | near | thaw | mist | mist ✅ | mist ✅ | mist ✅ |
| CAUSE | far | role | ceremony | election ❌ | None ❌ | team ❌ |
| PART_OF | near | thunder | tempest | chill ❌ | chill ❌ | None ❌ |
| PART_OF | far | custom | contract | ceremony ❌ | habit ❌ | None ❌ |
| PRECEDES | near | cloud | typhoon | glow ❌ | glow ❌ | glow ❌ |
| PRECEDES | far | election | ceremony | law ❌ | ceremony ✅ | role ❌ |
| RESEMBLES | near | drizzle | chill | chill ✅ | blizzard ❌ | chill ✅ |
| RESEMBLES | far | market | guild | election ❌ | election ❌ | None ❌ |

The only near hits for any symbolic arm (V3/V4 sym-floor: CAUSE `mist`,
RESEMBLES `chill`) are **cases the production baseline already got right**,
and they arise from incidental character-trigram coincidences with a demo
target (`mist`~`tempest` via `est`/`st `; `chill`~`squall` via `ll `), not from
a relation. They buy no new correct item and cost 3–4 foil admissions.
This is the same overfitting signature experiment 1 found with GloVe, now
reproduced by a non-vector feature.

## What worked / what failed

* **Worked:** the instrument is validated (sanity control passes, baseline
  reproduces 16/16). Negatives are safely rejected by all arms. V1/V2 with
  the production floor fail closed everywhere rather than confabulating.
* **Failed:** (a) exact relation type is constant inside a case and therefore
  cannot rank; (b) r2-token overlap is exactly zero on every candidate of
  every case, because in all 16 recorded items the gold candidate is a word
  that never appears as a demonstrated r2 token — the transfer target is
  deliberately unseen; (c) the identity-free character-ngram substitute only
  produces orthographic accidents, which flip the two already-correct near
  cases and admit foils; (d) the GloVe tie-break makes foil leakage worse
  (V4 sym-floor leaks 4/4 foils) without fixing PART_OF/PRECEDES.
* Falsifier **F3 is reinforced, not overturned**: the problem is not
  specifically GloVe. Token identity and subword identity carry no relational
  information either, because the roles are filled by unseen words.
  **F5 warning confirmed at the ranking primitive**: foil picks changed and in
  the sym-floor arms substantially increased. **F7 passed**, so these are
  informative nulls. n=1 per family/condition remains hypothesis-level
  evidence, not a suite verdict.

## Recommendation

**Do NOT implement.** No symbolic variant fixes the stated gap: any arm that
keeps foil/negative rejection intact leaves PART_OF/PRECEDES near and far at
0/4, and the only arms with any near hits obtain them from orthographic
accidents on already-correct items and leak 3–4 of 4 foils. This is a strict
regression against the design's "no regression / no foil leakage" requirement
and is worse than the production baseline on foils.

**Mechanism still missing:** a *typed, domain-general role slot* — a world-model
category shared by demo targets and unseen candidates (e.g. "ceremony" and
"rainbow" both instantiating the CAUSE effect slot, "contract"/"tempest" the
PART_OF whole slot, "guild"/"blizzard" the RESEMBLES similar slot) — with a
rule that admits a candidate of that type even though its token identity is
unseen. Neither a token multiset, nor subword overlap, nor a GloVe centroid
supplies it.

**Data limitation — the recorded cases cannot fully settle the positive
question.** The demos and candidates are disjoint vocabularies (weather demos
→ civic words in every "far" item, and unseen weather words in "near"), and
the recorded rows contain no type/category labels, ontology links or features
from which such a slot could be grounded. So this experiment **definitively
rejects the three tested symbolic signatures**, but it cannot demonstrate that
a properly typed role-slot matcher would work until the recorded cases carry
(or an external world-model supplies) that type information. The next scratch
step is to define such a slot explicitly and test it on the same 16 cases —
still with no production edits.

## Files

* `scratch/role_binding_symbolic_experiment.py` — script (scratch only)
* `scratch/role_binding_symbolic_results.json` — full per-case scores, floors,
  picks, margins-adjacent data and the 16/16 reproduction check
* `scratch/role_binding_symbolic_run.log` — captured stdout
* this document
