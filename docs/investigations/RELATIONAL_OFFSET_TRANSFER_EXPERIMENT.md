# H_offset — brain-grounded relational offset transfer: NEGATIVE result, new precise finding

**Status: scratch only. NO engine boot, NO `process_turn`, NO store write, NO production edit,
NO gate/threshold/item/gold change.** Worktree
`C:/Users/Likhith/Documents/projects/ravana_rsi_transfer`, branch
`goal/relational-offset-transfer` from main `0829813d2666f86717a8890b779d2bd2ebcb0185`.
No file under `ravana/src/`, `scripts/`, `tests/`, `weights/` or any item bank was touched.
The three modules imported (`scripts/tra_items.py`,
`ravana/src/ravana/ontology/attribute_encoder.py`, `scripts/front_gates.py`) were read-only;
their SHA-256 was asserted identical between main and worktree before any number was read.

## 1. The new hypothesis (not a repeat of the three prior negatives)

Prior negative results in this line all scored **item similarity**:

| Prior work | Surface | Outcome |
|---|---|---|
| `RSI_ROLE_BINDING_EXPERIMENT.md` | GloVe cosine to a single role label | near 1/4, far 1/4 |
| `RSI_SYMBOLIC_ROLE_EXPERIMENT.md` | symbolic token/role signature | 0 transfer (V3/V4 2/4 only on already-correct items, 3–4 foil leaks) |
| `TRA_CATEGORY_CENTRE_EXPERIMENT.md` | mean of taught r2 vectors (centroid) | near 1/40, far 1/40; centres in a 0.73–0.98 cosine cone |
| `GOAL_UNSEEN_TYPE_TRANSFER.md` H_b | candidate-side broad type lexicon | 7/80; **subject-dependent picks 0/80** |

**H_offset is a different computational primitive.** Structure-mapping theory says analogy
maps *relations*, not object attributes; the vector analogue is to learn a relation vector
from the taught pairs and *apply it to the query subject*, rather than compare the options
to a prototype of taught r2 tokens:

```
W     = unit( mean_i [ unit(v(r2_i)) - unit(v(r1_i)) ] )   # learned relation operator
pred  = unit( unit(v(subject)) + ALPHA * W )               # subject-specific predicted slot
score(o) = cos(unit(v(o)), pred)
```

This is **subject-dependent by construction**. That is the property the prior mechanisms
lacked and the property the earlier subject-independence control was designed to detect.

### Brain grounding (graded; web-verified this session)

| Finding | Source | Grade | What it licenses here |
|---|---|---|---|
| Analogy maps relations between objects, not attributes; mapping rules are content-independent | Gentner (1983), *Cognitive Science* 7(2):155–170, doi:10.1207/s15516709cog0702_3 ([Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog0702_3), [PsycNET](https://psycnet.apa.org/record/1984-14564-001)) | **S** (cognitive theory) | learn an r1→r2 operator, not an r2 prototype |
| Hippocampal relational network supports flexible, inferential expression in novel situations | Eichenbaum (2004), *Neuron* 44(1):109–120, doi:10.1016/j.neuron.2004.08.028 ([PubMed 15450164](https://pubmed.ncbi.nlm.nih.gov/15450164/)) | **S** (neuro review) | apply a learned relation to an unseen subject |
| Complementary learning systems: fast episodes + slow regularity extraction | McClelland, McNaughton & O'Reilly (1995), *Psychol Rev* 102(3):419–457, doi:10.1037/0033-295X.102.3.419 | **S** (framework) | keep demonstrations as distinct traces, write nothing |
| Pattern separation keeps similar episodes distinct | Yassa & Stark (2011), *Trends Neurosci* 34(10):515–525, doi:10.1016/j.tins.2011.06.006 | **S** (neuro) | motivates the `INSTANCE_MAX` arm |
| Difference-vector / 3CosAdd analogy; BART relation vectors | Rogers et al. (2017) 3CosAdd; Lu et al. (2019) BART, *PNAS*, doi:10.1073/pnas.1814779116 ([PNAS](https://www.pnas.org/doi/10.1073/pnas.1814779116)); Chen et al. (2017), *CogSci* parallelogram evaluation | **[M] computational modelling only** | supplies the *implementation shape* of the offset; **not brain evidence** |

The offset itself is an **extrapolation**: Gentner/Eichenbaum license "map a relation",
but no cited source says the brain performs a GloVe difference vector. This is stated as
such, following `neuroscience-to-architecture` §4 and the ledger convention in
`RSI_ROLE_BINDING_DESIGN.md` §1.

## 2. Instrument validated before any number was read

| Check | Result |
|---|---|
| Production module SHA-256 (main == worktree) | `tra_items.py` `7bc55db8637fb4b2…`, `attribute_encoder.py` `295610726018e5f8…`, `front_gates.py` `3ede0efc75d582e9…` — all identical |
| Frozen constants (set in script before run) | `FLOOR=0.20`, `MARGIN=0.10`, `ALPHA=1.0`, `ALT_SUBJECT='zeppelin'`, disjoint set fixed |
| Item bank | `build_items()` = 400 families, `validate_items()` = 0 errors; manifests recorded |
| Recorded rows | 13,440 total; 12,320 fake excluded; **1,120 live**; 160 graded rows (40 items × 4 conditions, `phase=query`, `arm=structural`, non-permuted) |
| Bank cross-check | all 160 subjects/candidates/golds match the regenerated bank: 0 mismatches |
| GloVe | 400,001 words × 64 D via unmodified `build_glove64_lookup`; all 80 gold words, all subjects and all candidates present; **0 OOV** |
| Probe hand-check | printed input→output for near/far/foil/negative; e.g. `tra_0000 near OFFSET subj=tempest cands=[squall,shower,breeze,sunset] gold=sunset → pick=None reason=strict_margin_not_met` |
| No leakage | `type_sets`/`world_model`/gold never enter any scorer; read only to grade and to audit information availability |

**Teaching-surface caveat:** the recorded live rows carry empty `teaching_surfaces` (the
recorder dropped them), so the teaching pairs were reconstructed deterministically from
`bank[item].taught_exemplars[:k]`, as in `GOAL_UNSEEN_TYPE_TRANSFER.md` §8. This is a
replay of recorded items, not a live `process_turn` transcript.

## 3. EVIDENCE (measured, n = 40 items × 4 conditions = 160 rows per arm)

| Arm | near | far | foil (reject) | negative (reject) | near+far |
|---|---|---|---|---|---|
| **OFFSET (new)** | **1/40** | **1/40** | 23/40 | 40/40 | **2/80** |
| INSTANCE_MAX (pattern separation) | 3/40 | 1/40 | 15/40 | 40/40 | 4/80 |
| CENTROID (prior negative, reproduced) | 4/40 | 4/40 | 13/40 | 40/40 | 8/80 |
| SUBJECT_COS (item-similarity control) | 4/40 | 0/40 | 30/40 | 40/40 | 4/80 |

Primary/sealed split is 20/20 items; no arm clears the TRA `near >= 0.75` rule.

### E1 — the new mechanism genuinely reads the subject (the prior line did not)

Same case, same candidates, subject replaced by the unrelated token `zeppelin`:

| Arm | picks emitted (near+far) | pick **changed** | pick identical |
|---|---|---|---|
| **OFFSET** | 20 | **18** | 2 |
| INSTANCE_MAX | 21 | 5 | 16 |
| CENTROID | 24 | **0** | 24 |
| SUBJECT_COS | 23 | 22 | 1 |

This is the first mechanism in this line that is demonstrably subject-dependent while
using only teaching-derived, deployable information (contrast: candidate-side H_b 0/80,
CENTROID 0/24). **It is not sufficient for transfer.**

### E2 — the failure is not the gate: the signal is absent

Removing floor and margin entirely and taking raw top-1 over the 80 near+far cases:

| Arm | raw top-1 correct | gold rank histogram (1/2/3/4; 12 k=0 cases have no scores) |
|---|---|---|
| OFFSET | 18/80 | 18 / 20 / 16 / 14 |
| INSTANCE_MAX | 19/80 | 19 / 15 / 18 / 16 |
| CENTROID | 14/80 | 14 / 17 / 21 / 16 |
| SUBJECT_COS | 16/80 | 16 / 14 / 20 / 18 |

Chance with four options is 25% (≈17 of 68 scored cases). Every arm is at or below chance
and the gold rank is approximately uniform. The strict-margin veto is therefore **not**
what blocks transfer; there is no discriminating score to veto.

### E3 — information audit (ORACLE `type_sets` used to audit, never to score)

- query subject's own authored type set contains the taught `type_label`: **0/80**
- gold candidate's authored type set contains it: **80/80**
- all 80 gold words have GloVe vectors: **80/80**
- 6 of 40 items are `k=0` (no teaching by construction)

The benchmark assigns the gold by fiat: `tra_items.py::_split_words` hash-picks the gold
from a reserved set, then `_split` writes `type_sets[gold] = [type_label, domain-event]`
and `type_sets[distractors] = [domain-other]`. There is no distributional, structural or
graph property of the gold word that makes it an "effect"/"whole"/"earlier-event". This is
why relative rank is flat in E2 and why H_offset cannot convert subject-dependence into
accuracy.

### E4 — boundary controls and saturation

- Disjoint options `apple/banana/chair/table` → `None` for all four arms (PASS).
- OOV subject → `None` for all arms (PASS).
- All-negated teaching → `None` (`no_admitted_trace_negated`) for all arms (PASS as a
  scratch proxy; in the real engine negation is handled by the assertion writer, which
  writes no trusted edge — this replay only sees the reconstructed surface).
- Foil rejection is **not clean** for any arm: OFFSET leaks 17/40 foils (better than
  CENTROID's 27/40, worse than SUBJECT_COS's 10/40). Negative 40/40 is negation-driven,
  not selective rejection.
- G6 (`front_gates.gate_saturation` unmodified, non-blocking): every arm FLAG
  (`tra_near` 2.5–10.0 in the low band; `tra_negative_reject=100`). FLAG is a caveat,
  never evidence (G6, `ravana-debugging` §1).
- `ALPHA` sweep `{0,0.5,1,2,4}` is reported as a **diagnostic only**: best near+far is
  4/80 at `ALPHA=0` (the plain subject-cosine baseline). No value was selected after
  seeing a result; the headline uses the pre-registered `ALPHA=1.0`.

## 4. Verdict

**Negative for the standing goal. Do not implement.** The exact finding:

> A brain-grounded relational transformation (learned r1→r2 offset applied to the query
> subject) is the first mechanism in this line that is genuinely **subject-dependent**
> (18 of 20 emitted picks flip when the subject is replaced by `zeppelin`, versus 0 of 24
> for the GloVe centroid). It still achieves only **2/80** near+far transfer, and with
> floor and margin removed its raw top-1 is **18/80 ≈ chance** with an approximately
> uniform gold-rank histogram. Subject-dependence is therefore achievable but carries no
> task-relevant information: the benchmark's gold type is oracle-fiat, present on the
> candidate side 80/80 times and absent from the subject 0/80 times. The remaining gap is
> an **input-information deficit upstream of articulation**, not a ranking, threshold,
> gate or binding defect.

This extends the three prior negatives from *item-similarity* representations to a
*relational-transformation* representation, closing that class of explanation without
lowering any gate. It does not establish that typed transfer is impossible in RAVANA; it
establishes that no signal available in this benchmark's input distinguishes the gold
from the distractors, at any threshold.

**What would have to change (not a mechanism tweak):** a legitimate, non-oracle
subject-specific input channel — an ostensive/contrastive turn about the test word (which
makes it taught, not unseen), frozen corpus snapshots containing the test token with
type-marking frames, or an independently authored type hierarchy connected to the
*subject*, not merely to the options. These are the same three requirements as
`GOAL_UNSEEN_TYPE_TRANSFER.md` §7; this experiment shows the relational mechanism does
not remove the need for them.

## 5. Honest caveats

- n = 40 recorded items (20 primary / 20 sealed), 4 conditions each — a static mechanism
  replay, not the 400-family benchmark; no Wilson CI, k-curve or cold-reload endpoint.
- Teaching surfaces are reconstructed from the frozen bank; the recorder dropped them.
  The negation channel is a scratch surface-level proxy, not the engine's assertion writer.
- GloVe64 is a projected 64-D space; a different embedding or a learned relation operator
  (BART-style) could behave differently. This rules out this offset in this space only.
- No production PR, no default-on flag, no TRA PASS claim. G6 FLAG is non-blocking.
- All retracted runs from prior experiments are untouched; this experiment had one
  instrument defect (a dict used as a dict key in `INSTANCE_MAX`, fixed before the first
  valid run) and no post-result constant change. Run log: `EXIT=0`, `errors` none.

## 6. Files

- `scratch/relational_offset_experiment.py` — 4 arms, controls, G6, info audit, handcheck; no engine
- `scratch/relational_offset_results.json` — 160 rows per arm + tallies, subject-independence, controls, audit
- `scratch/relational_offset_run.log` — captured stdout, `EXIT=0`
- this document
