# H_bridge — a non-leaking input representation that supplies transferable type evidence

**Status: scratch experiment + design doc. NO engine boot, NO `process_turn`,
NO store write, NO network, NO production edit, NO gate/threshold/item/gold
change.** Worktree `C:/Users/Likhith/Documents/projects/ravana_type_bridge`,
branch `experiment/ontology-type-bridge` from `main` `6a6fef6`. The three
modules imported (`scripts/tra_k_items.py`,
`scratch/trak_channel_from_substrate.py`, `scripts/front_gates.py`) were read
only; their newline-normalised SHA-256 was compared against the `main` git
blobs and is identical (`55a68182…`, `3d5a611f…`, `e3e37130…`). No file under
`ravana/src/`, `scripts/`, `tests/`, `weights/` or any item bank was touched.

## 1. The question this answers

Seven prior attacks scored **item similarity** — GloVe role centroid, symbolic
token/role signature, relational vector offset, category centre, candidate-side
type lexicon, substitution-plus-coherence. All were chance-level. The decisive
audit (`RELATIONAL_OFFSET_TRANSFER_EXPERIMENT.md` §3 E3) is that the benchmark
hash-picks the gold and **assigns its type by fiat**; the query subject's own
authored type set contains the taught role label **0/80** times while the gold
candidate's contains it **80/80**. No distributional, structural or graph
property of the subject can recover a label that was never in its input. Those
experiments therefore did not find a weak scorer; they located an
**input-information deficit**.

This experiment does not propose an eighth similarity metric. It asks a narrower
and answerable question:

> Is there a **non-leaking input representation** that supplies type evidence
> at all — even as a low-coverage precision route — and is that evidence
> **transferable** in the specific sense that it is corroborated by the taught
> demonstrations rather than being a bare lookup of the answer?

## 2. Brain mechanisms and evidence grade

| # | Finding | Source / locator | Grade | Computational implication here |
|---|---|---|---|---|
| N1 | A new word's type comes from a **subject-specific** contrastive input, not from other exemplars: `chromium` was a *colour* word because of "not the red tray, the chromium one" | Carey & Bartlett (1978), *Papers & Reports on Child Lang. Dev.* 15:17–29 (no DOI; primary PDF `languagelog.ldc.upenn.edu/myl/Acquiring_a_Single_New_Word.pdf`; catalogued by ERIC ED198703). Confirmed in Carey (2010), *Lang. Learn. Dev.*, PMC3101498 | **[S] human behavioural** | the query subject must have its **own** type input; a demo-side or candidate-side lexicon cannot supply it |
| N2 | Category-marking cues must be **in the input** for generalisation to novel category members | Lany & Saffran (2010), *Psychol Sci* 21(2):284–291, doi:10.1177/0956797609358570 | **[S] human behavioural** | same as N1; explains why the DEMO-only arm must fail |
| N3 | Sparse induction works only over a **pre-existing structured hypothesis space**; a few co-occurrences are diagnostic only against a prior type structure | Xu & Tenenbaum (2007), *Psychol Rev* 114(2):245–272, doi:10.1037/0033-295X.114.2.245 | **[S] human behavioural / computational** | the external ontology supplies the structured space; convergence of two sources is meaningful only inside it |
| N4 | One-trial assimilation becomes durable only if a **neocortical schema already exists** | Tse, Langston, Kakeyama, Bethus, Spooner, Wood, Witter & Morris (2007), *Science* 316(5821):76–82, doi:10.1126/science.1135935 | **[S] neurobiological (rat)** | the taught demonstrations provide the schema; the subject's external fact is the one-trial input assimilated into it |
| N5 | Item memory and **source memory dissociate**; schema-congruence is also a principal source of false memories | Johnson, Hashtroudi & Lindsay (1993), *Psychol Bull* 114(1):3–28, doi:10.1037/0033-2909.114.1.3 | **[S] human behavioural** | a schema may **propose** a candidate but only an actual source may **admit** it; convergence, not congruence alone, is the decision rule |
| N6 | Complementary learning systems: fast episodes + slow regularity extraction | McClelland, McNaughton & O'Reilly (1995), *Psychol Rev* 102(3):419–457, doi:10.1037/0033-295X.102.3.419 | **[S] framework** | keep demonstrations as distinct traces and the ontology as the slow structure |

**Evidence-type note.** N1–N3 are human behavioural; N4 is rodent neurobiology;
N5 is human behavioural; N6 is a computational/framework claim. The mapping to
a two-source convergence rule is an **extrapolation**, not a claim that the brain
computes set intersection over Wikidata facts. Following `neuroscience-to-architecture`
§4, the graded claim is: *the brain requires a subject-specific type input
(N1, N2), a structured prior space (N3, N6), and source-controlled assimilation
(N4, N5)*; the specific intersection implementation is the target system's own
primitive.

## 3. The proposed input representation (non-leaking)

Two independent sets of verbatim external ontology facts, both from the checked-in
CC0 Wikidata P31/P279 snapshot (`scratch/trak_substrate.json`, 980 facts, 140
covered words), projected by the unmodified adapter
`scratch/trak_channel_from_substrate.py`:

1. **Demo-side schema** — `is_a` facts whose subject is a *public* taught-exemplar
   object (`taught_exemplars`, no oracle field read).
2. **Subject-side evidence** — the admitted, full-provenance `is_a` fact for the
   **query subject** (`evidence_role='subject_type'`, from the declared TRA-K
   subject channel).

The decision representation is their **convergence**:

```
B(item, subject) = labels(subject-side external facts)
                 ∩ labels(demo-side external facts)
emit the unique label in B that appears in the displayed options; else abstain
```

No floor, margin, weight, ranking threshold or gate is introduced — the only
decision is *uniqueness inside the displayed option set*, and absence is a
fail-closed abstention. Anti-tautology property: `type_sets`, `world_model`, all
gold fields and the item bank's answer-key helpers are never read by any scorer;
this is **proved by an AST scan** of the three scorer function bodies in the
experiment (§4). The answer key is used only at item-construction time as an
*exclusion* filter, never as the source of an emitted label.

This is the "schemas propose; sources admit" rule of `ravana-debugging` §7a made
concrete: the demonstrations propose a type schema; the subject's own external
fact admits a label into it. Neither source is the answer key, and congruence
alone does not emit.

## 4. Instrument validated before any number was read

| Check | Result |
|---|---|
| Module SHA-256, worktree (LF-normalised) == `main` blob | `tra_k_items.py` `55a68182…`, `front_gates.py` `e3e37130…`, adapter `3d5a611f…` — all identical |
| Frozen constants set before run | `MAX_PER_ITEM=0`, `PERMUTE_OFFSET=137`, negation arm abstains — unchanged after every run |
| Item bank | `build_items()` 400 families; `validate_items()` **0 errors** |
| Channel | 44,986 external demo facts projected; 47,386 total channel facts |
| `channel_overlap()` | `ok=True`; `answer_key_leak=0`, `oracle_domain_tag_leak=0`, `provenance_missing=0`, `overlap_world_model=0` (7,796 `overlap_type_sets` are legitimate demo-word keys, reported not failed) |
| AST anti-leakage | **ok, 0 problems** across `subject_type_labels`, `demo_type_labels`, `pick` |
| Probe hand-check | printed input→output for six emitted rows, e.g. `tra_k_0006 near subject=gale options=[wind, party, natural phenomenon, intersection] demo includes wind, subject fact wind → pick=wind gold=wind`; all six MATCH |
| Script bug history | two instrument bugs (`by_subject` on the wrong loader; raw-byte SHA under CRLF) were fixed **before** the first valid run; the second fix left every endpoint byte-identical. No constant was changed after a result |

## 5. Evidence (measured; 400 families, 800 near+far typed queries)

| Arm | emit | coverage | correct | precision | foil reject | negative reject | unrelated reject | disjoint reject |
|---|---|---|---|---|---|---|---|---|
| DIRECT (subject fact only; upper bound) | 800/800 | 100% | 800 | 1.000 | 400/400 | 400/400 | 400/400 | 400/400 |
| **BRIDGE (new)** | **176/800** | **22.0%** | **176** | **1.000** | **400/400** | **400/400** | **400/400** | **400/400** |
| DEMO (demo facts only) | 194/800 | 24.25% | 10 | **0.052** | 313/400 | 400/400 | 400/400 | 400/400 |
| SUBJECT (alias of DIRECT) | 800/800 | 100% | 800 | 1.000 | 400/400 | 400/400 | 400/400 | 400/400 |

Chance on four options is 25%. Relevant contrasts:

- **E1 — the route fires and is exact.** BRIDGE emits on 176/800 and is right on
  all 176; it abstains entirely on the other 624. That is a **precision route**,
  not an aggregate improvement. Its G6 precision front is saturated and flagged.
- **E2 — the evidence is subject-anchored, not demo-side.** DEMO-only, with the
  exact same option sets, scores **10/194 = 5.2%**, *below* chance. Removing the
  subject's own external input destroys the signal. This is the N1/N2 prediction
  and rules out "the demonstrations alone already type the subject".
- **E3 — subject-specificity.** Rotating subject-side facts deterministically
  across items (offset 137) collapses emission from 176/800 to **7/800**
  (**96.02%** collapse). The picks are not a property of the option set or the
  domain.
- **E4 — information audit (oracle read only here).** The typed gold is the
  subject's own highest-credibility external label **800/800**. This is the
  honest boundary of the result, stated next.
- **E5 — controls.** Foil, unrelated, negated-teaching and disjoint-option arms
  reject 400/400 for BRIDGE. The negative arm abstains because the harness
  renders P2c channel facts as negated assertions, so no trusted type edge is
  written; the scorer models that surface (`NEGATION_ARM_ABSTAINS`), it does not
  relax a gate.
- **E6 — G6.** Unmodified `front_gates.gate_saturation` (non-blocking) FLAGs
  `tra_k_bridge_precision_pct`, `tra_k_direct_coverage_pct` and
  `tra_k_bridge_negative_reject_pct`. A FLAG is a caveat, never evidence.

## 6. Verdict — what this does and does not establish

**POSITIVE for the narrow claim.** A non-leaking, subject-specific external
ontology representation, restricted to labels corroborated by the taught
demonstrations, supplies type evidence that is **correct on every case it fires**
(176/176, 22% coverage), is **subject-dependent** (96% emission collapse under
permutation), and **fails closed** on every rejection control. This is the first
positive in this line after seven chance-level item-similarity results, and it
is obtained with no gate, threshold or item change and no engine boot.

**Honest limit — this is convergence, not new information.** Because TRA-K
*defines* the typed gold as the subject's own highest-credibility substrate fact
(`tra_k_items.typed_option_space._subject_label`), any route that reads that fact
is necessarily right when it emits. BRIDGE does **not** add a new information
source and does **not** beat DIRECT. What it adds is **corroboration**: a label
is emitted only when the taught demonstrations independently support the same
type, which is the operational form of schema assimilation (N4/N5). It must be
reported as a *precision/coverage* result, never as proof that RAVANA learned a
rule or that TRA is solved.

**Does not repair original TRA/RSI.** In original TRA the type is hash-picked and
assigned by fiat, the subject has no authored type support, and the candidate-side
real Wikidata type is unrelated to the role (`gold_has_substrate_role_type =
0/800` in a static audit). BRIDGE requires a legitimate subject-specific type
channel that original TRA does not provide. Under original TRA's input regime
abstention remains correct. The standing goal is unchanged; this experiment
identifies the representation that a legitimate knowledge-conditioned variant
must supply and measures exactly what it can and cannot do.

## 7. Falsifiers (pre-registered, measurable)

| # | Falsifier | Would mean |
|---|---|---|
| F1 | A subject-side input that is **not** externally sourced (e.g. the item bank's `type_sets`) also yields 100% precision | leakage, not transfer; re-run the AST and overlap audits |
| F2 | DEMO-only arm exceeds 50% precision on the same option sets | the effect is a demo/candidate-side lookup, not subject-anchored |
| F3 | Permutation of subject facts collapses emission by < 50% | the route is option-set driven; N1/N2 violated |
| F4 | Increasing channel breadth (more deployable demo facts) does **not** raise coverage above 22% | the ceiling is option construction or substrate sense coverage, not channel breadth |
| F5 | Any BRIDGE emit on foil / unrelated / disjoint / negated conditions | fail-closed property broken; withdraw the mechanism |
| F6 | On a frozen corpus channel containing the test word, definition-based cues recover the type while the ontology route does not | ontology is not the required input; switch to N2-style distributional evidence |

## 8. Implementation design — documented, NOT implemented

1. **Input.** Keep the external ontology channel as a pluggable, provenance-stamped
   file (`TRAK_CHANNEL_JSON`); no production reader changes.
2. **Primitive.** Add a pure helper `bridge_type_labels(subject_facts, demo_facts)`
   returning the intersection, default-OFF behind the existing `RAVANA_TRA_SLOTS`
   flag. Prove flag equivalence first (byte-identical outputs off vs on).
3. **Wiring.** Emit at the answer-selection gate only as an endorsement of a
   label already supported by both sources; never override a direct trusted
   typed edge, never override foil/negative rejection. Fail open on any doubt.
4. **Measurement before enabling.** Positive + negative endpoints, subject
   independence, disjoint/unrelated controls and G6 on identical state. Keep
   OFF unless coverage rises above the pre-registered 22% with all controls flat.
5. **No gate or threshold change.** Abstention is the decision; there is no
   floor or margin to tune. Do not cite the saturated DIRECT arm as capability.

## 9. Caveats

- n = 400 families / 800 typed queries on the **frozen bank**, a static mechanism
  replay. No engine, no cold reload, no Wilson CI, no k-curve; not a suite verdict.
- The substrate contains sense noise (e.g. `tempest → Python package`,
  `shower → industrial plant`) and only one-hop P31/P279 facts. Coverage 22% is
  bounded by that snapshot and by option construction; F4 tests the bound.
- The typed gold is defined from the subject fact, so precision is partly
  tautological (§6). The transferable content is the *demo corroboration*, not
  the label lookup.
- The negation control is modelled from the harness's own P2c surface rule, not
  from a live `process_turn`; in the engine the assertion writer handles it.
- No production PR, no default-on flag, no TRA/RSI PASS claim. G6 FLAG is
  non-blocking and is not evidence.

## 10. Files

- `scratch/ontology_type_bridge_experiment.py` — 4 arms, all controls, AST
  anti-leakage, G6, information audit, hand-check rows; no engine
- `scratch/ontology_type_bridge_results.json` — 800-row bridge audit, tallies,
  permutation control, module SHAs, G6
- `scratch/ontology_type_bridge_run.log` — captured stdout, `EXIT=0`
- this document
