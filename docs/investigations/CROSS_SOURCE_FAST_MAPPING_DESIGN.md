# H_lang — non-tautological type transfer from real-language evidence (RSI/TRA)

**Status: scratch design + static probe, NEGATIVE for the pre-registered discriminating
criterion. NO engine boot, NO `process_turn`, NO store write, NO network, NO production
edit, NO gate/threshold/item/gold change.** Worktree
`C:/Users/Likhith/Documents/projects/ravana_xsrc_transfer`, branch
`experiment/cross-source-type-transfer` from `main` `51eee3e7d313ee9bfd86e2659e23c1daa030c565`.
The three inputs read (`scripts/tra_k_items.py`, `scripts/front_gates.py`,
`scratch/trak_substrate.json`) were verified byte-identical to their `main` blobs
after LF-normalisation before any number was read. No file under `ravana/src/`,
`scripts/`, `tests/`, `weights/` or any item bank was touched.

---

## 1. What was wrong with the 176/176 bridge, and what must change

`ONTOLOGY_TYPE_BRIDGE.md` measured 176/176 precision at 22% coverage, but TRA-K
**defines** the typed gold as the subject's own highest-credibility substrate fact
(`tra_k_items.typed_option_space._subject_label`). The `DIRECT` arm reads that same
fact and scores 800/800. The bridge therefore cannot distinguish *transfer* from
*lookup*: correctness is guaranteed by construction whenever it fires. It is a
corroboration finding (demo schema agrees with the subject's own fact), not a new
information source.

A non-tautological transfer experiment needs three properties the bridge lacked:

1. **The scorer's subject-side input is not the answer-bearing record.** Here it is
   the verbatim Wikidata *item description* — real natural-language text — and the
   scorer never receives the subject's structural `type_label` at all (AST-enforced).
2. **The answer key is a relation computed by the grader from facts the scorer
   cannot read.** Here: *which displayed candidate shares the query subject's
   primary Wikidata type*. Neither item-bank `gold`, `type_sets`, `world_model`,
   `typed_queries` nor the item's authored `role`/`type_label` is consulted.
3. **Correctness must be able to fail.** Because the key depends on the subject's
   structural type while the scorer sees only its description, a language cue that
   is wrong or absent produces a wrong pick or an abstention. The oracle upper bound
   (direct subject-type lookup) is reported separately and labelled non-deployable.

This is a **new benchmark variant** of TRA-K, not a claim about original TRA/RSI.
A frozen-corpus audit (`TRAK_CORPUS_AUDIT.md`) already returned **NO-GO** for
language evidence on the original items (0/40 items with a corpus assertion for the
required label), so real-language evidence is supplied here from the CC0 Wikidata
item-description field instead, and the graded label vocabulary is replaced by the
ontology's own type vocabulary. Do not cite this as an original-TRA score.

## 2. Brain mechanisms (web-verified; evidence graded)

| # | Finding | Source / locator | Grade | Computational mapping here |
|---|---|---|---|---|
| N1 | A new word's type comes from a **subject-specific contrastive input**, not from other exemplars: `chromium` was a *colour* word because of "not the red tray, the chromium one" | Carey & Bartlett (1978), Stanford Papers & Reports on Child Lang. Dev. 15:17–29; primary account re-verified in Carey (2010), *Lang. Learn. Dev.* 6(3):241–264, doi:10.1080/15475441.2010.484379 ([PMC3101498](https://pmc.ncbi.nlm.nih.gov/articles/PMC3101498/)) | **[S] human behavioural** | the query subject's **own definition text** must be an input; a demo-side schema alone cannot type it |
| N2 | Fast mapping forms a usable lexical entry from one exposure when the referent/context is present; adults often retain one hypothesis | Heibeck & Markman (1987), *Child Dev* 58(4):1021–1034, doi:10.2307/1130543; Medina, Snedeker, Trueswell & Gleitman (2011), *PNAS* 108:9014–9019, doi:10.1073/pnas.1105040108 | **[S] human behavioural** | one description per subject is a legitimate teaching event; no repeated co-occurrence is required |
| N3 | The principle of contrast: a form difference marks a meaning difference; a novel label is assigned to a *gap* relative to known labels | Clark (1987), *The principle of contrast: a constraint on language acquisition*, in MacWhinney (ed.), *Mechanisms of Language Acquisition*, Erlbaum, pp. 1–33 ([PDF](https://faculty.wcas.northwestern.edu/robvoigt/courses/2021_fall/ling334/readings/Clark1987contrast.pdf)) | **[S] cognitive theory** | the displayed options are contrasted alternatives; a cue that also fits several options must **not** emit |
| N4 | Generalisation to novel category members requires **category-marking cues in the input** | Lany & Saffran (2010), *Psychol Sci* 21(2):284–291, doi:10.1177/0956797609358570 | **[S] human behavioural** | the subject description, not the candidate set, must carry the type-marking words |
| N5 | Sparse induction is diagnostic only against a **pre-existing structured hypothesis space** | Xu & Tenenbaum (2007), *Psychol Rev* 114(2):245–272, doi:10.1037/0033-295X.114.2.245 | **[S] behavioural / computational** | demonstrations provide the structured type space; the ontology supplies candidate types |
| N6 | One-trial assimilation becomes durable only when a **neocortical schema already exists** | Tse, Langston, Kakeyama, Bethus, Spooner, Wood, Witter & Morris (2007), *Science* 316(5821):76–82, doi:10.1126/science.1135935 | **[S] neurobiological (rat)** | demo-derived cue→type map is the schema into which the new description is assimilated |
| N7 | Item memory and source memory dissociate; schema-congruence is also a principal source of false memories | Johnson, Hashtroudi & Lindsay (1993), *Psychol Bull* 114(1):3–28, doi:10.1037/0033-2909.114.1.3; Dautriche, Goupil, Smith & Rabagliati (2021), *Open Mind* 5:1–19, doi:10.1162/opmi_a_00038 (toddlers withdraw words from an unreliable source) | **[S] human behavioural** | a schema may **propose** a candidate but only the subject's own observed text may admit it; permutation and source-invalidation controls are mandatory |
| N8 | Complementary learning systems: fast episodes + slow regularity extraction | McClelland, McNaughton & O'Reilly (1995), *Psychol Rev* 102(3):419–457, doi:10.1037/0033-295X.102.3.419 | **[S] framework** | keep demonstrations as separate teaching traces; the learned cue map is the slow regularity |

**Evidence-type note (per `neuroscience-to-architecture` §4).** N1–N5, N7 are human
behavioural/cognitive; N6 is rodent neurobiology; N8 is a computational framework.
The specific rule implemented below — count definition content words co-occurring
with taught types in ≥2 demonstrations — is an **extrapolation**, not a claim that
the brain counts word co-occurrences over Wikidata. The graded claim is only:
*subject-specific input + pre-existing schema + source-controlled admission*.

## 3. The experiment (H_lang)

**Inputs (deployable).** For each TRA-K item, the public question supplies the query
subject and four displayed candidates (`unseen_test_words[near|far].candidates`);
the external CC0 Wikidata snapshot supplies (a) the subject's item description
(`text_evidence`, P279, primary sense) — **text only**, (b) each taught demo word's
description and its primary type facts, (c) each candidate's primary type facts.
No item-bank oracle field reaches a scorer.

**Learned schema.** From the taught demonstrations, build `word -> {type: count}`
for description content words (stopwords frozen in the script).

**Decision rule (no thresholds tuned after a result).**
```
cues(subject)  = { type : support >= 2 from the subject's own description words }
pick           = the UNIQUE displayed candidate whose ontology types intersect cues
else            abstain
```
`MIN_DEMO_SUPPORT = MIN_CUE_SUPPORT = 2`, `PERMUTE_OFFSET = 137`, hash prefix
`RAVANA-LANG-FASTMAP-v1`, stopword list — all frozen in the script before the run.
Uniqueness is the only gate; no floor, margin, weight, ranking or confidence.

**Answer key (grader-only).** `key = the unique candidate sharing the subject's
primary Wikidata type`; cases without a unique key are logged but not scored
(125/800 gradeable). The scorer cannot compute this: it never receives the subject
type set. Non-deployable oracle (direct subject-type → candidate) is **not** scored
as capability because it is trivially 125/125 by construction.

**Arms.** `TEXT` (proposed, subject description + demo schema), `DEMO` (schema only,
no subject text). **Controls.** subject-description permutation (offset 137),
answer-key permutation (every key replaced by a *different* candidate from its own
displayed set), negated teaching (empty cue map), disjoint options
`apple/banana/chair/table`, AST anti-leakage of all four scorer function bodies,
unmodified G6 saturation gate, bank validation, primary/sealed split.

**Anti-leak proof.** `learn_cue_map`, `cue_predict`, `schema_from_demos`, `pick` are
AST-scanned for the identifiers `gold`, `type_sets`, `world_model`,
`typed_queries`, `unseen_test_words`, `type_label`, `item_id`, `role`, and for any
import of the item-bank or gate modules. Result: **ok, 0 problems**. The subject's
type labels never enter a scorer as a parameter; only the subject's text does.

## 4. Hypothesis, pre-registered expectation, and observed result

**Hypothesis.** A subject-specific language cue assimilated into a demo-learned
schema emits the type-sharing candidate at precision above a demo-only schema
lookup, and correctness depends on the subject's own text rather than on a
candidate-side or answer-key lookup.

**Pre-registered discriminating result (POSITIVE iff all hold).** TEXT emits ≥10;
TEXT precision ≥0.60; TEXT − DEMO precision ≥0.15 on the same gradeable cases;
subject-text permutation changes ≥80% of TEXT picks; gold permutation leaves picks
byte-identical while accuracy falls to ≤0.40; negated/disjoint arms emit 0.

**Measured (`cross_source_fast_mapping_results.json`, 800 cases → 125 graded,
70 primary):**

| Arm / control | All | Primary |
|---|---|---|
| TEXT (new) | 13 emits, **10/13 = 0.769** precision, 10.4% coverage | 4 emits, 3/4 = 0.750 |
| DEMO (schema only) | 21 emits, 15/21 = 0.714 | 14 emits, 8/14 = 0.571 |
| subject-text permutation | 12/13 picks changed (**0.923**) | — |
| answer-key permutation | picks byte-identical, all keys changed, accuracy **0.769 → 0.077** | — |
| negated teaching / empty subject text / disjoint options | 0 emits / 0 emits / 0 emits | — |
| AST anti-leak / bank validation / G6 | ok, 0 problems / 0 errors / PASS (non-blocking) | — |

**Verdict: NEGATIVE / NO-GO for the pre-registered criterion.** TEXT is genuinely
subject-anchored (0.923 permutation sensitivity), non-leaking (gold permutation
collapses accuracy while leaving picks unchanged), abstains on all anti-saturation
controls, and is above chance on its firing subset (10/13). But its margin over
DEMO is only **0.055**, far below the pre-registered 0.15; the demo schema is
already correlated with the answer because both are drawn from the same ontology.
Coverage is 10.4%, and 675/800 cases had no unique independently graded key.
Per the pre-registered rule this is **not** implemented and **not** cited as RSI/TRA
progress. Hand-checked rows are in the log (e.g. `tra_k_0130 tree`, text
"perennial woody plant", cue `plant life form`, options
`shrub/petal/blossom/foliage`, pick `shrub` = key).

**Instrument defects found and fixed before the valid run (reported for audit):**
(i) the first run crashed in the hand-check printer (`text`/`pick` key mismatch) after
all numbers had been computed — fixed; (ii) rotating keys across cases left 7.7%
of the original accuracy intact because candidate words recur across items — replaced
with a within-option-space relabel that provably changes every key; the result JSON
and log were regenerated from one clean `EXIT=0` run. No constant or threshold was
changed after the result was seen.

## 5. Falsifiers (pre-registered; each is measurable in the probe)

| # | Falsifier | What it would mean | Observed |
|---|---|---|---|
| F1 | Any scorer function references `gold`/`type_sets`/`world_model`/`typed_queries`/`type_label` | answer-key leakage, result withdrawn | **not triggered** (AST clean) |
| F2 | Gold-permutation leaves accuracy unchanged while picks are unchanged | scorer reads the key | **not triggered** (0.769 → 0.077) |
| F3 | Subject-text permutation changes <80% of picks | candidate-side lookup, not fast mapping | **not triggered** (0.923) |
| F4 | TEXT precision ≤ DEMO precision + 0.15 | the demo schema already contains the signal; no transfer increment | **TRIGGERED** (+0.055) — the decisive negative |
| F5 | TEXT precision ≤ 0.60 or emits <10 | unreplicated anecdote, not a precision route | not triggered (0.769, 13 emits) |
| F6 | Any emit on negated teaching, disjoint options, or absent subject text | fail-closed property broken | **not triggered** (0) |
| F7 | Primary and full-set precision differ by >0.20 | held-out pool overfitting | 0.750 vs 0.769 — not triggered, but primary n=4 is too small to conclude |
| F8 | A frozen corpus channel with the test word yields type cues where this route does not | wrong knowledge source; switch to corpus route | prior audit NO-GO (0/40), so not testable here |

## 6. What the negative result locates, and the next work item

The gap is **not** the scorer's subject-dependence or leak resistance — both pass.
It is that the demonstration schema and the graded answer are drawn from the same
ontology, so DEMO already captures most of the available signal. A future attempt
must make the answer key independent of the schema's source (e.g. grade with WordNet
hypernym membership while teaching types from Wikidata, or vice versa) *and* keep a
subject-specific language input. The exploratory cross-ontology variant already
checked statically: with WordNet hypernym keys and Wikidata type evidence it emits
on only 12/167 gradeable cases (6/12 correct), i.e. a coverage ceiling, not a
threshold problem. Neither result justifies a production change.

## 7. Files

- `scratch/cross_source_fast_mapping_experiment.py` — static probe, 2 arms + 5 controls, AST leak scan, G6, hand-check
- `scratch/cross_source_fast_mapping_results.json` — 800-case audit, tallies, controls, scorer sources, SHAs
- `scratch/cross_source_fast_mapping_run.log` — captured stdout, `EXIT=0`
- this document
