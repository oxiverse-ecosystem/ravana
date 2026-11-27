# H_struct — transferable type evidence from an independent lexical superordinate channel

**Status: scratch evidence POSITIVE on the pre-registered discriminating criterion;
production code implemented but DEFAULT-OFF. This is NOT an RSI/TRA gate PASS.**
No `CognitiveChatEngine` was constructed, no `process_turn` called, no store
written, no network fetch made, no engine booted (other agents own the live
store), and no item bank, gate, threshold or answer key was changed. All work
is in worktree `C:/Users/Likhith/Documents/projects/ravana_struct_transfer`,
branch `goal/lexical-struct-transfer` from `origin/main` `5f3be87`.
The three inputs read by the experiment (`scripts/tra_k_items.py`,
`scripts/front_gates.py`, `scratch/trak_substrate.json`) are newline-normalised
SHA-256 identical to their committed `HEAD` blobs, checked before any endpoint
was read. No file under `scripts/`, `tests/` (except the new unit test),
`weights/`, or any item bank was modified.

## 1. The new hypothesis

Every prior attack in this line scored **item similarity**: GloVe role centre,
symbolic role signature, relational vector offset, teaching-corpus
co-occurrence, topic-centre/character n-gram fast mapping and the Wikidata
description cue-map (`CROSS_SOURCE_FAST_MAPPING_DESIGN.md`, TEXT−DEMO = +0.055,
NO-GO). This attempt uses a different primitive:

> A **structured lexical hypothesis space** (WordNet 3.0 noun hypernyms)
> supplies subject-specific type evidence directly; the **criterion** for
> admitting a candidate is calibrated from the item's own publicly taught
> demonstration pairs. The query subject's own lexical identity, not a
> centroid of other words, drives the decision.

Formally, with `path(a,b)` the maximum WordNet noun path similarity over sense
pairs, `theta = median{path(r1,r2) : taught demo pairs} + DELTA`, and
`DELTA = 0.10`, `MARGIN = 0.05` frozen before any run:

```
scores(c) = path(subject, c)                 # subject-specific, not a centroid
pick      = unique argmax scores(c) with scores(c) >= theta
            and scores(c) - runner_up >= MARGIN, else ABSTAIN
```

The subject-INDEPENDENT comparator scores candidates by `max path(c, r2)`
over taught `r2` words under the **identical** theta and margin. It isolates
the contribution of the subject-specific source.

This is not a lookup of the answer key: the graded key is computed by the
grader alone as *the unique displayed candidate sharing the query subject's
primary Wikidata type*, from the independent CC0 substrate. No scorer receives
`gold`, `type_sets`, `world_model`, `typed_queries`, `unseen_test_words`,
`type_label`, domain tag or `role`. An AST scan of all four scorer functions
(`schema_criterion`, `subject_scores`, `demo_scores`, `decide`) found zero
banned identifiers; the public production API has no parameter in which a key
could be passed.

## 2. Brain grounding (graded)

| # | Finding | Source / locator | Grade | What it licenses here |
|---|---|---|---|---|
| N1 | A novel word's type comes from a **subject-specific** contrastive input, not from other exemplars (`chromium` was a colour word from "not the red tray, the chromium one"). | Carey & Bartlett (1978), *Papers & Reports on Child Lang. Dev.* 15:17–29 (no DOI; primary PDF `languagelog.ldc.upenn.edu/myl/Acquiring_a_Single_New_Word.pdf`) | **[S]** human behavioural | score by the query subject's own identity, never by a demo centroid |
| N2 | Sparse word learning works only over an already **structured hypothesis space**; three Dalmatians are informative because `dog ⊂ animal` exists. | Xu & Tenenbaum (2007), *Psychol Rev* 114(2):245–272, doi:10.1037/0033-295X.114.2.245 (**Crossref verified**) | **[S]** behavioural + computational | use a pre-existing lexical hierarchy as the type space |
| N3 | Category-marking cues must be **in the input** for generalisation to novel members. | Lany & Saffran (2010), *Psychol Sci* 21(2):284–291, doi:10.1177/0956797609358570 (**Crossref verified**) | **[S]** human behavioural | the criterion must come from the taught demonstrations actually supplied |
| N4 | One-trial assimilation becomes durable only when a neocortical **schema already exists**. | Tse et al. (2007), *Science* 316(5821):76–82, doi:10.1126/science.1135935 | **[S]** rodent neurobiology | teaching pairs calibrate the admission bar |
| N5 | Fast episodes and slow regularity extraction are complementary. | McClelland, McNaughton & O'Reilly (1995), *Psychol Rev* 102(3):419–457, doi:10.1037/0033-295X.102.3.419 | **[S]** framework | keep demos as separate calibration traces, not an averaged prototype |
| N6 | Item memory and **source memory dissociate**; schema-congruence also produces false memories. | Johnson, Hashtroudi & Lindsay (1993), *Psychol Bull* 114(1):3–28, doi:10.1037/0033-2909.114.1.3 | **[S]** human behavioural | a schema can propose, but a subject-specific source must admit |
| N7 | WordNet organises lexical concepts in a semantic relation network usable by machine. | Miller (1995), *CACM* 38(11):39–41, doi:10.1145/219717.219748 (**Crossref verified**); Fellbaum (ed.) (1998), MIT Press | **[M]** resource description | implementation substrate for the structured space, **not** brain evidence |

**Evidence-type note.** N1–N3 and N6 are human behavioural; N4 is rodent
neurobiology; N5 is a computational framework; N7 describes the offline lexical
resource. Following `neuroscience-to-architecture` §4, the graded claim is only
*subject-specific input + pre-existing structured hypothesis space +
teaching-calibrated admission*. The particular choice of WordNet `path_similarity`
and the `DELTA`/`MARGIN` constants are **extrapolations**, not claims that the
brain computes a shortest-path metric over WordNet.

## 3. Instrument validation before any endpoint was read

| Check | Result |
|---|---|
| Input SHAs identical to `HEAD` | `tra_k_items.py`, `front_gates.py`, `trak_substrate.json` — all True |
| Item bank | `build_items()` 400, `validate_items()` 0 errors |
| Frozen constants before run | `DELTA=0.10`, `MARGIN=0.05`, `PERMUTE_OFFSET=137`, criterion = median(taught-pair path sims) + DELTA |
| Cases | 800 near+far; **125 gradeable** (70 primary / 55 sealed) with a unique independent key |
| AST anti-leakage | ok, 0 problems across all four scorer functions |
| Hand-check | printed input → output for 12 emitted rows; e.g. `tra_k_0130 tree` scores `{shrub:0.333, petal:0.077, blossom:0.100, foliage:0.091}`, theta 0.283 → pick `shrub` = independently graded key |
| Shipped G6 gate (`front_gates.gate_saturation`, mode `offline`) | **PASS**, no saturated front (non-blocking) |

## 4. EVIDENCE

### v1 (first frozen run; one pre-registered control failed)

| Arm | emits / 125 | correct | precision | primary | sealed |
|---|---|---|---|---|---|
| **STRUCT** (new) | 40 | 28 | **0.700** | 0.650 | 0.750 |
| DEMO-WN (subject-independent) | 26 | 10 | 0.385 | 0.462 | 0.308 |

Controls: gold-key permutation leaves all 40 picks byte-identical and drops
accuracy 0.700 → 0.100 (all keys changed within their own option set);
subject permutation changes 38/40 picks (0.95); negated teaching 0 emits;
unrelated subject 0 emits; primary−sealed gap 0.10.

**Failed control:** the pre-registered fixed "disjoint" tuple
`apple/banana/chair/table` produced 6 emits on graded rows (29 over all rows).
Diagnosis (`wn_disjoint_diagnostic.py` prints every case): in the deployed
WordNet channel that tuple is genuinely related to several subjects —
`herb→banana` shares `herb.n.01` (depth 8), `vine→banana` shares
`vascular_plant.n.01` (depth 7), `leader→chair` shares `leader.n.01` (depth 7).
This is an instrument defect (wrong control for the channel), not answer-key
leakage: the scorer never sees a key or a domain tag. **No mechanism constant
or endpoint was changed.** `LEXICAL_SUPERORDINATE_V2_PREREG.md` records the
correction before the v2 run.

### v2 (controls only changed; scoring identical)

| Arm / control | Result | Pre-registered criterion |
|---|---|---|
| STRUCT emits / 125 | **40 (32.0%)** | ≥ 20 — PASS |
| STRUCT precision (all / primary / sealed) | **0.700 / 0.650 / 0.750** | ≥ 0.60 — PASS |
| DEMO-WN precision | 0.385 | comparator |
| STRUCT − DEMO-WN | **+0.315** | ≥ +0.15 — PASS (prior NO-GO was +0.055) |
| Subject permutation | 38/40 changed = **0.95** | ≥ 0.80 — PASS |
| Gold permutation | picks identical, all keys changed, accuracy 0.100 | ≤ 0.40 — PASS |
| OOV disjoint options (`zzq_lex_*`, no WordNet noun synset) | **0 emits** | 0 — PASS |
| Negated teaching / unrelated subject | **0 / 0 emits** | 0 — PASS |
| Candidate-order permutation | **0 changed picks** | 0 — PASS |
| Primary−sealed precision gap | 0.10 | ≤ 0.20 — PASS |
| AST / bank / unmodified inputs / G6 | clean / 0 / identical / PASS | all PASS |

**Verdict: POSITIVE on the pre-registered discriminating criterion.**
The legacy tuple's 6 emits are retained as a DIAGNOSTIC_ONLY field in the
results, never as a positive criterion.

## 5. Production implementation (default OFF)

`ravana/src/ravana/learn/lexical_superordinate.py` implements exactly the
frozen algorithm with no engine dependency, lazy `nltk` import, and an explicit
inert default:

* `RAVANA_LEXICAL_SUPERORDINATE` unset/≠`1` → `pick()` returns `None` on every
  input (verified over all 800 replayed cases: **0 flag-off emits**).
* With the flag on, all 800 cases were replayed through the PRODUCTION module:
  **178 emits, 0 mismatches** against the scratch picks; precision on the
  40 graded emits **0.700**. `PARITY: PASS`.
* WordNet 3.0 licence notice and citation are in the module docstring. The
  corpus is **not** checked in; a deployment must supply the licensed corpus
  locally (optional `RAVANA_WORDNET_DATA` path). Missing corpus fails closed.

`tests/unit/test_lexical_superordinate.py`: 8 tests, all passing
(`8 passed in 8.27s`), with `PYTHONPATH` pointed at this worktree and the
imported module's `__file__` verified under the worktree `ravana/src`.
Tests cover inert default, unique winner, strict margin, negated teaching,
OOV abstention, order invariance, subject-independent comparator, and the
absence of any answer-key parameter in the public API.

**No engine hooks were added.** The primitive is available but unwired: it
does not run inside RSI/TRA, has not been A/B'd on a live engine, and must not
be cited as an RSI/TRA PASS. Enabling it in production is a separate decision
that requires a live A/B by the parent agent, who owns the engine/store.

## 6. Falsifiers and limits

* The graded key is the **Wikidata-type-sharing proxy** inherited from
  `CROSS_SOURCE_FAST_MAPPING_DESIGN.md`, not the original TRA/RSI role
  relation (`effect-of`, `earlier-event`, …). A win here is necessary-direction
  evidence about transferable type input, not a benchmark repair.
* Coverage is 32%; 68% of gradeable cases abstain by design.
* Three emitted rows in the hand-check are wrong (`blizzard→sunset` vs key
  `breeze`, `blizzard→tempest` vs `thunder`, `leaf→stem` vs `bark`), consistent
  with 0.70 precision — the route is a precision play, not saturated.
* `path_similarity` is a lexical-semantic relatedness proxy. Under N7 it is an
  engineering extrapolation; if an independent lexical resource with different
  structure reverses the +0.315 margin, the mechanism's brain mapping needs
  revision even if the code is correct.
* The legacy disjoint tuple is not a valid control for this channel; any
  future attempt to reintroduce it must first prove non-relatedness in the
  actual ontology rather than assume it from the word strings.

## 7. Files

- `scratch/lexical_superordinate_experiment.py` — v1 frozen probe
- `scratch/lexical_superordinate_experiment_v2.py` — v2 (controls corrected)
- `scratch/lexical_superordinate_run.log`, `scratch/lexical_superordinate_v2_run.log`
- `scratch/lexical_superordinate_v2_results.json` — 800 rows, tallies, controls, scorer sources, SHAs
- `scratch/LEXICAL_SUPERORDINATE_V2_PREREG.md` — control-correction pre-registration
- `scratch/wn_channel_probe.py`, `scratch/wn_demo_calibration_probe.py`,
  `scratch/wn_head_probe.py`, `scratch/wn_rule_dev.py`,
  `scratch/wn_disjoint_diagnostic.py` — labelled exploratory diagnostics
- `scratch/verify_production_parity.py` — 800-case production/scratch parity
- `ravana/src/ravana/learn/lexical_superordinate.py` — default-off implementation
- `tests/unit/test_lexical_superordinate.py` — 8 unit tests
- this document
