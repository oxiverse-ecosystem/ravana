# TRA brain-grounded design — a new benchmark and mechanism for Typed Role Acquisition

**Status: STATIC research/design only.** No engine process was booted, no pytest
was run, and no source file, gate, threshold, item bank, benchmark or production
module was edited. The only artifact is this document, written into worktree
`C:/Users/Likhith/Documents/projects/ravana_tra_brain` on branch
`docs/tra-brain-design` from main `aa13c31`.

**Operating point.** RAVANA is a ~20 W CPU, no-LLM system: headline scores must be
gradeable by exact match / structured checks, and energy is a reported metric,
not an assumption (`docs/BENCHMARKING_SYSTEMS_2026.md §1`). Every primitive named
below already exists in the tree; nothing here requires a neural network at
inference.

---

## 0. Problem statement and why a new benchmark

RSI remains **FAIL**. Three scratch experiments (symbolic roles, typed roles,
schema wiring) established that ranking and wiring are not the gap: only an
answer-key oracle fixes the score. The engine has **no usable typed role slots**:

- 0 usable `is_a` edges touch RSI vocabulary;
- 0/7 gold candidates carry a type;
- item-bank `domain` / `far_domain` labels never reach the engine;
- `item["world_model"]` is the gold answer key, i.e. contamination if fed to the
  engine (`scripts/measure_learn5_rsi.py:295-317`; `ravana-debugging §2`).

RSI therefore cannot distinguish *failed type acquisition* from *successful
acquisition with no typed slot to express it in*. TRA (Typed Role Acquisition)
separates the two:

| | RSI | TRA (proposed) |
|---|---|---|
| Teaching content | `R(a,b)` demonstrations | `x is a TYPE`, `p is part of a TYPE`, `y is not a TYPE` |
| Query | which **object** completes `R(subject, ?)` | which **type slot** does an unseen word fill |
| Candidate set | object words | type labels (closed set, disjoint from teaching words) |
| Gold source | `world_model[(rel, subject)]` | withheld; never passed to engine or grader-side hook |
| Controls | foil / negative / far | foil / negative / k=0 / gold-shuffled / source-invalidated |

TRA's central claim is *typed-slot selection grows with taught exemplars*. If
that claim fails with clean controls, the gap is upstream of articulation and
the design must stop rather than tune the grader.

---

## 1. TRA item schema (specification only — not implemented here)

```
TRAItem := {
  item_id, family_id, split ∈ {primary, sealed},
  slot_family ∈ {IS_A, PART_OF_WHOLE, PART_OF_PART, EVENT_TYPE},
  teaching: [ "mist is a weather phenomenon.", ... ],   # natural assertions
  k ∈ {0, 1, 2, 4, 8},                                   # exemplars taught
  query: { subject: unseen_word,
           slots: [type_label, ...],                     # closed, ≥4
           gold_slot: type_label | None,
           foil_slot: type_label,                        # taught but not licensed
           counterexample_slot: type_label | None },
  manifest_sha256
}
```

Design rules, adopted from the measurement discipline in `ravana-debugging`:

1. **Split by fixed hash before any run**: `sha256(family_id|slot_family|subject)`;
   record the manifest hash; inspect sealed outcomes only in the reported
   measurement. Report primary-only and full-set scores side by side; the gap is
   the overfitting estimate.
2. **Sample size is part of the instrument**: ≥40 item families per pool, five
   k-levels, one k per item ⇒ ≥200 primary and ≥200 sealed items. If the 20 W
   budget forces fewer, report the reduced n as a hypothesis, never as a verdict.
3. **Teaching material is not scaffolding**: k=0 and definition-only arms run on
   the same cold store to prove nothing is pre-seeded. Domain/type labels are
   taught through conversation, never written into the cold-start graph.
4. **Fresh engine per condition** with its own `data_dir`; never a shared live
   scratch store, never a child pointed at the parent's store.

---

## 2. Mechanism map: brain finding → computational primitive → RAVANA object

Status labels: **S** = the cited source directly supports the brain/behaviour
claim; **A** = supported as an analogy (design shape, not neural homology);
**E** = engineering extrapolation with no direct support. No cell claims RAVANA
implements brain machinery.

| # | Brain finding (verified source) | Computational primitive | RAVANA object / concrete change | Ledger |
|---|---|---|---|---|
| M1 | Episodic and semantic memory are distinguishable systems; the hippocampus stores fast, distinct episodes while neocortex extracts regularities slowly (Tulving 1985 [1]; McClelland et al. 1995 [2]) | Keep each teaching assertion as an **immutable, separately provenanced episode**; derive a type view on demand, never write the abstraction back as a fact | Existing trusted `ConceptEdge`s written by `intake.intake(fact, Provenance(...))`; type view is a stateless per-call projection, no new edge kind | **S** for the distinction; **A** for the episode/abstraction split |
| M2 | A pre-existing neocortical schema lets a one-trial trace be assimilated rapidly; without one, consolidation is gradual (Tse et al. 2007 [3]) | Type strength accumulates over converged exemplars; k=1 yields a provisional hypothesis, k≥2 a stable slot centre | `TRUSTED_EDGE_KINDS` + `induce_schema_hypotheses`/`traces_from_edges` (`chat/schema_induction.py:70-115,275-316`) over distinct trace ids only | **S** for fast/slow dependence on schema; **E** for the count-based gate |
| M3 | Declarative memory is built from **relations among items**, enabling flexible inference in novel cases (Cohen & Eichenbaum 1993 [4]; Eichenbaum 2004 [5]); hippocampus–PFC interaction supports integrating new events into existing schemas (Preston & Eichenbaum 2013 [6]) | Represent a type as a **bound slot in a relation**, not as feature overlap with the query word: `TypeSlot(relation_type, slot_name, members, support)` | New *view* over existing edges; candidate ranking uses slot membership/role fit, not GloVe cosine to the candidate. No topology change | **S** for relational binding; **E** for the exact slot vocabulary |
| M4 | A new word can be mapped from a single exposure (Carey & Bartlett 1978 [7], bibliographic only; Dollaghan 1985 [8]; Kaminski, Call & Fischer 2004 [9]) | One natural assertion can create a **provisional** type hypothesis; k=1 must beat k=0 but need not be stable | k=1 arm; hypothesis carries `assertable=False` until supporting traces converge | **S** for one-shot mapping; **A** for applying it to type labels; **E** for the dog→engine transfer [9] |
| M5 | Fast mapping is robust for referent selection but poor for retention without augmentation (Horst & Samuelson 2008 [10]) | **Cold-reload retention is a mandatory endpoint**, not a nice-to-have; k=1 may select but fail to retain | `eng.save()` → cold reload → re-derive types from stored provenance; exact node identity, never substring | **S** for the dissociation; **A** for the persistence operationalisation |
| M6 | Words cannot be learned from observation alone; social/pragmatic context matters (Medina et al. 2011 [11]); pedagogical cues guide reorganization of sparse category evidence (Butler & Markman 2014 [15]) | Teaching must arrive as **conversational assertions through `process_turn`**, not silent graph injection | Real `Provenance(source='conversation', source_type='user')`; silent injection is a labelled non-deployable oracle arm | **S** for the behavioural effect; **E** for the API-level choice |
| M7 | Humans generalize from sparse data by hypothesising over a structured space and weighing likelihood against prior expectation (Tenenbaum & Griffiths 2001 [13]; Xu & Tenenbaum 2007 [12]; Tenenbaum, Griffiths & Kemp 2006 [14]) | Closed candidate slot set; support-weighted choice; **conservative** (abstain/tie → None) when evidence is sparse | Strict winner + role fit from converged traces; tie → `None`; no threshold lowering to force an answer | **S** for sparse structured induction; **E** for the scalar implementation |
| M8 | Source monitoring is a flexible, error-prone attribution of information to its origin; item and source memory dissociate (Johnson et al. 1993 [16]); toddlers withdraw words learned from a speaker later shown unreliable (Dautriche et al. 2021 [18]) | Provenance stamped **at write time**; source invalidation must retract taught types; inclusion/exclusion separates familiarity from recollection (Jacoby 1991 [17]) | Read `source_metadata` in `_is_trusted_edge`; add a source-invalidation condition; no answer-key-derived context | **S** for theory and withdrawal; **E** for engine operationalisation and experimenter-delivered cue |
| M9 | Schema congruence itself increases false memories | A type hypothesis may **propose** candidates but may not admit them; an observed trusted assertion is still required | Per-call `schema_hypothesis`, `assertable=False`, defer to directly taught edges, foil/negative veto preserved | **E** — engineering gate motivated by the source-monitoring literature [16], not a direct neural finding |

**Why cosine cannot be the type signal.** M3 is the load-bearing row: a type is a
position in a relational structure, whereas GloVe proximity measures lexical
affinity. RSI's existing candidate ranking already uses node-vector cosine and
still fails; replacing it with a cosinely-similar slot scorer would repeat the
same mistake with different labels. Node vectors may inform *slot typing of the
subject* only; they must never supply the answer key.

---

## 3. Verified citation ledger

Every DOI below was resolved against `api.crossref.org` during this research
pass; title, author list, year, journal, volume, issue and pages were read from
the Crossref record. Abstracts were used only where Crossref supplied them; no
full-text claim is made where only metadata was checked.

| # | Citation (author, year, journal, volume(issue):pages) | DOI | Crossref record |
|---|---|---|---|
| 1 | Tulving, E. (1985). Memory and consciousness. *Canadian Psychology, 26*(1), 1–12. | 10.1037/h0080017 | "Memory and consciousness.", Tulving, 1985, Can Psychol 26(1) |
| 2 | McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review, 102*(3), 419–457. | 10.1037/0033-295X.102.3.419 | title/authors/pages match; 1995-07 |
| 3 | Tse, D., Langston, R. F., Kakeyama, M., Bethus, I., Spooner, P. A., Wood, E. R., Witter, M. P., & Morris, R. G. M. (2007). Schemas and memory consolidation. *Science, 316*(5821), 76–82. | 10.1126/science.1135935 | full author list and abstract checked |
| 4 | Cohen, N. J., & Eichenbaum, H. (1993). *Memory, Amnesia, and the Hippocampal System*. Cambridge, MA: MIT Press. | **book — no DOI** | bibliographic record only; see §3.2 |
| 5 | Eichenbaum, H. (2004). Hippocampus: cognitive processes and neural representations that underlie declarative memory. *Neuron, 44*(1), 109–120. | 10.1016/j.neuron.2004.08.028 | title "Hippocampus", Eichenbaum, Neuron 44(1) |
| 6 | Preston, A. R., & Eichenbaum, H. (2013). Interplay of hippocampus and prefrontal cortex in memory. *Current Biology, 23*(17), R764–R773. | 10.1016/j.cub.2013.05.041 | title/authors/pages match |
| 7 | Carey, S., & Bartlett, E. (1978). Acquiring a single new word. *Papers and Reports on Child Language Development, 15*, 17–29. | **no DOI (1978 report)** | bibliographic record only; see §3.2 |
| 8 | Dollaghan, C. (1985). Child Meets Word: "Fast Mapping" in Preschool Children. *Journal of Speech, Language, and Hearing Research, 28*(3), 449–454. | 10.1044/jshr.2803.454 | abstract checked: 91% on first exposure, 81% on second |
| 9 | Kaminski, J., Call, J., & Fischer, J. (2004). Word Learning in a Domestic Dog: Evidence for "Fast Mapping". *Science, 304*(5677), 1682–1683. | 10.1126/science.1097859 | Crossref title query; Science 2004-06-11 |
| 10 | Horst, J. S., & Samuelson, L. K. (2008). Fast Mapping but Poor Retention by 24-Month-Old Infants. *Infancy, 13*(2), 128–157. | 10.1080/15250000701795598 | abstract checked: excellent selection, poor retention |
| 11 | Medina, T. N., Snedeker, J., Trueswell, J. C., & Gleitman, L. R. (2011). How words can and cannot be learned by observation. *PNAS, 108*(22), 9014–9019. | 10.1073/pnas.1105040108 | Crossref title query; full author list |
| 12 | Xu, F., & Tenenbaum, J. B. (2007). Word learning as Bayesian inference. *Psychological Review, 114*(2), 245–272. | 10.1037/0033-295X.114.2.245 | title/authors/pages match |
| 13 | Tenenbaum, J. B., & Griffiths, T. L. (2001). Generalization, similarity, and Bayesian inference. *Behavioral and Brain Sciences, 24*(4), 629–640. | 10.1017/S0140525X01000061 | Crossref title query; BBS 24(4) |
| 14 | Tenenbaum, J. B., Griffiths, T. L., & Kemp, C. (2006). Theory-based Bayesian models of inductive learning and reasoning. *Trends in Cognitive Sciences, 10*(7), 309–318. | 10.1016/j.tics.2006.05.009 | title/authors/pages match |
| 15 | Butler, L. P., & Markman, E. M. (2014). Preschoolers use pedagogical cues to guide radical reorganization of category knowledge. *Cognition, 130*(1), 116–127. | 10.1016/j.cognition.2013.10.002 | Crossref: online 2013, print 2014 |
| 16 | Johnson, M. K., Hashtroudi, S., & Lindsay, D. S. (1993). Source monitoring. *Psychological Bulletin, 114*(1), 3–28. | 10.1037/0033-2909.114.1.3 | title/authors/pages match |
| 17 | Jacoby, L. L. (1991). A process dissociation framework: separating automatic from intentional uses of memory. *Journal of Memory and Language, 30*(5), 513–541. | 10.1016/0749-596X(91)90025-F | title/authors/pages match |
| 18 | Dautriche, I., Goupil, L., Smith, K., & Rabagliati, H. (2021). Knowing How You Know: Toddlers Reevaluate Words Learned From an Unreliable Speaker. *Open Mind, 5*, 1–19. | 10.1162/opmi_a_00038 | full abstract checked |

### 3.1 Rejected during verification (do not propagate)

- `10.1016/j.anbehav.2008.04.013` — resolves to Leonard & Boake, *Associations
  between male courtship and female polyandry in wasps*, not a fast-mapping paper.
- `10.1073/pnas.1009010108` and `10.1037/0033-295X.108.4.629` — 404 at Crossref;
  the verified Medina DOI is `...1105040108` and the verified Tenenbaum &
  Griffiths DOI is `10.1017/S0140525X01000061`.
- The commonly cited "Kaminski, Call & Tomasello (2008), *Animal Behaviour*" is
  **wrong**: Crossref gives Kaminski, Call & Fischer (2004), *Science*.
- No DOI was found for [4] (book) or [7] (1978 conference report). Both are kept
  as bibliographic-only records and marked as such; they are not counted among
  the DOI-verified fast-mapping evidence.

### 3.2 Support vs extrapolation ledger

| Claim id | Claim used in this design | Status | Source | Boundary / why |
|---|---|---|---|---|
| L1 | Episodic and semantic memory are distinguishable; fast traces and slow abstractions are separable | **S** | [1][2] | Claims about human memory systems, not about software |
| L2 | One-trial assimilation depends on an already-established schema | **S** | [3] | Rat paired-associate task; extrapolation to k-sweeps is A |
| L3 | Declarative memory is relational, supporting flexible novel inference | **S** | [4][5][6] | Reviews/book; no claim about RAVANA's graph |
| L4 | A single exposure can create a word mapping | **S** | [7][8][9] | [7] has no DOI; [9] is a dog; only [8] is DOI-verified child data |
| L5 | Fast mapping is not durable retention | **S** | [10] | 24-month-olds, 5-min delay; cold reload is our analogue (A) |
| L6 | Observation alone is insufficient; pedagogical/social cues matter | **S** | [11][15] | Human children; "use process_turn" is our engineering choice (E) |
| L7 | Sparse generalization works by structured hypotheses with priors | **S** | [12][13][14] | Computational-level theories; our scalar scorer is E |
| L8 | Item and source memory dissociate; sources can be discredited and words withdrawn | **S** | [16][18] | Toddler data uses an experimenter cue — proxy (E) at engine level |
| L9 | Provenance must be stamped at write time and read from `source_metadata` | **A** | [16] → `_is_trusted_edge` | Brain finding motivates the shape; the field is an engineering object |
| L10 | A type view must be non-assertable until observed traces converge | **E** | [2][3][16] → schema gate | No brain experiment licenses this exact gate; it is a confabulation control |
| L11 | Cold reload after teaching is the correct retention endpoint | **E** | [5][10] → `eng.save()` | Persistence on disk ≠ biological consolidation |
| L12 | ~20 W/no-LLM is a valid operating point for this benchmark | **E** | repo design docs | Asserted resource target; not measured in this static pass |

---

## 4. Legitimate teaching signal vs answer-key contamination

### 4.1 Legitimate (deployable) teaching signal

A TRA teaching event counts as deployable only if **all** hold:

1. It is a **natural-language assertion** (`"mist is a weather phenomenon."`),
   delivered through the real `process_turn` path, with no machine-token triples.
2. It is parsed and written by the unchanged `intake.intake` with a real
   `Provenance(source='conversation', source_type='user')` at write time, and is
   admitted by the unchanged `_is_trusted_edge` predicate
   (`chat/schema_induction.py:70-115`) — never by a widened trust gate.
3. The resulting edge is recoverable after `eng.save()` + cold reload by **exact
   node identity and provenance**, with a sanitize-safe short id if the object
   is a multi-word label (the cold-reload sanitizer deletes ≥5-word labels).
4. The query path can select the type using only what is stored in the engine;
   no grader-side hook supplies the candidate.
5. The same assertion is available to **every** arm at the same k; k=0 and
   definition-only arms exist to prove no type was pre-seeded.

Negations are teaching signals too, but only when expressed as natural
assertions (`"gale is not a civic event."`) and filtered at the top-level call
so a hedged clause cannot mint a positive type edge.

### 4.2 Contamination (NON-DEPLOYABLE oracle — label and exclude)

| Source | Why it is contamination |
|---|---|
| `item["world_model"]` | The gold answer key; using it is answer lookup, not acquisition |
| `item["gold_slot"]`, `counterexample_slot` | Direct grading key; must never enter the engine or a decision hook |
| `item["domain"]`, `item["far_domain"]` | Item-bank labels that never reach the engine in RSI; injecting them creates a capability the system cannot have at test time |
| any item-bank type label passed as a feature | Same as above: a side channel unavailable outside the benchmark |
| a grader hook that returns a candidate from the key | Grades the hook, not the engine (G6 saturation) |
| silent graph injection of the type edge | Bypasses conversational learning; report only as an upper bound |

**Anti-tautology test (mandatory before any score).** Permute gold slots and
`world_model` in the item metadata while leaving graph provenance unchanged:
the staged context, its fingerprint, the decision, and the emitted answer must
be byte-identical. If any moves, the instrument is circular. Conversely, flip
the provenance of one stored trusted edge and require the graded endpoint to
move — a score that cannot move is measuring the oracle, not the system.

---

## 5. Pre-registered predictions (TRA)

Decision rule: TRA **PASSES** only if P1–P4 hold on **both** primary and sealed
pools with all validity gates green. A single failure is a FAIL, not a caveat.

- **P1 — growth.** `acc(k=0) ≈ chance = 1/|slots|` (≈0.25 for 4 slots);
  `acc(1) > acc(0)`, `acc(2) > acc(1)`, `acc(4) > acc(2)`, with
  `acc(8) − acc(1) ≥ 0.15` on primary and no sealed drop >0.10. Non-decreasing
  within Wilson 95% noise.
- **P2 — foil rejection.** On foil items (taught type present as a candidate but
  not licensed for the query subject) rejection ≥0.80 at **every** k, including
  k=8. A surface matcher that picks the demonstrated anchor must fail this.
- **P3 — negative rejection.** On negation/counterexample items, where the true
  type is in the candidate set but demonstrations assert the negation,
  rejection ≥0.80 at every k.
- **P4 — cold-reload retention.** At k=max, post-reload typed-slot accuracy ≥0.90
  of in-process accuracy, and the type view is re-derived from stored
  provenance only. Loss >0.10 is a write/provenance failure, not a learning
  result.
- **P5 — source invalidation (extrapolation).** After discrediting the teaching
  source, withdrawal/abstention on taught types ≥0.70; recorded as an
  experimenter-delivered proxy, not a human-equivalent demonstration.
- **P6 — answer-key immunity.** Gold-shuffled control changes any endpoint by
  ≤0.05. Larger movement ⇒ contamination; the run is INVALID.
- **P7 — deferral.** A directly taught type edge outranks a schema hypothesis;
  the new path never overrides a trusted direct edge.
- **P8 — instrumentation.** Report per-k, per-condition counts and denominators;
  ≥200 items per pool; flag any condition at 1.00/0.00 as G6 saturation with no
  headroom rather than capability evidence.

All thresholds above are pre-registered here, before any engine run. They are
not to be relaxed after seeing results.

---

## 6. Falsifiers and stop rules

| id | Observation | Meaning | Action |
|---|---|---|---|
| F1 | `acc(k=0) > chance + 0.15` | Pre-seeded types or side-channel leak | **STOP — INSTRUMENT INVALID, no score** |
| F2 | Gold-shuffled control moves an endpoint >0.05 | Answer key reaches the engine or grader | **STOP**; fix the write/seam, never the threshold |
| F3 | Provenance mutation does not flip graded endpoints | Grader disconnected from engine | **STOP — no score** |
| F4 | Forced single-token slot choice with echoed text stripped yields ~0 parseable decisions | No decision routine exists | Stop rewriting prompts; next work item is the mechanism |
| F5 | With ≥200/pool, `acc(k=max) − acc(k=1) ≤ 0` | No typed acquisition | Do not retune ranking; locate the read/write gap |
| F6 | Foil or negative rejection <0.80 at any k | Type hypothesis over-generalizes / confabulates | Revert schema proposal to default-OFF; repair veto |
| F7 | Cold-reload loss >0.10 | Write-time provenance or sanitizer defect | Fix the WRITE; do not loosen trust gate |
| F8 | All conditions saturated 1.00/0.00 | No headroom (G6) | Flag INVALID; harden items, do not tune thresholds |
| F9 | ON arm compute >12% above OFF at matched work counters | Confound, not consolidation | Revert and record partial result |
| F10 | Schema hook fires 0 times across all conditions | Read path never reached | Fix detection first; extraction tuning cannot help |
| F11 | Sealed score drops >0.10 below primary | Overfitting to primary split | Report sealed as the real baseline; no gate edits to restore the old number |

A negative result is a deliverable: commit the scratch script, its raw rows and
the doc, open a **docs-only** PR, and name the mechanism still missing. Never
open a production code PR from an unvalidated TRA design.

---

## 7. Honest caveats

- **No runtime evidence in this pass.** Every prediction above is unmeasured;
  the ~20 W/no-LLM operating point is a repo-level design target, not something
  verified here. No engine, no pytest, no source edits.
- **Crossref verifies metadata, not truth.** Bibliographic fields (and abstracts
  where supplied) were checked; full texts of all 18 papers were not re-read.
  [7] and [4] have no DOI and are explicitly bibliographic-only.
- **Analogy is not homology.** Human—and in one case canine—fast mapping is
  evidence about behaviour, not about `ConceptEdge`s. M1–M9 are analogies or
  engineering extrapolations except where marked S for the brain claim itself.
- **Category vs typed slot.** Xu & Tenenbaum's hierarchical categories and
  Butler & Markman's category reorganization concern conceptual categories;
  TRA's "type slot" is a relation argument position. The mapping is motivated,
  not established. A confound remains if candidates are discriminable by
  surface features alone — hence the foil and gold-shuffled controls.
- **Source invalidation is a proxy.** Dautriche et al.'s toddlers infer
  unreliability from the speaker's behaviour; TRA's experimenter cue supplies
  that inference for free. P5 is therefore weaker evidence than P1–P4.
- **Encoding cost.** Natural-language teaching of ≥200 items × 5 k-levels ×
  two pools may exceed a 20 W run budget; if reduced, the result is a
  hypothesis with an explicit n, not a benchmark verdict.
- **No new trust path is proposed.** If TRA requires admitting an unprovenanced
  edge to score, that is a regression, not a fix.

---

## Sources

1. https://doi.org/10.1037/h0080017
2. https://doi.org/10.1037/0033-295X.102.3.419
3. https://doi.org/10.1126/science.1135935
4. MIT Press, *Memory, Amnesia, and the Hippocampal System* (1993) — book, no DOI
5. https://doi.org/10.1016/j.neuron.2004.08.028
6. https://doi.org/10.1016/j.cub.2013.05.041
7. Carey & Bartlett (1978), Papers and Reports on Child Language Development 15:17–29 — no DOI
8. https://doi.org/10.1044/jshr.2803.454
9. https://doi.org/10.1126/science.1097859
10. https://doi.org/10.1080/15250000701795598
11. https://doi.org/10.1073/pnas.1105040108
12. https://doi.org/10.1037/0033-295X.114.2.245
13. https://doi.org/10.1017/S0140525X01000061
14. https://doi.org/10.1016/j.tics.2006.05.009
15. https://doi.org/10.1016/j.cognition.2013.10.002
16. https://doi.org/10.1037/0033-2909.114.1.3
17. https://doi.org/10.1016/0749-596X(91)90025-F
18. https://doi.org/10.1162/opmi_a_00038

Verification method: `https://api.crossref.org/works/<DOI>` and Crossref title
queries, read during this pass; rejected candidates listed in §3.1.
