# RSI role binding — brain-grounded minimal design and pre-registered falsifiers

**Status: research/design only.** No engine process was booted, no pytest was
run, no source file, gate, threshold, item bank or benchmark was edited. The
only artifact is this document, written into worktree
`C:/Users/Likhith/Documents/projects/ravana_rsi_brain` on branch
`docs/rsi-role-binding-brain` from main `dbb3e78`.

**Task.** RSI stores eight trusted demonstrations (e.g. `gale causes rainbow`)
and then asks `what is true of thaw?` against four options. After the wiring
fixes, the parent's focused live n=4/pool probe gives **near ≈ 0.50** (CAUSE
`thaw→mist` and RESEMBLES `drizzle→chill` correct; PART_OF `thunder→chill` and
PRECEDES `cloud→glow` wrong), **far = 0.00** across CAUSE / PART_OF / PRECEDES /
RESEMBLES, with **foil and negative correctly rejected**. Diagnosis inherited
from `RSI_SCHEMA_WIRING.md §2` and `RSI_ALL_RELATIONS.md`: the
`schema_induction` hypothesis ranks candidates with **GloVe node-vector
cosine** (`response_gen.schema_association_scores_from_graph`), i.e. lexical /
semantic proximity, not relational role binding. This document asks what a
brain-grounded, minimal, gate-preserving replacement looks like for a no-LLM
~20 W CPU system. It is a **design**, not an implementation or a result.

Baseline evidence (read, not re-measured here): `RSI_FOCUSED_LIVE.md`,
`RSI_ALL_RELATIONS.md §1`, `RSI_SCHEMA_WIRING.md §§0–3`,
`RSI_LIVE_RULE_INDUCTION.md §1`, `TRUSTED_EDGE_CHOICE.md §4`.

---

## 1. Mechanism map: brain finding → computational primitive → RAVANA object

Status labels: **S** = directly supported by the cited source (as a claim about
the brain or about human behaviour); **A** = supported as an *analogy* only
(the source licenses the design shape, not a neural homology claim); **E** =
extrapolation / engineering choice with no direct support. No cell of this
table claims RAVANA implements brain machinery.

| # | Brain finding (verified source) | Computational primitive | RAVANA object / concrete change | Ledger |
|---|---|---|---|---|
| M1 | Hippocampal declarative memory is built from **relations among items**, enabling **inferential and flexible expression** in novel situations (Cohen & Eichenbaum 1993; Eichenbaum 2004 [1,2]) | Represent each demonstration as a **bound role frame** `R(arg1:=x, arg2:=y)` plus provenance, not as an undifferentiated co-occurrence pair | `BoundRoleTrace(relation_type, slot_names, bindings, trace_id, provenance)` derived from a trusted edge; role names are fixed per relation family | **A**: relational binding is supported; the exact slot vocabulary is our engineering choice |
| M2 | Structure mapping transfers **relations, not object attributes**, and mapping rules depend on representation syntax rather than domain content (Gentner 1983 [3]) | Group and compare traces by **relation type and argument role**, never by shared words between demonstration and candidate | Alignment key = `(relation_type, role permutation)`; GloVe cosine may inform *slot typing*, never the final answer rank | **S** for the principle; **E** for using node vectors as slot-typing features |
| M3 | Analogical transfer improves when a **general schema is induced from two concrete analogs**; one analog with summary/diagram devices did not suffice; schema quality predicted transfer (Gick & Holyoak 1983 [4]) | Converge **≥ 2 distinct trusted traces** of the same relation into a derived, **unassertable** schema hypothesis before answering an unseen subject | Keep the existing `min_support = 2` over distinct trace ids and `assertable = False`; no write path | **S** for "two exemplars help"; **E** for `min_support == 2` as a hard engine threshold |
| M4 | Mapping is a **constraint-satisfaction** process with interacting structural (isomorphism), semantic and pragmatic constraints (Holyoak & Thagard 1989 [5]) | Score a candidate by **role-compatibility** with the abstracted role centre, with a structural-consistency term; keep a tie/ambiguity reject | `rank_candidates_for_unseen_subject` consumes a *role-fit* score instead of raw subject/candidate cosine; strict winner required | **A**: constraint satisfaction is supported; a single scalar role-fit is a deliberate minimal reduction |
| M5 | Rapid assimilation of one-trial traces requires an **already established neocortical schema**; without one, consolidation is gradual (Tse et al. 2007 [6]) | The schema hypothesis only proposes; it cannot assert. A trusted observation is still required before a candidate can be licensed | Preserve `admit_for_query()`: hypothesis + at least one observed trusted trace of that relation; defer to a directly taught edge | **A** for the fast/slow distinction; **E** for the exact gate implementation |
| M6 | Complementary learning systems: fast hippocampal storage of distinct episodes, slow interleaved cortical extraction of regularity (McClelland et al. 1995 [7]) | Keep demonstrations as **separate immutable edges**; derive the schema as a stateless view; never promote a hypothesis to a stored fact | Existing graph-edge store + per-call hypothesis, no mutation; matches `RSI_SCHEMA_WIRING.md §1` | **A** (fast store / slow extractor); **E** (stateless per-call view) |
| M7 | **Pattern separation** keeps similar experiences distinct (Yassa & Stark 2011 [8]) | Distinct trace ids per triple; duplicates cannot inflate support; two supported candidates produce no answer | Already present in `induce_schema_hypotheses` / `select_trusted_candidate`; keep unchanged | **A**; pattern separation is a hippocampal claim, trace-id dedup is a software analogue |
| M8 | Prefrontal cortex separates **controlled semantic retrieval** from **integration of retrieved relations**; frontopolar/RLPFC is recruited for relational integration (Bunge et al. 2005 [9]) | Treat "retrieve trusted traces" and "bind/align roles" as **two distinct stages with separate observables**; a failure in one must not be diagnosed as the other | Instrument trace count and role-score margin separately from the emitted answer (§5 F3/F4) | **A** for stage separability; **E** for using it as a debugging decision rule |
| M9 | Hippocampus–mPFC interaction supports **assimilation of new events into existing schemas** and contextual retrieval of the appropriate memory (Preston & Eichenbaum 2013 [10]; Bowman & Zeithamova 2018 [11]) | Role centres are learned from the demonstration set; candidate retrieval is filtered by that centre, not by generic familiarity | Per-family role centre computed per call from the trusted traces; no cross-family pooling | **A**; no mPFC homologue is claimed |
| M10 | **Schema congruence increases false memories**; schematic false alarms exceed non-schematic ones (Webb et al. 2016 [12]) | A schema must only *propose*; congruence alone is not evidence. Require strict winner + admitted trace + no writes | Unassertable hypothesis, congruence floor, tie → `None`, no candidate below floor | **S** for the false-memory effect; **E** for the specific gate design |
| M11 | **Source monitoring** is a flexible, error-prone attribution of information to its origin; item and source memory dissociate (Johnson et al. 1993 [13]) | Provenance is computed and stamped **at write time**; the reader cannot reconstruct it later | Unchanged `Provenance(source='conversation', source_type='user')` and `TRUSTED_EDGE_KINDS`; never relax `_is_trusted_edge` | **S** for the theory; **A** for the stamping design |
| M12 | Left anterior PFC activity increases with the demand to retrieve **specific perceptual/source detail**, not merely old/new familiarity (Ranganath et al. 2000 [14]) | Retrieval must expose *which* demonstration admitted the candidate, not just a familiarity score | Keep a per-call `supporting_trace_ids` field on the proposed answer for instrumentation; it is not written to the graph | **A** for specificity-vs-familiarity; **E** for the logging field |

### 1a. The four research questions, answered from the verified literature

* **(a) How relations are represented as roles with bindings.** Relational
  memory theory makes the relation, not the item, the unit of flexible
  inference [1,2]; structure-mapping makes relations and higher-order relations
  the mapped content while attributes are not mapped [3]; constraint
  satisfaction adds one-to-one structural consistency as an explicit mapping
  constraint [5]. **Design consequence:** `R(x, y)` with ordered named roles and
  a provenance stamp, not a similarity-weighted bag of node vectors.
* **(b) What cross-domain analogical mapping requires.** Content-independent
  mapping rules and systematicity [3], structural isomorphism and pragmatic
  centrality in addition to semantic similarity [5], and integration of
  retrieved relations as a step separable from retrieval itself [9].
  **Design consequence:** align by role structure first; lexicon must not be
  the answer key. Two exemplars help induce the schema that makes transfer
  possible [4].
* **(c) How many exemplars and what prior schemas.** Two prior analogs, not
  one, reliably yielded an induced schema and transfer in Gick & Holyoak [4];
  one-trial assimilation depended on a *pre-existing* schema in Tse et al. [6].
  **Design consequence:** keep `min_support=2` and a relation vocabulary
  already known to the engine; do not treat a single demonstration as a
  schema, and do not expect transfer into a completely empty schema.
* **(d) Why congruence causes false memories and what prevents confabulation.**
  Schematic items produced more false alarms than non-schematic ones [12];
  source attributions are constructed from features and are fallible [13];
  specific source detail requires extra anterior-prefrontal monitoring [14].
  Network-level integration of new memories into prior knowledge depends on
  hippocampal–prefrontal interaction [10]. **Design consequence:** congruence
  is a proposal signal only; source/observational admission, strict winner and
  fail-closed ties are the anti-confabulation controls. Rejecting on a tag
  alone would not be source monitoring; the gate must preserve its existing
  content checks.

---

## 2. Verified citation ledger

Every entry was resolved during this research pass against the publisher,
PubMed, PMC, APA PsycNET, Crossref metadata in the publisher page, or an
institutional repository. Nothing below is recalled from memory.

| # | Citation | DOI | Verification |
|---|---|---|---|
| 1 | Cohen, N. J., & Eichenbaum, H. (1993). *Memory, Amnesia, and the Hippocampal System*. Cambridge, MA: MIT Press. | book — no DOI | Google Books record; MIT Press 1993, 330 pp.; thesis text quoted in publisher listing and in Eichenbaum's "What H.M. Taught Us" reading |
| 2 | Eichenbaum, H. (2004). Hippocampus: cognitive processes and neural representations that underlie declarative memory. *Neuron, 44*(1), 109–120. | 10.1016/j.neuron.2004.08.028 | PubMed PMID 15450164; full-text PDF at the DOI |
| 3 | Gentner, D. (1983). Structure-mapping: A theoretical framework for analogy. *Cognitive Science, 7*(2), 155–170. | 10.1207/s15516709cog0702_3 | Wiley abstract page; APA PsycNET record 1984-14564-001; CiNii |
| 4 | Gick, M. L., & Holyoak, K. J. (1983). Schema induction and analogical transfer. *Cognitive Psychology, 15*(1), 1–38. | 10.1016/0010-0285(83)90002-6 | APA PsycNET 1983-20819-001; UMich Deep Blue handle 2027.42/25331; FORRT replication atlas |
| 5 | Holyoak, K. J., & Thagard, P. (1989). Analogical mapping by constraint satisfaction. *Cognitive Science, 13*(3), 295–355. | 10.1207/s15516709cog1303_1 | Wiley full abstract and citation block; APA PsycNET 1990-16295-001; dblp |
| 6 | Tse, D., Langston, R. F., Kakeyama, M., Bethus, I., Spooner, P. A., Wood, E. R., Witter, M. P., & Morris, R. G. M. (2007). Schemas and memory consolidation. *Science, 316*(5821), 76–82. | 10.1126/science.1135935 | science.org abstract and author list; PubMed PMID 17412951 |
| 7 | McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review, 102*(3), 419–457. | 10.1037/0033-295X.102.3.419 | APA PsycNET doiLanding and record 1995-42327-001; PubMed PMID 7624455 |
| 8 | Yassa, M. A., & Stark, C. E. L. (2011). Pattern separation in the hippocampus. *Trends in Neurosciences, 34*(10), 515–525. | 10.1016/j.tins.2011.06.006 | PMC3183227 full text; APA PsycNET 2011-16285-001; PubMed PMID 21788086 |
| 9 | Bunge, S. A., Wendelken, C., Badre, D., & Wagner, A. D. (2005). Analogical reasoning and prefrontal cortex: evidence for separable retrieval and integration mechanisms. *Cerebral Cortex, 15*(2), 239–249. | 10.1093/cercor/bhh126 | Publisher PDF (Berkeley Bunge Lab mirror) with DOI and abstract; cited with DOI in PMC3254816 |
| 10 | Preston, A. R., & Eichenbaum, H. (2013). Interplay of hippocampus and prefrontal cortex in memory. *Current Biology, 23*(17), R764–R773. | 10.1016/j.cub.2013.05.041 | PMC3789138 "View on publisher site" DOI link; citation block in arXiv 2104.00899 |
| 11 | Bowman, C. R., & Zeithamova, D. (2018). Abstract memory representations in the ventromedial prefrontal cortex and hippocampus support concept generalization. *Journal of Neuroscience, 38*(10), 2605–2614. | 10.1523/JNEUROSCI.2811-17.2018 | J Neurosci abstract and full-text PDF; PMC5858598; PubMed PMID 29437891 |
| 12 | Webb, C. E., Turney, I. C., & Dennis, N. A. (2016). What's the gist? The influence of schemas on the neural correlates underlying true and false memories. *Neuropsychologia, 93*(Pt A), 61–75. | 10.1016/j.neuropsychologia.2016.09.023 | PMC5354939 full text; APA PsycNET 2017-00257-008; PubMed PMID 27697593 |
| 13 | Johnson, M. K., Hashtroudi, S., & Lindsay, D. S. (1993). Source monitoring. *Psychological Bulletin, 114*(1), 3–28. | 10.1037/0033-2909.114.1.3 | APA PsycNET doiLanding and record 1993-40167-001; OpenAlex |
| 14 | Ranganath, C., Johnson, M. K., & D'Esposito, M. (2000). Left anterior prefrontal activation increases with demands to recall specific perceptual information. *Journal of Neuroscience, 20*(22), RC108. | 10.1523/JNEUROSCI.20-22-j0005.2000 | J Neurosci rapid communication page and PDF; PubMed PMID 11069977 |

### DOI corrections to the task brief (do not propagate the supplied values)

* The brief's `10.1038/nrn1406` for "Eichenbaum 2004" is **not** Eichenbaum.
  It resolves to **Wise, R. A. (2004), "Dopamine, learning and motivation",
  *Nature Reviews Neuroscience, 5*, 483–494** (nature.com/articles/nrn1406;
  PubMed PMID 15152198). The verified Eichenbaum 2004 declarative-memory
  review is [2] above, DOI `10.1016/j.neuron.2004.08.028`.
* The brief's `10.1146/annurev-neuro-062012-170349` for "Preston &
  Eichenbaum 2013" resolves to **Merchant, Harrington & Meck (2013), "Neural
  Basis of the Perception and Estimation of Time", *Annual Review of
  Neuroscience, 36*, 313–336** (annualreviews.org). The verified Preston &
  Eichenbaum 2013 review is [10] above, DOI `10.1016/j.cub.2013.05.041`.
* Related in-repo correction: `RSI_ALL_RELATIONS.md:106` attributes Tse et al.
  2007 to "Tse, Bone & McClelland, Science 316(5828):1179–1183". The publisher
  and PubMed give Tse *et al.*, *Science* **316**(5821), 76–82; use [6].

**Could not verify / deliberately excluded:** no DOI was found for the Cohen &
Eichenbaum 1993 book (book, cited by bibliographic record only). A candidate
Christoff et al. 2001 NeuroImage DOI was tested and found to belong to an
unrelated article, so that paper is **not cited here**. No claim is made that
RAVANA's GloVe cosine implements any identified neural computation.

---

## 3. Minimal, gate-preserving design

This section is a specification for a future implementation. It changes
nothing today. It is written so that each element is a pure, stateless
transformation over data already reachable from `eng.graph` and the answer
path described in `RSI_SCHEMA_WIRING.md`.

### 3.1 Represent bound role traces from trusted edges

```
BoundRoleTrace := {
  trace_id,                      # stable id of the trusted edge
  relation_type,                 # causal | part_of | temporal | analogical ...
  roles: (r1, r2),               # named by family, e.g. CAUSER/EFFECT
  bindings: {r1: node_x, r2: node_y},
  provenance: existing source_metadata (unmodified),
  slot_features: {r1: vector_x, r2: vector_y}   # cached, read-only
}
```

* Source: only edges already admitted by the unchanged `_is_trusted_edge`
  predicate. **No new trust path, no new edge_kind, no lowered threshold.**
* Role vocabulary is derived from `relation_type`, not from item wording:
  CAUSE = `(causer, effect)`, PART_OF = `(whole, part)`, PRECEDES =
  `(earlier, later)`, RESEMBLES = `(source, resembling)`. The role names are
  **our operationalisation (E)**; what is brain-supported is that a relation
  has argument roles that can be rebound [1–3].
* Each distinct trusted triple is one trace; duplicates do not increase
  support [8]. A trace is immutable: no hypothesis writes back to the graph
  [7].
* `slot_features` are the existing stored node vectors. They are *features of
  a filler in a role*, not the answer key (§3.3).

### 3.2 Induce an unassertable role schema

* For each `relation_type` with `≥ 2` distinct traces, build
  `RoleSchema(relation_type, roles, role_centre_r1, role_centre_r2,
  assertable=False)` where each role centre is the mean of the trace slot
  features for that role (unit-normalised), plus a **role-consistency range**
  from the observed within-role similarities.
* The schema is constructed per call and discarded. It is never asserted,
  persisted, or used to mint an edge. This keeps the existing CLS-style
  separation [7] and the schema-as-proposal rule [6,12].

### 3.3 Align two traces across domains

Alignment is an explicit three-check operation, not a cosine:

1. **Type check:** the query frame's relation and the schema relation must be
   identical. No cross-family pooling (this is what keeps a CAUSE demo from
   licensing a PART_OF answer). [3,5]
2. **Role check:** bind the query subject to the *same named role* it occupies
   in the demonstrated traces (subject → `r1` for the four configured
   families). Verify a one-to-one role mapping; a role swap is rejected unless
   the assertion itself was a known inverse phrase already canonicalised at
   write time. [3,5]
3. **Slot-feature check:** compute `sim(subject_vec, role_centre_r1)` for the
   *subject only*. This asks "can `thaw` occupy the causer role?", not "which
   option is closest to `thaw`?" Pass only if the subject clears the observed
   within-role range; otherwise fail closed.

This is the single substantive change: today the option ranking is driven by
candidate–subject / candidate–node cosine; after the change, cosine is used
**only as a slot-typing gate on the subject**, and the option is chosen by its
fit to the complementary role centre `role_centre_r2`.

**Why this can move far transfer.** Far items use held-out domain
vocabularies, so raw similarity between subject and gold option is weak. Role
fit asks a different question — does the option occupy the *effect/part/later/
resembling* slot given the demonstrated role structure — and can in principle
score a lexically distant but structurally appropriate candidate above floor.
That it actually does so is a prediction (§4), **not** an observed result.

### 3.4 Rank candidates without weakening foil/negative rejection

For each option `o`:

```
role_fit(o)       = cos(vec(o), role_centre_r2)
structural_bonus = 1 if type check and role check pass else -inf
score(o)          = structural_bonus * role_fit(o)
```

Admission rules, all inherited or tightened, never relaxed:

1. If a **directly taught trusted edge** already decides the query, defer to
   it; the hypothesis path returns `None` [6,13].
2. `admit_for_query()` still requires an observed trusted trace of that
   relation; congruence alone never licenses a candidate [6,12].
3. A candidate must beat the **congruence floor** derived from the weakest
   demonstrated instance (same rule as today, now applied to role fit).
4. **Strict winner**: ties or no candidate above floor → `None` → existing
   hedge/reject path. Duplicated evidence cannot create a winner [8].
5. The answer remains a proposal at the hedge surface; no `assertable=True`,
   no graph mutation, no change to CAR chains, source-monitor gate, trust
   predicate, Wilson/Welford gates, `THETA_*`, or benchmark files.
6. Flag-gated (`RAVANA_SCHEMA=1` or a successor flag); flag OFF byte-identical.
7. Instrument only: attach `supporting_trace_ids` to the proposal so a later
   audit can answer "which demonstration admitted this?" [14]. Logging is not
   a new gate.

### 3.5 Cost on a ~20 W CPU

All operations are table lookups and dense-vector dot products over a handful
of edges and four options: `O(k·d)` to form two role centres and `O(4·d)` to
rank, with `d` the existing embedding dimension and `k ≤ 8` demonstrations.
No LLM call, no training, no new model download beyond vectors already
present, no graph rewrite. The design is deliberately smaller than ACME's
network of competing mapping hypotheses [5]; this reduction is an
**extrapolation (E)** and is the first thing to fail if the role signal is too
weak (§5 F3).

### 3.6 Ledger for Section 3 design claims

| Design claim | Source | Verdict | Limit |
|---|---|---|---|
| Ordered named roles beat a bag of co-occurring words | [1–3] | **SUPPORT (framework)** | Human/harness theories; vector slot features are not human role representations |
| ≥ 2 distinct traces before a schema may propose | [4,6] | **SUPPORT (behavioural analogy)** | `min_support=2` is an operational threshold, not a measured neural constant |
| Align by relation type + role, not lexical overlap | [3,5] | **SUPPORT (framework)** | Our one-to-one check is a minimal stand-in for full structure mapping |
| Subject slot-feature gate; option ranked by complementary role centre | — | **EXTRAPOLATION** | No cited source licenses this exact scoring rule; §4/§5 decide its fate |
| Congruence floor + tie reject prevent confabulation | [12–14] | **SUPPORT (principle) / EXTRAPOLATION (implementation)** | Human false-memory/source-monitoring findings do not validate specific thresholds |
| Unassertable, stateless, no writes | [6,7] | **SUPPORT (analogy)** | Software analogue of CLS / assimilation, not a neural homology claim |
| Separating retrieval from binding as a debug axis | [9] | **SUPPORT (analogy)** | Human fMRI dissociation; used here as an instrumentation decision rule |

---

## 4. Pre-registered predictions (to be measured by the parent, not here)

Prerequisites: identical store and seed, one process, structural k=8, flag ON
versus flag OFF differential first, fake arm unchanged. Numbers are the
parent's existing focused-live summariser endpoints; this design adds no new
metric.

| ID | Prediction | Baseline to beat / hold | Outcome that supports role binding | Outcome that falsifies it |
|---|---|---|---|---|
| P1 | PART_OF near rises above 0 | 0.00 (wrong `thunder→chill`) | > 0 at k=8 | stays 0 with trusted PART_OF traces present and a role margin > 0 → articulation, not binding (F4) |
| P2 | PRECEDES near rises above 0 | 0.00 (wrong `cloud→glow`) | > 0 at k=8 | same as P1 |
| P3 | CAUSE near does **not regress** | ≈ 0.50 (`thaw→mist` correct) | ≥ baseline | any drop, or a new wrong CAUSE candidate emitted → gate regression, stop |
| P4 | RESEMBLES near does **not regress** | ≈ 0.50 (`drizzle→chill` correct) | ≥ baseline | any drop → stop |
| P5 | Far moves above 0 **if** cross-domain alignment works | 0.00 across four families | > 0 with subject-slot gate passing and role margins separating gold | 0 with passing gates → slot-centre extrapolation fails for far vocab (F3) |
| P6 | Foil rejection held | currently correctly rejected | no decrease; ideally unchanged | any foil now admitted → confabulation signal (F5) |
| P7 | Negative-rule rejection held | currently correctly rejected | no decrease | any negative now admitted → stop |
| P8 | `growth_headroom > 0` (k=8 − k=1 positive), not flat 1.00 | prior probes show 0 growth | positive but unsaturated | flat 0 or instant 1.00 → no learning signal / G6 saturation (F6) |
| P9 | Flag OFF byte-identical; fake oracle unchanged | — | exact match | any difference → hook leaked outside its flag |
| P10 | Directly taught edge still wins over hypothesis | `TRUSTED_EDGE_CHOICE.md §4` | deferral observed | hypothesis overrides an observed edge → stop |

Pre-registration note: these are **conditional predictions of a design**, not
claims of measured improvement. No run was performed in this worktree.

---

## 5. Falsifiers and stop rules

Each falsifier is a *decision procedure*, not a licence to tune. The universal
stop rule is: **do not edit a trust predicate, gate threshold, benchmark item,
gold answer, or harness to make a metric move.** A gain that appears only
after relaxing those surfaces is worse than a flat curve.

* **F1 — write-side, before binding.** Dump `source_metadata` on the exact
  resolved edge ids before and after each teaching turn. If PART_OF /
  PRECEDES / RESEMBLES edges lack the trusted stamp, there are no role traces
  to bind: stop here and fix the write route (see `RSI_ALL_RELATIONS.md §5`).
  A role-binding design cannot rescue an absent trace.
* **F2 — hypothesis-side.** If stamped traces exist but no hypothesis forms at
  support 2, inspect the grouping key (`relation_type` / `asserted_relation`)
  before touching the trust predicate. This is a bookkeeping defect, not a
  brain-grounding defect.
* **F3 — binding-side (the extrapolation fails).** If hypotheses form, the
  subject-slot gate passes, but role-fit scores do not separate the gold
  option from distractors while raw cosine does, then slot centres built from
  GloVe vectors are not a usable stand-in for relational roles. **Stop** the
  role-rank design; record that lexical similarity was the only working signal
  and that §3.3 is falsified for far transfer. Do not widen the floor or
  switch the score back to cosine and call it role binding.
* **F4 — retrieval/articulation, not binding.** If an offline diagnostic shows
  the gold option ranked first by role fit with a strict margin and an
  admitted supporting trace, yet the emitted answer still names no candidate
  (or near stays 0), the bottleneck is **retrieval/articulation**, not
  representation. Bunge et al. [9] motivate keeping these stages separable:
  instrument every early exit, correction guard, hedge emitter and final
  emission guard in one pass (`TRUSTED_EDGE_CHOICE.md §5` is the precedent)
  and stop editing the ranker. A converse diagnostic — gold never ranked
  first — exonerates articulation and keeps the defect in binding.
* **F5 — confabulation.** Any of: a candidate emitted with no qualifying
  hypothesis or no admitted trusted trace; a hypothesis becoming assertable;
  an edge written by the hypothesis path; foil or negative metrics dropping;
  far rising while foil/negative fall; a winner chosen from a tie; or a gain
  that requires relaxing trust/gates. **Stop immediately**, revert to the
  previous answer path, and treat the raised score as a false-memory analogue
  [12–14], not a capability gain.
* **F6 — saturation.** If any endpoint reaches 1.00 (especially fake and live
  together), label it G6 saturation — a scaffold handing the grader the token
  — and require `growth_headroom > 0` plus unchanged foil/negative before
  claiming learning. Do not "improve" a saturated table.
* **F7 — no effect at all.** If flag ON is byte-identical to flag OFF and fake
  stays 1.00, the new path was never reached. Verify the hook site; do not
  infer that the brain design was tested.

---

## 6. Honest caveats

* This document is a **design and literature review**. No implementation,
  engine run, pytest, live census or benchmark score was produced by this
  worktree. All numbers quoted are inherited from the cited in-repo notes.
* A ~20 W no-LLM graph system can do: immutable provenance-stamped triples,
  small fixed role vocabularies, per-call schema views, mean vectors and dot
  products, strict fail-closed ranking. It cannot do: human systematicity,
  higher-order relational projection, dynamic schema creation from an empty
  knowledge base, metacognitive source monitoring beyond its metadata, or
  recovery of information never stored.
* GloVe node-vector similarity is **associative**, not relational. Re-labelling
  it "role fit" does not make it structural alignment; F3 exists precisely to
  kill that self-deception.
* Human evidence is behavioural and neuroimaging evidence in humans/rodents.
  Mapping it to a Python graph is **analogy at best** (ledger `A`), never a
  claim that RAVANA has a hippocampus, mPFC, or human-like analogical
  competence.
* Even if P1–P5 hold on the n=4 focused probe, the full 120-item sweep and cold
  reload/save behaviour remain untested. Small-n gains are mechanism evidence
  only.
* The two DOIs supplied in the brief were incorrect and have been replaced
  with verified records (§2); this matters because an unverified citation in a
  design doc can launder a false mechanism claim.

---

## Sources

[1] Cohen & Eichenbaum 1993 (book, MIT Press). [2] Eichenbaum 2004, *Neuron*
44(1):109–120, doi:10.1016/j.neuron.2004.08.028. [3] Gentner 1983, *Cognitive
Science* 7(2):155–170, doi:10.1207/s15516709cog0702_3. [4] Gick & Holyoak
1983, *Cognitive Psychology* 15(1):1–38, doi:10.1016/0010-0285(83)90002-6.
[5] Holyoak & Thagard 1989, *Cognitive Science* 13(3):295–355,
doi:10.1207/s15516709cog1303_1. [6] Tse et al. 2007, *Science* 316(5821):76–82,
doi:10.1126/science.1135935. [7] McClelland, McNaughton & O'Reilly 1995,
*Psychological Review* 102(3):419–457, doi:10.1037/0033-295X.102.3.419. [8]
Yassa & Stark 2011, *Trends in Neurosciences* 34(10):515–525,
doi:10.1016/j.tins.2011.06.006. [9] Bunge et al. 2005, *Cerebral Cortex*
15(2):239–249, doi:10.1093/cercor/bhh126. [10] Preston & Eichenbaum 2013,
*Current Biology* 23(17):R764–R773, doi:10.1016/j.cub.2013.05.041. [11] Bowman
& Zeithamova 2018, *J Neurosci* 38(10):2605–2614,
doi:10.1523/JNEUROSCI.2811-17.2018. [12] Webb, Turney & Dennis 2016,
*Neuropsychologia* 93(Pt A):61–75, doi:10.1016/j.neuropsychologia.2016.09.023.
[13] Johnson, Hashtroudi & Lindsay 1993, *Psychological Bulletin*
114(1):3–28, doi:10.1037/0033-2909.114.1.3. [14] Ranganath, Johnson &
D'Esposito 2000, *J Neurosci* 20(22):RC108,
doi:10.1523/JNEUROSCI.20-22-j0005.2000.
