# GOAL_UNSEEN_TYPE_TRANSFER — autonomous attack on unseen-subject type/role transfer

**Status: scratch experiment + docs only. NO production implementation.**
No `CognitiveChatEngine` was constructed, no `process_turn` called, no store written,
no engine booted (a parent WFA full measurement was active), and no file under
`ravana/src/`, `scripts/`, `tests/` or any item bank, gate or threshold was edited.
Worktree `C:/Users/Likhith/Documents/projects/ravana_goal_transfer`, branch
`goal/unseen-type-transfer` from main `b75a24e`. The two modules imported were the
**already-merged, unmodified** typed-slot primitive
(`ravana/src/ravana/chat/schema_induction.py`) and the shipped saturation gate
(`scripts/front_gates.py`), both read-only; their SHA-256 was asserted identical
between main and worktree before any number was read.

**Standing goal.** Achieve what the current system cannot: generalise a taught
type/role to an **unseen subject** in RAVANA. This doc reports two distinct
hypotheses, each run on the 40 recorded live TRA cases (160 graded rows), with
exact numbers, foil/negative rejection, G6 flags, and a deployable-vs-changed-benchmark
verdict. Both are **negative for the standing goal**; one yields a genuine but
narrower capability whose implementation plan is given below, explicitly not applied.

---

## 1. Research citations (verified; ledger inherited from `TYPE_LEARNING_BRAIN.md`)

All DOIs below resolve on Crossref/PMC. The fast-mapping result is that the type of
a word comes from a **subject-specific input**, never from assertions about other words:

| # | Source | Locator | Relevance here |
|---|---|---|---|
| C1 | Carey & Bartlett (1978), *Acquiring a single new word* | Stanford Papers & Reports on Child Lang. Dev. 15:17–29 (no DOI) | `chromium` was known to be a **colour word** from the contrastive phrase "not the red tray, the chromium one" — the type came from the contrast, not other exemplars. |
| C2 | Dollaghan (1985), *J Speech Hear Res* 28(3):449–454 | doi:10.1044/jshr.2803.454 | 91% map novel word→referent on first exposure **with a referent present**. |
| C3 | Medina, Snedeker, Trueswell & Gleitman (2011), *PNAS* 108:9014–9019 | doi:10.1073/pnas.1105040108 | ~90% of natural word-learning instances are uninformative; one-trial mapping retains a **single** hypothesis. |
| C4 | Xu & Tenenbaum (2007), *Psychol Rev* 114(2):245–272 | doi:10.1037/0033-295X.114.2.245 | Sparse induction works only over a **pre-existing structured hypothesis space**; "suspicious coincidence" is informative only because `dalmatian ⊂ dog ⊂ animal` is already available. |
| C5 | Tenenbaum & Griffiths (2001), *BBS* 24(4):629–640 | doi:10.1017/S0140525X01000061 | Generalisation depends on the **representational structure**, not raw stimulus count. |
| C6 | Landauer & Dumais (1997), *Psychol Rev* 104(2):211–240 | doi:10.1037/0033-295X.104.2.211 | Meaning from language needs a **corpus containing the word**; a centroid of other words is not that. |
| C7 | Tse et al. (2007), *Science* 316:76–82 | doi:10.1126/science.1135935 | One-trial assimilation becomes durable **only if a neocortical schema already exists**. |
| C8 | McClelland, McNaughton & O'Reilly (1995), *Psychol Rev* 102(3):419–457 | doi:10.1037/0033-295X.102.3.419 | Fast episodes + slow interleaved regularity extraction are complementary. |
| C9 | Lany & Saffran (2010), *Psychol Sci* 21(2):284–291 | doi:10.1177/0956797609358570 | Generalisation occurred only when **category-marking cues were in the input**. |
| C10 | Wang & Trueswell (2019), *Cogn Psychol* 114:101226 | doi:10.1016/j.cogpsych.2019.101226 (PMC6925454) | Children induce subordinate-level meanings from ostensive events **only when exemplars share a domain and contrast non-overlappingly** — a structure absent from TRA teaching. |
| C11 | Remon, Loevenbruck, Deudon et al. (2020), *J Exp Child Psychol* 196:104859 | doi:10.1016/j.jecp.2020.104859 (PMC7262577) | Explicit ostensive naming creates a durable memory trace in **one learning event** — supports direct-type-teaching (H_a), not transfer from other words. |

**Prediction carried into the experiments.** Every cited mechanism obtains the
unseen word's type from an input that names or contexts *that word*. If the query
subject has no such input, the mutual information between the available signal and
the gold type is ~0 by construction; the correct behaviour is abstention, and the
problem is an **information deficit, not a ranking defect**
(`TYPE_LEARNING_BRAIN.md` §0, §4).

---

## 2. Instrument (validated before any number was read)

| Check | Result |
|---|---|
| Merged primitive SHA-256 (main == worktree) | `d17cf6461d208e2e…` identical; `__file__` under worktree `ravana/src` |
| Shipped G6 gate SHA-256 | `3ede0efc75d582e9…` (`scripts/front_gates.py`, unmodified) |
| Item bank SHA-256 (main == worktree) | `7bc55db8637fb4b2…`; `build_items()` 400, `validate_items()` 0 errors |
| Recorded rows SHA-256 | `c468f94263f0f7c4…`; 13,440 total, **12,320 fake excluded**, 1,120 `backend=live` |
| Graded selection | 160 rows = one `structural`/`query`/non-permuted row per (item, condition); 40 items × 4 conditions |
| Bank cross-check | all 160 subjects, candidate sets and gold match the regenerated bank (0 mismatches) |
| No engine / no key | `CognitiveChatEngine` never imported; no store write; gold/`type_sets`/`world_model` never passed to a scorer |
| Frozen constants | congruence floor `0.20`, strict margin `0.10` (from the primitive), curated write confidence `0.8`, two disjoint/unrelated controls — all fixed before run 1 |

**Information audit (recomputed, matches prior docs):** the query subject's own
authored type set contains the taught `type_label` **0/160** times; the **gold
candidate's** type set contains it **80/80** times. The answer lives on the
candidate side, exactly as `TYPE_ACQUISITION_SIGNAL_EXPERIMENT.md` E8 found.

**Instrument correction, run 1 → run 2.** Run 1 (`*_results_draft_r1.json`,
`*_run_draft_r1.log`) defined H_b's foil by offering an "other" type absent from
candidates; the fallback type was itself held by candidates, so that column
measured the fallback, not rejection. Run 2 states the same gold type for
foil/negative as for near/far (foil keeps the original TRA anchor-containing
option set; negative removes trusted edges for the type). No threshold, gate or
confidence value changed. Run 1 is retained for audit and never averaged in.

---

## 3. H_a — TRA-OST: ostensive subject-type teaching, then read with the merged primitive

**Design (a NEW benchmark variant).** Per case, add one provenance-stamped
conversational teaching turn stating the **query subject's** type
(`{subject} is a {type_label}.`) and materialise it as a trusted `is_a`
`TypedSlotEdgeView` (`edge_kind='curated'`, `source='conversation'`,
`source_type='user'`, confidence `0.8`) — the exact shape the merged
`slot_keyed_forced_choice` consumes. Conditions:
near/far = positive subject turn; foil = positive turn about a **different,
demonstrated** word (wrong-subject control); negative = **negated** turn (writes
no trusted edge). Two query option spaces are graded:
(i) the 6 canonical type labels (the new variant's task), and
(ii) the original TRA domain-word candidates (the transfer question).

### EVIDENCE (measured, n = 40 per condition)

| Arm / option space | near | far | foil (reject) | negative (reject) | near+far |
|---|---|---|---|---|---|
| **OST, type-label options** | **40/40** | **40/40** | 40/40 | 40/40 | **80/80** |
| OST, original TRA word options | 0/40 | 0/40 | 40/40 | 40/40 | 0/80 |
| No subject-type teaching (both spaces) | 0/40 | 0/40 | 40/40 | 40/40 | 0/80 |

- **Foil/negative rejection** is 40/40 in the type-label space **by abstention**:
  `slot_keyed_forced_choice` returned `pick=None` because the trusted edge was
  either from the wrong subject (foil) or absent (negated). `reason` was
  `no_trusted_typed_slot_above_floor` in every such row — a genuine fail-closed
  property of the merged primitive, not a loosened gate.
- **Controls PASS:** disjoint options `apple/banana/chair/table` → `None`;
  unrelated subject `zeppelin` → `None`; an untrusted `seed/web` edge → `None`;
  a non-`is_a` relation (`causes`) → `None`. All 160 type-label rows are identical
  across primary and sealed pools (20/20 each).
- **G6 (shipped `front_gates.gate_saturation`, non-blocking FLAG):**
  type-label arm — `tra_near=100.0`, `tra_negative_reject=100.0` → **FLAG** (both
  fronts saturated). Original-word arm — `tra_near=0.0`, `tra_negative_reject=100.0`
  → **FLAG**. A saturated PASS is not progress evidence (`ravana-debugging` §1).

### Verdict for H_a

**The merged typed-slot primitive CAN consume an ostensively taught,
provenance-stamped subject type: 80/80.** This is a genuine mechanism finding —
it closes the open F4 question in `TYPE_LEARNING_BRAIN.md` §6 (the architecture is
**not** the limit for *explicitly taught* types). But it is **a changed benchmark,
not unseen-subject transfer**, for three measured reasons:

1. The benchmark now **supplies the answer** as a teaching turn; the subject is
   no longer unseen in its own type. This is direct type learning-and-use.
2. The result is **G6-saturated** (100/100) with no headroom — a mechanism check,
   not capability growth.
3. It gives **0/80** on the original TRA domain-word task: teaching "tempest is
   an effect" does not tell the system which of `squall/shower/breeze/sunset` is
   the effect. The gold candidate's type is still never taught.

**Deployable capability (narrower, honest statement):** *given a provenance-stamped
explicit type assertion about a subject, use the already-merged typed-slot
primitive to answer a type-labelled query while rejecting wrong-subject and
negated teaching.* That is real and implementable, but it is **not** the standing
goal.

---

## 4. H_b — TRA-VOCAB: a broad conversational type channel before the query

**Design (a second NEW benchmark variant, explicitly separated from original
TRA).** Before the query, conversationally teach `is_a` types for a broad
vocabulary through the same provenance shape as H_a. In this scratch replay the
curriculum is populated from the item bank's authored ontology (84 words, **504
trusted `is_a` edges**), with all 40 recorded query subjects excluded so every
tested subject remains unseen. **In original TRA that ontology is oracle-only, so
this arm is changed-benchmark evidence by construction**, never an original-TRA
claim. Item teaching turns supply `TrustedTrace`s; `induce_schema_hypotheses`
(min_support 2) forms unassertable hypotheses; `rank_candidates_for_unseen_subject`
ranks candidates from scores supplied by the channel. The query states the target
type `T` (the new variant); near/far use the gold type, negative removes trusted
edges for `T`, foil keeps the original anchor-containing option set.

### EVIDENCE (measured, n = 40 per condition)

| Arm | near | far | foil (reject) | negative (reject) | near+far |
|---|---|---|---|---|---|
| `VOCAB_typed_query` (type lexicon lookup) | 3/40 | 4/40 | 28/40 | 40/40 | **7/80** |
| `VOCAB_typeblind` (any taught candidate) | 3/40 | 4/40 | 28/40 | 31/40 | 7/80 |
| ORACLE (also teach the subject's type) — **non-deployable** | — | — | — | — | 80/80 |

- **Foil/negative rejection:** typed-query foil 28/40 — **12 foil leaks**, all
  picked words are demonstrated anchors the channel typed as `T` (e.g. `breeze`
  ×4, `lung` ×3), i.e. the lookup endorses the planted anchor the TRA foil exists
  to catch. Negative 40/40 by abstention (all candidate scores 0). The type-blind
  arm leaks 9 negatives. No deployable arm has both near/far > 0 and clean
  foil/negative rejection.
- **Controls PASS:** disjoint options → `None`; unrelated subject with no scores
  → `None`.
- **G6:** typed-query `tra_negative_reject=100.0` → **FLAG** (abstention ceiling,
  not evidence); `tra_near` is in no band. Non-blocking.

### Decisive subject-independence control

For every near/far case the same candidates, scores and hypotheses were ranked
once with the real unseen subject and once with the unrelated token `zeppelin`.
**Subject-dependent picks: 0/80.** Whenever the mechanism emitted a pick it was
**identical** for the unseen subject and `zeppelin`; the other 62 cases abstained
for both. The 7 "hits" therefore carried **zero information about the subject** —
they are candidate-side type lookups that would fire for any token.

### Located bottleneck (not a threshold problem)

80 near/far cases → **52** have an admitted schema hypothesis (item `k ≥ 2`;
28 are `k ∈ {0,1}` with support < 2) → **23** of those have the gold candidate
typed by the channel → only **7** actually pick gold (16 abstain because another
candidate also holds `T`, so the strict-margin gate correctly refuses a tie).
Even a perfect broad type lexicon cannot type the unseen subject, because the
subject contributes nothing to the decision (C1–C3, C9, C10).

### Verdict for H_b

**Negative for the standing goal.** The channel is a genuine knowledge source and
a real capability for *candidate-side type lookup* (a changed benchmark whose
query states the type), but it is **not subject typing**: the subject-independent
control is 0/80, it leaks 12/40 foils, and near+far is 7/80. It confirms the
information deficit rather than overcoming it.

---

## 5. Decision

| Hypothesis | Unseen-subject transfer? | Genuine deployable capability? | Changed benchmark? |
|---|---|---|---|
| H_a (ostensive subject-type teaching) | **No** (0/80 original word task; answer supplied) | **Yes, narrower** — explicit taught-type use, 80/80 type-label space, clean wrong-subject/negation rejection | **Yes** — benchmark supplies the subject's type |
| H_b (broad conversational type channel) | **No** (subject-dependent 0/80; 7/80 lookup) | Only candidate-side type lookup, with 12/40 foil leaks | **Yes** — query states the target type; ontology is oracle-only in original TRA |

**The standing goal is NOT achieved by either route.** Both routes that move any
endpoint do so only after the benchmark is changed to supply information about the
test subject (H_a) or about the answer options (H_b). Under the **original** TRA
input regime — assertions about other words, a bare unseen token in a
relation-neutral forced-choice frame, no stored type evidence for that token —
the system's correct behaviour remains abstention, and all seven prior negative
experiments are consistent with this.

**No production edit, no default-on flag, no TRA PASS claim.** Per the
experiment-first rule, the only change made is scratch scripts + this document.

---

## 6. Precise implementation plan for the positive-but-narrower capability (H_a)

Documented for a future **docs-only PR / reviewed default-OFF prototype**; **not
implemented here**. All primitives exist; no gate or threshold changes.

1. **Teaching surface (write side).** In a new `TRA-OST` harness variant, emit one
   provenance-stamped conversational turn per taught type,
   `"{subject} is a {type_label}."`, through the unchanged
   `KnowledgeIntake.intake(fact, Provenance(source='conversation',
   source_type='user'))` path, producing `is_a` edges with `type_label` stamped in
   `source_metadata`. This is legitimate teaching material per
   `TRA_BENCHMARK_SPEC.md` §1/§2 P2c — **never** a copy of `type_sets`/`world_model`
   at query time.
2. **Read side.** Keep `slot_keyed_forced_choice` exactly as merged; it already
   ran 80/80 with clean wrong-subject/negation rejection. Change **nothing** in
   `schema_induction.py`.
3. **Flag wiring.** Ship behind the existing `RAVANA_TRA_SLOTS` default-OFF flag;
   prove flag equivalence (byte-identical outputs off vs on), as required by
   `ravana-debugging` §7a.
4. **Measurement before enabling.** Re-run with positive (near/far) and negative
   (foil/negative) endpoints, the disjoint/unrelated/untrusted controls, and the
   shipped G6 gate. Because the type-label arm is saturated by construction,
   record it as a **mechanism check with G6 FLAG**, not as learning progress; only
   a within-session k-curve with headroom could be called growth.
5. **Boundary.** This capability covers only subjects that have been explicitly
   taught a type. It must be presented as *explicit type learning and use*, never
   as unseen-subject transfer.

---

## 7. What new input the standing goal requires, and next step

The precise missing input is **subject-specific evidence for the unseen word**.
Under the current loop the subject appears in its own teaching surfaces 0/160
times and has no typed edge or definition. One of the following must become a
legitimate, non-oracle input; none is a mechanism or threshold tweak:

1. **An ostensive/contrastive teaching turn about the test subject** (C1, C2, C11)
   — but that makes the subject taught, not unseen (H_a). It yields capability
   §6, not transfer.
2. **A distributional channel over sentences that actually contain the test word**
   (C6, C9): frozen web snapshots counted for syntactic frames/type-frequency
   suffixes on the queried token, with held-out words excluded at item
   construction. This is **new data**, not a better scorer; a GloVe centroid is
   the measured-failing substitute (`TRA_CATEGORY_CENTRE_EXPERIMENT.md`).
3. **An independently authored structured type hierarchy supplied as a legitimate
   input** so that a handful of exemplars become diagnostic (C4, C5, C7). The
   item bank's `type_sets`/`world_model` are exactly such a hierarchy but are
   oracle-only; a deployable version must come from an external ontology or an
   independently learned type lexicon, never the answer key. H_b shows that even
   a broad candidate-side lexicon gives 0/80 subject-dependent decisions — the
   hierarchy must connect the **subject**, not just the options.

**Next step (in order, no tuning):** (a) decide which of (2) or (3) is an
acceptable new input for the benchmark; (b) if (3), author a candidate-type source
independent of the item bank and re-run this exact near/far/foil/negative +
subject-independence + disjoint + G6 harness; (c) if (2), capture frozen snapshots
containing the held-out tokens and add a frame-count channel; (d) only on a clean
positive with subject-dependent picks > 0 and foil/negative flat, consider a
default-OFF production prototype. If no new input is permitted, keep abstention
and relabel TRA as an information-availability test — do not lower the floor, the
margin or any gate.

---

## 8. Honest caveats

- n = 40 recorded items (20 primary / 20 sealed), 4 conditions each — mechanism
  replay, not the 400-family benchmark; no Wilson CI, k-curve or cold-reload
  endpoint.
- H_a/H_b teaching turns are constructed from the item bank's `type_label` and
  authored ontology because recorded rows carry empty `teaching_surfaces`; they
  exercise the merged primitives on the exact recorded subjects/options/gold, but
  are not transcripts of a live `process_turn`.
- H_b's ontology source is oracle-only in original TRA; that is why both moving
  arms are labelled changed-benchmark. No oracle arm is averaged into a verdict.
- `front_gates.gate_saturation` is non-blocking; FLAG is a caveat, never evidence.
- No threshold, gate, item or gold was edited after a result was seen; run 1 of
  H_b was retracted for the documented foil-fallback defect and superseded.

## 9. Files

- `scratch/goal_unseen_transfer_experiment.py` — both hypotheses, all controls, no engine/key
- `scratch/goal_unseen_type_results.json` — 160+160 rows, tallies, controls, G6, info audit
- `scratch/goal_unseen_transfer_run.log` — captured stdout, `EXIT=0`
- `scratch/goal_unseen_type_results_draft_r1.json`,
  `scratch/goal_unseen_transfer_run_draft_r1.log` — retracted H_b run 1, retained for audit
- this document
