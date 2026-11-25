# TYPE_LEARNING_BRAIN — what the brain actually uses to type an unseen word, and what RAVANA would need

**Status: STATIC research only.** No engine was booted, no `process_turn` was called, no
pytest was run, and no file under `ravana/src/`, `ravana_ml/src/`, `scripts/`, `tests/`
or any item bank, gate or threshold was edited. The only artifact is this document,
written into worktree `C:/Users/Likhith/Documents/projects/ravana_typelearn_brain` on
branch `docs/type-learning-brain` from main `bfe99a6`. All measured numbers quoted here
are read from previously committed investigation docs, not re-measured.

---

## 0. Bottom line (the single most important conclusion)

**Fast mapping is not type induction from other words.** Every brain mechanism the
literature supports obtains the type of a word from one of three inputs that RAVANA's
TRA/RSI loop does *not* supply: (i) an ostensive or contrastive referent present at the
learning moment (Carey & Bartlett 1978; Dollaghan 1985; Medina et al. 2011), (ii) a
**pre-existing structured hypothesis space** over kinds and properties that makes a
handful of exemplars informative (Xu & Tenenbaum 2007; Tenenbaum & Griffiths 2001; Tse
et al. 2007; McClelland et al. 1995), or (iii) distributional/morphological contexts
containing **the test word itself** (Lany & Saffran 2010; Feijoo et al. 2017; Gagliardi
et al. 2012). RAVANA's teaching turns contain assertions about *other* words; the query
contains a bare unseen token in a relation-neutral forced-choice frame; the graph holds
no type evidence for that token; and the only candidate signal, GloVe slot centres, are
non-separable (pairwise cosine 0.729–0.980, exact-centre top-2 margin 0.0198–0.0873
against a frozen 0.10 gate — `TRA_CATEGORY_CENTRE_EXPERIMENT.md`, E1).

Therefore the mutual information between the available input and the gold type is
approximately zero **by construction**. This is an **information deficit, not a ranking
defect**. No scorer, threshold, gate, schema hypothesis or embedding trick can create
information that was never presented. The minimal honest path is to make type knowledge
a real input: acquire explicit, provenance-stamped type assertions for a broad vocabulary
from genuine conversation and web reading (giving the system the structured prior a
child already has), and — for genuinely unseen words — add a real distributional channel
over sentences that contain that word. If neither is allowed, abstention is the correct
behaviour and the benchmark must be relabelled as an information-availability test.

---

## 1. Citation ledger — Crossref-verified 2026-11-25

All DOI records below were resolved through `https://api.crossref.org/works/<DOI>` on
2026-11-25; author, year, journal, volume, issue, pages and title were read from the
returned metadata, not reconstructed from memory. Two corrections were forced by this
check and are recorded in §1.1.

| # | Reference (as verified) | DOI / locator | What it directly supports |
|---|---|---|---|
| C1 | Carey, S. & Bartlett, E. (1978). *Acquiring a single new word.* Papers and Reports on Child Language Development, 15, 17–29 (Stanford University). | No DOI (proceedings volume). Primary PDF at `languagelog.ldc.upenn.edu/myl/Acquiring_a_Single_New_Word.pdf`; independently catalogued by Semantic Scholar (CorpusID 50145091) and cited in Carey's 2010 review. | One/few exposures create a *partial* lexical entry; fast mapping then "extended mapping" over months. Crucially, `chromium` was known to be a **colour word** from the contrastive phrase "not the red tray, the chromium one" — the type came from the contrastive situation, not from other exemplars. |
| C2 | Dollaghan, C. (1985). Child meets word: "Fast mapping" in preschool children. *Journal of Speech, Language, and Hearing Research*, 28(3), 449–454. | `10.1044/jshr.2803.454` | 91% inferred the novel word–referent connection on first exposure, 81% identified the referent on a second hearing. One-trial mapping **with a referent present**. |
| C3 | Medina, T. N., Snedeker, J., Trueswell, J. C. & Gleitman, L. R. (2011). How words can and cannot be learned by observation. *PNAS*, 108(22), 9014–9019. | `10.1073/pnas.1105040108` | ~90% of natural word-learning instances are uninformative and ~7% highly informative; learners retain a **single** meaning hypothesis and use a one-trial fast-mapping procedure, not cross-situational cross-tabulation. This is the direct answer to "can sparse language alone type a word?" — mostly no. |
| C4 | Xu, F. & Tenenbaum, J. B. (2007). Word learning as Bayesian inference. *Psychological Review*, 114(2), 245–272. | `10.1037/0033-295X.114.2.245` | Sparse induction works by likelihood over a **structured hypothesis space**; the "suspicious coincidence" of three Dalmatians is informative only because `dalmatian ⊂ dog ⊂ animal` is already available to the learner. |
| C5 | Tenenbaum, J. B. & Griffiths, T. L. (2001). Generalization, similarity, and Bayesian inference. *Behavioral and Brain Sciences*, 24(4), 629–640. | `10.1017/S0140525X01000061` | Generalization from multiple consequential stimuli depends on the **representational structure** of the hypothesis space (metric or set-theoretic); it is not a function of raw stimulus count. |
| C6 | Landauer, T. K. & Dumais, S. T. (1997). A solution to Plato's problem: The latent semantic analysis theory of acquisition, induction, and representation of knowledge. *Psychological Review*, 104(2), 211–240. | `10.1037/0033-295X.104.2.211` | Semantic similarity from **large text co-occurrence**; the canonical demonstration that meaning-from-language needs a corpus, not eight sentences. |
| C7 | Landauer, T. K., Foltz, P. W. & Laham, D. (1998). An introduction to latent semantic analysis. *Discourse Processes*, 25(2–3), 259–284. | `10.1080/01638539809545028` | Formal LSA setup and its corpus requirement. |
| C8 | McClelland, J. L. & Rogers, T. T. (2003). The parallel distributed processing approach to semantic cognition. *Nature Reviews Neuroscience*, 4, 310–322. | `10.1038/nrn1076` | Semantic knowledge as graded weights acquired by gradual adjustment to experience; conceptual coherence and category coherence are learned, not given. |
| C9 | Rogers, T. T., Lambon Ralph, M. A., Garrard, P., Bozeat, S., McClelland, J. L., Hodges, J. R. & Patterson, K. (2004). Structure and deterioration of semantic memory: A neuropsychological and computational investigation. *Psychological Review*, 111(1), 205–235. | `10.1037/0033-295X.111.1.205` | The actual 2004 Rogers/McClelland *journal* paper on PDP semantic memory (see §1.1). |
| C10 | Rogers, T. T. & McClelland, J. L. (2008). Précis of *Semantic Cognition: A Parallel Distributed Processing Approach*. *Behavioral and Brain Sciences*, 31(6), 689–714. | `10.1017/S0140525X0800589X` | Explicit statement that acquisition of semantic information involves **gradual** weight adjustment; six phenomena including early differentiation of concepts and reorganisations between ages 4 and 10. |
| C11 | Tse, D., Langston, R. F., Kakeyama, M., Bethus, I., Spooner, P. A., Wood, E. R., Witter, M. P., Morris, R. G. M. & Moser, E. I. (2007). Schemas and memory consolidation. *Science*, 316(5821), 76–82. | `10.1126/science.1135935` | One-trial assimilation becomes hippocampal-independent **only if a neocortical schema already exists**; schemas also causally support lasting one-trial associative memory. |
| C12 | McClelland, J. L., McNaughton, B. L. & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review*, 102(3), 419–457. | `10.1037/0033-295X.102.3.419` | Fast distinct episodes + slow interleaved extraction of regularities; the two computations are complementary, not interchangeable. |
| C13 | Yassa, M. A. & Stark, C. E. L. (2011). Pattern separation in the hippocampus. *Trends in Neurosciences*, 34(10), 515–525. | `10.1016/j.tins.2011.06.006` | Episodes must remain distinct; a blurred exemplar aggregate is not a memory of the individual traces. |
| C14 | Lany, J. & Saffran, J. R. (2010). From statistics to meaning: Infants' acquisition of lexical categories. *Psychological Science*, 21(2), 284–291. | `10.1177/0956797609358570` | Infants given reliable **statistical cues marking categories** in the input learned trained word–picture pairings and generalised to novel pairings; the control group did neither. The cue must be present in the input for the word. |
| C15 | Feijoo, S., Muñoz, C., Amadó, A. & Serrat, E. (2017). When meaning is not enough: Distributional and semantic cues to word categorization in child directed speech. *Frontiers in Psychology*, 8, 1242. | `10.3389/fpsyg.2017.01242` | Distributional + semantic cues jointly categorise nouns in natural child-directed speech; children weight **type frequency** over token frequency. Needs a corpus. |
| C16 | Gagliardi, A., Bennett, E., Lidz, J. & Feldman, N. H. (2012). Children's inferences in generalizing novel nouns and adjectives. *Proceedings of the Cognitive Science Society*, 34(34). | No DOI; eScholarship UC item `1rz1x81b`, ISSN 1069-7977 | Children already determine grammatical category from syntactic/morphological cues and use that category to bias the *meaning* hypothesis (kind hierarchy for nouns, property hierarchy for adjectives); a Bayesian model trained on the English lexicon predicts the pattern. |
| C17 | Horst, J. S. & Samuelson, L. K. (2008). Fast mapping but poor retention by 24-month-old infants. *Infancy*, 13(2), 128–157. | `10.1080/15250000701795598` | Fast mapping is robust for referent selection but transient for retention; retention improved when selection was augmented with ostensive naming. |
| C18 | Gick, M. L. & Holyoak, K. J. (1983). Schema induction and analogical transfer. *Cognitive Psychology*, 15(1), 1–38. | `10.1016/0010-0285(83)90002-6` | Abstract schema induction from multiple concrete exemplars enables transfer to a new case. |
| C19 | Holyoak, K. J. & Thagard, P. (1989). Analogical mapping by constraint satisfaction. *Cognitive Science*, 13(3), 295–355. | `10.1207/s15516709cog1303_1` | Candidate roles are chosen by relational fit under interacting constraints, not by feature overlap alone. |

### 1.1 Corrections forced by the verification pass

- **"Rogers & McClelland 2004 PNAS" in the task brief is wrong.** Crossref and MIT Press
  confirm Rogers & McClelland 2004 is an **MIT Press book** (no DOI), not a PNAS paper.
  The nearest verified journal articles are C9 (*Psychological Review* 111(1):205–235)
  and C8/C10. Both are cited here instead; the book is listed bibliographically only.
- Two DOIs guessed in an earlier pass were checked and **discarded**: `10.1073/pnas.0400087101`
  resolves to Barrat et al. on weighted networks, and `10.1037/h0022100` resolves to
  Tuckman on small groups. Neither appears in this document. Lany & Saffran's DOI was
  likewise corrected from a wrong guess to `10.1177/0956797609358570` after a
  publisher/PubMed lookup.
- C1 has no DOI; its bibliographic identity is established by the primary proceedings
  PDF and multiple independent catalogues.

---

## 2. Mechanism map: brain finding → computational primitive → RAVANA-compatible object

Status labels: **S** = the cited source directly supports the brain/behaviour claim;
**A** = supported as a design analogy (not neural homology); **E** = engineering
extrapolation with no direct support; **X** = extrapolated and *measured to fail* in an
existing scratch experiment. No cell claims RAVANA implements brain machinery.

| # | Brain finding (source) | Computational primitive | RAVANA-compatible object (verified present in tree) | Ledger | Available today? |
|---|---|---|---|---|---|
| M1 | One/few exposures create a partial entry, extended mapping is slow (C1, C2, C17) | Immutable episode with provenance; abstraction derived on demand, never written back as a fact | `ConceptEdge` + `source_metadata` (`edge_kind`, `source`, `source_type`, `extractor`) written through `KnowledgeIntake.intake(fact, Provenance(...))`; cold-reload retention verified in `TRA_SLOT_WRITE_READ_EXPERIMENT.md` | S for the dissociation, A for the storage split | Yes |
| M2 | One-trial assimilation requires an existing schema (C11); fast store + slow extraction (C12) | Require converged support before a slot may propose; keep the hypothesis unassertable | `schema_induction.traces_from_edges`, `induce_schema_hypotheses` (min_support=2, `assertable=False`), `admit_for_query`, `rank_candidates_for_unseen_subject` (`schema_induction.py:433,457,474,512`) | S for the dependence, A for the count gate | Yes, default-off |
| M3 | Distinct episodes must not blur (C13) | Distinct trace ids; duplicates cannot inflate support | `traces_from_edges` dedup over distinct trusted triples; trust predicate reads `source_metadata` (`_is_trusted_edge`, :102) | S for separation, A for the id scheme | Yes |
| M4 | Bayesian sparse induction over a structured hypothesis space (C4, C5) | Closed candidate set + support-weighted strict winner; abstain on ties/thin evidence | `slot_keyed_forced_choice` (`schema_induction.py:208`), congruence floor 0.20, strict margin 0.10, `pick=None` on no winner | S for structured induction, E for the scalar gate | Yes |
| M5 | Relational fit, not feature overlap, maps roles to new cases (C18, C19) | Group by `relation_type`/`slot_label`; score candidate by role fit | `typed_slot_views_from_graph` (:171) with `relation_type='is_a'`, `type_label`, `slot_label`; `_typed_slot_candidate` (`response_gen.py:7683`) | S for framework, E for graph-typed slots | Yes for **taught** words |
| M6 | Statistical/d morphological cues mark lexical categories and bias meaning (C14, C15, C16) | Count syntactic frames or type-frequency suffixes **for the queried word** | No general typer exists. `pet_slots.py:327` strips only a pet multiplicity suffix; `engine._definitions` supplies definition text for a subset of words (`brain_regions.py:559–565`) | S for the cue, **E for any current use** | **No** — requires new knowledge input |
| M7 | Semantic structure from large co-occurrence (C6, C7, C8, C10) | Distributional prototype from a corpus | `data/ravana_glove_cache.npz` (400,001 × 64 via `build_glove64_lookup`); slot centres = mean unit vectors of taught `r2` words | S for corpus-based semantics; **X** as deployed type signal | Present but measured unusable |
| M8 | Social/ostensive context matters; observation alone mostly fails (C1, C3, C17) | Type evidence must be delivered as real assertions in `process_turn`, with provenance | `RAVANA_TRA_SLOTS` / `RAVANA_SCHEMA` flags, `intake.intake` with `Provenance(source='conversation', source_type='user')`; web path `measure_fronts_web_frozen.py` for frozen snapshots | S for behaviour, E for the API choice | Yes as a channel; empty for the unseen test word |

**Read M6/M7 together.** The two mechanisms that could actually type a *new* word in the
brain — distributional statistics and prior grammatical-category knowledge — are exactly
the two RAVANA channels that are empty at query time. M1–M5 store and gate evidence;
they do not generate it.

---

## 3. SUPPORT vs EXTRAPOLATION ledger

| Claim used in RAVANA design | Source(s) | Verdict | Limit / consequence |
|---|---|---|---|
| A single natural assertion can create a provisional, provenance-stored type trace | C1, C2, C17 | **SUPPORT** (behaviour) | Human fast mapping requires a referent/contrast; RAVANA's assertion has neither. Storage is supported; typing an unseen subject is not. |
| Cold-reload retention is a mandatory endpoint | C17 | **SUPPORT** | Measured retention of *explicitly taught* edges (489/489) is not unseen-word transfer. |
| Eight exemplars of other words suffice to type an unseen word | — | **NO SUPPORT** | C4/C5 make the information value of exemplars contingent on a pre-existing hierarchy; C3 says most natural instances are uninformative. Five scratch experiments found 0–2/40 near/far. |
| GloVe slot-centre cosine is a stand-in for a type slot | C6, C7 | **EXTRAPOLATION → measured failure (X)** | LSA's result needs a corpus containing the word; a 64-D centroid of *other* words is not a relational slot. Pairwise cos 0.729–0.980; exact-centre sanity margin 0.0198–0.0873 < 0.10 gate; `C_pooled_cos` 1/40 near and far. |
| Distinct trace ids approximate hippocampal pattern separation | C13 | **A (analogy)** | No hippocampal measurement in RAVANA; the identity scheme is the whole claim. |
| `min_support=2` is a human-like threshold | C11, C18 | **E (operationalisation)** | Tse shows a schema must *pre-exist*; it does not measure "2 demonstrations". |
| Keep the schema hypothesis unassertable because schema congruence causes false memories | C11, C12 | **A** | Human false-memory paradigm; structural mapping to the gate only. |
| Provenance-stamped conversation ≈ social/ostensive input | C3, C17 | **E** | Conversation supplies provenance, not a referent. The social cue is approximated, not reproduced. |
| Conversational/web assertions about a broad vocabulary supply the "schema" that makes sparse evidence useful | C11, C4, C5 | **SUPPORT (conceptual) → E (implementation)** | Direction supported; the exact assertion vocabulary and the A/B that would validate it have not been run. |
| A forced-choice answer path can emit the typed slot | C19 | **A** | Verified only for words with a stored typed edge (`TRA_SLOT_REAL_ENGINE_CHECK.md`); that case teaches the tested word's own type and is an **oracle upper bound**, non-deployable per `ravana-debugging` §7. |

---

## 4. INFORMATION ANALYSIS — what signal actually exists

### 4.1 The three input surfaces

**Teaching turns (P2).** `TRA_BENCHMARK_SPEC.md` §1/§2 defines them as natural
assertions such as `"cloud causes rainbow."` with `type_label='effect'` carried in the
item bank. The spec explicitly says the bank's `type_sets` and `world_model` are "never
sent to the engine at query time" and are "leakage if exposed to the graded engine
path". In the worked example the teaching surfaces verbalise a **relation**, not a
**type**: no sentence says `rainbow is an effect`. The engine therefore receives:
- which r1→r2 pairs hold (a constant relation family within the item);
- no general statement that r2 belongs to a type;
- no statement whatsoever about the unseen test subject (`thaw`, `glow`, `ceremony`, …).

The only scratch configuration that ever moved a positive typed-slot endpoint taught
`"mist is a weather phenomenon."` — i.e. it supplied the type of the word being tested.
That is precisely the non-deployable oracle the measurement discipline forbids.

**Query prompt (P3).** `TRA_BENCHMARK_SPEC.md:125` fixes the surface:
`"Considering the examples above, what is true of {subject}? Options: {candidates}.
Answer briefly."` It contains a bare subject token, four candidate labels, and a
relation-neutral frame. It contains no `X is a ___` slot, no sentential context for
`X`, no morphological marker deliberately placed on `X`, and no distributional
evidence about `X`.

**Existing graph.** Trusted typed edges exist only for words explicitly taught (6 edges
in the recorded experiment: `mist/hail/drizzle→weather phenomenon`,
`election/ceremony/contract→civic event`). The measured store has ~0.9% typed edges
with all nodes typed `concrete`; the GloVe cache holds a context-free vector for most
words but only 1 of 6 TRA type labels is in GloVe (`TRA_CATEGORY_CENTRE_EXPERIMENT.md`,
"Word coverage"). No typed edge for the held-out subjects is present by construction.

### 4.2 Enumerate every channel that could carry the gold type T of subject S

| Channel | Present in RAVANA input? | Brain analogue | Verdict |
|---|---|---|---|
| (a) Direct assertion "`S is a T`" in teaching turns | **No** — S is "never demonstrated", and if supplied it is answer-key leakage | Ostensive naming / contrastive phrase (C1, C17) | Absent |
| (b) Diagnostic syntactic frame around S in the query | **No** — bare token in a forced-choice frame | Grammatical-category cue (C16) | Absent |
| (c) Suffix/morphology on S | **No** for the recorded vocab (`thaw`, `glow`, `chill`, `dew`, `sleet`, `ceremony`, `election`, `contract`, `habit`); no general suffix typer exists | Type-frequency morphology (C15, C16) | Absent |
| (d) Corpus contexts for S (distributional/SVD) | **No** in the engine path — a single GloVe prototype is not a context sample | LSA (C6, C7); category-statistics training (C14, C15) | Absent / collapsed |
| (e) Stored definition of S whose head noun is a candidate type | **Unknown, must be censused** (`engine._definitions`, `brain_regions.py:559`); not used by any graded TRA path | Lexical-semantic commitment | Open, likely thin |
| (f) Structural prior: taught exemplars constrain S via a known hierarchy | **No hierarchy** — relation type is constant within a family, so taught r2 words give no discriminating constraint across candidates | Bayesian hierarchy (C4, C5); schema (C11) | Absent |

With (a)–(d) and (f) absent, the conditional entropy H(T | I) is essentially H(T): the
best attainable decision rule is a prior guess over four options (chance ≈ 0.25), not a
classification. The observed 0–2/40 near/far with intact foil/negative rejection is
exactly the signature of a fail-closed system with no signal — not of a broken ranker.
The literature agrees: Xu & Tenenbaum's effect depends on a hierarchy the learner
*already has*; Tse's one-trial assimilation depends on a schema that *already exists*;
Lany & Saffran's generalisation depends on cues that *were in the input*. RAVANA is being
asked to do a task humans do not do under these conditions.

### 4.3 What the five failed experiments actually establish

| Experiment | Route | Result | What it rules out |
|---|---|---|---|
| RSI role centres (`RSI_ROLE_BINDING_EXPERIMENT.md`) | r1/r2 GloVe role centroids | near 1/4, far 1/4; r2 centre still an embedding neighbourhood | Embedding role alignment |
| Symbolic signatures (`RSI_SYMBOLIC_ROLE_EXPERIMENT.md`) | relation + token overlap + 3-gram | PART_OF/PRECEDES near+far 0/4 under both floors | Symbol overlap from taught tokens |
| Typed slots (`RSI_TYPED_ROLE_DESIGN.md`, `TRA_SLOT_*`) | explicit `is_a` provenance writes | 6 typed edges retained cold; only the taught word is classified | Storage and retention — **not** transfer |
| Category centroids (`TRA_CATEGORY_CENTRE_EXPERIMENT.md`) | mean unit vector of taught r2 per type | centres cos 0.729–0.980; near/far 0–1/40 | Centroid geometry as a type signal |
| Focused live TRA/RSI (`RSI_FOCUSED_LIVE.md`) | trusted-edge hook + query turn | near/far 0.00 on both pools, foil intact | Direct-edge retrieval as induction |

Collectively these exhaust the candidate internal signals. The common failure mode is
not a threshold; it is that no number computed from the taught set discriminates the
unseen subject's type.

---

## 5. MINIMAL DEPLOYABLE PATH

Ranked by evidence and by the ~20 W no-LLM constraint. All primitives below already
exist; none requires training a network at inference.

**Path A — recommended: make type knowledge a real input (not an oracle).**
Acquire explicit type assertions (`X is a Y`, `X is part of a Y`) for a broad vocabulary
through genuine conversation and web reading, written as provenance-stamped
`ConceptEdge`s through the unchanged `intake.intake` API. This gives RAVANA the
pre-existing structured prior that C4/C5 and C11 say sparse learning presupposes.
Requirements: (1) assertions come from real turns/pages, never from `type_sets`,
`world_model`, `domain`, `far_domain` or `gold_slot`; (2) held-out test words and their
types are excluded from the acquisition corpus at item-construction time and the split
is hash-frozen before any run; (3) ship default-off behind the existing flag, prove flag
equivalence, then A/B with positive (near/far) **and** negative (foil/negative) endpoints;
(4) keep cold-reload retention and source-invalidation controls. This is not answer-key
leakage — it is the system learning types the way a child learns that `chromium` is a
colour word: from an explicit, attributable statement.

**Path B — only for genuinely unseen words: add a distributional channel.**
C14/C15/C16 show the human cue is a *frame count* over sentences containing the target
word. A ~20 W implementation is feasible: read web text (frozen snapshots per
`measure_fronts_web_frozen.py`), count syntactic frames and type-frequency suffixes for
the queried token, and feed those counts as data to `slot_keyed_forced_choice`. This is
**new knowledge input**; without it, Path A covers only words already in the acquired
vocabulary. Do not substitute a GloVe centroid for these counts — that substitution is
the measured X failure in §3.

**Path C — honest abstention.** When neither Path A nor Path B yields type evidence for
S, return `None`. This is the behaviour the literature predicts for an evidence-free
query, and it preserves foil/negative rejection. Do not lower the 0.20 floor or the
0.10 margin to force coverage; `ravana-debugging` records that a coverage knob is not a
conservatism knob.

**Explicitly non-deployable:** feeding `type_sets`/`world_model`, the item's
`type_label`, or any gold-derived context into the engine or the scorer. That is answer-key
leakage, scores G6-saturated, and is labelled an oracle upper bound in every doc that
mentions it.

**Instrument consequence:** if TRA is to remain a *capability* benchmark, its teaching
turns must actually verbalise the type (`"rainbow is a weather event."`) and the held-out
word must have an independently acquired type source. As currently specified, TRA is an
information-availability test and should be relabelled until one of those conditions
holds. Relabelling is not tuning the benchmark to pass; it is correcting what the
instrument measures.

---

## 6. Falsifiers and the unbridgeability criterion

Pre-registered; each has a measurable prediction.

| ID | Test | If it passes | If it fails |
|---|---|---|---|
| F1 | Forced single-token type decision, echoed text stripped, self-consistent option list, direct type assertions in teaching turns; census trusted typed edges for the tested word | ≥1 trusted typed edge present and still ~0 correct ⇒ **mechanism/articulation gap**, not information gap | — |
| F2 | Census `engine._definitions` for ≥50% of held-out subjects; check whether the definition head noun equals a candidate label | Coverage present but near/far still 0 ⇒ **wiring gap** in an existing channel | Coverage 0 ⇒ definition channel absent; Path B required |
| F3 | Re-estimate slot centres from stored data; re-run exact-centre sanity | Pairwise cos drops below ~0.6 and exact-centre top-2 margin clears 0.10 ⇒ embeddings may carry a usable signal | cos stays >0.9 / margin stays <0.10 ⇒ **GloVe route unbridgeable**; stop tuning it |
| F4 | Non-deployable oracle arm: directly teach `S is a T` with full provenance, grade the same query | Success while graph-derived arm stays 0 ⇒ information/wiring gap; Path A is the fix | Even direct assertions yield ~0 with foil/negative intact ⇒ **genuine architecture limit** — the strongest possible proof the gap is unbridgeable |
| F5 | Real-input acquisition arm: read genuine conversation/web pages, count written trusted typed edges for held-out vocabulary | ≥N edges written and near/far rises with foil/negative flat ⇒ **deployable path confirmed** | 0 edges written ⇒ corpus/extractor ceiling, not a mechanism problem |
| F6 | Source invalidation of a taught type | Withdrawal of the type after discrediting the source ⇒ provenance is genuinely read | No withdrawal ⇒ provenance stored but not consumed |

**When is the gap unbridgeable under current constraints?** Only when (i) the teaching
turns and query contain no type-bearing cue for S, (ii) the graph has no typed edge or
definition for S, and (iii) the sole candidate signal is a non-separable centroid — all
three of which are **measured here** — *and* no new knowledge input is permitted. That is
not a permanent impossibility claim: it is the precise statement that the closed loop
is information-starved. The one observation that would establish true unbridgeability
even with new input is F4 failing (direct, provenance-stamped assertions about S still
produce no correct typed decisions). Until F4 is run, the honest verdict is
**"new knowledge input required; architecture not yet implicated."**

---

## 7. Compute and scope note

Every deployable primitive in §5 is a dict lookup, an edge census, a cosine over a
64-D vector, or an integer frame count — no LLM and no training at inference, consistent
with the ~20 W operating point. LSA (C6/C7) and PDP (C8/C10) are cited as *brain
grounds* for why distributional input and gradual schema formation matter; the doc does
**not** propose deploying either model. No engine was booted and no control was
re-measured to produce this document; all empirical figures are attributed to the
committed investigation docs named in-line.
